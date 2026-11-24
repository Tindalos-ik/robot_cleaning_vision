"""区域训练、场景折外特征和整图训练，按部署阈值报告验证指标。"""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from .features import extract_context, extract_regions, generate_windows, prepare_image, rgb_image
from .model import DEPLOYMENT_THRESHOLD, _network, score_summary


def _training_patches(image, objects, image_id, proposal, rng):
    """把一张训练图变成有标签的区域特征，返回特征、标签、原图ID及权重。

    正区域直接取项目目标框；负区域从自动窗口中随机选择。
    正负区域都使用与预测相同的HSV160提取函数，但不把框坐标作为网络输入。
    """
    height, width = image.shape[:2]
    boxes = np.asarray([[obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]
                        for obj in objects], dtype=np.float64).reshape(-1, 4)
    # 框使用原图像素坐标。随机选最多4个，避免目标特别多的图片主导训练。
    positive = boxes[rng.permutation(len(boxes))[:4]]
    positive = np.column_stack((np.floor(positive[:, :2]), np.ceil(positive[:, 2:])))
    positive = np.clip(positive, [0, 0, 0, 0], [width - 1, height - 1, width, height]).astype(np.int32)

    candidates = generate_windows(image.shape, proposal)
    # 背景必须避开这张图的全部目标，而不只是刚选中的正区域。
    # half_sizes取宽高的0.55，相当于把目标框外扩为原来的1.1倍。
    if len(boxes):
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        half_sizes = (boxes[:, 2:] - boxes[:, :2]) * 0.55
        expanded = np.column_stack((centers - half_sizes, centers + half_sizes))
        overlap = np.minimum(candidates[:, None, 2:], expanded[None, :, 2:])
        overlap -= np.maximum(candidates[:, None, :2], expanded[None, :, :2])
        intersects = (overlap > 0).all(axis=2).any(axis=1)
        candidates = candidates[~intersects]
    negative = candidates[rng.permutation(len(candidates))[:8]]

    parts, labels, weights, sample_boxes = [], [], [], []
    # 先限制每张原图的总贡献，再把预算平均分给该图采出的各个区域。
    # 之后每个训练折还会单独平衡正负类总权重。

    for selected, label, budget in ((positive, 1, 1.0),
                                     (negative, 0, 0.4 if len(boxes) else 0.7)):
        if len(selected):
            parts.append(extract_regions(image, selected))
            labels.extend([label] * len(selected))
            weights.extend([budget / len(selected)] * len(selected))
            sample_boxes.extend(selected.tolist())
    return {
        'features': np.concatenate(parts),
        'labels': np.asarray(labels, dtype=np.int64),
        'weights': np.asarray(weights, dtype=np.float32),
        'image_ids': np.full(len(labels), image_id, dtype=np.int64),
        'boxes': sample_boxes,
    }


def prepare_dataset(dataset, proposal, *, training=False, seed=42):
    """一次提取像素特征；验证集只提自动窗口，不用真值框生成训练区域。"""
    rng = np.random.default_rng(seed)
    bags, patches = [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        image = rgb_image(sample['image'])
        # 自动窗口供整图推理和折外评分；一张图的窗口数可变，每行特征固定160维。
        frame = prepare_image(image, proposal['long_edge'])
        boxes = generate_windows(frame.shape, proposal)
        bags.append({
            'image_id': sample['image_id'],
            'label': int(sample['label'] == 'FAIL'),
            'group_id': dataset.images[index]['group_id'],
            'features': extract_regions(frame, boxes),
            'boxes': boxes,
            'frame_size': (frame.shape[1], frame.shape[0]),
            'context': extract_context(image),
        })
        if training:
            # 只有train生成带真值的正负crop，val只保留像素特征与整图标签。
            patches.append(_training_patches(image, sample['objects'], sample['image_id'], proposal, rng))
        if (index + 1) % 100 == 0 or index + 1 == len(dataset):
            print(f'{dataset.split}：已提取 {index + 1}/{len(dataset)} 张', flush=True)
    if not training:
        return bags, None
    combined = {key: np.concatenate([part[key] for part in patches])
                for key in ('features', 'labels', 'weights', 'image_ids')}
    return bags, combined


def _balance_weights(labels, weights):
    """在当前训练数据内平衡两类总权重，不使用留出场景。"""
    weights = np.array(weights, dtype=np.float32, copy=True)
    for label in (0, 1):
        selected = labels == label
        weights[selected] *= len(labels) / (2 * weights[selected].sum())
    return weights


def _probabilities(network, features, mean, scale):
    normalized = (np.asarray(features, dtype=np.float32) - mean) / scale
    with torch.inference_mode():
        return network(torch.from_numpy(normalized)).sigmoid().numpy().ravel()


def _fit_network(features, labels, weights, hidden, config, seed, validation=None):
    """用 features、labels 和样本 weights 训练宽度为 hidden 的 MLP。

    config 指定优化参数，seed 控制初始化和打乱。提供 validation 特征和标签时，
    按类均衡的验证损失早停；FAIL 权重只影响训练。
    """
    # 只用当前训练折拟合统计量，再转换为推理时使用的float32。
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    scaler = StandardScaler().fit(features)
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    inputs = torch.from_numpy((features - mean) / scale)
    targets = torch.as_tensor(labels, dtype=torch.float32).reshape(-1, 1)
    weights = _balance_weights(labels, weights)
    fail_weight = config.get('fail_weight', 1.0)
    if fail_weight != 1.0:
        # 调整两类总权重之比，保持所有样本的平均权重为 1。
        weights[labels == 1] *= fail_weight
        weights *= len(labels) / weights.sum()
    weights = torch.from_numpy(weights).reshape(-1, 1)

    # 结构由输入维数和隐藏单元数确定；BCE直接接收未做Sigmoid的logit。
    network = _network(features.shape[1], hidden)
    optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    if validation is not None:
        # 验证输入沿用上面的训练统计量；验证标签只用于计算损失和早停。
        val_features, val_labels = validation
        val_inputs = torch.from_numpy((val_features - mean) / scale)
        val_targets = torch.as_tensor(val_labels, dtype=torch.float32).reshape(-1, 1)
        val_weights = torch.from_numpy(_balance_weights(val_labels, np.ones(len(val_labels)))).reshape(-1, 1)

    history = []
    best_loss = float('inf')
    best_epoch = 0
    best_state = None
    stale = 0
    epochs = config['epochs']
    log_epochs = set(np.ceil(np.linspace(1, epochs, min(10, epochs))).astype(int))

    # 每批完成“清梯度→前向→损失→反向传播→更新权重”，一轮遍历全部训练行。
    for epoch in range(1, epochs + 1):
        network.train()
        total_loss = 0.0
        for indices in torch.randperm(len(inputs), generator=generator).split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = (criterion(network(inputs[indices]), targets[indices]) * weights[indices]).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        network.eval()
        row = {'epoch': epoch, 'train_loss': total_loss / len(inputs)}
        if validation is not None:
            with torch.inference_mode():
                val_loss = float((criterion(network(val_inputs), val_targets) * val_weights).mean())
            row['val_loss'] = val_loss
            improved = val_loss < best_loss
        else:
            val_loss = 0.0
            improved = True
        history.append(row)
        if improved:
            # 必须深拷贝：state_dict直接引用参数，后续训练会修改原张量。
            best_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(network.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch in log_epochs:
            message = f"Epoch {epoch}/{epochs} | train_loss={row['train_loss']:.4f}"
            if validation is not None:
                message += f' | val_loss={val_loss:.4f}'
            print(message, flush=True)
        if validation is not None and stale >= config['patience']:
            print(f'早停：第{epoch}轮，恢复第{best_epoch}轮权重。', flush=True)
            break
    # 区域固定轮数时保留最后一轮；整图有validation时恢复验证损失最低的一轮。
    network.load_state_dict(best_state)
    return network, mean, scale, history, best_epoch


def _image_features(bags, network, mean, scale):
    rows = []
    # 先算每个窗口的分数，再做30维固定统计，与predict()的顺序完全一致。
    for bag in bags:
        probabilities = _probabilities(network, bag['features'], mean, scale)
        summary = score_summary(probabilities, bag['boxes'], bag['frame_size'])
        rows.append(np.r_[summary, bag['context']])
    return np.asarray(rows, dtype=np.float32)


def evaluation_metrics(labels, probabilities, threshold):
    """用 threshold 汇总 labels 与 probabilities；混淆矩阵的行是真值、列是预测。"""
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    fail_recall = float(tp / (tp + fn))
    pass_recall = float(tn / (tn + fp))
    return {
        'samples': len(labels),
        'threshold': threshold,
        'accuracy': float(np.mean(predicted == labels)),
        'fail_recall': fail_recall,
        'pass_recall': pass_recall,
        'balanced_accuracy': (fail_recall + pass_recall) / 2,
        'roc_auc': float(roc_auc_score(labels, probabilities)),
        'confusion_matrix_pass_fail': [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _prepare_image_training(train_dataset, val_dataset, proposal, region_config, folds, seed):
    """从 train_dataset 和 val_dataset 生成整图训练和验证输入。

    proposal 定义窗口，region_config 定义区域优化参数，folds 是场景折数，
    seed 用于区域采样和训练。返回区域检查点、两组矩阵和标签、区域训练记录。
    """
    torch.set_num_threads(1)
    train_bags, patches = prepare_dataset(train_dataset, proposal, training=True, seed=seed)
    val_bags, _ = prepare_dataset(val_dataset, proposal)
    train_ids = np.array([bag['image_id'] for bag in train_bags])
    labels = np.array([bag['label'] for bag in train_bags])
    groups = np.array([bag['group_id'] for bag in train_bags])
    val_labels = np.array([bag['label'] for bag in val_bags])

    # 折外分数：这张训练图片的区域分数，来自没有见过该场景的区域网络。
    # 同一group的所有图片和区域必须留在同一侧，不能仅按区域随机分折。
    oof = np.empty((len(train_bags), 88), dtype=np.float32)
    fold_reports = []
    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=123)
    for fold, (fit, held) in enumerate(splitter.split(train_ids, labels, groups), start=1):
        print(f'场景折 {fold}/{folds}：训练{len(fit)}张，留出{len(held)}张', flush=True)
        selected = np.isin(patches['image_ids'], train_ids[fit])
        # scaler、网络和类权重都只看到fit图片的区域，held区域不会进入训练。
        network, mean, scale, history, best_epoch = _fit_network(
            patches['features'][selected], patches['labels'][selected], patches['weights'][selected],
            32, region_config, seed)
        # held行只由当前折模型填入；三折结束后，每张训练图片恰好得到一次折外输入。
        oof[held] = _image_features([train_bags[index] for index in held], network, mean, scale)
        fold_reports.append({
            'fold': fold,
            'fit_image_ids': train_ids[fit].tolist(),
            'held_image_ids': train_ids[held].tolist(),
            'fit_group_ids': sorted(set(groups[fit])),
            'held_group_ids': sorted(set(groups[held])),
            'scaler_fit_patch_count': int(selected.sum()),
            'best_epoch': best_epoch,
            'history': history,
        })


    # 部署模型使用全部train训练的区域网络，而不是保存三个折模型。
    # 三折网络只服务于生成整图训练输入，使用后即可释放。
    print('训练最终区域网络：使用全部train区域', flush=True)
    region_network, region_mean, region_scale, region_history, _ = _fit_network(
        patches['features'], patches['labels'], patches['weights'], 32, region_config, seed)
    # val只经过最终区域网络的前向计算，没有参与其权重或scaler拟合。
    val_features = _image_features(val_bags, region_network, region_mean, region_scale)
    region_checkpoint = {
        'mode': 'color', 'proposal': proposal,
        'state_dict': region_network.state_dict(),
        'mean': torch.from_numpy(region_mean), 'scale': torch.from_numpy(region_scale),
    }
    report = {
        'fold_seed': 123, 'folds': fold_reports,
        'fold_policy': 'StratifiedGroupKFold on train group_id; fit-only patch scaler and weights',
        'train_image_ids': train_ids.tolist(),
        'val_image_ids': [bag['image_id'] for bag in val_bags],
        'region_history': region_history,
    }
    return region_checkpoint, (oof, labels), (val_features, val_labels), report


def _fit_image_checkpoint(region_checkpoint, train, validation, image_config, seed):
    """用区域 region_checkpoint 和整图 train、validation 完成两级模型。

    train 和 validation 都是（特征、标签）；image_config 设置整图优化参数，
    seed 控制初始化和打乱。返回检查点、验证分数及早停记录。
    """
    print('训练整图网络：使用场景折外的88维输入', flush=True)
    oof, labels = train
    val_features, _ = validation
    image_network, image_mean, image_scale, image_history, best_epoch = _fit_network(
        oof, labels, np.ones(len(labels)), 8, image_config, seed, validation)
    # 推理每次只送入一张图，验证也使用单行前向计算。
    probabilities = _deployment_probabilities(image_network, val_features, image_mean, image_scale)
    checkpoint = {
        'kind': 'regional-score-stack-v1',
        'summary_version': 'region-score-summary-30-v1',
        'include_context': True,
        'context_version': 'local-384-v1',
        'region_checkpoint': region_checkpoint,
        'state_dict': image_network.state_dict(),
        'mean': torch.from_numpy(image_mean), 'scale': torch.from_numpy(image_scale),
        'threshold': DEPLOYMENT_THRESHOLD,
    }
    return checkpoint, probabilities, image_history, best_epoch


def _deployment_probabilities(network, features, mean, scale):
    """按实际推理的单图形状，用 network 及标准化参数 mean、scale 为 features 打分。"""
    normalized = (features - mean) / scale
    with torch.inference_mode():
        return np.asarray([network(row[None, :]).sigmoid().item() for row in torch.from_numpy(normalized)])


def train_model(train_dataset, val_dataset, output_root, *, annotations, proposal, region_config,
                image_config, folds=3, seed=42, provenance=None):
    """用 train_dataset 训练两级模型，用 val_dataset 早停并按部署阈值评估。

    output_root 用于新运行目录；annotations 和 provenance 记录数据来源。
    proposal 定义窗口，region_config 和 image_config 定义优化参数；
    folds 是训练集场景折数，seed 用于采样、初始化及打乱。
    """
    region, train, validation, region_report = _prepare_image_training(
        train_dataset, val_dataset, proposal, region_config, folds, seed)
    checkpoint, probabilities, image_history, best_epoch = _fit_image_checkpoint(
        region, train, validation, image_config, seed)
    _, val_labels = validation
    metrics = evaluation_metrics(val_labels, probabilities, DEPLOYMENT_THRESHOLD)
    report = {
        'architecture': {'region': [160, 32, 1], 'image': [88, 8, 1]},
        'proposal': proposal,
        'sampling': {'positive_per_image': 4, 'negative_per_image': 8, 'exclude_margin': 1.1,
                     'positive_budget': 1.0, 'pass_negative_budget': 0.7, 'fail_negative_budget': 0.4,
                     'background_source': 'random automatic windows mapped by original image size'},
        'region_config': region_config, 'image_config': image_config,
        'annotations_sha256': hashlib.sha256(Path(annotations).read_bytes()).hexdigest(),
        'annotations': str(annotations),
        'data_provenance': provenance,
        **region_report,
        'seed': seed,
        'deployment_threshold': DEPLOYMENT_THRESHOLD,
        'image_history': image_history,
        'best_image_epoch': best_epoch, 'validation': metrics,
        'validation_predictions': [
            {'image_id': image_id, 'label': int(label), 'fail_probability': float(score)}
            for image_id, label, score in zip(region_report['val_image_ids'], val_labels, probabilities)
        ],
    }
    # 每次训练生成独立目录。默认的artifacts/mlp/model.pt完全不动，
    # 用户比较新结果后，再决定是否替换场景实测使用的模型。
    directory = Path(output_root) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'model.pt').open('xb') as stream:
        torch.save(checkpoint, stream)
    model_hash = hashlib.sha256((directory / 'model.pt').read_bytes()).hexdigest()
    for filename, payload in (('training.json', report),
                              ('metrics.json', {'model': 'model.pt', 'sha256': model_hash,
                                                'split': 'val', 'metrics': metrics})):
        with (directory / filename).open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f'已保存新模型与训练报告：{directory}', flush=True)
    return directory
