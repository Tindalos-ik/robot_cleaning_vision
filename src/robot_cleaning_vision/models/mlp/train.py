"""训练区域与整图 MLP，读取审核过的补充标注并按部署阈值评估。"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from dataprocessing.taco import ProjectDataset
from dataprocessing.supplemental.datasets import (
    SUPPLEMENTAL_ANNOTATIONS, TrainingDataset, add_supplemental_argument, supplemental_dataset,
)

from .training import prepare_dataset, train_model

# ===== 数据与保存位置 =====
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = PROJECT_ROOT / 'data/raw/taco'
ANNOTATIONS = DATA_ROOT / 'annotations_project.json'
LOCAL_TRAINING_ANNOTATIONS = SUPPLEMENTAL_ANNOTATIONS.with_name('local_training_annotations_project.json')
OUTPUT_ROOT = PROJECT_ROOT / 'artifacts/mlp/training'

# ===== 固定的特征与网络方案 =====
# 长边640的等比图上生成多尺度窗口；区域HSV160，整图为区域统计30＋背景58。
# 网络结构固定为区域160→32→1、整图88→8→1，与现有推理代码一致。
PROPOSAL = {
    'long_edge': 640,
    'scales': [0.12, 0.24, 0.48, 0.9],
    'aspects': [0.5, 1.0, 2.0],
    'stride': 0.8,
}

# 区域网络沿用选定方案的17轮。折外训练与最终区域训练使用同样轮数，
# 不根据当前留出折选择轮次，保证它的标签不影响对应区域模型的训练。
REGION_CONFIG = {
    'epochs': 17,
    'learning_rate': 0.001,
    'weight_decay': 0.001,
    'batch_size': 128,
}

# 整图网络用验证损失早停；训练结束后恢复验证损失最低的权重。
IMAGE_CONFIG = {
    'epochs': 200,
    'patience': 25,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'batch_size': 32,
    'fail_weight': 1.0,
}
FOLDS = 3  # 按原图的group_id分折，同一场景不能跨折。
SEED = 42


def data_provenance(supplemental, local_training_root=None):
    """记录 supplemental 来源及可选本地原图根目录 local_training_root、标注和摘要。"""
    report = {
        'supplemental': supplemental,
        'annotations': str(ANNOTATIONS),
        'annotations_sha256': hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest(),
    }
    if supplemental != 'none':
        report.update(
            supplemental_annotations=str(SUPPLEMENTAL_ANNOTATIONS),
            supplemental_annotations_sha256=hashlib.sha256(
                SUPPLEMENTAL_ANNOTATIONS.read_bytes()).hexdigest(),
        )
    if local_training_root is not None:
        report.update(
            local_training_root=str(local_training_root.resolve()),
            local_training_annotations=str(LOCAL_TRAINING_ANNOTATIONS),
            local_training_annotations_sha256=hashlib.sha256(
                LOCAL_TRAINING_ANNOTATIONS.read_bytes()).hexdigest(),
        )
    return report


def check_data(dataset, supplemental, provenance):
    """从 dataset 提取真实训练区域，按 supplemental 模式检查标签并打印 provenance。"""
    if not len(dataset):
        raise ValueError('检查数据不能为空')
    if supplemental != 'none':
        cox_labels = {image['label'] for image in dataset.images if image['source'] == 'coxpwd'}
        if cox_labels != {'PASS', 'FAIL'}:
            raise ValueError('Cox 数据必须同时包含 PASS 和 FAIL')
    bags, patches = prepare_dataset(dataset, PROPOSAL, training=True, seed=SEED)
    labels = np.asarray([bag['label'] for bag in bags])
    if not np.isfinite(patches['features']).all():
        raise ValueError('MLP 区域特征包含非有限值')
    report = {
        'data_provenance': provenance,
        'images': len(bags),
        'labels': {'PASS': int((labels == 0).sum()), 'FAIL': int((labels == 1).sum())},
        'groups': len({bag['group_id'] for bag in bags}),
        'regions': len(patches['labels']),
        'region_labels': {
            'background': int((patches['labels'] == 0).sum()),
            'target': int((patches['labels'] == 1).sum()),
        },
        'region_feature_shape': list(patches['features'].shape),
        'window_count': sum(len(bag['features']) for bag in bags),
    }
    if supplemental != 'none' and set(patches['labels'].tolist()) != {0, 1}:
        raise ValueError('补充检查数据必须生成背景和目标区域')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def _positive_float(value):
    """把命令行字符串 value 转成有限正数，拒绝无效的优化参数。"""
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('必须是有限正数')
    return result


def _select_training_data(supplemental, local_root, *, check_data=False):
    """读取 supplemental 指定的公开补充来源和 local_root 下的本地审核图片。

    正式训练始终加入 TACO train；check_data 为真且指定了补充图时只检查补充图。
    每个分区直接加入组合，以便共用读取器检查图像、标注和场景组的 ID 冲突。
    """
    extra = []
    if supplemental != 'none':
        extra.append(supplemental_dataset(supplemental))
    if local_root is not None:
        local = ProjectDataset(local_root, LOCAL_TRAINING_ANNOTATIONS, 'train')
        if not len(local):
            raise ValueError('本地审核标注没有训练图片')
        extra.append(local)
    if not check_data or not extra:
        extra.insert(0, ProjectDataset(DATA_ROOT, ANNOTATIONS, 'train'))
    return TrainingDataset(*extra) if len(extra) > 1 else extra[0]


def main(argv=None):
    """解析 argv（省略时读取进程参数），检查真实数据或执行训练。"""
    parser = argparse.ArgumentParser(description=__doc__)
    add_supplemental_argument(parser)
    parser.add_argument('--local-training-root', type=Path,
                        help='审核标注中图片相对路径的根目录，例如 data/raw/supplemental/local_training。')
    parser.add_argument('--image-fail-weight', type=_positive_float, default=IMAGE_CONFIG['fail_weight'],
                        help='整图训练时 FAIL 类相对 PASS 类的总损失权重。')
    parser.add_argument('--image-weight-decay', type=_positive_float, default=IMAGE_CONFIG['weight_decay'],
                        help='整图 Adam 优化器的 weight_decay。')
    parser.add_argument('--check-data', action='store_true', help='提取真实区域特征并打印数据统计，不训练或保存模型')
    args = parser.parse_args(argv)
    dataset = _select_training_data(args.supplemental, args.local_training_root, check_data=args.check_data)
    provenance = data_provenance(args.supplemental, args.local_training_root)
    if args.check_data:
        return check_data(dataset, args.supplemental, provenance)
    # 保留原 TACO 验证分区，训练阶段不读取测试图片。
    image_config = dict(IMAGE_CONFIG, fail_weight=args.image_fail_weight,
                        weight_decay=args.image_weight_decay)
    val_dataset = ProjectDataset(DATA_ROOT, ANNOTATIONS, 'val')
    return train_model(
        dataset,
        val_dataset,
        OUTPUT_ROOT,
        annotations=ANNOTATIONS,
        proposal=PROPOSAL,
        region_config=REGION_CONFIG,
        image_config=image_config,
        folds=FOLDS,
        seed=SEED,
        provenance=provenance,
    )


if __name__ == '__main__':
    main()
