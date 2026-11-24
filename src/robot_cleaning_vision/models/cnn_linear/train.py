"""复训 baseline2：冻结 ResNet-50，只用训练图拟合标准化参数和 L2 逻辑回归。"""

import argparse
from collections import Counter
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler

from dataprocessing.taco import ProjectDataset
from dataprocessing.supplemental.datasets import (
    INDOOR_HOLDOUT, INDOOR_HOLDOUT_SHA256, SUPPLEMENTAL_ROOT, TrainingDataset,
    indoor_dataset, supplemental_dataset,
)
from dataprocessing.taco.images import file_hash

from .model import (
    ARTIFACT_ROOT, LABELS, WEIGHTS,
    CNNLinearModel, FeatureExtractor, LinearHead, compute_device,
)


DATA_ROOT = ARTIFACT_ROOT.parents[1] / "data/raw/taco"
ANNOTATIONS = DATA_ROOT / "annotations_project.json"
LOCAL_ANNOTATIONS = INDOOR_HOLDOUT.with_name("local_training_annotations_project.json")
TRAINING_ROOT = ARTIFACT_ROOT / "training"
BATCH_SIZE = 16
SEED = 42
MINIMUM_FAIL_RECALL = 0.90
INPUT_SIZE = 384
REGULARIZATION_C = 0.001
CLASS_WEIGHT = "balanced"
SOLVER = "lbfgs"
MAX_ITER = 2000
CONFUSION_ORDER = ["PASS", "FAIL"]


@dataclass
class SplitFeatures:
    """保存正式分区的整图特征和数值标签。"""

    features: np.ndarray
    labels: np.ndarray


def extract_split(dataset, extractor: FeatureExtractor, *, batch_size: int = BATCH_SIZE) -> SplitFeatures:
    """用 extractor 提取 dataset 的图像特征及标签，每批最多 batch_size 张。"""
    if not len(dataset):
        raise ValueError("特征提取数据不能为空")
    labels = np.empty(len(dataset), dtype=np.int64)

    def read_batch(indices: range):
        """读取 indices 内的图像，并记录对应标签。"""
        for index in indices:
            sample = dataset[index]
            labels[index] = LABELS[sample["label"]]
            yield sample["image"]

    print(f"{dataset.split}: 提取 {len(dataset)} 张图的特征", flush=True)
    features = np.concatenate([
        extractor.extract(read_batch(range(start, min(start + batch_size, len(dataset)))))
        for start in range(0, len(dataset), batch_size)
    ])
    return SplitFeatures(features, labels)


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """用 val 的 labels、probabilities 选阈值：PASS 召回率优先，再比 FAIL 召回率。"""
    fpr, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    feasible = np.flatnonzero(tpr >= MINIMUM_FAIL_RECALL)
    best = min(feasible, key=lambda i: (fpr[i], -tpr[i]))
    upper = thresholds[best]
    if best + 1 == len(thresholds):
        return 0.0
    lower = thresholds[best + 1]
    # 取相邻得分的间隔中点，减少单图与批量浮点舍入对阈值边界的影响。
    return float(max(np.nextafter(lower, upper), lower + (upper - lower) / 2))


def sample_counts(labels: np.ndarray) -> dict:
    """统计实际读取的 labels 总数及各标签数量。"""
    return {
        "samples": len(labels),
        "label_counts": {label: int(np.count_nonzero(labels == value))
                         for label, value in LABELS.items()},
    }


def evaluate(labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict:
    """计算真值 labels 相对于 FAIL 概率 probabilities 和阈值预测 predictions 的指标。"""
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=LABELS["FAIL"], zero_division=0,
    )
    return {
        **sample_counts(labels),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_order": {"rows_true": CONFUSION_ORDER, "columns_predicted": CONFUSION_ORDER},
        "accuracy": float(matrix.trace() / matrix.sum()),
        "fail_precision": float(precision),
        "fail_recall": float(recall),
        "fail_f1": float(f1),
        "pass_recall": float(matrix[0, 0] / matrix[0].sum()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def _training_parts() -> tuple[ProjectDataset, ...]:
    """按顺序读取公用训练分区和完整的审核本地训练分区。"""
    parts = (
        ProjectDataset(DATA_ROOT, ANNOTATIONS, "train"),
        supplemental_dataset("all"),
        indoor_dataset("train"),
        ProjectDataset(SUPPLEMENTAL_ROOT / "local_training", LOCAL_ANNOTATIONS, "train"),
    )
    if not all(map(len, parts)):
        raise ValueError("每个训练来源都必须包含图像")
    return parts


def _single_image_scores(head: LinearHead, split: SplitFeatures) -> np.ndarray:
    """使用 head 逐图计算 split 的 FAIL 概率，与部署的线性头计算一致。"""
    return np.array([head.probabilities(row[None, :])[0] for row in split.features])


def _check_partitions(train: TrainingDataset, val: ProjectDataset) -> None:
    """检查 train、val 与 TACO test、冻结室内留出的图像 ID 和场景组互不重叠。"""
    if file_hash(INDOOR_HOLDOUT) != INDOOR_HOLDOUT_SHA256.read_text(encoding="utf-8").strip():
        raise ValueError("室内留出清单与冻结哈希不一致")
    holdout = json.loads(INDOOR_HOLDOUT.read_text(encoding="utf-8"))
    taco_test = ProjectDataset(DATA_ROOT, ANNOTATIONS, "test")
    seen_ids, seen_groups = set(), set()
    for records, id_key in ((train.images, "id"), (val.images, "id"),
                            (taco_test.images, "id"), (holdout, "image_id")):
        ids = {record[id_key] for record in records}
        groups = {record["group_id"] for record in records}
        if len(ids) != len(records) or seen_ids & ids or seen_groups & groups:
            raise ValueError("训练、验证或留出分区的图像 ID 或场景组冲突")
        seen_ids.update(ids)
        seen_groups.update(groups)


def _fit_head(train: SplitFeatures, *, c: float, class_weight: str | None) -> LinearHead:
    """用 train 拟合 StandardScaler 与 L2 头；c 控制正则强度，class_weight 指定类别权重。"""
    scaler = StandardScaler(copy=False)
    train_features = scaler.fit_transform(train.features.astype(np.float64))
    classifier = LogisticRegression(
        C=c, class_weight=class_weight, solver=SOLVER,
        max_iter=MAX_ITER, random_state=SEED,
    )
    classifier.fit(train_features, train.labels)
    return LinearHead(torch.from_numpy(scaler.mean_), torch.from_numpy(scaler.scale_),
                      torch.from_numpy(classifier.coef_[0]), float(classifier.intercept_[0]))


def main(argv=None):
    """用 argv 指定的命令行参数复训默认配置；省略 argv 时读取进程参数。"""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.parse_args(argv)
    train_dataset = TrainingDataset(*_training_parts())
    val_dataset = ProjectDataset(DATA_ROOT, ANNOTATIONS, "val")
    _check_partitions(train_dataset, val_dataset)
    torch.manual_seed(SEED)
    device = compute_device()
    print(f"baseline2: {WEIGHTS.name} | {device} | {INPUT_SIZE}×{INPUT_SIZE} | "
          f"C={REGULARIZATION_C}, class_weight={CLASS_WEIGHT!r}", flush=True)
    extractor = FeatureExtractor.pretrained(device, input_size=INPUT_SIZE)
    train = extract_split(train_dataset, extractor)
    # 验证集按部署时的单图批大小提取，避免批大小引起的浮点差异。
    val = extract_split(val_dataset, extractor, batch_size=1)
    head = _fit_head(train, c=REGULARIZATION_C, class_weight=CLASS_WEIGHT)
    val_probabilities = _single_image_scores(head, val)
    threshold = choose_threshold(val.labels, val_probabilities)
    val_predictions = (val_probabilities >= threshold).astype(np.int64)
    print(f"baseline2: val 选定阈值={threshold!r}", flush=True)
    print(json.dumps({
        "train": {**sample_counts(train.labels),
                  "sources": dict(Counter(record.get("source", "taco") for record in train_dataset.images))},
        "val": evaluate(val.labels, val_probabilities, val_predictions),
    }, ensure_ascii=False, indent=2, allow_nan=False), flush=True)

    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=TRAINING_ROOT))
    model = CNNLinearModel(extractor, head, threshold, output / "model.pt")
    model.save()
    print(f"baseline2: 模型已保存至 {model.model_path}", flush=True)


if __name__ == "__main__":
    main()
