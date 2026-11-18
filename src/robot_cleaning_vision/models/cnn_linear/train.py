"""固定 all-plus-indoor、384 和 C=0.001 训练 baseline2；新权重保存到 training/<时间戳_随机后缀>/model.pt。"""

import argparse
import hashlib
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
    INDOOR_HOLDOUT, INDOOR_HOLDOUT_SHA256, TrainingDataset,
    indoor_dataset, supplemental_dataset,
)

from .model import (
    ARTIFACT_ROOT, LABELS, PREPROCESSING, WEIGHTS,
    CNNLinearModel, FeatureExtractor, LinearHead, compute_device,
)


DATA_ROOT = ARTIFACT_ROOT.parents[1] / "data/raw/taco"
ANNOTATIONS = DATA_ROOT / "annotations_project.json"
TRAINING_ROOT = ARTIFACT_ROOT / "training"
BATCH_SIZE = 32
SEED = 42
MINIMUM_FAIL_RECALL = 0.90
REGULARIZATION_C = 0.001
SOLVER = "lbfgs"
MAX_ITER = 2000
CONFUSION_ORDER = ["PASS", "FAIL"]


@dataclass
class SplitFeatures:
    """保存正式分区的整图特征和数值标签。"""

    features: np.ndarray
    labels: np.ndarray


def extract_split(dataset, extractor: FeatureExtractor) -> SplitFeatures:
    """从 dataset 分批提取真实特征，extractor 为预训练特征提取器。"""
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
        extractor.extract(read_batch(range(start, min(start + BATCH_SIZE, len(dataset)))))
        for start in range(0, len(dataset), BATCH_SIZE)
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
        "fail_precision": float(precision),
        "fail_recall": float(recall),
        "fail_f1": float(f1),
        "pass_recall": float(matrix[0, 0] / matrix[0].sum()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def main(argv=None):
    """解析 argv 并训练固定配置；拒绝额外参数，每次仅在独立目录保存权重。"""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.parse_args(argv)
    with INDOOR_HOLDOUT.open("rb") as stream:
        holdout_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if holdout_hash != INDOOR_HOLDOUT_SHA256.read_text(encoding="utf-8").strip():
        raise ValueError("室内留出清单与冻结哈希不一致")
    torch.manual_seed(SEED)
    device = compute_device()
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"baseline2: {WEIGHTS.name} | {device} ({device_name}) | "
          f"{PREPROCESSING['target_size']} | all-plus-indoor | C={REGULARIZATION_C}", flush=True)
    extractor = FeatureExtractor.pretrained(device)
    train_dataset = TrainingDataset(
        ProjectDataset(DATA_ROOT, ANNOTATIONS, "train"),
        supplemental_dataset("all"),
        indoor_dataset("train"),
    )
    train = extract_split(train_dataset, extractor)
    val = extract_split(ProjectDataset(DATA_ROOT, ANNOTATIONS, "val"), extractor)

    scaler = StandardScaler(copy=False)
    train_features = scaler.fit_transform(train.features.astype(np.float64))
    classifier = LogisticRegression(
        C=REGULARIZATION_C, solver=SOLVER, max_iter=MAX_ITER, random_state=SEED,
    )
    classifier.fit(train_features, train.labels)
    head = LinearHead(torch.from_numpy(scaler.mean_), torch.from_numpy(scaler.scale_),
                      torch.from_numpy(classifier.coef_[0]), float(classifier.intercept_[0]))
    val_probabilities = head.probabilities(val.features)
    threshold = choose_threshold(val.labels, val_probabilities)
    val_predictions = (val_probabilities >= threshold).astype(np.int64)
    print(f"baseline2: val 选定阈值={threshold!r}", flush=True)

    test = extract_split(ProjectDataset(DATA_ROOT, ANNOTATIONS, "test"), extractor)
    test_probabilities = head.probabilities(test.features)
    test_predictions = (test_probabilities >= threshold).astype(np.int64)
    print(json.dumps({
        "train": sample_counts(train.labels),
        "val": evaluate(val.labels, val_probabilities, val_predictions),
        "test": evaluate(test.labels, test_probabilities, test_predictions),
    }, ensure_ascii=False, indent=2, allow_nan=False), flush=True)

    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=TRAINING_ROOT))
    model = CNNLinearModel(extractor, head, threshold, output / "model.pt")
    model.save()
    print(f"baseline2: 模型已保存至 {model.model_path}", flush=True)


if __name__ == "__main__":
    main()
