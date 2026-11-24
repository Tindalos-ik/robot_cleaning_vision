"""独立评估冻结的 YOLO11n Detect 权重，报告检测 AP 和真实部署判定及计数。"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from dataprocessing.taco.images import file_hash

from ...postprocessing import postprocess_detection
from .data import (
    AP_CONF, ARTIFACT_ROOT, CLASS_NAMES, DEPLOY_CONF, IMAGE_SIZE, MAX_DETECTIONS, NMS_IOU,
    DetectionImage, display_image, load_partitions, read_json, stage_dataset,
)
from .model import DetectorModel


def _divide(numerator: int, denominator: int) -> float | None:
    """返回 numerator / denominator；没有分母时指标未定义。"""
    return numerator / denominator if denominator else None


def deployment_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict:
    """用每行一张图的五类 truth、predicted 计数计算判定及计数指标。"""
    real_total, predicted_total = truth.sum(axis=1), predicted.sum(axis=1)
    real_fail, predicted_fail = real_total > 0, predicted_total > 0
    tn, fp, fn, tp = np.bincount(2 * real_fail.astype(np.intp) + predicted_fail, minlength=4).tolist()
    difference = predicted - truth
    classes = {}
    for index, name in enumerate(CLASS_NAMES):
        error = difference[:, index]
        classes[name] = {"true_total": int(truth[:, index].sum()),
                         "predicted_total": int(predicted[:, index].sum()),
                         "mae": float(np.abs(error).mean()),
                         "mean_signed_error": float(error.mean()),
                         "exact_accuracy": float((error == 0).mean())}
    total_error = predicted_total - real_total
    return {"confusion_order": ["PASS", "FAIL"], "confusion_matrix": [[tn, fp], [fn, tp]],
            "accuracy": (tn + tp) / len(truth), "pass_recall": _divide(tn, tn + fp),
            "fail_recall": _divide(tp, tp + fn), "fail_precision": _divide(tp, tp + fp),
            "fail_f1": _divide(2 * tp, 2 * tp + fp + fn),
            "count": {"exact_vector_accuracy": float((difference == 0).all(axis=1).mean()),
                      "exact_total_accuracy": float((total_error == 0).mean()),
                      "total_mae": float(np.abs(total_error).mean()),
                      "total_rmse": float(np.sqrt(np.mean(total_error ** 2))),
                      "total_mean_signed_error": float(total_error.mean()), "by_class": classes}}


def predict_counts(model: DetectorModel, rows: list[DetectionImage]) -> dict:
    """用部署 model 逐图预测 rows，经正式后处理比较五类数量及 PASS/FAIL。"""
    truth = np.empty((len(rows), len(CLASS_NAMES)), dtype=np.int64)
    predicted = np.empty_like(truth)
    for index, item in enumerate(rows):
        rgb, _ = display_image(item)
        frame = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        detections = model.predict_bgr(frame)
        result = postprocess_detection(detections.boxes, detections.class_ids, detections.scores,
                                       detections.transform, threshold=model.threshold)
        for column in range(len(CLASS_NAMES)):
            category = column + 1
            truth[index, column] = item.record["counts"][str(category)]
            predicted[index, column] = result.counts[category]
    return deployment_metrics(truth, predicted)


def _box_metrics(metrics, support: dict[str, int]) -> dict:
    """将 Ultralytics metrics 转为整体和逐类 AP；support 为各类的真值框数。"""
    box = metrics.box
    indices = {c: index for index, c in enumerate(box.ap_class_index)}
    classes = {}
    for category, name in enumerate(CLASS_NAMES):
        index = indices.get(category)
        classes[name] = {"instances": support[name], "ap50": float(box.ap50[index]) if index is not None else None,
                         "ap50_95": float(box.ap[index]) if index is not None else None}
    return {"map50": float(box.map50), "map50_95": float(box.map), "by_class": classes}


def main(argv=None) -> None:
    """从 argv 读取冻结权重和报告位置；省略 argv 时读取命令行。"""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--weights", type=Path, default=ARTIFACT_ROOT / "model.pt", help="冻结的原生五类 checkpoint")
    parser.add_argument("--training-report", type=Path, help="训练命令生成的 training.json，用于核对摘要")
    parser.add_argument("--output", type=Path, required=True, help="写入新评估 JSON 的路径")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu", help="AP 验证设备，例如 0 或 cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    digest = file_hash(args.weights)
    partitions, digests = load_partitions()
    report = {"model": "YOLO11n Detect", "model_id": "baseline3",
              "ultralytics": ultralytics.__version__, "torch": torch.__version__,
              "checkpoint_sha256": digest, "checkpoint_size": args.weights.stat().st_size,
              "class_names": dict(enumerate(CLASS_NAMES)), "manifest_sha256": digests,
              "ap_settings": {"imgsz": IMAGE_SIZE, "conf": AP_CONF, "iou": NMS_IOU, "max_det": MAX_DETECTIONS,
                              "agnostic_nms": False, "batch": 8, "rect": True, "device": args.device},
              "deployment_settings": {"imgsz": IMAGE_SIZE, "conf": DEPLOY_CONF, "iou": NMS_IOU,
                                      "agnostic_nms": False, "max_det": MAX_DETECTIONS, "rect": False}}
    if args.training_report is not None:
        training = read_json(args.training_report)
        if training["checkpoint_sha256"] != digest or training["manifest_sha256"] != digests:
            raise ValueError("当前权重或数据清单摘要与训练记录不符")
        report["training"] = training
    deploy = DetectorModel.load(args.weights)
    ap_model = YOLO(str(args.weights.resolve()), task="detect")
    native = ap_model.model
    report["architecture"] = {"parameters": sum(parameter.numel() for parameter in native.parameters()),
                              "stride": native.stride.int().tolist(), "reg_max": native.model[-1].reg_max}
    splits = ("val", "test", "indoor")
    with TemporaryDirectory(prefix="evaluation-", dir=ARTIFACT_ROOT) as temporary:
        root = Path(temporary)
        data, stats = stage_dataset(root, partitions, splits)
        report["data_audit"] = stats
        report["evaluation"] = {}
        for split in splits:
            metrics = ap_model.val(data=str(data), split=split, project=str(root), name=split, workers=0,
                                   plots=False, save_json=False, **report["ap_settings"])
            row = {"ap": _box_metrics(metrics, stats[split]["instances"]),
                   "deployment": predict_counts(deploy, partitions[split])}
            report["evaluation"][split] = row
            print(split, json.dumps(row, ensure_ascii=False), flush=True)
    if file_hash(args.weights) != digest:
        raise ValueError("冻结权重在评估期间发生变化")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"评估已保存至 {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
