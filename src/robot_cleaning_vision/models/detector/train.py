"""使用 COCO 预训练的 YOLO11n Detect 微调五类检测器，只用 TACO val 选择权重。"""

import argparse
from contextlib import chdir
import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from urllib.request import urlretrieve

import torch
import ultralytics
from ultralytics import YOLO

from dataprocessing.taco.images import file_hash

from .data import AP_CONF, ARTIFACT_ROOT, IMAGE_SIZE, MAX_DETECTIONS, NMS_IOU, load_partitions, stage_dataset

PRETRAINED_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
PRETRAINED_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
TRAINING_ROOT = ARTIFACT_ROOT / "training"
SEED = 42


def main(argv=None) -> None:
    """用 argv 指定设备和批量等训练选项；省略 argv 时读取进程参数。"""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pretrained", type=Path, help="已下载的官方 COCO yolo11n.pt；省略则下载")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu",
                        help="Ultralytics 设备，例如 0 或 cpu")
    parser.add_argument("--batch", type=int, default=8, help="每批图像数，按显存调整")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数上限")
    parser.add_argument("--patience", type=int, default=10, help="TACO val 未刷新最优时的提前停止轮数")
    parser.add_argument("--workers", type=int, default=0, help="数据加载进程数；Windows 默认在主进程加载")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="允许 Ultralytics 检查并启用 CUDA 混合精度")
    args = parser.parse_args(argv)
    partitions, digests = load_partitions()
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=TRAINING_ROOT))
    settings = dict(imgsz=IMAGE_SIZE, conf=AP_CONF, iou=NMS_IOU, agnostic_nms=False,
                    max_det=MAX_DETECTIONS, device=args.device, batch=args.batch, epochs=args.epochs,
                    patience=args.patience, workers=args.workers, seed=SEED, deterministic=True,
                    fraction=1.0, amp=args.amp, cache="disk", val=True, plots=False)
    best = {}

    def record_best(trainer) -> None:
        """从 trainer 记录写入原生 best.pt 的轮次和当轮 TACO val 检测指标。"""
        if trainer.fitness == trainer.best_fitness:
            best.update(epoch=trainer.epoch + 1, map50=float(trainer.metrics["metrics/mAP50(B)"]),
                        map50_95=float(trainer.fitness))

    print(f"baseline3: {run}", flush=True)
    with TemporaryDirectory(prefix="dataset-", dir=run) as temporary:
        root = Path(temporary)
        initial = args.pretrained.resolve() if args.pretrained is not None else root / "yolo11n.pt"
        if args.pretrained is None:
            urlretrieve(PRETRAINED_URL, initial)
        if file_hash(initial) != PRETRAINED_SHA256:
            raise ValueError(f"初始权重 SHA-256 与官方 COCO yolo11n.pt 不符：{initial}")
        data, stats = stage_dataset(root, partitions, ("train", "val"))
        print(json.dumps({name: stats[name] for name in ("train", "val")}, ensure_ascii=False), flush=True)
        model = YOLO(str(initial), task="detect")
        model.add_callback("on_model_save", record_best)
        # Ultralytics 的 AMP 预检可能下载单独的检查模型，文件随该临时目录清理。
        with chdir(root):
            model.train(data=str(data), project=str(run), name="fit", **settings)
    weights = run / "fit/weights/best.pt"
    (run / "fit/weights/last.pt").unlink()
    report = {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
              "device_name": torch.cuda.get_device_name(int(args.device)) if args.device.isdecimal() else args.device,
              "pretrained": {"url": PRETRAINED_URL, "sha256": PRETRAINED_SHA256}, "settings": settings,
              "actual_batch": model.trainer.batch_size, "actual_amp": bool(model.trainer.amp),
              "optimizer": type(model.trainer.optimizer).__name__,
              "manifest_sha256": digests, "data_audit": stats, "best_taco_val": best,
              "epochs_run": model.trainer.epoch + 1, "checkpoint_sha256": file_hash(weights)}
    (run / "training.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"baseline3: {weights}\nTACO val best: {best}", flush=True)


if __name__ == "__main__":
    main()
