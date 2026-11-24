"""加载五类 YOLO11n Detect 权重，在项目画布上执行一次检测。"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ...capture.camera import BGRFrame
from ...preprocessing import LetterboxTransform, preprocess

if TYPE_CHECKING:
    from ultralytics import YOLO

DEFAULT_MODEL_PATH = Path("artifacts/detector/model.pt")
IMAGE_SIZE = 640
CONFIDENCE = 0.25
NMS_IOU = 0.7
# checkpoint 中的英文名按项目类别 1–5 排列；界面使用项目中文名。
CHECKPOINT_NAMES = ("can", "bottle", "wrapper", "cup", "paper")


@dataclass(frozen=True)
class CanvasDetections:
    """已执行 NMS 的画布 xyxy 框、项目类别 ID、得分及坐标变换。"""

    boxes: NDArray[np.float32]
    class_ids: NDArray[np.int64]
    scores: NDArray[np.float32]
    transform: LetterboxTransform


class DetectorModel:
    """在单次图片或摄像头会话中复用原生 Ultralytics 预测器。"""

    model_id = "baseline3"
    model_name = "YOLO11n Detect"
    threshold = CONFIDENCE

    def __init__(self, yolo: "YOLO", path: Path) -> None:
        """保存已经验证的 YOLO 实例 yolo 和本地权重路径 path。"""
        self._yolo = yolo
        self.model_path = path

    @classmethod
    def load(cls, path: Path | None = None) -> "DetectorModel":
        """加载 path 指定的本地原生 .pt；省略时仅加载默认部署权重。"""
        path = (DEFAULT_MODEL_PATH if path is None else Path(path)).resolve()
        if path.suffix.lower() != ".pt":
            raise ValueError(f"baseline3 需要原生 .pt 权重：{path}")
        if not path.is_file():
            raise FileNotFoundError(f"找不到 baseline3 权重：{path}")

        from ultralytics import YOLO
        from ultralytics.nn.modules import Detect
        from ultralytics.nn.tasks import DetectionModel

        yolo = YOLO(str(path))
        native = yolo.model
        if (
            yolo.task != "detect"
            or type(native) is not DetectionModel
            or type(native.model[-1]) is not Detect
            or Path(native.yaml.get("yaml_file", "")).stem != "yolo11n"
            or native.yaml.get("scale") != "n"
        ):
            raise ValueError(f"baseline3 需要 YOLO11n Detect 原生权重：{path}")
        expected = dict(enumerate(CHECKPOINT_NAMES))
        num_classes = native.model[-1].nc
        if num_classes != len(expected):
            raise ValueError(f"baseline3 权重需要五类，实际为 {num_classes} 类：{path}")
        if native.names != expected:
            raise ValueError(f"baseline3 权重类别映射须为 {expected}，实际为 {native.names}：{path}")
        return cls(yolo, path)

    def predict_bgr(self, frame: BGRFrame) -> CanvasDetections:
        """将原始 BGR 帧 frame 缩放一次，并返回对应 640×640 画布上的检测。"""
        canvas, transform = preprocess(frame, (IMAGE_SIZE, IMAGE_SIZE), color="BGR", fill=114)
        # 输入已为正方形模型画布，Ultralytics 负责通道转换、归一化、解码和类别内 NMS。
        (result,) = self._yolo.predict(
            canvas, imgsz=IMAGE_SIZE, conf=CONFIDENCE, iou=NMS_IOU,
            rect=False, agnostic_nms=False, save=False, verbose=False,
        )
        data = result.boxes.data.cpu().numpy()
        # 框裁剪至有效图像区域，丢弃仅位于填充带的框；回放使用同一组画布坐标。
        left, top, right, bottom = transform.padding
        boxes = np.clip(
            data[:, :4], (left, top, left, top),
            (IMAGE_SIZE - right, IMAGE_SIZE - bottom, IMAGE_SIZE - right, IMAGE_SIZE - bottom),
            dtype=np.float32,
        )
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not valid.all():
            boxes, data = boxes[valid], data[valid]
        return CanvasDetections(boxes, data[:, 5].astype(np.int64) + 1, data[:, 4], transform)
