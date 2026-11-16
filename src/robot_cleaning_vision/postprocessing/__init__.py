"""预测筛选、坐标还原、计数与图像判定。"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from robot_cleaning_vision.preprocessing import LetterboxTransform


# 与数据整理的项目类别 ID 一致，不使用检测器特有的背景类或零起始 ID。
CATEGORY_NAMES = ("罐", "瓶", "包装物", "杯", "纸类物品")


@dataclass
class Detection:
    """原图像素坐标下的单个目标，bbox 为连续 xyxy 坐标。"""

    bbox: tuple[float, float, float, float]
    class_id: int
    score: float


@dataclass
class ImageResult:
    """分类的 objects/counts 为 None；检测的 image_score 为 None。"""

    decision: Literal["PASS", "FAIL"]
    image_score: float | None
    objects: list[Detection] | None
    counts: dict[int, int] | None


def _validate_probability(value: float, name: str) -> None:
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} 必须为 [0, 1] 内的有限数值")


def postprocess_classification(
    fail_probability: float, *, threshold: float,
) -> ImageResult:
    """FAIL 概率达到 threshold 时判为 FAIL，不推断目标或计数。"""
    _validate_probability(threshold, "threshold")
    _validate_probability(fail_probability, "fail_probability")
    return ImageResult(
        decision="FAIL" if fail_probability >= threshold else "PASS",
        image_score=float(fail_probability),
        objects=None,
        counts=None,
    )


def postprocess_detection(
    boxes: ArrayLike,
    class_ids: ArrayLike,
    scores: ArrayLike,
    transform: LetterboxTransform,
    *,
    threshold: float,
) -> ImageResult:
    """筛选已解码、已 NMS 的检测结果，复用预处理变换还原坐标。

    boxes 为预处理图像上的 (N, 4) xyxy 像素坐标，class_ids/scores 为
    (N,)，类别 ID 为 1–5。保留得分 >= threshold 的目标及其输入顺序；
    不修改输入，不裁剪坐标，不再执行 NMS。
    """
    _validate_probability(threshold, "threshold")
    boxes = np.asarray(boxes, dtype=np.float64)
    class_ids = np.asarray(class_ids)
    scores = np.asarray(scores, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes 必须为 (N, 4) 的 xyxy 坐标")
    size = len(boxes)
    if class_ids.shape != (size,) or scores.shape != (size,):
        raise ValueError("class_ids 和 scores 必须为与 boxes 等长的 (N,) 数组")
    if not np.all(np.isfinite(boxes)) or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("boxes 必须为有限数值且具有正宽高")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores 必须为 [0, 1] 内的有限数值")
    if size and (
        not np.issubdtype(class_ids.dtype, np.integer)
        or np.any((class_ids < 1) | (class_ids > len(CATEGORY_NAMES)))
    ):
        raise ValueError("class_ids 必须为 1–5 的整数项目类别 ID")

    keep = scores >= threshold
    restored = transform.inverse_boxes(boxes[keep])
    objects = [
        Detection(tuple(map(float, box)), int(class_id), float(score))
        for box, class_id, score in zip(restored, class_ids[keep], scores[keep])
    ]
    counts = dict.fromkeys(range(1, len(CATEGORY_NAMES) + 1), 0)
    for obj in objects:
        counts[obj.class_id] += 1
    return ImageResult(
        decision="FAIL" if objects else "PASS",
        image_score=None,
        objects=objects,
        counts=counts,
    )
