"""通用图像变换与坐标变换信息。"""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from PIL import Image


@dataclass(frozen=True)
class LetterboxTransform:
    """尺寸为 (高, 宽)，scale 为 (x, y)，padding 为 (左, 上, 右, 下)。"""

    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    target_size: tuple[int, int]
    scale: tuple[float, float]
    padding: tuple[int, int, int, int]

    def forward_boxes(self, boxes: ArrayLike) -> NDArray[np.float64]:
        """将 (..., 4) 的原图 xyxy 坐标映射到输出图，不修改输入或裁剪坐标。"""
        result = np.array(boxes, dtype=np.float64, copy=True)
        scale_x, scale_y = self.scale
        left, top, _, _ = self.padding
        result *= (scale_x, scale_y, scale_x, scale_y)
        result += (left, top, left, top)
        return result

    def inverse_boxes(self, boxes: ArrayLike) -> NDArray[np.float64]:
        """将 (..., 4) 的输出图 xyxy 坐标还原到原图，不修改输入或裁剪坐标。"""
        result = np.array(boxes, dtype=np.float64, copy=True)
        scale_x, scale_y = self.scale
        left, top, _, _ = self.padding
        result -= (left, top, left, top)
        result /= (scale_x, scale_y, scale_x, scale_y)
        return result


def preprocess(
    image: NDArray[np.uint8] | Image.Image,
    target_size: tuple[int, int],
    *,
    fill: int | tuple[int, int, int] = 0,
    color: Literal["BGR", "RGB"] = "RGB",
    dtype: DTypeLike = np.uint8,
    scale: float = 1.0,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> tuple[NDArray, LetterboxTransform]:
    """返回独立的 HWC 图像与 letterbox 信息，原图保持不变。

    输入为 BGR uint8 数组或 PIL RGB 图像；target_size 为 (高, 宽)。
    使用双线性插值等比例缩放，居中填充，奇数余量放在右侧或下侧。
    fill 为输出颜色顺序的 0–255 像素值，先填充再转换数值类型。
    数值变换为 (像素 * scale - mean) / std，省略的 mean/std 为 0/1；
    启用数值变换时 dtype 必须为浮点类型，统计量按输出通道顺序提供。
    输入格式、目标尺寸、颜色或归一化配置不符合约定时抛出 ValueError。
    """
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            raise ValueError("PIL 输入必须为 RGB 图像")
        pixels = np.asarray(image)
        source_color = "RGB"
    else:
        pixels = np.asarray(image)
        source_color = "BGR"
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("输入必须为形状 (高, 宽, 3) 的 uint8 图像")
    original_h, original_w = pixels.shape[:2]
    target_h, target_w = target_size
    if min(original_h, original_w, target_h, target_w) <= 0:
        raise ValueError("原图和目标尺寸必须为正")
    if color not in {"BGR", "RGB"}:
        raise ValueError("color 必须为 BGR 或 RGB")
    output_dtype = np.dtype(dtype)
    normalize = scale != 1.0 or mean is not None or std is not None
    if normalize and not np.issubdtype(output_dtype, np.floating):
        raise ValueError("数值归一化需要浮点 dtype")
    if std is not None and np.any(np.asarray(std) == 0):
        raise ValueError("std 不能为零")

    ratio = min(target_h / original_h, target_w / original_w)
    resized_h = max(1, round(original_h * ratio))
    resized_w = max(1, round(original_w * ratio))
    if color != source_color:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    if (resized_h, resized_w) != (original_h, original_w):
        pixels = cv2.resize(
            pixels, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR,
        )
    left = (target_w - resized_w) // 2
    top = (target_h - resized_h) // 2
    output = np.empty((target_h, target_w, 3), dtype=np.uint8)
    output[:] = fill
    output[top:top + resized_h, left:left + resized_w] = pixels
    output = output.astype(output_dtype, copy=False)
    if scale != 1.0:
        output *= scale
    if mean is not None:
        output -= np.asarray(mean, dtype=output_dtype)
    if std is not None:
        output /= np.asarray(std, dtype=output_dtype)

    transform = LetterboxTransform(
        original_size=(original_h, original_w),
        resized_size=(resized_h, resized_w),
        target_size=(target_h, target_w),
        scale=(resized_w / original_w, resized_h / original_h),
        padding=(left, top, target_w - resized_w - left, target_h - resized_h - top),
    )
    return output, transform
