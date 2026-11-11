"""统一物体标注、五类计数与图像标签，拒绝把未知目标当成负样本。"""

from __future__ import annotations

import math
from typing import Any


def convert_bbox(bbox: Any, width: int, height: int) -> tuple[float, float, float, float]:
    """严格转换 COCO xywh 为显示方向的 xyxy，不裁剪或旋转。

    参数：bbox 为四个有限数值 [x,y,w,h]，单位像素；width/height 为
    已处理 EXIF 方向的图像宽高，单位像素。右/下边界允许等于图像尺寸。
    返回：(xmin,ymin,xmax,ymax) 浮点元组，连续坐标，不采用闭区间加一。
    异常：ValueError 表示格式错误、非有限值、非正宽高或越界。
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox 必须含四个 xywh 数值")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        raise ValueError("bbox 必须为数值，不能是布尔值或字符串")
    x, y, w, h = map(float, bbox)
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        raise ValueError("bbox 含非有限值")
    if w <= 0 or h <= 0:
        raise ValueError("bbox 宽高必须大于零")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"bbox 超出 {width}x{height} 图像范围")
    return x, y, x + w, y + h


