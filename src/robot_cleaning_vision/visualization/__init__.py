"""仅绘制原图和后处理结果，返回独立的 BGR 标注图。"""

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robot_cleaning_vision.postprocessing import CATEGORY_NAMES, ImageResult


# RGB；类别 ID 1–5 的颜色固定，不随目标顺序或帧改变。
CATEGORY_COLORS = (
    (255, 170, 0),
    (0, 170, 255),
    (180, 100, 255),
    (0, 200, 160),
    (255, 100, 170),
)
_PASS = (0, 190, 80)
_FAIL = (240, 60, 60)
_TEXT = (235, 235, 235)
_BACKGROUND = (28, 30, 34)
_STATUS = {"completed": "已完成", "pending": "待处理", "error": "错误"}


class ResultRenderer:
    """初始化时加载调用方明确指定的中文字体；draw 不执行任何 I/O。"""

    def __init__(self, font_path: str | Path, *, font_size: int = 20) -> None:
        if font_size <= 0:
            raise ValueError("font_size 必须为正整数")
        self._font = ImageFont.truetype(str(font_path), font_size)
        self._line_height = sum(self._font.getmetrics()) + 8
        self._padding = 16
        self._sidebar_width = max(340, font_size * 16)

    def _wrap(self, text: str) -> list[str]:
        """按字体实际宽度换行，长模型名和错误信息不侵入原图。"""
        lines: list[str] = []
        line = ""
        width = self._sidebar_width - 2 * self._padding
        for char in text:
            if char == "\n":
                lines.append(line)
                line = ""
                continue
            if line and self._font.getlength(line + char) > width:
                lines.append(line)
                line = ""
            line += char
        lines.append(line)
        return lines

    def draw(
        self,
        frame: np.ndarray,
        result: ImageResult | None = None,
        *,
        model_name: str,
        status: Literal["completed", "pending", "error"] = "completed",
        elapsed_ms: float | None = None,
        error_message: str | None = None,
    ) -> np.ndarray:
        """输入 uint8 HWC BGR 原图；耗时为调用方最近完成帧的毫秒数。

        completed 必须提供后处理结果。pending/error 忽略 result，不绘制旧框、
        计数、概率或判定。输出保留原图尺寸与左上角位置，右侧追加信息栏；
        原图低于文字高度时仅向下补足画布。
        """
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or min(frame.shape[:2]) == 0
        ):
            raise ValueError("frame 必须为非空 HWC BGR uint8 数组")
        if status not in _STATUS:
            raise ValueError("status 必须为 completed、pending 或 error")
        if status == "completed" and result is None:
            raise ValueError("已完成状态必须提供后处理结果")
        if elapsed_ms is not None and (
            not np.isfinite(elapsed_ms) or elapsed_ms < 0
        ):
            raise ValueError("elapsed_ms 必须为非负有限毫秒数或 None")

        rows: list[tuple[str, tuple[int, int, int]]] = []

        def add(text: str, color: tuple[int, int, int] = _TEXT) -> None:
            rows.extend((line, color) for line in self._wrap(text))

        add(f"模型：{model_name}")
        add(f"状态：{_STATUS[status]}", _FAIL if status == "error" else _TEXT)
        add("最近完成帧耗时：" + (
            "—" if elapsed_ms is None else f"{elapsed_ms:.1f} ms"
        ))
        active = result if status == "completed" else None
        if active is not None:
            add(f"判定：{active.decision}", _PASS if active.decision == "PASS" else _FAIL)
            if active.objects is None:
                add(f"FAIL 概率：{active.image_score:.1%}")
            else:
                for class_id, name in enumerate(CATEGORY_NAMES, start=1):
                    add(f"{name}：{active.counts[class_id]}", CATEGORY_COLORS[class_id - 1])
        elif status == "error" and error_message:
            add(f"错误信息：{error_message}", _FAIL)

        height, width = frame.shape[:2]
        image = Image.fromarray(frame[:, :, ::-1])
        if active is not None and active.objects is not None:
            # 在原图大小的独立图层绘制，越界部分由画布自然裁切；不变换坐标。
            draw = ImageDraw.Draw(image)
            for obj in active.objects:
                color = CATEGORY_COLORS[obj.class_id - 1]
                draw.rectangle(obj.bbox, outline=color, width=2)
                label = f"{CATEGORY_NAMES[obj.class_id - 1]} {obj.score:.1%}"
                left, top, right, bottom = draw.textbbox((0, 0), label, font=self._font)
                label_width = right - left + 8
                label_height = bottom - top + 8
                x = max(0, min(round(obj.bbox[0]), width - label_width))
                y = max(0, min(round(obj.bbox[1]) - label_height, height - label_height))
                draw.rectangle(
                    (x, y, x + label_width - 1, y + label_height - 1), fill=color,
                )
                draw.text(
                    (x + 4 - left, y + 4 - top), label,
                    font=self._font, fill=(0, 0, 0),
                )

        canvas = Image.new(
            "RGB",
            (
                width + self._sidebar_width,
                max(height, 2 * self._padding + len(rows) * self._line_height),
            ),
            _BACKGROUND,
        )
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        for index, (text, color) in enumerate(rows):
            draw.text(
                (width + self._padding, self._padding + index * self._line_height),
                text, font=self._font, fill=color,
            )
        return np.array(canvas)[:, :, ::-1].copy()
