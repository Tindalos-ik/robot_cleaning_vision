"""保存原始 PNG 与可追加的场景元数据，不生成标注。"""

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .camera import BGRFrame

CSV_FIELDS = (
    "image_id", "scene_id", "file_name", "captured_at", "width", "height", "camera_source",
)


class SaveError(RuntimeError):
    """图片编码、文件写入或 CSV 追加失败。"""


@dataclass(frozen=True)
class ImageRecord:
    """成功保存的元数据。

    image_id 为唯一 PNG 文件名；scene_id 为物理场景名；file_name 为相对输出
    根目录的 POSIX 路径；captured_at 为含时区的 ISO 8601 时间字符串；
    width、height 为实际图像宽、高（像素）；camera_source 为来源的字符串标识。
    """

    image_id: str
    scene_id: str
    file_name: str
    captured_at: str
    width: int
    height: int
    camera_source: str


class ImageStore:
    """在输出根目录保存场景图片和 images.csv；用于单线程采集，不支持多进程共写。"""

    def __init__(self, output: str | Path, scene_id: str, camera_source: int | str) -> None:
        """准备一个物理场景的输出目录，不改写已有图片或 CSV。

        参数：output 为 str/Path 输出根目录，相对路径以进程工作目录为基准；
        scene_id 为 str 单级目录名，同一物理布置重复使用；camera_source 为 int
        设备索引或 str 视频流地址/文件路径，仅以字符串记录来源。
        返回：None。异常：scene_id 为空、为 . / .. 或含路径非法字符时抛出
        ValueError；目录无法创建时抛出 SaveError。
        """
        if (
            not scene_id.strip() or scene_id in {".", ".."}
            or any(char in '<>:"/\\|?*' or ord(char) < 32 for char in scene_id)
            or scene_id.endswith((".", " "))
        ):
            raise ValueError("scene_id 必须是非空的单级目录名，不能包含路径分隔符或非法字符。")
        self.output_dir = Path(output).resolve()
        self.scene_id = scene_id
        self.camera_source = str(camera_source)
        self.scene_dir = self.output_dir / scene_id
        self.csv_path = self.output_dir / "images.csv"
        try:
            self.scene_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SaveError(f"无法创建采集目录 {self.scene_dir}：{error}") from error

    def save(self, frame: BGRFrame, *, captured_at: datetime | None = None) -> ImageRecord:
        """保存当前原始帧，成功后追加一行 CSV；不在图片上绘制任何内容。

        参数：frame 为 BGR numpy.ndarray，shape=(height, width, 3)，dtype=uint8，
        宽、高为像素；不复制、不修改输入。captured_at 为带时区 datetime，表示
        该帧读取完成时间，默认 None 使用当前本地带时区时间。
        返回：ImageRecord，file_name 相对 output_dir，尺寸来自实际保存帧。
        异常：图像或时间格式错误抛出 ValueError；编码、写入、CSV 格式或追加
        失败抛出 SaveError。CSV 失败时删除本次图片，避免报告半成功。
        """
        if (
            not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
            or frame.size == 0 or frame.dtype != np.uint8
        ):
            raise ValueError("保存图像必须为非空 H×W×3、uint8 的 BGR 数组。")
        timestamp = captured_at if captured_at is not None else datetime.now().astimezone()
        if timestamp.utcoffset() is None:
            raise ValueError("captured_at 必须包含时区。")
        # 时间便于人工检索，UUID 避免跨次运行重名；xb 再保证绝不覆盖已有文件。
        image_id = f"{timestamp:%Y%m%dT%H%M%S_%f}_{uuid4().hex}.png"
        image_path = self.scene_dir / image_id
        record = ImageRecord(
            image_id=image_id, scene_id=self.scene_id,
            file_name=(Path(self.scene_id) / image_id).as_posix(),
            captured_at=timestamp.isoformat(), width=frame.shape[1], height=frame.shape[0],
            camera_source=self.camera_source,
        )
        created = False
        try:
            success, encoded = cv2.imencode(".png", frame)
            if not success:
                raise SaveError("OpenCV 无法将当前 BGR 帧编码为 PNG。")
            # Python 文件接口支持中文路径；无损 PNG 保留采集像素与实际分辨率。
            with image_path.open("xb") as image_file:
                created = True
                image_file.write(memoryview(encoded))
            self._append_record(record)
        except BaseException as error:
            if created:
                try:
                    image_path.unlink()
                except OSError as cleanup_error:
                    raise SaveError(
                        f"保存失败：{error}；清理未登记图片 {image_path} 也失败：{cleanup_error}"
                    ) from error
            if isinstance(error, (OSError, cv2.error, csv.Error)):
                raise SaveError(f"保存图片或元数据失败（{image_path}）：{error}") from error
            raise
        return record

    def _append_record(self, record: ImageRecord) -> None:
        """在图片写入完成后追加 CSV，保留原有记录且只在空文件写表头。

        参数：record 为本次 ImageRecord，file_name 相对输出根目录。
        返回：None。异常：表头不匹配抛出 SaveError；文件/CSV 异常原样传播，
        由 save() 统一报告。写入失败时截回追加前的位置，避免半行污染旧记录；
        不处理进程强杀、断电或多进程同时追加。
        """
        with self.csv_path.open("a+", encoding="utf-8", newline="") as csv_file:
            csv_file.seek(0)
            header = next(csv.reader(csv_file), None)
            if header is not None and header != list(CSV_FIELDS):
                raise SaveError(f"CSV 表头不匹配，未改写已有记录：{self.csv_path}")
            csv_file.seek(0, 2)
            original_end = csv_file.tell()
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            try:
                if original_end == 0:
                    writer.writeheader()
                writer.writerow(asdict(record))
                csv_file.flush()
            except BaseException:
                csv_file.seek(original_end)
                csv_file.truncate()
                raise
