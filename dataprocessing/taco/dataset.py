"""为三类模型提供同一份正式分区、RGB 图像与未经旋转的目标坐标。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .images import source_path


class ProjectDataset:
    """读取完整 COCO 产物，但只向消费者暴露显式指定的正式分区。"""

    def __init__(self, root: Path, annotations: Path, split: str) -> None:
        """构造指定正式分区的消费者。

        参数：
            root：只读原图根目录；annotations：完整 COCO JSON 文件路径。
            split：正式分区名，只接受 train、val、test。
        返回：
            None；仅加载标注，图像在索引访问时读取。
        异常：
            ValueError：分区名非法、JSON 无效或框结构无效。
            OSError：标注文件不可读取。
            KeyError：COCO 缺少消费者所需字段。
        """
        if split not in {"train", "val", "test"}:
            raise ValueError("split 必须显式指定为 train、val 或 test")
        self.root = Path(root)
        self.split = split
        data = json.loads(Path(annotations).read_text(encoding="utf-8"))
        self.images = [image for image in data["images"] if image["split"] == split]
        self.objects: dict[int, list[dict[str, Any]]] = defaultdict(list)
        image_ids = {image["id"] for image in self.images}
        for annotation in data["annotations"]:
            if annotation["image_id"] not in image_ids:
                continue
            obj = dict(annotation)
            x, y, width, height = obj["bbox"]
            obj.update(xmin=x, ymin=y, xmax=x + width, ymax=y + height)
            self.objects[obj["image_id"]].append(obj)

    def __len__(self) -> int:
        """返回指定正式分区中的图像数量，不含留出图像。"""
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """按整数索引返回图像及真值。

        参数：
            index：指定分区内的整数索引，支持 Python 列表的负索引。
        返回：
            含 PIL RGB 图像、image_id、label、counts、objects 的字典；
            仅像素应用 EXIF，bbox 和 xmin、ymin、xmax、ymax 均保持官方显示坐标。
        异常：
            IndexError：索引越界。
            OSError：原图不可读取或不能解码。
            ValueError：图像路径越出根目录，或 EXIF 转正后的尺寸不匹配标注。
        """
        record = self.images[index]
        with Image.open(source_path(self.root, record["file_name"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if image.size != (record["width"], record["height"]):
            raise ValueError(f"图像 {record['id']} 的显示尺寸与标注不匹配")
        return {
            "image": image,
            "image_id": record["id"],
            "label": record["label"],
            "counts": dict(record["counts"]),
            "objects": [
                {**obj, "bbox": list(obj["bbox"])}
                for obj in self.objects[record["id"]]
            ],
        }
