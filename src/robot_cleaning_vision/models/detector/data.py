"""校验项目五类框和正式分区，生成临时的 Ultralytics 检测数据集。"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps

from dataprocessing.supplemental.datasets import (
    INDOOR_ANNOTATIONS, INDOOR_HOLDOUT, INDOOR_HOLDOUT_SHA256,
    SUPPLEMENTAL_ANNOTATIONS, SUPPLEMENTAL_ROOT,
)
from dataprocessing.taco.annotations import convert_bbox
from dataprocessing.taco.images import file_hash, source_path

ARTIFACT_ROOT = Path(__file__).resolve().parents[4] / "artifacts/detector"
TACO_ROOT = ARTIFACT_ROOT.parents[1] / "data/raw/taco"
CLASS_NAMES = ("can", "bottle", "wrapper", "cup", "paper")
PROJECT_NAMES = ("罐", "瓶", "包装物", "杯", "纸类物品")
IMAGE_SIZE = 640
DEPLOY_CONF = 0.25
NMS_IOU = 0.7
AP_CONF = 0.001
MAX_DETECTIONS = 300
_LOCAL_ANNOTATIONS = SUPPLEMENTAL_ANNOTATIONS.with_name("local_training_annotations_project.json")
_SOURCES = (
    (TACO_ROOT, TACO_ROOT / "annotations_project.json", {"train": 621, "val": 134, "test": 134}),
    (SUPPLEMENTAL_ROOT, SUPPLEMENTAL_ANNOTATIONS, {"train": 131}),
    (SUPPLEMENTAL_ROOT, INDOOR_ANNOTATIONS, {"train": 52, "test": 52}),
    (SUPPLEMENTAL_ROOT / "local_training", _LOCAL_ANNOTATIONS, {"train": 195}),
)


@dataclass(frozen=True)
class DetectionImage:
    """保存单图的原始路径、正式分区、图像记录和已校验的 YOLO 标签文本。"""

    path: Path
    split: str
    record: dict
    labels: str
    difficult: int


def read_json(path: Path):
    """读取 path 指定的 UTF-8 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def yolo_labels(record: dict, boxes: list[dict]) -> tuple[str, int]:
    """用 record 的显示宽高转换 boxes，返回 YOLO 标签文本及保留的 difficult 框数。"""
    width, height = record["width"], record["height"]
    if width <= 0 or height <= 0:
        raise ValueError(f"图像尺寸无效：{record['file_name']}")
    counts = dict.fromkeys(map(str, range(1, 6)), 0)
    rows = set()
    difficult = 0
    for box in boxes:
        category = box["category_id"]
        if type(category) is not int or not 1 <= category <= len(CLASS_NAMES):
            raise ValueError(f"项目目标类别无效：{box['id']}")
        if box["iscrowd"] or box.get("ignore", 0):
            raise ValueError(f"正式分区含 crowd/ignore 目标：{box['id']}")
        x1, y1, x2, y2 = convert_bbox(box["bbox"], width, height)
        xywh = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                (x2 - x1) / width, (y2 - y1) / height)
        # Ultralytics 以 float32 读取坐标，预先拒绝会被它合并的重复框。
        row = (category - 1, *map(float, np.float32(xywh)))
        if row in rows:
            raise ValueError(f"重复目标框：{box['id']}")
        rows.add(row)
        counts[str(category)] += 1
        difficult += bool(box.get("difficult", False))
    if counts != record["counts"] or record["label"] != ("FAIL" if rows else "PASS"):
        raise ValueError(f"目标框与图像计数或标签不符：{record['file_name']}")
    return "".join(f"{c} {x:.9g} {y:.9g} {w:.9g} {h:.9g}\n"
                   for c, x, y, w, h in sorted(rows)), difficult


def load_partitions() -> tuple[dict[str, list[DetectionImage]], dict[str, str]]:
    """读取四份正式标注，校验分区、路径、场景组、框、冻结清单，返回样本及清单摘要。"""
    holdout_digest = file_hash(INDOOR_HOLDOUT)
    if holdout_digest != INDOOR_HOLDOUT_SHA256.read_text(encoding="ascii").strip():
        raise ValueError("Indoor holdout 清单的 SHA-256 与冻结摘要不符")
    frozen = read_json(INDOOR_HOLDOUT)
    partitions = {split: [] for split in ("train", "val", "test", "indoor")}
    digests = {"indoor_holdout": holdout_digest}
    image_ids, annotation_ids, paths = set(), set(), set()
    groups = {}
    for root, manifest, expected in _SOURCES:
        data = read_json(manifest)
        digests[manifest.relative_to(ARTIFACT_ROOT.parents[1]).as_posix()] = file_hash(manifest)
        categories = {row["id"]: row["name"] for row in data["categories"]}
        if len(data["categories"]) != len(PROJECT_NAMES) or categories != dict(enumerate(PROJECT_NAMES, 1)):
            raise ValueError(f"项目类别 ID/名称不符：{manifest}：{categories}")
        images = data["images"]
        counts = Counter(row["split"] for row in images)
        if counts != expected:
            raise ValueError(f"正式分区数量不符：{manifest}：{counts}，应为 {expected}")
        if manifest == SUPPLEMENTAL_ANNOTATIONS:
            sources = Counter(row["source"] for row in images)
            if sources != {"coxpwd": 110, "wikimedia": 21}:
                raise ValueError(f"补充来源数量不符：{sources}")
        manifest_ids = {row["id"] for row in images}
        objects = defaultdict(list)
        for box in data["annotations"]:
            if box["image_id"] not in manifest_ids or box["id"] in annotation_ids:
                raise ValueError(f"标注引用缺失或 ID 重复：{manifest}：{box['id']}")
            annotation_ids.add(box["id"])
            objects[box["image_id"]].append(box)
        for row in images:
            split = row["split"]
            if manifest == INDOOR_ANNOTATIONS and split == "test":
                split = "indoor"
            path = source_path(root, row["file_name"])
            path_key = path.as_posix().casefold()
            if row["id"] in image_ids or path_key in paths:
                raise ValueError(f"图像 ID 或路径重复：{row['id']}：{path}")
            image_ids.add(row["id"])
            paths.add(path_key)
            if groups.setdefault(row["group_id"], split) != split:
                raise ValueError(f"场景组跨分区：{row['group_id']}")
            labels, difficult = yolo_labels(row, objects[row["id"]])
            partitions[split].append(DetectionImage(path, split, row, labels, difficult))
    keys = ("file_name", "sha256", "group_id", "label", "counts", "source_categories", "background", "review_note")
    indoor = [{"image_id": item.record["id"], "original_id": item.record["source_image_id"],
               **{key: item.record[key] for key in keys}} for item in partitions["indoor"]]
    if indoor != frozen:
        raise ValueError("Indoor test 与冻结留出清单不一致")
    return partitions, digests


def display_image(item: DetectionImage) -> tuple[Image.Image, int]:
    """读 item 原图，返回 EXIF 转正后的 RGB 像素与原 EXIF 方向，坐标不旋转。"""
    with Image.open(item.path) as source:
        orientation = source.getexif().get(274, 1)
        ImageOps.exif_transpose(source, in_place=True)
        image = source.convert("RGB")
    if image.size != (item.record["width"], item.record["height"]):
        raise ValueError(f"EXIF 转正后的尺寸与显示坐标不符：{item.path}：{image.size}")
    return image, orientation


def stage_dataset(root: Path, partitions: dict[str, list[DetectionImage]],
                  splits: tuple[str, ...]) -> tuple[Path, dict]:
    """在临时 root 只导出 splits；审计 partitions 所有正式原件哈希、显示像素和分区。"""
    for split in splits:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    def stage(item: DetectionImage) -> tuple[str, str, int]:
        """校验 item 原图及显示尺寸，按需写副本，返回文件/像素哈希及 EXIF 方向。"""
        digest = file_hash(item.path)
        if "sha256" in item.record and digest != item.record["sha256"]:
            raise ValueError(f"原图 SHA-256 与审核清单不符：{item.path}")
        if "source_annotation_sha256" in item.record:
            annotation = source_path(SUPPLEMENTAL_ROOT / "local_training", item.record["source_annotation_path"])
            if file_hash(annotation) != item.record["source_annotation_sha256"]:
                raise ValueError(f"本地原始标注 SHA-256 与审核清单不符：{annotation}")
        image, orientation = display_image(item)
        rgb = np.asarray(image)
        pixel_hash = hashlib.sha256(f"{image.size}:".encode())
        pixel_hash.update(rgb)
        if item.split in splits:
            width, height = image.size
            ratio = IMAGE_SIZE / max(width, height)
            resized = (min(math.ceil(width * ratio), IMAGE_SIZE), min(math.ceil(height * ratio), IMAGE_SIZE))
            if min(resized) < 10:
                raise ValueError(f"图像缩放后任一边小于 Ultralytics 的 10 px 限制：{item.path}")
            if resized != image.size:
                rgb = cv2.resize(rgb, resized, interpolation=cv2.INTER_LINEAR)
            stem = str(item.record["id"])
            Image.fromarray(rgb).save(root / "images" / item.split / f"{stem}.png", compress_level=1)
            (root / "labels" / item.split / f"{stem}.txt").write_text(item.labels, encoding="ascii")
        return digest, pixel_hash.hexdigest(), orientation

    items = [item for partition in partitions.values() for item in partition]
    stats = {split: {"images": len(rows), "labels": dict(Counter(item.record["label"] for item in rows)),
                     "instances": dict(zip(CLASS_NAMES, (sum(item.record["counts"][str(i)] for item in rows)
                                                        for i in range(1, 6)))),
                     "difficult_boxes": sum(item.difficult for item in rows), "exif_transformed": 0}
             for split, rows in partitions.items()}
    file_splits, pixel_splits = {}, {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item, (digest, pixels, orientation) in zip(items, pool.map(stage, items), strict=True):
            for seen, key in ((file_splits, digest), (pixel_splits, pixels)):
                first = seen.setdefault(key, item)
                if first.split != item.split:
                    raise ValueError(f"文件或显示像素跨分区重复：{first.path} / {item.path}")
            stats[item.split]["exif_transformed"] += orientation in range(2, 9)
    dataset = root / "dataset.yaml"
    # Ultralytics 不解析自定义 indoor 键的相对路径；独立评估留空 train。
    paths = dict.fromkeys(("train", "val"))
    paths.update({split: (root / "images" / split).resolve().as_posix() for split in splits})
    dataset.write_text(yaml.safe_dump({"path": root.resolve().as_posix(), **paths,
                                       "names": dict(enumerate(CLASS_NAMES))}, sort_keys=False), encoding="utf-8")
    return dataset, stats
