"""以微型官方 TACO 输入验证完整流水线及统一消费者的公开行为。"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import Any

from PIL import Image

from . import ProjectDataset
from .pipeline import run_pipeline


class ProcessingTests(unittest.TestCase):
    """只经官方 JSON、流水线产物和数据集接口检查真实消费者行为。"""

    def setUp(self) -> None:
        """建立隔离的原图目录及官方格式标注，不依赖仓库内真实图像。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "raw"
        self.root.mkdir()
        self.output = Path(self.temporary.name) / "processed" / "coco.json"
        images: list[dict[str, Any]] = []
        for image_id in range(1, 17):
            path = self.root / "batch_1" / f"{image_id:03d}.jpg"
            path.parent.mkdir(exist_ok=True)
            exif = Image.Exif()
            exif[271] = "Fixture Camera"
            exif[272] = "Synthetic"
            session_id = 5 if image_id == 6 else image_id
            exif[36867] = f"2025:01:{session_id:02d} 12:00:00"
            exif[274] = 6 if image_id == 3 else 1
            raw_size = (12, 20) if image_id == 3 else (20, 12)
            pixels = random.Random(image_id).randbytes(raw_size[0] * raw_size[1] * 3)
            with Image.frombytes("RGB", raw_size, pixels) as image:
                image.save(path, exif=exif)
            source_id = 5 if image_id == 6 else image_id
            images.append({
                "id": image_id, "file_name": f"batch_1/{image_id:03d}.jpg",
                "width": 20, "height": 12, "license": 1,
                "flickr_url": f"https://www.flickr.com/photos/fixture/{source_id}/",
            })
        annotations: list[dict[str, Any]] = []
        for image_id in range(1, 17):
            if image_id == 7:
                continue
            category_id = 10 if image_id == 1 else 30 if image_id == 5 else 20
            bbox = [12, 2, 6, 4] if image_id == 3 else [19, 1, 3, 2] if image_id == 4 else [1, 2, 4, 3]
            annotation = {
                "id": 309 if image_id == 2 else 1000 + image_id,
                "image_id": image_id, "category_id": category_id,
                "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0,
                "segmentation": [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1],
                                  bbox[0] + bbox[2], bbox[1] + bbox[3]]],
            }
            annotations.append(annotation)
            if image_id == 2:
                annotations.append({**annotation, "bbox": [7, 2, 3, 3]})
        self.source: dict[str, Any] = {
            "info": {"description": "微型官方结构测试数据"},
            "licenses": [{"id": 1, "name": "fixture"}],
            "images": images, "annotations": annotations,
            "categories": [
                {"id": 10, "name": "Battery", "supercategory": "Battery"},
                {"id": 20, "name": "Drink can", "supercategory": "Can"},
                {"id": 30, "name": "Other carton", "supercategory": "Carton"},
            ],
        }
        (self.root / "annotations.json").write_text(
            json.dumps(self.source, ensure_ascii=False), encoding="utf-8"
        )

    def _run(self, output: Path | None = None) -> dict[str, Any]:
        """运行指定种子的流水线，断言返回的完整 COCO 与实际 JSON 文件一致。"""
        destination = self.output if output is None else output
        result = run_pipeline(self.root, destination, 42)
        self.assertIsInstance(result, dict)
        self.assertEqual(result, json.loads(destination.read_text(encoding="utf-8")))
        for key in ("images", "annotations", "categories", "project"):
            self.assertIn(key, result)
        return result

    def _sample(self, coco: dict[str, Any], image_id: int) -> dict[str, Any]:
        """从图像所属的正式分区获取消费者样本，避免依赖具体的随机分区结果。"""
        record = next(image for image in coco["images"] if image["id"] == image_id)
        dataset = ProjectDataset(self.root, self.output, record["split"])
        return next(dataset[index] for index in range(len(dataset))
                    if dataset.images[index]["id"] == image_id)

    def test_background_and_empty_images_are_pass(self) -> None:
        """背景类别和零目标图像均为 PASS，五类计数为零且不产生目标框。"""
        coco = self._run()
        for image_id in (1, 7):
            with self.subTest(image_id=image_id):
                sample = self._sample(coco, image_id)
                self.assertEqual(sample["label"], "PASS")
                self.assertEqual(sample["counts"], {str(category): 0 for category in range(1, 6)})
                self.assertEqual(sample["objects"], [])
                self.assertEqual(sample["image"].mode, "RGB")

    def test_review_holds_out_only_incomplete_images(self) -> None:
        """仅待定图片留出，同组标注完整的图片继续保留正式且保留共同组标识。"""
        coco = self._run()
        held_out = {image["id"]: image for image in coco["project"]["held_out"]}
        self.assertIn(5, held_out)
        self.assertNotIn(6, held_out)
        companion = next(image for image in coco["images"] if image["id"] == 6)
        self.assertEqual(held_out[5]["group_id"], companion["group_id"])
        self.assertTrue(held_out[5]["reasons"])
        self.assertNotIn(5, {image["id"] for image in coco["images"]})
        self.assertNotIn(5, {obj["image_id"] for obj in coco["annotations"]})
        self.assertIn(6, {obj["image_id"] for obj in coco["annotations"]})
        all_ids = {image["id"] for image in coco["images"]} | held_out.keys()
        self.assertEqual(all_ids, {image["id"] for image in self.source["images"]})

    def test_duplicate_object_ids_are_counted_individually(self) -> None:
        """重复原物体编号不去重，唯一 COCO 编号及原数组位置保留逐条来源。"""
        coco = self._run()
        sample = self._sample(coco, 2)
        self.assertEqual(sample["label"], "FAIL")
        self.assertEqual(sample["counts"], {"1": 2, "2": 0, "3": 0, "4": 0, "5": 0})
        self.assertEqual(len(sample["objects"]), 2)
        self.assertEqual([obj["object_id"] for obj in sample["objects"]], [309, 309])
        self.assertEqual(len({obj["id"] for obj in sample["objects"]}), 2)
        for obj in coco["annotations"]:
            self.assertEqual(obj["id"], obj["source_index"] + 1)
            source = self.source["annotations"][obj["source_index"]]
            self.assertEqual(obj["object_id"], source["id"])
            self.assertEqual(obj["image_id"], source["image_id"])

    def test_exif_transposes_pixels_but_not_coordinates(self) -> None:
        """读取时转正像素，官方框及多边形保持原值，同时原图字节完全不变。"""
        before = {path.relative_to(self.root): path.read_bytes()
                  for path in self.root.rglob("*") if path.is_file()}
        coco = self._run()
        sample = self._sample(coco, 3)
        self.assertEqual(sample["image"].size, (20, 12))
        self.assertEqual(sample["image"].mode, "RGB")
        obj = sample["objects"][0]
        self.assertEqual(obj["category_id"], 1)
        self.assertEqual(obj["bbox"], [12, 2, 6, 4])
        self.assertEqual([obj[key] for key in ("xmin", "ymin", "xmax", "ymax")], [12, 2, 18, 6])
        original = next(row for row in self.source["annotations"] if row["image_id"] == 3)
        self.assertEqual(obj["segmentation"], original["segmentation"])
        self.assertEqual(before, {path.relative_to(self.root): path.read_bytes()
                                  for path in self.root.rglob("*") if path.is_file()})
        with Image.open(self.root / "batch_1/003.jpg") as image:
            self.assertEqual(image.size, (12, 20))
            self.assertEqual(image.getexif()[274], 6)
        self.assertFalse(any(self.output.parent.rglob("*.jpg")))

    def test_invalid_box_is_held_out_without_clipping(self) -> None:
        """越界框不能被裁剪后偷偷加入训练，而应在可追溯留出记录中出现。"""
        coco = self._run()
        self.assertNotIn(4, {image["id"] for image in coco["images"]})
        self.assertNotIn(4, {obj["image_id"] for obj in coco["annotations"]})
        held_out = next(image for image in coco["project"]["held_out"] if image["id"] == 4)
        self.assertTrue(held_out["reasons"])
        stored = json.loads((self.root / "annotations.json").read_text(encoding="utf-8"))
        original = next(row for row in stored["annotations"] if row["image_id"] == 4)
        self.assertEqual(original["bbox"], [19, 1, 3, 2])

    def test_crowd_and_ignore_do_not_become_false_pass(self) -> None:
        """参数无；目标 crowd/ignore 必须留出，背景标记不影响 PASS；断言失败抛 AssertionError。"""
        for annotation in self.source["annotations"]:
            if annotation["image_id"] == 8:
                annotation["iscrowd"] = 1
            if annotation["image_id"] == 9:
                annotation["ignore"] = 1
            if annotation["image_id"] == 1:
                annotation["iscrowd"] = 1
        (self.root / "annotations.json").write_text(json.dumps(self.source), encoding="utf-8")
        coco = self._run()
        held = {image["id"] for image in coco["project"]["held_out"]}
        self.assertTrue({8, 9}.issubset(held))
        self.assertEqual(self._sample(coco, 1)["label"], "PASS")

    def test_consumer_rejects_unsafe_paths_and_changed_dimensions(self) -> None:
        """正式标注也不能越出原图目录，且读取时必须发现图像显示尺寸不符。"""
        coco = self._run()
        record = next(image for image in coco["images"] if image["id"] == 3)
        malformed = self.output.parent / "invalid.json"
        for field, value in (("file_name", "../escape.jpg"), ("width", 21)):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(coco))
                target = next(image for image in changed["images"] if image["id"] == 3)
                target[field] = value
                malformed.write_text(json.dumps(changed), encoding="utf-8")
                dataset = ProjectDataset(self.root, malformed, record["split"])
                index = next(index for index, image in enumerate(dataset.images) if image["id"] == 3)
                with self.assertRaises(ValueError):
                    dataset[index]

    def test_partitions_are_reproducible_and_strictly_isolated(self) -> None:
        """同种子复现完整产物，三个消费者严格按指定分区且场景组互不泄漏。"""
        coco = self._run()
        repeated = self._run(self.output.parent / "repeated.json")
        self.assertEqual(coco, repeated)
        seen_images: set[int] = set()
        seen_groups: set[str] = set()
        held_out_ids = {image["id"] for image in coco["project"]["held_out"]}
        for split in ("train", "val", "test"):
            dataset = ProjectDataset(self.root, self.output, split)
            expected = {image["id"] for image in coco["images"] if image["split"] == split}
            actual = {dataset[index]["image_id"] for index in range(len(dataset))}
            groups = {image["group_id"] for image in dataset.images}
            self.assertTrue(expected)
            self.assertEqual(actual, expected)
            self.assertFalse(actual & (seen_images | held_out_ids))
            self.assertFalse(groups & seen_groups)
            seen_images.update(actual)
            seen_groups.update(groups)
        self.assertEqual(seen_images, {image["id"] for image in coco["images"]})
        for split in ("", "all", "held_out", "review", "validation"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                ProjectDataset(self.root, self.output, split)


if __name__ == "__main__":
    unittest.main()
