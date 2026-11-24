"""检查基于真实审核标注的 YOLO 框、负图和分区导出规则。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from ultralytics.data.utils import check_det_dataset

from dataprocessing.supplemental.datasets import INDOOR_HOLDOUT

from robot_cleaning_vision.models.detector.data import (
    TACO_ROOT, display_image, load_partitions, read_json, stage_dataset, yolo_labels,
)


class DetectorDataTests(unittest.TestCase):
    """使用已审核标注与可选的本地真实原件检验数据转换。"""

    def test_difficult_boxes_remain_targets_but_ignore_is_rejected(self):
        """以实际 difficult 标注检查目标保留和 ignore 违规时拒绝导出；无参数。"""
        source = read_json(INDOOR_HOLDOUT.with_name("local_training_annotations_project.json"))
        difficult = next(box for box in source["annotations"] if box["difficult"])
        image = next(row for row in source["images"] if row["id"] == difficult["image_id"])
        boxes = [box for box in source["annotations"] if box["image_id"] == image["id"]]
        labels, difficult_count = yolo_labels(image, boxes)
        self.assertGreater(difficult_count, 0)
        self.assertEqual(sum(line.startswith(f"{difficult['category_id'] - 1} ") for line in labels.splitlines()),
                         image["counts"][str(difficult["category_id"])])
        with self.assertRaisesRegex(ValueError, "crowd/ignore"):
            yolo_labels(image, [{**difficult, "ignore": 1}])

    @unittest.skipUnless((TACO_ROOT / "annotations_project.json").exists(), "需要本地已处理的 TACO")
    def test_materialization_keeps_exif_boxes_and_negative_without_exporting_holdout(self):
        """用真实 EXIF 旋转图、负图和室内留出核验 Ultralytics 输入；无参数。"""
        partitions, _ = load_partitions()
        positive = next(item for item in partitions["train"] if item.record["id"] == 231)
        negative = next(item for item in partitions["train"] if item.record["id"] == 6)
        val, indoor = partitions["val"][0], partitions["indoor"][0]
        if not all(item.path.exists() for item in (positive, negative, val, indoor)):
            self.skipTest("需要真实 TACO 和 Indoor 原图")
        displayed, orientation = display_image(positive)
        self.assertEqual(orientation, 6)
        self.assertEqual(displayed.size, (2322, 4128))
        with Image.open(positive.path) as original:
            self.assertEqual(displayed.getpixel((0, 0)), original.convert("RGB").getpixel((0, 2321)))
        items = {"train": [positive, negative], "val": [val], "indoor": [indoor]}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, _ = stage_dataset(root, items, ("train", "val"))
            checked = check_det_dataset(str(dataset), autodownload=False)
            self.assertNotIn("indoor", checked)
            self.assertFalse((root / "images/indoor").exists())
            self.assertEqual((root / "labels/train/6.txt").read_bytes(), b"")
            label = (root / "labels/train/231.txt").read_text(encoding="ascii").split()
            self.assertEqual(label[0], "2")
            np.testing.assert_allclose([float(v) for v in label[1:]],
                                       [1374 / 2322, 2546 / 4128, 588 / 2322, 240 / 4128], rtol=1e-7)
            with Image.open(root / "images/train/231.png") as staged:
                self.assertEqual(staged.size, (360, 640))
                self.assertIsNone(staged.getexif().get(274))
            evaluation, _ = stage_dataset(root / "evaluation", items, ("val", "indoor"))
            checked = check_det_dataset(str(evaluation), autodownload=False)
            self.assertIsNone(checked["train"])
            self.assertTrue(Path(checked["indoor"]).is_absolute())
            self.assertTrue(Path(checked["indoor"]).is_dir())


if __name__ == "__main__":
    unittest.main()
