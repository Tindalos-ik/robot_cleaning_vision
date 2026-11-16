"""通过真实 CLI 验证离线分类、检测坐标与文件保护。"""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from robot_cleaning_vision.visualization import CATEGORY_COLORS


class OfflineCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.font = Path(os.environ.get("VISION_TEST_FONT", "C:/Windows/Fonts/msyh.ttc"))
        if not cls.font.is_file():
            raise unittest.SkipTest("请设置 VISION_TEST_FONT 为中文字体文件路径")

    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "原图.png"
        self.predictions = self.root / "预测.json"
        self.output = self.root / "标注.png"
        self.pixels = np.full((200, 400, 3), (17, 37, 67), dtype=np.uint8)
        Image.fromarray(self.pixels).save(self.source)
        self.original_bytes = self.source.read_bytes()

    def invoke(self, prediction: object, output: Path | None = None) -> subprocess.CompletedProcess:
        self.predictions.write_text(json.dumps(prediction, ensure_ascii=False), encoding="utf-8")
        return subprocess.run(
            [sys.executable, "-m", "robot_cleaning_vision.main", "process",
             "--input", str(self.source), "--predictions", str(self.predictions),
             "--output", str(output if output is not None else self.output),
             "--font", str(self.font), "--target-size", "400", "400",
             "--threshold", "0.5", "--elapsed-ms", "23.6"],
            capture_output=True,
        )

    def test_classification_keeps_original_pixels_and_file(self) -> None:
        completed = self.invoke({"task": "classification", "model_name": "分类器",
                                 "fail_probability": 0.8})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(b"FAIL", completed.stdout)
        with Image.open(self.output) as image:
            rendered = np.array(image)
        np.testing.assert_array_equal(rendered[:200, :400], self.pixels)
        self.assertGreater(rendered.shape[1], 400)
        self.assertEqual(self.source.read_bytes(), self.original_bytes)

    def test_detection_filters_then_restores_letterbox_coordinates(self) -> None:
        completed = self.invoke({
            "task": "detection", "model_name": "检测器",
            "boxes": [[40, 160, 140, 260], [250, 160, 350, 260]],
            "class_ids": [2, 3], "scores": [0.9, 0.49],
        })
        self.assertEqual(completed.returncode, 0, completed.stderr)
        with Image.open(self.output) as image:
            rendered = np.array(image)
        # 原图 200x400，上下各填充 100；框恢复到 y=60..160。
        np.testing.assert_array_equal(rendered[120, 40], CATEGORY_COLORS[1])
        np.testing.assert_array_equal(rendered[120, 250], self.pixels[120, 250])
        self.assertEqual(self.source.read_bytes(), self.original_bytes)

    def test_empty_detection_is_pass_and_keeps_original_pixels(self) -> None:
        completed = self.invoke({"task": "detection", "model_name": "检测器",
                                 "boxes": [], "class_ids": [], "scores": []})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(b"PASS", completed.stdout)
        with Image.open(self.output) as image:
            rendered = np.array(image)
        np.testing.assert_array_equal(rendered[:200, :400], self.pixels)

    def test_bad_predictions_and_existing_outputs_do_not_overwrite_files(self) -> None:
        for prediction in (
            {"task": "classification", "model_name": "分类器"},
            {"task": "classification", "model_name": "分类器", "fail_probability": 2},
            {"task": "detection", "model_name": "检测器", "boxes": [[0, 1, 2, 3]],
             "class_ids": [], "scores": []},
        ):
            with self.subTest(prediction=prediction):
                completed = self.invoke(prediction)
                self.assertEqual(completed.returncode, 1, completed.stderr)
                self.assertFalse(self.output.exists())
                self.assertNotIn(b"Traceback", completed.stderr)
        valid = {"task": "classification", "model_name": "分类器", "fail_probability": 0.2}
        self.output.write_bytes(b"existing output")
        self.assertEqual(self.invoke(valid).returncode, 1)
        self.assertEqual(self.output.read_bytes(), b"existing output")
        self.assertEqual(self.invoke(valid, self.source).returncode, 1)
        self.assertEqual(self.source.read_bytes(), self.original_bytes)


if __name__ == "__main__":
    unittest.main()
