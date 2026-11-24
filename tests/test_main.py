"""通过真实 CLI 检查运行推理、离线回放和文件保护。"""

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image, ImageChops, ImageOps

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


class DetectorDeploymentCommandTests(unittest.TestCase):
    def test_missing_checkpoint_reports_requested_path_without_creating_output(self) -> None:
        """显式权重不存在时，run 应在启动摄像头或保存结果前报错。"""
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, output = root / "model.pt", root / "output"
            completed = subprocess.run(
                [sys.executable, "-m", "robot_cleaning_vision.main", "run",
                 "--model", "baseline3", "--model-path", str(checkpoint), "--output", str(output)],
                capture_output=True, text=True, encoding="utf-8",
                env={**os.environ, "PYTHONUTF8": "1"}, timeout=30,
            )
            self.assertEqual(completed.returncode, 1, completed.stderr)
            self.assertIn(f"找不到 baseline3 权重：{checkpoint.resolve()}", completed.stderr)
            self.assertFalse(output.exists())

    def test_default_detector_saves_and_replays_real_positive_and_negative(self) -> None:
        """用杯子、空房间和难负样本校验默认权重、原图框、计数及回放；无参数。"""
        font = Path(os.environ.get("VISION_TEST_FONT", "C:/Windows/Fonts/msyh.ttc"))
        cup = Path("data/raw/taco/batch_14/000031.jpg")
        empty_room = Path("data/raw/supplemental/wikimedia/8941662.jpg")
        hard_negative = Path("data/raw/taco/batch_2/000071.JPG")
        weights = Path("artifacts/detector/model.pt")
        for path in (weights, font, cup, empty_room, hard_negative):
            if not path.is_file():
                self.skipTest(f"缺少本地验收素材：{path}")

        # 难负样本中的吸管可能误检；仍要求判定与计数、回放一致。
        for source, expected_decision in ((cup, "FAIL"), (empty_room, "PASS"), (hard_negative, None)):
            with self.subTest(source=source), TemporaryDirectory() as temporary:
                root = Path(temporary)
                command = [sys.executable, "-m", "robot_cleaning_vision.main"]
                completed = subprocess.run(
                    [*command, "run", "--model", "baseline3", "--input", str(source),
                     "--font", str(font), "--output", str(root)],
                    capture_output=True, timeout=120,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                (saved,) = root.iterdir()
                predictions = saved / "prediction.json"
                payload = json.loads(predictions.read_text(encoding="utf-8"))
                class_ids = payload["class_ids"]
                counts = Counter(class_ids)
                decision = "FAIL" if class_ids else "PASS"
                if expected_decision is not None:
                    self.assertEqual(decision, expected_decision, payload)
                self.assertIn(decision.encode(), completed.stdout)
                self.assertEqual(payload["model_id"], "baseline3")
                self.assertEqual(payload["task"], "detection")
                self.assertEqual(Path(payload["model_path"]), weights.resolve())
                self.assertEqual(payload["decision"], decision)
                self.assertEqual(payload["counts"], {str(i): counts[i] for i in range(1, 6)})
                self.assertEqual(payload["target_size"], [640, 640])
                self.assertEqual(payload["threshold"], 0.25)
                self.assertNotIn("fail_probability", payload)
                self.assertEqual(len(class_ids), len(payload["boxes"]))
                self.assertEqual(len(class_ids), len(payload["original_boxes"]))
                self.assertEqual(len(class_ids), len(payload["scores"]))
                self.assertTrue(all(0.25 <= score <= 1 for score in payload["scores"]))
                if source == cup:
                    # 杯子位于转正后的画面右下方；原图坐标不应落在 640 像素画布内。
                    self.assertTrue(any(
                        category == 4 and x1 < 2100 < x2 and y1 < 2500 < y2
                        for category, (x1, y1, x2, y2) in zip(class_ids, payload["original_boxes"])
                    ), payload)
                    np.testing.assert_allclose(
                        payload["boxes"],
                        np.asarray(payload["original_boxes"]) * (640 / 4032) + (80, 0, 80, 0),
                        atol=1e-5,
                    )
                if source == hard_negative and class_ids:
                    np.testing.assert_allclose(
                        payload["boxes"],
                        np.asarray(payload["original_boxes"]) * (640 / 3264) + (80, 0, 80, 0),
                        atol=1e-5,
                    )

                original = saved / "original.png"
                with Image.open(source) as raw, Image.open(original) as exported:
                    expected = ImageOps.exif_transpose(raw).convert("RGB")
                    self.assertEqual(exported.size, expected.size)
                    self.assertIsNone(ImageChops.difference(exported, expected).getbbox())
                replay = root / "replayed.png"
                completed = subprocess.run(
                    [*command, "process", "--input", str(original), "--predictions", str(predictions),
                     "--target-size", "640", "640", "--threshold", str(payload["threshold"]),
                     "--font", str(font), "--elapsed-ms", str(payload["elapsed_ms"]),
                     "--output", str(replay)],
                    capture_output=True, timeout=30,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn(decision.encode(), completed.stdout)
                with Image.open(saved / "result.png") as rendered, Image.open(replay) as replayed:
                    self.assertEqual(replayed.size, rendered.size)
                    self.assertIsNone(ImageChops.difference(rendered, replayed).getbbox())


if __name__ == "__main__":
    unittest.main()
