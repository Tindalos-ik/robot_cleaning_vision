"""使用实际 checkpoint、后处理和渲染器检验消费者拿到的图片及预测。"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
from PIL import Image

from robot_cleaning_vision.inference import classify_frame, run_image, save_prediction
from robot_cleaning_vision.models.detector.model import DetectorModel
from robot_cleaning_vision.models.mlp.model import MLPModel
from robot_cleaning_vision.visualization import ResultRenderer


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        font = Path(os.environ.get("VISION_TEST_FONT", "C:/Windows/Fonts/msyh.ttc"))
        if not font.is_file():
            raise unittest.SkipTest("请设置 VISION_TEST_FONT 为中文字体文件路径")
        cls.model = MLPModel.load()
        cls.renderer = ResultRenderer(font)

    def setUp(self) -> None:
        # 非对称红蓝区域：交换 RGB/BGR 不再是无效的灰度测试。
        self.rgb = np.zeros((96, 128, 3), dtype=np.uint8)
        self.rgb[:] = (230, 35, 15)
        self.rgb[12:72, 20:65] = (10, 60, 245)
        self.rgb[45:90, 70:122] = (90, 190, 30)
        self.frame = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)

    def test_bgr_prediction_matches_real_rgb_model_without_changing_original(self) -> None:
        original = self.frame.copy()
        expected = self.model.predict(self.rgb)
        prediction = classify_frame(self.frame, self.model, self.renderer)
        self.assertAlmostEqual(prediction.result.image_score, expected, places=12)
        self.assertEqual(prediction.threshold, self.model.threshold)
        self.assertEqual(
            prediction.result.decision,
            "FAIL" if expected >= self.model.threshold else "PASS",
        )
        self.assertIsNone(prediction.result.objects)
        self.assertIsNone(prediction.result.counts)
        self.assertGreater(prediction.elapsed_ms, 0)
        np.testing.assert_array_equal(self.frame, original)
        np.testing.assert_array_equal(prediction.annotated[:96, :128], original)

    def test_saved_original_json_and_rendered_image_are_the_same_displayed_frame(self) -> None:
        prediction = classify_frame(self.frame, self.model, self.renderer)
        # 处理另一帧不应让上一帧的保存结果指向新帧或重复执行预测。
        classify_frame(np.full_like(self.frame, 127), self.model, self.renderer)
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "results"
            first = save_prediction(prediction, output, source=0)
            files_before = {path.name: path.read_bytes() for path in first.iterdir()}
            second = save_prediction(prediction, output, source=0)
            self.assertNotEqual(first, second)
            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()}, files_before,
            )
            with Image.open(first / "original.png") as original:
                np.testing.assert_array_equal(np.array(original), self.rgb)
            with Image.open(first / "result.png") as annotated:
                np.testing.assert_array_equal(
                    np.array(annotated), cv2.cvtColor(prediction.annotated, cv2.COLOR_BGR2RGB),
                )
            payload = json.loads((first / "prediction.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["fail_probability"], prediction.result.image_score)
            self.assertEqual(payload["threshold"], self.model.threshold)
            self.assertEqual(payload["decision"], prediction.result.decision)
            self.assertEqual(payload["elapsed_ms"], prediction.elapsed_ms)
            self.assertEqual(payload["task"], "classification")
            self.assertNotIn("boxes", payload)
            self.assertNotIn("counts", payload)
        np.testing.assert_array_equal(self.frame, cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))

    def test_image_run_applies_exif_before_inference_and_preserves_input_file(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "camera_image.png"
            image = Image.fromarray(self.rgb)
            exif = Image.Exif()
            exif[274] = 6  # 顺时针 90 度，与常见手机竖拍方向一致。
            image.save(source, exif=exif)
            source_bytes = source.read_bytes()
            directory = run_image(source, self.model, self.renderer, root / "output")
            expected_rgb = np.rot90(self.rgb, k=3)
            with Image.open(directory / "original.png") as original:
                np.testing.assert_array_equal(np.array(original), expected_rgb)
            with Image.open(directory / "result.png") as annotated:
                np.testing.assert_array_equal(np.array(annotated)[:128, :96], expected_rgb)
            payload = json.loads((directory / "prediction.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(
                payload["fail_probability"], self.model.predict(expected_rgb), places=12,
            )
            self.assertEqual(payload["threshold"], self.model.threshold)
            self.assertEqual(source.read_bytes(), source_bytes)


class DetectorCheckpointTests(unittest.TestCase):
    def test_native_coco_checkpoint_is_rejected_before_inference(self) -> None:
        """用真实的 80 类 COCO checkpoint 检查五类模型的加载校验。"""
        path = Path("artifacts/detector/initialization/yolo11n.pt")
        if not path.is_file():
            self.skipTest("缺少本地官方 YOLO11n COCO 权重")
        with self.assertRaisesRegex(ValueError, "实际为 80 类"):
            DetectorModel.load(path)


if __name__ == "__main__":
    unittest.main()
