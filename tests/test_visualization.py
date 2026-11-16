"""验证像素保留、检测颜色与非完成状态的旧结果隔离。"""

import os
from pathlib import Path
import unittest

import numpy as np

from robot_cleaning_vision.postprocessing import Detection, ImageResult, postprocess_classification
from robot_cleaning_vision.visualization import CATEGORY_COLORS, ResultRenderer


class VisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        font = Path(os.environ.get("VISION_TEST_FONT", "C:/Windows/Fonts/msyh.ttc"))
        if not font.is_file():
            raise unittest.SkipTest("请设置 VISION_TEST_FONT 为中文字体文件路径")
        cls.renderer = ResultRenderer(font)

    def test_classification_preserves_small_noncontiguous_bgr_frame(self) -> None:
        source = np.random.default_rng(42).integers(0, 256, (37, 62, 3), dtype=np.uint8)
        frame = source[:, ::-1]
        before = source.copy()
        result = postprocess_classification(0.8, threshold=0.5)
        rendered = self.renderer.draw(frame, result, model_name="分类", elapsed_ms=12.3)
        np.testing.assert_array_equal(source, before)
        np.testing.assert_array_equal(rendered[:37, :62], frame)
        self.assertFalse(np.shares_memory(rendered, frame))
        self.assertEqual(rendered.dtype, np.uint8)
        self.assertTrue(rendered.flags.c_contiguous)
        self.assertGreater(rendered.shape[0], 37)
        self.assertGreater(rendered.shape[1], 62)

    def test_detection_draws_all_scores_with_fixed_colors_in_original_coordinates(self) -> None:
        frame = np.full((400, 700, 3), (7, 19, 43), dtype=np.uint8)
        objects = [
            Detection((20 + i * 130, 100, 110 + i * 130, 250), i + 1, score)
            for i, score in enumerate([0.0, 0.2, 0.5, 0.8, 1.0])
        ]
        result = ImageResult("FAIL", None, objects, {i: 1 for i in range(1, 6)})
        rendered = self.renderer.draw(frame, result, model_name="检测")
        for i, color in enumerate(CATEGORY_COLORS):
            np.testing.assert_array_equal(rendered[200, 20 + i * 130], color[::-1])
        np.testing.assert_array_equal(rendered[300, :700], frame[300])
        np.testing.assert_array_equal(frame, np.full_like(frame, (7, 19, 43)))
        # 图像级置信度不是检测显示的一部分。
        result.image_score = 0.99
        np.testing.assert_array_equal(
            rendered, self.renderer.draw(frame, result, model_name="检测"),
        )

    def test_pending_and_error_ignore_stale_results(self) -> None:
        frame = np.full((80, 100, 3), 123, dtype=np.uint8)
        stale = ImageResult("FAIL", None, [Detection((1, 1, 40, 40), 1, 0.9)], {1: 1})
        for status in ("pending", "error"):
            with self.subTest(status=status):
                options = dict(model_name="检测", status=status, elapsed_ms=42.5,
                               error_message="模型不可用")
                without = self.renderer.draw(frame, **options)
                with_stale = self.renderer.draw(frame, stale, **options)
                np.testing.assert_array_equal(without, with_stale)
                np.testing.assert_array_equal(with_stale[:80, :100], frame)

    def test_empty_detection_preserves_frame_and_displays_zero_counts(self) -> None:
        frame = np.zeros((400, 200, 3), dtype=np.uint8)
        result = ImageResult("PASS", None, [], dict.fromkeys(range(1, 6), 0))
        rendered = self.renderer.draw(frame, result, model_name="检测")
        np.testing.assert_array_equal(rendered[:, :200], frame)
        # 每个类别的计数行均使用其固定颜色，即使计数为零。
        sidebar = rendered[:, 200:]
        for color in CATEGORY_COLORS:
            self.assertTrue(np.any(np.all(sidebar == color[::-1], axis=2)))

    def test_invalid_frame_status_and_elapsed_are_rejected(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        for invalid in (frame.astype(float), frame[:, :, 0], frame[:0]):
            with self.assertRaises(ValueError):
                self.renderer.draw(invalid, model_name="模型", status="pending")
        with self.assertRaises(ValueError):
            self.renderer.draw(frame, model_name="模型")
        with self.assertRaises(ValueError):
            self.renderer.draw(frame, model_name="模型", status="unknown")
        for elapsed in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.renderer.draw(frame, model_name="模型", status="pending", elapsed_ms=elapsed)


if __name__ == "__main__":
    unittest.main()
