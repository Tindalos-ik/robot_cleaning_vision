"""保留模型的确定性消费者测试，不依赖训练流程或原始数据集。"""

import unittest

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

from robot_cleaning_vision.models.mlp import DEFAULT_MODEL_PATH, MLPModel
from robot_cleaning_vision.models.mlp.features import (
    extract_context, extract_regions, generate_windows, prepare_image, rgb_image,
)
from robot_cleaning_vision.models.mlp.model import score_summary


class FeatureTests(unittest.TestCase):
    def test_rgb_hsv_bins_and_quadrant_order(self):
        pixels = np.empty((64, 64, 3), dtype=np.uint8)
        colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255))
        for (top, left), color in zip(((0, 0), (0, 32), (32, 0), (32, 32)), colors):
            pixels[top:top + 32, left:left + 32] = color
        before = pixels.copy()
        features = extract_regions(pixels, np.array([[0, 0, 64, 64]]))[0]
        quadrants = np.zeros((4, 32), dtype=np.float32)
        for row, hue in enumerate((0, 5, 10, 0)):
            quadrants[row, hue] = 1
            quadrants[row, 16 + (0 if row == 3 else 7)] = 1
            quadrants[row, 31] = 1
        np.testing.assert_array_equal(features, np.r_[quadrants.mean(axis=0), quadrants.ravel()])
        np.testing.assert_array_equal(pixels, before)

    def test_local58_constant_black_and_white_definitions(self):
        for value in (0, 255):
            pixels = np.full((24, 48, 3), value, dtype=np.uint8)
            before = pixels.copy()
            block = np.zeros(11, dtype=np.float32)
            if value:
                block[[2, 10]] = 1  # 明度均值与明亮低饱和度像素比例。
            expected = np.r_[block, np.zeros(11), block, block, block, np.zeros(3)]
            result = extract_context(pixels)
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(pixels, before)

    def test_windows_are_ordered_deduplicated_and_include_frame(self):
        settings = {"scales": [0.5, 0.5], "aspects": [1.0], "stride": 1.0}
        expected = np.array([
            [0, 0, 2, 2], [2, 0, 4, 2], [4, 0, 6, 2],
            [0, 2, 2, 4], [2, 2, 4, 4], [4, 2, 6, 4], [0, 0, 6, 4],
        ], dtype=np.int32)
        np.testing.assert_array_equal(generate_windows((4, 6, 3), settings), expected)
        image = np.zeros((3, 7, 3), dtype=np.uint8)
        self.assertEqual(prepare_image(image).shape, (274, 640, 3))

    def test_exif_matches_explicit_orientation_without_mutation(self):
        pixels = np.zeros((40, 80, 3), dtype=np.uint8)
        pixels[:20, :40] = (255, 0, 0)
        image = Image.fromarray(pixels)
        image.getexif()[274] = 6
        corrected = np.asarray(ImageOps.exif_transpose(image))
        np.testing.assert_array_equal(rgb_image(image), corrected)
        np.testing.assert_array_equal(prepare_image(image), prepare_image(corrected))
        np.testing.assert_array_equal(extract_context(image), extract_context(corrected))
        np.testing.assert_array_equal(np.asarray(image), pixels)
        self.assertEqual(image.getexif()[274], 6)

    def test_invalid_array_inputs_are_rejected(self):
        for pixels in (np.zeros((8, 8)), np.zeros((8, 8, 3), dtype=np.float32),
                       np.zeros((0, 8, 3), dtype=np.uint8)):
            with self.subTest(shape=pixels.shape):
                with self.assertRaises(ValueError):
                    rgb_image(pixels)

    def test_score30_statistics_and_order_independence(self):
        probabilities = np.array([0, 0.5, 1], dtype=np.float32)
        boxes = np.array([[0, 0, 1, 1], [1, 0, 3, 2], [0, 0, 10, 10]])
        expected = np.array([
            1, .5, .5, .5, .5, .25, .5, .75, .9, .95, .99,
            .5, np.sqrt(1 / 6), 2 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3,
            0, 0, .5, .5, 1, 1, 102 / 105, 1, 1 / 3, 0, 1, np.log(4),
        ], dtype=np.float32)
        actual = score_summary(probabilities, boxes, (10, 10))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(
            score_summary(probabilities[::-1], boxes[::-1], (10, 10)), actual)


@unittest.skipUnless(DEFAULT_MODEL_PATH.is_file(), "Selected checkpoint not installed")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = torch.load(DEFAULT_MODEL_PATH, map_location="cpu", weights_only=True)
        cls.model = MLPModel.load()

    @staticmethod
    def checkpoint_forward(features, saved):
        mean = saved["mean"].numpy().astype(np.float32)
        scale = saved["scale"].numpy().astype(np.float32)
        tensor = torch.from_numpy((features - mean) / scale)
        state = saved["state_dict"]
        with torch.inference_mode():
            hidden = F.relu(F.linear(tensor, state["0.weight"], state["0.bias"]))
            return F.linear(hidden, state["2.weight"], state["2.bias"]).sigmoid().numpy().ravel()

    def test_real_checkpoint_scalers_networks_and_threshold(self):
        """核对默认权重的单图计算、重复预测和文件内阈值。"""
        pixels = np.random.default_rng(17).integers(0, 256, (37, 61, 3), dtype=np.uint8)
        before = pixels.copy()
        region = self.saved["region_checkpoint"]
        frame = prepare_image(pixels, region["proposal"]["long_edge"])
        boxes = generate_windows(frame.shape, region["proposal"])
        probabilities = self.checkpoint_forward(extract_regions(frame, boxes), region)
        features = np.r_[
            score_summary(probabilities, boxes, (frame.shape[1], frame.shape[0])),
            extract_context(pixels),
        ]
        expected = float(self.checkpoint_forward(features[None, :], self.saved)[0])
        actual = self.model.predict(pixels)
        repeated = self.model.predict(pixels)
        self.assertIs(type(actual), float)
        self.assertAlmostEqual(actual, expected, delta=1e-7)
        self.assertEqual(actual, repeated)
        self.assertEqual(self.model.threshold, self.saved["threshold"])
        np.testing.assert_array_equal(pixels, before)

    def test_model_pil_exif_and_rgb_array_agree(self):
        pixels = np.random.default_rng(9).integers(0, 256, (23, 49, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        image.getexif()[274] = 6
        corrected = np.asarray(ImageOps.exif_transpose(image))
        self.assertAlmostEqual(self.model.predict(image), self.model.predict(corrected), delta=1e-7)
        np.testing.assert_array_equal(np.asarray(image), pixels)
        self.assertEqual(image.getexif()[274], 6)


if __name__ == "__main__":
    unittest.main()
