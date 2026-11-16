"""覆盖 letterbox 的取整、通道与数值变换以及坐标往返。"""

import unittest

import cv2
import numpy as np
from PIL import Image

from robot_cleaning_vision.preprocessing import preprocess


class PreprocessingTests(unittest.TestCase):
    def test_letterbox_rounding_and_box_round_trip(self) -> None:
        image = np.arange(3 * 7 * 3, dtype=np.uint8).reshape(3, 7, 3)
        original = image.copy()
        output, transform = preprocess(image, (8, 10), color="BGR", fill=(11, 22, 33))

        self.assertEqual(transform.original_size, (3, 7))
        self.assertEqual(transform.resized_size, (4, 10))
        self.assertEqual(transform.target_size, (8, 10))
        self.assertEqual(transform.scale, (10 / 7, 4 / 3))
        self.assertEqual(transform.padding, (0, 2, 0, 2))
        np.testing.assert_array_equal(output[2:6], cv2.resize(image, (10, 4)))
        np.testing.assert_array_equal(output[:2], np.full((2, 10, 3), (11, 22, 33)))
        np.testing.assert_array_equal(output[6:], output[:2])
        np.testing.assert_array_equal(image, original)

        boxes = np.array([[0, 0, 7, 3], [1.25, 0.5, 6.5, 2.75]], dtype=np.float32)
        before = boxes.copy()
        mapped = transform.forward_boxes(boxes)
        np.testing.assert_allclose(mapped[0], [0, 2, 10, 6])
        np.testing.assert_allclose(mapped[1], [1.25 * 10 / 7, 2 + 0.5 * 4 / 3,
                                              6.5 * 10 / 7, 2 + 2.75 * 4 / 3])
        mapped_before = mapped.copy()
        np.testing.assert_allclose(transform.inverse_boxes(mapped), boxes)
        np.testing.assert_array_equal(mapped, mapped_before)
        np.testing.assert_array_equal(boxes, before)
        self.assertEqual(transform.forward_boxes(np.empty((0, 4))).shape, (0, 4))
        np.testing.assert_allclose(transform.inverse_boxes([0, 0, 10, 8]), [0, -1.5, 7, 4.5])

    def test_portrait_downscale_and_odd_padding(self) -> None:
        image = np.full((9, 4, 3), (3, 5, 7), dtype=np.uint8)
        output, transform = preprocess(image, (6, 8), color="BGR", fill=99)
        self.assertEqual(transform.resized_size, (6, 3))
        self.assertEqual(transform.padding, (2, 0, 3, 0))
        np.testing.assert_array_equal(output[:, :2], np.full((6, 2, 3), 99))
        np.testing.assert_array_equal(output[:, 2:5], np.full((6, 3, 3), (3, 5, 7)))
        np.testing.assert_array_equal(output[:, 5:], np.full((6, 3, 3), 99))

    def test_extreme_aspect_ratio_keeps_one_pixel(self) -> None:
        output, transform = preprocess(np.zeros((1, 100, 3), dtype=np.uint8), (3, 3))
        self.assertEqual(output.shape, (3, 3, 3))
        self.assertEqual(transform.resized_size, (1, 3))
        np.testing.assert_allclose(transform.forward_boxes([0, 0, 100, 1]), [0, 1, 3, 2])

    def test_bgr_and_pil_rgb_color_conversion(self) -> None:
        bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        rgb = bgr[..., ::-1].copy()
        pil = Image.fromarray(rgb)
        output, _ = preprocess(bgr, (1, 2))
        np.testing.assert_array_equal(output, rgb)
        output, _ = preprocess(pil, (1, 2), color="BGR")
        np.testing.assert_array_equal(output, bgr)
        output, _ = preprocess(pil, (1, 2))
        output[:] = 0
        np.testing.assert_array_equal(np.asarray(pil), rgb)
        output, _ = preprocess(bgr, (1, 2), color="BGR")
        output[:] = 0
        np.testing.assert_array_equal(bgr[..., ::-1], rgb)

    def test_normalization_in_output_channel_order_includes_padding(self) -> None:
        image = np.full((1, 2, 3), (51, 102, 204), dtype=np.uint8)
        output, _ = preprocess(
            image, (3, 2), fill=(10, 20, 30), dtype=np.float32,
            scale=1 / 255, mean=(0.1, 0.2, 0.3), std=(0.5, 0.25, 2.0),
        )
        self.assertEqual(output.dtype, np.float32)
        expected_pixel = (np.array([204, 102, 51]) / 255 - [0.1, 0.2, 0.3]) / [0.5, 0.25, 2]
        expected_fill = (np.array([10, 20, 30]) / 255 - [0.1, 0.2, 0.3]) / [0.5, 0.25, 2]
        np.testing.assert_allclose(output[1, 0], expected_pixel, rtol=1e-6)
        np.testing.assert_allclose(output[0, 0], expected_fill, rtol=1e-6)
        np.testing.assert_allclose(output[2, 1], expected_fill, rtol=1e-6)
        converted, _ = preprocess(image, (1, 2), dtype=np.float64, color="BGR")
        self.assertEqual(converted.dtype, np.float64)
        np.testing.assert_array_equal(converted, image)

    def test_integer_normalization_and_zero_std_are_rejected(self) -> None:
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            preprocess(image, (1, 1), scale=1 / 255)
        with self.assertRaises(ValueError):
            preprocess(image, (1, 1), dtype=np.float32, std=(1, 0, 1))


if __name__ == "__main__":
    unittest.main()
