"""覆盖分类边界、检测筛选、坐标还原与无目标语义。"""

import unittest

import numpy as np

from robot_cleaning_vision.postprocessing import (
    postprocess_classification,
    postprocess_detection,
)
from robot_cleaning_vision.preprocessing import preprocess


class PostprocessingTests(unittest.TestCase):
    def test_classification_threshold_and_not_applicable_fields(self) -> None:
        for probability, decision in [(0.0, "PASS"), (0.69, "PASS"),
                                      (0.7, "FAIL"), (1.0, "FAIL")]:
            with self.subTest(probability=probability):
                result = postprocess_classification(probability, threshold=0.7)
                self.assertEqual(result.decision, decision)
                self.assertEqual(result.image_score, probability)
                self.assertIsNone(result.objects)
                self.assertIsNone(result.counts)
        self.assertEqual(postprocess_classification(0, threshold=0).decision, "FAIL")
        self.assertEqual(postprocess_classification(1, threshold=1).decision, "FAIL")

    def test_detection_restores_filters_and_preserves_overlapping_objects(self) -> None:
        _, transform = preprocess(np.zeros((3, 7, 3), dtype=np.uint8), (8, 10))
        original_boxes = np.array([[0, 0, 7, 3], [0, 0, 7, 3],
                                   [1, 0.5, 6, 2], [-1, -1, 8, 4]])
        boxes = transform.forward_boxes(original_boxes)
        class_ids = np.array([2, 2, 5, 1])
        scores = np.array([0.8, 0.5, 0.49, 1.0])
        before = (boxes.copy(), class_ids.copy(), scores.copy())
        result = postprocess_detection(boxes, class_ids, scores, transform, threshold=0.5)
        self.assertEqual(result.decision, "FAIL")
        self.assertIsNone(result.image_score)
        self.assertEqual(result.counts, {1: 1, 2: 2, 3: 0, 4: 0, 5: 0})
        self.assertEqual([obj.class_id for obj in result.objects], [2, 2, 1])
        self.assertEqual([obj.score for obj in result.objects], [0.8, 0.5, 1.0])
        np.testing.assert_allclose([obj.bbox for obj in result.objects], original_boxes[[0, 1, 3]])
        for actual, expected in zip((boxes, class_ids, scores), before):
            np.testing.assert_array_equal(actual, expected)

    def test_empty_and_all_filtered_detection_are_pass_with_zero_counts(self) -> None:
        _, transform = preprocess(np.zeros((9, 4, 3), dtype=np.uint8), (6, 8))
        for boxes, ids, scores in [(np.empty((0, 4)), [], []),
                                    ([[0, 0, 1, 1]], [5], [0.2])]:
            with self.subTest(boxes=boxes):
                result = postprocess_detection(boxes, ids, scores, transform, threshold=0.3)
                self.assertEqual(result.decision, "PASS")
                self.assertIsNone(result.image_score)
                self.assertEqual(result.objects, [])
                self.assertEqual(result.counts, {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})

    def test_invalid_probabilities_and_detection_inputs_are_rejected(self) -> None:
        for invalid in [-0.01, 1.01, float("nan"), float("inf")]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    postprocess_classification(invalid, threshold=0.5)
                with self.assertRaises(ValueError):
                    postprocess_classification(0.5, threshold=invalid)
        _, transform = preprocess(np.zeros((2, 2, 3), dtype=np.uint8), (2, 2))
        invalid_inputs = [
            ([0, 0, 1, 1], [1], [0.9]),
            ([[0, 0, 1, 1]], [], [0.9]),
            ([[0, 0, 1, 1]], [1], [[0.9]]),
            ([[0, 0, 0, 1]], [1], [0.9]),
            ([[0, 0, float("nan"), 1]], [1], [0.9]),
            ([[0, 0, 1, 1]], [0], [0.9]),
            ([[0, 0, 1, 1]], [6], [0.9]),
            ([[0, 0, 1, 1]], [1.5], [0.9]),
            ([[0, 0, 1, 1]], [True], [0.9]),
            ([[0, 0, 1, 1]], [1], [float("nan")]),
            ([[0, 0, 1, 1]], [1], [1.1]),
        ]
        for boxes, ids, scores in invalid_inputs:
            with self.subTest(boxes=boxes, ids=ids, scores=scores):
                with self.assertRaises(ValueError):
                    postprocess_detection(boxes, ids, scores, transform, threshold=0.5)
        with self.assertRaises(ValueError):
            postprocess_detection([], [], [], transform, threshold=float("nan"))


if __name__ == "__main__":
    unittest.main()
