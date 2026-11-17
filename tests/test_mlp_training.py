"""检查背景采样、训练集标准化和验证阈值的实际数值行为。"""

import contextlib
import io
import unittest

import numpy as np

from robot_cleaning_vision.models.mlp.training import (
    _fit_network, _training_patches, choose_threshold,
)


class TrainingTests(unittest.TestCase):
    def test_backgrounds_avoid_targets_and_share_original_image_budget(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[20:40, 20:40] = (255, 0, 0)
        objects = [{'xmin': 20, 'ymin': 20, 'xmax': 40, 'ymax': 40}]
        proposal = {'scales': [0.25, 0.5], 'aspects': [1.0], 'stride': 0.5}
        samples = _training_patches(image, objects, 10, proposal, np.random.default_rng(42))
        labels = samples['labels']
        self.assertEqual(int((labels == 1).sum()), 1)
        self.assertAlmostEqual(float(samples['weights'][labels == 1].sum()), 1.0)
        self.assertAlmostEqual(float(samples['weights'][labels == 0].sum()), 0.4, places=6)
        for box, label in zip(samples['boxes'], labels):
            if label == 0:
                left, top = np.maximum(box[:2], [19, 19])
                right, bottom = np.minimum(box[2:], [41, 41])
                self.assertFalse(right > left and bottom > top)
        self.assertTrue(np.all(samples['image_ids'] == 10))

    def test_validation_outliers_do_not_enter_training_scaler(self):
        features = np.array([[-2, 4], [-1, 4], [1, 4], [2, 4]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1])
        validation = np.array([[-100, 400], [100, 400]], dtype=np.float32)
        config = {'epochs': 2, 'patience': 3, 'batch_size': 4,
                  'learning_rate': 0.001, 'weight_decay': 0.001}
        with contextlib.redirect_stdout(io.StringIO()):
            _, mean, scale, _, _ = _fit_network(features, labels, np.ones(4), 3,
                                               config, 42, (validation, np.array([0, 1])))
        np.testing.assert_array_equal(mean, [0, 4])
        np.testing.assert_allclose(scale, [np.sqrt(2.5), 1])

    def test_threshold_preserves_recall_and_uses_gap_between_classes(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        threshold = choose_threshold(labels, probabilities, minimum_recall=1.0)
        self.assertGreater(threshold, probabilities[1])
        self.assertLess(threshold, probabilities[2])
        np.testing.assert_array_equal(probabilities >= threshold, labels)


if __name__ == '__main__':
    unittest.main()
