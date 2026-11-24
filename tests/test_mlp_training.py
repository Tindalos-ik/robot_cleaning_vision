"""检查区域采样、训练集标准化和 FAIL 损失加权。"""

import contextlib
import io
import unittest

import numpy as np
import torch

from robot_cleaning_vision.models.mlp.training import (
    _fit_network, _training_patches,
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

    def test_fail_loss_weight_changes_scores(self):
        """同一特征对应两类时，最优分数等于该类在损失中的权重占比。"""
        features = np.zeros((2, 1), dtype=np.float32)
        labels = np.array([0, 1])
        config = {'epochs': 70, 'batch_size': 2, 'learning_rate': 0.06, 'weight_decay': 0.0}
        scores = []
        with contextlib.redirect_stdout(io.StringIO()):
            for weight in (1.0, 3.0):
                network, *_ = _fit_network(features, labels, np.ones(2), 1,
                                           dict(config, fail_weight=weight), 42)
                with torch.inference_mode():
                    scores.append(network(torch.from_numpy(features[:1])).sigmoid().item())
        np.testing.assert_allclose(scores, [0.5, 0.75], atol=0.01)


if __name__ == '__main__':
    unittest.main()
