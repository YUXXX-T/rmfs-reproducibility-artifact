import unittest

import numpy as np
import torch

from WorldModel.evaluation.diagnose_analytic_work_shared_efficiency import (
    _eta_prediction,
    _group_eta_targets,
    _oracle_shared_candidate_target,
    _predict_ridge,
    _select_ridge_by_seed,
    fit_global_efficiency,
)


def _row(nominal, true, target_raw, *, seed=1, load="low"):
    nominal = torch.tensor(nominal, dtype=torch.float32)
    true = torch.tensor(true, dtype=torch.float32)
    post = torch.tensor([2.0, 2.0], dtype=torch.float32)
    endpoint = post - true
    return {
        "nominal_station_relief": nominal,
        "true_station_relief": true,
        "current_station_work": post.clone(),
        "post_action_station_work": post,
        "endpoint_station_work": endpoint,
        "target_raw_work_drift": float(target_raw),
        "seed": seed,
        "load": load,
        "group_key": (str(seed), str(target_raw)),
    }


class SharedEfficiencyDiagnosticTests(unittest.TestCase):
    def test_global_efficiency_is_group_equal_weighted(self):
        groups = [
            [
                _row([1.0, 0.0], [0.5, 0.0], -0.5),
                _row([2.0, 0.0], [1.0, 0.0], -1.0),
            ],
            [
                _row([0.0, 3.0], [0.0, 1.5], -1.5),
                _row([0.0, 1.0], [0.0, 0.5], -0.5),
                _row([0.0, 2.0], [0.0, 1.0], -1.0),
            ],
        ]
        result = fit_global_efficiency(groups)
        self.assertAlmostEqual(result["eta"], 0.5, places=7)
        self.assertTrue(result["group_equal_weighting"])

    def test_oracle_group_eta_and_candidate_target_are_separate(self):
        groups = [[
            _row([1.0, 0.0], [0.5, 0.0], -0.80),
            _row([2.0, 0.0], [1.0, 0.0], -1.25),
            _row([3.0, 0.0], [1.5, 0.0], -1.90),
        ]]
        eta, defined = _group_eta_targets(groups)
        self.assertTrue(defined[0])
        self.assertAlmostEqual(eta[0], 0.5, places=7)
        target, audit = _oracle_shared_candidate_target(
            groups,
            eta,
            defined,
            capacity=np.asarray([2.0, 2.0]),
            work_weight=1.0,
        )
        self.assertAlmostEqual(float(target[0].mean()), 0.0, places=12)
        self.assertTrue(audit["group_mean_forced_to_zero"])

    def test_efficiency_prediction_uses_fixed_quadratic_potential(self):
        group = [
            _row([1.0, 0.5], [0.5, 0.25], -0.1),
            _row([0.5, 1.0], [0.25, 0.5], -0.2),
        ]
        raw, endpoint = _eta_prediction(
            group,
            0.5,
            capacity=np.asarray([2.0, 2.0]),
            work_weight=1.0,
        )
        expected_endpoint = np.asarray([
            [1.5, 1.75],
            [1.75, 1.5],
        ])
        self.assertTrue(np.allclose(endpoint, expected_endpoint))
        expected_raw = 0.5 * (
            np.square(expected_endpoint / 2.0).sum(axis=1)
            - np.square(np.asarray([2.0, 2.0]) / 2.0).sum()
        )
        self.assertTrue(np.allclose(raw, expected_raw))

    def test_ridge_alpha_selection_uses_only_training_seed_folds(self):
        features = np.asarray([
            [-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]
        ])
        target = 0.25 + 0.4 * features[:, 0]
        weights = np.ones(len(target))
        seeds = np.asarray([1, 1, 2, 2, 3, 3])
        model, audit = _select_ridge_by_seed(
            features,
            target,
            weights,
            seeds,
            alphas=(1e-6, 1e-3, 1.0),
        )
        predicted = _predict_ridge(model, features)
        self.assertLess(float(np.sqrt(np.mean(np.square(predicted - target)))), 1e-3)
        self.assertTrue(audit["validation_seeds_not_used_for_selection"])
        held_out = {
            fold["held_out_seed"]
            for row in audit["alpha_path"]
            for fold in row["folds"]
        }
        self.assertEqual(held_out, {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
