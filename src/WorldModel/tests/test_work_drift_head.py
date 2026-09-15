import inspect
import tempfile
import unittest

import torch

from WorldModel.core.work_drift_head import (
    SCHEMA_VERSION,
    WorkDriftHead,
    WorkDriftPrediction,
    all_non_tie_pairwise_logistic_loss,
    group_range_normalise,
)
from WorldModel.evaluation.work_drift_layer4_protocol import (
    formal_layer4_protocol,
)
from WorldModel.evaluation.evaluate_work_drift_head import _metric_contract
from WorldModel.evaluation.evaluate_lyapunov_validity import (
    evaluate_drift_prediction,
)
from WorldModel.training.train_work_drift_head import evaluate_rows


class WorkDriftHeadTests(unittest.TestCase):
    LAYER4_PROTOCOL_SHA256 = (
        "external-fingerprint-omitted"
    )

    def test_group_range_is_shift_and_positive_scale_invariant(self):
        values = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        expected = group_range_normalise(values)
        self.assertTrue(torch.allclose(
            expected,
            group_range_normalise(100.0 + 17.0 * values),
            atol=1e-6,
        ))
        self.assertTrue(torch.equal(
            group_range_normalise(torch.ones(4)),
            torch.zeros(4),
        ))

    def test_pairwise_loss_uses_all_non_ties_without_gap(self):
        target = torch.tensor([0.0, 1e-8, 2e-8])
        correct = torch.tensor([0.0, 1.0, 2.0])
        reversed_order = -correct
        good, good_pairs = all_non_tie_pairwise_logistic_loss(correct, target)
        bad, bad_pairs = all_non_tie_pairwise_logistic_loss(
            reversed_order, target
        )
        self.assertEqual(good_pairs, 3)
        self.assertEqual(bad_pairs, 3)
        self.assertLess(float(good), float(bad))

    def test_endpoint_work_reconstructs_raw_analytic_drift(self):
        head = WorkDriftHead(
            latent_dim=4,
            hidden_dim=8,
            num_stations=2,
            work_capacity=[2.0, 4.0],
            work_weight=1.0,
            residual_scale=[1.0, 1.0],
        )
        global_context = torch.randn(3, head.global_dim)
        station_context = torch.randn(3, 2, head.station_dim)
        current = torch.tensor([[2.0, 4.0], [1.0, 2.0], [0.0, 1.0]])
        prediction = head.predict_from_features(
            global_context, station_context, current
        )
        expected = (
            head.work_potential(prediction.endpoint_station_work)
            - head.work_potential(current)
        )
        self.assertEqual(prediction.endpoint_station_work.shape, (3, 2))
        self.assertTrue(torch.all(prediction.endpoint_station_work >= 0.0))
        self.assertTrue(torch.allclose(prediction.raw_work_drift, expected))

    def test_action_is_not_a_direct_head_input(self):
        signature = inspect.signature(WorkDriftHead.build_features)
        self.assertNotIn("action_embedding", signature.parameters)
        self.assertEqual(formal_layer4_protocol()["head"][
            "direct_action_embedding_to_head"
        ], False)
        self.assertEqual(
            formal_layer4_protocol()["protocol_sha256"],
            self.LAYER4_PROTOCOL_SHA256,
        )

    def test_checkpoint_round_trip(self):
        head = WorkDriftHead(
            latent_dim=4,
            hidden_dim=8,
            num_stations=2,
            work_capacity=[2.0, 3.0],
            residual_scale=[0.5, 0.75],
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "head_config": head.checkpoint_config(),
            "head_state_dict": head.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/head.pt"
            torch.save(payload, path)
            loaded, restored = WorkDriftHead.from_checkpoint(path)
        self.assertEqual(restored["schema_version"], SCHEMA_VERSION)
        self.assertEqual(loaded.checkpoint_config(), head.checkpoint_config())


class _OracleHead:
    def eval(self):
        return self

    def predict_from_features(self, global_context, station_context, current):
        return WorkDriftPrediction(
            endpoint_station_work=current,
            raw_work_drift=global_context[:, 0],
        )


class WorkDriftMetricsTests(unittest.TestCase):
    def test_oracle_group_range_metrics_are_perfect(self):
        rows = []
        for seed in (1, 2, 3):
            for group_index in range(2):
                group = (f"run{seed}", f"group{group_index}")
                for candidate, drift in enumerate((-2.0, -0.5, 1.0, 3.0)):
                    global_context = torch.zeros(20)
                    global_context[0] = drift
                    rows.append({
                        "global_context": global_context,
                        "station_context": torch.zeros(2, 9),
                        "current_station_work": torch.ones(2),
                        "endpoint_station_work": torch.ones(2),
                        "target_raw_work_drift": drift,
                        "group_key": group,
                        "seed": seed,
                        "load": ("low", "mid", "high")[seed - 1],
                        "candidate_key": str(candidate),
                    })
        report = evaluate_rows(
            _OracleHead(),
            rows,
            device="cpu",
            bootstrap_repeats=30,
            random_seed=7,
        )
        metric = report["continuous_group_range"]
        self.assertAlmostEqual(metric["normalised_rmse"], 0.0)
        self.assertAlmostEqual(metric["spearman"], 1.0)
        self.assertAlmostEqual(metric["all_non_tie_pair_concordance"], 1.0)
        self.assertAlmostEqual(metric["top1_min_drift_accuracy"], 1.0)
        self.assertGreater(
            metric[
                "normalised_selection_regret_improvement_over_uniform_random"
            ]["mean"],
            0.0,
        )
        self.assertIn(
            "normalised_selection_regret_improvement_over_uniform_random_"
            "ci95_seed_cluster_bootstrap",
            metric,
        )
        self.assertIn("nonzero_q1", report["raw_range_strata"]["strata"])
        self.assertEqual(report["independent_seed_clusters"], 3)

    def test_five_layer_adapter_requires_all_held_out_contracts(self):
        learned = {
            "schema_version": "work_drift_group_range_evaluation_v1",
            "status": "WORLD_MODEL_WORK_GROUP_RANGE_ESTIMATION_SUPPORTED",
            "supported": True,
            "checkpoint_contract": {"passed": True},
            "data_contract": {"passed": True},
            "metric_contract": {"passed": True},
            "validation": {
                "continuous_group_range": {
                    "normalised_rmse": 0.5,
                    "spearman": 0.8,
                    "all_non_tie_pair_concordance": 0.7,
                }
            },
        }
        layer = evaluate_drift_prediction(
            [], predicted_drift_field=None, learned_report=learned
        )
        self.assertTrue(layer["supported"])
        self.assertEqual(
            layer["status"], "WORLD_MODEL_DRIFT_ESTIMATION_SUPPORTED"
        )
        learned["data_contract"]["passed"] = False
        layer = evaluate_drift_prediction(
            [], predicted_drift_field=None, learned_report=learned
        )
        self.assertFalse(layer["supported"])
        self.assertEqual(
            layer["status"], "WORLD_MODEL_DRIFT_ESTIMATION_INCONCLUSIVE"
        )

    def test_formal_metric_contract_includes_load_and_seed_robustness(self):
        rows = []
        for seed in range(421, 431):
            for load in ("low", "mid", "high"):
                for group_index in range(2):
                    group = (f"run_{load}_{seed}", f"group{group_index}")
                    for candidate, drift in enumerate((-2.0, -0.5, 1.0, 3.0)):
                        global_context = torch.zeros(20)
                        global_context[0] = drift
                        rows.append({
                            "global_context": global_context,
                            "station_context": torch.zeros(2, 9),
                            "current_station_work": torch.ones(2),
                            "endpoint_station_work": torch.ones(2),
                            "target_raw_work_drift": drift,
                            "group_key": group,
                            "seed": seed,
                            "load": load,
                            "candidate_key": str(candidate),
                        })
        metrics = evaluate_rows(
            _OracleHead(), rows, device="cpu",
            bootstrap_repeats=50, random_seed=11,
        )
        contract = _metric_contract(metrics)
        self.assertTrue(contract["passed"])
        self.assertTrue(contract["guardrails_passed"])
        self.assertEqual(contract["supported_seed_clusters"], 10)

        for row in rows:
            if row["load"] == "high":
                row["global_context"][0] *= -1.0
        heterogeneous = _metric_contract(evaluate_rows(
            _OracleHead(), rows, device="cpu",
            bootstrap_repeats=50, random_seed=11,
        ))
        self.assertFalse(
            heterogeneous["checks"][
                "all_required_loads_directionally_supported"
            ]
        )
        self.assertFalse(heterogeneous["guardrails_passed"])


if __name__ == "__main__":
    unittest.main()
