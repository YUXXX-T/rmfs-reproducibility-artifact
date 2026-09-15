import unittest
from dataclasses import asdict

import torch

from WorldModel.core.lyapunov_head import (
    EndpointDemandPredictor,
    FixedLyapunovFunctional,
    LyapunovComponentHead,
    LyapunovPrediction,
    compute_lyapunov_training_loss,
    target_from_analytic_snapshot,
)
from WorldModel.core.lyapunov import LyapunovSnapshot
from WorldModel.core.lyapunov import LyapunovL0Config
from WorldModel.training.train_lyapunov_head import (
    _component_residual_scales,
    _component_training_support,
    _continuous_group_centered_drift_metrics,
    _demand_residual_scales,
    _load_label,
    _pairwise_drift_accuracy,
    _pairwise_gap_slices,
    _pairwise_rank_loss,
    _resolved_l0_config,
    _validate_disjoint_splits,
    _validate_l0_schema,
)
from WorldModel.training.train_lyapunov_head import _precompute


def _prediction(scale=1.0):
    return LyapunovPrediction(
        station_work=torch.tensor([1.0, 2.0]) * scale,
        station_queue_ratio=torch.tensor([0.5, 0.9]) * scale,
        traffic_excess=torch.tensor([0.4]) * scale,
        stationary_excess=torch.tensor([0.8]) * scale,
        plan_fail_excess=torch.tensor([0.5]) * scale,
        arrival_bins=torch.tensor([[0.0, 2.0], [1.0, 0.0]]) * scale,
    )


class LyapunovComponentHeadTests(unittest.TestCase):
    def test_endpoint_demand_predictor_reencodes_predicted_context(self):
        predictor = EndpointDemandPredictor(
            latent_dim=4,
            encoded_demand_dim=4,
            raw_demand_dim=3,
            hidden_dim=8,
        )
        encoder = torch.nn.Linear(3, 4)
        anchor = torch.tensor([0.2, 0.4, 0.6])
        raw, embedding = predictor.predict_embedding(
            torch.randn(5, 4), torch.randn(4), anchor, encoder
        )
        self.assertEqual(tuple(raw.shape), (3,))
        self.assertEqual(tuple(embedding.shape), (4,))
        self.assertTrue(bool((raw >= 0.0).all()))
        torch.testing.assert_close(raw, anchor)

    def test_failed_demand_gate_falls_back_to_raw_anchor(self):
        predictor = EndpointDemandPredictor(
            latent_dim=2, encoded_demand_dim=2,
            raw_demand_dim=2, hidden_dim=4,
        )
        with torch.no_grad():
            predictor.net[-1].bias.fill_(0.5)
        predictor.set_enabled(False)
        anchor = torch.tensor([0.2, 0.4])
        output = predictor(torch.randn(3, 2), torch.randn(2), anchor)
        torch.testing.assert_close(output, anchor)

    def test_zero_initialised_head_is_exact_analytic_anchor(self):
        head = LyapunovComponentHead(
            latent_dim=4,
            demand_dim=4,
            hidden_dim=8,
            num_stations=2,
            num_arrival_bins=3,
        )
        z = torch.randn(6, 4)
        anchor = LyapunovPrediction(
            station_work=torch.tensor([1.0, 2.0]),
            station_queue_ratio=torch.tensor([0.1, 0.2]),
            traffic_excess=torch.tensor([0.0]),
            stationary_excess=torch.tensor([0.3]),
            plan_fail_excess=torch.tensor([0.1]),
            arrival_bins=torch.tensor([
                [1.0, 0.0, 2.0], [0.0, 1.0, 0.0]
            ]),
        )
        prediction = head(
            z,
            torch.randn(4),
            station_node_ids=[1, 4],
            bottleneck_scores=torch.arange(6, dtype=torch.float32),
            anchor=anchor,
        )
        self.assertEqual(tuple(prediction.station_work.shape), (2,))
        self.assertEqual(tuple(prediction.arrival_bins.shape), (2, 3))
        for value in prediction:
            self.assertTrue(bool((value >= 0.0).all()))
        for predicted, expected in zip(prediction, anchor):
            torch.testing.assert_close(predicted, expected)

    def test_failed_component_gate_falls_back_to_analytic_anchor(self):
        head = LyapunovComponentHead(
            latent_dim=2, demand_dim=2, hidden_dim=4,
            num_stations=1, num_arrival_bins=1,
        )
        with torch.no_grad():
            head.global_net[-1].bias.fill_(0.5)
        head.set_component_gates({
            name: name != "traffic_excess"
            for name in LyapunovPrediction._fields
        })
        anchor = LyapunovPrediction(
            station_work=torch.ones(1),
            station_queue_ratio=torch.ones(1),
            traffic_excess=torch.zeros(1),
            stationary_excess=torch.zeros(1),
            plan_fail_excess=torch.zeros(1),
            arrival_bins=torch.ones(1, 1),
        )
        output = head(
            torch.randn(3, 2), torch.randn(2), [0], anchor=anchor
        )
        torch.testing.assert_close(output.traffic_excess, anchor.traffic_excess)

    def test_snapshot_conversion_uses_explicit_station_order(self):
        snapshot = LyapunovSnapshot(
            tick=0,
            total=0.0,
            components={},
            station_work={2: 3.0, 1: 4.0},
            station_queue_ratio={2: 0.2, 1: 0.1},
            arrival_bins={2: (2.0, 0.0), 1: (1.0, 3.0)},
            chains={},
            traffic_excess_rms=0.4,
            stationary_excess_rms=0.5,
            plan_fail_excess_rms=0.6,
        )
        target = target_from_analytic_snapshot(snapshot, [1, 2])
        torch.testing.assert_close(target.station_work, torch.tensor([4.0, 3.0]))
        torch.testing.assert_close(
            target.arrival_bins, torch.tensor([[1.0, 3.0], [2.0, 0.0]])
        )

    def test_endpoint_signature_requires_matching_demand(self):
        head = LyapunovComponentHead(
            latent_dim=3, demand_dim=2, num_stations=1, num_arrival_bins=2
        )
        anchor = LyapunovPrediction(
            station_work=torch.zeros(1),
            station_queue_ratio=torch.zeros(1),
            traffic_excess=torch.zeros(1),
            stationary_excess=torch.zeros(1),
            plan_fail_excess=torch.zeros(1),
            arrival_bins=torch.zeros(1, 2),
        )
        with self.assertRaisesRegex(ValueError, "e_demand"):
            head(
                torch.randn(4, 3), torch.randn(3), [0], anchor=anchor
            )

    def test_precompute_uses_graph_station_nodes_and_valid_horizon(self):
        class FakeModel:
            def __init__(self):
                self.rollout_station_nodes = None
                self.rollout_horizon = None

            def eval(self):
                return self

            def encode_state(self, node_history, edge_index, edge_features,
                             demand_context):
                nodes = node_history.size(1)
                return (
                    torch.zeros(nodes, 4),
                    torch.zeros(4),
                    torch.zeros(edge_index.size(1), 4),
                )

            def rollout(self, z, e0, edge_attr, action_node, action_global,
                        edge_index, station_node_ids, K):
                self.rollout_station_nodes = list(station_node_ids)
                self.rollout_horizon = K
                return [], None, None, [], z, z + 1.0

        def snapshot(work):
            return {
                "station_work": {1: work, 2: work + 1.0},
                "station_queue_ratio": {1: 0.0, 2: 0.0},
                "arrival_bins": {1: [0.0, 0.0], 2: [0.0, 0.0]},
                "traffic_excess_rms": 0.0,
                "stationary_excess_rms": 0.0,
                "plan_fail_excess_rms": 0.0,
            }

        model = FakeModel()
        sample = {
            "node_history": torch.zeros(4, 8, 10),
            "edge_index": torch.tensor([[0, 1], [1, 2]]),
            "edge_features": torch.zeros(2, 6),
            "demand_context": torch.zeros(9),
            "action_node": torch.zeros(8, 8),
            "action_global": torch.zeros(6),
            "station_node_ids": torch.tensor([5, 7]),
            "future_mask": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "future_demand_context": torch.zeros(9),
            "lyapunov_l0_start": snapshot(1.0),
            "lyapunov_l0_end": snapshot(0.5),
            "lyapunov_l0_progress": {
                "arrivals_by_station": {},
                "productive_by_station": {1: 0.5},
                "reverse_by_station": {},
                "replan_residual_by_station": {},
            },
        }
        rows = _precompute(model, [sample], [1, 2], "cpu")
        self.assertEqual(model.rollout_station_nodes, [5, 7])
        self.assertEqual(model.rollout_horizon, 2)
        self.assertEqual(rows[0]["station_node_ids"].tolist(), [5, 7])
        torch.testing.assert_close(rows[0]["d0_raw"], torch.zeros(9))


class FixedLyapunovFunctionalTests(unittest.TestCase):
    def setUp(self):
        self.functional = FixedLyapunovFunctional(
            work_capacity=[1.0, 2.0],
            arrival_capacity=[[1.0, 1.0], [1.0, 1.0]],
            station_safe_ratio=0.7,
        )

    def test_fixed_functional_is_non_negative_and_auditable(self):
        values = self.functional(_prediction())
        self.assertEqual(
            set(values), {"work", "station", "traffic", "stall", "arrival", "total"}
        )
        self.assertTrue(bool(values["total"] >= 0.0))
        torch.testing.assert_close(
            values["total"],
            sum(values[name] for name in self.functional.COMPONENT_NAMES),
        )

    def test_negative_physical_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "work_weight"):
            FixedLyapunovFunctional(
                work_capacity=[1.0],
                arrival_capacity=[[1.0]],
                work_weight=-1.0,
            )

    def test_exact_prediction_has_zero_supervision_loss(self):
        state = _prediction()
        losses = compute_lyapunov_training_loss(
            state,
            state,
            state,
            state,
            self.functional,
            balance_target=torch.tensor(0.0),
        )
        torch.testing.assert_close(losses["total"], torch.tensor(0.0))

    def test_drift_direction_error_is_penalised(self):
        current_true = _prediction(1.0)
        future_true = _prediction(1.5)
        current_pred = _prediction(1.0)
        future_pred = _prediction(0.5)
        losses = compute_lyapunov_training_loss(
            current_pred,
            future_pred,
            current_true,
            future_true,
            self.functional,
        )
        self.assertGreater(float(losses["sign"]), 0.0)


class LyapunovTrainingSchemaTests(unittest.TestCase):
    @staticmethod
    def _sample(work_capacity=2.0, config=None):
        snapshot = {
            "station_work": {1: 1.0, 2: 2.0},
            "work_capacity": work_capacity,
            "arrival_capacity": {1: [1.0, 2.0], 2: [1.0, 2.0]},
        }
        return {
            "lyapunov_l0_start": dict(snapshot),
            "lyapunov_l0_end": dict(snapshot),
            "lyapunov_l0_config": config or {},
        }

    def test_schema_treats_explicit_defaults_like_implicit_defaults(self):
        implicit = self._sample()
        explicit = self._sample(config=asdict(LyapunovL0Config()))
        _validate_l0_schema(
            [implicit, explicit],
            station_ids=[1, 2],
            reference_config=_resolved_l0_config(implicit),
            reference_work_capacity=[2.0, 2.0],
            reference_arrival_capacity=[[1.0, 2.0], [1.0, 2.0]],
            split_name="train",
        )

    def test_schema_rejects_changed_work_normalisation(self):
        reference = self._sample()
        changed = self._sample(work_capacity=3.0)
        with self.assertRaisesRegex(ValueError, "work_capacity changed"):
            _validate_l0_schema(
                [reference, changed],
                station_ids=[1, 2],
                reference_config=_resolved_l0_config(reference),
                reference_work_capacity=[2.0, 2.0],
                reference_arrival_capacity=[[1.0, 2.0], [1.0, 2.0]],
                split_name="train",
            )

    def test_pairwise_drift_accuracy_skips_target_ties(self):
        accuracy, pairs = _pairwise_drift_accuracy({
            ("seed201", "group-a"): [
                (0.1, 0.2),
                (0.8, 0.9),
                (0.7, 0.9),  # tied target: this pair is not scored
            ],
        })
        self.assertEqual(pairs, 2)
        self.assertEqual(accuracy, 1.0)

    def test_pairwise_rank_loss_ignores_sub_practical_gaps(self):
        predicted = [torch.tensor(0.0), torch.tensor(0.2), torch.tensor(0.4)]
        target = [0.0, 0.05, 1.0]
        loss, pairs = _pairwise_rank_loss(
            predicted, target, target_margin=0.1
        )
        self.assertEqual(pairs, 2)
        self.assertGreaterEqual(float(loss), 0.0)

    def test_pairwise_gap_slices_report_practical_thresholds(self):
        report = _pairwise_gap_slices({
            "g": [(0.0, 0.0), (0.2, 0.2), (1.0, 1.0)]
        })
        self.assertEqual(set(report), {"0", "0.1", "0.5"})
        self.assertEqual(report["0.5"]["pairs"], 2)

    def test_group_centered_drift_metrics_are_scale_invariant(self):
        records = [
            {
                "group_key": "g1", "predicted_drift": 1.0,
                "target_drift": 2.0,
            },
            {
                "group_key": "g1", "predicted_drift": 3.0,
                "target_drift": 4.0,
            },
            {
                "group_key": "g2", "predicted_drift": -2.0,
                "target_drift": -1.0,
            },
            {
                "group_key": "g2", "predicted_drift": 2.0,
                "target_drift": 3.0,
            },
        ]
        original = _continuous_group_centered_drift_metrics(records)
        scaled = _continuous_group_centered_drift_metrics([
            {
                **row,
                "predicted_drift": 7.0 * row["predicted_drift"],
                "target_drift": 7.0 * row["target_drift"],
            }
            for row in records
        ])
        self.assertAlmostEqual(
            original["normalized_rmse"], scaled["normalized_rmse"]
        )
        self.assertAlmostEqual(original["pearson"], scaled["pearson"])
        self.assertEqual(
            original["all_non_tie_pair_concordance"],
            scaled["all_non_tie_pair_concordance"],
        )

    def test_residual_scales_are_positive_for_zero_variance_component(self):
        row = {
            "current_target": _prediction(1.0),
            "future_target": _prediction(1.0),
            "d0_raw": torch.zeros(3),
            "future_demand": torch.zeros(3),
        }
        component = _component_residual_scales([row], floor=0.01)
        demand = _demand_residual_scales([row], floor=0.01)
        self.assertTrue(all(value >= 0.01 for value in component.values()))
        torch.testing.assert_close(demand, torch.full((3,), 0.01))

    def test_zero_delta_component_is_hard_masked_from_training(self):
        current = _prediction(1.0)
        future = LyapunovPrediction(
            station_work=current.station_work + 1.0,
            station_queue_ratio=current.station_queue_ratio,
            traffic_excess=current.traffic_excess,
            stationary_excess=current.stationary_excess,
            plan_fail_excess=current.plan_fail_excess,
            arrival_bins=current.arrival_bins,
        )
        support = _component_training_support([{
            "current_target": current,
            "future_target": future,
        }])
        self.assertTrue(support["station_work"]["active"])
        self.assertFalse(support["traffic_excess"]["active"])

    def test_load_label_is_recovered_from_run_or_path(self):
        self.assertEqual(_load_label({"run_id": "l0_mid_seed301"}), "mid")
        self.assertEqual(
            _load_label({"_source_path": "/tmp/l0_high_seed302/data.pt"}),
            "high",
        )

    def test_split_rejects_seed_leakage(self):
        train = [{"simulation_seed": 201, "_source_path": "train.pt"}]
        val = [{"simulation_seed": 201, "_source_path": "val.pt"}]
        with self.assertRaisesRegex(ValueError, "overlap"):
            _validate_disjoint_splits(train, val)


if __name__ == "__main__":
    unittest.main()
