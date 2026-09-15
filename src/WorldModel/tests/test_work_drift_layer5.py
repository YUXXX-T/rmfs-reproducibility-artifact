import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from Policies.TaskAssigner.WorldModelTaskAssigner.world_model_task_assigner import (
    WorldModelTaskAssigner,
)
from WorldModel.core.model import RMFSWorldModel
from WorldModel.evaluation.validate_work_drift_layer5 import (
    _load_online_rows,
    _metric_report,
    _trajectory_audit,
)
from WorldModel.evaluation.run_work_drift_layer5 import (
    BASELINE_LABEL,
    SCHEMA_VERSION as ONLINE_SCHEMA_VERSION,
    WORK_LABEL,
)
from WorldModel.evaluation.evaluate_online_v6 import _order_arrival_manifest
from Policies.OrderGenerator.RecordedOrderGenerator.recorded_order_generator import (
    RecordedOrderGenerator,
)
from WorldState.order_state import Order, OrderState
from WorldModel.evaluation.work_drift_layer5_protocol import (
    CERTIFICATION_SEEDS,
    EXPECTED_LAYER3_PROTOCOL_SHA256,
    EXPECTED_FORMAL_PROTOCOL_SHA256,
    FROZEN_LAMBDA,
    TICKS,
    derive_frozen_lambda,
    formal_protocol,
    validate_exact_seed_set,
)


def _synthetic_layer3_report():
    folds = []
    for fold, seed in enumerate(range(421, 431)):
        folds.append({
            "fold": fold,
            "held_out_clusters": [f"seed={seed}"],
            "standardised_coefficients": {
                "wm": 0.25,
                "candidate:work_group_range": 0.60 + 0.001 * fold,
            },
            "feature_ranges": {
                "wm": {"train_standard_deviation": 0.15},
                "candidate:work_group_range": {
                    "train_standard_deviation": 0.33
                },
            },
        })
    return {
        "layer3_incremental_information": {
            "candidate_component": "work_group_range",
            "supported": True,
            "primary_outcome": "realized_cost",
            "formal_candidate_contract": {
                "passed": True,
                "protocol": {
                    "protocol_sha256": EXPECTED_LAYER3_PROTOCOL_SHA256
                },
            },
            "independent_outcomes": {
                "realized_cost": {
                    "wm_plus_candidate_fold_diagnostics": folds,
                }
            },
        }
    }


class WorkDriftLayer5ProtocolTests(unittest.TestCase):
    def test_protocol_is_deterministic_and_freezes_conservative_rule(self):
        first = formal_protocol()
        second = formal_protocol()
        self.assertEqual(first, second)
        self.assertEqual(
            first["protocol_sha256"], EXPECTED_FORMAL_PROTOCOL_SHA256
        )
        self.assertEqual(first["integration"]["lambda"], 0.25)
        self.assertIn("group_range(wm_raw_cost)", first["integration"]["formula"])
        self.assertFalse(first["integration"]["hard_raw_gap_gate"])
        self.assertFalse(first["integration"]["load_gate"])
        self.assertEqual(
            first["world_model_semantics"]["feature_reservation_window"], 10
        )
        self.assertEqual(
            tuple(first["formal_test"]["seeds"]), CERTIFICATION_SEEDS
        )

    def test_lambda_audit_checks_direction_but_does_not_fit_magnitude(self):
        audit = derive_frozen_lambda(_synthetic_layer3_report())
        self.assertEqual(audit["value"], FROZEN_LAMBDA)
        self.assertFalse(audit["uses_layer3_coefficient_magnitude"])
        self.assertFalse(audit["uses_layer5_online_outcomes"])
        self.assertEqual(len(audit["fold_work_coefficients_direction_audit"]), 10)

    def test_exact_certification_seed_set(self):
        validate_exact_seed_set(list(reversed(CERTIFICATION_SEEDS)))
        with self.assertRaises(ValueError):
            validate_exact_seed_set(CERTIFICATION_SEEDS[:-1])


class WorkDriftLayer5ScoringTests(unittest.TestCase):
    def test_group_range_fusion_preserves_wm_order_at_zero_aux_and_is_no_gap(self):
        assigner = WorldModelTaskAssigner()
        assigner.work_drift_mode = "group_range_additive"
        assigner.work_drift_lambda = 0.25
        assigner.stats.update({
            "work_drift_contexts": 0,
            "work_drift_candidates": 0,
            "work_drift_max_candidate_count": 0,
            "work_drift_candidate_superset_contexts": 0,
            "work_drift_exact_tie_contexts": 0,
            "work_drift_wm_exact_tie_contexts": 0,
            "work_drift_modified_decisions": 0,
        })
        rows = [
            {"score": 1.0, "work_drift_raw": 5.0000000, "best_robot": 1},
            {"score": 2.0, "work_drift_raw": 5.0000001, "best_robot": 2},
            {"score": 3.0, "work_drift_raw": 5.0000002, "best_robot": 3},
        ]
        assigner._apply_work_drift_group_range(rows)
        self.assertAlmostEqual(rows[0]["work_drift_wm_group_range"], -0.5)
        self.assertAlmostEqual(rows[2]["work_drift_wm_group_range"], 0.5)
        self.assertAlmostEqual(rows[0]["work_drift_group_range"], -0.5)
        self.assertAlmostEqual(rows[2]["work_drift_group_range"], 0.5)
        self.assertEqual(assigner.stats["work_drift_exact_tie_contexts"], 0)

    def test_work_mode_rejects_other_decision_mechanisms(self):
        with self.assertRaises(ValueError):
            WorldModelTaskAssigner(
                work_drift_mode="group_range_additive",
                work_drift_head_path="head.pt",
                work_drift_layer4_report_path="layer4.json",
                work_drift_lambda=0.0,
            )
        with self.assertRaises(ValueError):
            WorldModelTaskAssigner(
                work_drift_mode="group_range_additive",
                work_drift_head_path="head.pt",
                work_drift_layer4_report_path="layer4.json",
                candidate_context_mode="stratified",
            )

    def test_continued_latent_matches_direct_rollout_endpoint(self):
        torch.manual_seed(7)
        model = RMFSWorldModel(
            node_feat_dim=10,
            edge_feat_dim=6,
            demand_dim=9,
            action_node_dim=8,
            action_global_dim=6,
            hidden_dim=8,
            num_spatial_layers=1,
            rollout_horizon=3,
            num_stations=2,
        ).eval()
        nodes = 5
        z = torch.randn(nodes, 8)
        demand = torch.randn(8)
        edge_attr = torch.randn(8, 8)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 0, 2, 4],
            [1, 2, 3, 4, 0, 2, 4, 1],
        ])
        action_node = torch.randn(nodes, 8)
        action_global = torch.randn(6)
        with torch.no_grad():
            *_, z3 = model.rollout(
                z,
                demand,
                edge_attr,
                action_node,
                action_global,
                edge_index,
                station_node_ids=[0, 4],
                K=3,
            )
            *_, z10 = model.rollout(
                z,
                demand,
                edge_attr,
                action_node,
                action_global,
                edge_index,
                station_node_ids=[0, 4],
                K=10,
            )
            continued = model.continue_latent_rollout(
                z3,
                demand,
                edge_attr,
                action_node,
                action_global,
                edge_index,
                start_step=3,
                end_step=10,
            )
        self.assertTrue(torch.equal(continued, z10))


class WorkDriftLayer5StatisticsTests(unittest.TestCase):
    def test_paired_metric_clusters_across_loads_by_seed(self):
        rows = []
        for seed in (431, 432):
            for load in ("low", "mid", "high"):
                rows.append({
                    "seed": seed,
                    "load": load,
                    "baseline": {"wm_label_cost": 2.0},
                    "work": {"wm_label_cost": 1.5},
                })
        report = _metric_report(
            rows,
            "wm_label_cost",
            direction="lower",
            relative=False,
            repeats=100,
            seed=1,
        )
        self.assertEqual(report["overall"]["n"], 2)
        self.assertAlmostEqual(report["overall"]["mean"], -0.5)
        self.assertTrue(report["favourable_point_estimate"])

    def test_formal_online_and_trajectory_audits_require_exact_matrix(self):
        protocol = formal_protocol()
        paths = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for load in ("low", "mid", "high"):
                per_seed = {}
                for seed in CERTIFICATION_SEEDS:
                    common = {
                        "online_robot_candidate_scope": "all_idle",
                        "fallback_greedy_calls": 0,
                        "local_greedy_compared": 0,
                        "wm_label_cost": 1.0,
                        "completed_orders": 10,
                        "avg_excess_delay": 1.0,
                        "open_order_count": 3,
                        "wait_or_stall": 0.1,
                        "unified_risk": 0.1,
                        "deadlock_ratio_max": 0.0,
                        "order_arrival_manifest_schema_version": (
                            "layer5_order_arrival_manifest_v1"
                        ),
                        "order_arrival_manifest_sha256": f"orders-{load}-{seed}",
                        "order_arrival_count": 20,
                        "order_arrival_replayed": False,
                    }
                    work = dict(common)
                    work.update({
                        "wm_label_cost": 0.9,
                        "work_drift_mode": "group_range_additive",
                        "work_drift_lambda": FROZEN_LAMBDA,
                        "work_drift_contexts": 2,
                        "work_drift_max_candidate_count": 48,
                        "work_drift_candidate_superset_contexts": 2,
                        "work_drift_modified_decisions": 1,
                        "order_arrival_replayed": True,
                    })
                    per_seed[str(seed)] = {
                        BASELINE_LABEL: dict(common),
                        WORK_LABEL: work,
                    }
                payload = {
                    "schema_version": ONLINE_SCHEMA_VERSION,
                    "meta": {
                        "formal": True,
                        "load": load,
                        "ticks": TICKS,
                        "seeds": list(CERTIFICATION_SEEDS),
                        "paired_arms": [BASELINE_LABEL, WORK_LABEL],
                        "layer5_protocol_sha256": protocol["protocol_sha256"],
                    },
                    "per_seed": per_seed,
                }
                path = root / f"{load}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths.append(path)
            rows, audit = _load_online_rows(paths, protocol)
            self.assertTrue(audit["passed"])
            self.assertEqual(len(rows), 30)

        trajectory = {
            "schema_version": "lyapunov_closed_loop_validation_v2",
            "semantics": {
                "five_layer_role": "layer5_normal_arrival_closed_loop_stability"
            },
            "groups": {
                f"{load}|{WORK_LABEL}": {"seeds": list(CERTIFICATION_SEEDS)}
                for load in ("low", "mid", "high")
            },
            "verdict": "PASS_CLOSED_LOOP_STABILITY",
            "passed": True,
        }
        self.assertTrue(_trajectory_audit(trajectory, protocol)["passed"])

    def test_frozen_order_manifest_replays_identical_ticks_and_payloads(self):
        original_next_id = Order._next_id
        try:
            Order._next_id = 0
            source_state = OrderState()
            source_state.add_order(Order({"A": 2}, station_id=1, created_at=0))
            source_state.add_order(Order({"B": 1}, station_id=3, created_at=5))
            source_engine = SimpleNamespace(
                world=SimpleNamespace(order_state=source_state)
            )
            source_manifest = _order_arrival_manifest(source_engine)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "orders.json"
                path.write_text(json.dumps(source_manifest), encoding="utf-8")
                Order._next_id = 0
                replay_state = OrderState()
                replay_world = SimpleNamespace(tick=0, order_state=replay_state)
                generator = RecordedOrderGenerator(
                    recorded_orders_path=str(path), immediate_dispatch=False
                )
                for tick in range(6):
                    replay_world.tick = tick
                    for order in generator.generate(replay_world):
                        replay_state.add_order(order)
                replay_engine = SimpleNamespace(world=replay_world)
                replay_manifest = _order_arrival_manifest(replay_engine)
            self.assertEqual(
                source_manifest["manifest_sha256"],
                replay_manifest["manifest_sha256"],
            )
            self.assertEqual(source_manifest["orders"], replay_manifest["orders"])
        finally:
            Order._next_id = original_next_id


if __name__ == "__main__":
    unittest.main()
