import copy
import unittest
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from WorldModel.data.build_td_stream_tuples import assemble_run
from WorldModel.model import TDRiskVHead
from WorldModel.training.train_td_risk_v import (
    _component_gate_decisions,
    _discounted_mc_average,
    _discounted_prefix,
    _mc_anchor_eligible,
    _split_samples,
    _series_gate_metrics,
    _temporal_direction_metrics,
    _validation_breakdowns,
    spearman_tie_safe,
    validate_stream_checkpoint,
)


class TDRiskVHeadTests(unittest.TestCase):
    @staticmethod
    def _passing_metric():
        checks = {
            name: True for name in TDRiskVHead.MULTIMETRIC_REQUIRED_CHECKS
        }
        return {
            "n": 40,
            "passed": True,
            "checks": checks,
            "failed_checks": [],
        }

    @classmethod
    def _v4_payload(cls, gates=(True, False, True)):
        head = TDRiskVHead(latent_dim=3, demand_dim=3, hidden_dim=5)
        head.set_component_gates(gates)
        gate_map = dict(zip(TDRiskVHead.RISK_COMPONENTS, gates))
        decisions = {}
        for name, enabled in gate_map.items():
            if enabled:
                decisions[name] = {
                    "enabled": True,
                    "fallback": None,
                    "gate_schema_version": (
                        TDRiskVHead.MULTIMETRIC_GATE_SCHEMA
                    ),
                    "aggregate": cls._passing_metric(),
                    "require_all_units": True,
                    "units": {
                        "seed:1": cls._passing_metric(),
                        "arm:A1": cls._passing_metric(),
                    },
                    "reasons": [],
                }
            else:
                decisions[name] = {
                    "enabled": False,
                    "fallback": "current_raw_risk",
                    "gate_schema_version": (
                        TDRiskVHead.MULTIMETRIC_GATE_SCHEMA
                    ),
                    "aggregate": {"passed": False},
                    "require_all_units": True,
                    "units": {},
                    "reasons": ["not_certified"],
                }
        return head, {
            "schema_version": "td_risk_residual_vhead_v4",
            "gate_schema_version": TDRiskVHead.MULTIMETRIC_GATE_SCHEMA,
            "td_risk_vhead": head.state_dict(),
            "td_risk_vhead_ema": head.state_dict(),
            "component_gate_decisions": decisions,
            "config": {
                "latent_dim": 3,
                "demand_dim": 3,
                "hidden_dim": 5,
                "physical_summary_dim": 0,
                "bottleneck_fraction": 0.2,
                "residual_scale": 1.0,
                "component_gates": gate_map,
                "component_gate_rule": {
                    "schema_version": TDRiskVHead.MULTIMETRIC_GATE_SCHEMA,
                    "require_all_validation_units": True,
                },
            },
        }

    def test_state_signature_contains_demand_and_structured_pools(self):
        z = torch.tensor([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ])
        demand = torch.tensor([7.0, 8.0])
        scores = torch.tensor([0.1, 0.9, 0.2, 0.3])
        g = TDRiskVHead.build_state_features(
            z,
            demand,
            station_node_ids=[0, 2],
            bottleneck_scores=scores,
            bottleneck_fraction=0.25,
        )

        self.assertEqual(tuple(g.shape), (14,))  # 7 * latent_dim
        torch.testing.assert_close(g[0:2], z.mean(dim=0))
        torch.testing.assert_close(g[2:4], z.max(dim=0).values)
        torch.testing.assert_close(g[4:6], z[[0, 2]].mean(dim=0))
        torch.testing.assert_close(g[6:8], z[[0, 2]].max(dim=0).values)
        torch.testing.assert_close(g[8:10], z[1])
        torch.testing.assert_close(g[10:12], z[1])
        torch.testing.assert_close(g[12:14], demand)

    def test_zero_initialised_head_is_exact_immediate_risk_baseline(self):
        head = TDRiskVHead(latent_dim=4, demand_dim=4, hidden_dim=8)
        x = torch.randn(5, head.input_dim)
        current = torch.rand(5, 3)
        y = head(x, current)
        self.assertEqual(tuple(y.shape), (5, 3))
        torch.testing.assert_close(y, current)
        self.assertTrue(bool(((y >= 0.0) & (y <= 1.0)).all()))

    def test_failed_component_gate_falls_back_to_immediate_risk(self):
        head = TDRiskVHead(latent_dim=2, demand_dim=2, hidden_dim=4)
        with torch.no_grad():
            head.net[-1].bias.copy_(torch.tensor([0.2, -0.2, 0.4]))
        head.set_component_gates([True, False, False])
        x = torch.zeros(2, head.input_dim)
        current = torch.tensor([[0.1, 0.5, 0.8], [0.2, 0.4, 0.7]])
        raw = head(x, current, apply_component_gates=False)
        gated = head(x, current)
        torch.testing.assert_close(gated[:, 0], raw[:, 0])
        torch.testing.assert_close(gated[:, 1:], current[:, 1:])

    def test_missing_pool_metadata_is_explicit_zero(self):
        z = torch.randn(5, 3)
        g = TDRiskVHead.build_state_features(z, torch.randn(3))
        torch.testing.assert_close(g[6:18], torch.zeros(12))

    def test_v3_checkpoint_loader_rejects_legacy_semantics(self):
        with self.assertRaisesRegex(ValueError, "legacy scalar"):
            TDRiskVHead.from_checkpoint({"schema_version": "tdv_v1"})
        with self.assertRaisesRegex(ValueError, "absolute-value"):
            TDRiskVHead.from_checkpoint({"schema_version": "td_risk_vhead_v2"})

    def test_v4_checkpoint_round_trip_preserves_certified_gates(self):
        head, payload = self._v4_payload()
        loaded, config = TDRiskVHead.from_checkpoint(payload)
        x = torch.randn(2, head.input_dim)
        current = torch.rand(2, 3)
        torch.testing.assert_close(head(x, current), loaded(x, current))
        self.assertEqual(loaded.component_gates.tolist(), [True, False, True])
        self.assertEqual(
            config["effective_component_gates"],
            {"stall": True, "deadlock": False, "handoff": True},
        )

    def test_v4_gate_mismatch_disables_only_inconsistent_component(self):
        _, payload = self._v4_payload((True, True, False))
        payload = copy.deepcopy(payload)
        payload["config"]["component_gates"]["stall"] = False
        loaded, config = TDRiskVHead.from_checkpoint(payload)
        self.assertEqual(loaded.component_gates.tolist(), [False, True, False])
        self.assertIn("stall", config["forced_disabled_components"])

    def test_v3_checkpoint_without_auditable_metrics_fails_closed(self):
        head = TDRiskVHead(latent_dim=3, demand_dim=3, hidden_dim=5)
        head.set_component_gates([True, False, True])
        payload = {
            "schema_version": "td_risk_residual_vhead_v3",
            "td_risk_vhead": head.state_dict(),
            "td_risk_vhead_ema": head.state_dict(),
            "component_gate_decisions": {
                name: {"enabled": bool(enabled)}
                for name, enabled in zip(
                    TDRiskVHead.RISK_COMPONENTS, [True, False, True]
                )
            },
            "config": {
                "latent_dim": 3,
                "demand_dim": 3,
                "hidden_dim": 5,
                "physical_summary_dim": 0,
                "bottleneck_fraction": 0.2,
                "residual_scale": 1.0,
                "component_gates": {
                    "stall": True, "deadlock": False, "handoff": True,
                },
            },
        }
        loaded, config = TDRiskVHead.from_checkpoint(payload)
        self.assertEqual(loaded.component_gates.tolist(), [False, False, False])
        self.assertEqual(
            set(config["forced_disabled_components"]), {"stall", "handoff"}
        )

    def test_real_v3_artifact_disables_degenerate_handoff(self):
        artifact = (
            REPO_ROOT / "WorldModel" / "checkpoints"
            / "phaseB_b3_bneckfix_w200_filtered" / "lyapunov_td_impl"
            / "td_risk_residual_v3_a1_full.pt"
        )
        if not artifact.exists():
            self.skipTest("real v3 artifact is not present")
        loaded, config = TDRiskVHead.from_checkpoint(artifact)
        self.assertEqual(loaded.component_gates.tolist(), [False, True, False])
        handoff = config["gate_certification_status"]["handoff"]
        self.assertIn("legacy_spearman_below_floor", handoff["reasons"])


class TDTargetTests(unittest.TestCase):
    def test_component_gates_fail_closed_without_mc_anchors(self):
        empty = np.empty((0, 3))
        decisions = _component_gate_decisions(empty, empty, empty, [])
        for name in TDRiskVHead.RISK_COMPONENTS:
            self.assertFalse(decisions[name]["enabled"])
            self.assertEqual(
                decisions[name]["reasons"],
                ["no_complete_held_out_mc_anchors"],
            )

    def test_sparse_near_zero_handoff_cannot_pass_on_mae_alone(self):
        n = 40
        target = np.zeros((n, 3), dtype=float)
        target[:, 0] = np.linspace(0.2, 0.8, n)
        target[:, 1] = np.linspace(0.1, 0.9, n)
        target[-2:, 2] = 0.2
        immediate = target.copy()
        immediate[:, :2] *= 0.8
        immediate[:-2, 2] = 0.02
        immediate[-2:, 2] = 0.1
        pred = target.copy()
        pred[:, 2] = 0.0
        rows = [
            {
                "seed": 1,
                "run_id": "sparse",
                "arm_label": "A1",
                "tick": tick,
            }
            for tick in range(n)
        ]
        handoff = _component_gate_decisions(
            pred, target, immediate, rows, min_unit_anchors=20,
        )["handoff"]
        self.assertTrue(handoff["aggregate"]["checks"]["mae_improvement"])
        self.assertFalse(handoff["enabled"])
        self.assertFalse(handoff["aggregate"]["checks"]["rank_floor"])
        self.assertFalse(handoff["aggregate"]["checks"]["std_ratio"])
        self.assertGreater(handoff["aggregate"]["relative_abs_bias"], 0.5)

    def test_well_calibrated_ranked_deadlock_component_passes(self):
        n = 40
        target_line = np.linspace(0.2, 0.8, n)
        target = np.repeat(target_line[:, None], 3, axis=1)
        immediate = np.clip(target * 0.7, 0.0, 1.0)
        pred = target.copy()
        rows = [
            {
                "seed": 1,
                "run_id": "valid",
                "arm_label": "A1",
                "tick": tick,
            }
            for tick in range(n)
        ]
        decision = _component_gate_decisions(
            pred, target, immediate, rows, min_unit_anchors=20,
        )["deadlock"]
        self.assertTrue(decision["enabled"])
        self.assertTrue(all(decision["aggregate"]["checks"].values()))

    def test_constant_target_marks_rank_and_std_not_applicable(self):
        metric = _series_gate_metrics(
            np.full(20, 0.5),
            np.full(20, 0.5),
            np.full(20, 0.3),
            min_anchors=20,
            min_absolute_improvement=0.0,
            min_relative_improvement=0.0,
            min_spearman=0.1,
            max_spearman_drop=0.05,
            max_relative_bias=0.5,
            min_std_ratio=0.1,
            max_rmse_relative_degrade=0.0,
            variation_epsilon=1e-8,
        )
        self.assertTrue(metric["passed"])
        self.assertFalse(metric["applicable"]["rank_floor"])
        self.assertFalse(metric["applicable"]["std_ratio"])
        self.assertTrue(metric["checks"]["rank_floor"])
        self.assertTrue(metric["checks"]["std_ratio"])

    def test_variable_target_constant_candidate_fails_rank_and_scale(self):
        target = np.linspace(0.0, 1.0, 20)
        metric = _series_gate_metrics(
            np.full(20, 0.5),
            target,
            1.0 - target,
            min_anchors=20,
            min_absolute_improvement=0.0,
            min_relative_improvement=0.0,
            min_spearman=0.1,
            max_spearman_drop=0.05,
            max_relative_bias=0.5,
            min_std_ratio=0.1,
            max_rmse_relative_degrade=0.0,
            variation_epsilon=1e-8,
        )
        self.assertTrue(metric["checks"]["mae_improvement"])
        self.assertFalse(metric["passed"])
        self.assertFalse(metric["checks"]["rank_floor"])
        self.assertFalse(metric["checks"]["std_ratio"])

    def test_component_gate_requires_each_validation_seed_by_default(self):
        target = np.array([
            [0.4, 0.4, 0.4], [0.6, 0.6, 0.6],
            [0.4, 0.4, 0.4], [0.6, 0.6, 0.6],
        ])
        immediate = np.array([
            [0.2, 0.2, 0.2], [0.8, 0.8, 0.8],
            [0.2, 0.2, 0.2], [0.8, 0.8, 0.8],
        ])
        pred = np.array([
            [0.4, 0.4, 0.4], [0.6, 0.6, 0.6],
            [0.15, 0.15, 0.15], [0.85, 0.85, 0.85],
        ])
        rows = [
            {"seed": 1, "run_id": "a"}, {"seed": 1, "run_id": "a"},
            {"seed": 2, "run_id": "b"}, {"seed": 2, "run_id": "b"},
        ]
        decisions = _component_gate_decisions(
            pred, target, immediate, rows, min_unit_anchors=2,
            max_spearman_drop=0.20,
        )
        for name in TDRiskVHead.RISK_COMPONENTS:
            self.assertFalse(decisions[name]["enabled"])
            self.assertTrue(decisions[name]["aggregate"]["passed"])
            self.assertFalse(decisions[name]["units"]["seed:2"]["passed"])

    def test_component_gate_requires_each_load_arm_by_default(self):
        target = np.full((4, 3), 0.5)
        immediate = np.full((4, 3), 0.3)
        pred = np.array([
            [0.5, 0.5, 0.5], [0.75, 0.75, 0.75],
            [0.5, 0.5, 0.5], [0.75, 0.75, 0.75],
        ])
        rows = [
            {"seed": 1, "run_id": "low1", "arm_label": "A1_LOW"},
            {"seed": 1, "run_id": "mid1", "arm_label": "A1_MID"},
            {"seed": 2, "run_id": "low2", "arm_label": "A1_LOW"},
            {"seed": 2, "run_id": "mid2", "arm_label": "A1_MID"},
        ]
        decisions = _component_gate_decisions(
            pred, target, immediate, rows, min_unit_anchors=2,
        )
        for name in TDRiskVHead.RISK_COMPONENTS:
            self.assertTrue(decisions[name]["aggregate"]["passed"])
            self.assertTrue(decisions[name]["units"]["seed:1"]["passed"])
            self.assertTrue(decisions[name]["units"]["seed:2"]["passed"])
            self.assertTrue(decisions[name]["units"]["arm:A1_LOW"]["passed"])
            self.assertFalse(decisions[name]["units"]["arm:A1_MID"]["passed"])
            self.assertFalse(decisions[name]["enabled"])

    def test_validation_breakdowns_report_seed_and_arm(self):
        target = np.full((2, 3), 0.5)
        immediate = np.full((2, 3), 0.2)
        pred = np.array([[0.5, 0.5, 0.5], [0.4, 0.4, 0.4]])
        rows = [
            {"seed": 209, "run_id": "low", "arm_label": "A1_LOW"},
            {"seed": 210, "run_id": "high", "arm_label": "A1_HIGH"},
        ]
        report = _validation_breakdowns(pred, target, immediate, rows)
        self.assertEqual(set(report["by_seed"]), {"209", "210"})
        self.assertEqual(set(report["by_arm"]), {"A1_LOW", "A1_HIGH"})
        self.assertGreater(
            report["by_arm"]["A1_LOW"]["absolute_mae_improvement"]["stall"],
            0.0,
        )

    def test_stream_checkpoint_provenance_uses_digest_not_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorded = root / "source" / "best_regret_world_model.pt"
            requested = root / "requested" / "best_regret_world_model.pt"
            recorded.parent.mkdir()
            requested.parent.mkdir()
            recorded.write_bytes(b"source-checkpoint")
            requested.write_bytes(b"different-checkpoint")
            runs = [{"header": {"checkpoint_path": str(recorded)}}]

            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                validate_stream_checkpoint(runs, str(requested))

            manifest = validate_stream_checkpoint(
                runs, str(requested), allow_mismatch=True
            )
            self.assertTrue(manifest["override_used"])
            self.assertEqual(len(manifest["mismatched"]), 1)

            requested.write_bytes(recorded.read_bytes())
            manifest = validate_stream_checkpoint(runs, str(requested))
            self.assertFalse(manifest["override_used"])
            self.assertEqual(len(manifest["verified"]), 1)

    def test_right_censored_segment_is_not_an_mc_anchor(self):
        self.assertTrue(_mc_anchor_eligible({"truncated": False}))
        self.assertFalse(_mc_anchor_eligible({"truncated": True}))
        self.assertFalse(_mc_anchor_eligible({
            "truncated": False, "mc_anchor_eligible": False
        }))

    def test_constant_risk_mc_average_preserves_level(self):
        costs = torch.full((20, 3), 0.25)
        mc = _discounted_mc_average(costs, 0.95)
        torch.testing.assert_close(mc, torch.full((3,), 0.25))

    def test_prefix_has_average_return_mass(self):
        costs = torch.ones(10, 3)
        value = _discounted_prefix(costs, 0.9, 4)
        expected = torch.full((3,), 1.0 - 0.9 ** 4)
        torch.testing.assert_close(value, expected)

    def test_tie_safe_spearman(self):
        self.assertAlmostEqual(
            spearman_tie_safe([0, 0, 1, 2], [4, 4, 5, 6]), 1.0
        )

    def test_seed_split_never_leaks_a_seed(self):
        rows = [
            {"seed": seed, "run_id": f"r{seed}_{load}"}
            for seed in (201, 202, 203)
            for load in ("low", "mid")
        ]
        train, val, _ = _split_samples(rows, 0.2, 42, {203})
        self.assertEqual({row["seed"] for row in val}, {203})
        self.assertNotIn(203, {row["seed"] for row in train})

    def test_temporal_direction_is_computed_within_run(self):
        pred = np.array([
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.4],
            [0.4, 0.1, 0.2],
        ])
        target = pred.copy()
        rows = [
            {"run_id": "a", "tick": 0},
            {"run_id": "a", "tick": 5},
            {"run_id": "b", "tick": 0},
        ]
        metrics = _temporal_direction_metrics(pred, target, rows, 0.0)
        self.assertEqual(metrics["stall"]["pairs"], 1)
        self.assertEqual(metrics["stall"]["accuracy"], 1.0)


class StreamTupleBuilderTests(unittest.TestCase):
    def test_raw_component_alignment_is_preserved(self):
        frames = {
            tick: {
                "node_history": torch.zeros(4, 2, 10),
                "edge_features": torch.zeros(1, 6),
                "demand_context": torch.zeros(9),
            }
            for tick in (0, 2, 4)
        }
        stream = {
            "run_id": "A1_LOW_seed201",
            "arm_label": "A1_LOW",
            "tick_seq": torch.arange(5),
            "risk_seq_full": torch.arange(5, dtype=torch.float32),
            "risk_components": torch.arange(15, dtype=torch.float32).reshape(5, 3),
            "frames_by_tick": frames,
            "frame_stride": 2,
            "edge_index": torch.zeros(2, 1, dtype=torch.long),
            "station_node_ids": [0],
        }
        run = assemble_run(stream, K=2, W=3, start_stride=2, min_start_tick=0)
        rec = run["tuples"][0]
        self.assertTrue(rec["mc_anchor_eligible"])
        torch.testing.assert_close(
            rec["risk_components_at_start"], stream["risk_components"][0]
        )
        torch.testing.assert_close(
            rec["risk_components_at_boot"], stream["risk_components"][2]
        )
        torch.testing.assert_close(
            rec["risk_components_seq"], stream["risk_components"][1:4]
        )
        self.assertEqual(run["policy_family"], "A1")


if __name__ == "__main__":
    unittest.main()
