from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from WorldModel.core.long_risk_schema import long_risk_runtime_contract
from WorldModel.evaluation import run_phase_c_pibt_repaired_sj_factorial as factorial
from WorldModel.evaluation.phase_c_pibt_planner_study_protocol import (
    ACTION_PATH_MODE,
    ACTION_ROUTE_ENCODING,
    PLANNER_NAME,
)
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


class _DummyAssigner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.candidate_context_mode = kwargs.get("candidate_context_mode", "prefix")
        self.energy_scoring_mode = kwargs.get("energy_scoring_mode", "off")


class _DummyPsiAssigner(_DummyAssigner):
    def psi_dispatch_metrics(self):
        return {
            "psi_dispatch_mode": "j_ascending",
            "psi_dispatch_head_loaded": True,
        }


class TestPhaseCPibtRepairedSjFactorial(unittest.TestCase):
    def test_factorial_is_complete_two_by_two(self):
        self.assertEqual(
            set(factorial.ARM_KEYS), {"s0_j0", "s0_j1", "s1_j0", "s1_j1"}
        )
        cells = {
            (spec.use_s1, spec.use_j1)
            for spec in factorial.ARM_SPECS.values()
        }
        self.assertEqual(cells, {(False, False), (False, True), (True, False), (True, True)})

    def test_assigner_factory_switches_only_requested_layers(self):
        checkpoint = Path("checkpoint.pt")
        psi_head = Path("psi.pt")
        psi_scale = Path("scale.json")
        with (
            patch.object(factorial, "WorldModelTaskAssigner", _DummyAssigner),
            patch.object(
                factorial,
                "PsiDispatchContextWorldModelTaskAssigner",
                _DummyPsiAssigner,
            ),
        ):
            for key, spec in factorial.ARM_SPECS.items():
                assigner = factorial._make_assigner(
                    spec,
                    checkpoint=checkpoint,
                    psi_head=psi_head,
                    psi_scale=psi_scale,
                )
                self.assertEqual(
                    assigner.energy_scoring_mode,
                    "conversion" if spec.use_s1 else "off",
                    key,
                )
                self.assertEqual(
                    isinstance(assigner, _DummyPsiAssigner), spec.use_j1, key
                )
                if spec.use_j1:
                    self.assertEqual(
                        assigner.kwargs["allow_phasec_s0_robot_scorer"],
                        not spec.use_s1,
                    )
                    self.assertEqual(assigner.kwargs["psi_context_mode"], "j_ascending")
                self.assertEqual(assigner.kwargs["action_path_mode"], ACTION_PATH_MODE)

    def test_policy_contract_hashes_factors_and_j1_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            psi_head = root / "psi.pt"
            psi_scale = root / "scale.json"
            checkpoint.write_bytes(b"checkpoint")
            psi_head.write_bytes(b"psi")
            psi_scale.write_text("{}", encoding="utf-8")
            contracts = {
                key: factorial._policy_contract(
                    spec,
                    checkpoint=checkpoint,
                    psi_head=psi_head,
                    psi_scale=psi_scale,
                )
                for key, spec in factorial.ARM_SPECS.items()
            }
            self.assertEqual(len({row["fingerprint_sha256"] for row in contracts.values()}), 4)
            self.assertNotIn("psi_head_checkpoint", contracts["s1_j0"])
            self.assertIn("psi_head_checkpoint", contracts["s0_j1"])
            self.assertFalse(contracts["s0_j1"]["long_risk_consumed"])
            self.assertTrue(contracts["s1_j0"]["long_risk_consumed"])

    @staticmethod
    def _base_metrics(checkpoint: Path) -> dict:
        return {
            "order_arrival_replayed": True,
            "order_arrival_manifest_sha256": "manifest",
            "order_arrival_count": 100,
            "path_planner_name": PLANNER_NAME,
            "path_planner_batch_interface": True,
            "path_planner_single_step": True,
            "pibt_strict_validation": True,
            "pibt_batch_calls": 10,
            "pibt_single_calls": 0,
            "pibt_planned_agents": 20,
            "pibt_move_decisions": 12,
            "pibt_wait_decisions": 8,
            "pibt_validation_failures": 0,
            "pibt_last_batch_audit": {"passed": True},
            "path_planner_engine_vertex_conflicts": 0,
            "path_planner_engine_swap_conflicts": 0,
            "model_assign_calls": 10,
            "fallback_greedy_calls": 0,
            "action_path_mode": ACTION_PATH_MODE,
            "action_route_encoding": ACTION_ROUTE_ENCODING,
            "world_model_path_planner_injected": True,
            "completed_orders": 50,
            "factorial_robot_selector_scope": "within_context_only",
            "factorial_candidate_context_mode": "prefix",
            "psi_dispatch_source_encoder_checkpoint_sha256": factorial.sha256_file(
                checkpoint
            ),
        }

    def test_runtime_audit_accepts_s0j0_and_s1j1_and_rejects_cross_contamination(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            manifest = {"manifest_sha256": "manifest", "total_orders": 100}
            station = {
                "mode": STATION_ADMISSION_PHYSICAL_ONLY,
                "passed": True,
                "physical_capacity_violation_count": 0,
            }

            s0 = self._base_metrics(checkpoint)
            s0.update({
                "factorial_robot_selector": "s0",
                "factorial_context_scheduler": "j0",
                "factorial_energy_scoring_mode": "off",
                "energy_conv_contexts": 0,
                "psi_dispatch_mode": "off",
                "psi_dispatch_head_loaded": False,
                "psi_dispatch_eval_calls": 0,
                "psi_dispatch_reordered_calls": 0,
                "psi_dispatch_robot_scorer_variant": "phasec_s0",
                "psi_dispatch_s1_within_context": False,
            })
            audit = factorial._run_audit(
                factorial.ARM_SPECS["s0_j0"],
                s0,
                manifest,
                station,
                checkpoint=checkpoint,
            )
            self.assertTrue(audit["passed"])

            s1j1 = self._base_metrics(checkpoint)
            s1j1.update({
                "factorial_robot_selector": "s1",
                "factorial_context_scheduler": "j1",
                "factorial_energy_scoring_mode": "conversion",
                "energy_conv_contexts": 10,
                "energy_drift_signal": "combo",
                "long_risk_schema_version": long_risk_runtime_contract()[
                    "schema_version"
                ],
                "psi_dispatch_mode": "j_ascending",
                "psi_dispatch_head_loaded": True,
                "psi_dispatch_eval_calls": 4,
                "psi_dispatch_contexts_seen": 20,
                "psi_dispatch_reordered_calls": 3,
                "psi_dispatch_encoder_contract_verified": True,
                "psi_dispatch_robot_scorer": (
                    "WorldModelTaskAssigner.select_robots_unmodified"
                ),
                "psi_dispatch_robot_scorer_variant": "s1_within_context",
                "psi_dispatch_s1_within_context": True,
            })
            audit = factorial._run_audit(
                factorial.ARM_SPECS["s1_j1"],
                s1j1,
                manifest,
                station,
                checkpoint=checkpoint,
            )
            self.assertTrue(audit["passed"])

            contaminated = dict(s1j1, factorial_energy_scoring_mode="off")
            self.assertFalse(
                factorial._run_audit(
                    factorial.ARM_SPECS["s1_j1"],
                    contaminated,
                    manifest,
                    station,
                    checkpoint=checkpoint,
                )["passed"]
            )
            missing_j1 = dict(s1j1, psi_dispatch_head_loaded=False)
            self.assertFalse(
                factorial._run_audit(
                    factorial.ARM_SPECS["s1_j1"],
                    missing_j1,
                    manifest,
                    station,
                    checkpoint=checkpoint,
                )["passed"]
            )

    def test_interaction_contrast_recovers_known_three_order_synergy(self):
        rows = {}
        for load in factorial.LOADS:
            for seed in factorial.SEEDS:
                rows[(load, seed)] = {
                    "s0_j0": {"completed_orders": 100.0},
                    "s0_j1": {"completed_orders": 105.0},
                    "s1_j0": {"completed_orders": 110.0},
                    "s1_j1": {"completed_orders": 118.0},
                }
        values = factorial._contrast_values(
            rows,
            coefficients=factorial.CONTRASTS["s1_by_j1_interaction"],
            metric="completed_orders",
            load="high",
        )
        self.assertEqual(values, [3.0] * len(factorial.SEEDS))
        report = factorial._contrast_summary(
            values, metric="completed_orders", bootstrap_seed=7
        )
        self.assertEqual(report["mean"], 3.0)
        self.assertEqual(report["wins"], len(factorial.SEEDS))
        self.assertAlmostEqual(report["exact_sign_flip_p_two_sided"], 2 / 1024)


if __name__ == "__main__":
    unittest.main()
