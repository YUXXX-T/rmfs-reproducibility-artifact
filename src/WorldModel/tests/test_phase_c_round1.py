import tempfile
import types
import unittest
from pathlib import Path

import torch

from WorldModel.data.build_phase_c_round1_dataset import _audit
from WorldModel.data.candidate_generator import NO_ASSIGN_ACTION_TYPE
from WorldModel.data.fuse_and_split import explicit_seed_group_split
from WorldModel.evaluation.decision_snapshot_probe import DecisionSnapshotProbe


class PhaseCRound1Tests(unittest.TestCase):
    def test_external_baseline_snapshot_source_is_rejected(self):
        engine = types.SimpleNamespace(world=types.SimpleNamespace())
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                ValueError, "not an external baseline"
            ):
                DecisionSnapshotProbe(
                    engine,
                    tmp,
                    meta={"arm_label": "Hungarian(Manhattan)"},
                    exclude_external_baselines=True,
                )

    def test_phase_c_dataset_audit_requires_native_no_assign(self):
        common = {
            "candidate_group_id": "g1",
            "rollout_continuation_mode": "isolated",
            "rollout_generated_orders": 0,
            "rollout_assigned_tasks": 0,
            "future_mask": torch.ones(10),
            "training_source_policy": "world_model_on_policy",
            "external_baseline_training_samples": False,
            "td_target_enabled": False,
            "td_value_head_enabled": False,
            "realized_cost": 1.0,
        }
        samples = [
            {**common, "action_type": "assign_robot", "realized_cost": 1.0},
            {**common, "action_type": "assign_robot", "realized_cost": 2.0},
            {
                **common,
                "action_type": NO_ASSIGN_ACTION_TYPE,
                "realized_cost": 3.0,
                "no_assign_audit": {
                    "immediate_context_unchanged": True,
                    "tasks_added_during_isolated_rollout": [],
                },
            },
        ]

        audit = _audit(samples, expected_groups=1, horizon=10)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["no_assign_samples"], 1)

        without_no_assign = _audit(samples[:2], expected_groups=1, horizon=10)
        self.assertFalse(without_no_assign["passed"])
        self.assertFalse(
            without_no_assign["checks"]["one_no_assign_per_group"]
        )

    def test_explicit_seed_split_keeps_whole_seeds_isolated(self):
        samples = []
        for seed in (461, 462, 463):
            for group in ("a", "b"):
                gid = f"seed{seed}::{group}"
                for cost in (1.0, 2.0):
                    samples.append({
                        "candidate_group_id": gid,
                        "source_seed": str(seed),
                        "source_run_id": f"run_seed{seed}",
                        "source_load_level": "low",
                        "realized_cost": cost,
                    })

        train, val, test, report = explicit_seed_group_split(
            samples,
            train_seeds=["461"],
            val_seeds=["462"],
            test_seeds=["463"],
        )

        self.assertTrue(all("seed461" in gid for gid in train))
        self.assertTrue(all("seed462" in gid for gid in val))
        self.assertTrue(all("seed463" in gid for gid in test))
        self.assertEqual(report["split_unit"], "source_seed")
        self.assertEqual(report["totals"]["train"]["groups"], 2)
        self.assertEqual(report["totals"]["val"]["groups"], 2)
        self.assertEqual(report["totals"]["test"]["groups"], 2)

    def test_explicit_seed_split_rejects_overlap(self):
        samples = [{
            "candidate_group_id": "g1",
            "source_seed": "461",
            "realized_cost": 1.0,
        }]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            explicit_seed_group_split(
                samples,
                train_seeds=["461"],
                val_seeds=["461"],
                test_seeds=["462"],
            )


if __name__ == "__main__":
    unittest.main()
