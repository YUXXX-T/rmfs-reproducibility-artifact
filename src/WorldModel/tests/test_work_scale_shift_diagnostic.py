import math
import unittest

from WorldModel.evaluation.diagnose_work_scale_shift import (
    diagnose_work_scale_shift,
)


def _samples():
    rows = []
    loads = ("low", "mid", "high")
    pattern = (-1.0, 0.0, 1.0)
    wm_pattern = (0.5, -1.0, 0.5)
    for seed in (1, 2, 3, 4):
        scale = 0.10 if seed == 4 else 0.01
        start_work = 10.0 if seed == 4 else 1.0
        for load_index, load in enumerate(loads):
            for group_index in range(4):
                group = f"seed{seed}_{load}_group{group_index}"
                offset = (seed + load_index + group_index) % 3
                for candidate in range(3):
                    index = (candidate + offset) % 3
                    direction = pattern[index]
                    wm = wm_pattern[index]
                    work_delta = scale * direction
                    end_work = start_work + work_delta
                    rows.append({
                        "run_id": f"diag_{load}_seed{seed}",
                        "simulation_seed": seed,
                        "load_level": load,
                        "candidate_group_id": group,
                        "candidate_key": f"{group}_candidate{candidate}",
                        "candidate_info": {"robot_id": candidate},
                        "fixed_context": {
                            "order_id": group_index,
                            "pod_id": group_index,
                            "station_id": load_index + 1,
                        },
                        "decision_tick": 5 * (group_index + 1),
                        "lyapunov_l0_valid": True,
                        "lyapunov_l0_delta": work_delta,
                        "lyapunov_l0_start": {
                            "components": {"work": start_work},
                            "station_work": {
                                1: math.sqrt(2.0 * start_work)
                            },
                            "work_capacity": 1.0,
                        },
                        "lyapunov_l0_end": {
                            "components": {"work": end_work},
                            "station_work": {
                                1: math.sqrt(2.0 * end_work)
                            },
                            "work_capacity": 1.0,
                        },
                        "lyapunov_l0_config": {"work_weight": 1.0},
                        "wm_score": wm,
                        "realized_cost": (
                            100.0 + 3.0 * load_index + 0.2 * wm
                            + 1.5 * direction
                        ),
                    })
    return rows


class WorkScaleShiftDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = diagnose_work_scale_shift(
            _samples(),
            wm_score_field="wm_score",
            focus_seed=4,
            bootstrap_repeats=30,
            placebo_repeats=10,
            random_seed=17,
            top_groups=3,
        )

    def test_reproduces_focus_seed_raw_scale_failure(self):
        report = self.report
        self.assertTrue(report["raw_scale_failure_reproduced"])
        comparison = report["focus_vs_reference"]["overall"]
        self.assertGreater(
            comparison["focus_to_reference_abs_range_ratio"], 9.0
        )
        self.assertGreater(
            comparison["focus_rows_outside_reference_range"], 0
        )
        raw = report["transform_summary"]["raw_centered"]
        self.assertLess(raw["focus_seed_improvement"], 0.0)
        self.assertEqual(raw["worst_seed"], "seed=4")
        attribution = report["prediction_error_attribution"]["raw_centered"]
        self.assertGreater(
            attribution["focus_seed"]["excess_sse_positive_is_harm"], 0.0
        )
        self.assertEqual(attribution["top_harmful_groups"][0]["seed"], 4)
        self.assertEqual(
            set(attribution["by_seed_and_load"]["seed=4"]),
            {"low", "mid", "high"},
        )

    def test_scale_invariant_transforms_recover_direction_without_certifying(self):
        report = self.report
        for name in ("relative_start", "group_range", "group_rank", "robust_tanh"):
            summary = report["transform_summary"][name]
            self.assertGreater(summary["focus_seed_improvement"], 0.0)
            self.assertGreater(summary["top1_accuracy_improvement"], 0.0)
            self.assertLess(
                summary["normalised_regret"]["wm_plus_transform"],
                summary["normalised_regret"]["wm_only"],
            )
            self.assertEqual(
                report["transform_evaluation"][name]["role"],
                "DEVELOPMENT_DIAGNOSTIC_ONLY",
            )
        self.assertFalse(report["semantics"]["certifies_alternative_transform"])
        self.assertFalse(report["semantics"]["permits_online_use"])

    def test_extreme_group_records_are_action_auditable(self):
        groups = self.report["extreme_groups"]["focus_seed_top_groups"]
        self.assertEqual(len(groups), 3)
        first = groups[0]
        self.assertEqual(first["seed"], 4)
        self.assertEqual(first["candidate_count"], 3)
        self.assertEqual(len(first["candidates"]), 3)
        self.assertIn("fixed_context", first)
        self.assertIn("group_rank", first["candidates"][0]["transforms"])
        ledger = first["candidates"][0]["work_ledger_audit"]
        self.assertEqual(ledger["status"], "AUDITED")
        self.assertAlmostEqual(ledger["closure_error"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
