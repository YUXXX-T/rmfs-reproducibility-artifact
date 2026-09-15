import unittest

from WorldModel.evaluation.phase_c_failure_diagnostic_probe import (
    FailureTrajectoryProbe,
)
from WorldModel.evaluation.phase_c_failure_diagnostic_protocol import (
    CASES,
    canonical_sha256,
    case_for,
    protocol_payload,
    run_id,
)
from WorldModel.evaluation.replay_phase_c_failure_counterfactuals import (
    _group_ranking_summary,
    _pairwise_concordance,
)
from WorldModel.evaluation.validate_phase_c_failure_diagnostic import (
    _persistent_deadlock_onset,
)


class PhaseCFailureDiagnosticTests(unittest.TestCase):
    def test_protocol_hash_excludes_only_claimed_hash(self):
        payload = protocol_payload("a" * 64)
        claimed = payload.pop("protocol_sha256")
        self.assertEqual(canonical_sha256(payload), claimed)

    def test_cases_and_run_ids_are_frozen(self):
        self.assertEqual(len(CASES), 3)
        self.assertEqual(case_for("high", 501)["snapshot_arm"], "phasec_s1")
        self.assertEqual(case_for("low", 506)["snapshot_arm"], "phasec")
        self.assertEqual(run_id("phasec", "mid", 506), "phasec_mid_seed506")
        with self.assertRaisesRegex(ValueError, "not frozen"):
            case_for("low", 501)

    def test_decision_summary_detects_energy_conversion_change(self):
        record = {
            "context_idx": 0,
            "order_id": 10,
            "pod_id": 20,
            "station_id": 2,
            "selected_robot": 4,
            "baseline_robot": 3,
            "scored_candidate_count": 2,
            "candidates": [
                {
                    "robot_id": 3,
                    "score": 1.0,
                    "score_conv": 0.5,
                    "route_len": 12,
                },
                {
                    "robot_id": 4,
                    "score": 1.2,
                    "score_conv": 0.4,
                    "route_len": 10,
                },
            ],
        }
        summary = FailureTrajectoryProbe._decision_summary(record)
        self.assertTrue(summary["energy_conversion_modified"])
        self.assertAlmostEqual(summary["base_rank_margin"], 0.2)
        self.assertEqual(summary["selected_robot"], 4)

    def test_persistent_deadlock_requires_two_high_windows(self):
        values = [0.0] * 20 + [0.6] * 100
        self.assertEqual(
            _persistent_deadlock_onset(values, window=10, threshold=0.5),
            19,
        )
        self.assertIsNone(
            _persistent_deadlock_onset(
                [0.0] * 20 + [0.6] * 10 + [0.0] * 20,
                window=10,
                threshold=0.5,
            )
        )

    def test_pairwise_concordance_and_pareto_ranking_summary(self):
        group = {
            "selected_robot": 1,
            "baseline_robot": 1,
            "candidates": [
                {
                    "robot_id": 1,
                    "wm_score": 0.1,
                    "policy_score": 0.1,
                    "horizons": {
                        "200": {
                            "discounted_realized_cost": 2.0,
                            "completed_orders_delta": 1.0,
                            "unified_risk_terminal_mean": 0.8,
                        }
                    },
                },
                {
                    "robot_id": 2,
                    "wm_score": 0.2,
                    "policy_score": 0.2,
                    "horizons": {
                        "200": {
                            "discounted_realized_cost": 1.0,
                            "completed_orders_delta": 2.0,
                            "unified_risk_terminal_mean": 0.4,
                        }
                    },
                },
            ],
        }
        summary = _group_ranking_summary(group, 200)
        self.assertFalse(summary["selected_top1_matches_true_cost"])
        self.assertTrue(summary["trace_policy_argmin_matches_selected"])
        self.assertTrue(summary["pareto_dominated_selected"])
        self.assertEqual(summary["pareto_dominating_robots"], [2])
        self.assertEqual(summary["selected_normalised_cost_regret"], 1.0)
        rows = [
            {"prediction": 0.1, "truth": 2.0},
            {"prediction": 0.2, "truth": 1.0},
        ]
        self.assertEqual(
            _pairwise_concordance(rows, "prediction", "truth"), 0.0
        )


if __name__ == "__main__":
    unittest.main()
