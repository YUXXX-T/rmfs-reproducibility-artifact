import unittest

from WorldModel.evaluation.run_phase_c_round1_online_pair import (
    _comparison,
    _pair_audit,
    _validate_candidate_schema,
)


class PhaseCOnlinePairTest(unittest.TestCase):
    def test_candidate_schema_is_fail_closed(self):
        valid = {
            "action_schema": {
                "schema_version": "wm_native_no_assign_action_v1",
                "supports_no_assign_candidate": True,
                "no_assign_encoding": "zero_action_tensors_v1",
                "complete_group_coverage": True,
                "zero_encoding_verified": True,
            }
        }
        _validate_candidate_schema(valid)
        invalid = {"action_schema": dict(valid["action_schema"])}
        invalid["action_schema"]["zero_encoding_verified"] = False
        with self.assertRaises(ValueError):
            _validate_candidate_schema(invalid)

    def test_pair_audit_requires_exact_manifest_and_no_assign(self):
        baseline = {
            "order_arrival_manifest_sha256": "abc",
            "order_arrival_count": 5,
            "fallback_greedy_ratio": 0.0,
            "online_robot_candidate_scope": "all_idle",
        }
        candidate = {
            "order_arrival_manifest_sha256": "abc",
            "order_arrival_count": 5,
            "fallback_greedy_ratio": 0.0,
            "online_robot_candidate_scope": "all_idle",
            "native_no_assign_enabled": True,
            "native_no_assign_contexts": 7,
            "native_no_assign_scored": 7,
        }
        self.assertTrue(_pair_audit(baseline, candidate)["passed"])
        candidate["order_arrival_manifest_sha256"] = "different"
        self.assertFalse(_pair_audit(baseline, candidate)["passed"])

    def test_comparison_uses_candidate_minus_baseline(self):
        result = _comparison(
            {"completed_orders": 2, "avg_excess_delay": 4.0},
            {"completed_orders": 5, "avg_excess_delay": 3.0},
        )
        self.assertEqual(
            result["completed_orders"]["candidate_minus_baseline"], 3
        )
        self.assertEqual(
            result["avg_excess_delay"]["candidate_minus_baseline"], -1.0
        )


if __name__ == "__main__":
    unittest.main()
