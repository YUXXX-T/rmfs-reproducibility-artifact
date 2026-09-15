import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from WorldModel.evaluation import (
    analyze_phase_c_dispatch_preserving_eta_queue as analyzer,
)
from WorldModel.evaluation.analyze_phase_c_dispatch_preserving_eta_queue import (
    DELTA_METRICS,
    ARM_LAYOUT,
    _difference_in_differences,
    _paired_delta,
)
from WorldModel.evaluation.run_phase_c_dispatch_preserving_eta_queue import (
    ADMISSION_MODES,
    ARM_KEYS,
    ARM_SPECS,
    DISPATCH_QUEUE_KEY,
    ETA_CONTROL_KEY,
)
from WorldState.station_state import (
    STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1,
    STATION_ADMISSION_DYNAMIC_ETA_V1,
)


def _row(completed, deadlock=0.1, clearance=0.8, open_orders=10):
    values = {
        "completed_orders": float(completed),
        "completed_orders_per_1000_sim_ticks": float(completed),
        "clearance_ratio": float(clearance),
        "open_order_count": float(open_orders),
        "deadlock_ratio_mean": float(deadlock),
        "stall_ratio_mean": float(deadlock) / 2.0,
        "capacity_rejections": 5.0,
        "collapsed": False,
    }
    assert set(DELTA_METRICS).issubset(values)
    return values


class DispatchPreservingEtaQueueEvaluationTests(unittest.TestCase):
    def test_arm_matrix_pairs_each_policy_under_identical_eta_modes(self):
        self.assertEqual(len(ARM_KEYS), 6)
        self.assertEqual(
            ADMISSION_MODES[ETA_CONTROL_KEY],
            STATION_ADMISSION_DYNAMIC_ETA_V1,
        )
        self.assertEqual(
            ADMISSION_MODES[DISPATCH_QUEUE_KEY],
            STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1,
        )
        for policy in ("s1_j1", "s0_j0", "greedy"):
            control = ARM_SPECS[f"{policy}_{ETA_CONTROL_KEY}"]
            queue = ARM_SPECS[f"{policy}_{DISPATCH_QUEUE_KEY}"]
            self.assertEqual(control.policy_key, queue.policy_key)
            self.assertEqual(control.policy_family, queue.policy_family)
            self.assertEqual(control.robot_selector, queue.robot_selector)
            self.assertEqual(control.context_scheduler, queue.context_scheduler)
            self.assertNotEqual(control.admission_mode, queue.admission_mode)

    def test_paired_delta_uses_seed_pairing(self):
        left = {1: _row(12), 2: _row(18)}
        right = {1: _row(10), 2: _row(14)}

        result = _paired_delta(left, right)

        self.assertEqual(result["paired_seed_count"], 2)
        self.assertEqual(result["completed_orders_delta_mean"], 3.0)
        self.assertEqual(
            result["throughput_wins_ties_losses"],
            {"wins": 2, "ties": 0, "losses": 0},
        )

    def test_difference_in_differences_is_queue_effect_difference(self):
        s1_control = {1: _row(10), 2: _row(20)}
        s1_queue = {1: _row(15), 2: _row(26)}
        baseline_control = {1: _row(9), 2: _row(18)}
        baseline_queue = {1: _row(11), 2: _row(21)}

        result = _difference_in_differences(
            s1_queue,
            s1_control,
            baseline_queue,
            baseline_control,
        )

        self.assertEqual(result["paired_seed_count"], 2)
        self.assertEqual(result["completed_orders_delta_mean"], 3.0)
        self.assertEqual(
            [row["completed_orders_delta"] for row in result["pairs"]],
            [3.0, 3.0],
        )

    def test_analyzer_writes_complete_paired_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            manifest_sha = "paired-manifest-sha"
            contract = {
                "schema_version": (
                    "phase_c_capacity_frontier_manifest_contract_v1"
                ),
                "load": "high",
                "seeds": [551],
                "target_ticks": 1500,
                "multipliers": [{"tag": "m100", "value": 1.0}],
                "nested_prefix_checks": [],
                "per_seed": {
                    "551": {
                        "scaled": {
                            "m100": {
                                "manifest_sha256": manifest_sha,
                                "total_orders": 100,
                            }
                        }
                    }
                },
            }
            (source / "capacity_frontier_manifest_contract.json").write_text(
                json.dumps(contract), encoding="utf-8"
            )

            for index, arm in enumerate(ARM_KEYS):
                policy, admission = ARM_LAYOUT[arm]
                path = (
                    output / "m100" / "per_arm" / arm / "high_seed551.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                is_queue = admission == DISPATCH_QUEUE_KEY
                completed = 90 + index
                payload = {
                    "schema_version": (
                        "phase_c_dispatch_preserving_eta_queue_arm_v1"
                    ),
                    "meta": {
                        "arm_key": arm,
                        "policy_key": policy,
                        "admission_key": admission,
                        "load": "high",
                        "seed": 551,
                        "ticks": 1500,
                        "policy_contract": {
                            "fingerprint_sha256": f"fingerprint-{policy}"
                        },
                    },
                    "audit": {"passed": True},
                    "manifest": {
                        "content_sha256": manifest_sha,
                        "total_orders": 100,
                    },
                    "metrics": {
                        "completed_orders": completed,
                        "open_order_count": 100 - completed,
                        "pending_order_count": 100 - completed,
                        "deadlock_ratio_mean": 0.01,
                        "stall_ratio_mean": 0.02,
                        "station_capacity_rejections": 5,
                        "waiting_assigned_ratio": 0.1 if is_queue else 0.0,
                        "waiting_duration_p95_ticks": 3 if is_queue else None,
                        "waiting_promotions": 2 if is_queue else 0,
                        "dispatch_queue_bypass_grants": 1 if is_queue else 0,
                        "unresolved_waiter_count_final": 0,
                    },
                    "station_admission_audit": {
                        "physical_capacity_violation_count": 0,
                        "dynamic_hard_limit_violation_count": 0,
                        "max_committed_load": {"1": 10},
                        "max_occupancy": {"1": 7},
                    },
                    "dispatch_waiting_audit": (
                        {
                            "passed": True,
                            "dispatch_priority_fallback_count": 0,
                        }
                        if is_queue else None
                    ),
                }
                path.write_text(json.dumps(payload), encoding="utf-8")

            argv = [
                "analyze",
                "--source-root", str(source),
                "--output-root", str(output),
                "--load", "high",
                "--seeds", "551",
            ]
            with patch.object(sys, "argv", argv):
                analyzer.main()

            summary_path = (
                output
                / "validation"
                / "dispatch_preserving_eta_queue_frontier.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["integrity"]["passed"])
            self.assertEqual(summary["meta"]["run_count"], 6)
            self.assertEqual(len(summary["queue_effects"]), 3)
            self.assertEqual(len(summary["difference_in_differences"]), 2)


if __name__ == "__main__":
    unittest.main()
