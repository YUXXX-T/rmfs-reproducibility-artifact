"""Unit tests for the passive FIFO token-flow diagnostic helpers."""

from __future__ import annotations

import unittest

from WorldModel.evaluation.analyze_phase_c_fifo_token_diagnostic import (
    _classify_station,
    _mode_reference_gate,
    _parse_case,
)
from WorldModel.evaluation.run_phase_c_fifo_token_diagnostic import (
    TokenFlowDiagnosticProbe,
)


class _FakeQueue:
    capacity = 7
    _assigned_agents = {2, 3}

    @staticmethod
    def occupancy():
        return 1

    @staticmethod
    def committed_load():
        return 2

    @staticmethod
    def physical_agent_ids():
        return {2}

    @staticmethod
    def waiting_agent_ids():
        return [8, 9]

    @staticmethod
    def waiting_reservation_sequence(agent_id):
        return {8: 11, 9: 12}[int(agent_id)]


class FifoTokenDiagnosticTest(unittest.TestCase):
    def test_parse_case(self):
        self.assertEqual(_parse_case("high:556"), ("high", 556))
        with self.assertRaises(ValueError):
            _parse_case("bad:556")

    def test_queue_core_is_read_only_and_complete(self):
        queue = _FakeQueue()
        before = set(queue._assigned_agents)
        row = TokenFlowDiagnosticProbe._queue_core(queue)
        self.assertEqual(row["capacity"], 7)
        self.assertEqual(row["committed_load"], 2)
        self.assertEqual(row["physical_agent_ids"], [2])
        self.assertEqual(row["waiting_agent_ids"], [8, 9])
        self.assertEqual(row["waiting_sequences"], [11, 12])
        self.assertEqual(queue._assigned_agents, before)

    def test_final_station_classification(self):
        full_wait = {
            "waiting_depth": 4,
            "committed_load": 7,
            "capacity": 7,
        }
        path_diag = {
            "event_counts": {"deliver_path_failure": 3},
            "max_in_transit_stationary_streak": 20,
            "max_exit_blocked_streak": 0,
            "max_stable_promotable_head_streak": 0,
        }
        self.assertEqual(
            _classify_station(full_wait, path_diag),
            "capacity_full_with_in_transit_path_stall",
        )

        exit_diag = {
            "event_counts": {},
            "max_in_transit_stationary_streak": 0,
            "max_exit_blocked_streak": 12,
            "max_stable_promotable_head_streak": 0,
        }
        self.assertEqual(
            _classify_station(full_wait, exit_diag),
            "capacity_full_with_exit_blocking",
        )

        promotable = {
            "waiting_depth": 1,
            "committed_load": 5,
            "capacity": 7,
        }
        fifo_diag = {
            "event_counts": {},
            "max_in_transit_stationary_streak": 0,
            "max_exit_blocked_streak": 0,
            "max_stable_promotable_head_streak": 3,
        }
        self.assertEqual(
            _classify_station(promotable, fifo_diag),
            "fifo_liveness_suspected",
        )

    def test_v1_deadlock_drift_is_warning_only(self):
        checks = {
            "completed_orders": {"passed": True},
            "completed_tasks": {"passed": True},
            "order_arrival_manifest_sha256": {"passed": True},
            "order_arrival_replayed": {"passed": True},
            "deadlock_ratio_mean": {
                "passed": False,
                "actual": 0.2,
                "expected": 0.1,
            },
            "deadlock_ratio_max": {
                "passed": False,
                "actual": 0.8,
                "expected": 0.4,
            },
        }
        v1 = _mode_reference_gate({
            "meta": {"mode": "committed_v1"},
            "reference_audit": {"passed": False, "checks": checks},
        })
        self.assertTrue(v1["hard_passed"])
        self.assertEqual(
            set(v1["warning_fields"]),
            {"deadlock_ratio_mean", "deadlock_ratio_max"},
        )

        v2 = _mode_reference_gate({
            "meta": {"mode": "fifo_v2"},
            "reference_audit": {"passed": False, "checks": checks},
        })
        self.assertFalse(v2["hard_passed"])
        self.assertEqual(v2["hard_failures"], [
            "deadlock_ratio_mean",
            "deadlock_ratio_max",
        ])


if __name__ == "__main__":
    unittest.main()
