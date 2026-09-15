import unittest

from WorldModel.core.analytic_work_relief import (
    WorkServiceDurations,
    work_potential,
)
from WorldModel.evaluation.analyze_phase_c_lyapunov_shadow import _auc
from WorldModel.evaluation.diagnose_phase_c_lyapunov_shadow import (
    reproduce_audit,
)
from WorldModel.evaluation.phase_c_postcert_lyapunov import (
    compute_no_assign_work_drift,
)
from WorldModel.core.analytic_work_relief import (
    compute_virtual_candidate_work_drift,
)


class PhaseCPostcertLyapunovTest(unittest.TestCase):
    def _snapshot(self):
        station_work = {1: 2.0, 2: 0.5}
        capacity = 4.0
        work_weight = 1.0
        return {
            "tick": 100,
            "total": work_potential(
                list(station_work.values()),
                work_capacity=capacity,
                work_weight=work_weight,
            ),
            "components": {
                "work": work_potential(
                    list(station_work.values()),
                    work_capacity=capacity,
                    work_weight=work_weight,
                ),
                "station": 0.0,
                "traffic": 0.0,
                "stall": 0.0,
                "arrival": 0.0,
            },
            "station_work": station_work,
            "station_queue_ratio": {1: 0.0, 2: 0.0},
            "arrival_bins": {1: [0.0, 0.0], 2: [0.0, 0.0]},
            "chains": {
                "10:20": {
                    "order_id": 10,
                    "pod_id": 20,
                    "station_id": 1,
                    "mass": 1.0,
                    "initial_work": 1.0,
                    "remaining_work": 1.0,
                    "remaining_fraction": 1.0,
                    "planned_remaining_work": 1.0,
                    "planned_remaining_fraction": 1.0,
                    "active_task_id": None,
                    "active_position": None,
                    "active_path_remaining": [],
                    "state": "pending",
                },
                "11:21": {
                    "order_id": 11,
                    "pod_id": 21,
                    "station_id": 1,
                    "mass": 1.0,
                    "initial_work": 10.0,
                    "remaining_work": 5.0,
                    "remaining_fraction": 0.5,
                    "planned_remaining_work": 5.0,
                    "planned_remaining_fraction": 0.5,
                    "active_task_id": 99,
                    "active_position": [0, 0],
                    "active_path_remaining": [],
                    "state": "pipeline",
                },
            },
            "work_capacity": capacity,
            "unresolved_pending_orders": 0,
            "traffic_diagnostics": {},
        }

    def test_assignment_dissipates_more_work_than_no_assign(self):
        snapshot = self._snapshot()
        no_assign = compute_no_assign_work_drift(
            snapshot,
            horizon=5,
            work_weight=1.0,
        )
        assignment = compute_virtual_candidate_work_drift(
            snapshot,
            candidate_info={"robot_id": 1, "robot_start": [0, 0]},
            fixed_context={
                "order_id": 10,
                "pod_id": 20,
                "pod_location": [1, 0],
                "station_id": 1,
                "station_location": [2, 0],
                "exit_position": [2, 0],
                "return_location": [0, 0],
            },
            service_durations=WorkServiceDurations(
                pickup=0,
                station_process=0,
                dropoff=0,
            ),
            horizon=5,
            work_weight=1.0,
        )
        self.assertLess(
            assignment.raw_work_drift,
            no_assign["raw_work_drift"],
        )

    def test_auc_is_tie_aware(self):
        self.assertEqual(_auc([2.0, 3.0], [0.0, 1.0]), 1.0)
        self.assertEqual(_auc([0.0, 1.0], [2.0, 3.0]), 0.0)
        self.assertEqual(_auc([1.0], [1.0]), 0.5)

    def test_reproduction_ignores_timing_but_not_actions(self):
        reference_report = {
            "arms": {
                "PhaseCWorldModel": {
                    "completed_orders": 3,
                    "wall_time_s": 10.0,
                    "assignment_time_ms_mean": 2.0,
                    "order_arrival_manifest_sha256": "abc",
                    "fallback_greedy_calls": 0,
                    "decision_trace_contexts": 1,
                }
            }
        }
        metrics = {
            "completed_orders": 3,
            "wall_time_s": 99.0,
            "assignment_time_ms_mean": 50.0,
            "order_arrival_manifest_sha256": "abc",
            "fallback_greedy_calls": 0,
            "decision_trace_contexts": 1,
        }
        trace = [{
            "tick": 3,
            "context_idx": 0,
            "selected_action_type": "no_assign",
            "action_status": "native_no_assign_selected",
        }]
        audit = reproduce_audit(
            reference_report=reference_report,
            reference_trace=trace,
            observed_metrics=metrics,
            observed_trace=trace,
        )
        self.assertTrue(audit["passed"])
        changed = [dict(trace[0], selected_action_type="assign_robot")]
        audit = reproduce_audit(
            reference_report=reference_report,
            reference_trace=trace,
            observed_metrics=metrics,
            observed_trace=changed,
        )
        self.assertFalse(audit["passed"])


if __name__ == "__main__":
    unittest.main()
