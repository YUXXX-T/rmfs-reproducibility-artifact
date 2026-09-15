import unittest

from WorldModel.core.analytic_work_relief import (
    STATION_WORK_TRAJECTORY_SCHEMA_VERSION,
    WorkServiceDurations,
    compute_nominal_work_relief,
    compute_snapshot_nominal_work_relief,
    compute_virtual_candidate_work_drift,
    extract_work_horizon_endpoint,
    work_potential,
)


def _snapshot():
    return {
        "station_work": {10: 2.5, 20: 1.0},
        "work_capacity": 2.0,
        "components": {"work": 0.0},
        "chains": {
            "1:1": {
                "state": "pipeline",
                "active_task_id": 101,
                "station_id": 10,
                "mass": 1.0,
                "initial_work": 20.0,
                "remaining_work": 12.0,
                "remaining_fraction": 0.6,
            },
            "2:2": {
                "state": "pipeline",
                "active_task_id": 102,
                "station_id": 10,
                "mass": 2.0,
                "initial_work": 10.0,
                "remaining_work": 3.0,
                "remaining_fraction": 0.3,
            },
            "3:3": {
                "state": "pending",
                "active_task_id": None,
                "station_id": 20,
                "mass": 1.0,
                "initial_work": 1.0,
                "remaining_work": 1.0,
                "remaining_fraction": 1.0,
            },
        },
    }


class AnalyticWorkReliefTests(unittest.TestCase):
    def test_snapshot_relief_is_monotone_saturating_and_excludes_pending(self):
        h2 = compute_snapshot_nominal_work_relief(
            _snapshot(), horizon=2, station_ids=[10, 20]
        )
        h20 = compute_snapshot_nominal_work_relief(
            _snapshot(), horizon=20, station_ids=[10, 20]
        )
        self.assertAlmostEqual(h2.nominal_station_relief[0], 0.5)
        self.assertAlmostEqual(h20.nominal_station_relief[0], 1.2)
        self.assertEqual(h2.nominal_station_relief[1], 0.0)
        self.assertEqual(h20.nominal_station_relief[1], 0.0)
        self.assertEqual(h20.available_station_relief, (1.2, 0.0))
        self.assertTrue(all(
            later >= earlier
            for earlier, later in zip(
                h2.nominal_station_relief,
                h20.nominal_station_relief,
            )
        ))
        self.assertEqual(h20.nominal_endpoint_station_work, (1.3, 1.0))

    def test_virtual_candidate_matches_the_same_stored_pipeline_chain(self):
        durations = WorkServiceDurations(
            pickup=2, station_process=5, dropoff=2
        )
        candidate = compute_nominal_work_relief(
            candidate_info={"robot_start": (0, 0)},
            fixed_context={
                "station_id": 10,
                "pod_location": (0, 2),
                "station_location": (0, 5),
                "exit_position": (0, 6),
                "return_location": (0, 1),
            },
            station_ids=[10],
            current_station_work=[1.0],
            service_durations=durations,
            horizon=10,
            work_capacity=2.0,
            work_weight=1.0,
        )
        chain_work = candidate.chain_free_flow_work
        snapshot = {
            "station_work": {10: 1.0},
            "chains": {
                "1:1": {
                    "state": "pipeline",
                    "active_task_id": 1,
                    "station_id": 10,
                    "mass": 1.0,
                    "initial_work": chain_work,
                    "remaining_work": chain_work,
                    "remaining_fraction": 1.0,
                }
            },
        }
        stored = compute_snapshot_nominal_work_relief(
            snapshot, horizon=10, station_ids=[10]
        )
        self.assertAlmostEqual(
            candidate.applied_relief_mass,
            stored.nominal_station_relief[0],
        )
        self.assertAlmostEqual(
            candidate.endpoint_station_work[0],
            stored.nominal_endpoint_station_work[0],
        )

    def test_virtual_online_candidate_reuses_full_post_action_snapshot_semantics(self):
        snapshot = _snapshot()
        snapshot["work_capacity"] = 2.0
        snapshot["components"]["work"] = work_potential(
            [2.5, 1.0], work_capacity=2.0, work_weight=1.0
        )
        snapshot["chains"]["3:3"] = {
            "state": "pending",
            "active_task_id": None,
            "station_id": 20,
            "mass": 1.0,
            "initial_work": 1.0,
            "remaining_work": 1.0,
            "remaining_fraction": 1.0,
            "planned_remaining_work": 1.0,
            "planned_remaining_fraction": 1.0,
        }
        durations = WorkServiceDurations(
            pickup=2, station_process=5, dropoff=2
        )
        drift = compute_virtual_candidate_work_drift(
            snapshot,
            candidate_info={"robot_start": (0, 0)},
            fixed_context={
                "order_id": 3,
                "pod_id": 3,
                "station_id": 20,
                "pod_location": (0, 2),
                "station_location": (0, 5),
                "exit_position": (0, 6),
                "return_location": (0, 1),
            },
            service_durations=durations,
            horizon=5,
            work_weight=1.0,
        )
        self.assertEqual(drift.horizon, 5)
        self.assertEqual(drift.active_pipeline_chains, 3)
        self.assertGreater(drift.candidate_nominal_relief_mass, 0.0)
        self.assertLess(drift.raw_work_drift, 0.0)
        self.assertEqual(drift.current_station_work, (2.5, 1.0))
        self.assertTrue(all(
            end <= start
            for start, end in zip(
                drift.current_station_work,
                drift.endpoint_station_work,
            )
        ))

        farther = compute_virtual_candidate_work_drift(
            snapshot,
            candidate_info={"robot_start": (20, 20)},
            fixed_context={
                "order_id": 3,
                "pod_id": 3,
                "station_id": 20,
                "pod_location": (0, 2),
                "station_location": (0, 5),
                "exit_position": (0, 6),
                "return_location": (0, 1),
            },
            service_durations=durations,
            horizon=5,
            work_weight=1.0,
        )
        self.assertLess(
            drift.raw_work_drift,
            farther.raw_work_drift,
            "a shorter free-flow chain must yield more negative H5 work drift",
        )

    def test_intermediate_horizon_endpoint_uses_station_trajectory_tick(self):
        rows = [
            [10.0 - 0.1 * tick, 5.0 - 0.05 * tick]
            for tick in range(1, 21)
        ]
        sample = {
            "future_mask": [1] * 20,
            "analytic_work_relief_trajectory_schema_version": (
                STATION_WORK_TRAJECTORY_SCHEMA_VERSION
            ),
            "lyapunov_l0_station_ids": [10, 20],
            "lyapunov_l0_station_work_trajectory": rows,
            "lyapunov_l0_start": {
                "tick": 100,
                "work_capacity": 2.0,
                "station_work": {10: 10.0, 20: 5.0},
                "components": {"work": work_potential(
                    [10.0, 5.0], work_capacity=2.0, work_weight=1.0
                )},
            },
            "lyapunov_l0_end": {
                "tick": 120,
                "work_capacity": 2.0,
                "station_work": {10: rows[-1][0], 20: rows[-1][1]},
                "components": {"work": 0.0, "station": 123.0},
            },
            "lyapunov_l0_config": {"work_weight": 1.0},
        }
        endpoint = extract_work_horizon_endpoint(sample, 5)
        self.assertEqual(endpoint["tick"], 105)
        self.assertEqual(endpoint["station_work"], {10: 9.5, 20: 4.75})
        self.assertAlmostEqual(
            endpoint["components"]["work"],
            work_potential(
                [9.5, 4.75], work_capacity=2.0, work_weight=1.0
            ),
        )
        self.assertEqual(endpoint["components"]["station"], 123.0)

    def test_pipeline_chain_without_active_task_is_rejected(self):
        snapshot = _snapshot()
        snapshot["chains"]["1:1"]["active_task_id"] = None
        with self.assertRaisesRegex(ValueError, "active physical task"):
            compute_snapshot_nominal_work_relief(
                snapshot, horizon=10, station_ids=[10, 20]
            )


if __name__ == "__main__":
    unittest.main()
