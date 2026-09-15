from __future__ import annotations

import unittest
from types import SimpleNamespace

from WorldModel.core.adaptive_pre_admission import (
    StationPipelineSnapshot,
    adaptive_pre_admission_decision,
    build_station_pipeline_snapshots,
    eligible_defer_increment,
)
from WorldState.agent_state import AgentState, AgentStatus
from WorldState.task_state import Task, TaskStatus, TaskType


class _Queue:
    def __init__(self, capacity, physical, committed, waiting):
        self.capacity = capacity
        self._physical = set(physical)
        self._committed = set(committed)
        self._waiting = list(waiting)

    def physical_agent_ids(self):
        return set(self._physical)

    def committed_agent_ids(self):
        return set(self._committed)

    def waiting_agent_ids(self):
        return list(self._waiting)


class AdaptivePreAdmissionTests(unittest.TestCase):
    def test_low_traffic_exposes_one_prefetch_window(self):
        snapshot = StationPipelineSnapshot(2, 7, 5, 2, 0, 0)
        decision = adaptive_pre_admission_decision(
            snapshot,
            traffic=0.0,
            global_waiting_ratio=0.0,
            defer_mass=0.0,
        )
        self.assertTrue(decision.execute)
        self.assertEqual(decision.pipeline_mass_after_execute, 8.0)
        self.assertEqual(decision.effective_pipeline_limit, 14.0)
        self.assertFalse(decision.debt_override)

    def test_high_traffic_defers_beyond_physical_window(self):
        snapshot = StationPipelineSnapshot(2, 7, 5, 2, 0, 0)
        decision = adaptive_pre_admission_decision(
            snapshot,
            traffic=1.0,
            global_waiting_ratio=0.0,
            defer_mass=0.0,
        )
        self.assertFalse(decision.execute)
        self.assertEqual(decision.effective_pipeline_limit, 7.0)
        self.assertEqual(decision.excess_mass, 1.0)

    def test_defer_debt_restores_headroom_without_eta_prediction(self):
        snapshot = StationPipelineSnapshot(2, 7, 5, 2, 0, 0)
        decision = adaptive_pre_admission_decision(
            snapshot,
            traffic=1.0,
            global_waiting_ratio=0.0,
            defer_mass=1.0,
        )
        self.assertTrue(decision.execute)
        self.assertTrue(decision.debt_override)
        self.assertEqual(decision.effective_pipeline_limit, 14.0)
        self.assertAlmostEqual(eligible_defer_increment(44), 1.0 / 44.0)

    def test_waiting_continuously_removes_prefetch_headroom(self):
        snapshot = StationPipelineSnapshot(2, 10, 4, 2, 0, 5)
        decision = adaptive_pre_admission_decision(
            snapshot,
            traffic=0.0,
            global_waiting_ratio=0.25,
            defer_mass=0.0,
        )
        self.assertAlmostEqual(decision.local_waiting_ratio, 0.5)
        self.assertAlmostEqual(decision.base_headroom, 0.375)
        self.assertAlmostEqual(decision.effective_pipeline_limit, 13.75)

    def test_pipeline_builder_counts_each_launched_chain_once(self):
        agents = [AgentState(index, (index, 0)) for index in range(6)]
        agents[4].status = AgentStatus.MOVING_TO_POD
        agents[5].status = AgentStatus.WAITING_ASSIGNED

        pick = Task(TaskType.PICK, order_id=11, pod_id=21,
                    source=(4, 0), destination=(8, 8))
        deliver = Task(TaskType.DELIVER, order_id=11, pod_id=21,
                       source=(8, 8), destination=(1, 1))
        for task in (pick, deliver):
            task.agent_id = 4
            task.status = TaskStatus.ASSIGNED
        deliver.station_id = 2

        world = SimpleNamespace(
            agents=agents,
            station_state=SimpleNamespace(stations={
                2: _Queue(
                    capacity=7,
                    physical={0, 1},
                    committed={0, 1, 2, 3},
                    waiting={5},
                )
            }),
            task_state=SimpleNamespace(tasks={
                pick.task_id: pick,
                deliver.task_id: deliver,
            }),
            get_agent=lambda agent_id: agents[int(agent_id)],
        )
        snapshot = build_station_pipeline_snapshots(world)[2]
        self.assertEqual(snapshot.physical, 2)
        self.assertEqual(snapshot.committed_in_transit, 2)
        self.assertEqual(snapshot.moving_to_pod, 1)
        self.assertEqual(snapshot.waiting_assigned, 1)
        self.assertEqual(snapshot.pipeline_mass, 6.0)


if __name__ == "__main__":
    unittest.main()
