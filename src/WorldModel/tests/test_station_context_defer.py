from __future__ import annotations

import unittest
from types import SimpleNamespace

from WorldModel.core.station_context_defer import (
    STATION_CONTEXT_DEFER_RISK_PIPELINE_PHI_V2,
    StationDispatchSnapshot,
    build_station_dispatch_snapshots,
    eligible_defer_increment,
    station_context_defer_decision,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.station_context_defer_assigner import (
    StationContextDeferDynamicPsiAssigner,
    StationContextDeferPipelinePhiDynamicPsiAssigner,
)
from WorldState.agent_state import AgentState, AgentStatus
from WorldState.task_state import Task, TaskStatus, TaskType


class _Queue:
    def __init__(self, capacity, physical, committed):
        self.capacity = int(capacity)
        self._physical = set(physical)
        self._committed = set(committed)

    def physical_agent_ids(self):
        return set(self._physical)

    def committed_agent_ids(self):
        return set(self._committed)


class StationContextDeferTests(unittest.TestCase):
    def test_normal_station_has_zero_defer_risk(self):
        snapshot = StationDispatchSnapshot(2, 7, 3, 1, 1, 0)
        decision = station_context_defer_decision(
            snapshot,
            service=1.0,
            traffic=1.0,
            service_debt=0.0,
            defer_mass=0.0,
            free_flow_time=44,
        )
        self.assertTrue(decision.execute)
        self.assertEqual(decision.station_risk, 0.0)
        self.assertEqual(decision.phi_pressure, 0.0)

    def test_pipeline_excess_defers_low_debt_context(self):
        snapshot = StationDispatchSnapshot(2, 7, 4, 2, 2, 0)
        decision = station_context_defer_decision(
            snapshot,
            service=0.4,
            traffic=0.4,
            service_debt=0.1,
            defer_mass=0.0,
            free_flow_time=44,
        )
        self.assertFalse(decision.execute)
        self.assertGreater(decision.pipeline_excess_score, 0.0)
        self.assertGreater(decision.station_risk, decision.context_credit)

    def test_ready_unadmitted_is_direct_contention(self):
        snapshot = StationDispatchSnapshot(2, 7, 4, 2, 0, 1)
        decision = station_context_defer_decision(
            snapshot,
            service=0.1,
            traffic=0.1,
            service_debt=0.1,
            defer_mass=0.0,
            free_flow_time=44,
        )
        self.assertFalse(decision.execute)
        self.assertEqual(decision.dominant_risk, "ready_unadmitted")
        self.assertAlmostEqual(decision.ready_contention_score, 0.5)

    def test_pipeline_phi_v2_keeps_ready_only_as_diagnostic(self):
        snapshot = StationDispatchSnapshot(2, 7, 7, 0, 0, 1)
        decision = station_context_defer_decision(
            snapshot,
            service=0.2,
            traffic=0.2,
            service_debt=0.3,
            defer_mass=0.0,
            free_flow_time=44,
            risk_mode=STATION_CONTEXT_DEFER_RISK_PIPELINE_PHI_V2,
        )
        self.assertAlmostEqual(decision.ready_contention_score, 1.0)
        self.assertFalse(decision.ready_contention_in_station_risk)
        self.assertTrue(decision.ready_contention_would_dominate)
        self.assertEqual(decision.diagnostic_dominant_risk, "ready_unadmitted")
        self.assertEqual(decision.dominant_risk, "pipeline_excess")
        self.assertLess(decision.station_risk, 1.0)
        self.assertTrue(decision.execute)

    def test_pipeline_phi_v2_still_defers_real_pipeline_excess(self):
        snapshot = StationDispatchSnapshot(2, 7, 7, 0, 4, 1)
        decision = station_context_defer_decision(
            snapshot,
            service=0.8,
            traffic=0.8,
            service_debt=0.1,
            defer_mass=0.0,
            free_flow_time=44,
            risk_mode=STATION_CONTEXT_DEFER_RISK_PIPELINE_PHI_V2,
        )
        self.assertFalse(decision.execute)
        self.assertFalse(decision.ready_contention_in_station_risk)
        self.assertGreater(decision.pipeline_excess_score, 0.0)
        self.assertGreater(decision.station_risk, decision.context_credit)

    def test_context_debt_can_execute_under_station_pressure(self):
        snapshot = StationDispatchSnapshot(2, 7, 4, 2, 2, 0)
        decision = station_context_defer_decision(
            snapshot,
            service=0.4,
            traffic=0.4,
            service_debt=0.5,
            defer_mass=0.0,
            free_flow_time=44,
        )
        self.assertTrue(decision.execute)
        self.assertTrue(decision.base_execute)

    def test_liveness_debt_eventually_overrides_risk(self):
        snapshot = StationDispatchSnapshot(2, 7, 7, 0, 7, 0)
        free_flow = 44.0
        service_debt = 0.2
        defer_mass = 0.0
        rejected = 0
        decision = None
        while rejected <= 44:
            decision = station_context_defer_decision(
                snapshot,
                service=1.0,
                traffic=1.0,
                service_debt=service_debt,
                defer_mass=defer_mass,
                free_flow_time=free_flow,
            )
            if decision.execute:
                break
            defer_mass = min(
                defer_mass + eligible_defer_increment(free_flow), 1.0
            )
            rejected += 1
        self.assertIsNotNone(decision)
        self.assertTrue(decision.execute)
        self.assertTrue(decision.debt_override)
        self.assertLessEqual(
            rejected, decision.worst_case_liveness_bound_ticks
        )

    def test_snapshot_builder_keeps_categories_mutually_exclusive(self):
        agents = [AgentState(index, (index, 0)) for index in range(6)]
        agents[3].status = AgentStatus.MOVING_TO_POD
        agents[4].status = AgentStatus.CARRYING
        agents[4].carried_pod_id = 24

        pick_pre = Task(TaskType.PICK, 10, 23, (3, 0), (8, 8))
        deliver_pre = Task(TaskType.DELIVER, 10, 23, (8, 8), (0, 0))
        pick_pre.agent_id = deliver_pre.agent_id = 3
        pick_pre.status = TaskStatus.IN_PROGRESS
        deliver_pre.status = TaskStatus.ASSIGNED
        deliver_pre.station_id = 2

        pick_ready = Task(TaskType.PICK, 11, 24, (4, 0), (9, 9))
        deliver_ready = Task(TaskType.DELIVER, 11, 24, (9, 9), (0, 0))
        pick_ready.agent_id = deliver_ready.agent_id = 4
        pick_ready.status = TaskStatus.COMPLETED
        deliver_ready.status = TaskStatus.ASSIGNED
        deliver_ready.station_id = 2

        tasks = {
            task.task_id: task
            for task in (pick_pre, deliver_pre, pick_ready, deliver_ready)
        }
        world = SimpleNamespace(
            agents=agents,
            station_state=SimpleNamespace(stations={
                2: _Queue(7, physical={0, 1}, committed={0, 1, 2})
            }),
            task_state=SimpleNamespace(tasks=tasks),
            get_agent=lambda agent_id: agents[int(agent_id)],
        )
        snapshot = build_station_dispatch_snapshots(world)[2]
        self.assertEqual(snapshot.physical, 2)
        self.assertEqual(snapshot.committed_in_transit, 1)
        self.assertEqual(snapshot.pre_pick, 1)
        self.assertEqual(snapshot.ready_unadmitted, 1)
        self.assertEqual(snapshot.pipeline_mass, 5)

    def test_defer_is_station_local_not_global(self):
        congested = StationDispatchSnapshot(1, 7, 7, 0, 2, 1)
        healthy = StationDispatchSnapshot(2, 7, 2, 1, 1, 0)
        blocked = station_context_defer_decision(
            congested,
            service=0.8,
            traffic=0.8,
            service_debt=0.1,
            defer_mass=0.0,
            free_flow_time=44,
        )
        executable = station_context_defer_decision(
            healthy,
            service=0.8,
            traffic=0.8,
            service_debt=0.1,
            defer_mass=0.0,
            free_flow_time=44,
        )
        self.assertFalse(blocked.execute)
        self.assertTrue(executable.execute)

    def test_defer_clock_reuses_first_registered_free_flow_time(self):
        assigner = StationContextDeferDynamicPsiAssigner.__new__(
            StationContextDeferDynamicPsiAssigner
        )
        assigner._station_context_defer_clock = {}
        key = (10, 20, 2)

        self.assertEqual(assigner._defer_clock_for_decision(key, 44.0), 44.0)
        assigner._station_context_defer_clock[key] = 44.0
        self.assertEqual(assigner._defer_clock_for_decision(key, 12.0), 44.0)
        self.assertEqual(assigner._defer_clock_for_decision(key, 100.0), 44.0)

    def test_v1_and_v2_assigners_freeze_distinct_risk_contracts(self):
        self.assertNotEqual(
            StationContextDeferDynamicPsiAssigner.station_context_defer_risk_mode,
            StationContextDeferPipelinePhiDynamicPsiAssigner.station_context_defer_risk_mode,
        )
        self.assertEqual(
            StationContextDeferPipelinePhiDynamicPsiAssigner.station_context_defer_risk_mode,
            STATION_CONTEXT_DEFER_RISK_PIPELINE_PHI_V2,
        )


if __name__ == "__main__":
    unittest.main()
