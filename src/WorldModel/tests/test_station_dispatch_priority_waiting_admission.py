import types
import unittest

from Engine.simulation_engine import SimulationEngine
from WorldState.agent_state import AgentState, AgentStatus
from WorldState.station_state import (
    STATION_ADMISSION_COMMITTED_FIFO_V2,
    STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1,
    STATION_ADMISSION_DYNAMIC_ETA_V1,
    SlotType,
    StationQueueState,
    StationSlot,
)
from WorldState.task_state import Task, TaskState, TaskStatus, TaskType


def _queue() -> StationQueueState:
    return StationQueueState(
        station_id=2,
        service=StationSlot((0, 0), SlotType.SERVICE, 0),
        queue_slots=[],
        buffer_slots=[],
        entry_position=(2, 0),
        exit_position=(0, 1),
    )


def _configure_dispatch_queue(queue: StationQueueState) -> None:
    queue.configure_dynamic_admission(
        service_ticks=5,
        max_committed_multiplier=2.0,
        healthy_extra_ratio=0.0,
        caution_extra_ratio=0.0,
        brake_extra_ratio=0.0,
        eta_near_ticks=8,
        eta_mid_ticks=20,
    )
    queue.set_admission_mode(
        STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1
    )


def _task(task_type, order_id, pod_id, agent_id):
    task = Task(
        task_type=task_type,
        order_id=order_id,
        pod_id=pod_id,
        source=(5, 5),
        destination=(0, 0),
    )
    task.agent_id = agent_id
    task.status = TaskStatus.ASSIGNED
    if task_type != TaskType.PICK:
        task.station_id = 2
    return task


class StationDispatchPriorityWaitingAdmissionTests(unittest.TestCase):
    def test_engine_stamps_deliver_chains_in_assigner_return_order(self):
        engine = SimulationEngine.__new__(SimulationEngine)
        engine._station_dispatch_sequence = 0
        tasks = []
        delivers = []
        for order_id, pod_id, agent_id in ((7, 70, 3), (8, 80, 1)):
            pick = _task(TaskType.PICK, order_id, pod_id, agent_id)
            deliver = _task(TaskType.DELIVER, order_id, pod_id, agent_id)
            ret = _task(TaskType.RETURN, order_id, pod_id, agent_id)
            tasks.extend((pick, deliver, ret))
            delivers.append(deliver)

        engine._stamp_station_dispatch_sequences(tasks)

        self.assertEqual(
            [task.station_dispatch_sequence for task in delivers], [1, 2]
        )
        self.assertTrue(
            all(task.station_dispatch_sequence is None for task in tasks[::3])
        )

    def test_dispatch_queue_sorts_by_task_priority_and_fallback_is_stable(self):
        queue = _queue()
        _configure_dispatch_queue(queue)

        queue.enqueue_waiting_reservation(20, priority_sequence=20)
        queue.enqueue_waiting_reservation(10, priority_sequence=10)
        fallback = queue.enqueue_waiting_reservation(30)

        self.assertTrue(queue.uses_dynamic_eta_overbooking)
        self.assertTrue(queue.uses_explicit_waiting)
        self.assertTrue(queue.uses_dispatch_priority_waiting)
        self.assertEqual(queue.waiting_agent_ids(), [10, 20, 30])
        self.assertGreaterEqual(fallback, 10**12)
        self.assertEqual(queue.committed_load(), 0)

    def test_later_far_eta_waiter_can_bypass_infeasible_earlier_near_waiter(self):
        queue = _queue()
        _configure_dispatch_queue(queue)
        self.assertTrue(queue.reserve(99, request_tick=0, eta_ticks=30))
        queue.enqueue_waiting_reservation(1, priority_sequence=10)
        queue.enqueue_waiting_reservation(2, priority_sequence=20)

        self.assertFalse(queue.reserve(1, request_tick=1, eta_ticks=0))
        self.assertTrue(queue.reserve(2, request_tick=1, eta_ticks=30))

        self.assertEqual(queue.waiting_agent_ids(), [1])
        self.assertEqual(queue.committed_agent_ids(), {2, 99})
        metrics = queue.admission_metrics()
        self.assertEqual(metrics["waiting_dispatch_priority_bypass_granted"], 1)
        self.assertEqual(metrics["waiting_granted"], 1)
        self.assertTrue(metrics["invariant_holds"])

    def test_engine_attempts_waiters_in_dispatch_order_not_agent_order(self):
        queue = _queue()
        _configure_dispatch_queue(queue)
        self.assertTrue(queue.reserve(99, request_tick=0, eta_ticks=30))

        earlier = AgentState(1, (3, 0))
        later = AgentState(2, (30, 0))
        task_state = TaskState()
        for agent, sequence in ((earlier, 10), (later, 20)):
            agent.carried_pod_id = 100 + agent.agent_id
            agent.mark_station_waiting(2, since_tick=0, sequence=sequence)
            deliver = _task(
                TaskType.DELIVER,
                order_id=agent.agent_id,
                pod_id=100 + agent.agent_id,
                agent_id=agent.agent_id,
            )
            deliver.station_dispatch_sequence = sequence
            task_state.add_task(deliver)
            agent.assigned_task_id = deliver.task_id
            queue.enqueue_waiting_reservation(
                agent.agent_id, priority_sequence=sequence
            )

        agents = [later, earlier]
        station_state = types.SimpleNamespace(
            stations={2: queue}, get_queue=lambda station_id: queue
        )
        world = types.SimpleNamespace(
            agents=agents,
            task_state=task_state,
            station_state=station_state,
        )
        world.get_agent = lambda agent_id: next(
            (agent for agent in agents if agent.agent_id == agent_id), None
        )
        planned_agents = []
        engine = SimulationEngine.__new__(SimulationEngine)
        engine.world = world
        engine.logger = types.SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
        )
        engine.path_planner = types.SimpleNamespace(
            plan=lambda agent, *args, **kwargs: (
                planned_agents.append(agent.agent_id) or [(2, 0)]
            )
        )
        engine._record_path_len = lambda *args, **kwargs: None

        engine._plan_and_activate(1)

        self.assertEqual(planned_agents, [2])
        self.assertEqual(earlier.status, AgentStatus.WAITING_ASSIGNED)
        self.assertEqual(later.status, AgentStatus.CARRYING)
        self.assertEqual(queue.waiting_agent_ids(), [1])

    def test_dispatch_requeue_keeps_original_priority(self):
        queue = _queue()
        _configure_dispatch_queue(queue)
        queue.enqueue_waiting_reservation(1, priority_sequence=10)
        queue.enqueue_waiting_reservation(2, priority_sequence=20)

        sequence = queue.requeue_waiting_reservation(
            1, request_tick=5, priority_sequence=10
        )

        self.assertEqual(sequence, 10)
        self.assertEqual(queue.waiting_agent_ids(), [1, 2])

    def test_legacy_eta_and_fifo_mode_selection_remains_separate(self):
        eta = _queue()
        eta.set_admission_mode(STATION_ADMISSION_DYNAMIC_ETA_V1)
        self.assertTrue(eta.uses_dynamic_eta_overbooking)
        self.assertFalse(eta.uses_explicit_waiting)
        with self.assertRaises(RuntimeError):
            eta.enqueue_waiting_reservation(1)

        fifo = _queue()
        fifo.set_admission_mode(STATION_ADMISSION_COMMITTED_FIFO_V2)
        first = fifo.enqueue_waiting_reservation(2)
        second = fifo.enqueue_waiting_reservation(1)
        self.assertEqual((first, second), (1, 2))
        self.assertEqual(fifo.waiting_agent_ids(), [2, 1])
        self.assertTrue(fifo.can_promote_waiting_reservation(2))
        self.assertFalse(fifo.can_promote_waiting_reservation(1))


if __name__ == "__main__":
    unittest.main()
