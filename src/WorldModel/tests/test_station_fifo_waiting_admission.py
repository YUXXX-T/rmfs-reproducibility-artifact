import types
import unittest

from Engine.simulation_engine import SimulationEngine
from WorldState.agent_state import AgentState, AgentStatus
from WorldState.station_state import (
    STATION_ADMISSION_COMMITTED_FIFO_V2,
    SlotType,
    StationQueueState,
    StationSlot,
)
from WorldState.task_state import Task, TaskState, TaskStatus, TaskType
from WorldModel.data.counterfactual_rollout import _plan_and_activate_step
from Policies.PathPlanner.PrioritizedPathPlanner.prioritized_path_planner import (
    PrioritizedPathPlanner,
)


def _queue(capacity_slots=1) -> StationQueueState:
    queue_slots = [
        StationSlot((i + 1, 0), SlotType.QUEUE, i)
        for i in range(max(0, capacity_slots - 1))
    ]
    return StationQueueState(
        station_id=2,
        service=StationSlot((0, 0), SlotType.SERVICE, 0),
        queue_slots=queue_slots,
        buffer_slots=[],
        entry_position=(capacity_slots + 1, 0),
        exit_position=(0, 1),
    )


class StationFifoWaitingAdmissionTests(unittest.TestCase):
    def test_new_status_preserves_historical_enum_values(self):
        self.assertEqual(AgentStatus.IDLE.value, 1)
        self.assertEqual(AgentStatus.MOVING_TO_POD.value, 2)
        self.assertEqual(AgentStatus.CARRYING.value, 3)
        self.assertEqual(AgentStatus.DELIVERING.value, 4)
        self.assertEqual(AgentStatus.QUEUING.value, 5)
        self.assertEqual(AgentStatus.RETURNING.value, 6)
        self.assertEqual(AgentStatus.EXITING.value, 7)
        self.assertEqual(AgentStatus.MOVING.value, 8)
        self.assertEqual(AgentStatus.WAITING_ASSIGNED.value, 9)

    def test_waiting_reservations_are_fifo_and_do_not_consume_capacity(self):
        queue = _queue(1)
        queue.set_admission_mode(STATION_ADMISSION_COMMITTED_FIFO_V2)

        self.assertTrue(queue.reserve(1, request_tick=1))
        self.assertFalse(queue.reserve(2, request_tick=2))
        self.assertFalse(queue.reserve(3, request_tick=3))
        self.assertEqual(queue.committed_load(), 1)
        self.assertEqual(queue.waiting_agent_ids(), [2, 3])
        self.assertFalse(queue.can_promote_waiting_reservation(2))
        self.assertFalse(queue.can_promote_waiting_reservation(3))

        queue.unreserve(1)
        self.assertTrue(queue.can_promote_waiting_reservation(2))
        self.assertFalse(queue.can_promote_waiting_reservation(3))
        self.assertFalse(queue.reserve(3, request_tick=4))
        self.assertTrue(queue.reserve(2, request_tick=4))
        self.assertEqual(queue.waiting_agent_ids(), [3])
        self.assertEqual(queue.committed_agent_ids(), {2})

        queue.unreserve(2)
        self.assertTrue(queue.reserve(3, request_tick=5))
        self.assertEqual(queue.waiting_agent_ids(), [])

    def test_cancelled_waiter_is_removed_without_reordering_other_waiters(self):
        queue = _queue(1)
        queue.set_admission_mode(STATION_ADMISSION_COMMITTED_FIFO_V2)
        self.assertTrue(queue.reserve(1))
        self.assertFalse(queue.reserve(2))
        self.assertFalse(queue.reserve(3))

        self.assertTrue(queue.cancel_waiting_reservation(2))
        self.assertEqual(queue.waiting_agent_ids(), [3])
        queue.unreserve(1)
        self.assertTrue(queue.reserve(3))

    def test_requeue_after_transactional_path_failure_goes_to_tail(self):
        queue = _queue(1)
        queue.set_admission_mode(STATION_ADMISSION_COMMITTED_FIFO_V2)
        self.assertTrue(queue.reserve(1))
        self.assertFalse(queue.reserve(2))
        queue.unreserve(1)

        # Agent 2 is promoted first, but its path transaction fails.  A new
        # request may proceed rather than being held behind a failed head.
        self.assertTrue(queue.reserve(2))
        queue.unreserve(2)
        sequence = queue.requeue_waiting_reservation(2, request_tick=9)
        self.assertIsInstance(sequence, int)
        self.assertFalse(queue.reserve(3, request_tick=10))
        self.assertEqual(queue.waiting_agent_ids(), [2, 3])

    def test_agent_waiting_state_is_non_idle_but_excluded_from_action_wait(self):
        agent = AgentState(agent_id=4, start_position=(5, 5))
        agent.mark_station_waiting(station_id=2, since_tick=17, sequence=8)

        self.assertEqual(agent.status, AgentStatus.WAITING_ASSIGNED)
        self.assertFalse(agent.is_idle)
        self.assertTrue(agent.is_waiting)
        self.assertEqual(agent.station_waiting_station_id, 2)
        self.assertEqual(agent.station_waiting_since_tick, 17)
        agent.clear_station_waiting()
        self.assertIsNone(agent.station_waiting_station_id)

    def test_engine_rejection_enters_waiting_without_starting_deliver(self):
        agent = AgentState(agent_id=1, start_position=(5, 5))
        agent.status = AgentStatus.MOVING_TO_POD
        agent.carried_pod_id = 10
        task_state = TaskState()
        deliver = Task(
            TaskType.DELIVER, order_id=1, pod_id=10,
            source=(5, 5), destination=(0, 0),
        )
        deliver.agent_id = agent.agent_id
        deliver.station_id = 2
        deliver.status = TaskStatus.ASSIGNED
        task_state.add_task(deliver)
        queue = _queue(1)
        queue.set_admission_mode(STATION_ADMISSION_COMMITTED_FIFO_V2)
        queue.reserve(99)

        world = types.SimpleNamespace(
            agents=[agent],
            task_state=task_state,
            station_state=types.SimpleNamespace(
                stations={2: queue},
                get_queue=lambda station_id: queue,
            ),
        )
        world.get_agent = lambda agent_id: agent if agent_id == 1 else None
        engine = SimulationEngine.__new__(SimulationEngine)
        engine.world = world
        engine.logger = types.SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
        )
        engine.path_planner = types.SimpleNamespace(
            plan=lambda *args, **kwargs: self.fail("path must not be planned")
        )
        engine._record_path_len = lambda *args, **kwargs: None

        engine._plan_and_activate(20)

        self.assertEqual(agent.status, AgentStatus.WAITING_ASSIGNED)
        self.assertEqual(deliver.status, TaskStatus.ASSIGNED)
        self.assertIsNone(task_state.get_active_task_for_agent(agent.agent_id))
        self.assertEqual(queue.waiting_agent_ids(), [1])

    def test_fifo_promotion_starts_deliver_and_only_then_consumes_token(self):
        agent = AgentState(agent_id=1, start_position=(5, 5))
        agent.mark_station_waiting(station_id=2, since_tick=10, sequence=1)
        agent.carried_pod_id = 10
        task_state = TaskState()
        deliver = Task(
            TaskType.DELIVER, order_id=1, pod_id=10,
            source=(5, 5), destination=(0, 0),
        )
        deliver.agent_id = 1
        deliver.station_id = 2
        deliver.status = TaskStatus.ASSIGNED
        task_state.add_task(deliver)
        queue = _queue(1)
        queue.set_admission_mode(STATION_ADMISSION_COMMITTED_FIFO_V2)
        queue.enqueue_waiting_reservation(1, request_tick=10)

        world = types.SimpleNamespace(
            agents=[agent],
            task_state=task_state,
            station_state=types.SimpleNamespace(
                stations={2: queue},
                get_queue=lambda station_id: queue,
            ),
        )
        world.get_agent = lambda agent_id: agent if agent_id == 1 else None
        engine = SimulationEngine.__new__(SimulationEngine)
        engine.world = world
        engine.logger = types.SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
        )
        engine.path_planner = types.SimpleNamespace(
            plan=lambda *args, **kwargs: [(4, 5), (3, 5), (2, 5), (1, 0)]
        )
        engine._record_path_len = lambda *args, **kwargs: None

        engine._plan_and_activate(11)

        self.assertEqual(agent.status, AgentStatus.CARRYING)
        self.assertEqual(deliver.status, TaskStatus.IN_PROGRESS)
        self.assertEqual(queue.committed_agent_ids(), {1})
        self.assertEqual(queue.waiting_agent_ids(), [])
        self.assertEqual(agent.assigned_task_id, deliver.task_id)

    def test_fifo_promotion_at_entry_accepts_zero_length_path(self):
        """Already-at-entry is a valid activation, not a path failure."""
        agent = AgentState(agent_id=1, start_position=(2, 0))
        agent.mark_station_waiting(station_id=2, since_tick=10, sequence=1)
        agent.carried_pod_id = 10
        task_state = TaskState()
        deliver = Task(
            TaskType.DELIVER, order_id=1, pod_id=10,
            source=(2, 0), destination=(0, 0),
        )
        deliver.agent_id = 1
        deliver.station_id = 2
        deliver.status = TaskStatus.ASSIGNED
        task_state.add_task(deliver)
        queue = _queue(1)
        queue.set_admission_mode(STATION_ADMISSION_COMMITTED_FIFO_V2)
        queue.enqueue_waiting_reservation(1, request_tick=10)

        world = types.SimpleNamespace(
            agents=[agent], task_state=task_state,
            station_state=types.SimpleNamespace(
                stations={2: queue}, get_queue=lambda station_id: queue,
            ),
        )
        world.get_agent = lambda agent_id: agent if agent_id == 1 else None
        engine = SimulationEngine.__new__(SimulationEngine)
        engine.world = world
        engine.logger = types.SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
        )
        engine.path_planner = types.SimpleNamespace(
            plan=lambda *args, **kwargs: self.fail(
                "planner must not be called when already at entry"
            )
        )
        engine._record_path_len = lambda *args, **kwargs: None

        engine._plan_and_activate(11)

        self.assertEqual(agent.status, AgentStatus.CARRYING)
        self.assertEqual(deliver.status, TaskStatus.IN_PROGRESS)
        self.assertFalse(agent.has_path)
        self.assertEqual(queue.committed_agent_ids(), {1})

    def test_waiting_assigned_never_executes_an_active_action(self):
        """Admission wait cannot accidentally consume a DELIVER countdown."""
        agent = AgentState(agent_id=1, start_position=(2, 0))
        agent.mark_station_waiting(station_id=2, since_tick=10, sequence=1)
        agent.carried_pod_id = 10
        agent.wait_ticks = 0
        deliver = Task(
            TaskType.DELIVER, order_id=1, pod_id=10,
            source=(2, 0), destination=(0, 0),
        )
        deliver.agent_id = 1
        deliver.station_id = 2
        deliver.status = TaskStatus.IN_PROGRESS
        task_state = TaskState()
        task_state.add_task(deliver)
        world = types.SimpleNamespace(
            agents=[agent], task_state=task_state,
            pod_state=types.SimpleNamespace(get_pod=lambda pod_id: None),
            order_state=types.SimpleNamespace(orders={}),
        )
        engine = SimulationEngine.__new__(SimulationEngine)
        engine.world = world
        engine.config = types.SimpleNamespace(
            simulation=types.SimpleNamespace(
                pickup_duration=0, station_process_duration=1,
                dropoff_duration=0,
            )
        )
        engine.logger = types.SimpleNamespace(info=lambda *a, **k: None)

        engine._handle_actions(11)

        self.assertEqual(agent.status, AgentStatus.WAITING_ASSIGNED)
        self.assertEqual(deliver.status, TaskStatus.IN_PROGRESS)
        self.assertEqual(agent.wait_ticks, 0)

    def test_prioritized_planner_reserves_waiting_cell_for_full_horizon(self):
        agent = AgentState(agent_id=1, start_position=(2, 2))
        agent.mark_station_waiting(station_id=2, since_tick=4, sequence=1)
        world = types.SimpleNamespace(
            tick=0,
            agents=[agent],
        )
        planner = PrioritizedPathPlanner(max_horizon=7, goal_reserve=2)
        planner._pre_reserve_existing_paths(world)
        self.assertIn((2, 2, 1), planner._vertex_res)
        self.assertIn((2, 2, 7), planner._vertex_res)
        self.assertIn((2, 2, 9), planner._vertex_res)

    def test_waiting_tail_is_not_polled_when_capacity_is_full(self):
        queue = _queue(1)
        queue.set_admission_mode(STATION_ADMISSION_COMMITTED_FIFO_V2)
        queue.reserve(99)
        agents = []
        tasks = TaskState()
        for aid, seq in ((1, 1), (2, 2)):
            agent = AgentState(aid, (5 + aid, 5))
            agent.carried_pod_id = 100 + aid
            agent.mark_station_waiting(2, 0, seq)
            deliver = Task(
                TaskType.DELIVER, aid, 100 + aid,
                agent.position, (0, 0),
            )
            deliver.agent_id = aid
            deliver.station_id = 2
            deliver.status = TaskStatus.ASSIGNED
            tasks.add_task(deliver)
            agents.append(agent)
            queue.enqueue_waiting_reservation(aid, request_tick=0)

        world = types.SimpleNamespace(
            agents=agents, task_state=tasks,
            station_state=types.SimpleNamespace(
                stations={2: queue}, get_queue=lambda station_id: queue,
            ),
        )
        world.get_agent = lambda agent_id: next(
            (a for a in agents if a.agent_id == agent_id), None
        )
        engine = SimulationEngine.__new__(SimulationEngine)
        engine.world = world
        engine.logger = types.SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
        )
        engine.path_planner = types.SimpleNamespace(
            plan=lambda *args, **kwargs: self.fail(
                "no waiter should be promoted while capacity is full"
            )
        )

        attempts_before = queue._admission_attempts
        engine._plan_and_activate(1)
        self.assertEqual(queue._admission_attempts, attempts_before)
        self.assertEqual(queue.waiting_agent_ids(), [1, 2])

    def test_counterfactual_step_uses_the_same_fifo_waiting_contract(self):
        agent = AgentState(agent_id=1, start_position=(5, 5))
        agent.mark_station_waiting(station_id=2, since_tick=10, sequence=1)
        agent.carried_pod_id = 10
        task_state = TaskState()
        deliver = Task(
            TaskType.DELIVER, order_id=1, pod_id=10,
            source=(5, 5), destination=(0, 0),
        )
        deliver.agent_id = 1
        deliver.station_id = 2
        deliver.status = TaskStatus.ASSIGNED
        deliver.free_flow_time = 1
        task_state.add_task(deliver)
        queue = _queue(1)
        queue.set_admission_mode(STATION_ADMISSION_COMMITTED_FIFO_V2)
        queue.enqueue_waiting_reservation(1, request_tick=10)

        world = types.SimpleNamespace(
            agents=[agent],
            task_state=task_state,
            station_state=types.SimpleNamespace(
                stations={2: queue},
                get_queue=lambda station_id: queue,
            ),
        )
        world.get_agent = lambda agent_id: agent if agent_id == 1 else None
        planner = types.SimpleNamespace(
            plan=lambda *args, **kwargs: [(4, 5), (3, 5), (2, 5), (1, 0)]
        )
        config = types.SimpleNamespace(simulation=types.SimpleNamespace())

        _plan_and_activate_step(world, planner, 11, config)

        self.assertEqual(agent.status, AgentStatus.CARRYING)
        self.assertEqual(deliver.status, TaskStatus.IN_PROGRESS)
        self.assertEqual(queue.committed_agent_ids(), {1})
        self.assertEqual(queue.waiting_agent_ids(), [])


if __name__ == "__main__":
    unittest.main()
