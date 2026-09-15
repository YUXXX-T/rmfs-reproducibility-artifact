import types
import unittest

from Policies.TaskAssigner.base_task_assigner import (
    fulfilled_pod_ids_for_order,
)
from Policies.TaskAssigner.GreedyTaskAssigner.greedy_task_assigner import (
    GreedyTaskAssigner,
)
from Policies.TaskAssigner.HungarianTaskAssigner.hungarian_task_assigner import (
    HungarianTaskAssigner,
)
from Policies.PodRetriever.DefaultPodRetriever import DefaultPodRetriever
from WorldModel.data.candidate_generator import generate_robot_candidates
from WorldModel.data.counterfactual_rollout import force_apply_candidate
from WorldState.agent_state import AgentStatus
from WorldState.order_state import Order, OrderState, OrderStatus
from WorldState.task_state import Task, TaskState, TaskStatus, TaskType


def _completed_chain(order_id, pod_id, agent_id=90):
    tasks = (
        Task(TaskType.PICK, order_id, pod_id, (0, 0), (0, 2)),
        Task(TaskType.DELIVER, order_id, pod_id, (0, 2), (0, 8)),
        Task(TaskType.RETURN, order_id, pod_id, (0, 8), (0, 2)),
    )
    for task in tasks:
        task.agent_id = agent_id
        task.status = TaskStatus.COMPLETED
    return tasks


class _World:
    def __init__(self, order, *, delivered=True, include_completed_chain=True):
        self.tick = 80
        self.config = types.SimpleNamespace(
            simulation=types.SimpleNamespace(
                task_execution_mode="parallel",
                pickup_duration=2,
                station_process_duration=5,
                dropoff_duration=2,
            )
        )
        self.order_state = OrderState()
        self.order_state.add_order(order)
        if delivered:
            order.delivered_pod_ids = [10]

        self.task_state = TaskState()
        if include_completed_chain:
            for task in _completed_chain(order.order_id, 10):
                self.task_state.add_task(task)

        self.pods = {
            pod_id: types.SimpleNamespace(
                pod_id=pod_id,
                is_carried=False,
                current_position=(0, pod_id - 8),
                home_position=(0, pod_id - 8),
                sku_inventory={"sku": 2},
            )
            for pod_id in (10, 11)
        }
        self.pod_state = types.SimpleNamespace(
            get_pod=lambda pod_id: self.pods.get(pod_id),
            get_available_pods=lambda: list(self.pods.values()),
        )
        self.station_state = types.SimpleNamespace(
            get_service_position=lambda _station_id: (0, 8),
            get_entry_position=lambda _station_id: (1, 8),
            get_exit_position=lambda _station_id: (0, 9),
        )
        self.map_state = types.SimpleNamespace(station_positions={1: (0, 8)})
        self.agent = types.SimpleNamespace(
            agent_id=7,
            status=AgentStatus.IDLE,
            position=(0, 0),
            assigned_task_id=None,
        )
        self.agents = [self.agent]

    def get_idle_agents(self):
        return [agent for agent in self.agents if agent.status == AgentStatus.IDLE]

    def get_agent(self, agent_id):
        return self.agent if agent_id == self.agent.agent_id else None


class CompletedPodAssignmentGuardTests(unittest.TestCase):
    def _order(self):
        order = Order({"sku": 2}, station_id=1)
        order.pod_ids = [10, 11]
        return order

    def test_completed_deliver_is_a_defensive_fulfilment_fallback(self):
        order = self._order()
        world = _World(order, delivered=False, include_completed_chain=True)

        self.assertEqual(fulfilled_pod_ids_for_order(world, order), {10})

    def test_propose_and_candidate_generation_skip_fulfilled_pod(self):
        order = self._order()
        world = _World(order)
        assigner = GreedyTaskAssigner()

        contexts = assigner.propose_assignment_contexts(world, max_contexts=2)
        self.assertEqual([context.pod_id for context in contexts], [11])

        groups = generate_robot_candidates(contexts, world, top_m=1)
        self.assertEqual(
            [group["fixed_context"]["pod_id"] for group in groups],
            [11],
        )

    def test_committing_last_unfulfilled_pod_closes_order_coverage(self):
        order = self._order()
        world = _World(order)
        assigner = GreedyTaskAssigner()
        contexts = assigner.propose_assignment_contexts(world, max_contexts=2)

        new_tasks = assigner.commit_assignments(world, contexts, {0: 7})

        self.assertEqual(len(new_tasks), 3)
        self.assertEqual({task.pod_id for task in new_tasks}, {11})
        self.assertEqual(order.status, OrderStatus.IN_PROGRESS)
        pod10_tasks = [
            task for task in world.task_state.tasks.values()
            if task.order_id == order.order_id and task.pod_id == 10
        ]
        self.assertEqual(len(pod10_tasks), 3)
        self.assertTrue(
            all(task.status == TaskStatus.COMPLETED for task in pod10_tasks)
        )

    def test_force_apply_rejects_stale_fulfilled_pod_candidate(self):
        order = self._order()
        world = _World(order)
        candidate = {"robot_id": 7, "robot_start": (0, 0)}
        context = {
            "order_id": order.order_id,
            "pod_id": 10,
            "pod_location": (0, 2),
            "station_id": 1,
            "station_location": (0, 8),
            "entry_position": (1, 8),
            "exit_position": (0, 9),
            "return_location": (0, 2),
        }
        before_task_ids = set(world.task_state.tasks)

        applied = force_apply_candidate(
            world, candidate, context, world.config,
        )

        self.assertFalse(applied)
        self.assertEqual(set(world.task_state.tasks), before_task_ids)
        self.assertEqual(world.agent.status, AgentStatus.IDLE)
        self.assertEqual(order.status, OrderStatus.PENDING)

    def test_hungarian_assigner_skips_fulfilled_pod(self):
        order = self._order()
        world = _World(order)
        assigner = HungarianTaskAssigner()

        new_tasks = assigner.assign(world)

        self.assertEqual(len(new_tasks), 3)
        self.assertEqual({task.pod_id for task in new_tasks}, {11})
        self.assertEqual(order.status, OrderStatus.IN_PROGRESS)
        deliver = next(
            task for task in new_tasks
            if task.task_type == TaskType.DELIVER
        )
        return_task = next(
            task for task in new_tasks
            if task.task_type == TaskType.RETURN
        )
        self.assertEqual(deliver.station_id, 1)
        self.assertEqual(return_task.station_id, 1)
        self.assertEqual(return_task.source, (0, 9))

    def test_default_retriever_never_returns_fulfilled_pod(self):
        order = self._order()
        world = _World(order)
        retriever = DefaultPodRetriever()

        selected = retriever.retrieve(order, world)

        self.assertNotIn(10, selected)
        self.assertEqual(selected, [11])


if __name__ == "__main__":
    unittest.main()
