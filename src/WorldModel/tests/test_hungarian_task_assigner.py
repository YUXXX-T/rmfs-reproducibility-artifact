import types
import unittest

from Config.config_loader import load_config
from Policies.TaskAssigner.HungarianTaskAssigner import HungarianTaskAssigner
from WorldModel.evaluation.evaluate import _build_engine
from WorldModel.evaluation.evaluate_online_v6 import (
    reset_global_ids,
    set_global_seed,
)
from WorldState.agent_state import AgentStatus
from WorldState.order_state import Order, OrderState, OrderStatus
from WorldState.task_state import TaskState, TaskType


class _MatchingWorld:
    def __init__(self):
        self.order_state = OrderState()
        first = Order({"sku": 1}, station_id=1)
        first.pod_ids = [10]
        second = Order({"sku": 1}, station_id=1)
        second.pod_ids = [11]
        self.order_state.add_order(first)
        self.order_state.add_order(second)

        self.task_state = TaskState()
        self.agents = [
            types.SimpleNamespace(
                agent_id=1,
                position=(0, 0),
                status=AgentStatus.IDLE,
                assigned_task_id=None,
            ),
            types.SimpleNamespace(
                agent_id=2,
                position=(0, 10),
                status=AgentStatus.IDLE,
                assigned_task_id=None,
            ),
        ]
        pods = {
            10: types.SimpleNamespace(
                pod_id=10,
                current_position=(0, 9),
                home_position=(0, 9),
                is_carried=False,
            ),
            11: types.SimpleNamespace(
                pod_id=11,
                current_position=(0, 1),
                home_position=(0, 1),
                is_carried=False,
            ),
        }
        self.pod_state = types.SimpleNamespace(
            get_pod=lambda pod_id: pods.get(pod_id)
        )
        self.station_state = types.SimpleNamespace(
            get_service_position=lambda _station_id: (2, 5),
            get_entry_position=lambda _station_id: (1, 5),
            get_exit_position=lambda _station_id: (2, 6),
        )
        self.map_state = types.SimpleNamespace(
            station_positions={1: (2, 5)}
        )

    def get_idle_agents(self):
        return [
            agent for agent in self.agents
            if agent.status == AgentStatus.IDLE
        ]

    def get_agent(self, agent_id):
        return next(
            (agent for agent in self.agents if agent.agent_id == agent_id),
            None,
        )


class HungarianTaskAssignerTests(unittest.TestCase):
    def test_global_manhattan_matching_and_shared_task_commit(self):
        world = _MatchingWorld()

        tasks = HungarianTaskAssigner().assign(world)

        picks = {
            task.agent_id: task.pod_id
            for task in tasks
            if task.task_type == TaskType.PICK
        }
        self.assertEqual(picks, {1: 11, 2: 10})
        self.assertEqual(len(tasks), 6)
        for task in tasks:
            if task.task_type in (TaskType.DELIVER, TaskType.RETURN):
                self.assertEqual(task.station_id, 1)
            if task.task_type == TaskType.RETURN:
                self.assertEqual(task.source, (2, 6))
        self.assertTrue(
            all(
                order.status == OrderStatus.IN_PROGRESS
                for order in world.order_state.orders.values()
            )
        )

    def test_small_closed_loop_completes_pick_deliver_return_and_order(self):
        seed = 9972
        config = load_config("Config/world_model_config_PP_48_low.json")
        config.robots.num_robots = 4
        config.simulation.max_ticks = 120
        config.simulation.seed = seed
        config.simulation.initial_order_pool_size = 1
        config.simulation.backlog_floor = 0
        config.simulation.backlog_refill_mode = "none"
        config.simulation.order_interval = 1000
        config.simulation.fixed_order_size = 1
        config.simulation.max_items_per_order = 1
        config.simulation.max_items_per_sku = 1
        set_global_seed(seed)
        reset_global_ids()

        engine = _build_engine(
            config,
            task_assigner=HungarianTaskAssigner(),
        )
        engine.run()

        completed_tasks = [
            task for task in engine.world.task_state.tasks.values()
            if task.status.name == "COMPLETED"
        ]
        self.assertEqual(engine.world.order_state.total_completed, 1)
        self.assertEqual(len(completed_tasks), 3)
        self.assertEqual(
            {task.task_type for task in completed_tasks},
            {TaskType.PICK, TaskType.DELIVER, TaskType.RETURN},
        )


if __name__ == "__main__":
    unittest.main()
