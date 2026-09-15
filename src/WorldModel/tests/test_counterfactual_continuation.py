import random
import types
import unittest
from unittest import mock

import numpy as np
import torch

from WorldModel.data.counterfactual_rollout import (
    _prepare_behavior_continuation_tick,
    evaluate_candidate_rollout,
    force_apply_candidate,
)
from WorldState.agent_state import AgentStatus
from WorldState.order_state import Order, OrderState, OrderStatus
from WorldState.task_state import Task, TaskState, TaskStatus, TaskType


def _config(backlog_floor=0, refill_mode="none"):
    return types.SimpleNamespace(
        simulation=types.SimpleNamespace(
            pickup_duration=2,
            station_process_duration=5,
            dropoff_duration=2,
            backlog_floor=backlog_floor,
            backlog_refill_mode=refill_mode,
        )
    )


class _Generator:
    def __init__(self):
        self.calls = 0

    def generate(self, world):
        self.calls += 1
        return [Order({"sku": 1}, station_id=1, created_at=world.tick)]


class _Assigner:
    def __init__(self):
        self.pending_seen = []
        self.last_tasks = []

    def assign(self, world):
        self.pending_seen.append(len(world.order_state.get_pending_orders()))
        task = Task(TaskType.PICK, -1, -1, (0, 0), (0, 1))
        task.agent_id = 7
        task.status = TaskStatus.ASSIGNED
        self.last_tasks = [task]
        return self.last_tasks


class CounterfactualContinuationTests(unittest.TestCase):
    def test_decision_tick_assigns_without_generating_orders(self):
        world = types.SimpleNamespace(
            tick=10,
            order_state=OrderState(),
        )
        existing = Order({"sku": 1}, station_id=1, created_at=10)
        world.order_state.add_order(existing)
        generator = _Generator()
        assigner = _Assigner()

        generated, assigned = _prepare_behavior_continuation_tick(
            world,
            generator,
            assigner,
            _config(),
            generate_orders=False,
        )

        self.assertEqual(generated, 0)
        self.assertEqual(assigned, 1)
        self.assertEqual(generator.calls, 0)
        self.assertEqual(assigner.pending_seen, [1])

    def test_later_tick_generates_before_assignment_and_stamps_tasks(self):
        world = types.SimpleNamespace(
            tick=11,
            order_state=OrderState(),
        )
        generator = _Generator()
        assigner = _Assigner()
        config = _config()

        generated, assigned = _prepare_behavior_continuation_tick(
            world,
            generator,
            assigner,
            config,
            generate_orders=True,
        )

        self.assertEqual(generated, 1)
        self.assertEqual(assigned, 1)
        self.assertEqual(assigner.pending_seen, [1])
        task = assigner.last_tasks[0]
        self.assertEqual(task.created_at, 11)
        self.assertEqual(task.assigned_at, 11)
        self.assertEqual(task.free_flow_time, 3)

    def test_forced_single_pod_keeps_multi_pod_order_dispatchable(self):
        order = Order({"sku": 2}, station_id=1)
        order.pod_ids = [10, 11]
        orders = OrderState()
        orders.add_order(order)
        tasks = TaskState()
        agent = types.SimpleNamespace(
            agent_id=7,
            status=AgentStatus.IDLE,
            position=(0, 0),
            assigned_task_id=None,
        )
        pod = types.SimpleNamespace(
            pod_id=10,
            is_carried=False,
            current_position=(0, 2),
        )
        world = types.SimpleNamespace(
            tick=4,
            order_state=orders,
            task_state=tasks,
            pod_state=types.SimpleNamespace(
                get_pod=lambda pod_id: pod if pod_id == 10 else None
            ),
            get_agent=lambda agent_id: agent if agent_id == 7 else None,
        )
        candidate = {"robot_id": 7, "robot_start": (0, 0)}
        context = {
            "order_id": order.order_id,
            "pod_id": 10,
            "pod_location": (0, 2),
            "station_id": 1,
            "station_location": (0, 8),
            "return_location": (0, 2),
        }

        self.assertTrue(
            force_apply_candidate(world, candidate, context, _config())
        )
        self.assertEqual(order.status, OrderStatus.PENDING)

    def test_candidate_rollout_restores_all_global_rng_states(self):
        class _World:
            def __init__(self):
                self.tick = 0
                self.map_state = types.SimpleNamespace(station_positions={1: (0, 1)})
                self.station_state = types.SimpleNamespace(get_queue=lambda _: None)
                self.order_state = types.SimpleNamespace(total_completed=0)
                self.agent = types.SimpleNamespace(
                    status=AgentStatus.IDLE,
                    plan_failed_streak=0,
                )

            def get_agent(self, _):
                return self.agent

        def noisy_step(world, _planner, _config):
            random.random()
            np.random.random()
            torch.rand(1)
            world.tick += 1
            return 0, 0, 0

        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        expected = (random.random(), np.random.random(), torch.rand(1))
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

        with (
            mock.patch(
                "WorldModel.data.counterfactual_rollout.force_apply_candidate",
                return_value=True,
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.step_world",
                side_effect=noisy_step,
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.extract_node_labels",
                return_value=torch.zeros(1, 6),
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.extract_system_labels",
                return_value=torch.zeros(7),
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.extract_station_labels",
                return_value=torch.zeros(1, 2),
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.extract_demand_context",
                return_value=torch.zeros(1),
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.compute_realized_cost",
                return_value=0.0,
            ),
        ):
            evaluate_candidate_rollout(
                world=_World(),
                candidate={"robot_id": 7},
                fixed_context={},
                config=_config(),
                path_planner=object(),
                horizon=1,
                node_map={(0, 0): 0},
                local_capacity=[1],
                bottleneck_score=[0],
                adj={},
            )

        actual = (random.random(), np.random.random(), torch.rand(1))
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])

    def test_l0_collection_persists_station_work_at_every_horizon_tick(self):
        class _World:
            def __init__(self):
                self.tick = 0
                self.map_state = types.SimpleNamespace(
                    station_positions={1: (0, 1), 2: (0, 2)}
                )
                self.station_state = types.SimpleNamespace(
                    get_queue=lambda _: None
                )
                self.order_state = types.SimpleNamespace(total_completed=0)
                self.agent = types.SimpleNamespace(
                    status=AgentStatus.IDLE,
                    plan_failed_streak=0,
                )

            def get_agent(self, _):
                return self.agent

        class _Snapshot:
            def __init__(self, tick):
                self.tick = int(tick)
                self.total = float(30 - 2 * tick)
                self.station_work = {
                    1: float(10 - tick),
                    2: float(20 - tick),
                }
                self.traffic_diagnostics = {}

            def physical_summary(self):
                return (self.total,) + (0.0,) * 11

            def to_dict(self):
                return {
                    "tick": self.tick,
                    "total": self.total,
                    "station_work": dict(self.station_work),
                    "work_capacity": 1.0,
                    "components": {"work": self.total},
                    "chains": {},
                }

        class _Progress:
            @staticmethod
            def to_dict():
                return {"horizon": 2}

        def advance(world, _planner, _config):
            world.tick += 1
            return 0, 0, 0

        with (
            mock.patch(
                "WorldModel.data.counterfactual_rollout.force_apply_candidate",
                return_value=True,
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.step_world",
                side_effect=advance,
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.extract_node_labels",
                return_value=torch.zeros(1, 6),
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.extract_system_labels",
                return_value=torch.zeros(7),
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.extract_station_labels",
                return_value=torch.zeros(2, 2),
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.extract_demand_context",
                return_value=torch.zeros(1),
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.compute_realized_cost",
                return_value=0.0,
            ),
            mock.patch(
                "WorldModel.core.lyapunov.compute_lyapunov_snapshot",
                side_effect=lambda world, _config: _Snapshot(world.tick),
            ),
            mock.patch(
                "WorldModel.core.lyapunov.compute_productive_progress",
                return_value=_Progress(),
            ),
        ):
            result = evaluate_candidate_rollout(
                world=_World(),
                candidate={"robot_id": 7},
                fixed_context={},
                config=_config(),
                path_planner=object(),
                horizon=2,
                node_map={(0, 0): 0},
                local_capacity=[1],
                bottleneck_score=[0],
                adj={},
                record_lyapunov_l0=True,
            )

        self.assertEqual(
            result["analytic_work_relief_trajectory_schema_version"],
            "analytic_work_relief_trajectory_v1",
        )
        torch.testing.assert_close(
            result["lyapunov_l0_station_ids"],
            torch.tensor([1, 2]),
        )
        torch.testing.assert_close(
            result["lyapunov_l0_station_work_trajectory"],
            torch.tensor([[9.0, 19.0], [8.0, 18.0]]),
        )


if __name__ == "__main__":
    unittest.main()
