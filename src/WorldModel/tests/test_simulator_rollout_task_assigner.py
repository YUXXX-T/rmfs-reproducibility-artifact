from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from Policies.TaskAssigner.base_task_assigner import AssignmentContext
from Policies.TaskAssigner.SimulatorRolloutTaskAssigner import (
    SimulatorRolloutTaskAssigner,
)


class _Agent:
    def __init__(self, agent_id, position):
        self.agent_id = agent_id
        self.position = position


class _World:
    def __init__(self, agents):
        self.tick = 7
        self._agents = list(agents)
        self.config = SimpleNamespace(
            simulation=SimpleNamespace(task_execution_mode="parallel")
        )

    def get_idle_agents(self):
        return list(self._agents)


def _context(index):
    return AssignmentContext(
        order_id=100 + index,
        pod_id=200 + index,
        pod_location=(0, index),
        station_id=1,
        station_location=(5, 5),
        entry_position=(5, 4),
        exit_position=(5, 6),
        return_location=(9, 9),
        order_size=1,
    )


class SimulatorRolloutTaskAssignerTest(TestCase):
    def _assigner(self):
        assigner = SimulatorRolloutTaskAssigner(
            rollout_horizon=10,
            candidate_limit=0,
        )
        assigner.path_planner = object()
        assigner._initialized = True
        assigner._node_map = {}
        assigner._local_capacity = []
        assigner._bottleneck_score = []
        assigner._adj = {}
        return assigner

    @patch(
        "Policies.TaskAssigner.SimulatorRolloutTaskAssigner."
        "simulator_rollout_task_assigner.evaluate_candidate_rollout"
    )
    def test_selects_minimum_cost_without_reusing_robot(self, rollout):
        costs = {
            (200, 1): 1.0,
            (200, 2): 2.0,
            (200, 3): 3.0,
            (201, 2): 4.0,
            (201, 3): 0.5,
        }

        def result(_world, candidate, fixed_context, *_args, **_kwargs):
            return {
                "future_mask": torch.ones(10),
                "realized_cost": costs[
                    (fixed_context["pod_id"], candidate["robot_id"])
                ],
                "rollout_vertex_conflicts": 0,
                "rollout_swap_conflicts": 0,
                "rollout_blocked_moves": 0,
            }

        rollout.side_effect = result
        world = _World([
            _Agent(1, (0, 0)),
            _Agent(2, (0, 1)),
            _Agent(3, (0, 2)),
        ])
        assigner = self._assigner()

        choices = assigner.select_robots(world, [_context(0), _context(1)])

        self.assertEqual(choices, {0: 1, 1: 3})
        self.assertEqual(assigner.stats["oracle_rollout_calls"], 5)
        self.assertEqual(assigner.stats["oracle_contexts_selected"], 2)
        self.assertEqual(assigner.stats["oracle_selected_nearest_count"], 1)
        self.assertEqual(assigner.stats["oracle_incomplete_rollouts"], 0)

    @patch(
        "Policies.TaskAssigner.SimulatorRolloutTaskAssigner."
        "simulator_rollout_task_assigner.evaluate_candidate_rollout"
    )
    def test_incomplete_rollout_fails_closed(self, rollout):
        rollout.return_value = {
            "future_mask": torch.zeros(10),
            "realized_cost": 0.0,
        }
        world = _World([_Agent(1, (0, 0))])
        assigner = self._assigner()

        with self.assertRaisesRegex(RuntimeError, "incomplete simulator-oracle"):
            assigner.select_robots(world, [_context(0)])

        self.assertEqual(assigner.stats["oracle_incomplete_rollouts"], 1)

