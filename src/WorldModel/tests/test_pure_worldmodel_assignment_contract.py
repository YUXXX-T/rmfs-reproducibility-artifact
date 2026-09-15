import inspect
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from Policies.policy_registry import get_policy
from Policies.TaskAssigner import GreedyTaskAssigner, WorldModelTaskAssigner
from Policies.TaskAssigner.base_task_assigner import BaseTaskAssigner
from Policies.TaskAssigner.context_assignment import (
    materialize_fixed_order_pod_contexts,
)


def _world(execution_mode="parallel"):
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            simulation=types.SimpleNamespace(
                task_execution_mode=execution_mode,
            )
        )
    )


class FixedContextWorldModelContractTests(unittest.TestCase):
    @staticmethod
    def _runtime_stub(*, ready):
        assigner = WorldModelTaskAssigner.__new__(WorldModelTaskAssigner)
        assigner._initialized = True
        assigner._feature_history = types.SimpleNamespace(is_ready=ready)
        assigner._update_features = lambda world: None
        assigner.stats = {
            "assign_calls": 0,
            "model_assign_calls": 0,
            "fallback_greedy_calls": 0,
            "warmup_defer_calls": 0,
            "assignment_time_total_ms": 0.0,
        }
        return assigner

    def test_public_entry_point_is_not_a_greedy_subclass(self):
        self.assertIs(
            get_policy("task_assigner", "WorldModelTaskAssigner"),
            WorldModelTaskAssigner,
        )
        self.assertTrue(issubclass(WorldModelTaskAssigner, BaseTaskAssigner))
        self.assertFalse(issubclass(WorldModelTaskAssigner, GreedyTaskAssigner))

    def test_training_top_m_never_truncates_online_idle_robots(self):
        assigner = WorldModelTaskAssigner.__new__(WorldModelTaskAssigner)
        assigner.top_m = 1
        assigner.stats = {
            "all_idle_candidate_contexts": 0,
            "all_idle_candidates_scored": 0,
        }
        robots = [
            types.SimpleNamespace(agent_id=robot_id)
            for robot_id in (7, 2, 9, 1)
        ]
        selected = assigner._select_robot_candidates(None, None, robots)
        self.assertEqual([robot.agent_id for robot in selected], [1, 2, 7, 9])
        self.assertEqual(assigner.stats["all_idle_candidates_scored"], 4)

    def test_history_warmup_defers_without_greedy(self):
        assigner = self._runtime_stub(ready=False)
        self.assertEqual(assigner.assign(_world()), [])
        self.assertEqual(assigner.stats["warmup_defer_calls"], 1)
        self.assertEqual(assigner.stats["fallback_greedy_calls"], 0)

    def test_missing_checkpoint_fails_before_random_model_initialisation(self):
        path = f"{tempfile.gettempdir()}/definitely_missing_fixed_context_wm.pt"
        assigner = WorldModelTaskAssigner(checkpoint_path=path)
        with self.assertRaisesRegex(FileNotFoundError, "random model"):
            assigner._init(None)

    def test_serial_mode_fails_instead_of_falling_back(self):
        assigner = self._runtime_stub(ready=True)
        with self.assertRaisesRegex(RuntimeError, "serial mode"):
            assigner.assign(_world("serial"))
        self.assertEqual(assigner.stats["fallback_greedy_calls"], 0)

    def test_fixed_context_proposal_is_policy_neutral_helper(self):
        assigner = WorldModelTaskAssigner.__new__(WorldModelTaskAssigner)
        sentinel = [object()]
        module = sys.modules[WorldModelTaskAssigner.__module__]
        with patch.object(
            module,
            "propose_fixed_assignment_contexts",
            return_value=sentinel,
        ) as helper:
            result = assigner.propose_assignment_contexts("world", max_contexts=3)
        self.assertIs(result, sentinel)
        helper.assert_called_once_with(assigner, "world", max_contexts=3)

    def test_pod_context_is_materialized_before_not_inside_robot_policy(self):
        order = types.SimpleNamespace(pod_ids=[])
        world = types.SimpleNamespace(
            order_state=types.SimpleNamespace(
                get_pending_orders=lambda: [order]
            )
        )
        retriever = types.SimpleNamespace(
            retrieve=lambda current_order, current_world: [11, 12]
        )
        materialize_fixed_order_pod_contexts(world, retriever)
        self.assertEqual(order.pod_ids, [11, 12])
        self.assertTrue(WorldModelTaskAssigner.requires_external_fixed_context)

    def test_decision_path_contains_no_greedy_robot_call_or_nearest_fallback(self):
        source = inspect.getsource(WorldModelTaskAssigner.select_robots)
        self.assertNotIn("super().select_robots", source)
        self.assertNotIn("best_robot = candidates[0].agent_id", source)


if __name__ == "__main__":
    unittest.main()
