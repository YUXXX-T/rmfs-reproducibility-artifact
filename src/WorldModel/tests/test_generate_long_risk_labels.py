from __future__ import annotations

import copy
import pickle
import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from WorldModel.data.candidate_generator import make_no_assign_candidate
from WorldModel.data.generate_long_risk_labels import process_snapshot
from WorldState.order_state import Order, OrderState
from WorldState.pod_state import Pod, PodState
from WorldState.task_state import Task, TaskState


class _NoAssignRolloutWorld:
    """Minimal picklable world used by the real label-rollout regression.

    ``get_agent`` deliberately fails for every call.  A native NO_ASSIGN
    rollout must never ask the world for an agent, while the validator still
    exercises the real order/pod/task checks on this cloned world.
    """

    last_clone = None

    def __init__(self, order_state, pod_state, task_state, tick=40):
        self.tick = tick
        self.order_state = order_state
        self.pod_state = pod_state
        self.task_state = task_state

    def get_agent(self, agent_id):
        raise AssertionError(
            f"NO_ASSIGN rollout attempted to resolve robot_id={agent_id!r}"
        )

    def __deepcopy__(self, memo):
        clone = type(self)(
            order_state=copy.deepcopy(self.order_state, memo),
            pod_state=copy.deepcopy(self.pod_state, memo),
            task_state=copy.deepcopy(self.task_state, memo),
            tick=self.tick,
        )
        memo[id(self)] = clone
        type(self).last_clone = clone
        return clone


class GenerateLongRiskLabelsTests(unittest.TestCase):
    def test_no_assign_candidate_key_matches_phase_c_dataset_builder(self):
        snapshot = {
            "candidate_group_id": "group7",
            "decision_tick": 40,
            "candidates": [
                {"action_type": "assign_robot", "robot_id": 3},
                make_no_assign_candidate(),
            ],
            "fixed_context": {
                "order_id": 11,
                "pod_id": 12,
                "station_id": 2,
            },
        }
        result = {
            "risk_peak": 0.2,
            "risk_cvar": 0.2,
            "risk_terminal": 0.2,
            "risk_event": 0.0,
            "risk_delta_current": 0.1,
            "r_at_decision": 0.1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.pkl"
            with path.open("wb") as handle:
                pickle.dump(snapshot, handle)
            with patch(
                "WorldModel.data.generate_long_risk_labels._rollout_candidate",
                return_value=result,
            ):
                labels = process_snapshot(str(path), 200, 50)

        keys = {row["candidate_key"] for row in labels}
        self.assertIn("group7_r3_o11_p12_s2_t40", keys)
        self.assertIn("group7_rNO_ASSIGN_o11_p12_s2_t40", keys)
        self.assertNotIn("group7_rNone_o11_p12_s2_t40", keys)

    def test_real_no_assign_rollout_defers_without_robot_lookup_or_task(self):
        """Exercise the native NO_ASSIGN branch through process_snapshot.

        The old implementation unconditionally called
        ``force_apply_candidate`` and therefore reached ``get_agent(None)``.
        This test intentionally makes that lookup fail, while keeping the
        surrounding snapshot/clone/validation/label path real.  Only the
        expensive physical step and risk extraction are stubbed so the test
        remains deterministic and fast.
        """
        old_task_next_id = Task._next_id
        old_order_next_id = Order._next_id
        try:
            Task._next_id = 0
            Order._next_id = 0
            _NoAssignRolloutWorld.last_clone = None

            order = Order({"sku": 1}, station_id=2, created_at=40)
            order.pod_ids = [7]
            order_state = OrderState()
            order_state.add_order(order)

            pod_state = PodState()
            pod_state.add_pod(Pod(7, (0, 1)))
            task_state = TaskState()
            world = _NoAssignRolloutWorld(
                order_state=order_state,
                pod_state=pod_state,
                task_state=task_state,
            )

            fixed_context = {
                "order_id": order.order_id,
                "pod_id": 7,
                "pod_location": (0, 1),
                "station_id": 2,
                "station_location": (0, 2),
                "entry_position": (0, 3),
                "exit_position": (0, 4),
                "return_location": (0, 1),
                "order_size": 1,
            }
            snapshot = {
                "candidate_group_id": "group_no_assign",
                "decision_tick": 40,
                "candidates": [make_no_assign_candidate()],
                "fixed_context": fixed_context,
                "world_snapshot": world,
                "config": types.SimpleNamespace(
                    simulation=types.SimpleNamespace(
                        backlog_refill_mode="none",
                    )
                ),
                "path_planner_state": types.SimpleNamespace(),
                "order_generator_state": types.SimpleNamespace(),
                "task_next_id": Task._next_id,
                "order_next_id": Order._next_id,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.random.get_rng_state(),
            }

            def advance(world_state, _planner, _config):
                world_state.tick += 1
                return 0, 0, 0

            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "no_assign_snapshot.pkl"
                with path.open("wb") as handle:
                    pickle.dump(snapshot, handle)

                with (
                    patch(
                        "WorldModel.data.generate_long_risk_labels.step_world",
                        side_effect=advance,
                    ),
                    patch(
                        "WorldModel.data.generate_long_risk_labels.compute_unified_risk",
                        return_value={"unified_risk": 0.25},
                    ),
                    patch(
                        "WorldModel.data.generate_long_risk_labels._make_continuation_assigner",
                        return_value=types.SimpleNamespace(stats={}),
                    ),
                ):
                    labels = process_snapshot(
                        str(path),
                        W=1,
                        terminal_window=1,
                    )

            self.assertEqual(len(labels), 1)
            row = labels[0]
            self.assertEqual(
                row["candidate_key"],
                "group_no_assign_rNO_ASSIGN_o0_p7_s2_t40",
            )
            for field in (
                "risk_peak",
                "risk_cvar",
                "risk_terminal",
                "risk_delta_group",
                "risk_event",
            ):
                self.assertIn(field, row)
                self.assertTrue(np.isfinite(float(row[field])))
            self.assertEqual(row["risk_event"], 0.0)
            self.assertIsNone(row["robot_id"])
            self.assertEqual(task_state.tasks, {})
            self.assertIsNotNone(_NoAssignRolloutWorld.last_clone)
            self.assertEqual(
                _NoAssignRolloutWorld.last_clone.task_state.tasks,
                {},
            )
        finally:
            Task._next_id = old_task_next_id
            Order._next_id = old_order_next_id


if __name__ == "__main__":
    unittest.main()
