"""Semantic and ordering tests for the isolated psi-dispatch layer."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
import unittest
from unittest import mock

from Policies.TaskAssigner.base_task_assigner import AssignmentContext
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_context_assigner import (
    PsiDispatchContextWorldModelTaskAssigner,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.world_model_task_assigner import (
    WorldModelTaskAssigner,
)
from WorldModel.core.psi_dispatch import (
    DispatchServiceDurations,
    bounded_age_score,
    bounded_backlog_score,
    build_context_debt_features,
    dispatch_cost,
)
from WorldState.order_state import Order, OrderState
from WorldState.task_state import TaskStatus


class _Distance:
    @staticmethod
    def distance(start, end):
        return abs(int(start[0]) - int(end[0])) + abs(
            int(start[1]) - int(end[1])
        )


def _context(order_id: int, station_id: int, pod_id: int) -> AssignmentContext:
    return AssignmentContext(
        order_id=order_id,
        pod_id=pod_id,
        pod_location=(pod_id, 0),
        station_id=station_id,
        station_location=(station_id, 2),
        entry_position=(station_id, 1),
        exit_position=(station_id, 1),
        return_location=(0, 0),
        order_size=1,
    )


def _world(*orders: Order, tick: int = 100):
    order_state = OrderState()
    for order in orders:
        order_state.add_order(order)
    return SimpleNamespace(
        tick=tick,
        agents=[SimpleNamespace(position=(0, 0))],
        map_state=SimpleNamespace(
            station_positions={1: (1, 2), 2: (2, 2)},
        ),
        order_state=order_state,
        task_state=SimpleNamespace(tasks={}),
        config=SimpleNamespace(
            simulation=SimpleNamespace(
                pickup_duration=1,
                station_process_duration=2,
                dropoff_duration=1,
            )
        ),
        get_idle_agents=lambda: [SimpleNamespace(position=(0, 0))],
    )


class PsiDispatchTests(unittest.TestCase):
    def test_s0_context_ordering_requires_explicit_factorial_opt_in(self):
        common = {
            "psi_head_checkpoint": "head.pt",
            "psi_scale_contract": "scale.json",
            "psi_context_mode": "j_ascending",
            "checkpoint_path": "model.pt",
            "energy_scoring_mode": "off",
            "candidate_context_mode": "prefix",
        }
        with self.assertRaises(ValueError):
            PsiDispatchContextWorldModelTaskAssigner(**common)

        assigner = PsiDispatchContextWorldModelTaskAssigner(
            allow_phasec_s0_robot_scorer=True,
            **common,
        )
        self.assertEqual(assigner.psi_robot_scorer_variant, "phasec_s0")
        self.assertEqual(assigner.energy_scoring_mode, "off")

    def test_s1_context_ordering_remains_the_default_contract(self):
        assigner = PsiDispatchContextWorldModelTaskAssigner(
            psi_head_checkpoint="head.pt",
            psi_scale_contract="scale.json",
            psi_context_mode="j_ascending",
            checkpoint_path="model.pt",
            energy_scoring_mode="conversion",
            candidate_context_mode="prefix",
        )
        self.assertEqual(
            assigner.psi_robot_scorer_variant, "s1_within_context"
        )
        self.assertEqual(assigner.energy_scoring_mode, "conversion")

    def test_bounded_terms_are_monotone_and_dimensionless(self):
        self.assertAlmostEqual(bounded_backlog_score(0, 12), 0.0)
        self.assertAlmostEqual(bounded_backlog_score(12, 12), 0.5)
        self.assertLess(bounded_backlog_score(120, 12), 1.0)
        self.assertGreater(
            bounded_backlog_score(120, 12), bounded_backlog_score(12, 12)
        )
        self.assertAlmostEqual(bounded_age_score(0, 44), 0.0)
        self.assertAlmostEqual(bounded_age_score(44, 44), 0.5)
        self.assertLess(bounded_age_score(440, 44), 1.0)


    def test_pending_work_is_independent_of_proposed_context_prefix(self):
        first = Order({"sku": 1}, station_id=1, created_at=0)
        first.pod_ids = [10]
        second = Order({"sku": 1}, station_id=1, created_at=90)
        second.pod_ids = [11]
        third = Order({"sku": 1}, station_id=2, created_at=90)
        # An unmaterialised pending order is still one conservative unit of work.
        world = _world(first, second, third)
        world.task_state.tasks = {
            1: SimpleNamespace(
                order_id=first.order_id,
                pod_id=10,
                status=TaskStatus.IN_PROGRESS,
            )
        }
        contexts = [_context(second.order_id, 1, 11)]
        features = build_context_debt_features(
            world,
            contexts,
            world.get_idle_agents(),
            _Distance(),
            DispatchServiceDurations(1, 2, 1),
        )
        debt = features[(second.order_id, 11, 1)]
        # first is active and excluded; second remains one unresolved chain.
        self.assertEqual(debt.unserved_chain_count, 1)
        self.assertEqual(debt.station_id, 1)
        self.assertGreater(debt.service_debt, 0.0)

    def test_backlog_uses_station_queue_capacity_when_available(self):
        order = Order({"sku": 1}, station_id=1, created_at=90)
        order.pod_ids = [20, 21]
        world = _world(order)
        world.station_state = SimpleNamespace(
            stations={1: SimpleNamespace(capacity=4)}
        )
        context = _context(order.order_id, 1, 20)
        features = build_context_debt_features(
            world,
            [context],
            world.get_idle_agents(),
            _Distance(),
            DispatchServiceDurations(1, 2, 1),
        )
        debt = features[(order.order_id, 20, 1)]
        self.assertEqual(debt.station_capacity, 4.0)
        self.assertEqual(debt.unserved_chain_count, 2)
        self.assertAlmostEqual(debt.backlog_score, 1.0 / 3.0)


    def test_dispatch_cost_has_declared_signs(self):
        baseline = dispatch_cost(0.2, 0.2, 0.2)
        more_load = dispatch_cost(0.8, 0.8, 0.2)
        more_debt = dispatch_cost(0.2, 0.2, 0.8)
        self.assertGreater(more_load, baseline)
        self.assertLess(more_debt, baseline)
        with self.assertRaises(ValueError):
            dispatch_cost(1.1, 0.2, 0.2)


    def test_applied_j_order_changes_only_context_order_and_preserves_parent_s1(self):
        assigner = PsiDispatchContextWorldModelTaskAssigner.__new__(
            PsiDispatchContextWorldModelTaskAssigner
        )
        assigner.psi_context_mode = "j_ascending"
        assigner.psi_trace_enabled = True
        assigner.psi_trace_max_records = 10
        assigner.psi_dispatch_trace_records = []
        assigner._psi_last_final_budget = 1
        assigner._psi_distance = _Distance()
        assigner._psi_durations = DispatchServiceDurations(1, 2, 1)
        assigner._evaluate_station_channels = lambda _world: {
            1: {"service": 0.9, "traffic": 0.9},
            2: {"service": 0.1, "traffic": 0.1},
        }
        assigner.stats = defaultdict(float)

        old = Order({"sku": 1}, station_id=1, created_at=90)
        old.pod_ids = [101]
        young = Order({"sku": 1}, station_id=2, created_at=99)
        young.pod_ids = [102]
        world = _world(old, young, tick=100)
        contexts = [
            _context(old.order_id, 1, 101),
            _context(young.order_id, 2, 102),
        ]

        with mock.patch.object(
            WorldModelTaskAssigner,
            "select_robots",
            return_value={0: 7},
        ) as parent:
            choices = assigner.select_robots(world, contexts)

        self.assertEqual(choices, {0: 7})
        self.assertEqual([context.station_id for context in contexts], [2, 1])
        parent.assert_called_once_with(world, contexts)
        trace_rows = assigner.psi_dispatch_trace_records[0]["contexts"]
        self.assertEqual(
            next(row for row in trace_rows if row["station_id"] == 2)["j_rank"],
            0,
        )


    def test_shadow_does_not_mutate_context_order(self):
        assigner = PsiDispatchContextWorldModelTaskAssigner.__new__(
            PsiDispatchContextWorldModelTaskAssigner
        )
        assigner.psi_context_mode = "shadow"
        assigner.psi_trace_enabled = False
        assigner.psi_trace_max_records = 0
        assigner.psi_dispatch_trace_records = []
        assigner._psi_last_final_budget = 1
        assigner._psi_distance = _Distance()
        assigner._psi_durations = DispatchServiceDurations(1, 2, 1)
        assigner._evaluate_station_channels = lambda _world: {
            1: {"service": 0.9, "traffic": 0.9},
            2: {"service": 0.1, "traffic": 0.1},
        }
        assigner.stats = defaultdict(float)
        first = Order({"sku": 1}, station_id=1, created_at=99)
        first.pod_ids = [201]
        second = Order({"sku": 1}, station_id=2, created_at=99)
        second.pod_ids = [202]
        world = _world(first, second)
        contexts = [
            _context(first.order_id, 1, 201),
            _context(second.order_id, 2, 202),
        ]
        original = list(contexts)
        with mock.patch.object(
            WorldModelTaskAssigner,
            "select_robots",
            return_value={0: 7},
        ):
            assigner.select_robots(world, contexts)
        self.assertEqual(contexts, original)


if __name__ == "__main__":
    unittest.main()
