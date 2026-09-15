"""Tests for the isolated Phase-C ``phi_state`` context ranker."""

from __future__ import annotations

import types
import unittest
from unittest import mock

from Policies.TaskAssigner.base_task_assigner import AssignmentContext
from Policies.TaskAssigner.WorldModelTaskAssigner.phi_context_rank_assigner import (
    PhiContextRankWorldModelTaskAssigner,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.world_model_task_assigner import (
    WorldModelTaskAssigner,
)
from WorldModel.evaluation.phase_c_phi_context_ablation_protocol import (
    LOCKED_REFERENCE_SEEDS,
    PHI_CONTEXT_FACTOR,
    PHI_VALIDATION_SEEDS,
    SEEDS,
    protocol_payload,
)


def _context(order: int, station: int) -> AssignmentContext:
    return AssignmentContext(
        order_id=order,
        pod_id=100 + order,
        pod_location=(order, 0),
        station_id=station,
        station_location=(station, 1),
        entry_position=None,
        exit_position=None,
        return_location=(order, 2),
        order_size=1,
    )


def _runtime(mode: str) -> PhiContextRankWorldModelTaskAssigner:
    assigner = PhiContextRankWorldModelTaskAssigner.__new__(
        PhiContextRankWorldModelTaskAssigner
    )
    assigner.phi_context_mode = mode
    assigner.phi_context_factor = 2.0
    assigner.phi_trace_enabled = True
    assigner.phi_trace_max_records = 10
    assigner.phi_context_trace_records = []
    assigner._phi_last_final_budget = 2
    assigner.stats = {
        "phi_context_eval_calls": 0,
        "phi_context_eval_time_total_ms": 0.0,
        "phi_context_contexts_seen": 0,
        "phi_context_unique_stations_sum": 0,
        "phi_context_multi_station_calls": 0,
        "phi_context_reordered_calls": 0,
        "phi_context_changed_positions": 0,
        "phi_context_replacement_count": 0,
        "phi_context_replacement_calls": 0,
        "phi_context_pressure_count": 0,
        "phi_context_pressure_sum": 0.0,
        "phi_context_pressure_min": float("inf"),
        "phi_context_pressure_max": float("-inf"),
        "phi_context_pressure_spread_sum": 0.0,
        "phi_context_pressure_spread_count": 0,
        "phi_context_baseline_budget_pressure_sum": 0.0,
        "phi_context_applied_budget_pressure_sum": 0.0,
        "phi_context_budget_pressure_count": 0,
        "phi_context_selected_pressure_sum": 0.0,
        "phi_context_selected_pressure_count": 0,
        "phi_context_trace_dropped": 0,
    }
    assigner._current_station_service_pressure = lambda _world: {
        1: 0.8,
        2: 0.2,
        3: 0.5,
    }
    return assigner


class PhiContextRankTests(unittest.TestCase):
    def test_pressure_order_is_stable_and_low_pressure_first(self):
        contexts = [_context(1, 1), _context(2, 2), _context(3, 2)]
        order = PhiContextRankWorldModelTaskAssigner._stable_pressure_order(
            contexts, {1: 0.9, 2: 0.1}
        )
        self.assertEqual(order, [1, 2, 0])

    def test_applied_mode_reorders_contexts_before_parent_robot_scoring(self):
        assigner = _runtime("service_ascending")
        contexts = [_context(1, 1), _context(2, 2), _context(3, 3)]
        world = types.SimpleNamespace(
            tick=25,
            get_idle_agents=lambda: [object(), object()],
        )
        with mock.patch.object(
            WorldModelTaskAssigner,
            "select_robots",
            return_value={0: 7, 1: 8},
        ) as parent:
            choices = assigner.select_robots(world, contexts)
        self.assertEqual([context.station_id for context in contexts], [2, 3, 1])
        self.assertEqual(choices, {0: 7, 1: 8})
        parent.assert_called_once_with(world, contexts)
        self.assertEqual(assigner.stats["phi_context_replacement_count"], 1)
        self.assertLess(
            assigner.stats["phi_context_applied_budget_pressure_sum"],
            assigner.stats["phi_context_baseline_budget_pressure_sum"],
        )

    def test_shadow_mode_records_but_does_not_change_execution_order(self):
        assigner = _runtime("shadow")
        contexts = [_context(1, 1), _context(2, 2), _context(3, 3)]
        original = list(contexts)
        world = types.SimpleNamespace(
            tick=25,
            get_idle_agents=lambda: [object(), object()],
        )
        with mock.patch.object(
            WorldModelTaskAssigner,
            "select_robots",
            return_value={0: 7, 1: 8},
        ):
            assigner.select_robots(world, contexts)
        self.assertEqual(contexts, original)
        self.assertEqual(assigner.stats["phi_context_reordered_calls"], 1)

    def test_constructor_rejects_confounded_station_injection(self):
        with self.assertRaisesRegex(ValueError, "station injection"):
            PhiContextRankWorldModelTaskAssigner(
                phi_head_checkpoint="head.pt",
                phi_scale_contract="scale.json",
                phi_context_mode="service_ascending",
                station_injection_weight=0.1,
            )

    def test_protocol_uses_fresh_seeds_and_freezes_single_mechanism(self):
        self.assertTrue(set(SEEDS).isdisjoint(LOCKED_REFERENCE_SEEDS))
        self.assertTrue(set(SEEDS).isdisjoint(PHI_VALIDATION_SEEDS))
        hashes = {
            "source_policy_bundle": "a",
            "model_checkpoint": "b",
            "phi_head_checkpoint": "c",
            "phi_scale_contract": "d",
            "config_low": "e",
            "config_mid": "f",
            "config_high": "g",
        }
        protocol = protocol_payload(hashes)
        policy = protocol["policy_contract"]["PhaseCS1PhiContext"]
        self.assertEqual(policy["context_superset_factor"], PHI_CONTEXT_FACTOR)
        self.assertFalse(policy["no_assign"])
        self.assertFalse(policy["hard_gate"])
        self.assertTrue(protocol["forbidden"]["psi_pre_online_use"])


if __name__ == "__main__":
    unittest.main()
