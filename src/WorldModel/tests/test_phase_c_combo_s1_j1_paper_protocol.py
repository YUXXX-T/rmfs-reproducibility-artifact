from __future__ import annotations

import math
import unittest

from WorldModel.core.long_risk_schema import (
    LONG_RISK_SCHEMA_VERSION,
    long_risk_runtime_contract,
)
from WorldModel.evaluation.phase_c_combo_s1_j1_paper_protocol import (
    FRONTIER_COLLAPSE_EFFICIENCY,
    FRONTIER_POINTS,
    FRONTIER_SEEDS,
    FRESH_SEEDS,
    LEGACY_SEEDS,
    LOADS,
    RUN_ARMS,
    TICKS,
    formal_protocol,
    selector_config,
    sha256_file,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    PSI_HEAD_CHECKPOINT,
    PSI_SCALE_CONTRACT,
)
from WorldModel.evaluation.run_phase_c_combo_s1_j1_paper_experiments import (
    LEGACY_POLICY_METADATA,
    _infer_base_tick_interval,
    _metrics_with_derived,
    _order_identity,
    _run_audit,
)
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


class PhaseCComboS1J1PaperProtocolTest(unittest.TestCase):
    def test_main_replacement_prefers_old_pairing_and_has_clean_fallback(self):
        protocol = formal_protocol()
        main = protocol["main_replacement"]
        self.assertEqual(LEGACY_SEEDS, tuple(range(551, 561)))
        self.assertEqual(FRESH_SEEDS, tuple(range(641, 651)))
        self.assertEqual(LOADS, ("low", "mid", "high"))
        self.assertEqual(TICKS, 1500)
        self.assertEqual(main["preferred_seeds"], list(LEGACY_SEEDS))
        self.assertEqual(main["fallback_fresh_seeds"], list(FRESH_SEEDS))
        self.assertEqual(main["preferred_new_simulation_count"], 30)
        self.assertEqual(main["fallback_new_simulation_count"], 120)

    def test_corrected_combo_s1_and_static_j1_are_frozen(self):
        config = selector_config()
        self.assertEqual(config["energy_scoring_mode"], "conversion")
        self.assertEqual(config["energy_drift_signal"], "combo")
        protocol = formal_protocol()
        runtime = protocol["runtime_policy"]
        self.assertEqual(runtime["combo_s1_config"], config)
        self.assertIn("static ascending J", runtime["context_scheduler"])
        self.assertEqual(
            runtime["long_risk_runtime_contract"]["quantile_combo_weights"],
            {"peak_q95": 0.2, "cvar_q90": 0.3, "terminal_q90": 0.5},
        )
        config["energy_drift_signal"] = "event_logit"
        self.assertEqual(selector_config()["energy_drift_signal"], "combo")

    def test_protocol_is_physical_only_and_has_no_station_controller(self):
        station = formal_protocol()["station_admission"]
        self.assertEqual(station["mode"], STATION_ADMISSION_PHYSICAL_ONLY)
        self.assertTrue(station["physical_occupancy_cap_enforced"])
        self.assertFalse(station["in_transit_committed_cap_enforced"])
        self.assertFalse(station["station_queue"])
        self.assertFalse(station["eta_controller"])

    def test_frontier_scope_and_threshold_semantics_are_frozen(self):
        frontier = formal_protocol()["paired_arrival_frontier"]
        self.assertEqual(FRONTIER_SEEDS, LEGACY_SEEDS)
        self.assertEqual(
            FRONTIER_POINTS,
            (
                ("m080", 0.8),
                ("m100", 1.0),
                ("m120", 1.2),
                ("m140", 1.4),
                ("m160", 1.6),
            ),
        )
        self.assertEqual(frontier["new_simulation_count"], 200)
        self.assertEqual(
            frontier["collapse_efficiency_ratio"],
            FRONTIER_COLLAPSE_EFFICIENCY,
        )
        self.assertIn("empirical best", frontier["collapse_reference"])
        self.assertIn(
            "contiguous", frontier["critical_multiplier_definition"]
        )
        self.assertEqual(tuple(frontier["arms"]), RUN_ARMS)

    def test_order_identity_ignores_only_arrival_tick(self):
        left = {
            "tick": 10,
            "order_id": 7,
            "station_id": 2,
            "sku_demands": {"A": 1},
        }
        right = {**left, "tick": 5}
        different = {**right, "station_id": 3}
        self.assertEqual(_order_identity(left), _order_identity(right))
        self.assertNotEqual(_order_identity(left), _order_identity(different))

    def test_scaled_ticks_admit_one_consistent_base_stream(self):
        base_tick = 1000
        observed = {
            tag: math.floor(base_tick / multiplier)
            for tag, multiplier in FRONTIER_POINTS
        }
        self.assertEqual(
            _infer_base_tick_interval(
                observed, target_ticks=TICKS, base_ticks=2400
            ),
            (base_tick, base_tick),
        )

        later_base_tick = 1700
        later_observed = {
            tag: math.floor(later_base_tick / multiplier)
            for tag, multiplier in FRONTIER_POINTS
            if later_base_tick < TICKS * multiplier
        }
        interval = _infer_base_tick_interval(
            later_observed, target_ticks=TICKS, base_ticks=2400
        )
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertLessEqual(interval[0], later_base_tick)
        self.assertGreaterEqual(interval[1], later_base_tick)

        inconsistent = dict(observed)
        inconsistent["m140"] += 100
        self.assertIsNone(
            _infer_base_tick_interval(
                inconsistent, target_ticks=TICKS, base_ticks=2400
            )
        )

    def test_combo_runtime_audit_rejects_event_logit(self):
        manifest = {"manifest_sha256": "abc", "total_orders": 100}
        station_audit = {
            "passed": True,
            "mode": STATION_ADMISSION_PHYSICAL_ONLY,
            "physical_capacity_violation_count": 0,
            "committed_capacity_contract": (
                "not enforced for in-transit commitments"
            ),
        }
        metrics = {
            "order_arrival_replayed": True,
            "order_arrival_manifest_sha256": "abc",
            "order_arrival_count": 100,
            "model_assign_calls": 10,
            "fallback_greedy_calls": 0,
            "native_no_assign_enabled": False,
            "long_risk_schema_version": LONG_RISK_SCHEMA_VERSION,
            "long_risk_runtime_contract": long_risk_runtime_contract(),
            "energy_scoring_mode": "conversion",
            "energy_drift_signal": "combo",
            "energy_conv_contexts": 20,
            "psi_dispatch_s1_within_context": True,
            "psi_dispatch_mode": "j_ascending",
            "psi_dispatch_head_loaded": True,
            "psi_dispatch_head_checkpoint_sha256": sha256_file(
                PSI_HEAD_CHECKPOINT
            ),
            "psi_dispatch_scale_contract_sha256": sha256_file(
                PSI_SCALE_CONTRACT
            ),
            "psi_dispatch_eval_calls": 5,
            "psi_dispatch_contexts_seen": 20,
            "psi_dispatch_robot_scorer": (
                "WorldModelTaskAssigner.select_robots_unmodified"
            ),
            "psi_dispatch_robot_scorer_variant": "s1_within_context",
        }
        accepted = _run_audit(
            "combo_j1",
            metrics,
            manifest,
            station_audit,
            replayed=True,
        )
        self.assertTrue(accepted["passed"])

        wrong = dict(metrics)
        wrong["energy_drift_signal"] = "event_logit"
        rejected = _run_audit(
            "combo_j1",
            wrong,
            manifest,
            station_audit,
            replayed=True,
        )
        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["checks"]["combo_signal"])

    def test_completion_ratio_and_legacy_metadata_contract(self):
        metrics = _metrics_with_derived(
            {"completed_orders": 80}, {"total_orders": 100}
        )
        self.assertEqual(metrics["completion_ratio"], 0.8)
        self.assertEqual(
            LEGACY_POLICY_METADATA["legacy_event_j1"],
            ("s1_j1", "world_model", "s1", "j1"),
        )


if __name__ == "__main__":
    unittest.main()
