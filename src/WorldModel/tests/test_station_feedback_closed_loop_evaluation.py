from __future__ import annotations

import unittest
from types import SimpleNamespace

from WorldModel.core.station_context_defer import (
    STATION_CONTEXT_DEFER_PIPELINE_PHI_SCHEMA_VERSION,
    STATION_CONTEXT_DEFER_RISK_PIPELINE_PHI_V2,
)
from WorldModel.core.station_feedback_control import (
    STATION_FEEDBACK_MODE_ACTIVE,
    STATION_FEEDBACK_MODE_ACTIVE_V2,
    STATION_FEEDBACK_MODE_ACTIVE_V3,
    STATION_FEEDBACK_MODE_OFF,
    STATION_FEEDBACK_MODE_SHADOW,
    STATION_FEEDBACK_MODE_SHADOW_V2,
    STATION_FEEDBACK_MODE_SHADOW_V3,
    StationFeedbackConfig,
    station_feedback_mode_is_active,
    station_feedback_mode_uses_early_brake,
)
from WorldModel.evaluation.analyze_phase_c_station_feedback_early_brake import (
    _paired_integrity,
)
from WorldModel.evaluation.analyze_phase_c_station_feedback_release_aware import (
    _shadow_equivalence as _release_shadow_equivalence,
)
from WorldModel.evaluation.analyze_phase_c_station_feedback_closed_loop import (
    BEHAVIOR_EQUIVALENCE_KEYS,
    _parse_points,
    _shadow_equivalence,
)
from WorldModel.evaluation.run_phase_c_station_feedback_closed_loop import (
    ARM_KEYS,
    _audit,
)
from WorldModel.evaluation.run_phase_c_station_feedback_early_brake import (
    ARM_KEYS as EARLY_ARM_KEYS,
    _feedback_config as _early_feedback_config,
)
from WorldModel.evaluation.run_phase_c_station_feedback_release_aware import (
    ARM_KEYS as RELEASE_ARM_KEYS,
    _feedback_config as _release_feedback_config,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.station_feedback_defer_assigner import (
    STATION_FEEDBACK_ASSIGNER_SCHEMA_VERSION,
    STATION_FEEDBACK_BATCH_FILTER_SCHEMA_VERSION,
)
from WorldState.station_state import STATION_ADMISSION_DYNAMIC_ETA_V1


def _metrics(
    mode: str,
    config: StationFeedbackConfig | None = None,
) -> dict:
    enabled = mode != STATION_FEEDBACK_MODE_OFF
    active = station_feedback_mode_is_active(mode)
    if config is None:
        config = StationFeedbackConfig.for_service_ticks(
            5,
            early_brake_enabled=station_feedback_mode_uses_early_brake(mode),
        )
    return {
        "order_arrival_replayed": True,
        "order_arrival_manifest_sha256": "manifest",
        "order_arrival_count": 12,
        "ticks": 1500,
        "model_assign_calls": 1,
        "energy_conv_contexts": 1,
        "fallback_greedy_calls": 0,
        "dynamic_probe_batches": 1,
        "dynamic_probe_choices_reindexed": True,
        "station_context_defer_enabled": True,
        "station_context_defer_schema_version": (
            STATION_CONTEXT_DEFER_PIPELINE_PHI_SCHEMA_VERSION
        ),
        "station_context_defer_risk_mode": (
            STATION_CONTEXT_DEFER_RISK_PIPELINE_PHI_V2
        ),
        "station_context_defer_ready_contention_in_station_risk": False,
        "station_context_defer_liveness_bound_violations": 0,
        "station_context_defer_physical_safety_layer": (
            "external_station_admission_contract"
        ),
        "station_feedback_admission_contract_owned_by_engine": True,
        "psi_dispatch_schema_version": STATION_FEEDBACK_ASSIGNER_SCHEMA_VERSION,
        "station_feedback_schema_version": config.schema_version,
        "station_feedback_batch_filter_schema_version": (
            STATION_FEEDBACK_BATCH_FILTER_SCHEMA_VERSION
        ),
        "station_feedback_batch_filter_enabled": enabled,
        "station_feedback_batch_filter_active": active,
        "station_feedback_batch_filter_before_dynamic_j": active,
        "station_feedback_batch_filter_preserves_full_batch_debt_features": True,
        "station_feedback_batch_filter_station_tick_deduplicated": True,
        "station_feedback_mode": mode,
        "station_feedback_enabled": enabled,
        "station_feedback_config": (
            config.as_dict()
        ),
        "station_feedback_early_brake_enabled": bool(
            config.early_brake_enabled
        ),
        "station_feedback_decision_override_enabled": active,
        "station_feedback_station_local": True,
        "station_feedback_order_generator_modified": False,
        "station_feedback_task_lifecycle_modified": False,
        "station_feedback_station_admission_modified": False,
        "station_feedback_path_planning_modified": False,
        "station_feedback_s1_modified": False,
        "station_feedback_world_model_modified": False,
        "station_feedback_new_trainable_parameters": 0,
        "station_feedback_context_evaluations": int(enabled),
        "station_feedback_forced_defers": int(active),
        "station_feedback_contexts_materialized_in_defer_state": 0,
        "station_feedback_liveness_clock_paused_while_deferred": active,
    }


class StationFeedbackClosedLoopEvaluationTests(unittest.TestCase):
    def test_arm_keys_are_isolated(self):
        self.assertEqual(len(set(ARM_KEYS.values())), 3)

    def test_early_brake_arm_keys_are_isolated(self):
        self.assertEqual(len(EARLY_ARM_KEYS), 4)
        self.assertEqual(len(set(EARLY_ARM_KEYS.values())), 4)

    def test_release_aware_arm_keys_are_isolated(self):
        self.assertEqual(len(RELEASE_ARM_KEYS), 4)
        self.assertEqual(len(set(RELEASE_ARM_KEYS.values())), 4)

    def test_early_brake_profile_is_explicitly_mode_gated(self):
        self.assertFalse(
            _early_feedback_config(5, STATION_FEEDBACK_MODE_OFF)
            .early_brake_enabled
        )
        self.assertFalse(
            _early_feedback_config(5, STATION_FEEDBACK_MODE_ACTIVE)
            .early_brake_enabled
        )
        self.assertTrue(
            _early_feedback_config(5, STATION_FEEDBACK_MODE_SHADOW_V2)
            .early_brake_enabled
        )
        self.assertTrue(
            _early_feedback_config(5, STATION_FEEDBACK_MODE_ACTIVE_V2)
            .early_brake_enabled
        )

    def test_release_aware_profile_is_explicitly_mode_gated(self):
        off = _release_feedback_config(5, STATION_FEEDBACK_MODE_OFF)
        v2 = _release_feedback_config(5, STATION_FEEDBACK_MODE_ACTIVE_V2)
        shadow = _release_feedback_config(5, STATION_FEEDBACK_MODE_SHADOW_V3)
        active = _release_feedback_config(5, STATION_FEEDBACK_MODE_ACTIVE_V3)

        self.assertFalse(off.early_brake_enabled)
        self.assertNotIn("release_signal", off.as_dict())
        self.assertTrue(v2.early_brake_enabled)
        self.assertNotIn("release_signal", v2.as_dict())
        self.assertTrue(shadow.early_brake_enabled)
        self.assertTrue(active.early_brake_enabled)
        self.assertEqual(shadow.release_signal, active.release_signal)
        self.assertEqual(shadow.early_brake_rule, active.early_brake_rule)

    def test_audit_accepts_each_explicit_mode_contract(self):
        config = StationFeedbackConfig.for_service_ticks(5)
        manifest = {"manifest_sha256": "manifest", "total_orders": 12}
        reference = {"manifest_sha256": "manifest"}
        station_audit = {
            "mode": STATION_ADMISSION_DYNAMIC_ETA_V1,
            "passed": True,
            "physical_capacity_violation_count": 0,
            "dynamic_hard_limit_violation_count": 0,
        }
        for mode in (
            STATION_FEEDBACK_MODE_OFF,
            STATION_FEEDBACK_MODE_SHADOW,
            STATION_FEEDBACK_MODE_ACTIVE,
        ):
            with self.subTest(mode=mode):
                audit = _audit(
                    args=SimpleNamespace(
                        station_feedback_mode=mode,
                        ticks=1500,
                    ),
                    metrics=_metrics(mode),
                    manifest=manifest,
                    station_audit=station_audit,
                    feedback_config=config,
                    reference=reference,
                )
                self.assertTrue(audit["passed"], audit["checks"])

    def test_audit_accepts_v2_shadow_and_active_profiles(self):
        manifest = {"manifest_sha256": "manifest", "total_orders": 12}
        reference = {"manifest_sha256": "manifest"}
        station_audit = {
            "mode": STATION_ADMISSION_DYNAMIC_ETA_V1,
            "passed": True,
            "physical_capacity_violation_count": 0,
            "dynamic_hard_limit_violation_count": 0,
        }
        for mode in (
            STATION_FEEDBACK_MODE_SHADOW_V2,
            STATION_FEEDBACK_MODE_ACTIVE_V2,
        ):
            with self.subTest(mode=mode):
                config = _early_feedback_config(5, mode)
                audit = _audit(
                    args=SimpleNamespace(
                        station_feedback_mode=mode,
                        ticks=1500,
                        max_committed_multiplier=2.0,
                    ),
                    metrics=_metrics(mode),
                    manifest=manifest,
                    station_audit=station_audit,
                    feedback_config=config,
                    reference=reference,
                )
                self.assertTrue(audit["passed"], audit["checks"])

    def test_audit_accepts_v3_shadow_and_active_profiles(self):
        manifest = {"manifest_sha256": "manifest", "total_orders": 12}
        reference = {"manifest_sha256": "manifest"}
        station_audit = {
            "mode": STATION_ADMISSION_DYNAMIC_ETA_V1,
            "passed": True,
            "physical_capacity_violation_count": 0,
            "dynamic_hard_limit_violation_count": 0,
        }
        for mode in (
            STATION_FEEDBACK_MODE_SHADOW_V3,
            STATION_FEEDBACK_MODE_ACTIVE_V3,
        ):
            with self.subTest(mode=mode):
                config = _release_feedback_config(5, mode)
                audit = _audit(
                    args=SimpleNamespace(
                        station_feedback_mode=mode,
                        ticks=1500,
                        max_committed_multiplier=2.0,
                    ),
                    metrics=_metrics(mode, config),
                    manifest=manifest,
                    station_audit=station_audit,
                    feedback_config=config,
                    reference=reference,
                )
                self.assertTrue(audit["passed"], audit["checks"])

    def test_active_audit_rejects_materialized_braked_context(self):
        metrics = _metrics(STATION_FEEDBACK_MODE_ACTIVE)
        metrics["station_feedback_contexts_materialized_in_defer_state"] = 1
        audit = _audit(
            args=SimpleNamespace(
                station_feedback_mode=STATION_FEEDBACK_MODE_ACTIVE,
                ticks=1500,
            ),
            metrics=metrics,
            manifest={"manifest_sha256": "manifest", "total_orders": 12},
            station_audit={
                "mode": STATION_ADMISSION_DYNAMIC_ETA_V1,
                "passed": True,
                "physical_capacity_violation_count": 0,
                "dynamic_hard_limit_violation_count": 0,
            },
            feedback_config=StationFeedbackConfig.for_service_ticks(5),
            reference={"manifest_sha256": "manifest"},
        )
        self.assertFalse(audit["passed"])
        self.assertFalse(
            audit["checks"]["active_never_materializes_braked_context"]
        )

    def test_shadow_equivalence_detects_behavior_change(self):
        base = {
            "manifest": {"content_sha256": "same"},
            "metrics": {key: 1 for key in (
                "completed_orders",
                "completed_tasks",
                "avg_task_duration",
                "avg_excess_delay",
                "open_order_count",
                "pending_order_count",
                "deadlock_ratio_mean",
                "deadlock_ratio_max",
                "stall_ratio_mean",
                "stall_ratio_max",
                "station_capacity_rejections",
                "station_over_capacity_grants",
                "station_committed_over_capacity_tick_count",
                "station_ticks_with_any_committed_over_capacity",
                "station_context_defer_decisions",
                "station_context_defer_selected",
                "station_context_defer_debt_updates",
                "station_context_defer_liveness_bound_violations",
            )},
        }
        shadow = {
            "manifest": dict(base["manifest"]),
            "metrics": dict(base["metrics"]),
        }
        payloads = {
            ("m100", STATION_FEEDBACK_MODE_OFF, 551): base,
            ("m100", STATION_FEEDBACK_MODE_SHADOW, 551): shadow,
        }
        self.assertTrue(
            _shadow_equivalence(payloads, [(1.0, "m100")], [551])["passed"]
        )
        shadow["metrics"]["completed_orders"] = 2
        self.assertFalse(
            _shadow_equivalence(payloads, [(1.0, "m100")], [551])["passed"]
        )

    def test_early_integrity_uses_off_shadow_v2_as_noninterference_pair(self):
        def payload():
            return {
                "manifest": {"content_sha256": "same"},
                "metrics": {
                    key: 1 for key in BEHAVIOR_EQUIVALENCE_KEYS
                },
            }

        payloads = {
            ("m100", mode, 551): payload()
            for mode in EARLY_ARM_KEYS
        }
        passed = _paired_integrity(
            payloads, [(1.0, "m100")], [551]
        )
        self.assertTrue(passed["manifest_pairing_passed"])
        self.assertTrue(passed["off_shadow_v2_equivalence_passed"])

        payloads[("m100", STATION_FEEDBACK_MODE_SHADOW_V2, 551)][
            "metrics"
        ]["completed_orders"] = 2
        failed = _paired_integrity(
            payloads, [(1.0, "m100")], [551]
        )
        self.assertFalse(failed["off_shadow_v2_equivalence_passed"])

    def test_release_integrity_uses_off_shadow_v3_pair(self):
        def payload():
            return {
                "manifest": {"content_sha256": "same"},
                "metrics": {
                    key: 1 for key in BEHAVIOR_EQUIVALENCE_KEYS
                },
            }

        payloads = {
            ("m100", mode, 551): payload()
            for mode in RELEASE_ARM_KEYS
        }
        passed = _release_shadow_equivalence(
            payloads, [(1.0, "m100")], [551]
        )
        self.assertTrue(passed["passed"])
        payloads[("m100", STATION_FEEDBACK_MODE_SHADOW_V3, 551)][
            "metrics"
        ]["completed_orders"] = 2
        failed = _release_shadow_equivalence(
            payloads, [(1.0, "m100")], [551]
        )
        self.assertFalse(failed["passed"])

    def test_point_parser_preserves_multiplier_and_tag(self):
        self.assertEqual(
            _parse_points(["0.8:m080", "1.2:m120"]),
            [(0.8, "m080"), (1.2, "m120")],
        )


if __name__ == "__main__":
    unittest.main()
