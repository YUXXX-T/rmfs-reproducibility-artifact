"""Run one paired station-feedback closed-loop capacity-frontier arm.

This development runner isolates exactly one new factor around the frozen
Dynamic-J + pipeline+phi V2 + S1 stack: station feedback is either ``off``,
``shadow_v1``, or ``active_v1``.  The V2 runner applies active feedback as a
station-level batch filter before Dynamic-J, while shadow mode audits the same
filter without changing decisions.  Station admission remains ETA V3 and is
owned by the engine.  Every run must replay an existing capacity-frontier
manifest and match the corresponding historical reference result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from Policies.TaskAssigner.WorldModelTaskAssigner.station_feedback_defer_assigner import (
    STATION_FEEDBACK_ASSIGNER_SCHEMA_VERSION,
    STATION_FEEDBACK_BATCH_FILTER_SCHEMA_VERSION,
    StationFeedbackClosedLoopPipelinePhiDynamicPsiAssigner,
)
from WorldModel.core.station_context_defer import (
    STATION_CONTEXT_DEFER_PIPELINE_PHI_SCHEMA_VERSION,
    STATION_CONTEXT_DEFER_RISK_PIPELINE_PHI_V2,
)
from WorldModel.core.station_feedback_control import (
    STATION_FEEDBACK_MODE_ACTIVE,
    STATION_FEEDBACK_MODE_OFF,
    STATION_FEEDBACK_MODE_SHADOW,
    STATION_FEEDBACK_MODES,
    STATION_FEEDBACK_RELEASE_SIGNAL_SERVICE_DUE,
    StationFeedbackConfig,
    station_feedback_mode_is_active,
    station_feedback_mode_is_shadow,
    station_feedback_mode_uses_early_brake,
    station_feedback_mode_uses_release_aware_brake,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    LOADS,
    S1_CONFIG,
    TOP_M,
    canonical_sha256,
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_psi_dynamic_admission import (
    _atomic_json,
    _read_json,
)
from WorldModel.evaluation.run_phase_c_station_admission_restoration import (
    DEFAULT_BUNDLE,
    FORMAL_SEEDS,
    FORMAL_TICKS,
    StationAdmissionRestorationAuditProbe,
    _dynamic_config,
    _engine_contract,
    _load_run_inputs,
    _manifest_path,
    _max_mapping,
    _validate_manifest,
)
from WorldState.station_state import STATION_ADMISSION_DYNAMIC_ETA_V1


LEGACY_SCHEMA_VERSION = "phase_c_station_feedback_closed_loop_arm_v1"
SCHEMA_VERSION = "phase_c_station_feedback_closed_loop_arm_v2"
SUPPORTED_SCHEMA_VERSIONS = (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION)
EXPERIMENT_CONTRACT_VERSION = (
    "phase_c_station_feedback_closed_loop_capacity_frontier_v2"
)

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_SOURCE_ROOT = (
    BASE_ROOT / "station_admission_capacity_frontier_source_high_551_560_v1"
    / "m100"
)
DEFAULT_REFERENCE_ROOT = (
    BASE_ROOT / "station_admission_capacity_frontier_high_551_560_v1"
    / "m100"
)
DEFAULT_OUTPUT_ROOT = (
    BASE_ROOT / "station_feedback_closed_loop_capacity_frontier_high_551_560_v2"
    / "m100"
)
DEFAULT_REFERENCE_ARM = "s1_j1_eta_v3"

ARM_KEYS = {
    STATION_FEEDBACK_MODE_OFF: (
        "s1_dynamic_j_pipeline_phi_eta_v3_feedback_off"
    ),
    STATION_FEEDBACK_MODE_SHADOW: (
        "s1_dynamic_j_pipeline_phi_eta_v3_feedback_shadow_v1"
    ),
    STATION_FEEDBACK_MODE_ACTIVE: (
        "s1_dynamic_j_pipeline_phi_eta_v3_feedback_active_v1"
    ),
}


def _output_path(
    root: Path,
    feedback_mode: str,
    load: str,
    seed: int,
    arm_keys: Mapping[str, str] = ARM_KEYS,
) -> Path:
    return (
        root / "per_arm" / arm_keys[feedback_mode]
        / f"{load}_seed{seed}.json"
    )


def _reference_path(
    root: Path,
    arm: str,
    load: str,
    seed: int,
) -> Path:
    return root / "per_arm" / arm / f"{load}_seed{seed}.json"


def _runtime_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "station_feedback_control": Path(
            "WorldModel/core/station_feedback_control.py"
        ),
        "station_feedback_assigner": Path(
            "Policies/TaskAssigner/WorldModelTaskAssigner/"
            "station_feedback_defer_assigner.py"
        ),
        "station_context_defer": Path(
            "WorldModel/core/station_context_defer.py"
        ),
        "station_context_defer_assigner": Path(
            "Policies/TaskAssigner/WorldModelTaskAssigner/"
            "station_context_defer_assigner.py"
        ),
        "dynamic_j_assigner": Path(
            "Policies/TaskAssigner/WorldModelTaskAssigner/"
            "psi_dispatch_dynamic_probe_assigner.py"
        ),
        "world_model_assigner": Path(
            "Policies/TaskAssigner/WorldModelTaskAssigner/"
            "world_model_task_assigner.py"
        ),
        "station_state": Path("WorldState/station_state.py"),
        "simulation_engine": Path("Engine/simulation_engine.py"),
        "evaluate_online": Path(
            "WorldModel/evaluation/evaluate_online_v6.py"
        ),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _service_ticks(config_path: Path) -> int:
    payload = _read_json(config_path)
    simulation = payload.get("simulation") or {}
    value = int(simulation.get("station_process_duration", 0))
    if value <= 0:
        raise ValueError(
            f"invalid station_process_duration in {config_path}: {value}"
        )
    return value


def _feedback_config(
    service_ticks: int,
    feedback_mode: str | None = None,
) -> StationFeedbackConfig:
    # V1 deliberately derives all thresholds from the configured station
    # service duration.  There are no outcome-tuned CLI threshold overrides.
    del feedback_mode
    return StationFeedbackConfig.for_service_ticks(service_ticks)


def _policy_contract(
    *,
    feedback_mode: str,
    feedback_config: StationFeedbackConfig,
    feedback_trace_max_records: int,
    model_checkpoint: Path,
    psi_head_checkpoint: Path,
    psi_scale_contract: Path,
    runtime_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if station_feedback_mode_is_active(feedback_mode):
        selection_order = (
            "station feedback batch filter -> Dynamic-J -> "
            "pipeline+phi V2 -> S1 -> ETA V3 admission"
        )
    elif station_feedback_mode_is_shadow(feedback_mode):
        selection_order = (
            "station feedback batch audit -> Dynamic-J -> "
            "pipeline+phi V2 -> station feedback audit -> S1 -> "
            "ETA V3 admission"
        )
    else:
        selection_order = (
            "Dynamic-J -> pipeline+phi V2 -> S1 -> ETA V3 admission"
        )
    contract = {
        "policy": (
            "StationFeedbackClosedLoopPipelinePhiDynamicPsiAssigner"
        ),
        "checkpoint": model_checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(model_checkpoint),
        "psi_head_checkpoint": psi_head_checkpoint.as_posix(),
        "psi_head_checkpoint_sha256": sha256_file(psi_head_checkpoint),
        "psi_scale_contract": psi_scale_contract.as_posix(),
        "psi_scale_contract_sha256": sha256_file(psi_scale_contract),
        "top_m": int(TOP_M),
        "s1_config": dict(S1_CONFIG),
        "energy_conv_random_flip_seed": 0,
        "context_scheduler": "dynamic_j_interleaved",
        "station_context_defer": "pipeline_phi_v2",
        "station_feedback_mode": str(feedback_mode),
        "station_feedback_config": feedback_config.as_dict(),
        "station_feedback_batch_filter_schema_version": (
            STATION_FEEDBACK_BATCH_FILTER_SCHEMA_VERSION
        ),
        "station_feedback_batch_filter_active": (
            station_feedback_mode_is_active(feedback_mode)
        ),
        "station_feedback_trace_max_records": int(
            feedback_trace_max_records
        ),
        "station_admission": "engine_owned_eta_v3",
        "selection_order": selection_order,
        "runtime_code_sha256": dict(runtime_hashes),
    }
    return {**contract, "fingerprint_sha256": canonical_sha256(contract)}


def _reference_contract(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    path = _reference_path(
        args.reference_root,
        args.reference_arm,
        args.load,
        int(args.seed),
    )
    if not path.is_file():
        raise FileNotFoundError(
            "missing paired capacity-frontier reference result: "
            f"{path}"
        )
    payload = _read_json(path)
    reference_manifest = payload.get("manifest") or {}
    expected_sha = str(manifest.get("manifest_sha256"))
    expected_count = int(manifest.get("total_orders", -1))
    if (
        str(reference_manifest.get("content_sha256")) != expected_sha
        or int(reference_manifest.get("total_orders", -2)) != expected_count
    ):
        raise RuntimeError(
            f"reference manifest differs from source manifest: {path}"
        )
    if not bool((payload.get("audit") or {}).get("passed")):
        raise RuntimeError(f"reference result audit did not pass: {path}")
    metrics = payload.get("metrics") or {}
    return {
        "path": path,
        "file_sha256": sha256_file(path),
        "arm": str(args.reference_arm),
        "manifest_sha256": expected_sha,
        "total_orders": expected_count,
        "metrics": {
            key: metrics.get(key)
            for key in (
                "completed_orders",
                "open_order_count",
                "pending_order_count",
                "deadlock_ratio_mean",
                "stall_ratio_mean",
                "station_capacity_rejections",
            )
        },
    }


def _audit(
    *,
    args: argparse.Namespace,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    station_audit: Mapping[str, Any],
    feedback_config: StationFeedbackConfig,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    mode = str(args.station_feedback_mode)
    feedback_enabled = mode != STATION_FEEDBACK_MODE_OFF
    active = station_feedback_mode_is_active(mode)
    early_brake = station_feedback_mode_uses_early_brake(mode)
    release_aware = station_feedback_mode_uses_release_aware_brake(mode)
    checks = {
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "manifest_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_count_matches": int(
            metrics.get("order_arrival_count", -1)
        ) == int(manifest.get("total_orders", -2)),
        "reference_manifest_matches": (
            reference.get("manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "ticks_match": int(metrics.get("ticks", -1)) == int(args.ticks),
        "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
        "s1_used": int(metrics.get("energy_conv_contexts", 0)) > 0,
        "no_greedy_fallback": int(
            metrics.get("fallback_greedy_calls", 0)
        ) == 0,
        "dynamic_j_batches_seen": int(
            metrics.get("dynamic_probe_batches", 0)
        ) > 0,
        "dynamic_j_is_interleaved": bool(
            metrics.get("dynamic_probe_choices_reindexed", False)
        ),
        "pipeline_phi_enabled": bool(
            metrics.get("station_context_defer_enabled", False)
        ),
        "pipeline_phi_schema": (
            metrics.get("station_context_defer_schema_version")
            == STATION_CONTEXT_DEFER_PIPELINE_PHI_SCHEMA_VERSION
        ),
        "pipeline_phi_risk_mode": (
            metrics.get("station_context_defer_risk_mode")
            == STATION_CONTEXT_DEFER_RISK_PIPELINE_PHI_V2
        ),
        "ready_contention_is_diagnostic_only": not bool(
            metrics.get(
                "station_context_defer_ready_contention_in_station_risk",
                True,
            )
        ),
        "pipeline_phi_liveness_holds": int(
            metrics.get(
                "station_context_defer_liveness_bound_violations", -1
            )
        ) == 0,
        "admission_is_external_to_assigner": (
            metrics.get("station_context_defer_physical_safety_layer")
            == "external_station_admission_contract"
            and bool(
                metrics.get(
                    "station_feedback_admission_contract_owned_by_engine",
                    False,
                )
            )
        ),
        "feedback_assigner_schema": (
            metrics.get("psi_dispatch_schema_version")
            == STATION_FEEDBACK_ASSIGNER_SCHEMA_VERSION
        ),
        "feedback_controller_schema": (
            metrics.get("station_feedback_schema_version")
            == feedback_config.schema_version
        ),
        "feedback_batch_filter_schema": (
            metrics.get("station_feedback_batch_filter_schema_version")
            == STATION_FEEDBACK_BATCH_FILTER_SCHEMA_VERSION
        ),
        "feedback_batch_filter_enabled_flag": bool(
            metrics.get("station_feedback_batch_filter_enabled", False)
        ) == feedback_enabled,
        "feedback_batch_filter_active_flag": bool(
            metrics.get("station_feedback_batch_filter_active", False)
        ) == active,
        "feedback_batch_filter_before_dynamic_j": bool(
            metrics.get(
                "station_feedback_batch_filter_before_dynamic_j", False
            )
        ) == active,
        "feedback_batch_filter_preserves_full_batch_debt": bool(
            metrics.get(
                "station_feedback_batch_filter_preserves_full_batch_debt_features",
                False,
            )
        ),
        "feedback_batch_filter_station_tick_deduplicated": bool(
            metrics.get(
                "station_feedback_batch_filter_station_tick_deduplicated",
                False,
            )
        ),
        "feedback_mode": metrics.get("station_feedback_mode") == mode,
        "feedback_enabled_flag": bool(
            metrics.get("station_feedback_enabled", False)
        ) == feedback_enabled,
        "feedback_config": (
            metrics.get("station_feedback_config")
            == feedback_config.as_dict()
        ),
        "feedback_early_brake_profile": (
            bool(feedback_config.early_brake_enabled) == early_brake
            and bool(
                metrics.get("station_feedback_early_brake_enabled", False)
            ) == early_brake
        ),
        "feedback_release_aware_profile": (
            (
                feedback_config.release_signal
                == STATION_FEEDBACK_RELEASE_SIGNAL_SERVICE_DUE
            ) == release_aware
        ),
        "feedback_early_brake_is_below_eta_hard_cap": (
            float(feedback_config.early_brake_committed_multiplier)
            < float(args.max_committed_multiplier)
            if early_brake else True
        ),
        "feedback_override_flag": bool(
            metrics.get("station_feedback_decision_override_enabled", False)
        ) == active,
        "feedback_station_local": bool(
            metrics.get("station_feedback_station_local", False)
        ),
        "feedback_does_not_modify_order_generator": not bool(
            metrics.get("station_feedback_order_generator_modified", True)
        ),
        "feedback_does_not_modify_task_lifecycle": not bool(
            metrics.get("station_feedback_task_lifecycle_modified", True)
        ),
        "feedback_does_not_modify_admission": not bool(
            metrics.get("station_feedback_station_admission_modified", True)
        ),
        "feedback_does_not_modify_path_planning": not bool(
            metrics.get("station_feedback_path_planning_modified", True)
        ),
        "feedback_does_not_modify_s1": not bool(
            metrics.get("station_feedback_s1_modified", True)
        ),
        "feedback_does_not_modify_world_model": not bool(
            metrics.get("station_feedback_world_model_modified", True)
        ),
        "feedback_has_no_trainable_parameters": int(
            metrics.get("station_feedback_new_trainable_parameters", -1)
        ) == 0,
        "feedback_evaluation_activity": (
            int(metrics.get("station_feedback_context_evaluations", 0)) > 0
            if feedback_enabled
            else int(metrics.get("station_feedback_context_evaluations", 0))
            == 0
        ),
        "off_or_shadow_never_forces_defer": (
            int(metrics.get("station_feedback_forced_defers", 0)) == 0
            if not active else True
        ),
        "active_never_materializes_braked_context": (
            int(
                metrics.get(
                    "station_feedback_contexts_materialized_in_defer_state",
                    -1,
                )
            ) == 0
            if active else True
        ),
        "feedback_liveness_pause_flag": bool(
            metrics.get(
                "station_feedback_liveness_clock_paused_while_deferred",
                False,
            )
        ) == active,
        "eta_v3_admission_mode": (
            station_audit.get("mode") == STATION_ADMISSION_DYNAMIC_ETA_V1
        ),
        "eta_v3_admission_audit": bool(station_audit.get("passed")),
        "physical_capacity_never_exceeded": int(
            station_audit.get("physical_capacity_violation_count", -1)
        ) == 0,
        "dynamic_hard_limit_never_exceeded": int(
            station_audit.get("dynamic_hard_limit_violation_count", -1)
        ) == 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _resume_compatible(
    path: Path,
    *,
    args: argparse.Namespace,
    bundle_sha: str,
    runtime_hashes: Mapping[str, str],
    policy_fingerprint: str,
    manifest_sha: str,
    dynamic_config: Mapping[str, Any],
    reference_file_sha: str,
    arm_keys: Mapping[str, str] = ARM_KEYS,
    schema_version: str = SCHEMA_VERSION,
    contract_version: str = EXPERIMENT_CONTRACT_VERSION,
) -> bool:
    if not path.is_file():
        return False
    payload = _read_json(path)
    meta = payload.get("meta") or {}
    reference = payload.get("paired_reference") or {}
    checks = {
        "schema": payload.get("schema_version") == schema_version,
        "contract": (
            meta.get("experiment_contract_version")
            == contract_version
        ),
        "arm": meta.get("arm_key") == arm_keys[args.station_feedback_mode],
        "feedback_mode": (
            meta.get("station_feedback_mode")
            == args.station_feedback_mode
        ),
        "load": meta.get("load") == args.load,
        "seed": int(meta.get("seed", -1)) == int(args.seed),
        "ticks": int(meta.get("ticks", -1)) == int(args.ticks),
        "admission": (
            meta.get("station_admission")
            == STATION_ADMISSION_DYNAMIC_ETA_V1
        ),
        "bundle": meta.get("frozen_bundle_sha256") == bundle_sha,
        "runtime": meta.get("runtime_code_sha256") == dict(runtime_hashes),
        "policy": (
            (meta.get("policy_contract") or {}).get("fingerprint_sha256")
            == policy_fingerprint
        ),
        "manifest": (
            (payload.get("manifest") or {}).get("content_sha256")
            == manifest_sha
        ),
        "dynamic_config": (
            meta.get("dynamic_admission_config") == dict(dynamic_config)
        ),
        "reference": reference.get("file_sha256") == reference_file_sha,
        "audit": bool((payload.get("audit") or {}).get("passed")),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"cannot resume incompatible output {path}: {failed}")
    print(
        f"[resume] {args.station_feedback_mode} {args.load} "
        f"seed={args.seed}: {path}"
    )
    return True


def _run(
    args: argparse.Namespace,
    *,
    arm_keys: Mapping[str, str] = ARM_KEYS,
    schema_version: str = SCHEMA_VERSION,
    contract_version: str = EXPERIMENT_CONTRACT_VERSION,
    feedback_config_factory=_feedback_config,
) -> None:
    (
        _bundle,
        source_protocol,
        model_checkpoint,
        psi_head_checkpoint,
        psi_scale_contract,
        config_path,
    ) = _load_run_inputs(args)

    manifest_path = _manifest_path(args.source_root, args.load, int(args.seed))
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "closed-loop runs never generate orders; missing manifest: "
            f"{manifest_path}"
        )
    manifest = _validate_manifest(manifest_path)
    reference = _reference_contract(args, manifest)
    service_ticks = _service_ticks(config_path)
    feedback_config = feedback_config_factory(
        service_ticks, args.station_feedback_mode
    )
    dynamic_config = _dynamic_config(args)
    runtime_hashes = _runtime_hashes()
    policy_contract = _policy_contract(
        feedback_mode=args.station_feedback_mode,
        feedback_config=feedback_config,
        feedback_trace_max_records=args.feedback_trace_max_records,
        model_checkpoint=model_checkpoint,
        psi_head_checkpoint=psi_head_checkpoint,
        psi_scale_contract=psi_scale_contract,
        runtime_hashes=runtime_hashes,
    )
    bundle_sha = sha256_file(args.frozen_bundle)
    output = _output_path(
        args.output_root,
        args.station_feedback_mode,
        args.load,
        int(args.seed),
        arm_keys,
    )
    if _resume_compatible(
        output,
        args=args,
        bundle_sha=bundle_sha,
        runtime_hashes=runtime_hashes,
        policy_fingerprint=policy_contract["fingerprint_sha256"],
        manifest_sha=str(manifest["manifest_sha256"]),
        dynamic_config=dynamic_config,
        reference_file_sha=str(reference["file_sha256"]),
        arm_keys=arm_keys,
        schema_version=schema_version,
        contract_version=contract_version,
    ):
        return

    assigner = StationFeedbackClosedLoopPipelinePhiDynamicPsiAssigner(
        station_feedback_mode=args.station_feedback_mode,
        station_feedback_config=feedback_config,
        station_feedback_service_ticks=service_ticks,
        station_feedback_trace_max_records=args.feedback_trace_max_records,
        psi_head_checkpoint=str(psi_head_checkpoint),
        psi_scale_contract=str(psi_scale_contract),
        dynamic_trace_enabled=int(args.policy_trace_max_records) > 0,
        dynamic_trace_max_records=int(args.policy_trace_max_records),
        dynamic_interleaved=True,
        checkpoint_path=str(model_checkpoint),
        top_m=TOP_M,
        energy_conv_random_flip_seed=0,
        **S1_CONFIG,
    )

    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    print(
        f"[run] feedback={args.station_feedback_mode} load={args.load} "
        f"seed={args.seed} ticks={args.ticks} admission=ETA_V3"
    )
    with _engine_contract(
        STATION_ADMISSION_DYNAMIC_ETA_V1,
        dynamic_config=dynamic_config,
        station_trace_stride=args.station_trace_stride,
        station_trace_max_records=args.station_trace_max_records,
        fifo_trace_max_records=0,
    ) as holder:
        metrics = _run_one_assigner(
            str(config_path),
            assigner,
            int(args.seed),
            int(args.ticks),
            trace_label=arm_keys[args.station_feedback_mode],
            recorded_orders_path=str(manifest_path),
        )
    station_probe = holder.get("station_probe")
    if not isinstance(station_probe, StationAdmissionRestorationAuditProbe):
        raise RuntimeError("station admission audit probe was not attached")
    station_audit = station_probe.summary()

    metrics.update(assigner.dynamic_probe_metrics())
    metrics.update({
        "station_admission_mode": STATION_ADMISSION_DYNAMIC_ETA_V1,
        "station_capacity_rejections": station_audit.get(
            "capacity_rejections"
        ),
        "station_over_capacity_grants": station_audit.get(
            "over_capacity_grants"
        ),
        "station_committed_over_capacity_tick_count": station_audit.get(
            "committed_over_capacity_station_tick_count"
        ),
        "station_ticks_with_any_committed_over_capacity": station_audit.get(
            "ticks_with_any_committed_over_capacity"
        ),
    })
    audit = _audit(
        args=args,
        metrics=metrics,
        manifest=manifest,
        station_audit=station_audit,
        feedback_config=feedback_config,
        reference=reference,
    )
    if not audit["passed"]:
        failed = [key for key, passed in audit["checks"].items() if not passed]
        raise RuntimeError(
            f"station-feedback audit failed mode={args.station_feedback_mode} "
            f"load={args.load} seed={args.seed}: {failed}"
        )

    payload = {
        "schema_version": schema_version,
        "meta": {
            "experiment_contract_version": contract_version,
            "source_protocol_sha256": source_protocol.get("protocol_sha256"),
            "frozen_bundle": args.frozen_bundle.as_posix(),
            "frozen_bundle_sha256": bundle_sha,
            "arm_key": arm_keys[args.station_feedback_mode],
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "development_only": True,
            "station_feedback_mode": args.station_feedback_mode,
            "station_feedback_config": feedback_config.as_dict(),
            "station_feedback_service_ticks_source": (
                "simulation.station_process_duration"
            ),
            "station_admission": STATION_ADMISSION_DYNAMIC_ETA_V1,
            "dynamic_admission_config": dynamic_config,
            "policy_contract": policy_contract,
            "runtime_code_sha256": runtime_hashes,
            "isolation_contract": {
                "order_generator_modified": False,
                "station_admission_modified": False,
                "task_lifecycle_modified": False,
                "path_planning_modified": False,
                "world_model_modified": False,
                "s1_modified": False,
                "new_trainable_parameters": 0,
            },
            "engine_log_level": "ERROR",
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "paired_reference": {
            **reference,
            "path": reference["path"].as_posix(),
        },
        "audit": audit,
        "station_admission_audit": station_audit,
        "metrics": metrics,
        "station_feedback_transition_trace": list(
            assigner.station_feedback_trace_records
        ),
        "dynamic_trace": list(assigner.dynamic_probe_trace_records),
    }
    _atomic_json(output, payload)
    print(f"[done] {output}")
    print(json.dumps({
        "completed_orders": metrics.get("completed_orders"),
        "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
        "stall_ratio_mean": metrics.get("stall_ratio_mean"),
        "capacity_rejections": metrics.get("station_capacity_rejections"),
        "max_committed_load": _max_mapping(
            station_audit.get("max_committed_load")
        ),
        "max_occupancy": _max_mapping(
            station_audit.get("max_occupancy")
        ),
        "feedback_forced_defers": metrics.get(
            "station_feedback_forced_defers"
        ),
        "feedback_state_station_ticks": metrics.get(
            "station_feedback_state_station_ticks"
        ),
        "feedback_transitions": metrics.get("station_feedback_transitions"),
    }, indent=2, ensure_ascii=False))


def _validate_args(args: argparse.Namespace) -> None:
    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive")
    if args.feedback_trace_max_records < 0:
        raise SystemExit("--feedback-trace-max-records must be non-negative")
    if args.policy_trace_max_records < 0:
        raise SystemExit("--policy-trace-max-records must be non-negative")
    if args.station_trace_stride <= 0:
        raise SystemExit("--station-trace-stride must be positive")
    if args.station_trace_max_records < 0:
        raise SystemExit("--station-trace-max-records must be non-negative")
    if not args.development:
        if args.ticks != FORMAL_TICKS:
            raise SystemExit(
                f"formal closed-loop evaluation freezes --ticks={FORMAL_TICKS}"
            )
        if int(args.seed) not in FORMAL_SEEDS:
            raise SystemExit(
                f"formal closed-loop evaluation freezes seeds={FORMAL_SEEDS}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--station-feedback-mode",
        choices=STATION_FEEDBACK_MODES,
        required=True,
    )
    parser.add_argument("--load", choices=LOADS, default="high")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ticks", type=int, default=FORMAL_TICKS)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT
    )
    parser.add_argument(
        "--reference-arm", default=DEFAULT_REFERENCE_ARM
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frozen-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--feedback-trace-max-records", type=int, default=500)
    parser.add_argument("--policy-trace-max-records", type=int, default=0)
    parser.add_argument("--station-trace-stride", type=int, default=10)
    parser.add_argument("--station-trace-max-records", type=int, default=300)
    parser.add_argument("--max-committed-multiplier", type=float, default=2.0)
    parser.add_argument("--healthy-extra-ratio", type=float, default=1.0)
    parser.add_argument("--caution-extra-ratio", type=float, default=0.5)
    parser.add_argument("--brake-extra-ratio", type=float, default=0.0)
    parser.add_argument("--eta-near-ticks", type=int, default=8)
    parser.add_argument("--eta-mid-ticks", type=int, default=20)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    _validate_args(args)
    _run(args)


if __name__ == "__main__":
    main()


__all__ = [
    "ARM_KEYS",
    "EXPERIMENT_CONTRACT_VERSION",
    "LEGACY_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
]
