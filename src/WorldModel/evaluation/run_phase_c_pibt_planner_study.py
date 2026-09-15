"""Run and audit the matched Phase-C PP-to-PIBT planner study."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from Policies.TaskAssigner import (
    GreedyTaskAssigner,
    HungarianTaskAssigner,
    WorldModelTaskAssigner,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_context_assigner import (
    PsiDispatchContextWorldModelTaskAssigner,
)
from WorldModel.core.long_risk_schema import long_risk_runtime_contract
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
from WorldModel.evaluation.phase_c_combo_s1_j1_paper_protocol import selector_config
from WorldModel.evaluation.phase_c_pibt_planner_study_protocol import (
    ADAPTED_COMPARISONS,
    ADAPTED_NEW_ARMS,
    ADAPTED_OUTPUT_ROOT,
    ACTION_PATH_MODE,
    ACTION_ROUTE_ENCODING,
    ALL_ARMS,
    ARM_LABELS,
    BUNDLE_SCHEMA_VERSION,
    COMBO_ARMS,
    EVAL_SEEDS,
    LOADS,
    LONG_RISK_HORIZON,
    LONG_RISK_TERMINAL_WINDOW,
    MANIFEST_AUDIT_SCHEMA_VERSION,
    MODEL_ARMS,
    PHASE_B_STAGE1_CHECKPOINT,
    PIBT_TRAINED_CHECKPOINT,
    PIBT_TRAIN_ROOT,
    PLANNER_NAME,
    PLANNER_PARAMS,
    PP_TRAINED_CHECKPOINT,
    PRIMARY_CONTRAST_METRICS,
    PSI_SCALE_CONTRACT,
    RUN_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SUMMARY_METRICS,
    SUMMARY_SCHEMA_VERSION,
    TICKS,
    TOP_M,
    TRAINING_AUDIT_SCHEMA_VERSION,
    TRAIN_SEEDS,
    TRAIN_SPLIT,
    ZERO_SHOT_ARMS,
    ZERO_SHOT_COMPARISONS,
    ZERO_SHOT_OUTPUT_ROOT,
    canonical_sha256,
    checkpoint_for_arm,
    evaluation_protocol,
    manifest_root,
    psi_artifacts_for_arm,
    sha256_file,
    training_protocol,
)
from WorldModel.evaluation.phase_c_s1_hungarian_protocol import PHASEC_CONFIG
from WorldModel.evaluation.phase_c_s1_long_risk_correction_protocol import (
    LOAD_CONFIGS,
)
from WorldModel.evaluation.run_phase_c_combo_s1_j1_paper_experiments import (
    _validate_manifest,
)
from WorldModel.evaluation.run_phase_c_s1_j1_long_risk_correction import (
    _max_mapping,
    _physical_only_audit_contract,
)
from WorldModel.evaluation.run_phase_c_s1_long_risk_correction import (
    _bootstrap_ci,
    _csv_text,
    _mean_std,
    _number,
    _read_json,
    _write_json_exact,
    _write_text_exact,
)
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


EVAL_BUNDLE_FILENAME = "phase_c_pibt_planner_study_frozen_protocol.json"
TRAINING_BUNDLE_FILENAME = "phase_c_pibt_training_frozen_protocol.json"
MANIFEST_CONTRACT_FILENAME = "phase_c_pibt_input_manifests.json"


SOURCE_FILES = {
    "protocol": "WorldModel/evaluation/phase_c_pibt_planner_study_protocol.py",
    "runner": "WorldModel/evaluation/run_phase_c_pibt_planner_study.py",
    "submission": "WorldModel/evaluation/run_phase_c_pibt_planner_study_60cpu.slurm",
    "pibt_planner": "Policies/PathPlanner/PIBTPlanner/pibt_path_planner.py",
    "base_planner": "Policies/PathPlanner/base_path_planner.py",
    "planner_registry": "Policies/PathPlanner/__init__.py",
    "simulation_engine": "Engine/simulation_engine.py",
    "agent_state": "WorldState/agent_state.py",
    "metric_tracker": "Metrics/tracker.py",
    "online_evaluator": "WorldModel/evaluation/evaluate_online_v6.py",
    "counterfactual_rollout": "WorldModel/data/counterfactual_rollout.py",
    "snapshot_probe": "WorldModel/evaluation/decision_snapshot_probe.py",
    "snapshot_collector": "WorldModel/evaluation/collect_phase_c_round1.py",
    "dataset_builder": "WorldModel/data/build_phase_c_round1_dataset.py",
    "long_risk_generator": "WorldModel/data/generate_long_risk_labels.py",
    "pibt_station_head_dataset": (
        "WorldModel/data/build_pibt_station_congestion_head_dataset.py"
    ),
    "station_head_trainer": "WorldModel/training/train_station_congestion_head.py",
    "data_fusion": "WorldModel/data/fuse_and_split.py",
    "trainer": "WorldModel/training/run_train_v6.py",
    "world_model_assigner": (
        "Policies/TaskAssigner/WorldModelTaskAssigner/"
        "world_model_task_assigner.py"
    ),
    "static_j_assigner": (
        "Policies/TaskAssigner/WorldModelTaskAssigner/"
        "psi_dispatch_context_assigner.py"
    ),
    "long_risk_schema": "WorldModel/core/long_risk_schema.py",
    "pibt_tests": "WorldModel/tests/test_pibt_rmfs_integration.py",
    "study_tests": "WorldModel/tests/test_phase_c_pibt_planner_study.py",
}


def _eval_bundle_path(root: Path) -> Path:
    return root / EVAL_BUNDLE_FILENAME


def _training_bundle_path(root: Path) -> Path:
    return root / TRAINING_BUNDLE_FILENAME


def _manifest_contract_path(root: Path) -> Path:
    return root / "validation" / MANIFEST_CONTRACT_FILENAME


def _artifact_snapshot(paths: Mapping[str, Path | str]) -> dict[str, Any]:
    result = {}
    for name, raw_path in paths.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {
            "path": path.as_posix(),
            "sha256": sha256_file(path),
        }
    return result


def _manifest_cell_key(load: str, seed: int) -> str:
    return f"{load}:seed{int(seed)}"


def _manifest_path_for_roots(
    load: str,
    seed: int,
    *,
    high_root: Path,
    low_mid_root: Path,
) -> Path:
    root = high_root if load == "high" else low_mid_root
    return root / "order_manifests" / f"orders_{load}_seed{int(seed)}.json"


def _load_manifest_contract(
    path: Path,
    *,
    verify_manifest_files: bool = True,
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != MANIFEST_AUDIT_SCHEMA_VERSION:
        raise ValueError(f"wrong manifest-contract schema: {path}")
    claimed = payload.get("contract_sha256")
    body = dict(payload)
    body.pop("contract_sha256", None)
    if claimed != canonical_sha256(body):
        raise ValueError(f"manifest-contract hash mismatch: {path}")

    cells = payload.get("cells") or {}
    expected = {
        _manifest_cell_key(load, seed)
        for load in LOADS
        for seed in EVAL_SEEDS
    }
    if set(cells) != expected:
        missing = sorted(expected - set(cells))
        extra = sorted(set(cells) - expected)
        raise ValueError(
            f"manifest-contract cell mismatch: missing={missing}, extra={extra}"
        )
    if verify_manifest_files:
        for key, row in cells.items():
            manifest_path = Path(str(row.get("path")))
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = _validate_manifest(manifest_path)
            if sha256_file(manifest_path) != row.get("file_sha256"):
                raise ValueError(f"manifest file hash mismatch ({key})")
            if manifest.get("manifest_sha256") != row.get("content_sha256"):
                raise ValueError(f"manifest content hash mismatch ({key})")
            if int(manifest.get("total_orders", -1)) != int(
                row.get("total_orders", -2)
            ):
                raise ValueError(f"manifest order count mismatch ({key})")
    return payload


def _audit_manifests(args: argparse.Namespace) -> None:
    high_root = Path(args.high_manifest_root)
    low_mid_root = Path(args.low_mid_manifest_root)
    cells: dict[str, dict[str, Any]] = {}
    for load in LOADS:
        for seed in EVAL_SEEDS:
            path = _manifest_path_for_roots(
                load,
                seed,
                high_root=high_root,
                low_mid_root=low_mid_root,
            )
            manifest = _validate_manifest(path)
            cells[_manifest_cell_key(load, seed)] = {
                "load": load,
                "seed": int(seed),
                "path": path.as_posix(),
                "file_sha256": sha256_file(path),
                "content_sha256": manifest.get("manifest_sha256"),
                "total_orders": int(manifest.get("total_orders", 0)),
            }

    payload: dict[str, Any] = {
        "schema_version": MANIFEST_AUDIT_SCHEMA_VERSION,
        "purpose": (
            "freeze the exact paired low/mid/high order streams shared by "
            "zero-shot and PIBT-adapted evaluation"
        ),
        "loads": list(LOADS),
        "seeds": list(EVAL_SEEDS),
        "ticks": TICKS,
        "roots": {
            "high": high_root.as_posix(),
            "low_mid": low_mid_root.as_posix(),
        },
        "cells": cells,
    }
    payload["contract_sha256"] = canonical_sha256(payload)
    output = Path(
        args.manifest_contract
        or _manifest_contract_path(Path(args.output_root))
    )
    _write_json_exact(output, payload)
    _load_manifest_contract(output)
    print(f"[complete] {output}")


def _prepare_manifest(args: argparse.Namespace) -> None:
    if args.load not in LOADS or int(args.seed) not in EVAL_SEEDS:
        raise ValueError("load/seed is outside the frozen evaluation block")
    if int(args.ticks) != TICKS:
        raise ValueError(f"PIBT study freezes ticks={TICKS}")
    path = _manifest_path_for_roots(
        args.load,
        int(args.seed),
        high_root=Path(args.high_manifest_root),
        low_mid_root=Path(args.low_mid_manifest_root),
    )
    if path.is_file():
        _validate_manifest(path)
        print(f"[skip] existing manifest {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[run] manifest source Greedy+default-PP physical-only "
        f"load={args.load} seed={args.seed}"
    )
    with _physical_only_audit_contract() as holder:
        metrics = _run_one_assigner(
            LOAD_CONFIGS[args.load],
            GreedyTaskAssigner(),
            int(args.seed),
            int(args.ticks),
            trace_label="PIBTStudyManifestSourceGreedyPPPhysicalOnly",
            save_order_manifest=str(path),
        )
    station_probe = holder.get("station_probe")
    if station_probe is None:
        raise RuntimeError("manifest-source station audit probe was not attached")
    station_audit = station_probe.summary()
    if not bool(station_audit.get("passed")):
        raise RuntimeError(
            f"manifest-source station audit failed: {args.load} seed={args.seed}"
        )
    manifest = _validate_manifest(path)
    if int(metrics.get("order_arrival_count", -1)) != int(
        manifest.get("total_orders", -2)
    ):
        raise RuntimeError("generated manifest order accounting mismatch")
    print(f"[done] generated {path}")


def _freeze_evaluation(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    checkpoint = Path(args.checkpoint)
    campaign = args.campaign
    manifest_contract_path = Path(
        args.manifest_contract or _manifest_contract_path(root)
    )
    manifest_contract = _load_manifest_contract(manifest_contract_path)
    if campaign == "zero_shot" and checkpoint.resolve() != Path(
        PP_TRAINED_CHECKPOINT
    ).resolve():
        raise ValueError("zero_shot campaign must use the frozen PP checkpoint")
    if campaign == "adapted" and checkpoint.resolve() != Path(
        PIBT_TRAINED_CHECKPOINT
    ).resolve():
        raise ValueError("adapted campaign must use the PIBT-trained checkpoint")

    combo_arm = (
        "combo_ppwm_pibt" if campaign == "zero_shot" else "combo_pibtwm_pibt"
    )
    psi_artifacts = psi_artifacts_for_arm(combo_arm)
    if psi_artifacts is None:
        raise AssertionError("combo arm must define J1 artifacts")
    psi_head, psi_scale = psi_artifacts
    if not psi_head.is_file() or not psi_scale.is_file():
        raise FileNotFoundError(
            f"missing {campaign} J1 artifacts: head={psi_head}, scale={psi_scale}"
        )
    _validate_psi_artifacts(
        psi_head,
        psi_scale,
        checkpoint,
        require_formal=(campaign == "adapted"),
    )
    artifacts: dict[str, Path | str] = dict(SOURCE_FILES)
    artifacts.update({
        "world_model_checkpoint": checkpoint,
        "psi_head_checkpoint": psi_head,
        "psi_scale_contract": psi_scale,
        "input_manifest_contract": manifest_contract_path,
    })
    source_root = None
    if campaign == "adapted":
        source_root = Path(args.source_zero_shot_root)
        artifacts.update({
            "zero_shot_bundle": _eval_bundle_path(source_root),
            "zero_shot_summary": source_root / "validation/summary.json",
            "zero_shot_hashes": (
                source_root / "validation/validated_outputs.sha256"
            ),
        })

    protocol = evaluation_protocol(campaign, checkpoint)
    protocol["input_manifest_contract"] = {
        "path": manifest_contract_path.as_posix(),
        "file_sha256": sha256_file(manifest_contract_path),
        "contract_sha256": manifest_contract["contract_sha256"],
    }
    protocol["source_zero_shot_root"] = (
        source_root.as_posix() if source_root is not None else None
    )
    protocol_sha = canonical_sha256(protocol)
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "artifacts": _artifact_snapshot(artifacts),
    }
    root.mkdir(parents=True, exist_ok=True)
    _write_json_exact(_eval_bundle_path(root), bundle)
    print(f"[freeze] {_eval_bundle_path(root)}")
    print(f"[freeze] protocol_sha256={protocol_sha}")


def _validate_psi_artifacts(
    head_path: Path,
    scale_path: Path,
    model_checkpoint: Path,
    *,
    require_formal: bool = False,
) -> dict[str, Any]:
    import torch

    payload = torch.load(head_path, map_location="cpu", weights_only=False)
    scale = _read_json(scale_path)
    expected_encoder_sha = sha256_file(model_checkpoint)
    checks = {
        "source_encoder_sha": payload.get("source_encoder_checkpoint_sha256")
        == expected_encoder_sha,
        "head_scale_contract": payload.get("scale_contract_sha256")
        == (payload.get("scale_contract") or {}).get("contract_sha256"),
        "external_scale_contract": payload.get("scale_contract_sha256")
        == scale.get("contract_sha256"),
        "region_representation": (payload.get("representation") or {}).get(
            "name"
        ) == "station_region_mean_max",
        "formal_training_contract": (
            not require_formal
            or bool((payload.get("audit") or {}).get("formal_training_contract"))
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("J1 artifact preflight failed: " + ", ".join(failed))
    return {
        "source_encoder_checkpoint_sha256": expected_encoder_sha,
        "scale_contract_sha256": scale.get("contract_sha256"),
        "checks": checks,
    }


def _freeze_training(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    artifacts: dict[str, Path | str] = dict(SOURCE_FILES)
    artifacts["phase_b_stage1_checkpoint"] = PHASE_B_STAGE1_CHECKPOINT
    artifacts["lyapunov_config"] = "Config/lyapunov_l1_config.json"
    artifacts["static_j1_scale_contract"] = PSI_SCALE_CONTRACT
    for load, path in LOAD_CONFIGS.items():
        artifacts[f"config_{load}"] = path
    protocol = training_protocol()
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "artifacts": _artifact_snapshot(artifacts),
    }
    root.mkdir(parents=True, exist_ok=True)
    _write_json_exact(_training_bundle_path(root), bundle)
    print(f"[freeze] {_training_bundle_path(root)}")


def _verify_bundle(path: Path, *, expected_campaign: str | None = None) -> dict:
    payload = _read_json(path)
    if payload.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"wrong bundle schema: {path}")
    protocol = payload.get("protocol") or {}
    if payload.get("protocol_sha256") != canonical_sha256(protocol):
        raise ValueError(f"protocol hash mismatch: {path}")
    if expected_campaign and protocol.get("campaign") != expected_campaign:
        raise ValueError(f"wrong campaign in {path}")
    for name, row in (payload.get("artifacts") or {}).items():
        artifact = Path(str(row.get("path")))
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        if sha256_file(artifact) != row.get("sha256"):
            raise ValueError(f"artifact hash mismatch ({name}): {artifact}")
    return payload


def _bundle_manifest_contract(bundle: Mapping[str, Any]) -> dict[str, Any]:
    row = (bundle.get("artifacts") or {}).get("input_manifest_contract") or {}
    path = Path(str(row.get("path")))
    contract = _load_manifest_contract(path)
    protocol_row = (bundle.get("protocol") or {}).get(
        "input_manifest_contract"
    ) or {}
    checks = {
        "artifact_file_sha": row.get("sha256") == sha256_file(path),
        "protocol_path": protocol_row.get("path") == path.as_posix(),
        "protocol_file_sha": protocol_row.get("file_sha256")
        == sha256_file(path),
        "protocol_contract_sha": protocol_row.get("contract_sha256")
        == contract.get("contract_sha256"),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("bundle manifest contract mismatch: " + ", ".join(failed))
    return contract


def _make_assigner(arm: str):
    if arm == "greedy_pibt":
        return GreedyTaskAssigner()
    if arm == "hungarian_pibt":
        return HungarianTaskAssigner()
    checkpoint = checkpoint_for_arm(arm)
    if checkpoint is None:
        raise ValueError(arm)
    if arm.startswith("phasec_"):
        return WorldModelTaskAssigner(
            checkpoint_path=str(checkpoint),
            top_m=TOP_M,
            action_path_mode=ACTION_PATH_MODE,
            **dict(PHASEC_CONFIG),
        )
    if arm.startswith("combo_"):
        psi_artifacts = psi_artifacts_for_arm(arm)
        if psi_artifacts is None:
            raise AssertionError(f"missing J1 artifacts for {arm}")
        psi_head, psi_scale = psi_artifacts
        return PsiDispatchContextWorldModelTaskAssigner(
            psi_head_checkpoint=str(psi_head),
            psi_scale_contract=str(psi_scale),
            psi_context_mode="j_ascending",
            psi_trace_enabled=False,
            psi_trace_max_records=0,
            allow_phasec_s0_robot_scorer=False,
            checkpoint_path=str(checkpoint),
            top_m=TOP_M,
            action_path_mode=ACTION_PATH_MODE,
            energy_conv_random_flip_seed=0,
            **selector_config(),
        )
    raise ValueError(arm)


def _policy_contract(arm: str) -> dict[str, Any]:
    checkpoint = checkpoint_for_arm(arm)
    contract = {
        "arm": arm,
        "label": ARM_LABELS[arm],
        "path_planner": PLANNER_NAME,
        "path_planner_params": dict(PLANNER_PARAMS),
        "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
        "checkpoint": checkpoint.as_posix() if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "action_path_mode": ACTION_PATH_MODE if arm in MODEL_ARMS else None,
        "action_route_encoding": (
            ACTION_ROUTE_ENCODING if arm in MODEL_ARMS else None
        ),
        "robot_selector": (
            "greedy"
            if arm == "greedy_pibt"
            else "hungarian"
            if arm == "hungarian_pibt"
            else "phasec_s0"
            if arm.startswith("phasec_")
            else "corrected_quantile_combo_s1"
        ),
        "context_scheduler": "static_j1" if arm in COMBO_ARMS else "j0",
    }
    if arm in COMBO_ARMS:
        psi_artifacts = psi_artifacts_for_arm(arm)
        if psi_artifacts is None:
            raise AssertionError(f"missing J1 artifacts for {arm}")
        psi_head, psi_scale = psi_artifacts
        contract.update({
            "psi_head_checkpoint": psi_head.as_posix(),
            "psi_head_checkpoint_sha256": sha256_file(psi_head),
            "psi_scale_contract": psi_scale.as_posix(),
            "psi_scale_contract_sha256": sha256_file(psi_scale),
        })
    contract["fingerprint_sha256"] = canonical_sha256(contract)
    return contract


def _output_path(root: Path, arm: str, load: str, seed: int) -> Path:
    return root / "per_arm" / arm / f"{load}_seed{seed}.json"


def _resume_ok(
    path: Path,
    *,
    bundle: dict,
    arm: str,
    load: str,
    seed: int,
    manifest_sha: str,
    policy_sha: str,
) -> bool:
    if not path.is_file():
        return False
    payload = _read_json(path)
    meta = payload.get("meta") or {}
    return all((
        payload.get("schema_version") == RUN_SCHEMA_VERSION,
        meta.get("protocol_sha256") == bundle.get("protocol_sha256"),
        meta.get("arm_key") == arm,
        meta.get("load") == load,
        int(meta.get("seed", -1)) == seed,
        int(meta.get("ticks", -1)) == TICKS,
        (payload.get("manifest") or {}).get("content_sha256") == manifest_sha,
        (meta.get("policy_contract") or {}).get("fingerprint_sha256")
        == policy_sha,
        bool((payload.get("audit") or {}).get("passed")),
    ))


def _run_audit(
    arm: str,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    station_audit: Mapping[str, Any],
) -> dict[str, Any]:
    last_audit = metrics.get("pibt_last_batch_audit") or {}
    planned = int(metrics.get("pibt_planned_agents", 0))
    decisions = int(metrics.get("pibt_move_decisions", 0)) + int(
        metrics.get("pibt_wait_decisions", 0)
    )
    checks = {
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "manifest_sha": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_count": int(metrics.get("order_arrival_count", -1))
        == int(manifest.get("total_orders", -2)),
        "station_audit": bool(station_audit.get("passed")),
        "station_mode": station_audit.get("mode")
        == STATION_ADMISSION_PHYSICAL_ONLY,
        "physical_capacity": int(
            station_audit.get("physical_capacity_violation_count", -1)
        ) == 0,
        "planner_name": metrics.get("path_planner_name") == PLANNER_NAME,
        "batch_interface": bool(metrics.get("path_planner_batch_interface")),
        "single_step": bool(metrics.get("path_planner_single_step")),
        "strict_validation": bool(metrics.get("pibt_strict_validation")),
        "batch_called": int(metrics.get("pibt_batch_calls", 0)) > 0,
        "sequential_interface_unused": int(
            metrics.get("pibt_single_calls", -1)
        ) == 0,
        "planned_agents": planned > 0,
        "decision_accounting": decisions == planned,
        "invariant_clean": int(metrics.get("pibt_validation_failures", -1)) == 0,
        "last_batch_clean": bool(last_audit.get("passed")),
        "engine_vertex_conflict_free": int(
            metrics.get("path_planner_engine_vertex_conflicts", -1)
        ) == 0,
        "engine_swap_conflict_free": int(
            metrics.get("path_planner_engine_swap_conflicts", -1)
        ) == 0,
        "metric_completed_orders_present": _number(
            metrics.get("completed_orders")
        ) is not None,
        "metric_task_duration_present": _number(
            metrics.get("avg_task_duration")
        ) is not None,
        "metric_deadlock_present": _number(
            metrics.get("deadlock_ratio_mean")
        ) is not None,
    }
    if arm in MODEL_ARMS:
        checks.update({
            "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
            "no_greedy_fallback": int(
                metrics.get("fallback_greedy_calls", 0)
            ) == 0,
            "action_path_mode": int(metrics.get("action_path_mode", -1))
            == ACTION_PATH_MODE,
            "action_route_encoding": metrics.get("action_route_encoding")
            == ACTION_ROUTE_ENCODING,
            "world_model_path_planner_injected": bool(
                metrics.get("world_model_path_planner_injected")
            ),
        })
    else:
        checks["model_not_used"] = int(metrics.get("model_assign_calls", 0)) == 0
    if arm in COMBO_ARMS:
        checks.update({
            "combo_signal": metrics.get("energy_drift_signal") == "combo",
            "j1_mode": metrics.get("psi_dispatch_mode") == "j_ascending",
            "j1_loaded": bool(metrics.get("psi_dispatch_head_loaded")),
            "j1_encoder_contract": bool(
                metrics.get("psi_dispatch_encoder_contract_verified")
            ),
            "j1_source_encoder_sha": (
                metrics.get("psi_dispatch_source_encoder_checkpoint_sha256")
                == sha256_file(checkpoint_for_arm(arm))
            ),
        })
    return {"passed": all(checks.values()), "checks": checks}


def _run_arm(args: argparse.Namespace) -> None:
    campaign_arms = (
        ZERO_SHOT_ARMS if args.campaign == "zero_shot" else ADAPTED_NEW_ARMS
    )
    if args.arm not in campaign_arms:
        raise ValueError(f"arm {args.arm} is not part of {args.campaign}")
    if args.load not in LOADS or int(args.seed) not in EVAL_SEEDS:
        raise ValueError("load/seed is outside the frozen evaluation block")
    if int(args.ticks) != TICKS:
        raise ValueError(f"PIBT study freezes ticks={TICKS}")

    root = Path(args.output_root)
    bundle_path = Path(args.frozen_bundle or _eval_bundle_path(root))
    bundle = _verify_bundle(bundle_path, expected_campaign=args.campaign)
    manifest_contract = _bundle_manifest_contract(bundle)
    manifest_path = Path(args.manifest_path)
    manifest = _validate_manifest(manifest_path)
    manifest_cell = (manifest_contract.get("cells") or {}).get(
        _manifest_cell_key(args.load, int(args.seed))
    ) or {}
    manifest_checks = {
        "path": manifest_path.resolve()
        == Path(str(manifest_cell.get("path"))).resolve(),
        "file_sha256": sha256_file(manifest_path)
        == manifest_cell.get("file_sha256"),
        "content_sha256": manifest.get("manifest_sha256")
        == manifest_cell.get("content_sha256"),
        "total_orders": int(manifest.get("total_orders", -1))
        == int(manifest_cell.get("total_orders", -2)),
    }
    if not all(manifest_checks.values()):
        failed = [name for name, passed in manifest_checks.items() if not passed]
        raise ValueError(
            "run manifest is outside the frozen paired contract: "
            + ", ".join(failed)
        )
    policy = _policy_contract(args.arm)
    output = _output_path(root, args.arm, args.load, int(args.seed))
    if _resume_ok(
        output,
        bundle=bundle,
        arm=args.arm,
        load=args.load,
        seed=int(args.seed),
        manifest_sha=str(manifest["manifest_sha256"]),
        policy_sha=str(policy["fingerprint_sha256"]),
    ):
        print(f"[skip] {output}")
        return

    assigner = _make_assigner(args.arm)
    print(
        f"[run] campaign={args.campaign} arm={args.arm} "
        f"load={args.load} seed={args.seed}"
    )
    with _physical_only_audit_contract() as holder:
        metrics = _run_one_assigner(
            LOAD_CONFIGS[args.load],
            assigner,
            int(args.seed),
            int(args.ticks),
            trace_label=ARM_LABELS[args.arm],
            recorded_orders_path=str(manifest_path),
            path_planner_override=PLANNER_NAME,
            path_planner_params_override=dict(PLANNER_PARAMS),
        )
    if hasattr(assigner, "psi_dispatch_metrics"):
        metrics.update(assigner.psi_dispatch_metrics())
    planned = int(metrics.get("pibt_planned_agents", 0))
    metrics["pibt_wait_decision_ratio"] = round(
        int(metrics.get("pibt_wait_decisions", 0)) / max(planned, 1),
        8,
    )
    metrics["pibt_priority_inheritance_per_agent"] = round(
        int(metrics.get("pibt_priority_inheritance_calls", 0))
        / max(planned, 1),
        8,
    )
    metrics["pibt_backtracks_per_agent"] = round(
        int(metrics.get("pibt_backtracks", 0)) / max(planned, 1),
        8,
    )
    station_probe = holder.get("station_probe")
    if station_probe is None:
        raise RuntimeError("physical-only station audit probe was not attached")
    station_audit = station_probe.summary()
    metrics.update({
        "station_admission_mode": STATION_ADMISSION_PHYSICAL_ONLY,
        "station_capacity_rejections": int(
            station_audit.get("capacity_rejections", 0)
        ),
        "station_max_occupancy": _max_mapping(
            station_audit.get("max_occupancy")
        ),
        "station_max_committed_load": _max_mapping(
            station_audit.get("max_committed_load")
        ),
    })
    audit = _run_audit(args.arm, metrics, manifest, station_audit)
    if not audit["passed"]:
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise RuntimeError("PIBT run audit failed: " + ", ".join(failed))

    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": bundle["protocol_sha256"],
            "frozen_bundle": bundle_path.as_posix(),
            "frozen_bundle_sha256": sha256_file(bundle_path),
            "campaign": args.campaign,
            "arm_key": args.arm,
            "arm": ARM_LABELS[args.arm],
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "base_config": LOAD_CONFIGS[args.load],
            "path_planner_override": PLANNER_NAME,
            "path_planner_params_override": dict(PLANNER_PARAMS),
            "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
            "policy_contract": policy,
            "long_risk_runtime_contract": (
                long_risk_runtime_contract() if args.arm in MODEL_ARMS else None
            ),
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
            "contract_sha256": manifest_contract.get("contract_sha256"),
        },
        "station_audit": station_audit,
        "audit": audit,
        "metrics": metrics,
    }
    _write_json_exact(output, payload)
    print(f"[done] {output}")


def _paired_report(
    rows: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
    baseline: str,
    candidate: str,
    metric: str,
    seed_base: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = {}
    csv_rows = []
    clustered = []
    for seed in EVAL_SEEDS:
        values = []
        for load in LOADS:
            left = _number(rows[(load, seed)][baseline].get(metric))
            right = _number(rows[(load, seed)][candidate].get(metric))
            if left is not None and right is not None:
                values.append(right - left)
        if len(values) == len(LOADS):
            clustered.append(sum(values) / len(values))
    for index, load in enumerate(LOADS):
        values = []
        for seed in EVAL_SEEDS:
            left = _number(rows[(load, seed)][baseline].get(metric))
            right = _number(rows[(load, seed)][candidate].get(metric))
            if left is not None and right is not None:
                values.append(right - left)
        summary = _mean_std(values)
        summary["ci95_low"], summary["ci95_high"] = _bootstrap_ci(
            values, seed_base + index
        )
        report[load] = summary
        csv_rows.append({"scope": load, **summary})
    overall = _mean_std(clustered)
    overall["ci95_low"], overall["ci95_high"] = _bootstrap_ci(
        clustered, seed_base + 90
    )
    report["overall_seed_cluster"] = overall
    csv_rows.append({"scope": "overall_seed_cluster", **overall})
    return report, csv_rows


def _load_campaign_rows(
    campaign: str,
    root: Path,
    source_root: Path | None,
    bundle: Mapping[str, Any],
) -> tuple[dict, list[Path]]:
    arms = ZERO_SHOT_ARMS if campaign == "zero_shot" else ALL_ARMS
    manifest_contract = _bundle_manifest_contract(bundle)
    source_bundle = None
    if campaign == "adapted":
        if source_root is None:
            raise RuntimeError("missing zero-shot source root")
        source_bundle = _verify_bundle(
            _eval_bundle_path(source_root), expected_campaign="zero_shot"
        )
        source_contract = _bundle_manifest_contract(source_bundle)
        if source_contract.get("contract_sha256") != manifest_contract.get(
            "contract_sha256"
        ):
            raise ValueError("adapted and zero-shot manifest contracts differ")
    rows = {}
    files = []
    for load in LOADS:
        for seed in EVAL_SEEDS:
            cell = {}
            for arm in arms:
                arm_root = (
                    source_root
                    if campaign == "adapted" and arm in ZERO_SHOT_ARMS
                    else root
                )
                if arm_root is None:
                    raise RuntimeError("missing zero-shot source root")
                path = _output_path(arm_root, arm, load, seed)
                payload = _read_json(path)
                expected_bundle = (
                    source_bundle
                    if campaign == "adapted" and arm in ZERO_SHOT_ARMS
                    else bundle
                )
                if expected_bundle is None:
                    raise RuntimeError("missing expected run bundle")
                meta = payload.get("meta") or {}
                manifest_row = payload.get("manifest") or {}
                contract_cell = (manifest_contract.get("cells") or {}).get(
                    _manifest_cell_key(load, seed)
                ) or {}
                checks = {
                    "schema": payload.get("schema_version")
                    == RUN_SCHEMA_VERSION,
                    "protocol": meta.get("protocol_sha256")
                    == expected_bundle.get("protocol_sha256"),
                    "campaign": meta.get("campaign")
                    == (
                        "zero_shot"
                        if campaign == "adapted" and arm in ZERO_SHOT_ARMS
                        else campaign
                    ),
                    "arm": meta.get("arm_key") == arm,
                    "load": meta.get("load") == load,
                    "seed": int(meta.get("seed", -1)) == seed,
                    "ticks": int(meta.get("ticks", -1)) == TICKS,
                    "manifest_file": manifest_row.get("file_sha256")
                    == contract_cell.get("file_sha256"),
                    "manifest_content": manifest_row.get("content_sha256")
                    == contract_cell.get("content_sha256"),
                    "manifest_count": int(
                        manifest_row.get("total_orders", -1)
                    ) == int(contract_cell.get("total_orders", -2)),
                    "manifest_contract": manifest_row.get("contract_sha256")
                    == manifest_contract.get("contract_sha256"),
                }
                if not all(checks.values()):
                    failed = [
                        name for name, passed in checks.items() if not passed
                    ]
                    raise ValueError(
                        f"incompatible source run {path}: {', '.join(failed)}"
                    )
                if not bool((payload.get("audit") or {}).get("passed")):
                    raise ValueError(f"failed source audit: {path}")
                cell[arm] = payload.get("metrics") or {}
                files.append(path)
            rows[(load, seed)] = cell
    return rows, files


def _summarize(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    bundle = _verify_bundle(
        Path(args.frozen_bundle or _eval_bundle_path(root)),
        expected_campaign=args.campaign,
    )
    source_root = (
        Path(args.source_zero_shot_root)
        if args.campaign == "adapted"
        else None
    )
    rows, input_files = _load_campaign_rows(
        args.campaign, root, source_root, bundle
    )
    arms = ZERO_SHOT_ARMS if args.campaign == "zero_shot" else ALL_ARMS
    comparisons = (
        ZERO_SHOT_COMPARISONS
        if args.campaign == "zero_shot"
        else ADAPTED_COMPARISONS
    )

    aggregate_json = {}
    aggregate_rows = []
    for arm in arms:
        aggregate_json[arm] = {}
        for load in LOADS:
            aggregate_json[arm][load] = {}
            for metric in SUMMARY_METRICS:
                values = [
                    value
                    for seed in EVAL_SEEDS
                    if (value := _number(rows[(load, seed)][arm].get(metric)))
                    is not None
                ]
                summary = _mean_std(values)
                aggregate_json[arm][load][metric] = summary
                aggregate_rows.append({
                    "arm_key": arm,
                    "arm": ARM_LABELS[arm],
                    "load": load,
                    "metric": metric,
                    **summary,
                })

    comparisons_json = {}
    comparison_rows = []
    for comparison_index, (name, baseline, candidate) in enumerate(comparisons):
        block = {
            "baseline": baseline,
            "candidate": candidate,
            "definition": "candidate_minus_baseline",
            "metrics": {},
        }
        for metric_index, metric in enumerate(PRIMARY_CONTRAST_METRICS):
            report, csv_rows = _paired_report(
                rows,
                baseline,
                candidate,
                metric,
                9_000_000 + comparison_index * 100_000 + metric_index * 100,
            )
            block["metrics"][metric] = report
            for row in csv_rows:
                comparison_rows.append({
                    "comparison": name,
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    **row,
                })
        comparisons_json[name] = block

    validation = root / "validation"
    aggregate_csv = validation / "aggregate_summary.csv"
    paired_csv = validation / "paired_contrasts.csv"
    summary_path = validation / "summary.json"
    _write_text_exact(
        aggregate_csv,
        _csv_text(
            aggregate_rows,
            ["arm_key", "arm", "load", "metric", "n", "mean", "std"],
        ),
    )
    _write_text_exact(
        paired_csv,
        _csv_text(
            comparison_rows,
            [
                "comparison", "baseline", "candidate", "metric", "scope",
                "n", "mean", "std", "ci95_low", "ci95_high",
            ],
        ),
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "campaign": args.campaign,
        "protocol_sha256": bundle["protocol_sha256"],
        "frozen_bundle_sha256": sha256_file(
            Path(args.frozen_bundle or _eval_bundle_path(root))
        ),
        "artifact_audit": {
            "passed": True,
            "loads": len(LOADS),
            "seeds": len(EVAL_SEEDS),
            "arms": len(arms),
            "outputs": len(input_files),
        },
        "metric_semantics": bundle["protocol"]["metric_semantics"],
        "aggregate": aggregate_json,
        "comparisons": comparisons_json,
    }
    _write_json_exact(summary_path, summary)
    validated = sorted(
        set(
            input_files
            + [
                Path(args.frozen_bundle or _eval_bundle_path(root)),
                Path(
                    bundle["artifacts"]["input_manifest_contract"]["path"]
                ),
                aggregate_csv,
                paired_csv,
                summary_path,
            ]
        ),
        key=lambda path: path.as_posix(),
    )
    _write_text_exact(
        validation / "validated_outputs.sha256",
        "".join(
            f"{sha256_file(path)}  {path.as_posix()}\n" for path in validated
        ),
    )
    print(f"[complete] {summary_path}")


def _audit_training(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    stage = args.training_stage
    bundle_path = Path(args.frozen_bundle or _training_bundle_path(root))
    bundle = _verify_bundle(bundle_path, expected_campaign="pibt_on_policy_round1")
    protocol = bundle.get("protocol") or {}
    checks: dict[str, bool] = {}
    files: list[Path] = [bundle_path]
    snapshot_root = root / "snapshots_461_470"
    dataset_root = root / "datasets_461_470"
    long_risk_root = root / "long_risk_w200_461_470"
    long_risk_hashes = long_risk_root / "long_risk_outputs.sha256"

    snapshot_hashes = snapshot_root / "snapshot_collection_outputs.sha256"
    dataset_hashes = dataset_root / "replayed_datasets.sha256"
    checks["snapshot_hash_contract"] = snapshot_hashes.is_file()
    checks["dataset_hash_contract"] = dataset_hashes.is_file()
    checks["long_risk_hash_contract"] = long_risk_hashes.is_file()
    files.extend((snapshot_hashes, dataset_hashes, long_risk_hashes))

    for load in LOADS:
        collection_path = snapshot_root / f"collection_{load}.json"
        collection = _read_json(collection_path)
        collection_protocol = collection.get("protocol") or {}
        checks[f"collection_{load}_planner"] = (
            collection_protocol.get("path_planner_override") == PLANNER_NAME
            and collection_protocol.get("path_planner_params_override")
            == PLANNER_PARAMS
            and collection_protocol.get("behavior_checkpoint_sha256")
            == sha256_file(PHASE_B_STAGE1_CHECKPOINT)
        )
        checks[f"collection_{load}_scope"] = (
            collection.get("load") == load
            and collection_protocol.get("load") == load
            and collection_protocol.get("seeds") == list(TRAIN_SEEDS)
            and set(collection.get("per_seed") or {})
            == {str(seed) for seed in TRAIN_SEEDS}
        )
        files.append(collection_path)

        for seed in TRAIN_SEEDS:
            key = f"{load}_{seed}"
            run_root = snapshot_root / load / f"seed{seed}"
            metrics_path = run_root / "metrics.json"
            metrics_payload = _read_json(metrics_path)
            metrics = metrics_payload.get("metrics") or {}
            planned = int(metrics.get("pibt_planned_agents", 0))
            decisions = int(metrics.get("pibt_move_decisions", 0)) + int(
                metrics.get("pibt_wait_decisions", 0)
            )
            checks[f"{key}_run_provenance"] = (
                metrics_payload.get("protocol_sha256")
                == collection_protocol.get("protocol_sha256")
                and metrics_payload.get("load") == load
                and int(metrics_payload.get("seed", -1)) == seed
            )
            checks[f"{key}_planner"] = all((
                metrics.get("path_planner_name") == PLANNER_NAME,
                bool(metrics.get("path_planner_batch_interface")),
                bool(metrics.get("path_planner_single_step")),
                bool(metrics.get("pibt_strict_validation")),
                int(metrics.get("pibt_batch_calls", 0)) > 0,
                int(metrics.get("pibt_single_calls", -1)) == 0,
                planned > 0,
                decisions == planned,
                int(metrics.get("pibt_validation_failures", -1)) == 0,
                bool((metrics.get("pibt_last_batch_audit") or {}).get("passed")),
                int(metrics.get("path_planner_engine_vertex_conflicts", -1))
                == 0,
                int(metrics.get("path_planner_engine_swap_conflicts", -1))
                == 0,
            ))
            index_paths = list((run_root / "snapshots").glob("snapindex_*.json"))
            checks[f"{key}_one_index"] = len(index_paths) == 1
            if len(index_paths) != 1:
                raise RuntimeError(f"wrong snapindex count for {key}")
            index = _read_json(index_paths[0])
            checks[f"{key}_index_planner"] = (
                index.get("path_planner_override") == PLANNER_NAME
                and index.get("path_planner_params_override") == PLANNER_PARAMS
                and index.get("load") == load
                and int(index.get("seed", -1)) == seed
            )

            data_root = dataset_root / load / f"seed{seed}"
            data_path = data_root / "data.pt"
            long_risk_path = long_risk_root / load / f"seed{seed}" / (
                "long_risk_labels.pt"
            )
            gen_path = data_root / "gen_config.json"
            meta_path = data_root / "data_meta.json"
            gen = _read_json(gen_path)
            meta = _read_json(meta_path)
            checks[f"{key}_dataset_file"] = data_path.is_file()
            checks[f"{key}_long_risk_file"] = long_risk_path.is_file()
            checks[f"{key}_dataset_planner"] = (
                gen.get("path_planner_override") == PLANNER_NAME
                and gen.get("path_planner_params_override") == PLANNER_PARAMS
                and gen.get("load_level") == load
                and int(gen.get("seed", -1)) == seed
                and gen.get("rollout_continuation_mode") == "isolated"
                and gen.get("training_source_policy") == "world_model_on_policy"
                and not bool(gen.get("external_baseline_training_samples"))
                and int(gen.get("action_path_mode", -1)) == ACTION_PATH_MODE
                and gen.get("action_route_encoding") == ACTION_ROUTE_ENCODING
                and not bool(gen.get("action_path_planner_preview_used", True))
            )
            checks[f"{key}_dataset_audit"] = bool(
                (meta.get("audit") or {}).get("passed")
            )
            if long_risk_path.is_file():
                try:
                    import torch

                    long_risk_labels = torch.load(
                        long_risk_path, map_location="cpu", weights_only=False
                    )
                    candidate_keys = [
                        str(row.get("candidate_key"))
                        for row in long_risk_labels
                        if isinstance(row, Mapping) and row.get("candidate_key")
                    ]
                    checks[f"{key}_long_risk_nonempty"] = bool(candidate_keys)
                    checks[f"{key}_long_risk_unique_keys"] = (
                        len(candidate_keys) == len(set(candidate_keys))
                    )
                    checks[f"{key}_long_risk_no_rNone"] = all(
                        "_rNone_" not in value for value in candidate_keys
                    )
                    checks[f"{key}_long_risk_has_no_assign"] = any(
                        "_rNO_ASSIGN_" in value for value in candidate_keys
                    )
                    checks[f"{key}_long_risk_contract"] = all(
                        int(row.get("long_risk_horizon", -1))
                        == LONG_RISK_HORIZON
                        and int(row.get("long_risk_terminal_window", -1))
                        == LONG_RISK_TERMINAL_WINDOW
                        and row.get("continuation_policy") == "greedy"
                        and row.get("source_path_planner") == PLANNER_NAME
                        and row.get("source_path_planner_params")
                        == PLANNER_PARAMS
                        for row in long_risk_labels
                        if isinstance(row, Mapping)
                    )
                except Exception:
                    checks[f"{key}_long_risk_nonempty"] = False
                    checks[f"{key}_long_risk_unique_keys"] = False
                    checks[f"{key}_long_risk_no_rNone"] = False
                    checks[f"{key}_long_risk_has_no_assign"] = False
                    checks[f"{key}_long_risk_contract"] = False
            files.extend(
                (
                    metrics_path,
                    index_paths[0],
                    data_path,
                    gen_path,
                    meta_path,
                    long_risk_path,
                )
            )

    if stage in ("data", "complete"):
        fused = root / "fused_seed_split_v1"
        splits_path = fused / "splits.json"
        splits = _read_json(splits_path)
        quality_path = fused / "quality_report.json"
        quality = _read_json(quality_path)
        checks["split_unit"] = splits.get("split_unit") == "source_seed"
        checks["split_seeds"] = splits.get("explicit_seeds") == {
            key: [str(seed) for seed in values]
            for key, values in TRAIN_SPLIT.items()
        }
        checks["fused_long_risk_samples"] = int(
            quality.get("total_long_risk_samples", 0)
        ) > 0
        checks["fused_long_risk_groups"] = int(
            quality.get("total_long_risk_groups", 0)
        ) > 0
        for name in (
            "phase_c_round1_fused.pt",
            "splits.json",
            "manifest.json",
            "quality_report.json",
            "validated_outputs.sha256",
        ):
            path = fused / name
            checks[f"fused_{name}"] = path.is_file()
            files.append(path)

    if stage == "complete":
        model_root = root / "model_round1_v1"
        summary_path = model_root / "train_summary.json"
        train_summary = _read_json(summary_path)
        action_schema = train_summary.get("action_schema") or {}
        stage1_summary = train_summary.get("stage1") or {}
        stage2_summary = train_summary.get("stage2") or {}
        training_spec = protocol.get("training") or {}
        checks["model_action_schema"] = all((
            action_schema.get("supports_no_assign_candidate"),
            action_schema.get("complete_group_coverage"),
            action_schema.get("zero_encoding_verified"),
        ))
        try:
            import torch

            checkpoint_payload = torch.load(
                model_root / "world_model.pt",
                map_location="cpu",
                weights_only=False,
            )
            checks["model_checkpoint_action_schema"] = (
                (checkpoint_payload.get("action_schema") or {})
                == action_schema
            )
            stage1_payload = torch.load(
                PHASE_B_STAGE1_CHECKPOINT,
                map_location="cpu",
                weights_only=False,
            )
            adapted_payload = torch.load(
                PIBT_TRAINED_CHECKPOINT,
                map_location="cpu",
                weights_only=False,
            )
            stage1_state = stage1_payload.get("state_dict", stage1_payload)
            adapted_state = adapted_payload.get("state_dict", adapted_payload)
            long_risk_keys = sorted(
                key for key in stage1_state if key.startswith("long_risk_head.")
            )
            checks["model_long_risk_head_present"] = bool(long_risk_keys) and all(
                key in adapted_state for key in long_risk_keys
            )
            checks["model_long_risk_head_updated"] = bool(long_risk_keys) and any(
                not torch.equal(stage1_state[key], adapted_state[key])
                for key in long_risk_keys
                if key in adapted_state
            )
        except Exception:
            checks["model_checkpoint_action_schema"] = False
            checks["model_long_risk_head_present"] = False
            checks["model_long_risk_head_updated"] = False
        checks["model_stage1_contract"] = (
            bool(stage1_summary.get("skipped"))
            and stage1_summary.get("checkpoint")
            == training_spec.get("stage1_checkpoint")
        )
        checks["model_stage2_contract"] = all((
            int(stage2_summary.get("horizon", -1))
            == int(training_spec.get("stage2_horizon", -2)),
            int(stage2_summary.get("epochs", -1))
            == int(training_spec.get("stage2_epochs", -2)),
            float(stage2_summary.get("lr", -1.0))
            == float(training_spec.get("stage2_lr", -2.0)),
            float(stage2_summary.get("alpha_rank", -1.0))
            == float(training_spec.get("stage2_alpha_rank", -2.0)),
            int(stage2_summary.get("freeze_epochs", -1))
            == int(training_spec.get("stage2_freeze_epochs", -2)),
            stage2_summary.get("unfreeze_mode")
            == training_spec.get("stage2_unfreeze_mode"),
            int(stage2_summary.get("early_stopping_patience", -1))
            == int(training_spec.get("early_stopping_patience", -2)),
            stage2_summary.get("early_stopping_monitor")
            == training_spec.get("early_stopping_monitor"),
        ))
        checks["model_long_risk_eval"] = bool(
            stage2_summary.get("val_long_risk_eval")
        )
        checks["trained_checkpoint_location"] = (
            PIBT_TRAINED_CHECKPOINT.resolve()
            == (model_root / "best_regret_world_model.pt").resolve()
        )
        for name in (
            "world_model.pt",
            "best_rank_world_model.pt",
            "best_regret_world_model.pt",
            "train_summary.json",
            "trained_outputs.sha256",
        ):
            path = model_root / name
            checks[f"model_{name}"] = path.is_file()
            files.append(path)

        station_root = root / "station_congestion_head_region_pibt_461_470_v1"
        station_dataset = station_root / "station_congestion_latents.pt"
        station_scale = station_root / "station_congestion_scale_contract.json"
        station_summary = station_root / "dataset_summary.json"
        station_head = station_root / "linear_head_v1/best_station_congestion_head.pt"
        station_train_summary = station_root / "linear_head_v1/train_summary.json"
        for name, path in (
            ("dataset", station_dataset),
            ("scale", station_scale),
            ("summary", station_summary),
            ("head", station_head),
            ("train_summary", station_train_summary),
        ):
            checks[f"station_head_{name}_file"] = path.is_file()
            files.append(path)
        checks["station_head_pp_scale_reused"] = (
            station_scale.is_file()
            and Path(PSI_SCALE_CONTRACT).is_file()
            and sha256_file(station_scale) == sha256_file(PSI_SCALE_CONTRACT)
        )
        if station_summary.is_file():
            station_summary_payload = _read_json(station_summary)
            checks["station_head_dataset_audit"] = bool(
                (station_summary_payload.get("audit") or {}).get("passed")
            )
            checks["station_head_snapshot_provenance"] = bool(
                (station_summary_payload.get("audit") or {}).get(
                    "pibt_snapshot_provenance_verified"
                )
            )
        else:
            checks["station_head_dataset_audit"] = False
            checks["station_head_snapshot_provenance"] = False
        if station_head.is_file():
            try:
                import torch

                head_payload = torch.load(
                    station_head, map_location="cpu", weights_only=False
                )
                checks["station_head_encoder_sha"] = (
                    head_payload.get("source_encoder_checkpoint_sha256")
                    == sha256_file(PIBT_TRAINED_CHECKPOINT)
                )
                checks["station_head_scale_contract"] = (
                    head_payload.get("scale_contract_sha256")
                    == (head_payload.get("scale_contract") or {}).get(
                        "contract_sha256"
                    )
                )
                checks["station_head_formal_contract"] = bool(
                    (head_payload.get("audit") or {}).get(
                        "formal_training_contract"
                    )
                )
            except Exception:
                checks["station_head_encoder_sha"] = False
                checks["station_head_scale_contract"] = False
                checks["station_head_formal_contract"] = False

    passed = all(checks.values())
    report = {
        "schema_version": TRAINING_AUDIT_SCHEMA_VERSION,
        "stage": stage,
        "protocol_sha256": bundle["protocol_sha256"],
        "passed": passed,
        "checks": checks,
        "snapshot_runs": len(LOADS) * len(TRAIN_SEEDS),
        "dataset_runs": len(LOADS) * len(TRAIN_SEEDS),
        "trained_checkpoint": (
            PIBT_TRAINED_CHECKPOINT.as_posix() if stage == "complete" else None
        ),
        "trained_checkpoint_sha256": (
            sha256_file(PIBT_TRAINED_CHECKPOINT)
            if stage == "complete" and PIBT_TRAINED_CHECKPOINT.is_file()
            else None
        ),
    }
    validation = root / "validation"
    report_names = {
        "replay": "training_replay_audit.json",
        "data": "training_data_audit.json",
        "complete": "training_audit.json",
    }
    hash_names = {
        "replay": "validated_training_replay_outputs.sha256",
        "data": "validated_training_data_outputs.sha256",
        "complete": "validated_training_outputs.sha256",
    }
    report_path = validation / report_names[stage]
    _write_json_exact(report_path, report)
    if not passed:
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError("PIBT training audit failed: " + ", ".join(failed))
    files.append(report_path)
    _write_text_exact(
        validation / hash_names[stage],
        "".join(
            f"{sha256_file(path)}  {path.as_posix()}\n"
            for path in sorted(set(files), key=lambda item: item.as_posix())
        ),
    )
    print(f"[complete] {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "prepare-manifest",
            "audit-manifests",
            "freeze-evaluation",
            "freeze-training",
            "arm",
            "summary",
            "audit-training",
        ),
    )
    parser.add_argument("--campaign", choices=("zero_shot", "adapted"))
    parser.add_argument("--output-root", default=str(ZERO_SHOT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint", default=str(PP_TRAINED_CHECKPOINT))
    parser.add_argument("--source-zero-shot-root", default=str(ZERO_SHOT_OUTPUT_ROOT))
    parser.add_argument("--frozen-bundle", default=None)
    parser.add_argument(
        "--high-manifest-root", default=str(manifest_root("high"))
    )
    parser.add_argument(
        "--low-mid-manifest-root", default=str(manifest_root("low"))
    )
    parser.add_argument("--manifest-contract", default=None)
    parser.add_argument(
        "--training-stage",
        choices=("replay", "data", "complete"),
        default="complete",
    )
    parser.add_argument("--arm", choices=ALL_ARMS)
    parser.add_argument("--load", choices=LOADS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--manifest-path", default=None)
    args = parser.parse_args()

    if args.mode == "prepare-manifest":
        if None in (args.load, args.seed):
            parser.error("--mode prepare-manifest requires --load and --seed")
        _prepare_manifest(args)
        return
    if args.mode == "audit-manifests":
        _audit_manifests(args)
        return
    if args.mode == "freeze-training":
        _freeze_training(args)
        return
    if args.mode == "audit-training":
        _audit_training(args)
        return
    if args.campaign is None:
        parser.error(f"--mode {args.mode} requires --campaign")
    if args.mode == "freeze-evaluation":
        _freeze_evaluation(args)
        return
    if args.mode == "arm":
        if None in (args.arm, args.load, args.seed, args.manifest_path):
            parser.error("--mode arm requires --arm, --load, --seed, --manifest-path")
        _run_arm(args)
        return
    _summarize(args)


if __name__ == "__main__":
    main()
