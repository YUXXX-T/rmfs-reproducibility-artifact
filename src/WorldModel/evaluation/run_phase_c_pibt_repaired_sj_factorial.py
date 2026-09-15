"""Run the isolated 2x2 S1/J1 factorial with the repaired PIBT checkpoint.

The four formal cells differ only in two dispatch layers:

* S0/J0: Phase-C robot scoring and the original context order.
* S0/J1: Phase-C robot scoring with static ascending-J context ordering.
* S1/J0: corrected quantile-combo within-context conversion only.
* S1/J1: corrected quantile-combo conversion plus static ascending J.

All cells use the same PIBT-specific World Model, repaired LongRiskHead,
rebound static-J1 head, path planner, physical-only station admission, and
frozen 551--560 order manifests.  This module is intentionally separate from
the canonical PIBT study so the new factorial cannot change historical runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

from Policies.TaskAssigner import WorldModelTaskAssigner
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_context_assigner import (
    PsiDispatchContextWorldModelTaskAssigner,
)
from WorldModel.core.long_risk_schema import long_risk_runtime_contract
from WorldModel.evaluation import run_phase_c_pibt_planner_study as study
from WorldModel.evaluation.phase_c_combo_s1_j1_paper_protocol import (
    selector_config,
)
from WorldModel.evaluation.phase_c_pibt_planner_study_protocol import (
    ACTION_PATH_MODE,
    ACTION_ROUTE_ENCODING,
    EVAL_SEEDS,
    LOADS,
    LOAD_CONFIGS,
    PLANNER_NAME,
    PLANNER_PARAMS,
    TICKS,
    TOP_M,
    canonical_sha256,
)
from WorldModel.evaluation.phase_c_s1_hungarian_protocol import PHASEC_CONFIG
from WorldModel.evaluation.run_phase_c_pibt_repaired_longrisk_combo import (
    _validate_rebound_output,
    _validate_repair_output,
)
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


SCHEMA_VERSION = "phase_c_pibt_repaired_sj_factorial_run_v1"
BUNDLE_SCHEMA_VERSION = "phase_c_pibt_repaired_sj_factorial_bundle_v1"
SUMMARY_SCHEMA_VERSION = "phase_c_pibt_repaired_sj_factorial_summary_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_OUTPUT_ROOT = (
    BASE_ROOT / "phasec_pibt_repaired_longrisk_sj_factorial_551_560_v1"
)
REPAIR_ROOT = Path(
    "WorldModel/checkpoints/"
    "phaseC_wm_onpolicy_pibt_longrisk_head_repair_681_690_v1"
)
DEFAULT_CHECKPOINT = (
    REPAIR_ROOT / "training/long_risk_head_only_v1/best_long_risk_world_model.pt"
)
DEFAULT_PSI_HEAD = (
    REPAIR_ROOT / "training/station_head_rebound_v1/best_station_congestion_head.pt"
)
DEFAULT_PSI_SCALE = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_pibt_round1_v2/"
    "station_congestion_head_region_pibt_461_470_v1/"
    "station_congestion_scale_contract.json"
)
DEFAULT_MANIFEST_CONTRACT = (
    BASE_ROOT
    / "phasec_pibt_cross_planner_ppweights_551_560_v2/validation/"
    "phase_c_pibt_input_manifests.json"
)
BUNDLE_FILENAME = "phase_c_pibt_repaired_sj_factorial_frozen_protocol.json"


@dataclass(frozen=True)
class ArmSpec:
    key: str
    label: str
    use_s1: bool
    use_j1: bool


ARM_SPECS = {
    spec.key: spec
    for spec in (
        ArmSpec("s0_j0", "Phase C (S0+J0) + PIBT", False, False),
        ArmSpec("s0_j1", "J1-only (S0+J1) + PIBT", False, True),
        ArmSpec("s1_j0", "S1-only (S1+J0) + PIBT", True, False),
        ArmSpec("s1_j1", "Combo (S1+J1) + PIBT", True, True),
    )
}
ARM_KEYS = tuple(ARM_SPECS)
SEEDS = tuple(int(seed) for seed in EVAL_SEEDS)

SUMMARY_METRICS = (
    "completed_orders",
    "completed_tasks",
    "avg_task_duration",
    "avg_excess_delay",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "stall_ratio_mean",
    "congestion_events",
    "severe_events",
    "risk_rate_per_100",
    "pibt_wait_decision_ratio",
    "station_capacity_rejections",
    "energy_conv_modified_decisions",
    "energy_conv_modified_decision_rate",
    "psi_dispatch_reordered_calls",
)
PRIMARY_METRICS = (
    "completed_orders",
    "deadlock_ratio_mean",
    "congestion_events",
    "severe_events",
    "pibt_wait_decision_ratio",
)
METRIC_DIRECTIONS = {
    "completed_orders": "higher",
    "deadlock_ratio_mean": "lower",
    "congestion_events": "lower",
    "severe_events": "lower",
    "pibt_wait_decision_ratio": "lower",
}

CONTRASTS = {
    "s1_effect_at_j0": {"s1_j0": 1.0, "s0_j0": -1.0},
    "s1_effect_at_j1": {"s1_j1": 1.0, "s0_j1": -1.0},
    "j1_effect_at_s0": {"s0_j1": 1.0, "s0_j0": -1.0},
    "j1_effect_at_s1": {"s1_j1": 1.0, "s1_j0": -1.0},
    "s1_by_j1_interaction": {
        "s1_j1": 1.0,
        "s0_j1": -1.0,
        "s1_j0": -1.0,
        "s0_j0": 1.0,
    },
    "full_method_minus_phasec": {"s1_j1": 1.0, "s0_j0": -1.0},
}

SOURCE_FILES = {
    "runner": Path(__file__),
    "submission": Path(
        "WorldModel/evaluation/"
        "run_phase_c_pibt_repaired_sj_factorial_60cpu.slurm"
    ),
    "tests": Path(
        "WorldModel/tests/test_phase_c_pibt_repaired_sj_factorial.py"
    ),
    "canonical_pibt_runner": Path(
        "WorldModel/evaluation/run_phase_c_pibt_planner_study.py"
    ),
    "canonical_pibt_protocol": Path(
        "WorldModel/evaluation/phase_c_pibt_planner_study_protocol.py"
    ),
    "repaired_longrisk_runner": Path(
        "WorldModel/evaluation/run_phase_c_pibt_repaired_longrisk_combo.py"
    ),
    "combo_s1_protocol": Path(
        "WorldModel/evaluation/phase_c_combo_s1_j1_paper_protocol.py"
    ),
    "phasec_s0_protocol": Path(
        "WorldModel/evaluation/phase_c_s1_hungarian_protocol.py"
    ),
    "s1_signal_protocol": Path(
        "WorldModel/evaluation/phase_c_s1_long_risk_correction_protocol.py"
    ),
    "world_model_assigner": Path(
        "Policies/TaskAssigner/WorldModelTaskAssigner/"
        "world_model_task_assigner.py"
    ),
    "static_j1_assigner": Path(
        "Policies/TaskAssigner/WorldModelTaskAssigner/"
        "psi_dispatch_context_assigner.py"
    ),
    "pibt_planner": Path(
        "Policies/PathPlanner/PIBTPlanner/pibt_path_planner.py"
    ),
    "simulation_engine": Path("Engine/simulation_engine.py"),
    "online_evaluator": Path("WorldModel/evaluation/evaluate_online_v6.py"),
    "long_risk_schema": Path("WorldModel/core/long_risk_schema.py"),
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _bundle_path(root: Path) -> Path:
    return root / BUNDLE_FILENAME


def _output_path(root: Path, arm: str, load: str, seed: int) -> Path:
    return root / "per_arm" / arm / f"{load}_seed{seed}.json"


def _manifest_contract(path: Path) -> dict[str, Any]:
    contract = study._load_manifest_contract(path)
    if tuple(contract.get("loads") or ()) != tuple(LOADS):
        raise ValueError("manifest contract load block does not match low/mid/high")
    if tuple(int(seed) for seed in contract.get("seeds") or ()) != SEEDS:
        raise ValueError("manifest contract seed block does not match 551--560")
    return contract


def _manifest_path(contract: Mapping[str, Any], load: str, seed: int) -> Path:
    row = (contract.get("cells") or {}).get(f"{load}:seed{seed}") or {}
    path = Path(str(row.get("path") or ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _selector_config(spec: ArmSpec) -> dict[str, Any]:
    return selector_config() if spec.use_s1 else dict(PHASEC_CONFIG)


def _policy_contract(
    spec: ArmSpec,
    *,
    checkpoint: Path,
    psi_head: Path,
    psi_scale: Path,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "arm": spec.key,
        "label": spec.label,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "path_planner": PLANNER_NAME,
        "path_planner_params": dict(PLANNER_PARAMS),
        "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
        "action_path_mode": ACTION_PATH_MODE,
        "action_route_encoding": ACTION_ROUTE_ENCODING,
        "robot_selector": "corrected_quantile_combo_s1" if spec.use_s1 else "phasec_s0",
        "robot_selector_scope": "within_context_only",
        "robot_selector_config": _selector_config(spec),
        "context_scheduler": "static_j1" if spec.use_j1 else "j0",
        "long_risk_consumed": bool(spec.use_s1),
    }
    if spec.use_j1:
        body.update({
            "psi_head_checkpoint": psi_head.as_posix(),
            "psi_head_checkpoint_sha256": sha256_file(psi_head),
            "psi_scale_contract": psi_scale.as_posix(),
            "psi_scale_contract_sha256": sha256_file(psi_scale),
        })
    return {**body, "fingerprint_sha256": canonical_sha256(body)}


def _make_assigner(
    spec: ArmSpec,
    *,
    checkpoint: Path,
    psi_head: Path,
    psi_scale: Path,
):
    common = {
        "checkpoint_path": str(checkpoint),
        "top_m": TOP_M,
        "action_path_mode": ACTION_PATH_MODE,
        "energy_conv_random_flip_seed": 0,
        **_selector_config(spec),
    }
    if not spec.use_j1:
        return WorldModelTaskAssigner(**common)
    return PsiDispatchContextWorldModelTaskAssigner(
        psi_head_checkpoint=str(psi_head),
        psi_scale_contract=str(psi_scale),
        psi_context_mode="j_ascending",
        psi_trace_enabled=False,
        psi_trace_max_records=0,
        allow_phasec_s0_robot_scorer=not spec.use_s1,
        **common,
    )


def _factor_metrics(spec: ArmSpec, assigner: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "factorial_robot_selector": "s1" if spec.use_s1 else "s0",
        "factorial_robot_selector_scope": "within_context_only",
        "factorial_context_scheduler": "j1" if spec.use_j1 else "j0",
        "factorial_candidate_context_mode": str(
            getattr(assigner, "candidate_context_mode", "")
        ),
        "factorial_energy_scoring_mode": str(
            getattr(assigner, "energy_scoring_mode", "")
        ),
        "factorial_long_risk_consumed": bool(spec.use_s1),
    }
    if spec.use_j1:
        metrics.update(assigner.psi_dispatch_metrics())
    else:
        metrics.update({
            "psi_dispatch_mode": "off",
            "psi_dispatch_head_loaded": False,
            "psi_dispatch_eval_calls": 0,
            "psi_dispatch_contexts_seen": 0,
            "psi_dispatch_reordered_calls": 0,
            "psi_dispatch_robot_scorer_variant": (
                "s1_within_context" if spec.use_s1 else "phasec_s0"
            ),
            "psi_dispatch_s1_within_context": bool(spec.use_s1),
        })
    return metrics


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _run_audit(
    spec: ArmSpec,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    station_audit: Mapping[str, Any],
    *,
    checkpoint: Path,
) -> dict[str, Any]:
    last_batch = metrics.get("pibt_last_batch_audit") or {}
    planned = int(metrics.get("pibt_planned_agents", 0))
    decisions = int(metrics.get("pibt_move_decisions", 0)) + int(
        metrics.get("pibt_wait_decisions", 0)
    )
    checks: dict[str, bool] = {
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "manifest_sha": metrics.get("order_arrival_manifest_sha256")
        == manifest.get("manifest_sha256"),
        "manifest_count": int(metrics.get("order_arrival_count", -1))
        == int(manifest.get("total_orders", -2)),
        "physical_only_station_mode": station_audit.get("mode")
        == STATION_ADMISSION_PHYSICAL_ONLY,
        "station_audit": bool(station_audit.get("passed")),
        "physical_capacity_clean": int(
            station_audit.get("physical_capacity_violation_count", -1)
        )
        == 0,
        "pibt_planner": metrics.get("path_planner_name") == PLANNER_NAME,
        "pibt_batch_interface": bool(metrics.get("path_planner_batch_interface")),
        "pibt_single_step": bool(metrics.get("path_planner_single_step")),
        "pibt_strict_validation": bool(metrics.get("pibt_strict_validation")),
        "pibt_batch_called": int(metrics.get("pibt_batch_calls", 0)) > 0,
        "pibt_sequential_unused": int(metrics.get("pibt_single_calls", -1)) == 0,
        "pibt_decision_accounting": planned > 0 and decisions == planned,
        "pibt_validation_clean": int(metrics.get("pibt_validation_failures", -1))
        == 0,
        "pibt_last_batch_clean": bool(last_batch.get("passed")),
        "pibt_vertex_conflict_free": int(
            metrics.get("path_planner_engine_vertex_conflicts", -1)
        )
        == 0,
        "pibt_swap_conflict_free": int(
            metrics.get("path_planner_engine_swap_conflicts", -1)
        )
        == 0,
        "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
        "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
        "action_path_mode": int(metrics.get("action_path_mode", -1))
        == ACTION_PATH_MODE,
        "action_route_encoding": metrics.get("action_route_encoding")
        == ACTION_ROUTE_ENCODING,
        "world_model_planner_injected": bool(
            metrics.get("world_model_path_planner_injected")
        ),
        "completed_orders_present": _number(metrics.get("completed_orders"))
        is not None,
        "selector_factor": metrics.get("factorial_robot_selector")
        == ("s1" if spec.use_s1 else "s0"),
        "scheduler_factor": metrics.get("factorial_context_scheduler")
        == ("j1" if spec.use_j1 else "j0"),
        "within_context_scope": metrics.get("factorial_robot_selector_scope")
        == "within_context_only",
        "prefix_context_source": metrics.get("factorial_candidate_context_mode")
        == "prefix",
        "energy_mode": metrics.get("factorial_energy_scoring_mode")
        == ("conversion" if spec.use_s1 else "off"),
        "s1_activity": (
            int(metrics.get("energy_conv_contexts", 0)) > 0
            if spec.use_s1
            else int(metrics.get("energy_conv_contexts", 0)) == 0
        ),
        "s1_signal": (
            metrics.get("energy_drift_signal") == "combo" if spec.use_s1 else True
        ),
        "long_risk_schema": (
            metrics.get("long_risk_schema_version")
            == long_risk_runtime_contract()["schema_version"]
            if spec.use_s1
            else True
        ),
        "robot_scorer_variant": metrics.get("psi_dispatch_robot_scorer_variant")
        == ("s1_within_context" if spec.use_s1 else "phasec_s0"),
        "s1_flag": bool(metrics.get("psi_dispatch_s1_within_context"))
        == spec.use_s1,
    }
    if spec.use_j1:
        checks.update({
            "j1_mode": metrics.get("psi_dispatch_mode") == "j_ascending",
            "j1_head_loaded": bool(metrics.get("psi_dispatch_head_loaded")),
            "j1_evaluated": int(metrics.get("psi_dispatch_eval_calls", 0)) > 0,
            "j1_contexts_seen": int(metrics.get("psi_dispatch_contexts_seen", 0))
            > 0,
            "j1_encoder_contract": bool(
                metrics.get("psi_dispatch_encoder_contract_verified")
            ),
            "j1_source_encoder": metrics.get(
                "psi_dispatch_source_encoder_checkpoint_sha256"
            )
            == sha256_file(checkpoint),
            "j1_parent_scorer_preserved": metrics.get("psi_dispatch_robot_scorer")
            == "WorldModelTaskAssigner.select_robots_unmodified",
        })
    else:
        checks.update({
            "j0_mode": metrics.get("psi_dispatch_mode") == "off",
            "j0_head_absent": not bool(metrics.get("psi_dispatch_head_loaded")),
            "j0_not_evaluated": int(metrics.get("psi_dispatch_eval_calls", 0)) == 0,
            "j0_not_reordered": int(metrics.get("psi_dispatch_reordered_calls", 0))
            == 0,
        })
    return {"passed": all(checks.values()), "checks": checks}


def preflight(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    psi_head = Path(args.psi_head)
    psi_scale = Path(args.psi_scale)
    manifest_contract = Path(args.manifest_contract)
    required = list(SOURCE_FILES.values()) + [
        checkpoint,
        checkpoint.parent / "tensor_audit.json",
        checkpoint.parent / "train_summary.json",
        psi_head,
        psi_head.parent / "rebind_summary.json",
        psi_scale,
        manifest_contract,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing PIBT factorial prerequisites:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    _validate_repair_output(checkpoint, require_planner_specific=True)
    _validate_rebound_output(psi_head)
    study._validate_psi_artifacts(
        psi_head, psi_scale, checkpoint, require_formal=True
    )
    contract = _manifest_contract(manifest_contract)
    for load, seed in product(LOADS, SEEDS):
        _manifest_path(contract, load, seed)
    print("PIBT repaired S/J factorial preflight PASS")


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    psi_head = Path(args.psi_head)
    psi_scale = Path(args.psi_scale)
    manifest_path = Path(args.manifest_contract)
    manifest = _manifest_contract(manifest_path)
    arms = {
        key: _policy_contract(
            spec,
            checkpoint=checkpoint,
            psi_head=psi_head,
            psi_scale=psi_scale,
        )
        for key, spec in ARM_SPECS.items()
    }
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "purpose": "paired 2x2 S1/J1 component-generalization factorial under PIBT",
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "ticks": TICKS,
        "new_simulation_count": len(ARM_KEYS) * len(LOADS) * len(SEEDS),
        "planner": {"name": PLANNER_NAME, "params": dict(PLANNER_PARAMS)},
        "station_admission": {
            "mode": STATION_ADMISSION_PHYSICAL_ONLY,
            "physical_occupancy_cap": True,
            "in_transit_committed_cap": False,
        },
        "factors": {
            "S": {
                "S0": "Phase-C within-context robot scorer",
                "S1": "corrected quantile-combo within-context conversion",
            },
            "J": {
                "J0": "original prefix context order",
                "J1": "static ascending-J order once per proposal batch",
            },
        },
        "arms": arms,
        "contrasts": CONTRASTS,
        "long_risk_runtime_contract": long_risk_runtime_contract(),
        "manifest_contract": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "contract_sha256": manifest.get("contract_sha256"),
        },
        "forbidden": {
            "post_run_tuning": True,
            "mixed_manifests": True,
            "committed_or_eta_admission": True,
            "dynamic_j": True,
            "station_feedback": True,
            "greedy_fallback": True,
        },
    }


def freeze(args: argparse.Namespace) -> None:
    preflight(args)
    root = Path(args.output_root)
    protocol = _protocol(args)
    artifacts = dict(SOURCE_FILES)
    artifacts.update({
        "world_model_checkpoint": Path(args.checkpoint),
        "long_risk_tensor_audit": Path(args.checkpoint).parent
        / "tensor_audit.json",
        "long_risk_train_summary": Path(args.checkpoint).parent
        / "train_summary.json",
        "psi_head_checkpoint": Path(args.psi_head),
        "psi_rebind_summary": Path(args.psi_head).parent / "rebind_summary.json",
        "psi_scale_contract": Path(args.psi_scale),
        "input_manifest_contract": Path(args.manifest_contract),
    })
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "artifacts": study._artifact_snapshot(artifacts),
    }
    root.mkdir(parents=True, exist_ok=True)
    study._write_json_exact(_bundle_path(root), bundle)
    print(f"[freeze] {_bundle_path(root)}")


def _verify_bundle(path: Path) -> dict[str, Any]:
    bundle = _read_json(path)
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"wrong factorial bundle schema: {path}")
    protocol = bundle.get("protocol") or {}
    if bundle.get("protocol_sha256") != canonical_sha256(protocol):
        raise ValueError(f"factorial protocol hash mismatch: {path}")
    for name, row in (bundle.get("artifacts") or {}).items():
        artifact = Path(str(row.get("path") or ""))
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        if sha256_file(artifact) != row.get("sha256"):
            raise ValueError(f"factorial artifact hash mismatch ({name})")
    manifest_row = protocol.get("manifest_contract") or {}
    manifest_path = Path(str(manifest_row.get("path") or ""))
    contract = _manifest_contract(manifest_path)
    if manifest_row.get("file_sha256") != sha256_file(manifest_path):
        raise ValueError("factorial manifest-contract file hash mismatch")
    if manifest_row.get("contract_sha256") != contract.get("contract_sha256"):
        raise ValueError("factorial manifest-contract semantic hash mismatch")
    return bundle


def _resume_ok(
    path: Path,
    *,
    bundle: Mapping[str, Any],
    spec: ArmSpec,
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
        payload.get("schema_version") == SCHEMA_VERSION,
        meta.get("protocol_sha256") == bundle.get("protocol_sha256"),
        meta.get("arm_key") == spec.key,
        meta.get("load") == load,
        int(meta.get("seed", -1)) == seed,
        int(meta.get("ticks", -1)) == TICKS,
        (payload.get("manifest") or {}).get("content_sha256") == manifest_sha,
        (meta.get("policy_contract") or {}).get("fingerprint_sha256")
        == policy_sha,
        bool((payload.get("audit") or {}).get("passed")),
    ))


def run_arm(args: argparse.Namespace) -> None:
    if args.arm not in ARM_SPECS:
        raise ValueError(args.arm)
    if args.load not in LOADS or int(args.seed) not in SEEDS:
        raise ValueError("load/seed is outside the frozen factorial block")
    if int(args.ticks) != TICKS:
        raise ValueError(f"formal factorial freezes ticks={TICKS}")
    root = Path(args.output_root)
    bundle_path = _bundle_path(root)
    bundle = _verify_bundle(bundle_path)
    contract_path = Path(
        str((bundle["protocol"].get("manifest_contract") or {}).get("path"))
    )
    contract = _manifest_contract(contract_path)
    manifest_path = _manifest_path(contract, args.load, int(args.seed))
    manifest = study._validate_manifest(manifest_path)
    cell = (contract.get("cells") or {}).get(
        f"{args.load}:seed{int(args.seed)}"
    ) or {}
    if not all((
        sha256_file(manifest_path) == cell.get("file_sha256"),
        manifest.get("manifest_sha256") == cell.get("content_sha256"),
        int(manifest.get("total_orders", -1)) == int(cell.get("total_orders", -2)),
    )):
        raise ValueError("factorial run manifest is outside the frozen contract")

    checkpoint = Path(args.checkpoint)
    psi_head = Path(args.psi_head)
    psi_scale = Path(args.psi_scale)
    spec = ARM_SPECS[args.arm]
    policy = _policy_contract(
        spec, checkpoint=checkpoint, psi_head=psi_head, psi_scale=psi_scale
    )
    frozen_policy = (bundle["protocol"].get("arms") or {}).get(spec.key) or {}
    if policy.get("fingerprint_sha256") != frozen_policy.get("fingerprint_sha256"):
        raise ValueError("runtime policy differs from the frozen factorial policy")
    output = _output_path(root, spec.key, args.load, int(args.seed))
    if _resume_ok(
        output,
        bundle=bundle,
        spec=spec,
        load=args.load,
        seed=int(args.seed),
        manifest_sha=str(manifest.get("manifest_sha256")),
        policy_sha=str(policy.get("fingerprint_sha256")),
    ):
        print(f"[skip] {output}")
        return

    assigner = _make_assigner(
        spec, checkpoint=checkpoint, psi_head=psi_head, psi_scale=psi_scale
    )
    print(f"[run] arm={spec.key} load={args.load} seed={args.seed}")
    with study._physical_only_audit_contract() as holder:
        metrics = study._run_one_assigner(
            LOAD_CONFIGS[args.load],
            assigner,
            int(args.seed),
            int(args.ticks),
            trace_label=spec.label,
            recorded_orders_path=str(manifest_path),
            path_planner_override=PLANNER_NAME,
            path_planner_params_override=dict(PLANNER_PARAMS),
        )
    metrics.update(_factor_metrics(spec, assigner))
    planned = int(metrics.get("pibt_planned_agents", 0))
    metrics["pibt_wait_decision_ratio"] = round(
        int(metrics.get("pibt_wait_decisions", 0)) / max(planned, 1), 8
    )
    metrics["pibt_priority_inheritance_per_agent"] = round(
        int(metrics.get("pibt_priority_inheritance_calls", 0)) / max(planned, 1),
        8,
    )
    metrics["pibt_backtracks_per_agent"] = round(
        int(metrics.get("pibt_backtracks", 0)) / max(planned, 1), 8
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
        "station_max_occupancy": study._max_mapping(
            station_audit.get("max_occupancy")
        ),
        "station_max_committed_load": study._max_mapping(
            station_audit.get("max_committed_load")
        ),
    })
    audit = _run_audit(
        spec, metrics, manifest, station_audit, checkpoint=checkpoint
    )
    if not audit["passed"]:
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise RuntimeError("PIBT factorial run audit failed: " + ", ".join(failed))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": bundle["protocol_sha256"],
            "frozen_bundle": bundle_path.as_posix(),
            "frozen_bundle_sha256": sha256_file(bundle_path),
            "arm_key": spec.key,
            "arm": spec.label,
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "base_config": LOAD_CONFIGS[args.load],
            "policy_contract": policy,
            "long_risk_runtime_contract": long_risk_runtime_contract(),
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
            "contract_sha256": contract.get("contract_sha256"),
        },
        "station_audit": station_audit,
        "audit": audit,
        "metrics": metrics,
    }
    study._write_json_exact(output, payload)
    print(f"[done] {output}")


def _mean_std(values: Iterable[float]) -> dict[str, Any]:
    clean = [float(value) for value in values]
    if not clean:
        return {"n": 0, "mean": None, "std": None}
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / max(
        len(clean) - 1, 1
    )
    return {"n": len(clean), "mean": mean, "std": math.sqrt(variance)}


def _exact_sign_flip_p(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values]
    if not clean:
        return None
    observed = abs(sum(clean) / len(clean))
    extreme = 0
    total = 1 << len(clean)
    for mask in range(total):
        mean = sum(
            value if mask & (1 << index) else -value
            for index, value in enumerate(clean)
        ) / len(clean)
        if abs(mean) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def _contrast_values(
    rows: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
    *,
    coefficients: Mapping[str, float],
    metric: str,
    load: str,
) -> list[float]:
    values = []
    for seed in SEEDS:
        terms = []
        for arm, coefficient in coefficients.items():
            value = _number(rows[(load, seed)][arm].get(metric))
            if value is None:
                break
            terms.append(float(coefficient) * value)
        else:
            values.append(sum(terms))
    return values


def _contrast_summary(
    values: list[float], *, metric: str, bootstrap_seed: int
) -> dict[str, Any]:
    result = _mean_std(values)
    low, high = study._bootstrap_ci(values, bootstrap_seed)
    direction = METRIC_DIRECTIONS[metric]
    wins = ties = losses = 0
    for value in values:
        signed = value if direction == "higher" else -value
        wins += int(signed > 1e-12)
        ties += int(abs(signed) <= 1e-12)
        losses += int(signed < -1e-12)
    result.update({
        "ci95_low": low,
        "ci95_high": high,
        "exact_sign_flip_p_two_sided": _exact_sign_flip_p(values),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "direction": direction,
    })
    return result


def _read_run(
    path: Path,
    *,
    bundle: Mapping[str, Any],
    contract: Mapping[str, Any],
    arm: str,
    load: str,
    seed: int,
) -> dict[str, Any]:
    payload = _read_json(path)
    meta = payload.get("meta") or {}
    manifest = payload.get("manifest") or {}
    cell = (contract.get("cells") or {}).get(f"{load}:seed{seed}") or {}
    checks = {
        "schema": payload.get("schema_version") == SCHEMA_VERSION,
        "protocol": meta.get("protocol_sha256") == bundle.get("protocol_sha256"),
        "arm": meta.get("arm_key") == arm,
        "load": meta.get("load") == load,
        "seed": int(meta.get("seed", -1)) == seed,
        "ticks": int(meta.get("ticks", -1)) == TICKS,
        "manifest": manifest.get("content_sha256") == cell.get("content_sha256"),
        "audit": bool((payload.get("audit") or {}).get("passed")),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"invalid factorial run {path}: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )
    return payload


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def summarize(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    bundle = _verify_bundle(_bundle_path(root))
    contract_path = Path(
        str((bundle["protocol"].get("manifest_contract") or {}).get("path"))
    )
    contract = _manifest_contract(contract_path)
    rows: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    input_files: list[Path] = []
    per_seed_rows: list[dict[str, Any]] = []
    for load, seed in product(LOADS, SEEDS):
        rows[(load, seed)] = {}
        for arm in ARM_KEYS:
            path = _output_path(root, arm, load, seed)
            payload = _read_run(
                path,
                bundle=bundle,
                contract=contract,
                arm=arm,
                load=load,
                seed=seed,
            )
            input_files.append(path)
            metrics = payload.get("metrics") or {}
            rows[(load, seed)][arm] = metrics
            per_seed_rows.append({
                "arm": arm,
                "load": load,
                "seed": seed,
                **{metric: metrics.get(metric) for metric in SUMMARY_METRICS},
            })

    aggregate: list[dict[str, Any]] = []
    for arm, load, metric in product(ARM_KEYS, LOADS, SUMMARY_METRICS):
        values = [
            value
            for seed in SEEDS
            if (value := _number(rows[(load, seed)][arm].get(metric))) is not None
        ]
        aggregate.append({
            "arm": arm,
            "load": load,
            "metric": metric,
            **_mean_std(values),
        })

    contrast_json: dict[str, Any] = {}
    contrast_rows: list[dict[str, Any]] = []
    for contrast_index, (name, coefficients) in enumerate(CONTRASTS.items()):
        contrast_json[name] = {"coefficients": coefficients, "metrics": {}}
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            metric_report: dict[str, Any] = {}
            clustered: list[float] = []
            for seed in SEEDS:
                seed_values = []
                for load in LOADS:
                    terms = []
                    for arm, coefficient in coefficients.items():
                        value = _number(rows[(load, seed)][arm].get(metric))
                        if value is None:
                            break
                        terms.append(float(coefficient) * value)
                    else:
                        seed_values.append(sum(terms))
                if len(seed_values) == len(LOADS):
                    clustered.append(sum(seed_values) / len(seed_values))
            for load_index, load in enumerate(LOADS):
                values = _contrast_values(
                    rows,
                    coefficients=coefficients,
                    metric=metric,
                    load=load,
                )
                report = _contrast_summary(
                    values,
                    metric=metric,
                    bootstrap_seed=(
                        20260820
                        + contrast_index * 1000
                        + metric_index * 20
                        + load_index
                    ),
                )
                metric_report[load] = report
                contrast_rows.append({
                    "contrast": name,
                    "metric": metric,
                    "scope": load,
                    **report,
                })
            overall = _contrast_summary(
                clustered,
                metric=metric,
                bootstrap_seed=(
                    20260820 + contrast_index * 1000 + metric_index * 20 + 10
                ),
            )
            metric_report["overall_seed_cluster"] = overall
            contrast_rows.append({
                "contrast": name,
                "metric": metric,
                "scope": "overall_seed_cluster",
                **overall,
            })
            contrast_json[name]["metrics"][metric] = metric_report

    validation = root / "validation"
    aggregate_path = validation / "aggregate_summary.csv"
    contrast_path = validation / "factorial_contrasts.csv"
    per_seed_path = validation / "per_seed_metrics.csv"
    summary_path = validation / "summary.json"
    hashes_path = validation / "validated_outputs.sha256"
    validation.mkdir(parents=True, exist_ok=True)
    study._write_text_exact(
        aggregate_path,
        _csv_text(aggregate, ["arm", "load", "metric", "n", "mean", "std"]),
    )
    study._write_text_exact(
        contrast_path,
        _csv_text(
            contrast_rows,
            [
                "contrast",
                "metric",
                "scope",
                "n",
                "mean",
                "std",
                "ci95_low",
                "ci95_high",
                "exact_sign_flip_p_two_sided",
                "wins",
                "ties",
                "losses",
                "direction",
            ],
        ),
    )
    study._write_text_exact(
        per_seed_path,
        _csv_text(
            per_seed_rows,
            ["arm", "load", "seed", *SUMMARY_METRICS],
        ),
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "purpose": "estimate S1, J1, and S1-by-J1 effects under PIBT",
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "ticks": TICKS,
        "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "psi_head_sha256": sha256_file(Path(args.psi_head)),
        "manifest_contract_sha256": contract.get("contract_sha256"),
        "arms": {key: spec.label for key, spec in ARM_SPECS.items()},
        "aggregate": aggregate,
        "contrasts": contrast_json,
        "integrity": {
            "expected_runs": len(ARM_KEYS) * len(LOADS) * len(SEEDS),
            "validated_runs": len(input_files),
            "all_run_audits_passed": True,
            "same_checkpoint_all_arms": True,
            "exact_manifest_pairing": True,
        },
    }
    study._write_json_exact(summary_path, summary)
    validated = sorted(
        input_files
        + [
            _bundle_path(root),
            aggregate_path,
            contrast_path,
            per_seed_path,
            summary_path,
        ],
        key=lambda path: path.as_posix(),
    )
    study._write_text_exact(
        hashes_path,
        "".join(f"{sha256_file(path)}  {path.as_posix()}\n" for path in validated),
    )
    print(f"[complete] {summary_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=("preflight", "freeze", "run-arm", "summary")
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--psi-head", default=str(DEFAULT_PSI_HEAD))
    parser.add_argument("--psi-scale", default=str(DEFAULT_PSI_SCALE))
    parser.add_argument(
        "--manifest-contract", default=str(DEFAULT_MANIFEST_CONTRACT)
    )
    parser.add_argument("--arm", choices=ARM_KEYS)
    parser.add_argument("--load", choices=LOADS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int, default=TICKS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.mode == "preflight":
        preflight(args)
    elif args.mode == "freeze":
        freeze(args)
    elif args.mode == "run-arm":
        if args.arm is None or args.load is None or args.seed is None:
            raise SystemExit("--mode run-arm requires --arm, --load, and --seed")
        run_arm(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()


__all__ = [
    "ARM_KEYS",
    "ARM_SPECS",
    "CONTRASTS",
    "_contrast_summary",
    "_make_assigner",
    "_policy_contract",
    "_run_audit",
]
