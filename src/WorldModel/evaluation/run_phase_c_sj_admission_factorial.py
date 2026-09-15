"""Run the paired Phase-C S/J x station-admission factorial.

The experiment separates two World Model policy factors that were conflated in
several earlier comparisons:

* S0/S1 controls robot selection inside one fixed assignment context.
* J0/J1/Dynamic-J controls ordering across assignment contexts.

Every policy cell is replayed against the same frozen order manifest under
physical-only, committed V1, ETA V3, and FIFO V2 station admission.  Greedy
and Hungarian baselines are replayed under the same four admission modes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from Policies.TaskAssigner import (
    GreedyTaskAssigner,
    HungarianTaskAssigner,
    WorldModelTaskAssigner,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_context_assigner import (
    PsiDispatchContextWorldModelTaskAssigner,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_dynamic_probe_assigner import (
    DynamicPsiDispatchProbeAssigner,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    LOADS,
    S1_CONFIG,
    TOP_M,
    canonical_sha256,
    sha256_file,
)
from WorldModel.evaluation.phase_c_s1_hungarian_protocol import PHASEC_CONFIG
from WorldModel.evaluation.run_phase_c_psi_dynamic_admission import (
    _atomic_json,
    _read_json,
)
from WorldModel.evaluation.run_phase_c_station_admission_restoration import (
    DEFAULT_BUNDLE,
    DEFAULT_SOURCE_ROOT,
    DEFAULT_V1_REFERENCE_ROOT,
    DEFAULT_V3_REFERENCE_ROOT,
    FORMAL_SEEDS,
    FORMAL_TICKS,
    StationAdmissionRestorationAuditProbe,
    _aggregate_arm,
    _dynamic_config,
    _engine_contract,
    _load_run_inputs,
    _manifest_path,
    _max_mapping,
    _validate_historical_contract,
    _validate_manifest,
)
from WorldState.station_state import (
    STATION_ADMISSION_COMMITTED_FIFO_V2,
    STATION_ADMISSION_COMMITTED_V1,
    STATION_ADMISSION_DYNAMIC_ETA_V1,
    STATION_ADMISSION_PHYSICAL_ONLY,
)


SCHEMA_VERSION = "phase_c_sj_admission_factorial_arm_v1"
SUMMARY_SCHEMA_VERSION = "phase_c_sj_admission_factorial_summary_v1"
FACTORIAL_CONTRACT_VERSION = "phase_c_sj_admission_factorial_contract_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "station_admission_sj_factorial_551_560_v1"

ROBOT_SELECTORS = ("s0", "s1")
CONTEXT_SCHEDULERS = ("j0", "j1", "dynamic_j")
ADMISSION_KEYS = (
    "physical_only",
    "committed_v1",
    "eta_v3",
    "fifo_v2",
)
ADMISSION_MODES = {
    "physical_only": STATION_ADMISSION_PHYSICAL_ONLY,
    "committed_v1": STATION_ADMISSION_COMMITTED_V1,
    "eta_v3": STATION_ADMISSION_DYNAMIC_ETA_V1,
    "fifo_v2": STATION_ADMISSION_COMMITTED_FIFO_V2,
}
ADMISSION_LABELS = {
    "physical_only": "physical-only (unbounded in-transit commitments)",
    "committed_v1": "committed V1 (committed cap = physical capacity)",
    "eta_v3": "ETA V3 (dynamic overbooking)",
    "fifo_v2": "FIFO V2 (committed cap with waiting lifecycle)",
}


@dataclass(frozen=True)
class ArmSpec:
    key: str
    label: str
    policy_key: str
    policy_family: str
    robot_selector: str | None
    context_scheduler: str | None
    admission_key: str
    admission_mode: str


def _build_arm_specs() -> dict[str, ArmSpec]:
    specs: list[ArmSpec] = []
    for selector in ROBOT_SELECTORS:
        for scheduler in CONTEXT_SCHEDULERS:
            policy_key = f"{selector}_{scheduler}"
            for admission_key in ADMISSION_KEYS:
                specs.append(ArmSpec(
                    key=f"{policy_key}_{admission_key}",
                    label=(
                        f"{selector.upper()} + {scheduler.replace('_', ' ').upper()}"
                        f" + {ADMISSION_LABELS[admission_key]}"
                    ),
                    policy_key=policy_key,
                    policy_family="world_model",
                    robot_selector=selector,
                    context_scheduler=scheduler,
                    admission_key=admission_key,
                    admission_mode=ADMISSION_MODES[admission_key],
                ))
    for baseline in ("greedy", "hungarian"):
        for admission_key in ADMISSION_KEYS:
            specs.append(ArmSpec(
                key=f"{baseline}_{admission_key}",
                label=f"{baseline.title()} + {ADMISSION_LABELS[admission_key]}",
                policy_key=baseline,
                policy_family=baseline,
                robot_selector=None,
                context_scheduler=None,
                admission_key=admission_key,
                admission_mode=ADMISSION_MODES[admission_key],
            ))
    return {spec.key: spec for spec in specs}


ARM_SPECS = _build_arm_specs()
ARM_KEYS = tuple(ARM_SPECS)
POLICY_KEYS = tuple(
    [
        f"{selector}_{scheduler}"
        for selector in ROBOT_SELECTORS
        for scheduler in CONTEXT_SCHEDULERS
    ]
    + ["greedy", "hungarian"]
)


def _factorial_contract() -> dict[str, Any]:
    contract = {
        "schema_version": FACTORIAL_CONTRACT_VERSION,
        "loads": list(LOADS),
        "formal_seeds": list(FORMAL_SEEDS),
        "formal_ticks": FORMAL_TICKS,
        "robot_selector_factor": {
            "s0": "Phase-C scorer; energy_scoring_mode=off",
            "s1": (
                "frozen within-context conversion; it selects a robot only "
                "inside the currently supplied context"
            ),
        },
        "context_scheduler_factor": {
            "j0": "original candidate_context_mode=prefix; no psi reorder",
            "j1": "one static ascending-J context order per proposal batch",
            "dynamic_j": (
                "interleaved ascending-J selection with virtual pending and "
                "idle-robot updates after each selected context"
            ),
        },
        "station_admission_factor": {
            key: {
                "mode": ADMISSION_MODES[key],
                "description": ADMISSION_LABELS[key],
            }
            for key in ADMISSION_KEYS
        },
        "world_model_policy_cells": [
            f"{selector}_{scheduler}"
            for selector in ROBOT_SELECTORS
            for scheduler in CONTEXT_SCHEDULERS
        ],
        "external_baselines": ["greedy", "hungarian"],
        "arm_count": len(ARM_KEYS),
        "arm_keys": list(ARM_KEYS),
        "manifest_contract": (
            "all arms for one load/seed replay the identical frozen manifest"
        ),
        "s1_config": dict(S1_CONFIG),
        "s0_config": dict(PHASEC_CONFIG),
    }
    return {**contract, "contract_sha256": canonical_sha256(contract)}


FACTORIAL_CONTRACT = _factorial_contract()


def _output_path(root: Path, arm: str, load: str, seed: int) -> Path:
    return root / "per_arm" / arm / f"{load}_seed{seed}.json"


def _runtime_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "world_model_assigner": Path(
            "Policies/TaskAssigner/WorldModelTaskAssigner/"
            "world_model_task_assigner.py"
        ),
        "static_j_assigner": Path(
            "Policies/TaskAssigner/WorldModelTaskAssigner/"
            "psi_dispatch_context_assigner.py"
        ),
        "dynamic_j_assigner": Path(
            "Policies/TaskAssigner/WorldModelTaskAssigner/"
            "psi_dispatch_dynamic_probe_assigner.py"
        ),
        "greedy_assigner": Path(
            "Policies/TaskAssigner/GreedyTaskAssigner/greedy_task_assigner.py"
        ),
        "hungarian_assigner": Path(
            "Policies/TaskAssigner/HungarianTaskAssigner/"
            "hungarian_task_assigner.py"
        ),
        "station_state": Path("WorldState/station_state.py"),
        "simulation_engine": Path("Engine/simulation_engine.py"),
        "evaluate_online": Path(
            "WorldModel/evaluation/evaluate_online_v6.py"
        ),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _selector_config(selector: str) -> dict[str, Any]:
    if selector == "s0":
        return dict(PHASEC_CONFIG)
    if selector == "s1":
        return dict(S1_CONFIG)
    raise ValueError(selector)


def _policy_contract(
    spec: ArmSpec,
    *,
    model_checkpoint: Path,
    psi_head_checkpoint: Path,
    psi_scale_contract: Path,
    runtime_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if spec.policy_family in ("greedy", "hungarian"):
        code_key = f"{spec.policy_family}_assigner"
        contract = {
            "policy_key": spec.policy_key,
            "policy": (
                "GreedyTaskAssigner"
                if spec.policy_family == "greedy"
                else "HungarianTaskAssigner"
            ),
            "policy_code_sha256": runtime_hashes[code_key],
        }
    else:
        assert spec.robot_selector is not None
        assert spec.context_scheduler is not None
        scheduler = spec.context_scheduler
        contract = {
            "policy_key": spec.policy_key,
            "policy": {
                "j0": "WorldModelTaskAssigner",
                "j1": "PsiDispatchContextWorldModelTaskAssigner",
                "dynamic_j": "DynamicPsiDispatchProbeAssigner",
            }[scheduler],
            "checkpoint": model_checkpoint.as_posix(),
            "checkpoint_sha256": sha256_file(model_checkpoint),
            "top_m": int(TOP_M),
            "robot_selector": spec.robot_selector,
            "robot_selector_scope": "within_context_only",
            "robot_selector_config": _selector_config(spec.robot_selector),
            "energy_conv_random_flip_seed": 0,
            "context_scheduler": scheduler,
            "candidate_context_mode": "prefix",
            "world_model_assigner_sha256": runtime_hashes[
                "world_model_assigner"
            ],
        }
        if scheduler in ("j1", "dynamic_j"):
            contract.update({
                "psi_head_checkpoint": psi_head_checkpoint.as_posix(),
                "psi_head_checkpoint_sha256": sha256_file(
                    psi_head_checkpoint
                ),
                "psi_scale_contract": psi_scale_contract.as_posix(),
                "psi_scale_contract_sha256": sha256_file(
                    psi_scale_contract
                ),
                "psi_dispatch_code_sha256": runtime_hashes[
                    "static_j_assigner"
                ],
            })
        if scheduler == "j1":
            contract["psi_context_mode"] = "j_ascending"
            contract["j_refresh"] = "once_per_proposal_batch"
        if scheduler == "dynamic_j":
            contract.update({
                "dynamic_interleaved": True,
                "j_refresh": "after_each_virtual_selection",
                "dynamic_j_code_sha256": runtime_hashes[
                    "dynamic_j_assigner"
                ],
            })
    return {**contract, "fingerprint_sha256": canonical_sha256(contract)}


def _make_assigner(
    spec: ArmSpec,
    *,
    model_checkpoint: Path,
    psi_head_checkpoint: Path,
    psi_scale_contract: Path,
    trace_max_records: int,
):
    if spec.policy_family == "greedy":
        return GreedyTaskAssigner()
    if spec.policy_family == "hungarian":
        return HungarianTaskAssigner()

    assert spec.robot_selector is not None
    assert spec.context_scheduler is not None
    config = _selector_config(spec.robot_selector)
    common = {
        "checkpoint_path": str(model_checkpoint),
        "top_m": TOP_M,
        "energy_conv_random_flip_seed": 0,
        **config,
    }
    if spec.context_scheduler == "j0":
        return WorldModelTaskAssigner(**common)
    psi_common = {
        "psi_head_checkpoint": str(psi_head_checkpoint),
        "psi_scale_contract": str(psi_scale_contract),
        "allow_phasec_s0_robot_scorer": spec.robot_selector == "s0",
        **common,
    }
    if spec.context_scheduler == "j1":
        return PsiDispatchContextWorldModelTaskAssigner(
            psi_context_mode="j_ascending",
            psi_trace_enabled=int(trace_max_records) > 0,
            psi_trace_max_records=int(trace_max_records),
            **psi_common,
        )
    if spec.context_scheduler == "dynamic_j":
        return DynamicPsiDispatchProbeAssigner(
            dynamic_trace_enabled=int(trace_max_records) > 0,
            dynamic_trace_max_records=int(trace_max_records),
            dynamic_interleaved=True,
            **psi_common,
        )
    raise ValueError(spec.context_scheduler)


def _factor_metrics(spec: ArmSpec, assigner) -> dict[str, Any]:
    if spec.policy_family != "world_model":
        return {
            "factorial_policy_key": spec.policy_key,
            "factorial_robot_selector": None,
            "factorial_context_scheduler": None,
        }

    assert spec.robot_selector is not None
    assert spec.context_scheduler is not None
    result = {
        "factorial_policy_key": spec.policy_key,
        "factorial_robot_selector": spec.robot_selector,
        "factorial_robot_selector_scope": "within_context_only",
        "factorial_context_scheduler": spec.context_scheduler,
        "factorial_candidate_context_mode": str(
            getattr(assigner, "candidate_context_mode", "")
        ),
        "factorial_energy_scoring_mode": str(
            getattr(assigner, "energy_scoring_mode", "")
        ),
    }
    if spec.context_scheduler == "j1":
        result.update(assigner.psi_dispatch_metrics())
    elif spec.context_scheduler == "dynamic_j":
        result.update(assigner.dynamic_probe_metrics())
    else:
        result.update({
            "psi_dispatch_mode": "off",
            "psi_dispatch_head_loaded": False,
            "psi_dispatch_eval_calls": 0,
            "psi_dispatch_contexts_seen": 0,
            "psi_dispatch_reordered_calls": 0,
            "psi_dispatch_robot_scorer_variant": (
                "phasec_s0"
                if spec.robot_selector == "s0"
                else "s1_within_context"
            ),
            "psi_dispatch_s1_within_context": (
                spec.robot_selector == "s1"
            ),
        })
    return result


def _policy_audit(
    spec: ArmSpec,
    metrics: Mapping[str, Any],
) -> dict[str, bool]:
    model_calls = int(metrics.get("model_assign_calls") or 0)
    s1_contexts = int(metrics.get("energy_conv_contexts") or 0)
    fallback_calls = int(metrics.get("fallback_greedy_calls") or 0)
    if spec.policy_family != "world_model":
        return {
            "external_baseline_has_no_model_calls": model_calls == 0,
            "external_baseline_has_no_s1_contexts": s1_contexts == 0,
            "external_baseline_has_no_greedy_fallback": fallback_calls == 0,
        }

    assert spec.robot_selector is not None
    assert spec.context_scheduler is not None
    selector_variant = (
        "phasec_s0"
        if spec.robot_selector == "s0"
        else "s1_within_context"
    )
    checks = {
        "model_used": model_calls > 0,
        "no_greedy_fallback": fallback_calls == 0,
        "native_no_assign_disabled": not bool(
            metrics.get("native_no_assign_enabled", False)
        ),
        "candidate_context_mode_is_prefix": (
            metrics.get("factorial_candidate_context_mode") == "prefix"
        ),
        "robot_selector_label": (
            metrics.get("factorial_robot_selector") == spec.robot_selector
        ),
        "robot_selector_scope_is_within_context": (
            metrics.get("factorial_robot_selector_scope")
            == "within_context_only"
        ),
        "context_scheduler_label": (
            metrics.get("factorial_context_scheduler")
            == spec.context_scheduler
        ),
        "energy_mode_matches_selector": (
            metrics.get("factorial_energy_scoring_mode")
            == ("off" if spec.robot_selector == "s0" else "conversion")
        ),
        "s1_activity_matches_selector": (
            s1_contexts == 0
            if spec.robot_selector == "s0"
            else s1_contexts > 0
        ),
        "robot_scorer_variant_matches": (
            metrics.get("psi_dispatch_robot_scorer_variant")
            == selector_variant
        ),
        "s1_within_context_flag_matches": bool(
            metrics.get("psi_dispatch_s1_within_context", False)
        ) == (spec.robot_selector == "s1"),
    }
    if spec.context_scheduler == "j0":
        checks.update({
            "j0_has_no_psi_head": not bool(
                metrics.get("psi_dispatch_head_loaded", False)
            ),
            "j0_has_no_psi_evaluation": int(
                metrics.get("psi_dispatch_eval_calls") or 0
            ) == 0,
            "j0_has_no_context_reorder": int(
                metrics.get("psi_dispatch_reordered_calls") or 0
            ) == 0,
            "j0_mode_is_off": metrics.get("psi_dispatch_mode") == "off",
        })
    elif spec.context_scheduler == "j1":
        checks.update({
            "j1_mode_is_static_ascending": (
                metrics.get("psi_dispatch_mode") == "j_ascending"
            ),
            "j1_head_loaded": bool(
                metrics.get("psi_dispatch_head_loaded", False)
            ),
            "j1_evaluated": int(
                metrics.get("psi_dispatch_eval_calls") or 0
            ) > 0,
            "j1_contexts_seen": int(
                metrics.get("psi_dispatch_contexts_seen") or 0
            ) > 0,
            "j1_parent_robot_scorer_preserved": (
                metrics.get("psi_dispatch_robot_scorer")
                == "WorldModelTaskAssigner.select_robots_unmodified"
            ),
        })
    else:
        checks.update({
            "dynamic_mode_is_interleaved": (
                metrics.get("psi_dispatch_mode")
                == "j_dynamic_interleaved_probe"
            ),
            "dynamic_head_loaded": bool(
                metrics.get("psi_dispatch_head_loaded", False)
            ),
            "dynamic_j_evaluated": int(
                metrics.get("psi_dispatch_eval_calls") or 0
            ) > 0,
            "dynamic_batches_seen": int(
                metrics.get("dynamic_probe_batches") or 0
            ) > 0,
            "dynamic_virtual_backlog_updated": int(
                metrics.get("dynamic_probe_virtual_backlog_updates") or 0
            ) > 0,
            "dynamic_choices_reindexed": bool(
                metrics.get("dynamic_probe_choices_reindexed", False)
            ),
            "dynamic_parent_robot_scorer_preserved": (
                metrics.get("psi_dispatch_robot_scorer")
                == "WorldModelTaskAssigner.select_robots_unmodified"
            ),
        })
    return checks


def _run_audit(
    spec: ArmSpec,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    station_audit: Mapping[str, Any],
    fifo_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    checks = {
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "manifest_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_count_matches": int(
            metrics.get("order_arrival_count", -1)
        ) == int(manifest.get("total_orders", -2)),
        "station_admission_audit": bool(station_audit.get("passed")),
        "station_admission_mode": (
            station_audit.get("mode") == spec.admission_mode
        ),
        "physical_capacity_never_exceeded": int(
            station_audit.get("physical_capacity_violation_count", -1)
        ) == 0,
        **_policy_audit(spec, metrics),
    }
    if spec.admission_key == "fifo_v2":
        checks.update({
            "fifo_waiting_lifecycle": bool(
                fifo_audit and fifo_audit.get("passed")
            ),
            "fifo_mode": bool(
                fifo_audit
                and fifo_audit.get("mode")
                == STATION_ADMISSION_COMMITTED_FIFO_V2
            ),
        })
    else:
        checks["fifo_waiting_not_attached"] = fifo_audit is None
    return {"passed": all(checks.values()), "checks": checks}


def _resume_compatible(
    path: Path,
    *,
    spec: ArmSpec,
    args: argparse.Namespace,
    bundle_sha: str,
    runtime_hashes: Mapping[str, str],
    policy_fingerprint: str,
    manifest_sha: str,
    dynamic_config: Mapping[str, Any],
) -> bool:
    if not path.is_file():
        return False
    payload = _read_json(path)
    meta = payload.get("meta") or {}
    checks = {
        "schema": payload.get("schema_version") == SCHEMA_VERSION,
        "factorial_contract": (
            meta.get("factorial_contract_sha256")
            == FACTORIAL_CONTRACT["contract_sha256"]
        ),
        "arm": meta.get("arm_key") == spec.key,
        "load": meta.get("load") == args.load,
        "seed": int(meta.get("seed", -1)) == int(args.seed),
        "ticks": int(meta.get("ticks", -1)) == int(args.ticks),
        "admission": meta.get("station_admission") == spec.admission_mode,
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
            meta.get("dynamic_admission_config")
            == (
                dict(dynamic_config)
                if spec.admission_key == "eta_v3"
                else None
            )
        ),
        "audit": bool((payload.get("audit") or {}).get("passed")),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"cannot resume incompatible output {path}: {failed}")
    print(f"[resume] {spec.key} {args.load} seed={args.seed}: {path}")
    return True


def _manifest_contract(args: argparse.Namespace) -> dict[str, Any]:
    path = _manifest_path(args.source_root, args.load, int(args.seed))
    if not path.is_file():
        raise FileNotFoundError(
            "factorial runs never generate orders; missing frozen manifest: "
            f"{path}"
        )
    manifest = _validate_manifest(path)
    historical = _validate_historical_contract(
        manifest,
        v1_root=args.manifest_reference_root,
        v3_root=args.v3_reference_root,
        load=args.load,
        seed=int(args.seed),
        require_v3=args.require_v3_reference,
        skip=args.skip_historical_contract,
    )
    return {
        "path": path,
        "payload": manifest,
        "historical": historical,
    }


def _validate_manifest_only(args: argparse.Namespace) -> None:
    contract = _manifest_contract(args)
    manifest = contract["payload"]
    print(json.dumps({
        "passed": True,
        "load": args.load,
        "seed": int(args.seed),
        "manifest": contract["path"].as_posix(),
        "content_sha256": manifest.get("manifest_sha256"),
        "total_orders": manifest.get("total_orders"),
        "historical_contract": contract["historical"],
    }, indent=2, ensure_ascii=False))


def _run_arm(args: argparse.Namespace) -> None:
    if args.arm is None:
        raise SystemExit("--arm is required for --mode arm")
    spec = ARM_SPECS[args.arm]
    (
        _bundle,
        source_protocol,
        model_checkpoint,
        psi_head_checkpoint,
        psi_scale_contract,
        config_path,
    ) = _load_run_inputs(args)
    manifest_contract = _manifest_contract(args)
    manifest_path = manifest_contract["path"]
    manifest = manifest_contract["payload"]
    runtime_hashes = _runtime_hashes()
    policy_contract = _policy_contract(
        spec,
        model_checkpoint=model_checkpoint,
        psi_head_checkpoint=psi_head_checkpoint,
        psi_scale_contract=psi_scale_contract,
        runtime_hashes=runtime_hashes,
    )
    dynamic_config = _dynamic_config(args)
    bundle_sha = sha256_file(args.frozen_bundle)
    output = _output_path(
        args.output_root, spec.key, args.load, int(args.seed)
    )
    if _resume_compatible(
        output,
        spec=spec,
        args=args,
        bundle_sha=bundle_sha,
        runtime_hashes=runtime_hashes,
        policy_fingerprint=policy_contract["fingerprint_sha256"],
        manifest_sha=str(manifest["manifest_sha256"]),
        dynamic_config=dynamic_config,
    ):
        return

    assigner = _make_assigner(
        spec,
        model_checkpoint=model_checkpoint,
        psi_head_checkpoint=psi_head_checkpoint,
        psi_scale_contract=psi_scale_contract,
        trace_max_records=args.policy_trace_max_records,
    )
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    print(
        f"[run] arm={spec.key} load={args.load} seed={args.seed} "
        f"ticks={args.ticks} admission={spec.admission_mode}"
    )
    with _engine_contract(
        spec.admission_mode,
        dynamic_config=dynamic_config,
        station_trace_stride=args.station_trace_stride,
        station_trace_max_records=args.station_trace_max_records,
        fifo_trace_max_records=args.fifo_trace_max_records,
    ) as holder:
        metrics = _run_one_assigner(
            str(config_path),
            assigner,
            int(args.seed),
            int(args.ticks),
            trace_label=spec.label,
            recorded_orders_path=str(manifest_path),
        )
    station_probe = holder.get("station_probe")
    if not isinstance(station_probe, StationAdmissionRestorationAuditProbe):
        raise RuntimeError("station admission audit probe was not attached")
    station_audit = station_probe.summary()
    fifo_probe = holder.get("fifo_probe")
    fifo_audit = fifo_probe.summary() if fifo_probe is not None else None

    metrics.update(_factor_metrics(spec, assigner))
    metrics.update({
        "station_admission_mode": spec.admission_mode,
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
    if fifo_audit is not None:
        metrics.update({
            "waiting_assigned_agent_ticks": fifo_audit.get(
                "waiting_agent_ticks"
            ),
            "waiting_assigned_ratio": fifo_audit.get(
                "waiting_assigned_ratio"
            ),
            "waiting_promotions": fifo_audit.get("waiting_promotions"),
            "waiting_duration_p95_ticks": fifo_audit.get(
                "waiting_duration_p95_ticks"
            ),
            "unresolved_waiter_count_final": fifo_audit.get(
                "unresolved_waiter_count_final"
            ),
        })

    audit = _run_audit(
        spec, metrics, manifest, station_audit, fifo_audit
    )
    if not audit["passed"]:
        failed = [
            key for key, passed in audit["checks"].items() if not passed
        ]
        raise RuntimeError(
            f"factorial arm audit failed {spec.key} {args.load} "
            f"seed={args.seed}: {failed}"
        )

    if spec.context_scheduler == "dynamic_j":
        policy_trace = assigner.dynamic_probe_trace_records
    elif spec.context_scheduler == "j1":
        policy_trace = assigner.psi_dispatch_trace_records
    else:
        policy_trace = []
    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "factorial_contract_sha256": FACTORIAL_CONTRACT[
                "contract_sha256"
            ],
            "source_protocol_sha256": source_protocol.get(
                "protocol_sha256"
            ),
            "frozen_bundle": args.frozen_bundle.as_posix(),
            "frozen_bundle_sha256": bundle_sha,
            "arm_key": spec.key,
            "arm_label": spec.label,
            "policy_key": spec.policy_key,
            "policy_family": spec.policy_family,
            "robot_selector": spec.robot_selector,
            "context_scheduler": spec.context_scheduler,
            "policy_contract": policy_contract,
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "formal": not bool(args.development),
            "admission_key": spec.admission_key,
            "station_admission": spec.admission_mode,
            "dynamic_admission_config": (
                dynamic_config if spec.admission_key == "eta_v3" else None
            ),
            "physical_only_semantics": (
                "reserve checks physical occupancy only; in-transit "
                "commitments have no count cap; physical check-in still "
                "requires a free configured station slot"
                if spec.admission_key == "physical_only"
                else None
            ),
            "source_manifest_root": args.source_root.as_posix(),
            "runtime_code_sha256": runtime_hashes,
            "engine_log_level": "ERROR",
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "historical_manifest_contract": manifest_contract["historical"],
        "audit": audit,
        "station_admission_audit": station_audit,
        "fifo_waiting_audit": fifo_audit,
        "metrics": metrics,
        "policy_trace": policy_trace,
    }
    _atomic_json(output, payload)
    print(f"[done] {output}")
    print(json.dumps({
        "completed_orders": metrics.get("completed_orders"),
        "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
        "stall_ratio_mean": metrics.get("stall_ratio_mean"),
        "max_committed_load": _max_mapping(
            station_audit.get("max_committed_load")
        ),
        "max_occupancy": _max_mapping(station_audit.get("max_occupancy")),
        "policy_audit": audit["passed"],
    }, indent=2, ensure_ascii=False))


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


_T_CRITICAL_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _mean_ci95(values: Sequence[float]) -> list[float] | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    if len(values) == 1:
        return [round(mean, 6), round(mean, 6)]
    df = len(values) - 1
    critical = _T_CRITICAL_975.get(df, 1.96)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [round(mean - half, 6), round(mean + half, 6)]


def _payloads_by_seed(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    return {
        int((payload.get("meta") or {}).get("seed")): payload
        for payload in payloads
    }


def _linear_contrast(
    name: str,
    coefficients: Mapping[str, float],
    payloads_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    relative_denominator_arm: str | None = None,
) -> dict[str, Any]:
    indexed = {
        arm: _payloads_by_seed(payloads_by_arm[arm])
        for arm in coefficients
    }
    common_seeds = sorted(set.intersection(*(
        set(rows) for rows in indexed.values()
    ))) if indexed else []
    rows = []
    throughput_wins = throughput_ties = throughput_losses = 0
    deadlock_wins = deadlock_ties = deadlock_losses = 0
    for seed in common_seeds:
        metric_rows = {
            arm: indexed[arm][seed].get("metrics") or {}
            for arm in coefficients
        }
        deltas = {}
        for metric in (
            "completed_orders", "deadlock_ratio_mean", "stall_ratio_mean"
        ):
            value = sum(
                float(coefficient)
                * float(metric_rows[arm].get(metric, 0.0))
                for arm, coefficient in coefficients.items()
            )
            deltas[metric] = value
        throughput_delta = deltas["completed_orders"]
        deadlock_delta = deltas["deadlock_ratio_mean"]
        throughput_wins += int(throughput_delta > 1e-12)
        throughput_ties += int(abs(throughput_delta) <= 1e-12)
        throughput_losses += int(throughput_delta < -1e-12)
        deadlock_wins += int(deadlock_delta < -1e-12)
        deadlock_ties += int(abs(deadlock_delta) <= 1e-12)
        deadlock_losses += int(deadlock_delta > 1e-12)
        rows.append({
            "seed": seed,
            "completed_orders_contrast": round(throughput_delta, 6),
            "deadlock_ratio_contrast": round(deadlock_delta, 6),
            "stall_ratio_contrast": round(
                deltas["stall_ratio_mean"], 6
            ),
        })

    throughput_values = [
        float(row["completed_orders_contrast"]) for row in rows
    ]
    deadlock_values = [
        float(row["deadlock_ratio_contrast"]) for row in rows
    ]
    stall_values = [float(row["stall_ratio_contrast"]) for row in rows]
    throughput_mean = _mean(throughput_values)
    denominator_mean = None
    if relative_denominator_arm is not None and rows:
        denominator = indexed[relative_denominator_arm]
        denominator_mean = statistics.fmean(
            float((denominator[seed].get("metrics") or {}).get(
                "completed_orders", 0.0
            ))
            for seed in common_seeds
        )
    return {
        "name": name,
        "coefficients": dict(coefficients),
        "paired_seed_count": len(rows),
        "completed_orders_contrast_mean": throughput_mean,
        "completed_orders_contrast_ci95": _mean_ci95(throughput_values),
        "completed_orders_relative_delta_percent": (
            round(float(throughput_mean) * 100.0 / denominator_mean, 6)
            if throughput_mean is not None and denominator_mean
            else None
        ),
        "throughput_wins_ties_losses": {
            "wins": throughput_wins,
            "ties": throughput_ties,
            "losses": throughput_losses,
        },
        "deadlock_ratio_contrast_mean": _mean(deadlock_values),
        "deadlock_ratio_contrast_ci95": _mean_ci95(deadlock_values),
        "deadlock_lower_wins_ties_losses": {
            "wins": deadlock_wins,
            "ties": deadlock_ties,
            "losses": deadlock_losses,
        },
        "stall_ratio_contrast_mean": _mean(stall_values),
        "stall_ratio_contrast_ci95": _mean_ci95(stall_values),
        "pairs": rows,
    }


def _pair(
    name: str,
    left: str,
    right: str,
    payloads_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    return _linear_contrast(
        name,
        {left: 1.0, right: -1.0},
        payloads_by_arm,
        relative_denominator_arm=right,
    )


def _comparison_plan() -> dict[str, list[tuple]]:
    s1_effect = []
    context_effect = []
    interactions = []
    admission_effect = []
    baselines = []
    for scheduler in CONTEXT_SCHEDULERS:
        for admission in ADMISSION_KEYS:
            s1_effect.append((
                f"s1_minus_s0__{scheduler}__{admission}",
                f"s1_{scheduler}_{admission}",
                f"s0_{scheduler}_{admission}",
            ))
    for selector in ROBOT_SELECTORS:
        for admission in ADMISSION_KEYS:
            j0 = f"{selector}_j0_{admission}"
            j1 = f"{selector}_j1_{admission}"
            dynamic = f"{selector}_dynamic_j_{admission}"
            context_effect.extend([
                (f"j1_minus_j0__{selector}__{admission}", j1, j0),
                (
                    f"dynamic_j_minus_j0__{selector}__{admission}",
                    dynamic,
                    j0,
                ),
                (
                    f"dynamic_j_minus_j1__{selector}__{admission}",
                    dynamic,
                    j1,
                ),
            ])
    for admission in ADMISSION_KEYS:
        for scheduler in ("j1", "dynamic_j"):
            interactions.append((
                f"s_by_{scheduler}_interaction__{admission}",
                {
                    f"s1_{scheduler}_{admission}": 1.0,
                    f"s0_{scheduler}_{admission}": -1.0,
                    f"s1_j0_{admission}": -1.0,
                    f"s0_j0_{admission}": 1.0,
                },
            ))
    for policy in POLICY_KEYS:
        committed = f"{policy}_committed_v1"
        admission_effect.extend([
            (
                f"physical_minus_committed__{policy}",
                f"{policy}_physical_only",
                committed,
            ),
            (
                f"eta_minus_committed__{policy}",
                f"{policy}_eta_v3",
                committed,
            ),
            (
                f"fifo_minus_committed__{policy}",
                f"{policy}_fifo_v2",
                committed,
            ),
        ])
    for selector in ROBOT_SELECTORS:
        for scheduler in CONTEXT_SCHEDULERS:
            policy = f"{selector}_{scheduler}"
            for admission in ADMISSION_KEYS:
                model_arm = f"{policy}_{admission}"
                baselines.extend([
                    (
                        f"{policy}_minus_greedy__{admission}",
                        model_arm,
                        f"greedy_{admission}",
                    ),
                    (
                        f"{policy}_minus_hungarian__{admission}",
                        model_arm,
                        f"hungarian_{admission}",
                    ),
                ])
    return {
        "s1_effect_within_same_j_and_admission": s1_effect,
        "context_scheduler_effect_within_same_s_and_admission": context_effect,
        "s_by_j_interactions": interactions,
        "admission_effect_within_same_policy": admission_effect,
        "model_vs_external_same_admission": baselines,
    }


def _build_comparisons(
    payloads_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for category, entries in _comparison_plan().items():
        rows = {}
        for entry in entries:
            if len(entry) == 3:
                name, left, right = entry
                rows[name] = _pair(
                    name, left, right, payloads_by_arm
                )
            else:
                name, coefficients = entry
                rows[name] = _linear_contrast(
                    name, coefficients, payloads_by_arm
                )
        result[category] = rows
    return result


def _format_ci(value: Any) -> str:
    if not isinstance(value, Sequence) or len(value) != 2:
        return "n/a"
    return f"[{value[0]}, {value[1]}]"


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Phase-C S/J x station-admission factorial",
        "",
        (
            "S0/S1 is the within-context robot selector. J0/J1/Dynamic-J "
            "is the independent cross-context scheduler. In particular, "
            "S1+J1 means S1 chooses a robot inside each context while J1 "
            "statically ranks contexts once per proposal batch."
        ),
        "",
        "## All arms",
        "",
        "| Arm | Completed | Deadlock | Stall | Max committed | Max occupancy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_KEYS:
        row = summary["arms"][arm]
        lines.append(
            f"| {arm} | {row.get('completed_orders_mean')} | "
            f"{row.get('deadlock_ratio_mean')} | "
            f"{row.get('stall_ratio_mean')} | "
            f"{row.get('max_committed_load')} | "
            f"{row.get('max_occupancy')} |"
        )
    for category, comparisons in summary["comparisons"].items():
        lines.extend([
            "",
            f"## {category.replace('_', ' ').title()}",
            "",
            "| Contrast | Orders delta | 95% CI | Relative | W/T/L | Deadlock delta |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for name, row in comparisons.items():
            wtl = row["throughput_wins_ties_losses"]
            relative = row.get("completed_orders_relative_delta_percent")
            relative_text = "n/a" if relative is None else f"{relative}%"
            lines.append(
                f"| {name} | "
                f"{row.get('completed_orders_contrast_mean')} | "
                f"{_format_ci(row.get('completed_orders_contrast_ci95'))} | "
                f"{relative_text} | "
                f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
                f"{row.get('deadlock_ratio_contrast_mean')} |"
            )
    integrity = summary["integrity"]
    lines.extend([
        "",
        "## Integrity",
        "",
        f"- Manifest pairing passed: {integrity['manifest_pairing_passed']}",
        f"- Policy fingerprints paired across admission: "
        f"{integrity['policy_pairing_passed']}",
        f"- All arm audits passed: {integrity['all_arm_audits_passed']}",
        f"- Physical capacity violations: "
        f"{integrity['physical_capacity_violations_all_arms']}",
        "",
    ])
    return "\n".join(lines)


def _summarize(args: argparse.Namespace) -> None:
    payloads_by_arm: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in ARM_KEYS
    }
    missing = []
    for arm in ARM_KEYS:
        for seed in args.seeds:
            path = _output_path(args.output_root, arm, args.load, seed)
            if not path.is_file():
                missing.append(path.as_posix())
                continue
            payload = _read_json(path)
            payload["_path"] = path.as_posix()
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"wrong factorial result schema: {path}")
            if not bool((payload.get("audit") or {}).get("passed")):
                raise RuntimeError(f"failed arm audit in result: {path}")
            payloads_by_arm[arm].append(payload)
    if missing and not args.allow_incomplete:
        raise FileNotFoundError(
            "missing factorial results:\n" + "\n".join(missing)
        )

    manifest_checks = {}
    manifest_pairing_passed = True
    for seed in args.seeds:
        manifest_path = _manifest_path(args.source_root, args.load, seed)
        if not manifest_path.is_file():
            manifest_checks[str(seed)] = {"missing_source_manifest": True}
            manifest_pairing_passed = False
            continue
        manifest = _validate_manifest(manifest_path)
        historical = _validate_historical_contract(
            manifest,
            v1_root=args.manifest_reference_root,
            v3_root=args.v3_reference_root,
            load=args.load,
            seed=seed,
            require_v3=args.require_v3_reference,
            skip=args.skip_historical_contract,
        )
        arm_shas = {
            str((payload.get("manifest") or {}).get("content_sha256"))
            for arm in ARM_KEYS
            for payload in payloads_by_arm[arm]
            if int((payload.get("meta") or {}).get("seed", -1)) == seed
        }
        expected = str(manifest.get("manifest_sha256"))
        passed = (
            arm_shas == {expected}
            if not args.allow_incomplete
            else (not arm_shas or arm_shas == {expected})
        )
        manifest_pairing_passed &= passed
        manifest_checks[str(seed)] = {
            "passed": passed,
            "source_content_sha256": expected,
            "source_total_orders": int(manifest.get("total_orders", 0)),
            "arm_content_sha256_values": sorted(arm_shas),
            "historical_contract": historical,
        }

    policy_checks = {}
    policy_pairing_passed = True
    for policy in POLICY_KEYS:
        arms = [f"{policy}_{admission}" for admission in ADMISSION_KEYS]
        fingerprints = {
            str(
                ((payload.get("meta") or {}).get("policy_contract") or {}).get(
                    "fingerprint_sha256"
                )
            )
            for arm in arms
            for payload in payloads_by_arm[arm]
        }
        passed = (
            len(fingerprints) == 1
            and "None" not in fingerprints
            and "" not in fingerprints
        ) if not args.allow_incomplete else (
            len(fingerprints) <= 1
            and "None" not in fingerprints
            and "" not in fingerprints
        )
        policy_pairing_passed &= passed
        policy_checks[policy] = {
            "passed": passed,
            "arms": arms,
            "fingerprints": sorted(fingerprints),
        }

    arms = {
        arm: _aggregate_arm(payloads_by_arm[arm]) for arm in ARM_KEYS
    }
    comparisons = _build_comparisons(payloads_by_arm)
    all_audits_passed = all(
        bool((payload.get("audit") or {}).get("passed"))
        for payloads in payloads_by_arm.values()
        for payload in payloads
    )
    physical_violations = sum(
        int(row.get("physical_capacity_violations_sum", 0))
        for row in arms.values()
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "meta": {
            "load": args.load,
            "seeds": [int(seed) for seed in args.seeds],
            "ticks": int(args.ticks),
            "source_root": args.source_root.as_posix(),
            "output_root": args.output_root.as_posix(),
            "runner": Path(__file__).as_posix(),
            "runner_sha256": sha256_file(Path(__file__)),
            "missing_results": missing,
        },
        "factorial_contract": FACTORIAL_CONTRACT,
        "integrity": {
            "manifest_pairing_passed": bool(manifest_pairing_passed),
            "manifest_checks": manifest_checks,
            "policy_pairing_passed": bool(policy_pairing_passed),
            "policy_checks": policy_checks,
            "all_arm_audits_passed": bool(all_audits_passed),
            "physical_capacity_violations_all_arms": int(
                physical_violations
            ),
        },
        "arms": arms,
        "comparisons": comparisons,
    }
    if not args.allow_incomplete and not all((
        manifest_pairing_passed,
        policy_pairing_passed,
        all_audits_passed,
        physical_violations == 0,
    )):
        raise RuntimeError("factorial summary integrity checks failed")

    validation = args.output_root / "validation"
    json_path = validation / f"station_admission_sj_factorial_{args.load}.json"
    markdown_path = validation / (
        f"station_admission_sj_factorial_{args.load}.md"
    )
    _atomic_json(json_path, summary)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = _markdown_summary(summary)
    if markdown_path.is_file():
        if markdown_path.read_text(encoding="utf-8") != markdown:
            raise FileExistsError(
                f"refusing to overwrite changed summary: {markdown_path}"
            )
    else:
        markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    print(json.dumps({
        "summary_json": json_path.as_posix(),
        "summary_markdown": markdown_path.as_posix(),
        "integrity": summary["integrity"],
    }, indent=2, ensure_ascii=False))


def _validate_args(args: argparse.Namespace) -> None:
    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive")
    if args.station_trace_stride <= 0:
        raise SystemExit("--station-trace-stride must be positive")
    if any(value < 0 for value in (
        args.policy_trace_max_records,
        args.station_trace_max_records,
        args.fifo_trace_max_records,
    )):
        raise SystemExit("trace record limits must be non-negative")
    if args.mode in ("manifest", "arm") and args.seed is None:
        raise SystemExit("--seed is required for manifest and arm modes")
    if not args.development:
        if args.ticks != FORMAL_TICKS:
            raise SystemExit(f"formal factorial freezes --ticks={FORMAL_TICKS}")
        seeds = args.seeds if args.mode == "summary" else [args.seed]
        invalid = [seed for seed in seeds if seed not in FORMAL_SEEDS]
        if invalid:
            raise SystemExit(
                f"formal seeds must be in {list(FORMAL_SEEDS)}: {invalid}"
            )
        if args.skip_historical_contract:
            raise SystemExit(
                "formal factorial cannot skip the historical manifest contract"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("manifest", "arm", "summary"), required=True
    )
    parser.add_argument("--arm", choices=ARM_KEYS, default=None)
    parser.add_argument("--load", choices=LOADS, default="high")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS)
    )
    parser.add_argument("--ticks", type=int, default=FORMAL_TICKS)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frozen-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--manifest-reference-root",
        type=Path,
        default=DEFAULT_V1_REFERENCE_ROOT,
    )
    parser.add_argument(
        "--v3-reference-root", type=Path, default=DEFAULT_V3_REFERENCE_ROOT
    )
    parser.add_argument("--require-v3-reference", action="store_true")
    parser.add_argument("--skip-historical-contract", action="store_true")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--policy-trace-max-records", type=int, default=0)
    parser.add_argument("--station-trace-stride", type=int, default=5)
    parser.add_argument("--station-trace-max-records", type=int, default=300)
    parser.add_argument("--fifo-trace-max-records", type=int, default=0)
    parser.add_argument("--max-committed-multiplier", type=float, default=2.0)
    parser.add_argument("--healthy-extra-ratio", type=float, default=1.0)
    parser.add_argument("--caution-extra-ratio", type=float, default=0.5)
    parser.add_argument("--brake-extra-ratio", type=float, default=0.0)
    parser.add_argument("--eta-near-ticks", type=int, default=8)
    parser.add_argument("--eta-mid-ticks", type=int, default=20)
    args = parser.parse_args()
    _validate_args(args)
    if args.mode == "manifest":
        _validate_manifest_only(args)
    elif args.mode == "arm":
        _run_arm(args)
    else:
        _summarize(args)


if __name__ == "__main__":
    main()


__all__ = [
    "ADMISSION_KEYS",
    "ARM_KEYS",
    "ARM_SPECS",
    "CONTEXT_SCHEDULERS",
    "FACTORIAL_CONTRACT",
    "ROBOT_SELECTORS",
]
