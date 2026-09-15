"""Run the corrected Combo-S1 + static-J1 paper replacement and frontier."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from functools import lru_cache
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
from WorldModel.core.long_risk_schema import (
    LONG_RISK_SCHEMA_VERSION,
    long_risk_runtime_contract,
)
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
from WorldModel.evaluation.phase_c_combo_s1_j1_paper_protocol import (
    BUNDLE_SCHEMA_VERSION,
    DERIVED_REPORT_METRICS,
    FRONTIER_AUDIT_SCHEMA_VERSION,
    FRONTIER_COLLAPSE_EFFICIENCY,
    FRONTIER_COMPARISONS,
    FRONTIER_LOAD,
    FRONTIER_OUTPUT_ROOT,
    FRONTIER_POINTS,
    FRONTIER_SEEDS,
    FRONTIER_SOURCE_ROOT,
    FRONTIER_SUMMARY_SCHEMA_VERSION,
    FRONTIER_SUSTAINABLE_MAX_COLLAPSE_RATE,
    FRONTIER_SUSTAINABLE_MEAN_CLEARANCE,
    FRESH_SEEDS,
    LEGACY_AUDIT_SCHEMA_VERSION,
    LEGACY_HIGH_MANIFEST_ROOT,
    LEGACY_HIGH_REFERENCE_ROOT,
    LEGACY_LOW_MID_MANIFEST_ROOT,
    LEGACY_LOW_MID_REFERENCE_ROOT,
    LEGACY_REFERENCE_ARM_DIRS,
    LEGACY_REFERENCE_LABELS,
    LEGACY_SEEDS,
    LOADS,
    MAIN_FRESH_COMPARISONS,
    MAIN_FRESH_OUTPUT_ROOT,
    MAIN_LEGACY_COMPARISONS,
    MAIN_LEGACY_OUTPUT_ROOT,
    MAIN_SUMMARY_SCHEMA_VERSION,
    RUN_ARMS,
    RUN_ARM_LABELS,
    RUN_SCHEMA_VERSION,
    TICKS,
    formal_protocol,
    selector_config,
    sha256_file,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    PSI_HEAD_CHECKPOINT,
    PSI_SCALE_CONTRACT,
)
from WorldModel.evaluation.phase_c_s1_long_risk_correction_protocol import (
    CANDIDATE_CHECKPOINT,
    LOAD_CONFIGS,
    PHASEC_CONFIG,
    TOP_M,
    canonical_sha256,
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
from WorldModel.evaluation.run_phase_c_station_admission_restoration import (
    _validate_manifest,
)
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


BUNDLE_FILENAME = "phase_c_combo_s1_j1_paper_frozen_protocol.json"
FRONTIER_CONTRACT_FILENAME = "capacity_frontier_manifest_contract.json"

REQUIRED_ARTIFACTS = {
    "candidate_checkpoint": str(CANDIDATE_CHECKPOINT),
    "psi_head_checkpoint": Path(PSI_HEAD_CHECKPOINT).as_posix(),
    "psi_scale_contract": Path(PSI_SCALE_CONTRACT).as_posix(),
    "config_low": LOAD_CONFIGS["low"],
    "config_mid": LOAD_CONFIGS["mid"],
    "config_high": LOAD_CONFIGS["high"],
    "long_risk_schema": "WorldModel/core/long_risk_schema.py",
    "world_model_core": "WorldModel/core/model.py",
    "world_model_assigner": (
        "Policies/TaskAssigner/WorldModelTaskAssigner/"
        "world_model_task_assigner.py"
    ),
    "static_j_assigner": (
        "Policies/TaskAssigner/WorldModelTaskAssigner/"
        "psi_dispatch_context_assigner.py"
    ),
    "base_task_assigner": "Policies/TaskAssigner/base_task_assigner.py",
    "context_assignment": "Policies/TaskAssigner/context_assignment.py",
    "online_evaluator": "WorldModel/evaluation/evaluate_online_v6.py",
    "simulation_engine": "Engine/simulation_engine.py",
    "station_state": "WorldState/station_state.py",
    "station_audit": (
        "WorldModel/evaluation/run_phase_c_station_admission_restoration.py"
    ),
    "protocol": (
        "WorldModel/evaluation/phase_c_combo_s1_j1_paper_protocol.py"
    ),
    "runner": (
        "WorldModel/evaluation/run_phase_c_combo_s1_j1_paper_experiments.py"
    ),
    "submission_script": (
        "WorldModel/evaluation/"
        "run_phase_c_combo_s1_j1_main_and_frontier_60cpu.slurm"
    ),
    "protocol_test": (
        "WorldModel/tests/test_phase_c_combo_s1_j1_paper_protocol.py"
    ),
}


LEGACY_POLICY_METADATA = {
    "greedy": ("greedy", "greedy", None, None),
    "hungarian": ("hungarian", "hungarian", None, None),
    "phasec": ("s0_j0", "world_model", "s0", "j0"),
    "s0_j1": ("s0_j1", "world_model", "s0", "j1"),
    "legacy_event_j1": ("s1_j1", "world_model", "s1", "j1"),
}


def _bundle_path(root: Path) -> Path:
    return root / BUNDLE_FILENAME


def _freeze_bundle(root: Path) -> dict[str, Any]:
    protocol = formal_protocol()
    artifacts: dict[str, dict[str, str]] = {}
    for name, raw_path in REQUIRED_ARTIFACTS.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts[name] = {
            "path": path.as_posix(),
            "sha256": sha256_file(path),
        }
    payload = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "artifacts": artifacts,
    }
    _write_json_exact(_bundle_path(root), payload)
    return payload


def _verify_bundle(path: Path) -> dict[str, Any]:
    bundle = _read_json(path)
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"wrong bundle schema: {path}")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("bundle lacks protocol")
    if canonical_sha256(protocol) != str(bundle.get("protocol_sha256", "")):
        raise ValueError("bundle protocol hash mismatch")
    if protocol != formal_protocol():
        raise ValueError("current formal protocol differs from frozen protocol")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("bundle lacks artifacts")
    for name, raw_path in REQUIRED_ARTIFACTS.items():
        artifact = artifacts.get(name)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"bundle lacks artifact {name}")
        path_value = Path(str(artifact.get("path", "")))
        if path_value.as_posix() != Path(raw_path).as_posix():
            raise ValueError(f"artifact path changed for {name}: {path_value}")
        if not path_value.is_file():
            raise FileNotFoundError(path_value)
        if sha256_file(path_value) != str(artifact.get("sha256", "")):
            raise ValueError(f"frozen artifact changed: {path_value}")
    return bundle


def _manifest_path(root: Path, load: str, seed: int) -> Path:
    return root / "order_manifests" / f"orders_{load}_seed{seed}.json"


def _output_path(root: Path, arm: str, load: str, seed: int) -> Path:
    return root / "per_arm" / arm / f"{load}_seed{seed}.json"


def _legacy_manifest_root(args: argparse.Namespace, load: str) -> Path:
    return (
        Path(args.legacy_high_manifest_root)
        if load == "high"
        else Path(args.legacy_low_mid_manifest_root)
    )


def _legacy_reference_root(args: argparse.Namespace, load: str) -> Path:
    return (
        Path(args.legacy_high_reference_root)
        if load == "high"
        else Path(args.legacy_low_mid_reference_root)
    )


@lru_cache(maxsize=None)
def _policy_contract(arm: str) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "arm": arm,
        "label": RUN_ARM_LABELS[arm],
        "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
        "in_transit_committed_cap": None,
    }
    if arm in ("greedy", "hungarian"):
        contract.update({
            "policy_family": "analytic_baseline",
            "checkpoint": None,
        })
    elif arm == "phasec":
        contract.update({
            "policy_family": "world_model",
            "checkpoint": str(CANDIDATE_CHECKPOINT),
            "checkpoint_sha256": sha256_file(CANDIDATE_CHECKPOINT),
            "robot_selector": "phasec_s0",
            "context_scheduler": "j0_prefix",
            "robot_selector_config": dict(PHASEC_CONFIG),
        })
    elif arm == "combo_j1":
        contract.update({
            "policy_family": "world_model",
            "checkpoint": str(CANDIDATE_CHECKPOINT),
            "checkpoint_sha256": sha256_file(CANDIDATE_CHECKPOINT),
            "robot_selector": "corrected_quantile_combo_s1",
            "robot_selector_config": selector_config(),
            "energy_drift_signal": "combo",
            "context_scheduler": "static_j1",
            "psi_context_mode": "j_ascending",
            "j_refresh": "once_per_proposal_batch",
            "psi_head_checkpoint": Path(PSI_HEAD_CHECKPOINT).as_posix(),
            "psi_head_checkpoint_sha256": sha256_file(PSI_HEAD_CHECKPOINT),
            "psi_scale_contract": Path(PSI_SCALE_CONTRACT).as_posix(),
            "psi_scale_contract_sha256": sha256_file(PSI_SCALE_CONTRACT),
            "long_risk_runtime_contract": long_risk_runtime_contract(),
        })
    else:
        raise ValueError(arm)
    return {**contract, "fingerprint_sha256": canonical_sha256(contract)}


def _make_assigner(arm: str):
    if arm == "greedy":
        return GreedyTaskAssigner()
    if arm == "hungarian":
        return HungarianTaskAssigner()
    if arm == "phasec":
        return WorldModelTaskAssigner(
            checkpoint_path=str(CANDIDATE_CHECKPOINT),
            top_m=TOP_M,
            **dict(PHASEC_CONFIG),
        )
    if arm == "combo_j1":
        return PsiDispatchContextWorldModelTaskAssigner(
            psi_head_checkpoint=str(PSI_HEAD_CHECKPOINT),
            psi_scale_contract=str(PSI_SCALE_CONTRACT),
            psi_context_mode="j_ascending",
            psi_trace_enabled=False,
            psi_trace_max_records=0,
            allow_phasec_s0_robot_scorer=False,
            checkpoint_path=str(CANDIDATE_CHECKPOINT),
            top_m=TOP_M,
            energy_conv_random_flip_seed=0,
            **selector_config(),
        )
    raise ValueError(arm)


def _resume_ok(
    path: Path,
    *,
    arm: str,
    load: str,
    seed: int,
    ticks: int,
    bundle_sha256: str,
    protocol_sha256: str,
    manifest_sha256: str,
    policy_fingerprint: str,
) -> bool:
    if not path.is_file():
        return False
    payload = _read_json(path)
    meta = payload.get("meta") or {}
    manifest = payload.get("manifest") or {}
    checks = {
        "schema": payload.get("schema_version") == RUN_SCHEMA_VERSION,
        "arm": meta.get("arm_key") == arm,
        "load": meta.get("load") == load,
        "seed": int(meta.get("seed", -1)) == seed,
        "ticks": int(meta.get("ticks", -1)) == ticks,
        "bundle": meta.get("frozen_bundle_sha256") == bundle_sha256,
        "protocol": meta.get("protocol_sha256") == protocol_sha256,
        "policy": (
            (meta.get("policy_contract") or {}).get("fingerprint_sha256")
            == policy_fingerprint
        ),
        "manifest": manifest.get("content_sha256") == manifest_sha256,
        "station": (
            meta.get("station_admission") == STATION_ADMISSION_PHYSICAL_ONLY
        ),
        "audit": bool((payload.get("audit") or {}).get("passed")),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"cannot resume incompatible output {path}: {failed}")
    print(f"[resume] arm={arm} load={load} seed={seed}: {path}")
    return True


def _run_audit(
    arm: str,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    station_audit: Mapping[str, Any],
    *,
    replayed: bool,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "manifest_replay_mode": (
            bool(metrics.get("order_arrival_replayed")) == bool(replayed)
        ),
        "manifest_content_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_count_matches": (
            int(metrics.get("order_arrival_count", -1))
            == int(manifest.get("total_orders", -2))
        ),
        "station_audit_passed": bool(station_audit.get("passed")),
        "physical_only_mode": (
            station_audit.get("mode") == STATION_ADMISSION_PHYSICAL_ONLY
        ),
        "physical_capacity_never_exceeded": int(
            station_audit.get("physical_capacity_violation_count", -1)
        ) == 0,
        "no_committed_cap_contract": (
            station_audit.get("committed_capacity_contract")
            == "not enforced for in-transit commitments"
        ),
    }
    if arm in ("greedy", "hungarian"):
        checks["analytic_baseline_has_no_model_calls"] = int(
            metrics.get("model_assign_calls", 0)
        ) == 0
    else:
        checks.update({
            "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
            "no_greedy_fallback": int(
                metrics.get("fallback_greedy_calls", 0)
            ) == 0,
            "native_no_assign_disabled": not bool(
                metrics.get("native_no_assign_enabled")
            ),
            "long_risk_schema_matches": (
                metrics.get("long_risk_schema_version")
                == LONG_RISK_SCHEMA_VERSION
            ),
            "long_risk_contract_matches": (
                metrics.get("long_risk_runtime_contract")
                == long_risk_runtime_contract()
            ),
        })
    if arm == "phasec":
        checks.update({
            "phasec_energy_mode_off": metrics.get("energy_scoring_mode") == "off",
            "phasec_no_conversion": int(
                metrics.get("energy_conv_contexts", 0)
            ) == 0,
        })
    if arm == "combo_j1":
        checks.update({
            "combo_conversion_mode": (
                metrics.get("energy_scoring_mode") == "conversion"
            ),
            "combo_signal": metrics.get("energy_drift_signal") == "combo",
            "combo_contexts_observed": int(
                metrics.get("energy_conv_contexts", 0)
            ) > 0,
            "combo_s1_reported": bool(
                metrics.get("psi_dispatch_s1_within_context")
            ),
            "j1_mode": metrics.get("psi_dispatch_mode") == "j_ascending",
            "j1_head_loaded": bool(metrics.get("psi_dispatch_head_loaded")),
            "j1_head_hash": (
                metrics.get("psi_dispatch_head_checkpoint_sha256")
                == sha256_file(PSI_HEAD_CHECKPOINT)
            ),
            "j1_scale_hash": (
                metrics.get("psi_dispatch_scale_contract_sha256")
                == sha256_file(PSI_SCALE_CONTRACT)
            ),
            "j1_evaluated": int(
                metrics.get("psi_dispatch_eval_calls", 0)
            ) > 0,
            "j1_contexts_seen": int(
                metrics.get("psi_dispatch_contexts_seen", 0)
            ) > 0,
            "j1_parent_scorer_preserved": (
                metrics.get("psi_dispatch_robot_scorer")
                == "WorldModelTaskAssigner.select_robots_unmodified"
            ),
            "j1_s1_variant": (
                metrics.get("psi_dispatch_robot_scorer_variant")
                == "s1_within_context"
            ),
        })
    return {"passed": all(checks.values()), "checks": checks}


def _run_arm(args: argparse.Namespace) -> None:
    if args.arm not in RUN_ARMS:
        raise ValueError(args.arm)
    if args.load not in LOADS:
        raise ValueError(args.load)
    if int(args.seed) <= 0:
        raise ValueError("seed must be positive")
    if int(args.ticks) != TICKS:
        raise ValueError(f"paper protocol freezes ticks={TICKS}")
    if args.generate_manifest and args.arm != "greedy":
        raise ValueError("only Greedy can generate a fresh manifest")

    root = Path(args.output_root)
    bundle_path = Path(args.frozen_bundle or _bundle_path(root))
    bundle = _verify_bundle(bundle_path)
    protocol_sha256 = str(bundle["protocol_sha256"])
    bundle_sha256 = sha256_file(bundle_path)
    manifest_path = Path(args.manifest_path)
    policy_contract = _policy_contract(args.arm)
    output_path = _output_path(root, args.arm, args.load, int(args.seed))

    manifest_exists = manifest_path.is_file()
    if manifest_exists:
        manifest_before = _validate_manifest(manifest_path)
        manifest_sha = str(manifest_before["manifest_sha256"])
        if _resume_ok(
            output_path,
            arm=args.arm,
            load=args.load,
            seed=int(args.seed),
            ticks=int(args.ticks),
            bundle_sha256=bundle_sha256,
            protocol_sha256=protocol_sha256,
            manifest_sha256=manifest_sha,
            policy_fingerprint=str(policy_contract["fingerprint_sha256"]),
        ):
            return
    elif not args.generate_manifest:
        raise FileNotFoundError(manifest_path)

    generated_now = bool(args.generate_manifest and not manifest_exists)
    replayed = not generated_now
    run_kwargs: dict[str, Any]
    if generated_now:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        run_kwargs = {"save_order_manifest": str(manifest_path)}
    else:
        run_kwargs = {"recorded_orders_path": str(manifest_path)}

    assigner = _make_assigner(args.arm)
    print(
        f"[run] arm={args.arm} load={args.load} seed={args.seed} "
        f"manifest_mode={'generate' if generated_now else 'replay'}"
    )
    with _physical_only_audit_contract() as holder:
        metrics = _run_one_assigner(
            LOAD_CONFIGS[args.load],
            assigner,
            int(args.seed),
            int(args.ticks),
            trace_label=RUN_ARM_LABELS[args.arm],
            **run_kwargs,
        )
    if hasattr(assigner, "psi_dispatch_metrics"):
        metrics.update(assigner.psi_dispatch_metrics())
    manifest = _validate_manifest(manifest_path)
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
        "station_ticks_any_committed_over_capacity": int(
            station_audit.get("ticks_with_any_committed_over_capacity", 0)
        ),
    })
    audit = _run_audit(
        args.arm, metrics, manifest, station_audit, replayed=replayed
    )
    if not audit["passed"]:
        failed = [
            name for name, passed in audit["checks"].items() if not passed
        ]
        raise RuntimeError(
            f"audit failed for {args.arm} {args.load} seed={args.seed}: "
            + ", ".join(failed)
        )

    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": protocol_sha256,
            "frozen_bundle": bundle_path.as_posix(),
            "frozen_bundle_sha256": bundle_sha256,
            "arm_key": args.arm,
            "arm": RUN_ARM_LABELS[args.arm],
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "config": LOAD_CONFIGS[args.load],
            "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
            "manifest_mode": "generated" if generated_now else "replayed",
            "generate_manifest_requested": bool(args.generate_manifest),
            "policy_contract": policy_contract,
            "long_risk_runtime_contract": (
                long_risk_runtime_contract()
                if args.arm in ("phasec", "combo_j1") else None
            ),
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "station_audit": station_audit,
        "audit": audit,
        "metrics": metrics,
    }
    _write_json_exact(output_path, payload)
    print(f"[done] {output_path}")


def _validate_legacy_reference_payload(
    path: Path,
    *,
    expected_arm: str,
    expected_arm_dir: str,
    load: str,
    seed: int,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = _read_json(path)
    meta = payload.get("meta") or {}
    embedded_manifest = payload.get("manifest") or {}
    metrics = payload.get("metrics") or {}
    expected_policy = LEGACY_POLICY_METADATA[expected_arm]
    policy_key, policy_family, robot_selector, context_scheduler = expected_policy
    checks = {
        "arm": meta.get("arm_key") == expected_arm_dir,
        "formal": bool(meta.get("formal")),
        "policy_key": meta.get("policy_key") == policy_key,
        "policy_family": meta.get("policy_family") == policy_family,
        "robot_selector": meta.get("robot_selector") == robot_selector,
        "context_scheduler": (
            meta.get("context_scheduler") == context_scheduler
        ),
        "load": meta.get("load") == load,
        "seed": int(meta.get("seed", -1)) == seed,
        "ticks": int(meta.get("ticks", -1)) == TICKS,
        "physical_only": (
            meta.get("station_admission") == STATION_ADMISSION_PHYSICAL_ONLY
        ),
        "audit": bool((payload.get("audit") or {}).get("passed")),
        "manifest_meta": (
            embedded_manifest.get("content_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_metrics": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_count": (
            int(metrics.get("order_arrival_count", -1))
            == int(manifest.get("total_orders", -2))
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"invalid legacy reference {path}: {failed}")
    return payload


def _audit_legacy_sources(args: argparse.Namespace) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    input_files: list[Path] = []
    legacy_event_runtime_hashes: set[tuple[str, str, str]] = set()
    for load in LOADS:
        manifest_root = _legacy_manifest_root(args, load)
        reference_root = _legacy_reference_root(args, load)
        for seed in LEGACY_SEEDS:
            manifest_path = _manifest_path(manifest_root, load, seed)
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = _validate_manifest(manifest_path)
            input_files.append(manifest_path)
            references: dict[str, Any] = {}
            for arm, arm_dir in LEGACY_REFERENCE_ARM_DIRS.items():
                path = (
                    reference_root / "per_arm" / arm_dir
                    / f"{load}_seed{seed}.json"
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                reference_payload = _validate_legacy_reference_payload(
                    path,
                    expected_arm=arm,
                    expected_arm_dir=arm_dir,
                    load=load,
                    seed=seed,
                    manifest=manifest,
                )
                if arm == "legacy_event_j1":
                    runtime_hashes = (
                        reference_payload.get("meta") or {}
                    ).get("runtime_code_sha256") or {}
                    hash_triplet = (
                        str(runtime_hashes.get("runner", "")),
                        str(runtime_hashes.get("world_model_assigner", "")),
                        str(runtime_hashes.get("static_j_assigner", "")),
                    )
                    if not all(hash_triplet):
                        raise ValueError(
                            f"legacy event arm lacks runtime hashes: {path}"
                        )
                    legacy_event_runtime_hashes.add(hash_triplet)
                input_files.append(path)
                references[arm] = {
                    "path": path.as_posix(),
                    "file_sha256": sha256_file(path),
                }
            entries.append({
                "load": load,
                "seed": seed,
                "manifest": manifest_path.as_posix(),
                "manifest_file_sha256": sha256_file(manifest_path),
                "manifest_content_sha256": manifest["manifest_sha256"],
                "total_orders": int(manifest["total_orders"]),
                "references": references,
            })
    if len(legacy_event_runtime_hashes) != 1:
        raise ValueError(
            "legacy event-logit references do not share one frozen runtime: "
            f"{sorted(legacy_event_runtime_hashes)}"
        )
    legacy_runner_hash, legacy_assigner_hash, legacy_j1_hash = next(
        iter(legacy_event_runtime_hashes)
    )
    payload = {
        "schema_version": LEGACY_AUDIT_SCHEMA_VERSION,
        "passed": True,
        "loads": list(LOADS),
        "seeds": list(LEGACY_SEEDS),
        "manifest_count": len(LOADS) * len(LEGACY_SEEDS),
        "reference_output_count": (
            len(LOADS) * len(LEGACY_SEEDS)
            * len(LEGACY_REFERENCE_ARM_DIRS)
        ),
        "legacy_event_semantics": (
            "historical s1_j1_physical_only used event_logit at runtime despite "
            "the old metadata string 'combo'"
        ),
        "legacy_event_runtime_hashes": {
            "runner": legacy_runner_hash,
            "world_model_assigner": legacy_assigner_hash,
            "static_j_assigner": legacy_j1_hash,
        },
        "entries": entries,
    }
    output = Path(args.output_root) / "validation" / "legacy_source_audit.json"
    _write_json_exact(output, payload)
    print(f"[complete] legacy source audit={output}")
    return payload


def _order_identity(order: Mapping[str, Any]) -> str:
    payload = {
        key: value for key, value in order.items() if key != "tick"
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _infer_base_tick_interval(
    observed_ticks: Mapping[str, int],
    *,
    target_ticks: int,
    base_ticks: int,
) -> tuple[int, int] | None:
    """Infer whether scaled observations can share one integer base tick."""

    lower = 0
    upper = int(base_ticks) - 1
    for tag, multiplier in FRONTIER_POINTS:
        scale = Fraction(str(multiplier))
        cutoff = scale * int(target_ticks)
        if tag not in observed_ticks:
            # The construction selects every base order before this cutoff.
            lower = max(lower, _ceil_fraction(cutoff))
            continue
        tick = int(observed_ticks[tag])
        if not 0 <= tick < int(target_ticks):
            return None
        # tick == floor(base_tick / multiplier).
        lower = max(lower, _ceil_fraction(scale * tick))
        upper = min(
            upper,
            _ceil_fraction(scale * (tick + 1)) - 1,
            _ceil_fraction(cutoff) - 1,
        )
    if lower > upper:
        return None
    return lower, upper


def _audit_frontier_sources(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.frontier_source_root)
    contract_path = root / FRONTIER_CONTRACT_FILENAME
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract = _read_json(contract_path)
    expected_contract_sha = str(contract.get("contract_sha256", ""))
    contract_for_hash = dict(contract)
    contract_for_hash.pop("contract_sha256", None)
    nested_rows = contract.get("nested_prefix_checks")
    if not isinstance(nested_rows, list):
        nested_rows = []
    nested_index: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in nested_rows:
        if not isinstance(row, Mapping):
            continue
        key = (int(row.get("seed", -1)), str(row.get("tag", "")))
        if key in nested_index:
            raise ValueError(f"duplicate frontier nested check: {key}")
        nested_index[key] = row
    scaling_formula = contract.get("scaling_formula") or {}
    expected_base_ticks = int(
        TICKS * max(multiplier for _, multiplier in FRONTIER_POINTS)
    )
    checks = {
        "schema": (
            contract.get("schema_version")
            == "phase_c_capacity_frontier_manifest_contract_v1"
        ),
        "load": contract.get("load") == FRONTIER_LOAD,
        "ticks": int(contract.get("target_ticks", -1)) == TICKS,
        "seeds": [int(value) for value in contract.get("seeds", [])]
        == list(FRONTIER_SEEDS),
        "points": [
            (str(row.get("tag")), float(row.get("value")))
            for row in contract.get("multipliers", [])
        ] == list(FRONTIER_POINTS),
        "contract_sha256": (
            canonical_sha256(contract_for_hash) == expected_contract_sha
            and len(expected_contract_sha) == 64
        ),
        "base_ticks": int(contract.get("base_ticks", -1))
        == expected_base_ticks,
        "scaling_selection": (
            scaling_formula.get("selection")
            == "base_tick < target_ticks * multiplier"
        ),
        "scaling_tick": (
            scaling_formula.get("scaled_tick")
            == "floor(base_tick / multiplier)"
        ),
        "unit_reference_required": bool(
            contract.get("unit_reference_required")
        ),
        "base_source_contract_required": bool(
            contract.get("base_source_contract_required")
        ),
        "nested_contract_cardinality": len(nested_index)
        == len(FRONTIER_POINTS) * len(FRONTIER_SEEDS),
        "nested_contract_checks": len(nested_rows) == len(nested_index) and all(
            bool(row.get("previous_prefix_is_subset"))
            for row in nested_rows
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"invalid frontier contract {contract_path}: {failed}")

    entries: list[dict[str, Any]] = []
    seed_pairing: list[dict[str, Any]] = []
    base_ticks = int(contract["base_ticks"])
    for seed in FRONTIER_SEEDS:
        seed_contract = (contract.get("per_seed") or {}).get(str(seed)) or {}
        scaled_contract = seed_contract.get("scaled") or {}
        previous_keys: set[str] = set()
        previous_count = -1
        point_order_ticks: dict[str, dict[str, int]] = {}
        for tag, multiplier in FRONTIER_POINTS:
            path = _manifest_path(root / tag, FRONTIER_LOAD, seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            manifest = _validate_manifest(path)
            expected = scaled_contract.get(tag) or {}
            selection = expected.get("selection") or {}
            nested_expected = nested_index.get((seed, tag)) or {}
            unit_reference = expected.get("unit_reference") or {}
            orders = manifest.get("orders") or []
            keys = {_order_identity(order) for order in orders}
            tick_map = {
                _order_identity(order): int(order.get("tick", -1))
                for order in orders
            }
            point_order_ticks[tag] = tick_map
            ticks = list(tick_map.values())
            row_checks = {
                "multiplier": float(expected.get("multiplier", -1.0))
                == float(multiplier),
                "content_sha256": (
                    expected.get("manifest_sha256")
                    == manifest.get("manifest_sha256")
                ),
                "total_orders": int(expected.get("total_orders", -1))
                == int(manifest.get("total_orders", -2)),
                "unique_order_identity": len(keys) == len(orders),
                "nested_prefix": previous_keys.issubset(keys),
                "monotone_count": len(orders) >= previous_count,
                "tick_range": all(0 <= tick < TICKS for tick in ticks),
                "selection_count": int(
                    selection.get("selected_order_count", -1)
                ) == len(orders),
                "selection_cutoff": float(
                    selection.get("cutoff_base_tick_exclusive", -1.0)
                ) == float(TICKS * multiplier),
                "selection_min_tick": selection.get("scaled_min_tick")
                == (min(ticks) if ticks else None),
                "selection_max_tick": selection.get("scaled_max_tick")
                == (max(ticks) if ticks else None),
                "nested_contract_seed": int(
                    nested_expected.get("seed", -1)
                ) == seed,
                "nested_contract_tag": nested_expected.get("tag") == tag,
                "nested_contract_multiplier": float(
                    nested_expected.get("multiplier", -1.0)
                ) == float(multiplier),
                "nested_contract_count": int(
                    nested_expected.get("selected_order_count", -1)
                ) == len(orders),
                "nested_contract_passed": bool(
                    nested_expected.get("previous_prefix_is_subset")
                ),
                "unit_reference": (
                    tag != "m100"
                    or (
                        bool(unit_reference.get("checked"))
                        and bool(unit_reference.get("passed"))
                        and unit_reference.get("content_sha256")
                        == manifest.get("manifest_sha256")
                        and int(unit_reference.get("total_orders", -1))
                        == len(orders)
                    )
                ),
            }
            if not all(row_checks.values()):
                failed = [
                    name for name, passed in row_checks.items() if not passed
                ]
                raise ValueError(
                    f"invalid frontier manifest {path}: {failed}"
                )
            entries.append({
                "tag": tag,
                "multiplier": multiplier,
                "seed": seed,
                "path": path.as_posix(),
                "file_sha256": sha256_file(path),
                "content_sha256": manifest["manifest_sha256"],
                "total_orders": int(manifest["total_orders"]),
                "checks": row_checks,
            })
            previous_keys = keys
            previous_count = len(orders)
        all_order_keys = set().union(*(
            set(values) for values in point_order_ticks.values()
        ))
        inconsistent_keys = []
        ambiguous_interval_count = 0
        for order_key in all_order_keys:
            observed_ticks = {
                tag: tick_map[order_key]
                for tag, tick_map in point_order_ticks.items()
                if order_key in tick_map
            }
            interval = _infer_base_tick_interval(
                observed_ticks,
                target_ticks=TICKS,
                base_ticks=base_ticks,
            )
            if interval is None:
                inconsistent_keys.append(order_key)
            elif interval[0] != interval[1]:
                ambiguous_interval_count += 1
        base_source = seed_contract.get("base_source_contract") or {}
        seed_checks = {
            "tick_formula_consistent": not inconsistent_keys,
            "base_total_orders": int(
                seed_contract.get("base_total_orders", -1)
            ) == len(all_order_keys),
            "highest_point_contains_base_stream": len(
                point_order_ticks[FRONTIER_POINTS[-1][0]]
            ) == len(all_order_keys),
            "base_max_tick_in_range": 0 <= int(
                seed_contract.get("base_max_tick", -1)
            ) < base_ticks,
            "base_manifest_hash_recorded": len(str(
                seed_contract.get("base_manifest_sha256", "")
            )) == 64,
            "base_source_contract": (
                bool(base_source.get("checked"))
                and bool(base_source.get("passed"))
            ),
        }
        if not all(seed_checks.values()):
            failed = [
                name for name, passed in seed_checks.items() if not passed
            ]
            raise ValueError(
                f"invalid paired frontier seed contract seed={seed}: {failed}; "
                f"inconsistent_orders={len(inconsistent_keys)}"
            )
        seed_pairing.append({
            "seed": seed,
            "unique_base_order_count": len(all_order_keys),
            "exact_base_tick_interval_count": (
                len(all_order_keys) - ambiguous_interval_count
            ),
            "ambiguous_but_consistent_interval_count": (
                ambiguous_interval_count
            ),
            "checks": seed_checks,
        })
    payload = {
        "schema_version": FRONTIER_AUDIT_SCHEMA_VERSION,
        "passed": True,
        "contract": contract_path.as_posix(),
        "contract_file_sha256": sha256_file(contract_path),
        "contract_content_sha256": expected_contract_sha,
        "manifest_count": len(entries),
        "pairing": (
            "nested order identities from one base stream per seed with "
            "multiplier-specific arrival-tick compression"
        ),
        "seed_pairing": seed_pairing,
        "entries": entries,
    }
    output = Path(args.output_root) / "validation" / "frontier_source_audit.json"
    _write_json_exact(output, payload)
    print(f"[complete] frontier source audit={output}")
    return payload


def _metrics_with_derived(
    metrics: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(metrics)
    completed = _number(metrics.get("completed_orders"))
    offered = _number(manifest.get("total_orders"))
    result["completion_ratio"] = (
        completed / offered
        if completed is not None and offered is not None and offered > 0
        else float("nan")
    )
    return result


def _paired_metric_report(
    rows: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
    *,
    scopes: Sequence[str],
    seeds: Sequence[int],
    baseline: str,
    candidate: str,
    metric: str,
    seed_offset: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    per_seed_cluster: list[float] = []
    for seed in seeds:
        values = []
        for scope in scopes:
            left = _number(rows[(scope, seed)][baseline].get(metric))
            right = _number(rows[(scope, seed)][candidate].get(metric))
            if left is not None and right is not None:
                values.append(right - left)
        if len(values) == len(scopes):
            per_seed_cluster.append(sum(values) / len(values))
    for scope_index, scope in enumerate(scopes):
        values = []
        for seed in seeds:
            left = _number(rows[(scope, seed)][baseline].get(metric))
            right = _number(rows[(scope, seed)][candidate].get(metric))
            if left is not None and right is not None:
                values.append(right - left)
        summary = _mean_std(values)
        summary["ci95_low"], summary["ci95_high"] = _bootstrap_ci(
            values, seed_offset + scope_index
        )
        report[scope] = summary
        csv_rows.append({"scope": scope, **summary})
    overall = _mean_std(per_seed_cluster)
    overall["ci95_low"], overall["ci95_high"] = _bootstrap_ci(
        per_seed_cluster, seed_offset + 90
    )
    report["overall_seed_cluster"] = overall
    csv_rows.append({"scope": "overall_seed_cluster", **overall})
    return report, csv_rows


def _aggregate_reports(
    rows: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
    *,
    scopes: Sequence[str],
    seeds: Sequence[int],
    arms: Sequence[str],
    labels: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate_json: dict[str, Any] = {}
    aggregate_rows: list[dict[str, Any]] = []
    for arm in arms:
        aggregate_json[arm] = {}
        for scope in scopes:
            aggregate_json[arm][scope] = {}
            for metric in DERIVED_REPORT_METRICS:
                values = [
                    value
                    for seed in seeds
                    if (
                        value := _number(rows[(scope, seed)][arm].get(metric))
                    ) is not None
                ]
                summary = _mean_std(values)
                aggregate_json[arm][scope][metric] = summary
                aggregate_rows.append({
                    "arm_key": arm,
                    "arm": labels[arm],
                    "scope": scope,
                    "metric": metric,
                    **summary,
                })
    return aggregate_json, aggregate_rows


def _contrast_reports(
    rows: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
    *,
    scopes: Sequence[str],
    seeds: Sequence[int],
    comparisons: Sequence[tuple[str, str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contrast_json: dict[str, Any] = {}
    contrast_rows: list[dict[str, Any]] = []
    for comparison_index, (name, baseline, candidate) in enumerate(comparisons):
        comparison = {
            "baseline": baseline,
            "candidate": candidate,
            "definition": "candidate_minus_baseline",
            "metrics": {},
        }
        for metric_index, metric in enumerate(DERIVED_REPORT_METRICS):
            report, csv_rows = _paired_metric_report(
                rows,
                scopes=scopes,
                seeds=seeds,
                baseline=baseline,
                candidate=candidate,
                metric=metric,
                seed_offset=(
                    11_000_000
                    + comparison_index * 100_000
                    + metric_index * 100
                ),
            )
            comparison["metrics"][metric] = report
            for row in csv_rows:
                contrast_rows.append({
                    "comparison": name,
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    **row,
                })
        contrast_json[name] = comparison
    return contrast_json, contrast_rows


def _load_current_output(
    path: Path,
    *,
    arm: str,
    load: str,
    seed: int,
    bundle_sha256: str,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = _read_json(path)
    meta = payload.get("meta") or {}
    metrics = payload.get("metrics") or {}
    embedded_manifest = payload.get("manifest") or {}
    expected_policy = _policy_contract(arm)
    checks = {
        "schema": payload.get("schema_version") == RUN_SCHEMA_VERSION,
        "arm": meta.get("arm_key") == arm,
        "load": meta.get("load") == load,
        "seed": int(meta.get("seed", -1)) == seed,
        "ticks": int(meta.get("ticks", -1)) == TICKS,
        "bundle": meta.get("frozen_bundle_sha256") == bundle_sha256,
        "protocol": meta.get("protocol_sha256") == canonical_sha256(
            formal_protocol()
        ),
        "policy": (
            (meta.get("policy_contract") or {}).get("fingerprint_sha256")
            == expected_policy["fingerprint_sha256"]
        ),
        "station": (
            meta.get("station_admission") == STATION_ADMISSION_PHYSICAL_ONLY
        ),
        "audit": bool((payload.get("audit") or {}).get("passed")),
        "embedded_manifest": (
            embedded_manifest.get("content_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "count": int(metrics.get("order_arrival_count", -1))
        == int(manifest.get("total_orders", -2)),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"invalid current output {path}: {failed}")
    return payload


def _write_validated_hashes(root: Path, files: Sequence[Path]) -> Path:
    output = root / "validation" / "validated_outputs.sha256"
    unique = sorted(set(files), key=lambda path: path.as_posix())
    _write_text_exact(
        output,
        "".join(
            f"{sha256_file(path)}  {path.as_posix()}\n" for path in unique
        ),
    )
    return output


def _summarize_main(args: argparse.Namespace) -> None:
    mode = args.main_mode
    if mode not in ("legacy", "fresh"):
        raise ValueError(mode)
    root = Path(args.output_root)
    bundle_path = Path(args.frozen_bundle or _bundle_path(root))
    bundle = _verify_bundle(bundle_path)
    bundle_sha = sha256_file(bundle_path)
    seeds = LEGACY_SEEDS if mode == "legacy" else FRESH_SEEDS
    if args.seeds:
        if tuple(args.seeds) != tuple(seeds):
            raise ValueError(
                f"{mode} main summary freezes seeds={list(seeds)}"
            )
    arms = (
        tuple(LEGACY_REFERENCE_ARM_DIRS) + ("combo_j1",)
        if mode == "legacy" else RUN_ARMS
    )
    labels = (
        {**LEGACY_REFERENCE_LABELS, "combo_j1": RUN_ARM_LABELS["combo_j1"]}
        if mode == "legacy" else RUN_ARM_LABELS
    )
    comparisons = (
        MAIN_LEGACY_COMPARISONS if mode == "legacy"
        else MAIN_FRESH_COMPARISONS
    )

    rows: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    input_files: list[Path] = [bundle_path]
    legacy_audit_path = root / "validation" / "legacy_source_audit.json"
    if mode == "legacy":
        if not legacy_audit_path.is_file():
            raise FileNotFoundError(legacy_audit_path)
        legacy_audit = _read_json(legacy_audit_path)
        if not bool(legacy_audit.get("passed")):
            raise ValueError("legacy source audit did not pass")
        input_files.append(legacy_audit_path)

    for load in LOADS:
        for seed in seeds:
            if mode == "legacy":
                manifest_path = _manifest_path(
                    _legacy_manifest_root(args, load), load, seed
                )
            else:
                manifest_path = _manifest_path(root, load, seed)
            manifest = _validate_manifest(manifest_path)
            input_files.append(manifest_path)
            arm_metrics: dict[str, Mapping[str, Any]] = {}
            if mode == "legacy":
                reference_root = _legacy_reference_root(args, load)
                for arm, arm_dir in LEGACY_REFERENCE_ARM_DIRS.items():
                    path = (
                        reference_root / "per_arm" / arm_dir
                        / f"{load}_seed{seed}.json"
                    )
                    payload = _validate_legacy_reference_payload(
                        path,
                        expected_arm=arm,
                        expected_arm_dir=arm_dir,
                        load=load,
                        seed=seed,
                        manifest=manifest,
                    )
                    input_files.append(path)
                    arm_metrics[arm] = _metrics_with_derived(
                        payload.get("metrics") or {}, manifest
                    )
                current_arms = ("combo_j1",)
            else:
                current_arms = RUN_ARMS
            for arm in current_arms:
                path = _output_path(root, arm, load, seed)
                payload = _load_current_output(
                    path,
                    arm=arm,
                    load=load,
                    seed=seed,
                    bundle_sha256=bundle_sha,
                    manifest=manifest,
                )
                input_files.append(path)
                arm_metrics[arm] = _metrics_with_derived(
                    payload.get("metrics") or {}, manifest
                )
            rows[(load, seed)] = arm_metrics

    aggregate_json, aggregate_rows = _aggregate_reports(
        rows,
        scopes=LOADS,
        seeds=seeds,
        arms=arms,
        labels=labels,
    )
    contrast_json, contrast_rows = _contrast_reports(
        rows,
        scopes=LOADS,
        seeds=seeds,
        comparisons=comparisons,
    )

    validation = root / "validation"
    aggregate_csv = validation / "main_aggregate_summary.csv"
    contrast_csv = validation / "main_paired_contrasts.csv"
    summary_json = validation / "main_replacement_summary.json"
    _write_text_exact(
        aggregate_csv,
        _csv_text(
            aggregate_rows,
            ["arm_key", "arm", "scope", "metric", "n", "mean", "std"],
        ),
    )
    _write_text_exact(
        contrast_csv,
        _csv_text(
            contrast_rows,
            [
                "comparison", "baseline", "candidate", "metric", "scope",
                "n", "mean", "std", "ci95_low", "ci95_high",
            ],
        ),
    )
    summary = {
        "schema_version": MAIN_SUMMARY_SCHEMA_VERSION,
        "main_mode": mode,
        "seeds": list(seeds),
        "loads": list(LOADS),
        "ticks": TICKS,
        "protocol_sha256": bundle["protocol_sha256"],
        "frozen_bundle_sha256": bundle_sha,
        "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
        "legacy_event_semantics": (
            "event_logit runtime; never interpreted as quantile combo"
            if mode == "legacy" else None
        ),
        "aggregate": aggregate_json,
        "comparisons": contrast_json,
        "artifact_audit": {
            "passed": True,
            "manifest_count": len(LOADS) * len(seeds),
            "arm_output_count": len(LOADS) * len(seeds) * len(arms),
        },
    }
    _write_json_exact(summary_json, summary)
    input_files.extend([aggregate_csv, contrast_csv, summary_json])
    hashes = _write_validated_hashes(root, input_files)
    print(f"[complete] main summary={summary_json}")
    print(f"[complete] main validated hashes={hashes}")


def _summarize_frontier(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    source_root = Path(args.frontier_source_root)
    bundle_path = Path(args.frozen_bundle or _bundle_path(root))
    bundle = _verify_bundle(bundle_path)
    bundle_sha = sha256_file(bundle_path)
    audit = _audit_frontier_sources(args)
    audit_path = root / "validation" / "frontier_source_audit.json"

    scopes = tuple(tag for tag, _ in FRONTIER_POINTS)
    rows: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    offered: dict[tuple[str, int], int] = {}
    input_files: list[Path] = [bundle_path, audit_path]
    for tag, _ in FRONTIER_POINTS:
        cell_root = root / tag
        for seed in FRONTIER_SEEDS:
            manifest_path = _manifest_path(
                source_root / tag, FRONTIER_LOAD, seed
            )
            manifest = _validate_manifest(manifest_path)
            input_files.append(manifest_path)
            offered[(tag, seed)] = int(manifest["total_orders"])
            arm_metrics: dict[str, Mapping[str, Any]] = {}
            for arm in RUN_ARMS:
                path = _output_path(cell_root, arm, FRONTIER_LOAD, seed)
                payload = _load_current_output(
                    path,
                    arm=arm,
                    load=FRONTIER_LOAD,
                    seed=seed,
                    bundle_sha256=bundle_sha,
                    manifest=manifest,
                )
                input_files.append(path)
                arm_metrics[arm] = _metrics_with_derived(
                    payload.get("metrics") or {}, manifest
                )
            rows[(tag, seed)] = arm_metrics

    aggregate_json, aggregate_rows = _aggregate_reports(
        rows,
        scopes=scopes,
        seeds=FRONTIER_SEEDS,
        arms=RUN_ARMS,
        labels=RUN_ARM_LABELS,
    )
    contrast_json, contrast_rows = _contrast_reports(
        rows,
        scopes=scopes,
        seeds=FRONTIER_SEEDS,
        comparisons=FRONTIER_COMPARISONS,
    )

    frontier_status: dict[str, Any] = {}
    status_rows: list[dict[str, Any]] = []
    completion_efficiency: dict[tuple[str, int, str], float] = {}
    empirical_ceiling: dict[tuple[str, int], float] = {}
    for tag, _ in FRONTIER_POINTS:
        for seed in FRONTIER_SEEDS:
            completed_by_arm = {
                arm: float(rows[(tag, seed)][arm]["completed_orders"])
                for arm in RUN_ARMS
            }
            ceiling = max(completed_by_arm.values())
            empirical_ceiling[(tag, seed)] = ceiling
            for arm, completed in completed_by_arm.items():
                completion_efficiency[(tag, seed, arm)] = (
                    completed / ceiling if ceiling > 0 else 1.0
                )
    for arm in RUN_ARMS:
        arm_status: dict[str, Any] = {}
        point_passes: list[tuple[float, bool]] = []
        for tag, multiplier in FRONTIER_POINTS:
            ratios = [
                float(rows[(tag, seed)][arm]["completion_ratio"])
                for seed in FRONTIER_SEEDS
            ]
            efficiencies = [
                completion_efficiency[(tag, seed, arm)]
                for seed in FRONTIER_SEEDS
            ]
            mean_clearance = sum(ratios) / len(ratios)
            collapse_count = sum(
                efficiency < FRONTIER_COLLAPSE_EFFICIENCY
                for efficiency in efficiencies
            )
            collapse_rate = collapse_count / len(ratios)
            mean_offered = sum(
                offered[(tag, seed)] for seed in FRONTIER_SEEDS
            ) / len(FRONTIER_SEEDS)
            sustainable = (
                mean_clearance >= FRONTIER_SUSTAINABLE_MEAN_CLEARANCE
                and collapse_rate
                <= FRONTIER_SUSTAINABLE_MAX_COLLAPSE_RATE
            )
            point_passes.append((float(multiplier), sustainable))
            row = {
                "tag": tag,
                "multiplier": multiplier,
                "arm": arm,
                "mean_offered_orders": mean_offered,
                "mean_arrival_rate_per_tick": mean_offered / TICKS,
                "mean_clearance_ratio": mean_clearance,
                "mean_empirical_completion_ceiling": sum(
                    empirical_ceiling[(tag, seed)] for seed in FRONTIER_SEEDS
                ) / len(FRONTIER_SEEDS),
                "mean_completion_efficiency": (
                    sum(efficiencies) / len(efficiencies)
                ),
                "collapse_count": collapse_count,
                "collapse_rate": collapse_rate,
                "sustainable": sustainable,
            }
            arm_status[tag] = row
            status_rows.append(row)
        contiguous_critical = None
        still_contiguous = True
        for multiplier, passed in point_passes:
            if still_contiguous and passed:
                contiguous_critical = multiplier
            else:
                still_contiguous = False
        isolated_passing = [
            multiplier for multiplier, passed in point_passes if passed
        ]
        frontier_status[arm] = {
            "points": arm_status,
            "contiguous_critical_multiplier": contiguous_critical,
            "highest_passing_multiplier": (
                max(isolated_passing) if isolated_passing else None
            ),
        }

    validation = root / "validation"
    aggregate_csv = validation / "frontier_aggregate_summary.csv"
    contrast_csv = validation / "frontier_paired_contrasts.csv"
    status_csv = validation / "frontier_status.csv"
    summary_json = validation / "paired_arrival_frontier_summary.json"
    _write_text_exact(
        aggregate_csv,
        _csv_text(
            aggregate_rows,
            ["arm_key", "arm", "scope", "metric", "n", "mean", "std"],
        ),
    )
    _write_text_exact(
        contrast_csv,
        _csv_text(
            contrast_rows,
            [
                "comparison", "baseline", "candidate", "metric", "scope",
                "n", "mean", "std", "ci95_low", "ci95_high",
            ],
        ),
    )
    _write_text_exact(
        status_csv,
        _csv_text(
            status_rows,
            [
                "tag", "multiplier", "arm", "mean_offered_orders",
                "mean_arrival_rate_per_tick", "mean_clearance_ratio",
                "mean_empirical_completion_ceiling",
                "mean_completion_efficiency",
                "collapse_count", "collapse_rate", "sustainable",
            ],
        ),
    )
    summary = {
        "schema_version": FRONTIER_SUMMARY_SCHEMA_VERSION,
        "protocol_sha256": bundle["protocol_sha256"],
        "frozen_bundle_sha256": bundle_sha,
        "source_audit": audit,
        "load": FRONTIER_LOAD,
        "seeds": list(FRONTIER_SEEDS),
        "ticks": TICKS,
        "points": [
            {"tag": tag, "multiplier": multiplier}
            for tag, multiplier in FRONTIER_POINTS
        ],
        "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
        "thresholds": {
            "collapse_efficiency_ratio": FRONTIER_COLLAPSE_EFFICIENCY,
            "sustainable_mean_clearance_ratio": (
                FRONTIER_SUSTAINABLE_MEAN_CLEARANCE
            ),
            "sustainable_max_collapse_rate": (
                FRONTIER_SUSTAINABLE_MAX_COLLAPSE_RATE
            ),
        },
        "collapse_definition": (
            "per multiplier/seed, completed_orders divided by the empirical "
            "best completed_orders among all frozen arms"
        ),
        "critical_multiplier_definition": (
            "largest contiguous sustainable point from the lowest multiplier"
        ),
        "aggregate": aggregate_json,
        "comparisons": contrast_json,
        "frontier_status": frontier_status,
        "artifact_audit": {
            "passed": True,
            "manifest_count": len(FRONTIER_POINTS) * len(FRONTIER_SEEDS),
            "output_count": (
                len(FRONTIER_POINTS) * len(FRONTIER_SEEDS) * len(RUN_ARMS)
            ),
        },
    }
    _write_json_exact(summary_json, summary)
    input_files.extend([
        aggregate_csv, contrast_csv, status_csv, summary_json,
        source_root / FRONTIER_CONTRACT_FILENAME,
    ])
    hashes = _write_validated_hashes(root, input_files)
    print(f"[complete] frontier summary={summary_json}")
    print(f"[complete] frontier validated hashes={hashes}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "freeze", "audit-legacy", "audit-frontier", "arm",
            "summary-main", "summary-frontier",
        ),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=MAIN_LEGACY_OUTPUT_ROOT)
    parser.add_argument("--frozen-bundle", type=Path, default=None)
    parser.add_argument("--arm", choices=RUN_ARMS, default=None)
    parser.add_argument("--load", choices=LOADS, default="high")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--generate-manifest", action="store_true")
    parser.add_argument("--main-mode", choices=("legacy", "fresh"), default=None)
    parser.add_argument(
        "--legacy-high-manifest-root",
        type=Path,
        default=LEGACY_HIGH_MANIFEST_ROOT,
    )
    parser.add_argument(
        "--legacy-low-mid-manifest-root",
        type=Path,
        default=LEGACY_LOW_MID_MANIFEST_ROOT,
    )
    parser.add_argument(
        "--legacy-high-reference-root",
        type=Path,
        default=LEGACY_HIGH_REFERENCE_ROOT,
    )
    parser.add_argument(
        "--legacy-low-mid-reference-root",
        type=Path,
        default=LEGACY_LOW_MID_REFERENCE_ROOT,
    )
    parser.add_argument(
        "--frontier-source-root", type=Path, default=FRONTIER_SOURCE_ROOT
    )
    args = parser.parse_args()

    if args.mode == "freeze":
        bundle = _freeze_bundle(Path(args.output_root))
        print(f"[complete] bundle={_bundle_path(Path(args.output_root))}")
        print(f"[complete] protocol_sha256={bundle['protocol_sha256']}")
        return
    if args.mode == "audit-legacy":
        _audit_legacy_sources(args)
        return
    if args.mode == "audit-frontier":
        _audit_frontier_sources(args)
        return
    if args.mode == "arm":
        if args.arm is None or args.seed is None or args.manifest_path is None:
            parser.error("--mode arm requires --arm, --seed, and --manifest-path")
        _run_arm(args)
        return
    if args.mode == "summary-main":
        if args.main_mode is None:
            parser.error("--mode summary-main requires --main-mode")
        _summarize_main(args)
        return
    if Path(args.output_root) == MAIN_LEGACY_OUTPUT_ROOT:
        args.output_root = FRONTIER_OUTPUT_ROOT
    _summarize_frontier(args)


if __name__ == "__main__":
    main()
