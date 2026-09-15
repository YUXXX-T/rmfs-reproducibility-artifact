"""Run the physical-only corrected S1 x static-J1 paired campaign."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_context_assigner import (
    PsiDispatchContextWorldModelTaskAssigner,
)
from WorldModel.core.long_risk_schema import (
    LONG_RISK_SCHEMA_VERSION,
    long_risk_runtime_contract,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    PSI_HEAD_CHECKPOINT,
    PSI_SCALE_CONTRACT,
)
from WorldModel.evaluation.phase_c_s1_j1_long_risk_correction_protocol import (
    ALL_ARM_KEYS,
    ALL_ARM_LABELS,
    BUNDLE_SCHEMA_VERSION,
    COMPARISONS,
    INTERACTIONS,
    NEW_ARM_KEYS,
    NEW_ARM_LABELS,
    NEW_ARM_SPECS,
    OUTPUT_ROOT,
    PER_SEED_SCHEMA_VERSION,
    SOURCE_ROOT,
    SUMMARY_SCHEMA_VERSION,
    formal_protocol,
    selector_config,
)
from WorldModel.evaluation.phase_c_s1_long_risk_correction_protocol import (
    ARM_KEYS as SOURCE_ARM_KEYS,
    BUNDLE_SCHEMA_VERSION as SOURCE_BUNDLE_SCHEMA_VERSION,
    CANDIDATE_CHECKPOINT,
    LOAD_CONFIGS,
    LOADS,
    PER_SEED_SCHEMA_VERSION as SOURCE_PER_SEED_SCHEMA_VERSION,
    REPORT_METRICS,
    SEEDS,
    SUMMARY_SCHEMA_VERSION as SOURCE_SUMMARY_SCHEMA_VERSION,
    TICKS,
    TOP_M,
    canonical_sha256,
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_s1_long_risk_correction import (
    _bootstrap_ci,
    _csv_text,
    _mean_std,
    _number,
    _read_json,
    _verify_bundle as _verify_source_bundle,
    _write_json_exact,
    _write_text_exact,
)
from WorldModel.evaluation.run_phase_c_station_admission_restoration import (
    StationAdmissionRestorationAuditProbe,
)
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


BUNDLE_FILENAME = "phase_c_s1_j1_long_risk_correction_frozen_protocol.json"
SOURCE_BUNDLE_FILENAME = "phase_c_s1_long_risk_correction_frozen_protocol.json"
SOURCE_SUMMARY_RELATIVE = Path("validation/correction_summary.json")
SOURCE_HASHES_RELATIVE = Path("validation/validated_outputs.sha256")

REGULAR_ARTIFACTS = {
    "candidate_checkpoint": str(CANDIDATE_CHECKPOINT),
    "config_low": LOAD_CONFIGS["low"],
    "config_mid": LOAD_CONFIGS["mid"],
    "config_high": LOAD_CONFIGS["high"],
    "psi_head_checkpoint": Path(PSI_HEAD_CHECKPOINT).as_posix(),
    "psi_scale_contract": Path(PSI_SCALE_CONTRACT).as_posix(),
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
    "psi_dispatch_core": "WorldModel/core/psi_dispatch.py",
    "station_congestion_core": "WorldModel/core/station_congestion_head.py",
    "station_congestion_endpoint": (
        "WorldModel/evaluation/station_congestion_endpoint.py"
    ),
    "online_evaluator": "WorldModel/evaluation/evaluate_online_v6.py",
    "simulation_engine": "Engine/simulation_engine.py",
    "station_state": "WorldState/station_state.py",
    "station_admission_audit": (
        "WorldModel/evaluation/run_phase_c_station_admission_restoration.py"
    ),
    "base_task_assigner": "Policies/TaskAssigner/base_task_assigner.py",
    "context_assignment": "Policies/TaskAssigner/context_assignment.py",
    "world_model_graph_builder": "WorldModel/graph/graph_builder.py",
    "base_s1_protocol": (
        "WorldModel/evaluation/phase_c_s1_hungarian_protocol.py"
    ),
    "psi_dispatch_protocol": (
        "WorldModel/evaluation/phase_c_psi_dispatch_ablation_protocol.py"
    ),
    "source_protocol": (
        "WorldModel/evaluation/"
        "phase_c_s1_long_risk_correction_protocol.py"
    ),
    "source_runner": (
        "WorldModel/evaluation/run_phase_c_s1_long_risk_correction.py"
    ),
    "protocol": (
        "WorldModel/evaluation/"
        "phase_c_s1_j1_long_risk_correction_protocol.py"
    ),
    "runner": (
        "WorldModel/evaluation/"
        "run_phase_c_s1_j1_long_risk_correction.py"
    ),
    "submission_script": (
        "WorldModel/evaluation/"
        "run_phase_c_s1_j1_long_risk_correction_60cpu.slurm"
    ),
}


def _bundle_path(root: Path) -> Path:
    return root / BUNDLE_FILENAME


def _source_paths(source_root: Path, load: str, seed: int) -> tuple[Path, dict[str, Path]]:
    manifest = source_root / "order_manifests" / f"orders_{load}_seed{seed}.json"
    outputs = {
        arm: source_root / "per_arm" / arm / f"{load}_seed{seed}.json"
        for arm in SOURCE_ARM_KEYS
    }
    return manifest, outputs


def _new_output_path(root: Path, arm: str, load: str, seed: int) -> Path:
    return root / "per_arm" / arm / f"{load}_seed{seed}.json"


def _parse_source_hashes(path: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2 or len(pieces[0]) != 64:
            raise ValueError(f"invalid source hash line {line_number}: {raw_line}")
        rows.append((pieces[0].lower(), Path(pieces[1].strip())))
    if not rows:
        raise ValueError(f"empty source hash manifest: {path}")
    return rows


def _source_snapshot(source_root: Path, *, verify_files: bool) -> dict[str, Any]:
    source_bundle = source_root / SOURCE_BUNDLE_FILENAME
    source_summary = source_root / SOURCE_SUMMARY_RELATIVE
    source_hashes = source_root / SOURCE_HASHES_RELATIVE
    for path in (source_bundle, source_summary, source_hashes):
        if not path.is_file():
            raise FileNotFoundError(path)

    bundle = _read_json(source_bundle)
    if bundle.get("schema_version") != SOURCE_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"wrong source bundle schema: {source_bundle}")
    summary = _read_json(source_summary)
    if summary.get("schema_version") != SOURCE_SUMMARY_SCHEMA_VERSION:
        raise ValueError(f"wrong source summary schema: {source_summary}")
    artifact_audit = summary.get("artifact_audit") or {}
    if not bool(artifact_audit.get("passed")):
        raise ValueError("source correction artifact audit did not pass")
    if int(artifact_audit.get("manifests", -1)) != len(LOADS) * len(SEEDS):
        raise ValueError("source manifest count differs from frozen protocol")
    if int(artifact_audit.get("per_arm_outputs", -1)) != (
        len(SOURCE_ARM_KEYS) * len(LOADS) * len(SEEDS)
    ):
        raise ValueError("source arm count differs from frozen protocol")
    if summary.get("frozen_bundle_sha256") != sha256_file(source_bundle):
        raise ValueError("source summary points to a different frozen bundle")

    entries = _parse_source_hashes(source_hashes)
    resolved_source_root = source_root.resolve()
    normalized_entries = []
    for expected_sha, entry_path in entries:
        normalized = entry_path.as_posix()
        try:
            entry_path.resolve().relative_to(resolved_source_root)
        except ValueError as exc:
            raise ValueError(
                f"source hash entry escapes source root: {entry_path}"
            ) from exc
        if verify_files:
            if not entry_path.is_file():
                raise FileNotFoundError(entry_path)
            actual_sha = sha256_file(entry_path)
            if actual_sha != expected_sha:
                raise ValueError(f"source artifact hash mismatch: {entry_path}")
        normalized_entries.append({"sha256": expected_sha, "path": normalized})

    if verify_files:
        _verify_source_bundle(source_bundle)

    return {
        "source_root": source_root.as_posix(),
        "source_bundle_sha256": sha256_file(source_bundle),
        "source_summary_sha256": sha256_file(source_summary),
        "source_hash_manifest_sha256": sha256_file(source_hashes),
        "validated_entry_count": len(normalized_entries),
        "validated_entries_sha256": canonical_sha256(
            {"entries": normalized_entries}
        ),
        "source_artifact_audit": dict(artifact_audit),
    }


def _source_hash_lookup(source_root: Path) -> dict[str, str]:
    return {
        path.resolve().as_posix(): expected_sha
        for expected_sha, path in _parse_source_hashes(
            source_root / SOURCE_HASHES_RELATIVE
        )
    }


def _freeze_bundle(root: Path, source_root: Path) -> dict[str, Any]:
    snapshot = _source_snapshot(source_root, verify_files=True)
    protocol = formal_protocol(source_root)
    artifact_paths = {
        **REGULAR_ARTIFACTS,
        "source_frozen_bundle": (
            source_root / SOURCE_BUNDLE_FILENAME
        ).as_posix(),
        "source_summary": (source_root / SOURCE_SUMMARY_RELATIVE).as_posix(),
        "source_validated_hashes": (
            source_root / SOURCE_HASHES_RELATIVE
        ).as_posix(),
    }
    artifacts: dict[str, dict[str, str]] = {}
    for name, raw_path in artifact_paths.items():
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
        "source_snapshot": snapshot,
        "artifacts": artifacts,
    }
    _write_json_exact(_bundle_path(root), payload)
    return payload


def _verify_bundle(path: Path, *, full_source: bool = False) -> dict[str, Any]:
    bundle = _read_json(path)
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"wrong bundle schema: {path}")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("bundle lacks protocol")
    if canonical_sha256(protocol) != str(bundle.get("protocol_sha256", "")):
        raise ValueError("bundle protocol hash mismatch")
    source_root = Path(str((protocol.get("formal_test") or {}).get("source_root", "")))
    if not source_root.as_posix():
        raise ValueError("bundle protocol lacks source root")

    expected_paths = {
        **REGULAR_ARTIFACTS,
        "source_frozen_bundle": (
            source_root / SOURCE_BUNDLE_FILENAME
        ).as_posix(),
        "source_summary": (source_root / SOURCE_SUMMARY_RELATIVE).as_posix(),
        "source_validated_hashes": (
            source_root / SOURCE_HASHES_RELATIVE
        ).as_posix(),
    }
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("bundle lacks artifacts")
    for name, expected_path in expected_paths.items():
        artifact = artifacts.get(name)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"bundle lacks artifact {name}")
        actual_path = Path(str(artifact.get("path", "")))
        if actual_path.as_posix() != Path(expected_path).as_posix():
            raise ValueError(f"artifact path changed for {name}: {actual_path}")
        if not actual_path.is_file():
            raise FileNotFoundError(actual_path)
        if sha256_file(actual_path) != str(artifact.get("sha256", "")):
            raise ValueError(f"frozen artifact changed: {actual_path}")

    snapshot = _source_snapshot(source_root, verify_files=full_source)
    if snapshot != bundle.get("source_snapshot"):
        raise ValueError("source correction snapshot changed")
    return bundle


def _make_assigner(arm: str) -> PsiDispatchContextWorldModelTaskAssigner:
    spec = NEW_ARM_SPECS[arm]
    config = selector_config(arm)
    return PsiDispatchContextWorldModelTaskAssigner(
        psi_head_checkpoint=str(PSI_HEAD_CHECKPOINT),
        psi_scale_contract=str(PSI_SCALE_CONTRACT),
        psi_context_mode="j_ascending",
        psi_trace_enabled=False,
        psi_trace_max_records=0,
        allow_phasec_s0_robot_scorer=spec["selector"] == "s0",
        checkpoint_path=str(CANDIDATE_CHECKPOINT),
        top_m=TOP_M,
        energy_conv_random_flip_seed=0,
        **config,
    )


def _policy_contract(arm: str) -> dict[str, Any]:
    spec = NEW_ARM_SPECS[arm]
    contract = {
        "arm": arm,
        "label": NEW_ARM_LABELS[arm],
        "checkpoint": str(CANDIDATE_CHECKPOINT),
        "checkpoint_sha256": sha256_file(CANDIDATE_CHECKPOINT),
        "psi_head_checkpoint": Path(PSI_HEAD_CHECKPOINT).as_posix(),
        "psi_head_checkpoint_sha256": sha256_file(PSI_HEAD_CHECKPOINT),
        "psi_scale_contract": Path(PSI_SCALE_CONTRACT).as_posix(),
        "psi_scale_contract_sha256": sha256_file(PSI_SCALE_CONTRACT),
        "context_scheduler": "static_j1",
        "psi_context_mode": "j_ascending",
        "j_refresh": "once_per_proposal_batch",
        "robot_selector": spec["selector"],
        "energy_drift_signal": spec["signal"],
        "robot_selector_config": selector_config(arm),
        "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
        "in_transit_committed_cap": None,
    }
    return {**contract, "fingerprint_sha256": canonical_sha256(contract)}


@contextmanager
def _physical_only_audit_contract() -> Iterator[dict[str, Any]]:
    """Audit the constructor-default physical-only mode without changing it."""

    import WorldModel.evaluation.evaluate_online_v6 as eval_module

    original_builder = eval_module._build_engine
    holder: dict[str, Any] = {}

    def build(cfg, task_assigner=None):
        # Logging-only change: physical-only failed-path warnings can otherwise
        # dominate Slurm output. Simulation decisions are unchanged.
        cfg.simulation.log_level = "ERROR"
        engine = original_builder(cfg, task_assigner=task_assigner)
        modes = {
            queue.admission_mode
            for queue in engine.world.station_state.stations.values()
        }
        if modes != {STATION_ADMISSION_PHYSICAL_ONLY}:
            raise RuntimeError(
                "J1 correction expected constructor-default physical-only "
                f"station admission, got {sorted(modes)}"
            )
        probe = StationAdmissionRestorationAuditProbe(
            engine,
            STATION_ADMISSION_PHYSICAL_ONLY,
            trace_stride=10,
            trace_max_records=0,
        )
        engine.on_tick_callbacks.append(probe.on_tick)
        holder["engine"] = engine
        holder["station_probe"] = probe
        return engine

    eval_module._build_engine = build
    try:
        yield holder
    finally:
        eval_module._build_engine = original_builder


def _max_mapping(value: Any) -> int:
    if not isinstance(value, Mapping) or not value:
        return 0
    return max(int(item) for item in value.values())


def _audit_metrics(
    arm: str,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    station_audit: Mapping[str, Any],
) -> dict[str, Any]:
    spec = NEW_ARM_SPECS[arm]
    signal = spec["signal"]
    checks: dict[str, bool] = {
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "manifest_content_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_count_matches": int(
            metrics.get("order_arrival_count", -1)
        ) == int(manifest.get("total_orders", -2)),
        "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
        "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
        "long_risk_schema_matches": (
            metrics.get("long_risk_schema_version")
            == LONG_RISK_SCHEMA_VERSION
        ),
        "long_risk_contract_matches": (
            metrics.get("long_risk_runtime_contract")
            == long_risk_runtime_contract()
        ),
        "j1_mode": metrics.get("psi_dispatch_mode") == "j_ascending",
        "j1_head_loaded": bool(metrics.get("psi_dispatch_head_loaded")),
        "j1_evaluated": int(metrics.get("psi_dispatch_eval_calls", 0)) > 0,
        "j1_contexts_seen": int(metrics.get("psi_dispatch_contexts_seen", 0)) > 0,
        "j1_parent_robot_scorer_preserved": (
            metrics.get("psi_dispatch_robot_scorer")
            == "WorldModelTaskAssigner.select_robots_unmodified"
        ),
        "j1_head_hash": (
            metrics.get("psi_dispatch_head_checkpoint_sha256")
            == sha256_file(PSI_HEAD_CHECKPOINT)
        ),
        "j1_scale_hash": (
            metrics.get("psi_dispatch_scale_contract_sha256")
            == sha256_file(PSI_SCALE_CONTRACT)
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
    if signal is None:
        checks.update({
            "s0_energy_mode_off": metrics.get("energy_scoring_mode") == "off",
            "s0_no_conversion_contexts": int(
                metrics.get("energy_conv_contexts", 0)
            ) == 0,
            "s0_j1_variant": (
                metrics.get("psi_dispatch_robot_scorer_variant")
                == "phasec_s0"
            ),
            "s0_not_reported_as_s1": not bool(
                metrics.get("psi_dispatch_s1_within_context")
            ),
        })
    else:
        checks.update({
            "s1_energy_mode_conversion": (
                metrics.get("energy_scoring_mode") == "conversion"
            ),
            "s1_contexts_observed": int(
                metrics.get("energy_conv_contexts", 0)
            ) > 0,
            "s1_signal_matches": metrics.get("energy_drift_signal") == signal,
            "s1_j1_variant": (
                metrics.get("psi_dispatch_robot_scorer_variant")
                == "s1_within_context"
            ),
            "s1_reported": bool(
                metrics.get("psi_dispatch_s1_within_context")
            ),
        })
    return {"passed": all(checks.values()), "checks": checks}


def _resume_ok(
    path: Path,
    *,
    arm: str,
    load: str,
    seed: int,
    ticks: int,
    bundle_sha256: str,
    protocol_sha256: str,
    policy_fingerprint: str,
    manifest_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    payload = _read_json(path)
    meta = payload.get("meta") or {}
    manifest = payload.get("manifest") or {}
    checks = {
        "schema": payload.get("schema_version") == PER_SEED_SCHEMA_VERSION,
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
        "physical_only": (
            meta.get("station_admission")
            == STATION_ADMISSION_PHYSICAL_ONLY
        ),
        "audit": bool((payload.get("audit") or {}).get("passed")),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"cannot resume incompatible output {path}: {failed}")
    print(f"[resume] arm={arm} load={load} seed={seed}: {path}")
    return True


def _run_arm(args: argparse.Namespace) -> None:
    if args.arm not in NEW_ARM_KEYS:
        raise ValueError(args.arm)
    if args.load not in LOADS:
        raise ValueError(args.load)
    if int(args.seed) not in SEEDS:
        raise ValueError(f"formal seed must be one of {list(SEEDS)}")
    if int(args.ticks) != TICKS:
        raise ValueError(f"formal ticks must equal {TICKS}")

    root = Path(args.output_root)
    bundle_path = Path(args.frozen_bundle or _bundle_path(root))
    bundle = _verify_bundle(bundle_path, full_source=False)
    protocol_sha256 = str(bundle["protocol_sha256"])
    bundle_sha256 = sha256_file(bundle_path)
    source_root = Path(
        str(bundle["protocol"]["formal_test"]["source_root"])
    )
    manifest_path, _ = _source_paths(source_root, args.load, int(args.seed))
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _read_json(manifest_path)
    expected_hashes = _source_hash_lookup(source_root)
    expected_file_sha = expected_hashes.get(manifest_path.resolve().as_posix())
    if expected_file_sha is None:
        raise ValueError(f"manifest absent from source validation: {manifest_path}")
    if sha256_file(manifest_path) != expected_file_sha:
        raise ValueError(f"source manifest changed: {manifest_path}")

    policy_contract = _policy_contract(args.arm)
    output_path = _new_output_path(
        root, args.arm, args.load, int(args.seed)
    )
    if _resume_ok(
        output_path,
        arm=args.arm,
        load=args.load,
        seed=int(args.seed),
        ticks=int(args.ticks),
        bundle_sha256=bundle_sha256,
        protocol_sha256=protocol_sha256,
        policy_fingerprint=str(policy_contract["fingerprint_sha256"]),
        manifest_sha256=str(manifest.get("manifest_sha256", "")),
    ):
        return

    assigner = _make_assigner(args.arm)
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    print(
        f"[run] arm={args.arm} load={args.load} seed={args.seed} "
        f"ticks={args.ticks} admission={STATION_ADMISSION_PHYSICAL_ONLY}"
    )
    with _physical_only_audit_contract() as holder:
        metrics = _run_one_assigner(
            LOAD_CONFIGS[args.load],
            assigner,
            int(args.seed),
            int(args.ticks),
            trace_label=NEW_ARM_LABELS[args.arm],
            recorded_orders_path=str(manifest_path),
        )
    metrics.update(assigner.psi_dispatch_metrics())
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
    audit = _audit_metrics(args.arm, metrics, manifest, station_audit)
    if not audit["passed"]:
        failed = [
            name for name, passed in audit["checks"].items() if not passed
        ]
        raise RuntimeError(
            f"audit failed for {args.arm} {args.load} seed={args.seed}: "
            + ", ".join(failed)
        )

    payload = {
        "schema_version": PER_SEED_SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": protocol_sha256,
            "frozen_bundle": bundle_path.as_posix(),
            "frozen_bundle_sha256": bundle_sha256,
            "formal": True,
            "arm_key": args.arm,
            "arm": NEW_ARM_LABELS[args.arm],
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "top_m": TOP_M,
            "config": LOAD_CONFIGS[args.load],
            "checkpoint": str(CANDIDATE_CHECKPOINT),
            "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
            "physical_only_mode_source": "StationQueue constructor default",
            "policy_contract": policy_contract,
            "energy_drift_signal": NEW_ARM_SPECS[args.arm]["signal"],
            "long_risk_runtime_contract": long_risk_runtime_contract(),
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
            "source_validation_sha256": expected_file_sha,
        },
        "station_audit": station_audit,
        "audit": audit,
        "metrics": metrics,
    }
    _write_json_exact(output_path, payload)
    print(f"[done] {output_path}")


def interaction_delta_from_values(
    s0_j0: float,
    s0_j1: float,
    s1_j0: float,
    s1_j1: float,
) -> float:
    return (float(s1_j1) - float(s1_j0)) - (
        float(s0_j1) - float(s0_j0)
    )


def _load_combined_rows(
    root: Path,
    source_root: Path,
    bundle: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, int], dict[str, Mapping[str, Any]]],
    list[Path],
    dict[str, bool],
]:
    protocol_sha256 = str(bundle["protocol_sha256"])
    bundle_sha256 = sha256_file(_bundle_path(root))
    rows: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    input_files: list[Path] = [
        _bundle_path(root),
        source_root / SOURCE_BUNDLE_FILENAME,
        source_root / SOURCE_SUMMARY_RELATIVE,
        source_root / SOURCE_HASHES_RELATIVE,
    ]
    checks: dict[str, bool] = {}
    for load in LOADS:
        for seed in SEEDS:
            manifest_path, source_outputs = _source_paths(
                source_root, load, seed
            )
            manifest = _read_json(manifest_path)
            input_files.append(manifest_path)
            arm_metrics: dict[str, Mapping[str, Any]] = {}
            for arm in SOURCE_ARM_KEYS:
                path = source_outputs[arm]
                payload = _read_json(path)
                input_files.append(path)
                meta = payload.get("meta") or {}
                metrics = payload.get("metrics") or {}
                key = f"source_{arm}_{load}_{seed}"
                checks[f"{key}_identity"] = (
                    payload.get("schema_version")
                    == SOURCE_PER_SEED_SCHEMA_VERSION
                    and meta.get("arm_key") == arm
                    and meta.get("load") == load
                    and int(meta.get("seed", -1)) == seed
                    and int(meta.get("ticks", -1)) == TICKS
                )
                checks[f"{key}_audit"] = bool(
                    (payload.get("audit") or {}).get("passed")
                )
                checks[f"{key}_manifest"] = (
                    metrics.get("order_arrival_manifest_sha256")
                    == manifest.get("manifest_sha256")
                    and int(metrics.get("order_arrival_count", -1))
                    == int(manifest.get("total_orders", -2))
                )
                arm_metrics[arm] = metrics

            for arm in NEW_ARM_KEYS:
                path = _new_output_path(root, arm, load, seed)
                payload = _read_json(path)
                input_files.append(path)
                meta = payload.get("meta") or {}
                metrics = payload.get("metrics") or {}
                key = f"new_{arm}_{load}_{seed}"
                checks[f"{key}_identity"] = (
                    payload.get("schema_version") == PER_SEED_SCHEMA_VERSION
                    and meta.get("protocol_sha256") == protocol_sha256
                    and meta.get("frozen_bundle_sha256") == bundle_sha256
                    and meta.get("arm_key") == arm
                    and meta.get("load") == load
                    and int(meta.get("seed", -1)) == seed
                    and int(meta.get("ticks", -1)) == TICKS
                    and meta.get("station_admission")
                    == STATION_ADMISSION_PHYSICAL_ONLY
                )
                checks[f"{key}_audit"] = bool(
                    (payload.get("audit") or {}).get("passed")
                )
                checks[f"{key}_station_audit"] = bool(
                    (payload.get("station_audit") or {}).get("passed")
                )
                checks[f"{key}_manifest"] = (
                    metrics.get("order_arrival_manifest_sha256")
                    == manifest.get("manifest_sha256")
                    and int(metrics.get("order_arrival_count", -1))
                    == int(manifest.get("total_orders", -2))
                )
                arm_metrics[arm] = metrics
            rows[(load, seed)] = arm_metrics
    return rows, input_files, checks


def _paired_metric_report(
    rows: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
    *,
    baseline: str,
    candidate: str,
    metric: str,
    seed_offset: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    per_seed_cluster: list[float] = []
    for seed in SEEDS:
        values = []
        for load in LOADS:
            left = _number(rows[(load, seed)][baseline].get(metric))
            right = _number(rows[(load, seed)][candidate].get(metric))
            if left is not None and right is not None:
                values.append(right - left)
        if len(values) == len(LOADS):
            per_seed_cluster.append(sum(values) / len(values))
    for load_index, load in enumerate(LOADS):
        values = []
        for seed in SEEDS:
            left = _number(rows[(load, seed)][baseline].get(metric))
            right = _number(rows[(load, seed)][candidate].get(metric))
            if left is not None and right is not None:
                values.append(right - left)
        summary = _mean_std(values)
        summary["ci95_low"], summary["ci95_high"] = _bootstrap_ci(
            values, seed_offset + load_index
        )
        report[load] = summary
        csv_rows.append({"scope": load, **summary})
    overall = _mean_std(per_seed_cluster)
    overall["ci95_low"], overall["ci95_high"] = _bootstrap_ci(
        per_seed_cluster, seed_offset + 90
    )
    report["overall_seed_cluster"] = overall
    csv_rows.append({"scope": "overall_seed_cluster", **overall})
    return report, csv_rows


def _interaction_metric_report(
    rows: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
    *,
    s0_j0: str,
    s0_j1: str,
    s1_j0: str,
    s1_j1: str,
    metric: str,
    seed_offset: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    def delta(load: str, seed: int) -> float | None:
        values = [
            _number(rows[(load, seed)][arm].get(metric))
            for arm in (s0_j0, s0_j1, s1_j0, s1_j1)
        ]
        if any(value is None for value in values):
            return None
        return interaction_delta_from_values(*values)  # type: ignore[arg-type]

    report: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    per_seed_cluster = []
    for seed in SEEDS:
        values = [delta(load, seed) for load in LOADS]
        if all(value is not None for value in values):
            per_seed_cluster.append(
                sum(float(value) for value in values) / len(values)
            )
    for load_index, load in enumerate(LOADS):
        values = [
            value
            for seed in SEEDS
            if (value := delta(load, seed)) is not None
        ]
        summary = _mean_std(values)
        summary["ci95_low"], summary["ci95_high"] = _bootstrap_ci(
            values, seed_offset + load_index
        )
        report[load] = summary
        csv_rows.append({"scope": load, **summary})
    overall = _mean_std(per_seed_cluster)
    overall["ci95_low"], overall["ci95_high"] = _bootstrap_ci(
        per_seed_cluster, seed_offset + 90
    )
    report["overall_seed_cluster"] = overall
    csv_rows.append({"scope": "overall_seed_cluster", **overall})
    return report, csv_rows


def _summarize(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    bundle_path = Path(args.frozen_bundle or _bundle_path(root))
    bundle = _verify_bundle(bundle_path, full_source=True)
    source_root = Path(
        str(bundle["protocol"]["formal_test"]["source_root"])
    )
    rows, input_files, audit_checks = _load_combined_rows(
        root, source_root, bundle
    )
    if not all(audit_checks.values()):
        failed = [name for name, passed in audit_checks.items() if not passed]
        raise RuntimeError("summary audit failed: " + ", ".join(failed[:30]))

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_json: dict[str, Any] = {}
    for arm in ALL_ARM_KEYS:
        aggregate_json[arm] = {}
        for load in LOADS:
            aggregate_json[arm][load] = {}
            for metric in REPORT_METRICS:
                values = [
                    value
                    for seed in SEEDS
                    if (
                        value := _number(rows[(load, seed)][arm].get(metric))
                    ) is not None
                ]
                summary = _mean_std(values)
                aggregate_json[arm][load][metric] = summary
                aggregate_rows.append({
                    "arm_key": arm,
                    "arm": ALL_ARM_LABELS[arm],
                    "load": load,
                    "metric": metric,
                    **summary,
                })

    contrast_rows: list[dict[str, Any]] = []
    contrast_json: dict[str, Any] = {}
    for comparison_index, (name, baseline, candidate) in enumerate(COMPARISONS):
        comparison_report = {
            "baseline": baseline,
            "candidate": candidate,
            "definition": "candidate_minus_baseline",
            "metrics": {},
        }
        for metric_index, metric in enumerate(REPORT_METRICS):
            report, csv_rows = _paired_metric_report(
                rows,
                baseline=baseline,
                candidate=candidate,
                metric=metric,
                seed_offset=(
                    2_000_000
                    + comparison_index * 100_000
                    + metric_index * 100
                ),
            )
            comparison_report["metrics"][metric] = report
            for row in csv_rows:
                contrast_rows.append({
                    "comparison": name,
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    **row,
                })
        contrast_json[name] = comparison_report

    interaction_rows: list[dict[str, Any]] = []
    interaction_json: dict[str, Any] = {}
    for interaction_index, (
        name,
        s0_j0,
        s0_j1,
        s1_j0,
        s1_j1,
    ) in enumerate(INTERACTIONS):
        interaction_report = {
            "definition": "(S1+J1-S1+J0)-(S0+J1-S0+J0)",
            "s0_j0": s0_j0,
            "s0_j1": s0_j1,
            "s1_j0": s1_j0,
            "s1_j1": s1_j1,
            "metrics": {},
        }
        for metric_index, metric in enumerate(REPORT_METRICS):
            report, csv_rows = _interaction_metric_report(
                rows,
                s0_j0=s0_j0,
                s0_j1=s0_j1,
                s1_j0=s1_j0,
                s1_j1=s1_j1,
                metric=metric,
                seed_offset=(
                    8_000_000
                    + interaction_index * 100_000
                    + metric_index * 100
                ),
            )
            interaction_report["metrics"][metric] = report
            for row in csv_rows:
                interaction_rows.append({
                    "interaction": name,
                    "s0_j0": s0_j0,
                    "s0_j1": s0_j1,
                    "s1_j0": s1_j0,
                    "s1_j1": s1_j1,
                    "metric": metric,
                    **row,
                })
        interaction_json[name] = interaction_report

    mechanism = {}
    for arm in NEW_ARM_KEYS:
        mechanism[arm] = {
            "psi_contexts_seen": sum(
                int(rows[(load, seed)][arm].get("psi_dispatch_contexts_seen", 0))
                for load in LOADS for seed in SEEDS
            ),
            "psi_replacement_count": sum(
                int(rows[(load, seed)][arm].get("psi_dispatch_replacement_count", 0))
                for load in LOADS for seed in SEEDS
            ),
            "energy_contexts": sum(
                int(rows[(load, seed)][arm].get("energy_conv_contexts", 0))
                for load in LOADS for seed in SEEDS
            ),
            "energy_active_contexts": sum(
                int(rows[(load, seed)][arm].get("energy_conv_active_contexts", 0))
                for load in LOADS for seed in SEEDS
            ),
            "energy_modified_decisions": sum(
                int(rows[(load, seed)][arm].get("energy_conv_modified_decisions", 0))
                for load in LOADS for seed in SEEDS
            ),
            "max_physical_occupancy": max(
                int(rows[(load, seed)][arm].get("station_max_occupancy", 0))
                for load in LOADS for seed in SEEDS
            ),
            "max_committed_load": max(
                int(rows[(load, seed)][arm].get("station_max_committed_load", 0))
                for load in LOADS for seed in SEEDS
            ),
        }

    validation_dir = root / "validation"
    aggregate_csv = validation_dir / "aggregate_summary.csv"
    contrast_csv = validation_dir / "paired_contrasts.csv"
    interaction_csv = validation_dir / "interaction_contrasts.csv"
    summary_json = validation_dir / "s1_j1_correction_summary.json"
    _write_text_exact(
        aggregate_csv,
        _csv_text(
            aggregate_rows,
            ["arm_key", "arm", "load", "metric", "n", "mean", "std"],
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
        interaction_csv,
        _csv_text(
            interaction_rows,
            [
                "interaction", "s0_j0", "s0_j1", "s1_j0", "s1_j1",
                "metric", "scope", "n", "mean", "std", "ci95_low",
                "ci95_high",
            ],
        ),
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "protocol_sha256": bundle["protocol_sha256"],
        "frozen_bundle_sha256": sha256_file(bundle_path),
        "source_snapshot": bundle["source_snapshot"],
        "artifact_audit": {
            "passed": True,
            "checks": len(audit_checks),
            "manifests_reused": len(LOADS) * len(SEEDS),
            "source_outputs_reused": (
                len(SOURCE_ARM_KEYS) * len(LOADS) * len(SEEDS)
            ),
            "new_outputs": len(NEW_ARM_KEYS) * len(LOADS) * len(SEEDS),
            "combined_outputs": len(ALL_ARM_KEYS) * len(LOADS) * len(SEEDS),
        },
        "station_admission": {
            "mode": STATION_ADMISSION_PHYSICAL_ONLY,
            "physical_occupancy_enforced": True,
            "in_transit_committed_cap": False,
            "station_queue": False,
        },
        "long_risk_runtime_contract": long_risk_runtime_contract(),
        "mechanism": mechanism,
        "aggregate": aggregate_json,
        "comparisons": contrast_json,
        "interactions": interaction_json,
    }
    _write_json_exact(summary_json, summary)

    validated_files = sorted(
        set(input_files + [aggregate_csv, contrast_csv, interaction_csv, summary_json]),
        key=lambda path: path.as_posix(),
    )
    _write_text_exact(
        validation_dir / "validated_outputs.sha256",
        "".join(
            f"{sha256_file(path)}  {path.as_posix()}\n"
            for path in validated_files
        ),
    )
    print(f"[complete] summary={summary_json}")
    print(f"[complete] aggregate={aggregate_csv}")
    print(f"[complete] contrasts={contrast_csv}")
    print(f"[complete] interactions={interaction_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "arm", "summary"), required=True)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--frozen-bundle", default=None)
    parser.add_argument("--arm", choices=NEW_ARM_KEYS)
    parser.add_argument("--load", choices=LOADS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int, default=TICKS)
    args = parser.parse_args()

    root = Path(args.output_root)
    if args.mode == "freeze":
        bundle = _freeze_bundle(root, Path(args.source_root))
        print(f"[complete] bundle={_bundle_path(root)}")
        print(f"[complete] protocol_sha256={bundle['protocol_sha256']}")
        return
    if args.mode == "arm":
        if args.arm is None or args.load is None or args.seed is None:
            parser.error("--mode arm requires --arm, --load, and --seed")
        _run_arm(args)
        return
    _summarize(args)


if __name__ == "__main__":
    main()
