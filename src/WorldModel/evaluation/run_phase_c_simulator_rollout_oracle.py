"""Run the direct Phase-C simulator-rollout candidate oracle.

The policy under test does not load a World Model checkpoint.  It keeps the
original fixed-context prefix order and ranks every currently idle robot by
the exact H=10 isolated counterfactual rollout cost used to supervise Phase C.
The experiment replays the frozen high-load 551--560 order manifests under
physical-only and committed-V1 station admission.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from Policies.TaskAssigner.SimulatorRolloutTaskAssigner import (
    SimulatorRolloutTaskAssigner,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    LOADS,
    canonical_sha256,
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_psi_dynamic_probe import (
    _artifact,
    _atomic_json,
    _load_bundle,
    _read_json,
)
from WorldModel.evaluation.run_phase_c_station_admission_restoration import (
    BASE_ROOT,
    DEFAULT_BUNDLE,
    DEFAULT_SOURCE_ROOT,
    DEFAULT_V1_REFERENCE_ROOT,
    DEFAULT_V3_REFERENCE_ROOT,
    FORMAL_SEEDS,
    FORMAL_TICKS,
    _aggregate_arm,
    _engine_contract,
    _manifest_path,
    _paired_comparison,
    _validate_historical_contract,
    _validate_manifest,
)
from WorldState.station_state import (
    STATION_ADMISSION_COMMITTED_V1,
    STATION_ADMISSION_PHYSICAL_ONLY,
)


SCHEMA_VERSION = "phase_c_simulator_rollout_oracle_arm_v1"
SUMMARY_SCHEMA_VERSION = "phase_c_simulator_rollout_oracle_summary_v1"
FORMAL_LOAD = "high"
FORMAL_HORIZON = 10
FORMAL_CANDIDATE_LIMIT = 0
FORMAL_MAX_CONTEXTS_PER_TICK = 0
FORMAL_RESERVATION_WINDOW = 1

DEFAULT_OUTPUT_ROOT = (
    BASE_ROOT / "phase_c_simulator_rollout_oracle_high_551_560_v1"
)


@dataclass(frozen=True)
class ArmSpec:
    key: str
    label: str
    admission_mode: str


ARM_SPECS = {
    spec.key: spec
    for spec in (
        ArmSpec(
            "sim_oracle_physical_only",
            "H10 simulator oracle + physical-only",
            STATION_ADMISSION_PHYSICAL_ONLY,
        ),
        ArmSpec(
            "sim_oracle_committed_v1",
            "H10 simulator oracle + committed V1",
            STATION_ADMISSION_COMMITTED_V1,
        ),
    )
}
ARM_KEYS = tuple(ARM_SPECS)


def _output_path(root: Path, arm: str, load: str, seed: int) -> Path:
    return root / "per_arm" / arm / f"{load}_seed{seed}.json"


def _runtime_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "oracle_assigner": Path(
            "Policies/TaskAssigner/SimulatorRolloutTaskAssigner/"
            "simulator_rollout_task_assigner.py"
        ),
        "counterfactual_rollout": Path(
            "WorldModel/data/counterfactual_rollout.py"
        ),
        "realized_cost": Path("WorldModel/core/costs.py"),
        "context_assignment": Path(
            "Policies/TaskAssigner/context_assignment.py"
        ),
        "station_state": Path("WorldState/station_state.py"),
        "simulation_engine": Path("Engine/simulation_engine.py"),
        "evaluate_online": Path(
            "WorldModel/evaluation/evaluate_online_v6.py"
        ),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _load_inputs(args: argparse.Namespace):
    bundle, protocol = _load_bundle(args.frozen_bundle)
    # This experiment intentionally resolves only the simulator config.  It
    # does not resolve or load the model/head checkpoint artifacts.
    config_path = _artifact(bundle, f"config_{args.load}")
    return bundle, protocol, config_path


def _policy_contract(
    args: argparse.Namespace,
    runtime_hashes: Mapping[str, str],
) -> dict[str, Any]:
    contract = {
        "policy": "SimulatorRolloutTaskAssigner",
        "uses_world_model": False,
        "loads_checkpoint": False,
        "context_scheduler": "prefix_j0",
        "candidate_scope": (
            "all_idle"
            if int(args.candidate_limit) == 0
            else f"nearest_{int(args.candidate_limit)}"
        ),
        "rollout_horizon": int(args.rollout_horizon),
        "rollout_continuation_mode": "isolated",
        "future_orders_in_rollout": False,
        "continuation_assigner_in_rollout": False,
        "max_contexts_per_tick": int(args.max_contexts_per_tick),
        "reservation_window": int(args.reservation_window),
        "realized_cost": (
            "Phase-C discounted 7-channel label cost; deadlock uses max"
        ),
        "policy_code_sha256": runtime_hashes["oracle_assigner"],
        "counterfactual_code_sha256": runtime_hashes[
            "counterfactual_rollout"
        ],
        "realized_cost_code_sha256": runtime_hashes["realized_cost"],
    }
    return {
        **contract,
        "fingerprint_sha256": canonical_sha256(contract),
    }


def _make_assigner(args: argparse.Namespace) -> SimulatorRolloutTaskAssigner:
    return SimulatorRolloutTaskAssigner(
        rollout_horizon=int(args.rollout_horizon),
        candidate_limit=int(args.candidate_limit),
        max_contexts_per_tick=int(args.max_contexts_per_tick),
        reservation_window=int(args.reservation_window),
        decision_trace_enabled=int(args.decision_trace_max_records) > 0,
        decision_trace_max_records=int(args.decision_trace_max_records),
    )


def _formal_argument_checks(args: argparse.Namespace) -> None:
    if args.development:
        return
    expected = {
        "load": (args.load, FORMAL_LOAD),
        "ticks": (int(args.ticks), FORMAL_TICKS),
        "rollout_horizon": (
            int(args.rollout_horizon), FORMAL_HORIZON
        ),
        "candidate_limit": (
            int(args.candidate_limit), FORMAL_CANDIDATE_LIMIT
        ),
        "max_contexts_per_tick": (
            int(args.max_contexts_per_tick),
            FORMAL_MAX_CONTEXTS_PER_TICK,
        ),
        "reservation_window": (
            int(args.reservation_window), FORMAL_RESERVATION_WINDOW
        ),
    }
    failed = [
        f"{name}={actual!r} (expected {target!r})"
        for name, (actual, target) in expected.items()
        if actual != target
    ]
    if failed:
        raise ValueError(
            "formal simulator-oracle contract mismatch: " + "; ".join(failed)
        )
    if args.mode in ("manifest", "arm") and int(args.seed) not in FORMAL_SEEDS:
        raise ValueError(
            f"formal seed must be in {list(FORMAL_SEEDS)}: {args.seed}"
        )


def _manifest_contract(args: argparse.Namespace) -> tuple[dict, dict, Path]:
    manifest_path = _manifest_path(args.source_root, args.load, int(args.seed))
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _validate_manifest(manifest_path)
    historical = _validate_historical_contract(
        manifest,
        v1_root=args.manifest_reference_root,
        v3_root=args.v3_reference_root,
        load=args.load,
        seed=int(args.seed),
        require_v3=bool(args.require_v3_reference),
        skip=bool(args.skip_historical_contract),
    )
    return manifest, historical, manifest_path


def _prepare_manifest(args: argparse.Namespace) -> None:
    _load_inputs(args)
    manifest, historical, manifest_path = _manifest_contract(args)
    print(json.dumps({
        "manifest": manifest_path.as_posix(),
        "content_sha256": manifest.get("manifest_sha256"),
        "total_orders": manifest.get("total_orders"),
        "historical_contract": historical,
    }, indent=2, ensure_ascii=False))


def _resume_compatible(
    path: Path,
    *,
    args: argparse.Namespace,
    spec: ArmSpec,
    bundle_sha: str,
    runtime_hashes: Mapping[str, str],
    policy_fingerprint: str,
    manifest_sha: str,
) -> bool:
    if not path.is_file():
        return False
    payload = _read_json(path)
    meta = payload.get("meta") or {}
    policy = meta.get("policy_contract") or {}
    checks = {
        "schema": payload.get("schema_version") == SCHEMA_VERSION,
        "arm": meta.get("arm_key") == spec.key,
        "load": meta.get("load") == args.load,
        "seed": int(meta.get("seed", -1)) == int(args.seed),
        "ticks": int(meta.get("ticks", -1)) == int(args.ticks),
        "formal": bool(meta.get("formal")) == (not args.development),
        "bundle": meta.get("frozen_bundle_sha256") == bundle_sha,
        "admission": meta.get("station_admission") == spec.admission_mode,
        "runtime": meta.get("runtime_code_sha256") == dict(runtime_hashes),
        "policy": policy.get("fingerprint_sha256") == policy_fingerprint,
        "manifest": (payload.get("manifest") or {}).get(
            "content_sha256"
        ) == manifest_sha,
        "audit": bool((payload.get("audit") or {}).get("passed")),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"cannot resume incompatible output {path}: {failed}")
    print(f"[resume] {spec.key} {args.load} seed={args.seed}: {path}")
    return True


def _audit(
    *,
    metrics: Mapping[str, Any],
    oracle: Mapping[str, Any],
    manifest: Mapping[str, Any],
    station: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    checks = {
        "manifest_hash": metrics.get("order_arrival_manifest_sha256")
        == manifest.get("manifest_sha256"),
        "manifest_count": int(metrics.get("order_arrival_count", -1))
        == int(manifest.get("total_orders", -2)),
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "station_admission": bool(station.get("passed")),
        "no_world_model": oracle.get("uses_world_model") is False,
        "no_checkpoint": oracle.get("loads_checkpoint") is False,
        "isolated_rollout": oracle.get("rollout_continuation_mode")
        == "isolated",
        "no_future_orders": oracle.get("future_orders_in_rollout") is False,
        "no_continuation_assigner": oracle.get(
            "continuation_assigner_in_rollout"
        ) is False,
        "complete_rollouts": int(oracle.get("complete_rollouts", -1))
        == int(oracle.get("rollout_calls", -2)),
        "zero_incomplete_rollouts": int(
            oracle.get("incomplete_rollouts", -1)
        ) == 0,
        "rollouts_exercised": int(oracle.get("rollout_calls", 0)) > 0,
        "contexts_selected": int(oracle.get("contexts_selected", 0)) > 0,
        "selection_commit_match": int(
            oracle.get("contexts_selected", -1)
        ) == int(oracle.get("contexts_committed", -2)),
        "horizon": int(oracle.get("rollout_horizon", -1))
        == int(args.rollout_horizon),
        "candidate_scope": oracle.get("candidate_scope")
        == (
            "all_idle"
            if int(args.candidate_limit) == 0
            else f"nearest_{int(args.candidate_limit)}"
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }


def _run_arm(args: argparse.Namespace) -> None:
    if args.arm not in ARM_SPECS:
        raise ValueError(f"unknown --arm: {args.arm}")
    spec = ARM_SPECS[args.arm]
    _bundle, protocol, config_path = _load_inputs(args)
    manifest, historical, manifest_path = _manifest_contract(args)
    runtime_hashes = _runtime_hashes()
    policy_contract = _policy_contract(args, runtime_hashes)
    bundle_sha = sha256_file(args.frozen_bundle)
    output = _output_path(
        args.output_root, spec.key, args.load, int(args.seed)
    )
    if _resume_compatible(
        output,
        args=args,
        spec=spec,
        bundle_sha=bundle_sha,
        runtime_hashes=runtime_hashes,
        policy_fingerprint=policy_contract["fingerprint_sha256"],
        manifest_sha=str(manifest["manifest_sha256"]),
    ):
        return

    assigner = _make_assigner(args)
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    print(
        f"[run] arm={spec.key} load={args.load} seed={args.seed} "
        f"ticks={args.ticks} H={args.rollout_horizon} "
        f"candidate_limit={args.candidate_limit}"
    )
    with _engine_contract(
        spec.admission_mode,
        dynamic_config={},
        station_trace_stride=int(args.station_trace_stride),
        station_trace_max_records=int(args.station_trace_max_records),
        fifo_trace_max_records=0,
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
    if station_probe is None:
        raise RuntimeError("station admission audit probe was not attached")
    station_audit = station_probe.summary()
    oracle_metrics = assigner.oracle_metrics()
    metrics.update({
        "online_robot_candidate_scope": oracle_metrics["candidate_scope"],
        "station_admission_mode": spec.admission_mode,
        "station_capacity_rejections": station_audit.get(
            "capacity_rejections"
        ),
        "station_over_capacity_grants": station_audit.get(
            "over_capacity_grants"
        ),
    })
    audit = _audit(
        metrics=metrics,
        oracle=oracle_metrics,
        manifest=manifest,
        station=station_audit,
        args=args,
    )
    if not audit["passed"]:
        raise RuntimeError(
            f"simulator oracle audit failed {spec.key} {args.load} "
            f"seed={args.seed}: {audit['failed']}"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": protocol.get("protocol_sha256"),
            "frozen_bundle": args.frozen_bundle.as_posix(),
            "frozen_bundle_sha256": bundle_sha,
            "arm_key": spec.key,
            "arm_label": spec.label,
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "formal": not bool(args.development),
            "station_admission": spec.admission_mode,
            "policy_contract": policy_contract,
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
        "historical_manifest_contract": historical,
        "audit": audit,
        "station_admission_audit": station_audit,
        "oracle_metrics": oracle_metrics,
        "decision_trace": assigner.decision_trace_records,
        "metrics": metrics,
    }
    _atomic_json(output, payload)
    print(f"[done] {output}")
    print(json.dumps({
        "completed_orders": metrics.get("completed_orders"),
        "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
        "wall_time_s": metrics.get("wall_time_s"),
        "oracle_rollout_calls": oracle_metrics.get("rollout_calls"),
        "oracle_rollout_time_ms_mean": oracle_metrics.get(
            "rollout_time_ms_mean"
        ),
        "oracle_selected_nearest_ratio": oracle_metrics.get(
            "selected_nearest_ratio"
        ),
    }, indent=2, ensure_ascii=False))


def _weighted_mean(
    payloads: Sequence[Mapping[str, Any]],
    value_key: str,
    weight_key: str,
) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for payload in payloads:
        row = payload.get("oracle_metrics") or {}
        value = row.get(value_key)
        weight = row.get(weight_key)
        if value is None or weight is None:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    return round(numerator / denominator, 6) if denominator > 0 else None


def _aggregate_oracle(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [payload.get("oracle_metrics") or {} for payload in payloads]
    rollout_calls = sum(int(row.get("rollout_calls", 0)) for row in rows)
    selected = sum(int(row.get("contexts_selected", 0)) for row in rows)
    committed = sum(int(row.get("contexts_committed", 0)) for row in rows)
    nearest = sum(int(row.get("selected_nearest_count", 0)) for row in rows)
    return {
        "run_count": len(rows),
        "rollout_calls_sum": rollout_calls,
        "complete_rollouts_sum": sum(
            int(row.get("complete_rollouts", 0)) for row in rows
        ),
        "incomplete_rollouts_sum": sum(
            int(row.get("incomplete_rollouts", 0)) for row in rows
        ),
        "contexts_selected_sum": selected,
        "contexts_committed_sum": committed,
        "candidates_per_context_weighted_mean": _weighted_mean(
            payloads,
            "candidates_per_context_mean",
            "contexts_scored",
        ),
        "rollout_time_ms_weighted_mean": _weighted_mean(
            payloads,
            "rollout_time_ms_mean",
            "rollout_calls",
        ),
        "selected_cost_weighted_mean": _weighted_mean(
            payloads,
            "selected_cost_mean",
            "contexts_selected",
        ),
        "selected_nearest_count_sum": nearest,
        "selected_nearest_ratio": round(nearest / max(selected, 1), 6),
        "exact_tie_contexts_sum": sum(
            int(row.get("exact_tie_contexts", 0)) for row in rows
        ),
        "selected_rollout_blocked_moves_sum": sum(
            int(row.get("selected_rollout_blocked_moves", 0))
            for row in rows
        ),
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Phase-C direct simulator-rollout oracle",
        "",
        (
            "This experiment ranks robots with the exact H=10 isolated "
            "counterfactual simulator cost. It loads no World Model "
            "checkpoint and keeps the original prefix/J0 context order."
        ),
        "",
        (
            "It is a candidate-ranking oracle for the Phase-C supervision "
            "target, not full closed-loop MPC: future orders and continuation "
            "task assignment are disabled inside each rollout."
        ),
        "",
        "| Admission | Throughput | Deadlock | Stall | Wall time sum (s) | Rollouts | ms/rollout | Nearest match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_KEYS:
        aggregate = summary["arms"][arm]
        oracle = summary["oracle"][arm]
        lines.append(
            f"| {ARM_SPECS[arm].label} | "
            f"{aggregate.get('completed_orders_mean')} | "
            f"{aggregate.get('deadlock_ratio_mean')} | "
            f"{aggregate.get('stall_ratio_mean')} | "
            f"{aggregate.get('wall_time_s_sum')} | "
            f"{oracle.get('rollout_calls_sum')} | "
            f"{oracle.get('rollout_time_ms_weighted_mean')} | "
            f"{oracle.get('selected_nearest_ratio')} |"
        )
    comparison = summary["comparisons"][
        "physical_only_minus_committed_v1"
    ]
    wtl = comparison["throughput_wins_ties_losses"]
    lines.extend([
        "",
        "## Paired admission contrast",
        "",
        (
            "- Physical-only minus committed V1 completed orders: "
            f"{comparison.get('completed_orders_delta_mean')}."
        ),
        (
            "- Throughput wins/ties/losses: "
            f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']}."
        ),
        (
            "- Deadlock-ratio delta: "
            f"{comparison.get('deadlock_ratio_delta_mean')}."
        ),
        "",
        "## Integrity",
        "",
        f"- Manifest pairing passed: {summary['integrity']['manifest_pairing_passed']}",
        f"- Policy pairing passed: {summary['integrity']['policy_pairing_passed']}",
        f"- All run audits passed: {summary['integrity']['all_run_audits_passed']}",
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
            path = _output_path(args.output_root, arm, args.load, int(seed))
            if not path.is_file():
                missing.append(path.as_posix())
                continue
            payload = _read_json(path)
            payload["_path"] = path.as_posix()
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"wrong simulator-oracle schema: {path}")
            if not bool((payload.get("audit") or {}).get("passed")):
                raise RuntimeError(f"failed simulator-oracle audit: {path}")
            payloads_by_arm[arm].append(payload)
    if missing and not args.allow_incomplete:
        raise FileNotFoundError(
            "missing simulator-oracle results:\n" + "\n".join(missing)
        )

    manifest_checks = {}
    manifest_pairing_passed = True
    for seed in args.seeds:
        manifest_path = _manifest_path(args.source_root, args.load, int(seed))
        manifest = _validate_manifest(manifest_path)
        expected = str(manifest.get("manifest_sha256"))
        arm_hashes = {
            str((payload.get("manifest") or {}).get("content_sha256"))
            for arm in ARM_KEYS
            for payload in payloads_by_arm[arm]
            if int((payload.get("meta") or {}).get("seed", -1))
            == int(seed)
        }
        passed = arm_hashes == {expected} if not args.allow_incomplete else (
            not arm_hashes or arm_hashes == {expected}
        )
        manifest_pairing_passed &= passed
        manifest_checks[str(seed)] = {
            "passed": passed,
            "expected": expected,
            "arm_hashes": sorted(arm_hashes),
        }

    fingerprints = {
        str(
            ((payload.get("meta") or {}).get("policy_contract") or {}).get(
                "fingerprint_sha256"
            )
        )
        for arm in ARM_KEYS
        for payload in payloads_by_arm[arm]
    }
    policy_pairing_passed = (
        len(fingerprints) == 1
        and "None" not in fingerprints
        and "" not in fingerprints
    )
    all_audits = all(
        bool((payload.get("audit") or {}).get("passed"))
        for rows in payloads_by_arm.values()
        for payload in rows
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "meta": {
            "load": args.load,
            "seeds": [int(seed) for seed in args.seeds],
            "ticks": int(args.ticks),
            "rollout_horizon": int(args.rollout_horizon),
            "candidate_limit": int(args.candidate_limit),
            "source_root": args.source_root.as_posix(),
            "output_root": args.output_root.as_posix(),
            "missing_results": missing,
        },
        "semantics": {
            "uses_world_model": False,
            "loads_checkpoint": False,
            "context_scheduler": "prefix_j0",
            "candidate_scope": (
                "all_idle"
                if int(args.candidate_limit) == 0
                else f"nearest_{int(args.candidate_limit)}"
            ),
            "rollout_continuation_mode": "isolated",
            "full_closed_loop_mpc": False,
        },
        "integrity": {
            "manifest_pairing_passed": bool(manifest_pairing_passed),
            "manifest_checks": manifest_checks,
            "policy_pairing_passed": bool(policy_pairing_passed),
            "policy_fingerprints": sorted(fingerprints),
            "all_run_audits_passed": bool(all_audits),
        },
        "arms": {
            arm: _aggregate_arm(payloads_by_arm[arm]) for arm in ARM_KEYS
        },
        "oracle": {
            arm: _aggregate_oracle(payloads_by_arm[arm]) for arm in ARM_KEYS
        },
        "comparisons": {
            "physical_only_minus_committed_v1": _paired_comparison(
                "physical_only_minus_committed_v1",
                "sim_oracle_physical_only",
                "sim_oracle_committed_v1",
                payloads_by_arm,
            )
        },
    }
    if not args.allow_incomplete and not all((
        manifest_pairing_passed,
        policy_pairing_passed,
        all_audits,
    )):
        raise RuntimeError("simulator-oracle summary integrity checks failed")

    validation = args.output_root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    json_path = validation / "phase_c_simulator_rollout_oracle.json"
    md_path = validation / "phase_c_simulator_rollout_oracle.md"
    _atomic_json(json_path, summary)
    markdown = _summary_markdown(summary)
    if md_path.is_file() and md_path.read_text(encoding="utf-8") != markdown:
        raise FileExistsError(f"existing summary differs: {md_path}")
    md_path.write_text(markdown, encoding="utf-8")
    print(f"[summary] {json_path}")
    print(f"[summary] {md_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("manifest", "arm", "summary"), required=True
    )
    parser.add_argument("--arm", choices=ARM_KEYS)
    parser.add_argument("--load", choices=LOADS, default=FORMAL_LOAD)
    parser.add_argument("--seed", type=int, default=FORMAL_SEEDS[0])
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS)
    )
    parser.add_argument("--ticks", type=int, default=FORMAL_TICKS)
    parser.add_argument(
        "--rollout-horizon", type=int, default=FORMAL_HORIZON
    )
    parser.add_argument(
        "--candidate-limit", type=int, default=FORMAL_CANDIDATE_LIMIT
    )
    parser.add_argument(
        "--max-contexts-per-tick",
        type=int,
        default=FORMAL_MAX_CONTEXTS_PER_TICK,
    )
    parser.add_argument(
        "--reservation-window",
        type=int,
        default=FORMAL_RESERVATION_WINDOW,
    )
    parser.add_argument(
        "--source-root", type=Path, default=DEFAULT_SOURCE_ROOT
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--frozen-bundle", type=Path, default=DEFAULT_BUNDLE
    )
    parser.add_argument(
        "--manifest-reference-root",
        type=Path,
        default=DEFAULT_V1_REFERENCE_ROOT,
    )
    parser.add_argument(
        "--v3-reference-root",
        type=Path,
        default=DEFAULT_V3_REFERENCE_ROOT,
    )
    parser.add_argument("--require-v3-reference", action="store_true")
    parser.add_argument("--skip-historical-contract", action="store_true")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--decision-trace-max-records", type=int, default=0
    )
    parser.add_argument("--station-trace-stride", type=int, default=5)
    parser.add_argument(
        "--station-trace-max-records", type=int, default=300
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.rollout_horizon <= 0:
        raise ValueError("--rollout-horizon must be positive")
    if args.candidate_limit < 0:
        raise ValueError("--candidate-limit must be non-negative")
    if args.max_contexts_per_tick < 0:
        raise ValueError("--max-contexts-per-tick must be non-negative")
    if args.reservation_window <= 0:
        raise ValueError("--reservation-window must be positive")
    if args.decision_trace_max_records < 0:
        raise ValueError("--decision-trace-max-records must be non-negative")
    _formal_argument_checks(args)

    if args.mode == "manifest":
        _prepare_manifest(args)
    elif args.mode == "arm":
        if args.arm is None:
            raise ValueError("--arm is required for --mode arm")
        _run_arm(args)
    else:
        _summarize(args)


if __name__ == "__main__":
    main()
