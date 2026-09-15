"""Run paired Stage-1 and Phase-C Round-1 World Model online arms.

The Stage-1 arm generates the realised exogenous order-arrival manifest.  The
Phase-C arm replays that exact manifest, so policy-induced state differences do
not silently change the offered demand.  Both arms remain pure World Model
policies: no Greedy/Hungarian fallback, TD target/head, WorkDrift head,
Lyapunov score, or post-hoc NO_ASSIGN gate is enabled.

Formal mode is fail-closed against the frozen seeds, loads, ticks, checkpoints,
protocol and configuration hashes.  It writes one atomic paired result and one
compact candidate decision trace per seed, which makes multi-day collection
safe to resume without interpreting a partial seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from Policies.TaskAssigner import WorldModelTaskAssigner
from WorldModel.evaluation.evaluate_online_v6 import (
    ORDER_ARRIVAL_MANIFEST_SCHEMA_VERSION,
    _run_one_assigner,
    aggregate_seeds,
)
from WorldModel.evaluation.phase_c_round1_cert_protocol import (
    BASELINE_HORIZON,
    CANDIDATE_HORIZON,
    DECISION_TRACE_MAX_RECORDS,
    DECISION_TRACE_SCHEMA_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_CANDIDATE_SHA256,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOAD_CONFIGS,
    ONLINE_REPORT_SCHEMA_VERSION,
    PER_SEED_REPORT_SCHEMA_VERSION,
    REQUIRED_LOADS,
    TICKS,
    TOP_M_METADATA_ONLY,
    formal_protocol,
    sha256_file,
    validate_exact_seed_set,
)


SCHEMA_VERSION = ONLINE_REPORT_SCHEMA_VERSION
BASELINE_LABEL = "Stage1WorldModel"
CANDIDATE_LABEL = "PhaseCWorldModel"
NO_ASSIGN_SCHEMA = "wm_native_no_assign_action_v1"
NO_ASSIGN_ENCODING = "zero_action_tensors_v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _checkpoint_meta(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"unsupported World Model checkpoint: {path}")
    config = dict(checkpoint.get("model_config") or {})
    action_schema = dict(checkpoint.get("action_schema") or {})
    return {
        "path": path.as_posix(),
        "sha256": _sha256_file(path),
        "model_config": config,
        "rollout_horizon": int(config.get("rollout_horizon", -1)),
        "label_schema_version": checkpoint.get("label_schema_version"),
        "action_schema": action_schema,
    }


def _validate_candidate_schema(meta: Mapping[str, Any]) -> None:
    schema = meta["action_schema"]
    checks = {
        "schema_version": schema.get("schema_version") == NO_ASSIGN_SCHEMA,
        "supports_no_assign_candidate": bool(
            schema.get("supports_no_assign_candidate")
        ),
        "no_assign_encoding": (
            schema.get("no_assign_encoding") == NO_ASSIGN_ENCODING
        ),
        "complete_group_coverage": bool(schema.get("complete_group_coverage")),
        "zero_encoding_verified": bool(schema.get("zero_encoding_verified")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "Phase-C checkpoint lacks the frozen native NO_ASSIGN contract: "
            + ", ".join(failed)
        )


def _verify_bundle(
    bundle: Mapping[str, Any],
    *,
    baseline_path: Path,
    candidate_path: Path,
    config_path: Path,
    load: str,
) -> dict[str, Any]:
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unexpected frozen Phase-C online bundle schema")
    protocol = bundle.get("formal_protocol") or {}
    if protocol != formal_protocol():
        raise ValueError("frozen Phase-C protocol differs from source protocol")
    artifacts = bundle.get("artifacts") or {}
    artifact_checks = (
        (
            "baseline_checkpoint",
            baseline_path,
            EXPECTED_BASELINE_SHA256,
        ),
        (
            "candidate_checkpoint",
            candidate_path,
            EXPECTED_CANDIDATE_SHA256,
        ),
    )
    for name, path, frozen_digest in artifact_checks:
        expected = (artifacts.get(name) or {}).get("sha256")
        actual = sha256_file(path) if path.is_file() else None
        if expected != frozen_digest or actual != frozen_digest:
            raise ValueError(f"frozen artifact changed: {name}")
    config_artifact = ((artifacts.get("load_configs") or {}).get(load) or {})
    if (
        config_artifact.get("path") != Path(LOAD_CONFIGS[load]).as_posix()
        or not config_path.is_file()
        or sha256_file(config_path) != config_artifact.get("sha256")
    ):
        raise ValueError(f"frozen {load} configuration changed")
    return dict(protocol)


def _pair_audit(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_hash = baseline.get("order_arrival_manifest_sha256")
    candidate_hash = candidate.get("order_arrival_manifest_sha256")
    baseline_count = baseline.get("order_arrival_count")
    candidate_count = candidate.get("order_arrival_count")
    checks = {
        "paired_order_hash": bool(baseline_hash)
        and baseline_hash == candidate_hash,
        "paired_order_count": baseline_count == candidate_count,
        "baseline_no_fallback": float(
            baseline.get("fallback_greedy_ratio", 0.0)
        )
        == 0.0,
        "candidate_no_fallback": float(
            candidate.get("fallback_greedy_ratio", 0.0)
        )
        == 0.0,
        "baseline_all_idle": (
            baseline.get("online_robot_candidate_scope") == "all_idle"
        ),
        "candidate_all_idle": (
            candidate.get("online_robot_candidate_scope") == "all_idle"
        ),
        "baseline_no_native_no_assign": not bool(
            baseline.get("native_no_assign_enabled", False)
        ),
        "candidate_native_no_assign_enabled": bool(
            candidate.get("native_no_assign_enabled", False)
        ),
        "candidate_no_assign_exercised": int(
            candidate.get("native_no_assign_contexts", 0)
        )
        > 0,
        "candidate_no_assign_scored_every_context": int(
            candidate.get("native_no_assign_scored", -1)
        )
        == int(candidate.get("native_no_assign_contexts", -2)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "order_manifest_sha256": baseline_hash,
        "order_count": baseline_count,
    }


def _comparison(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    fields = (
        "completed_orders",
        "completed_tasks",
        "avg_task_duration",
        "avg_excess_delay",
        "completed_order_flow_time_p95",
        "open_order_count",
        "pending_order_count",
        "open_order_age_p95",
        "pending_order_age_p95",
        "wm_label_cost",
        "wait_or_stall",
        "station_pressure",
        "bottleneck_CVaR",
        "unified_risk",
        "deadlock_ratio_mean",
        "deadlock_ratio_max",
        "handoff_ratio_mean",
        "handoff_ratio_max",
    )
    deltas = {}
    for field in fields:
        left = baseline.get(field)
        right = candidate.get(field)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            deltas[field] = {
                "baseline": left,
                "candidate": right,
                "candidate_minus_baseline": right - left,
            }
    return deltas


def _compact_candidate_trace(
    assigner: WorldModelTaskAssigner,
    metrics: Mapping[str, Any],
    *,
    load: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = list(getattr(assigner, "decision_trace_records", ()) or ())
    compact = []
    invalid_action_rows = 0
    invalid_ticks = 0
    for index, record in enumerate(records):
        action_type = record.get("selected_action_type")
        if action_type not in {"assign_robot", "no_assign"}:
            invalid_action_rows += 1
        tick = int(record.get("tick", -1))
        if tick < 0:
            invalid_ticks += 1
        compact.append({
            "schema_version": DECISION_TRACE_SCHEMA_VERSION,
            "load": load,
            "seed": int(seed),
            "record_index": int(index),
            "tick": tick,
            "context_idx": int(record.get("context_idx", -1)),
            "selected_action_type": action_type,
            "action_status": record.get("action_status"),
        })
    no_assign = sum(
        row["selected_action_type"] == "no_assign" for row in compact
    )
    assign_robot = sum(
        row["selected_action_type"] == "assign_robot" for row in compact
    )
    trace_contexts = int(metrics.get("decision_trace_contexts", -1))
    decision_contexts = int(metrics.get("decision_contexts_total", -1))
    no_assign_contexts = int(metrics.get("native_no_assign_contexts", -1))
    no_assign_scored = int(metrics.get("native_no_assign_scored", -1))
    checks = {
        "nonempty": len(compact) > 0,
        "no_dropped_records": int(metrics.get("decision_trace_dropped", 0)) == 0,
        "trace_count_matches_assigner": len(compact) == trace_contexts,
        "trace_count_matches_decision_contexts": len(compact) == decision_contexts,
        "trace_count_matches_no_assign_contexts": len(compact) == no_assign_contexts,
        "no_assign_scored_every_context": len(compact) == no_assign_scored,
        "selected_no_assign_count_matches": no_assign
        == int(metrics.get("native_no_assign_selected", -1)),
        "selected_assignment_count_matches": assign_robot
        == int(metrics.get("native_no_assign_assignment_selected", -1)),
        "valid_action_rows": invalid_action_rows == 0,
        "valid_ticks": invalid_ticks == 0,
    }
    summary = {
        "records": len(compact),
        "no_assign_selected": no_assign,
        "assign_robot_selected": assign_robot,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return compact, summary


def _write_text_atomic(path: Path, text: str, *, compare_existing: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        partial.unlink()
    partial.write_text(text, encoding="utf-8")
    if path.exists():
        if not compare_existing or path.read_text(encoding="utf-8") != text:
            partial.unlink(missing_ok=True)
            raise FileExistsError(f"refusing to overwrite changed artifact: {path}")
        partial.unlink()
        return
    partial.replace(path)


def _write_json_atomic(
    path: Path, payload: Mapping[str, Any], *, compare_existing: bool
) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        compare_existing=compare_existing,
    )


def _write_trace_atomic(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    text = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in rows
    )
    _write_text_atomic(path, text, compare_existing=True)


def _resume_seed_payload(
    path: Path,
    *,
    formal: bool,
    load: str,
    seed: int,
    ticks: int,
    protocol_sha256: str | None,
    bundle_sha256: str | None,
    manifest_path: Path,
    trace_path: Path | None,
) -> dict[str, Any]:
    payload = _read_json(path)
    meta = payload.get("meta") or {}
    checks = {
        "schema": payload.get("schema_version")
        == PER_SEED_REPORT_SCHEMA_VERSION,
        "formal": bool(meta.get("formal")) == bool(formal),
        "load": meta.get("load") == load,
        "seed": int(meta.get("seed", -1)) == int(seed),
        "ticks": int(meta.get("ticks", -1)) == int(ticks),
        "protocol": meta.get("formal_protocol_sha256") == protocol_sha256,
        "bundle": meta.get("frozen_bundle_sha256") == bundle_sha256,
        "paired_arms": set(payload.get("arms") or {})
        == {BASELINE_LABEL, CANDIDATE_LABEL},
        "pair_audit": bool((payload.get("pair_audit") or {}).get("passed")),
    }
    artifacts = payload.get("artifacts") or {}
    manifest = artifacts.get("order_manifest") or {}
    checks["manifest"] = (
        manifest_path.is_file()
        and manifest.get("sha256") == sha256_file(manifest_path)
    )
    if trace_path is not None:
        trace = artifacts.get("candidate_decision_trace") or {}
        checks["trace"] = (
            trace_path.is_file()
            and trace.get("sha256") == sha256_file(trace_path)
            and bool((trace.get("summary") or {}).get("passed"))
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"cannot resume changed/incomplete seed artifact {path}: "
            + ", ".join(failed)
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-bundle")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--load", choices=REQUIRED_LOADS, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--ticks", type=int, required=True)
    parser.add_argument("--top-m", type=int, default=1)
    parser.add_argument("--streams-dir", required=True)
    parser.add_argument("--trace-dir")
    parser.add_argument("--per-seed-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--formal",
        action="store_true",
        help="enforce the exact frozen Phase-C online certification contract",
    )
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise SystemExit("seeds must be a non-empty unique list")
    if int(args.ticks) <= 0 or int(args.top_m) <= 0:
        raise SystemExit("ticks and top-m must be positive")

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    config_path = Path(args.config)
    for path in (baseline_path, candidate_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    protocol = None
    bundle_path = Path(args.frozen_bundle) if args.frozen_bundle else None
    bundle_sha256 = None
    if args.formal:
        if bundle_path is None or not bundle_path.is_file():
            raise SystemExit("formal Phase-C online testing requires --frozen-bundle")
        if not args.trace_dir or not args.per_seed_dir:
            raise SystemExit(
                "formal Phase-C online testing requires --trace-dir and --per-seed-dir"
            )
        validate_exact_seed_set(seeds)
        if int(args.ticks) != TICKS:
            raise SystemExit(f"formal Phase-C testing freezes --ticks={TICKS}")
        if int(args.top_m) != TOP_M_METADATA_ONLY:
            raise SystemExit(
                "formal Phase-C testing freezes metadata "
                f"--top-m={TOP_M_METADATA_ONLY}"
            )
        expected_config = Path(LOAD_CONFIGS[args.load]).as_posix()
        if config_path.as_posix() != expected_config:
            raise SystemExit(
                f"formal {args.load} load requires {expected_config}, "
                f"got {config_path.as_posix()}"
            )
        bundle = _read_json(bundle_path)
        protocol = _verify_bundle(
            bundle,
            baseline_path=baseline_path,
            candidate_path=candidate_path,
            config_path=config_path,
            load=args.load,
        )
        bundle_sha256 = sha256_file(bundle_path)

    baseline_checkpoint = _checkpoint_meta(baseline_path)
    candidate_checkpoint = _checkpoint_meta(candidate_path)
    _validate_candidate_schema(candidate_checkpoint)
    if baseline_checkpoint["rollout_horizon"] != BASELINE_HORIZON:
        raise ValueError(
            "Phase-C Round-1 baseline must retain native Stage-1 "
            f"H={BASELINE_HORIZON}, got "
            f"H={baseline_checkpoint['rollout_horizon']}"
        )
    if candidate_checkpoint["rollout_horizon"] != CANDIDATE_HORIZON:
        raise ValueError(
            "Phase-C Round-1 candidate must retain native "
            f"H={CANDIDATE_HORIZON}, got "
            f"H={candidate_checkpoint['rollout_horizon']}"
        )
    if args.formal and (
        baseline_checkpoint["sha256"] != EXPECTED_BASELINE_SHA256
        or candidate_checkpoint["sha256"] != EXPECTED_CANDIDATE_SHA256
    ):
        raise ValueError("formal Phase-C checkpoint hash changed")

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite paired Phase-C result: {output}")
    runtime_dir = Path(args.streams_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = runtime_dir.parent / "order_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    per_seed_dir = Path(args.per_seed_dir) if args.per_seed_dir else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
    if per_seed_dir is not None:
        per_seed_dir.mkdir(parents=True, exist_ok=True)

    protocol_sha256 = protocol.get("protocol_sha256") if protocol else None
    trace_required = bool(args.formal or trace_dir is not None)
    per_seed: dict[int, dict[str, Any]] = {}
    pair_audits: dict[int, dict[str, Any]] = {}
    comparisons: dict[int, dict[str, Any]] = {}
    per_seed_artifacts: dict[int, dict[str, Any]] = {}

    for seed in seeds:
        manifest_path = manifest_dir / (
            f"phasec_r1_{args.load}_seed{seed}_orders.json"
        )
        trace_path = (
            trace_dir
            / f"phasec_r1_{args.load}_seed{seed}_candidate_decisions.jsonl"
            if trace_dir is not None
            else None
        )
        seed_path = (
            per_seed_dir / f"phasec_r1_{args.load}_seed{seed}_paired.json"
            if per_seed_dir is not None
            else None
        )

        if seed_path is not None and seed_path.is_file():
            payload = _resume_seed_payload(
                seed_path,
                formal=bool(args.formal),
                load=args.load,
                seed=seed,
                ticks=int(args.ticks),
                protocol_sha256=protocol_sha256,
                bundle_sha256=bundle_sha256,
                manifest_path=manifest_path,
                trace_path=trace_path if trace_required else None,
            )
            arms = payload["arms"]
            per_seed[seed] = arms
            pair_audits[seed] = payload["pair_audit"]
            comparisons[seed] = payload["comparison"]
            per_seed_artifacts[seed] = {
                **payload["artifacts"],
                "per_seed_report": {
                    "path": seed_path.as_posix(),
                    "sha256": sha256_file(seed_path),
                },
            }
            print(f"[resume] accepted frozen {args.load} seed={seed}")
            continue

        print(
            f"\n--- Phase C paired {args.load} seed={seed}: "
            "native Stage-1 World Model ---"
        )
        baseline_assigner = WorldModelTaskAssigner(
            checkpoint_path=str(baseline_path),
            top_m=int(args.top_m),
            include_no_assign_candidate=False,
        )
        baseline_metrics = _run_one_assigner(
            str(config_path),
            baseline_assigner,
            seed,
            int(args.ticks),
            trace_label=BASELINE_LABEL,
            save_order_manifest=str(manifest_path),
        )

        print(
            f"--- Phase C paired {args.load} seed={seed}: "
            "Phase-C World Model + native NO_ASSIGN ---"
        )
        candidate_assigner = WorldModelTaskAssigner(
            checkpoint_path=str(candidate_path),
            top_m=int(args.top_m),
            include_no_assign_candidate=True,
            decision_trace_enabled=trace_required,
            decision_trace_max_records=DECISION_TRACE_MAX_RECORDS,
        )
        candidate_metrics = _run_one_assigner(
            str(config_path),
            candidate_assigner,
            seed,
            int(args.ticks),
            trace_label=CANDIDATE_LABEL,
            recorded_orders_path=str(manifest_path),
        )

        audit = _pair_audit(baseline_metrics, candidate_metrics)
        if not audit["passed"]:
            failed = [
                key for key, passed in audit["checks"].items() if not passed
            ]
            raise RuntimeError(
                f"paired Phase-C mechanism audit failed for seed {seed}: "
                + ", ".join(failed)
            )

        trace_artifact = None
        if trace_required:
            if trace_path is None:
                raise RuntimeError("trace path missing in trace-enabled run")
            compact_trace, trace_summary = _compact_candidate_trace(
                candidate_assigner,
                candidate_metrics,
                load=args.load,
                seed=seed,
            )
            if not trace_summary["passed"]:
                failed = [
                    key
                    for key, passed in trace_summary["checks"].items()
                    if not passed
                ]
                raise RuntimeError(
                    f"candidate trace audit failed for {args.load} seed={seed}: "
                    + ", ".join(failed)
                )
            _write_trace_atomic(trace_path, compact_trace)
            trace_artifact = {
                "path": trace_path.as_posix(),
                "sha256": sha256_file(trace_path),
                "schema_version": DECISION_TRACE_SCHEMA_VERSION,
                "summary": trace_summary,
            }

        arms = {
            BASELINE_LABEL: baseline_metrics,
            CANDIDATE_LABEL: candidate_metrics,
        }
        comparison = _comparison(baseline_metrics, candidate_metrics)
        artifacts = {
            "order_manifest": {
                "path": manifest_path.as_posix(),
                "sha256": sha256_file(manifest_path),
                "manifest_sha256": audit["order_manifest_sha256"],
                "order_count": audit["order_count"],
            },
        }
        if trace_artifact is not None:
            artifacts["candidate_decision_trace"] = trace_artifact
        seed_payload = {
            "schema_version": PER_SEED_REPORT_SCHEMA_VERSION,
            "meta": {
                "formal": bool(args.formal),
                "role": (
                    "FORMAL_PHASE_C_ROUND1_PAIRED_CLOSED_LOOP_SEED"
                    if args.formal
                    else "DEVELOPMENT_PHASE_C_ROUND1_PAIRED_SEED"
                ),
                "load": args.load,
                "seed": int(seed),
                "ticks": int(args.ticks),
                "config": config_path.as_posix(),
                "config_sha256": sha256_file(config_path),
                "formal_protocol_sha256": protocol_sha256,
                "frozen_bundle_sha256": bundle_sha256,
                "baseline_checkpoint_sha256": baseline_checkpoint["sha256"],
                "candidate_checkpoint_sha256": candidate_checkpoint["sha256"],
            },
            "arms": arms,
            "pair_audit": audit,
            "comparison": comparison,
            "artifacts": artifacts,
        }
        if seed_path is not None:
            _write_json_atomic(seed_path, seed_payload, compare_existing=True)
            artifacts = {
                **artifacts,
                "per_seed_report": {
                    "path": seed_path.as_posix(),
                    "sha256": sha256_file(seed_path),
                },
            }

        per_seed[seed] = arms
        pair_audits[seed] = audit
        comparisons[seed] = comparison
        per_seed_artifacts[seed] = artifacts
        print(
            f"[done] {args.load} seed={seed} orders "
            f"{baseline_metrics.get('completed_orders')} -> "
            f"{candidate_metrics.get('completed_orders')}; cost "
            f"{baseline_metrics.get('wm_label_cost')} -> "
            f"{candidate_metrics.get('wm_label_cost')}; NO_ASSIGN selected "
            f"{candidate_metrics.get('native_no_assign_selected_ratio', 0.0)}"
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "formal": bool(args.formal),
            "role": (
                "FORMAL_PHASE_C_ROUND1_PAIRED_CLOSED_LOOP_COLLECTION"
                if args.formal
                else "PRE_CERTIFICATION_MECHANISM_AND_DIRECTIONAL_SMOKE"
            ),
            "load": args.load,
            "config": config_path.as_posix(),
            "config_sha256": _sha256_file(config_path),
            "seeds": seeds,
            "ticks": int(args.ticks),
            "top_m": int(args.top_m),
            "top_m_is_online_limit": False,
            "online_robot_candidate_scope": "all_idle",
            "normal_arrivals": True,
            "paired_order_arrivals": (
                "Stage-1-generated manifest replayed exactly in Phase-C arm"
            ),
            "paired_arms": [BASELINE_LABEL, CANDIDATE_LABEL],
            "baseline_checkpoint": baseline_checkpoint,
            "candidate_checkpoint": candidate_checkpoint,
            "baseline_action_space": "all_idle_robots_only",
            "candidate_action_space": "all_idle_robots_plus_native_no_assign",
            "baseline_native_horizon": BASELINE_HORIZON,
            "candidate_native_horizon": CANDIDATE_HORIZON,
            "external_assignment_baseline": False,
            "td_target_or_head": False,
            "work_drift_or_residual_head": False,
            "lyapunov_online_scoring": False,
            "hard_no_assign_gate": False,
            "formal_protocol_sha256": protocol_sha256,
            "frozen_bundle": bundle_path.as_posix() if bundle_path else None,
            "frozen_bundle_sha256": bundle_sha256,
            "runtime_dir": runtime_dir.as_posix(),
            "order_manifest_dir": manifest_dir.as_posix(),
            "candidate_trace_dir": trace_dir.as_posix() if trace_dir else None,
            "per_seed_dir": per_seed_dir.as_posix() if per_seed_dir else None,
        },
        "per_seed": {str(seed): values for seed, values in per_seed.items()},
        "pair_audits": {
            str(seed): values for seed, values in pair_audits.items()
        },
        "comparisons": {
            str(seed): values for seed, values in comparisons.items()
        },
        "per_seed_artifacts": {
            str(seed): values for seed, values in per_seed_artifacts.items()
        },
        "aggregate": aggregate_seeds(per_seed),
    }
    _write_json_atomic(output, result, compare_existing=False)
    print(f"\nsaved paired Phase-C online result: {output}")


if __name__ == "__main__":
    main()
