"""Run the frozen three-arm Phase-C dispatch-potential Kill Gate 1.

Stage-1 creates the realised order manifest.  The frozen Phase-C Round-1
policy and the training-free dispatch bridge replay that exact manifest.  The
bridge changes no checkpoint and adds no learned head.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from Policies.TaskAssigner import WorldModelTaskAssigner
from WorldModel.evaluation.evaluate_online_v6 import (
    _run_one_assigner,
    aggregate_seeds,
)
from WorldModel.evaluation.phase_c_dispatch_gate1_protocol import (
    BASELINE_CHECKPOINT,
    BRIDGE_LABEL,
    CANDIDATE_CHECKPOINT,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_CANDIDATE_SHA256,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOAD_CONFIGS,
    ONLINE_REPORT_SCHEMA_VERSION,
    PER_SEED_REPORT_SCHEMA_VERSION,
    REQUIRED_LOADS,
    ROUND1_LABEL,
    STAGE1_LABEL,
    TICKS,
    TOP_M_METADATA_ONLY,
    TRACE_MAX_RECORDS,
    TRACE_SCHEMA_VERSION,
    formal_protocol,
    sha256_file,
    validate_exact_seed_set,
)
from WorldModel.evaluation.run_phase_c_round1_online_pair import (
    _checkpoint_meta,
    _comparison,
    _pair_audit,
    _validate_candidate_schema,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write_text_atomic(path: Path, value: str, *, compare_existing: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)
    partial.write_text(value, encoding="utf-8")
    if path.exists():
        if not compare_existing or path.read_text(encoding="utf-8") != value:
            partial.unlink(missing_ok=True)
            raise FileExistsError(f"refusing to overwrite changed artifact: {path}")
        partial.unlink()
        return
    partial.replace(path)


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
    *,
    compare_existing: bool,
) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        compare_existing=compare_existing,
    )


def _write_trace_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    value = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in rows
    )
    _write_text_atomic(path, value, compare_existing=True)


def _verify_bundle(
    bundle: Mapping[str, Any],
    *,
    baseline: Path,
    candidate: Path,
    config: Path,
    load: str,
) -> dict[str, Any]:
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unexpected dispatch Gate-1 frozen bundle schema")
    protocol = bundle.get("formal_protocol") or {}
    if protocol != formal_protocol():
        raise ValueError("frozen dispatch Gate-1 protocol differs from source")
    artifacts = bundle.get("artifacts") or {}
    checks = (
        ("baseline_checkpoint", baseline, EXPECTED_BASELINE_SHA256),
        ("candidate_checkpoint", candidate, EXPECTED_CANDIDATE_SHA256),
    )
    for name, path, expected in checks:
        frozen = (artifacts.get(name) or {}).get("sha256")
        if not path.is_file() or frozen != expected or sha256_file(path) != expected:
            raise ValueError(f"frozen Gate-1 artifact changed: {name}")
    frozen_config = ((artifacts.get("load_configs") or {}).get(load) or {})
    if (
        frozen_config.get("path") != Path(LOAD_CONFIGS[load]).as_posix()
        or not config.is_file()
        or sha256_file(config) != frozen_config.get("sha256")
    ):
        raise ValueError(f"frozen Gate-1 {load} config changed")
    return dict(protocol)


def _trace_candidate_type(candidate: Mapping[str, Any]) -> str:
    return "no_assign" if candidate.get("robot_id") is None else "assign_robot"


def _compact_trace(
    assigner: WorldModelTaskAssigner,
    metrics: Mapping[str, Any],
    *,
    arm: str,
    load: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = list(getattr(assigner, "decision_trace_records", ()) or ())
    compact: list[dict[str, Any]] = []
    invalid_actions = 0
    invalid_ticks = 0
    composite_argmin_violations = 0
    crossing_trace_violations = 0
    group_span_violations = 0
    crossing_count = 0
    dominance_count = 0

    for index, record in enumerate(records):
        action_type = record.get("selected_action_type")
        if action_type not in {"assign_robot", "no_assign"}:
            invalid_actions += 1
        tick = int(record.get("tick", -1))
        if tick < 0:
            invalid_ticks += 1
        audit = record.get("dispatch_potential")
        if arm == BRIDGE_LABEL:
            if not isinstance(audit, dict):
                composite_argmin_violations += 1
                audit = {}
            span = float(audit.get("wm_group_span", 1e9))
            if span > 1.0 + 1e-12:
                group_span_violations += 1
            if bool(audit.get("crossing_reached")):
                crossing_count += 1
                if action_type != "assign_robot":
                    crossing_trace_violations += 1
            if bool(audit.get("potential_dominates_wm")):
                dominance_count += 1

            candidates = list(record.get("candidates") or ())
            if candidates:
                ranked = sorted(
                    candidates,
                    key=lambda row: (
                        float(row.get("dispatch_composite_score", 1e30)),
                        1 if _trace_candidate_type(row) == "no_assign" else 0,
                        (
                            int(row["robot_id"])
                            if row.get("robot_id") is not None
                            else 10**9
                        ),
                    ),
                )
                expected_type = _trace_candidate_type(ranked[0])
                if expected_type != action_type:
                    composite_argmin_violations += 1

        compact.append({
            "schema_version": TRACE_SCHEMA_VERSION,
            "arm": arm,
            "load": load,
            "seed": int(seed),
            "record_index": int(index),
            "tick": tick,
            "context_idx": int(record.get("context_idx", -1)),
            "order_id": int(record.get("order_id", -1)),
            "pod_id": int(record.get("pod_id", -1)),
            "station_id": int(record.get("station_id", -1)),
            "selected_action_type": action_type,
            "action_status": record.get("action_status"),
            "dispatch_potential_mode": record.get("dispatch_potential_mode"),
            "dispatch_potential": audit,
            "selected_dispatch_composite_score": record.get(
                "selected_dispatch_composite_score"
            ),
        })

    no_assign = sum(row["selected_action_type"] == "no_assign" for row in compact)
    assign_robot = sum(
        row["selected_action_type"] == "assign_robot" for row in compact
    )
    checks = {
        "nonempty": bool(compact),
        "no_dropped_records": int(metrics.get("decision_trace_dropped", 0)) == 0,
        "trace_count_matches_assigner": len(compact)
        == int(metrics.get("decision_trace_contexts", -1)),
        "trace_count_matches_decision_contexts": len(compact)
        == int(metrics.get("decision_contexts_total", -1)),
        "trace_count_matches_no_assign_contexts": len(compact)
        == int(metrics.get("native_no_assign_contexts", -1)),
        "selected_no_assign_count_matches": no_assign
        == int(metrics.get("native_no_assign_selected", -1)),
        "selected_assignment_count_matches": assign_robot
        == int(metrics.get("native_no_assign_assignment_selected", -1)),
        "valid_actions": invalid_actions == 0,
        "valid_ticks": invalid_ticks == 0,
    }
    if arm == BRIDGE_LABEL:
        checks.update({
            "bridge_mode": metrics.get("dispatch_potential_mode") == "bridge",
            "dispatch_trace_count": len(compact)
            == int(metrics.get("dispatch_contexts", -1)),
            "group_span_integrity": group_span_violations == 0
            and int(metrics.get("dispatch_group_span_violations", -1)) == 0,
            "composite_argmin_integrity": composite_argmin_violations == 0,
            "crossing_trace_integrity": crossing_trace_violations == 0,
            "crossing_count_matches": crossing_count
            == int(metrics.get("dispatch_crossing_contexts", -1)),
            "dominance_count_matches": dominance_count
            == int(metrics.get("dispatch_dominance_contexts", -1)),
        })
    summary = {
        "records": len(compact),
        "no_assign_selected": no_assign,
        "assign_robot_selected": assign_robot,
        "crossing_contexts": crossing_count,
        "dominance_contexts": dominance_count,
        "group_span_violations": group_span_violations,
        "composite_argmin_violations": composite_argmin_violations,
        "crossing_trace_violations": crossing_trace_violations,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return compact, summary


def _resume_seed(
    path: Path,
    *,
    load: str,
    seed: int,
    ticks: int,
    protocol_sha256: str,
    bundle_sha256: str,
) -> dict[str, Any]:
    payload = _read_json(path)
    meta = payload.get("meta") or {}
    checks = {
        "schema": payload.get("schema_version") == PER_SEED_REPORT_SCHEMA_VERSION,
        "load": meta.get("load") == load,
        "seed": int(meta.get("seed", -1)) == int(seed),
        "ticks": int(meta.get("ticks", -1)) == int(ticks),
        "protocol": meta.get("protocol_sha256") == protocol_sha256,
        "bundle": meta.get("bundle_sha256") == bundle_sha256,
        "arms": set(payload.get("arms") or {})
        == {STAGE1_LABEL, ROUND1_LABEL, BRIDGE_LABEL},
        "audit": bool((payload.get("three_arm_audit") or {}).get("passed")),
    }
    for artifact in (payload.get("artifacts") or {}).values():
        artifact_path = Path(str(artifact.get("path", "")))
        checks[f"artifact:{artifact_path.name}"] = (
            artifact_path.is_file()
            and artifact.get("sha256") == sha256_file(artifact_path)
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"cannot resume {path}: " + ", ".join(failed))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-bundle", required=True)
    parser.add_argument("--baseline", default=BASELINE_CHECKPOINT)
    parser.add_argument("--candidate", default=CANDIDATE_CHECKPOINT)
    parser.add_argument("--config", required=True)
    parser.add_argument("--load", choices=REQUIRED_LOADS, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--top-m", type=int, default=TOP_M_METADATA_ONLY)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--per-seed-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds]
    if args.formal:
        validate_exact_seed_set(seeds)
        if int(args.ticks) != TICKS or int(args.top_m) != TOP_M_METADATA_ONLY:
            raise SystemExit("formal Gate-1 ticks/top-m differ from protocol")
        expected_config = Path(LOAD_CONFIGS[args.load]).as_posix()
        if Path(args.config).as_posix() != expected_config:
            raise SystemExit(f"formal {args.load} config must be {expected_config}")

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    config_path = Path(args.config)
    bundle_path = Path(args.frozen_bundle)
    for path in (baseline_path, candidate_path, config_path, bundle_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    bundle = _read_json(bundle_path)
    protocol = _verify_bundle(
        bundle,
        baseline=baseline_path,
        candidate=candidate_path,
        config=config_path,
        load=args.load,
    )
    bundle_sha256 = sha256_file(bundle_path)
    baseline_meta = _checkpoint_meta(baseline_path)
    candidate_meta = _checkpoint_meta(candidate_path)
    _validate_candidate_schema(candidate_meta)
    if (
        baseline_meta["sha256"] != EXPECTED_BASELINE_SHA256
        or candidate_meta["sha256"] != EXPECTED_CANDIDATE_SHA256
    ):
        raise ValueError("Gate-1 checkpoint hash changed")

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite Gate-1 report: {output}")
    runtime_dir = Path(args.runtime_dir)
    manifest_dir = runtime_dir.parent / "order_manifests"
    trace_dir = Path(args.trace_dir)
    per_seed_dir = Path(args.per_seed_dir)
    for path in (runtime_dir, manifest_dir, trace_dir, per_seed_dir):
        path.mkdir(parents=True, exist_ok=True)

    per_seed: dict[int, dict[str, Any]] = {}
    audits: dict[int, dict[str, Any]] = {}
    comparisons: dict[int, dict[str, Any]] = {}
    artifacts_by_seed: dict[int, dict[str, Any]] = {}

    for seed in seeds:
        manifest_path = manifest_dir / (
            f"phasec_dispatch_gate1_{args.load}_seed{seed}_orders.json"
        )
        round1_trace_path = trace_dir / (
            f"phasec_dispatch_gate1_{args.load}_seed{seed}_round1.jsonl"
        )
        bridge_trace_path = trace_dir / (
            f"phasec_dispatch_gate1_{args.load}_seed{seed}_bridge.jsonl"
        )
        seed_path = per_seed_dir / (
            f"phasec_dispatch_gate1_{args.load}_seed{seed}.json"
        )
        if seed_path.is_file():
            payload = _resume_seed(
                seed_path,
                load=args.load,
                seed=seed,
                ticks=int(args.ticks),
                protocol_sha256=protocol["protocol_sha256"],
                bundle_sha256=bundle_sha256,
            )
            per_seed[seed] = payload["arms"]
            audits[seed] = payload["three_arm_audit"]
            comparisons[seed] = payload["comparisons"]
            artifacts_by_seed[seed] = payload["artifacts"]
            print(f"[resume] accepted Gate-1 {args.load} seed={seed}")
            continue

        print(f"\n--- Gate-1 {args.load} seed={seed}: Stage-1 ---")
        stage1_assigner = WorldModelTaskAssigner(
            checkpoint_path=str(baseline_path),
            top_m=int(args.top_m),
            include_no_assign_candidate=False,
        )
        stage1 = _run_one_assigner(
            str(config_path),
            stage1_assigner,
            seed,
            int(args.ticks),
            trace_label=STAGE1_LABEL,
            save_order_manifest=str(manifest_path),
        )

        print(f"--- Gate-1 {args.load} seed={seed}: Round-1 ---")
        round1_assigner = WorldModelTaskAssigner(
            checkpoint_path=str(candidate_path),
            top_m=int(args.top_m),
            include_no_assign_candidate=True,
            decision_trace_enabled=True,
            decision_trace_max_records=TRACE_MAX_RECORDS,
        )
        round1 = _run_one_assigner(
            str(config_path),
            round1_assigner,
            seed,
            int(args.ticks),
            trace_label=ROUND1_LABEL,
            recorded_orders_path=str(manifest_path),
        )

        print(f"--- Gate-1 {args.load} seed={seed}: dispatch bridge ---")
        bridge_assigner = WorldModelTaskAssigner(
            checkpoint_path=str(candidate_path),
            top_m=int(args.top_m),
            include_no_assign_candidate=True,
            dispatch_potential_mode="bridge",
            decision_trace_enabled=True,
            decision_trace_max_records=TRACE_MAX_RECORDS,
        )
        bridge = _run_one_assigner(
            str(config_path),
            bridge_assigner,
            seed,
            int(args.ticks),
            trace_label=BRIDGE_LABEL,
            recorded_orders_path=str(manifest_path),
        )

        round1_rows, round1_trace = _compact_trace(
            round1_assigner,
            round1,
            arm=ROUND1_LABEL,
            load=args.load,
            seed=seed,
        )
        bridge_rows, bridge_trace = _compact_trace(
            bridge_assigner,
            bridge,
            arm=BRIDGE_LABEL,
            load=args.load,
            seed=seed,
        )
        if not round1_trace["passed"] or not bridge_trace["passed"]:
            raise RuntimeError(
                f"Gate-1 trace audit failed for {args.load} seed={seed}"
            )
        _write_trace_atomic(round1_trace_path, round1_rows)
        _write_trace_atomic(bridge_trace_path, bridge_rows)

        stage1_round1 = _pair_audit(stage1, round1)
        stage1_bridge = _pair_audit(stage1, bridge)
        manifest_hashes = {
            stage1.get("order_arrival_manifest_sha256"),
            round1.get("order_arrival_manifest_sha256"),
            bridge.get("order_arrival_manifest_sha256"),
        }
        three_arm_checks = {
            "stage1_round1_pair": stage1_round1["passed"],
            "stage1_bridge_pair": stage1_bridge["passed"],
            "one_exact_manifest": len(manifest_hashes) == 1 and None not in manifest_hashes,
            "round1_trace": round1_trace["passed"],
            "bridge_trace": bridge_trace["passed"],
            "bridge_no_fallback": float(bridge.get("fallback_greedy_ratio", 0.0)) == 0.0,
            "bridge_group_span": int(bridge.get("dispatch_group_span_violations", -1)) == 0,
        }
        three_arm_audit = {
            "checks": three_arm_checks,
            "passed": all(three_arm_checks.values()),
            "round1_trace": round1_trace,
            "bridge_trace": bridge_trace,
        }
        if not three_arm_audit["passed"]:
            failed = [key for key, value in three_arm_checks.items() if not value]
            raise RuntimeError("Gate-1 three-arm audit failed: " + ", ".join(failed))

        arms = {
            STAGE1_LABEL: stage1,
            ROUND1_LABEL: round1,
            BRIDGE_LABEL: bridge,
        }
        pair_comparisons = {
            "round1_minus_stage1": _comparison(stage1, round1),
            "bridge_minus_round1": _comparison(round1, bridge),
            "bridge_minus_stage1": _comparison(stage1, bridge),
        }
        artifacts = {
            "order_manifest": {
                "path": manifest_path.as_posix(),
                "sha256": sha256_file(manifest_path),
            },
            "round1_trace": {
                "path": round1_trace_path.as_posix(),
                "sha256": sha256_file(round1_trace_path),
                "summary": round1_trace,
            },
            "bridge_trace": {
                "path": bridge_trace_path.as_posix(),
                "sha256": sha256_file(bridge_trace_path),
                "summary": bridge_trace,
            },
        }
        seed_payload = {
            "schema_version": PER_SEED_REPORT_SCHEMA_VERSION,
            "meta": {
                "formal": bool(args.formal),
                "role": "PREREGISTERED_GATE1_DEVELOPMENT_SEED",
                "load": args.load,
                "seed": int(seed),
                "ticks": int(args.ticks),
                "protocol_sha256": protocol["protocol_sha256"],
                "bundle_sha256": bundle_sha256,
            },
            "arms": arms,
            "three_arm_audit": three_arm_audit,
            "comparisons": pair_comparisons,
            "artifacts": artifacts,
        }
        _write_json_atomic(seed_path, seed_payload, compare_existing=True)
        per_seed[seed] = arms
        audits[seed] = three_arm_audit
        comparisons[seed] = pair_comparisons
        artifacts_by_seed[seed] = {
            **artifacts,
            "per_seed_report": {
                "path": seed_path.as_posix(),
                "sha256": sha256_file(seed_path),
            },
        }
        print(
            f"[done] {args.load} seed={seed}: completed orders "
            f"{stage1.get('completed_orders')} -> {round1.get('completed_orders')} "
            f"-> {bridge.get('completed_orders')}"
        )

    result = {
        "schema_version": ONLINE_REPORT_SCHEMA_VERSION,
        "meta": {
            "formal": bool(args.formal),
            "role": "PREREGISTERED_PHASE_C_DISPATCH_KILL_GATE1",
            "load": args.load,
            "seeds": seeds,
            "ticks": int(args.ticks),
            "arms": [STAGE1_LABEL, ROUND1_LABEL, BRIDGE_LABEL],
            "protocol_sha256": protocol["protocol_sha256"],
            "frozen_bundle": bundle_path.as_posix(),
            "frozen_bundle_sha256": bundle_sha256,
            "baseline_checkpoint": baseline_meta,
            "candidate_checkpoint": candidate_meta,
            "external_assignment_baseline": False,
            "td_target_or_head": False,
            "work_drift_or_residual_head": False,
            "new_neural_head": False,
            "hard_liveness_gate": False,
            "lyapunov_online_score": True,
            "lyapunov_score_semantics": "analytic_dispatch_potential_v1",
        },
        "per_seed": {str(seed): value for seed, value in per_seed.items()},
        "three_arm_audits": {str(seed): value for seed, value in audits.items()},
        "comparisons": {str(seed): value for seed, value in comparisons.items()},
        "per_seed_artifacts": {
            str(seed): value for seed, value in artifacts_by_seed.items()
        },
        "aggregate": aggregate_seeds(per_seed),
    }
    _write_json_atomic(output, result, compare_existing=False)
    print(f"\nsaved dispatch Gate-1 report: {output}")


if __name__ == "__main__":
    main()
