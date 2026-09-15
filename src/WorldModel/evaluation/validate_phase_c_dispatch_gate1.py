"""Validate the preregistered Phase-C dispatch-potential Kill Gate 1."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from WorldModel.evaluation.phase_c_dispatch_gate1_protocol import (
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    BRIDGE_LABEL,
    BRIDGE_ROUND1_BOUNDS,
    BRIDGE_STAGE1_BOUNDS,
    EXPECTED_CANDIDATE_SHA256,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOAD_CONFIGS,
    MAX_ELIGIBLE_DEFER_MASS,
    ONLINE_REPORT_SCHEMA_VERSION,
    PER_SEED_REPORT_SCHEMA_VERSION,
    REQUIRED_LOADS,
    ROUND1_LABEL,
    SEEDS,
    STAGE1_LABEL,
    TICKS,
    TRACE_SCHEMA_VERSION,
    VALIDATION_SCHEMA_VERSION,
    formal_protocol,
    sha256_file,
)


METRICS = (
    "completion_fraction",
    "deadlock_ratio_max",
    "avg_excess_delay_relative",
    "completed_flow_time_relative",
    "open_order_fraction",
    "pending_order_fraction",
    "open_order_age_fraction",
    "pending_order_age_fraction",
    "handoff_ratio_mean",
    "handoff_ratio_max",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _number(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric metric: {name}")
    return float(value)


def _comparison_values(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    metric: str,
) -> tuple[float, float, float]:
    left_orders = _number(left, "order_arrival_count")
    right_orders = _number(right, "order_arrival_count")
    if left_orders <= 0 or right_orders <= 0:
        raise ValueError("order arrival count must be positive")
    if metric == "completion_fraction":
        a = _number(left, "completed_orders") / left_orders
        b = _number(right, "completed_orders") / right_orders
    elif metric == "open_order_fraction":
        a = _number(left, "open_order_count") / left_orders
        b = _number(right, "open_order_count") / right_orders
    elif metric == "pending_order_fraction":
        a = _number(left, "pending_order_count") / left_orders
        b = _number(right, "pending_order_count") / right_orders
    elif metric == "open_order_age_fraction":
        a = _number(left, "open_order_age_p95") / float(TICKS)
        b = _number(right, "open_order_age_p95") / float(TICKS)
    elif metric == "pending_order_age_fraction":
        a = _number(left, "pending_order_age_p95") / float(TICKS)
        b = _number(right, "pending_order_age_p95") / float(TICKS)
    elif metric.endswith("_relative"):
        source = {
            "avg_excess_delay_relative": "avg_excess_delay",
            "completed_flow_time_relative": "completed_order_flow_time_p95",
        }[metric]
        raw_a = _number(left, source)
        raw_b = _number(right, source)
        a = 0.0
        b = (raw_b - raw_a) / max(abs(raw_a), 1.0)
    else:
        a = _number(left, metric)
        b = _number(right, metric)
    return a, b, b - a


def _mean_ci(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "ci95": None}
    if array.size == 1:
        value = float(array[0])
        return {"n": 1, "mean": value, "ci95": [value, value]}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0,
        array.size,
        size=(BOOTSTRAP_REPEATS, array.size),
    )
    means = array[indices].mean(axis=1)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def _metric_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    left_label: str,
    right_label: str,
    metric: str,
    seed_offset: int,
) -> dict[str, Any]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    by_load: dict[str, list[float]] = defaultdict(list)
    raw_rows = []
    for row in rows:
        left, right, delta = _comparison_values(
            row["arms"][left_label],
            row["arms"][right_label],
            metric,
        )
        seed = int(row["seed"])
        load = str(row["load"])
        by_seed[seed].append(delta)
        by_load[load].append(delta)
        raw_rows.append({
            "seed": seed,
            "load": load,
            "left": left,
            "right": right,
            "right_minus_left": delta,
        })
    seed_means = [float(np.mean(by_seed[key])) for key in sorted(by_seed)]
    return {
        "left_arm": left_label,
        "right_arm": right_label,
        "overall_seed_cluster": _mean_ci(
            seed_means,
            seed=BOOTSTRAP_SEED + seed_offset,
        ),
        "by_load": {
            load: _mean_ci(
                values,
                seed=BOOTSTRAP_SEED + seed_offset + 100 + index,
            )
            for index, (load, values) in enumerate(sorted(by_load.items()))
        },
        "rows": raw_rows,
    }


def _terminal_assignment(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    terminal = [row for row in rows if 1000 <= int(row.get("tick", -1)) < 1500]
    assignments = sum(
        row.get("selected_action_type") == "assign_robot" for row in terminal
    )
    return {
        "terminal_contexts": len(terminal),
        "terminal_assignments": int(assignments),
        "terminal_bin_robot_assignment_when_contexts_exist": (
            not terminal or assignments > 0
        ),
    }


def _liveness_audit(
    bridge: Mapping[str, Any],
    bridge_trace: Sequence[Mapping[str, Any]],
    round1_trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bridge_terminal = _terminal_assignment(bridge_trace)
    round1_terminal = _terminal_assignment(round1_trace)
    crossing_rows = [
        row
        for row in bridge_trace
        if bool((row.get("dispatch_potential") or {}).get("crossing_reached"))
    ]
    crossing_assignments = sum(
        row.get("selected_action_type") == "assign_robot"
        for row in crossing_rows
    )
    max_mass = _number(bridge, "dispatch_max_continuous_eligible_mass")
    checks = {
        "terminal_bin_robot_assignment_when_contexts_exist": bridge_terminal[
            "terminal_bin_robot_assignment_when_contexts_exist"
        ],
        "max_eligible_defer_streak_bound_passed": bool(
            bridge.get("max_eligible_defer_streak_bound_passed")
        ) and max_mass <= MAX_ELIGIBLE_DEFER_MASS + 1e-12,
        "crossing_assignment_rate_100pct": (
            crossing_assignments == len(crossing_rows)
            and int(bridge.get("dispatch_crossing_violations", -1)) == 0
        ),
        "group_span_integrity": int(
            bridge.get("dispatch_group_span_violations", -1)
        ) == 0 and float(bridge.get("dispatch_group_span_max", 1e9)) <= 1.0 + 1e-12,
        "no_fallback": float(bridge.get("fallback_greedy_ratio", 1.0)) == 0.0,
        "all_idle_scope": bridge.get("online_robot_candidate_scope") == "all_idle",
        "trace_not_dropped": int(bridge.get("decision_trace_dropped", 0)) == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "bridge_terminal": bridge_terminal,
        "round1_terminal": round1_terminal,
        "crossing_contexts": len(crossing_rows),
        "crossing_assignments": int(crossing_assignments),
        "max_continuous_eligible_defer_mass": max_mass,
        "bound": MAX_ELIGIBLE_DEFER_MASS,
    }


def _branch(
    row: Mapping[str, Any],
    liveness: Mapping[str, Any],
) -> dict[str, Any]:
    stage1 = row["arms"][STAGE1_LABEL]
    round1 = row["arms"][ROUND1_LABEL]
    bridge = row["arms"][BRIDGE_LABEL]
    _, _, br_completion = _comparison_values(
        round1, bridge, "completion_fraction"
    )
    _, _, br_deadlock = _comparison_values(round1, bridge, "deadlock_ratio_max")
    _, _, rs_completion = _comparison_values(
        stage1, round1, "completion_fraction"
    )
    _, _, rs_deadlock = _comparison_values(stage1, round1, "deadlock_ratio_max")
    _, _, bs_completion = _comparison_values(
        stage1, bridge, "completion_fraction"
    )
    _, _, bs_deadlock = _comparison_values(stage1, bridge, "deadlock_ratio_max")

    if not liveness["passed"]:
        name = "DEFER_ABSORPTION"
    else:
        preexisting_completion = (
            rs_completion <= -0.05 and bs_completion <= -0.05
        )
        preexisting_deadlock = rs_deadlock > 0.02 and bs_deadlock > 0.02
        bridge_within_hard_bounds = br_completion >= -0.10 and br_deadlock <= 0.02
        if (
            (preexisting_completion or preexisting_deadlock)
            and bridge_within_hard_bounds
        ):
            name = "PREEXISTING_CONGESTION_BRANCH"
        elif br_completion < -0.05 or br_deadlock > 0.02:
            name = "INDUCED_CONGESTION_REGRESSION"
        else:
            name = "PASS_OR_OTHER_NONABSORBING"
    return {
        "branch": name,
        "bridge_minus_round1_completion_fraction": br_completion,
        "bridge_minus_round1_deadlock_ratio_max": br_deadlock,
        "round1_minus_stage1_completion_fraction": rs_completion,
        "round1_minus_stage1_deadlock_ratio_max": rs_deadlock,
        "bridge_minus_stage1_completion_fraction": bs_completion,
        "bridge_minus_stage1_deadlock_ratio_max": bs_deadlock,
    }


def _bridge_round1_contract(reports: Mapping[str, Any]) -> dict[str, Any]:
    b = BRIDGE_ROUND1_BOUNDS
    completion = reports["completion_fraction"]
    deadlock = reports["deadlock_ratio_max"]
    checks = {
        "completion_ci95_lower": completion["overall_seed_cluster"]["ci95"][0]
        > b["completion_fraction_ci95_lower_strict"],
        "completion_each_load": all(
            value["mean"] > b["completion_fraction_each_load_strict"]
            for value in completion["by_load"].values()
        ),
        "completion_each_run_hard_floor": all(
            row["right_minus_left"] >= b["completion_fraction_each_run_min"]
            for row in completion["rows"]
        ),
        "deadlock_ci95_upper": deadlock["overall_seed_cluster"]["ci95"][1]
        <= b["deadlock_ratio_max_ci95_upper"],
        "deadlock_each_load": all(
            value["mean"] <= b["deadlock_ratio_max_each_load_upper"]
            for value in deadlock["by_load"].values()
        ),
        "avg_excess_delay": reports["avg_excess_delay_relative"][
            "overall_seed_cluster"
        ]["ci95"][1] <= b["avg_excess_delay_relative_ci95_upper"],
        "completed_flow_time": reports["completed_flow_time_relative"][
            "overall_seed_cluster"
        ]["ci95"][1] <= b["completed_flow_time_relative_ci95_upper"],
        "open_order_fraction": reports["open_order_fraction"][
            "overall_seed_cluster"
        ]["ci95"][1] <= b["open_order_fraction_ci95_upper"],
        "pending_order_fraction": reports["pending_order_fraction"][
            "overall_seed_cluster"
        ]["ci95"][1] <= b["pending_order_fraction_ci95_upper"],
        "open_order_age_fraction": reports["open_order_age_fraction"][
            "overall_seed_cluster"
        ]["ci95"][1] <= b["open_order_age_fraction_ci95_upper"],
        "pending_order_age_fraction": reports["pending_order_age_fraction"][
            "overall_seed_cluster"
        ]["ci95"][1] <= b["pending_order_age_fraction_ci95_upper"],
        "handoff_ratio_mean": reports["handoff_ratio_mean"][
            "overall_seed_cluster"
        ]["ci95"][1] <= b["handoff_ratio_mean_ci95_upper"],
        "handoff_ratio_max": reports["handoff_ratio_max"][
            "overall_seed_cluster"
        ]["ci95"][1] <= b["handoff_ratio_max_ci95_upper"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def _bridge_stage1_contract(reports: Mapping[str, Any]) -> dict[str, Any]:
    b = BRIDGE_STAGE1_BOUNDS
    completion = reports["completion_fraction"]
    deadlock = reports["deadlock_ratio_max"]
    checks = {
        "completion_ci95_lower": completion["overall_seed_cluster"]["ci95"][0]
        > b["completion_fraction_ci95_lower_strict"],
        "completion_each_load": all(
            value["mean"] > b["completion_fraction_each_load_strict"]
            for value in completion["by_load"].values()
        ),
        "deadlock_ci95_upper": deadlock["overall_seed_cluster"]["ci95"][1]
        <= b["deadlock_ratio_max_ci95_upper"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-bundle", required=True)
    parser.add_argument("--online-reports", nargs="+", required=True)
    parser.add_argument("--per-seed-dir", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--nonperturbation-smoke", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bundle_path = Path(args.frozen_bundle)
    bundle = _read_json(bundle_path)
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unexpected frozen Gate-1 bundle")
    if bundle.get("formal_protocol") != formal_protocol():
        raise ValueError("frozen Gate-1 protocol differs from source")
    protocol_sha256 = bundle["formal_protocol"]["protocol_sha256"]

    reports = [_read_json(Path(path)) for path in args.online_reports]
    if {str((report.get("meta") or {}).get("load")) for report in reports} != set(
        REQUIRED_LOADS
    ):
        raise ValueError("Gate-1 reports must cover low/mid/high exactly")
    rows = []
    per_seed_dir = Path(args.per_seed_dir)
    trace_dir = Path(args.trace_dir)
    liveness_rows = []
    branch_rows = []
    artifact_checks = []
    for report in reports:
        meta = report.get("meta") or {}
        load = str(meta.get("load"))
        if (
            report.get("schema_version") != ONLINE_REPORT_SCHEMA_VERSION
            or tuple(int(value) for value in meta.get("seeds", ())) != SEEDS
            or int(meta.get("ticks", -1)) != TICKS
            or meta.get("protocol_sha256") != protocol_sha256
        ):
            raise ValueError(f"invalid Gate-1 online report for load={load}")
        for seed in SEEDS:
            seed_path = per_seed_dir / f"phasec_dispatch_gate1_{load}_seed{seed}.json"
            seed_payload = _read_json(seed_path)
            seed_meta = seed_payload.get("meta") or {}
            if (
                seed_payload.get("schema_version") != PER_SEED_REPORT_SCHEMA_VERSION
                or seed_meta.get("load") != load
                or int(seed_meta.get("seed", -1)) != seed
                or int(seed_meta.get("ticks", -1)) != TICKS
                or seed_meta.get("protocol_sha256") != protocol_sha256
            ):
                raise ValueError(f"invalid Gate-1 seed report: {seed_path}")
            arms = seed_payload.get("arms") or {}
            if set(arms) != {STAGE1_LABEL, ROUND1_LABEL, BRIDGE_LABEL}:
                raise ValueError(f"missing Gate-1 arm: {seed_path}")
            if not bool((seed_payload.get("three_arm_audit") or {}).get("passed")):
                raise ValueError(f"three-arm audit failed: {seed_path}")
            round1_path = trace_dir / (
                f"phasec_dispatch_gate1_{load}_seed{seed}_round1.jsonl"
            )
            bridge_path = trace_dir / (
                f"phasec_dispatch_gate1_{load}_seed{seed}_bridge.jsonl"
            )
            round1_trace = _read_jsonl(round1_path)
            bridge_trace = _read_jsonl(bridge_path)
            for arm_name, trace_rows, trace_path in (
                (ROUND1_LABEL, round1_trace, round1_path),
                (BRIDGE_LABEL, bridge_trace, bridge_path),
            ):
                if any(
                    row.get("schema_version") != TRACE_SCHEMA_VERSION
                    or row.get("arm") != arm_name
                    or row.get("load") != load
                    or int(row.get("seed", -1)) != seed
                    for row in trace_rows
                ):
                    raise ValueError(f"invalid Gate-1 trace contract: {trace_path}")
            report_arms = ((report.get("per_seed") or {}).get(str(seed)) or {})
            if report_arms != arms:
                raise ValueError(
                    f"online/per-seed arm payload mismatch: {load} seed={seed}"
                )
            frozen_artifacts = seed_payload.get("artifacts") or {}
            for name, actual_path in (
                ("round1_trace", round1_path),
                ("bridge_trace", bridge_path),
            ):
                frozen = frozen_artifacts.get(name) or {}
                if (
                    frozen.get("path") != actual_path.as_posix()
                    or frozen.get("sha256") != sha256_file(actual_path)
                ):
                    raise ValueError(
                        f"seed artifact hash mismatch: {load} seed={seed} {name}"
                    )
            liveness = _liveness_audit(
                arms[BRIDGE_LABEL],
                bridge_trace,
                round1_trace,
            )
            row = {"load": load, "seed": seed, "arms": arms}
            rows.append(row)
            liveness_rows.append({"load": load, "seed": seed, **liveness})
            branch_rows.append({"load": load, "seed": seed, **_branch(row, liveness)})
            artifact_checks.append({
                "load": load,
                "seed": seed,
                "seed_report_sha256": sha256_file(seed_path),
                "round1_trace_sha256": sha256_file(round1_path),
                "bridge_trace_sha256": sha256_file(bridge_path),
            })

    comparisons = {}
    for comparison_name, left_label, right_label, offset in (
        ("bridge_minus_round1", ROUND1_LABEL, BRIDGE_LABEL, 0),
        ("bridge_minus_stage1", STAGE1_LABEL, BRIDGE_LABEL, 1000),
        ("round1_minus_stage1", STAGE1_LABEL, ROUND1_LABEL, 2000),
    ):
        comparisons[comparison_name] = {
            metric: _metric_report(
                rows,
                left_label=left_label,
                right_label=right_label,
                metric=metric,
                seed_offset=offset + index * 10,
            )
            for index, metric in enumerate(METRICS)
        }

    nonperturbation = _read_json(Path(args.nonperturbation_smoke))
    implementation_checks = {
        "nonperturbation_smoke": bool(nonperturbation.get("passed")),
        "nonperturbation_smoke_contract": (
            int(nonperturbation.get("seed", -1)) == 9981
            and int(nonperturbation.get("ticks", -1)) == 100
            and nonperturbation.get("config")
            == Path(LOAD_CONFIGS["low"]).as_posix()
            and (nonperturbation.get("checkpoint") or {}).get("sha256")
            == EXPECTED_CANDIDATE_SHA256
        ),
        "all_three_arm_audits": all(
            bool((report.get("three_arm_audits") or {}).get(str(seed), {}).get("passed"))
            for report in reports
            for seed in SEEDS
        ),
        "all_liveness_runs": all(row["passed"] for row in liveness_rows),
    }
    implementation = {
        "passed": all(implementation_checks.values()),
        "checks": implementation_checks,
    }
    bridge_round1 = _bridge_round1_contract(
        comparisons["bridge_minus_round1"]
    )
    bridge_stage1 = _bridge_stage1_contract(
        comparisons["bridge_minus_stage1"]
    )
    if not implementation["passed"]:
        verdict = "INVALID_OR_DEFER_ABSORPTION"
        passed = False
    elif not bridge_round1["passed"]:
        verdict = "GATE1_FAIL_NONINFERIORITY_OR_INDUCED_CONGESTION"
        passed = False
    elif not bridge_stage1["passed"]:
        verdict = "DEFER_VALUE_NOT_DEMONSTRATED"
        passed = False
        for row in branch_rows:
            if row["branch"] == "PASS_OR_OTHER_NONABSORBING":
                row["branch"] = "DEFER_VALUE_NOT_DEMONSTRATED"
    else:
        verdict = "GATE1_PASS_PROCEED_TO_LONG_CONTINUATION_GATE2"
        passed = True

    branch_counts = {
        branch: sum(row["branch"] == branch for row in branch_rows)
        for branch in (
            "DEFER_ABSORPTION",
            "INDUCED_CONGESTION_REGRESSION",
            "PREEXISTING_CONGESTION_BRANCH",
            "DEFER_VALUE_NOT_DEMONSTRATED",
            "PASS_OR_OTHER_NONABSORBING",
        )
    }
    payload = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "passed": passed,
        "verdict": verdict,
        "protocol_sha256": protocol_sha256,
        "frozen_bundle_sha256": sha256_file(bundle_path),
        "implementation_and_liveness": implementation,
        "bridge_vs_round1_contract": bridge_round1,
        "bridge_vs_stage1_contract": bridge_stage1,
        "comparisons": comparisons,
        "liveness_by_run": liveness_rows,
        "diagnostic_branches": branch_rows,
        "diagnostic_branch_counts": branch_counts,
        "artifact_checks": artifact_checks,
        "nonperturbation_smoke": {
            "path": Path(args.nonperturbation_smoke).as_posix(),
            "sha256": sha256_file(args.nonperturbation_smoke),
            "passed": bool(nonperturbation.get("passed")),
        },
        "scope": (
            "development kill gate only; passing authorises Gate-2 long-"
            "continuation falsification, not formal deployment certification"
        ),
    }
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite Gate-1 validation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("dispatch Gate-1 verdict =", verdict)
    print("passed =", passed)
    print("branches =", branch_counts)
    print("report =", output)


if __name__ == "__main__":
    main()
