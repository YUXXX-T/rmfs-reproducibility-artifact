"""Validate the frozen paired Layer-5 online results and trajectory report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from WorldModel.evaluation.run_work_drift_layer5 import (
    BASELINE_LABEL,
    SCHEMA_VERSION as ONLINE_SCHEMA_VERSION,
    WORK_LABEL,
)
from WorldModel.evaluation.evaluate_online_v6 import (
    ORDER_ARRIVAL_MANIFEST_SCHEMA_VERSION,
)
from WorldModel.evaluation.work_drift_layer5_protocol import (
    BOOTSTRAP_REPEATS,
    CERTIFICATION_SEEDS,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    FROZEN_LAMBDA,
    RANDOM_SEED,
    REQUIRED_LOADS,
    TICKS,
    formal_protocol,
    sha256_file,
)


SCHEMA_VERSION = "work_drift_layer5_closed_loop_cert_v1"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _mean_ci(values: Sequence[float], *, repeats: int, seed: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "ci95": None}
    if array.size == 1:
        value = float(array[0])
        return {"n": 1, "mean": value, "ci95": [value, value]}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(repeats), array.size))
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
    rows: Sequence[Mapping],
    metric: str,
    *,
    direction: str,
    relative: bool,
    repeats: int,
    seed: int,
) -> dict:
    by_seed: dict[int, list[float]] = defaultdict(list)
    by_load: dict[str, list[float]] = defaultdict(list)
    raw_rows = []
    for row in rows:
        base = float(row["baseline"][metric])
        work = float(row["work"][metric])
        delta = work - base
        value = delta
        if relative:
            floor = 1.0 if metric != "unified_risk" else 1e-3
            value = delta / max(abs(base), floor)
        by_seed[int(row["seed"])].append(value)
        by_load[str(row["load"])].append(value)
        raw_rows.append({
            "seed": int(row["seed"]),
            "load": str(row["load"]),
            "baseline": base,
            "work": work,
            "delta_work_minus_baseline": delta,
            "comparison_value": value,
        })
    seed_means = [float(np.mean(by_seed[seed_id])) for seed_id in sorted(by_seed)]
    overall = _mean_ci(seed_means, repeats=repeats, seed=seed)
    load_reports = {
        load: _mean_ci(
            values,
            repeats=repeats,
            seed=seed + 1009 * (index + 1),
        )
        for index, (load, values) in enumerate(sorted(by_load.items()))
    }
    mean = overall["mean"]
    favourable = (
        mean is not None
        and (mean <= 0.0 if direction == "lower" else mean >= 0.0)
    )
    return {
        "metric": metric,
        "direction": direction,
        "comparison": (
            "relative_work_minus_baseline" if relative
            else "absolute_work_minus_baseline"
        ),
        "bootstrap_unit": "simulation_seed_paired_across_loads",
        "overall": overall,
        "by_load": load_reports,
        "favourable_point_estimate": bool(favourable),
        "rows": raw_rows,
    }


def _load_online_rows(paths: Sequence[Path], protocol: Mapping) -> tuple[list[dict], dict]:
    rows = []
    audits = {
        "schemas": True,
        "formal_flags": True,
        "protocol_hashes": True,
        "exact_loads": True,
        "exact_seeds": True,
        "exact_ticks": True,
        "paired_arms": True,
        "all_idle_scope": True,
        "no_greedy_or_fallback": True,
        "exact_paired_order_manifests": True,
        "work_mode_isolated": True,
        "candidate_supersets_above_10": True,
        "work_signal_exercised": True,
    }
    observed_loads = []
    modified_total = 0
    for path in paths:
        report = _read_json(path)
        meta = report.get("meta") or {}
        if report.get("schema_version") != ONLINE_SCHEMA_VERSION:
            audits["schemas"] = False
        if not bool(meta.get("formal")):
            audits["formal_flags"] = False
        if meta.get("layer5_protocol_sha256") != protocol.get("protocol_sha256"):
            audits["protocol_hashes"] = False
        load = str(meta.get("load"))
        observed_loads.append(load)
        if int(meta.get("ticks", -1)) != TICKS:
            audits["exact_ticks"] = False
        if set(meta.get("paired_arms") or ()) != {BASELINE_LABEL, WORK_LABEL}:
            audits["paired_arms"] = False
        seeds = tuple(sorted(int(value) for value in meta.get("seeds") or ()))
        if seeds != CERTIFICATION_SEEDS:
            audits["exact_seeds"] = False
        per_seed = report.get("per_seed") or {}
        if tuple(sorted(int(value) for value in per_seed)) != CERTIFICATION_SEEDS:
            audits["exact_seeds"] = False
        for seed_text, arms in per_seed.items():
            if set(arms) != {BASELINE_LABEL, WORK_LABEL}:
                audits["paired_arms"] = False
                continue
            baseline = arms[BASELINE_LABEL]
            work = arms[WORK_LABEL]
            for metric in (baseline, work):
                if metric.get("online_robot_candidate_scope") != "all_idle":
                    audits["all_idle_scope"] = False
                if int(metric.get("fallback_greedy_calls", 0)) != 0:
                    audits["no_greedy_or_fallback"] = False
                if int(metric.get("local_greedy_compared", 0)) != 0:
                    audits["no_greedy_or_fallback"] = False
            if work.get("work_drift_mode") != "group_range_additive":
                audits["work_mode_isolated"] = False
            if float(work.get("work_drift_lambda", -1.0)) != FROZEN_LAMBDA:
                audits["work_mode_isolated"] = False
            if int(work.get("work_drift_contexts", 0)) <= 0:
                audits["work_signal_exercised"] = False
            if int(work.get("work_drift_max_candidate_count", 0)) <= 10:
                audits["candidate_supersets_above_10"] = False
            if int(work.get("work_drift_candidate_superset_contexts", 0)) <= 0:
                audits["candidate_supersets_above_10"] = False
            modified_total += int(work.get("work_drift_modified_decisions", 0))
            baseline_order_hash = baseline.get("order_arrival_manifest_sha256")
            work_order_hash = work.get("order_arrival_manifest_sha256")
            if (
                not baseline_order_hash
                or baseline_order_hash != work_order_hash
                or baseline.get("order_arrival_count")
                != work.get("order_arrival_count")
                or baseline.get("order_arrival_manifest_schema_version")
                != ORDER_ARRIVAL_MANIFEST_SCHEMA_VERSION
                or work.get("order_arrival_manifest_schema_version")
                != ORDER_ARRIVAL_MANIFEST_SCHEMA_VERSION
                or bool(baseline.get("order_arrival_replayed"))
                or not bool(work.get("order_arrival_replayed"))
            ):
                audits["exact_paired_order_manifests"] = False
            rows.append({
                "seed": int(seed_text),
                "load": load,
                "baseline": baseline,
                "work": work,
            })
    if tuple(sorted(observed_loads)) != tuple(sorted(REQUIRED_LOADS)):
        audits["exact_loads"] = False
    if len(rows) != len(REQUIRED_LOADS) * len(CERTIFICATION_SEEDS):
        audits["exact_seeds"] = False
    if modified_total <= 0:
        audits["work_signal_exercised"] = False
    audits["modified_decisions_total"] = modified_total
    audits["passed"] = all(
        bool(value) for key, value in audits.items()
        if key not in {"modified_decisions_total", "passed"}
    )
    return rows, audits


def _trajectory_audit(report: Mapping, protocol: Mapping) -> dict:
    semantics = report.get("semantics") or {}
    groups = report.get("groups") or {}
    required_keys = {f"{load}|{WORK_LABEL}" for load in REQUIRED_LOADS}
    exact_groups = set(groups) == required_keys
    exact_seeds = True
    for key in required_keys:
        seeds = tuple(sorted(int(value) for value in (groups.get(key) or {}).get("seeds", ())))
        if seeds != CERTIFICATION_SEEDS:
            exact_seeds = False
    return {
        "schema_version": report.get("schema_version"),
        "compatible_role": (
            semantics.get("five_layer_role")
            == "layer5_normal_arrival_closed_loop_stability"
        ),
        "source_verdict": report.get("verdict"),
        "source_passed": bool(report.get("passed")),
        "exact_work_arm_load_groups": exact_groups,
        "exact_seeds_per_group": exact_seeds,
        "protocol_sha256": protocol["protocol_sha256"],
        "passed": bool(
            semantics.get("five_layer_role")
            == "layer5_normal_arrival_closed_loop_stability"
            and report.get("verdict") == "PASS_CLOSED_LOOP_STABILITY"
            and bool(report.get("passed"))
            and exact_groups
            and exact_seeds
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-bundle", required=True)
    parser.add_argument("--online-reports", nargs="+", required=True)
    parser.add_argument("--trajectory-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite formal Layer-5 report: {output}")
    bundle_path = Path(args.frozen_bundle)
    bundle = _read_json(bundle_path)
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise SystemExit("unexpected frozen Layer-5 bundle schema")
    protocol = bundle.get("formal_protocol") or {}
    if protocol != formal_protocol():
        raise SystemExit("frozen Layer-5 protocol differs from source protocol")

    online_paths = [Path(value) for value in args.online_reports]
    rows, implementation = _load_online_rows(online_paths, protocol)
    trajectory_path = Path(args.trajectory_report)
    trajectory_source = _read_json(trajectory_path)
    trajectory = _trajectory_audit(trajectory_source, protocol)

    metric_specs = {
        "wm_label_cost": ("lower", False),
        "completed_orders": ("higher", False),
        "avg_excess_delay": ("lower", False),
        "open_order_count": ("lower", False),
        "wait_or_stall": ("lower", False),
        "completed_orders_relative": ("higher", True),
        "avg_excess_delay_relative": ("lower", True),
        "open_order_count_relative": ("lower", True),
        "unified_risk": ("lower", False),
        "deadlock_ratio_max": ("lower", False),
    }
    metrics = {}
    for index, (name, (direction, relative)) in enumerate(metric_specs.items()):
        source_name = name.removesuffix("_relative")
        metrics[name] = _metric_report(
            rows,
            source_name,
            direction=direction,
            relative=relative,
            repeats=BOOTSTRAP_REPEATS,
            seed=RANDOM_SEED + 10007 * (index + 1),
        )

    primary = metrics["wm_label_cost"]
    primary_ci = primary["overall"]["ci95"] or [float("inf"), float("inf")]
    primary_loads = primary["by_load"]
    primary_passed = bool(
        primary["overall"]["mean"] is not None
        and primary_ci[1] < 0.0
        and all(
            value.get("mean") is not None and float(value["mean"]) <= 0.0
            for value in primary_loads.values()
        )
    )

    secondary_names = (
        "completed_orders", "avg_excess_delay", "open_order_count", "wait_or_stall"
    )
    secondary_count = sum(
        bool(metrics[name]["favourable_point_estimate"])
        for name in secondary_names
    )
    secondary_passed = secondary_count >= 3

    def ci(name: str) -> list[float]:
        return metrics[name]["overall"]["ci95"] or [float("-inf"), float("inf")]

    guardrail_checks = {
        "completed_orders_relative_ci95_lower": (
            ci("completed_orders_relative")[0] >= -0.02
        ),
        "avg_excess_delay_relative_ci95_upper": (
            ci("avg_excess_delay_relative")[1] <= 0.05
        ),
        "open_order_count_relative_ci95_upper": (
            ci("open_order_count_relative")[1] <= 0.05
        ),
        "unified_risk_absolute_ci95_upper": ci("unified_risk")[1] <= 0.02,
        "deadlock_ratio_max_absolute_ci95_upper": (
            ci("deadlock_ratio_max")[1] <= 0.01
        ),
    }
    guardrails_passed = all(guardrail_checks.values())

    all_checks = {
        "implementation_contract": bool(implementation["passed"]),
        "primary_incremental_value": primary_passed,
        "secondary_support": secondary_passed,
        "noninferiority_guardrails": guardrails_passed,
        "work_arm_closed_loop_trajectory": bool(trajectory["passed"]),
    }
    if not implementation["passed"]:
        verdict = "INVALID_LAYER5_PROTOCOL_OR_DATA"
    elif not guardrails_passed or not trajectory["passed"]:
        verdict = "CLOSED_LOOP_POLICY_FAILURE"
    elif primary_passed and secondary_passed:
        verdict = "PASS_CLOSED_LOOP_STABILITY"
    else:
        verdict = "INCREMENTAL_CLOSED_LOOP_VALUE_INCONCLUSIVE"
    passed = verdict == "PASS_CLOSED_LOOP_STABILITY"

    recommendations = []
    if verdict == "INVALID_LAYER5_PROTOCOL_OR_DATA":
        recommendations.append("repair protocol/data integrity before interpreting outcomes")
    elif verdict == "CLOSED_LOOP_POLICY_FAILURE":
        recommendations.append(
            "keep WorkDrift out of deployment and diagnose the frozen integration; "
            "do not redesign L solely from this policy-level failure"
        )
    elif verdict == "INCREMENTAL_CLOSED_LOOP_VALUE_INCONCLUSIVE":
        recommendations.append(
            "retain Layers 1--4 as supported but do not deploy the auxiliary; "
            "a new preregistered test would be required for a changed integration"
        )
    else:
        recommendations.append(
            "the frozen auxiliary is supported for the tested configuration family; "
            "continue distribution-shift monitoring outside this certificate"
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "layer": 5,
        "name": "frozen_predicted_work_drift_normal_arrival_closed_loop",
        "semantics": {
            "five_layer_role": "layer5_normal_arrival_closed_loop_stability",
            "paired_policy_test": True,
            "normal_arrivals": True,
            "paired_order_manifest_replay": True,
            "fixed_context": True,
            "all_idle_candidates": True,
            "future_unknown_orders_in_rollout": False,
            "continuation_policy_in_rollout": False,
            "td_risk_v_head": False,
            "greedy_or_external_assignment_policy": False,
            "hard_gap_or_load_gate": False,
        },
        "formal_protocol": protocol,
        "parameters": protocol["formal_test"],
        "implementation_audit": implementation,
        "paired_metrics": metrics,
        "primary_contract": {
            "metric": "wm_label_cost",
            "passed": primary_passed,
        },
        "secondary_contract": {
            "favourable": secondary_count,
            "required": 3,
            "passed": secondary_passed,
        },
        "guardrails": {
            "checks": guardrail_checks,
            "passed": guardrails_passed,
        },
        "trajectory_audit": trajectory,
        "groups": trajectory_source.get("groups", {}),
        "failed_policy_groups": trajectory_source.get("failed_policy_groups", []),
        "incomplete_groups": trajectory_source.get("incomplete_groups", []),
        "undercovered_components": trajectory_source.get("undercovered_components", []),
        "checks": all_checks,
        "verdict": verdict,
        "passed": passed,
        "recommendations": recommendations,
        "provenance": {
            "frozen_bundle": str(bundle_path),
            "frozen_bundle_sha256": sha256_file(bundle_path),
            "online_reports": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in online_paths
            ],
            "trajectory_report": str(trajectory_path),
            "trajectory_report_sha256": sha256_file(trajectory_path),
            "post_test_tuning": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {output}")
    print("Layer 5 verdict =", verdict)
    print("Layer 5 passed =", passed)
    print("primary delta =", primary["overall"]["mean"])
    print("primary CI95 =", primary["overall"]["ci95"])
    print("secondary support =", secondary_count, "/ 4")
    print("guardrails passed =", guardrails_passed)
    print("trajectory passed =", trajectory["passed"])


if __name__ == "__main__":
    main()
