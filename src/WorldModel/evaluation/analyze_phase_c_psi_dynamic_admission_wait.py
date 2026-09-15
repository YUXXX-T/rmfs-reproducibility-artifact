"""Validate and summarise the FIFO ``WAITING_ASSIGNED`` development arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from WorldModel.evaluation.run_phase_c_psi_dynamic_admission_wait import (
    ARM_KEY,
    SCHEMA_VERSION,
)


LOADS = ("low", "mid", "high")
SEEDS = tuple(range(551, 561))
METRICS = (
    "completed_orders",
    "completed_tasks",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "stall_ratio_mean",
    "stall_ratio_max",
    "pending_order_count",
    "open_order_count",
)
REFERENCES = (
    "s1_baseline",
    "greedy_manifest",
    "s1_psi_dynamic_committed_v1",
    "s1_psi_dynamic_legacy",
    "s1_psi_dispatch_static",
    "s1_psi_shadow",
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _number(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _mean_ci95(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.mean(values)
    half = 0.0 if len(values) == 1 else 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _outcome(completed_delta: float, deadlock_delta: float) -> str:
    if completed_delta > 0 and deadlock_delta < 0:
        return "completion_up_deadlock_down"
    if completed_delta < 0 and deadlock_delta > 0:
        return "completion_down_deadlock_up"
    if completed_delta > 0 and deadlock_delta > 0:
        return "completion_up_deadlock_up"
    if completed_delta < 0 and deadlock_delta < 0:
        return "completion_down_deadlock_down"
    return "tie_or_single_metric_change"


def analyse(root: Path, require_complete: bool = True) -> dict[str, Any]:
    arm_root = root / "per_arm" / ARM_KEY
    runs: list[dict[str, Any]] = []
    missing: list[str] = []
    for load in LOADS:
        for seed in SEEDS:
            path = arm_root / f"{load}_seed{seed}.json"
            if not path.is_file():
                missing.append(f"{load}{seed}")
                continue
            payload = _read(path)
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"wrong schema: {path}")
            meta = payload.get("meta") or {}
            metrics = payload.get("metrics") or {}
            admission = payload.get("station_admission_audit") or {}
            references = payload.get("reference_metrics") or {}
            deltas: dict[str, Any] = {}
            for reference in REFERENCES:
                ref = references.get(reference)
                if not isinstance(ref, dict):
                    continue
                metric_delta = {
                    key: _number(metrics.get(key)) - _number(ref.get(key))
                    for key in METRICS
                }
                deltas[reference] = {
                    "metrics": metric_delta,
                    "outcome": _outcome(
                        metric_delta["completed_orders"],
                        metric_delta["deadlock_ratio_mean"],
                    ),
                }
            station_final = admission.get("station_metrics_final") or []
            runs.append({
                "path": path.as_posix(),
                "load": str(meta.get("load")),
                "seed": int(meta.get("seed", -1)),
                "ticks": int(meta.get("ticks", -1)),
                "runtime_code_hashes": {
                    key: meta.get(key)
                    for key in (
                        "runner_code_sha256",
                        "station_state_code_sha256",
                        "engine_code_sha256",
                        "prioritized_planner_code_sha256",
                        "counterfactual_code_sha256",
                    )
                },
                "audit_passed": bool((payload.get("audit") or {}).get("passed")),
                "admission_audit_passed": bool(admission.get("passed")),
                "capacity_violation_count": int(admission.get("capacity_violation_count", -1)),
                "token_mismatch_count": int(admission.get("token_mismatch_count", -1)),
                "waiting_semantic_violation_count": int(
                    admission.get("waiting_semantic_violation_count", -1)
                ),
                "fifo_order_violation_count": int(
                    admission.get("fifo_order_violation_count", -1)
                ),
                "waiting_agent_ticks": int(admission.get("waiting_agent_ticks", 0)),
                "waiting_assigned_ratio": float(
                    admission.get("waiting_assigned_ratio", 0.0)
                ),
                "waiting_promotions": int(admission.get("waiting_promotions", 0)),
                "waiting_cancellations": int(
                    admission.get("waiting_cancellations_observed", 0)
                ),
                "waiting_age_p95_ticks": admission.get("waiting_age_p95_ticks"),
                "waiting_age_max_ticks": admission.get("waiting_age_max_ticks"),
                "waiting_duration_p95_ticks": admission.get("waiting_duration_p95_ticks"),
                "waiting_duration_max_ticks": admission.get("waiting_duration_max_ticks"),
                "unresolved_waiter_count_final": int(
                    admission.get("unresolved_waiter_count_final", 0)
                ),
                "oldest_waiter_age_final_ticks": admission.get(
                    "oldest_waiter_age_final_ticks"
                ),
                "max_waiting_depth": admission.get("max_waiting_depth") or {},
                "max_committed_load": admission.get("max_committed_load") or {},
                "capacity_rejections": sum(
                    int(row.get("rejected_capacity", 0)) for row in station_final
                ),
                "waiting_enqueued": sum(
                    int(row.get("waiting_enqueued", 0)) for row in station_final
                ),
                "waiting_granted_counter": sum(
                    int(row.get("waiting_granted", 0)) for row in station_final
                ),
                "waiting_cancelled_counter": sum(
                    int(row.get("waiting_cancelled", 0)) for row in station_final
                ),
                "waiting_requeued": sum(
                    int(row.get("waiting_requeued", 0)) for row in station_final
                ),
                "waiting_fairness_rejected": sum(
                    int(row.get("waiting_fairness_rejected", 0))
                    for row in station_final
                ),
                "metrics": {key: metrics.get(key) for key in METRICS},
                "deltas": deltas,
            })

    if require_complete and missing:
        raise RuntimeError(f"missing {len(missing)} runs: {missing}")

    by_load: dict[str, Any] = {}
    for load in LOADS:
        load_runs = [run for run in runs if run["load"] == load]
        reference_summary: dict[str, Any] = {}
        for reference in REFERENCES:
            paired = [run for run in load_runs if reference in run["deltas"]]
            reference_summary[reference] = {
                "paired_runs": len(paired),
                "metric_deltas": {
                    metric: _mean_ci95([
                        float(run["deltas"][reference]["metrics"][metric])
                        for run in paired
                    ])
                    for metric in METRICS
                },
                "outcome_counts": {
                    name: sum(
                        run["deltas"][reference]["outcome"] == name
                        for run in paired
                    )
                    for name in (
                        "completion_up_deadlock_down",
                        "completion_down_deadlock_up",
                        "completion_up_deadlock_up",
                        "completion_down_deadlock_down",
                        "tie_or_single_metric_change",
                    )
                },
            }
        waiting_runs = [run for run in load_runs if run["waiting_agent_ticks"] > 0]
        by_load[load] = {
            "runs": len(load_runs),
            "runs_with_waiting": len(waiting_runs),
            "capacity_rejections": sum(run["capacity_rejections"] for run in load_runs),
            "waiting_enqueued": sum(run["waiting_enqueued"] for run in load_runs),
            "waiting_granted_counter": sum(
                run["waiting_granted_counter"] for run in load_runs
            ),
            "waiting_cancelled_counter": sum(
                run["waiting_cancelled_counter"] for run in load_runs
            ),
            "waiting_requeued": sum(run["waiting_requeued"] for run in load_runs),
            "waiting_fairness_rejected": sum(
                run["waiting_fairness_rejected"] for run in load_runs
            ),
            "waiting_agent_ticks": _mean_ci95(
                [float(run["waiting_agent_ticks"]) for run in load_runs]
            ),
            "waiting_assigned_ratio": _mean_ci95(
                [float(run["waiting_assigned_ratio"]) for run in load_runs]
            ),
            "waiting_promotions": _mean_ci95(
                [float(run["waiting_promotions"]) for run in load_runs]
            ),
            "waiting_duration_p95_ticks": _mean_ci95([
                float(run["waiting_duration_p95_ticks"])
                for run in waiting_runs
                if run["waiting_duration_p95_ticks"] is not None
            ]),
            "waiting_age_max_ticks": _mean_ci95([
                float(run["waiting_age_max_ticks"])
                for run in waiting_runs
                if run["waiting_age_max_ticks"] is not None
            ]),
            "unresolved_waiter_count_final": _mean_ci95([
                float(run["unresolved_waiter_count_final"])
                for run in load_runs
            ]),
            "oldest_waiter_age_final_ticks": _mean_ci95([
                float(run["oldest_waiter_age_final_ticks"])
                for run in load_runs
                if run["oldest_waiter_age_final_ticks"] is not None
            ]),
            "max_waiting_depth_by_station": {
                str(station): max(
                    int(run["max_waiting_depth"].get(str(station), 0))
                    for run in load_runs
                )
                for station in sorted({
                    station
                    for run in load_runs
                    for station in run["max_waiting_depth"]
                })
            },
            "references": reference_summary,
        }

    runtime_fingerprints = {
        tuple(sorted(run["runtime_code_hashes"].items())) for run in runs
    }
    mechanism_passed = bool(runs) and len(runtime_fingerprints) == 1 and all(
        run["audit_passed"]
        and run["admission_audit_passed"]
        and run["capacity_violation_count"] == 0
        and run["token_mismatch_count"] == 0
        and run["waiting_semantic_violation_count"] == 0
        and run["fifo_order_violation_count"] == 0
        for run in runs
    )
    return {
        "schema_version": "phase_c_psi_dynamic_fifo_wait_report_v2",
        "root": root.as_posix(),
        "development_only": True,
        "summary": {
            "runs": len(runs),
            "expected_runs": len(LOADS) * len(SEEDS),
            "missing": missing,
            "complete": not missing,
            "mechanism_and_lifecycle_passed": mechanism_passed,
            "total_capacity_rejections": sum(run["capacity_rejections"] for run in runs),
            "total_waiting_enqueued": sum(run["waiting_enqueued"] for run in runs),
            "total_waiting_granted": sum(
                run["waiting_granted_counter"] for run in runs
            ),
            "total_waiting_requeued": sum(run["waiting_requeued"] for run in runs),
            "capacity_violation_count": sum(run["capacity_violation_count"] for run in runs),
            "token_mismatch_count": sum(run["token_mismatch_count"] for run in runs),
            "waiting_semantic_violation_count": sum(
                run["waiting_semantic_violation_count"] for run in runs
            ),
            "fifo_order_violation_count": sum(
                run["fifo_order_violation_count"] for run in runs
            ),
            "runs_with_waiting": sum(
                run["waiting_agent_ticks"] > 0 for run in runs
            ),
            "runtime_code_fingerprint_count": len(runtime_fingerprints),
        },
        "by_load": by_load,
        "runs": runs,
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Dynamic-J FIFO station admission waiting",
        "",
        "Development-only paired replay; this is not a new certification.",
        "",
        f"- runs: {summary['runs']}/{summary['expected_runs']}",
        f"- FIFO admission/task lifecycle: {summary['mechanism_and_lifecycle_passed']}",
        f"- runs with explicit waiting: {summary['runs_with_waiting']}",
        f"- FIFO requests enqueued/granted/requeued: "
        f"{summary['total_waiting_enqueued']}/"
        f"{summary['total_waiting_granted']}/"
        f"{summary['total_waiting_requeued']}",
        f"- capacity violations: {summary['capacity_violation_count']}",
        f"- token mismatches: {summary['token_mismatch_count']}",
        f"- waiting semantic violations: {summary['waiting_semantic_violation_count']}",
        f"- FIFO order violations: {summary['fifo_order_violation_count']}",
        f"- runtime code fingerprints: "
        f"{summary['runtime_code_fingerprint_count']}",
        "",
        "| load | reference | n | completed delta | deadlock delta | "
        "completion up/deadlock down | completion down/deadlock up |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for load in LOADS:
        for reference in REFERENCES:
            row = report["by_load"][load]["references"][reference]
            completed = row["metric_deltas"]["completed_orders"]["mean"]
            deadlock = row["metric_deltas"]["deadlock_ratio_mean"]["mean"]
            counts = row["outcome_counts"]
            lines.append(
                f"| {load} | {reference} | {row['paired_runs']} | "
                f"{completed if completed is not None else 'N/A'} | "
                f"{deadlock if deadlock is not None else 'N/A'} | "
                f"{counts['completion_up_deadlock_down']} | "
                f"{counts['completion_down_deadlock_up']} |"
            )
    lines.extend([
        "",
        "Waiting is a simulator admission state and is excluded from physical "
        "deadlock/stall counters.  Its duration and queue depth are reported "
        "separately above and in the JSON report.",
        "",
    ])
    for load in LOADS:
        row = report["by_load"][load]
        wait = row["waiting_duration_p95_ticks"]["mean"]
        lines.append(
            f"- {load}: runs with waiting={row['runs_with_waiting']}, "
            f"mean waiting agent-ticks={row['waiting_agent_ticks']['mean']}, "
            f"mean waiting ratio={row['waiting_assigned_ratio']['mean']}, "
            f"mean per-run wait p95={wait}, "
            f"mean unresolved waiters at end="
            f"{row['unresolved_waiter_count_final']['mean']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = analyse(args.input_root, require_complete=not args.allow_partial)
    validation = args.input_root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    json_path = validation / "dynamic_fifo_wait_validation.json"
    md_path = validation / "dynamic_fifo_wait_validation.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        _markdown(report) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    (validation / "dynamic_fifo_wait_validation.sha256").write_text(
        f"{digest}  {json_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
