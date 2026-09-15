"""Validate and summarise the committed-capacity Dynamic-J development arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from WorldModel.evaluation.run_phase_c_psi_dynamic_admission import (
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
    "pending_order_count",
    "open_order_count",
)
REFERENCES = (
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
    if len(values) == 1:
        half = 0.0
    else:
        half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
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
                "audit_passed": bool((payload.get("audit") or {}).get("passed")),
                "admission_audit_passed": bool(admission.get("passed")),
                "capacity_violation_count": int(
                    admission.get("capacity_violation_count", -1)
                ),
                "token_mismatch_count": int(
                    admission.get("token_mismatch_count", -1)
                ),
                "capacity_rejections": sum(
                    int(row.get("rejected_capacity", 0))
                    for row in station_final
                ),
                "max_committed_load": admission.get("max_committed_load") or {},
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
            paired = [
                run for run in load_runs if reference in run["deltas"]
            ]
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
        by_load[load] = {
            "runs": len(load_runs),
            "capacity_rejections": sum(
                run["capacity_rejections"] for run in load_runs
            ),
            "references": reference_summary,
        }

    mechanism_passed = bool(runs) and all(
        run["audit_passed"]
        and run["admission_audit_passed"]
        and run["capacity_violation_count"] == 0
        and run["token_mismatch_count"] == 0
        for run in runs
    )
    return {
        "schema_version": "phase_c_psi_dynamic_committed_admission_report_v1",
        "root": root.as_posix(),
        "development_only": True,
        "summary": {
            "runs": len(runs),
            "expected_runs": len(LOADS) * len(SEEDS),
            "missing": missing,
            "complete": not missing,
            "mechanism_and_lifecycle_passed": mechanism_passed,
            "total_capacity_rejections": sum(
                run["capacity_rejections"] for run in runs
            ),
            "capacity_violation_count": sum(
                run["capacity_violation_count"] for run in runs
            ),
            "token_mismatch_count": sum(
                run["token_mismatch_count"] for run in runs
            ),
        },
        "by_load": by_load,
        "runs": runs,
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Dynamic-J committed-capacity station admission",
        "",
        "Development-only paired replay; this is not a new certification.",
        "",
        f"- runs: {summary['runs']}/{summary['expected_runs']}",
        f"- admission invariant and token lifecycle: "
        f"{summary['mechanism_and_lifecycle_passed']}",
        f"- capacity violations: {summary['capacity_violation_count']}",
        f"- token mismatches: {summary['token_mismatch_count']}",
        f"- capacity rejections: {summary['total_capacity_rejections']}",
        "",
        "| load | reference | n | completed delta | deadlock delta | "
        "better/both | worse/both |",
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
        "A capacity rejection means an additional DELIVER robot was held back "
        "because physical occupants plus admitted in-transit robots already "
        "equalled station capacity.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    report = analyse(args.input_root, require_complete=not args.allow_partial)
    validation = args.input_root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    json_path = validation / "dynamic_admission_validation.json"
    md_path = validation / "dynamic_admission_validation.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    (validation / "dynamic_admission_validation.sha256").write_text(
        f"{digest}  {json_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
