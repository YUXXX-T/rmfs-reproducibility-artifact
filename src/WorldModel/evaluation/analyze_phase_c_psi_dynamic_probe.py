"""Summarise isolated dynamic-J online probe outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "completed_orders",
    "completed_tasks",
    "pending_order_count",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "stall_ratio_mean",
    "stall_ratio_max",
    "wall_time_s",
    "assignment_time_ms_mean",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        metric: _num(left.get("metrics", {}).get(metric))
        - _num(right.get("metrics", {}).get(metric))
        for metric in METRICS
    }


def _classify(delta: dict[str, float]) -> str:
    completed = delta["completed_orders"]
    deadlock = delta["deadlock_ratio_mean"]
    if completed > 0 and deadlock < 0:
        return "both_better"
    if completed < 0 and deadlock > 0:
        return "both_worse"
    if completed > 0 and deadlock > 0:
        return "throughput_up_deadlock_up"
    if completed < 0 and deadlock < 0:
        return "throughput_down_deadlock_down"
    return "mixed_or_tie"


def analyse(root: Path) -> dict[str, Any]:
    paths = sorted(
        (root / "per_arm" / "s1_psi_dynamic_probe").glob("*.json")
    )
    runs = []
    for path in paths:
        payload = _read(path)
        metrics = payload.get("metrics") or {}
        refs = payload.get("reference_metrics") or {}
        static_ref = refs.get("s1_psi_dispatch")
        shadow_ref = refs.get("s1_psi_shadow")
        dynamic_payload = {"metrics": metrics}
        delta_static = (
            _delta(dynamic_payload, {"metrics": static_ref})
            if static_ref is not None
            else None
        )
        delta_shadow = (
            _delta(dynamic_payload, {"metrics": shadow_ref})
            if shadow_ref is not None
            else None
        )
        trace = payload.get("dynamic_probe_trace") or []
        runs.append({
            "path": path.as_posix(),
            "load": payload.get("meta", {}).get("load"),
            "seed": int(payload.get("meta", {}).get("seed", -1)),
            "audit_passed": bool((payload.get("audit") or {}).get("passed")),
            "ticks": int(payload.get("meta", {}).get("ticks", -1)),
            "completed_orders": metrics.get("completed_orders"),
            "dynamic_probe_batches": metrics.get("dynamic_probe_batches", 0),
            "dynamic_probe_steps": metrics.get("dynamic_probe_steps", 0),
            "dynamic_probe_selected": metrics.get("dynamic_probe_selected", 0),
            "dynamic_probe_no_selection": metrics.get(
                "dynamic_probe_no_selection", 0
            ),
            "dynamic_probe_order_changed_batches": metrics.get(
                "dynamic_probe_order_changed_batches", 0
            ),
            "dynamic_probe_changed_positions": metrics.get(
                "dynamic_probe_changed_positions", 0
            ),
            "trace_records": len(trace),
            "delta_dynamic_minus_static": delta_static,
            "delta_dynamic_minus_shadow": delta_shadow,
            "outcome_class_vs_static": (
                _classify(delta_static) if delta_static is not None else None
            ),
        })

    def mean_delta(name: str, metric: str) -> float:
        values = [
            _num(run[name].get(metric))
            for run in runs
            if isinstance(run.get(name), dict)
        ]
        return statistics.mean(values) if values else 0.0

    summary = {
        "files": len(runs),
        # ``all([])`` is True; an empty output directory must never be
        # reported as a successful collection.
        "all_audits_passed": bool(runs)
        and all(run["audit_passed"] for run in runs),
        "outcome_noninferiority_not_adjudicated": True,
        "dynamic_minus_static_mean": {
            metric: mean_delta("delta_dynamic_minus_static", metric)
            for metric in METRICS
        },
        "dynamic_minus_shadow_mean": {
            metric: mean_delta("delta_dynamic_minus_shadow", metric)
            for metric in METRICS
        },
        "both_better_count": sum(
            run["outcome_class_vs_static"] == "both_better"
            for run in runs
        ),
        "both_worse_count": sum(
            run["outcome_class_vs_static"] == "both_worse"
            for run in runs
        ),
    }
    return {
        "schema_version": "phase_c_psi_dynamic_probe_report_v1",
        "root": root.as_posix(),
        "summary": summary,
        "runs": runs,
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# Phase-C dynamic-J online probe",
        "",
        "This is an exploratory comparison. It does not adjudicate formal "
        "non-inferiority.",
        "",
        f"- outputs: {s['files']}",
        f"- audits passed: {s['all_audits_passed']}",
        f"- both better vs static: {s['both_better_count']}",
        f"- both worse vs static: {s['both_worse_count']}",
        "",
        "| run | outcome vs static | completed Δ | deadlock Δ | pending Δ | "
        "dynamic batches | J-order changes |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        delta = run["delta_dynamic_minus_static"]
        # Static/shadow reference arms are optional for this isolated probe.
        # Keep the report useful when only the dynamic per-arm JSONs were
        # synchronized to the analysis host instead of failing while
        # formatting a missing reference delta.
        if isinstance(delta, dict):
            completed_delta = f"{delta['completed_orders']:+.3f}"
            deadlock_delta = f"{delta['deadlock_ratio_mean']:+.4f}"
            pending_delta = f"{delta['pending_order_count']:+.3f}"
        else:
            completed_delta = deadlock_delta = pending_delta = "N/A"
        outcome = run["outcome_class_vs_static"] or "N/A"
        lines.append(
            f"| {run['load']}{run['seed']} | {outcome} | "
            f"{completed_delta} | "
            f"{deadlock_delta} | "
            f"{pending_delta} | "
            f"{run['dynamic_probe_batches']} | "
            f"{run['dynamic_probe_order_changed_batches']} |"
        )
    lines.extend([
        "",
        "The dynamic arm refreshes virtual pending debt and the available-robot "
        "view between one-context S1 calls. Service/traffic remain frozen at "
        "the beginning of each tick, so any improvement or regression is still "
        "a probe result rather than a final controller claim.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    report = analyse(args.input_root)
    out = args.input_root / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dynamic_probe_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out / "dynamic_probe_validation.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
