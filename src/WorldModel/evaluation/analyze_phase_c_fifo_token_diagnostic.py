"""Aggregate matched V1/V2 station token-flow diagnostic replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

from WorldModel.evaluation.run_phase_c_fifo_token_diagnostic import (
    SCHEMA_VERSION,
    sha256_file,
)


REPORT_SCHEMA_VERSION = "phase_c_fifo_token_diagnostic_report_v2"
DEFAULT_ROOT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "fifo_token_diag_551_560_v1"
)
PAIR_METRICS = (
    "completed_orders",
    "completed_tasks",
    "completed_order_flow_time_p95",
    "avg_task_duration",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "stall_ratio_mean",
    "stall_ratio_max",
    "open_order_count",
    "pending_order_count",
)
V1_WARNING_KEYS = frozenset({
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
})


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _ci95(values: Iterable[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    if not rows:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    center = mean(rows)
    if len(rows) == 1:
        return {
            "n": 1,
            "mean": center,
            "ci95_low": center,
            "ci95_high": center,
        }
    # The diagnostic report is descriptive.  A normal approximation keeps it
    # dependency-free; formal outcome testing remains in the frozen reports.
    half = 1.96 * stdev(rows) / math.sqrt(len(rows))
    return {
        "n": len(rows),
        "mean": center,
        "ci95_low": center - half,
        "ci95_high": center + half,
    }


def _parse_case(spec: str) -> tuple[str, int]:
    load, raw_seed = spec.strip().split(":", 1)
    if load not in {"low", "mid", "high"}:
        raise ValueError(f"invalid load in case {spec!r}")
    return load, int(raw_seed)


def _discover(root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted((root / "runs").glob("*/*/diagnostic_summary.json")):
        payload = _read_json(path)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unexpected schema in {path}")
        meta = payload.get("meta") or {}
        mode = str(meta.get("mode"))
        load = str(meta.get("load"))
        seed = int(meta.get("seed"))
        key = (mode, load, seed)
        if key in rows:
            raise ValueError(f"duplicate diagnostic summary for {key}")
        trace = payload.get("trace") or {}
        trace_path = Path(str(trace.get("path", "")))
        if not trace_path.is_file():
            # Server reports use project-relative paths.  Resolve against cwd
            # before rejecting a trace that was moved with the result root.
            candidate = root / "runs" / mode / f"{load}_seed{seed}" / trace_path.name
            trace_path = candidate
        trace_ok = bool(
            trace_path.is_file()
            and sha256_file(trace_path) == trace.get("sha256")
        )
        payload["_path"] = path.as_posix()
        payload["_trace_integrity_passed"] = trace_ok
        rows[key] = payload
    return rows


def _max_station_value(flow: dict[str, Any], key: str) -> int:
    values = [
        int(row.get(key, 0))
        for row in (flow.get("by_station") or {}).values()
    ]
    return max(values, default=0)


def _mode_reference_gate(payload: dict[str, Any]) -> dict[str, Any]:
    """Re-evaluate an embedded audit without rerunning the simulator.

    Older diagnostic summaries predate the mode-aware gate and therefore have
    ``reference_audit.passed=False`` for V1 deadlock drift.  This function
    intentionally reconstructs the gate from the per-field checks so those
    existing summaries can be analyzed in place.
    """
    meta = payload.get("meta") or {}
    mode = str(meta.get("mode"))
    audit = payload.get("reference_audit") or {}
    checks = audit.get("checks") or {}
    if not checks or bool(audit.get("reference_missing")):
        return {
            "hard_passed": False,
            "warning_fields": [],
            "warnings": [],
            "legacy_passed": bool(audit.get("passed")),
            "reason": "missing_reference_checks",
        }
    warning_keys = V1_WARNING_KEYS if mode == "committed_v1" else frozenset()
    hard_failures = []
    warnings = []
    for key, row in checks.items():
        passed = bool(row.get("passed"))
        if key in warning_keys:
            if not passed:
                warnings.append({
                    "key": key,
                    "actual": row.get("actual"),
                    "expected": row.get("expected"),
                    "delta": (
                        float(row["actual"]) - float(row["expected"])
                        if row.get("actual") is not None
                        and row.get("expected") is not None
                        else None
                    ),
                })
        elif not passed:
            hard_failures.append(key)
    # Preserve any richer warning payload produced by the updated runner.
    for warning in audit.get("warnings") or []:
        if warning.get("key") not in {row["key"] for row in warnings}:
            warnings.append(warning)
    return {
        "hard_passed": not hard_failures,
        "hard_failures": hard_failures,
        "warning_fields": [row["key"] for row in warnings],
        "warnings": warnings,
        "legacy_passed": bool(audit.get("passed")),
        "reason": "mode_aware_reference_gate",
    }


def _classify_station(
    station: dict[str, Any], station_diag: dict[str, Any]
) -> str:
    waiting = int(station.get("waiting_depth", 0))
    full = int(station.get("committed_load", 0)) >= int(
        station.get("capacity", 0)
    )
    path_failures = int(
        (station_diag.get("event_counts") or {}).get(
            "deliver_path_failure", 0
        )
    )
    token_stall = int(
        station_diag.get("max_in_transit_stationary_streak", 0)
    )
    exit_stall = int(station_diag.get("max_exit_blocked_streak", 0))
    promotable_streak = int(
        station_diag.get("max_stable_promotable_head_streak", 0)
    )
    if waiting <= 0:
        return "no_final_waiter"
    if not full:
        if promotable_streak > 1:
            return "fifo_liveness_suspected"
        return "horizon_edge_or_activation_retry"
    if token_stall >= 10 and path_failures > 0:
        return "capacity_full_with_in_transit_path_stall"
    if exit_stall >= 10:
        return "capacity_full_with_exit_blocking"
    return "capacity_full_service_or_in_transit_backlog"


def _compact_run(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload["meta"]
    flow = payload.get("token_flow") or {}
    fifo = payload.get("fifo_invariant_audit") or {}
    reference_gate = _mode_reference_gate(payload)
    by_station = flow.get("by_station") or {}
    final_stations = []
    for station in flow.get("final_stations") or []:
        sid = str(int(station["station_id"]))
        diag = by_station.get(sid) or {}
        final_stations.append({
            "station_id": int(station["station_id"]),
            "capacity": int(station.get("capacity", 0)),
            "occupancy": int(station.get("occupancy", 0)),
            "committed_load": int(station.get("committed_load", 0)),
            "in_transit_agent_ids": station.get("in_transit_agent_ids") or [],
            "waiting_depth": int(station.get("waiting_depth", 0)),
            "oldest_waiting_age": max(
                [int(row.get("age", 0)) for row in station.get("waiting") or []],
                default=0,
            ),
            "classification": _classify_station(station, diag),
            **{
                key: int(diag.get(key, 0))
                for key in (
                    "max_waiting_depth",
                    "max_waiting_age",
                    "max_token_age",
                    "max_in_transit_stationary_streak",
                    "max_full_wait_streak",
                    "max_stable_promotable_head_streak",
                    "max_exit_blocked_streak",
                )
            },
            "event_counts": diag.get("event_counts") or {},
        })
    return {
        "mode": meta["mode"],
        "load": meta["load"],
        "seed": int(meta["seed"]),
        "path": payload["_path"],
        "reference_hard_passed": bool(reference_gate["hard_passed"]),
        "reference_warning_fields": reference_gate["warning_fields"],
        "reference_warnings": reference_gate["warnings"],
        # Retain the old field for consumers that already read it, but make
        # the new mode-aware field authoritative in validation decisions.
        "reference_match_legacy": bool(reference_gate["legacy_passed"]),
        "trace_integrity_passed": bool(payload["_trace_integrity_passed"]),
        "fifo_invariant_passed": (
            bool(fifo.get("passed")) if meta["mode"] == "fifo_v2" else None
        ),
        "metrics": {
            key: (payload.get("metrics") or {}).get(key) for key in PAIR_METRICS
        },
        "event_counts": flow.get("event_counts") or {},
        "requeue_reasons": flow.get("requeue_reasons") or {},
        "max_in_transit_stationary_streak": _max_station_value(
            flow, "max_in_transit_stationary_streak"
        ),
        "max_full_wait_streak": _max_station_value(
            flow, "max_full_wait_streak"
        ),
        "max_stable_promotable_head_streak": _max_station_value(
            flow, "max_stable_promotable_head_streak"
        ),
        "max_exit_blocked_streak": _max_station_value(
            flow, "max_exit_blocked_streak"
        ),
        "unresolved_waiter_count_final": int(
            fifo.get("unresolved_waiter_count_final", 0)
        ),
        "final_stations": final_stations,
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# FIFO token-flow diagnostic",
        "",
        "Matched development replay; no policy or simulator behaviour change.",
        "",
        f"- cases: {summary['complete_pairs']}/{summary['expected_pairs']}",
        f"- strict reference hard checks: {summary['reference_hard_passes']}/{summary['runs']}",
        f"- V1 baseline-drift warning runs: {summary['v1_warning_runs']}",
        f"- trace integrity: {summary['trace_integrity_passes']}/{summary['runs']}",
        f"- FIFO invariant passes: {summary['fifo_invariant_passes']}/{summary['fifo_runs']}",
        f"- suspected FIFO liveness failures: {summary['fifo_liveness_suspected']}",
        "",
        "| case | completed V2-V1 | flow-p95 V2-V1 | final waiters | "
        "max token-stall | max exit-block | max stable-promotable | classification |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for pair in report["pairs"]:
        classifications = sorted({
            row["classification"]
            for row in pair["v2"]["final_stations"]
            if row["classification"] != "no_final_waiter"
        })
        lines.append(
            f"| {pair['load']}{pair['seed']} | "
            f"{pair['metric_deltas']['completed_orders']:+.1f} | "
            f"{pair['metric_deltas']['completed_order_flow_time_p95']:+.1f} | "
            f"{pair['v2']['unresolved_waiter_count_final']} | "
            f"{pair['v2']['max_in_transit_stationary_streak']} | "
            f"{pair['v2']['max_exit_blocked_streak']} | "
            f"{pair['v2']['max_stable_promotable_head_streak']} | "
            f"{', '.join(classifications) if classifications else 'none'} |"
        )
    lines.extend([
        "",
        "V1 deadlock baseline drift warnings (informational; not fatal):",
        "",
    ])
    for pair in report["pairs"]:
        warnings = pair["v1"].get("reference_warnings") or []
        if warnings:
            fields = ", ".join(
                f"{row['key']}={row.get('delta')}" for row in warnings
            )
            lines.append(f"- {pair['load']}{pair['seed']}: {fields}")
    lines.extend([
        "",
        "Requeue reasons (FIFO V2):",
        "",
    ])
    for key, value in sorted(report["aggregate"]["v2_requeue_reasons"].items()):
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--expected-cases",
        default="",
        help="comma-separated load:seed list; empty means infer completed pairs",
    )
    args = parser.parse_args()

    rows = _discover(args.input_root)
    if args.expected_cases.strip():
        expected_cases = [
            _parse_case(spec)
            for spec in args.expected_cases.split(",")
            if spec.strip()
        ]
    else:
        cases = {(load, seed) for _, load, seed in rows}
        expected_cases = sorted(cases)

    missing = []
    pairs = []
    compact_runs = []
    for load, seed in expected_cases:
        v1_payload = rows.get(("committed_v1", load, seed))
        v2_payload = rows.get(("fifo_v2", load, seed))
        if v1_payload is None:
            missing.append(f"committed_v1:{load}:{seed}")
        if v2_payload is None:
            missing.append(f"fifo_v2:{load}:{seed}")
        if v1_payload is None or v2_payload is None:
            continue
        v1 = _compact_run(v1_payload)
        v2 = _compact_run(v2_payload)
        compact_runs.extend([v1, v2])
        metric_deltas = {}
        for key in PAIR_METRICS:
            left = v2["metrics"].get(key)
            right = v1["metrics"].get(key)
            metric_deltas[key] = (
                float(left) - float(right)
                if left is not None and right is not None
                else None
            )
        pairs.append({
            "load": load,
            "seed": seed,
            "v1": v1,
            "v2": v2,
            "metric_deltas": metric_deltas,
        })

    requeue_reasons: Counter[str] = Counter()
    classifications: Counter[str] = Counter()
    for pair in pairs:
        requeue_reasons.update(pair["v2"]["requeue_reasons"])
        for station in pair["v2"]["final_stations"]:
            classifications[station["classification"]] += 1

    fifo_runs = [row for row in compact_runs if row["mode"] == "fifo_v2"]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input_root": args.input_root.as_posix(),
        "development_only": True,
        "summary": {
            "runs": len(compact_runs),
            "expected_runs": 2 * len(expected_cases),
            "complete_pairs": len(pairs),
            "expected_pairs": len(expected_cases),
            "missing": missing,
            "reference_hard_passes": sum(
                bool(row["reference_hard_passed"]) for row in compact_runs
            ),
            # Compatibility alias; unlike the old report this is now
            # mode-aware (V1 deadlock drift is excluded from the gate).
            "reference_matches": sum(
                bool(row["reference_hard_passed"]) for row in compact_runs
            ),
            "v2_strict_reference_passes": sum(
                bool(row["reference_hard_passed"])
                for row in compact_runs
                if row["mode"] == "fifo_v2"
            ),
            "v1_hard_reference_passes": sum(
                bool(row["reference_hard_passed"])
                for row in compact_runs
                if row["mode"] == "committed_v1"
            ),
            "v1_warning_runs": sum(
                bool(row["reference_warning_fields"])
                for row in compact_runs
                if row["mode"] == "committed_v1"
            ),
            "v1_warning_fields": sorted({
                field
                for row in compact_runs
                if row["mode"] == "committed_v1"
                for field in row["reference_warning_fields"]
            }),
            "trace_integrity_passes": sum(
                bool(row["trace_integrity_passed"]) for row in compact_runs
            ),
            "fifo_runs": len(fifo_runs),
            "fifo_invariant_passes": sum(
                bool(row["fifo_invariant_passed"]) for row in fifo_runs
            ),
            "fifo_liveness_suspected": sum(
                row["max_stable_promotable_head_streak"] > 1
                for row in fifo_runs
            ),
            "complete": bool(
                not missing
                and all(
                    row["reference_hard_passed"]
                    for row in compact_runs
                )
                and all(row["trace_integrity_passed"] for row in compact_runs)
                and all(row["fifo_invariant_passed"] for row in fifo_runs)
            ),
        },
        "paired_metric_deltas": {
            key: _ci95(
                pair["metric_deltas"][key]
                for pair in pairs
                if pair["metric_deltas"][key] is not None
            )
            for key in PAIR_METRICS
        },
        "aggregate": {
            "v2_requeue_reasons": dict(sorted(requeue_reasons.items())),
            "final_station_classifications": dict(
                sorted(classifications.items())
            ),
        },
        "pairs": pairs,
    }

    validation = args.input_root / "validation"
    json_path = validation / "fifo_token_diagnostic_validation.json"
    md_path = validation / "fifo_token_diagnostic_validation.md"
    sha_path = validation / "fifo_token_diagnostic_validation.sha256"
    _atomic_json(json_path, report)
    md_path.write_text(_markdown(report), encoding="utf-8")
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    sha_path.write_text(
        f"{digest}  {json_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"[complete] {json_path}")
    if not report["summary"]["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
