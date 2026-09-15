"""Trace-level counterfactual replay for the Phase-C psi dispatcher.

This tool does not run the simulator and does not modify any policy.  It uses
the frozen ``psi_dispatch_trace`` rows emitted by the development ablation and
asks a narrower question:

    If one context is selected, and only that station's unresolved pending
    chain count is reduced before selecting the next context, does the J order
    change?

The replay deliberately keeps the recorded service/traffic channels and age
scores fixed.  It therefore measures a *backlog-only lower bound* on within-
batch J drift.  It does not pretend to know which robot S1 selected or how a
future physical state would evolve.  A positive result is a reason to run the
small online robot-level probe; it is not a closed-loop performance claim.

The script is intentionally standalone and reads only JSON artifacts.  It is
safe to run against the existing ``per_arm`` directory and writes a new report
directory.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_METRICS = (
    "completed_orders",
    "completed_tasks",
    "pending_order_count",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "stall_ratio_mean",
    "stall_ratio_max",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["order_id"]),
        int(row["pod_id"]),
        int(row["station_id"]),
    )


def _key_text(row: Mapping[str, Any]) -> str:
    return (
        f"o{int(row['order_id'])}/p{int(row['pod_id'])}/"
        f"s{int(row['station_id'])}"
    )


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _static_order(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reproduce the frozen static J order from a trace row."""

    return [
        dict(row)
        for row in sorted(
            rows,
            key=lambda row: (
                _finite(row.get("j_score")),
                -_finite(row.get("service_debt")),
                -_finite(row.get("age_score")),
                int(row.get("original_index", 0)),
            ),
        )
    ]


def _infer_station_capacities(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, float], float]:
    """Infer the frozen queue capacity from the recorded bounded backlog.

    The trace schema does not store ``station_capacity``.  Since
    ``backlog_score = x/(1+x)`` with ``x=chains/capacity``, the capacity can be
    recovered whenever both the chain count and score are non-zero.  The
    median is used to suppress floating-point repetition noise.  A fallback of
    one is used only when a station has no recoverable observation.
    """

    estimates: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        station = int(row["station_id"])
        chains = _finite(row.get("unserved_chain_count"))
        score = _finite(row.get("backlog_score"))
        if chains > 0.0 and score > 1e-9 and score < 1.0 - 1e-9:
            capacity = chains * (1.0 - score) / score
            if math.isfinite(capacity) and capacity > 0.0:
                estimates[station].append(capacity)

    capacities: dict[int, float] = {}
    residuals: list[float] = []
    stations = {int(row["station_id"]) for row in rows}
    for station in stations:
        values = estimates.get(station, [])
        capacity = statistics.median(values) if values else 1.0
        capacity = max(float(capacity), 1.0)
        capacities[station] = capacity
        for value in values:
            residuals.append(abs(value - capacity))
    residual = statistics.mean(residuals) if residuals else 0.0
    return capacities, float(residual)


def _backlog_score(chains: float, capacity: float) -> float:
    ratio = max(float(chains), 0.0) / max(float(capacity), 1.0)
    return ratio / (1.0 + ratio)


def _dynamic_order_backlog_only(
    rows: Sequence[Mapping[str, Any]],
    budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, float]]:
    """Recompute J after each hypothetical context removal.

    Only unresolved station-chain work is updated.  Service, traffic, age and
    free-flow values remain at their recorded state values.  This is the
    deliberately conservative diagnostic described in the module docstring.
    """

    capacities, capacity_residual = _infer_station_capacities(rows)
    pending: dict[int, int] = {}
    for row in rows:
        station = int(row["station_id"])
        pending[station] = max(
            pending.get(station, 0),
            int(round(_finite(row.get("unserved_chain_count")))),
        )

    remaining = [dict(row) for row in rows]
    selected: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    limit = min(max(int(budget), 0), len(remaining))

    for step in range(limit):
        ranked: list[tuple[float, float, float, int, dict[str, Any], float]] = []
        for row in remaining:
            station = int(row["station_id"])
            backlog = _backlog_score(
                pending.get(station, 0),
                capacities.get(station, 1.0),
            )
            debt = 0.5 * (backlog + _finite(row.get("age_score")))
            j_score = 0.5 * (
                _finite(row.get("service"))
                + _finite(row.get("traffic"))
            ) - debt
            ranked.append(
                (
                    j_score,
                    -debt,
                    -_finite(row.get("age_score")),
                    int(row.get("original_index", 0)),
                    row,
                    debt,
                )
            )
        ranked.sort(key=lambda item: item[:4])
        j_score, _, _, _, chosen, debt = ranked[0]
        chosen = dict(chosen)
        station = int(chosen["station_id"])
        selected.append(chosen)

        # Capture the complete current dynamic ranking for auditability.
        steps.append(
            {
                "step": int(step),
                "selected": _key_text(chosen),
                "selected_station": station,
                "selected_dynamic_j": float(j_score),
                "selected_recorded_j": _finite(chosen.get("j_score")),
                "selected_dynamic_debt": float(debt),
                "pending_before": {
                    str(key): int(value) for key, value in sorted(pending.items())
                },
                "ranked_keys": [_key_text(item[4]) for item in ranked],
                "ranked_j": [float(item[0]) for item in ranked],
            }
        )

        # A selected context represents one unresolved order/pod chain in the
        # frozen debt definition.  The cap prevents negative virtual work when
        # the trace contains multiple contexts sharing a chain.
        pending[station] = max(pending.get(station, 0) - 1, 0)
        removed = False
        next_remaining = []
        for row in remaining:
            if not removed and _key(row) == _key(chosen):
                removed = True
                continue
            next_remaining.append(row)
        remaining = next_remaining

    return selected, steps, capacities, capacity_residual


def _record_replay(record: Mapping[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in record.get("contexts", [])]
    budget = int(record.get("final_budget", 0))
    static = _static_order(rows)[: min(max(budget, 0), len(rows))]
    dynamic, steps, capacities, capacity_residual = (
        _dynamic_order_backlog_only(rows, budget)
    )
    static_keys = [_key(row) for row in static]
    dynamic_keys = [_key(row) for row in dynamic]
    changed_positions = [
        index
        for index, (left, right) in enumerate(
            zip(static_keys, dynamic_keys)
        )
        if left != right
    ]
    first_difference = (
        min(changed_positions) if changed_positions else None
    )

    # Compare each dynamic choice to the frozen static rank.  This is useful
    # even when the selected prefix happens to remain identical.
    static_rank = {key: index for index, key in enumerate(static_keys)}
    dynamic_rank_displacements = []
    for index, key in enumerate(dynamic_keys):
        if key in static_rank:
            dynamic_rank_displacements.append(
                abs(index - static_rank[key])
            )

    max_j_drift = max(
        (
            abs(
                float(step["selected_dynamic_j"])
                - float(step["selected_recorded_j"])
            )
            for step in steps
        ),
        default=0.0,
    )

    return {
        "tick": int(record.get("tick", -1)),
        "context_count": len(rows),
        "final_budget": budget,
        "static_keys": [_key_text(row) for row in static],
        "dynamic_backlog_only_keys": [_key_text(row) for row in dynamic],
        "changed_positions": changed_positions,
        "changed_position_count": len(changed_positions),
        "first_difference_position": first_difference,
        "max_rank_displacement": max(dynamic_rank_displacements, default=0),
        "max_selected_j_drift": float(max_j_drift),
        "station_capacities_inferred": {
            str(key): float(value) for key, value in sorted(capacities.items())
        },
        "station_capacity_inference_mean_abs_residual": float(
            capacity_residual
        ),
        "steps": steps,
    }


def _outcome_delta(
    applied: Mapping[str, Any],
    shadow: Mapping[str, Any],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in DEFAULT_METRICS:
        left = _finite(applied.get("metrics", {}).get(metric))
        right = _finite(shadow.get("metrics", {}).get(metric))
        result[metric] = left - right
    return result


def _classify(delta: Mapping[str, float]) -> str:
    completed = float(delta.get("completed_orders", 0.0))
    deadlock = float(delta.get("deadlock_ratio_mean", 0.0))
    if completed > 0.0 and deadlock < 0.0:
        return "both_better"
    if completed < 0.0 and deadlock > 0.0:
        return "both_worse"
    if completed > 0.0 and deadlock > 0.0:
        return "throughput_up_deadlock_up"
    if completed < 0.0 and deadlock < 0.0:
        return "throughput_down_deadlock_down"
    return "mixed_or_tie"


def analyse_file(path: Path, shadow_path: Path | None) -> dict[str, Any]:
    payload = _read_json(path)
    traces = payload.get("psi_dispatch_trace", [])
    replay_rows = [_record_replay(record) for record in traces]
    eligible = [row for row in replay_rows if row["final_budget"] > 1]
    changed = [row for row in eligible if row["changed_position_count"] > 0]
    first = next(
        (row for row in replay_rows if row["changed_position_count"] > 0),
        None,
    )
    delta = (
        _outcome_delta(payload, _read_json(shadow_path))
        if shadow_path is not None and shadow_path.is_file()
        else None
    )

    return {
        "file": path.as_posix(),
        "arm": payload.get("meta", {}).get("arm_key"),
        "load": payload.get("meta", {}).get("load"),
        "seed": int(payload.get("meta", {}).get("seed", -1)),
        "completed_orders": _finite(payload.get("metrics", {}).get("completed_orders")),
        "trace_records": len(replay_rows),
        "eligible_records": len(eligible),
        "dynamic_order_changed_records": len(changed),
        "dynamic_order_changed_fraction": (
            len(changed) / len(eligible) if eligible else 0.0
        ),
        "first_dynamic_difference": first,
        "outcome_delta_applied_minus_shadow": delta,
        "outcome_class": _classify(delta) if delta is not None else None,
        "records": replay_rows,
    }


def _fmt_order(keys: Sequence[str]) -> str:
    return " → ".join(keys) if keys else "(empty)"


def make_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Psi-dispatch dynamic J trace replay",
        "",
        "This is a backlog-only counterfactual replay of the frozen per-arm "
        "traces. After each hypothetical context selection, the selected "
        "station's unresolved-chain count is decremented and the remaining "
        "J values are recomputed. Service, traffic, age/free-flow and robot "
        "matching are held fixed.",
        "",
        f"- Files analysed: {report['summary']['files']}",
        f"- Trace records: {report['summary']['trace_records']}",
        f"- Eligible records (budget > 1): {report['summary']['eligible_records']}",
        f"- Records whose selected prefix changes: "
        f"{report['summary']['changed_records']} "
        f"({report['summary']['changed_fraction']:.3f})",
        "",
        "## Per-run summary",
        "",
        "| run | outcome | changed records / eligible | first changed tick | first static order | first dynamic order |",
        "|---|---|---:|---:|---|---|",
    ]
    for run in report["runs"]:
        first = run.get("first_dynamic_difference")
        if first is None:
            tick = "-"
            static = dynamic = "-"
        else:
            tick = str(first["tick"])
            static = _fmt_order(first["static_keys"])
            dynamic = _fmt_order(first["dynamic_backlog_only_keys"])
        lines.append(
            "| {load}{seed} | {outcome} | {changed}/{eligible} | {tick} | "
            "{static} | {dynamic} |".format(
                load=run["load"],
                seed=run["seed"],
                outcome=run.get("outcome_class") or "-",
                changed=run["dynamic_order_changed_records"],
                eligible=run["eligible_records"],
                tick=tick,
                static=static,
                dynamic=dynamic,
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The replay is intentionally not a closed-loop performance test. "
            "A changed prefix means that the one-shot static J order is not "
            "invariant to removing the previously selected station chain. "
            "Because robot IDs and candidate scores are not present in the "
            "per-arm schema, this report cannot decide whether the eventual "
            "closed-loop split is caused by context ranking or by S1's "
            "sequential robot matching.",
            "",
            "The next experiment should therefore be a short online probe "
            "that records selected robot IDs and recomputes J after each "
            "hypothetical selection on the same decision state.",
            "",
        ]
    )
    return "\n".join(lines)


def run(input_root: Path, output_root: Path) -> dict[str, Any]:
    dispatch_dir = input_root / "s1_psi_dispatch"
    if not dispatch_dir.is_dir():
        raise FileNotFoundError(f"missing dispatch trace directory: {dispatch_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in sorted(dispatch_dir.glob("*.json")):
        stem = path.stem
        shadow_path = input_root / "s1_psi_shadow" / f"{stem}.json"
        runs.append(analyse_file(path, shadow_path))

    total_records = sum(int(run["trace_records"]) for run in runs)
    total_eligible = sum(int(run["eligible_records"]) for run in runs)
    total_changed = sum(
        int(run["dynamic_order_changed_records"]) for run in runs
    )
    report = {
        "schema_version": "phase_c_psi_dispatch_dynamic_replay_v1",
        "mode": "trace_counterfactual_backlog_only",
        "input_root": input_root.as_posix(),
        "summary": {
            "files": len(runs),
            "trace_records": total_records,
            "eligible_records": total_eligible,
            "changed_records": total_changed,
            "changed_fraction": (
                total_changed / total_eligible if total_eligible else 0.0
            ),
        },
        "runs": runs,
        "caveats": [
            "service and traffic are held at the recorded tick values",
            "age/free-flow is held fixed; idle-robot removal is not simulated",
            "robot IDs and S1 candidate scores are absent from per-arm traces",
            "this is not a closed-loop outcome replay",
        ],
    }
    json_path = output_root / "dynamic_replay_report.json"
    md_path = output_root / "dynamic_replay_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(make_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.input_root, args.output_root)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"[written] {args.output_root / 'dynamic_replay_report.json'}")
    print(f"[written] {args.output_root / 'dynamic_replay_report.md'}")


if __name__ == "__main__":
    main()
