#!/usr/bin/env python3
"""Regenerate aggregate tables and complete paired confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from bootstrap_ci import mean_ci


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260905
METRIC_DIRECTIONS = {
    "completed_orders": "higher",
    "completed_tasks": "higher",
    "avg_task_duration": "lower",
    "avg_excess_delay": "lower",
    "open_order_count": "lower",
    "pending_order_count": "lower",
    "congestion_events": "lower",
    "severe_events": "lower",
    "stall_ratio_mean": "lower",
    "deadlock_ratio_mean": "lower",
    "risk_rate_per_100": "lower",
    "completion_fraction": "higher",
    "final_backlog_total": "lower",
    "final_backlog_fraction": "lower",
    "completed_orders_per_congestion_event": "higher",
    "completed_orders_per_severe_event": "higher",
}
AGGREGATE_METRICS = tuple(METRIC_DIRECTIONS) + (
    "wall_time_s",
    "assignment_time_ms_mean",
)
SIX_METRICS = (
    "completed_orders",
    "completed_tasks",
    "avg_task_duration",
    "avg_excess_delay",
    "open_order_count",
    "pending_order_count",
    "stall_ratio_mean",
    "stall_ratio_max",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "congestion_events",
    "severe_events",
    "risk_rate_per_100",
    "assignment_time_ms_mean",
    "wall_time_s",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return round(statistics.mean(values), 6), round(statistics.pstdev(values), 6)


def aggregate_rows(
    rows: list[dict[str, str]], metrics: Iterable[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["load"], row["arm"])].append(row)
    output = []
    load_rank = {"low": 0, "mid": 1, "high": 2}
    for (load, arm), group in sorted(grouped.items(), key=lambda item: (load_rank[item[0][0]], item[0][1])):
        record: dict[str, Any] = {"load": load, "arm": arm, "n": len(group)}
        for metric in metrics:
            values = [value for row in group if (value := number(row.get(metric))) is not None]
            mean, std = mean_std(values)
            record[f"{metric}_n"] = len(values)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
        output.append(record)
    return output


def paired_rows(
    rows: list[dict[str, str]],
    *,
    proposed: str,
    baselines: list[tuple[str, str]],
    metrics: dict[str, str],
) -> list[dict[str, Any]]:
    indexed = {(row["load"], int(row["seed"]), row["arm"]): row for row in rows}
    output = []
    for load in ("low", "mid", "high"):
        seeds = sorted({int(row["seed"]) for row in rows if row["load"] == load})
        for baseline, paper_name in baselines:
            for metric, direction in metrics.items():
                deltas = []
                wins = losses = ties = 0
                for seed in seeds:
                    proposed_row = indexed.get((load, seed, proposed))
                    baseline_row = indexed.get((load, seed, baseline))
                    if proposed_row is None or baseline_row is None:
                        continue
                    left = number(proposed_row.get(metric))
                    right = number(baseline_row.get(metric))
                    if left is None or right is None:
                        continue
                    delta = left - right
                    deltas.append(delta)
                    favorable = delta if direction == "higher" else -delta
                    if favorable > 0:
                        wins += 1
                    elif favorable < 0:
                        losses += 1
                    else:
                        ties += 1
                if not deltas:
                    continue
                lo, hi = mean_ci(deltas, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED)
                output.append(
                    {
                        "load": load,
                        "proposed": proposed,
                        "baseline": baseline,
                        "baseline_paper_name": paper_name,
                        "metric": metric,
                        "direction": direction,
                        "paired_seeds": len(deltas),
                        "arm_minus_baseline_mean": round(statistics.mean(deltas), 6),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                    }
                )
    return output


def collapse_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for arm in sorted({row["arm"] for row in rows}):
        group = [row for row in rows if row["load"] == "high" and row["arm"] == arm]
        collapsed = [
            row
            for row in group
            if number(row.get("completed_orders")) is not None
            and number(row.get("deadlock_ratio_mean")) is not None
            and number(row["completed_orders"]) < 300
            and number(row["deadlock_ratio_mean"]) >= 0.4
        ]
        output.append(
            {
                "load": "high",
                "arm": arm,
                "n_seeds": len(group),
                "collapsed_runs": len(collapsed),
                "collapse_fraction": round(len(collapsed) / len(group), 6),
                "definition": "completed_orders < 300 AND deadlock_ratio_mean >= 0.4",
            }
        )
    return output


def runtime_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["campaign"], row["variant"], row["load"], row["arm"])].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        record: dict[str, Any] = dict(zip(("campaign", "variant", "load", "arm"), key))
        for metric in ("wall_time_s", "assignment_time_ms_mean"):
            values = [value for row in group if (value := number(row.get(metric))) is not None]
            mean, std = mean_std(values)
            record[f"{metric}_n"] = len(values)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("all", "main", "paired-ci", "station6", "runtime"), default="all")
    parser.add_argument("--freeze", action="store_true", help="write committed artifact locations")
    args = parser.parse_args()

    main_rows = read_csv(ROOT / "artifacts/raw/main_per_seed.csv")
    six_rows = read_csv(ROOT / "artifacts/raw/station6_per_seed.csv")
    runtime = read_csv(ROOT / "runtime/raw_measurements.csv")
    base = ROOT / "artifacts" if args.freeze else ROOT / "artifacts/generated"
    table_dir = base / "tables"
    stat_dir = base / "statistics"

    if args.only in ("all", "main"):
        write_csv(table_dir / "table_main.csv", aggregate_rows(main_rows, AGGREGATE_METRICS))
        write_csv(table_dir / "table_baselines.csv", aggregate_rows(main_rows, METRIC_DIRECTIONS))
        write_csv(stat_dir / "collapse_summary.csv", collapse_rows(main_rows))
    if args.only in ("all", "paired-ci"):
        paired = paired_rows(
            main_rows,
            proposed="ComboS1J1",
            baselines=[("JSQ", "JSQ"), ("PhaseC", "WM-Base")],
            metrics=METRIC_DIRECTIONS,
        )
        write_csv(stat_dir / "paired_confidence_intervals.csv", paired)
        settings = {
            "schema_version": "paired_bootstrap_settings_v1",
            "unit": "paired simulation seed within one load",
            "statistic": "mean(proposed - baseline)",
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 2.5% and 97.5%",
        }
        stat_dir.mkdir(parents=True, exist_ok=True)
        (stat_dir / "bootstrap_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    if args.only in ("all", "station6"):
        write_csv(table_dir / "table_station6.csv", aggregate_rows(six_rows, SIX_METRICS))
        six_directions = {metric: METRIC_DIRECTIONS.get(metric, "lower") for metric in SIX_METRICS}
        paired = paired_rows(
            six_rows,
            proposed="combo_s1_j1",
            baselines=[("greedy", "Greedy"), ("phasec", "WM-Base")],
            metrics=six_directions,
        )
        write_csv(stat_dir / "station6_paired_confidence_intervals.csv", paired)
    if args.only in ("all", "runtime"):
        destination = ROOT / "runtime/summary.csv" if args.freeze else base / "runtime/summary.csv"
        write_csv(destination, runtime_rows(runtime))

    print(f"Wrote reproduced tables under {base}")


if __name__ == "__main__":
    main()
