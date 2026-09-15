#!/usr/bin/env python3
"""Build the compact data bundle used by the interactive results explorer."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs/assets/data"

METRICS = {
    "completed_orders": {
        "label": "Completed orders",
        "short_label": "Orders",
        "unit": "orders",
        "direction": "higher",
        "decimals": 1,
        "group": "Throughput",
    },
    "completed_tasks": {
        "label": "Completed tasks",
        "short_label": "Tasks",
        "unit": "tasks",
        "direction": "higher",
        "decimals": 1,
        "group": "Throughput",
    },
    "completion_fraction": {
        "label": "Completion fraction",
        "short_label": "Completion",
        "unit": "fraction",
        "direction": "higher",
        "decimals": 3,
        "group": "Throughput",
    },
    "avg_task_duration": {
        "label": "Average task duration",
        "short_label": "Task duration",
        "unit": "ticks",
        "direction": "lower",
        "decimals": 2,
        "group": "Delay",
    },
    "avg_excess_delay": {
        "label": "Average excess delay",
        "short_label": "Excess delay",
        "unit": "ticks",
        "direction": "lower",
        "decimals": 2,
        "group": "Delay",
    },
    "open_order_count": {
        "label": "Open orders at termination",
        "short_label": "Open orders",
        "unit": "orders",
        "direction": "lower",
        "decimals": 1,
        "group": "Backlog",
    },
    "pending_order_count": {
        "label": "Pending orders at termination",
        "short_label": "Pending orders",
        "unit": "orders",
        "direction": "lower",
        "decimals": 1,
        "group": "Backlog",
    },
    "final_backlog_total": {
        "label": "Final backlog",
        "short_label": "Backlog",
        "unit": "orders",
        "direction": "lower",
        "decimals": 1,
        "group": "Backlog",
    },
    "final_backlog_fraction": {
        "label": "Final backlog fraction",
        "short_label": "Backlog fraction",
        "unit": "fraction",
        "direction": "lower",
        "decimals": 3,
        "group": "Backlog",
    },
    "congestion_events": {
        "label": "Congestion events",
        "short_label": "Congestion",
        "unit": "events",
        "direction": "lower",
        "decimals": 1,
        "group": "Congestion",
    },
    "severe_events": {
        "label": "Severe events",
        "short_label": "Severe events",
        "unit": "events",
        "direction": "lower",
        "decimals": 1,
        "group": "Congestion",
    },
    "stall_ratio_mean": {
        "label": "Mean stall ratio",
        "short_label": "Stall ratio",
        "unit": "ratio",
        "direction": "lower",
        "decimals": 3,
        "group": "Congestion",
    },
    "stall_ratio_max": {
        "label": "Maximum stall ratio",
        "short_label": "Max stall ratio",
        "unit": "ratio",
        "direction": "lower",
        "decimals": 3,
        "group": "Congestion",
    },
    "deadlock_ratio_mean": {
        "label": "Mean deadlock ratio",
        "short_label": "Deadlock ratio",
        "unit": "ratio",
        "direction": "lower",
        "decimals": 3,
        "group": "Congestion",
    },
    "deadlock_ratio_max": {
        "label": "Maximum deadlock ratio",
        "short_label": "Max deadlock ratio",
        "unit": "ratio",
        "direction": "lower",
        "decimals": 3,
        "group": "Congestion",
    },
    "risk_rate_per_100": {
        "label": "Risk events per 100 ticks",
        "short_label": "Risk rate",
        "unit": "events / 100 ticks",
        "direction": "lower",
        "decimals": 2,
        "group": "Congestion",
    },
    "completed_orders_per_congestion_event": {
        "label": "Orders per congestion event",
        "short_label": "Orders / congestion",
        "unit": "orders / event",
        "direction": "higher",
        "decimals": 2,
        "group": "Efficiency",
    },
    "completed_orders_per_severe_event": {
        "label": "Orders per severe event",
        "short_label": "Orders / severe event",
        "unit": "orders / event",
        "direction": "higher",
        "decimals": 2,
        "group": "Efficiency",
    },
    "wall_time_s": {
        "label": "Wall-clock runtime",
        "short_label": "Runtime",
        "unit": "seconds",
        "direction": "lower",
        "decimals": 1,
        "group": "Runtime",
    },
    "assignment_time_ms_mean": {
        "label": "Mean assignment latency",
        "short_label": "Assignment latency",
        "unit": "milliseconds",
        "direction": "lower",
        "decimals": 2,
        "group": "Runtime",
    },
}

ARM_LABELS = {
    "combo_s1_j1": "ComboS1J1",
    "ComboS1J1": "ComboS1J1",
    "greedy": "Greedy",
    "Greedy": "Greedy",
    "hungarian": "Hungarian",
    "Hungarian": "Hungarian",
    "jsq": "JSQ",
    "JSQ": "JSQ",
    "phasec": "WM-Base",
    "PhaseC": "WM-Base",
}

SOURCES = {
    "main": {
        "aggregate": ROOT / "artifacts/tables/table_main.csv",
        "paired": ROOT / "artifacts/statistics/paired_confidence_intervals.csv",
        "label": "Four-station PP",
        "detail": "20×20 · 48 robots · 4 stations · 50 paired seeds · 1,500 ticks",
    },
    "station6": {
        "aggregate": ROOT / "artifacts/tables/table_station6.csv",
        "paired": ROOT / "artifacts/statistics/station6_paired_confidence_intervals.csv",
        "label": "Six-station adaptation",
        "detail": "20×20 · 48 robots · 6 stations · 10 held-out seeds · 1,500 ticks",
    },
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def optional_number(value: str | None) -> float | int | None:
    if value is None or value.strip() == "":
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def aggregate_rows(path: Path) -> list[dict[str, Any]]:
    output = []
    for row in read_rows(path):
        values = {}
        for metric in METRICS:
            mean = optional_number(row.get(f"{metric}_mean"))
            if mean is None:
                continue
            values[metric] = {
                "mean": mean,
                "std": optional_number(row.get(f"{metric}_std")),
                "n": optional_number(row.get(f"{metric}_n")),
            }
        output.append(
            {
                "load": row["load"],
                "arm": ARM_LABELS.get(row["arm"], row["arm"]),
                "n": int(row["n"]),
                "values": values,
            }
        )
    return output


def paired_rows(path: Path) -> list[dict[str, Any]]:
    output = []
    for row in read_rows(path):
        metric = row["metric"]
        if metric not in METRICS:
            continue
        output.append(
            {
                "load": row["load"],
                "proposed": ARM_LABELS.get(row["proposed"], row["proposed"]),
                "baseline": row["baseline_paper_name"],
                "metric": metric,
                "direction": row["direction"],
                "n": int(row["paired_seeds"]),
                "effect": optional_number(row["arm_minus_baseline_mean"]),
                "low": optional_number(row["ci95_low"]),
                "high": optional_number(row["ci95_high"]),
                "wins": int(row["wins"]),
                "losses": int(row["losses"]),
                "ties": int(row["ties"]),
            }
        )
    return output


def build_payload() -> dict[str, Any]:
    datasets = {}
    for key, config in SOURCES.items():
        datasets[key] = {
            "label": config["label"],
            "detail": config["detail"],
            "aggregate": aggregate_rows(config["aggregate"]),
            "paired": paired_rows(config["paired"]),
        }
    return {
        "schema_version": "rmfs_results_explorer_v1",
        "generated_from": [
            "artifacts/tables/table_main.csv",
            "artifacts/tables/table_station6.csv",
            "artifacts/statistics/paired_confidence_intervals.csv",
            "artifacts/statistics/station6_paired_confidence_intervals.csv",
        ],
        "metrics": METRICS,
        "datasets": datasets,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    destination = OUTPUT_DIR / "results_explorer.json"
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for config in SOURCES.values():
        for source in (config["aggregate"], config["paired"]):
            shutil.copyfile(source, OUTPUT_DIR / source.name)
    print(
        "Built interactive results data: "
        f"{sum(len(value['aggregate']) for value in payload['datasets'].values())} "
        "aggregate rows and "
        f"{sum(len(value['paired']) for value in payload['datasets'].values())} paired rows"
    )


if __name__ == "__main__":
    main()
