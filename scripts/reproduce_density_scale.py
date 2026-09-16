#!/usr/bin/env python3
"""Reproduce fixed-four-station density/scale tables and static figures."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOADS = ("low", "mid", "high")
MAP_SIZES = (20, 30, 40)
DENSITIES = (0.12, 0.15, 0.18)
ARMS = ("greedy", "hungarian", "jsq", "phasec", "combo_s1_j1")
METRICS = (
    "completed_orders",
    "avg_excess_delay",
    "deadlock_ratio_mean",
    "risk_rate_per_100",
)
COMPARISONS = (
    ("phasec", "greedy"),
    ("phasec", "jsq"),
    ("combo_s1_j1", "greedy"),
    ("combo_s1_j1", "jsq"),
    ("combo_s1_j1", "phasec"),
)
LABELS = {
    "greedy": "Greedy",
    "hungarian": "Hungarian",
    "jsq": "JSQ",
    "phasec": "WM-Base",
    "combo_s1_j1": "Proposed (S1+J1)",
}
COLORS = {
    "greedy": "#8B969F",
    "hungarian": "#D28A18",
    "jsq": "#4B8B65",
    "phasec": "#4D88C7",
    "combo_s1_j1": "#0F4D92",
}
MARKERS = {
    "greedy": "o",
    "hungarian": "s",
    "jsq": "^",
    "phasec": "D",
    "combo_s1_j1": "*",
}
METRIC_LABELS = {
    "completed_orders": "Completed orders $\\uparrow$",
    "avg_excess_delay": "Average excess delay $\\downarrow$",
    "deadlock_ratio_mean": "Mean deadlock ratio $\\downarrow$",
    "risk_rate_per_100": "Risk events / 100 ticks $\\downarrow$",
}
BOOTSTRAP_DRAWS = 50_000
BOOTSTRAP_SEED = 20260916
T_CRITICAL_DF9 = 2.2621571628540993


def read_rows(path: Path) -> list[dict[str, object]]:
    integer_fields = {
        "seed", "map_rows", "map_cols", "num_robots", "num_stations",
        "order_arrival_count", "completed_orders", "completed_tasks",
        "open_order_count", "pending_order_count", "congestion_events", "severe_events",
    }
    categorical = {"variant", "arm", "load"}
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = {}
            for key, value in raw.items():
                if key in categorical:
                    row[key] = value
                elif key in integer_fields:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def validate(rows: list[dict[str, object]]) -> None:
    expected = {
        (size, density, load, arm, seed)
        for size in MAP_SIZES
        for density in DENSITIES
        for load in LOADS
        for arm in ARMS
        for seed in range(701, 711)
    }
    actual = {
        (
            int(row["map_rows"]),
            round(float(row["robot_density"]), 2),
            str(row["load"]),
            str(row["arm"]),
            int(row["seed"]),
        )
        for row in rows
    }
    if len(rows) != 1350 or actual != expected:
        raise ValueError("density/scale rows do not form the expected 1,350-run grid")
    if any(
        int(row["map_rows"]) != int(row["map_cols"])
        or int(row["num_stations"]) != 4
        for row in rows
    ):
        raise ValueError("campaign must use square maps and a fixed four-station interface")

    arrival_counts: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in rows:
        size = int(row["map_rows"])
        robots = int(row["num_robots"])
        density = round(float(row["robot_density"]), 2)
        if robots != round(size * size * density):
            raise ValueError("robot count is inconsistent with the recorded map density")
        if row["variant"] != f"map{size}_r{robots}_s4":
            raise ValueError("variant name is inconsistent with the map and robot count")
        arrival_counts[(str(row["load"]), int(row["seed"]))].add(
            int(row["order_arrival_count"])
        )
    if len(arrival_counts) != 30 or any(len(values) != 1 for values in arrival_counts.values()):
        raise ValueError("arrival counts are not paired across policies and scale variants")


def mean_ci_t(values: Iterable[float]) -> tuple[float, float, float]:
    array = list(values)
    mean = statistics.mean(array)
    half = T_CRITICAL_DF9 * statistics.stdev(array) / math.sqrt(len(array))
    return mean, mean - half, mean + half


def stratified_bootstrap(
    clusters: dict[tuple[str, int], float], rng: np.random.Generator
) -> tuple[float, float, float]:
    by_load = [
        np.asarray([clusters[(load, seed)] for seed in range(701, 711)], dtype=float)
        for load in LOADS
    ]
    boot = np.zeros(BOOTSTRAP_DRAWS, dtype=float)
    for values in by_load:
        sampled = values[
            rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
        ]
        boot += sampled.mean(axis=1) / len(LOADS)
    low, high = np.quantile(boot, [0.025, 0.975])
    return (
        float(np.mean(np.concatenate(by_load))),
        float(low),
        float(high),
    )


def cell_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["map_rows"], row["robot_density"], row["num_robots"],
            row["load"], row["arm"],
        )
        groups[key].append(row)
    output = []
    for key in sorted(groups):
        group = groups[key]
        record: dict[str, object] = dict(
            zip(("map_size", "robot_density", "num_robots", "load", "arm"), key)
        )
        record["n"] = len(group)
        for metric in METRICS:
            mean, low, high = mean_ci_t(float(row[metric]) for row in group)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        output.append(record)
    return output


def cluster_metric(
    rows: list[dict[str, object]], metric: str
) -> dict[tuple[str, int], float]:
    values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        values[(str(row["load"]), int(row["seed"]))].append(float(row[metric]))
    return {key: statistics.mean(group) for key, group in values.items()}


def map_summary(
    rows: list[dict[str, object]], rng: np.random.Generator
) -> list[dict[str, object]]:
    output = []
    for size in MAP_SIZES:
        for arm in ARMS:
            subset = [
                row for row in rows
                if int(row["map_rows"]) == size and row["arm"] == arm
            ]
            record: dict[str, object] = {"map_size": size, "arm": arm, "n_runs": len(subset)}
            for metric in METRICS:
                mean, low, high = stratified_bootstrap(cluster_metric(subset, metric), rng)
                record[f"{metric}_mean"] = mean
                record[f"{metric}_ci_low"] = low
                record[f"{metric}_ci_high"] = high
            output.append(record)
    return output


def paired_effects(
    rows: list[dict[str, object]], rng: np.random.Generator
) -> list[dict[str, object]]:
    indexed = {
        (str(row["load"]), int(row["seed"]), str(row["variant"]), str(row["arm"])): row
        for row in rows
    }
    output = []
    for scope in ("all", "20", "30", "40"):
        scoped = rows if scope == "all" else [row for row in rows if int(row["map_rows"]) == int(scope)]
        variants = sorted({str(row["variant"]) for row in scoped})
        for method, baseline in COMPARISONS:
            clusters: dict[tuple[str, int], float] = {}
            for load in LOADS:
                for seed in range(701, 711):
                    differences = [
                        float(indexed[(load, seed, variant, method)]["completed_orders"])
                        - float(indexed[(load, seed, variant, baseline)]["completed_orders"])
                        for variant in variants
                    ]
                    clusters[(load, seed)] = statistics.mean(differences)
            mean, low, high = stratified_bootstrap(clusters, rng)
            values = list(clusters.values())
            output.append(
                {
                    "map_scope": scope,
                    "method": method,
                    "baseline": baseline,
                    "paired_mean_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_load_seed_clusters": sum(value > 0 for value in values),
                    "tied_load_seed_clusters": sum(value == 0 for value in values),
                    "clusters": len(values),
                    "cluster_definition": "load-seed; mean over included map-density variants",
                }
            )
    return output


def cell_membership(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, float, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["map_rows"]), float(row["robot_density"]),
            str(row["load"]), str(row["arm"]),
        )
        groups[key].append(row)
    means = {
        key: {metric: statistics.mean(float(row[metric]) for row in group) for metric in METRICS}
        for key, group in groups.items()
    }
    output = []
    for size in MAP_SIZES:
        for density in DENSITIES:
            for load in LOADS:
                vectors = {
                    arm: np.asarray(
                        [
                            -means[(size, density, load, arm)]["completed_orders"],
                            means[(size, density, load, arm)]["avg_excess_delay"],
                            means[(size, density, load, arm)]["deadlock_ratio_mean"],
                            means[(size, density, load, arm)]["risk_rate_per_100"],
                        ]
                    )
                    for arm in ARMS
                }
                best = max(means[(size, density, load, arm)]["completed_orders"] for arm in ARMS)
                for arm in ARMS:
                    vector = vectors[arm]
                    dominated = any(
                        other != arm
                        and np.all(other_vector <= vector)
                        and np.any(other_vector < vector)
                        for other, other_vector in vectors.items()
                    )
                    output.append(
                        {
                            "map_size": size,
                            "robot_density": density,
                            "load": load,
                            "arm": arm,
                            "completed_orders_mean": means[(size, density, load, arm)]["completed_orders"],
                            "throughput_winner": means[(size, density, load, arm)]["completed_orders"] == best,
                            "four_metric_pareto_member": not dominated,
                        }
                    )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.edgecolor": "#737E87",
            "axes.linewidth": 0.65,
            "axes.grid": True,
            "grid.color": "#D8DEE3",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=2.2, width=0.55, pad=1.8)


def save_figure(fig: plt.Figure, stem: Path, docs_dir: Path | None) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=240, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={
            "Title": stem.name.replace("_", " ").title(),
            "Author": "Anonymous Authors",
            "Creator": "RMFS reproducibility artifact",
        },
    )
    if docs_dir is not None:
        docs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(png, docs_dir / png.name)
    plt.close(fig)


def plot_factorial(rows: list[dict[str, object]], output: Path, docs_dir: Path | None) -> None:
    lookup = {
        (int(row["map_size"]), float(row["robot_density"]), str(row["load"]), str(row["arm"])): row
        for row in rows
    }
    fig, axes = plt.subplots(3, 3, figsize=(9.0, 6.1), sharex=True, sharey=True)
    robot_labels = {0.12: "48 / 108 / 192", 0.15: "60 / 135 / 240", 0.18: "72 / 162 / 288"}
    for row_index, load in enumerate(LOADS):
        for column_index, density in enumerate(DENSITIES):
            axis = axes[row_index, column_index]
            for arm in ARMS:
                values = [lookup[(size, density, load, arm)] for size in MAP_SIZES]
                y = np.asarray([float(value["completed_orders_mean"]) for value in values])
                low = np.asarray([float(value["completed_orders_ci_low"]) for value in values])
                high = np.asarray([float(value["completed_orders_ci_high"]) for value in values])
                axis.fill_between(MAP_SIZES, low, high, color=COLORS[arm], alpha=0.075)
                axis.plot(
                    MAP_SIZES, y, color=COLORS[arm], marker=MARKERS[arm],
                    markersize=4.4 if arm != "combo_s1_j1" else 6.2,
                    linewidth=1.25 if arm != "combo_s1_j1" else 1.75,
                    label=LABELS[arm], zorder=3,
                )
            if row_index == 0:
                axis.set_title(
                    f"Robot density {density:.2f}\n({robot_labels[density]} robots)",
                    weight="bold", pad=4,
                )
            axis.text(
                0.02, 0.96, f"{load.capitalize()} load", transform=axis.transAxes,
                ha="left", va="top", fontsize=7.0, weight="bold",
            )
            axis.set_xticks(MAP_SIZES)
            axis.set_xlim(18.6, 41.4)
            axis.set_ylim(125, 525)
            style_axis(axis)
            if column_index == 0:
                axis.set_ylabel("Completed orders")
            if row_index == 2:
                axis.set_xlabel("Map size ($N \\times N$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=5, frameon=False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.09, top=0.865, wspace=0.14, hspace=0.20)
    save_figure(fig, output / "density_scale_completed_orders_factorial", docs_dir)


def plot_endpoints(rows: list[dict[str, object]], output: Path, docs_dir: Path | None) -> None:
    lookup = {(int(row["map_size"]), str(row["arm"])): row for row in rows}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 4.8), sharex=True)
    for axis, metric, tag in zip(axes.flat, METRICS, ("(a)", "(b)", "(c)", "(d)")):
        for arm in ARMS:
            values = [lookup[(size, arm)] for size in MAP_SIZES]
            y = np.asarray([float(value[f"{metric}_mean"]) for value in values])
            low = np.asarray([float(value[f"{metric}_ci_low"]) for value in values])
            high = np.asarray([float(value[f"{metric}_ci_high"]) for value in values])
            axis.fill_between(MAP_SIZES, low, high, color=COLORS[arm], alpha=0.08)
            axis.plot(
                MAP_SIZES, y, color=COLORS[arm], marker=MARKERS[arm],
                markersize=4.5 if arm != "combo_s1_j1" else 6.4,
                linewidth=1.3 if arm != "combo_s1_j1" else 1.8,
                label=LABELS[arm], zorder=3,
            )
        axis.set_title(f"{tag} {METRIC_LABELS[metric]}", loc="left", weight="bold", pad=4)
        axis.set_xticks(MAP_SIZES)
        axis.set_xlim(18.8, 41.2)
        style_axis(axis)
    for axis in axes[1, :]:
        axis.set_xlabel("Map size ($N \\times N$)")
    axes[0, 0].set_ylim(245, 390)
    axes[0, 1].set_ylim(20, 70)
    axes[1, 0].set_ylim(0.18, 0.57)
    axes[1, 1].set_ylim(45, 90)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.12, top=0.84, wspace=0.18, hspace=0.31)
    save_figure(fig, output / "density_scale_four_endpoint_summary", docs_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze", action="store_true",
        help="refresh committed tables, statistics, figures, and MkDocs PNG copies",
    )
    args = parser.parse_args()
    rows = read_rows(ROOT / "artifacts/raw/density_scale_per_seed.csv")
    validate(rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cells = cell_summary(rows)
    maps = map_summary(rows, rng)
    pairs = paired_effects(rows, rng)
    membership = cell_membership(rows)

    if args.freeze:
        table_dir = ROOT / "artifacts/tables"
        stat_dir = ROOT / "artifacts/statistics"
        figure_dir = ROOT / "artifacts/figures"
        docs_dir: Path | None = ROOT / "docs/assets"
    else:
        base = ROOT / "artifacts/generated"
        table_dir = base / "tables"
        stat_dir = base / "statistics"
        figure_dir = base / "figures"
        docs_dir = None
    write_csv(table_dir / "table_density_scale.csv", maps)
    write_csv(table_dir / "table_density_scale_cells.csv", cells)
    write_csv(stat_dir / "density_scale_paired_effects.csv", pairs)
    write_csv(stat_dir / "density_scale_cell_membership.csv", membership)
    configure_style()
    plot_factorial(cells, figure_dir, docs_dir)
    plot_endpoints(maps, figure_dir, docs_dir)
    print("Validated 1,350 runs and reproduced density/scale evidence")


if __name__ == "__main__":
    main()
