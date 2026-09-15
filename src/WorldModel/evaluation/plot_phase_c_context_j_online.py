"""Visualize the fresh-seed Phase-C context-J online comparison.

The script reads only frozen per-arm JSON artifacts.  It does not import the
simulator or mutate any experiment result.  Figures are written as both PNG
and PDF, accompanied by seed-level CSV diagnostics and a JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np


DEFAULT_ROOT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "ctxj_online_591_600_v1"
)

ARM_DIRS = {
    "Greedy": "greedy_manifest",
    "Hungarian": "hungarian",
    "Phase C": "phasec",
    "S1": "s1",
    "Old J": "learned_j_old",
    "New J": "learned_j_new",
}
ARMS = list(ARM_DIRS)
LOADS = ["low", "mid", "high"]

COLORS = {
    "Greedy": "#7F7F7F",
    "Hungarian": "#4C78A8",
    "Phase C": "#F58518",
    "S1": "#54A24B",
    "Old J": "#B279A2",
    "New J": "#E45756",
}
MARKERS = {
    "Greedy": "o",
    "Hungarian": "s",
    "Phase C": "^",
    "S1": "D",
    "Old J": "P",
    "New J": "X",
}
LOAD_COLORS = {"low": "#59A14F", "mid": "#4C78A8", "high": "#E45756"}
LOAD_MARKERS = {"low": "o", "mid": "s", "high": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def apply_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=300)
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)


def infer_seeds(root: Path) -> list[int]:
    pattern = re.compile(r"^(?:low|mid|high)_seed(\d+)\.json$")
    seeds = {
        int(match.group(1))
        for path in (root / "per_arm" / ARM_DIRS["New J"]).glob("*.json")
        if (match := pattern.match(path.name))
    }
    result = sorted(seeds)
    if not result:
        raise RuntimeError(f"no seed artifacts found under {root}")
    return result


def load_rows(root: Path, seeds: list[int]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    reports: dict[tuple[str, str, int], dict] = {}
    for arm, arm_dir in ARM_DIRS.items():
        for load in LOADS:
            for seed in seeds:
                path = root / "per_arm" / arm_dir / f"{load}_seed{seed}.json"
                with path.open("r", encoding="utf-8") as handle:
                    report = json.load(handle)
                metrics = report["metrics"]
                arrivals = float(metrics["order_arrival_count"])
                row = {
                    "arm": arm,
                    "load": load,
                    "seed": seed,
                    "completed_orders": float(metrics["completed_orders"]),
                    "completion_fraction": float(metrics["completed_orders"]) / arrivals,
                    "completed_tasks": float(metrics["completed_tasks"]),
                    "avg_task_duration": float(metrics["avg_task_duration"]),
                    "open_order_count": float(metrics["open_order_count"]),
                    "pending_order_count": float(metrics["pending_order_count"]),
                    "deadlock_ratio_mean": float(metrics["deadlock_ratio_mean"]),
                    "deadlock_ratio_max": float(metrics["deadlock_ratio_max"]),
                    "stall_ratio_mean": float(metrics["stall_ratio_mean"]),
                    "stall_ratio_max": float(metrics["stall_ratio_max"]),
                    "dynamic_batches": float(metrics.get("dynamic_probe_batches", 0.0) or 0.0),
                    "dynamic_changed_batches": float(
                        metrics.get("dynamic_probe_order_changed_batches", 0.0) or 0.0
                    ),
                }
                rows.append(row)
                reports[(arm, load, seed)] = report
    expected = len(ARMS) * len(LOADS) * len(seeds)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(rows)}")
    return rows, reports


def make_index(rows: list[dict]) -> dict[tuple[str, str, int], dict]:
    return {(row["arm"], row["load"], row["seed"]): row for row in rows}


def values(
    index: dict[tuple[str, str, int], dict],
    arm: str,
    load: str,
    seeds: list[int],
    metric: str,
) -> np.ndarray:
    return np.asarray([index[(arm, load, seed)][metric] for seed in seeds], dtype=float)


def plot_seed_performance(
    index: dict[tuple[str, str, int], dict], seeds: list[int], output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0), sharex="col")
    specs = [
        ("completed_orders", "Completed orders", 1.0),
        ("completion_fraction", "Completion fraction (%)", 100.0),
    ]
    for row_id, (metric, ylabel, scale) in enumerate(specs):
        for col, load in enumerate(LOADS):
            ax = axes[row_id, col]
            for arm in ARMS:
                y = scale * values(index, arm, load, seeds, metric)
                ax.plot(
                    seeds,
                    y,
                    color=COLORS[arm],
                    marker=MARKERS[arm],
                    markersize=4.2 if arm != "New J" else 5.2,
                    linewidth=1.2 if arm != "New J" else 1.8,
                    alpha=0.82 if arm != "New J" else 1.0,
                    label=arm,
                )
            ax.set_title(load.capitalize())
            ax.set_ylabel(ylabel if col == 0 else "")
            ax.set_xticks(seeds)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="x", visible=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Fresh-seed online performance: every paired seed", y=1.045, fontsize=11)
    fig.subplots_adjust(top=0.87, hspace=0.32, wspace=0.23)
    save_figure(fig, output_dir, "fig01_seed_performance")


def annotated_delta_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    seeds: list[int],
    title: str,
) -> None:
    limit = max(1.0, float(np.max(np.abs(matrix))))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(seeds)), [str(seed) for seed in seeds], rotation=45)
    ax.set_yticks(np.arange(len(LOADS)), [load.capitalize() for load in LOADS])
    ax.set_title(title)
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if abs(value) > 0.62 * limit else "black"
            ax.text(j, i, f"{value:+.0f}", ha="center", va="center",
                    fontsize=6.4, color=color)
    plt.colorbar(image, ax=ax, fraction=0.035, pad=0.025)


def plot_order_delta_heatmaps(
    index: dict[tuple[str, str, int], dict], seeds: list[int], output_dir: Path
) -> None:
    baselines = ["Greedy", "Hungarian", "Phase C", "S1"]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 5.1))
    for ax, baseline in zip(axes.flat, baselines):
        matrix = np.asarray(
            [
                [
                    index[("New J", load, seed)]["completed_orders"]
                    - index[(baseline, load, seed)]["completed_orders"]
                    for seed in seeds
                ]
                for load in LOADS
            ],
            dtype=float,
        )
        annotated_delta_heatmap(
            ax, matrix, seeds, f"New J - {baseline}: completed orders"
        )
    fig.text(0.5, 0.015, "Green: New J completes more orders; red: fewer orders",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.93,
                        hspace=0.43, wspace=0.25)
    save_figure(fig, output_dir, "fig02_newj_order_delta_heatmaps")


def annotate_largest_points(
    ax: plt.Axes,
    index: dict[tuple[str, str, int], dict],
    seeds: list[int],
    load: str,
    metric: str,
    scale: float,
    count: int = 4,
) -> None:
    candidates = []
    for arm in ARMS:
        for seed in seeds:
            value = scale * index[(arm, load, seed)][metric]
            candidates.append((value, arm, seed))
    offsets = [(4, 4), (4, -10), (-30, 4), (-30, -10)]
    abbreviations = {
        "Greedy": "G",
        "Hungarian": "H",
        "Phase C": "P",
        "S1": "S",
        "Old J": "O",
        "New J": "N",
    }
    for offset, (value, arm, seed) in zip(offsets, sorted(candidates, reverse=True)[:count]):
        ax.annotate(
            f"{abbreviations[arm]}{seed}",
            (seed, value),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.7,
            color=COLORS[arm],
        )


def plot_seed_deadlock(
    index: dict[tuple[str, str, int], dict], seeds: list[int], output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0), sharex="col")
    specs = [
        ("deadlock_ratio_mean", "Mean deadlock ratio (%)"),
        ("deadlock_ratio_max", "Maximum deadlock ratio (%)"),
    ]
    for row_id, (metric, ylabel) in enumerate(specs):
        for col, load in enumerate(LOADS):
            ax = axes[row_id, col]
            for arm in ARMS:
                y = 100.0 * values(index, arm, load, seeds, metric)
                ax.plot(
                    seeds,
                    y,
                    color=COLORS[arm],
                    marker=MARKERS[arm],
                    markersize=4.0 if arm != "New J" else 5.0,
                    linewidth=1.1 if arm != "New J" else 1.7,
                    alpha=0.82 if arm != "New J" else 1.0,
                    label=arm,
                )
            annotate_largest_points(ax, index, seeds, load, metric, 100.0)
            ax.set_title(load.capitalize())
            ax.set_ylabel(ylabel if col == 0 else "")
            ax.set_xticks(seeds)
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylim(bottom=-0.02)
            ax.grid(axis="x", visible=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Congestion branches by seed (labels mark largest spikes)",
                 y=1.045, fontsize=11)
    fig.subplots_adjust(top=0.87, hspace=0.32, wspace=0.23)
    save_figure(fig, output_dir, "fig03_seed_deadlock")


def plot_deadlock_heatmaps(
    index: dict[tuple[str, str, int], dict], seeds: list[int], output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.0))
    metrics = [
        ("deadlock_ratio_mean", "Mean deadlock (%)"),
        ("deadlock_ratio_max", "Maximum deadlock (%)"),
    ]
    for row_id, (metric, row_label) in enumerate(metrics):
        all_values = np.asarray(
            [100.0 * index[(arm, load, seed)][metric]
             for arm in ARMS for load in LOADS for seed in seeds],
            dtype=float,
        )
        vmax = max(0.01, float(all_values.max()))
        for col, load in enumerate(LOADS):
            ax = axes[row_id, col]
            matrix = np.asarray(
                [[100.0 * index[(arm, load, seed)][metric] for seed in seeds]
                 for arm in ARMS],
                dtype=float,
            )
            image = ax.imshow(
                matrix,
                cmap="YlOrRd",
                norm=PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax),
                aspect="auto",
            )
            ax.set_xticks(np.arange(len(seeds)), [str(seed) for seed in seeds], rotation=45)
            ax.set_yticks(np.arange(len(ARMS)), ARMS if col == 0 else [])
            ax.set_title(f"{load.capitalize()} - {row_label}")
            ax.grid(False)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    text = "0" if value < 0.005 else f"{value:.2f}"
                    color = "white" if value > 0.42 * vmax else "black"
                    ax.text(j, i, text, ha="center", va="center", fontsize=5.5,
                            color=color)
            plt.colorbar(image, ax=ax, fraction=0.032, pad=0.02, label="%")
    fig.suptitle("Seed-level congestion heatmaps", y=0.995, fontsize=11)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.10, top=0.91,
                        hspace=0.43, wspace=0.27)
    save_figure(fig, output_dir, "fig04_deadlock_heatmaps")


def plot_paired_outcome_quadrants(
    index: dict[tuple[str, str, int], dict], seeds: list[int], output_dir: Path
) -> dict[str, dict[str, int]]:
    baselines = ["Greedy", "Hungarian", "Phase C", "S1"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.9))
    counts: dict[str, dict[str, int]] = {}
    for ax, baseline in zip(axes.flat, baselines):
        quadrant_counts = defaultdict(int)
        candidates = []
        for load in LOADS:
            xs = []
            ys = []
            for seed in seeds:
                new = index[("New J", load, seed)]
                base = index[(baseline, load, seed)]
                x = new["completed_orders"] - base["completed_orders"]
                # Positive y means lower deadlock for New J.
                y = 100.0 * (base["deadlock_ratio_mean"] - new["deadlock_ratio_mean"])
                xs.append(x)
                ys.append(y)
                if x > 0 and y > 0:
                    quadrant_counts["better_both"] += 1
                elif x < 0 and y < 0:
                    quadrant_counts["worse_both"] += 1
                else:
                    quadrant_counts["mixed_or_tied"] += 1
                candidates.append((abs(x) + 10.0 * abs(y), load, seed, x, y))
            ax.scatter(
                xs,
                ys,
                s=38,
                color=LOAD_COLORS[load],
                marker=LOAD_MARKERS[load],
                edgecolor="white",
                linewidth=0.5,
                label=load.capitalize(),
            )
        ax.axvline(0.0, color="#555555", linewidth=0.8)
        ax.axhline(0.0, color="#555555", linewidth=0.8)
        ax.set_xlabel("Completed orders: New J - baseline")
        ax.set_ylabel("Mean deadlock reduction (percentage points)")
        ax.set_title(
            f"vs {baseline}: better both={quadrant_counts['better_both']}, "
            f"worse both={quadrant_counts['worse_both']}"
        )
        offsets = [(4, 4), (4, -10), (-28, 4), (-28, -10)]
        for offset, (_, load, seed, x, y) in zip(
            offsets, sorted(candidates, reverse=True)[:4]
        ):
            ax.annotate(f"{load[0].upper()}{seed}", (x, y), xytext=offset,
                        textcoords="offset points", fontsize=6.6)
        ax.text(0.98, 0.96, "desired", transform=ax.transAxes, ha="right",
                va="top", fontsize=7.5, color="#2E7D32")
        counts[baseline] = dict(quadrant_counts)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Paired throughput-congestion outcomes for every seed",
                 y=1.045, fontsize=11)
    fig.subplots_adjust(top=0.89, hspace=0.38, wspace=0.28)
    save_figure(fig, output_dir, "fig05_paired_outcome_quadrants")
    return counts


def plot_order_task_tradeoff(
    index: dict[tuple[str, str, int], dict], seeds: list[int], output_dir: Path
) -> None:
    baselines = ["Greedy", "Phase C", "S1"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.35))
    for ax, baseline in zip(axes, baselines):
        candidates = []
        for load in LOADS:
            xs = []
            ys = []
            for seed in seeds:
                new = index[("New J", load, seed)]
                base = index[(baseline, load, seed)]
                x = new["completed_orders"] - base["completed_orders"]
                y = new["completed_tasks"] - base["completed_tasks"]
                xs.append(x)
                ys.append(y)
                candidates.append((abs(x) + 0.05 * abs(y), load, seed, x, y))
            ax.scatter(xs, ys, s=36, color=LOAD_COLORS[load],
                       marker=LOAD_MARKERS[load], edgecolor="white", linewidth=0.5,
                       label=load.capitalize())
        ax.axvline(0.0, color="#555555", linewidth=0.8)
        ax.axhline(0.0, color="#555555", linewidth=0.8)
        ax.set_title(f"New J - {baseline}")
        ax.set_xlabel("Delta completed orders")
        ax.set_ylabel("Delta completed tasks" if ax is axes[0] else "")
        offsets = [(4, 4), (4, -10), (-28, 4)]
        for offset, (_, load, seed, x, y) in zip(
            offsets, sorted(candidates, reverse=True)[:3]
        ):
            ax.annotate(f"{load[0].upper()}{seed}", (x, y), xytext=offset,
                        textcoords="offset points", fontsize=6.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Completed-order gains can hide lower completed work volume",
                 y=1.07, fontsize=11)
    fig.subplots_adjust(top=0.82, wspace=0.25)
    save_figure(fig, output_dir, "fig06_order_task_tradeoff")


def trace_agreement(
    reports: dict[tuple[str, str, int], dict], seeds: list[int]
) -> list[dict]:
    rows: list[dict] = []
    for load in LOADS:
        for seed in seeds:
            old_trace = {
                int(record["tick"]): record
                for record in reports[("Old J", load, seed)].get("dynamic_trace", [])
            }
            new_trace = {
                int(record["tick"]): record
                for record in reports[("New J", load, seed)].get("dynamic_trace", [])
            }
            common_ticks = sorted(set(old_trace) & set(new_trace))
            same_frames = []
            for tick in common_ticks:
                old = old_trace[tick]
                new = new_trace[tick]
                if old.get("original_order") == new.get("original_order"):
                    same_frames.append((old, new))
            first_match = sum(
                bool(old.get("dynamic_order"))
                and bool(new.get("dynamic_order"))
                and old["dynamic_order"][0] == new["dynamic_order"][0]
                for old, new in same_frames
            )
            full_match = sum(
                old.get("dynamic_order") == new.get("dynamic_order")
                for old, new in same_frames
            )
            denominator = len(same_frames)
            rows.append(
                {
                    "load": load,
                    "seed": seed,
                    "common_trace_ticks": len(common_ticks),
                    "same_context_frames": denominator,
                    "first_choice_match_rate": first_match / denominator if denominator else np.nan,
                    "full_sequence_match_rate": full_match / denominator if denominator else np.nan,
                }
            )
    return rows


def plot_trace_agreement(
    agreement_rows: list[dict], seeds: list[int], output_dir: Path
) -> None:
    lookup = {(row["load"], row["seed"]): row for row in agreement_rows}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.65))
    specs = [
        ("first_choice_match_rate", "Old/New J first-context agreement (%)"),
        ("full_sequence_match_rate", "Old/New J full-sequence agreement (%)"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        matrix = np.asarray(
            [[100.0 * lookup[(load, seed)][metric] for seed in seeds] for load in LOADS],
            dtype=float,
        )
        image = ax.imshow(matrix, cmap="YlGn", vmin=90.0, vmax=100.0, aspect="auto")
        ax.set_xticks(np.arange(len(seeds)), [str(seed) for seed in seeds], rotation=45)
        ax.set_yticks(np.arange(len(LOADS)), [load.capitalize() for load in LOADS])
        ax.set_title(title)
        ax.grid(False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                        fontsize=6.2, color="black")
        plt.colorbar(image, ax=ax, fraction=0.035, pad=0.025, label="%")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.86, wspace=0.25)
    save_figure(fig, output_dir, "fig07_old_new_online_agreement")


def plot_congestion_components(
    index: dict[tuple[str, str, int], dict], seeds: list[int], output_dir: Path
) -> None:
    """Separate spatial congestion from terminal service backlog."""
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 8.1), sharex="col")
    specs = [
        ("stall_ratio_mean", "Mean stall ratio (%)", 100.0),
        ("open_order_count", "Open orders at tick 1500", 1.0),
        ("pending_order_count", "Pending orders at tick 1500", 1.0),
    ]
    for row_id, (metric, ylabel, scale) in enumerate(specs):
        for col, load in enumerate(LOADS):
            ax = axes[row_id, col]
            for arm in ARMS:
                y = scale * values(index, arm, load, seeds, metric)
                ax.plot(
                    seeds,
                    y,
                    color=COLORS[arm],
                    marker=MARKERS[arm],
                    markersize=3.8 if arm != "New J" else 4.8,
                    linewidth=1.05 if arm != "New J" else 1.6,
                    alpha=0.82 if arm != "New J" else 1.0,
                    label=arm,
                )
            ax.set_title(load.capitalize())
            ax.set_ylabel(ylabel if col == 0 else "")
            ax.set_xticks(seeds)
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylim(bottom=-0.02)
            ax.grid(axis="x", visible=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(
        "Congestion components: spatial stall versus terminal order backlog",
        y=1.035,
        fontsize=11,
    )
    fig.subplots_adjust(top=0.90, hspace=0.34, wspace=0.23)
    save_figure(fig, output_dir, "fig08_congestion_components_by_seed")


def write_figure_notes(output_dir: Path) -> None:
    notes = """# Context-J fresh-seed online figures

All comparisons use the same order manifest within each `(load, seed)` pair.

- `fig01_seed_performance`: raw completed orders and completion fraction for every seed.
- `fig02_newj_order_delta_heatmaps`: exact per-seed completed-order delta. Green means New J is better.
- `fig03_seed_deadlock`: mean and maximum deadlock ratio. Labels use `L/M/H` for load, followed by seed.
- `fig04_deadlock_heatmaps`: the same deadlock data as annotated heatmaps. Mean is time-averaged; maximum is the worst tick.
- `fig05_paired_outcome_quadrants`: x is New-J completed-order gain; y is baseline deadlock minus New-J deadlock. The upper-right quadrant is desirable.
- `fig06_order_task_tradeoff`: distinguishes completed orders from completed task-chain volume.
- `fig07_old_new_online_agreement`: compares Old-J and New-J only when they saw the same context frame.
- `fig08_congestion_components_by_seed`: separates spatial stall from terminal open/pending backlog. Hungarian mid/high primarily fails through backlog, not deadlock.

The CSV files contain the plotted seed-level values. Ratios in the JSON/CSV remain in `[0, 1]`; figures display deadlock and stall ratios as percentages.
"""
    (output_dir / "README.md").write_text(notes, encoding="utf-8")


def write_diagnostics(
    root: Path,
    index: dict[tuple[str, str, int], dict],
    seeds: list[int],
    agreement_rows: list[dict],
    quadrant_counts: dict[str, dict[str, int]],
    output_dir: Path,
) -> None:
    baseline_names = ["Greedy", "Hungarian", "Phase C", "S1", "Old J"]
    csv_path = output_dir / "seed_level_diagnostics.csv"
    fields = [
        "load", "seed", "arm", "completed_orders", "completion_fraction",
        "completed_tasks", "deadlock_ratio_mean", "deadlock_ratio_max",
        "open_order_count", "pending_order_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for load in LOADS:
            for seed in seeds:
                for arm in ARMS:
                    row = index[(arm, load, seed)]
                    writer.writerow({field: row[field] for field in fields})

    delta_path = output_dir / "newj_paired_deltas.csv"
    delta_fields = [
        "load", "seed", "baseline", "delta_completed_orders",
        "delta_completion_fraction", "delta_completed_tasks",
        "deadlock_mean_reduction", "deadlock_max_reduction",
    ]
    with delta_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=delta_fields)
        writer.writeheader()
        for load in LOADS:
            for seed in seeds:
                new = index[("New J", load, seed)]
                for baseline in baseline_names:
                    base = index[(baseline, load, seed)]
                    writer.writerow(
                        {
                            "load": load,
                            "seed": seed,
                            "baseline": baseline,
                            "delta_completed_orders": new["completed_orders"] - base["completed_orders"],
                            "delta_completion_fraction": new["completion_fraction"] - base["completion_fraction"],
                            "delta_completed_tasks": new["completed_tasks"] - base["completed_tasks"],
                            "deadlock_mean_reduction": base["deadlock_ratio_mean"] - new["deadlock_ratio_mean"],
                            "deadlock_max_reduction": base["deadlock_ratio_max"] - new["deadlock_ratio_max"],
                        }
                    )

    agreement_path = output_dir / "old_new_trace_agreement.csv"
    with agreement_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(agreement_rows[0]))
        writer.writeheader()
        writer.writerows(agreement_rows)

    aggregate = {}
    for arm in ARMS:
        aggregate[arm] = {}
        for load in LOADS:
            aggregate[arm][load] = {
                metric: float(values(index, arm, load, seeds, metric).mean())
                for metric in [
                    "completed_orders", "completion_fraction", "completed_tasks",
                    "deadlock_ratio_mean", "deadlock_ratio_max", "stall_ratio_mean",
                    "open_order_count", "pending_order_count",
                ]
            }

    same_frames = sum(row["same_context_frames"] for row in agreement_rows)
    first_matches = sum(
        row["first_choice_match_rate"] * row["same_context_frames"]
        for row in agreement_rows
    )
    full_matches = sum(
        row["full_sequence_match_rate"] * row["same_context_frames"]
        for row in agreement_rows
    )
    summary = {
        "schema_version": "phase_c_context_j_online_figure_summary_v1",
        "source_root": str(root),
        "seeds": seeds,
        "aggregate": aggregate,
        "paired_outcome_quadrants": quadrant_counts,
        "old_new_trace_agreement": {
            "same_context_frames": same_frames,
            "first_choice_match_rate": first_matches / same_frames,
            "full_sequence_match_rate": full_matches / same_frames,
        },
        "sign_convention": {
            "newj_paired_deltas": "positive order/task delta means New J is higher",
            "deadlock_reduction": "positive means New J has lower deadlock",
        },
    }
    with (output_dir / "figure_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or (root / "figures")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    seeds = infer_seeds(root)
    rows, reports = load_rows(root, seeds)
    index = make_index(rows)

    plot_seed_performance(index, seeds, output_dir)
    plot_order_delta_heatmaps(index, seeds, output_dir)
    plot_seed_deadlock(index, seeds, output_dir)
    plot_deadlock_heatmaps(index, seeds, output_dir)
    quadrant_counts = plot_paired_outcome_quadrants(index, seeds, output_dir)
    plot_order_task_tradeoff(index, seeds, output_dir)
    agreement_rows = trace_agreement(reports, seeds)
    plot_trace_agreement(agreement_rows, seeds, output_dir)
    plot_congestion_components(index, seeds, output_dir)
    write_diagnostics(
        root,
        index,
        seeds,
        agreement_rows,
        quadrant_counts,
        output_dir,
    )
    write_figure_notes(output_dir)
    print(f"wrote context-J online figures to {output_dir}")


if __name__ == "__main__":
    main()
