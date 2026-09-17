#!/usr/bin/env python3
"""Regenerate the three-load six-station runtime audit figures.

The committed inputs are anonymous CSV extracts from the dedicated runtime
benchmark. They retain every plotted value but omit the absolute paths,
host name, and static fingerprints present in the collection JSON.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/raw/figure_inputs"
DEFAULT_OUTPUT = ROOT / "artifacts/generated/station6_runtime_crossload"
LOADS = ("low", "mid", "high")
LABELS = ("greedy", "jsq", "hungarian", "proposed_cpu", "proposed_cuda_0")
NAMES = (
    "Greedy (CPU)",
    "JSQ (CPU)",
    "Hungarian (CPU)",
    "Proposed (CPU)",
    "Proposed (GPU)",
)
COLORS = ("#797f88", "#a88863", "#67948b", "#42678a", "#c45c42")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_inputs() -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    summary = read_csv(INPUT / "station6_runtime_chart_summary.csv")
    if len(summary) != 15:
        raise AssertionError(f"expected 15 summary rows, found {len(summary)}")
    if {(row["load"], row["label"]) for row in summary} != {
        (load, label) for load in LOADS for label in LABELS
    }:
        raise AssertionError("runtime summary does not contain the 3x5 design")
    for row in summary:
        if int(row["runs"]) != 10 or int(row["assignment_calls"]) != 15_000:
            raise AssertionError("unexpected run/call count in runtime summary")
        if row["audit_passed"].lower() != "true":
            raise AssertionError("runtime summary contains a failed audit")

    runs: dict[str, list[dict[str, str]]] = {}
    for load in LOADS:
        rows = read_csv(INPUT / f"{load}_all_seeds_cpu_gpu_runs.csv")
        if len(rows) != 50:
            raise AssertionError(f"expected 50 {load} rows, found {len(rows)}")
        expected = {
            (label, seed) for label in LABELS for seed in range(721, 731)
        }
        observed = {(row["label"], int(row["seed"])) for row in rows}
        if observed != expected:
            raise AssertionError(f"incomplete {load} method/seed matrix")
        for row in rows:
            if row["load"] != load or int(row["ticks"]) != 1500:
                raise AssertionError(f"unexpected load/tick contract in {load}")
            if int(row["assignment_calls"]) != 1500:
                raise AssertionError(f"unexpected assignment count in {load}")
            if row["audit_passed"].lower() != "true":
                raise AssertionError(f"failed run audit in {load}")
        runs[load] = rows

    # Validate the summary against the per-run values where aggregation is
    # possible without the 1,500-call latency samples.
    by_summary = {(row["load"], row["label"]): row for row in summary}
    for load, rows in runs.items():
        for label in LABELS:
            group = [row for row in rows if row["label"] == label]
            frozen = by_summary[(load, label)]
            checks = {
                "wall_mean_s": statistics.fmean(float(row["wall_time_s"]) for row in group),
                "wall_median_s": statistics.median(float(row["wall_time_s"]) for row in group),
                "wall_min_s": min(float(row["wall_time_s"]) for row in group),
                "wall_max_s": max(float(row["wall_time_s"]) for row in group),
                "completed_orders_mean": statistics.fmean(
                    float(row["completed_orders"]) for row in group
                ),
            }
            for field, actual in checks.items():
                if abs(actual - float(frozen[field])) > 1e-9:
                    raise AssertionError(f"summary mismatch: {load}/{label}/{field}")
    return summary, runs


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def assignment_plot(summary: list[dict[str, str]], output: Path) -> None:
    by = {(row["load"], row["label"]): row for row in summary}
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.5), sharey=True)
    for axis, load in zip(axes, LOADS):
        for index, (label, color) in enumerate(zip(LABELS, COLORS)):
            row = by[(load, label)]
            mean = float(row["assignment_mean_ms"])
            p95 = float(row["assignment_p95_ms"])
            axis.plot(
                [mean, p95], [index, index], color=color, linewidth=2.1,
                alpha=0.65, zorder=2,
            )
            axis.scatter(
                [mean], [index], marker="o", s=80, color=color,
                edgecolors="white", linewidths=0.7, zorder=4,
            )
            axis.scatter(
                [p95], [index], marker="D", s=21, color="#273038", zorder=5,
            )
        axis.set_title(f"{load.capitalize()} load")
        axis.set_xscale("log")
        axis.set_xlim(0.2, 1000)
        axis.set_xticks([0.2, 1, 10, 100, 1000])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.grid(axis="x", which="major", color="#e5e9ed", linewidth=0.8)
        axis.tick_params(axis="both", length=0)
        axis.invert_yaxis()
        axis.set_yticks(range(len(NAMES)), NAMES)
    axes[0].tick_params(axis="y", labelleft=True)
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)
    fig.suptitle(
        "Online dispatch runtime | 20×20 map, 48 robots, 6 stations",
        fontsize=12,
        y=0.99,
    )
    fig.supxlabel("Assignment latency (ms/tick; log scale)", y=0.125, fontsize=10)
    fig.legend(
        handles=[
            Line2D(
                [0], [0], color="#797f88", marker="o", markersize=8,
                linewidth=0, label="Circle: mean over 15,000 calls",
            ),
            Line2D(
                [0], [0], color="#273038", marker="D", markersize=5,
                linewidth=0, label="Diamond: P95 over 15,000 calls",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.text(
        0.5,
        0.005,
        "CPU: Intel Xeon Gold 6240C; GPU: NVIDIA RTX 4090 (cuda:0); "
        "4 PyTorch threads; 10 seeds/load; CUDA synchronized; init excluded",
        ha="center",
        va="center",
        fontsize=7.1,
        color="#555e68",
    )
    fig.subplots_adjust(left=0.205, right=0.988, top=0.87, bottom=0.27, wspace=0.17)
    save(fig, output, "station6_runtime_assignment")


def completed_order_mismatches(
    runs: dict[str, list[dict[str, str]]],
) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for load, rows in runs.items():
        values = {
            (row["label"], int(row["seed"])): int(float(row["completed_orders"]))
            for row in rows
        }
        result[load] = {
            seed
            for seed in range(721, 731)
            if values[("proposed_cpu", seed)] != values[("proposed_cuda_0", seed)]
        }
    return result


def wall_plot(runs: dict[str, list[dict[str, str]]], output: Path) -> None:
    mismatches = completed_order_mismatches(runs)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.6), sharey=True)
    for axis, load in zip(axes, LOADS):
        for index, (label, color) in enumerate(zip(LABELS, COLORS)):
            group = sorted(
                (row for row in runs[load] if row["label"] == label),
                key=lambda row: int(row["seed"]),
            )
            values = [float(row["wall_time_s"]) for row in group]
            axis.plot(
                [index - 0.31, index + 0.31],
                [statistics.median(values)] * 2,
                color=color,
                linewidth=2.6,
                zorder=4,
            )
            for point, row in enumerate(group):
                x = index + (point - 4.5) * 0.064
                mismatch = (
                    label.startswith("proposed")
                    and int(row["seed"]) in mismatches[load]
                )
                axis.scatter(
                    x,
                    float(row["wall_time_s"]),
                    s=25,
                    facecolors="none" if mismatch else color,
                    edgecolors="#b53135" if mismatch else "white",
                    linewidths=1.1 if mismatch else 0.45,
                    zorder=5,
                    marker="o",
                )
        axis.set_title(f"{load.capitalize()} load")
        axis.set_xticks(range(5), ("G", "J", "H", "P-C", "P-G"))
        axis.tick_params(axis="both", length=0)
        axis.set_yscale("log")
        axis.grid(axis="y", which="major", color="#e5e9ed", linewidth=0.8)
        axis.set_ylim(30, 2000)
        axis.set_xlabel("Method / execution device")
    axes[0].set_ylabel("Full simulation wall time (s; log scale)")
    fig.suptitle(
        "End-to-end runtime varies with the simulated trajectory",
        fontsize=12,
        y=0.99,
    )
    fig.text(
        0.5,
        0.10,
        "G=Greedy, J=JSQ, H=Hungarian (CPU only); "
        "P-C=Proposed CPU; P-G=Proposed GPU",
        ha="center",
        fontsize=8,
    )
    fig.text(
        0.5,
        0.055,
        "Each dot = one held-out seed; colored bar = median; red open dots = "
        "CPU/GPU differ in completed orders",
        ha="center",
        fontsize=7.7,
        color="#555e68",
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.86, bottom=0.26, wspace=0.12)
    save(fig, output, "station6_runtime_wall")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary, runs = load_inputs()
    set_style()
    assignment_plot(summary, args.output_dir.resolve())
    wall_plot(runs, args.output_dir.resolve())
    mismatches = completed_order_mismatches(runs)
    print(f"wrote cross-load runtime figures to {args.output_dir.resolve()}")
    print("Proposed CPU/GPU completed-order mismatches:", mismatches)


if __name__ == "__main__":
    main()
