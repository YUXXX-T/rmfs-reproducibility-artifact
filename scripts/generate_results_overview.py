#!/usr/bin/env python3
"""Generate static, GitHub-renderable result summaries from canonical CSVs."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOADS = ("low", "mid", "high")
LOAD_LABELS = {"low": "Low", "mid": "Mid", "high": "High"}
ARM_LABELS = {
    "Greedy": "Greedy",
    "greedy": "Greedy",
    "Hungarian": "Hungarian",
    "hungarian": "Hungarian",
    "JSQ": "JSQ",
    "PhaseC": "WM-Base",
    "phasec": "WM-Base",
    "ComboS1J1": "ComboS1J1",
    "combo_s1_j1": "ComboS1J1",
}
COLORS = {
    "Greedy": "#4C78A8",
    "Hungarian": "#F58518",
    "JSQ": "#54A24B",
    "WM-Base": "#9C755F",
    "ComboS1J1": "#E45756",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def aggregate_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["load"], ARM_LABELS[row["arm"]]): row
        for row in rows
    }


def grouped_bars(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    arms: tuple[str, ...],
    metric: str,
    title: str,
    ylabel: str,
    decimals: int,
) -> None:
    indexed = aggregate_index(rows)
    centers = np.arange(len(LOADS), dtype=float)
    total_width = 0.82
    width = total_width / len(arms)
    offsets = (np.arange(len(arms)) - (len(arms) - 1) / 2) * width

    for offset, arm in zip(offsets, arms):
        means = np.asarray(
            [float(indexed[(load, arm)][f"{metric}_mean"]) for load in LOADS]
        )
        stds = np.asarray(
            [float(indexed[(load, arm)][f"{metric}_std"]) for load in LOADS]
        )
        bars = ax.bar(
            centers + offset,
            means,
            width=width * 0.91,
            color=COLORS[arm],
            label=arm,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        ax.errorbar(
            centers + offset,
            means,
            yerr=stds,
            fmt="none",
            ecolor="#32373D",
            elinewidth=0.75,
            capsize=2,
            capthick=0.75,
            alpha=0.68,
            zorder=3,
        )
        for bar, value in zip(bars, means):
            label = f"{value:.{decimals}f}"
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.7,
                rotation=90,
                color="#30343B",
            )

    ax.set_title(title, loc="left", pad=9)
    ax.set_ylabel(ylabel)
    ax.set_xticks(centers, [LOAD_LABELS[load] for load in LOADS])
    ax.set_xlabel("Offered-load regime")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E7E9EC", linewidth=0.8)
    ax.set_ylim(bottom=0)


def save_figure(
    fig: plt.Figure,
    stem: str,
    *,
    figure_dir: Path,
    doc_asset_dir: Path | None,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{stem}.png"
    pdf = figure_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Title": stem.replace("_", " ").title(),
            "Author": "Anonymous Authors",
            "Creator": "RMFS reproducibility artifact",
        },
    )
    if doc_asset_dir is not None:
        doc_asset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(png, doc_asset_dir / png.name)
    print(f"Wrote {png.relative_to(ROOT)} and {pdf.relative_to(ROOT)}")


def generate_aggregate_overview(
    figure_dir: Path, doc_asset_dir: Path | None
) -> None:
    main = read_csv(ROOT / "artifacts/tables/table_main.csv")
    station6 = read_csv(ROOT / "artifacts/tables/table_station6.csv")
    main_arms = ("Greedy", "Hungarian", "JSQ", "WM-Base", "ComboS1J1")
    station6_arms = ("Greedy", "Hungarian", "WM-Base", "ComboS1J1")

    fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.2), constrained_layout=True)
    grouped_bars(
        axes[0, 0], main, arms=main_arms, metric="completed_orders",
        title="(a) Four-station evaluation (50 paired seeds)",
        ylabel="Completed orders after 1,500 ticks", decimals=1,
    )
    grouped_bars(
        axes[0, 1], main, arms=main_arms, metric="deadlock_ratio_mean",
        title="(b) Four-station deadlock exposure",
        ylabel="Mean deadlock ratio", decimals=2,
    )
    grouped_bars(
        axes[1, 0], station6, arms=station6_arms, metric="completed_orders",
        title="(c) Six-station adaptation (10 held-out seeds)",
        ylabel="Completed orders after 1,500 ticks", decimals=1,
    )
    grouped_bars(
        axes[1, 1], station6, arms=station6_arms, metric="deadlock_ratio_mean",
        title="(d) Six-station deadlock exposure",
        ylabel="Mean deadlock ratio", decimals=2,
    )
    axes[0, 0].legend(ncol=3, loc="upper left", fontsize=8.5)
    axes[0, 1].legend(ncol=3, loc="upper left", fontsize=8.5)
    axes[1, 0].legend(ncol=2, loc="upper left", fontsize=8.5)
    axes[1, 1].legend(ncol=2, loc="upper left", fontsize=8.5)
    fig.suptitle(
        "Aggregate policy outcomes across load and station-layout settings",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "Bars show seed means; whiskers show between-seed SD. Numeric labels are means.",
        ha="center",
        fontsize=9,
        color="#555B64",
    )
    save_figure(
        fig,
        "results_overview",
        figure_dir=figure_dir,
        doc_asset_dir=doc_asset_dir,
    )
    plt.close(fig)


def select_paired(
    rows: list[dict[str, str]], metric: str
) -> list[dict[str, str]]:
    selected = [row for row in rows if row["metric"] == metric]
    baseline_rank = {"JSQ": 0, "Greedy": 0, "WM-Base": 1}
    load_rank = {load: rank for rank, load in enumerate(LOADS)}
    return sorted(
        selected,
        key=lambda row: (
            load_rank[row["load"]],
            baseline_rank[row["baseline_paper_name"]],
        ),
    )


def forest_panel(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    metric: str,
    title: str,
    xlabel: str,
    favorable: str,
) -> None:
    selected = select_paired(rows, metric)
    y = np.arange(len(selected))[::-1]
    means = np.asarray([float(row["arm_minus_baseline_mean"]) for row in selected])
    lows = np.asarray([float(row["ci95_low"]) for row in selected])
    highs = np.asarray([float(row["ci95_high"]) for row in selected])
    labels = [
        f"{LOAD_LABELS[row['load']]} vs {row['baseline_paper_name']}"
        for row in selected
    ]
    colors = [COLORS["JSQ"] if row["baseline_paper_name"] == "JSQ" else
              COLORS["Greedy"] if row["baseline_paper_name"] == "Greedy" else
              COLORS["WM-Base"] for row in selected]

    extent = max(abs(float(lows.min())), abs(float(highs.max())), 1e-9)
    if favorable == "right":
        ax.axvspan(0, extent * 1.28, color="#E8F4EC", zorder=0)
    else:
        ax.axvspan(-extent * 1.28, 0, color="#E8F4EC", zorder=0)
    ax.axvline(0, color="#4A4F56", linewidth=1.0, linestyle="--", zorder=1)
    for position, mean, low, high, color in zip(y, means, lows, highs, colors):
        ax.errorbar(
            mean,
            position,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            markersize=5.5,
            color=color,
            ecolor=color,
            elinewidth=1.8,
            capsize=3,
            zorder=3,
        )
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.7, len(selected) - 0.3)
    ax.set_xlim(-extent * 1.14, extent * 1.14)
    ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, color="#E7E9EC", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(
        0.98 if favorable == "right" else 0.02,
        0.04,
        "favors ComboS1J1",
        transform=ax.transAxes,
        ha="right" if favorable == "right" else "left",
        va="bottom",
        fontsize=8,
        color="#2F6B43",
    )


def generate_paired_overview(
    figure_dir: Path, doc_asset_dir: Path | None
) -> None:
    main = read_csv(ROOT / "artifacts/statistics/paired_confidence_intervals.csv")
    station6 = read_csv(
        ROOT / "artifacts/statistics/station6_paired_confidence_intervals.csv"
    )
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.0), constrained_layout=True)
    forest_panel(
        axes[0, 0], main, metric="completed_orders",
        title="(a) Four stations: throughput effect",
        xlabel="Completed orders: ComboS1J1 minus baseline",
        favorable="right",
    )
    forest_panel(
        axes[0, 1], main, metric="deadlock_ratio_mean",
        title="(b) Four stations: deadlock effect",
        xlabel="Mean deadlock ratio: ComboS1J1 minus baseline",
        favorable="left",
    )
    forest_panel(
        axes[1, 0], station6, metric="completed_orders",
        title="(c) Six stations: throughput effect",
        xlabel="Completed orders: ComboS1J1 minus baseline",
        favorable="right",
    )
    forest_panel(
        axes[1, 1], station6, metric="deadlock_ratio_mean",
        title="(d) Six stations: deadlock effect",
        xlabel="Mean deadlock ratio: ComboS1J1 minus baseline",
        favorable="left",
    )
    fig.suptitle(
        "Paired policy effects across the same arrival seeds",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "Points are paired mean differences; whiskers are percentile bootstrap 95% CIs. "
        "Intervals crossing zero are inconclusive.",
        ha="center",
        fontsize=9,
        color="#555B64",
    )
    save_figure(
        fig,
        "paired_effects_overview",
        figure_dir=figure_dir,
        doc_asset_dir=doc_asset_dir,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="refresh committed artifacts and their MkDocs copies",
    )
    args = parser.parse_args()
    if args.freeze:
        figure_dir = ROOT / "artifacts/figures"
        doc_asset_dir: Path | None = ROOT / "docs/assets"
    else:
        figure_dir = ROOT / "artifacts/generated/figures"
        doc_asset_dir = None
    configure_style()
    generate_aggregate_overview(figure_dir, doc_asset_dir)
    generate_paired_overview(figure_dir, doc_asset_dir)


if __name__ == "__main__":
    main()
