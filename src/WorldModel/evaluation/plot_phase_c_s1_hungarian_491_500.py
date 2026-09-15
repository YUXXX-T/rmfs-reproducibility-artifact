"""Plot the frozen Phase-C/S1/Hungarian certification results.

The script consumes only the validated per-arm JSON artifacts.  It produces
publication-ready PDF/PNG figures plus small machine-readable summaries used
to audit the plotted values.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np


DEFAULT_ROOT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "phasec_s1_hungarian_cert_491_500_v1"
)

ARM_DIRS = {
    "Greedy": "greedy",
    "Hungarian": "hungarian",
    "Phase C": "phasec",
    "Phase C + S1": "phasec_s1",
}
ARMS = list(ARM_DIRS)
LOADS = ["low", "mid", "high"]
SEEDS = list(range(491, 501))

COLORS = {
    "Greedy": "#7F7F7F",
    "Hungarian": "#4C78A8",
    "Phase C": "#F58518",
    "Phase C + S1": "#54A24B",
}
LOAD_COLORS = {"low": "#59A14F", "mid": "#4C78A8", "high": "#E45756"}
LOAD_MARKERS = {"low": "o", "mid": "s", "high": "^"}


def apply_icra_style() -> None:
    """Apply the repository's lightweight publication figure style locally.

    This script deliberately avoids importing ``WorldModel`` because that
    package imports PyTorch at module initialization, while plotting the JSON
    artifacts should remain usable in a CPU-only analysis environment.
    """
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
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig: plt.Figure, path_stem: str, dpi_png: int = 300) -> None:
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png", dpi=dpi_png)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def infer_seeds(root: Path) -> list[int]:
    validation_path = root / "phase_c_s1_hungarian_validation.json"
    with validation_path.open("r", encoding="utf-8") as handle:
        validation = json.load(handle)
    seeds = (validation.get("scope") or {}).get("seeds")
    if not seeds:
        raise ValueError(f"validation report does not declare seeds: {validation_path}")
    result = sorted(int(seed) for seed in seeds)
    if len(result) < 2 or len(result) != len(set(result)):
        raise ValueError(f"invalid seed block in {validation_path}: {result}")
    return result


def load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for arm, arm_dir in ARM_DIRS.items():
        for load in LOADS:
            for seed in SEEDS:
                path = root / "per_arm" / arm_dir / f"{load}_seed{seed}.json"
                with path.open("r", encoding="utf-8") as handle:
                    report = json.load(handle)
                metrics = report["metrics"]
                arrivals = float(metrics["order_arrival_count"])
                rows.append(
                    {
                        "arm": arm,
                        "load": load,
                        "seed": seed,
                        "completed_orders": float(metrics["completed_orders"]),
                        "completion_fraction": float(metrics["completed_orders"]) / arrivals,
                        "deadlock_ratio_mean": float(metrics["deadlock_ratio_mean"]),
                        "deadlock_ratio_max": float(metrics["deadlock_ratio_max"]),
                        "stall_ratio_max": float(metrics["stall_ratio_max"]),
                        "severe_events_per_100": float(metrics["severe_events"]) / 15.0,
                        "congestion_events_per_100": float(metrics["congestion_events"]) / 15.0,
                        "avg_excess_delay": float(metrics["avg_excess_delay"]),
                        "open_order_count": float(metrics["open_order_count"]),
                        "pending_order_count": float(metrics["pending_order_count"]),
                        "modified_decisions": float(
                            metrics.get("energy_conv_modified_decisions", 0.0) or 0.0
                        ),
                        "modified_decision_rate": float(
                            metrics.get("energy_conv_modified_decision_rate", 0.0) or 0.0
                        ),
                        "active_contexts": float(
                            metrics.get("energy_conv_active_contexts", 0.0) or 0.0
                        ),
                        "contexts": float(metrics.get("energy_conv_contexts", 0.0) or 0.0),
                    }
                )
    expected = len(ARMS) * len(LOADS) * len(SEEDS)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(rows)}")
    return rows


def row_index(rows: list[dict]) -> dict[tuple[str, str, int], dict]:
    return {(row["arm"], row["load"], row["seed"]): row for row in rows}


def select(rows: list[dict], arm: str, load: str, metric: str) -> np.ndarray:
    return np.asarray(
        [
            row[metric]
            for row in rows
            if row["arm"] == arm and row["load"] == load
        ],
        dtype=float,
    )


def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    return mean, float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    save_fig(fig, str(output_dir / name))


def plot_aggregate(rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    metrics = [
        ("completion_fraction", "Completion fraction", (0.0, 1.03)),
        ("deadlock_ratio_mean", "Mean deadlock ratio", (0.0, 0.72)),
    ]
    x = np.arange(len(LOADS), dtype=float)
    offsets = np.linspace(-0.18, 0.18, len(ARMS))

    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        for offset, arm in zip(offsets, ARMS):
            means = []
            errors = []
            for load in LOADS:
                mean, ci = mean_ci95(select(rows, arm, load, metric))
                means.append(mean)
                errors.append(ci)
            ax.errorbar(
                x + offset,
                means,
                yerr=errors,
                color=COLORS[arm],
                marker="o",
                markersize=4.2,
                linewidth=1.5,
                capsize=2.5,
                label=arm,
            )
        ax.set_xticks(x, [load.capitalize() for load in LOADS])
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="x", visible=False)

    axes[0].set_title("Order completion ratio")
    axes[1].set_title("Congestion")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(top=0.78, wspace=0.32)
    save(fig, output_dir, "fig01_aggregate_by_load")


def plot_paired_s1_hungarian(rows: list[dict], output_dir: Path) -> None:
    idx = row_index(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05))
    specs = [
        ("completion_fraction", "Hungarian completion", "Phase C + S1 completion"),
        ("deadlock_ratio_mean", "Hungarian deadlock", "Phase C + S1 deadlock"),
    ]
    for ax, (metric, xlabel, ylabel) in zip(axes, specs):
        for load in LOADS:
            xs = [idx[("Hungarian", load, seed)][metric] for seed in SEEDS]
            ys = [idx[("Phase C + S1", load, seed)][metric] for seed in SEEDS]
            ax.scatter(
                xs,
                ys,
                s=30,
                color=LOAD_COLORS[load],
                marker=LOAD_MARKERS[load],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
                label=load.capitalize(),
            )
        ax.plot([0, 1], [0, 1], "--", color="#555555", linewidth=0.9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Label only the three largest departures from equality in each panel.
    offsets = [(4, 4), (4, -10), (-24, 4)]
    for ax, (metric, _, _) in zip(axes, specs):
        candidates = []
        for load in LOADS:
            for seed in SEEDS:
                x = idx[("Hungarian", load, seed)][metric]
                y = idx[("Phase C + S1", load, seed)][metric]
                candidates.append((abs(y - x), load, seed, x, y))
        for offset, (_, load, seed, x, y) in zip(
            offsets, sorted(candidates, reverse=True)[:3]
        ):
            ax.annotate(
                f"{load[0].upper()}{seed}",
                (x, y),
                xytext=offset,
                textcoords="offset points",
                fontsize=6.8,
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.subplots_adjust(top=0.86, wspace=0.38)
    save(fig, output_dir, "fig02_paired_s1_vs_hungarian")


def heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    value_format: str,
    limit: float,
) -> None:
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(SEEDS)), [str(seed) for seed in SEEDS], rotation=45)
    ax.set_yticks(np.arange(len(LOADS)), [load.capitalize() for load in LOADS])
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if abs(value) > 0.62 * limit else "black"
            ax.text(j, i, format(value, value_format), ha="center", va="center",
                    fontsize=5.8, color=color)
    ax.grid(False)
    return image


def comparison_matrix(
    idx: dict[tuple[str, str, int], dict], baseline: str, metric: str
) -> np.ndarray:
    matrix = np.zeros((len(LOADS), len(SEEDS)), dtype=float)
    for i, load in enumerate(LOADS):
        for j, seed in enumerate(SEEDS):
            candidate = idx[("Phase C + S1", load, seed)][metric]
            base = idx[(baseline, load, seed)][metric]
            # Positive always means that S1 is better.
            matrix[i, j] = candidate - base if metric == "completion_fraction" else base - candidate
    return matrix


def plot_delta_heatmaps(rows: list[dict], output_dir: Path) -> None:
    idx = row_index(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.2))
    panels = [
        ("Hungarian", "completion_fraction", "Completion advantage vs Hungarian", "+.2f", 0.85),
        ("Phase C", "completion_fraction", "Completion advantage vs Phase C", "+.2f", 0.85),
        ("Hungarian", "deadlock_ratio_mean", "Deadlock reduction vs Hungarian", "+.2f", 0.85),
        ("Phase C", "deadlock_ratio_mean", "Deadlock reduction vs Phase C", "+.2f", 0.85),
    ]
    for ax, (baseline, metric, title, fmt, limit) in zip(axes.flat, panels):
        heatmap(ax, comparison_matrix(idx, baseline, metric), title, fmt, limit)
    fig.text(0.5, 0.015, "S1 advantage: green = better, red = worse",
             ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.92,
                        wspace=0.24, hspace=0.48)
    save(fig, output_dir, "fig03_seed_delta_heatmaps")


def plot_paired_delta_basin(rows: list[dict], output_dir: Path) -> float:
    idx = row_index(rows)
    fig, ax = plt.subplots(figsize=(4.25, 3.45))
    all_x: list[float] = []
    all_y: list[float] = []
    point_rows: list[tuple[float, str, int, float, float]] = []
    for load in LOADS:
        x = np.asarray(
            [
                idx[("Phase C + S1", load, seed)]["completion_fraction"]
                - idx[("Hungarian", load, seed)]["completion_fraction"]
                for seed in SEEDS
            ]
        )
        y = np.asarray(
            [
                idx[("Phase C + S1", load, seed)]["deadlock_ratio_mean"]
                - idx[("Hungarian", load, seed)]["deadlock_ratio_mean"]
                for seed in SEEDS
            ]
        )
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
        point_rows.extend(
            (abs(xx) + abs(yy), load, seed, float(xx), float(yy))
            for seed, xx, yy in zip(SEEDS, x, y)
        )
        ax.scatter(
            x,
            y,
            s=40,
            color=LOAD_COLORS[load],
            marker=LOAD_MARKERS[load],
            edgecolor="white",
            linewidth=0.5,
            label=load.capitalize(),
        )

    offsets = [(4, 4), (4, -10), (-25, 4), (-25, -10)]
    for offset, (_, load, seed, xx, yy) in zip(
        offsets, sorted(point_rows, reverse=True)[:4]
    ):
        ax.annotate(
            f"{load[0].upper()}{seed}",
            (xx, yy),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.7,
        )

    corr = float(np.corrcoef(np.asarray(all_x), np.asarray(all_y))[0, 1])
    ax.axvline(0.0, color="#555555", linewidth=0.8)
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.axhline(0.02, color="#E45756", linestyle="--", linewidth=0.9,
               label="Deadlock NI margin")
    ax.set_xlabel("Completion fraction: S1 − Hungarian")
    ax.set_ylabel("Deadlock ratio: S1 − Hungarian")
    ax.set_title(f"Paired closed-loop outcomes (Pearson r = {corr:+.3f})")
    ax.text(0.98, 0.20, "desirable", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#2E7D32")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    save(fig, output_dir, "fig04_paired_delta_basin")
    return corr


def plot_s1_mechanism(rows: list[dict], output_dir: Path) -> dict[str, float]:
    idx = row_index(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85))
    rates: list[float] = []
    completion_effects: list[float] = []
    deadlock_reductions: list[float] = []
    effect_rows: list[tuple[str, int, float, float, float]] = []
    for load in LOADS:
        x = []
        dc = []
        dd = []
        for seed in SEEDS:
            s1 = idx[("Phase C + S1", load, seed)]
            phasec = idx[("Phase C", load, seed)]
            x.append(100.0 * s1["modified_decision_rate"])
            dc.append(s1["completion_fraction"] - phasec["completion_fraction"])
            dd.append(phasec["deadlock_ratio_mean"] - s1["deadlock_ratio_mean"])
        rates.extend(x)
        completion_effects.extend(dc)
        deadlock_reductions.extend(dd)
        effect_rows.extend(
            (load, seed, float(xx), float(yy_c), float(yy_d))
            for seed, xx, yy_c, yy_d in zip(SEEDS, x, dc, dd)
        )
        for ax, y in zip(axes, [dc, dd]):
            ax.scatter(
                x,
                y,
                s=35,
                color=LOAD_COLORS[load],
                marker=LOAD_MARKERS[load],
                edgecolor="white",
                linewidth=0.5,
                label=load.capitalize(),
            )

    offsets = [(4, 4), (4, -10), (-25, 4), (-25, -10)]
    for panel, value_index in [(0, 3), (1, 4)]:
        ranked = sorted(effect_rows, key=lambda row: abs(row[value_index]), reverse=True)
        for offset, row in zip(offsets, ranked[:4]):
            load, seed, xx = row[0], row[1], row[2]
            yy = row[value_index]
            axes[panel].annotate(
                f"{load[0].upper()}{seed}",
                (xx, yy),
                xytext=offset,
                textcoords="offset points",
                fontsize=6.7,
            )

    corr_completion = float(np.corrcoef(rates, completion_effects)[0, 1])
    corr_deadlock = float(np.corrcoef(rates, deadlock_reductions)[0, 1])
    axes[0].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[0].set_ylabel("Completion advantage over Phase C")
    axes[1].set_ylabel("Deadlock reduction vs Phase C")
    for ax in axes:
        ax.set_xlabel("Conversion-modified decisions (%)")
    axes[0].set_title(f"Completion-ratio effect (r = {corr_completion:+.2f})")
    axes[1].set_title(f"Deadlock effect (r = {corr_deadlock:+.2f})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(top=0.78, wspace=0.32)
    save(fig, output_dir, "fig05_s1_modification_vs_effect")
    return {
        "modified_rate_vs_completion_effect_pearson": corr_completion,
        "modified_rate_vs_deadlock_reduction_pearson": corr_deadlock,
    }


def plot_s1_modification_vs_baselines(
    rows: list[dict], output_dir: Path
) -> dict[str, dict[str, float]]:
    """Relate S1's internal conversion rate to closed-loop baseline effects."""
    idx = row_index(rows)
    baselines = ("Greedy", "Hungarian")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), sharex="col")
    correlations: dict[str, dict[str, float]] = {}

    for column, baseline_name in enumerate(baselines):
        all_rates: list[float] = []
        all_completion_effects: list[float] = []
        all_deadlock_reductions: list[float] = []

        for load in LOADS:
            rates: list[float] = []
            completion_effects: list[float] = []
            deadlock_reductions: list[float] = []
            for seed in SEEDS:
                s1 = idx[("Phase C + S1", load, seed)]
                baseline = idx[(baseline_name, load, seed)]
                rates.append(100.0 * s1["modified_decision_rate"])
                completion_effects.append(
                    s1["completion_fraction"] - baseline["completion_fraction"]
                )
                deadlock_reductions.append(
                    baseline["deadlock_ratio_mean"] - s1["deadlock_ratio_mean"]
                )

            all_rates.extend(rates)
            all_completion_effects.extend(completion_effects)
            all_deadlock_reductions.extend(deadlock_reductions)
            for row, values in enumerate(
                (completion_effects, deadlock_reductions)
            ):
                axes[row, column].scatter(
                    rates,
                    values,
                    s=35,
                    color=LOAD_COLORS[load],
                    marker=LOAD_MARKERS[load],
                    edgecolor="white",
                    linewidth=0.5,
                    label=load.capitalize(),
                )

        completion_corr = float(
            np.corrcoef(all_rates, all_completion_effects)[0, 1]
        )
        deadlock_corr = float(
            np.corrcoef(all_rates, all_deadlock_reductions)[0, 1]
        )
        correlations[baseline_name.lower()] = {
            "modified_rate_vs_completion_advantage_pearson": completion_corr,
            "modified_rate_vs_deadlock_reduction_pearson": deadlock_corr,
        }

        axes[0, column].set_title(
            f"vs {baseline_name}: completion (r = {completion_corr:+.2f})"
        )
        axes[1, column].set_title(
            f"vs {baseline_name}: deadlock (r = {deadlock_corr:+.2f})"
        )

    for ax in axes.flat:
        ax.axhline(0.0, color="#555555", linewidth=0.8)
    axes[0, 0].set_ylabel("Completion advantage\n(S1 - baseline)")
    axes[1, 0].set_ylabel("Deadlock reduction\n(baseline - S1)")
    for ax in axes[1, :]:
        ax.set_xlabel("S1 conversion-modified contexts (%)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(top=0.88, hspace=0.42, wspace=0.30)
    save(fig, output_dir, "fig08_s1_modification_vs_baselines")
    return correlations


def plot_focus_cases(rows: list[dict], output_dir: Path) -> None:
    idx = row_index(rows)
    effects = []
    for load in LOADS:
        for seed in SEEDS:
            s1 = idx[("Phase C + S1", load, seed)]
            phasec = idx[("Phase C", load, seed)]
            completion_advantage = (
                s1["completion_fraction"] - phasec["completion_fraction"]
            )
            deadlock_reduction = (
                phasec["deadlock_ratio_mean"] - s1["deadlock_ratio_mean"]
            )
            effects.append(
                (completion_advantage + deadlock_reduction, load, seed)
            )
    ordered = sorted(effects)
    cases = [(load, seed) for _, load, seed in ordered[:4]]
    cases += [(load, seed) for _, load, seed in ordered[-4:]]
    methods = ["Hungarian", "Phase C", "Phase C + S1"]
    labels = [f"{load[0].upper()}{seed}" for load, seed in cases]
    x = np.arange(len(cases), dtype=float)
    width = 0.25
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.35), sharex=True)
    for offset, arm in zip([-width, 0.0, width], methods):
        completion = [idx[(arm, load, seed)]["completion_fraction"] for load, seed in cases]
        deadlock = [idx[(arm, load, seed)]["deadlock_ratio_mean"] for load, seed in cases]
        axes[0].bar(x + offset, completion, width=width, color=COLORS[arm], label=arm)
        axes[1].bar(x + offset, deadlock, width=width, color=COLORS[arm], label=arm)
    axes[0].set_ylabel("Completion fraction")
    axes[1].set_ylabel("Mean deadlock ratio")
    axes[0].set_ylim(0.0, 1.02)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Load/seed (left: S1 regressions; right: S1 improvements)")
    axes[0].set_title("Focused Phase-C vs S1 trajectory branches")
    axes[0].legend(frameon=False, ncol=3, loc="upper center")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.subplots_adjust(hspace=0.16)
    save(fig, output_dir, "fig06_low_mid_focus_cases")


def plot_formal_forest(root: Path, output_dir: Path) -> None:
    with (root / "phase_c_s1_hungarian_validation.json").open(
        "r", encoding="utf-8"
    ) as handle:
        validation = json.load(handle)
    comparison = validation["comparisons"]["phasec_s1_minus_hungarian"]["metrics"]
    categories = ["Overall", "Low", "Mid", "High"]
    y = np.arange(len(categories))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharey=True)

    specs = [
        ("completed_orders", "Completed orders: S1 − Hungarian", "right"),
        ("deadlock_ratio_mean", "Deadlock ratio: S1 − Hungarian", "left"),
    ]
    for ax, (metric, xlabel, desirable) in zip(axes, specs):
        report = comparison[metric]
        entries = [report["overall_seed_cluster"]] + [
            report["by_load"][load] for load in LOADS
        ]
        means = np.asarray([entry["mean"] for entry in entries], dtype=float)
        low_err = np.asarray(
            [entry["mean"] - entry["ci95"][0] for entry in entries], dtype=float
        )
        high_err = np.asarray(
            [entry["ci95"][1] - entry["mean"] for entry in entries], dtype=float
        )
        point_colors = ["#222222"] + [LOAD_COLORS[load] for load in LOADS]
        for yy, mean, lo, hi, color in zip(y, means, low_err, high_err, point_colors):
            ax.errorbar(
                mean,
                yy,
                xerr=np.asarray([[lo], [hi]]),
                fmt="o",
                color=color,
                markersize=5,
                linewidth=1.3,
                capsize=2.5,
            )
        ax.axvline(0.0, color="#555555", linewidth=0.9)
        if metric == "deadlock_ratio_mean":
            ax.axvline(0.02, color="#E45756", linestyle="--", linewidth=1.0,
                       label="NI margin (+0.02)")
            ax.legend(frameon=False, loc="lower right")
        ax.set_xlabel(xlabel)
        ax.set_yticks(y, categories)
        ax.grid(axis="y", visible=False)
        edge = "right" if desirable == "right" else "left"
        ax.text(
            0.98 if edge == "right" else 0.02,
            0.04,
            "desirable",
            transform=ax.transAxes,
            ha=edge,
            va="bottom",
            fontsize=7.5,
            color="#2E7D32",
        )
    axes[0].set_title("Completed-order effect")
    axes[1].set_title("Primary congestion test")
    fig.subplots_adjust(wspace=0.28)
    save(fig, output_dir, "fig07_formal_paired_effects")


def write_diagnostics(
    root: Path,
    rows: list[dict],
    output_dir: Path,
    paired_corr: float,
    mechanism_corrs: dict[str, float],
    baseline_corrs: dict[str, dict[str, float]],
) -> None:
    idx = row_index(rows)
    csv_path = output_dir / "seed_level_diagnostics.csv"
    fieldnames = [
        "load",
        "seed",
        "s1_completion_fraction",
        "greedy_completion_fraction",
        "hungarian_completion_fraction",
        "phasec_completion_fraction",
        "s1_minus_greedy_completion",
        "s1_minus_hungarian_completion",
        "s1_minus_phasec_completion",
        "s1_deadlock_ratio_mean",
        "greedy_deadlock_ratio_mean",
        "hungarian_deadlock_ratio_mean",
        "phasec_deadlock_ratio_mean",
        "s1_minus_greedy_deadlock",
        "s1_minus_hungarian_deadlock",
        "s1_minus_phasec_deadlock",
        "s1_modified_decisions",
        "s1_modified_decision_rate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for load in LOADS:
            for seed in SEEDS:
                s1 = idx[("Phase C + S1", load, seed)]
                g = idx[("Greedy", load, seed)]
                h = idx[("Hungarian", load, seed)]
                p = idx[("Phase C", load, seed)]
                writer.writerow(
                    {
                        "load": load,
                        "seed": seed,
                        "s1_completion_fraction": s1["completion_fraction"],
                        "greedy_completion_fraction": g["completion_fraction"],
                        "hungarian_completion_fraction": h["completion_fraction"],
                        "phasec_completion_fraction": p["completion_fraction"],
                        "s1_minus_greedy_completion": s1["completion_fraction"]
                        - g["completion_fraction"],
                        "s1_minus_hungarian_completion": s1["completion_fraction"]
                        - h["completion_fraction"],
                        "s1_minus_phasec_completion": s1["completion_fraction"]
                        - p["completion_fraction"],
                        "s1_deadlock_ratio_mean": s1["deadlock_ratio_mean"],
                        "greedy_deadlock_ratio_mean": g["deadlock_ratio_mean"],
                        "hungarian_deadlock_ratio_mean": h["deadlock_ratio_mean"],
                        "phasec_deadlock_ratio_mean": p["deadlock_ratio_mean"],
                        "s1_minus_greedy_deadlock": s1["deadlock_ratio_mean"]
                        - g["deadlock_ratio_mean"],
                        "s1_minus_hungarian_deadlock": s1["deadlock_ratio_mean"]
                        - h["deadlock_ratio_mean"],
                        "s1_minus_phasec_deadlock": s1["deadlock_ratio_mean"]
                        - p["deadlock_ratio_mean"],
                        "s1_modified_decisions": s1["modified_decisions"],
                        "s1_modified_decision_rate": s1["modified_decision_rate"],
                    }
                )

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for arm in ARMS:
        aggregate[arm] = {}
        for load in LOADS:
            aggregate[arm][load] = {}
            for metric in ["completion_fraction", "deadlock_ratio_mean", "severe_events_per_100"]:
                values = select(rows, arm, load, metric)
                mean, ci = mean_ci95(values)
                aggregate[arm][load][metric] = {"mean": mean, "ci95_half_width": ci}

    summary = {
        "schema_version": "phase_c_s1_hungarian_figure_summary_v1",
        "source": str(root / "phase_c_s1_hungarian_validation.json"),
        "aggregate": aggregate,
        "paired_s1_minus_hungarian_completion_vs_deadlock_pearson": paired_corr,
        "mechanism_correlations": mechanism_corrs,
        "baseline_effect_correlations": baseline_corrs,
        "interpretation_guard": (
            "Phase C + S1 includes frozen long-risk beta and conversion. "
            "Conversion-modified decision counts do not isolate either component causally."
        ),
    }
    with (output_dir / "figure_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main() -> None:
    global SEEDS
    args = parse_args()
    root = args.root.resolve()
    SEEDS = infer_seeds(root)
    output_dir = (args.output_dir or (root / "figures")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_icra_style()
    rows = load_rows(root)
    plot_aggregate(rows, output_dir)
    plot_paired_s1_hungarian(rows, output_dir)
    plot_delta_heatmaps(rows, output_dir)
    paired_corr = plot_paired_delta_basin(rows, output_dir)
    mechanism_corrs = plot_s1_mechanism(rows, output_dir)
    baseline_corrs = plot_s1_modification_vs_baselines(rows, output_dir)
    plot_focus_cases(rows, output_dir)
    plot_formal_forest(root, output_dir)
    write_diagnostics(
        root,
        rows,
        output_dir,
        paired_corr,
        mechanism_corrs,
        baseline_corrs,
    )
    print(f"wrote Phase-C/S1/Hungarian figures to {output_dir}")


if __name__ == "__main__":
    main()
