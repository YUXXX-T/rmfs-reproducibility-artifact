"""Cross-load plots for the Phase-C S/J x station-admission factorial.

This script is analysis-only: it reads the validated low/mid/high summaries and
the already-produced station traces.  It does not invoke the simulator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_LOW_MID_ROOT = BASE_ROOT / "station_admission_sj_factorial_low_mid_551_560_v1"
DEFAULT_HIGH_ROOT = BASE_ROOT / "station_admission_sj_factorial_551_560_v1"
DEFAULT_OUTPUT = DEFAULT_LOW_MID_ROOT / "figures_cross_load"

LOADS = ("low", "mid", "high")
LOAD_LABELS = {"low": "Low", "mid": "Mid", "high": "High"}
LOAD_COLORS = {"low": "#56B4E9", "mid": "#E69F00", "high": "#D55E00"}
ADMISSIONS = ("physical_only", "committed_v1", "eta_v3", "fifo_v2")
ADMISSION_LABELS = {
    "physical_only": "Physical-only",
    "committed_v1": "Committed V1",
    "eta_v3": "ETA V3",
    "fifo_v2": "FIFO V2",
}
ADMISSION_COLORS = {
    "physical_only": "#D55E00",
    "committed_v1": "#0072B2",
    "eta_v3": "#009E73",
    "fifo_v2": "#CC79A7",
}
KEY_POLICIES = ("s1_j1", "s0_j0", "s1_dynamic_j", "greedy", "hungarian")
POLICY_LABELS = {
    "s1_j1": "S1+J1",
    "s0_j0": "Phase C",
    "s1_dynamic_j": "S1+Dynamic-J",
    "greedy": "Greedy",
    "hungarian": "Hungarian",
}
POLICY_COLORS = {
    "s1_j1": "#E69F00",
    "s0_j0": "#0072B2",
    "s1_dynamic_j": "#F0E442",
    "greedy": "#009E73",
    "hungarian": "#7F7F7F",
}
COLLAPSE_THRESHOLD = 0.80
HEALTHY_THRESHOLD = 0.95
T_CRITICAL_9_DF = 2.262


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-mid-root", type=Path, default=DEFAULT_LOW_MID_ROOT)
    parser.add_argument("--high-root", type=Path, default=DEFAULT_HIGH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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


def mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    vals = [float(value) for value in values]
    mean = statistics.fmean(vals)
    if len(vals) < 2:
        return mean, mean, mean
    half = T_CRITICAL_9_DF * statistics.stdev(vals) / math.sqrt(len(vals))
    return mean, mean - half, mean + half


def load_summaries(low_mid_root: Path, high_root: Path) -> dict[str, dict[str, Any]]:
    summaries = {}
    for load in LOADS:
        root = low_mid_root if load in {"low", "mid"} else high_root
        path = root / "validation" / f"station_admission_sj_factorial_{load}.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        integrity = summary["integrity"]
        required = (
            integrity["manifest_pairing_passed"],
            integrity["policy_pairing_passed"],
            integrity["all_arm_audits_passed"],
            integrity["physical_capacity_violations_all_arms"] == 0,
            not summary["meta"]["missing_results"],
        )
        if not all(required):
            raise RuntimeError(f"summary integrity failed: {path}")
        summaries[load] = summary
    return summaries


def arm_runs(summary: dict[str, Any], arm: str) -> dict[int, dict[str, Any]]:
    return {int(row["seed"]): row for row in summary["arms"][arm]["runs"]}


def empirical_ceiling(summary: dict[str, Any]) -> dict[int, float]:
    """Per-seed attainable completion ceiling from controlled-admission arms.

    This avoids applying the high-load absolute `<400` rule to low/mid, whose
    frozen manifests contain different numbers of orders and late arrivals.
    """

    controlled = [arm for arm in summary["arms"] if not arm.endswith("physical_only")]
    seeds = [int(seed) for seed in summary["meta"]["seeds"]]
    by_arm = {arm: arm_runs(summary, arm) for arm in controlled}
    return {
        seed: max(float(by_arm[arm][seed]["completed_orders"]) for arm in controlled)
        for seed in seeds
    }


def completion_efficiency(summary: dict[str, Any], arm: str) -> list[float]:
    ceiling = empirical_ceiling(summary)
    runs = arm_runs(summary, arm)
    return [float(runs[seed]["completed_orders"]) / ceiling[seed] for seed in sorted(ceiling)]


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=300)
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)


def plot_physical_efficiency(summaries: dict[str, dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 8.0), sharex=True)
    x = np.arange(len(KEY_POLICIES))
    width = 0.24
    for index, load in enumerate(LOADS):
        means, lows, highs, collapses = [], [], [], []
        for policy in KEY_POLICIES:
            values = completion_efficiency(summaries[load], f"{policy}_physical_only")
            mean, low, high = mean_ci(values)
            means.append(100.0 * mean)
            lows.append(100.0 * (mean - low))
            highs.append(100.0 * (high - mean))
            collapses.append(sum(value < COLLAPSE_THRESHOLD for value in values))
        offset = (index - 1) * width
        axes[0].bar(
            x + offset,
            means,
            width,
            yerr=np.asarray([lows, highs]),
            capsize=2.5,
            color=LOAD_COLORS[load],
            alpha=0.88,
            label=LOAD_LABELS[load],
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        axes[1].bar(
            x + offset,
            collapses,
            width,
            color=LOAD_COLORS[load],
            alpha=0.88,
            label=LOAD_LABELS[load],
        )
    axes[0].axhline(100.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Completion efficiency (%)\nmean ± paired-seed 95% CI")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("Collapse seeds / 10\n(efficiency < 80%)")
    axes[1].set_ylim(0, 10.8)
    axes[1].set_yticks(range(0, 11, 2))
    axes[1].set_xticks(x, [POLICY_LABELS[p] for p in KEY_POLICIES], rotation=20, ha="right")
    axes[0].legend(loc="lower left", ncol=3, frameon=False)
    axes[0].grid(axis="x", visible=False)
    axes[1].grid(axis="x", visible=False)
    fig.suptitle("Physical-only policy robustness changes sharply with workload", fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.12, top=0.93, hspace=0.18)
    save_figure(fig, output, "fig01_physical_efficiency_and_collapse")


def contrast_row(summary: dict[str, Any], baseline: str) -> dict[str, Any]:
    if baseline in {"greedy", "hungarian"}:
        category = "model_vs_external_same_admission"
        name = f"s1_j1_minus_{baseline}__physical_only"
    elif baseline == "s0_j0":
        left = arm_runs(summary, "s1_j1_physical_only")
        right = arm_runs(summary, "s0_j0_physical_only")
        values = [left[seed]["completed_orders"] - right[seed]["completed_orders"] for seed in sorted(left)]
        mean, low, high = mean_ci(values)
        return {"completed_orders_contrast_mean": mean, "completed_orders_contrast_ci95": [low, high]}
    elif baseline == "s1_dynamic_j":
        source = summary["comparisons"]["context_scheduler_effect_within_same_s_and_admission"]
        row = source["dynamic_j_minus_j1__s1__physical_only"]
        return {
            "completed_orders_contrast_mean": -float(row["completed_orders_contrast_mean"]),
            "completed_orders_contrast_ci95": [
                -float(row["completed_orders_contrast_ci95"][1]),
                -float(row["completed_orders_contrast_ci95"][0]),
            ],
        }
    else:
        raise KeyError(baseline)
    return summary["comparisons"][category][name]


def plot_contrast_forest(summaries: dict[str, dict[str, Any]], output: Path) -> None:
    baselines = ("greedy", "hungarian", "s0_j0", "s1_dynamic_j")
    baseline_labels = {
        "greedy": "Greedy",
        "hungarian": "Hungarian",
        "s0_j0": "Phase C",
        "s1_dynamic_j": "S1+Dynamic-J",
    }
    rows = []
    for load in LOADS:
        for baseline in baselines:
            row = contrast_row(summaries[load], baseline)
            mean = float(row["completed_orders_contrast_mean"])
            low, high = [float(value) for value in row["completed_orders_contrast_ci95"]]
            rows.append((load, baseline, mean, low, high))
    fig, ax = plt.subplots(figsize=(10.8, 7.1))
    y = np.arange(len(rows))
    labels = []
    for index, (load, baseline, mean, low, high) in enumerate(rows):
        ax.errorbar(
            mean,
            index,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color=LOAD_COLORS[load],
            capsize=3,
            linewidth=1.3,
            markersize=5,
        )
        labels.append(f"{LOAD_LABELS[load]}: S1+J1 − {baseline_labels[baseline]}")
        ax.text(high + 7, index, f"{mean:+.1f} [{low:+.1f}, {high:+.1f}]", va="center", fontsize=7.3)
    ax.axvline(0.0, color="#444444", linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Completed-order contrast (paired mean and 95% CI)")
    ax.set_title("S1+J1 physical-only advantage is workload-dependent")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(row[3] for row in rows) - 30, max(row[4] for row in rows) + 115)
    fig.subplots_adjust(left=0.28, right=0.97, bottom=0.10, top=0.92)
    save_figure(fig, output, "fig02_s1j1_physical_contrast_forest")


def plot_admission_tradeoff(summaries: dict[str, dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.3), sharex=True)
    selected = ("s1_j1", "greedy", "s0_j0")
    x = np.arange(len(ADMISSIONS))
    for column, load in enumerate(LOADS):
        summary = summaries[load]
        for policy in selected:
            efficiencies = []
            deadlocks = []
            for admission in ADMISSIONS:
                efficiencies.append(
                    100.0 * statistics.fmean(completion_efficiency(summary, f"{policy}_{admission}"))
                )
                deadlocks.append(100.0 * float(summary["arms"][f"{policy}_{admission}"]["deadlock_ratio_mean"]))
            axes[0, column].plot(
                x,
                efficiencies,
                marker="o",
                linewidth=1.6,
                markersize=4.5,
                color=POLICY_COLORS[policy],
                label=POLICY_LABELS[policy],
            )
            axes[1, column].plot(
                x,
                deadlocks,
                marker="o",
                linewidth=1.6,
                markersize=4.5,
                color=POLICY_COLORS[policy],
                label=POLICY_LABELS[policy],
            )
        axes[0, column].set_title(f"{LOAD_LABELS[load]} load")
        axes[0, column].set_ylim(0, 107)
        axes[1, column].set_ylim(bottom=0)
        axes[1, column].set_xticks(x, [ADMISSION_LABELS[a] for a in ADMISSIONS], rotation=30, ha="right")
        axes[0, column].grid(axis="x", visible=False)
        axes[1, column].grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("Completion efficiency (%)")
    axes[1, 0].set_ylabel("Mean deadlock ratio (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Admission controls collapse but compresses observable policy differences", fontsize=12, y=1.035)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.88, wspace=0.20, hspace=0.20)
    save_figure(fig, output, "fig03_admission_tradeoff")


def plot_seed_heatmaps(summaries: dict[str, dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), sharey=True)
    for ax, load in zip(axes, LOADS):
        summary = summaries[load]
        matrix = 100.0 * np.asarray(
            [completion_efficiency(summary, f"{policy}_physical_only") for policy in KEY_POLICIES]
        )
        image = ax.imshow(matrix, cmap="RdYlGn", vmin=20, vmax=100, aspect="auto")
        seeds = summary["meta"]["seeds"]
        ax.set_xticks(range(len(seeds)), [str(seed) for seed in seeds], rotation=45, ha="right")
        ax.set_title(f"{LOAD_LABELS[load]} load")
        ax.set_xlabel("Seed")
        ax.grid(False)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value < 45 else "black",
                )
    axes[0].set_yticks(range(len(KEY_POLICIES)), [POLICY_LABELS[p] for p in KEY_POLICIES])
    fig.colorbar(image, ax=axes, fraction=0.016, pad=0.02, label="Completion efficiency (%)")
    fig.suptitle("Physical-only outcomes are bifurcations, not uniform service-rate shifts", fontsize=12, y=1.00)
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.18, top=0.88, wspace=0.08)
    save_figure(fig, output, "fig04_physical_seed_efficiency_heatmaps")


def load_trace(root: Path, load: str, policy: str, seed: int) -> dict[str, Any]:
    path = root / "per_arm" / f"{policy}_physical_only" / f"{load}_seed{seed}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def station_series(payload: dict[str, Any], station_id: int) -> tuple[list[int], list[float], list[float]]:
    ticks, occupancy, committed = [], [], []
    for row in payload["station_admission_audit"]["trace"]:
        station = next(item for item in row["stations"] if int(item["station_id"]) == station_id)
        ticks.append(int(row["tick"]))
        occupancy.append(float(station["occupancy"]))
        committed.append(float(station["committed_load"]))
    return ticks, occupancy, committed


def plot_station_lock_examples(low_mid_root: Path, output: Path) -> None:
    cases = [
        ("low", 559, 3, "s1_j1", "s0_j0", "Low seed 559: S1+J1 collapses; Phase C stays healthy"),
        ("mid", 553, 4, "s1_j1", "greedy", "Mid seed 553: Greedy collapses; S1+J1 stays healthy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.0), sharex="col")
    for column, (load, seed, station_id, left, right, title) in enumerate(cases):
        payloads = {
            left: load_trace(low_mid_root, load, left, seed),
            right: load_trace(low_mid_root, load, right, seed),
        }
        for policy, payload in payloads.items():
            ticks, occupancy, committed = station_series(payload, station_id)
            label = POLICY_LABELS[policy]
            color = POLICY_COLORS[policy]
            axes[0, column].plot(ticks, occupancy, color=color, linewidth=1.4, label=label)
            axes[1, column].plot(ticks, committed, color=color, linewidth=1.4, label=label)
        axes[0, column].axhline(7, color="#444444", linestyle="--", linewidth=0.8, label="Physical capacity")
        axes[1, column].axhline(7, color="#444444", linestyle="--", linewidth=0.8)
        axes[0, column].set_title(f"{title}\nStation {station_id}")
        axes[0, column].set_ylim(bottom=0)
        axes[1, column].set_ylim(bottom=0)
        axes[1, column].set_xlabel("Tick")
        axes[0, column].grid(axis="x", visible=False)
        axes[1, column].grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("Physical occupancy")
    axes[1, 0].set_ylabel("Committed load")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("The winning policy is the one that avoids a sustained single-station lock", fontsize=12, y=1.045)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.09, top=0.86, wspace=0.14, hspace=0.16)
    save_figure(fig, output, "fig05_station_lock_examples")


def write_csvs(summaries: dict[str, dict[str, Any]], output: Path) -> None:
    with (output / "cross_load_key_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "load",
                "policy",
                "admission",
                "completed_orders_mean",
                "completion_efficiency_mean",
                "completion_efficiency_ci95_low",
                "completion_efficiency_ci95_high",
                "collapse_seed_count_lt_0_8",
                "healthy_seed_count_ge_0_95",
                "deadlock_ratio_mean",
                "stall_ratio_mean",
                "max_committed_load",
                "max_occupancy",
            ]
        )
        for load in LOADS:
            summary = summaries[load]
            for policy in KEY_POLICIES:
                for admission in ADMISSIONS:
                    arm = f"{policy}_{admission}"
                    values = completion_efficiency(summary, arm)
                    mean, low, high = mean_ci(values)
                    aggregate = summary["arms"][arm]
                    writer.writerow(
                        [
                            load,
                            policy,
                            admission,
                            aggregate["completed_orders_mean"],
                            mean,
                            low,
                            high,
                            sum(value < COLLAPSE_THRESHOLD for value in values),
                            sum(value >= HEALTHY_THRESHOLD for value in values),
                            aggregate["deadlock_ratio_mean"],
                            aggregate["stall_ratio_mean"],
                            aggregate["max_committed_load"],
                            aggregate["max_occupancy"],
                        ]
                    )

    with (output / "cross_load_physical_seed_efficiency.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["load", "seed", "policy", "completed_orders", "empirical_ceiling", "efficiency"])
        for load in LOADS:
            summary = summaries[load]
            ceiling = empirical_ceiling(summary)
            for policy in KEY_POLICIES:
                runs = arm_runs(summary, f"{policy}_physical_only")
                for seed in sorted(ceiling):
                    completed = float(runs[seed]["completed_orders"])
                    writer.writerow([load, seed, policy, completed, ceiling[seed], completed / ceiling[seed]])


def make_overview(output: Path) -> None:
    names = (
        ("fig01_physical_efficiency_and_collapse.png", "Physical-only efficiency and collapse"),
        ("fig02_s1j1_physical_contrast_forest.png", "S1+J1 paired contrasts"),
        ("fig03_admission_tradeoff.png", "Admission trade-off"),
        ("fig05_station_lock_examples.png", "Representative station-lock traces"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.2))
    for ax, (filename, title) in zip(axes.flat, names):
        ax.imshow(mpimg.imread(output / filename))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Phase-C S/J × station-admission factorial: low / mid / high overview", fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.96, wspace=0.02, hspace=0.08)
    fig.savefig(output / "fig00_cross_load_overview.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output / "fig00_cross_load_overview.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def write_readme(output: Path) -> None:
    text = """# Cross-load factorial figures

These figures use the validated 551–560 summaries only; no simulation is rerun.

For cross-load collapse analysis, completion efficiency is normalized by the
per-load/per-seed empirical controlled-admission ceiling: the maximum completed
orders among all non-physical-only arms.  A collapse is efficiency `< 0.80`.
This replaces the high-load-only absolute `<400 orders` rule, which is invalid
for low/mid manifests with different order totals.

- `fig00_cross_load_overview`: compact summary page.
- `fig01_physical_efficiency_and_collapse`: normalized performance and collapse counts.
- `fig02_s1j1_physical_contrast_forest`: paired raw-order contrasts and 95% CIs.
- `fig03_admission_tradeoff`: completion efficiency and deadlock across admissions.
- `fig04_physical_seed_efficiency_heatmaps`: seed-level bifurcations.
- `fig05_station_lock_examples`: representative low-load reversal and mid-load win.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(args.low_mid_root, args.high_root)
    plot_physical_efficiency(summaries, args.output_dir)
    plot_contrast_forest(summaries, args.output_dir)
    plot_admission_tradeoff(summaries, args.output_dir)
    plot_seed_heatmaps(summaries, args.output_dir)
    plot_station_lock_examples(args.low_mid_root, args.output_dir)
    write_csvs(summaries, args.output_dir)
    make_overview(args.output_dir)
    write_readme(args.output_dir)


if __name__ == "__main__":
    main()
