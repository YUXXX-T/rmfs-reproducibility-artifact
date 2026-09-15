"""Plot one load of the Phase-C S/J x station-admission factorial.

The plotting pass is deliberately independent of the simulator.  It reads the
validated summary JSON and the frozen per-arm result JSON files, computes
seed-level confidence intervals/contrasts, and writes publication-friendly
PNG/PDF figures plus CSV diagnostics.
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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_ROOT = BASE_ROOT / "station_admission_sj_factorial_551_560_v1"
COLLAPSE_EFFICIENCY_THRESHOLD = 0.80

SEEDS = list(range(551, 561))
ADMISSIONS = ("physical_only", "committed_v1", "eta_v3", "fifo_v2")
ADMISSION_LABELS = {
    "physical_only": "Physical-only",
    "committed_v1": "Committed V1",
    "eta_v3": "ETA V3",
    "fifo_v2": "FIFO V2",
}
POLICIES = (
    "s0_j0",
    "s0_j1",
    "s0_dynamic_j",
    "s1_j0",
    "s1_j1",
    "s1_dynamic_j",
    "greedy",
    "hungarian",
)
POLICY_LABELS = {
    "s0_j0": "S0 + J0\n(Phase C)",
    "s0_j1": "S0 + J1",
    "s0_dynamic_j": "S0 + Dynamic-J",
    "s1_j0": "S1 + J0",
    "s1_j1": "S1 + J1",
    "s1_dynamic_j": "S1 + Dynamic-J",
    "greedy": "Greedy",
    "hungarian": "Hungarian",
}
POLICY_SHORT = {
    "s0_j0": "S0J0",
    "s0_j1": "S0J1",
    "s0_dynamic_j": "S0DJ",
    "s1_j0": "S1J0",
    "s1_j1": "S1J1",
    "s1_dynamic_j": "S1DJ",
    "greedy": "Greedy",
    "hungarian": "Hungarian",
}

COLORS = {
    "physical_only": "#D55E00",
    "committed_v1": "#0072B2",
    "eta_v3": "#009E73",
    "fifo_v2": "#CC79A7",
}
POLICY_COLORS = {
    "s0_j0": "#7F7F7F",
    "s0_j1": "#BDBDBD",
    "s0_dynamic_j": "#636363",
    "s1_j0": "#56B4E9",
    "s1_j1": "#E69F00",
    "s1_dynamic_j": "#F0E442",
    "greedy": "#009E73",
    "hungarian": "#0072B2",
}

T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


TRACE_CASES = {
    "low": (
        ("S1+J1 physical-only, seed 559", "s1_j1_physical_only", 559, "#D55E00"),
        ("Phase C physical-only, seed 559", "s0_j0_physical_only", 559, "#0072B2"),
        ("Greedy physical-only, seed 551", "greedy_physical_only", 551, "#009E73"),
        ("S1+J1 physical-only, seed 551", "s1_j1_physical_only", 551, "#E69F00"),
    ),
    "mid": (
        ("S1+J1 physical-only, seed 553", "s1_j1_physical_only", 553, "#D55E00"),
        ("Greedy physical-only, seed 553", "greedy_physical_only", 553, "#009E73"),
        ("S1+J1 physical-only, seed 557", "s1_j1_physical_only", 557, "#E69F00"),
        ("Greedy physical-only, seed 557", "greedy_physical_only", 557, "#0072B2"),
    ),
    "high": (
        ("S1+J1 physical-only, seed 551", "s1_j1_physical_only", 551, "#D55E00"),
        ("S1+J1 physical-only, seed 555", "s1_j1_physical_only", 555, "#E69F00"),
        ("Greedy physical-only, seed 555", "greedy_physical_only", 555, "#009E73"),
        ("Greedy physical-only, seed 556", "greedy_physical_only", 556, "#0072B2"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--load", choices=("low", "mid", "high"), default="high")
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


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=300)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def make_contact_sheet(output: Path, load: str) -> None:
    """Make one compact page for the four most diagnostic figures."""
    names = [
        ("fig01_policy_admission_heatmaps.png", "Policy × admission surface"),
        ("fig03_physical_only_seed_lines.png", "Physical-only seed behavior"),
        ("fig05_physical_only_contrast_forest.png", "Paired physical-only contrasts"),
        ("fig08_distribution_and_collapse_rate.png", "Distribution and collapse rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.0))
    for ax, (filename, title) in zip(axes.flat, names):
        image = mpimg.imread(output / filename)
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{load.title()}-load Phase-C S/J × station-admission factorial overview", fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.96, wspace=0.02, hspace=0.08)
    fig.savefig(output / "fig00_overview.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output / "fig00_overview.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def arm_key(policy: str, admission: str) -> str:
    return f"{policy}_{admission}"


def mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    mean = statistics.fmean(vals)
    if len(vals) < 2:
        return (mean, mean, mean)
    critical = T_CRITICAL_975.get(len(vals) - 1, 1.96)
    half = critical * statistics.stdev(vals) / math.sqrt(len(vals))
    return (mean, mean - half, mean + half)


def load_data(root: Path, load: str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    summary_path = root / "validation" / f"station_admission_sj_factorial_{load}.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seeds = [int(s) for s in summary["meta"]["seeds"]]
    runs: dict[str, list[dict[str, Any]]] = {}
    for policy in POLICIES:
        for admission in ADMISSIONS:
            arm = arm_key(policy, admission)
            path = root / "per_arm" / arm
            rows = []
            for seed in seeds:
                result_path = path / f"{load}_seed{seed}.json"
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                metrics = payload["metrics"]
                audit = payload.get("audit", {})
                if not audit.get("passed", False):
                    raise RuntimeError(f"failed audit: {result_path}")
                rows.append(
                    {
                        "seed": seed,
                        "completed_orders": float(metrics["completed_orders"]),
                        "deadlock_ratio_mean": float(metrics["deadlock_ratio_mean"]),
                        "stall_ratio_mean": float(metrics["stall_ratio_mean"]),
                        "open_order_count": float(metrics["open_order_count"]),
                        "pending_order_count": float(metrics["pending_order_count"]),
                        "capacity_rejections": float(metrics["station_capacity_rejections"]),
                        "over_capacity_grants": float(metrics["station_over_capacity_grants"]),
                        "max_committed_load": max(
                            [
                                float(value)
                                for value in (
                                    (payload.get("station_admission_audit") or {}).get(
                                        "max_committed_load", {}
                                    )
                                    or {}
                                ).values()
                            ]
                            or [0.0]
                        ),
                        "max_occupancy": max(
                            [
                                float(value)
                                for value in (
                                    (payload.get("station_admission_audit") or {}).get(
                                        "max_occupancy", {}
                                    )
                                    or {}
                                ).values()
                            ]
                            or [0.0]
                        ),
                        "payload": payload,
                    }
                )
            runs[arm] = rows
    if not summary["integrity"]["manifest_pairing_passed"]:
        raise RuntimeError("summary manifest pairing failed")
    if not summary["integrity"]["policy_pairing_passed"]:
        raise RuntimeError("summary policy pairing failed")
    return summary, runs


def row_values(runs: dict[str, list[dict[str, Any]]], arm: str, metric: str) -> list[float]:
    return [float(row[metric]) for row in runs[arm]]


def empirical_controlled_ceiling(runs: dict[str, list[dict[str, Any]]]) -> dict[int, float]:
    """Per-seed completion ceiling from all non-physical-only arms."""

    controlled_arms = [arm for arm in runs if not arm.endswith("physical_only")]
    seeds = sorted(int(row["seed"]) for row in next(iter(runs.values())))
    by_arm = {
        arm: {int(row["seed"]): float(row["completed_orders"]) for row in rows}
        for arm, rows in runs.items()
    }
    return {seed: max(by_arm[arm][seed] for arm in controlled_arms) for seed in seeds}


def plot_heatmaps(summary: dict[str, Any], runs: dict[str, list[dict[str, Any]]], output: Path, load: str) -> None:
    matrices = []
    for metric in ("completed_orders", "deadlock_ratio_mean", "stall_ratio_mean"):
        matrix = np.asarray(
            [
                [mean_ci(row_values(runs, arm_key(policy, admission), metric))[0]
                 for admission in ADMISSIONS]
                for policy in POLICIES
            ],
            dtype=float,
        )
        matrices.append(matrix)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
    titles = ["Completed orders", "Mean deadlock ratio", "Mean stall ratio"]
    cmaps = ["YlGn", "YlOrRd", "YlOrRd"]
    for ax, matrix, title, cmap in zip(axes, matrices, titles, cmaps):
        if "ratio" in title:
            matrix_plot = 100.0 * matrix
            label = "%"
        else:
            matrix_plot = matrix
            label = "orders"
        image = ax.imshow(matrix_plot, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(ADMISSIONS)), [ADMISSION_LABELS[x] for x in ADMISSIONS], rotation=35, ha="right")
        ax.set_yticks(range(len(POLICIES)), [POLICY_SHORT[x] for x in POLICIES])
        ax.set_title(title)
        ax.grid(False)
        for i in range(matrix_plot.shape[0]):
            for j in range(matrix_plot.shape[1]):
                value = matrix_plot[i, j]
                threshold = (float(np.nanmax(matrix_plot)) + float(np.nanmin(matrix_plot))) / 2.0
                color = "white" if (value > threshold and "deadlock" not in title and "stall" not in title) else "black"
                ax.text(j, i, f"{value:.1f}" if "orders" in label else f"{value:.1f}", ha="center", va="center", fontsize=7, color=color)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=label)
    fig.suptitle(f"{load.title()}-load factorial: policy × station-admission response surface", y=1.02, fontsize=12)
    fig.subplots_adjust(wspace=0.38, top=0.84, bottom=0.25, left=0.07, right=0.98)
    save_figure(fig, output, "fig01_policy_admission_heatmaps")


def plot_admission_bars(runs: dict[str, list[dict[str, Any]]], output: Path, load: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12.2, 10.0), sharex=True)
    x = np.arange(len(POLICIES))
    width = 0.19
    for ax, metric, title, scale in zip(
        axes,
        ("completed_orders", "deadlock_ratio_mean", "stall_ratio_mean"),
        ("Completed orders (mean ± 95% CI)", "Mean deadlock ratio (%)", "Mean stall ratio (%)"),
        (1.0, 100.0, 100.0),
    ):
        for j, admission in enumerate(ADMISSIONS):
            means = []
            lows = []
            highs = []
            for policy in POLICIES:
                m, lo, hi = mean_ci(row_values(runs, arm_key(policy, admission), metric))
                means.append(scale * m)
                lows.append(scale * (m - lo))
                highs.append(scale * (hi - m))
            ax.bar(
                x + (j - 1.5) * width,
                means,
                width,
                yerr=np.asarray([lows, highs]),
                capsize=2,
                color=COLORS[admission],
                alpha=0.88,
                label=ADMISSION_LABELS[admission],
                error_kw={"elinewidth": 0.7, "capthick": 0.7},
            )
        ax.set_ylabel(title)
        ax.grid(axis="x", visible=False)
        if metric != "completed_orders":
            ax.set_ylim(bottom=0)
    axes[-1].set_xticks(x, [POLICY_SHORT[p] for p in POLICIES], rotation=25, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(f"{load.title()}-load factorial outcomes across all 32 arms", y=1.035, fontsize=12)
    fig.subplots_adjust(top=0.91, hspace=0.20, bottom=0.12, left=0.08, right=0.98)
    save_figure(fig, output, "fig02_all_arm_bars")


def plot_physical_seed_lines(runs: dict[str, list[dict[str, Any]]], output: Path, load: str) -> None:
    key_policies = ("s1_j1", "s0_j0", "s1_dynamic_j", "greedy", "hungarian")
    labels = {
        "s1_j1": "S1 + J1",
        "s0_j0": "S0 + J0 (Phase C)",
        "s1_dynamic_j": "S1 + Dynamic-J",
        "greedy": "Greedy",
        "hungarian": "Hungarian",
    }
    colors = {"s1_j1": "#D55E00", "s0_j0": "#0072B2", "s1_dynamic_j": "#E69F00", "greedy": "#009E73", "hungarian": "#7F7F7F"}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True)
    seeds = [int(row["seed"]) for row in runs[arm_key(key_policies[0], "physical_only")]]
    for ax, metric, title, scale in zip(
        axes,
        ("completed_orders", "deadlock_ratio_mean"),
        ("Physical-only completed orders by seed", "Physical-only mean deadlock by seed"),
        (1.0, 100.0),
    ):
        for policy in key_policies:
            values = np.asarray(row_values(runs, arm_key(policy, "physical_only"), metric)) * scale
            ax.plot(seeds, values, marker="o", linewidth=1.7 if policy == "s1_j1" else 1.15, markersize=4.8, label=labels[policy], color=colors[policy], alpha=0.92)
        ax.set_title(title)
        ax.set_ylabel("orders" if metric == "completed_orders" else "%")
        ax.set_xticks(seeds)
        ax.grid(axis="x", visible=False)
        if metric != "completed_orders":
            ax.set_ylim(bottom=0)
    handles, labels_out = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"{load.title()}-load physical-only condition is seed-bifurcated", y=1.045, fontsize=12)
    fig.subplots_adjust(top=0.87, hspace=0.28, bottom=0.10, left=0.08, right=0.98)
    save_figure(fig, output, "fig03_physical_only_seed_lines")


def contrast_rows(summary: dict[str, Any], names: list[tuple[str, str]]) -> list[dict[str, Any]]:
    result = []
    for category, name in names:
        row = summary["comparisons"][category][name]
        result.append({"label": name, **row})
    return result


def plot_seed_deltas(runs: dict[str, list[dict[str, Any]]], output: Path, load: str) -> None:
    contrasts = [
        ("S1+J1 − Greedy", "s1_j1", "greedy"),
        ("S1+J1 − Phase C", "s1_j1", "s0_j0"),
        ("S1+J1 − S1+Dynamic-J", "s1_j1", "s1_dynamic_j"),
        ("S1+J1 − Hungarian", "s1_j1", "hungarian"),
    ]
    seeds = [int(row["seed"]) for row in runs[arm_key(contrasts[0][1], "physical_only")]]
    matrix = np.asarray(
        [
            [
                row_values(runs, arm_key(left, "physical_only"), "completed_orders")[i]
                - row_values(runs, arm_key(right, "physical_only"), "completed_orders")[i]
                for i in range(len(seeds))
            ]
            for _, left, right in contrasts
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(11.8, 4.2))
    limit = max(1.0, float(np.max(np.abs(matrix))))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(seeds)), [str(s) for s in seeds])
    ax.set_yticks(range(len(contrasts)), [x[0] for x in contrasts])
    ax.set_title(f"{load.title()}-load physical-only paired completed-order deltas by seed")
    ax.set_xlabel("Seed")
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if abs(value) > 0.62 * limit else "black"
            ax.text(j, i, f"{value:+.0f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="orders")
    fig.text(0.5, 0.01, "Green = S1+J1 completes more; red = fewer. Large positive cells are collapse avoidance, not uniform speedup.", ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.19, top=0.87)
    save_figure(fig, output, "fig04_physical_only_seed_delta_heatmap")


def plot_contrast_forest(summary: dict[str, Any], output: Path, load: str) -> None:
    selected = [
        ("S1 effect at J0", "s1_effect_within_same_j_and_admission", "s1_minus_s0__j0__physical_only"),
        ("J1 effect at S1", "context_scheduler_effect_within_same_s_and_admission", "j1_minus_j0__s1__physical_only"),
        ("Dynamic-J − J1 at S1", "context_scheduler_effect_within_same_s_and_admission", "dynamic_j_minus_j1__s1__physical_only"),
        ("S×J1 interaction", "s_by_j_interactions", "s_by_j1_interaction__physical_only"),
        ("S1+J1 − Greedy", "model_vs_external_same_admission", "s1_j1_minus_greedy__physical_only"),
        ("S1+J1 − Hungarian", "model_vs_external_same_admission", "s1_j1_minus_hungarian__physical_only"),
    ]
    rows = []
    for label, category, name in selected:
        value = summary["comparisons"][category][name]
        mean = float(value["completed_orders_contrast_mean"])
        lo, hi = [float(x) for x in value["completed_orders_contrast_ci95"]]
        rows.append((label, mean, lo, hi))
    fig, ax = plt.subplots(figsize=(10.7, 5.0))
    y = np.arange(len(rows))
    for i, (label, mean, lo, hi) in enumerate(rows):
        color = "#009E73" if mean >= 0 else "#D55E00"
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o", color=color, capsize=3, markersize=5, linewidth=1.4)
        ax.text(hi + 8 if hi >= 0 else hi - 8, i, f"{mean:+.1f} [{lo:+.1f}, {hi:+.1f}]", va="center", ha="left" if hi >= 0 else "right", fontsize=7.5)
    ax.axvline(0, color="#444444", linewidth=0.9)
    ax.set_yticks(y, [x[0] for x in rows])
    ax.set_xlabel("Completed-order contrast (orders; paired 95% CI)")
    ax.set_title(f"{load.title()}-load key physical-only paired contrasts")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(r[2] for r in rows) - 55, max(r[3] for r in rows) + 100)
    fig.subplots_adjust(left=0.24, right=0.96, bottom=0.14, top=0.88)
    save_figure(fig, output, "fig05_physical_only_contrast_forest")


def plot_admission_effects(summary: dict[str, Any], output: Path, load: str) -> None:
    policies = ("s1_j1", "s0_j0", "s1_dynamic_j", "greedy", "hungarian")
    effects = ("physical_minus_committed", "eta_minus_committed", "fifo_minus_committed")
    labels = {"physical_minus_committed": "Physical-only − V1", "eta_minus_committed": "ETA V3 − V1", "fifo_minus_committed": "FIFO V2 − V1"}
    vals = np.zeros((len(policies), len(effects)))
    cis = np.zeros((len(policies), len(effects), 2))
    for i, policy in enumerate(policies):
        for j, effect in enumerate(effects):
            name = f"{effect}__{policy}"
            row = summary["comparisons"]["admission_effect_within_same_policy"][name]
            vals[i, j] = row["completed_orders_contrast_mean"]
            cis[i, j] = row["completed_orders_contrast_ci95"]
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    x = np.arange(len(policies))
    width = 0.23
    for j, effect in enumerate(effects):
        means = vals[:, j]
        lower = means - cis[:, j, 0]
        upper = cis[:, j, 1] - means
        ax.bar(x + (j - 1) * width, means, width, yerr=np.asarray([lower, upper]), capsize=2.5, color=["#D55E00", "#009E73", "#CC79A7"][j], alpha=0.86, label=labels[effect], error_kw={"elinewidth": 0.7, "capthick": 0.7})
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(x, [POLICY_SHORT[p] for p in policies], rotation=20, ha="right")
    ax.set_ylabel("Completed-order delta (orders; paired 95% CI)")
    ax.set_title(f"{load.title()}-load admission effects relative to committed V1")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(top=0.83, bottom=0.16, left=0.09, right=0.98)
    save_figure(fig, output, "fig06_admission_effects")


def trace_series(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    trace = (payload.get("station_admission_audit") or {}).get("trace") or []
    ticks = []
    max_occ = []
    max_committed = []
    max_excess = []
    station_full = []
    for record in trace:
        stations = record.get("stations") or []
        if not stations:
            continue
        ticks.append(float(record["tick"]))
        max_occ.append(max(float(s.get("occupancy", 0)) for s in stations))
        max_committed.append(max(float(s.get("committed_load", 0)) for s in stations))
        max_excess.append(max(float(s.get("committed_excess", 0)) for s in stations))
        station_full.append(sum(float(s.get("occupancy", 0)) >= float(s.get("capacity", 7)) for s in stations))
    return {"tick": np.asarray(ticks), "max_occupancy": np.asarray(max_occ), "max_committed": np.asarray(max_committed), "max_excess": np.asarray(max_excess), "full_station_count": np.asarray(station_full)}


def plot_station_traces(root: Path, output: Path, load: str) -> None:
    cases = TRACE_CASES[load]
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.5), sharex=True)
    for title, arm, seed, color in cases:
        payload = json.loads((root / "per_arm" / arm / f"{load}_seed{seed}.json").read_text(encoding="utf-8"))
        series = trace_series(payload)
        axes[0].plot(series["tick"], series["max_occupancy"], linewidth=1.2, color=color, label=title)
        axes[1].plot(series["tick"], series["max_committed"], linewidth=1.2, color=color, label=title)
    axes[0].axhline(7, color="#444444", linestyle="--", linewidth=0.9, label="Physical capacity = 7")
    axes[0].set_ylabel("Max station occupancy")
    axes[1].set_ylabel("Max committed load")
    axes[1].set_xlabel("Simulation tick")
    axes[0].set_title(f"{load.title()}-load representative traces: occupancy remains physical")
    axes[1].set_title("In-transit committed load can grow far beyond physical capacity")
    axes[0].legend(frameon=False, ncol=2, fontsize=7.5, loc="upper left")
    axes[1].legend(frameon=False, ncol=2, fontsize=7.5, loc="upper left")
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
    fig.subplots_adjust(hspace=0.28, top=0.88, bottom=0.11, left=0.08, right=0.98)
    save_figure(fig, output, "fig07_station_trace_mechanism")


def plot_distribution(runs: dict[str, list[dict[str, Any]]], output: Path, load: str) -> None:
    arms = ["s1_j1", "s0_j0", "s1_dynamic_j", "greedy", "hungarian"]
    admissions = ("physical_only", "committed_v1", "eta_v3", "fifo_v2")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    data_orders = [row_values(runs, arm_key(policy, admission), "completed_orders") for policy in arms for admission in admissions]
    positions = np.arange(len(data_orders))
    bp = axes[0].boxplot(data_orders, positions=positions, widths=0.62, patch_artist=True, showfliers=True, flierprops={"markersize": 2.5})
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[admissions[i % len(admissions)]])
        patch.set_alpha(0.72)
    axes[0].set_xticks([i * 4 + 1.5 for i in range(len(arms))], [POLICY_SHORT[p] for p in arms], rotation=22, ha="right")
    axes[0].set_ylabel("Completed orders")
    axes[0].set_title("Seed distribution of completed orders")
    for i in range(1, len(arms)):
        axes[0].axvline(i * 4 - 0.5, color="#BBBBBB", linewidth=0.7)
    # High retains its historical <400 definition. Low/mid use a per-seed
    # controlled-admission ceiling because their manifest totals vary widely.
    collapse = np.zeros((len(arms), len(admissions)))
    ceiling = empirical_controlled_ceiling(runs)
    for i, policy in enumerate(arms):
        for j, admission in enumerate(admissions):
            rows = runs[arm_key(policy, admission)]
            if load == "high":
                collapsed = [float(row["completed_orders"]) < 400.0 for row in rows]
            else:
                collapsed = [
                    float(row["completed_orders"]) / ceiling[int(row["seed"])]
                    < COLLAPSE_EFFICIENCY_THRESHOLD
                    for row in rows
                ]
            collapse[i, j] = 100.0 * sum(collapsed) / len(collapsed)
    x = np.arange(len(arms))
    width = 0.19
    for j, admission in enumerate(admissions):
        axes[1].bar(x + (j - 1.5) * width, collapse[:, j], width, color=COLORS[admission], label=ADMISSION_LABELS[admission])
    axes[1].set_xticks(x, [POLICY_SHORT[p] for p in arms], rotation=22, ha="right")
    if load == "high":
        axes[1].set_ylabel("Seeds below 400 orders (%)")
        axes[1].set_title("Collapse probability (threshold = 400 completed orders)")
    else:
        axes[1].set_ylabel("Collapse seeds (%)")
        axes[1].set_title("Collapse probability (completion efficiency < 80%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False, ncol=2, fontsize=7.5)
    axes[0].legend([bp["boxes"][j] for j in range(4)], [ADMISSION_LABELS[a] for a in admissions], frameon=False, ncol=2, fontsize=7.5)
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"{load.title()}-load distributions and collapse probability", y=1.02, fontsize=12)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.07, right=0.98, wspace=0.25)
    save_figure(fig, output, "fig08_distribution_and_collapse_rate")


def write_csvs(summary: dict[str, Any], runs: dict[str, list[dict[str, Any]]], output: Path) -> None:
    with (output / "seed_level_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["policy", "admission", "seed", "completed_orders", "deadlock_ratio_mean", "stall_ratio_mean", "open_order_count", "pending_order_count", "capacity_rejections", "over_capacity_grants", "max_committed_load", "max_occupancy"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for policy in POLICIES:
            for admission in ADMISSIONS:
                for row in runs[arm_key(policy, admission)]:
                    writer.writerow({"policy": policy, "admission": admission, **{field: row[field] for field in fields[2:]}})
    with (output / "selected_contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["category", "name", "mean", "ci95_low", "ci95_high", "relative_percent", "wins", "ties", "losses", "deadlock_delta", "stall_delta"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for category, rows in summary["comparisons"].items():
            for name, row in rows.items():
                writer.writerow({"category": category, "name": name, "mean": row.get("completed_orders_contrast_mean"), "ci95_low": (row.get("completed_orders_contrast_ci95") or [None, None])[0], "ci95_high": (row.get("completed_orders_contrast_ci95") or [None, None])[1], "relative_percent": row.get("completed_orders_relative_delta_percent"), "wins": row.get("throughput_wins_ties_losses", {}).get("wins"), "ties": row.get("throughput_wins_ties_losses", {}).get("ties"), "losses": row.get("throughput_wins_ties_losses", {}).get("losses"), "deadlock_delta": row.get("deadlock_ratio_contrast_mean"), "stall_delta": row.get("stall_ratio_contrast_mean")})


def write_figure_readme(output: Path, load: str) -> None:
    collapse_description = (
        "fraction below 400 completed orders"
        if load == "high"
        else "fraction with completion efficiency below 80% of the per-seed controlled-admission ceiling"
    )
    text = f"""# {load.title()}-load S/J × station-admission figures

Source: `station_admission_sj_factorial_{load}.json`, plus the 320 audited
per-arm JSON results (seeds 551–560, 1500 ticks). All arms for a seed replay
the same frozen order-arrival manifest. Ratios are plotted as percentages.

| Figure | Meaning |
|---|---|
| `fig00_overview` | One-page contact sheet of the four most diagnostic views. |
| `fig01_policy_admission_heatmaps` | Complete policy × admission response surface. |
| `fig02_all_arm_bars` | All 32 arm means with paired-seed 95% CIs. |
| `fig03_physical_only_seed_lines` | Seed-by-seed behavior under the true physical-only restoration condition. |
| `fig04_physical_only_seed_delta_heatmap` | Exact paired order deltas for S1+J1 against key baselines. |
| `fig05_physical_only_contrast_forest` | Main paired contrasts and confidence intervals. |
| `fig06_admission_effects` | Physical-only, ETA V3, and FIFO V2 relative to committed V1. |
| `fig07_station_trace_mechanism` | Representative occupancy/commitment traces explaining station-lock collapse. |
| `fig08_distribution_and_collapse_rate` | Seed distributions and {collapse_description}. |

PNG and PDF versions are provided. `seed_level_metrics.csv` and
`selected_contrasts.csv` contain the plotted numerical data.
"""
    (output / "README.md").write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = (args.output_dir or (args.root / f"figures_{args.load}")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    apply_style()
    summary, runs = load_data(root, args.load)
    plot_heatmaps(summary, runs, output, args.load)
    plot_admission_bars(runs, output, args.load)
    plot_physical_seed_lines(runs, output, args.load)
    plot_seed_deltas(runs, output, args.load)
    plot_contrast_forest(summary, output, args.load)
    plot_admission_effects(summary, output, args.load)
    plot_station_traces(root, output, args.load)
    plot_distribution(runs, output, args.load)
    make_contact_sheet(output, args.load)
    write_csvs(summary, runs, output)
    write_figure_readme(output, args.load)
    print(f"wrote {args.load}-load factorial figures to {output}")


if __name__ == "__main__":
    main()
