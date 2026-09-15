"""Plot the Phase-C behavior-continuation H=10 diagnostics.

The figures distinguish three questions:

1. What behavior continuation changes relative to an isolated target.
2. Whether the frozen model predicts the behavior-continuation endpoint.
3. Whether station signals describe congestion state or actually rank robot
   actions inside one fixed (order, pod, station) context.

The old isolated H=10 report is shown only as an unpaired contextual
reference.  It uses different states/seeds and must not be read as a causal
continuation-mode ablation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.stats import ConstantInputWarning, rankdata, spearmanr


ROOT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "behavior_h10_531_540_v1"
)
DEFAULT_REPORT = ROOT / "phase_c_behavior_h10_validation.json"
DEFAULT_ISOLATED = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "system_dynamics_h100_mid_seed470_v1/"
    "system_dynamics_horizon_eval.json"
)
DEFAULT_OUTPUT = ROOT / "figures"

SCHEMA_VERSION = "phase_c_behavior_h10_plot_metrics_v1"
LOADS = ("low", "mid", "high")
LOAD_COLORS = {
    "low": "#4C78A8",
    "mid": "#ECA82C",
    "high": "#D95F59",
}
PRED_COLOR = "#D55E00"
TRUE_COLOR = "#2B2D30"

SYSTEM_CHANNELS = (
    "total_wait_time",
    "average_excess_delay",
    "station_queue_delta",
    "station_load_imbalance",
    "bottleneck_CVaR",
    "completed_orders_delta",
    "deadlock_or_severe_congestion_risk",
)
DISPLAY = {
    "total_wait_time": "Stall / wait fraction",
    "average_excess_delay": "Average excess delay (ticks)",
    "station_queue_delta": "Station queue delta",
    "station_load_imbalance": "Station load imbalance",
    "bottleneck_CVaR": "Bottleneck CVaR",
    "completed_orders_delta": "Completed orders at step 10",
    "deadlock_or_severe_congestion_risk": "Severe-congestion risk",
    "completed_orders_cumulative": "Completed orders, cumulative H=10",
    "station_queue_by_step": "Station queue occupancy",
    "station_load_by_step": "Station assigned load",
    "node_congestion_by_step": "Node congestion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--isolated-report", type=Path, default=DEFAULT_ISOLATED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def apply_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def finite_spearman(prediction: Sequence[float], target: Sequence[float]) -> float | None:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if (
        prediction.size < 3
        or np.ptp(prediction) <= 1e-12
        or np.ptp(target) <= 1e-12
    ):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        value = float(spearmanr(prediction, target).statistic)
    return value if math.isfinite(value) else None


def rank_auc(target: Sequence[bool], score: Sequence[float]) -> float | None:
    target = np.asarray(target, dtype=bool)
    score = np.asarray(score, dtype=float)
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(score, method="average")
    return float(
        (ranks[target].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def mean_ci(values: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean()) if values.size else 0.0
    if values.size < 2:
        return mean, 0.0
    half = 1.96 * float(values.std(ddof=1)) / math.sqrt(values.size)
    return mean, half


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  wrote {output_dir / f'{stem}.png'}")


def runs_for(report: Mapping[str, Any], load: str) -> list[dict]:
    return [row for row in report["runs"] if row["load"] == load]


def endpoint_records(report: Mapping[str, Any], load: str) -> list[dict]:
    return [
        record
        for run in runs_for(report, load)
        for record in run["h10_endpoint_records"]
    ]


def run_metric(
    run: Mapping[str, Any],
    key: str,
    step: int,
    channel: str | None = None,
) -> Mapping[str, Any]:
    section = run["metrics"][key][str(step)]
    return section["channels"][channel] if channel is not None else section


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=1.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        linespacing=1.25,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "linewidth": 1.3, "color": "#555555"},
    )


def plot_protocol_semantics(output_dir: Path, dpi: int) -> dict:
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.01, 0.87, "Frozen model forecast", weight="bold", va="center")
    draw_box(ax, (0.18, 0.79), 0.15, 0.15, "Current decision state\nencode -> z(t)",
             facecolor="#E9F2FA", edgecolor="#4C78A8")
    draw_box(ax, (0.42, 0.79), 0.20, 0.15, "Force candidate a=(c,r)\nlatent transition for H=10",
             facecolor="#E9F2FA", edgecolor="#4C78A8")
    draw_box(ax, (0.72, 0.79), 0.20, 0.15, "Decode z(t+1 ... t+10)\nsystem / node / station",
             facecolor="#E9F2FA", edgecolor="#4C78A8")
    arrow(ax, (0.33, 0.865), (0.42, 0.865))
    arrow(ax, (0.62, 0.865), (0.72, 0.865))

    ax.text(0.01, 0.54, "Isolated simulator target", weight="bold", va="center")
    draw_box(ax, (0.18, 0.45), 0.15, 0.17, "Clone same state\nforce candidate",
             facecolor="#F1F3F4", edgecolor="#777777")
    draw_box(ax, (0.42, 0.45), 0.20, 0.17, "Physics only for H ticks\nno future orders\nno continuation scheduler",
             facecolor="#F1F3F4", edgecolor="#777777")
    draw_box(ax, (0.72, 0.45), 0.20, 0.17, "Isolated labels\nY_iso(t+1 ... t+10)",
             facecolor="#F1F3F4", edgecolor="#777777")
    arrow(ax, (0.33, 0.535), (0.42, 0.535))
    arrow(ax, (0.62, 0.535), (0.72, 0.535))

    ax.text(0.01, 0.20, "Behavior simulator target", weight="bold", va="center")
    draw_box(ax, (0.18, 0.10), 0.15, 0.20, "Clone same state\nforce candidate",
             facecolor="#FDF1DF", edgecolor="#D38B25")
    draw_box(ax, (0.42, 0.10), 0.20, 0.20,
             "At step 0: assign remaining work\nLater steps: generate orders\n+ behavior scheduler + physics",
             facecolor="#FDF1DF", edgecolor="#D38B25")
    draw_box(ax, (0.72, 0.10), 0.20, 0.20, "Closed-loop labels\nY_beh(t+1 ... t+10)",
             facecolor="#FDF1DF", edgecolor="#D38B25")
    arrow(ax, (0.33, 0.20), (0.42, 0.20))
    arrow(ax, (0.62, 0.20), (0.72, 0.20))

    fig.suptitle(
        "What changes between isolated and behavior continuation",
        fontsize=13,
        weight="bold",
    )
    save_figure(fig, output_dir, "figure_01_protocol_semantics", dpi)
    return {
        "isolated": {
            "future_orders": False,
            "continuation_scheduler": False,
            "physics": True,
        },
        "behavior": {
            "future_orders": True,
            "continuation_scheduler": True,
            "physics": True,
        },
        "model_forecast_unchanged": True,
    }


def plot_endpoint_means(report: Mapping[str, Any], output_dir: Path, dpi: int) -> dict:
    specs = [
        ("system", "total_wait_time"),
        ("system", "average_excess_delay"),
        ("system", "station_queue_delta"),
        ("system", "station_load_imbalance"),
        ("system", "bottleneck_CVaR"),
        ("system", "deadlock_or_severe_congestion_risk"),
        ("metric", "station_queue_by_step"),
        ("metric", "completed_orders_cumulative"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.8))
    values_out: dict[str, Any] = {}
    x = np.arange(len(LOADS), dtype=float)
    width = 0.34
    for ax, (kind, key) in zip(axes.flat, specs):
        target_means, target_errs = [], []
        pred_means, pred_errs = [], []
        values_out[key] = {}
        for load in LOADS:
            rows = runs_for(report, load)
            if kind == "system":
                metrics = [run_metric(row, "system_by_step", 10, key) for row in rows]
            else:
                metrics = [run_metric(row, key, 10) for row in rows]
            pred, pred_ci = mean_ci([metric["pred_mean"] for metric in metrics])
            target, target_ci = mean_ci([metric["target_mean"] for metric in metrics])
            pred_means.append(pred)
            pred_errs.append(pred_ci)
            target_means.append(target)
            target_errs.append(target_ci)
            values_out[key][load] = {
                "prediction_mean": pred,
                "prediction_ci95_half": pred_ci,
                "target_mean": target,
                "target_ci95_half": target_ci,
            }
        ax.bar(
            x - width / 2,
            target_means,
            width,
            yerr=target_errs,
            capsize=2.5,
            color=TRUE_COLOR,
            alpha=0.86,
            label="Simulator target",
        )
        ax.bar(
            x + width / 2,
            pred_means,
            width,
            yerr=pred_errs,
            capsize=2.5,
            color=PRED_COLOR,
            alpha=0.86,
            label="Model prediction",
        )
        ax.set_xticks(x, LOADS)
        ax.set_title(DISPLAY.get(key, key))
        ax.axhline(0.0, color="#777777", linewidth=0.7)
    axes.flat[0].legend(frameon=False, loc="best")
    fig.suptitle(
        "Behavior continuation at H=10: predicted versus observed level\n"
        "Bars are equal-seed means; error bars are 95% normal CIs across 10 seeds",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_dir, "figure_02_behavior_h10_true_vs_predicted", dpi)
    return values_out


def plot_horizon_trajectories(
    report: Mapping[str, Any], output_dir: Path, dpi: int
) -> dict:
    specs = [
        ("system", "average_excess_delay"),
        ("system", "station_load_imbalance"),
        ("system", "deadlock_or_severe_congestion_risk"),
        ("metric", "station_queue_by_step"),
        ("metric", "station_load_by_step"),
        ("metric", "completed_orders_cumulative"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.3, 7.2), sharex=True)
    output: dict[str, Any] = {}
    steps = np.arange(1, 11)
    for ax, (kind, key) in zip(axes.flat, specs):
        output[key] = {}
        for load in LOADS:
            rows = runs_for(report, load)
            pred_values = []
            target_values = []
            for step in steps:
                if kind == "system":
                    metrics = [
                        run_metric(row, "system_by_step", int(step), key)
                        for row in rows
                    ]
                else:
                    metrics = [run_metric(row, key, int(step)) for row in rows]
                pred_values.append(np.mean([metric["pred_mean"] for metric in metrics]))
                target_values.append(np.mean([metric["target_mean"] for metric in metrics]))
            color = LOAD_COLORS[load]
            ax.plot(
                steps,
                target_values,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=3,
                label=f"{load} target",
            )
            ax.plot(
                steps,
                pred_values,
                color=color,
                linewidth=1.7,
                linestyle="--",
                marker="s",
                markersize=2.7,
                label=f"{load} prediction",
            )
            output[key][load] = {
                "steps": steps.tolist(),
                "prediction_mean": [float(value) for value in pred_values],
                "target_mean": [float(value) for value in target_values],
            }
        ax.set_title(DISPLAY.get(key, key))
        ax.set_xlabel("Rollout step")
        ax.set_xticks([1, 2, 4, 6, 8, 10])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Receding-horizon local forecast: target and prediction across H=1...10\n"
        "Solid = behavior simulator; dashed = frozen model",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    save_figure(fig, output_dir, "figure_03_behavior_horizon_true_vs_predicted", dpi)
    return output


def pooled_system_metrics(
    report: Mapping[str, Any], load: str, channel: str
) -> dict:
    rows = endpoint_records(report, load)
    prediction = np.asarray(
        [row["system_prediction"][channel] for row in rows], dtype=float
    )
    target = np.asarray([row["system_target"][channel] for row in rows], dtype=float)
    error = prediction - target
    return {
        "n": int(prediction.size),
        "mae": float(np.mean(np.abs(error))),
        "target_std": float(np.std(target)),
        "normalized_mae": (
            float(np.mean(np.abs(error)) / np.std(target))
            if np.std(target) > 1e-12 else None
        ),
        "spearman": finite_spearman(prediction, target),
    }


def plot_isolated_reference(
    report: Mapping[str, Any],
    isolated: Mapping[str, Any],
    output_dir: Path,
    dpi: int,
) -> dict:
    behavior = {
        channel: pooled_system_metrics(report, "mid", channel)
        for channel in SYSTEM_CHANNELS
    }
    isolated_metrics = isolated["metrics_by_horizon"]["10"]["channels"]
    labels = [DISPLAY[channel].replace(" (ticks)", "") for channel in SYSTEM_CHANNELS]
    y = np.arange(len(labels))
    width = 0.36
    behavior_sp = [behavior[channel]["spearman"] or 0.0 for channel in SYSTEM_CHANNELS]
    isolated_sp = [isolated_metrics[channel]["spearman"] for channel in SYSTEM_CHANNELS]
    behavior_nmae = [
        behavior[channel]["normalized_mae"] or 0.0 for channel in SYSTEM_CHANNELS
    ]
    isolated_nmae = [
        isolated_metrics[channel].get("mae_over_target_std") or 0.0
        for channel in SYSTEM_CHANNELS
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    axes[0].barh(y - width / 2, isolated_sp, width, color="#7A7A7A", label="Isolated reference")
    axes[0].barh(y + width / 2, behavior_sp, width, color="#D38B25", label="Behavior continuation")
    axes[0].axvline(0.0, color="#555555", linewidth=0.8)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Tie-aware Spearman")
    axes[0].set_title("Endpoint ranking")
    axes[0].legend(frameon=False)

    axes[1].barh(y - width / 2, isolated_nmae, width, color="#7A7A7A")
    axes[1].barh(y + width / 2, behavior_nmae, width, color="#D38B25")
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("MAE / target standard deviation")
    axes[1].set_title("Scale-normalized error (lower is better)")
    fig.suptitle(
        "Behavior H=10 versus the earlier isolated H=10 report\n"
        "Context only: different seeds, states, and target semantics; this is not a paired A/B test",
        fontsize=12.5,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, output_dir, "figure_04_behavior_vs_isolated_reference", dpi)
    return {
        "warning": "unpaired contextual reference only",
        "behavior_mid": behavior,
        "isolated_mid_seed470": {
            channel: {
                "mae": isolated_metrics[channel]["mae"],
                "normalized_mae": isolated_metrics[channel].get(
                    "mae_over_target_std"
                ),
                "spearman": isolated_metrics[channel]["spearman"],
            }
            for channel in SYSTEM_CHANNELS
        },
    }


def target_binned_curve(
    prediction: Sequence[float], target: Sequence[float], bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    order = np.argsort(target, kind="stable")
    chunks = [chunk for chunk in np.array_split(order, bins) if chunk.size]
    return (
        np.asarray([target[chunk].mean() for chunk in chunks]),
        np.asarray([prediction[chunk].mean() for chunk in chunks]),
    )


def station_values(record: Mapping[str, Any], field: str, channel: int) -> list[float]:
    matrix = record[field] or []
    return [float(row[channel]) for row in matrix]


def plot_tail_calibration(
    report: Mapping[str, Any], output_dir: Path, dpi: int
) -> dict:
    specs: list[tuple[str, str, Callable[[Mapping[str, Any]], list[float]], float]] = [
        (
            "Station queue occupancy",
            "station_queue",
            lambda row: station_values(row, "station_prediction", 0),
            0.5,
        ),
        (
            "System station-load imbalance",
            "station_load_imbalance",
            lambda row: [float(row["system_prediction"]["station_load_imbalance"])],
            1.0,
        ),
        (
            "Severe-congestion risk",
            "deadlock_or_severe_congestion_risk",
            lambda row: [
                float(row["system_prediction"]["deadlock_or_severe_congestion_risk"])
            ],
            0.8,
        ),
        (
            "Bottleneck CVaR",
            "bottleneck_CVaR",
            lambda row: [float(row["system_prediction"]["bottleneck_CVaR"])],
            0.15,
        ),
    ]
    target_extractors = {
        "station_queue": lambda row: station_values(row, "station_target", 0),
        "station_load_imbalance": lambda row: [
            float(row["system_target"]["station_load_imbalance"])
        ],
        "deadlock_or_severe_congestion_risk": lambda row: [
            float(row["system_target"]["deadlock_or_severe_congestion_risk"])
        ],
        "bottleneck_CVaR": lambda row: [
            float(row["system_target"]["bottleneck_CVaR"])
        ],
    }
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.1))
    output: dict[str, Any] = {}
    for ax, (title, key, pred_fn, threshold) in zip(axes, specs):
        output[key] = {}
        global_min = math.inf
        global_max = -math.inf
        for load in LOADS:
            prediction: list[float] = []
            target: list[float] = []
            for row in endpoint_records(report, load):
                prediction.extend(pred_fn(row))
                target.extend(target_extractors[key](row))
            x, y = target_binned_curve(prediction, target)
            ax.plot(
                x,
                y,
                color=LOAD_COLORS[load],
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=load,
            )
            pred_array = np.asarray(prediction, dtype=float)
            target_array = np.asarray(target, dtype=float)
            tail = target_array >= threshold
            tail_auc = rank_auc(tail, pred_array)
            tail_bias = (
                float(np.mean(pred_array[tail] - target_array[tail]))
                if tail.any() else None
            )
            under_rate = (
                float(np.mean(pred_array[tail] < target_array[tail]))
                if tail.any() else None
            )
            output[key][load] = {
                "threshold": threshold,
                "tail_count": int(tail.sum()),
                "tail_auc": tail_auc,
                "tail_bias": tail_bias,
                "tail_underprediction_rate": under_rate,
                "binned_target_mean": x.tolist(),
                "binned_prediction_mean": y.tolist(),
            }
            global_min = min(global_min, float(x.min()), float(y.min()))
            global_max = max(global_max, float(x.max()), float(y.max()))
        margin = max((global_max - global_min) * 0.05, 0.01)
        ax.plot(
            [global_min - margin, global_max + margin],
            [global_min - margin, global_max + margin],
            color="#444444",
            linestyle=":",
            linewidth=1.2,
            label="perfect calibration",
        )
        ax.set_title(title)
        ax.set_xlabel("Observed target (binned)")
        ax.set_ylabel("Mean prediction")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(
        "Congestion-tail calibration at H=10\n"
        "Curves below the diagonal mean systematic underprediction",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, output_dir, "figure_05_congestion_tail_calibration", dpi)
    return output


ACTION_METRICS = (
    "average_excess_delay",
    "deadlock_or_severe_congestion_risk",
    "bottleneck_CVaR",
    "context_station_queue",
    "context_station_load",
)
ACTION_LABELS = (
    "Excess delay",
    "Severe risk",
    "Bottleneck CVaR",
    "Context station queue",
    "Context station load",
)
ACTION_EPS = {
    "average_excess_delay": 0.1,
    "deadlock_or_severe_congestion_risk": 1e-5,
    "bottleneck_CVaR": 1e-5,
    "context_station_queue": 1e-5,
    "context_station_load": 1e-5,
}


def action_value(record: Mapping[str, Any], key: str, field: str) -> float:
    if key in SYSTEM_CHANNELS:
        return float(record[f"system_{field}"][key])
    station_id = int(record["station_id"])
    matrix = record[f"station_{field}"] or []
    index = station_id - 1
    if not 0 <= index < len(matrix):
        raise ValueError(
            f"station id {station_id} does not index {len(matrix)} station rows"
        )
    channel = 0 if key == "context_station_queue" else 1
    return float(matrix[index][channel])


def within_context_stats(report: Mapping[str, Any], load: str, key: str) -> dict:
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for run in runs_for(report, load):
        for record in run["h10_endpoint_records"]:
            groups[(int(run["seed"]), record["candidate_group_id"])].append(record)
    epsilon = ACTION_EPS[key]
    correct = 0
    pairs = 0
    tied_pairs = 0
    sensitive_groups = 0
    for records in groups.values():
        prediction = np.asarray(
            [action_value(row, key, "prediction") for row in records], dtype=float
        )
        target = np.asarray(
            [action_value(row, key, "target") for row in records], dtype=float
        )
        if np.ptp(target) > epsilon:
            sensitive_groups += 1
        for left, right in itertools.combinations(range(len(records)), 2):
            target_delta = target[left] - target[right]
            if abs(target_delta) <= epsilon:
                tied_pairs += 1
                continue
            prediction_delta = prediction[left] - prediction[right]
            pairs += 1
            correct += int(prediction_delta * target_delta > 0.0)
    return {
        "groups": len(groups),
        "action_sensitive_groups": sensitive_groups,
        "action_sensitive_group_fraction": (
            sensitive_groups / len(groups) if groups else 0.0
        ),
        "informative_pairs": pairs,
        "target_tied_pairs": tied_pairs,
        "pairwise_accuracy": correct / pairs if pairs else None,
    }


def plot_within_context(
    report: Mapping[str, Any], output_dir: Path, dpi: int
) -> dict:
    output = {
        key: {load: within_context_stats(report, load, key) for load in LOADS}
        for key in ACTION_METRICS
    }
    x = np.arange(len(ACTION_METRICS), dtype=float)
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
    for load_index, load in enumerate(LOADS):
        offset = (load_index - 1) * width
        accuracy = [
            output[key][load]["pairwise_accuracy"]
            if output[key][load]["pairwise_accuracy"] is not None else 0.0
            for key in ACTION_METRICS
        ]
        sensitivity = [
            output[key][load]["action_sensitive_group_fraction"]
            for key in ACTION_METRICS
        ]
        axes[0].bar(
            x + offset,
            accuracy,
            width,
            color=LOAD_COLORS[load],
            label=load,
        )
        axes[1].bar(
            x + offset,
            sensitivity,
            width,
            color=LOAD_COLORS[load],
            label=load,
        )
    for ax in axes:
        ax.set_xticks(x, ACTION_LABELS, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
    axes[0].axhline(0.5, color="#555555", linestyle="--", linewidth=1.1)
    axes[0].set_ylabel("Pairwise ordering accuracy")
    axes[0].set_title("When the simulator target differs")
    axes[1].set_ylabel("Fraction of contexts with a non-tied target")
    axes[1].set_title("How often changing robot affects the H=10 target")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(
        "Within one fixed (order, pod, station) context: robot-action signal\n"
        "High station-level state accuracy does not imply robot-ranking signal",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, output_dir, "figure_06_within_context_action_signal", dpi)
    return output


def run_metadata(report: Mapping[str, Any], report_path: Path) -> list[dict]:
    root = report_path.parent / "runs"
    output = []
    report_by_key = {
        (row["load"], int(row["seed"])): row for row in report["runs"]
    }
    for (load, seed), run in sorted(report_by_key.items()):
        run_dir = root / f"{load}_seed{seed}"
        meta_paths = list(run_dir.glob("*_meta.json"))
        if not meta_paths:
            continue
        metadata = read_json(meta_paths[0])
        simulation = metadata["simulation_result"]
        ticks = sorted({
            int(record["decision_tick"])
            for record in run["h10_endpoint_records"]
        })
        output.append({
            "load": load,
            "seed": seed,
            "samples": int(run["samples"]),
            "groups": int(run["candidate_groups"]),
            "completed_orders": int(simulation["completed_orders"]),
            "elapsed_seconds": float(simulation["elapsed_seconds"]),
            "last_sampled_tick": max(ticks) if ticks else None,
        })
    return output


def plot_coverage(
    report: Mapping[str, Any], report_path: Path, output_dir: Path, dpi: int
) -> dict:
    rows = run_metadata(report, report_path)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))
    output: dict[str, Any] = {"runs": rows, "correlations": {}}
    for load in LOADS:
        subset = [row for row in rows if row["load"] == load]
        groups = np.asarray([row["groups"] for row in subset], dtype=float)
        completed = np.asarray(
            [row["completed_orders"] for row in subset], dtype=float
        )
        sp = finite_spearman(groups, completed)
        output["correlations"][load] = {
            "groups_vs_completed_orders_spearman": sp,
        }
        axes[0].scatter(
            groups,
            completed,
            color=LOAD_COLORS[load],
            s=48,
            alpha=0.88,
            label=f"{load}, rho={sp:.2f}" if sp is not None else load,
        )
        for row in subset:
            axes[0].annotate(
                str(row["seed"]),
                (row["groups"], row["completed_orders"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
    axes[0].set_xlabel("Collected candidate groups")
    axes[0].set_ylabel("Completed orders in the behavior run")
    axes[0].set_title("Collection availability is outcome-dependent")
    axes[0].legend(frameon=False)

    seeds = sorted({row["seed"] for row in rows})
    x = np.arange(len(seeds), dtype=float)
    width = 0.24
    for index, load in enumerate(LOADS):
        by_seed = {
            row["seed"]: row["last_sampled_tick"]
            for row in rows if row["load"] == load
        }
        axes[1].bar(
            x + (index - 1) * width,
            [by_seed.get(seed, 0) for seed in seeds],
            width,
            color=LOAD_COLORS[load],
            label=load,
        )
    axes[1].set_xticks(x, seeds, rotation=45)
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Last decision tick with a collected context")
    axes[1].set_title("Poor runs often stop yielding eligible contexts early")
    axes[1].legend(frameon=False)
    fig.suptitle(
        "Behavior H=10 coverage audit",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_dir, "figure_07_sampling_coverage", dpi)
    return output


def main() -> None:
    args = parse_args()
    apply_style()
    report = read_json(args.report)
    isolated = read_json(args.isolated_report)
    if int(report.get("completed_runs", 0)) != 30 or bool(report.get("partial")):
        raise RuntimeError("behavior report is incomplete")

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "source_report": args.report.as_posix(),
        "isolated_reference": args.isolated_report.as_posix(),
        "figure_01_protocol_semantics": plot_protocol_semantics(
            args.output_dir, args.dpi
        ),
        "figure_02_endpoint_means": plot_endpoint_means(
            report, args.output_dir, args.dpi
        ),
        "figure_03_horizon_trajectories": plot_horizon_trajectories(
            report, args.output_dir, args.dpi
        ),
        "figure_04_isolated_reference": plot_isolated_reference(
            report, isolated, args.output_dir, args.dpi
        ),
        "figure_05_tail_calibration": plot_tail_calibration(
            report, args.output_dir, args.dpi
        ),
        "figure_06_within_context": plot_within_context(
            report, args.output_dir, args.dpi
        ),
        "figure_07_sampling_coverage": plot_coverage(
            report, args.report, args.output_dir, args.dpi
        ),
    }
    metrics_path = args.output_dir / "behavior_h10_plot_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"  wrote {metrics_path}")


if __name__ == "__main__":
    main()
