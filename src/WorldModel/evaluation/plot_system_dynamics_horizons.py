"""Plot audited long-horizon system-dynamics diagnostics.

The script keeps the frozen World Model and existing validation reports
unchanged.  It builds (or reuses) a compact cache containing the full H=100
system prediction/target trajectories, applies tie-aware rank metrics, and
produces publication-ready PNG/PDF figures plus the exact plotted values.

All channels except completed_orders_delta are evaluated at the endpoint
``t + H``.  completed_orders_delta is a per-tick event in the collector, so
the throughput figures and rank summaries use its cumulative sum over
``[t + 1, t + H]``.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ConstantInputWarning, rankdata, spearmanr


ROOT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "system_dynamics_h100_mid_seed470_v1"
)
DEFAULT_DATA = ROOT / "data.pt"
DEFAULT_CHECKPOINT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "model_round1_v1/best_regret_world_model.pt"
)
DEFAULT_CACHE = ROOT / "system_dynamics_h100_trajectory_cache.pt"
DEFAULT_OUTPUT = ROOT / "figures"
DEFAULT_REPORT = ROOT / "system_dynamics_horizon_eval.json"

SCHEMA_VERSION = "wm_system_dynamics_horizon_plot_metrics_v1"
CACHE_SCHEMA_VERSION = "wm_system_dynamics_h100_trajectory_cache_v1"
SELECTED_HORIZONS = (10, 20, 50, 80, 100)
SYSTEM_CHANNELS = (
    "total_wait_time",
    "average_excess_delay",
    "station_queue_delta",
    "station_load_imbalance",
    "bottleneck_CVaR",
    "completed_orders_delta",
    "deadlock_or_severe_congestion_risk",
)
DISPLAY_NAMES = {
    "total_wait_time": "Stall / wait fraction",
    "average_excess_delay": "Average excess delay",
    "station_queue_delta": "Station queue delta",
    "station_load_imbalance": "Station load imbalance",
    "bottleneck_CVaR": "Bottleneck CVaR",
    "completed_orders_delta": "Completed orders (cumulative)",
    "deadlock_or_severe_congestion_risk": "Severe-congestion risk",
}
COLORS = {
    "total_wait_time": "#4C78A8",
    "average_excess_delay": "#F58518",
    "station_queue_delta": "#72B7B2",
    "station_load_imbalance": "#B279A2",
    "bottleneck_CVaR": "#54A24B",
    "completed_orders_delta": "#ECA82C",
    "deadlock_or_severe_congestion_risk": "#E45756",
    "prediction": "#D55E00",
    "target": "#202124",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-horizon", type=int, default=100)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
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
            "grid.alpha": 0.22,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_cache(
    data_path: Path,
    checkpoint_path: Path,
    cache_path: Path,
    *,
    device_name: str,
    max_horizon: int,
) -> dict:
    from WorldModel.data.dataset import WorldModelDataset
    from WorldModel.evaluation.evaluate import _load_model
    from WorldModel.evaluation.validate_system_dynamics_horizons import (
        _rollout_system,
    )

    device = resolve_device(device_name)
    samples = WorldModelDataset.from_file(str(data_path)).samples
    if not samples:
        raise RuntimeError("system-dynamics dataset is empty")

    model, label_schema = _load_model(str(checkpoint_path))
    model = model.to(device).eval()

    predictions = []
    targets = []
    group_ids = []
    action_types = []
    prefix_errors = []

    with torch.inference_mode():
        for index, sample in enumerate(samples):
            target = sample["future_system_labels"]
            mask = sample.get("future_mask")
            if target.shape[0] < max_horizon or target.shape[1] != 7:
                raise RuntimeError(
                    f"sample[{index}] label shape {tuple(target.shape)} "
                    f"does not cover H={max_horizon}"
                )
            if mask is not None and not bool(
                torch.all(mask[:max_horizon] >= 0.5).item()
            ):
                raise RuntimeError(f"sample[{index}] has an invalid future step")

            prediction, z_start = _rollout_system(
                model, sample, device, max_horizon
            )
            if prediction.shape != (max_horizon, 7):
                raise RuntimeError(
                    f"sample[{index}] prediction shape={tuple(prediction.shape)}"
                )

            if index < 3:
                short_prediction, short_z_start = _rollout_system(
                    model, sample, device, min(10, max_horizon)
                )
                prefix_errors.append(
                    {
                        "sample_index": index,
                        "prediction_max_abs_error": float(
                            (
                                prediction[: short_prediction.shape[0]]
                                - short_prediction
                            )
                            .abs()
                            .max()
                            .item()
                        ),
                        "z_start_max_abs_error": float(
                            (z_start - short_z_start).abs().max().item()
                        ),
                    }
                )

            predictions.append(prediction.float())
            targets.append(target[:max_horizon].detach().cpu().float())
            group_ids.append(str(sample.get("candidate_group_id", "")))
            action_types.append(str(sample.get("action_type", "unknown")))
            if (index + 1) % 100 == 0:
                print(f"  trajectory inference {index + 1}/{len(samples)}")

    max_prefix_error = max(
        (
            max(
                row["prediction_max_abs_error"],
                row["z_start_max_abs_error"],
            )
            for row in prefix_errors
        ),
        default=0.0,
    )
    if max_prefix_error > 1e-6:
        raise RuntimeError(
            "same-start/prediction-prefix audit failed; refusing to plot "
            f"non-deterministic predictions (max error={max_prefix_error:.3e})"
        )

    step_embed = model.transition.step_embed
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "data": str(data_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_label_schema": label_schema,
        "device": str(device),
        "channels": list(SYSTEM_CHANNELS),
        "max_horizon": int(max_horizon),
        "samples": len(samples),
        "groups": len(set(group_ids)),
        "predictions": torch.stack(predictions),
        "targets": torch.stack(targets),
        "group_ids": group_ids,
        "action_types": action_types,
        "audit": {
            "passed": True,
            "all_future_steps_valid": True,
            "prefix_records": prefix_errors,
            "max_prefix_error": max_prefix_error,
            "checkpoint_training_horizon": int(model.rollout_horizon),
            "learned_step_embedding_capacity": int(
                step_embed.num_embeddings
            ),
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"  wrote trajectory cache: {cache_path}")
    return payload


def load_or_build_cache(args: argparse.Namespace) -> dict:
    if args.cache.exists() and not args.rebuild_cache:
        cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    else:
        cache = build_cache(
            args.data,
            args.checkpoint,
            args.cache,
            device_name=args.device,
            max_horizon=args.max_horizon,
        )

    if cache.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported cache schema: {cache.get('schema_version')}")
    if tuple(cache.get("channels", ())) != SYSTEM_CHANNELS:
        raise RuntimeError("trajectory cache channel contract mismatch")
    if not bool((cache.get("audit") or {}).get("passed")):
        raise RuntimeError("trajectory cache audit did not pass")
    if int(cache.get("max_horizon", 0)) < max(SELECTED_HORIZONS):
        raise RuntimeError("trajectory cache does not cover selected horizons")
    return cache


def finite_float(value: float) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def tie_aware_spearman(prediction: np.ndarray, target: np.ndarray) -> Optional[float]:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if prediction.size < 3 or np.ptp(prediction) <= 1e-12 or np.ptp(target) <= 1e-12:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        value = spearmanr(prediction, target).statistic
    return finite_float(value)


def rank_auc(target: np.ndarray, prediction: np.ndarray) -> Optional[float]:
    target = np.asarray(target, dtype=float) >= 0.5
    prediction = np.asarray(prediction, dtype=float)
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(prediction, method="average")
    positive_rank_sum = float(ranks[target].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def metric_bundle(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    error = prediction - target
    return {
        "n": int(prediction.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "spearman_tie_aware": tie_aware_spearman(prediction, target),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction)),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
        "target_unique_values": int(np.unique(target).size),
    }


def group_balanced_series(series: np.ndarray, group_ids: list[str]) -> dict:
    """Summarise a (sample, step) series with equal weight per context."""
    grouped = group_series_matrix(series, group_ids)
    return {
        "group_balanced_mean": grouped.mean(axis=0).tolist(),
        "group_balanced_std": grouped.std(axis=0).tolist(),
        "pooled_mean": series.mean(axis=0).tolist(),
        "pooled_std": series.std(axis=0).tolist(),
        "groups": grouped.shape[0],
    }


def group_series_matrix(series: np.ndarray, group_ids: list[str]) -> np.ndarray:
    """Return one mean trajectory per starting context."""
    group_order = list(dict.fromkeys(group_ids))
    return np.stack(
        [
            series[np.asarray([gid == group for gid in group_ids])].mean(axis=0)
            for group in group_order
        ],
        axis=0,
    )


def build_true_target_summaries(
    cache: dict,
    *,
    data_delay_scale: float,
) -> tuple[dict, dict]:
    """Return true trajectories and context-balanced endpoint distributions."""
    targets = cache["targets"].detach().cpu().numpy().astype(float)
    targets[:, :, 1] *= data_delay_scale
    targets[:, :, 5] = np.cumsum(targets[:, :, 5], axis=1)
    group_ids = [str(value) for value in cache["group_ids"]]

    trajectories = {}
    endpoint_groups = {}
    for channel, name in enumerate(SYSTEM_CHANNELS):
        trajectories[name] = group_balanced_series(targets[:, :, channel], group_ids)
        endpoint_groups[name] = {
            str(horizon): [
                float(value)
                for value in np.asarray(
                    [
                        targets[
                            np.asarray(
                                [gid == group for gid in group_ids]
                            ),
                            horizon - 1,
                            channel,
                        ].mean()
                        for group in list(dict.fromkeys(group_ids))
                    ]
                )
            ]
            for horizon in SELECTED_HORIZONS
        }
    return trajectories, endpoint_groups


def build_prediction_target_trajectory_summaries(
    cache: dict,
    *,
    model_delay_scale: float,
    data_delay_scale: float,
) -> dict:
    """Build group-balanced true/predicted trajectories and errors."""
    predictions = cache["predictions"].detach().cpu().numpy().astype(float)
    targets = cache["targets"].detach().cpu().numpy().astype(float)
    predictions[:, :, 1] *= model_delay_scale
    targets[:, :, 1] *= data_delay_scale
    predictions[:, :, 5] = np.cumsum(predictions[:, :, 5], axis=1)
    targets[:, :, 5] = np.cumsum(targets[:, :, 5], axis=1)
    group_ids = [str(value) for value in cache["group_ids"]]

    result = {}
    for channel, name in enumerate(SYSTEM_CHANNELS):
        pred_groups = group_series_matrix(predictions[:, :, channel], group_ids)
        target_groups = group_series_matrix(targets[:, :, channel], group_ids)
        error_groups = pred_groups - target_groups
        result[name] = {
            "target_mean": target_groups.mean(axis=0).tolist(),
            "target_std": target_groups.std(axis=0).tolist(),
            "prediction_mean": pred_groups.mean(axis=0).tolist(),
            "prediction_std": pred_groups.std(axis=0).tolist(),
            "error_mean": error_groups.mean(axis=0).tolist(),
            "error_std": error_groups.std(axis=0).tolist(),
        }
    return result


def endpoint_values(
    predictions: np.ndarray,
    targets: np.ndarray,
    horizon: int,
    channel: int,
) -> tuple[np.ndarray, np.ndarray]:
    if channel == 5:
        return (
            predictions[:, :horizon, channel].sum(axis=1),
            targets[:, :horizon, channel].sum(axis=1),
        )
    return predictions[:, horizon - 1, channel], targets[:, horizon - 1, channel]


def build_metrics(cache: dict, report_path: Path) -> dict:
    predictions = cache["predictions"].detach().cpu().numpy().astype(float)
    targets = cache["targets"].detach().cpu().numpy().astype(float)
    max_horizon = int(cache["max_horizon"])

    source_report = None
    if report_path.exists():
        source_report = json.loads(report_path.read_text(encoding="utf-8"))
    model_delay_scale = float(
        (source_report or {}).get("model_delay_scale", 10.0)
    )
    data_delay_scale = float(
        (source_report or {}).get("data_delay_scale", 100.0)
    )
    predictions[:, :, 1] *= model_delay_scale
    targets[:, :, 1] *= data_delay_scale
    true_target_trajectories, true_endpoint_group_values = (
        build_true_target_summaries(
            cache,
            data_delay_scale=data_delay_scale,
        )
    )

    by_horizon = {}
    rank_curves = {name: [] for name in SYSTEM_CHANNELS}
    risk_auc_curve = []
    risk_out_of_bounds_curve = []

    for horizon in range(1, max_horizon + 1):
        horizon_metrics = {}
        for channel, name in enumerate(SYSTEM_CHANNELS):
            prediction, target = endpoint_values(
                predictions, targets, horizon, channel
            )
            bundle = metric_bundle(prediction, target)
            horizon_metrics[name] = bundle
            rank_curves[name].append(bundle["spearman_tie_aware"])

        risk_prediction, risk_target = endpoint_values(
            predictions, targets, horizon, 6
        )
        risk_auc_curve.append(rank_auc(risk_target, risk_prediction))
        risk_out_of_bounds_curve.append(
            float(np.mean((risk_prediction < 0.0) | (risk_prediction > 1.0)))
        )
        if horizon in SELECTED_HORIZONS:
            by_horizon[str(horizon)] = horizon_metrics

    return {
        "schema_version": SCHEMA_VERSION,
        "source_cache": str(cache.get("cache_path", "")),
        "source_data": cache["data"],
        "source_checkpoint": cache["checkpoint"],
        "source_report": str(report_path) if report_path.exists() else None,
        "source_data_sha256": (
            source_report.get("data_sha256") if source_report else None
        ),
        "source_checkpoint_sha256": (
            source_report.get("checkpoint_sha256") if source_report else None
        ),
        "samples": int(cache["samples"]),
        "groups": int(cache["groups"]),
        "selected_horizons": list(SELECTED_HORIZONS),
        "max_horizon": max_horizon,
        "semantics": {
            "state_channels": "endpoint value at t + H",
            "completed_orders_delta": (
                "per-tick event accumulated over [t + 1, t + H]"
            ),
            "average_excess_delay": "physical ticks",
            "model_delay_scale": model_delay_scale,
            "data_delay_scale": data_delay_scale,
            "spearman": "tie-aware average-rank Spearman",
        },
        "step_contract": cache["audit"],
        "true_target_trajectories": true_target_trajectories,
        "true_endpoint_group_values": true_endpoint_group_values,
        "metrics_by_selected_horizon": by_horizon,
        "curves": {
            "horizons": list(range(1, max_horizon + 1)),
            "spearman_tie_aware": rank_curves,
            "risk_auc": risk_auc_curve,
            "risk_prediction_out_of_bounds_fraction": (
                risk_out_of_bounds_curve
            ),
        },
        "warnings": [
            "H > 10 is outside the checkpoint training horizon.",
            "Steps 10-19 use embedding rows unseen during H=10 training.",
            "Transitions after H=20 reuse learned-table index 19.",
            (
                "Legacy report Spearman values are not used because the "
                "legacy rank helper does not average ties."
            ),
        ],
    }


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return [str(png), str(pdf)]


def add_horizon_contract(ax: plt.Axes, *, label: bool = False) -> None:
    ax.axvspan(10, 100, color="#6B7280", alpha=0.055, zorder=0)
    ax.axvline(10, color="#4B5563", linestyle="--", linewidth=0.9)
    ax.axvline(20, color="#9CA3AF", linestyle=":", linewidth=0.9)
    if label:
        top = ax.get_ylim()[1]
        ax.annotate(
            "training limit",
            (10, top),
            xytext=(3, -3),
            textcoords="offset points",
            va="top",
            fontsize=7,
            color="#4B5563",
        )


def selected_metric(metrics: dict, horizon: int, channel: str, field: str):
    return metrics["metrics_by_selected_horizon"][str(horizon)][channel][field]


def plot_rank_heatmap(metrics: dict, output_dir: Path, dpi: int) -> list[str]:
    rows = list(SYSTEM_CHANNELS)
    values = np.full((len(rows), len(SELECTED_HORIZONS)), np.nan)
    for row, channel in enumerate(rows):
        for column, horizon in enumerate(SELECTED_HORIZONS):
            value = selected_metric(
                metrics, horizon, channel, "spearman_tie_aware"
            )
            if value is not None:
                values[row, column] = value

    fig, ax = plt.subplots(figsize=(7.15, 3.65))
    cmap = matplotlib.colormaps["RdBu"].copy()
    cmap.set_bad("#E5E7EB")
    image = ax.imshow(
        np.ma.masked_invalid(values),
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
    )
    ax.grid(False)
    ax.set_xticks(
        np.arange(len(SELECTED_HORIZONS)),
        [f"H={value}" for value in SELECTED_HORIZONS],
    )
    ax.set_yticks(
        np.arange(len(rows)),
        [DISPLAY_NAMES[name] for name in rows],
    )
    ax.axvline(0.5, color="white", linewidth=2.0)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = "N/A" if not np.isfinite(value) else f"{value:+.2f}"
            color = "white" if np.isfinite(value) and abs(value) >= 0.48 else "#111827"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="semibold",
                color=color,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Tie-aware Spearman")
    ax.set_title(
        "Congestion rank fidelity collapses while throughput ordering persists"
    )
    fig.text(
        0.5,
        0.015,
        (
            "H=10 trained; H>10 extrapolation. Endpoint metrics; completed "
            "orders are cumulative over the rollout prefix."
        ),
        ha="center",
        fontsize=7.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))
    return save_figure(fig, output_dir, "figure_01_rank_heatmap", dpi)


def plot_rank_curves(metrics: dict, output_dir: Path, dpi: int) -> list[str]:
    horizons = np.asarray(metrics["curves"]["horizons"], dtype=float)
    curves = metrics["curves"]["spearman_tie_aware"]
    panels = [
        (
            "Congestion-state channels",
            (
                "total_wait_time",
                "bottleneck_CVaR",
                "deadlock_or_severe_congestion_risk",
            ),
        ),
        (
            "Service and station channels",
            (
                "average_excess_delay",
                "station_queue_delta",
                "station_load_imbalance",
            ),
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85), sharey=True)
    for panel_index, (title, channels) in enumerate(panels):
        ax = axes[panel_index]
        for channel in channels:
            values = np.asarray(
                [np.nan if value is None else value for value in curves[channel]],
                dtype=float,
            )
            ax.plot(
                horizons,
                values,
                label=DISPLAY_NAMES[channel],
                color=COLORS[channel],
                linewidth=1.65,
            )
            ax.scatter(
                SELECTED_HORIZONS,
                [values[value - 1] for value in SELECTED_HORIZONS],
                color=COLORS[channel],
                s=14,
                zorder=3,
            )
        ax.axhline(0.0, color="#6B7280", linewidth=0.8)
        add_horizon_contract(ax, label=(panel_index == 1))
        ax.set_xlim(1, 100)
        ax.set_ylim(-0.55, 1.02)
        ax.set_xlabel("Rollout horizon H")
        ax.set_title(title)
        ax.legend(loc="best", frameon=False)
    axes[0].set_ylabel("Tie-aware Spearman")
    fig.suptitle("Where the frozen H=10 transition loses physical ordering", y=1.02)
    fig.tight_layout()
    return save_figure(fig, output_dir, "figure_02_rank_curves", dpi)


def plot_calibration(metrics: dict, output_dir: Path, dpi: int) -> list[str]:
    channels = (
        "total_wait_time",
        "station_load_imbalance",
        "bottleneck_CVaR",
        "deadlock_or_severe_congestion_risk",
    )
    x = np.asarray(SELECTED_HORIZONS, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.0))
    for ax, channel in zip(axes.flat, channels):
        prediction_mean = np.asarray(
            [
                selected_metric(metrics, h, channel, "prediction_mean")
                for h in SELECTED_HORIZONS
            ]
        )
        target_mean = np.asarray(
            [
                selected_metric(metrics, h, channel, "target_mean")
                for h in SELECTED_HORIZONS
            ]
        )
        target_std = np.asarray(
            [
                selected_metric(metrics, h, channel, "target_std")
                for h in SELECTED_HORIZONS
            ]
        )
        ax.fill_between(
            x,
            target_mean - target_std,
            target_mean + target_std,
            color="#9CA3AF",
            alpha=0.18,
            linewidth=0,
            label="Target +/- 1 SD",
        )
        ax.plot(
            x,
            target_mean,
            marker="o",
            color=COLORS["target"],
            linewidth=1.6,
            label="Simulator target",
        )
        ax.plot(
            x,
            prediction_mean,
            marker="s",
            color=COLORS["prediction"],
            linewidth=1.6,
            label="World Model",
        )
        ax.axvline(10, color="#4B5563", linestyle="--", linewidth=0.8)
        ax.axvline(20, color="#9CA3AF", linestyle=":", linewidth=0.8)
        ax.set_title(DISPLAY_NAMES[channel])
        ax.set_xlabel("Rollout horizon H")
        ax.set_ylabel("Endpoint value")
        ax.set_xticks(SELECTED_HORIZONS)
        ax.annotate(
            f"H100 bias {prediction_mean[-1] - target_mean[-1]:+.2f}",
            xy=(100, prediction_mean[-1]),
            xytext=(-5, 8),
            textcoords="offset points",
            ha="right",
            fontsize=7.5,
            color=COLORS["prediction"],
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Endpoint calibration drifts after H=10", y=0.99)
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.97))
    return save_figure(fig, output_dir, "figure_03_calibration_drift", dpi)


def plot_risk_diagnostics(metrics: dict, output_dir: Path, dpi: int) -> list[str]:
    horizons = np.asarray(metrics["curves"]["horizons"], dtype=float)
    auc = np.asarray(
        [
            np.nan if value is None else value
            for value in metrics["curves"]["risk_auc"]
        ],
        dtype=float,
    )
    out_of_bounds = np.asarray(
        metrics["curves"]["risk_prediction_out_of_bounds_fraction"],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8))
    ax = axes[0]
    ax.plot(horizons, auc, color="#4C78A8", linewidth=1.8)
    ax.scatter(
        SELECTED_HORIZONS,
        [auc[value - 1] for value in SELECTED_HORIZONS],
        color=["#4C78A8", "#4C78A8", "#E45756", "#E45756", "#E45756"],
        s=26,
        zorder=3,
    )
    ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=0.9)
    add_horizon_contract(ax)
    ax.set_xlim(1, 100)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Rollout horizon H")
    ax.set_ylabel("Risk AUC")
    ax.set_title("Risk ordering falls below random")

    ax = axes[1]
    ax.plot(horizons, out_of_bounds, color="#E45756", linewidth=1.8)
    ax.scatter(
        SELECTED_HORIZONS,
        [out_of_bounds[value - 1] for value in SELECTED_HORIZONS],
        color="#E45756",
        s=26,
        zorder=3,
    )
    add_horizon_contract(ax, label=True)
    ax.set_xlim(1, 100)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Rollout horizon H")
    ax.set_ylabel("Predictions outside [0, 1]")
    ax.set_title("Risk output saturates outside its label range")
    fig.suptitle("Severe-congestion risk failure", y=1.02)
    fig.tight_layout()
    return save_figure(fig, output_dir, "figure_04_risk_failure", dpi)


def plot_cumulative_throughput(metrics: dict, output_dir: Path, dpi: int) -> list[str]:
    horizons = np.asarray(metrics["curves"]["horizons"], dtype=float)
    channel = "completed_orders_delta"

    dense = metrics["curves"]["cumulative_completed_orders"]
    prediction_mean = np.asarray(dense["prediction_mean"], dtype=float)
    target_mean = np.asarray(dense["target_mean"], dtype=float)
    target_std = np.asarray(dense["target_std"], dtype=float)
    mae = np.asarray(dense["mae"], dtype=float)
    rank = np.asarray(
        [
            np.nan if value is None else value
            for value in metrics["curves"]["spearman_tie_aware"][channel]
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65))
    ax = axes[0]
    ax.fill_between(
        horizons,
        np.maximum(0.0, target_mean - target_std),
        target_mean + target_std,
        color="#9CA3AF",
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(horizons, target_mean, color=COLORS["target"], label="Simulator")
    ax.plot(
        horizons,
        prediction_mean,
        color=COLORS["completed_orders_delta"],
        label="World Model",
    )
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylabel("Cumulative completed orders")
    ax.set_title("Mean completion count")

    ax = axes[1]
    ax.plot(horizons, mae, color="#F58518", linewidth=1.8)
    ax.set_ylabel("MAE (orders)")
    ax.set_title("Absolute calibration error")

    ax = axes[2]
    ax.plot(horizons, rank, color="#4C78A8", linewidth=1.8)
    ax.axhline(0.0, color="#6B7280", linewidth=0.8)
    ax.set_ylim(-0.1, 1.0)
    ax.set_ylabel("Tie-aware Spearman")
    ax.set_title("Completion-count ordering")

    for index, ax in enumerate(axes):
        add_horizon_contract(ax, label=(index == 2))
        ax.set_xlim(1, 100)
        ax.set_xlabel("Rollout horizon H")
        ax.scatter(
            SELECTED_HORIZONS,
            [
                (
                    prediction_mean[value - 1]
                    if index == 0
                    else mae[value - 1]
                    if index == 1
                    else rank[value - 1]
                )
                for value in SELECTED_HORIZONS
            ],
            color=(
                COLORS["completed_orders_delta"]
                if index == 0
                else "#F58518"
                if index == 1
                else "#4C78A8"
            ),
            s=16,
            zorder=3,
        )

    fig.suptitle("Throughput ordering survives longer than calibration", y=1.03)
    fig.tight_layout()
    return save_figure(fig, output_dir, "figure_05_cumulative_throughput", dpi)


def plot_true_target_trajectories(
    metrics: dict, output_dir: Path, dpi: int
) -> list[str]:
    """Plot simulator target trajectories only, without model predictions."""
    horizons = np.arange(1, int(metrics["max_horizon"]) + 1, dtype=float)
    trajectories = metrics["true_target_trajectories"]
    fig, axes = plt.subplots(4, 2, figsize=(7.15, 7.15), sharex=True)
    axes_flat = list(axes.flat)

    for axis_index, channel in enumerate(SYSTEM_CHANNELS):
        ax = axes_flat[axis_index]
        summary = trajectories[channel]
        mean = np.asarray(summary["group_balanced_mean"], dtype=float)
        std = np.asarray(summary["group_balanced_std"], dtype=float)
        color = COLORS[channel]
        ax.fill_between(
            horizons,
            mean - std,
            mean + std,
            color=color,
            alpha=0.18,
            linewidth=0,
            label="Context mean +/- 1 SD",
        )
        ax.plot(
            horizons,
            mean,
            color=color,
            linewidth=1.8,
            label="Simulator target",
        )
        ax.scatter(
            SELECTED_HORIZONS,
            [mean[horizon - 1] for horizon in SELECTED_HORIZONS],
            color=color,
            edgecolor="white",
            linewidth=0.6,
            s=23,
            zorder=3,
        )
        ax.axvline(10, color="#4B5563", linestyle="--", linewidth=0.8)
        ax.axvline(20, color="#9CA3AF", linestyle=":", linewidth=0.8)
        ax.set_title(DISPLAY_NAMES[channel])
        ax.set_xlim(1, int(metrics["max_horizon"]))
        ylabel = {
            "average_excess_delay": "Physical ticks",
            "completed_orders_delta": "Completed orders",
            "deadlock_or_severe_congestion_risk": "Risk [0, 1]",
        }.get(channel, "Label value")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        if axis_index in (0, 1):
            ax.legend(frameon=False, loc="best")

    axes_flat[-1].axis("off")
    for ax in axes_flat[-2:]:
        if ax in axes_flat[:-1]:
            ax.set_xlabel("Rollout step k")
    # The last visible row has two axes; the second one is intentionally empty.
    axes_flat[6].set_xlabel("Rollout step k")
    fig.suptitle(
        "Simulator ground-truth trajectories under isolated rollout",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Solid curve: equal-weight mean over 53 contexts; shaded area: "
            "context-to-context SD. Markers show H=10/20/50/80/100."
        ),
        ha="center",
        fontsize=7.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.975))
    return save_figure(fig, output_dir, "figure_06_true_target_trajectories", dpi)


def plot_true_endpoint_distributions(
    metrics: dict, output_dir: Path, dpi: int
) -> list[str]:
    """Plot context-balanced true endpoint distributions at selected H."""
    fig, axes = plt.subplots(4, 2, figsize=(7.15, 7.4))
    axes_flat = list(axes.flat)
    positions = np.arange(1, len(SELECTED_HORIZONS) + 1, dtype=float)

    for axis_index, channel in enumerate(SYSTEM_CHANNELS):
        ax = axes_flat[axis_index]
        values_by_horizon = metrics["true_endpoint_group_values"][channel]
        values = [
            np.asarray(values_by_horizon[str(horizon)], dtype=float)
            for horizon in SELECTED_HORIZONS
        ]
        box = ax.boxplot(
            values,
            positions=positions,
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.2},
            whiskerprops={"color": "#6B7280", "linewidth": 0.8},
            capprops={"color": "#6B7280", "linewidth": 0.8},
            boxprops={
                "facecolor": COLORS[channel],
                "edgecolor": COLORS[channel],
                "alpha": 0.45,
            },
        )
        ax.plot(
            positions,
            [float(np.mean(value)) for value in values],
            color=COLORS[channel],
            marker="o",
            linewidth=1.2,
            markersize=3.5,
        )
        ax.set_title(DISPLAY_NAMES[channel])
        ax.set_xticks(positions, [f"H={horizon}" for horizon in SELECTED_HORIZONS])
        ylabel = {
            "average_excess_delay": "Physical ticks",
            "completed_orders_delta": "Completed orders",
            "deadlock_or_severe_congestion_risk": "Risk [0, 1]",
        }.get(channel, "Label value")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)

    axes_flat[-1].axis("off")
    fig.suptitle(
        "True endpoint distributions across rollout horizons",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Each box contains 53 context means (one equal-weight value per "
            "starting context); no World Model prediction is plotted."
        ),
        ha="center",
        fontsize=7.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.975))
    return save_figure(fig, output_dir, "figure_07_true_endpoint_distributions", dpi)


def plot_true_vs_predicted_trajectories(
    metrics: dict, output_dir: Path, dpi: int
) -> list[str]:
    """Overlay simulator targets and frozen-model trajectories."""
    horizons = np.arange(1, int(metrics["max_horizon"]) + 1, dtype=float)
    trajectories = metrics["true_vs_pred_trajectories"]
    fig, axes = plt.subplots(4, 2, figsize=(7.15, 7.15), sharex=True)
    axes_flat = list(axes.flat)

    for axis_index, channel in enumerate(SYSTEM_CHANNELS):
        ax = axes_flat[axis_index]
        summary = trajectories[channel]
        target_mean = np.asarray(summary["target_mean"], dtype=float)
        target_std = np.asarray(summary["target_std"], dtype=float)
        prediction_mean = np.asarray(summary["prediction_mean"], dtype=float)
        prediction_std = np.asarray(summary["prediction_std"], dtype=float)
        ax.fill_between(
            horizons,
            target_mean - target_std,
            target_mean + target_std,
            color="#6B7280",
            alpha=0.13,
            linewidth=0,
        )
        ax.fill_between(
            horizons,
            prediction_mean - prediction_std,
            prediction_mean + prediction_std,
            color=COLORS["prediction"],
            alpha=0.10,
            linewidth=0,
        )
        ax.plot(
            horizons,
            target_mean,
            color=COLORS["target"],
            linewidth=1.7,
            label="Simulator target",
        )
        ax.plot(
            horizons,
            prediction_mean,
            color=COLORS["prediction"],
            linewidth=1.7,
            label="World Model",
        )
        ax.axvline(10, color="#4B5563", linestyle="--", linewidth=0.8)
        ax.axvline(20, color="#9CA3AF", linestyle=":", linewidth=0.8)
        ax.set_title(DISPLAY_NAMES[channel])
        ylabel = {
            "average_excess_delay": "Physical ticks",
            "completed_orders_delta": "Completed orders",
            "deadlock_or_severe_congestion_risk": "Risk [0, 1]",
        }.get(channel, "Value")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, int(metrics["max_horizon"]))
        if axis_index == 0:
            ax.legend(frameon=False, loc="best")

    axes_flat[-1].axis("off")
    axes_flat[6].set_xlabel("Rollout step k")
    fig.suptitle(
        "True versus predicted system trajectories",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Equal-weight means over 53 contexts; shaded bands are context "
            "SD. Dashed H=10 and dotted H=20 mark the checkpoint boundaries."
        ),
        ha="center",
        fontsize=7.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.975))
    return save_figure(fig, output_dir, "figure_08_true_vs_predicted_trajectories", dpi)


def plot_trajectory_errors(
    metrics: dict, output_dir: Path, dpi: int
) -> list[str]:
    """Plot signed prediction error against rollout step."""
    horizons = np.arange(1, int(metrics["max_horizon"]) + 1, dtype=float)
    trajectories = metrics["true_vs_pred_trajectories"]
    fig, axes = plt.subplots(4, 2, figsize=(7.15, 7.15), sharex=True)
    axes_flat = list(axes.flat)

    for axis_index, channel in enumerate(SYSTEM_CHANNELS):
        ax = axes_flat[axis_index]
        summary = trajectories[channel]
        error_mean = np.asarray(summary["error_mean"], dtype=float)
        error_std = np.asarray(summary["error_std"], dtype=float)
        color = COLORS[channel]
        ax.fill_between(
            horizons,
            error_mean - error_std,
            error_mean + error_std,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(horizons, error_mean, color=color, linewidth=1.8)
        ax.axhline(0.0, color="#111827", linewidth=0.8)
        ax.axvline(10, color="#4B5563", linestyle="--", linewidth=0.8)
        ax.axvline(20, color="#9CA3AF", linestyle=":", linewidth=0.8)
        ax.set_title(DISPLAY_NAMES[channel])
        ylabel = {
            "average_excess_delay": "Error (ticks)",
            "completed_orders_delta": "Error (orders)",
            "deadlock_or_severe_congestion_risk": "Error (risk)",
        }.get(channel, "Prediction - target")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, int(metrics["max_horizon"]))

    axes_flat[-1].axis("off")
    axes_flat[6].set_xlabel("Rollout step k")
    fig.suptitle(
        "Autoregressive trajectory error under isolated rollout",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Positive values mean overprediction. A persistent drift after "
            "H=10 indicates rollout calibration loss, not label noise."
        ),
        ha="center",
        fontsize=7.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.975))
    return save_figure(fig, output_dir, "figure_09_trajectory_errors", dpi)


def dense_completed_metrics(cache: dict) -> dict:
    predictions = cache["predictions"][:, :, 5].detach().cpu().numpy().astype(float)
    targets = cache["targets"][:, :, 5].detach().cpu().numpy().astype(float)
    prediction_cumulative = np.cumsum(predictions, axis=1)
    target_cumulative = np.cumsum(targets, axis=1)
    error = prediction_cumulative - target_cumulative
    return {
        "prediction_mean": prediction_cumulative.mean(axis=0).tolist(),
        "target_mean": target_cumulative.mean(axis=0).tolist(),
        "target_std": target_cumulative.std(axis=0).tolist(),
        "mae": np.abs(error).mean(axis=0).tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.max_horizon < max(SELECTED_HORIZONS):
        raise SystemExit("--max-horizon must be at least 100")
    apply_style()
    cache = load_or_build_cache(args)
    cache["cache_path"] = str(args.cache)
    metrics = build_metrics(cache, args.report)
    metrics["curves"]["cumulative_completed_orders"] = (
        dense_completed_metrics(cache)
    )
    metrics["true_vs_pred_trajectories"] = (
        build_prediction_target_trajectory_summaries(
            cache,
            model_delay_scale=float(metrics["semantics"]["model_delay_scale"]),
            data_delay_scale=float(metrics["semantics"]["data_delay_scale"]),
        )
    )

    artifacts = []
    artifacts.extend(plot_rank_heatmap(metrics, args.output_dir, args.dpi))
    artifacts.extend(plot_rank_curves(metrics, args.output_dir, args.dpi))
    artifacts.extend(plot_calibration(metrics, args.output_dir, args.dpi))
    artifacts.extend(plot_risk_diagnostics(metrics, args.output_dir, args.dpi))
    artifacts.extend(plot_cumulative_throughput(metrics, args.output_dir, args.dpi))
    artifacts.extend(
        plot_true_target_trajectories(metrics, args.output_dir, args.dpi)
    )
    artifacts.extend(
        plot_true_endpoint_distributions(metrics, args.output_dir, args.dpi)
    )
    artifacts.extend(
        plot_true_vs_predicted_trajectories(metrics, args.output_dir, args.dpi)
    )
    artifacts.extend(plot_trajectory_errors(metrics, args.output_dir, args.dpi))

    metrics["figure_artifacts"] = artifacts
    metrics["figure_descriptions"] = {
        "figure_01_rank_heatmap": (
            "Selected-horizon tie-aware rank fidelity for all seven system "
            "channels; completion events are accumulated over the prefix."
        ),
        "figure_02_rank_curves": (
            "Per-step endpoint rank fidelity, with H=10 training and H=20 "
            "step-table boundaries marked."
        ),
        "figure_03_calibration_drift": (
            "Predicted and simulator endpoint means for four congestion "
            "channels, including target variation."
        ),
        "figure_04_risk_failure": (
            "Severe-risk AUC and the fraction of predictions outside the "
            "physical [0, 1] label interval."
        ),
        "figure_05_cumulative_throughput": (
            "Cumulative completed-order mean, absolute error, and ordering "
            "over the rollout prefix."
        ),
        "figure_06_true_target_trajectories": (
            "Simulator target trajectories only, averaged with equal weight "
            "per starting context."
        ),
        "figure_07_true_endpoint_distributions": (
            "Simulator target endpoint distributions at selected horizons; "
            "each box is balanced over contexts."
        ),
        "figure_08_true_vs_predicted_trajectories": (
            "Direct simulator-target versus frozen-World-Model trajectory "
            "comparison for all seven channels."
        ),
        "figure_09_trajectory_errors": (
            "Signed autoregressive prediction error and context variation "
            "over rollout step."
        ),
    }
    metrics_path = args.output_dir / "system_dynamics_horizon_plot_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote plotted metrics: {metrics_path}")
    for artifact in artifacts:
        print(f"  wrote figure: {artifact}")


if __name__ == "__main__":
    main()
