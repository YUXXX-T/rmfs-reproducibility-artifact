"""Visualize station prediction accuracy from dynamics_eval_test_v3.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DEFAULT_EVAL = "DataGen/wm_checkpoints/v6_formal/dynamics_eval_test_v3.json"
OUTPUT_DIR = Path(__file__).parent / "output"


def load_eval(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_station_quality(data: dict, out: Path):
    station = data["aggregate_metrics"]["future_station"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: MAE / RMSE / Spearman
    metrics_left = {
        "Queue MAE": station["queue_mae"],
        "Queue RMSE": station["queue_rmse"],
        "Load MAE": station["assigned_load_mae"],
        "Load RMSE": station["assigned_load_rmse"],
    }
    x = np.arange(len(metrics_left))
    bar_colors = ["#1f77b4", "#6baed6", "#ff7f0e", "#ffbb78"]
    bars = ax1.bar(x, list(metrics_left.values()), color=bar_colors, width=0.5, edgecolor="white")
    for bar, v in zip(bars, metrics_left.values()):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(metrics_left.keys()), fontsize=10)
    ax1.set_ylabel("Error", fontsize=11)
    ax1.set_title("预测误差", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    spearman_val = station["queue_spearman"]
    imbalance_corr = station["station_imbalance_correlation"]
    ax_inset = ax1.inset_axes([0.62, 0.55, 0.35, 0.38])
    corr_bars = ax_inset.bar([0, 1], [spearman_val, imbalance_corr], color=["#2ca02c", "#9467bd"], width=0.5)
    for bar, v in zip(corr_bars, [spearman_val, imbalance_corr]):
        ax_inset.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax_inset.set_xticks([0, 1])
    ax_inset.set_xticklabels(["Queue\nSpearman", "Imbalance\nCorr."], fontsize=7)
    ax_inset.set_ylim(0.95, 1.005)
    ax_inset.set_title("Correlation", fontsize=8)

    # Right: Improvement over naive
    improvements = {
        "Queue": station["queue_improvement_over_naive"],
        "Assigned Load": station["assigned_load_improvement_over_naive"],
    }
    naive_maes = {
        "Queue": station["queue_naive_mae"],
        "Assigned Load": station["assigned_load_naive_mae"],
    }
    model_maes = {
        "Queue": station["queue_mae"],
        "Assigned Load": station["assigned_load_mae"],
    }

    x2 = np.arange(2)
    width = 0.3
    bars_naive = ax2.bar(x2 - width / 2, list(naive_maes.values()), width, label="Naive MAE", color="#d62728", edgecolor="white")
    bars_model = ax2.bar(x2 + width / 2, list(model_maes.values()), width, label="Model MAE", color="#2ca02c", edgecolor="white")

    for bar, v in zip(bars_naive, naive_maes.values()):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{v:.2f}", ha="center", fontsize=10)
    for bar, v in zip(bars_model, model_maes.values()):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

    for i, (name, imp) in enumerate(improvements.items()):
        ax2.annotate(
            f"-{imp * 100:.1f}%",
            xy=(x2[i], max(list(naive_maes.values())[i], list(model_maes.values())[i]) + 0.3),
            ha="center",
            fontsize=14,
            fontweight="bold",
            color="#2ca02c",
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(list(improvements.keys()), fontsize=11)
    ax2.legend(fontsize=10)
    ax2.set_ylabel("MAE", fontsize=11)
    ax2.set_title("相比 Naive 基线的改进", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("站台预测质量", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "station_prediction_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_station_examples(data: dict, out: Path):
    examples = data["prediction_examples"]
    if len(examples) < 2:
        print("Not enough examples to plot station predictions")
        return

    sample_indices = [0, len(examples) - 1]
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(14, 5 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, sample_indices):
        ex = examples[idx]
        horizons = ex["future_station_by_horizon"]
        group_id = ex.get("candidate_group_id", f"sample_{idx}")

        k = horizons[0]["k"]
        pred_vals = horizons[0]["pred"]
        target_vals = horizons[0]["target"]
        num_stations = len(pred_vals) // 2
        queue_pred = pred_vals[:num_stations]
        queue_target = target_vals[:num_stations]
        load_pred = pred_vals[num_stations:]
        load_target = target_vals[num_stations:]

        x = np.arange(num_stations)
        width = 0.2

        ax.bar(x - 1.5 * width, queue_target, width, label="Queue Target", color="#1f77b4", alpha=0.6)
        ax.bar(x - 0.5 * width, queue_pred, width, label="Queue Pred", color="#1f77b4")
        ax.bar(x + 0.5 * width, load_target, width, label="Load Target", color="#ff7f0e", alpha=0.6)
        ax.bar(x + 1.5 * width, load_pred, width, label="Load Pred", color="#ff7f0e")

        ax.set_xlabel("Station", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i}" for i in range(num_stations)], fontsize=10)
        ax.legend(fontsize=9, ncol=4, loc="upper right")
        ax.set_title(f"样本 {group_id} — Horizon k={k}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("站台预测示例 (Pred vs Target)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "station_prediction_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(eval_path: str | None = None):
    if eval_path is None:
        project_root = Path(__file__).resolve().parents[2]
        eval_path = str(project_root / DEFAULT_EVAL)

    data = load_eval(eval_path)
    print(f"Loaded evaluation from {eval_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_station_quality(data, OUTPUT_DIR)
    plot_station_examples(data, OUTPUT_DIR)
    print(f"Saved station prediction plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
