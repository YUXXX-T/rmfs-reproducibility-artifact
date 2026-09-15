"""
ICRA Offline Dynamics Evaluation — Compact Summary
===================================================
Reads dynamics_eval_test_v3.json and produces:

  fig_offline_prediction.pdf  — 4 subplots:
    (a) System label Spearman correlations (7-channel bar)
    (b) Node-level: wait AUC, congestion AUC, density Spearman
    (c) Station: queue/load MAE vs naive baseline
    (d) Ranking: pairwise acc, top-1 regret, cost Spearman

Usage:
    python -m WorldModel.visualize.plot_dynamics_eval ^
      --json DataGen/wm_checkpoints/v6_formal/dynamics_eval_test_v3.json ^
      --out WorldModel/visualize/output
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from WorldModel.visualize.icra_style import (
    apply_icra_style,
    save_fig,
    label_subplot,
    COLOR_CONG_WM,
    COLOR_OLD_WM,
    COLOR_GREEDY,
)

OUTPUT_DIR = Path(__file__).parent / "output"

SYSTEM_CHANNELS = [
    "total_wait_time",
    "average_excess_delay",
    "station_queue_delta",
    "station_load_imbalance",
    "bottleneck_CVaR",
    "completed_orders_delta",
    "deadlock_or_severe_congestion_risk",
]
SHORT_NAMES = [
    "Wait",
    "Excess Delay",
    "Queue Delta",
    "Load Imbal.",
    "Bottleneck",
    "Orders Delta",
    "Deadlock Risk",
]


def load_eval(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_compact(data: dict, out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.4))
    bar_color = COLOR_CONG_WM
    naive_color = COLOR_GREEDY
    accent = COLOR_OLD_WM

    # ── (a) System label Spearman ────────────────────────────────
    ax = axes[0]
    sys = data["aggregate_metrics"]["future_system"]
    spearman_vals = [sys[ch]["spearman"] for ch in SYSTEM_CHANNELS]
    x = np.arange(len(SYSTEM_CHANNELS))
    bars = ax.bar(x, spearman_vals, color=bar_color, width=0.7, edgecolor="white",
                  linewidth=0.3)
    for b, v in zip(bars, spearman_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=5.5, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, fontsize=5, rotation=45, ha="right")
    ax.set_ylabel("Spearman " + r"$\rho$")
    ax.set_ylim(0, 1.18)
    ax.set_title("System Labels", fontsize=8, pad=3)
    label_subplot(ax, "a")

    # ── (b) Node-level metrics ───────────────────────────────────
    ax = axes[1]
    node = data["aggregate_metrics"]["future_node"]
    node_metrics = {
        "Wait\nAUC": node["wait_auc"],
        "Congestion\nAUC": node["congestion_auc"],
        "Density\nSpearman": node["density_spearman"],
    }
    x = np.arange(len(node_metrics))
    vals = list(node_metrics.values())
    bars = ax.bar(x, vals, color=[bar_color, accent, "#7570b3"],
                  width=0.55, edgecolor="white", linewidth=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(list(node_metrics.keys()), fontsize=6.5)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Node Predictions", fontsize=8, pad=3)
    label_subplot(ax, "b")

    # ── (c) Station: model MAE vs naive MAE ──────────────────────
    ax = axes[2]
    sta = data["aggregate_metrics"]["future_station"]
    cats = ["Queue", "Load"]
    model_maes = [sta["queue_mae"], sta["assigned_load_mae"]]
    naive_maes = [sta["queue_naive_mae"], sta["assigned_load_naive_mae"]]
    improvements = [sta["queue_improvement_over_naive"],
                    sta["assigned_load_improvement_over_naive"]]

    x = np.arange(len(cats))
    w = 0.3
    ax.bar(x - w / 2, naive_maes, w, label="Naive", color=naive_color,
           edgecolor="white", linewidth=0.3)
    ax.bar(x + w / 2, model_maes, w, label="Model", color=bar_color,
           edgecolor="white", linewidth=0.3)
    for xi, imp in zip(x, improvements):
        ax.text(xi + w / 2, model_maes[int(xi)] + 0.05,
                f"-{imp*100:.0f}%", ha="center", fontsize=6.5,
                fontweight="bold", color=bar_color)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("MAE")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
    ax.set_title("Station Pred.", fontsize=8, pad=3)
    label_subplot(ax, "c")

    # ── (d) Ranking metrics ──────────────────────────────────────
    ax = axes[3]
    rank = data["ranking"]
    rank_metrics = {
        "Pair\nAcc": rank["pairwise_rank_accuracy"],
        "Top-1\nAcc": rank["top1_accuracy"],
        "Cost\nSpearman": rank["cost_spearman"],
    }
    regret_mean = rank["top1_regret_mean"]

    x = np.arange(len(rank_metrics))
    vals = list(rank_metrics.values())
    bars = ax.bar(x, vals, color=[bar_color, accent, "#7570b3"],
                  width=0.55, edgecolor="white", linewidth=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(list(rank_metrics.keys()), fontsize=6.5)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Ranking Quality", fontsize=8, pad=3)
    label_subplot(ax, "d")

    ax.text(0.95, 0.15,
            f"Regret\n{regret_mean:.3f}",
            transform=ax.transAxes, fontsize=6.5,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="gray", alpha=0.8))

    fig.tight_layout(w_pad=0.8)
    save_fig(fig, str(out / "fig_offline_prediction"))
    print(f"  Saved fig_offline_prediction.pdf/.png")


def main():
    apply_icra_style()

    parser = argparse.ArgumentParser(
        description="ICRA Offline Dynamics Evaluation",
    )
    parser.add_argument(
        "--json", type=str, required=True,
        help="dynamics_eval_test_v3.json path",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.out) if args.out else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    data = load_eval(args.json)
    print(f"Loaded {args.json}")
    print(f"  {data['meta']['num_samples']} samples, "
          f"{data['meta']['num_groups']} groups")

    plot_compact(data, out)


if __name__ == "__main__":
    main()
