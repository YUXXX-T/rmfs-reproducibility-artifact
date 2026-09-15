"""
ICRA Online Evaluation Figures
==============================
Reads multiple online eval JSONs (e.g. low / mid / high load) and produces:

  fig_online_load_comparison.pdf   — 4-subplot grouped bar chart
  fig_online_paired_delta.pdf      — 2×3 per-seed paired delta

Usage:
    python -m WorldModel.visualize.plot_online_eval ^
      --jsons online_low.json online_mid.json online_high.json ^
      --load-labels Low Mid High ^
      --methods Greedy "Old-cost WM" "Cong-cost WM" ^
      --method-keys Greedy v6_formal v6_cong ^
      --out WorldModel/visualize/output

Single-file backward-compatible:
    python -m WorldModel.visualize.plot_online_eval ^
      --jsons online_eval.json ^
      --load-labels Single
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from WorldModel.visualize.icra_style import (
    apply_icra_style,
    save_fig,
    mean_ci95,
    label_subplot,
    METHOD_COLORS,
    COLOR_GREEDY,
    COLOR_OLD_WM,
    COLOR_CONG_WM,
)

OUTPUT_DIR = Path(__file__).parent / "output"


# ── Data loading ─────────────────────────────────────────────────

def load_eval(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _collect_per_seed_metric(
    data: dict, method_key: str, metric: str,
) -> List[float]:
    """Extract per-seed values for a given method and metric."""
    ps = data["per_seed"]
    vals = []
    for seed_data in ps.values():
        entry = seed_data.get(method_key)
        if entry is not None and metric in entry:
            vals.append(entry[metric])
    return vals


# ── Fig 1: Load comparison (4 subplots × grouped bars) ──────────

def plot_load_comparison(
    all_data: List[dict],
    load_labels: List[str],
    method_keys: List[str],
    method_names: List[str],
    out: Path,
):
    metrics = [
        ("completed_orders", "Completed Orders", True),
        ("wm_label_cost", "WM Label Cost", False),
        ("station_pressure", "Station Pressure", False),
        ("deadlock_risk", "Deadlock Risk", False),
    ]

    colors = [METHOD_COLORS.get(n, "#333333") for n in method_names]
    n_loads = len(load_labels)
    n_methods = len(method_keys)
    bar_w = 0.7 / n_methods

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.3), sharey=False)
    letters = "abcd"

    for ax, (metric, ylabel, higher_better), letter in zip(axes, metrics, letters):
        for j, (mkey, mname, col) in enumerate(
            zip(method_keys, method_names, colors)
        ):
            means, cis = [], []
            for data in all_data:
                vals = _collect_per_seed_metric(data, mkey, metric)
                if not vals:
                    means.append(0)
                    cis.append(0)
                else:
                    m, ci = mean_ci95(vals)
                    means.append(m)
                    cis.append(ci)

            x = np.arange(n_loads)
            offset = (j - (n_methods - 1) / 2) * bar_w
            ax.bar(
                x + offset, means, bar_w * 0.9,
                yerr=cis, capsize=2, color=col,
                edgecolor="white", linewidth=0.4,
                label=mname if ax == axes[0] else None,
                error_kw={"linewidth": 0.7},
            )

        ax.set_xticks(np.arange(n_loads))
        ax.set_xticklabels(load_labels)
        ax.set_ylabel(ylabel)
        arrow = r"$\uparrow$" if higher_better else r"$\downarrow$"
        label_subplot(ax, letter)
        ax.set_title(f"{ylabel} ({arrow})", fontsize=8, pad=3)

    axes[0].legend(loc="upper left", framealpha=0.8, fontsize=7)
    fig.tight_layout(w_pad=1.0)
    save_fig(fig, str(out / "fig_online_load_comparison"))
    print(f"  Saved fig_online_load_comparison.pdf/.png")


# ── Fig 2: Paired per-seed delta (2 rows × N load columns) ──────

def plot_paired_delta(
    all_data: List[dict],
    load_labels: List[str],
    method_keys: List[str],
    method_names: List[str],
    greedy_key: str,
    out: Path,
):
    delta_metrics = [
        ("completed_orders", r"$\Delta$ Throughput", True),
        ("wm_label_cost", r"$\Delta$ WM Cost", False),
    ]

    wm_keys = [(k, n) for k, n in zip(method_keys, method_names)
               if k != greedy_key]
    if not wm_keys:
        print("  Skipping paired delta (no WM methods)")
        return

    n_loads = len(load_labels)
    n_rows = len(delta_metrics)
    colors_wm = [METHOD_COLORS.get(n, "#333333") for _, n in wm_keys]

    fig, axes = plt.subplots(
        n_rows, n_loads,
        figsize=(2.2 * n_loads, 1.8 * n_rows),
        squeeze=False,
    )

    letters_flat = "abcdefghijkl"
    idx = 0
    for row, (metric, ylabel, higher_better) in enumerate(delta_metrics):
        for col, (data, load_lbl) in enumerate(zip(all_data, load_labels)):
            ax = axes[row, col]
            ps = data["per_seed"]
            seeds = sorted(ps.keys())
            x = np.arange(len(seeds))

            for wi, (wk, wn) in enumerate(wm_keys):
                deltas = []
                for s in seeds:
                    g_val = ps[s].get(greedy_key, {}).get(metric, 0)
                    w_val = ps[s].get(wk, {}).get(metric, 0)
                    deltas.append(w_val - g_val)

                w = 0.7 / len(wm_keys)
                offset = (wi - (len(wm_keys) - 1) / 2) * w
                bar_colors = [
                    colors_wm[wi] if (d > 0) == higher_better
                    else "#d62728"
                    for d in deltas
                ]
                ax.bar(
                    x + offset, deltas, w * 0.9,
                    color=bar_colors, edgecolor="white", linewidth=0.3,
                    label=wn if row == 0 and col == 0 else None,
                )

            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"s{s}" for s in seeds], fontsize=6, rotation=45,
            )
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if row == 0:
                ax.set_title(load_lbl, fontsize=9)

            label_subplot(ax, letters_flat[idx], x=-0.18, y=1.12)
            idx += 1

    if len(wm_keys) > 1:
        axes[0, 0].legend(fontsize=6, loc="best", framealpha=0.8)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    save_fig(fig, str(out / "fig_online_paired_delta"))
    print(f"  Saved fig_online_paired_delta.pdf/.png")


# ── CLI ──────────────────────────────────────────────────────────

def main():
    apply_icra_style()

    parser = argparse.ArgumentParser(
        description="ICRA Online Evaluation Figures",
    )
    parser.add_argument(
        "--jsons", nargs="+", required=True,
        help="Online eval JSON files (one per load level)",
    )
    parser.add_argument(
        "--load-labels", nargs="+", default=None,
        help="X-axis labels for each JSON (e.g. Low Mid High)",
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["Greedy", "Old-cost WM", "Cong-cost WM"],
        help="Display names for methods (used for legend/colors)",
    )
    parser.add_argument(
        "--method-keys", nargs="+",
        default=["Greedy", "WorldModel", "WorldModel_cong"],
        help="Keys in per_seed[seed] dict for each method",
    )
    parser.add_argument(
        "--greedy-key", type=str, default="Greedy",
        help="Key for the greedy baseline in per_seed",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output directory",
    )
    args = parser.parse_args()

    if args.load_labels is None:
        args.load_labels = [f"Load{i}" for i in range(len(args.jsons))]
    if len(args.load_labels) != len(args.jsons):
        parser.error("--load-labels must match --jsons length")
    if len(args.methods) != len(args.method_keys):
        parser.error("--methods must match --method-keys length")

    out = Path(args.out) if args.out else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    all_data = [load_eval(p) for p in args.jsons]
    print(f"Loaded {len(all_data)} evaluation file(s)")

    plot_load_comparison(
        all_data, args.load_labels,
        args.method_keys, args.methods, out,
    )
    plot_paired_delta(
        all_data, args.load_labels,
        args.method_keys, args.methods,
        args.greedy_key, out,
    )


if __name__ == "__main__":
    main()
