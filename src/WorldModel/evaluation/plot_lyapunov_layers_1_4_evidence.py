"""Plot the frozen evidence chain for Lyapunov Layers 1--4.

The figure intentionally consumes immutable JSON reports rather than raw
``data.pt`` files, so it can be regenerated on a lightweight review machine
without loading PyTorch checkpoints or simulation tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
    "lyapunov_td_impl"
)
DEFAULT_L12 = (
    ROOT
    / "lyapunov_oracle_isolated_v3/formal_401_403/"
    "five_layer_l1_l3_load_component_v2_dev.json"
)
DEFAULT_L3 = (
    ROOT
    / "lyapunov_oracle_isolated_v3/cert_work_group_range_421_430/"
    "five_layer_l1_l3_work_group_range_v2_cert_421_430.json"
)
DEFAULT_L4 = (
    ROOT
    / "layer4_work_drift_group_range_v1/"
    "work_drift_group_range_v1_test_421_430.json"
)
DEFAULT_OUTPUT = ROOT / "lyapunov_layers_1_4_evidence.png"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _errorbar(ax, labels, values, intervals, *, color, reference=None):
    y = np.arange(len(labels))
    lower = np.array([v - ci[0] for v, ci in zip(values, intervals)])
    upper = np.array([ci[1] - v for v, ci in zip(values, intervals)])
    ax.errorbar(
        values,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=2.2,
        capsize=4,
        markersize=7,
        zorder=3,
    )
    if reference is not None:
        ax.axvline(reference, color="#7f8c8d", linestyle="--", linewidth=1.3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    for yi, value in zip(y, values):
        ax.annotate(
            f"{value:.3f}",
            (value, yi),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
        )


def build_figure(l12: dict, l3: dict, l4: dict):
    blue = "#2463a7"
    cyan = "#2a9d8f"
    green = "#2f855a"
    orange = "#d97706"
    gray = "#6b7280"

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.5))
    fig.subplots_adjust(
        left=0.13,
        right=0.975,
        top=0.89,
        bottom=0.175,
        wspace=0.30,
        hspace=0.54,
    )
    fig.suptitle(
        "Lyapunov Work-Drift Evidence Chain (Layers 1–4)",
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.925,
        "Isolated analytic potential → controllable action drift → incremental value over WM → latent drift estimator",
        ha="center",
        fontsize=10.5,
        color="#374151",
    )

    # Layer 1: the contract is primarily an audited physical/semantic object.
    ax = axes[0, 0]
    layer1 = l12["layer1_state_potential"]
    checks = [
        ("Isolated semantics", layer1["isolated_semantics_audit"]["passed"]),
        ("Analytic invariants", layer1["analytic_invariant_audit"]["passed"]),
        ("Stored formula", layer1["stored_formula_audit"]["passed"]),
        ("Observed transitions", layer1["observed_transition_contract"]["passed"]),
    ]
    labels = [name for name, _ in checks]
    values = [float(passed) for _, passed in checks]
    y = np.arange(len(labels))
    ax.barh(y, values, color=green, height=0.58)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.13)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "PASS"])
    ax.grid(axis="x", alpha=0.18)
    for yi, passed in zip(y, values):
        ax.text(passed + 0.025, yi, "PASS" if passed else "FAIL", va="center", color=green)
    ax.set_title("Layer 1 | State potential contract", loc="left", fontweight="bold")
    ax.text(
        0.0,
        -0.19,
        f"Status: {layer1['status']}\n"
        f"Evidence base: {l12['samples']:,} samples, {l12['candidate_groups']:,} candidate groups, "
        "9 isolated load-seed arms (401–403)",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        va="top",
    )

    # Layer 2: action-controllable spread, shown by load.
    ax = axes[0, 1]
    layer2 = l12["layer2_action_controllable_drift"]
    loads = ["low", "mid", "high"]
    ranges = [
        layer2["by_load"][load]["normalised_within_group_drift_range"]["mean"]
        for load in loads
    ]
    varying_rates = [layer2["by_load"][load]["varying_group_rate"] for load in loads]
    bars = ax.bar(loads, ranges, color=[blue, cyan, orange], width=0.62)
    overall_mean = layer2["normalised_within_group_drift_range"]["mean"]
    ci = layer2["normalised_range_mean_ci95_cluster_bootstrap"]
    ax.axhspan(ci[0], ci[1], color=gray, alpha=0.12, label="overall seed-cluster 95% CI")
    ax.axhline(overall_mean, color=gray, linestyle="--", linewidth=1.4)
    for bar, value, rate in zip(bars, ranges, varying_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.055,
            f"range={value:.2f}\nvarying={rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_ylabel("Mean normalized within-group drift range")
    ax.set_ylim(0, max(ranges) * 1.30)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("Layer 2 | Action-controllable short-horizon drift", loc="left", fontweight="bold")
    ax.text(
        0.0,
        -0.19,
        f"{layer2['varying_groups']}/{layer2['groups']} groups vary ({layer2['varying_group_rate']:.1%}); "
        f"overall mean CI [{ci[0]:.2f}, {ci[1]:.2f}] across 3 seed clusters",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        va="top",
    )

    # Layer 3: frozen incremental information over the true WM score.
    ax = axes[1, 0]
    layer3 = l3["layer3_incremental_information"]
    contract = layer3["formal_candidate_contract"]
    primary = contract["primary_evidence"]
    l3_labels = ["Overall", "Low", "Mid", "High"]
    l3_values = [
        primary["normalised_rmse_improvement"],
        primary["load_evidence"]["low"]["normalised_rmse_improvement"],
        primary["load_evidence"]["mid"]["normalised_rmse_improvement"],
        primary["load_evidence"]["high"]["normalised_rmse_improvement"],
    ]
    l3_intervals = [
        primary["normalised_rmse_improvement_ci95"],
        primary["load_evidence"]["low"]["ci95"],
        primary["load_evidence"]["mid"]["ci95"],
        primary["load_evidence"]["high"]["ci95"],
    ]
    _errorbar(ax, l3_labels, l3_values, l3_intervals, color=blue, reference=0.0)
    ax.set_xlim(-0.015, 0.29)
    ax.set_xlabel("Normalized RMSE improvement over pure WM (higher is better)")
    ranking = primary["ranking"]
    ax.set_title("Layer 3 | Frozen incremental information", loc="left", fontweight="bold")
    ax.text(
        0.0,
        -0.30,
        "Top-1 accuracy: "
        f"{ranking['wm_only']['top1_accuracy']:.3f} → {ranking['wm_plus_predictor']['top1_accuracy']:.3f};  "
        "pairwise concordance: "
        f"{ranking['wm_only']['pairwise_concordance']:.3f} → "
        f"{ranking['wm_plus_predictor']['pairwise_concordance']:.3f}\n"
        f"10 unseen seed clusters (421–430), secondary support "
        f"{contract['supportive_secondary_outcomes']}/3, guardrails passed",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        va="top",
    )

    # Layer 4: held-out endpoint estimator quality and ranking preservation.
    ax = axes[1, 1]
    metric = l4["validation"]["continuous_group_range"]
    l4_labels = ["NRMSE ↓", "Spearman ↑", "Pairwise ↑", "Top-1 accuracy ↑", "Regret improvement ↑"]
    l4_values = [
        metric["normalised_rmse"],
        metric["spearman"],
        metric["all_non_tie_pair_concordance"],
        metric["top1_min_drift_accuracy"],
        metric["normalised_selection_regret_improvement_over_uniform_random"]["mean"],
    ]
    l4_intervals = [
        metric["normalised_rmse_ci95_seed_cluster_bootstrap"],
        metric["spearman_ci95_seed_cluster_bootstrap"],
        metric["all_non_tie_pair_concordance_ci95_seed_cluster_bootstrap"],
        metric["top1_min_drift_accuracy_ci95_seed_cluster_bootstrap"],
        metric[
            "normalised_selection_regret_improvement_over_uniform_random_ci95_seed_cluster_bootstrap"
        ],
    ]
    _errorbar(ax, l4_labels, l4_values, l4_intervals, color=green)
    baselines = [1.0, 0.0, 0.5, metric["random_tie_aware_top1_baseline"], 0.0]
    y = np.arange(len(l4_labels))
    ax.scatter(baselines, y, marker="x", s=55, color=gray, label="null/random reference", zorder=4)
    ax.set_xlim(-0.04, 1.07)
    ax.set_xlabel("Held-out metric value (dot = estimate; bar = seed-cluster 95% CI)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Layer 4 | WM-latent work-drift estimator", loc="left", fontweight="bold")
    ax.text(
        0.0,
        -0.30,
        f"{l4['validation']['samples']:,} candidates in {l4['validation']['groups']:,} groups; "
        "held-out seeds 421–430, low/mid/high\n"
        "Training: 411–418; development/early-stop: 419–420; all formal metric checks passed",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        va="top",
    )

    fig.text(
        0.5,
        0.025,
        "Evidence reports: formal_401_403 / cert_work_group_range_421_430 / layer4_work_drift_group_range_v1",
        ha="center",
        fontsize=8.5,
        color="#6b7280",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer12-report", type=Path, default=DEFAULT_L12)
    parser.add_argument("--layer3-report", type=Path, default=DEFAULT_L3)
    parser.add_argument("--layer4-report", type=Path, default=DEFAULT_L4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    for path in (args.layer12_report, args.layer3_report, args.layer4_report):
        if not path.is_file():
            raise FileNotFoundError(path)

    figure = build_figure(
        _read(args.layer12_report),
        _read(args.layer3_report),
        _read(args.layer4_report),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, facecolor="white")
    svg = args.output.with_suffix(".svg")
    figure.savefig(svg, facecolor="white")
    plt.close(figure)
    print(f"saved: {args.output}")
    print(f"saved: {svg}")


if __name__ == "__main__":
    main()
