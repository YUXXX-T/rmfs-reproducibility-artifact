"""Plot corrected psi_pre v2 OOF metrics without collapsing channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((args.root / "psi_pre_v2_validation.json").read_text(encoding="utf-8"))
    arrays = torch.load(args.root / "oof_predictions.pt", map_location="cpu", weights_only=False)
    channels = report["target_channels"]
    labels = [name.replace("region_", "") for name in channels]
    roles = report["diagnostic_channels"]
    x = np.arange(len(channels))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, key, color in (
        ("psi_pre v2", "psi_pre_v2", "#1f77b4"),
        ("corrected decoder H=10", "decoder_h10", "#d62728"),
        ("corrected state persistence", "decoder_state_persistence", "#9467bd"),
    ):
        values = [report["channels"][channel][key]["pooled_spearman"] for channel in channels]
        ax.plot(x, values, marker="o", linewidth=2, label=name, color=color)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(0.5, color="#2ca02c", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("OOF Spearman")
    ax.set_title("psi_pre v2: named station/traffic channels (no scalar collapse)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, args.root / "figures" / "figure_01_v2_spearman")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.25
    for offset, (name, key, color) in enumerate(
        (("psi_pre v2", "psi_pre_v2", "#1f77b4"),
         ("corrected decoder H=10", "decoder_h10", "#d62728"),
         ("corrected state persistence", "decoder_state_persistence", "#9467bd"))
    ):
        values = [report["channels"][channel][key]["mae"] for channel in channels]
        ax.bar(x + (offset - 1) * width, values, width, label=name, color=color)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Endpoint MAE (probability/normalized label scale)")
    ax.set_title("Corrected decoder baseline versus psi_pre v2")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, args.root / "figures" / "figure_02_v2_mae")
    plt.close(fig)

    pred = arrays["predictions"].numpy()
    target = arrays["targets"].numpy()
    frame = arrays["frame_group"].numpy()
    within = []
    for index in range(len(channels)):
        values = []
        for group in np.unique(frame):
            idx = np.flatnonzero(frame == group)
            if idx.size < 2:
                continue
            left = pred[idx, index]
            right = target[idx, index]
            left_rank = np.argsort(np.argsort(left, kind="mergesort"), kind="mergesort")
            right_rank = np.argsort(np.argsort(right, kind="mergesort"), kind="mergesort")
            if np.std(left_rank) > 0 and np.std(right_rank) > 0:
                values.append(float(np.corrcoef(left_rank, right_rank)[0, 1]))
        within.append(float(np.mean(values)) if values else np.nan)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#2ca02c" if value >= 0.5 else "#ff7f0e" for value in within]
    ax.bar(x, within, color=colors)
    ax.axhline(0.5, color="#2ca02c", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Within-frame station Spearman")
    ax.set_title("Station discrimination at the same decision frame")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(["service reference 0.50", "zero"], loc="best")
    _save(fig, args.root / "figures" / "figure_03_v2_within_station")
    plt.close(fig)

    metrics = {
        "schema_version": "phase_c_psi_pre_plot_metrics_v2",
        "target_channels": channels,
        "primary_channels": report["primary_channels"],
        "diagnostic_channels": roles,
        "within_frame_station_spearman": dict(zip(channels, within)),
        "diagnostic_gate": report.get("diagnostic_gate", {}),
    }
    (args.root / "figures").mkdir(parents=True, exist_ok=True)
    (args.root / "figures" / "psi_pre_v2_plot_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"[complete] v2 plots written under {args.root / 'figures'}")


if __name__ == "__main__":
    main()
