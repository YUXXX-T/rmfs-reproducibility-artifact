"""Plot the out-of-fold behavior-aligned ``psi_pre`` diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from WorldModel.evaluation.phase_c_psi_pre_protocol import TARGET_CHANNELS


def _load(path: Path) -> tuple[dict, dict]:
    report = json.loads((path / "psi_pre_validation.json").read_text(encoding="utf-8"))
    arrays = torch.load(path / "oof_predictions.pt", map_location="cpu", weights_only=False)
    return report, arrays


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

    report, arrays = _load(args.root)
    channels = report["channels"]
    labels = [name.replace("region_", "") for name in TARGET_CHANNELS]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for name, key, color in (
        ("psi_pre", "psi_pre", "#1f77b4"),
        ("state decoder persistence", "decoder_state_persistence", "#9467bd"),
        ("decoder H=10", "decoder_h10", "#d62728"),
        ("H=1 persistence", "decoder_h1_persistence", "#7f7f7f"),
    ):
        values = [channels[channel][key]["pooled_spearman"] for channel in TARGET_CHANNELS]
        ax.plot(x, values, marker="o", linewidth=2, label=name, color=color)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(0.5, color="#2ca02c", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("OOF Spearman")
    ax.set_title("Behavior H=10 endpoint: station/region ordering")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, args.root / "figures" / "figure_01_oof_spearman")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    width = 0.20
    for offset, (name, key, color) in enumerate(
        (("psi_pre", "psi_pre", "#1f77b4"),
         ("state decoder persistence", "decoder_state_persistence", "#9467bd"),
         ("decoder H=10", "decoder_h10", "#d62728"),
         ("H=1 persistence", "decoder_h1_persistence", "#7f7f7f"))
    ):
        values = [channels[channel][key]["mae"] for channel in TARGET_CHANNELS]
        ax.bar(x + (offset - 1.5) * width, values, width, label=name, color=color)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Endpoint MAE")
    ax.set_title("Behavior H=10 endpoint calibration error")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, args.root / "figures" / "figure_02_endpoint_mae")
    plt.close(fig)

    pred = arrays["predictions"].numpy()
    target = arrays["targets"].numpy()
    frame = arrays["frame_group"].numpy()
    # Within-frame station rank is the key spatial test.  Show each channel's
    # OOF rank correlation, with a zero line for channels that cannot localise.
    within = []
    for index in range(len(TARGET_CHANNELS)):
        values = []
        for group in np.unique(frame):
            idx = np.flatnonzero(frame == group)
            if idx.size < 2:
                continue
            a = pred[idx, index]
            b = target[idx, index]
            ra = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort")
            rb = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort")
            if np.std(ra) > 0 and np.std(rb) > 0:
                values.append(float(np.corrcoef(ra, rb)[0, 1]))
        within.append(float(np.mean(values)) if values else np.nan)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    colors = ["#2ca02c" if value >= 0.5 else "#ff7f0e" for value in within]
    ax.bar(x, within, color=colors)
    ax.axhline(0.5, color="#2ca02c", linestyle="--", linewidth=1, label="pre-registered service bar")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Within decision-frame station Spearman")
    ax.set_title("Does psi_pre distinguish stations at the same decision state?")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, args.root / "figures" / "figure_03_within_station_rank")
    plt.close(fig)

    # Load-specific endpoint ordering for the two primary service channels and
    # the four max-style traffic channels.
    selected = [0, 1, 3, 5, 7, 9]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    load_names = ("low", "mid", "high")
    load_codes = arrays["load_code"].numpy()
    for axis, channel_index in zip(axes.flat, selected):
        values = []
        for code in range(3):
            idx = np.flatnonzero(load_codes == code)
            if idx.size < 2:
                values.append(np.nan)
            else:
                values.append(float(np.corrcoef(
                    np.argsort(np.argsort(pred[idx, channel_index], kind="mergesort"), kind="mergesort"),
                    np.argsort(np.argsort(target[idx, channel_index], kind="mergesort"), kind="mergesort"),
                )[0, 1]))
        axis.bar(load_names, values, color=["#4c78a8", "#f58518", "#e45756"])
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(labels[channel_index])
        axis.set_ylim(-1.0, 1.0)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("psi_pre OOF Spearman by load")
    fig.tight_layout()
    _save(fig, args.root / "figures" / "figure_04_load_breakdown")
    plt.close(fig)

    metrics = {
        "schema_version": "phase_c_psi_pre_plot_metrics_v1",
        "channels": list(TARGET_CHANNELS),
        "within_frame_station_spearman": dict(zip(TARGET_CHANNELS, within)),
        "kill_gate": report.get("kill_gate", {}),
    }
    (args.root / "figures").mkdir(parents=True, exist_ok=True)
    (args.root / "figures" / "psi_pre_plot_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"[complete] plots written under {args.root / 'figures'}")


if __name__ == "__main__":
    main()
