"""
ICRA Training Curves — Stage 2 Focus
=====================================
Reads one or more train_log_stage2.jsonl files and produces:

  fig_training_stage2.pdf  — 3 subplots:
    (a) Train loss + ranking loss
    (b) Val rank accuracy
    (c) Val top-1 regret

Usage (multi-run comparison):
    python -m WorldModel.visualize.plot_training_curves ^
      --stage2-logs run1/train_log_stage2.jsonl ^
                    run2/train_log_stage2.jsonl ^
                    run3/train_log_stage2.jsonl ^
      --run-names "Old cost" "Frozen old" "Congestion" ^
      --out WorldModel/visualize/output

Single run:
    python -m WorldModel.visualize.plot_training_curves ^
      --stage2-logs run3/train_log_stage2.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from WorldModel.visualize.icra_style import (
    apply_icra_style,
    save_fig,
    label_subplot,
)

OUTPUT_DIR = Path(__file__).parent / "output"

RUN_COLORS = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"]


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _extract(records, key):
    return [r.get(key) for r in records]


def plot_stage2(
    all_logs: List[List[dict]],
    run_names: List[str],
    out: Path,
):
    n_runs = len(all_logs)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))

    for i, (records, name) in enumerate(zip(all_logs, run_names)):
        col = RUN_COLORS[i % len(RUN_COLORS)]
        epochs = _extract(records, "epoch")

        # (a) Loss
        ax = axes[0]
        loss = _extract(records, "loss")
        rank = _extract(records, "rank")
        ax.plot(epochs, loss, color=col, linewidth=1.2,
                label=f"{name}" if n_runs > 1 else "Total loss")
        if rank and any(r is not None for r in rank):
            rank_clean = [r if r is not None else 0 for r in rank]
            ax.plot(epochs, rank_clean, color=col, linestyle=":",
                    linewidth=0.9, alpha=0.7,
                    label=f"{name} rank" if n_runs > 1 else "Rank loss")

        # (b) Val rank accuracy
        ax = axes[1]
        val_acc = _extract(records, "val_rank_accuracy")
        if val_acc and any(v is not None for v in val_acc):
            valid = [(e, v) for e, v in zip(epochs, val_acc) if v is not None]
            if valid:
                es, vs = zip(*valid)
                ax.plot(es, vs, color=col, linewidth=1.2,
                        label=name if n_runs > 1 else None)
                best_idx = int(np.argmax(vs))
                ax.plot(es[best_idx], vs[best_idx], "v", color=col,
                        markersize=5, zorder=5)
                ax.annotate(
                    f"{vs[best_idx]:.3f}",
                    (es[best_idx], vs[best_idx]),
                    textcoords="offset points", xytext=(4, -10),
                    fontsize=6, color=col,
                )

        # (c) Val top-1 regret
        ax = axes[2]
        regret = _extract(records, "val_top1_regret_mean")
        if regret and any(r is not None for r in regret):
            valid = [(e, r) for e, r in zip(epochs, regret) if r is not None]
            if valid:
                es, rs = zip(*valid)
                ax.plot(es, rs, color=col, linewidth=1.2,
                        label=name if n_runs > 1 else None)
                best_idx = int(np.argmin(rs))
                ax.plot(es[best_idx], rs[best_idx], "^", color=col,
                        markersize=5, zorder=5)
                ax.annotate(
                    f"{rs[best_idx]:.3f}",
                    (es[best_idx], rs[best_idx]),
                    textcoords="offset points", xytext=(4, 6),
                    fontsize=6, color=col,
                )

    # ── Frozen-phase background shading ────────────────────────────
    freeze_end = 0
    for records in all_logs:
        frozen = _extract(records, "frozen")
        epochs = _extract(records, "epoch")
        for e, f in zip(epochs, frozen):
            if f and e is not None:
                freeze_end = max(freeze_end, e)
    if freeze_end > 0:
        for ax in axes:
            ax.axvspan(0, freeze_end + 0.5, color="#cccccc", alpha=0.25, zorder=0)
        axes[0].text(
            freeze_end * 0.5, 0.97, "frozen", transform=axes[0].get_xaxis_transform(),
            ha="center", va="top", fontsize=6, color="#666666", style="italic",
        )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train Loss", fontsize=8, pad=3)
    label_subplot(axes[0], "a")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Rank Accuracy")
    axes[1].set_title("Val Rank Accuracy", fontsize=8, pad=3)
    label_subplot(axes[1], "b")

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Top-1 Regret")
    axes[2].set_title("Val Top-1 Regret", fontsize=8, pad=3)
    label_subplot(axes[2], "c")

    if n_runs > 1:
        axes[1].legend(fontsize=6, loc="best", framealpha=0.8)
    axes[0].legend(fontsize=6, loc="best", framealpha=0.8)

    fig.tight_layout(w_pad=1.0)
    save_fig(fig, str(out / "fig_training_stage2"))
    print(f"  Saved fig_training_stage2.pdf/.png")


def main():
    apply_icra_style()

    parser = argparse.ArgumentParser(
        description="ICRA Training Curves — Stage 2",
    )
    parser.add_argument(
        "--stage2-logs", nargs="+", required=True,
        help="train_log_stage2.jsonl files (one per run)",
    )
    parser.add_argument(
        "--run-names", nargs="+", default=None,
        help="Display names for each run",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.run_names is None:
        if len(args.stage2_logs) == 1:
            args.run_names = ["Stage 2"]
        else:
            args.run_names = [f"Run {i+1}" for i in range(len(args.stage2_logs))]
    if len(args.run_names) != len(args.stage2_logs):
        parser.error("--run-names must match --stage2-logs length")

    out = Path(args.out) if args.out else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    all_logs = []
    for p in args.stage2_logs:
        records = load_jsonl(p)
        all_logs.append(records)
        print(f"  Loaded {len(records)} epochs from {p}")

    plot_stage2(all_logs, args.run_names, out)


if __name__ == "__main__":
    main()
