#!/usr/bin/env python3
"""Generate aggregate two-panel station-lock mechanism Figure 5."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("Greedy", "ComboS1J1")
COLORS = {"Greedy": "#67747C", "ComboS1J1": "#0F4D92"}
EXPECTED = {
    "Greedy": {"locks": 50, "collapsed": 22, "pairs": 22, "median": 445.0},
    "ComboS1J1": {"locks": 49, "collapsed": 15, "pairs": 14, "median": 460.0},
}


def optional_float(value: str) -> float | None:
    return None if value.strip() == "" else float(value)


def read_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                {
                    "load": row["load"],
                    "seed": int(row["seed"]),
                    "policy": row["policy"],
                    "lock_onset_tick": optional_float(row["lock_onset_tick"]),
                    "lock_onset_tick_share_0_9": optional_float(row["lock_onset_tick_share_0_9"]),
                    "collapse_tick": optional_float(row["collapse_tick"]),
                    "lead_time": optional_float(row["lead_time"]),
                    "collapse_detected": row["collapse_detected"].lower() == "true",
                    "paper_collapsed": row["paper_collapsed"].lower() == "true",
                    "completed_orders": float(row["completed_orders"]),
                    "deadlock_ratio_mean": float(row["deadlock_ratio_mean"]),
                }
            )
    return records


def validate(records: list[dict]) -> dict:
    if len(records) != 100:
        raise ValueError(f"expected 100 high-load policy rows, got {len(records)}")
    summary = {"schema_version": "fig05_mechanism_summary_v1", "denominator": 50, "groups": []}
    for policy in POLICIES:
        rows = [row for row in records if row["policy"] == policy]
        if len(rows) != 50 or {row["seed"] for row in rows} != set(range(900, 950)):
            raise ValueError(f"{policy}: incomplete seed block")
        endpoint = [row for row in rows if row["paper_collapsed"]]
        onset = [row for row in endpoint if row["collapse_tick"] is not None]
        pairs = [row for row in endpoint if row["lead_time"] is not None]
        locks = [row for row in rows if row["lock_onset_tick"] is not None]
        leads = np.asarray([row["lead_time"] for row in pairs], dtype=float)
        expected = EXPECTED[policy]
        observed = {
            "policy": policy,
            "n_seeds": len(rows),
            "lock_onsets_share_0_8": len(locks),
            "paper_collapsed_runs": len(endpoint),
            "paper_collapsed_with_onset": len(onset),
            "lead_pairs": len(pairs),
            "median_lead_ticks": float(np.median(leads)),
            "positive_leads": int(np.sum(leads > 0)),
        }
        if observed["lock_onsets_share_0_8"] != expected["locks"]:
            raise ValueError(f"{policy}: unexpected lock count")
        if observed["paper_collapsed_runs"] != expected["collapsed"]:
            raise ValueError(f"{policy}: unexpected collapse count")
        if observed["lead_pairs"] != expected["pairs"] or observed["median_lead_ticks"] != expected["median"]:
            raise ValueError(f"{policy}: unexpected lead distribution")
        if observed["positive_leads"] != len(pairs):
            raise ValueError(f"{policy}: non-positive paired lead found")
        summary["groups"].append(observed)
    return summary


def cumulative_step(events: list[float], denominator: int = 50, horizon: float = 1500.0):
    if not events:
        return np.asarray([0.0, horizon]), np.asarray([0.0, 0.0])
    ticks, counts = np.unique(np.asarray(events, dtype=float), return_counts=True)
    fractions = np.cumsum(counts) / denominator
    return (
        np.concatenate(([0.0], ticks, [horizon])),
        np.concatenate(([0.0], fractions, [fractions[-1]])),
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.4,
            "axes.edgecolor": "#737E87",
            "axes.linewidth": 0.65,
            "axes.grid": True,
            "grid.color": "#D8DEE3",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.74,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(axis, *, xgrid: bool = False) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", visible=xgrid)
    axis.grid(axis="y", visible=True)
    axis.tick_params(which="both", length=2.3, width=0.52, pad=1.8)


def draw_events(axis, records: list[dict]) -> None:
    for policy in POLICIES:
        rows = [row for row in records if row["policy"] == policy]
        lock_events = [row["lock_onset_tick"] for row in rows if row["lock_onset_tick"] is not None]
        collapse_events = [
            row["collapse_tick"]
            for row in rows
            if row["paper_collapsed"] and row["collapse_tick"] is not None
        ]
        for label, events, linestyle in (
            ("lock onset", lock_events, "-"),
            ("collapse onset", collapse_events, "--"),
        ):
            x, y = cumulative_step(events)
            axis.step(
                x,
                y,
                where="post",
                color=COLORS[policy],
                linestyle=linestyle,
                linewidth=1.45 if linestyle == "-" else 1.25,
                label=f"{policy} | {label}",
            )
    axis.set(xlim=(0, 1500), ylim=(0, 1.03), xlabel="Simulation tick", ylabel="Seed fraction with event")
    axis.set_xticks([0, 500, 1000, 1500])
    axis.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    axis.set_title("(a) Event accumulation", loc="left", weight="bold")
    axis.text(0.99, 0.04, "50 high-load seeds; denominator = 50", transform=axis.transAxes,
              ha="right", va="bottom", fontsize=6.3, color="#737E87")
    axis.legend(loc="upper left", frameon=False, ncol=2, columnspacing=0.9,
                handlelength=1.75, handletextpad=0.35)
    style_axis(axis)


def draw_leads(axis, records: list[dict]) -> None:
    rng = np.random.default_rng(20260910)
    for position, policy in enumerate(POLICIES):
        values = np.asarray(
            [row["lead_time"] for row in records if row["policy"] == policy and row["paper_collapsed"] and row["lead_time"] is not None],
            dtype=float,
        )
        color = COLORS[policy]
        violin = axis.violinplot([values], positions=[position], vert=False, widths=0.60,
                                 showmeans=False, showmedians=False, showextrema=False, bw_method=0.22)
        body = violin["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_linewidth(0.75)
        body.set_alpha(0.24 if policy == "Greedy" else 0.30)
        box = axis.boxplot([values], positions=[position], vert=False, widths=0.20,
                           patch_artist=True, showfliers=False, whis=(5, 95),
                           medianprops={"color": "#1E2A35", "linewidth": 1.0},
                           whiskerprops={"color": color, "linewidth": 0.75},
                           capprops={"color": color, "linewidth": 0.75},
                           boxprops={"edgecolor": color, "linewidth": 0.85})
        box["boxes"][0].set_facecolor("white")
        jitter = rng.uniform(-0.14, 0.14, size=len(values))
        axis.scatter(values, np.full(len(values), position) + jitter, s=8, color=color,
                     edgecolor="white", linewidth=0.28, alpha=0.72, zorder=4)
        median = float(np.median(values))
        axis.scatter([median], [position], marker="D", s=20, facecolor=color,
                     edgecolor="white", linewidth=0.65, zorder=5)
        axis.text(1490, position + 0.31, f"n={len(values)}, median={int(median)} ticks",
                  ha="right", va="bottom", fontsize=6.3, color=color, weight="bold")
    axis.axvline(0, color="#737E87", linewidth=0.75, zorder=1)
    axis.set(xlim=(-40, 1550), ylim=(-0.48, 1.48), xlabel=r"$\Delta t=t_{\mathrm{collapse}}-t_{\mathrm{lock}}$ (ticks)")
    axis.set_yticks([0, 1], list(POLICIES))
    axis.set_title("(b) Lead-time distribution", loc="left", weight="bold")
    style_axis(axis, xgrid=True)
    axis.grid(axis="y", visible=False)


def generate(input_path: Path, output_dir: Path) -> None:
    records = read_records(input_path)
    summary = validate(records)
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(5.20, 2.10), gridspec_kw={"width_ratios": [1.45, 1.0]})
    draw_events(axes[0], records)
    draw_leads(axes[1], records)
    figure.subplots_adjust(left=0.095, right=0.995, bottom=0.24, top=0.84, wspace=0.10)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "Aggregate station-lock mechanism", "Author": "Anonymous Authors", "Creator": "rmfs-wm-artifact"}
    figure.savefig(output_dir / "fig05_station_lock_mechanism.pdf", bbox_inches="tight", pad_inches=0.035, metadata=metadata)
    figure.savefig(output_dir / "fig05_station_lock_mechanism.png", dpi=450, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)
    (output_dir / "fig05_station_lock_mechanism.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/raw/station_lock_events.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/generated/figures")
    args = parser.parse_args()
    generate(args.input, args.output_dir)
    print(f"Wrote Fig. 5 under {args.output_dir}")


if __name__ == "__main__":
    main()
