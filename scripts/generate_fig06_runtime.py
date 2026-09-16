"""Runtime-benchmark figure for the six-station 20x20 adaptation experiment.

Reads (read-only)
-----------------
artifacts/raw/figure_inputs/fig06_runtime_benchmark_runs.csv
artifacts/raw/figure_inputs/fig06_runtime_benchmark_pooled.csv

Outputs (under ``artifacts/generated/fig06`` by default)
---------------------------------------------------------
fig06_runtime_benchmark.{pdf,png}
    (a) steady-state ``task_assigner.assign`` wall latency per dispatcher,
        per held-out seed (median and p95 of 1,500 synchronised calls);
    (b) paired CPU vs GPU latency of the Proposed dispatcher, per seed;
    (c) what drives the Proposed latency: mean per-tick cost against the
        number of World-Model inference calls per tick.
fig06s_runtime_wall_time.{pdf,png}
    Supplementary: end-to-end wall time of the whole 1,500-tick episode
    against completed orders, showing that the low-throughput wall-time tail
    is shared by every dispatcher.
data/fig06_runtime_benchmark_runs.csv, data/fig06_runtime_benchmark_pooled.csv
    Every plotted value.

Run by FILE PATH with any python that has numpy/scipy/pandas/matplotlib
(no torch, nothing from the model side is imported)::

    python scripts/generate_fig06_runtime.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/raw/figure_inputs"
OUT = ROOT / "artifacts/generated/fig06"
DATA_OUT = OUT / "data"
RUNS_CSV = INPUT / "fig06_runtime_benchmark_runs.csv"
POOLED_CSV = INPUT / "fig06_runtime_benchmark_pooled.csv"
FROZEN_STATS = INPUT / "fig06_runtime_benchmark_stats.json"

# ---- paper palette (identical to generate_icra_figures.py) -----------------
COLORS = {
    "ink": "#1E2A35",
    "muted": "#737E87",
    "grid": "#D8DEE3",
    "paper": "#FFFFFF",
}
CONFIGS = ["greedy", "hungarian", "jsq", "proposed_cpu", "proposed_cuda_0"]
CONFIG_LABEL = {
    "greedy": "Greedy",
    "hungarian": "Hungarian",
    "jsq": "JSQ",
    "proposed_cpu": "Proposed (CPU)",
    "proposed_cuda_0": "Proposed (GPU)",
}
CONFIG_COLOR = {
    "greedy": "#8B969F",
    "hungarian": "#D28A18",
    "jsq": "#4B8B65",
    "proposed_cpu": "#0F4D92",
    "proposed_cuda_0": "#0F4D92",   # same entity, device encoded by fill
}
CONFIG_MARKER = {
    "greedy": "o",
    "hungarian": "s",
    "jsq": "^",
    "proposed_cpu": "*",
    "proposed_cuda_0": "*",
}
CONFIG_FILLED = {
    "greedy": True, "hungarian": True, "jsq": True,
    "proposed_cpu": True, "proposed_cuda_0": False,
}
LOW_THROUGHPUT_MARKER = 300.0


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.8,
            "axes.titlesize": 7.5,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.9,
            "legend.fontsize": 5.8,
            "axes.edgecolor": COLORS["muted"],
            "axes.linewidth": 0.65,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.45,
            "grid.alpha": 0.74,
            "figure.facecolor": COLORS["paper"],
            "savefig.facecolor": COLORS["paper"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=xgrid)
    ax.grid(axis="y", visible=ygrid)
    ax.tick_params(which="both", length=2.3, width=0.52, pad=1.8)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def marker_kwargs(cfg: str, size: float) -> dict:
    color = CONFIG_COLOR[cfg]
    if CONFIG_FILLED[cfg]:
        return dict(marker=CONFIG_MARKER[cfg], s=size, facecolor=color,
                    edgecolor=COLORS["paper"], linewidths=0.45)
    # hollow: transparent face so a coincident filled mark underneath stays visible
    return dict(marker=CONFIG_MARKER[cfg], s=size, facecolor="none",
                edgecolor=color, linewidths=0.7)


def legend_handle(cfg: str, label: str | None = None, size: float | None = None) -> Line2D:
    color = CONFIG_COLOR[cfg]
    mk = CONFIG_MARKER[cfg]
    ms = size if size is not None else (5.4 if mk == "*" else 3.8)
    if CONFIG_FILLED[cfg]:
        return Line2D([], [], marker=mk, linestyle="none", markersize=ms,
                      markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.3,
                      label=label or CONFIG_LABEL[cfg])
    return Line2D([], [], marker=mk, linestyle="none", markersize=ms,
                  markerfacecolor="none", markeredgecolor=color, markeredgewidth=0.8,
                  label=label or CONFIG_LABEL[cfg])


# ---- data ------------------------------------------------------------------
def load_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the anonymous, figure-complete extract of the benchmark.

    The collection JSON is intentionally not committed because its audit
    metadata contains workstation paths and static file fingerprints.  These
    two tables preserve every value plotted in Fig. 6/6s.
    """
    frame = pd.read_csv(RUNS_CSV).sort_values(["config", "seed"]).reset_index(drop=True)
    assert len(frame) == 50 and (frame.groupby("config").size() == 10).all()
    assert set(frame["config"]) == set(CONFIGS)
    assert set(frame["seed"].astype(int)) == set(range(721, 731))
    assert (frame["assign_calls"].astype(int) == 1500).all()

    pooled = pd.read_csv(POOLED_CSV).set_index("config").loc[CONFIGS]
    assert (pooled["calls"].astype(int) == 15000).all()
    return frame, pooled


def paired_stats(frame: pd.DataFrame) -> dict:
    cpu = frame[frame.config == "proposed_cpu"].set_index("seed")
    gpu = frame[frame.config == "proposed_cuda_0"].set_index("seed")
    assert (cpu.index == gpu.index).all()
    out = {}
    for stat in ("assign_median_ms", "assign_mean_ms", "assign_p95_ms", "wall_s"):
        a, b = cpu[stat].to_numpy(), gpu[stat].to_numpy()
        ratio = b / a
        out[stat] = {
            "gpu_over_cpu_median_ratio": float(np.median(ratio)),
            "ratio_min": float(ratio.min()),
            "ratio_max": float(ratio.max()),
            "gpu_faster_seeds": int((b < a).sum()),
            "wilcoxon_two_sided_p": float(stats.wilcoxon(a, b).pvalue),
        }
    # latency driver: mean per-tick cost vs World-Model inference calls per tick.
    # ``internal_inference_ms_mean`` is the policy's own timer divided by the
    # number of INFERENCE calls (evaluate_online_v6: mitt / mic), i.e. cost per
    # inference, not per assign call.  Share of inference inside assign() is
    # therefore (per-inference ms x inference calls) / assign total ms.
    for key, sub in (("cpu", cpu), ("gpu", gpu)):
        x = sub["model_inference_calls"].to_numpy(dtype=float) / sub["assign_calls"].to_numpy()
        y = sub["assign_mean_ms"].to_numpy()
        slope, intercept, r, p, _ = stats.linregress(x, y)
        inf_total_ms = sub["internal_inference_ms_mean"].to_numpy() * sub["model_inference_calls"].to_numpy()
        share = inf_total_ms / (sub["assign_total_s"].to_numpy() * 1000.0)
        out[f"driver_{key}"] = {
            "slope_ms_per_inference_call": float(slope),
            "intercept_ms_per_tick": float(intercept),
            "pearson_r": float(r), "p": float(p),
            "inference_calls_per_tick_min": float(x.min()),
            "inference_calls_per_tick_max": float(x.max()),
            "per_inference_ms_internal_median": float(np.median(sub["internal_inference_ms_mean"])),
            "inference_share_of_assign_total_median": float(np.median(share)),
            "inference_share_min": float(share.min()),
            "inference_share_max": float(share.max()),
        }
    return out


# ---- figure ----------------------------------------------------------------
def figure_runtime(frame: pd.DataFrame, pooled: pd.DataFrame, ps: dict) -> None:
    configure_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(7.15, 2.15),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0], "wspace": 0.42},
    )
    rng = np.random.default_rng(20260917)

    # (a) per-config latency rows: per-seed median (filled) -> p95 (hollow)
    ax = axes[0]
    ypos = {cfg: i for i, cfg in enumerate(reversed(CONFIGS))}
    for cfg in CONFIGS:
        sub = frame[frame.config == cfg].sort_values("seed")
        y0 = ypos[cfg]
        jitter = rng.uniform(-0.22, 0.22, size=len(sub))
        color = CONFIG_COLOR[cfg]
        for j, (_, r) in enumerate(sub.iterrows()):
            ax.plot([r.assign_median_ms, r.assign_p95_ms], [y0 + jitter[j]] * 2,
                    color=color, linewidth=0.55, alpha=0.55, solid_capstyle="round", zorder=2)
        ax.scatter(sub.assign_median_ms, y0 + jitter, zorder=4, **marker_kwargs(cfg, 15))
        ax.scatter(sub.assign_p95_ms, y0 + jitter, zorder=4, marker=CONFIG_MARKER[cfg], s=15,
                   facecolor=COLORS["paper"], edgecolor=color, linewidths=0.7)
        # pooled median over all 15,000 calls: ink tick
        pm = pooled.loc[cfg, "pooled_median_ms"]
        ax.plot([pm, pm], [y0 - 0.34, y0 + 0.34], color=COLORS["ink"], linewidth=1.05, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(0.008, 2000)
    ax.set_xticks([0.01, 0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100", "1000"])
    ax.set_yticks([ypos[c] for c in CONFIGS])
    ax.set_yticklabels([CONFIG_LABEL[c] for c in CONFIGS])
    ax.set_ylim(-0.6, len(CONFIGS) - 0.4 + 0.42)   # headroom for the top-row label
    ax.set_xlabel("assign() latency per tick (ms, log)")
    style_axis(ax, xgrid=True, ygrid=False)
    # selective direct labels: pooled medians for the two ends of the story
    for cfg in ("greedy", "proposed_cpu"):
        pm = pooled.loc[cfg, "pooled_median_ms"]
        ax.annotate(f"{pm:.1f} ms", xy=(pm, ypos[cfg] + 0.36), xytext=(0, 1.0),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=5.6, color=COLORS["ink"])
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=3.6, markerfacecolor=COLORS["muted"],
               markeredgecolor=COLORS["muted"], label="seed median"),
        Line2D([], [], marker="o", linestyle="none", markersize=3.6, markerfacecolor="none",
               markeredgecolor=COLORS["muted"], label="seed p95"),
        Line2D([], [], color=COLORS["ink"], linewidth=1.05, label="pooled median"),
    ]
    # lower-left is empty: Proposed rows never go below ~2 ms
    ax.legend(handles=handles, loc="lower left", frameon=False, handlelength=1.2,
              borderpad=0.2, labelspacing=0.25, handletextpad=0.5)
    ax.set_title("(a) Dispatch latency, 10 held-out seeds", loc="left")

    # (b) paired CPU vs GPU, Proposed
    ax = axes[1]
    cpu = frame[frame.config == "proposed_cpu"].set_index("seed")
    gpu = frame[frame.config == "proposed_cuda_0"].set_index("seed")
    lim = (1.0, 1500.0)
    ax.plot(lim, lim, color=COLORS["muted"], linewidth=0.7, zorder=1)
    stat_marks = [
        ("assign_median_ms", "o", "median"),
        ("assign_mean_ms", "D", "mean"),
        ("assign_p95_ms", "^", "p95"),
    ]
    for col, mk, _ in stat_marks:
        ax.scatter(cpu[col], gpu[col], marker=mk, s=15, facecolor=CONFIG_COLOR["proposed_cpu"],
                   edgecolor=COLORS["paper"], linewidths=0.45, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ticks = [1, 10, 100, 1000]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks]); ax.set_yticklabels([str(t) for t in ticks])
    ax.set_xlabel("CPU latency (ms, log)")
    ax.set_ylabel("GPU latency (ms, log)")
    ax.set_aspect("equal", adjustable="box")
    style_axis(ax, xgrid=True, ygrid=True)
    r_med = ps["assign_median_ms"]["gpu_over_cpu_median_ratio"]
    r_mean = ps["assign_mean_ms"]["gpu_over_cpu_median_ratio"]
    p_mean = ps["assign_mean_ms"]["wilcoxon_two_sided_p"]
    # data sit on the diagonal; both off-diagonal corners are free
    ax.text(0.96, 0.05,
            f"GPU/CPU: median {r_med:.2f}, mean {r_mean:.2f}\nWilcoxon p={p_mean:.2f} (mean)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=COLORS["ink"])
    handles = [Line2D([], [], marker=mk, linestyle="none", markersize=3.6,
                      markerfacecolor=CONFIG_COLOR["proposed_cpu"],
                      markeredgecolor=CONFIG_COLOR["proposed_cpu"], markeredgewidth=0.3,
                      label=lab) for _, mk, lab in stat_marks]
    ax.legend(handles=handles, loc="upper left", frameon=False, handlelength=1.0,
              borderpad=0.2, labelspacing=0.2, handletextpad=0.4)
    ax.set_title("(b) Proposed: CPU vs GPU", loc="left")

    # (c) latency driver: mean per-tick cost vs inference calls per tick
    ax = axes[2]
    for cfg in ("proposed_cpu", "proposed_cuda_0"):
        sub = frame[frame.config == cfg]
        x = sub.model_inference_calls.to_numpy(dtype=float) / sub.assign_calls.to_numpy()
        ax.scatter(x, sub.assign_mean_ms, zorder=3, **marker_kwargs(cfg, 24))
    d = ps["driver_cpu"]
    xs = np.linspace(1.0, 10.5, 50)
    ax.plot(xs, d["intercept_ms_per_tick"] + d["slope_ms_per_inference_call"] * xs,
            color=CONFIG_COLOR["proposed_cpu"], linewidth=0.9, alpha=0.75, zorder=2)
    ax.set_xlim(1.0, 10.8); ax.set_ylim(0, 150)   # headroom above the (10, ~117) point
    ax.set_xlabel("WM inference calls per tick")
    ax.set_ylabel("mean assign() latency (ms)")
    style_axis(ax, xgrid=False, ygrid=True)
    # upper-left is empty (points climb toward the upper right along the fit)
    ax.text(0.04, 0.96,
            f"CPU fit: {d['intercept_ms_per_tick']:.0f} ms + "
            f"{d['slope_ms_per_inference_call']:.1f} ms/call, r={d['pearson_r']:.2f}\n"
            f"WM forward pass ≈{100 * d['inference_share_of_assign_total_median']:.0f}% "
            f"of assign() time",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color=COLORS["ink"])
    handles = [legend_handle("proposed_cpu", "CPU"), legend_handle("proposed_cuda_0", "GPU")]
    ax.legend(handles=handles, loc="lower right", frameon=False, handlelength=1.0,
              borderpad=0.2, labelspacing=0.2, handletextpad=0.4)
    ax.set_title("(c) What drives Proposed latency", loc="left")

    save_figure(fig, "fig06_runtime_benchmark")


def figure_wall_time(frame: pd.DataFrame) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    for cfg in CONFIGS:
        sub = frame[frame.config == cfg]
        ax.scatter(sub.completed_orders, sub.wall_s, zorder=3, **marker_kwargs(cfg, 22))
    ax.axvline(LOW_THROUGHPUT_MARKER, color=COLORS["muted"], linewidth=0.7, zorder=1)
    # the low-throughput runs cluster at (150-220 orders, ~1100-1350 s); everything
    # left of the threshold line below ~800 s is empty, so the label goes there
    # (the legend owns the upper-right corner)
    ax.text(LOW_THROUGHPUT_MARKER - 10, 32, "low-throughput runs\n(< 300 orders)", ha="right", va="bottom",
            fontsize=5.6, color=COLORS["muted"])
    ax.set_yscale("log")
    ax.set_ylim(25, 2000)
    ax.set_yticks([30, 100, 300, 1000])
    ax.set_yticklabels(["30", "100", "300", "1000"])
    ax.set_xlim(100, 800)
    ax.set_xlabel("completed orders (1,500 ticks)")
    ax.set_ylabel("episode wall time (s, log)")
    style_axis(ax, xgrid=False, ygrid=True)
    handles = [legend_handle(c) for c in CONFIGS]
    # upper-right (right of ~550 orders, above ~700 s) holds no data: the
    # higher-throughput cluster sits at 40-250 s and the low-throughput cluster is far left
    ax.legend(handles=handles, loc="upper right", frameon=False,
              handlelength=1.0, borderpad=0.2, labelspacing=0.2, handletextpad=0.4,
              ncol=2, columnspacing=0.9)
    ax.set_title("Episode wall time tracks the low-throughput regime", loc="left",
                 fontsize=6.6)
    save_figure(fig, "fig06s_runtime_wall_time")


def write_extracts(frame: pd.DataFrame, pooled: pd.DataFrame, ps: dict) -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_OUT / "fig06_runtime_benchmark_runs.csv", index=False)
    pooled.to_csv(DATA_OUT / "fig06_runtime_benchmark_pooled.csv")
    (DATA_OUT / "fig06_runtime_benchmark_stats.json").write_text(
        json.dumps(ps, indent=2), encoding="utf-8")


def assert_stats_match(actual: dict, frozen: dict, path: str = "stats") -> None:
    """Compare recomputed statistics while tolerating CSV float round-off."""
    if set(actual) != set(frozen):
        raise AssertionError(f"{path} keys differ")
    for key in actual:
        left, right = actual[key], frozen[key]
        child = f"{path}.{key}"
        if isinstance(left, dict):
            if not isinstance(right, dict):
                raise AssertionError(f"{child} type differs")
            assert_stats_match(left, right, child)
        elif isinstance(left, float):
            if not np.isclose(left, right, rtol=1e-12, atol=1e-12):
                raise AssertionError(f"{child} differs: {left} != {right}")
        elif left != right:
            raise AssertionError(f"{child} differs: {left} != {right}")


def main() -> None:
    global OUT, DATA_OUT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="generated figure directory (default: artifacts/generated/fig06)",
    )
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    DATA_OUT = OUT / "data"

    frame, pooled = load_runs()
    ps = paired_stats(frame)
    frozen = json.loads(FROZEN_STATS.read_text(encoding="utf-8"))
    assert_stats_match(ps, frozen)
    write_extracts(frame, pooled, ps)
    figure_runtime(frame, pooled, ps)
    figure_wall_time(frame)
    print("hardware: Intel Xeon Gold 6240C | torch 2.5.1, 4 threads | RTX 4090")
    print(pooled[["pooled_median_ms", "pooled_mean_ms", "pooled_p95_ms", "pooled_max_ms",
                  "wall_median_s"]].round(2).to_string())
    print(json.dumps(ps, indent=2))
    print(f"[info] wrote {OUT / 'fig06_runtime_benchmark.pdf'} (+png)")
    print(f"[info] wrote {OUT / 'fig06s_runtime_wall_time.pdf'} (+png)")


if __name__ == "__main__":
    main()
