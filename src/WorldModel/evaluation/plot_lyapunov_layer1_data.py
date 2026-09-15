"""Create data-driven Layer-1 figures from the formal 401--403 raw arms.

The two figures answer separate questions:

1. Does the analytic state potential respond to observed unfinished-work load
   and its concentration across stations?
2. On observed quiet service transitions, does physical productive progress
   ever increase the work potential?

Unlike the overview audit graphic, these plots read the raw ``data.pt``
counterfactual samples and display empirical state/transition values.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_DATA_ROOT = Path(
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
    "lyapunov_td_impl/lyapunov_oracle_isolated_v3/formal_401_403"
)
DEFAULT_OUTPUT_DIR = Path(
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
    "lyapunov_td_impl/paper_figures/layer1"
)
LOADS = ("low", "mid", "high")
COLORS = {"low": "#2f6fb0", "mid": "#2a9d8f", "high": "#d97706"}


def _finite(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _infer_load(sample: Mapping, source: Path) -> str:
    value = sample.get("load") or sample.get("load_level")
    if value is not None and str(value).lower() in LOADS:
        return str(value).lower()
    text = " ".join((str(sample.get("run_id", "")), str(source))).lower()
    for load in LOADS:
        if re.search(rf"(?:^|[_/\\]){load}(?:[_/\\]|$)", text) or load in text:
            return load
    return "unknown"


def _load_samples(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected a list of samples")
        for value in payload:
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}: non-mapping sample")
            row = dict(value)
            row["_source_path"] = str(path)
            row["_load"] = _infer_load(row, path)
            rows.append(row)
    return rows


def _group_key(sample: Mapping) -> tuple[str, str]:
    return (
        str(sample.get("run_id") or sample.get("_source_path") or "unknown"),
        str(sample.get("candidate_group_id", "unknown")),
    )


def _state_rows(samples: Iterable[Mapping]) -> list[dict]:
    """Return one start state per candidate group to avoid candidate weighting."""

    unique: dict[tuple[str, str], dict] = {}
    for sample in samples:
        if sample.get("lyapunov_l0_valid") is False:
            continue
        snapshot = sample.get("lyapunov_l0_start")
        if not isinstance(snapshot, Mapping):
            continue
        components = snapshot.get("components")
        station_work = snapshot.get("station_work")
        if not isinstance(components, Mapping) or not isinstance(station_work, Mapping):
            continue
        key = _group_key(sample)
        if key in unique:
            continue
        work_values = np.asarray(
            [_finite(value) for value in station_work.values()], dtype=np.float64
        )
        if work_values.size == 0 or bool((work_values < 0.0).any()):
            continue
        total_work = float(work_values.sum())
        concentration = (
            float(np.square(work_values).sum() / (total_work * total_work))
            if total_work > 0.0
            else 0.0
        )
        unique[key] = {
            "load": str(sample.get("_load", "unknown")),
            "total": _finite(snapshot.get("total")),
            "work": _finite(components.get("work")),
            "station": _finite(components.get("station")),
            "arrival": _finite(components.get("arrival")),
            "total_station_work": total_work,
            "work_concentration": concentration,
            "work_capacity": _finite(snapshot.get("work_capacity"), 1.0),
            "num_stations": int(work_values.size),
        }
    return list(unique.values())


def _service_rows(samples: Iterable[Mapping], tolerance: float) -> list[dict]:
    rows: list[dict] = []
    for sample in samples:
        start = sample.get("lyapunov_l0_start")
        end = sample.get("lyapunov_l0_end")
        progress = sample.get("lyapunov_l0_progress")
        if not all(isinstance(value, Mapping) for value in (start, end, progress)):
            continue
        start_components = start.get("components")
        end_components = end.get("components")
        if not isinstance(start_components, Mapping) or not isinstance(end_components, Mapping):
            continue
        start_work = _finite(start_components.get("work"), float("nan"))
        end_work = _finite(end_components.get("work"), float("nan"))
        if not math.isfinite(start_work) or not math.isfinite(end_work):
            continue
        productive = _finite(progress.get("productive_total"))
        reverse = _finite(progress.get("reverse_total"))
        arrival = _finite(progress.get("arrival_total"))
        residual = _finite(progress.get("replan_residual_total"))
        quiet = max(abs(reverse), abs(arrival), abs(residual)) <= tolerance
        if productive <= tolerance or not quiet:
            continue
        threshold = tolerance * max(1.0, abs(start_work))
        rows.append({
            "load": str(sample.get("_load", "unknown")),
            "productive": productive,
            "start_work": start_work,
            "end_work": end_work,
            "delta_work": end_work - start_work,
            "violation": bool(end_work > start_work + threshold),
        })
    return rows


def _save(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _state_figure(states: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    fig.suptitle(
        "Layer 1 — Empirical State Potential under Isolated Collection",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0]
    values = [
        np.asarray([row["total"] for row in states if row["load"] == load])
        for load in LOADS
    ]
    parts = ax.violinplot(values, positions=np.arange(3), showextrema=False, widths=0.82)
    for body, load in zip(parts["bodies"], LOADS):
        body.set_facecolor(COLORS[load])
        body.set_edgecolor(COLORS[load])
        body.set_alpha(0.55)
    for index, (load, array) in enumerate(zip(LOADS, values)):
        q05, median, q95 = np.quantile(array, [0.05, 0.5, 0.95])
        ax.vlines(index, q05, q95, color=COLORS[load], linewidth=3)
        ax.scatter(index, median, s=55, color=COLORS[load], edgecolor="white", zorder=3)
        ax.text(index, q95, f" n={array.size}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(np.arange(3), LOADS)
    ax.set_ylabel("Start-state total potential  L(s)")
    ax.set_title("A. Observed state-potential distribution by load", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    for load in LOADS:
        rows = [row for row in states if row["load"] == load]
        ax.scatter(
            [row["total_station_work"] for row in rows],
            [row["work"] for row in rows],
            c=[row["work_concentration"] for row in rows],
            cmap="viridis",
            vmin=0.25,
            vmax=1.0,
            s=19,
            alpha=0.62,
            label=load,
            edgecolors="none",
        )
    capacities = [row["work_capacity"] for row in states if row["work_capacity"] > 0]
    station_counts = [row["num_stations"] for row in states if row["num_stations"] > 0]
    capacity = float(np.median(capacities))
    stations = int(round(float(np.median(station_counts))))
    xmax = max(row["total_station_work"] for row in states)
    grid = np.linspace(0.0, xmax, 200)
    balanced = 0.5 * np.square(grid / capacity) / max(stations, 1)
    concentrated = 0.5 * np.square(grid / capacity)
    ax.plot(grid, balanced, "--", color="#374151", linewidth=1.4, label="balanced lower envelope")
    ax.plot(grid, concentrated, ":", color="#374151", linewidth=1.4, label="single-station upper envelope")
    scalar = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0.25, 1.0))
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, pad=0.02)
    colorbar.set_label("Station-work concentration  ΣW²/(ΣW)²")
    ax.set_xlabel("Total effective remaining work  Σj Wj(s)")
    ax.set_ylabel("Observed work potential  Lwork(s)")
    ax.set_title("B. Work mass and concentration determine work potential", loc="left", fontweight="bold")
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    # The load scatter uses a continuous colour; keep only analytic envelopes.
    ax.legend(handles[-2:], labels[-2:], fontsize=8, frameon=False, loc="upper left")

    fig.text(
        0.5,
        0.015,
        f"One start state per candidate group; {len(states):,} independent state contexts. "
        f"Frozen analytic capacity Cwork={capacity:g}, stations={stations}.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.subplots_adjust(top=0.84, bottom=0.14, wspace=0.30)
    return fig


def _service_figure(rows: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    violations = sum(int(row["violation"]) for row in rows)
    fig.suptitle(
        "Layer 1 — Observed Physical-Service Transition Contract",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0]
    for load in LOADS:
        subset = [row for row in rows if row["load"] == load]
        ax.scatter(
            [row["start_work"] for row in subset],
            [row["end_work"] for row in subset],
            s=18,
            alpha=0.48,
            color=COLORS[load],
            label=f"{load} (n={len(subset)})",
            edgecolors="none",
        )
    maximum = max(max(row["start_work"], row["end_work"]) for row in rows)
    ax.plot([0, maximum], [0, maximum], "--", color="#111827", linewidth=1.4, label="no-dissipation boundary")
    ax.set_xlabel("Start  Lwork(s)")
    ax.set_ylabel("End  Lwork(s after H)")
    ax.set_title("A. Productive service keeps endpoint below the identity line", loc="left", fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    for load in LOADS:
        subset = [row for row in rows if row["load"] == load]
        ax.scatter(
            [row["productive"] for row in subset],
            [-row["delta_work"] for row in subset],
            s=18,
            alpha=0.48,
            color=COLORS[load],
            label=load,
            edgecolors="none",
        )
    ax.axhline(0.0, linestyle="--", color="#111827", linewidth=1.4)
    ax.set_xlabel("Observed productive work dissipation")
    ax.set_ylabel("Work-potential decrease  −ΔLwork")
    ax.set_title("B. Physical progress versus observed potential dissipation", loc="left", fontweight="bold")
    ax.grid(alpha=0.2)

    fig.text(
        0.5,
        0.015,
        f"Quiet service transitions: {len(rows):,}; observed violations: {violations}. "
        "Filtered to zero external arrival, reverse work and replan residual.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.subplots_adjust(top=0.84, bottom=0.14, wspace=0.27)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--strict-formal-401-403",
        action="store_true",
        help="require 9 arms, 3603 samples, 598 groups and 838 service transitions",
    )
    args = parser.parse_args()

    paths = sorted(args.data_root.glob("l1_v3_*_seed*/data.pt"))
    if not paths:
        raise FileNotFoundError(f"no formal data.pt arms under {args.data_root}")
    samples = _load_samples(paths)
    states = _state_rows(samples)
    service = _service_rows(samples, args.tolerance)
    load_counts = Counter(row["load"] for row in states)
    transition_counts = Counter(row["load"] for row in service)
    violations = sum(int(row["violation"]) for row in service)

    if args.strict_formal_401_403:
        expected = (9, 3603, 598, 838, 0)
        observed = (len(paths), len(samples), len(states), len(service), violations)
        if observed != expected:
            raise SystemExit(
                "formal Layer-1 data contract mismatch: "
                f"expected files/samples/groups/service/violations={expected}, "
                f"observed={observed}"
            )

    state_path = args.output_dir / "layer1_state_potential_data.png"
    service_path = args.output_dir / "layer1_service_transition_data.png"
    _save(_state_figure(states), state_path, args.dpi)
    _save(_service_figure(service), service_path, args.dpi)

    summary = {
        "schema_version": "lyapunov_layer1_data_figure_summary_v1",
        "data_root": str(args.data_root),
        "source_files": [str(path) for path in paths],
        "samples": len(samples),
        "candidate_group_start_states": len(states),
        "state_groups_by_load": dict(sorted(load_counts.items())),
        "quiet_productive_service_transitions": len(service),
        "service_transitions_by_load": dict(sorted(transition_counts.items())),
        "service_monotonicity_violations": violations,
        "figures": [str(state_path), str(service_path)],
    }
    summary_path = args.output_dir / "layer1_data_figure_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {state_path}")
    print(f"saved: {state_path.with_suffix('.svg')}")
    print(f"saved: {service_path}")
    print(f"saved: {service_path.with_suffix('.svg')}")
    print(f"saved: {summary_path}")
    print(
        "formal counts: "
        f"files={len(paths)} samples={len(samples)} groups={len(states)} "
        f"service={len(service)} violations={violations}"
    )


if __name__ == "__main__":
    main()
