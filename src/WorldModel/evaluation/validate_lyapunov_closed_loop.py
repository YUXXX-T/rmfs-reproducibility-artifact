"""Validate analytic Lyapunov behaviour on normal-arrival online streams.

This tool consumes ``tdstream_*.pt`` files produced by
``evaluate_online_v6.py --td-stream-dir --td-stream-lyapunov-l0``.  It checks
the physical work ledger, reports both a continuous current-L/drift relation
and descriptive current-L quantiles, and aggregates drift/backlog slopes
across independent runs.  Quantile regions describe the Foster-style compact
set; they are not raw action-gap gates.

The verdict is deliberately policy-scoped.  Positive closed-loop drift can
show that the evaluated policy/load pair is unstable; by itself it does not
prove that the analytic functional is the cause.  Analytic action-selection
value must be established separately by
``evaluate_lyapunov_validity.py``.  The legacy oracle report remains useful
for historical diagnostics only.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch


SCHEMA_VERSION = "lyapunov_closed_loop_validation_v2"
L_COMPONENT_NAMES = (
    "L_work", "L_station", "L_traffic", "L_stall", "L_arrival",
)
L_COMPONENT_WEIGHT_FIELDS = {
    "L_work": ("work_weight",),
    "L_station": ("station_weight",),
    "L_traffic": ("traffic_weight",),
    # The stored stall component combines stationary-stall and plan-failure
    # terms in the current analytic functional.
    "L_stall": ("stall_weight", "plan_fail_weight"),
    "L_arrival": ("arrival_weight",),
}


def _parse_int_list(text: str) -> Tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        )
    return values


def _expand_paths(values: Iterable[str]) -> list[str]:
    result = []
    for value in values:
        matches = sorted(glob.glob(value)) if any(ch in value for ch in "*?") else []
        for path in matches or [value]:
            resolved = str(Path(path).expanduser().resolve())
            if resolved not in result:
                result.append(resolved)
    return result


def _quantiles(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        end = cursor + 1
        while end < values.size and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + end - 1) + 1.0
        cursor = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or right.size != left.size:
        return None
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if left_rank.std() <= 1e-12 or right_rank.std() <= 1e-12:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _moving_block_mean_ci(
    values: np.ndarray,
    *,
    repeats: int,
    block_length: int,
    seed: int,
) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    if values.size == 1 or repeats <= 1:
        value = float(values.mean())
        return [value, value]
    block = max(1, min(int(block_length), values.size))
    blocks_needed = int(math.ceil(values.size / block))
    max_start = values.size - block
    rng = np.random.default_rng(int(seed))
    means = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        sample = np.concatenate([
            values[start:start + block] for start in starts
        ])[:values.size]
        means[repeat] = sample.mean()
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _independent_mean_ci(
    values: Sequence[float], *, repeats: int, seed: int,
) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    if array.size == 1 or repeats <= 1:
        value = float(array.mean())
        return [value, value]
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(repeats, array.size))
    means = array[indices].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _linear_slope(ticks: np.ndarray, values: np.ndarray) -> float | None:
    if ticks.size < 3 or values.size != ticks.size:
        return None
    x = ticks.astype(np.float64)
    y = values.astype(np.float64)
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    if denominator <= 1e-12:
        return None
    return float(np.dot(x, y - y.mean()) / denominator)


def _linear_relation(
    predictor: np.ndarray,
    response: np.ndarray,
) -> dict:
    """Fit a continuous response relation without a raw-value gate.

    The returned slope is scale-covariant, while Pearson/Spearman are
    invariant to multiplying the Lyapunov functional by a positive constant.
    This is descriptive evidence for a Foster-style drift relation; it is not
    a proof that one fitted straight line globally bounds the stochastic
    system.
    """
    x = np.asarray(predictor, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    if x.size < 3 or y.size != x.size:
        return {"n": int(x.size)}
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    centered_x = x - x_mean
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 1e-12:
        return {
            "n": int(x.size),
            "intercept": y_mean,
            "slope": None,
            "pearson": None,
            "spearman": _spearman(x, y),
            "predictor_std": float(x.std()),
            "response_std": float(y.std()),
        }
    slope = float(np.dot(centered_x, y - y_mean) / denominator)
    intercept = float(y_mean - slope * x_mean)
    pearson = None
    if x.std() > 1e-12 and y.std() > 1e-12:
        pearson = float(np.corrcoef(x, y)[0, 1])
    return {
        "n": int(x.size),
        "intercept": intercept,
        "slope": slope,
        "pearson": pearson,
        "spearman": _spearman(x, y),
        "predictor_std": float(x.std()),
        "response_std": float(y.std()),
    }


def _moving_block_relation_ci(
    predictor: np.ndarray,
    response: np.ndarray,
    *,
    repeats: int,
    block_length: int,
    seed: int,
) -> dict | None:
    """Moving-block bootstrap CI for the continuous drift relation."""
    x = np.asarray(predictor, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    if x.size < 3 or y.size != x.size or repeats <= 0:
        return None
    block = max(1, min(int(block_length), x.size))
    blocks_needed = int(math.ceil(x.size / block))
    max_start = x.size - block
    rng = np.random.default_rng(int(seed))
    slopes = []
    p95_predictions = []
    reference_p95 = float(np.quantile(x, 0.95))
    for _ in range(repeats):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        indices = np.concatenate([
            np.arange(start, start + block) for start in starts
        ])[:x.size]
        relation = _linear_relation(x[indices], y[indices])
        slope = relation.get("slope")
        intercept = relation.get("intercept")
        if slope is None or intercept is None:
            continue
        slopes.append(float(slope))
        p95_predictions.append(float(intercept + slope * reference_p95))
    if not slopes:
        return None
    return {
        "slope_ci95": [
            float(np.quantile(slopes, 0.025)),
            float(np.quantile(slopes, 0.975)),
        ],
        "predicted_drift_at_observed_L_p95_ci95": [
            float(np.quantile(p95_predictions, 0.025)),
            float(np.quantile(p95_predictions, 0.975)),
        ],
        "observed_L_p95": reference_p95,
        "bootstrap_repeats_used": len(slopes),
    }


def _future_window_mean(values: np.ndarray, horizon: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= horizon:
        return np.empty(0, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    starts = np.arange(values.size - horizon)
    return (
        cumulative[starts + horizon + 1] - cumulative[starts + 1]
    ) / float(horizon)


def _infer_load(payload: Mapping, source_path: str) -> str:
    text = " ".join((
        str(payload.get("config", "")),
        str(payload.get("run_id", "")),
        source_path,
    )).lower()
    match = re.search(r"(?:^|[_/\\])(low|mid|high)(?:[_/\\.]|$)", text)
    return match.group(1) if match else "unknown"


def _column(array: np.ndarray, names: Sequence[str], name: str) -> np.ndarray:
    if name not in names:
        raise ValueError(f"stream lacks required column {name!r}")
    return array[:, list(names).index(name)]


def _configured_active_components(
    payload: Mapping,
    observed_components: Mapping[str, np.ndarray],
) -> tuple[list[str], str]:
    """Return components that are part of the configured functional.

    Coverage must not require deliberately disabled terms.  In particular,
    the L1/v3 defaults assign zero weight to traffic and stall, so treating
    their all-zero trajectories as missing evidence would make Layer 5
    impossible to pass by construction.  Legacy streams without the frozen
    config fall back to observed non-zero components and are labelled as
    inferred rather than configured.
    """
    config = payload.get("lyapunov_l0_config")
    if isinstance(config, Mapping):
        active = []
        for component, fields in L_COMPONENT_WEIGHT_FIELDS.items():
            weights = [float(config.get(field, 0.0)) for field in fields]
            if any(weight > 0.0 for weight in weights):
                active.append(component)
        return active, "lyapunov_l0_config"
    active = [
        name for name, values in observed_components.items()
        if bool((np.abs(values) > 1e-12).any())
    ]
    return active, "inferred_from_observed_nonzero_values"


def analyze_stream(
    payload: Mapping,
    *,
    source_path: str = "",
    drift_horizons: Sequence[int] = (1, 5, 10, 25, 50),
    primary_horizon: int = 10,
    risk_horizon: int = 25,
    bins: int = 5,
    high_quantile: float = 0.80,
    burn_in: int = 50,
    bootstrap_repeats: int = 1000,
    min_bin_points: int = 20,
    invariant_tolerance: float = 1e-4,
    random_seed: int = 2026,
) -> dict:
    if not payload.get("lyapunov_l0_enabled", False):
        raise ValueError(f"{source_path}: Lyapunov L0 stream recording is disabled")
    required = (
        "tick_seq", "lyapunov_l0_summary", "lyapunov_l0_summary_names",
        "productive_progress", "productive_progress_names", "risk_seq_full",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"{source_path}: missing stream fields {missing}")

    ticks = torch.as_tensor(payload["tick_seq"]).detach().cpu().numpy().astype(
        np.float64
    )
    summary = torch.as_tensor(
        payload["lyapunov_l0_summary"]
    ).detach().cpu().numpy().astype(np.float64)
    summary_names = list(payload["lyapunov_l0_summary_names"])
    progress = torch.as_tensor(
        payload["productive_progress"]
    ).detach().cpu().numpy().astype(np.float64)
    progress_names = list(payload["productive_progress_names"])
    risk = torch.as_tensor(
        payload["risk_seq_full"]
    ).detach().cpu().numpy().astype(np.float64)

    lengths = {ticks.size, summary.shape[0], progress.shape[0], risk.size}
    if len(lengths) != 1:
        raise ValueError(f"{source_path}: inconsistent per-tick stream lengths")
    n = int(ticks.size)
    if n < 2:
        raise ValueError(f"{source_path}: stream is too short")

    components = {
        name: _column(summary, summary_names, name)
        for name in L_COMPONENT_NAMES
    }
    total = _column(summary, summary_names, "L_total")
    work = _column(summary, summary_names, "station_work_sum")
    all_arrays = [ticks, summary.reshape(-1), progress.reshape(-1), risk]
    finite_pass = all(np.isfinite(array).all() for array in all_arrays)
    nonnegative_min = min(
        [float(total.min())]
        + [float(values.min()) for values in components.values()]
    )
    nonnegative_pass = nonnegative_min >= -float(invariant_tolerance)
    reconstructed = sum(components.values())
    decomposition_error = np.abs(total - reconstructed)

    productive = _column(progress, progress_names, "productive")
    reverse = _column(progress, progress_names, "reverse")
    arrival = _column(progress, progress_names, "arrival")
    residual = _column(progress, progress_names, "replan_residual")
    expected_work_delta = arrival[1:] - productive[1:] + reverse[1:] + residual[1:]
    work_balance_error = np.diff(work) - expected_work_delta
    invariant_pass = bool(
        finite_pass
        and nonnegative_pass
        and float(decomposition_error.max()) <= invariant_tolerance
        and float(np.abs(work_balance_error).max()) <= invariant_tolerance
    )

    start_index = min(max(int(burn_in), 0), max(n - 2, 0))
    drift = {}
    for horizon in sorted(set(int(value) for value in drift_horizons)):
        indices = np.arange(start_index, n - horizon, dtype=np.int64)
        if indices.size == 0:
            drift[str(horizon)] = {"n": 0}
            continue
        state_l = total[indices]
        delta = total[indices + horizon] - total[indices]
        continuous_relation = _linear_relation(state_l, delta)
        relation_ci = _moving_block_relation_ci(
            state_l,
            delta,
            repeats=bootstrap_repeats,
            block_length=max(horizon, 5),
            seed=random_seed + 1877 * horizon,
        )
        if (
            continuous_relation.get("slope") is not None
            and continuous_relation.get("intercept") is not None
        ):
            observed_l_p95 = float(np.quantile(state_l, 0.95))
            continuous_relation[
                "predicted_drift_at_observed_L_p95"
            ] = float(
                continuous_relation["intercept"]
                + continuous_relation["slope"] * observed_l_p95
            )
            continuous_relation["observed_L_p95"] = observed_l_p95
        continuous_relation["moving_block_bootstrap"] = relation_ci
        edges = np.quantile(state_l, np.linspace(0.0, 1.0, bins + 1))
        rows = []
        for bin_index in range(bins):
            low = float(edges[bin_index])
            high = float(edges[bin_index + 1])
            if bin_index == bins - 1:
                mask = (state_l >= low) & (state_l <= high)
            else:
                mask = (state_l >= low) & (state_l < high)
            values = delta[mask]
            rows.append({
                "bin": bin_index,
                "L_low": low,
                "L_high": high,
                **_quantiles(values.tolist()),
            })
        threshold = float(np.quantile(state_l, high_quantile))
        high_values = delta[state_l >= threshold]
        low_threshold = float(np.quantile(state_l, 1.0 - high_quantile))
        low_values = delta[state_l <= low_threshold]
        high_ci = _moving_block_mean_ci(
            high_values,
            repeats=bootstrap_repeats,
            block_length=max(horizon, 5),
            seed=random_seed + 1009 * horizon,
        )
        if high_values.size < min_bin_points:
            high_status = "INSUFFICIENT_POINTS"
        elif high_ci is not None and high_ci[1] < 0.0:
            high_status = "NEGATIVE_DRIFT_SUPPORTED"
        elif high_ci is not None and high_ci[0] > 0.0:
            high_status = "POSITIVE_DRIFT_DETECTED"
        else:
            high_status = "INCONCLUSIVE"
        drift[str(horizon)] = {
            "n": int(indices.size),
            "overall": _quantiles(delta.tolist()),
            "continuous_relation": continuous_relation,
            "bins": rows,
            "high_L": {
                "quantile": float(high_quantile),
                "threshold": threshold,
                **_quantiles(high_values.tolist()),
                "mean_ci95_block_bootstrap": high_ci,
                "status": high_status,
            },
            "low_L": {
                "quantile": float(1.0 - high_quantile),
                "threshold": low_threshold,
                **_quantiles(low_values.tolist()),
            },
        }

    tail_ticks = ticks[start_index:]
    tail_work = work[start_index:]
    backlog_slope = _linear_slope(tail_ticks, tail_work)
    normalized_slope = (
        float(backlog_slope * 100.0 / max(float(tail_work.mean()), 1.0))
        if backlog_slope is not None else None
    )

    future_risk = _future_window_mean(risk, int(risk_horizon))
    risk_indices = np.arange(start_index, future_risk.size, dtype=np.int64)
    if risk_indices.size:
        risk_state = total[risk_indices]
        risk_future = future_risk[risk_indices]
        high_threshold = float(np.quantile(risk_state, high_quantile))
        low_threshold = float(np.quantile(risk_state, 1.0 - high_quantile))
        high_future = risk_future[risk_state >= high_threshold]
        low_future = risk_future[risk_state <= low_threshold]
        risk_report = {
            "n": int(risk_indices.size),
            "horizon": int(risk_horizon),
            "spearman_L_vs_future_risk": _spearman(risk_state, risk_future),
            "high_L_future_risk": _quantiles(high_future.tolist()),
            "low_L_future_risk": _quantiles(low_future.tolist()),
            "high_minus_low_mean": (
                float(high_future.mean() - low_future.mean())
                if high_future.size and low_future.size else None
            ),
        }
    else:
        risk_report = {"n": 0, "horizon": int(risk_horizon)}

    active_components = {
        name: {
            "active_samples": int((np.abs(values) > 1e-12).sum()),
            "active_rate": float((np.abs(values) > 1e-12).mean()),
        }
        for name, values in components.items()
    }
    configured_active, configured_active_source = (
        _configured_active_components(payload, components)
    )
    flow_slice = slice(start_index, None)
    flow_totals = {
        "productive": float(productive[flow_slice].sum()),
        "arrival": float(arrival[flow_slice].sum()),
        "reverse": float(reverse[flow_slice].sum()),
        "replan_residual": float(residual[flow_slice].sum()),
    }
    flow_totals["productive_to_arrival_ratio"] = (
        float(flow_totals["productive"] / flow_totals["arrival"])
        if flow_totals["arrival"] > 1e-12 else None
    )

    primary = drift.get(str(primary_horizon), {"n": 0})
    return {
        "source_path": source_path,
        "run_id": str(payload.get("run_id", Path(source_path).stem)),
        "seed": payload.get("seed"),
        "load": _infer_load(payload, source_path),
        "arm": str(payload.get("arm_label", "unknown")),
        "config": str(payload.get("config", "")),
        "ticks": n,
        "burn_in": start_index,
        "invariants": {
            "pass": invariant_pass,
            "finite": finite_pass,
            "nonnegative": nonnegative_pass,
            "minimum_component_or_total": nonnegative_min,
            "decomposition_abs_error": _quantiles(
                decomposition_error.tolist()
            ),
            "work_balance_abs_error": _quantiles(
                np.abs(work_balance_error).tolist()
            ),
            "tolerance": float(invariant_tolerance),
        },
        "component_coverage": active_components,
        "configured_active_components": configured_active,
        "configured_active_components_source": configured_active_source,
        "drift": drift,
        "primary_continuous_drift_relation": primary.get(
            "continuous_relation"
        ),
        "primary_high_L_drift": primary.get("high_L"),
        "backlog": {
            "station_work_start": float(tail_work[0]),
            "station_work_end": float(tail_work[-1]),
            "station_work": _quantiles(tail_work.tolist()),
            "tail_slope_per_tick": backlog_slope,
            "tail_normalized_slope_per_100_ticks": normalized_slope,
        },
        "flows_after_burn_in": flow_totals,
        "future_risk": risk_report,
    }


def evaluate_payloads(
    payloads: Sequence[Tuple[str, Mapping]],
    *,
    drift_horizons: Sequence[int] = (1, 5, 10, 25, 50),
    primary_horizon: int = 10,
    risk_horizon: int = 25,
    bins: int = 5,
    high_quantile: float = 0.80,
    burn_in: int = 50,
    bootstrap_repeats: int = 1000,
    min_bin_points: int = 20,
    min_runs_per_group: int = 5,
    min_component_active_samples: int = 20,
    invariant_tolerance: float = 1e-4,
    random_seed: int = 2026,
) -> dict:
    runs = [
        analyze_stream(
            payload,
            source_path=path,
            drift_horizons=drift_horizons,
            primary_horizon=primary_horizon,
            risk_horizon=risk_horizon,
            bins=bins,
            high_quantile=high_quantile,
            burn_in=burn_in,
            bootstrap_repeats=bootstrap_repeats,
            min_bin_points=min_bin_points,
            invariant_tolerance=invariant_tolerance,
            random_seed=random_seed + index * 7919,
        )
        for index, (path, payload) in enumerate(payloads)
    ]

    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["load"], run["arm"])].append(run)
    group_reports = {}
    for group_index, ((load, arm), members) in enumerate(sorted(grouped.items())):
        high_means = [
            float(run["primary_high_L_drift"]["mean"])
            for run in members
            if isinstance(run.get("primary_high_L_drift"), Mapping)
            and run["primary_high_L_drift"].get("n", 0) >= min_bin_points
        ]
        slopes = [
            float(run["backlog"]["tail_slope_per_tick"])
            for run in members
            if run["backlog"].get("tail_slope_per_tick") is not None
        ]
        drift_relation_slopes = [
            float(run["primary_continuous_drift_relation"]["slope"])
            for run in members
            if isinstance(
                run.get("primary_continuous_drift_relation"), Mapping
            )
            and run["primary_continuous_drift_relation"].get("slope")
            is not None
        ]
        p95_drift_predictions = [
            float(
                run["primary_continuous_drift_relation"]
                ["predicted_drift_at_observed_L_p95"]
            )
            for run in members
            if isinstance(
                run.get("primary_continuous_drift_relation"), Mapping
            )
            and run["primary_continuous_drift_relation"].get(
                "predicted_drift_at_observed_L_p95"
            ) is not None
        ]
        high_ci = _independent_mean_ci(
            high_means,
            repeats=bootstrap_repeats,
            seed=random_seed + 104729 * (group_index + 1),
        )
        slope_ci = _independent_mean_ci(
            slopes,
            repeats=bootstrap_repeats,
            seed=random_seed + 130363 * (group_index + 1),
        )
        drift_relation_slope_ci = _independent_mean_ci(
            drift_relation_slopes,
            repeats=bootstrap_repeats,
            seed=random_seed + 155921 * (group_index + 1),
        )
        p95_drift_prediction_ci = _independent_mean_ci(
            p95_drift_predictions,
            repeats=bootstrap_repeats,
            seed=random_seed + 196613 * (group_index + 1),
        )
        if len(members) < min_runs_per_group:
            status = "INSUFFICIENT_RUNS"
        elif len(high_means) < min_runs_per_group:
            status = "INSUFFICIENT_HIGH_L_POINTS"
        elif (
            high_ci is not None and high_ci[0] > 0.0
        ) or (
            p95_drift_prediction_ci is not None
            and p95_drift_prediction_ci[0] > 0.0
        ):
            status = "POLICY_POSITIVE_HIGH_L_DRIFT"
        elif slope_ci is not None and slope_ci[0] > 0.0:
            status = "POLICY_BACKLOG_GROWTH"
        elif (
            (
                high_ci is not None and high_ci[1] < 0.0
            ) or (
                p95_drift_prediction_ci is not None
                and p95_drift_prediction_ci[1] < 0.0
            )
        ) and slope_ci is not None and slope_ci[1] <= 0.0:
            status = "CLOSED_LOOP_STABILITY_SUPPORTED"
        else:
            status = "INCONCLUSIVE"
        group_reports[f"{load}|{arm}"] = {
            "load": load,
            "arm": arm,
            "runs": len(members),
            "seeds": sorted({
                int(run["seed"]) for run in members if run.get("seed") is not None
            }),
            "high_L_drift_run_means": _quantiles(high_means),
            "high_L_drift_mean_ci95_across_runs": high_ci,
            "continuous_drift_slope_run_values": _quantiles(
                drift_relation_slopes
            ),
            "continuous_drift_slope_mean_ci95_across_runs": (
                drift_relation_slope_ci
            ),
            "continuous_predicted_drift_at_L_p95_run_values": _quantiles(
                p95_drift_predictions
            ),
            "continuous_predicted_drift_at_L_p95_ci95_across_runs": (
                p95_drift_prediction_ci
            ),
            "backlog_slope_run_values": _quantiles(slopes),
            "backlog_slope_mean_ci95_across_runs": slope_ci,
            "status": status,
        }

    active_counts = {name: 0 for name in L_COMPONENT_NAMES}
    for run in runs:
        for name in L_COMPONENT_NAMES:
            active_counts[name] += int(
                run["component_coverage"][name]["active_samples"]
            )
    configured_active = sorted({
        name
        for run in runs
        for name in run.get("configured_active_components", ())
    })
    undercovered = [
        name for name in configured_active
        if active_counts[name] < int(min_component_active_samples)
    ]
    invariant_failures = [
        run["run_id"] for run in runs if not run["invariants"]["pass"]
    ]
    group_failures = [
        name for name, row in group_reports.items()
        if row["status"] in {
            "POLICY_POSITIVE_HIGH_L_DRIFT", "POLICY_BACKLOG_GROWTH",
        }
    ]
    group_incomplete = [
        name for name, row in group_reports.items()
        if row["status"] in {
            "INSUFFICIENT_RUNS", "INSUFFICIENT_HIGH_L_POINTS", "INCONCLUSIVE",
        }
    ]
    if invariant_failures:
        verdict = "INVALID_STREAM_INVARIANTS"
    elif group_failures:
        verdict = "CLOSED_LOOP_POLICY_FAILURE"
    elif undercovered or group_incomplete:
        verdict = "INSUFFICIENT_DATA"
    elif group_reports and all(
        row["status"] == "CLOSED_LOOP_STABILITY_SUPPORTED"
        for row in group_reports.values()
    ):
        verdict = "PASS_CLOSED_LOOP_STABILITY"
    else:
        verdict = "INCONCLUSIVE"

    recommendations = []
    if invariant_failures:
        recommendations.append(
            "Fix snapshot/progress bookkeeping before interpreting any drift."
        )
    if undercovered:
        recommendations.append(
            "Collect targeted stress states for inactive components: "
            + ", ".join(undercovered)
        )
    if group_failures:
        recommendations.append(
            "A closed-loop policy/load failure is detected. Compare analytic "
            "oracle, learned-drift error, guard activation, and matched-random "
            "arms before attributing the failure to L itself."
        )
    if group_incomplete:
        recommendations.append(
            "Increase independent seeds/run length; overlapping tick windows "
            "must not be counted as independent runs."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "semantics": {
            "five_layer_role": "layer5_normal_arrival_closed_loop_stability",
            "scope": "normal-arrival closed-loop trajectories",
            "primary_horizon": int(primary_horizon),
            "high_L_quantile": float(high_quantile),
            "policy_scoped": True,
            "causal_L_redesign_claim": False,
            "raw_action_gap_gate": False,
            "continuous_drift_relation": (
                "regress observed closed-loop Delta L on current L and "
                "report block-bootstrap uncertainty; quantile bins are "
                "descriptive compact-region diagnostics, not action-gap "
                "selection thresholds"
            ),
            "note": (
                "Positive drift diagnoses the evaluated policy/load pair. "
                "Use the isolated analytic oracle to decide whether L itself "
                "needs redesign."
            ),
        },
        "parameters": {
            "drift_horizons": list(drift_horizons),
            "risk_horizon": int(risk_horizon),
            "bins": int(bins),
            "burn_in": int(burn_in),
            "bootstrap_repeats": int(bootstrap_repeats),
            "min_bin_points": int(min_bin_points),
            "min_runs_per_group": int(min_runs_per_group),
            "min_component_active_samples": int(min_component_active_samples),
            "invariant_tolerance": float(invariant_tolerance),
        },
        "runs": runs,
        "groups": group_reports,
        "component_active_samples": active_counts,
        "configured_active_components": configured_active,
        "undercovered_components": undercovered,
        "invariant_failure_runs": invariant_failures,
        "failed_policy_groups": group_failures,
        "incomplete_groups": group_incomplete,
        "verdict": verdict,
        "passed": verdict == "PASS_CLOSED_LOOP_STABILITY",
        "recommendations": recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--streams", nargs="+", required=True)
    parser.add_argument(
        "--drift-horizons", type=_parse_int_list, default=(1, 5, 10, 25, 50)
    )
    parser.add_argument("--primary-horizon", type=int, default=10)
    parser.add_argument("--risk-horizon", type=int, default=25)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--high-quantile", type=float, default=0.80)
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--min-bin-points", type=int, default=20)
    parser.add_argument("--min-runs-per-group", type=int, default=5)
    parser.add_argument("--min-component-active-samples", type=int, default=20)
    parser.add_argument("--invariant-tolerance", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--fail-on-failure", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.primary_horizon <= 0 or args.risk_horizon <= 0:
        raise SystemExit("horizons must be positive")
    if args.primary_horizon not in args.drift_horizons:
        raise SystemExit("--primary-horizon must appear in --drift-horizons")
    if args.bins < 2:
        raise SystemExit("--bins must be >= 2")
    if not 0.5 < args.high_quantile < 1.0:
        raise SystemExit("--high-quantile must lie in (0.5, 1)")
    if args.burn_in < 0:
        raise SystemExit("--burn-in must be non-negative")
    if args.bootstrap_repeats <= 0:
        raise SystemExit("--bootstrap-repeats must be positive")
    if min(
        args.min_bin_points, args.min_runs_per_group,
        args.min_component_active_samples,
    ) < 0:
        raise SystemExit("minimum-count gates must be non-negative")
    if args.invariant_tolerance <= 0.0:
        raise SystemExit("--invariant-tolerance must be positive")

    paths = _expand_paths(args.streams)
    payloads = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise SystemExit(f"{path}: expected a mapping payload")
        payloads.append((path, payload))
    report = evaluate_payloads(
        payloads,
        drift_horizons=args.drift_horizons,
        primary_horizon=args.primary_horizon,
        risk_horizon=args.risk_horizon,
        bins=args.bins,
        high_quantile=args.high_quantile,
        burn_in=args.burn_in,
        bootstrap_repeats=args.bootstrap_repeats,
        min_bin_points=args.min_bin_points,
        min_runs_per_group=args.min_runs_per_group,
        min_component_active_samples=args.min_component_active_samples,
        invariant_tolerance=args.invariant_tolerance,
        random_seed=args.random_seed,
    )
    report["source_files"] = paths
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"saved: {output}")
    else:
        print(text)
    print(f"closed-loop Lyapunov verdict: {report['verdict']}")
    if args.fail_on_failure and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
