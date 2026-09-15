"""Lightweight, auditable implementations of the mechanism event contracts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def rolling_mean(values: Sequence[float], window: int = 5) -> np.ndarray:
    """Return the centered rolling mean used by the frozen trace analysis."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if array.size == 0:
        return array
    half = window // 2
    return np.asarray(
        [
            array[max(0, index - half) : min(array.size, index + half + 1)].mean()
            for index in range(array.size)
        ]
    )


def first_sustained_crossing(
    ticks: Sequence[float],
    series: Sequence[float],
    threshold: float,
    *,
    sustain_samples: int = 3,
) -> float | None:
    """Locate the first sample of the first sustained threshold crossing."""
    ticks_array = np.asarray(ticks, dtype=float)
    series_array = np.asarray(series, dtype=float)
    if ticks_array.shape != series_array.shape or ticks_array.ndim != 1:
        raise ValueError("ticks and series must be aligned one-dimensional arrays")
    if sustain_samples <= 0:
        raise ValueError("sustain_samples must be positive")
    run = 0
    for index, value in enumerate(series_array):
        run = run + 1 if value >= threshold else 0
        if run >= sustain_samples:
            return float(ticks_array[index - sustain_samples + 1])
    return None


def station_attempt_share(cumulative_attempts: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert cumulative station attempts to per-interval maximum shares."""
    attempts = np.asarray(cumulative_attempts, dtype=float)
    if attempts.ndim != 2 or attempts.shape[0] < 2 or attempts.shape[1] < 1:
        raise ValueError("cumulative_attempts must have shape (ticks>=2, stations>=1)")
    interval = np.clip(np.diff(attempts, axis=0), 0.0, None)
    totals = interval.sum(axis=1)
    return np.divide(
        interval.max(axis=1),
        totals,
        out=np.zeros_like(totals),
        where=totals > 0,
    )


def detect_station_lock(
    ticks: Sequence[float],
    cumulative_attempts: Sequence[Sequence[float]],
    *,
    threshold: float = 0.8,
    smoothing_window: int = 5,
    sustain_samples: int = 3,
) -> float | None:
    """Detect station-lock onset from admission-attempt concentration."""
    ticks_array = np.asarray(ticks, dtype=float)
    attempts = np.asarray(cumulative_attempts, dtype=float)
    if ticks_array.ndim != 1 or ticks_array.size != attempts.shape[0]:
        raise ValueError("ticks must align with cumulative-attempt rows")
    share = rolling_mean(station_attempt_share(attempts), smoothing_window)
    return first_sustained_crossing(
        ticks_array[1:], share, threshold, sustain_samples=sustain_samples
    )


def is_paper_collapsed(
    completed_orders: float,
    deadlock_ratio_mean: float,
    *,
    completed_orders_threshold: float = 300,
    deadlock_ratio_threshold: float = 0.4,
) -> bool:
    """Apply the run-level paper collapse endpoint (a strict conjunction)."""
    return (
        float(completed_orders) < completed_orders_threshold
        and float(deadlock_ratio_mean) >= deadlock_ratio_threshold
    )


def detect_throughput_collapse(
    ticks: Sequence[float],
    completed: Sequence[float],
    pending: Sequence[float],
    in_progress: Sequence[float],
    *,
    smoothing_window: int = 5,
    early_fraction: float = 1.0 / 3.0,
    rate_fraction: float = 0.25,
    backlog_multiplier: float = 1.5,
    minimum_backlog: float = 1.0,
    sustain_samples: int = 3,
) -> float | None:
    """Localize collapse onset using only completion-rate and backlog signals."""
    ticks_array = np.asarray(ticks, dtype=float)
    completed_array = np.asarray(completed, dtype=float)
    pending_array = np.asarray(pending, dtype=float)
    progress_array = np.asarray(in_progress, dtype=float)
    shapes = {
        array.shape
        for array in (ticks_array, completed_array, pending_array, progress_array)
    }
    if len(shapes) != 1 or ticks_array.ndim != 1:
        raise ValueError("all trace inputs must be aligned one-dimensional arrays")
    if ticks_array.size < 7:
        return None
    delta_tick = np.diff(ticks_array)
    rate = np.divide(
        np.diff(completed_array),
        delta_tick,
        out=np.zeros_like(delta_tick),
        where=delta_tick > 0,
    )
    rate = rolling_mean(rate, smoothing_window)
    backlog = (pending_array + progress_array)[1:]
    early_n = max(2, int(rate.size * early_fraction))
    healthy_rate = float(np.median(rate[:early_n]))
    early_backlog = float(np.median(backlog[:early_n]))
    if healthy_rate <= 0:
        return None
    rate_threshold = rate_fraction * healthy_rate
    backlog_threshold = max(minimum_backlog, backlog_multiplier * early_backlog)
    run = 0
    for index in range(early_n, rate.size):
        if rate[index] < rate_threshold and backlog[index] >= backlog_threshold:
            run += 1
        else:
            run = 0
        if run >= sustain_samples:
            return float(ticks_array[1:][index - sustain_samples + 1])
    return None


__all__ = [
    "detect_station_lock",
    "detect_throughput_collapse",
    "first_sustained_crossing",
    "is_paper_collapsed",
    "rolling_mean",
    "station_attempt_share",
]
