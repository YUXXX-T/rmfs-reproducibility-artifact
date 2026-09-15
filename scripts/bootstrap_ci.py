#!/usr/bin/env python3
"""Deterministic paired-seed percentile bootstrap used by the paper artifact."""

from __future__ import annotations

import math
import random
from typing import Iterable


def mean_ci(
    values: Iterable[float],
    *,
    iterations: int = 10_000,
    seed: int = 20260905,
) -> tuple[float, float]:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("at least one paired delta is required")
    if iterations < 2:
        raise ValueError("iterations must be at least two")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(iterations)
    )
    lo = means[math.floor(0.025 * (iterations - 1))]
    hi = means[math.ceil(0.975 * (iterations - 1))]
    return round(lo, 6), round(hi, 6)


__all__ = ["mean_ci"]
