from __future__ import annotations

import csv
from pathlib import Path

import pytest

from bootstrap_ci import mean_ci


ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_is_deterministic_and_validates_inputs() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert mean_ci(values, iterations=2000, seed=17) == mean_ci(
        values, iterations=2000, seed=17
    )
    assert mean_ci([7.5]) == (7.5, 7.5)
    with pytest.raises(ValueError):
        mean_ci([])
    with pytest.raises(ValueError):
        mean_ci(values, iterations=1)


def test_frozen_high_load_proposed_minus_greedy_ci() -> None:
    with (ROOT / "artifacts/raw/main_per_seed.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["load"], int(row["seed"]), row["arm"]): row for row in rows}
    order_delta = [
        float(indexed[("high", seed, "ComboS1J1")]["completed_orders"])
        - float(indexed[("high", seed, "Greedy")]["completed_orders"])
        for seed in range(900, 950)
    ]
    deadlock_delta = [
        float(indexed[("high", seed, "ComboS1J1")]["deadlock_ratio_mean"])
        - float(indexed[("high", seed, "Greedy")]["deadlock_ratio_mean"])
        for seed in range(900, 950)
    ]
    assert sum(order_delta) / 50 == pytest.approx(65.06)
    assert mean_ci(order_delta) == (15.76, 114.86)
    assert sum(deadlock_delta) / 50 == pytest.approx(-0.128522, abs=1e-6)
    assert mean_ci(deadlock_delta) == (-0.223687, -0.035251)
