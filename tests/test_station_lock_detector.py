from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from generate_fig05_mechanism import read_records, validate
from rmfs_artifact.detection import detect_station_lock, station_attempt_share


ROOT = Path(__file__).resolve().parents[1]


def test_station_lock_uses_per_interval_attempt_share() -> None:
    cumulative = np.asarray(
        [[0, 0], [5, 5], [14, 6], [23, 7], [32, 8], [41, 9]], dtype=float
    )
    shares = station_attempt_share(cumulative)
    np.testing.assert_allclose(shares, [0.5, 0.9, 0.9, 0.9, 0.9])
    assert detect_station_lock(
        [0, 10, 20, 30, 40, 50],
        cumulative,
        threshold=0.8,
        smoothing_window=1,
        sustain_samples=3,
    ) == 20.0


def test_primary_and_sensitivity_thresholds_are_registered() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/detection/station_lock.yaml").read_text(encoding="utf-8")
    )
    assert contract["primary_threshold"] == 0.8
    assert contract["sensitivity_thresholds"] == [0.6, 0.7, 0.8, 0.9]
    assert contract["sustain_samples"] == 3
    assert contract["trace_stride_ticks"] == 10


def test_frozen_fig05_event_summary_is_self_consistent() -> None:
    records = read_records(ROOT / "artifacts/raw/station_lock_events.csv")
    summary = validate(records)
    groups = {group["policy"]: group for group in summary["groups"]}
    assert summary["denominator"] == 50
    assert groups["Greedy"]["paper_collapsed_runs"] == 22
    assert groups["Greedy"]["lead_pairs"] == 22
    assert groups["Greedy"]["median_lead_ticks"] == 445.0
    assert groups["ComboS1J1"]["paper_collapsed_runs"] == 15
    assert groups["ComboS1J1"]["lead_pairs"] == 14
    assert groups["ComboS1J1"]["median_lead_ticks"] == 460.0
    with (ROOT / "artifacts/raw/station_lock_events.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert all(float(row["lead_time"]) > 0 for row in rows if row["lead_time"].strip())
