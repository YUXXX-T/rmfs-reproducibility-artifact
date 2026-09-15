from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from rmfs_artifact.detection import detect_throughput_collapse, is_paper_collapsed


ROOT = Path(__file__).resolve().parents[1]


def test_run_level_collapse_is_a_two_metric_conjunction() -> None:
    assert is_paper_collapsed(299, 0.4)
    assert not is_paper_collapsed(300, 0.4)
    assert not is_paper_collapsed(299, 0.399999)
    contract = yaml.safe_load(
        (ROOT / "configs/detection/collapse.yaml").read_text(encoding="utf-8")
    )
    endpoint = contract["run_level_label"]
    assert endpoint["conjunction"] is True
    assert endpoint["completed_orders"] == {"operator": "less_than", "threshold": 300}
    assert endpoint["deadlock_ratio_mean"] == {
        "operator": "greater_than_or_equal",
        "threshold": 0.4,
    }


def test_temporal_detector_requires_low_rate_and_high_backlog() -> None:
    ticks = np.arange(0, 130, 10, dtype=float)
    completed = np.asarray([0, 4, 8, 12, 16, 20, 24, 24, 24, 24, 24, 24, 24], dtype=float)
    pending = np.asarray([4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12], dtype=float)
    in_progress = np.zeros_like(pending)
    onset = detect_throughput_collapse(
        ticks, completed, pending, in_progress, smoothing_window=1
    )
    assert onset == 70.0
    assert detect_throughput_collapse(
        ticks, completed, np.ones_like(pending), in_progress, smoothing_window=1
    ) is None


def test_frozen_endpoint_counts_match_paper_aligned_definition() -> None:
    with (ROOT / "artifacts/raw/main_per_seed.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = {}
    for arm in ("Greedy", "ComboS1J1"):
        group = [row for row in rows if row["load"] == "high" and row["arm"] == arm]
        counts[arm] = sum(
            is_paper_collapsed(row["completed_orders"], row["deadlock_ratio_mean"])
            for row in group
        )
    assert counts == {"Greedy": 22, "ComboS1J1": 15}
