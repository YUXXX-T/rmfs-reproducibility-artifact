from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_interactive_result_bundle_covers_frozen_tables() -> None:
    path = ROOT / "docs/assets/data/results_explorer.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "rmfs_results_explorer_v1"
    assert set(payload["datasets"]) == {"main", "station6"}
    assert len(payload["datasets"]["main"]["aggregate"]) == 15
    assert len(payload["datasets"]["station6"]["aggregate"]) == 12
    assert len(payload["datasets"]["main"]["paired"]) == 96
    assert len(payload["datasets"]["station6"]["paired"]) == 87
    assert payload["metrics"]["completed_orders"]["direction"] == "higher"
    assert payload["metrics"]["deadlock_ratio_mean"]["direction"] == "lower"


def test_pages_csv_downloads_match_canonical_evidence() -> None:
    pairs = (
        ("artifacts/tables/table_main.csv", "docs/assets/data/table_main.csv"),
        ("artifacts/tables/table_station6.csv", "docs/assets/data/table_station6.csv"),
        (
            "artifacts/statistics/paired_confidence_intervals.csv",
            "docs/assets/data/paired_confidence_intervals.csv",
        ),
        (
            "artifacts/statistics/station6_paired_confidence_intervals.csv",
            "docs/assets/data/station6_paired_confidence_intervals.csv",
        ),
    )
    for canonical, download in pairs:
        assert (ROOT / canonical).read_bytes() == (ROOT / download).read_bytes()
