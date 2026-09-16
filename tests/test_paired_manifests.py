from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_index_is_complete_and_well_formed() -> None:
    with (ROOT / "manifests/metadata.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 210
    assert set(rows[0]) == {"campaign", "relative_path", "schema_version", "total_orders"}
    assert {row["campaign"] for row in rows} == {
        "main_50seed", "station6_10seed", "density_scale_10seed"
    }
    for row in rows:
        path = ROOT / "manifests" / row["relative_path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) == {"schema_version", "orders", "total_orders"}
        assert payload["schema_version"] == "layer5_order_arrival_manifest_v1"
        assert int(row["total_orders"]) == payload["total_orders"] == len(payload["orders"])


def test_each_paired_campaign_has_one_manifest_per_load_and_seed() -> None:
    main = {path.stem for path in (ROOT / "manifests/main_50seed").glob("*.json")}
    assert main == {
        f"greedy_{load}_seed{seed}_orders"
        for load in ("low", "mid", "high")
        for seed in range(900, 950)
    }
    station6 = list((ROOT / "manifests/station6_10seed").rglob("*.json"))
    assert len(station6) == 30
    for load in ("low", "mid", "high"):
        for seed in range(721, 731):
            assert any(
                path.parent.name == load
                and path.name == f"orders_{load}_seed{seed}.json"
                for path in station6
            )
    density_scale = {path.name for path in (ROOT / "manifests/density_scale_10seed").glob("*.json")}
    assert density_scale == {
        f"orders_{load}_seed{seed}.json"
        for load in ("low", "mid", "high")
        for seed in range(701, 711)
    }
