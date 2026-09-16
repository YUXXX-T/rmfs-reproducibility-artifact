from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_density_scale_factorial_and_pairing() -> None:
    rows = read_rows("artifacts/raw/density_scale_per_seed.csv")
    assert len(rows) == 1350
    assert {int(row["num_stations"]) for row in rows} == {4}
    assert {int(row["map_rows"]) for row in rows} == {20, 30, 40}
    assert {float(row["robot_density"]) for row in rows} == {0.12, 0.15, 0.18}
    for load in ("low", "mid", "high"):
        for seed in range(701, 711):
            cell = [row for row in rows if row["load"] == load and int(row["seed"]) == seed]
            assert len(cell) == 45
            assert len({int(row["order_arrival_count"]) for row in cell}) == 1


def test_density_scale_inputs_are_published_without_static_fingerprints() -> None:
    import json

    paths = sorted((ROOT / "manifests/density_scale_10seed").glob("*.json"))
    assert len(paths) == 30
    indexed = read_rows("manifests/metadata.csv")
    assert sum(row["campaign"] == "density_scale_10seed" for row in indexed) == 30
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) == {"schema_version", "orders", "total_orders"}
        assert len(payload["orders"]) == payload["total_orders"]
    configs = list((ROOT / "configs/evaluation/density_scale_variants").rglob("*.json"))
    assert len(configs) == 27
    for path in configs:
        config = json.loads(path.read_text(encoding="utf-8"))
        size = int(path.parent.name.split("_")[0][3:])
        robots = int(path.parent.name.split("_")[1][1:])
        assert (config["map"]["rows"], config["map"]["cols"]) == (size, size)
        assert config["robots"]["num_robots"] == robots
        assert len(config["map"]["stations"]) == 4


def test_density_scale_reported_effects_and_membership() -> None:
    effects = read_rows("artifacts/statistics/density_scale_paired_effects.csv")
    lookup = {(row["map_scope"], row["method"], row["baseline"]): row for row in effects}
    overall = lookup[("all", "combo_s1_j1", "jsq")]
    assert round(float(overall["paired_mean_difference"]), 2) == 34.31
    assert round(float(overall["ci95_low"]), 2) == 20.12
    assert round(float(overall["ci95_high"]), 2) == 49.03
    assert int(overall["positive_load_seed_clusters"]) == 25

    membership = read_rows("artifacts/statistics/density_scale_cell_membership.csv")
    proposed = [row for row in membership if row["arm"] == "combo_s1_j1"]
    assert len(proposed) == 27
    assert sum(row["throughput_winner"] == "True" for row in proposed) == 15
    assert sum(row["four_metric_pareto_member"] == "True" for row in proposed) == 25
