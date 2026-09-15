from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_workload_contracts_match_exact_simulator_configs() -> None:
    expected = {
        "low": (5, 4, 2),
        "mid": (4, 8, 4),
        "high": (3, 12, 6),
    }
    for load, values in expected.items():
        workload = yaml.safe_load(
            (ROOT / f"configs/workloads/{load}.yaml").read_text(encoding="utf-8")
        )
        simulator = json.loads(
            (ROOT / f"configs/simulator/world_model_config_PP_48_{load}.json").read_text(
                encoding="utf-8"
            )
        )["simulation"]
        observed = (
            workload["order_interval_ticks"],
            workload["initial_order_pool_size"],
            workload["pending_backlog_floor"],
        )
        assert observed == values
        assert observed == (
            simulator["order_interval"],
            simulator["initial_order_pool_size"],
            simulator["backlog_floor"],
        )
        assert workload["fixed_distinct_skus_per_order"] == 2
        assert simulator["max_items_per_order"] == 2
        assert workload["quantity_per_sku"] == {
            "minimum": 1,
            "maximum": simulator["max_items_per_sku"],
        }
        assert workload["station_service_ticks"] == simulator["station_process_duration"] == 5


def test_zipf_generator_implementation_exposes_registered_semantics() -> None:
    source = (
        ROOT
        / "src/Policies/OrderGenerator/ZipfOrderGenerator/zipf_order_generator.py"
    ).read_text(encoding="utf-8")
    assert "world_state.tick % self.order_interval" in source
    assert "np.random.choice" in source
    assert "self.zipf_param" in source
    assert "replace=False" in source
    assert "random.randint(1, self.max_items_per_sku)" in source
