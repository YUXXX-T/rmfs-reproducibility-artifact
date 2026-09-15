from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _schema(name: str) -> dict:
    return yaml.safe_load((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def _assert_contiguous(channels: list[dict], size: int) -> None:
    assert [channel["index"] for channel in channels] == list(range(size))
    assert len({channel["name"] for channel in channels}) == size


def test_feature_dimensions_and_channel_indices() -> None:
    state = _schema("state_features.yaml")
    action = _schema("action_features.yaml")
    system = _schema("system_labels.yaml")
    node = _schema("node_labels.yaml")
    assert state["history_length"] == 4
    assert state["node_features"]["shape"] == ["history", "nodes", 10]
    assert state["edge_features"]["shape"] == ["directed_edges", 6]
    _assert_contiguous(state["node_features"]["channels"], 10)
    _assert_contiguous(state["edge_features"]["channels"], 6)
    assert action["node_field"]["shape"] == ["nodes", 8]
    assert action["global"]["shape"] == [6]
    _assert_contiguous(action["node_field"]["channels"], 8)
    _assert_contiguous(action["global"]["channels"], 6)
    assert system["shape"] == ["horizon", 7]
    _assert_contiguous(system["channels"], 7)
    assert node["node"]["shape"] == ["horizon", "nodes", 6]
    assert node["station"]["shape"] == ["horizon", "stations", 2]
    _assert_contiguous(node["node"]["channels"], 6)
    _assert_contiguous(node["station"]["channels"], 2)


def test_schemas_reference_the_shipped_implementation() -> None:
    for filename in (
        "state_features.yaml",
        "action_features.yaml",
        "system_labels.yaml",
        "node_labels.yaml",
    ):
        payload = _schema(filename)
        assert (ROOT / payload["canonical_source"]).is_file(), filename
    implementation = (ROOT / "src/WorldModel/graph/graph_builder.py").read_text(
        encoding="utf-8"
    )
    assert "feat_dim: int = 10" in implementation
    assert "Returns (action_node (N, 8), action_global (6,))" in implementation
    assert "action_node = [[0.0] * 8 for _ in range(N)]" in implementation
