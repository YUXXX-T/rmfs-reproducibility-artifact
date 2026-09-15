from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_artifact_index_is_path_based_and_cross_platform() -> None:
    payload = json.loads(
        (ROOT / "artifacts/artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "rmfs_artifact_index_v2"
    assert payload["entry_count"] == len(payload["entries"])
    assert all(set(entry) == {"path", "role"} for entry in payload["entries"])
    assert all(
        entry["role"] == entry["path"].split("/", 1)[0]
        for entry in payload["entries"]
    )
