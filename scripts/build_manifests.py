#!/usr/bin/env python3
"""Build compact, path-based indexes for committed artifact files."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest_root = ROOT / "manifests"
    manifest_files = list((manifest_root / "main_50seed").rglob("*.json"))
    manifest_files += list((manifest_root / "station6_10seed").rglob("*.json"))
    artifact_root = ROOT / "artifacts"
    files = [
        path
        for path in artifact_root.rglob("*")
        if path.is_file()
        and "generated" not in path.parts
        and path.name != "artifact_manifest.json"
    ]
    entries = []
    for path in sorted(files):
        relative = path.relative_to(artifact_root).as_posix()
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "role": relative.split("/", 1)[0],
            }
        )
    payload = {
        "schema_version": "rmfs_artifact_index_v1",
        "generated_from": "committed anonymous evidence",
        "entry_count": len(entries),
        "entries": entries,
    }
    (artifact_root / "artifact_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Indexed {len(manifest_files)} arrival manifests and {len(entries)} artifacts")


if __name__ == "__main__":
    main()
