#!/usr/bin/env python3
"""Build a compact, cross-platform index of committed artifact files."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest_root = ROOT / "manifests"
    campaigns = ("main_50seed", "station6_10seed", "density_scale_10seed")
    manifest_files = [
        path
        for campaign in campaigns
        for path in sorted((manifest_root / campaign).rglob("*.json"))
    ]
    if len(manifest_files) != 210:
        raise ValueError(f"expected 210 manifests, got {len(manifest_files)}")
    with (manifest_root / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("campaign", "relative_path", "schema_version", "total_orders"))
        for path in manifest_files:
            source = json.loads(path.read_text(encoding="utf-8"))
            if int(source["total_orders"]) != len(source["orders"]):
                raise ValueError(f"arrival manifest count mismatch: {path}")
            writer.writerow(
                (
                    path.relative_to(manifest_root).parts[0],
                    path.relative_to(manifest_root).as_posix(),
                    source["schema_version"],
                    source["total_orders"],
                )
            )
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
                "role": relative.split("/", 1)[0],
            }
        )
    payload = {
        "schema_version": "rmfs_artifact_index_v2",
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
