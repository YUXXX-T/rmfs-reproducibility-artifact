#!/usr/bin/env python3
"""Verify that separately distributed checkpoints are installed as listed."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((ROOT / "checkpoints/checkpoint_manifest.json").read_text(encoding="utf-8"))
    missing = []
    for asset in manifest["assets"]:
        path = ROOT / asset["path"]
        if not path.is_file():
            missing.append(asset["name"])
            continue
        expected_bytes = asset.get("bytes")
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise SystemExit(f"FAIL: unexpected byte size for {asset['name']}")
        print(f"PASS: {asset['name']}")
    if missing:
        print("Not installed (allowed for lightweight reproduction): " + ", ".join(missing))


if __name__ == "__main__":
    main()
