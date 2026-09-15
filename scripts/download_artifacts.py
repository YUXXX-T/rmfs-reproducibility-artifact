#!/usr/bin/env python3
"""Download separately hosted assets once release/DOI URLs are populated."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", action="append", help="asset name; repeatable")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "checkpoints/checkpoint_manifest.json").read_text(encoding="utf-8"))
    assets = manifest["assets"]
    if args.list or not args.name:
        for asset in assets:
            size = asset.get("bytes")
            detail = f", {size} bytes" if size is not None else ""
            print(f"{asset['name']}: {asset['distribution']}{detail}")
        if not args.name:
            return
    selected = set(args.name)
    for asset in assets:
        if asset["name"] not in selected:
            continue
        url = asset.get("url")
        if not url:
            raise SystemExit(f"URL not yet published for {asset['name']}; use the review release bundle")
        target = ROOT / asset["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, target)
        expected_bytes = asset.get("bytes")
        if expected_bytes is not None and target.stat().st_size != expected_bytes:
            raise SystemExit(f"unexpected byte size for {asset['name']}: {target}")
        print(f"Downloaded {target}")


if __name__ == "__main__":
    main()
