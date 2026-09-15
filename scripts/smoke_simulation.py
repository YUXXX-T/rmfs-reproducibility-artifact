#!/usr/bin/env python3
"""Materialize a short override and invoke the unmodified simulator entry point."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def set_dotted(config: dict, dotted: str, value) -> None:
    target = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def main() -> None:
    smoke_path = ROOT / "configs/evaluation/smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    base_path = (smoke_path.parent / smoke["inherits"]).resolve()
    config = json.loads(base_path.read_text(encoding="utf-8"))
    for dotted, value in smoke["overrides"].items():
        set_dotted(config, dotted, value)
    with tempfile.TemporaryDirectory(prefix="rmfs-artifact-smoke-") as temporary:
        materialized = Path(temporary) / "smoke.json"
        materialized.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        subprocess.run(
            [sys.executable, str(ROOT / "src/main.py"), "--config", str(materialized)],
            cwd=ROOT,
            env=environment,
            check=True,
        )
    print("PASS: 20-tick baseline simulator smoke run")


if __name__ == "__main__":
    main()
