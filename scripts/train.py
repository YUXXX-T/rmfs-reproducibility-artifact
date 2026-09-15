#!/usr/bin/env python3
"""Document or execute canonical training entry points."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "world-model": "WorldModel.training.run_train_v6",
    "j1-dispatch": "WorldModel.training.train_station_congestion_head",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=MODULES)
    parser.add_argument("--execute", action="store_true")
    args, remainder = parser.parse_known_args()
    command = [sys.executable, "-m", MODULES[args.component], *remainder]
    print(" ".join(command))
    if args.execute:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
