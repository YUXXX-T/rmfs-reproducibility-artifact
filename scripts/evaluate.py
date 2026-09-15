#!/usr/bin/env python3
"""Document or execute canonical evaluation entry points."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "main-pp": "WorldModel.evaluation.run_phase_c_physical_only_pp_50seed",
    "main-trace": "WorldModel.evaluation.run_phase_c_physical_only_pp_trace",
    "pibt": "WorldModel.evaluation.run_phase_c_pibt_planner_study",
    "station6": "WorldModel.evaluation.run_station6_20x20_adaptation",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", choices=MODULES)
    parser.add_argument("--execute", action="store_true")
    args, remainder = parser.parse_known_args()
    command = [sys.executable, "-m", MODULES[args.campaign], *remainder]
    print(" ".join(command))
    if args.execute:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
