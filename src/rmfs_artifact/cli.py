"""Command-line dispatcher for common artifact operations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {
    "tables": "reproduce_tables.py",
    "figures": "reproduce_figures.py",
    "mechanism": "reproduce_mechanism.py",
    "verify": "verify_artifacts.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="rmfs-artifact")
    parser.add_argument("command", choices=SCRIPTS)
    args, remainder = parser.parse_known_args()
    script = ROOT / "scripts" / SCRIPTS[args.command]
    raise SystemExit(subprocess.call([sys.executable, str(script), *remainder], cwd=ROOT))


if __name__ == "__main__":
    main()
