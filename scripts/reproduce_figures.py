#!/usr/bin/env python3
"""Figure reproduction entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=("fig05", "fig06", "all"), default="all")
    args = parser.parse_args()
    if args.figure in ("fig05", "all"):
        script = Path(__file__).with_name("generate_fig05_mechanism.py")
        subprocess.run([sys.executable, str(script)], check=True)
    if args.figure in ("fig06", "all"):
        script = Path(__file__).with_name("generate_fig06_runtime.py")
        subprocess.run([sys.executable, str(script)], check=True)
    if args.figure == "all":
        print(
            "Figs. 2--4 remain frozen outputs; Figs. 5, 6, and 6s were "
            "regenerated from compact evidence."
        )


if __name__ == "__main__":
    main()
