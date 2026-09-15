#!/usr/bin/env python3
"""Figure reproduction entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=("fig05", "all"), default="all")
    args = parser.parse_args()
    script = Path(__file__).with_name("generate_fig05_mechanism.py")
    subprocess.run([sys.executable, str(script)], check=True)
    if args.figure == "all":
        print("Figs. 2--4 remain frozen outputs; Fig. 5 was regenerated from compact event data.")


if __name__ == "__main__":
    main()
