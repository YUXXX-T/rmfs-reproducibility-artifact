#!/usr/bin/env python3
"""Convenience wrapper for the frozen runtime summary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    script = Path(__file__).with_name("reproduce_tables.py")
    raise SystemExit(subprocess.call([sys.executable, str(script), "--only", "runtime", *sys.argv[1:]]))
