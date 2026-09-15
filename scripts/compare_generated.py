#!/usr/bin/env python3
"""Compare generated tabular/statistical outputs with committed evidence."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    names = (
        "tables/table_main.csv",
        "tables/table_baselines.csv",
        "tables/table_station6.csv",
        "statistics/collapse_summary.csv",
        "statistics/paired_confidence_intervals.csv",
        "statistics/station6_paired_confidence_intervals.csv",
        "statistics/bootstrap_settings.json",
    )
    failures = []
    for name in names:
        frozen = ROOT / "artifacts" / name
        generated = ROOT / "artifacts/generated" / name
        if not generated.is_file():
            failures.append(f"missing generated file: {name}")
        elif frozen.read_bytes() != generated.read_bytes():
            failures.append(f"content differs: {name}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS: {len(names)} generated files match committed evidence")


if __name__ == "__main__":
    main()
