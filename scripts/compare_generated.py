#!/usr/bin/env python3
"""Compare generated tabular/statistical outputs with committed evidence."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def normalized_text_bytes(path: Path) -> bytes:
    """Return text bytes with platform-specific line endings normalized."""

    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def parse_number(value: str) -> float | None:
    """Parse a finite CSV value as a number, or return None for text."""

    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def csv_equivalent(frozen: Path, generated: Path) -> tuple[bool, str]:
    """Compare CSV structure exactly and numeric cells with round-off tolerance."""

    with frozen.open(encoding="utf-8", newline="") as handle:
        expected_rows = list(csv.reader(handle))
    with generated.open(encoding="utf-8", newline="") as handle:
        actual_rows = list(csv.reader(handle))

    if len(expected_rows) != len(actual_rows):
        return False, f"row count differs ({len(expected_rows)} != {len(actual_rows)})"

    for row_number, (expected, actual) in enumerate(
        zip(expected_rows, actual_rows), start=1
    ):
        if len(expected) != len(actual):
            return False, (
                f"column count differs at row {row_number} "
                f"({len(expected)} != {len(actual)})"
            )
        for column_number, (expected_value, actual_value) in enumerate(
            zip(expected, actual), start=1
        ):
            if expected_value == actual_value:
                continue
            expected_number = parse_number(expected_value)
            actual_number = parse_number(actual_value)
            if (
                expected_number is not None
                and actual_number is not None
                and math.isclose(
                    expected_number,
                    actual_number,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
            ):
                continue
            return False, (
                f"row {row_number}, column {column_number}: "
                f"{expected_value!r} != {actual_value!r}"
            )
    return True, ""


def main() -> None:
    names = (
        "tables/table_main.csv",
        "tables/table_baselines.csv",
        "tables/table_station6.csv",
        "tables/table_density_scale.csv",
        "tables/table_density_scale_cells.csv",
        "statistics/collapse_summary.csv",
        "statistics/paired_confidence_intervals.csv",
        "statistics/station6_paired_confidence_intervals.csv",
        "statistics/density_scale_paired_effects.csv",
        "statistics/density_scale_cell_membership.csv",
        "statistics/bootstrap_settings.json",
    )
    failures = []
    for name in names:
        frozen = ROOT / "artifacts" / name
        generated = ROOT / "artifacts/generated" / name
        if not generated.is_file():
            failures.append(f"missing generated file: {name}")
            continue
        if frozen.suffix == ".csv":
            matches, detail = csv_equivalent(frozen, generated)
            if not matches:
                failures.append(f"content differs: {name} ({detail})")
        elif normalized_text_bytes(frozen) != normalized_text_bytes(generated):
            failures.append(f"content differs: {name}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS: {len(names)} generated files match committed evidence")


if __name__ == "__main__":
    main()
