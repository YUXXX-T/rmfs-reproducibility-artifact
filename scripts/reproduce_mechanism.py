#!/usr/bin/env python3
"""Validate paper-aligned collapse and station-lock event statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_fig05_mechanism import read_records, validate


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    records = read_records(ROOT / "artifacts/raw/station_lock_events.csv")
    summary = validate(records)
    endpoint_counts = {
        policy: sum(row["paper_collapsed"] for row in records if row["policy"] == policy)
        for policy in ("Greedy", "ComboS1J1")
    }
    summary["collapse_definition"] = "completed_orders < 300 AND deadlock_ratio_mean >= 0.4"
    summary["endpoint_counts"] = endpoint_counts
    if not args.check:
        target = ROOT / "artifacts/generated/statistics/mechanism_summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {target}")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
