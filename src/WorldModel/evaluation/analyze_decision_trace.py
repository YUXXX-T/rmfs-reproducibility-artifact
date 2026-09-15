"""
Analyze trace-only assignment diagnostics.

The input is the JSONL emitted by evaluate_online_v6 with
--decision-trace-jsonl. Each row is one assignment context and contains the
candidate-set scores observed before the online policy commits a task.
"""

import argparse
import csv
import json
import math
import os
from typing import Iterable, List, Optional


DEFAULT_FEATURES = [
    "baseline_potential",
    "candidate_min_potential",
    "candidate_potential_spread",
    "potential_gain_vs_baseline",
    "best_potential_base_gap",
    "base_margin",
    "baseline_potential_rank",
    "candidate_min_risk_score",
    "candidate_max_risk_score",
    "candidate_risk_spread",
    "baseline_base_score",
]


DEFAULT_OUTCOMES = [
    "station_pressure_mean",
    "bottleneck_CVaR_mean",
    "completed_orders_delta_sum",
    "congestion_events",
    "severe_events",
    "deadlock_ratio_mean",
    "stall_ratio_mean",
    "wm_label_cost",
]


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _rank(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _outcome_keys(horizons: List[int]) -> List[str]:
    keys = []
    for h in horizons:
        for suffix in DEFAULT_OUTCOMES:
            keys.append(f"future_{h}_{suffix}")
    return keys


def _bin_feature(records: List[dict], feature: str, bins: int,
                 outcome_keys: List[str]) -> List[dict]:
    pairs = []
    for record in records:
        value = _to_float(record.get(feature))
        if value is not None:
            pairs.append((value, record))
    pairs.sort(key=lambda x: x[0])
    if not pairs:
        return []

    rows = []
    n = len(pairs)
    for b in range(bins):
        start = b * n // bins
        end = (b + 1) * n // bins
        chunk = pairs[start:end]
        if not chunk:
            continue
        values = [v for v, _ in chunk]
        chunk_records = [r for _, r in chunk]
        row = {
            "feature": feature,
            "bin": b,
            "count": len(chunk),
            "value_min": round(min(values), 6),
            "value_max": round(max(values), 6),
            "value_mean": round(sum(values) / len(values), 6),
            "baseline_is_best_potential_rate": round(
                sum(1 for r in chunk_records
                    if r.get("baseline_is_best_potential")) / len(chunk_records),
                6,
            ),
        }
        for key in outcome_keys:
            avg = _mean(_to_float(r.get(key)) for r in chunk_records)
            row[key] = round(avg, 6) if avg is not None else None
        rows.append(row)
    return rows


def analyze(records: List[dict], features: List[str], horizons: List[int],
            bins: int) -> dict:
    outcome_keys = _outcome_keys(horizons)
    bin_rows = []
    correlations = []

    for feature in features:
        bin_rows.extend(_bin_feature(records, feature, bins, outcome_keys))
        for outcome in outcome_keys:
            xs = []
            ys = []
            for record in records:
                x = _to_float(record.get(feature))
                y = _to_float(record.get(outcome))
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
            corr = _spearman(xs, ys)
            correlations.append({
                "feature": feature,
                "outcome": outcome,
                "n": len(xs),
                "spearman": round(corr, 6) if corr is not None else None,
            })

    return {
        "num_records": len(records),
        "features": features,
        "horizons": horizons,
        "bins": bins,
        "bin_rows": bin_rows,
        "spearman": correlations,
    }


def _write_csv(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Bin trace-only assignment diagnostics by candidate-set features"
    )
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--save-json", required=True)
    parser.add_argument("--save-csv", required=True)
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--horizons", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()

    if args.bins <= 1:
        parser.error("--bins must be > 1")
    if any(h <= 0 for h in args.horizons):
        parser.error("--horizons must be positive integers")

    records = _load_jsonl(args.trace_jsonl)
    summary = analyze(records, args.features, args.horizons, args.bins)
    summary["source"] = args.trace_jsonl

    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _write_csv(args.save_csv, summary["bin_rows"])

    print(f"records={summary['num_records']} bins={len(summary['bin_rows'])}")
    print(f"json={args.save_json}")
    print(f"csv={args.save_csv}")


if __name__ == "__main__":
    main()
