"""S0 zero-cost trace reuse for Phase 6.1 energy conversion.

This script reads decision-trace JSONL emitted by evaluate_online_v6 and does
not run simulation or model inference. It recomputes per-context terminal
potential advantages from candidate long-risk terminal predictions, then checks:

1. Whether context-level DeltaV spread is on a comparable scale with normalized
   base-score margin.
2. Whether executed actions with high DeltaV are followed by worse future_100
   congestion outcomes.
"""

import argparse
import csv
import json
import math
import os
from typing import Iterable, List, Optional


DEFAULT_OUTCOMES = [
    "future_100_station_pressure_mean",
    "future_100_bottleneck_CVaR_mean",
    "future_100_completed_orders_delta_sum",
    "future_100_congestion_events",
    "future_100_severe_events",
    "future_100_deadlock_ratio_mean",
    "future_100_stall_ratio_mean",
    "future_100_wm_label_cost",
]


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _percentile(values: List[float], q: float) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    q = max(0.0, min(1.0, q))
    idx = int(round((len(vals) - 1) * q))
    return vals[idx]


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
            if line:
                records.append(json.loads(line))
    return records


def _selected_candidate(record: dict, candidates: List[dict]) -> Optional[dict]:
    selected_robot = record.get("selected_robot")
    for cand in candidates:
        if cand.get("is_selected"):
            return cand
    if selected_robot is not None:
        for cand in candidates:
            if cand.get("robot_id") == selected_robot:
                return cand
    if not candidates:
        return None
    return min(candidates, key=lambda c: _to_float(c.get("base_score")) or 0.0)


def _context_metrics(
    record: dict,
    terminal_key: str,
    base_key: str,
    outcomes: List[str],
) -> Optional[dict]:
    raw_candidates = record.get("candidates") or []
    candidates = []
    for cand in raw_candidates:
        terminal = _to_float(cand.get(terminal_key))
        base = _to_float(cand.get(base_key))
        if terminal is None or base is None:
            continue
        item = dict(cand)
        item["_terminal"] = terminal
        item["_base"] = base
        candidates.append(item)

    if len(candidates) < 2:
        return None

    selected = _selected_candidate(record, candidates)
    if selected is None:
        return None

    terminals = [c["_terminal"] for c in candidates]
    bases = [c["_base"] for c in candidates]
    terminal_min = min(terminals)
    terminal_max = max(terminals)
    delta_spread = terminal_max - terminal_min

    base_mu = sum(bases) / len(bases)
    base_std = _std(bases)
    if base_std > 1e-9:
        for cand in candidates:
            cand["_base_z"] = (cand["_base"] - base_mu) / base_std
    else:
        ranks = _rank(bases)
        denom = max(len(ranks) - 1, 1)
        for cand, rank in zip(candidates, ranks):
            cand["_base_z"] = rank / denom

    sorted_by_base = sorted(candidates, key=lambda c: c["_base"])
    best_base = sorted_by_base[0]
    second_base = sorted_by_base[1]
    base_raw_margin = second_base["_base"] - best_base["_base"]
    base_z_margin = second_base["_base_z"] - best_base["_base_z"]
    base_z_spread = max(c["_base_z"] for c in candidates) - min(
        c["_base_z"] for c in candidates
    )

    selected_terminal = selected["_terminal"]
    selected_delta_v = selected_terminal - terminal_min
    selected_delta_norm = (
        selected_delta_v / delta_spread if delta_spread > 1e-12 else 0.0
    )
    terminal_ranks = _rank(terminals)
    selected_idx = candidates.index(selected)
    selected_delta_rank = terminal_ranks[selected_idx] / max(
        len(candidates) - 1, 1
    )

    row = {
        "seed": record.get("seed"),
        "tick": record.get("tick"),
        "context_idx": record.get("context_idx"),
        "order_id": record.get("order_id"),
        "station_id": record.get("station_id"),
        "selected_robot": record.get("selected_robot"),
        "candidate_count": len(candidates),
        "selected_terminal_potential": selected_terminal,
        "candidate_min_terminal_potential": terminal_min,
        "candidate_max_terminal_potential": terminal_max,
        "selected_delta_v": selected_delta_v,
        "selected_delta_v_norm": selected_delta_norm,
        "selected_delta_v_rank": selected_delta_rank,
        "context_delta_v_spread": delta_spread,
        "base_raw_margin": base_raw_margin,
        "base_z_margin": base_z_margin,
        "base_z_spread": base_z_spread,
        "delta_spread_to_base_z_margin": (
            delta_spread / base_z_margin if abs(base_z_margin) > 1e-12 else None
        ),
        "baseline_is_min_terminal": abs(
            (record.get("baseline_potential") or terminal_min) - terminal_min
        ) <= 1e-9,
    }
    for key in outcomes:
        row[key] = _to_float(record.get(key))
    return row


def _bin_rows(
    rows: List[dict],
    feature: str,
    bins: int,
    outcomes: List[str],
) -> List[dict]:
    valid = []
    for row in rows:
        value = _to_float(row.get(feature))
        if value is not None:
            valid.append((value, row))
    valid.sort(key=lambda x: x[0])
    if not valid:
        return []

    out = []
    n = len(valid)
    for b in range(bins):
        start = b * n // bins
        end = (b + 1) * n // bins
        chunk = valid[start:end]
        if not chunk:
            continue
        values = [v for v, _ in chunk]
        chunk_rows = [r for _, r in chunk]
        row = {
            "feature": feature,
            "bin": b,
            "count": len(chunk),
            "value_min": round(min(values), 6),
            "value_max": round(max(values), 6),
            "value_mean": round(sum(values) / len(values), 6),
            "selected_delta_v_mean": round(
                _mean(r.get("selected_delta_v") for r in chunk_rows) or 0.0,
                6,
            ),
            "selected_delta_v_norm_mean": round(
                _mean(r.get("selected_delta_v_norm") for r in chunk_rows) or 0.0,
                6,
            ),
            "selected_delta_v_rank_mean": round(
                _mean(r.get("selected_delta_v_rank") for r in chunk_rows) or 0.0,
                6,
            ),
            "context_delta_v_spread_mean": round(
                _mean(r.get("context_delta_v_spread") for r in chunk_rows) or 0.0,
                6,
            ),
            "base_z_margin_mean": round(
                _mean(r.get("base_z_margin") for r in chunk_rows) or 0.0,
                6,
            ),
        }
        for key in outcomes:
            avg = _mean(r.get(key) for r in chunk_rows)
            row[key] = round(avg, 6) if avg is not None else None
        out.append(row)
    return out


def _is_non_decreasing(values: List[Optional[float]], tolerance: float) -> bool:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return False
    return all(vals[i + 1] + tolerance >= vals[i] for i in range(len(vals) - 1))


def _correlations(rows: List[dict], features: List[str], outcomes: List[str]) -> List[dict]:
    out = []
    for feature in features:
        for outcome in outcomes:
            xs = []
            ys = []
            for row in rows:
                x = _to_float(row.get(feature))
                y = _to_float(row.get(outcome))
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
            corr = _spearman(xs, ys)
            out.append({
                "feature": feature,
                "outcome": outcome,
                "n": len(xs),
                "spearman": round(corr, 6) if corr is not None else None,
            })
    return out


def analyze(
    records: List[dict],
    terminal_key: str,
    base_key: str,
    bins: int,
    outcomes: List[str],
    scale_min_ratio: float,
    scale_max_ratio: float,
    monotonic_tolerance: float,
) -> dict:
    rows = []
    skipped = 0
    for record in records:
        row = _context_metrics(record, terminal_key, base_key, outcomes)
        if row is None:
            skipped += 1
        else:
            rows.append(row)

    features = [
        "selected_delta_v",
        "selected_delta_v_norm",
        "selected_delta_v_rank",
        "context_delta_v_spread",
        "base_z_margin",
    ]
    bin_rows = []
    for feature in features:
        bin_rows.extend(_bin_rows(rows, feature, bins, outcomes))

    ratios = [
        r.get("delta_spread_to_base_z_margin")
        for r in rows
        if _to_float(r.get("delta_spread_to_base_z_margin")) is not None
    ]
    ratio_median = _percentile(ratios, 0.5)
    scale_gate = (
        ratio_median is not None
        and scale_min_ratio <= ratio_median <= scale_max_ratio
    )

    selected_bins = [
        r for r in bin_rows if r.get("feature") == "selected_delta_v"
    ]
    direction = {}
    for outcome in [
        "future_100_severe_events",
        "future_100_deadlock_ratio_mean",
        "future_100_wm_label_cost",
        "future_100_station_pressure_mean",
    ]:
        vals = [r.get(outcome) for r in selected_bins]
        top_minus_bottom = None
        if len(vals) >= 2 and vals[0] is not None and vals[-1] is not None:
            top_minus_bottom = vals[-1] - vals[0]
        direction[outcome] = {
            "bin_means": vals,
            "top_minus_bottom": (
                round(top_minus_bottom, 6)
                if top_minus_bottom is not None else None
            ),
            "monotonic_non_decreasing": _is_non_decreasing(
                vals, monotonic_tolerance
            ),
        }

    corr = _correlations(rows, features, outcomes)
    corr_map = {
        (c["feature"], c["outcome"]): c["spearman"]
        for c in corr
    }
    severe_corr = corr_map.get(("selected_delta_v", "future_100_severe_events"))
    deadlock_corr = corr_map.get(
        ("selected_delta_v", "future_100_deadlock_ratio_mean")
    )
    severe_dir = direction.get("future_100_severe_events", {})
    deadlock_dir = direction.get("future_100_deadlock_ratio_mean", {})

    relaxed_direction_gate = (
        severe_corr is not None
        and deadlock_corr is not None
        and severe_corr > 0.0
        and deadlock_corr > 0.0
        and (severe_dir.get("top_minus_bottom") or 0.0) > 0.0
        and (deadlock_dir.get("top_minus_bottom") or 0.0) > 0.0
    )
    strict_direction_gate = (
        relaxed_direction_gate
        and bool(severe_dir.get("monotonic_non_decreasing"))
        and bool(deadlock_dir.get("monotonic_non_decreasing"))
    )

    summary = {
        "num_input_records": len(records),
        "num_contexts": len(rows),
        "skipped_records": skipped,
        "terminal_key": terminal_key,
        "base_key": base_key,
        "bins": bins,
        "scale_check": {
            "context_delta_v_spread_mean": round(
                _mean(r.get("context_delta_v_spread") for r in rows) or 0.0,
                6,
            ),
            "context_delta_v_spread_p50": round(
                _percentile(
                    [r.get("context_delta_v_spread") for r in rows], 0.5
                ) or 0.0,
                6,
            ),
            "context_delta_v_spread_p90": round(
                _percentile(
                    [r.get("context_delta_v_spread") for r in rows], 0.9
                ) or 0.0,
                6,
            ),
            "base_z_margin_mean": round(
                _mean(r.get("base_z_margin") for r in rows) or 0.0,
                6,
            ),
            "base_z_margin_p50": round(
                _percentile([r.get("base_z_margin") for r in rows], 0.5) or 0.0,
                6,
            ),
            "delta_spread_to_base_z_margin_p50": round(ratio_median or 0.0, 6),
            "delta_spread_to_base_z_margin_p90": round(
                _percentile(ratios, 0.9) or 0.0, 6
            ),
            "scale_min_ratio": scale_min_ratio,
            "scale_max_ratio": scale_max_ratio,
            "scale_gate_pass": scale_gate,
        },
        "direction_check": direction,
        "gate": {
            "strict_direction_gate_pass": strict_direction_gate,
            "relaxed_direction_gate_pass": relaxed_direction_gate,
            "s0_gate_pass_relaxed": bool(scale_gate and relaxed_direction_gate),
            "s0_gate_pass_strict": bool(scale_gate and strict_direction_gate),
            "recommendation": (
                "Proceed to S1 if relaxed gate passes; inspect strict "
                "monotonicity before claiming a strong gate."
                if scale_gate and relaxed_direction_gate
                else "Do not proceed to S1 before revising terminal potential "
                "composition or normalization."
            ),
        },
        "bin_rows": bin_rows,
        "spearman": corr,
    }
    return summary, rows


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
        description="S0 trace-only validation for Phase 6.1 energy conversion"
    )
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--save-json", required=True)
    parser.add_argument("--save-csv", required=True)
    parser.add_argument("--save-context-csv", default=None)
    parser.add_argument("--terminal-key", default="long_risk_terminal_q90")
    parser.add_argument("--base-key", default="base_score")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--scale-min-ratio", type=float, default=0.05)
    parser.add_argument("--scale-max-ratio", type=float, default=4.0)
    parser.add_argument("--monotonic-tolerance", type=float, default=0.0)
    args = parser.parse_args()

    if args.bins <= 1:
        parser.error("--bins must be > 1")
    if args.scale_min_ratio < 0:
        parser.error("--scale-min-ratio must be >= 0")
    if args.scale_max_ratio <= 0:
        parser.error("--scale-max-ratio must be > 0")
    if args.scale_min_ratio > args.scale_max_ratio:
        parser.error("--scale-min-ratio must be <= --scale-max-ratio")

    records = _load_jsonl(args.trace_jsonl)
    summary, context_rows = analyze(
        records,
        terminal_key=args.terminal_key,
        base_key=args.base_key,
        bins=args.bins,
        outcomes=DEFAULT_OUTCOMES,
        scale_min_ratio=args.scale_min_ratio,
        scale_max_ratio=args.scale_max_ratio,
        monotonic_tolerance=args.monotonic_tolerance,
    )
    summary["source"] = args.trace_jsonl

    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _write_csv(args.save_csv, summary["bin_rows"])
    if args.save_context_csv:
        _write_csv(args.save_context_csv, context_rows)

    print(f"contexts={summary['num_contexts']} skipped={summary['skipped_records']}")
    print(
        "scale ratio p50="
        f"{summary['scale_check']['delta_spread_to_base_z_margin_p50']}"
    )
    print(
        "relaxed_gate="
        f"{summary['gate']['s0_gate_pass_relaxed']} "
        "strict_gate="
        f"{summary['gate']['s0_gate_pass_strict']}"
    )
    print(f"json={args.save_json}")
    print(f"csv={args.save_csv}")
    if args.save_context_csv:
        print(f"context_csv={args.save_context_csv}")


if __name__ == "__main__":
    main()
