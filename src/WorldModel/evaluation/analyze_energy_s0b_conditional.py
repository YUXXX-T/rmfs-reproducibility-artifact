"""Conditional S0b analysis for Phase 6.1 energy conversion.

This is a zero-cost offline analysis over an existing decision-trace JSONL.
It implements P0 + P1 + P2 from
energy_s0_review_and_V_redesign.md:

P0: tie / discriminability diagnostics for eight potential combinations.
P1: stratified direction checks by two state proxies.
P2: residualized direction checks after regressing outcomes on confounders.

The script does not import torch and does not run simulation or model inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


EPS = 1e-6


OUTCOMES = [
    "future_100_severe_events",
    "future_100_deadlock_ratio_mean",
    "future_100_station_pressure_mean",
    "future_100_wm_label_cost",
    "future_100_completed_orders_delta_sum",
]


STATE_PROXIES = [
    ("baseline_base_score", "baseline_base_score"),
    ("candidate_min_potential", "candidate_min_potential"),
]


COMBOS = [
    {
        "name": "C1_terminal_only",
        "desc": "terminal_q90",
        "terms": [
            ("long_risk_terminal_q90", 1.0),
        ],
    },
    {
        "name": "C2_tail_mix",
        "desc": "terminal_q90 + 0.5*cvar_q90 + 0.5*peak_q95",
        "terms": [
            ("long_risk_terminal_q90", 1.0),
            ("long_risk_cvar_q90", 0.5),
            ("long_risk_peak_q95", 0.5),
        ],
    },
    {
        "name": "C3_combo_head",
        "desc": "long_risk_combo_q90",
        "terms": [
            ("long_risk_combo_q90", 1.0),
        ],
    },
    {
        "name": "C4_risk_short",
        "desc": "risk_max",
        "terms": [
            ("risk_max", 1.0),
        ],
    },
    {
        "name": "C5_inj_raw",
        "desc": "station_injection_raw",
        "terms": [
            ("station_injection_raw", 1.0),
        ],
    },
    {
        "name": "C6_term_plus_inj",
        "desc": "terminal_q90 + station_injection_raw",
        "terms": [
            ("long_risk_terminal_q90", 1.0),
            ("station_injection_raw", 1.0),
        ],
    },
    {
        "name": "C7_short_plus_long",
        "desc": "risk_max + 0.5*terminal_q90",
        "terms": [
            ("risk_max", 1.0),
            ("long_risk_terminal_q90", 0.5),
        ],
    },
    {
        "name": "C8_guard_combo",
        "desc": "potential_score",
        "terms": [
            ("potential_score", 1.0),
        ],
    },
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


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _percentile(values: Iterable[Optional[float]], q: float) -> Optional[float]:
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


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _combo_value(candidate: dict, combo: dict) -> Optional[float]:
    total = 0.0
    for key, weight in combo["terms"]:
        val = _to_float(candidate.get(key))
        if val is None:
            return None
        total += weight * val
    return total


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


def _base_z_margin(candidates: List[dict]) -> Optional[float]:
    bases = []
    for cand in candidates:
        val = _to_float(cand.get("base_score"))
        if val is None:
            return None
        bases.append(val)
    if len(bases) < 2:
        return None
    std = _std(bases)
    if std > 1e-9:
        zs = [(v - sum(bases) / len(bases)) / std for v in bases]
    else:
        ranks = _rank(bases)
        denom = max(len(ranks) - 1, 1)
        zs = [r / denom for r in ranks]
    order = sorted(range(len(bases)), key=lambda i: bases[i])
    return zs[order[1]] - zs[order[0]]


def _record_combo_rows(records: List[dict]) -> Tuple[List[dict], List[dict]]:
    rows = []
    p0_missing = []
    for ridx, record in enumerate(records):
        raw_candidates = record.get("candidates") or []
        selected = _selected_candidate(record, raw_candidates)
        if selected is None or len(raw_candidates) < 2:
            continue
        base_z_margin = _base_z_margin(raw_candidates)
        if base_z_margin is None:
            continue

        for combo in COMBOS:
            values = []
            candidates = []
            for cand in raw_candidates:
                val = _combo_value(cand, combo)
                if val is None:
                    continue
                item = dict(cand)
                item["_combo_value"] = val
                candidates.append(item)
                values.append(val)
            selected_value = None
            selected_robot = selected.get("robot_id")
            for cand in candidates:
                if cand.get("robot_id") == selected_robot:
                    selected_value = cand["_combo_value"]
                    break
            if selected_value is None or len(values) < 2:
                p0_missing.append({
                    "record_idx": ridx,
                    "combo": combo["name"],
                    "reason": "missing_selected_or_candidates",
                })
                continue

            min_value = min(values)
            max_value = max(values)
            spread = max_value - min_value
            selected_delta = selected_value - min_value
            row = {
                "combo": combo["name"],
                "combo_desc": combo["desc"],
                "seed": record.get("seed"),
                "tick": _to_float(record.get("tick")),
                "context_idx": record.get("context_idx"),
                "order_id": record.get("order_id"),
                "station_id": record.get("station_id"),
                "candidate_count": len(values),
                "selected_value": selected_value,
                "min_value": min_value,
                "max_value": max_value,
                "spread": spread,
                "selected_delta": selected_delta,
                "selected_delta_positive": selected_delta > EPS,
                "selected_delta_zero": abs(selected_delta) <= EPS,
                "tie": spread < EPS,
                "baseline_base_score": _to_float(record.get("baseline_base_score")),
                "candidate_min_potential": _to_float(
                    record.get("candidate_min_potential")
                ),
                "base_z_margin": base_z_margin,
            }
            for outcome in OUTCOMES:
                row[outcome] = _to_float(record.get(outcome))
            rows.append(row)
    return rows, p0_missing


def _assign_strata(rows: List[dict], proxy_key: str, bins: int = 5) -> Dict[int, int]:
    valid = [
        (idx, _to_float(row.get(proxy_key)))
        for idx, row in enumerate(rows)
        if _to_float(row.get(proxy_key)) is not None
    ]
    valid.sort(key=lambda x: x[1])
    strata = {}
    n = len(valid)
    for rank, (idx, _) in enumerate(valid):
        strata[idx] = min(bins - 1, rank * bins // max(n, 1))
    return strata


def _ols_residuals(rows: List[dict], outcome: str) -> Dict[int, float]:
    xs = []
    ys = []
    ids = []
    for idx, row in enumerate(rows):
        y = _to_float(row.get(outcome))
        controls = [
            1.0,
            _to_float(row.get("baseline_base_score")),
            _to_float(row.get("candidate_min_potential")),
            _to_float(row.get("base_z_margin")),
            _to_float(row.get("tick")),
        ]
        if y is None or any(v is None for v in controls):
            continue
        xs.append(controls)
        ys.append(y)
        ids.append(idx)
    if len(xs) < 8:
        return {}
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    beta, *_ = np.linalg.lstsq(x_arr, y_arr, rcond=None)
    pred = x_arr @ beta
    residuals = y_arr - pred
    return {idx: float(resid) for idx, resid in zip(ids, residuals)}


def _corr_for_subset(
    rows: List[dict],
    indices: List[int],
    y_key: str,
    x_key: str = "selected_delta",
) -> Tuple[int, Optional[float], Optional[float], Optional[float]]:
    xs = []
    ys = []
    for idx in indices:
        x = _to_float(rows[idx].get(x_key))
        y = _to_float(rows[idx].get(y_key))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    if not xs:
        return 0, None, None, None
    corr = _spearman(xs, ys)
    return len(xs), corr, _mean(xs), _mean(ys)


def _p0_rows(rows: List[dict]) -> List[dict]:
    out = []
    for combo in [c["name"] for c in COMBOS]:
        subset = [r for r in rows if r["combo"] == combo]
        if not subset:
            continue
        tie_count = sum(1 for r in subset if r["tie"])
        zero_count = sum(1 for r in subset if r["selected_delta_zero"])
        pos_count = sum(1 for r in subset if r["selected_delta_positive"])
        out.append({
            "protocol": "P0_tie_diagnostic",
            "combo": combo,
            "n": len(subset),
            "tie_rate": round(tie_count / len(subset), 6),
            "selected_delta_zero_rate": round(zero_count / len(subset), 6),
            "selected_delta_positive_rate": round(pos_count / len(subset), 6),
            "spread_mean": round(_mean(r["spread"] for r in subset) or 0.0, 6),
            "spread_p50": round(_percentile((r["spread"] for r in subset), 0.5) or 0.0, 6),
            "spread_p90": round(_percentile((r["spread"] for r in subset), 0.9) or 0.0, 6),
            "selected_delta_mean": round(
                _mean(r["selected_delta"] for r in subset) or 0.0,
                6,
            ),
            "p0_pass": tie_count / len(subset) < 0.5,
        })
    return out


def _p1_p2_rows(rows: List[dict], strata_bins: int = 5) -> List[dict]:
    out = []
    for combo in [c["name"] for c in COMBOS]:
        combo_rows = [r for r in rows if r["combo"] == combo]
        if not combo_rows:
            continue
        residuals_by_outcome = {
            outcome: _ols_residuals(combo_rows, outcome)
            for outcome in OUTCOMES
        }
        for proxy_name, proxy_key in STATE_PROXIES:
            strata = _assign_strata(combo_rows, proxy_key, bins=strata_bins)
            for stratum_name, stratum_id in [
                ("calm", 0),
                ("danger", strata_bins - 1),
            ]:
                base_indices = [
                    idx for idx, sid in strata.items() if sid == stratum_id
                ]
                indices = [
                    idx for idx in base_indices
                    if combo_rows[idx].get("selected_delta_positive")
                ]
                for outcome in OUTCOMES:
                    n, corr, x_mean, y_mean = _corr_for_subset(
                        combo_rows, indices, outcome
                    )
                    out.append({
                        "protocol": "P1_stratified_direct",
                        "combo": combo,
                        "state_proxy": proxy_name,
                        "stratum": stratum_name,
                        "delta_scope": "selected_delta_gt_0",
                        "outcome": outcome,
                        "n": n,
                        "spearman": round(corr, 6) if corr is not None else None,
                        "selected_delta_mean": round(x_mean, 6) if x_mean is not None else None,
                        "outcome_mean": round(y_mean, 6) if y_mean is not None else None,
                    })

                    residuals = residuals_by_outcome.get(outcome, {})
                    for idx, resid in residuals.items():
                        combo_rows[idx][f"residual_{outcome}"] = resid
                    n2, corr2, x_mean2, y_mean2 = _corr_for_subset(
                        combo_rows,
                        indices,
                        f"residual_{outcome}",
                    )
                    out.append({
                        "protocol": "P2_residualized",
                        "combo": combo,
                        "state_proxy": proxy_name,
                        "stratum": stratum_name,
                        "delta_scope": "selected_delta_gt_0",
                        "outcome": outcome,
                        "n": n2,
                        "spearman": round(corr2, 6) if corr2 is not None else None,
                        "selected_delta_mean": round(x_mean2, 6) if x_mean2 is not None else None,
                        "outcome_mean": round(y_mean2, 6) if y_mean2 is not None else None,
                    })
    return out


def _gate_summary(p0: List[dict], p_rows: List[dict]) -> dict:
    p0_pass = {row["combo"]: bool(row["p0_pass"]) for row in p0}
    outcomes = {"future_100_severe_events", "future_100_deadlock_ratio_mean"}
    candidates = []
    for combo in [c["name"] for c in COMBOS]:
        if not p0_pass.get(combo, False):
            continue
        for protocol in ("P1_stratified_direct", "P2_residualized"):
            proxy_rows = [
                row for row in p_rows
                if row["combo"] == combo
                and row["protocol"] == protocol
                and row["stratum"] == "danger"
                and row["outcome"] in outcomes
                and row["spearman"] is not None
                and row["spearman"] >= 0.10
            ]
            by_outcome = {}
            for row in proxy_rows:
                by_outcome.setdefault(row["outcome"], set()).add(
                    row["state_proxy"]
                )
            for outcome, proxies in by_outcome.items():
                if len(proxies) >= 2:
                    candidates.append({
                        "combo": combo,
                        "protocol": protocol,
                        "outcome": outcome,
                        "state_proxies": sorted(proxies),
                    })
    return {
        "gated_drift_feasible": bool(candidates),
        "passing_candidates": candidates,
        "rule": (
            "Pass if at least one combo has P0 tie_rate < 0.5 and, in the "
            "danger stratum with selected_delta>0, severe or deadlock "
            "Spearman >= +0.10 under both state proxies."
        ),
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
        description="Run S0b P0+P1+P2 conditional analysis over decision trace"
    )
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--save-json", required=True)
    parser.add_argument("--save-csv", required=True)
    parser.add_argument("--save-context-csv", default=None)
    args = parser.parse_args()

    records = _load_jsonl(args.trace_jsonl)
    rows, missing = _record_combo_rows(records)
    p0 = _p0_rows(rows)
    p1_p2 = _p1_p2_rows(rows)
    all_results = p0 + p1_p2
    gate = _gate_summary(p0, p1_p2)

    summary = {
        "source": args.trace_jsonl,
        "num_input_records": len(records),
        "num_combo_context_rows": len(rows),
        "num_missing_combo_rows": len(missing),
        "combos": COMBOS,
        "state_proxies": STATE_PROXIES,
        "outcomes": OUTCOMES,
        "p0": p0,
        "results": p1_p2,
        "gate": gate,
    }

    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _write_csv(args.save_csv, all_results)
    if args.save_context_csv:
        _write_csv(args.save_context_csv, rows)

    print(f"records={len(records)} combo_context_rows={len(rows)}")
    print(f"missing_combo_rows={len(missing)}")
    print(f"gated_drift_feasible={gate['gated_drift_feasible']}")
    if gate["passing_candidates"]:
        for cand in gate["passing_candidates"]:
            print("PASS", cand)
    print(f"json={args.save_json}")
    print(f"csv={args.save_csv}")
    if args.save_context_csv:
        print(f"context_csv={args.save_context_csv}")


if __name__ == "__main__":
    main()
