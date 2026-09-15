"""Aggregate post-cert Lyapunov shadows into a preregistered diagnosis."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Iterable


REPORT_SCHEMA_VERSION = "phase_c_postcert_lyapunov_shadow_analysis_v1"

VERDICT_MISS = "LYAPUNOV_MISSES_NO_ASSIGN_ABSORPTION"
VERDICT_NONSPECIFIC = "LYAPUNOV_FLAGS_ABSORPTION_BUT_NONDISCRIMINATIVE"
VERDICT_SPECIFIC = "LYAPUNOV_DETECTS_ABSORPTION_WITH_CONTROL_SPECIFICITY"
VERDICT_INSUFFICIENT = "INSUFFICIENT_REPRODUCTION_OR_COVERAGE"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_records(path: Path) -> list[dict]:
    rows = []
    if path.suffix == ".gz":
        handle_context = gzip.open(path, mode="rt", encoding="utf-8")
    else:
        handle_context = path.open(mode="r", encoding="utf-8")
    with handle_context as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _quantile(values: Iterable[float], probability: float) -> float | None:
    rows = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not rows:
        return None
    position = min(max(float(probability), 0.0), 1.0) * (len(rows) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return float(rows[left])
    weight = position - left
    return float(rows[left] * (1.0 - weight) + rows[right] * weight)


def _distribution(values: Iterable[float]) -> dict:
    rows = [float(value) for value in values if math.isfinite(float(value))]
    if not rows:
        return {
            "n": 0,
            "mean": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(rows),
        "mean": float(sum(rows) / len(rows)),
        "p05": _quantile(rows, 0.05),
        "p50": _quantile(rows, 0.50),
        "p95": _quantile(rows, 0.95),
        "min": float(min(rows)),
        "max": float(max(rows)),
    }


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = 0.5 * (cursor + 1 + end)
        for offset in range(cursor, end):
            ranks[order[offset]] = average
        cursor = end
    return ranks


def _auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    values = [float(value) for value in positive + negative]
    ranks = _average_ranks(values)
    n_pos = len(positive)
    n_neg = len(negative)
    rank_sum = sum(ranks[:n_pos])
    statistic = rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(statistic / (n_pos * n_neg))


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum(
        (x - mean_left) * (y - mean_right) for x, y in zip(left, right)
    )
    denom_left = math.sqrt(sum((x - mean_left) ** 2 for x in left))
    denom_right = math.sqrt(sum((y - mean_right) ** 2 for y in right))
    if denom_left <= 0.0 or denom_right <= 0.0:
        return None
    return float(numerator / (denom_left * denom_right))


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(_average_ranks(left), _average_ranks(right))


def _advantage(row: dict, horizon: int, relative: bool = True) -> float:
    entry = row["horizons"][str(int(horizon))]
    key = (
        "analytic_assign_advantage_relative"
        if relative
        else "analytic_assign_advantage"
    )
    return float(entry[key])


def _selected_no_assign(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("selected_action_type") == "no_assign"]


def _tail(rows: list[dict], start: int = 1000, end: int = 1500) -> list[dict]:
    return [row for row in rows if start <= int(row["tick"]) < end]


def _resolve_record_path(summary_path: Path, report: dict) -> Path:
    raw = Path(report["records"]["path"])
    if raw.is_file():
        return raw
    sibling = summary_path.with_name(summary_path.stem + "_records.jsonl.gz")
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--primary-horizon", type=int, default=10)
    parser.add_argument("--tail-start", type=int, default=1000)
    parser.add_argument("--tail-end", type=int, default=1500)
    parser.add_argument("--minimum-failure-contexts", type=int, default=50)
    parser.add_argument("--minimum-control-contexts", type=int, default=20)
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not 0 <= int(args.tail_start) < int(args.tail_end):
        raise ValueError("invalid tail interval")

    runs = []
    for raw in args.inputs:
        summary_path = Path(raw)
        report = _read_json(summary_path)
        records_path = _resolve_record_path(summary_path, report)
        records = _read_records(records_path)
        runs.append({
            "summary_path": summary_path,
            "records_path": records_path,
            "report": report,
            "records": records,
            "role": str(report["meta"]["case_role"]),
            "load": str(report["meta"]["load"]),
            "seed": int(report["meta"]["seed"]),
        })

    reproduction_passed = all(
        bool(run["report"]["reproduction_audit"]["passed"]) for run in runs
    )
    failure_roles = {"no_assign_failure", "near_absorbing_failure"}
    control_roles = {"healthy_control", "positive_control"}
    failure_rows = []
    control_tail_rows = []
    control_all_rows = []
    for run in runs:
        if run["role"] in failure_roles:
            failure_rows.extend(_selected_no_assign(_tail(
                run["records"], int(args.tail_start), int(args.tail_end)
            )))
        if run["role"] in control_roles:
            control_all_rows.extend(_selected_no_assign(run["records"]))
            control_tail_rows.extend(_selected_no_assign(_tail(
                run["records"], int(args.tail_start), int(args.tail_end)
            )))
    controls = (
        control_tail_rows
        if len(control_tail_rows) >= int(args.minimum_control_contexts)
        else control_all_rows
    )

    horizon = int(args.primary_horizon)
    failure_adv = [_advantage(row, horizon) for row in failure_rows]
    control_adv = [_advantage(row, horizon) for row in controls]
    failure_raw = [_advantage(row, horizon, relative=False) for row in failure_rows]
    control_raw = [_advantage(row, horizon, relative=False) for row in controls]
    failure_opposition = float(
        sum(value > 1e-12 for value in failure_raw) / max(len(failure_raw), 1)
    )
    control_opposition = float(
        sum(value > 1e-12 for value in control_raw) / max(len(control_raw), 1)
    )
    specificity_auc = _auc(failure_adv, control_adv)

    long_streak_rows = [
        row
        for run in runs
        for row in run["records"]
        if int(row.get("chain_no_assign_streak_after", 0)) >= 20
    ]
    zero_streak_rows = [
        row
        for run in runs
        for row in run["records"]
        if int(row.get("chain_no_assign_streak_after", 0)) == 0
    ]
    streak_auc = _auc(
        [_advantage(row, horizon) for row in long_streak_rows],
        [_advantage(row, horizon) for row in zero_streak_rows],
    )

    all_rows = [row for run in runs for row in run["records"]]
    wm_values = [float(row["wm_no_assign_advantage"]) for row in all_rows]
    analytic_values = [_advantage(row, horizon) for row in all_rows]
    wm_analytic_spearman = _spearman(wm_values, analytic_values)
    naive_noassign_rate_failure = float(
        sum(bool(row["naive_immediate_l0_prefers_no_assign"]) for row in failure_rows)
        / max(len(failure_rows), 1)
    )

    enough_coverage = (
        len(failure_rows) >= int(args.minimum_failure_contexts)
        and len(controls) >= int(args.minimum_control_contexts)
    )
    if not reproduction_passed or not enough_coverage:
        verdict = VERDICT_INSUFFICIENT
    elif failure_opposition < 0.80:
        verdict = VERDICT_MISS
    elif specificity_auc is None or specificity_auc < 0.70:
        verdict = VERDICT_NONSPECIFIC
    else:
        verdict = VERDICT_SPECIFIC

    per_run = {}
    for run in runs:
        noassign = _selected_no_assign(run["records"])
        tail_noassign = _selected_no_assign(_tail(
            run["records"], int(args.tail_start), int(args.tail_end)
        ))
        per_run[f"{run['load']}_seed{run['seed']}"] = {
            "case_role": run["role"],
            "records": len(run["records"]),
            "selected_no_assign": len(noassign),
            "tail_selected_no_assign": len(tail_noassign),
            "tail_no_assign_ratio": float(
                len(tail_noassign)
                / max(len(_tail(
                    run["records"], int(args.tail_start), int(args.tail_end)
                )), 1)
            ),
            "tail_relative_advantage": _distribution(
                _advantage(row, horizon) for row in tail_noassign
            ),
            "max_chain_no_assign_streak": max(
                (
                    int(row.get("chain_no_assign_streak_after", 0))
                    for row in run["records"]
                ),
                default=0,
            ),
            "reproduction_passed": bool(
                run["report"]["reproduction_audit"]["passed"]
            ),
        }

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": "POSTCERT_FAILURE_MECHANISM_DIAGNOSTIC_NOT_POLICY_TUNING",
        "verdict": verdict,
        "preregistered_rule": {
            "failure_tail": [int(args.tail_start), int(args.tail_end)],
            "primary_horizon": horizon,
            "minimum_failure_contexts": int(args.minimum_failure_contexts),
            "minimum_control_contexts": int(args.minimum_control_contexts),
            "failure_opposition_required": 0.80,
            "specificity_auc_required": 0.70,
            "certification_seeds_must_not_set_future_lambda_or_threshold": True,
        },
        "checks": {
            "all_replays_exact": reproduction_passed,
            "enough_failure_contexts": (
                len(failure_rows) >= int(args.minimum_failure_contexts)
            ),
            "enough_control_contexts": (
                len(controls) >= int(args.minimum_control_contexts)
            ),
            "failure_opposition_rate": failure_opposition,
            "control_opposition_rate": control_opposition,
            "failure_vs_control_specificity_auc": specificity_auc,
        },
        "failure_tail_relative_advantage": _distribution(failure_adv),
        "control_relative_advantage": _distribution(control_adv),
        "failure_tail_raw_advantage": _distribution(failure_raw),
        "control_raw_advantage": _distribution(control_raw),
        "streak_diagnostics": {
            "long_streak_contexts": len(long_streak_rows),
            "zero_streak_contexts": len(zero_streak_rows),
            "long_streak_vs_zero_streak_auc": streak_auc,
        },
        "wm_vs_analytic": {
            "spearman": wm_analytic_spearman,
            "meaning": (
                "positive means the analytic urgency rises where the frozen WM "
                "also gives NO_ASSIGN a larger score advantage"
            ),
        },
        "naive_full_l0_warning": {
            "failure_tail_immediate_arrival_term_prefers_no_assign_rate": (
                naive_noassign_rate_failure
            ),
            "warning": (
                "The immediate arrival-barrier delta is not the H-step work "
                "dissipation signal. Adding naive instantaneous L0 can reinforce "
                "NO_ASSIGN even when analytic L_work opposes it."
            ),
        },
        "per_run": per_run,
        "next_step": (
            "If and only if the verdict has control specificity, choose a fusion "
            "rule on new development seeds. Otherwise repair sequential NO_ASSIGN "
            "coverage/semantics before any Lyapunov policy test."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(f"saved: {output_path}")
    print("verdict =", verdict)
    print("failure contexts =", len(failure_rows))
    print("control contexts =", len(controls))
    print("failure opposition rate =", failure_opposition)
    print("control opposition rate =", control_opposition)
    print("specificity AUC =", specificity_auc)
    print("long-streak AUC =", streak_auc)
    print(
        "naive immediate L0 prefers NO_ASSIGN rate =",
        naive_noassign_rate_failure,
    )


if __name__ == "__main__":
    main()
