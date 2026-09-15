"""Validate the paired station-conditioned context-defer development arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from WorldModel.evaluation.run_phase_c_station_context_defer import (
    ARM_KEY,
    SCHEMA_VERSION,
)


LOADS = ("low", "mid", "high")
SEEDS = tuple(range(551, 561))
METRICS = (
    "completed_orders",
    "completed_tasks",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "stall_ratio_mean",
    "stall_ratio_max",
    "pending_order_count",
    "open_order_count",
)

# Reuse the project's established paired non-inferiority margins.  These are
# report-only guardrails; they are never read by the online controller.
COMPLETION_RELATIVE_NONINFERIORITY = -0.05
DEADLOCK_MEAN_NONINFERIORITY_MARGIN = 0.02
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260812


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _number(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _mean_ci95(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.mean(values)
    half = (
        0.0 if len(values) == 1
        else 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    )
    return {
        "n": len(values),
        "mean": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile on zero values")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _cluster_bootstrap(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    """Bootstrap seed clusters so three loads are not treated as independent."""

    by_seed: dict[int, list[float]] = {}
    for row in rows:
        if key == "completion_relative_delta":
            value = float(row[key])
        elif key.startswith("delta_"):
            value = float(row["deltas"][key.removeprefix("delta_")])
        else:
            raise KeyError(key)
        by_seed.setdefault(int(row["seed"]), []).append(value)
    seeds = sorted(by_seed)
    if not seeds:
        return {
            "n": 0,
            "clusters": 0,
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_unit": "seed_clustered_across_loads",
        }
    rng = random.Random(BOOTSTRAP_SEED + sum(ord(char) for char in key))
    draws: list[float] = []
    for _ in range(BOOTSTRAP_REPEATS):
        sampled = [rng.choice(seeds) for _ in seeds]
        values = [value for seed in sampled for value in by_seed[seed]]
        draws.append(statistics.mean(values))
    observed = statistics.mean(
        value for values in by_seed.values() for value in values
    )
    return {
        "n": sum(len(values) for values in by_seed.values()),
        "clusters": len(seeds),
        "mean": observed,
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_unit": "seed_clustered_across_loads",
    }


def analyse(root: Path, allow_partial: bool) -> dict[str, Any]:
    arm_root = root / "per_arm" / ARM_KEY
    runs = []
    missing = []
    for load in LOADS:
        for seed in SEEDS:
            path = arm_root / f"{load}_seed{seed}.json"
            if not path.is_file():
                missing.append(f"{load}{seed}")
                continue
            payload = _read(path)
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"wrong schema: {path}")
            meta = payload.get("meta") or {}
            metrics = payload.get("metrics") or {}
            reference = (payload.get("reference") or {}).get("metrics") or {}
            admission = payload.get("station_admission_audit") or {}
            deltas = {
                key: _number(metrics.get(key)) - _number(reference.get(key))
                for key in METRICS
            }
            completion_relative_delta = (
                deltas["completed_orders"]
                / max(_number(reference.get("completed_orders")), 1.0)
            )
            runs.append({
                "path": path.as_posix(),
                "load": str(meta.get("load")),
                "seed": int(meta.get("seed", -1)),
                "audit_passed": bool((payload.get("audit") or {}).get("passed")),
                "admission_passed": bool(admission.get("passed")),
                "capacity_violation_count": int(
                    admission.get("capacity_violation_count", -1)
                ),
                "token_mismatch_count": int(
                    admission.get("token_mismatch_count", -1)
                ),
                "metrics": {key: metrics.get(key) for key in METRICS},
                "reference": {key: reference.get(key) for key in METRICS},
                "deltas": deltas,
                "completion_relative_delta": completion_relative_delta,
                "defer": {
                    "evaluations": metrics.get(
                        "station_context_defer_evaluations"
                    ),
                    "decisions": metrics.get(
                        "station_context_defer_decisions"
                    ),
                    "rate": metrics.get("station_context_defer_rate"),
                    "all_remaining_batches": metrics.get(
                        "station_context_defer_all_remaining_batches"
                    ),
                    "debt_override_selected": metrics.get(
                        "station_context_defer_debt_override_selected"
                    ),
                    "eligible_ticks_max": metrics.get(
                        "station_context_defer_eligible_ticks_max"
                    ),
                    "liveness_bound_violations": metrics.get(
                        "station_context_defer_liveness_bound_violations"
                    ),
                },
            })
    if missing and not allow_partial:
        raise RuntimeError(f"missing {len(missing)} runs: {missing}")

    def group(values: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "runs": len(values),
            "metric_deltas": {
                metric: _mean_ci95([
                    float(row["deltas"][metric]) for row in values
                ])
                for metric in METRICS
            },
            "completion_wins_ties_losses": {
                "wins": sum(row["deltas"]["completed_orders"] > 0 for row in values),
                "ties": sum(row["deltas"]["completed_orders"] == 0 for row in values),
                "losses": sum(row["deltas"]["completed_orders"] < 0 for row in values),
            },
            "completion_up_deadlock_down": sum(
                row["deltas"]["completed_orders"] > 0
                and row["deltas"]["deadlock_ratio_mean"] < 0
                for row in values
            ),
            "completion_down_deadlock_up": sum(
                row["deltas"]["completed_orders"] < 0
                and row["deltas"]["deadlock_ratio_mean"] > 0
                for row in values
            ),
            "defer_rate": _mean_ci95([
                _number(row["defer"]["rate"]) for row in values
            ]),
            "all_remaining_batches_mean": _mean_ci95([
                _number(row["defer"]["all_remaining_batches"])
                for row in values
            ]),
            "max_eligible_defer_ticks": max(
                (_number(row["defer"]["eligible_ticks_max"]) for row in values),
                default=0.0,
            ),
        }

    mechanism_passed = bool(runs) and all(
        row["audit_passed"]
        and row["admission_passed"]
        and row["capacity_violation_count"] == 0
        and row["token_mismatch_count"] == 0
        and int(row["defer"]["liveness_bound_violations"] or 0) == 0
        for row in runs
    )
    completion_relative = _cluster_bootstrap(
        runs, "completion_relative_delta"
    )
    deadlock_delta = _cluster_bootstrap(
        runs, "delta_deadlock_ratio_mean"
    )
    by_load = {
        load: group([row for row in runs if row["load"] == load])
        for load in LOADS
    }
    outcome_checks = {
        "completion_relative_ci95_lower_ge_minus_0_05": (
            completion_relative["ci95_low"] is not None
            and completion_relative["ci95_low"]
            >= COMPLETION_RELATIVE_NONINFERIORITY
        ),
        "deadlock_mean_ci95_upper_le_0_02": (
            deadlock_delta["ci95_high"] is not None
            and deadlock_delta["ci95_high"]
            <= DEADLOCK_MEAN_NONINFERIORITY_MARGIN
        ),
        "deadlock_each_load_point_le_0_02": all(
            by_load[load]["metric_deltas"]["deadlock_ratio_mean"]["mean"]
            is not None
            and by_load[load]["metric_deltas"]["deadlock_ratio_mean"]["mean"]
            <= DEADLOCK_MEAN_NONINFERIORITY_MARGIN
            for load in LOADS
        ),
    }
    outcome_guardrails = {
        "margins": {
            "completion_relative_ci95_lower": (
                COMPLETION_RELATIVE_NONINFERIORITY
            ),
            "deadlock_mean_ci95_upper": (
                DEADLOCK_MEAN_NONINFERIORITY_MARGIN
            ),
            "deadlock_each_load_point_upper": (
                DEADLOCK_MEAN_NONINFERIORITY_MARGIN
            ),
        },
        "completion_relative_delta": completion_relative,
        "deadlock_ratio_mean_delta": deadlock_delta,
        "checks": outcome_checks,
        "passed": bool(runs) and not missing and all(outcome_checks.values()),
    }
    return {
        "schema_version": "phase_c_station_context_defer_report_v1",
        "root": root.as_posix(),
        "development_only": True,
        "summary": {
            "runs": len(runs),
            "expected_runs": len(LOADS) * len(SEEDS),
            "missing": missing,
            "complete": not missing,
            "mechanism_and_lifecycle_passed": mechanism_passed,
            "outcome_guardrails_passed": outcome_guardrails["passed"],
            "development_gate_passed": (
                mechanism_passed and outcome_guardrails["passed"]
            ),
        },
        "overall": group(runs),
        "by_load": by_load,
        "outcome_guardrails": outcome_guardrails,
        "runs": runs,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Station-conditioned context defer validation",
        "",
        "Paired against Dynamic J + S1 + committed admission V1. The only "
        "new policy action is station-local context defer.",
        "",
        f"- runs: {report['summary']['runs']}/"
        f"{report['summary']['expected_runs']}",
        f"- mechanism/liveness audits: "
        f"{report['summary']['mechanism_and_lifecycle_passed']}",
        f"- outcome guardrails: "
        f"{report['summary']['outcome_guardrails_passed']}",
        f"- development gate: "
        f"{report['summary']['development_gate_passed']}",
        "",
        "| load | n | completed delta | deadlock delta | stall delta | "
        "defer rate | better both | worse both |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for load in (*LOADS, "overall"):
        row = report[load] if load == "overall" else report["by_load"][load]
        lines.append(
            f"| {load} | {row['runs']} | "
            f"{_fmt(row['metric_deltas']['completed_orders']['mean'], 2)} | "
            f"{_fmt(row['metric_deltas']['deadlock_ratio_mean']['mean'])} | "
            f"{_fmt(row['metric_deltas']['stall_ratio_mean']['mean'])} | "
            f"{_fmt(row['defer_rate']['mean'])} | "
            f"{row['completion_up_deadlock_down']} | "
            f"{row['completion_down_deadlock_up']} |"
        )
    lines.extend([
        "",
        "Positive completed delta and negative deadlock/stall delta favour "
        "the new context-defer arm.",
        "",
        "Frozen outcome guardrails: completion relative-delta CI95 lower "
        ">= -0.05; deadlock-mean delta CI95 upper <= +0.02; every load's "
        "deadlock-mean point delta <= +0.02.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = analyse(args.input_root, args.allow_partial)
    validation = args.input_root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    json_path = validation / "station_context_defer_validation.json"
    md_path = validation / "station_context_defer_validation.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    (validation / "station_context_defer_validation.sha256").write_text(
        f"{digest}  {json_path.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "summary": report["summary"],
        "overall": report["overall"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
