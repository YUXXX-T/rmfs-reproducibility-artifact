"""Validate the pipeline+phi station-context defer V2 paired experiment.

The report keeps two paired contrasts separate:

* V2 versus Dynamic-J + S1 + committed admission V1 (primary baseline);
* V2 versus the over-conservative ready-max defer V1 (mechanism contrast).

Low/mid/high observations sharing a seed are resampled as one cluster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Mapping

from WorldModel.core.station_context_defer import (
    STATION_CONTEXT_DEFER_PIPELINE_PHI_SCHEMA_VERSION,
    STATION_CONTEXT_DEFER_RISK_PIPELINE_PHI_V2,
)
from WorldModel.evaluation.run_phase_c_station_context_defer_pipeline_phi import (
    ARM_KEY,
    BASELINE_ARM_KEY,
    READY_MAX_V1_ARM_KEY,
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
CONTRASTS = {
    "vs_dynamic_j_admission_v1": "reference",
    "vs_ready_max_defer_v1": "ready_max_v1_reference",
}

# Analysis-only guardrails frozen before the V2 online outcomes are observed.
COMPLETION_RELATIVE_NONINFERIORITY = -0.05
DEADLOCK_MEAN_NONINFERIORITY_MARGIN = 0.02
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260813


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
        0.0
        if len(values) == 1
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
    contrast: str,
    value_path: tuple[str, ...],
) -> dict[str, Any]:
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        value: Any = row["contrasts"][contrast]
        for key in value_path:
            value = value[key]
        value = float(value)
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
    salt = sum(ord(char) for char in f"{contrast}:{'/'.join(value_path)}")
    rng = random.Random(BOOTSTRAP_SEED + salt)
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


def _reference_metrics(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    block = payload.get(key) or {}
    metrics = block.get("metrics") or {}
    if not isinstance(metrics, dict):
        raise ValueError(f"{key}.metrics must be an object")
    return metrics


def _contrast(metrics: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    deltas = {
        metric: _number(metrics.get(metric)) - _number(reference.get(metric))
        for metric in METRICS
    }
    reference_completed = max(_number(reference.get("completed_orders")), 1.0)
    completion_relative_delta = deltas["completed_orders"] / reference_completed
    return {
        "deltas": deltas,
        "completion_relative_delta": completion_relative_delta,
        "completion_up_deadlock_down": (
            deltas["completed_orders"] > 0
            and deltas["deadlock_ratio_mean"] < 0
        ),
        "completion_down_deadlock_up": (
            deltas["completed_orders"] < 0
            and deltas["deadlock_ratio_mean"] > 0
        ),
    }


def _group(rows: list[dict[str, Any]], contrast: str) -> dict[str, Any]:
    return {
        "runs": len(rows),
        "metric_deltas": {
            metric: _mean_ci95([
                float(row["contrasts"][contrast]["deltas"][metric])
                for row in rows
            ])
            for metric in METRICS
        },
        "completion_relative_delta": _mean_ci95([
            float(row["contrasts"][contrast]["completion_relative_delta"])
            for row in rows
        ]),
        "completion_wins_ties_losses": {
            "wins": sum(
                row["contrasts"][contrast]["deltas"]["completed_orders"] > 0
                for row in rows
            ),
            "ties": sum(
                row["contrasts"][contrast]["deltas"]["completed_orders"] == 0
                for row in rows
            ),
            "losses": sum(
                row["contrasts"][contrast]["deltas"]["completed_orders"] < 0
                for row in rows
            ),
        },
        "completion_up_deadlock_down": sum(
            row["contrasts"][contrast]["completion_up_deadlock_down"]
            for row in rows
        ),
        "completion_down_deadlock_up": sum(
            row["contrasts"][contrast]["completion_down_deadlock_up"]
            for row in rows
        ),
    }


def analyse(root: Path, allow_partial: bool = False) -> dict[str, Any]:
    arm_root = root / "per_arm" / ARM_KEY
    runs: list[dict[str, Any]] = []
    missing: list[str] = []
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
            audit = payload.get("audit") or {}
            admission = payload.get("station_admission_audit") or {}
            baseline = _reference_metrics(payload, "reference")
            ready_v1 = _reference_metrics(payload, "ready_max_v1_reference")
            ready_v1_defer = (
                payload.get("ready_max_v1_reference") or {}
            ).get("defer") or {}
            reference_arm = (payload.get("reference") or {}).get("arm")
            ready_v1_arm = (
                payload.get("ready_max_v1_reference") or {}
            ).get("arm")
            if reference_arm != BASELINE_ARM_KEY:
                raise ValueError(f"wrong primary reference arm: {path}")
            if ready_v1_arm != READY_MAX_V1_ARM_KEY:
                raise ValueError(f"wrong V1 reference arm: {path}")
            checks = audit.get("checks") or {}
            defer_evaluations = int(
                metrics.get("station_context_defer_evaluations", 0)
            )
            ready_would_dominate = int(
                metrics.get(
                    "station_context_defer_ready_would_dominate_evaluations",
                    0,
                )
            )
            all_defer_batches = int(
                metrics.get(
                    "station_context_defer_all_remaining_batches", 0
                )
            )
            total_batches = int(
                metrics.get("station_context_defer_batches", 0)
            )
            v1_defer_rate_raw = ready_v1_defer.get("rate")
            v1_batches_raw = ready_v1_defer.get("batches")
            v1_all_defer_raw = ready_v1_defer.get("all_remaining_batches")
            v1_defer_available = (
                v1_defer_rate_raw is not None
                and v1_batches_raw is not None
                and v1_all_defer_raw is not None
                and int(v1_batches_raw) > 0
            )
            runs.append({
                "path": path.as_posix(),
                "load": str(meta.get("load")),
                "seed": int(meta.get("seed", -1)),
                "ticks": int(meta.get("ticks", -1)),
                "audit_passed": bool(audit.get("passed")),
                "risk_mode_passed": bool(checks.get("pipeline_phi_risk_mode")),
                "ready_diagnostic_only_passed": bool(
                    checks.get("ready_contention_is_diagnostic_only")
                ),
                "admission_passed": bool(admission.get("passed")),
                "capacity_violation_count": int(
                    admission.get("capacity_violation_count", -1)
                ),
                "token_mismatch_count": int(
                    admission.get("token_mismatch_count", -1)
                ),
                "liveness_bound_violations": int(
                    metrics.get(
                        "station_context_defer_liveness_bound_violations", -1
                    )
                ),
                "metrics": {metric: metrics.get(metric) for metric in METRICS},
                "contrasts": {
                    "vs_dynamic_j_admission_v1": _contrast(metrics, baseline),
                    "vs_ready_max_defer_v1": _contrast(metrics, ready_v1),
                },
                "defer": {
                    "evaluations": defer_evaluations,
                    "decisions": int(
                        metrics.get("station_context_defer_decisions", 0)
                    ),
                    "rate": _number(
                        metrics.get("station_context_defer_rate")
                    ),
                    "batches": total_batches,
                    "all_remaining_batches": all_defer_batches,
                    "all_remaining_batch_rate": (
                        all_defer_batches / total_batches
                        if total_batches > 0 else 0.0
                    ),
                    "ready_would_dominate_evaluations": ready_would_dominate,
                    "ready_would_dominate_rate": (
                        ready_would_dominate / defer_evaluations
                        if defer_evaluations > 0 else 0.0
                    ),
                    "risk_mean": _number(
                        metrics.get("station_context_defer_risk_mean")
                    ),
                    "pipeline_excess_mean": _number(
                        metrics.get(
                            "station_context_defer_pipeline_excess_mean"
                        )
                    ),
                    "ready_contention_mean": _number(
                        metrics.get(
                            "station_context_defer_ready_contention_mean"
                        )
                    ),
                    "phi_pressure_mean": _number(
                        metrics.get("station_context_defer_phi_pressure_mean")
                    ),
                    "eligible_ticks_max": int(
                        metrics.get(
                            "station_context_defer_eligible_ticks_max", 0
                        )
                    ),
                },
                "ready_max_v1_defer": {
                    "available": v1_defer_available,
                    "rate": (
                        float(v1_defer_rate_raw)
                        if v1_defer_rate_raw is not None else None
                    ),
                    "batches": int(v1_batches_raw or 0),
                    "all_remaining_batches": int(
                        v1_all_defer_raw or 0
                    ),
                    "all_remaining_batch_rate": (
                        int(v1_all_defer_raw or 0) / int(v1_batches_raw)
                        if v1_defer_available else None
                    ),
                },
            })

    if missing and not allow_partial:
        raise RuntimeError(f"missing {len(missing)} runs: {missing}")

    mechanism_passed = bool(runs) and all(
        row["audit_passed"]
        and row["risk_mode_passed"]
        and row["ready_diagnostic_only_passed"]
        and row["admission_passed"]
        and row["capacity_violation_count"] == 0
        and row["token_mismatch_count"] == 0
        and row["liveness_bound_violations"] == 0
        for row in runs
    )

    contrasts: dict[str, Any] = {}
    for contrast in CONTRASTS:
        completion_relative = _cluster_bootstrap(
            runs, contrast, ("completion_relative_delta",)
        )
        deadlock_delta = _cluster_bootstrap(
            runs, contrast, ("deltas", "deadlock_ratio_mean")
        )
        overall = _group(runs, contrast)
        by_load = {
            load: _group(
                [row for row in runs if row["load"] == load], contrast
            )
            for load in LOADS
        }
        contrasts[contrast] = {
            "reference_arm": (
                BASELINE_ARM_KEY
                if contrast == "vs_dynamic_j_admission_v1"
                else READY_MAX_V1_ARM_KEY
            ),
            "overall": overall,
            "by_load": by_load,
            "clustered_completion_relative_delta": completion_relative,
            "clustered_deadlock_ratio_mean_delta": deadlock_delta,
        }

    primary = contrasts["vs_dynamic_j_admission_v1"]
    primary_checks = {
        "completion_relative_ci95_lower_ge_minus_0_05": (
            primary["clustered_completion_relative_delta"]["ci95_low"]
            is not None
            and primary["clustered_completion_relative_delta"]["ci95_low"]
            >= COMPLETION_RELATIVE_NONINFERIORITY
        ),
        "deadlock_mean_ci95_upper_le_0_02": (
            primary["clustered_deadlock_ratio_mean_delta"]["ci95_high"]
            is not None
            and primary["clustered_deadlock_ratio_mean_delta"]["ci95_high"]
            <= DEADLOCK_MEAN_NONINFERIORITY_MARGIN
        ),
        "deadlock_each_load_point_le_0_02": all(
            primary["by_load"][load]["metric_deltas"]
            ["deadlock_ratio_mean"]["mean"] is not None
            and primary["by_load"][load]["metric_deltas"]
            ["deadlock_ratio_mean"]["mean"]
            <= DEADLOCK_MEAN_NONINFERIORITY_MARGIN
            for load in LOADS
        ),
    }
    primary_guardrails = {
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
        "checks": primary_checks,
        "passed": bool(runs) and not missing and all(primary_checks.values()),
    }

    defer_summary = {
        "defer_rate": _mean_ci95([
            float(row["defer"]["rate"]) for row in runs
        ]),
        "all_remaining_batch_rate": _mean_ci95([
            float(row["defer"]["all_remaining_batch_rate"]) for row in runs
        ]),
        "ready_would_dominate_rate": _mean_ci95([
            float(row["defer"]["ready_would_dominate_rate"])
            for row in runs
        ]),
        "risk_mean": _mean_ci95([
            float(row["defer"]["risk_mean"]) for row in runs
        ]),
        "pipeline_excess_mean": _mean_ci95([
            float(row["defer"]["pipeline_excess_mean"]) for row in runs
        ]),
        "ready_contention_mean_diagnostic": _mean_ci95([
            float(row["defer"]["ready_contention_mean"]) for row in runs
        ]),
        "phi_pressure_mean": _mean_ci95([
            float(row["defer"]["phi_pressure_mean"]) for row in runs
        ]),
        "max_eligible_defer_ticks": max(
            (int(row["defer"]["eligible_ticks_max"]) for row in runs),
            default=0,
        ),
        "paired_delta_vs_ready_max_v1": {
            "defer_rate": _mean_ci95([
                float(row["defer"]["rate"])
                - float(row["ready_max_v1_defer"]["rate"])
                for row in runs
                if row["ready_max_v1_defer"]["available"]
            ]),
            "all_remaining_batch_rate": _mean_ci95([
                float(row["defer"]["all_remaining_batch_rate"])
                - float(
                    row["ready_max_v1_defer"]
                    ["all_remaining_batch_rate"]
                )
                for row in runs
                if row["ready_max_v1_defer"]["available"]
            ]),
        },
    }
    return {
        "schema_version": (
            "phase_c_station_context_defer_pipeline_phi_report_v2"
        ),
        "root": root.as_posix(),
        "development_only": True,
        "single_variable_change": (
            "ready_contention removed from station_risk max; retained "
            "inside pipeline mass and as a diagnostic"
        ),
        "summary": {
            "runs": len(runs),
            "expected_runs": len(LOADS) * len(SEEDS),
            "missing": missing,
            "complete": not missing,
            "mechanism_and_lifecycle_passed": mechanism_passed,
            "primary_outcome_guardrails_passed": primary_guardrails["passed"],
            "development_gate_passed": (
                mechanism_passed and primary_guardrails["passed"]
            ),
        },
        "controller_contract": {
            "schema": STATION_CONTEXT_DEFER_PIPELINE_PHI_SCHEMA_VERSION,
            "risk_mode": STATION_CONTEXT_DEFER_RISK_PIPELINE_PHI_V2,
            "ready_contention_in_station_risk": False,
        },
        "defer_mechanism": defer_summary,
        "contrasts": contrasts,
        "primary_outcome_guardrails": primary_guardrails,
        "runs": runs,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Station-context defer pipeline+phi V2 validation",
        "",
        "Single-variable paired development experiment: `ready_contention` "
        "remains in pipeline mass and diagnostics, but is removed from the "
        "independent station-risk maximum.",
        "",
        f"- runs: {summary['runs']}/{summary['expected_runs']}",
        f"- mechanism/lifecycle audits: "
        f"{summary['mechanism_and_lifecycle_passed']}",
        f"- primary outcome guardrails: "
        f"{summary['primary_outcome_guardrails_passed']}",
        f"- development gate: {summary['development_gate_passed']}",
        "",
        "| contrast | load | n | completed delta | deadlock delta | "
        "stall delta | better both | worse both |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for contrast in CONTRASTS:
        block = report["contrasts"][contrast]
        for load in (*LOADS, "overall"):
            row = block["overall"] if load == "overall" else block["by_load"][load]
            lines.append(
                f"| {contrast} | {load} | {row['runs']} | "
                f"{_fmt(row['metric_deltas']['completed_orders']['mean'], 2)} | "
                f"{_fmt(row['metric_deltas']['deadlock_ratio_mean']['mean'])} | "
                f"{_fmt(row['metric_deltas']['stall_ratio_mean']['mean'])} | "
                f"{row['completion_up_deadlock_down']} | "
                f"{row['completion_down_deadlock_up']} |"
            )
    mechanism = report["defer_mechanism"]
    lines.extend([
        "",
        "## V2 defer mechanism",
        "",
        f"- defer rate mean: {_fmt(mechanism['defer_rate']['mean'])}",
        f"- all-remaining-deferred batch rate mean: "
        f"{_fmt(mechanism['all_remaining_batch_rate']['mean'])}",
        f"- ready would have dominated the old max, diagnostic rate: "
        f"{_fmt(mechanism['ready_would_dominate_rate']['mean'])}",
        f"- paired defer-rate delta versus ready-max V1: "
        f"{_fmt(mechanism['paired_delta_vs_ready_max_v1']['defer_rate']['mean'])}",
        f"- paired all-defer-batch-rate delta versus ready-max V1: "
        f"{_fmt(mechanism['paired_delta_vs_ready_max_v1']['all_remaining_batch_rate']['mean'])}",
        f"- maximum eligible defer streak: "
        f"{mechanism['max_eligible_defer_ticks']}",
        "",
        "Positive completed delta and negative deadlock/stall delta favour V2. "
        "The frozen primary guardrails apply only to V2 versus Dynamic-J + "
        "S1 + committed admission V1.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = analyse(args.input_root, allow_partial=args.allow_partial)
    validation = args.input_root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    json_path = validation / "station_context_defer_pipeline_phi_validation.json"
    md_path = validation / "station_context_defer_pipeline_phi_validation.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    (
        validation / "station_context_defer_pipeline_phi_validation.sha256"
    ).write_text(f"{digest}  {json_path.name}\n", encoding="utf-8")
    print(json.dumps({
        "summary": report["summary"],
        "defer_mechanism": report["defer_mechanism"],
        "primary": report["contrasts"]["vs_dynamic_j_admission_v1"],
        "mechanism_contrast": report["contrasts"]["vs_ready_max_defer_v1"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
