"""Validate the frozen Phase-C Round-1 paired online certification matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from WorldModel.evaluation.evaluate_online_v6 import (
    ORDER_ARRIVAL_MANIFEST_SCHEMA_VERSION,
)
from WorldModel.evaluation.phase_c_round1_cert_protocol import (
    BOOTSTRAP_REPEATS,
    CERTIFICATION_SEEDS,
    CERT_REPORT_SCHEMA_VERSION,
    DECISION_TRACE_SCHEMA_VERSION,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    GUARDRAILS,
    LOAD_CONFIGS,
    ONLINE_REPORT_SCHEMA_VERSION,
    PER_SEED_REPORT_SCHEMA_VERSION,
    RANDOM_SEED,
    REQUIRED_LOADS,
    SECONDARY_METRICS,
    SECONDARY_REQUIRED,
    TICKS,
    TOP_M_METADATA_ONLY,
    TRAJECTORY_BIN_EDGES,
    formal_protocol,
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_round1_online_pair import (
    BASELINE_LABEL,
    CANDIDATE_LABEL,
    _pair_audit,
)


METRIC_DIRECTIONS = {
    "completion_fraction": "higher",
    "completed_tasks_relative": "higher",
    "avg_excess_delay_relative": "lower",
    "open_order_fraction": "lower",
    "pending_order_fraction": "lower",
    "wait_or_stall": "lower",
    "deadlock_ratio_max": "lower",
    "completed_flow_time_relative": "lower",
    "open_order_age_fraction": "lower",
    "pending_order_age_fraction": "lower",
    "handoff_ratio_mean": "lower",
    "handoff_ratio_max": "lower",
    "station_pressure": "lower",
    "bottleneck_CVaR": "lower",
    "unified_risk": "lower",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _canonical_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    canonical = {
        "schema_version": manifest.get("schema_version"),
        "orders": manifest.get("orders"),
    }
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mean_ci(values: Sequence[float], *, repeats: int, seed: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "ci95": None}
    if array.size == 1:
        value = float(array[0])
        return {"n": 1, "mean": value, "ci95": [value, value]}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(repeats), array.size))
    means = array[indices].mean(axis=1)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def _number(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric online metric: {key}")
    return float(value)


def _metric_values(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    metric: str,
) -> tuple[float, float, float]:
    baseline_orders = _number(baseline, "order_arrival_count")
    candidate_orders = _number(candidate, "order_arrival_count")
    if baseline_orders <= 0 or candidate_orders <= 0:
        raise ValueError("paired order-arrival count must be positive")

    if metric == "completion_fraction":
        left = _number(baseline, "completed_orders") / baseline_orders
        right = _number(candidate, "completed_orders") / candidate_orders
    elif metric == "open_order_fraction":
        left = _number(baseline, "open_order_count") / baseline_orders
        right = _number(candidate, "open_order_count") / candidate_orders
    elif metric == "pending_order_fraction":
        left = _number(baseline, "pending_order_count") / baseline_orders
        right = _number(candidate, "pending_order_count") / candidate_orders
    elif metric == "open_order_age_fraction":
        left = _number(baseline, "open_order_age_p95") / float(TICKS)
        right = _number(candidate, "open_order_age_p95") / float(TICKS)
    elif metric == "pending_order_age_fraction":
        left = _number(baseline, "pending_order_age_p95") / float(TICKS)
        right = _number(candidate, "pending_order_age_p95") / float(TICKS)
    elif metric.endswith("_relative"):
        source = {
            "completed_tasks_relative": "completed_tasks",
            "avg_excess_delay_relative": "avg_excess_delay",
            "completed_flow_time_relative": "completed_order_flow_time_p95",
        }[metric]
        raw_left = _number(baseline, source)
        raw_right = _number(candidate, source)
        left = 0.0
        right = (raw_right - raw_left) / max(abs(raw_left), 1.0)
    else:
        left = _number(baseline, metric)
        right = _number(candidate, metric)
    return left, right, right - left


def _metric_report(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    direction = METRIC_DIRECTIONS[metric]
    by_seed: dict[int, list[float]] = defaultdict(list)
    by_load: dict[str, list[float]] = defaultdict(list)
    raw_rows = []
    for row in rows:
        left, right, delta = _metric_values(
            row["baseline"], row["candidate"], metric
        )
        seed_id = int(row["seed"])
        load = str(row["load"])
        by_seed[seed_id].append(delta)
        by_load[load].append(delta)
        raw_rows.append({
            "seed": seed_id,
            "load": load,
            "baseline_comparison_value": left,
            "candidate_comparison_value": right,
            "candidate_minus_baseline": delta,
        })
    seed_means = [float(np.mean(by_seed[key])) for key in sorted(by_seed)]
    overall = _mean_ci(seed_means, repeats=repeats, seed=seed)
    by_load_report = {
        load: _mean_ci(
            values,
            repeats=repeats,
            seed=seed + 1009 * (index + 1),
        )
        for index, (load, values) in enumerate(sorted(by_load.items()))
    }
    mean = overall["mean"]
    favourable = bool(
        mean is not None
        and (mean >= 0.0 if direction == "higher" else mean <= 0.0)
    )
    return {
        "metric": metric,
        "direction": direction,
        "comparison": "candidate_minus_baseline",
        "bootstrap_unit": "simulation_seed_paired_across_loads",
        "overall": overall,
        "by_load": by_load_report,
        "by_seed": {
            str(key): float(np.mean(values))
            for key, values in sorted(by_seed.items())
        },
        "favourable_point_estimate": favourable,
        "rows": raw_rows,
    }


def _load_online_rows(
    paths: Sequence[Path],
    *,
    protocol: Mapping[str, Any],
    bundle_sha256: str,
    config_artifacts: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    audits: dict[str, Any] = {
        "schemas": True,
        "formal_flags": True,
        "roles": True,
        "protocol_hashes": True,
        "bundle_hashes": True,
        "exact_loads": True,
        "exact_seeds": True,
        "exact_ticks": True,
        "exact_top_m_metadata": True,
        "frozen_configs": True,
        "paired_arms": True,
        "checkpoint_hashes": True,
        "native_horizons": True,
        "pure_world_model_only": True,
        "all_idle_scope": True,
        "no_fallback": True,
        "native_no_assign_contract": True,
        "exact_paired_order_manifests": True,
        "pair_audits": True,
    }
    observed_loads = []
    for path in paths:
        report = _read_json(path)
        meta = report.get("meta") or {}
        if report.get("schema_version") != ONLINE_REPORT_SCHEMA_VERSION:
            audits["schemas"] = False
        if not bool(meta.get("formal")):
            audits["formal_flags"] = False
        if meta.get("role") != (
            "FORMAL_PHASE_C_ROUND1_PAIRED_CLOSED_LOOP_COLLECTION"
        ):
            audits["roles"] = False
        if meta.get("formal_protocol_sha256") != protocol.get("protocol_sha256"):
            audits["protocol_hashes"] = False
        if meta.get("frozen_bundle_sha256") != bundle_sha256:
            audits["bundle_hashes"] = False
        load = str(meta.get("load"))
        observed_loads.append(load)
        if load not in REQUIRED_LOADS:
            audits["exact_loads"] = False
            continue
        if int(meta.get("ticks", -1)) != TICKS:
            audits["exact_ticks"] = False
        if int(meta.get("top_m", -1)) != TOP_M_METADATA_ONLY:
            audits["exact_top_m_metadata"] = False
        frozen_config = config_artifacts.get(load) or {}
        if (
            meta.get("config") != Path(LOAD_CONFIGS[load]).as_posix()
            or meta.get("config_sha256") != frozen_config.get("sha256")
        ):
            audits["frozen_configs"] = False
        if set(meta.get("paired_arms") or ()) != {
            BASELINE_LABEL,
            CANDIDATE_LABEL,
        }:
            audits["paired_arms"] = False
        if (
            (meta.get("baseline_checkpoint") or {}).get("sha256")
            != protocol["prerequisites"]["baseline_checkpoint_sha256"]
            or (meta.get("candidate_checkpoint") or {}).get("sha256")
            != protocol["prerequisites"]["candidate_checkpoint_sha256"]
        ):
            audits["checkpoint_hashes"] = False
        if (
            int(meta.get("baseline_native_horizon", -1)) != 3
            or int(meta.get("candidate_native_horizon", -1)) != 10
        ):
            audits["native_horizons"] = False
        if any(
            bool(meta.get(key))
            for key in (
                "external_assignment_baseline",
                "td_target_or_head",
                "work_drift_or_residual_head",
                "lyapunov_online_scoring",
                "hard_no_assign_gate",
            )
        ):
            audits["pure_world_model_only"] = False

        seeds = tuple(sorted(int(value) for value in meta.get("seeds") or ()))
        if seeds != CERTIFICATION_SEEDS:
            audits["exact_seeds"] = False
        per_seed = report.get("per_seed") or {}
        if tuple(sorted(int(value) for value in per_seed)) != CERTIFICATION_SEEDS:
            audits["exact_seeds"] = False
        pair_audits = report.get("pair_audits") or {}
        artifacts_by_seed = report.get("per_seed_artifacts") or {}
        for seed_text, arms in per_seed.items():
            if set(arms) != {BASELINE_LABEL, CANDIDATE_LABEL}:
                audits["paired_arms"] = False
                continue
            baseline = arms[BASELINE_LABEL]
            candidate = arms[CANDIDATE_LABEL]
            for metrics in (baseline, candidate):
                if metrics.get("online_robot_candidate_scope") != "all_idle":
                    audits["all_idle_scope"] = False
                if (
                    int(metrics.get("fallback_greedy_calls", 0)) != 0
                    or float(metrics.get("fallback_greedy_ratio", 0.0)) != 0.0
                ):
                    audits["no_fallback"] = False
            contexts = int(candidate.get("native_no_assign_contexts", -1))
            if (
                bool(baseline.get("native_no_assign_enabled", False))
                or not bool(candidate.get("native_no_assign_enabled", False))
                or contexts <= 0
                or int(candidate.get("native_no_assign_scored", -2)) != contexts
                or int(candidate.get("native_no_assign_selected", -1))
                + int(candidate.get("native_no_assign_assignment_selected", -1))
                != contexts
            ):
                audits["native_no_assign_contract"] = False
            pair = _pair_audit(baseline, candidate)
            if (
                not pair["passed"]
                or not bool((pair_audits.get(seed_text) or {}).get("passed"))
            ):
                audits["pair_audits"] = False
            if (
                baseline.get("order_arrival_manifest_schema_version")
                != ORDER_ARRIVAL_MANIFEST_SCHEMA_VERSION
                or candidate.get("order_arrival_manifest_schema_version")
                != ORDER_ARRIVAL_MANIFEST_SCHEMA_VERSION
                or bool(baseline.get("order_arrival_replayed"))
                or not bool(candidate.get("order_arrival_replayed"))
            ):
                audits["exact_paired_order_manifests"] = False
            rows.append({
                "seed": int(seed_text),
                "load": load,
                "baseline": baseline,
                "candidate": candidate,
                "artifacts": artifacts_by_seed.get(seed_text) or {},
            })
    if tuple(sorted(observed_loads)) != tuple(sorted(REQUIRED_LOADS)):
        audits["exact_loads"] = False
    if len(rows) != len(REQUIRED_LOADS) * len(CERTIFICATION_SEEDS):
        audits["exact_seeds"] = False
    audits["passed"] = all(
        bool(value) for key, value in audits.items() if key != "passed"
    )
    return rows, audits


def _bin_index(tick: int) -> int | None:
    for index, (left, right) in enumerate(
        zip(TRAJECTORY_BIN_EDGES[:-1], TRAJECTORY_BIN_EDGES[1:])
    ):
        if int(left) <= tick < int(right):
            return index
    return None


def _audit_artifacts(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest_dir: Path,
    trace_dir: Path,
    per_seed_dir: Path,
    protocol_sha256: str,
    bundle_sha256: str,
) -> dict[str, Any]:
    expected_manifests = {
        f"phasec_r1_{load}_seed{seed}_orders.json"
        for load in REQUIRED_LOADS
        for seed in CERTIFICATION_SEEDS
    }
    expected_traces = {
        f"phasec_r1_{load}_seed{seed}_candidate_decisions.jsonl"
        for load in REQUIRED_LOADS
        for seed in CERTIFICATION_SEEDS
    }
    expected_seed_reports = {
        f"phasec_r1_{load}_seed{seed}_paired.json"
        for load in REQUIRED_LOADS
        for seed in CERTIFICATION_SEEDS
    }
    actual_manifests = {
        path.name for path in manifest_dir.glob("phasec_r1_*_seed*_orders.json")
    }
    actual_traces = {
        path.name
        for path in trace_dir.glob("phasec_r1_*_seed*_candidate_decisions.jsonl")
    }
    actual_seed_reports = {
        path.name for path in per_seed_dir.glob("phasec_r1_*_seed*_paired.json")
    }
    checks: dict[str, bool] = {
        "exact_manifest_files": actual_manifests == expected_manifests,
        "exact_trace_files": actual_traces == expected_traces,
        "exact_per_seed_files": actual_seed_reports == expected_seed_reports,
        "no_partial_artifacts": not any(
            path.is_file()
            for directory in (manifest_dir, trace_dir, per_seed_dir)
            for path in directory.glob("*.partial")
        ),
        "manifest_contents": True,
        "per_seed_reports": True,
        "trace_schema_and_counts": True,
    }
    errors = []
    trajectory: dict[str, Any] = {}
    terminal_assignments_passed = True

    for row in rows:
        load = str(row["load"])
        seed = int(row["seed"])
        key = f"{load}|seed={seed}"
        manifest_path = manifest_dir / f"phasec_r1_{load}_seed{seed}_orders.json"
        trace_path = (
            trace_dir
            / f"phasec_r1_{load}_seed{seed}_candidate_decisions.jsonl"
        )
        seed_path = per_seed_dir / f"phasec_r1_{load}_seed{seed}_paired.json"
        artifacts = row.get("artifacts") or {}
        candidate = row["candidate"]

        try:
            manifest = _read_json(manifest_path)
            manifest_hash = _canonical_manifest_sha256(manifest)
            manifest_artifact = artifacts.get("order_manifest") or {}
            if not (
                manifest.get("schema_version")
                == ORDER_ARRIVAL_MANIFEST_SCHEMA_VERSION
                and int(manifest.get("total_orders", -1))
                == len(manifest.get("orders") or ())
                and manifest.get("manifest_sha256") == manifest_hash
                and manifest_hash
                == row["baseline"].get("order_arrival_manifest_sha256")
                == candidate.get("order_arrival_manifest_sha256")
                and int(manifest.get("total_orders", -1))
                == int(row["baseline"].get("order_arrival_count", -2))
                == int(candidate.get("order_arrival_count", -3))
                and manifest_artifact.get("sha256") == sha256_file(manifest_path)
            ):
                checks["manifest_contents"] = False
                errors.append(f"{key}: manifest audit failed")
        except Exception as exc:  # fail closed while preserving a report
            checks["manifest_contents"] = False
            errors.append(f"{key}: manifest error: {exc}")

        try:
            seed_report = _read_json(seed_path)
            seed_meta = seed_report.get("meta") or {}
            seed_artifact = artifacts.get("per_seed_report") or {}
            seed_payload_artifacts = seed_report.get("artifacts") or {}
            expected_payload_artifacts = {
                name: value
                for name, value in artifacts.items()
                if name != "per_seed_report"
            }
            if not (
                seed_report.get("schema_version")
                == PER_SEED_REPORT_SCHEMA_VERSION
                and bool(seed_meta.get("formal"))
                and seed_meta.get("load") == load
                and int(seed_meta.get("seed", -1)) == seed
                and int(seed_meta.get("ticks", -1)) == TICKS
                and seed_meta.get("formal_protocol_sha256")
                == protocol_sha256
                and seed_meta.get("frozen_bundle_sha256") == bundle_sha256
                and seed_report.get("arms")
                == {
                    BASELINE_LABEL: row["baseline"],
                    CANDIDATE_LABEL: candidate,
                }
                and seed_payload_artifacts == expected_payload_artifacts
                and seed_artifact.get("sha256") == sha256_file(seed_path)
            ):
                checks["per_seed_reports"] = False
                errors.append(f"{key}: per-seed report audit failed")
        except Exception as exc:
            checks["per_seed_reports"] = False
            errors.append(f"{key}: per-seed report error: {exc}")

        bins = [
            {"tick_range": [int(left), int(right)], "contexts": 0,
             "no_assign": 0, "assign_robot": 0}
            for left, right in zip(
                TRAJECTORY_BIN_EDGES[:-1], TRAJECTORY_BIN_EDGES[1:]
            )
        ]
        trace_valid = True
        no_assign_total = 0
        assign_total = 0
        records = 0
        try:
            with trace_path.open("r", encoding="utf-8") as handle:
                for expected_index, line in enumerate(handle):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    action = record.get("selected_action_type")
                    tick = int(record.get("tick", -1))
                    index = _bin_index(tick)
                    if not (
                        record.get("schema_version")
                        == DECISION_TRACE_SCHEMA_VERSION
                        and record.get("load") == load
                        and int(record.get("seed", -1)) == seed
                        and int(record.get("record_index", -1)) == expected_index
                        and action in {"no_assign", "assign_robot"}
                        and index is not None
                    ):
                        trace_valid = False
                        continue
                    records += 1
                    bins[index]["contexts"] += 1
                    if action == "no_assign":
                        no_assign_total += 1
                        bins[index]["no_assign"] += 1
                    else:
                        assign_total += 1
                        bins[index]["assign_robot"] += 1
            trace_artifact = artifacts.get("candidate_decision_trace") or {}
            summary = trace_artifact.get("summary") or {}
            trace_valid = trace_valid and bool(summary.get("passed"))
            trace_valid = trace_valid and (
                trace_artifact.get("sha256") == sha256_file(trace_path)
                and records == int(candidate.get("decision_trace_contexts", -1))
                == int(candidate.get("decision_contexts_total", -2))
                == int(candidate.get("native_no_assign_contexts", -3))
                == int(candidate.get("native_no_assign_scored", -4))
                and no_assign_total
                == int(candidate.get("native_no_assign_selected", -1))
                and assign_total
                == int(candidate.get("native_no_assign_assignment_selected", -1))
                and int(candidate.get("decision_trace_dropped", 0)) == 0
            )
        except Exception as exc:
            trace_valid = False
            errors.append(f"{key}: trace error: {exc}")
        if not trace_valid:
            checks["trace_schema_and_counts"] = False
            errors.append(f"{key}: trace schema/count audit failed")
        for bin_row in bins:
            contexts = int(bin_row["contexts"])
            bin_row["no_assign_rate"] = (
                float(bin_row["no_assign"]) / contexts if contexts else None
            )
        terminal = bins[-1]
        terminal_passed = bool(
            int(terminal["contexts"]) == 0
            or int(terminal["assign_robot"]) > 0
        )
        terminal_assignments_passed = (
            terminal_assignments_passed and terminal_passed
        )
        trajectory[key] = {
            "records": records,
            "no_assign_selected": no_assign_total,
            "assign_robot_selected": assign_total,
            "bins": bins,
            "terminal_bin_robot_assignment_when_contexts_exist": terminal_passed,
            "trace_integrity_passed": trace_valid,
        }

    integrity_passed = all(checks.values())
    return {
        "checks": checks,
        "integrity_passed": integrity_passed,
        "terminal_assignments_passed": terminal_assignments_passed,
        "passed": integrity_passed and terminal_assignments_passed,
        "trajectory": trajectory,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-bundle", required=True)
    parser.add_argument("--online-reports", nargs="+", required=True)
    parser.add_argument("--order-manifest-dir", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--per-seed-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(
            f"refusing to overwrite formal Phase-C certification report: {output}"
        )
    bundle_path = Path(args.frozen_bundle)
    bundle = _read_json(bundle_path)
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise SystemExit("unexpected frozen Phase-C online bundle schema")
    protocol = bundle.get("formal_protocol") or {}
    if protocol != formal_protocol():
        raise SystemExit("frozen Phase-C protocol differs from source protocol")
    bundle_sha256 = sha256_file(bundle_path)
    config_artifacts = (bundle.get("artifacts") or {}).get("load_configs") or {}

    online_paths = [Path(value) for value in args.online_reports]
    rows, implementation = _load_online_rows(
        online_paths,
        protocol=protocol,
        bundle_sha256=bundle_sha256,
        config_artifacts=config_artifacts,
    )
    artifact_audit = _audit_artifacts(
        rows,
        manifest_dir=Path(args.order_manifest_dir),
        trace_dir=Path(args.trace_dir),
        per_seed_dir=Path(args.per_seed_dir),
        protocol_sha256=str(protocol["protocol_sha256"]),
        bundle_sha256=bundle_sha256,
    )

    metrics = {}
    for index, metric in enumerate(METRIC_DIRECTIONS):
        metrics[metric] = _metric_report(
            rows,
            metric,
            repeats=BOOTSTRAP_REPEATS,
            seed=RANDOM_SEED + 10007 * (index + 1),
        )

    primary = metrics["completion_fraction"]
    primary_ci = primary["overall"]["ci95"] or [float("-inf"), float("inf")]
    primary_passed = bool(
        primary["overall"]["mean"] is not None
        and float(primary_ci[0]) > 0.0
        and all(
            (primary["by_load"].get(load) or {}).get("mean") is not None
            and float(primary["by_load"][load]["mean"]) > 0.0
            for load in REQUIRED_LOADS
        )
    )

    secondary_count = sum(
        bool(metrics[name]["favourable_point_estimate"])
        for name in SECONDARY_METRICS
    )
    secondary_passed = secondary_count >= SECONDARY_REQUIRED

    def upper(name: str) -> float:
        ci = metrics[name]["overall"]["ci95"]
        return float(ci[1]) if ci else float("inf")

    guardrail_checks = {
        "avg_excess_delay_relative_ci95_upper": (
            upper("avg_excess_delay_relative")
            <= GUARDRAILS["avg_excess_delay_relative_ci95_upper"]
        ),
        "completed_flow_time_relative_ci95_upper": (
            upper("completed_flow_time_relative")
            <= GUARDRAILS["completed_flow_time_relative_ci95_upper"]
        ),
        "open_order_fraction_ci95_upper": (
            upper("open_order_fraction")
            <= GUARDRAILS["open_order_fraction_ci95_upper"]
        ),
        "pending_order_fraction_ci95_upper": (
            upper("pending_order_fraction")
            <= GUARDRAILS["pending_order_fraction_ci95_upper"]
        ),
        "open_order_age_fraction_ci95_upper": (
            upper("open_order_age_fraction")
            <= GUARDRAILS["open_order_age_fraction_ci95_upper"]
        ),
        "pending_order_age_fraction_ci95_upper": (
            upper("pending_order_age_fraction")
            <= GUARDRAILS["pending_order_age_fraction_ci95_upper"]
        ),
        "deadlock_ratio_max_absolute_ci95_upper": (
            upper("deadlock_ratio_max")
            <= GUARDRAILS["deadlock_ratio_max_absolute_ci95_upper"]
        ),
        "handoff_ratio_mean_absolute_ci95_upper": (
            upper("handoff_ratio_mean")
            <= GUARDRAILS["handoff_ratio_mean_absolute_ci95_upper"]
        ),
        "handoff_ratio_max_absolute_ci95_upper": (
            upper("handoff_ratio_max")
            <= GUARDRAILS["handoff_ratio_max_absolute_ci95_upper"]
        ),
    }
    guardrails_passed = all(guardrail_checks.values())

    load_checks = {}
    for load in REQUIRED_LOADS:
        load_checks[load] = {
            "completion_fraction_positive": (
                float(metrics["completion_fraction"]["by_load"][load]["mean"])
                > 0.0
            ),
            "avg_excess_delay_relative_at_most_0.05": (
                float(metrics["avg_excess_delay_relative"]["by_load"][load]["mean"])
                <= 0.05
            ),
            "open_order_fraction_at_most_0.02": (
                float(metrics["open_order_fraction"]["by_load"][load]["mean"])
                <= 0.02
            ),
            "deadlock_ratio_max_delta_at_most_0.02": (
                float(metrics["deadlock_ratio_max"]["by_load"][load]["mean"])
                <= 0.02
            ),
        }
        load_checks[load]["passed"] = all(load_checks[load].values())
    load_robustness_passed = all(
        bool(load_checks[load]["passed"]) for load in REQUIRED_LOADS
    )

    high_pressure = float(
        metrics["station_pressure"]["by_load"]["high"]["mean"]
    )
    high_pressure_increased = high_pressure > 0.0
    high_conditions = {
        "completion_fraction_improves": float(
            metrics["completion_fraction"]["by_load"]["high"]["mean"]
        )
        > 0.0,
        "avg_excess_delay_does_not_worsen": float(
            metrics["avg_excess_delay_relative"]["by_load"]["high"]["mean"]
        )
        <= 0.0,
        "bottleneck_cvar_does_not_worsen": float(
            metrics["bottleneck_CVaR"]["by_load"]["high"]["mean"]
        )
        <= 0.0,
        "deadlock_max_does_not_worsen": float(
            metrics["deadlock_ratio_max"]["by_load"]["high"]["mean"]
        )
        <= 0.0,
    }
    station_pressure_passed = bool(
        not high_pressure_increased or all(high_conditions.values())
    )

    implementation_passed = bool(
        implementation["passed"] and artifact_audit["integrity_passed"]
    )
    trajectory_passed = bool(artifact_audit["terminal_assignments_passed"])
    if not implementation_passed:
        verdict = "INVALID_PROTOCOL_OR_DATA"
    elif (
        not guardrails_passed
        or not load_robustness_passed
        or not station_pressure_passed
        or not trajectory_passed
    ):
        verdict = "CLOSED_LOOP_POLICY_FAILURE"
    elif primary_passed and secondary_passed:
        verdict = "PASS_PHASE_C_ROUND1_CLOSED_LOOP_CERTIFICATION"
    else:
        verdict = "INCREMENTAL_CLOSED_LOOP_VALUE_INCONCLUSIVE"
    passed = verdict == "PASS_PHASE_C_ROUND1_CLOSED_LOOP_CERTIFICATION"

    if verdict == "INVALID_PROTOCOL_OR_DATA":
        recommendation = "repair protocol/data integrity before interpreting outcomes"
    elif verdict == "CLOSED_LOOP_POLICY_FAILURE":
        recommendation = (
            "do not deploy this checkpoint under the certified policy contract; "
            "diagnose the frozen closed-loop failure without tuning on seeds 471--480"
        )
    elif verdict == "INCREMENTAL_CLOSED_LOOP_VALUE_INCONCLUSIVE":
        recommendation = (
            "retain the Phase-C offline evidence but do not claim closed-loop "
            "superiority from this one-shot certificate"
        )
    else:
        recommendation = (
            "the pure Phase-C World Model is supported for the frozen tested "
            "configuration family; continue separate distribution-shift monitoring"
        )

    report = {
        "schema_version": CERT_REPORT_SCHEMA_VERSION,
        "name": "phase_c_round1_pure_world_model_closed_loop_certification",
        "formal_protocol": protocol,
        "semantics": {
            "paired_normal_arrival_test": True,
            "baseline": "Stage-1 H=3 all-idle robot actions",
            "candidate": "Phase-C H=10 all-idle robots plus native NO_ASSIGN",
            "external_assignment_algorithm": False,
            "td_target_or_head": False,
            "work_drift_or_residual_head": False,
            "lyapunov_online_score": False,
            "post_test_tuning": False,
        },
        "implementation_audit": implementation,
        "artifact_and_trace_audit": artifact_audit,
        "paired_metrics": metrics,
        "primary_contract": {
            "metric": "completion_fraction",
            "overall_ci95_lower_above_zero": float(primary_ci[0]) > 0.0,
            "each_load_point_estimate_above_zero": all(
                float(primary["by_load"][load]["mean"]) > 0.0
                for load in REQUIRED_LOADS
            ),
            "passed": primary_passed,
        },
        "secondary_contract": {
            "metrics": list(SECONDARY_METRICS),
            "favourable": secondary_count,
            "required": SECONDARY_REQUIRED,
            "passed": secondary_passed,
        },
        "guardrails": {
            "checks": guardrail_checks,
            "passed": guardrails_passed,
        },
        "load_robustness": {
            "by_load": load_checks,
            "passed": load_robustness_passed,
        },
        "station_pressure_contract": {
            "high_load_candidate_minus_baseline": high_pressure,
            "increased": high_pressure_increased,
            "conditional_checks": high_conditions,
            "passed": station_pressure_passed,
        },
        "checks": {
            "implementation_and_data_integrity": implementation_passed,
            "primary": primary_passed,
            "secondary": secondary_passed,
            "guardrails": guardrails_passed,
            "load_robustness": load_robustness_passed,
            "no_assign_terminal_trajectory": trajectory_passed,
            "conditional_station_pressure": station_pressure_passed,
        },
        "verdict": verdict,
        "passed": passed,
        "recommendation": recommendation,
        "scope_limit": protocol["scope_limit"],
        "provenance": {
            "frozen_bundle": bundle_path.as_posix(),
            "frozen_bundle_sha256": bundle_sha256,
            "online_reports": [
                {"path": path.as_posix(), "sha256": sha256_file(path)}
                for path in online_paths
            ],
            "post_test_checkpoint_or_no_assign_tuning": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {output}")
    print("Phase C verdict =", verdict)
    print("Phase C passed =", passed)
    print("primary improvement =", primary["overall"]["mean"])
    print("primary CI95 =", primary["overall"]["ci95"])
    print("secondary support =", secondary_count, "/", len(SECONDARY_METRICS))
    print("guardrails passed =", guardrails_passed)
    print("load robustness passed =", load_robustness_passed)
    print("NO_ASSIGN trajectory passed =", trajectory_passed)


if __name__ == "__main__":
    main()
