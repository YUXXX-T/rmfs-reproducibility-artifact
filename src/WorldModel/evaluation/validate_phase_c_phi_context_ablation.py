"""Validate the paired 541--550 S1 versus S1+phi-context ablation."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from WorldModel.evaluation.phase_c_phi_context_ablation_protocol import (
    ARM_KEYS,
    ARM_LABELS,
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    COMPLETION_RELATIVE_NONINFERIORITY,
    DEADLOCK_MEAN_NONINFERIORITY_MARGIN,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOADS,
    OUTPUT_ROOT,
    PER_ARM_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    S1_ARM,
    S1_PHI_ARM,
    SCHEMA_VERSION,
    SEEDS,
    TICKS,
    canonical_sha256,
    sha256_file,
)


METRICS = (
    "completed_orders",
    "completed_tasks",
    "avg_task_duration",
    "avg_excess_delay",
    "open_order_count",
    "pending_order_count",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "stall_ratio_mean",
    "stall_ratio_max",
    "assignment_time_ms_mean",
    "wall_time_s",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite changed output: {path}")
        return
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = _read_json(path)
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError("wrong phi-context frozen bundle schema")
    protocol = dict(bundle.get("protocol") or {})
    claimed = str(protocol.pop("protocol_sha256", ""))
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong phi-context protocol schema")
    if canonical_sha256(protocol) != claimed:
        raise ValueError("phi-context protocol hash mismatch")
    if bundle.get("protocol_sha256") != claimed:
        raise ValueError("bundle/protocol hash mismatch")
    protocol["protocol_sha256"] = claimed
    return bundle, protocol


def _percentile(values: Iterable[float], q: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        raise ValueError("cannot compute a percentile on zero values")
    if len(rows) == 1:
        return rows[0]
    position = (len(rows) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return rows[lower]
    fraction = position - lower
    return rows[lower] * (1.0 - fraction) + rows[upper] * fraction


def _cluster_bootstrap(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, float]:
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(float(row[key]))
    seeds = sorted(by_seed)
    if seeds != list(SEEDS):
        raise ValueError(f"bootstrap seed coverage changed: {seeds}")
    rng = random.Random(BOOTSTRAP_SEED + sum(ord(ch) for ch in key))
    draws = []
    for _ in range(BOOTSTRAP_REPEATS):
        sampled = [rng.choice(seeds) for _ in seeds]
        values = [value for seed in sampled for value in by_seed[seed]]
        draws.append(sum(values) / len(values))
    observed = sum(float(row[key]) for row in rows) / len(rows)
    return {
        "mean": observed,
        "ci95_lower": _percentile(draws, 0.025),
        "ci95_upper": _percentile(draws, 0.975),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_unit": "seed_clustered_across_loads",
    }


def _load_runs(root: Path, protocol_sha: str, bundle_sha: str) -> dict:
    runs = {}
    for arm in ARM_KEYS:
        for load in LOADS:
            for seed in SEEDS:
                path = root / "per_arm" / arm / f"{load}_seed{seed}.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = _read_json(path)
                meta = payload.get("meta") or {}
                checks = {
                    "schema": payload.get("schema_version")
                    == PER_ARM_SCHEMA_VERSION,
                    "protocol": meta.get("protocol_sha256") == protocol_sha,
                    "bundle": meta.get("frozen_bundle_sha256") == bundle_sha,
                    "arm": meta.get("arm_key") == arm,
                    "load": meta.get("load") == load,
                    "seed": int(meta.get("seed", -1)) == seed,
                    "ticks": int(meta.get("ticks", -1)) == TICKS,
                    "audit": bool((payload.get("audit") or {}).get("passed")),
                }
                if not all(checks.values()):
                    failed = [key for key, passed in checks.items() if not passed]
                    raise ValueError(f"invalid run {path}: {failed}")
                runs[(arm, load, seed)] = payload
    return runs


def validate(root: Path, frozen_bundle: Path) -> dict[str, Any]:
    _, protocol = _load_protocol(frozen_bundle)
    protocol_sha = str(protocol["protocol_sha256"])
    bundle_sha = sha256_file(frozen_bundle)
    runs = _load_runs(root, protocol_sha, bundle_sha)

    rows = []
    for load in LOADS:
        for seed in SEEDS:
            baseline = runs[(S1_ARM, load, seed)]
            candidate = runs[(S1_PHI_ARM, load, seed)]
            if baseline["manifest"]["content_sha256"] != candidate["manifest"][
                "content_sha256"
            ]:
                raise ValueError(f"paired manifest mismatch: {load} seed={seed}")
            base_metrics = baseline["metrics"]
            phi_metrics = candidate["metrics"]
            row: dict[str, Any] = {"load": load, "seed": seed}
            for metric in METRICS:
                row[f"s1_{metric}"] = float(base_metrics[metric])
                row[f"s1_phi_{metric}"] = float(phi_metrics[metric])
                row[f"delta_{metric}"] = (
                    float(phi_metrics[metric]) - float(base_metrics[metric])
                )
            row["completion_relative_delta"] = (
                row["delta_completed_orders"]
                / max(row["s1_completed_orders"], 1.0)
            )
            row["phi_context_reordered_calls"] = int(
                phi_metrics["phi_context_reordered_calls"]
            )
            row["phi_context_replacement_count"] = int(
                phi_metrics["phi_context_replacement_count"]
            )
            row["phi_context_baseline_budget_pressure_mean"] = float(
                phi_metrics["phi_context_baseline_budget_pressure_mean"]
            )
            row["phi_context_applied_budget_pressure_mean"] = float(
                phi_metrics["phi_context_applied_budget_pressure_mean"]
            )
            row["phi_context_eval_time_ms_mean"] = float(
                phi_metrics["phi_context_eval_time_ms_mean"]
            )
            rows.append(row)

    aggregate: dict[str, Any] = {
        "completion_relative_delta": _cluster_bootstrap(
            rows, "completion_relative_delta"
        ),
    }
    for metric in METRICS:
        aggregate[f"delta_{metric}"] = _cluster_bootstrap(
            rows, f"delta_{metric}"
        )

    by_load = {}
    for load in LOADS:
        selected = [row for row in rows if row["load"] == load]
        by_load[load] = {
            "pairs": len(selected),
            "completion_relative_delta_mean": sum(
                row["completion_relative_delta"] for row in selected
            ) / len(selected),
            **{
                f"delta_{metric}_mean": sum(
                    row[f"delta_{metric}"] for row in selected
                ) / len(selected)
                for metric in METRICS
            },
        }

    phi_runs = [
        runs[(S1_PHI_ARM, load, seed)]["metrics"]
        for load in LOADS for seed in SEEDS
    ]
    mechanism = {
        "phi_runs": len(phi_runs),
        "head_loaded_runs": sum(
            int(bool(metrics.get("phi_context_head_loaded")))
            for metrics in phi_runs
        ),
        "context_reordered_total": sum(
            int(metrics.get("phi_context_reordered_calls", 0))
            for metrics in phi_runs
        ),
        "context_replacement_total": sum(
            int(metrics.get("phi_context_replacement_count", 0))
            for metrics in phi_runs
        ),
        "mean_extra_contexts_proposed": sum(
            float(metrics.get("phi_context_superset_extra_mean", 0.0))
            for metrics in phi_runs
        ) / len(phi_runs),
        "mean_phi_eval_time_ms": sum(
            float(metrics.get("phi_context_eval_time_ms_mean", 0.0))
            for metrics in phi_runs
        ) / len(phi_runs),
        "baseline_budget_pressure_mean": sum(
            float(metrics.get(
                "phi_context_baseline_budget_pressure_mean", 0.0
            )) for metrics in phi_runs
        ) / len(phi_runs),
        "applied_budget_pressure_mean": sum(
            float(metrics.get(
                "phi_context_applied_budget_pressure_mean", 0.0
            )) for metrics in phi_runs
        ) / len(phi_runs),
    }
    mechanism_checks = {
        "head_loaded_every_run": mechanism["head_loaded_runs"] == len(phi_runs),
        "context_reordered": mechanism["context_reordered_total"] > 0,
        "context_replaced": mechanism["context_replacement_total"] > 0,
        "pressure_reduced_or_equal": (
            mechanism["applied_budget_pressure_mean"]
            <= mechanism["baseline_budget_pressure_mean"] + 1e-8
        ),
    }
    mechanism["checks"] = mechanism_checks
    mechanism["passed"] = all(mechanism_checks.values())

    completion_ci = aggregate["completion_relative_delta"]
    deadlock_ci = aggregate["delta_deadlock_ratio_mean"]
    outcome_checks = {
        "completion_relative_ci95_lower_ge_minus_0_05": (
            completion_ci["ci95_lower"]
            >= COMPLETION_RELATIVE_NONINFERIORITY
        ),
        "deadlock_mean_ci95_upper_le_0_02": (
            deadlock_ci["ci95_upper"]
            <= DEADLOCK_MEAN_NONINFERIORITY_MARGIN
        ),
        "deadlock_each_load_point_le_0_02": all(
            by_load[load]["delta_deadlock_ratio_mean_mean"]
            <= DEADLOCK_MEAN_NONINFERIORITY_MARGIN
            for load in LOADS
        ),
    }
    outcome_guardrails = {
        "checks": outcome_checks,
        "passed": all(outcome_checks.values()),
    }

    paired_outcomes = {
        "completion_up_deadlock_down": sum(
            row["delta_completed_orders"] > 0.0
            and row["delta_deadlock_ratio_mean"] < 0.0
            for row in rows
        ),
        "completion_down_deadlock_up": sum(
            row["delta_completed_orders"] < 0.0
            and row["delta_deadlock_ratio_mean"] > 0.0
            for row in rows
        ),
        "other": 0,
    }
    paired_outcomes["other"] = len(rows) - sum(
        paired_outcomes[key]
        for key in ("completion_up_deadlock_down", "completion_down_deadlock_up")
    )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_sha256": protocol_sha,
        "frozen_bundle": frozen_bundle.as_posix(),
        "frozen_bundle_sha256": bundle_sha,
        "pair_count": len(rows),
        "arms": {
            "baseline": ARM_LABELS[S1_ARM],
            "candidate": ARM_LABELS[S1_PHI_ARM],
        },
        "mechanism": mechanism,
        "outcome_guardrails": outcome_guardrails,
        "overall_passed": bool(
            mechanism["passed"] and outcome_guardrails["passed"]
        ),
        "aggregate": aggregate,
        "by_load": by_load,
        "paired_outcomes": paired_outcomes,
        "pairs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--frozen-bundle",
        type=Path,
        default=OUTPUT_ROOT / "phase_c_phi_context_frozen_protocol.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "validation",
    )
    args = parser.parse_args()

    report = validate(args.input_root, args.frozen_bundle)
    report_path = args.output_dir / "phi_context_ablation_validation.json"
    _atomic_json(report_path, report)
    manifest_path = args.output_dir / "validation_outputs.sha256"
    _atomic_json(
        args.output_dir / "validation_summary.json",
        {
            "schema_version": "phase_c_phi_context_validation_summary_v1",
            "overall_passed": report["overall_passed"],
            "mechanism_passed": report["mechanism"]["passed"],
            "outcome_guardrails_passed": report["outcome_guardrails"]["passed"],
            "pair_count": report["pair_count"],
            "report": report_path.name,
            "report_sha256": sha256_file(report_path),
        },
    )
    summary_path = args.output_dir / "validation_summary.json"
    manifest_text = (
        f"{sha256_file(report_path)}  {report_path.name}\n"
        f"{sha256_file(summary_path)}  {summary_path.name}\n"
    )
    if manifest_path.is_file() and manifest_path.read_text(
        encoding="utf-8"
    ) != manifest_text:
        raise FileExistsError(f"refusing to overwrite changed {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")

    print("=" * 72)
    print("Phase-C S1 + phi_state context ablation")
    print("=" * 72)
    print(f"pairs: {report['pair_count']}")
    print(f"mechanism passed: {report['mechanism']['passed']}")
    print(
        "outcome guardrails passed: "
        f"{report['outcome_guardrails']['passed']}"
    )
    print(f"overall passed: {report['overall_passed']}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
