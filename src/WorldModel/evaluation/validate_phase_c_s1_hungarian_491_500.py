"""Validate and aggregate the frozen Phase-C/S1/Hungarian comparison."""

from __future__ import annotations

import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

from WorldModel.evaluation.phase_c_s1_hungarian_protocol import (
    AGGREGATE_SCHEMA_VERSION,
    ARMS,
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    DEADLOCK_NONINFERIORITY_MARGIN,
    GREEDY_LABEL,
    HUNGARIAN_LABEL,
    LOADS,
    METRIC_DIRECTIONS,
    OUTPUT_ROOT,
    PER_SEED_SCHEMA_VERSION,
    PHASEC_LABEL,
    PHASEC_S1_LABEL,
    SEEDS,
    canonical_sha256,
    sha256_file,
)


ARM_KEYS = {
    "greedy": GREEDY_LABEL,
    "hungarian": HUNGARIAN_LABEL,
    "phasec": PHASEC_LABEL,
    "phasec_s1": PHASEC_S1_LABEL,
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _bootstrap_ci(values: list[float], seed_offset: int) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    means = []
    n = len(values)
    for _ in range(BOOTSTRAP_REPEATS):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _mean_std(values: Iterable[float]) -> dict[str, float | int]:
    rows = list(values)
    return {
        "n": len(rows),
        "mean": statistics.mean(rows) if rows else float("nan"),
        "std": statistics.pstdev(rows) if len(rows) > 1 else 0.0,
    }


def _metric(metrics: Mapping[str, Any], name: str) -> float | None:
    if name == "completion_fraction":
        completed = _number(metrics.get("completed_orders"))
        arrivals = _number(metrics.get("order_arrival_count"))
        if completed is None or arrivals is None or arrivals <= 0:
            return None
        return completed / arrivals
    if name == "congestion_events_per_100":
        value = _number(metrics.get("congestion_events"))
        ticks = _number(metrics.get("ticks"))
        if value is None or ticks is None or ticks <= 0:
            return None
        return 100.0 * value / ticks
    if name == "severe_events_per_100":
        value = _number(metrics.get("severe_events"))
        ticks = _number(metrics.get("ticks"))
        if value is None or ticks is None or ticks <= 0:
            return None
        return 100.0 * value / ticks
    return _number(metrics.get(name))


REPORT_METRICS = tuple(METRIC_DIRECTIONS) + (
    "completion_fraction",
    "congestion_events_per_100",
    "severe_events_per_100",
)


def _comparison(
    rows: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
    *,
    baseline: str,
    candidate: str,
    seed_offset: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "baseline": baseline,
        "candidate": candidate,
        "definition": "candidate_minus_baseline",
        "metrics": {},
    }
    for metric_index, metric in enumerate(REPORT_METRICS):
        by_load = {}
        per_seed_cluster = []
        metric_rows = []
        for seed in SEEDS:
            seed_deltas = []
            for load in LOADS:
                arm_rows = rows[(load, seed)]
                left = _metric(arm_rows[baseline], metric)
                right = _metric(arm_rows[candidate], metric)
                if left is None or right is None:
                    continue
                delta = right - left
                seed_deltas.append(delta)
                metric_rows.append({
                    "load": load,
                    "seed": seed,
                    "baseline": left,
                    "candidate": right,
                    "candidate_minus_baseline": delta,
                })
            if len(seed_deltas) == len(LOADS):
                per_seed_cluster.append(sum(seed_deltas) / len(seed_deltas))
        for load_index, load in enumerate(LOADS):
            values = [
                row["candidate_minus_baseline"]
                for row in metric_rows if row["load"] == load
            ]
            summary = _mean_std(values)
            summary["ci95"] = _bootstrap_ci(
                values,
                seed_offset + metric_index * 100 + load_index,
            )
            by_load[load] = summary
        overall = _mean_std(per_seed_cluster)
        overall["ci95"] = _bootstrap_ci(
            per_seed_cluster,
            seed_offset + metric_index * 100 + 90,
        )
        direction = METRIC_DIRECTIONS.get(metric)
        if direction is None:
            direction = "higher" if metric == "completion_fraction" else "lower"
        report["metrics"][metric] = {
            "direction": direction,
            "overall_seed_cluster": overall,
            "by_load": by_load,
            "rows": metric_rows,
        }
    return report


def _write_exact(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"existing validation output differs: {path}")
        print(f"[audit] unchanged {path}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"[write] {path}")


def main() -> None:
    root = Path(os.environ.get("PHASEC_S1_CERT", OUTPUT_ROOT))
    bundle_path = root / "phase_c_s1_hungarian_frozen_protocol.json"
    bundle = _read_json(bundle_path)
    bundle_sha256 = sha256_file(bundle_path)
    protocol = bundle.get("protocol") or {}
    protocol_sha256 = str(bundle.get("protocol_sha256", ""))
    if canonical_sha256(protocol) != protocol_sha256:
        raise ValueError("frozen protocol hash mismatch")

    rows: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    files: list[Path] = [bundle_path, root / "frozen_inputs.sha256"]
    audit_checks: dict[str, bool] = {}
    s1_contexts = 0
    s1_active_contexts = 0
    s1_modified = 0

    for load in LOADS:
        for seed in SEEDS:
            manifest_path = root / "order_manifests" / f"orders_{load}_seed{seed}.json"
            manifest = _read_json(manifest_path)
            files.append(manifest_path)
            expected_content_hash = manifest.get("manifest_sha256")
            expected_count = int(manifest.get("total_orders", -1))
            arm_rows: dict[str, Mapping[str, Any]] = {}
            for arm_key, arm_label in ARM_KEYS.items():
                path = root / "per_arm" / arm_key / f"{load}_seed{seed}.json"
                payload = _read_json(path)
                files.append(path)
                meta = payload.get("meta") or {}
                arm_audit = payload.get("audit") or {}
                metrics = payload.get("metrics") or {}
                key = f"{arm_key}_{load}_{seed}"
                audit_checks[f"{key}_schema"] = (
                    payload.get("schema_version") == PER_SEED_SCHEMA_VERSION
                )
                audit_checks[f"{key}_protocol"] = (
                    meta.get("protocol_sha256") == protocol_sha256
                    and meta.get("frozen_bundle_sha256") == bundle_sha256
                )
                audit_checks[f"{key}_identity"] = (
                    meta.get("load") == load
                    and int(meta.get("seed", -1)) == seed
                    and meta.get("arm") == arm_label
                )
                audit_checks[f"{key}_arm_audit"] = bool(
                    arm_audit.get("passed")
                )
                audit_checks[f"{key}_manifest"] = (
                    metrics.get("order_arrival_manifest_sha256")
                    == expected_content_hash
                    and int(metrics.get("order_arrival_count", -2))
                    == expected_count
                    and (payload.get("manifest") or {}).get("file_sha256")
                    == sha256_file(manifest_path)
                )
                arm_rows[arm_label] = metrics
                if arm_key == "phasec_s1":
                    s1_contexts += int(metrics.get("energy_conv_contexts", 0))
                    s1_active_contexts += int(
                        metrics.get("energy_conv_active_contexts", 0)
                    )
                    s1_modified += int(
                        metrics.get("energy_conv_modified_decisions", 0)
                    )
            rows[(load, seed)] = arm_rows

    comparisons = {
        "phasec_s1_minus_hungarian": _comparison(
            rows,
            baseline=HUNGARIAN_LABEL,
            candidate=PHASEC_S1_LABEL,
            seed_offset=10000,
        ),
        "phasec_minus_hungarian": _comparison(
            rows,
            baseline=HUNGARIAN_LABEL,
            candidate=PHASEC_LABEL,
            seed_offset=20000,
        ),
        "phasec_s1_minus_phasec": _comparison(
            rows,
            baseline=PHASEC_LABEL,
            candidate=PHASEC_S1_LABEL,
            seed_offset=30000,
        ),
        "hungarian_minus_greedy": _comparison(
            rows,
            baseline=GREEDY_LABEL,
            candidate=HUNGARIAN_LABEL,
            seed_offset=40000,
        ),
    }

    aggregate: dict[str, Any] = {}
    for arm in ARMS:
        aggregate[arm] = {}
        for load in LOADS:
            metric_report = {}
            for metric in REPORT_METRICS:
                values = [
                    _metric(rows[(load, seed)][arm], metric)
                    for seed in SEEDS
                ]
                metric_report[metric] = _mean_std(
                    value for value in values if value is not None
                )
            aggregate[arm][load] = metric_report

    primary_metric = comparisons["phasec_s1_minus_hungarian"]["metrics"][
        "deadlock_ratio_mean"
    ]
    overall_upper = float(primary_metric["overall_seed_cluster"]["ci95"][1])
    each_load_point = all(
        float(primary_metric["by_load"][load]["mean"])
        <= DEADLOCK_NONINFERIORITY_MARGIN
        for load in LOADS
    )
    primary = {
        "metric": "deadlock_ratio_mean",
        "contrast": f"{PHASEC_S1_LABEL}-minus-{HUNGARIAN_LABEL}",
        "margin": DEADLOCK_NONINFERIORITY_MARGIN,
        "overall_ci95_upper": overall_upper,
        "overall_ci95_upper_within_margin": (
            overall_upper <= DEADLOCK_NONINFERIORITY_MARGIN
        ),
        "each_load_point_estimate_within_margin": each_load_point,
    }
    primary["passed"] = bool(
        primary["overall_ci95_upper_within_margin"]
        and primary["each_load_point_estimate_within_margin"]
    )
    mechanism = {
        "s1_contexts": s1_contexts,
        "s1_active_contexts": s1_active_contexts,
        "s1_modified_decisions": s1_modified,
        "passed": s1_contexts > 0 and s1_active_contexts > 0 and s1_modified > 0,
    }
    artifact_audit = {
        "checks": audit_checks,
        "passed": all(audit_checks.values()),
        "per_arm_outputs": len(LOADS) * len(SEEDS) * len(ARMS),
        "manifests": len(LOADS) * len(SEEDS),
    }
    passed = bool(artifact_audit["passed"] and mechanism["passed"] and primary["passed"])
    report = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "protocol_sha256": protocol_sha256,
        "passed": passed,
        "verdict": (
            "PHASEC_S1_DEADLOCK_NONINFERIOR_TO_HUNGARIAN"
            if passed
            else "PHASEC_S1_DEADLOCK_WORSE_OR_AUDIT_FAILED"
        ),
        "artifact_audit": artifact_audit,
        "s1_mechanism": mechanism,
        "primary": primary,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "scope": {
            "native_no_assign": False,
            "new_dispatch_potential": False,
            "checkpoint_retraining": False,
            "loads": list(LOADS),
            "seeds": list(SEEDS),
            "throughput_is_hard_gate": False,
        },
    }

    report_path = root / "phase_c_s1_hungarian_validation.json"
    report_text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    _write_exact(report_path, report_text)
    files.append(report_path)
    hash_text = "".join(
        f"{sha256_file(path)}  {path.as_posix()}\n"
        for path in sorted(files, key=lambda value: value.as_posix())
    )
    _write_exact(root / "validated_outputs.sha256", hash_text)

    print("Phase-C/S1/Hungarian validation complete")
    print(f"artifact audit = {artifact_audit['passed']}")
    print(f"S1 mechanism = {mechanism}")
    print(f"primary = {primary}")
    print(f"verdict = {report['verdict']}")
    print(f"report = {report_path}")


if __name__ == "__main__":
    main()
