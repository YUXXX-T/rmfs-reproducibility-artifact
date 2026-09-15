"""Replay one frozen Phase-C arm with a non-perturbing Lyapunov shadow."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
from WorldModel.evaluation.phase_c_postcert_lyapunov import (
    LyapunovShadowWorldModelTaskAssigner,
    SHADOW_SCHEMA_VERSION,
    shadow_config_dict,
)


REPORT_SCHEMA_VERSION = "phase_c_postcert_lyapunov_shadow_report_v1"
REPRODUCTION_SCHEMA_VERSION = "phase_c_postcert_reproduction_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
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


def _tick_bin(tick: int, width: int = 250) -> str:
    start = max(0, int(tick) // int(width) * int(width))
    return f"{start}-{start + int(width)}"


def summarize_shadow(records: list[dict], horizons: list[int]) -> dict:
    by_bin: dict[str, list[dict]] = {}
    for row in records:
        by_bin.setdefault(_tick_bin(int(row["tick"])), []).append(row)

    def subset_summary(rows: list[dict]) -> dict:
        no_assign = [
            row for row in rows if row.get("selected_action_type") == "no_assign"
        ]
        result = {
            "contexts": len(rows),
            "no_assign_selected": len(no_assign),
            "no_assign_selected_ratio": float(
                len(no_assign) / max(len(rows), 1)
            ),
            "wm_no_assign_advantage": _distribution(
                row["wm_no_assign_advantage"] for row in rows
            ),
            "chain_no_assign_streak_after": _distribution(
                row.get("chain_no_assign_streak_after", 0) for row in rows
            ),
            "global_no_assign_streak_after": _distribution(
                row.get("global_no_assign_streak_after", 0) for row in rows
            ),
            "current_lyapunov_total": _distribution(
                row["current_lyapunov"]["total"] for row in rows
            ),
            "current_work_component": _distribution(
                row["current_lyapunov"]["components"].get("work", 0.0)
                for row in rows
            ),
            "current_station_component": _distribution(
                row["current_lyapunov"]["components"].get("station", 0.0)
                for row in rows
            ),
            "current_arrival_component": _distribution(
                row["current_lyapunov"]["components"].get("arrival", 0.0)
                for row in rows
            ),
            "naive_immediate_l0_prefers_no_assign_rate": float(
                sum(bool(row["naive_immediate_l0_prefers_no_assign"]) for row in rows)
                / max(len(rows), 1)
            ),
            "horizons": {},
        }
        for horizon in horizons:
            key = str(int(horizon))
            advantages = [
                float(row["horizons"][key]["analytic_assign_advantage"])
                for row in rows
            ]
            no_assign_advantages = [
                float(row["horizons"][key]["analytic_assign_advantage"])
                for row in no_assign
            ]
            result["horizons"][key] = {
                "analytic_assign_advantage": _distribution(advantages),
                "selected_no_assign_advantage": _distribution(
                    no_assign_advantages
                ),
                "selected_no_assign_opposition_rate": float(
                    sum(value > 1e-12 for value in no_assign_advantages)
                    / max(len(no_assign_advantages), 1)
                ),
            }
        return result

    return {
        "overall": subset_summary(records),
        "by_tick_bin": {
            key: subset_summary(rows) for key, rows in sorted(by_bin.items())
        },
        "terminal_1000_1500": subset_summary([
            row for row in records if 1000 <= int(row["tick"]) < 1500
        ]),
    }


def _compact_observed_trace(records: list[dict]) -> list[dict]:
    return [
        {
            "tick": int(row.get("tick", -1)),
            "context_idx": int(row.get("context_idx", -1)),
            "selected_action_type": row.get("selected_action_type"),
            "action_status": row.get("action_status"),
        }
        for row in records
    ]


def _compact_reference_trace(records: list[dict]) -> list[dict]:
    return [
        {
            "tick": int(row.get("tick", -1)),
            "context_idx": int(row.get("context_idx", -1)),
            "selected_action_type": row.get("selected_action_type"),
            "action_status": row.get("action_status"),
        }
        for row in records
    ]


def _is_timing_key(path: tuple[str, ...]) -> bool:
    key = path[-1] if path else ""
    return key == "wall_time_s" or "time_ms" in key


def _compare_values(
    expected: Any,
    observed: Any,
    *,
    path: tuple[str, ...] = (),
    failures: list[str],
) -> None:
    if _is_timing_key(path):
        return
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        for key, value in expected.items():
            if key not in observed:
                failures.append("missing:" + ".".join((*path, str(key))))
                continue
            _compare_values(
                value,
                observed[key],
                path=(*path, str(key)),
                failures=failures,
            )
        return
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            failures.append("length:" + ".".join(path))
            return
        for index, (left, right) in enumerate(zip(expected, observed)):
            _compare_values(
                left,
                right,
                path=(*path, str(index)),
                failures=failures,
            )
        return
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        if not math.isclose(
            float(expected), float(observed), rel_tol=1e-7, abs_tol=1e-8
        ):
            failures.append(
                "value:"
                + ".".join(path)
                + f":{expected!r}!={observed!r}"
            )
        return
    if expected != observed:
        failures.append(
            "value:" + ".".join(path) + f":{expected!r}!={observed!r}"
        )


def reproduce_audit(
    *,
    reference_report: dict,
    reference_trace: list[dict],
    observed_metrics: dict,
    observed_trace: list[dict],
) -> dict:
    expected_metrics = reference_report["arms"]["PhaseCWorldModel"]
    metric_failures: list[str] = []
    _compare_values(
        expected_metrics,
        observed_metrics,
        failures=metric_failures,
    )
    compact_expected = _compact_reference_trace(reference_trace)
    compact_observed = _compact_observed_trace(observed_trace)
    trace_equal = compact_expected == compact_observed
    first_trace_mismatch = None
    if not trace_equal:
        limit = min(len(compact_expected), len(compact_observed))
        for index in range(limit):
            if compact_expected[index] != compact_observed[index]:
                first_trace_mismatch = {
                    "index": index,
                    "expected": compact_expected[index],
                    "observed": compact_observed[index],
                }
                break
        if first_trace_mismatch is None:
            first_trace_mismatch = {
                "expected_records": len(compact_expected),
                "observed_records": len(compact_observed),
            }
    checks = {
        "exact_action_trace": trace_equal,
        "non_timing_metrics_match": not metric_failures,
        "order_manifest_sha256": (
            observed_metrics.get("order_arrival_manifest_sha256")
            == expected_metrics.get("order_arrival_manifest_sha256")
        ),
        "no_fallback": int(observed_metrics.get("fallback_greedy_calls", -1)) == 0,
        "all_shadow_records_attached": (
            len(observed_trace)
            == int(observed_metrics.get("decision_trace_contexts", -1))
        ),
    }
    return {
        "schema_version": REPRODUCTION_SCHEMA_VERSION,
        "checks": checks,
        "passed": all(checks.values()),
        "metric_failures": metric_failures[:50],
        "metric_failure_count": len(metric_failures),
        "first_trace_mismatch": first_trace_mismatch,
        "reference_trace_records": len(compact_expected),
        "observed_trace_records": len(compact_observed),
    }


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lyapunov-config", required=True)
    parser.add_argument("--load", choices=("low", "mid", "high"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--top-m", type=int, default=10)
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--recorded-orders", required=True)
    parser.add_argument("--reference-per-seed", required=True)
    parser.add_argument("--reference-trace", required=True)
    parser.add_argument(
        "--case-role",
        choices=(
            "no_assign_failure",
            "healthy_control",
            "near_absorbing_failure",
            "congestion_failure",
            "positive_control",
        ),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--records-output", default=None)
    args = parser.parse_args()

    paths = {
        "config": Path(args.config),
        "checkpoint": Path(args.checkpoint),
        "lyapunov_config": Path(args.lyapunov_config),
        "recorded_orders": Path(args.recorded_orders),
        "reference_per_seed": Path(args.reference_per_seed),
        "reference_trace": Path(args.reference_trace),
        "output": Path(args.output),
    }
    for name, path in paths.items():
        if name == "output":
            continue
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    if paths["output"].exists():
        raise FileExistsError(f"refusing to overwrite {paths['output']}")
    if int(args.ticks) <= 0 or int(args.top_m) <= 0:
        raise ValueError("ticks and top-m must be positive")
    horizons = sorted({int(value) for value in args.horizons})
    if not horizons or min(horizons) <= 0:
        raise ValueError("horizons must be positive")

    reference_report = _read_json(paths["reference_per_seed"])
    meta = reference_report.get("meta") or {}
    if (
        str(meta.get("load")) != args.load
        or int(meta.get("seed", -1)) != int(args.seed)
        or int(meta.get("ticks", -1)) != int(args.ticks)
    ):
        raise ValueError("reference per-seed report does not match requested run")
    reference_trace = _read_jsonl(paths["reference_trace"])
    with paths["lyapunov_config"].open("r", encoding="utf-8") as handle:
        lyapunov_config = json.load(handle)

    assigner = LyapunovShadowWorldModelTaskAssigner(
        checkpoint_path=str(paths["checkpoint"]),
        top_m=int(args.top_m),
        decision_trace_max_records=200000,
        lyapunov_l0_config=lyapunov_config,
        shadow_horizons=horizons,
    )
    metrics = _run_one_assigner(
        str(paths["config"]),
        assigner,
        int(args.seed),
        int(args.ticks),
        trace_label="PhaseCWorldModelLyapunovShadow",
        recorded_orders_path=str(paths["recorded_orders"]),
    )
    reproduction = reproduce_audit(
        reference_report=reference_report,
        reference_trace=reference_trace,
        observed_metrics=metrics,
        observed_trace=assigner.decision_trace_records,
    )
    if not reproduction["passed"]:
        raise RuntimeError(
            "Lyapunov shadow perturbed or failed to reproduce the frozen arm: "
            + json.dumps(reproduction, ensure_ascii=False)
        )
    if len(assigner.lyapunov_shadow_records) != len(
        assigner.decision_trace_records
    ):
        raise RuntimeError("not every frozen decision received a shadow record")

    records_path = (
        Path(args.records_output)
        if args.records_output
        else paths["output"].with_name(paths["output"].stem + "_records.jsonl.gz")
    )
    if records_path.exists():
        raise FileExistsError(f"refusing to overwrite {records_path}")
    _write_records(records_path, assigner.lyapunov_shadow_records)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": "POSTCERT_READ_ONLY_FAILURE_MECHANISM_DIAGNOSTIC",
        "meta": {
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "case_role": args.case_role,
            "not_policy_certification": True,
            "certification_seed_not_used_for_tuning": True,
        },
        "inputs": {
            name: {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
            if name != "output"
        },
        "shadow_config": shadow_config_dict(assigner),
        "reproduction_audit": reproduction,
        "metrics": metrics,
        "shadow_summary": summarize_shadow(
            assigner.lyapunov_shadow_records,
            horizons,
        ),
        "records": {
            "path": records_path.as_posix(),
            "sha256": sha256_file(records_path),
            "records": len(assigner.lyapunov_shadow_records),
            "schema_version": SHADOW_SCHEMA_VERSION,
        },
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    with paths["output"].open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(f"saved: {paths['output']}")
    print(f"records: {records_path}")
    print("reproduction audit PASS")
    print(
        "terminal NO_ASSIGN ratio =",
        report["shadow_summary"]["terminal_1000_1500"][
            "no_assign_selected_ratio"
        ],
    )
    for horizon in horizons:
        row = report["shadow_summary"]["terminal_1000_1500"]["horizons"][
            str(horizon)
        ]
        print(
            f"H={horizon} terminal opposition rate =",
            row["selected_no_assign_opposition_rate"],
            "advantage median =",
            row["selected_no_assign_advantage"]["p50"],
        )


if __name__ == "__main__":
    main()
