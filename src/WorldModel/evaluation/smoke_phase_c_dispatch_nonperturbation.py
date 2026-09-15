"""Verify that dispatch diagnostic mode is exactly action-nonperturbing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from Policies.TaskAssigner import WorldModelTaskAssigner
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
from WorldModel.evaluation.phase_c_dispatch_gate1_protocol import (
    CANDIDATE_CHECKPOINT,
    sha256_file,
)


TIMING_OR_EXPECTED_META = {
    "wall_time_s",
    "assignment_time_ms_mean",
    "model_inference_time_ms_mean",
    "order_arrival_replayed",
    "order_arrival_manifest_path",
    "dispatch_potential_mode",
    "max_eligible_defer_streak_bound_passed",
}


def _action_trace(assigner: WorldModelTaskAssigner) -> list[dict[str, Any]]:
    return [
        {
            "tick": int(record.get("tick", -1)),
            "context_idx": int(record.get("context_idx", -1)),
            "order_id": int(record.get("order_id", -1)),
            "pod_id": int(record.get("pod_id", -1)),
            "station_id": int(record.get("station_id", -1)),
            "selected_action_type": record.get("selected_action_type"),
            "selected_robot": record.get("selected_robot"),
            "candidate_ids": record.get("candidate_ids"),
        }
        for record in assigner.decision_trace_records
    ]


def _physical_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in TIMING_OR_EXPECTED_META
        and not key.startswith("dispatch_")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CANDIDATE_CHECKPOINT)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=9981)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest = work_dir / f"seed{args.seed}_orders.json"

    baseline_assigner = WorldModelTaskAssigner(
        checkpoint_path=args.checkpoint,
        include_no_assign_candidate=True,
        decision_trace_enabled=True,
    )
    baseline = _run_one_assigner(
        args.config,
        baseline_assigner,
        int(args.seed),
        int(args.ticks),
        save_order_manifest=str(manifest),
    )
    diagnostic_assigner = WorldModelTaskAssigner(
        checkpoint_path=args.checkpoint,
        include_no_assign_candidate=True,
        dispatch_potential_mode="diagnostic",
        decision_trace_enabled=True,
    )
    diagnostic = _run_one_assigner(
        args.config,
        diagnostic_assigner,
        int(args.seed),
        int(args.ticks),
        recorded_orders_path=str(manifest),
    )

    baseline_trace = _action_trace(baseline_assigner)
    diagnostic_trace = _action_trace(diagnostic_assigner)
    baseline_metrics = _physical_metrics(baseline)
    diagnostic_metrics = _physical_metrics(diagnostic)
    checks = {
        "same_order_manifest": (
            baseline.get("order_arrival_manifest_sha256")
            == diagnostic.get("order_arrival_manifest_sha256")
        ),
        "exact_action_trace": baseline_trace == diagnostic_trace,
        "exact_non_timing_metrics": baseline_metrics == diagnostic_metrics,
        "diagnostic_exercised": int(
            diagnostic.get("dispatch_contexts", 0)
        ) > 0,
        "group_span_integrity": int(
            diagnostic.get("dispatch_group_span_violations", -1)
        ) == 0,
        "no_crossing_algebra_violation": int(
            diagnostic.get("dispatch_crossing_violations", -1)
        ) == 0,
    }
    payload = {
        "schema_version": "phase_c_dispatch_diagnostic_nonperturbation_v1",
        "seed": int(args.seed),
        "ticks": int(args.ticks),
        "config": Path(args.config).as_posix(),
        "checkpoint": {
            "path": Path(args.checkpoint).as_posix(),
            "sha256": sha256_file(args.checkpoint),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "baseline_trace_records": len(baseline_trace),
        "diagnostic_trace_records": len(diagnostic_trace),
        "manifest": {
            "path": manifest.as_posix(),
            "sha256": sha256_file(manifest),
        },
        "metric_difference_keys": sorted(
            key
            for key in set(baseline_metrics) | set(diagnostic_metrics)
            if baseline_metrics.get(key) != diagnostic_metrics.get(key)
        ),
    }
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite diagnostic smoke: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not payload["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit("dispatch diagnostic nonperturbation FAIL: " + ", ".join(failed))
    print("Phase-C dispatch diagnostic nonperturbation PASS")
    print("trace records =", len(baseline_trace))
    print("report =", output)


if __name__ == "__main__":
    main()
