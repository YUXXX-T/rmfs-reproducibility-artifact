"""Frozen scope for targeted Phase-C closed-loop failure diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = "phasec_failure_diagnostic_protocol_v1"
FROZEN_BUNDLE_SCHEMA_VERSION = "phasec_failure_diagnostic_bundle_v1"
RUN_SCHEMA_VERSION = "phasec_failure_diagnostic_run_v1"
VALIDATION_SCHEMA_VERSION = "phasec_failure_diagnostic_validation_v1"
COUNTERFACTUAL_SCHEMA_VERSION = "phasec_failure_counterfactual_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
SOURCE_ROOT = Path(os.environ.get(
    "PHASEC_FAILURE_SOURCE_ROOT",
    str(BASE_ROOT / "phasec_s1_hungarian_replication_501_510_v1"),
))
OUTPUT_ROOT = Path(os.environ.get(
    "PHASEC_FAILURE_OUTPUT_ROOT",
    str(BASE_ROOT / "phasec_failure_diagnostic_501_506_v1"),
))
SOURCE_BUNDLE = SOURCE_ROOT / "phase_c_s1_hungarian_frozen_protocol.json"

TICKS = 1500
FRAME_STRIDE = 5
TRACE_HORIZONS = (50, 100, 200)
COUNTERFACTUAL_HORIZONS = (50, 100, 200)
COUNTERFACTUAL_DELAY_SCALE = 10.0
ARMS = ("greedy", "hungarian", "phasec", "phasec_s1")

# Each case includes all four closed-loop arms. Snapshot capture is restricted
# to the arm that answers the case's primary ranking question.
CASES = (
    {
        "load": "high",
        "seed": 501,
        "snapshot_arm": "phasec_s1",
        "question": (
            "did one of six S1 conversion changes trigger the divergence from "
            "the successful Phase-C trajectory"
        ),
    },
    {
        "load": "low",
        "seed": 506,
        "snapshot_arm": "phasec",
        "question": (
            "did the base Phase-C within-context robot ranking contribute to "
            "the shared Phase-C/S1 congestion failure"
        ),
    },
    {
        "load": "mid",
        "seed": 506,
        "snapshot_arm": "phasec",
        "question": (
            "is the second shared Phase-C/S1 failure consistent with the same "
            "base-ranking mechanism"
        ),
    },
)

DETERMINISTIC_METRIC_KEYS = (
    "completed_orders",
    "completed_tasks",
    "ticks",
    "open_order_count",
    "pending_order_count",
    "completed_orders_delta_sum",
    "congestion_events",
    "severe_events",
    "stall_ratio_mean",
    "stall_ratio_max",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "handoff_ratio_mean",
    "handoff_ratio_max",
    "order_arrival_count",
    "order_arrival_manifest_sha256",
)

WM_DETERMINISTIC_METRIC_KEYS = (
    "assign_calls",
    "model_assign_calls",
    "model_inference_calls",
    "fallback_greedy_calls",
    "warmup_defer_calls",
    "decision_contexts_total",
    "energy_conv_contexts",
    "energy_conv_modified_decisions",
)

DIAGNOSTIC_SOURCE_FILES = (
    "WorldModel/evaluation/phase_c_failure_diagnostic_protocol.py",
    "WorldModel/evaluation/phase_c_failure_diagnostic_probe.py",
    "WorldModel/evaluation/freeze_phase_c_failure_diagnostic.py",
    "WorldModel/evaluation/run_phase_c_failure_diagnostic.py",
    "WorldModel/evaluation/replay_phase_c_failure_counterfactuals.py",
    "WorldModel/evaluation/validate_phase_c_failure_diagnostic.py",
    "WorldModel/evaluation/run_phase_c_failure_diagnostic_cpu.slurm",
    "WorldModel/tests/test_phase_c_failure_diagnostic.py",
)


def case_for(load: str, seed: int) -> dict:
    for case in CASES:
        if case["load"] == load and int(case["seed"]) == int(seed):
            return dict(case)
    raise ValueError(f"diagnostic case is not frozen: {load} seed={seed}")


def run_id(arm: str, load: str, seed: int) -> str:
    if arm not in ARMS:
        raise ValueError(arm)
    case_for(load, seed)
    return f"{arm}_{load}_seed{int(seed)}"


def canonical_sha256(payload: Mapping) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_payload(source_bundle_sha256: str) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "diagnose low completion with high deadlock and test whether "
            "within-context World-Model ranking errors precede the failures"
        ),
        "source_comparison": {
            "root": SOURCE_ROOT.as_posix(),
            "frozen_bundle": SOURCE_BUNDLE.as_posix(),
            "frozen_bundle_sha256": source_bundle_sha256,
            "exact_order_manifest_replay": True,
            "reuse_existing_per_arm_references": True,
        },
        "cases": [dict(case) for case in CASES],
        "arms": list(ARMS),
        "ticks": TICKS,
        "td_frame_stride": FRAME_STRIDE,
        "decision_trace_future_horizons": list(TRACE_HORIZONS),
        "counterfactual_replay": {
            "scope": "every captured rankable context",
            "forced_first_action": "each online robot candidate",
            "shared_continuation": "greedy_behavior",
            "horizons": list(COUNTERFACTUAL_HORIZONS),
            "system_label_delay_scale": COUNTERFACTUAL_DELAY_SCALE,
            "primary_evidence": (
                "predicted-vs-realized ranking plus completion/risk Pareto audit"
            ),
        },
        "snapshot_contract": {
            "capture_only_declared_snapshot_arm": True,
            "capture_every_rankable_online_context": True,
            "candidate_scope": "all_idle_online",
            "trace_alignment_required": True,
            "native_no_assign": False,
            "min_robot_candidates": 2,
        },
        "non_perturbation_audit": {
            "reference": "existing per_arm JSON for the same arm/load/seed",
            "metric_keys": list(DETERMINISTIC_METRIC_KEYS),
            "wm_metric_keys": list(WM_DETERMINISTIC_METRIC_KEYS),
            "exact_value_match_required": True,
        },
        "interpretation_limits": {
            "decision_trace_future_is_observed_policy_future": True,
            "snapshot_supports_within_context_counterfactuals": True,
            "counterfactual_continuation_is_not_original_policy_future": True,
            "snapshot_does_not_prove_cross_context_joint_optimality": True,
            "diagnostic_results_are_not_new_certification_seeds": True,
        },
    }
    return {**payload, "protocol_sha256": canonical_sha256(payload)}
