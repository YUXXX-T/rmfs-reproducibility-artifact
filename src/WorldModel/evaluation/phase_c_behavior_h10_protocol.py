"""Frozen protocol for closed-loop behavior-continuation H=10 validation.

This protocol is deliberately separate from every frozen Phase-C/S1 result.
It tests whether the frozen H=10 World Model remains locally calibrated when
the simulator continues to generate orders and the behavior scheduler runs
after the forced candidate action.  No psi/Q modification is part of this
experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "phase_c_behavior_h10_protocol_v1"
FROZEN_BUNDLE_SCHEMA_VERSION = "phase_c_behavior_h10_bundle_v1"
RUN_SCHEMA_VERSION = "phase_c_behavior_h10_run_v1"
REPORT_SCHEMA_VERSION = "phase_c_behavior_h10_report_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
OUTPUT_ROOT = BASE_ROOT / "behavior_h10_531_540_v1"
MODEL_CHECKPOINT = BASE_ROOT / "model_round1_v1/best_regret_world_model.pt"

LOADS = ("low", "mid", "high")
LOAD_CONFIGS = {
    "low": Path("Config/world_model_config_PP_48_low.json"),
    "mid": Path("Config/world_model_config_PP_48_mid.json"),
    "high": Path("Config/world_model_config_PP_48_high.json"),
}
SEEDS = tuple(range(531, 541))
TICKS = 1500
HORIZON = 10
HISTORY_LEN = 4
SAMPLE_INTERVAL = 25
TOP_M = 5
MIN_GROUP = 2
DELAY_SCALE = 10.0
STATION_QUEUE_DELTA_SCALE = None
STALLED_RATIO_THRESHOLD = 0.3
RISK_DURATION = 8
CANDIDATE_ROBOT_MODE = "nearest"
ROLLOUT_CONTINUATION_MODE = "behavior"
BEHAVIOR_POLICY = "GreedyTaskAssigner"


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def formal_protocol() -> dict[str, Any]:
    """Return the immutable test contract.

    The protocol intentionally has no outcome-derived threshold.  The first
    run is a mechanism/coverage validation; numerical results are reported
    by load, seed, decision group, and horizon step for a later preregistered
    gate review.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "validate local H=10 system-dynamics calibration under ongoing "
            "order arrivals and behavior-policy continuation before any "
            "station potential psi_pre is connected to Q(c,r)"
        ),
        "inputs": {
            "model_checkpoint": MODEL_CHECKPOINT.as_posix(),
            "loads": {key: value.as_posix() for key, value in LOAD_CONFIGS.items()},
            "seeds": list(SEEDS),
        },
        "collection": {
            "ticks": TICKS,
            "history_len": HISTORY_LEN,
            "horizon": HORIZON,
            "sample_interval": SAMPLE_INTERVAL,
            "top_m": TOP_M,
            "min_group": MIN_GROUP,
            "delay_scale": DELAY_SCALE,
            "station_queue_delta_scale": STATION_QUEUE_DELTA_SCALE,
            "stalled_ratio_threshold": STALLED_RATIO_THRESHOLD,
            "risk_duration": RISK_DURATION,
            "candidate_robot_mode": CANDIDATE_ROBOT_MODE,
            "include_no_assign": False,
            "rollout_continuation_mode": ROLLOUT_CONTINUATION_MODE,
            "behavior_policy": BEHAVIOR_POLICY,
            "future_order_generation": True,
            "continuation_scheduler": True,
            "new_orders_are_generated_after_step_zero": True,
        },
        "validation": {
            "fresh_encode_unit": "each candidate sample / decision group",
            "model_rollout_horizon": HORIZON,
            "delay_metric_unit": "physical_ticks",
            "completed_orders_delta_semantics": (
                "per-step completion event; cumulative prefix is also reported"
            ),
            "primary_outputs": [
                "system_channels_by_step",
                "system_channels_at_h10",
                "cumulative_completed_orders_at_h10",
                "node_density_and_congestion_metrics",
                "station_queue_and_assigned_load_metrics",
            ],
        },
        "interpretation_limits": [
            (
                "This is a closed-loop target test for a frozen "
                "candidate-conditioned model; it does not retrain the "
                "encoder/transition."
            ),
            (
                "A failure can reflect a continuation-semantic mismatch as "
                "well as transition/decoder error; isolated and behavior "
                "reports must not be conflated."
            ),
            (
                "No psi_pre, DeltaPsi, dispatch gate, liveness term, or "
                "Q(c,r) change is allowed in this protocol."
            ),
        ],
        "forbidden": {
            "seeds_501_510": True,
            "checkpoint_reselection": True,
            "encoder_or_transition_update": True,
            "psi_pre_or_delta_psi": True,
            "q_score_change": True,
            "post_result_parameter_tuning": True,
        },
    }


__all__ = [
    "BASE_ROOT",
    "BEHAVIOR_POLICY",
    "CANDIDATE_ROBOT_MODE",
    "DELAY_SCALE",
    "FROZEN_BUNDLE_SCHEMA_VERSION",
    "HISTORY_LEN",
    "HORIZON",
    "LOAD_CONFIGS",
    "LOADS",
    "MIN_GROUP",
    "MODEL_CHECKPOINT",
    "OUTPUT_ROOT",
    "REPORT_SCHEMA_VERSION",
    "ROLLOUT_CONTINUATION_MODE",
    "RUN_SCHEMA_VERSION",
    "SAMPLE_INTERVAL",
    "SCHEMA_VERSION",
    "SEEDS",
    "TICKS",
    "TOP_M",
    "canonical_sha256",
    "formal_protocol",
    "sha256_file",
]
