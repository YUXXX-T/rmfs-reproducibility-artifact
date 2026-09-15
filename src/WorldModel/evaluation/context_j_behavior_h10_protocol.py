"""Frozen protocol for the expanded multi-context behavior labels.

This protocol is intentionally independent from the historical
``behavior_h10_531_540_v1`` bundle.  It collects all dispatchable contexts at
every tenth tick under the frozen Greedy behavior continuation, without
adding NO_ASSIGN or changing the World Model/S1 policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "context_j_behavior_h10_protocol_v1"
# Keep the generic bundle schema compatible with the frozen behavior validator
# used for the model-side H=10 audit.  The protocol hash and purpose identify
# this bundle; no old bundle is modified or reused.
FROZEN_BUNDLE_SCHEMA_VERSION = "phase_c_behavior_h10_bundle_v1"
RUN_SCHEMA_VERSION = "context_j_behavior_h10_run_v1"
REPORT_SCHEMA_VERSION = "context_j_behavior_h10_report_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
OUTPUT_ROOT = BASE_ROOT / "context_j_behavior_h10_571_590_v1"
MODEL_CHECKPOINT = BASE_ROOT / "model_round1_v1/best_regret_world_model.pt"

LOADS = ("low", "mid", "high")
LOAD_CONFIGS = {
    "low": Path("Config/world_model_config_PP_48_low.json"),
    "mid": Path("Config/world_model_config_PP_48_mid.json"),
    "high": Path("Config/world_model_config_PP_48_high.json"),
}
SEEDS = tuple(range(571, 591))
TRAIN_SEEDS = tuple(range(571, 583))
VAL_SEEDS = tuple(range(583, 587))
TEST_SEEDS = tuple(range(587, 591))

TICKS = 1500
HORIZON = 10
HISTORY_LEN = 4
SAMPLE_INTERVAL = 10
TOP_M = 5
MIN_GROUP = 2
DELAY_SCALE = 10.0
STATION_QUEUE_DELTA_SCALE = None
STATION_QUEUE_SCALE = None
STATION_LOAD_SCALE = None
STALLED_RATIO_THRESHOLD = 0.3
RISK_DURATION = 8
RESERVATION_WINDOW = 1
ASSIGNMENT_MODE = "parallel"
CANDIDATE_ROBOT_MODE = "nearest"
ROLLOUT_CONTINUATION_MODE = "behavior"
BEHAVIOR_POLICY = "GreedyTaskAssigner"
INCLUDE_NO_ASSIGN = False
MAX_GROUPS_PER_TICK = None


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
    """Return the complete immutable collection contract."""

    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "expand multi-context behavior-continuation labels for context "
            "dispatch-J head training; the frozen encoder, transition, "
            "decoder, S1 robot scorer, and order stream are unchanged"
        ),
        "inputs": {
            "model_checkpoint": MODEL_CHECKPOINT.as_posix(),
            "loads": {name: path.as_posix() for name, path in LOAD_CONFIGS.items()},
            "seeds": list(SEEDS),
            "split_seeds": {
                "train": list(TRAIN_SEEDS),
                "val": list(VAL_SEEDS),
                "test": list(TEST_SEEDS),
            },
        },
        "collection": {
            "ticks": TICKS,
            "history_len": HISTORY_LEN,
            "horizon": HORIZON,
            "sample_interval": SAMPLE_INTERVAL,
            "reservation_window": RESERVATION_WINDOW,
            "top_m": TOP_M,
            "min_group": MIN_GROUP,
            "delay_scale": DELAY_SCALE,
            "station_queue_delta_scale": STATION_QUEUE_DELTA_SCALE,
            "station_queue_scale": STATION_QUEUE_SCALE,
            "station_load_scale": STATION_LOAD_SCALE,
            "stalled_ratio_threshold": STALLED_RATIO_THRESHOLD,
            "risk_duration": RISK_DURATION,
            "assignment_mode": ASSIGNMENT_MODE,
            "candidate_robot_mode": CANDIDATE_ROBOT_MODE,
            "rollout_continuation_mode": ROLLOUT_CONTINUATION_MODE,
            "behavior_policy": BEHAVIOR_POLICY,
            "include_no_assign": INCLUDE_NO_ASSIGN,
            "max_groups_per_tick": MAX_GROUPS_PER_TICK,
            "future_order_generation": True,
            "continuation_scheduler": True,
            "all_dispatchable_contexts": True,
        },
        "label_contract": {
            "robot_candidates": (
                "top-m nearest robots proposed for each fixed order/pod/station "
                "context"
            ),
            "context_target": (
                "minimum true behavior-continuation composite cost among the "
                "robot candidates in that context"
            ),
            "cross_context_group": "all distinct contexts sharing run and decision tick",
            "system_cost": "DEFAULT_LAMBDAS applied to H=10 future system labels",
            "local_station_cost": (
                "discounted target-station queue plus assigned-load trajectory; "
                "gamma=0.95, weights=(1,1)"
            ),
            "lower_score_is_better": True,
            "analytic_J_is_not_used_as_label": True,
            "no_assign_rows": False,
        },
        "quality_gates": {
            "expected_runs": len(SEEDS) * len(LOADS),
            "minimum_non_tie_pairs": {
                "train": 150,
                "val": 40,
                "test": 40,
            },
            "pair_epsilon": 0.01,
            "all_runs_required": True,
        },
        "forbidden": {
            "protected_seeds_501_510": True,
            "protected_seeds_551_570": True,
            "checkpoint_reselection": True,
            "encoder_or_transition_update": True,
            "psi_or_delta_psi_connection": True,
            "q_score_change": True,
            "backlog_override": True,
            "post_result_parameter_tuning": True,
        },
    }


__all__ = [
    "ASSIGNMENT_MODE",
    "BASE_ROOT",
    "BEHAVIOR_POLICY",
    "CANDIDATE_ROBOT_MODE",
    "DELAY_SCALE",
    "FROZEN_BUNDLE_SCHEMA_VERSION",
    "HORIZON",
    "LOAD_CONFIGS",
    "LOADS",
    "MAX_GROUPS_PER_TICK",
    "MIN_GROUP",
    "MODEL_CHECKPOINT",
    "OUTPUT_ROOT",
    "REPORT_SCHEMA_VERSION",
    "ROLLOUT_CONTINUATION_MODE",
    "RUN_SCHEMA_VERSION",
    "SAMPLE_INTERVAL",
    "SCHEMA_VERSION",
    "SEEDS",
    "TEST_SEEDS",
    "TICKS",
    "TOP_M",
    "TRAIN_SEEDS",
    "VAL_SEEDS",
    "canonical_sha256",
    "formal_protocol",
    "sha256_file",
]
