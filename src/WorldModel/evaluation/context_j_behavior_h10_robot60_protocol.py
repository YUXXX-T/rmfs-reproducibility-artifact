"""Frozen protocol constants for 60-robot context-J label collection.

The bundle mirrors the proven 571--590 H=10 behavior-continuation pipeline,
but uses independently derived 60-robot configurations, fresh seeds 611--630,
and the deployed committed-capacity FIFO-V2 station admission semantics.
Seeds 601--610 remain a read-only pressure/regression block and are forbidden
from fitting or checkpoint selection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "context_j_behavior_h10_robot60_protocol_v1"
FROZEN_BUNDLE_SCHEMA_VERSION = "phase_c_behavior_h10_bundle_v1"
RUN_SCHEMA_VERSION = "context_j_behavior_h10_robot60_run_v1"
REPORT_SCHEMA_VERSION = "context_j_behavior_h10_robot60_report_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
OUTPUT_ROOT = BASE_ROOT / "context_j_behavior_h10_robot60_611_630_v1"
MODEL_CHECKPOINT = BASE_ROOT / "model_round1_v1/best_regret_world_model.pt"

SOURCE_ROBOT_COUNT = 48
TARGET_ROBOT_COUNT = 60
LOADS = ("low", "mid", "high")
SOURCE_LOAD_CONFIGS = {
    "low": Path("Config/world_model_config_PP_48_low.json"),
    "mid": Path("Config/world_model_config_PP_48_mid.json"),
    "high": Path("Config/world_model_config_PP_48_high.json"),
}
SEEDS = tuple(range(611, 631))
TRAIN_SEEDS = tuple(range(611, 623))
VAL_SEEDS = tuple(range(623, 627))
TEST_SEEDS = tuple(range(627, 631))

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
STATION_ADMISSION = "committed_capacity_fifo_v2"
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


def derived_config_path(output_root: Path, load: str) -> Path:
    return output_root / "frozen_configs" / f"world_model_config_PP_60_{load}.json"


def formal_protocol(
    *,
    output_root: Path = OUTPUT_ROOT,
    load_configs: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    configs = dict(load_configs or {
        load: derived_config_path(output_root, load) for load in LOADS
    })
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "fit a 60-robot context dispatch-J head from fresh multi-context "
            "H=10 behavior-continuation labels while keeping the frozen World "
            "Model, station phi, S1 robot scorer, and J architecture unchanged"
        ),
        "inputs": {
            "model_checkpoint": MODEL_CHECKPOINT.as_posix(),
            "source_loads": {
                name: path.as_posix() for name, path in SOURCE_LOAD_CONFIGS.items()
            },
            "loads": {name: configs[name].as_posix() for name in LOADS},
            "source_robot_count": SOURCE_ROBOT_COUNT,
            "target_robot_count": TARGET_ROBOT_COUNT,
            "allowed_config_changes": ["robots.num_robots"],
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
            "station_admission": STATION_ADMISSION,
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
                "minimum true FIFO-V2 behavior-continuation composite cost "
                "among the robot candidates in that context"
            ),
            "cross_context_group": (
                "all distinct contexts sharing run and decision tick"
            ),
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
            "minimum_non_tie_pairs": {"train": 150, "val": 40, "test": 40},
            "pair_epsilon": 0.01,
            "all_runs_required": True,
            "all_runs_use_60_robots": True,
            "fifo_v2_invariant_required": True,
        },
        "forbidden": {
            "pressure_seeds_601_610_for_training": True,
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
    "BASE_ROOT",
    "BEHAVIOR_POLICY",
    "FROZEN_BUNDLE_SCHEMA_VERSION",
    "LOADS",
    "MODEL_CHECKPOINT",
    "OUTPUT_ROOT",
    "REPORT_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SEEDS",
    "SOURCE_LOAD_CONFIGS",
    "SOURCE_ROBOT_COUNT",
    "STATION_ADMISSION",
    "TARGET_ROBOT_COUNT",
    "TEST_SEEDS",
    "TRAIN_SEEDS",
    "VAL_SEEDS",
    "canonical_sha256",
    "derived_config_path",
    "formal_protocol",
    "sha256_file",
]
