"""Frozen development protocol for station-potential endpoint transport."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from WorldModel.evaluation.phase_c_s1_hungarian_protocol import S1_CONFIG


SCHEMA_VERSION = "station_congestion_endpoint_transport_protocol_v1"
FROZEN_BUNDLE_SCHEMA_VERSION = (
    "station_congestion_endpoint_transport_bundle_v1"
)
COLLECTION_SCHEMA_VERSION = "station_congestion_endpoint_collection_v1"
REPLAY_SCHEMA_VERSION = "station_congestion_endpoint_replay_shard_v1"
ANALYSIS_SCHEMA_VERSION = "station_congestion_endpoint_analysis_v1"

BASE_ROOT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1"
)
OUTPUT_ROOT = BASE_ROOT / "station_congestion_endpoint_h10_dev_521_530_v1"
MODEL_CHECKPOINT = (
    BASE_ROOT / "model_round1_v1/best_regret_world_model.pt"
)
HEAD_ROOT = BASE_ROOT / "station_congestion_head_region_dev_511_520_v1"
HEAD_CHECKPOINT = HEAD_ROOT / "linear_head_v1/best_station_congestion_head.pt"
SCALE_CONTRACT = HEAD_ROOT / "station_congestion_scale_contract.json"

LOADS = ("low", "mid", "high")
LOAD_CONFIGS = {
    "low": Path("Config/world_model_config_PP_48_low.json"),
    "mid": Path("Config/world_model_config_PP_48_mid.json"),
    "high": Path("Config/world_model_config_PP_48_high.json"),
}
SEEDS = tuple(range(521, 531))
TICKS = 1500
HORIZON = 10

BEHAVIOUR_POLICY = "PhaseCS1RobotOnly"
BEHAVIOUR_TOP_M = 10
SNAPSHOT_INTERVAL = 100
SNAPSHOT_TOP_M = 4
SNAPSHOT_MAX_CONTEXTS_PER_TICK = 1
SNAPSHOT_CANDIDATE_ROBOT_MODE = "stratified"
MIN_SNAPSHOTS_PER_RUN = 8
REPLAY_SHARDS = 2

BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260804


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
    return {
        "schema_version": SCHEMA_VERSION,
        "development_only": True,
        "claim": (
            "with H fixed at 10, separate real-endpoint station-head "
            "decoding, frozen World-Model endpoint latent transport, and "
            "station-potential delta transport on fresh post-520 seeds"
        ),
        "inputs": {
            "model_checkpoint": MODEL_CHECKPOINT.as_posix(),
            "station_head_checkpoint": HEAD_CHECKPOINT.as_posix(),
            "station_scale_contract": SCALE_CONTRACT.as_posix(),
            "loads": {
                key: value.as_posix() for key, value in LOAD_CONFIGS.items()
            },
        },
        "state_collection": {
            "loads": list(LOADS),
            "seeds": list(SEEDS),
            "ticks": TICKS,
            "behaviour_policy": BEHAVIOUR_POLICY,
            "behaviour_top_m": BEHAVIOUR_TOP_M,
            "behaviour_config": dict(S1_CONFIG),
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "snapshot_top_m": SNAPSHOT_TOP_M,
            "snapshot_max_contexts_per_tick": (
                SNAPSHOT_MAX_CONTEXTS_PER_TICK
            ),
            "snapshot_candidate_robot_mode": (
                SNAPSHOT_CANDIDATE_ROBOT_MODE
            ),
            "minimum_complete_snapshots_per_run": MIN_SNAPSHOTS_PER_RUN,
            "order_stream": "seeded native generator, saved per run",
        },
        "counterfactual": {
            "horizon": HORIZON,
            "candidate_actions": "assignment candidates only",
            "defer_context_included": False,
            "native_global_no_assign_included": False,
            "continuation_mode": "isolated",
            "future_order_generation": False,
            "continuation_scheduler": False,
            "replay_shards_per_run": REPLAY_SHARDS,
        },
        "measurements": {
            "real_endpoint_decoding": (
                "Psi(z_H_real) versus physical Y_H_real"
            ),
            "endpoint_latent_transport": (
                "Psi(z_H_pred) versus Psi(z_H_real)"
            ),
            "delta_transport": (
                "Psi(z_H_pred)-Psi(z_0) versus "
                "Psi(z_H_real)-Psi(z_0)"
            ),
            "channels": ["traffic", "service"],
            "subsets": ["all_stations", "context_station"],
            "ranking_units": [
                "stations_within_snapshot_candidate",
                "candidates_within_snapshot_context_station",
            ],
            "cluster_unit": "source simulation run",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "decision_rule": {
            "performance_gate_frozen": False,
            "purpose": (
                "mechanism kill-point before DEFER_CONTEXT, Q(c,r), or any "
                "online action change"
            ),
        },
        "forbidden": {
            "seeds_501_510": True,
            "head_retraining_or_reselection": True,
            "encoder_or_transition_update": True,
            "online_assigner_change": True,
            "q_score_change": True,
            "defer_substitution_with_global_zero_action": True,
            "horizon_selection_after_results": True,
        },
    }


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "BEHAVIOUR_POLICY",
    "BEHAVIOUR_TOP_M",
    "BOOTSTRAP_REPEATS",
    "BOOTSTRAP_SEED",
    "COLLECTION_SCHEMA_VERSION",
    "FROZEN_BUNDLE_SCHEMA_VERSION",
    "HEAD_CHECKPOINT",
    "HORIZON",
    "LOAD_CONFIGS",
    "LOADS",
    "MIN_SNAPSHOTS_PER_RUN",
    "MODEL_CHECKPOINT",
    "OUTPUT_ROOT",
    "REPLAY_SCHEMA_VERSION",
    "REPLAY_SHARDS",
    "SCALE_CONTRACT",
    "SCHEMA_VERSION",
    "SEEDS",
    "SNAPSHOT_CANDIDATE_ROBOT_MODE",
    "SNAPSHOT_INTERVAL",
    "SNAPSHOT_MAX_CONTEXTS_PER_TICK",
    "SNAPSHOT_TOP_M",
    "TICKS",
    "canonical_sha256",
    "formal_protocol",
    "sha256_file",
]
