"""Frozen protocol for repairing the formal Phase-C LongRiskHead lineage.

The source Phase-C checkpoint is retained as the dynamics/cost model.  Only
``long_risk_head`` is trained from W=200 closed-loop supervision collected
under the source Phase-C behavior policy.  Evaluation seeds are disjoint from
all training, validation, and offline-test seeds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from WorldModel.core.long_risk_schema import long_risk_runtime_contract
from WorldModel.evaluation.phase_c_combo_s1_j1_paper_protocol import (
    selector_config,
)
from WorldModel.evaluation.phase_c_s1_hungarian_protocol import (
    LOAD_CONFIGS,
    PHASEC_CONFIG,
)
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


SCHEMA_VERSION = "phase_c_long_risk_head_repair_protocol_v1"
TRAINING_BUNDLE_SCHEMA_VERSION = (
    "phase_c_long_risk_head_repair_training_bundle_v1"
)
EVALUATION_BUNDLE_SCHEMA_VERSION = (
    "phase_c_long_risk_head_repair_evaluation_bundle_v1"
)
COLLECTION_SCHEMA_VERSION = "phase_c_long_risk_head_repair_collection_v1"
RUN_SCHEMA_VERSION = "phase_c_long_risk_head_repair_run_v1"
SUMMARY_SCHEMA_VERSION = "phase_c_long_risk_head_repair_summary_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
OUTPUT_ROOT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_longrisk_head_repair_681_700_v1"
)

SOURCE_CHECKPOINT = (
    BASE_ROOT / "model_round1_v1/best_regret_world_model.pt"
)
PHASE_B_STAGE1_CHECKPOINT = Path(
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
    "stage1_world_model.pt"
)
SOURCE_PSI_HEAD = (
    BASE_ROOT
    / "station_congestion_head_region_dev_511_520_v1"
    / "linear_head_v1/best_station_congestion_head.pt"
)
PSI_SCALE_CONTRACT = (
    BASE_ROOT
    / "station_congestion_head_region_dev_511_520_v1"
    / "station_congestion_scale_contract.json"
)

TRAINING_ROOT = OUTPUT_ROOT / "training"
SNAPSHOT_ROOT = TRAINING_ROOT / "snapshots_681_690"
DATASET_ROOT = TRAINING_ROOT / "datasets_681_690"
LONG_RISK_ROOT = TRAINING_ROOT / "long_risk_w200_681_690"
FUSED_ROOT = TRAINING_ROOT / "fused_seed_split_v1"
MODEL_ROOT = TRAINING_ROOT / "long_risk_head_only_v1"
REPAIRED_CHECKPOINT = MODEL_ROOT / "best_long_risk_world_model.pt"
REBOUND_PSI_ROOT = TRAINING_ROOT / "station_head_rebound_v1"
REBOUND_PSI_HEAD = REBOUND_PSI_ROOT / "best_station_congestion_head.pt"
EVALUATION_ROOT = OUTPUT_ROOT / "paired_online_691_700_v1"

LOADS = ("low", "mid", "high")
TRAIN_SEEDS = tuple(range(681, 688))
VAL_SEEDS = (688, 689)
OFFLINE_TEST_SEEDS = (690,)
DATA_SEEDS = TRAIN_SEEDS + VAL_SEEDS + OFFLINE_TEST_SEEDS
ONLINE_TEST_SEEDS = tuple(range(691, 701))

TICKS = 1500
SNAPSHOT_INTERVAL = 20
SNAPSHOT_TOP_M = 10
MAX_CONTEXTS_PER_TICK = 2
ROLLOUT_HORIZON = 10
LONG_RISK_HORIZON = 200
LONG_RISK_TERMINAL_WINDOW = 50
LONG_RISK_CONTINUATION = "greedy"
LYAPUNOV_CONFIG = Path("Config/lyapunov_l1_config.json")
TOP_M = 10

ARM_LABELS = {
    "greedy": "GreedySequentialManhattan",
    "hungarian": "HungarianBatchManhattan",
    "source_phasec": "SourcePhaseCS0J0UntrainedLongRiskHead",
    "repaired_phasec": "RepairedPhaseCS0J0NonPerturbationControl",
    "source_combo_j1": "SourceUntrainedLongRiskHeadComboS1StaticJ1",
    "repaired_event_j1": "RepairedEventLogitS1StaticJ1",
    "repaired_combo_j0": "RepairedQuantileComboS1J0",
    "repaired_combo_j1": "RepairedQuantileComboS1StaticJ1",
}
ARM_KEYS = tuple(ARM_LABELS)

REPORT_METRICS = (
    "completed_orders",
    "completed_tasks",
    "avg_task_duration",
    "avg_excess_delay",
    "open_order_count",
    "pending_order_count",
    "stall_ratio_mean",
    "stall_ratio_max",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "congestion_events",
    "severe_events",
    "risk_rate_per_100",
    "assignment_time_ms_mean",
    "wall_time_s",
)

COMPARISONS = tuple(
    (f"repaired_combo_j1_minus_{baseline}", baseline, "repaired_combo_j1")
    for baseline in (
        "greedy",
        "hungarian",
        "source_phasec",
        "repaired_phasec",
        "source_combo_j1",
        "repaired_event_j1",
        "repaired_combo_j0",
    )
)


def canonical_sha256(payload: Mapping) -> str:
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combo_selector_config(signal: str = "combo") -> dict:
    if signal not in ("combo", "event_logit"):
        raise ValueError(signal)
    return {**selector_config(), "energy_drift_signal": signal}


def formal_protocol() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "replace the unsupervised LongRiskHead embedded in the formal "
            "Phase-C checkpoint with a W=200 supervised head while keeping "
            "every non-head tensor bitwise unchanged"
        ),
        "lineage": {
            "source_checkpoint": SOURCE_CHECKPOINT.as_posix(),
            "stage1_reference_checkpoint": (
                PHASE_B_STAGE1_CHECKPOINT.as_posix()
            ),
            "source_long_risk_status": (
                "random Stage-1 initialization preserved through formal "
                "Phase-C training; not a supervised long-risk predictor"
            ),
            "repaired_checkpoint": REPAIRED_CHECKPOINT.as_posix(),
            "trainable_prefixes": ["long_risk_head."],
            "required_unchanged_prefixes": [
                "state_encoder.",
                "demand_encoder.",
                "action_encoder.",
                "transition.",
                "node_decoder.",
                "system_decoder.",
                "station_decoder.",
                "cost_head.",
            ],
            "checkpoint_selection": (
                "minimum held-out validation combo top-1 regret, then maximum "
                "combo pairwise accuracy, then minimum validation loss"
            ),
        },
        "data": {
            "behavior_policy": "source Phase-C S0/J0, top_m=1",
            "behavior_checkpoint": SOURCE_CHECKPOINT.as_posix(),
            "loads": list(LOADS),
            "train_seeds": list(TRAIN_SEEDS),
            "validation_seeds": list(VAL_SEEDS),
            "offline_test_seeds": list(OFFLINE_TEST_SEEDS),
            "ticks": TICKS,
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "snapshot_top_m": SNAPSHOT_TOP_M,
            "max_contexts_per_tick": MAX_CONTEXTS_PER_TICK,
            "candidate_set": "stratified idle robots plus native NO_ASSIGN",
            "short_rollout_horizon": ROLLOUT_HORIZON,
            "long_risk_horizon": LONG_RISK_HORIZON,
            "terminal_window": LONG_RISK_TERMINAL_WINDOW,
            "long_risk_continuation_policy": LONG_RISK_CONTINUATION,
            "action_route_encoding": "canonical_bfs_service_cell_v1",
            "action_path_mode": 0,
            "path_planner": "PrioritizedPlanning",
        },
        "head_targets": {
            "output_order": long_risk_runtime_contract()["output_order"],
            "combo_weights": long_risk_runtime_contract()[
                "quantile_combo_weights"
            ],
            "losses": {
                "peak_q90": "pinball_tau_0.90",
                "peak_q95": "pinball_tau_0.95",
                "cvar_q90": "pinball_tau_0.90",
                "terminal_q90": "pinball_tau_0.90",
                "delta_group_q90": "pinball_tau_0.90",
                "event_logit": "focal_binary_cross_entropy",
            },
        },
        "j1_rebinding": {
            "source_head": SOURCE_PSI_HEAD.as_posix(),
            "scale_contract": PSI_SCALE_CONTRACT.as_posix(),
            "target_head": REBOUND_PSI_HEAD.as_posix(),
            "permitted_only_if_all_non_long_risk_tensors_are_bitwise_equal": True,
            "station_head_weights_retrained": False,
        },
        "online_test": {
            "loads": list(LOADS),
            "seeds": list(ONLINE_TEST_SEEDS),
            "ticks": TICKS,
            "paired_manifest_source": "greedy",
            "arms": dict(ARM_LABELS),
            "comparisons": [
                {"name": name, "baseline": left, "candidate": right}
                for name, left, right in COMPARISONS
            ],
            "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
            "source_vs_repaired_s0_nonperturbation_control": True,
        },
        "runtime": {
            "phasec_config": dict(PHASEC_CONFIG),
            "combo_s1_config": combo_selector_config("combo"),
            "event_s1_config": combo_selector_config("event_logit"),
            "long_risk_contract": long_risk_runtime_contract(),
        },
        "reported_metrics": list(REPORT_METRICS),
        "forbidden": {
            "joint_encoder_or_transition_finetuning": True,
            "cost_head_change": True,
            "training_on_online_test_seeds": True,
            "post_run_signal_tuning": True,
            "event_logit_described_as_quantile_combo": True,
            "committed_or_eta_station_admission": True,
            "overwrite_existing_phase_c_or_pibt_artifacts": True,
        },
    }


def protocol_sha256() -> str:
    return canonical_sha256(formal_protocol())


__all__ = [
    "ARM_KEYS",
    "ARM_LABELS",
    "COLLECTION_SCHEMA_VERSION",
    "COMPARISONS",
    "DATASET_ROOT",
    "DATA_SEEDS",
    "EVALUATION_BUNDLE_SCHEMA_VERSION",
    "EVALUATION_ROOT",
    "FUSED_ROOT",
    "LOADS",
    "LOAD_CONFIGS",
    "LONG_RISK_CONTINUATION",
    "LONG_RISK_HORIZON",
    "LONG_RISK_ROOT",
    "LONG_RISK_TERMINAL_WINDOW",
    "LYAPUNOV_CONFIG",
    "MAX_CONTEXTS_PER_TICK",
    "MODEL_ROOT",
    "OFFLINE_TEST_SEEDS",
    "ONLINE_TEST_SEEDS",
    "OUTPUT_ROOT",
    "PHASE_B_STAGE1_CHECKPOINT",
    "PHASEC_CONFIG",
    "PSI_SCALE_CONTRACT",
    "REBOUND_PSI_HEAD",
    "REBOUND_PSI_ROOT",
    "REPAIRED_CHECKPOINT",
    "REPORT_METRICS",
    "ROLLOUT_HORIZON",
    "RUN_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SNAPSHOT_INTERVAL",
    "SNAPSHOT_ROOT",
    "SNAPSHOT_TOP_M",
    "SOURCE_CHECKPOINT",
    "SOURCE_PSI_HEAD",
    "SUMMARY_SCHEMA_VERSION",
    "TICKS",
    "TOP_M",
    "TRAINING_BUNDLE_SCHEMA_VERSION",
    "TRAINING_ROOT",
    "TRAIN_SEEDS",
    "VAL_SEEDS",
    "canonical_sha256",
    "combo_selector_config",
    "formal_protocol",
    "protocol_sha256",
    "sha256_file",
]
