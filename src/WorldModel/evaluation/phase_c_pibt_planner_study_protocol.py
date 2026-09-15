"""Frozen design for the Phase-C PP-to-PIBT planner study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from WorldModel.evaluation.phase_c_combo_s1_j1_paper_protocol import (
    selector_config,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    PSI_HEAD_CHECKPOINT,
    PSI_SCALE_CONTRACT,
)
from WorldModel.evaluation.phase_c_s1_hungarian_protocol import PHASEC_CONFIG
from WorldModel.evaluation.phase_c_s1_long_risk_correction_protocol import (
    CANDIDATE_CHECKPOINT as PP_TRAINED_CHECKPOINT,
    LOAD_CONFIGS,
    REPORT_METRICS,
    TOP_M,
)
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


SCHEMA_VERSION = "phase_c_pibt_planner_study_protocol_v2"
BUNDLE_SCHEMA_VERSION = "phase_c_pibt_planner_study_bundle_v2"
RUN_SCHEMA_VERSION = "phase_c_pibt_planner_study_run_v2"
SUMMARY_SCHEMA_VERSION = "phase_c_pibt_planner_study_summary_v2"
TRAINING_AUDIT_SCHEMA_VERSION = "phase_c_pibt_training_audit_v2"
MANIFEST_AUDIT_SCHEMA_VERSION = "phase_c_pibt_manifest_audit_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
PIBT_TRAIN_ROOT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_pibt_round1_v2"
)
PIBT_TRAINED_CHECKPOINT = (
    PIBT_TRAIN_ROOT / "model_round1_v1/best_regret_world_model.pt"
)
PIBT_PSI_HEAD_ROOT = (
    PIBT_TRAIN_ROOT / "station_congestion_head_region_pibt_461_470_v1"
)
PIBT_PSI_HEAD_CHECKPOINT = (
    PIBT_PSI_HEAD_ROOT / "linear_head_v1/best_station_congestion_head.pt"
)
PIBT_PSI_SCALE_CONTRACT = (
    PIBT_PSI_HEAD_ROOT / "station_congestion_scale_contract.json"
)
PHASE_B_STAGE1_CHECKPOINT = Path(
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
    "stage1_world_model.pt"
)

ZERO_SHOT_OUTPUT_ROOT = (
    BASE_ROOT / "phasec_pibt_cross_planner_ppweights_551_560_v2"
)
ADAPTED_OUTPUT_ROOT = (
    BASE_ROOT / "phasec_pibt_retrained_weights_551_560_v2"
)

HIGH_MANIFEST_ROOT = (
    BASE_ROOT / "psi_dispatch_dynamic_eta_overbooking_stress_source_551_560_v1"
)
LOW_MID_MANIFEST_ROOT = (
    BASE_ROOT / "station_admission_sj_factorial_low_mid_source_551_560_v1"
)

LOADS = ("low", "mid", "high")
EVAL_SEEDS = tuple(range(551, 561))
TRAIN_SEEDS = tuple(range(461, 471))
TRAIN_SPLIT = {
    "train": tuple(range(461, 468)),
    "val": (468, 469),
    "test": (470,),
}
TICKS = 1500

PLANNER_NAME = "PIBTPlanner"
PLANNER_PARAMS = {"strict_validation": True}
ACTION_PATH_MODE = 0
ACTION_ROUTE_ENCODING = "canonical_bfs_service_cell_v1"
LONG_RISK_HORIZON = 200
LONG_RISK_TERMINAL_WINDOW = 50

ZERO_SHOT_ARMS = (
    "greedy_pibt",
    "hungarian_pibt",
    "phasec_ppwm_pibt",
    "combo_ppwm_pibt",
)
ADAPTED_NEW_ARMS = (
    "phasec_pibtwm_pibt",
    "combo_pibtwm_pibt",
)
ALL_ARMS = ZERO_SHOT_ARMS + ADAPTED_NEW_ARMS

ARM_LABELS = {
    "greedy_pibt": "Greedy+PIBT",
    "hungarian_pibt": "Hungarian+PIBT",
    "phasec_ppwm_pibt": "PhaseC(PP-trained WM)+PIBT",
    "combo_ppwm_pibt": "Combo-S1+J1(PP-trained WM)+PIBT",
    "phasec_pibtwm_pibt": "PhaseC(PIBT-trained WM)+PIBT",
    "combo_pibtwm_pibt": "Combo-S1+J1(PIBT-trained WM)+PIBT",
}

MODEL_ARMS = {
    "phasec_ppwm_pibt",
    "combo_ppwm_pibt",
    "phasec_pibtwm_pibt",
    "combo_pibtwm_pibt",
}
COMBO_ARMS = {"combo_ppwm_pibt", "combo_pibtwm_pibt"}

SUMMARY_METRICS = tuple(REPORT_METRICS) + (
    "path_planner_engine_deadlock_event_ticks",
    "path_planner_engine_vertex_conflicts",
    "path_planner_engine_swap_conflicts",
    "path_planner_engine_avg_plan_ms",
    "pibt_batch_calls",
    "pibt_single_calls",
    "pibt_planned_agents",
    "pibt_priority_inheritance_calls",
    "pibt_backtracks",
    "pibt_validation_failures",
    "pibt_max_batch_size",
    "pibt_wait_decision_ratio",
    "pibt_priority_inheritance_per_agent",
    "pibt_backtracks_per_agent",
)

PRIMARY_CONTRAST_METRICS = (
    "completed_orders",
    "avg_task_duration",
    "avg_excess_delay",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "congestion_events",
    "severe_events",
    "pibt_wait_decision_ratio",
)

ZERO_SHOT_COMPARISONS = (
    ("combo_ppwm_minus_greedy", "greedy_pibt", "combo_ppwm_pibt"),
    (
        "combo_ppwm_minus_hungarian",
        "hungarian_pibt",
        "combo_ppwm_pibt",
    ),
    ("combo_ppwm_minus_phasec", "phasec_ppwm_pibt", "combo_ppwm_pibt"),
    ("phasec_ppwm_minus_greedy", "greedy_pibt", "phasec_ppwm_pibt"),
)

ADAPTED_COMPARISONS = (
    (
        "phasec_pibtwm_minus_phasec_ppwm",
        "phasec_ppwm_pibt",
        "phasec_pibtwm_pibt",
    ),
    (
        "combo_pibtwm_minus_combo_ppwm",
        "combo_ppwm_pibt",
        "combo_pibtwm_pibt",
    ),
    (
        "combo_pibtwm_minus_greedy",
        "greedy_pibt",
        "combo_pibtwm_pibt",
    ),
    (
        "combo_pibtwm_minus_hungarian",
        "hungarian_pibt",
        "combo_pibtwm_pibt",
    ),
    (
        "combo_pibtwm_minus_phasec_pibtwm",
        "phasec_pibtwm_pibt",
        "combo_pibtwm_pibt",
    ),
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_root(load: str) -> Path:
    if load == "high":
        return HIGH_MANIFEST_ROOT
    if load in ("low", "mid"):
        return LOW_MID_MANIFEST_ROOT
    raise ValueError(load)


def checkpoint_for_arm(arm: str) -> Path | None:
    if arm in ("greedy_pibt", "hungarian_pibt"):
        return None
    if arm in ("phasec_ppwm_pibt", "combo_ppwm_pibt"):
        return Path(PP_TRAINED_CHECKPOINT)
    if arm in ("phasec_pibtwm_pibt", "combo_pibtwm_pibt"):
        return PIBT_TRAINED_CHECKPOINT
    raise ValueError(arm)


def psi_artifacts_for_arm(arm: str) -> tuple[Path, Path] | None:
    if arm not in COMBO_ARMS:
        return None
    if arm == "combo_ppwm_pibt":
        return Path(PSI_HEAD_CHECKPOINT), Path(PSI_SCALE_CONTRACT)
    if arm == "combo_pibtwm_pibt":
        return PIBT_PSI_HEAD_CHECKPOINT, PIBT_PSI_SCALE_CONTRACT
    raise ValueError(arm)


def evaluation_protocol(campaign: str, checkpoint: Path) -> dict[str, Any]:
    if campaign not in ("zero_shot", "adapted"):
        raise ValueError(campaign)
    arms = ZERO_SHOT_ARMS if campaign == "zero_shot" else ADAPTED_NEW_ARMS
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": campaign,
        "purpose": (
            "separate zero-shot PP-trained World-Model transfer to PIBT from "
            "PIBT-on-policy World-Model fine-tuning"
        ),
        "planner": {
            "name": PLANNER_NAME,
            "params": dict(PLANNER_PARAMS),
            "joint_one_step_batch_required": True,
            "nonrequested_robots_pinned": True,
            "queue_entry_goal_is_authoritative": True,
            "runtime_invariant_validation": True,
        },
        "action_route_encoding": {
            "action_path_mode": ACTION_PATH_MODE,
            "encoding": ACTION_ROUTE_ENCODING,
            "planner_preview_used": False,
            "purpose": (
                "keep candidate action geometry planner-independent while "
                "the state trajectory, rollout targets, and realised labels "
                "remain planner-specific"
            ),
        },
        "loads": list(LOADS),
        "seeds": list(EVAL_SEEDS),
        "ticks": TICKS,
        "manifests": {
            "high": HIGH_MANIFEST_ROOT.as_posix(),
            "low_mid": LOW_MID_MANIFEST_ROOT.as_posix(),
            "pairing": "exact replay within every load/seed cell",
            "missing_source_generation": (
                "Greedy + default PP + physical-only, used only to record the "
                "order stream; every formal arm then replays the frozen file"
            ),
        },
        "station_admission": {
            "mode": STATION_ADMISSION_PHYSICAL_ONLY,
            "physical_occupancy_cap": True,
            "in_transit_committed_cap": False,
        },
        "arms": {arm: ARM_LABELS[arm] for arm in arms},
        "new_simulation_count": len(arms) * len(LOADS) * len(EVAL_SEEDS),
        "world_model_checkpoint": checkpoint.as_posix(),
        "world_model_checkpoint_sha256": sha256_file(checkpoint),
        "phasec_robot_selector": dict(PHASEC_CONFIG),
        "combo_robot_selector": selector_config(),
        "combo_context_scheduler": {
            "mode": "static_j1_ascending",
            "psi_head": (
                Path(PSI_HEAD_CHECKPOINT).as_posix()
                if campaign == "zero_shot"
                else PIBT_PSI_HEAD_CHECKPOINT.as_posix()
            ),
            "psi_scale_contract": (
                Path(PSI_SCALE_CONTRACT).as_posix()
                if campaign == "zero_shot"
                else PIBT_PSI_SCALE_CONTRACT.as_posix()
            ),
            "encoder_binding": (
                "PP-trained J1 is a declared zero-shot transfer condition"
                if campaign == "zero_shot"
                else "PIBT J1 head is trained on and SHA-bound to the PIBT encoder"
            ),
        },
        "metric_semantics": {
            "completed_orders": "unchanged order lifecycle definition",
            "task_duration": "unchanged created_at-to-completed_at definition",
            "deadlock": (
                "active non-waiting robot stationary for >=10 ticks; deliberate "
                "PIBT wait actions are not silently excluded"
            ),
            "stuck_this_tick": (
                "unchanged: includes an explicit path-planner wait action"
            ),
            "plan_failed_streak": (
                "unchanged: only empty/failed planning; PIBT valid waits do not "
                "masquerade as path failures"
            ),
            "planner_diagnostics": (
                "additional audit-only fields; no existing metric is redefined"
            ),
        },
        "reported_metrics": list(SUMMARY_METRICS),
        "forbidden": {
            "planner_specific_metric_redefinition": True,
            "mixed_order_manifests": True,
            "committed_or_eta_station_admission": True,
            "post_run_parameter_tuning": True,
            "seed_or_load_dropping": True,
        },
    }


def training_protocol() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": "pibt_on_policy_round1",
        "purpose": (
            "construct the matched planner-specific World-Model checkpoint "
            "using the original Phase-C round-1 data/training recipe"
        ),
        "planner": {"name": PLANNER_NAME, "params": dict(PLANNER_PARAMS)},
        "behavior_checkpoint": PHASE_B_STAGE1_CHECKPOINT.as_posix(),
        "behavior_checkpoint_sha256": sha256_file(PHASE_B_STAGE1_CHECKPOINT),
        "loads": list(LOADS),
        "configs": dict(LOAD_CONFIGS),
        "seeds": list(TRAIN_SEEDS),
        "split": {key: list(values) for key, values in TRAIN_SPLIT.items()},
        "ticks": TICKS,
        "snapshot_run_count": len(LOADS) * len(TRAIN_SEEDS),
        "isolated_replay_run_count": len(LOADS) * len(TRAIN_SEEDS),
        "snapshot_interval": 20,
        "snapshot_top_m": 10,
        "max_contexts_per_tick": 2,
        "candidate_robot_mode": "stratified",
        "rollout_horizon": 10,
        "candidate_rollout": {
            "continuation_mode": "isolated",
            "future_unknown_orders": False,
            "continuation_scheduler": False,
            "path_planner": PLANNER_NAME,
        },
        "long_risk_labels": {
            "required": True,
            "horizon": LONG_RISK_HORIZON,
            "terminal_window": LONG_RISK_TERMINAL_WINDOW,
            "continuation_policy": "greedy_assignment_with_snapshot_PIBT_planner",
            "all_candidates_including_no_assign": True,
        },
        "action_route_encoding": {
            "action_path_mode": ACTION_PATH_MODE,
            "encoding": ACTION_ROUTE_ENCODING,
            "planner_preview_used": False,
        },
        "training": {
            "stage1_checkpoint": PHASE_B_STAGE1_CHECKPOINT.as_posix(),
            "skip_stage1": True,
            "stage2_horizon": 10,
            "stage2_epochs": 30,
            "stage2_lr": 1e-4,
            "stage2_alpha_rank": 0.1,
            "stage2_freeze_epochs": 0,
            "stage2_unfreeze_mode": "all",
            "early_stopping_patience": 8,
            "early_stopping_monitor": "val_top1_regret_mean",
            "balanced_sampler": True,
            "pair_balance_by": "source_load_level",
        },
        "station_context_head": {
            "planner_specific": True,
            "source": "PIBT on-policy decision snapshots",
            "encoder_sha_bound": True,
            "target_definition": "unchanged static-J1 service/traffic targets",
            "scale_contract": Path(PSI_SCALE_CONTRACT).as_posix(),
            "scale_contract_file_sha256": sha256_file(PSI_SCALE_CONTRACT),
            "scale_contract_refit": False,
        },
        "interpretation_limit": (
            "the state encoder, transition model, LongRiskHead, and static-J1 "
            "station head are planner-specific; candidate action geometry is "
            "kept on the frozen canonical BFS service-cell encoding"
        ),
    }


__all__ = [name for name in globals() if name.isupper()] + [
    "canonical_sha256",
    "checkpoint_for_arm",
    "evaluation_protocol",
    "manifest_root",
    "psi_artifacts_for_arm",
    "sha256_file",
    "training_protocol",
]
