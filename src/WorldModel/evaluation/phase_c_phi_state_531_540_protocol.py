"""Frozen protocol for fresh current-state ``phi_state`` validation.

This protocol is deliberately separate from the earlier 511--520 feature
selection experiment.  It evaluates the already-trained, station-conditioned
region head on a fresh seed block (531--540) using contemporaneous simulator
state only.  It does not train a head, alter S1/the assigner, use endpoint
labels, or compute ``DeltaPsi``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "phase_c_phi_state_validation_protocol_v1"
FROZEN_BUNDLE_SCHEMA_VERSION = "phase_c_phi_state_validation_bundle_v1"
RUN_SCHEMA_VERSION = "phase_c_phi_state_validation_run_v1"
REPORT_SCHEMA_VERSION = "phase_c_phi_state_validation_report_v1"
TRACE_SCHEMA_VERSION = "phase_c_station_congestion_trace_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
SOURCE_ROOT = Path(os.environ.get(
    "PHASEC_PHI_STATE_SOURCE_ROOT",
    str(BASE_ROOT / "phasec_s1_hungarian_cert_491_500_v1"),
))
SOURCE_BUNDLE = Path(os.environ.get(
    "PHASEC_PHI_STATE_SOURCE_BUNDLE",
    str(SOURCE_ROOT / "phase_c_s1_hungarian_frozen_protocol.json"),
))
HEAD_ROOT = Path(os.environ.get(
    "PHASEC_PHI_STATE_HEAD_ROOT",
    str(BASE_ROOT / "station_congestion_head_region_dev_511_520_v1"),
))
HEAD_CHECKPOINT = Path(os.environ.get(
    "PHASEC_PHI_STATE_HEAD_CHECKPOINT",
    str(HEAD_ROOT / "linear_head_v1" / "best_station_congestion_head.pt"),
))
SCALE_CONTRACT = Path(os.environ.get(
    "PHASEC_PHI_STATE_SCALE_CONTRACT",
    str(HEAD_ROOT / "station_congestion_scale_contract.json"),
))
OUTPUT_ROOT = Path(os.environ.get(
    "PHASEC_PHI_STATE_OUT",
    str(BASE_ROOT / "phi_state_validate_531_540_v2"),
))

LOADS = ("low", "mid", "high")
SEEDS = tuple(range(531, 541))
LOCKED_REFERENCE_SEEDS = tuple(range(501, 511))
TRAINING_SEEDS = tuple(range(511, 517))
ARMS = ("greedy", "hungarian", "phasec", "phasec_s1")
TICKS = 1500
FRAME_STRIDE = 5
REGION_HOPS = (1, 3, 5)
PRIMARY_REGION_HOPS = 3
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20260807

SOURCE_FILES = (
    "WorldModel/evaluation/phase_c_phi_state_531_540_protocol.py",
    "WorldModel/evaluation/freeze_phase_c_phi_state_531_540.py",
    "WorldModel/evaluation/run_phase_c_phi_state_531_540_arm.py",
    "WorldModel/evaluation/validate_phase_c_phi_state_531_540.py",
    "WorldModel/evaluation/run_phase_c_phi_state_531_540_cpu.slurm",
    "WorldModel/evaluation/station_congestion_trace_probe.py",
    "WorldModel/evaluation/td_stream_probe.py",
    "WorldModel/evaluation/evaluate_online_v6.py",
    "WorldModel/evaluation/run_phase_c_s1_hungarian_arm.py",
    "WorldModel/evaluation/phase_c_s1_hungarian_protocol.py",
    "WorldModel/evaluation/phase_c_station_congestion_correlation_protocol.py",
    "WorldModel/data/build_station_congestion_head_dataset.py",
    "WorldModel/core/station_congestion_head.py",
    "WorldModel/core/model.py",
    "WorldModel/graph/graph_builder.py",
    "Policies/TaskAssigner/context_assignment.py",
    "Policies/TaskAssigner/WorldModelTaskAssigner/world_model_task_assigner.py",
    "Policies/TaskAssigner/HungarianTaskAssigner/hungarian_task_assigner.py",
    "Policies/TaskAssigner/GreedyTaskAssigner/greedy_task_assigner.py",
    "Engine/simulation_engine.py",
)


def canonical_sha256(payload: Mapping[str, Any]) -> str:
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


def run_id(arm: str, load: str, seed: int) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if load not in LOADS:
        raise ValueError(f"unknown load: {load}")
    if int(seed) not in SEEDS:
        raise ValueError(f"seed is outside fresh validation block: {seed}")
    return f"{arm}_{load}_seed{int(seed)}"


def protocol_payload(
    source_bundle_sha256: str,
    head_checkpoint_sha256: str,
    scale_contract_sha256: str,
) -> dict[str, Any]:
    if set(SEEDS) & set(LOCKED_REFERENCE_SEEDS):
        raise RuntimeError("fresh validation seeds overlap locked reference")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "validate the frozen station-conditioned phi_state head on a "
            "fresh closed-loop seed block using same-tick physical labels"
        ),
        "frozen_inputs": {
            "source_policy_bundle": SOURCE_BUNDLE.as_posix(),
            "source_policy_bundle_sha256": str(source_bundle_sha256),
            "head_checkpoint": HEAD_CHECKPOINT.as_posix(),
            "head_checkpoint_sha256": str(head_checkpoint_sha256),
            "scale_contract": SCALE_CONTRACT.as_posix(),
            "scale_contract_sha256": str(scale_contract_sha256),
            "scale_fit_seeds": list(TRAINING_SEEDS),
            "scale_refit_on_validation": False,
        },
        "development": {
            "loads": list(LOADS),
            "seeds": list(SEEDS),
            "ticks": TICKS,
            "frame_stride": FRAME_STRIDE,
            "arms": list(ARMS),
            "manifest_source": "greedy",
            "exact_manifest_replay": True,
        },
        "representation": {
            "kind": "station_region_mean_max",
            "region_hops": list(REGION_HOPS),
            "primary_region_hops": PRIMARY_REGION_HOPS,
            "station_node_ids_required": True,
            "variable_station_count_supported": True,
        },
        "target_semantics": {
            "traffic": [
                "stationary_ticks_max",
                "node_density_cvar90",
                "bottleneck_density_excess",
            ],
            "service": [
                "assigned_agent_capacity_ratio",
                "in_progress_pressure",
            ],
            "same_tick_only": True,
            "endpoint_or_future_target": False,
            "delta_psi": False,
            "blocked_max": "diagnostic_only_not_head_target",
            "cross_channel_cancellation": False,
            "channel_roles": {
                "service": "primary_station_pressure_for_cross_context_use",
                "traffic": "auxiliary_diagnostic_future_risk_context",
            },
        },
        "analysis": {
            "pooled_and_per_run_spearman": True,
            "bootstrap_unit": "run_id",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "within_tick_station_ranking": True,
            "load_and_arm_breakdown": True,
            "report_mae_rmse_bias_prediction_spread": True,
            "reference_comparison": "511_520_state_level_validation_test",
            "primary_service_bars": {
                "pooled_spearman_ge": 0.80,
                "per_run_cluster_ci95_lower_ge": 0.70,
                "within_tick_station_spearman_mean_ge": 0.60,
                "within_tick_positive_fraction_ge": 0.80,
            },
            "traffic_channel_is_hard_gate": False,
        },
        "forbidden": {
            "world_model_retraining": True,
            "online_assigner_change": True,
            "s1_change": True,
            "delta_psi_online_use": True,
            "psi_pre_as_primary_target": True,
            "future_label_as_phi_state": True,
            "locked_501_510_outcomes": True,
            "overwrite_existing_outputs": True,
        },
    }
    return {**payload, "protocol_sha256": canonical_sha256(payload)}


__all__ = [
    "ARMS",
    "BOOTSTRAP_REPEATS",
    "BOOTSTRAP_SEED",
    "FRAME_STRIDE",
    "FROZEN_BUNDLE_SCHEMA_VERSION",
    "HEAD_CHECKPOINT",
    "HEAD_ROOT",
    "LOADS",
    "LOCKED_REFERENCE_SEEDS",
    "OUTPUT_ROOT",
    "PRIMARY_REGION_HOPS",
    "REGION_HOPS",
    "REPORT_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "SCALE_CONTRACT",
    "SCHEMA_VERSION",
    "SEEDS",
    "SOURCE_BUNDLE",
    "SOURCE_FILES",
    "SOURCE_ROOT",
    "TICKS",
    "TRAINING_SEEDS",
    "TRACE_SCHEMA_VERSION",
    "canonical_sha256",
    "protocol_payload",
    "run_id",
    "sha256_file",
]
