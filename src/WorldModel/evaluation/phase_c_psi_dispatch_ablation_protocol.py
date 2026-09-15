"""Frozen development protocol for the service/traffic/debt context layer.

This is an opt-in follow-up to the frozen S1 policy.  It does not modify the
World Model, ``e_demand`` schema, S1 robot scorer, or any earlier seed block.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from WorldModel.evaluation.phase_c_s1_hungarian_protocol import S1_CONFIG


SCHEMA_VERSION = "phase_c_psi_dispatch_ablation_protocol_v1"
FROZEN_BUNDLE_SCHEMA_VERSION = "phase_c_psi_dispatch_bundle_v1"
PER_ARM_SCHEMA_VERSION = "phase_c_psi_dispatch_arm_v1"
REPORT_SCHEMA_VERSION = "phase_c_psi_dispatch_report_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
OUTPUT_ROOT = Path(os.environ.get(
    "PHASEC_PSI_DISPATCH_OUT",
    str(BASE_ROOT / "psi_dispatch_ablation_551_560_v1"),
))
SOURCE_POLICY_ROOT = BASE_ROOT / "phasec_s1_hungarian_cert_491_500_v1"
SOURCE_POLICY_BUNDLE = SOURCE_POLICY_ROOT / (
    "phase_c_s1_hungarian_frozen_protocol.json"
)
MODEL_CHECKPOINT = BASE_ROOT / "model_round1_v1/best_regret_world_model.pt"
PSI_HEAD_ROOT = BASE_ROOT / "station_congestion_head_region_dev_511_520_v1"
PSI_HEAD_CHECKPOINT = PSI_HEAD_ROOT / (
    "linear_head_v1/best_station_congestion_head.pt"
)
PSI_SCALE_CONTRACT = PSI_HEAD_ROOT / "station_congestion_scale_contract.json"

LOADS = ("low", "mid", "high")
LOAD_CONFIGS = {
    "low": Path("Config/world_model_config_PP_48_low.json"),
    "mid": Path("Config/world_model_config_PP_48_mid.json"),
    "high": Path("Config/world_model_config_PP_48_high.json"),
}
SEEDS = tuple(range(551, 561))
LOCKED_REFERENCE_SEEDS = tuple(range(501, 511))
PHI_VALIDATION_SEEDS = tuple(range(531, 541))
OLD_PHI_DEV_SEEDS = tuple(range(541, 551))
TICKS = 1500
TOP_M = 10
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260809

GREEDY_ARM = "greedy_manifest"
S1_ARM = "s1"
PSI_SHADOW_ARM = "s1_psi_shadow"
PSI_ARM = "s1_psi_dispatch"
ARM_KEYS = (GREEDY_ARM, S1_ARM, PSI_SHADOW_ARM, PSI_ARM)
ARM_LABELS = {
    GREEDY_ARM: "GreedyManifestSource",
    S1_ARM: "PhaseCS1Baseline",
    PSI_SHADOW_ARM: "PhaseCS1PsiDispatchShadow",
    PSI_ARM: "PhaseCS1PsiDispatch",
}


SOURCE_FILES = (
    "Policies/TaskAssigner/__init__.py",
    "Policies/TaskAssigner/base_task_assigner.py",
    "Policies/TaskAssigner/GreedyTaskAssigner/greedy_task_assigner.py",
    "Policies/TaskAssigner/WorldModelTaskAssigner/__init__.py",
    "Policies/TaskAssigner/WorldModelTaskAssigner/world_model_task_assigner.py",
    "Policies/TaskAssigner/WorldModelTaskAssigner/psi_dispatch_context_assigner.py",
    "Policies/TaskAssigner/context_assignment.py",
    "WorldModel/__init__.py",
    "WorldModel/core/psi_dispatch.py",
    "WorldModel/core/station_congestion_head.py",
    "WorldModel/evaluation/station_congestion_endpoint.py",
    "WorldModel/evaluation/evaluate_online_v6.py",
    "WorldModel/evaluation/phase_c_s1_hungarian_protocol.py",
    "WorldModel/evaluation/phase_c_psi_dispatch_ablation_protocol.py",
    "WorldModel/evaluation/freeze_phase_c_psi_dispatch_ablation.py",
    "WorldModel/evaluation/run_phase_c_psi_dispatch_arm.py",
    "WorldModel/evaluation/validate_phase_c_psi_dispatch.py",
    "WorldModel/evaluation/run_phase_c_psi_dispatch_cpu.slurm",
    "WorldModel/core/model.py",
    "WorldModel/graph/graph_builder.py",
    "Engine/simulation_engine.py",
)


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


def run_id(arm: str, load: str, seed: int) -> str:
    if arm not in ARM_KEYS:
        raise ValueError(f"unknown psi-dispatch arm: {arm}")
    if load not in LOADS:
        raise ValueError(f"unknown load: {load}")
    if int(seed) not in SEEDS:
        raise ValueError(f"seed is outside development block: {seed}")
    return f"{arm}_{load}_seed{int(seed)}"


def protocol_payload(artifact_hashes: Mapping[str, str]) -> dict[str, Any]:
    forbidden_overlap = (
        set(SEEDS) & set(LOCKED_REFERENCE_SEEDS)
        or set(SEEDS) & set(PHI_VALIDATION_SEEDS)
        or set(SEEDS) & set(OLD_PHI_DEV_SEEDS)
    )
    if forbidden_overlap:
        raise RuntimeError(f"psi-dispatch seeds overlap frozen blocks: {forbidden_overlap}")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "causal development test of a station service/traffic plus "
            "exact pending service-debt cross-context ordering layer"
        ),
        "frozen_inputs": {
            "source_policy_bundle": SOURCE_POLICY_BUNDLE.as_posix(),
            "source_policy_bundle_sha256": artifact_hashes["source_policy_bundle"],
            "model_checkpoint": MODEL_CHECKPOINT.as_posix(),
            "model_checkpoint_sha256": artifact_hashes["model_checkpoint"],
            "psi_head_checkpoint": PSI_HEAD_CHECKPOINT.as_posix(),
            "psi_head_checkpoint_sha256": artifact_hashes["psi_head_checkpoint"],
            "psi_scale_contract": PSI_SCALE_CONTRACT.as_posix(),
            "psi_scale_contract_sha256": artifact_hashes["psi_scale_contract"],
            "load_configs": {
                load: {
                    "path": path.as_posix(),
                    "sha256": artifact_hashes[f"config_{load}"],
                }
                for load, path in LOAD_CONFIGS.items()
            },
        },
        "development": {
            "loads": list(LOADS),
            "seeds": list(SEEDS),
            "ticks": TICKS,
            "top_m": TOP_M,
            "arms": [ARM_LABELS[key] for key in ARM_KEYS],
            "manifest_source": ARM_LABELS[GREEDY_ARM],
            "exact_manifest_replay": True,
            "one_manifest_per_load_seed": True,
        },
        "psi_contract": {
            "station_tensor": "S x 2",
            "station_channels": ["service", "traffic"],
            "context_tensor": "C x 3",
            "context_channels": ["service", "traffic", "service_debt"],
            "service": (
                "frozen station head service channel: normalized assigned "
                "capacity and in-progress pressure"
            ),
            "traffic": (
                "frozen station head traffic channel: stationary/density/"
                "bottleneck traffic around each station"
            ),
            "service_debt": {
                "station_work": (
                    "pending unresolved order/pod chains, independent of "
                    "idle count and proposal prefix"
                ),
                "context_urgency": "age/(age+free_flow_chain_time)",
                "backlog_transform": (
                    "x/(1+x), x=chains/station_queue_capacity; "
                    "fallback=agents/stations"
                ),
                "combination": "0.5*backlog_score + 0.5*age_score",
            },
            "dispatch_cost": (
                "J(c) = 0.5*service[s(c)] + 0.5*traffic[s(c)] "
                "- service_debt[c]; lower is better"
            ),
            "context_order": (
                "ascending J; service_debt, age, then original order are "
                "deterministic tie-breaks"
            ),
            "within_context_robot_scorer": (
                "unchanged certified S1 WorldModelTaskAssigner scorer"
            ),
        },
        "policy_contract": {
            ARM_LABELS[S1_ARM]: {
                "checkpoint": "frozen Phase-C best_regret",
                "s1_config": dict(S1_CONFIG),
                "context_order": "legacy fixed-context prefix",
            },
            ARM_LABELS[PSI_SHADOW_ARM]: {
                "checkpoint": "same frozen Phase-C best_regret",
                "mode": "shadow",
                "j_is_evaluated": True,
                "execution_order_changed": False,
            },
            ARM_LABELS[PSI_ARM]: {
                "checkpoint": "same frozen Phase-C best_regret",
                "mode": "j_ascending",
                "all_dispatchable_contexts_seen": True,
                "execution_order_changed": True,
            },
        },
        "analysis": {
            "primary_contrast": f"{ARM_LABELS[PSI_ARM]}-minus-{ARM_LABELS[S1_ARM]}",
            "shadow_contrast": f"{ARM_LABELS[PSI_SHADOW_ARM]}-minus-{ARM_LABELS[S1_ARM]}",
            "bootstrap_unit": "seed clustered across loads",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "mechanism_checks": {
                "head_loaded_every_run": True,
                "j_evaluated_every_model_run": True,
                "service_debt_finite": True,
                "parent_robot_scorer_preserved": True,
                "no_assign_added": False,
                "hard_gate_added": False,
                "e_demand_modified": False,
                "attention_added": False,
            },
            "reported_metrics": [
                "completed_orders",
                "completed_tasks",
                "avg_task_duration",
                "avg_excess_delay",
                "open_order_count",
                "pending_order_count",
                "deadlock_ratio_mean",
                "deadlock_ratio_max",
                "stall_ratio_mean",
                "stall_ratio_max",
                "assignment_time_ms_mean",
                "wall_time_s",
            ],
        },
        "forbidden": {
            "seeds_501_510": True,
            "seeds_531_540": True,
            "seeds_541_550": True,
            "world_model_retraining": True,
            "e_demand_schema_change": True,
            "s1_retuning": True,
            "psi_pre_online_use": True,
            "delta_psi": True,
            "native_no_assign": True,
            "lyapunov_or_dispatch_gate": True,
            "handcrafted_station_injection": True,
            "post_outcome_weight_tuning": True,
            "overwrite_existing_outputs": True,
        },
    }
    return {**payload, "protocol_sha256": canonical_sha256(payload)}


__all__ = [
    "ARM_KEYS",
    "ARM_LABELS",
    "BOOTSTRAP_REPEATS",
    "BOOTSTRAP_SEED",
    "FROZEN_BUNDLE_SCHEMA_VERSION",
    "GREEDY_ARM",
    "LOADS",
    "LOAD_CONFIGS",
    "MODEL_CHECKPOINT",
    "OLD_PHI_DEV_SEEDS",
    "PER_ARM_SCHEMA_VERSION",
    "PHI_VALIDATION_SEEDS",
    "PSI_ARM",
    "PSI_HEAD_CHECKPOINT",
    "PSI_SCALE_CONTRACT",
    "PSI_SHADOW_ARM",
    "REPORT_SCHEMA_VERSION",
    "S1_ARM",
    "SCHEMA_VERSION",
    "SEEDS",
    "SOURCE_FILES",
    "SOURCE_POLICY_BUNDLE",
    "TICKS",
    "TOP_M",
    "canonical_sha256",
    "protocol_payload",
    "run_id",
    "sha256_file",
]
