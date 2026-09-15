"""Development protocol for the first online ``phi_state`` causal ablation.

The experiment is deliberately narrower than a new Phase-C certification.  It
holds the frozen Round-1 World Model and S1 conversion fixed and changes only
the order in which competing fixed contexts consume the idle-robot budget.
The 501--510 reference block and the 531--540 probe-validation block remain
read-only and are not reused as online outcome seeds.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from WorldModel.evaluation.phase_c_s1_hungarian_protocol import S1_CONFIG


SCHEMA_VERSION = "phase_c_phi_context_ablation_protocol_v1"
FROZEN_BUNDLE_SCHEMA_VERSION = "phase_c_phi_context_ablation_bundle_v1"
PER_ARM_SCHEMA_VERSION = "phase_c_phi_context_ablation_arm_v1"
REPORT_SCHEMA_VERSION = "phase_c_phi_context_ablation_report_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
OUTPUT_ROOT = Path(os.environ.get(
    "PHASEC_PHI_CONTEXT_OUT",
    str(BASE_ROOT / "phi_context_ablation_541_550_v1"),
))
SOURCE_POLICY_ROOT = BASE_ROOT / "phasec_s1_hungarian_cert_491_500_v1"
SOURCE_POLICY_BUNDLE = SOURCE_POLICY_ROOT / (
    "phase_c_s1_hungarian_frozen_protocol.json"
)
MODEL_CHECKPOINT = BASE_ROOT / "model_round1_v1/best_regret_world_model.pt"
PHI_HEAD_ROOT = BASE_ROOT / "station_congestion_head_region_dev_511_520_v1"
PHI_HEAD_CHECKPOINT = PHI_HEAD_ROOT / (
    "linear_head_v1/best_station_congestion_head.pt"
)
PHI_SCALE_CONTRACT = PHI_HEAD_ROOT / "station_congestion_scale_contract.json"

LOADS = ("low", "mid", "high")
LOAD_CONFIGS = {
    "low": Path("Config/world_model_config_PP_48_low.json"),
    "mid": Path("Config/world_model_config_PP_48_mid.json"),
    "high": Path("Config/world_model_config_PP_48_high.json"),
}
SEEDS = tuple(range(541, 551))
LOCKED_REFERENCE_SEEDS = tuple(range(501, 511))
PHI_VALIDATION_SEEDS = tuple(range(531, 541))
TICKS = 1500
TOP_M = 10
PHI_CONTEXT_FACTOR = 2.0
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260808

GREEDY_ARM = "greedy_manifest"
S1_ARM = "s1"
S1_PHI_ARM = "s1_phi_context"
ARM_KEYS = (GREEDY_ARM, S1_ARM, S1_PHI_ARM)
ARM_LABELS = {
    GREEDY_ARM: "GreedyManifestSource",
    S1_ARM: "PhaseCS1Baseline",
    S1_PHI_ARM: "PhaseCS1PhiContext",
}

COMPLETION_RELATIVE_NONINFERIORITY = -0.05
DEADLOCK_MEAN_NONINFERIORITY_MARGIN = 0.02

SOURCE_FILES = (
    "Policies/TaskAssigner/__init__.py",
    "Policies/TaskAssigner/base_task_assigner.py",
    "Policies/TaskAssigner/GreedyTaskAssigner/greedy_task_assigner.py",
    "Policies/TaskAssigner/WorldModelTaskAssigner/__init__.py",
    "Policies/TaskAssigner/WorldModelTaskAssigner/phi_context_rank_assigner.py",
    "Policies/TaskAssigner/WorldModelTaskAssigner/world_model_task_assigner.py",
    "Policies/TaskAssigner/context_assignment.py",
    "WorldModel/__init__.py",
    "WorldModel/core/station_congestion_head.py",
    "WorldModel/core/lyapunov.py",
    "WorldModel/data/build_station_congestion_head_dataset.py",
    "WorldModel/data/candidate_generator.py",
    "WorldModel/evaluation/station_congestion_endpoint.py",
    "WorldModel/evaluation/evaluate_online_v6.py",
    "WorldModel/evaluation/phase_c_s1_hungarian_protocol.py",
    "WorldModel/evaluation/phase_c_phi_context_ablation_protocol.py",
    "WorldModel/evaluation/freeze_phase_c_phi_context_ablation.py",
    "WorldModel/evaluation/run_phase_c_phi_context_ablation_arm.py",
    "WorldModel/evaluation/validate_phase_c_phi_context_ablation.py",
    "WorldModel/evaluation/run_phase_c_phi_context_ablation_cpu.slurm",
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
        raise ValueError(f"unknown phi-context arm: {arm}")
    if load not in LOADS:
        raise ValueError(f"unknown load: {load}")
    if int(seed) not in SEEDS:
        raise ValueError(f"seed is outside the development block: {seed}")
    return f"{arm}_{load}_seed{int(seed)}"


def protocol_payload(artifact_hashes: Mapping[str, str]) -> dict[str, Any]:
    if set(SEEDS) & set(LOCKED_REFERENCE_SEEDS):
        raise RuntimeError("phi-context seeds overlap locked 501--510 outcomes")
    if set(SEEDS) & set(PHI_VALIDATION_SEEDS):
        raise RuntimeError("phi-context seeds overlap 531--540 head validation")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "causal development ablation of frozen current-state station "
            "service pressure for cross-context ordering"
        ),
        "frozen_inputs": {
            "source_policy_bundle": SOURCE_POLICY_BUNDLE.as_posix(),
            "source_policy_bundle_sha256": artifact_hashes[
                "source_policy_bundle"
            ],
            "model_checkpoint": MODEL_CHECKPOINT.as_posix(),
            "model_checkpoint_sha256": artifact_hashes["model_checkpoint"],
            "phi_head_checkpoint": PHI_HEAD_CHECKPOINT.as_posix(),
            "phi_head_checkpoint_sha256": artifact_hashes[
                "phi_head_checkpoint"
            ],
            "phi_scale_contract": PHI_SCALE_CONTRACT.as_posix(),
            "phi_scale_contract_sha256": artifact_hashes[
                "phi_scale_contract"
            ],
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
        "policy_contract": {
            ARM_LABELS[S1_ARM]: {
                "checkpoint": "frozen Phase-C best_regret",
                "s1_config": dict(S1_CONFIG),
                "energy_conv_random_flip_seed": 0,
                "context_order": "legacy fixed-context prefix",
            },
            ARM_LABELS[S1_PHI_ARM]: {
                "checkpoint": "same frozen Phase-C best_regret",
                "s1_config": dict(S1_CONFIG),
                "energy_conv_random_flip_seed": 0,
                "phi_channel": "service",
                "phi_semantics": "predicted current station pressure is a cost",
                "context_superset_factor": PHI_CONTEXT_FACTOR,
                "context_order": (
                    "stable ascending phi_state(service); original order "
                    "breaks exact pressure ties"
                ),
                "within_context_robot_scorer": (
                    "unchanged parent WorldModelTaskAssigner.select_robots"
                ),
                "no_assign": False,
                "hard_gate": False,
                "high_pressure_only_case": "assignment still executes",
            },
        },
        "analysis": {
            "primary_contrast": (
                f"{ARM_LABELS[S1_PHI_ARM]}-minus-{ARM_LABELS[S1_ARM]}"
            ),
            "bootstrap_unit": "seed clustered across loads",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "mechanism_bars": {
                "phi_head_loaded_every_run": True,
                "context_reordered_total_gt": 0,
                "context_replacement_total_gt": 0,
                "applied_budget_pressure_not_above_baseline": True,
            },
            "outcome_guardrails": {
                "completed_orders_relative_delta_ci95_lower_ge": (
                    COMPLETION_RELATIVE_NONINFERIORITY
                ),
                "deadlock_ratio_mean_delta_ci95_upper_le": (
                    DEADLOCK_MEAN_NONINFERIORITY_MARGIN
                ),
                "deadlock_ratio_mean_each_load_point_le": (
                    DEADLOCK_MEAN_NONINFERIORITY_MARGIN
                ),
                "throughput_improvement_is_required": False,
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
            "world_model_retraining": True,
            "s1_retuning": True,
            "psi_pre_online_use": True,
            "delta_psi": True,
            "native_no_assign": True,
            "lyapunov_or_dispatch_gate": True,
            "handcrafted_station_injection": True,
            "post_outcome_phi_weight_tuning": True,
            "overwrite_existing_outputs": True,
        },
    }
    return {**payload, "protocol_sha256": canonical_sha256(payload)}


__all__ = [
    "ARM_KEYS",
    "ARM_LABELS",
    "BOOTSTRAP_REPEATS",
    "BOOTSTRAP_SEED",
    "COMPLETION_RELATIVE_NONINFERIORITY",
    "DEADLOCK_MEAN_NONINFERIORITY_MARGIN",
    "FROZEN_BUNDLE_SCHEMA_VERSION",
    "GREEDY_ARM",
    "LOADS",
    "LOAD_CONFIGS",
    "MODEL_CHECKPOINT",
    "OUTPUT_ROOT",
    "PER_ARM_SCHEMA_VERSION",
    "PHI_CONTEXT_FACTOR",
    "PHI_HEAD_CHECKPOINT",
    "PHI_SCALE_CONTRACT",
    "REPORT_SCHEMA_VERSION",
    "S1_ARM",
    "S1_PHI_ARM",
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
