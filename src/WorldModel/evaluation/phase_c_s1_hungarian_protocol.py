"""Frozen Phase-C/S1 versus distance-baseline online comparison protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = "phase_c_s1_hungarian_protocol_v1"
FROZEN_BUNDLE_SCHEMA_VERSION = "phase_c_s1_hungarian_bundle_v1"
PER_SEED_SCHEMA_VERSION = "phase_c_s1_hungarian_seed_v1"
AGGREGATE_SCHEMA_VERSION = "phase_c_s1_hungarian_aggregate_v1"

BASE_ROOT = "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1"
OUTPUT_ROOT = f"{BASE_ROOT}/phasec_s1_hungarian_cert_491_500_v1"
CANDIDATE_CHECKPOINT = f"{BASE_ROOT}/model_round1_v1/best_regret_world_model.pt"

LOADS = ("low", "mid", "high")
LOAD_CONFIGS = {
    "low": "Config/world_model_config_PP_48_low.json",
    "mid": "Config/world_model_config_PP_48_mid.json",
    "high": "Config/world_model_config_PP_48_high.json",
}
SEEDS = tuple(range(491, 501))
TICKS = 1500
TOP_M = 10

GREEDY_LABEL = "GreedySequentialManhattan"
HUNGARIAN_LABEL = "HungarianBatchManhattan"
PHASEC_LABEL = "PhaseCRobotOnly"
PHASEC_S1_LABEL = "PhaseCS1RobotOnly"
ARMS = (
    GREEDY_LABEL,
    HUNGARIAN_LABEL,
    PHASEC_LABEL,
    PHASEC_S1_LABEL,
)

# Exact frozen S1/C3 within-context conversion configuration.  These values
# come from the previously evaluated GATED_C3_L025 arm.  No post-Phase-C
# tuning is permitted in this comparison.
S1_CONFIG = {
    "include_no_assign_candidate": False,
    "risk_threshold": None,
    "risk_weight": None,
    "long_risk_beta": {"terminal": 0.5, "cvar": 0.2},
    "risk_defer_mode": "off",
    "station_injection_weight": 0.0,
    "potential_guard_mode": "off",
    "pool_scoring_mode": "off",
    "candidate_set_guard_mode": "off",
    "margin_substitute_mode": "off",
    "energy_scoring_mode": "conversion",
    "energy_potential_form": "endpoint",
    "energy_score_lambda": 0.0,
    "energy_discount": 0.95,
    "energy_drift_signal": "combo",
    "energy_gate_mode": "sigmoid",
    "energy_gate_window": 200,
    "energy_gate_warmup": 50,
    "energy_gate_quantile": 0.8,
    "energy_conv_lambda": 0.25,
    "energy_conv_random_flip_rate": 0.0,
    "energy_weight_wait": 0.0,
    "energy_weight_excess": 0.0,
    "energy_weight_station": 1.0,
    "energy_weight_bottleneck": 1.0,
    "energy_weight_severe": 2.0,
    "energy_weight_completed": 0.0,
    "energy_weight_long_terminal": 0.0,
    "energy_weight_long_cvar": 0.0,
    "energy_weight_long_peak": 0.0,
    "energy_weight_long_delta": 0.0,
    "energy_service_relief_scale": 1.0,
    "energy_service_weight_order_age": 1.0,
    "energy_service_weight_station_pending": 1.0,
    "energy_service_weight_order_size": 0.0,
    "lyapunov_l0_mode": "off",
    "dispatch_potential_mode": "off",
    "work_drift_mode": "off",
    "candidate_robot_mode": "nearest",
    "candidate_context_mode": "prefix",
}

PHASEC_CONFIG = {
    "include_no_assign_candidate": False,
    "risk_threshold": None,
    "risk_weight": None,
    "long_risk_beta": {},
    "risk_defer_mode": "off",
    "station_injection_weight": 0.0,
    "potential_guard_mode": "off",
    "pool_scoring_mode": "off",
    "candidate_set_guard_mode": "off",
    "margin_substitute_mode": "off",
    "energy_scoring_mode": "off",
    "lyapunov_l0_mode": "off",
    "dispatch_potential_mode": "off",
    "work_drift_mode": "off",
    "candidate_robot_mode": "nearest",
    "candidate_context_mode": "prefix",
}

METRIC_DIRECTIONS = {
    "completed_orders": "higher",
    "completed_tasks": "higher",
    "avg_task_duration": "lower",
    "avg_excess_delay": "lower",
    "open_order_count": "lower",
    "pending_order_count": "lower",
    "wait_or_stall": "lower",
    "stall_ratio_mean": "lower",
    "stall_ratio_max": "lower",
    "deadlock_ratio_mean": "lower",
    "deadlock_ratio_max": "lower",
    "congestion_events": "lower",
    "severe_events": "lower",
    "risk_rate_per_100": "lower",
    "wall_time_s": "lower",
    "assignment_time_ms_mean": "lower",
}

# The user's decisive requirement is congestion non-inferiority to the
# distance-only Hungarian baseline.  Throughput remains reported but is not a
# hard gate.  The 0.02 margin matches the existing Phase-C guardrail scale.
DEADLOCK_NONINFERIORITY_MARGIN = 0.02
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260729


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


def formal_protocol() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "claim": (
            "compare the frozen Phase-C robot-only policy and the same "
            "checkpoint with the previously frozen S1/C3 within-context "
            "conversion against sequential Greedy Manhattan and independent "
            "batch Hungarian Manhattan under exact replayed order streams"
        ),
        "formal_test": {
            "loads": list(LOADS),
            "seeds": list(SEEDS),
            "ticks": TICKS,
            "top_m": TOP_M,
            "arms": list(ARMS),
            "manifest_source": GREEDY_LABEL,
            "one_manifest_per_load_seed": True,
            "exact_manifest_replay_required": True,
            "bootstrap_unit": "simulation_seed_clustered_across_loads",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "policy_semantics": {
            GREEDY_LABEL: (
                "repository GreedyTaskAssigner: sequential nearest-idle-robot "
                "Manhattan selection with parallel task chains"
            ),
            HUNGARIAN_LABEL: (
                "independent global linear-sum assignment minimizing total "
                "idle-robot-to-pod Manhattan distance"
            ),
            PHASEC_LABEL: (
                "best_regret Phase-C checkpoint, top_m=10, robot candidates "
                "only, no native NO_ASSIGN and no Lyapunov/dispatch/work head"
            ),
            PHASEC_S1_LABEL: (
                "the same Phase-C checkpoint plus the exact frozen S1/C3 "
                "within-context conversion; robot candidates only"
            ),
            "s1_config": S1_CONFIG,
            "phasec_config": PHASEC_CONFIG,
        },
        "primary_analysis": {
            "contrast": f"{PHASEC_S1_LABEL}-minus-{HUNGARIAN_LABEL}",
            "throughput_is_hard_gate": False,
            "deadlock_ratio_mean_noninferiority_margin": (
                DEADLOCK_NONINFERIORITY_MARGIN
            ),
            "overall_cluster_ci95_upper_must_not_exceed_margin": True,
            "each_load_point_estimate_must_not_exceed_margin": True,
            "additional_congestion_metrics_reported_without_tuning": [
                "deadlock_ratio_max",
                "stall_ratio_mean",
                "stall_ratio_max",
                "congestion_events",
                "severe_events",
                "risk_rate_per_100",
                "avg_excess_delay",
            ],
        },
        "secondary_analysis": {
            "phasec_s1_minus_phasec": True,
            "phasec_minus_hungarian": True,
            "hungarian_minus_greedy": True,
            "metrics": METRIC_DIRECTIONS,
        },
        "forbidden": {
            "native_no_assign": True,
            "dispatch_potential": True,
            "new_lyapunov_score": True,
            "work_drift_or_residual_head": True,
            "post_test_s1_tuning": True,
            "seed_or_load_dropping": True,
            "checkpoint_reselection": True,
        },
    }

