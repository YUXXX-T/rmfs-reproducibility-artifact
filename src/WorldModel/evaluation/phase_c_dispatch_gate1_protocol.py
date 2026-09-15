"""Frozen Phase-C Round-2 dispatch-potential Kill Gate 1 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = "phase_c_dispatch_gate1_protocol_v1"
FROZEN_BUNDLE_SCHEMA_VERSION = "phase_c_dispatch_gate1_bundle_v1"
ONLINE_REPORT_SCHEMA_VERSION = "phase_c_dispatch_gate1_three_arm_online_v1"
PER_SEED_REPORT_SCHEMA_VERSION = "phase_c_dispatch_gate1_seed_v1"
TRACE_SCHEMA_VERSION = "phase_c_dispatch_gate1_trace_v1"
VALIDATION_SCHEMA_VERSION = "phase_c_dispatch_gate1_validation_v1"

BASE_ROOT = "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1"
GATE_ROOT = f"{BASE_ROOT}/round2_dispatch_gate1_481_486_v1"
BASELINE_CHECKPOINT = (
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
    "stage1_world_model.pt"
)
CANDIDATE_CHECKPOINT = f"{BASE_ROOT}/model_round1_v1/best_regret_world_model.pt"
EXPECTED_BASELINE_SHA256 = (
    "external-fingerprint-omitted"
)
EXPECTED_CANDIDATE_SHA256 = (
    "external-fingerprint-omitted"
)

SEEDS = tuple(range(481, 487))
REQUIRED_LOADS = ("low", "mid", "high")
LOAD_CONFIGS = {
    "low": "Config/world_model_config_PP_48_low.json",
    "mid": "Config/world_model_config_PP_48_mid.json",
    "high": "Config/world_model_config_PP_48_high.json",
}
TICKS = 1500
TOP_M_METADATA_ONLY = 10
TRACE_MAX_RECORDS = 300000
TRAJECTORY_BIN_EDGES = (0, 250, 500, 1000, 1500)
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260727

STAGE1_LABEL = "Stage1WorldModel"
ROUND1_LABEL = "PhaseCWorldModel"
BRIDGE_LABEL = "DispatchBridge"

LIVENESS_CROSSING = 2.0 ** 0.5
DOMINANCE_DEBT = 1.5
MAX_ELIGIBLE_DEFER_MASS = 2.0 * LIVENESS_CROSSING

BRIDGE_ROUND1_BOUNDS = {
    "completion_fraction_ci95_lower_strict": -0.05,
    "completion_fraction_each_load_strict": -0.05,
    "completion_fraction_each_run_min": -0.10,
    "deadlock_ratio_max_ci95_upper": 0.02,
    "deadlock_ratio_max_each_load_upper": 0.02,
    "avg_excess_delay_relative_ci95_upper": 0.05,
    "completed_flow_time_relative_ci95_upper": 0.10,
    "open_order_fraction_ci95_upper": 0.02,
    "pending_order_fraction_ci95_upper": 0.02,
    "open_order_age_fraction_ci95_upper": 0.05,
    "pending_order_age_fraction_ci95_upper": 0.05,
    "handoff_ratio_mean_ci95_upper": 0.01,
    "handoff_ratio_max_ci95_upper": 0.05,
}

BRIDGE_STAGE1_BOUNDS = {
    "completion_fraction_ci95_lower_strict": 0.0,
    "completion_fraction_each_load_strict": 0.0,
    "deadlock_ratio_max_ci95_upper": 0.02,
}

# Filled after canonicalising this source-level protocol.  Any semantic edit
# requires a new protocol version and a deliberately updated digest.
EXPECTED_PROTOCOL_SHA256 = (
    "external-fingerprint-omitted"
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


def _protocol_body() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": "PREREGISTERED_DEVELOPMENT_KILL_GATE_NOT_FORMAL_CERTIFICATION",
        "claim": (
            "a parameter-free dispatch potential removes the native "
            "NO_ASSIGN absorbing branch without hiding an induced congestion "
            "regression, while retaining positive value over never deferring"
        ),
        "evidence_isolation": {
            "round1_certification_seeds_471_480": (
                "mechanism evidence only; forbidden for formula, scale, "
                "threshold, checkpoint, seed, or load selection"
            ),
            "gate1_seeds": list(SEEDS),
            "post_result_tuning": False,
            "seed_or_load_dropping": False,
        },
        "control_law": {
            "score": "exact_group_range(C_WM) + delta_L_dispatch",
            "wm_group_range_includes_defer_and_all_robot_candidates": True,
            "wm_group_span_upper": 1.0,
            "raw_tie_maps_to_zero": True,
            "exact_composite_tie_selects_assignment": True,
            "tunable_lambda": False,
            "delta_l_second_normalisation": False,
            "distance": "static_graph_shortest_path_to_station_entry",
            "pending_pressure_source": "all_materialised_pending_chains",
            "station_capacity": "max(num_agents/num_stations,1)",
            "debt_increment": "1/T_ff only when eligible and deferred",
            "assignment_debt_transition": "clear",
            "temporarily_ineligible_debt_transition": "hold",
            "liveness_crossing": LIVENESS_CROSSING,
            "dominance_debt": DOMINANCE_DEBT,
            "max_continuous_eligible_defer_mass": MAX_ELIGIBLE_DEFER_MASS,
        },
        "policy_semantics": {
            "stage1": "all idle robots; no NO_ASSIGN",
            "round1": "all idle robots plus frozen native global-zero NO_ASSIGN",
            "bridge": (
                "same Round-1 checkpoint/action encoding plus the analytic "
                "dispatch control law; no learned head and no hard gate"
            ),
            "context_proposal": (
                "full pending ledger, then debt-desc/order-age-desc/stable-id "
                "priority, then pod de-duplication and idle-count budget"
            ),
            "robot_robot_ordering": "frozen World Model only",
            "decision_frequency": "every simulator tick",
        },
        "test": {
            "loads": list(REQUIRED_LOADS),
            "seeds": list(SEEDS),
            "ticks": TICKS,
            "arms": [STAGE1_LABEL, ROUND1_LABEL, BRIDGE_LABEL],
            "paired_arrivals": (
                "Stage-1 generates one manifest; Round-1 and bridge replay it"
            ),
            "top_m_metadata_only": TOP_M_METADATA_ONLY,
            "trace_max_records": TRACE_MAX_RECORDS,
            "trajectory_bin_edges": list(TRAJECTORY_BIN_EDGES),
            "bootstrap_unit": "source seed clustered across loads",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "implementation_integrity": {
            "diagnostic_nonperturbation_smoke_required": True,
            "diagnostic_smoke": {
                "load": "low",
                "seed": 9981,
                "ticks": 100,
                "exact_action_trace": True,
                "exact_non_timing_metrics": True,
            },
            "group_span_violations": 0,
            "crossing_assignment_rate": 1.0,
            "crossing_violations": 0,
            "fallback_greedy_calls": 0,
            "all_idle_scope": True,
            "exact_paired_manifest": True,
            "decision_trace_dropped": 0,
        },
        "liveness_acceptance": {
            "terminal_bin_robot_assignment_when_contexts_exist": True,
            "each_seed_load_required": True,
            "max_eligible_defer_streak_bound_passed": True,
            "max_continuous_eligible_defer_mass": MAX_ELIGIBLE_DEFER_MASS,
        },
        "bridge_vs_round1": dict(BRIDGE_ROUND1_BOUNDS),
        "bridge_vs_stage1": dict(BRIDGE_STAGE1_BOUNDS),
        "diagnostic_branches": [
            "DEFER_ABSORPTION",
            "INDUCED_CONGESTION_REGRESSION",
            "PREEXISTING_CONGESTION_BRANCH",
            "DEFER_VALUE_NOT_DEMONSTRATED",
            "PASS_OR_OTHER_NONABSORBING",
        ],
        "forbidden": {
            "greedy_or_hungarian_inside_policy": True,
            "td_target_or_head": True,
            "work_drift_or_residual_head": True,
            "new_neural_head": True,
            "future_demand_predictor": True,
            "hard_liveness_threshold": True,
            "post_result_scale_tuning": True,
        },
    }


def formal_protocol() -> dict:
    body = _protocol_body()
    digest = canonical_sha256(body)
    if EXPECTED_PROTOCOL_SHA256 != "TO_BE_FILLED" and digest != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "dispatch Gate-1 protocol changed without a version/hash update: "
            f"expected={EXPECTED_PROTOCOL_SHA256} actual={digest}"
        )
    return {**body, "protocol_sha256": digest}


def validate_exact_seed_set(values: Sequence[int]) -> None:
    if tuple(int(value) for value in values) != SEEDS:
        raise ValueError(f"Gate-1 seeds must be exactly {list(SEEDS)}")


__all__ = [name for name in globals() if name.isupper()] + [
    "canonical_sha256",
    "formal_protocol",
    "sha256_file",
    "validate_exact_seed_set",
]
