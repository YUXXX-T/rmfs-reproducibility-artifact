"""Frozen Phase-C Round-1 paired closed-loop certification protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = "phase_c_round1_online_cert_protocol_v1"
FROZEN_BUNDLE_SCHEMA_VERSION = "phase_c_round1_online_cert_bundle_v1"
ONLINE_REPORT_SCHEMA_VERSION = "phase_c_round1_paired_online_v1"
CERT_REPORT_SCHEMA_VERSION = "phase_c_round1_closed_loop_certification_v1"

BASE_ROOT = "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1"
BASELINE_CHECKPOINT = (
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
    "stage1_world_model.pt"
)
CANDIDATE_CHECKPOINT = f"{BASE_ROOT}/model_round1_v1/best_regret_world_model.pt"
TRAINING_PROTOCOL = f"{BASE_ROOT}/phase_c_round1_frozen_protocol.json"
TRAINING_SUMMARY = f"{BASE_ROOT}/model_round1_v1/train_summary.json"
OFFLINE_BASELINE_REPORT = (
    f"{BASE_ROOT}/model_round1_v1/offline_seed470_v1/stage1_seed470.json"
)
OFFLINE_CANDIDATE_REPORT = (
    f"{BASE_ROOT}/model_round1_v1/offline_seed470_v1/"
    "phasec_best_regret_seed470.json"
)
PILOT_REPORTS = {
    load: f"{BASE_ROOT}/online_pair_pilot_{load}_seed9972_t500_v1/paired_online.json"
    for load in ("low", "mid", "high")
}

EXPECTED_BASELINE_SHA256 = (
    "external-fingerprint-omitted"
)
EXPECTED_CANDIDATE_SHA256 = (
    "external-fingerprint-omitted"
)
EXPECTED_TRAINING_PROTOCOL_SHA256 = (
    "external-fingerprint-omitted"
)

TRAINING_SEEDS = tuple(range(461, 468))
VALIDATION_SEEDS = (468, 469)
OFFLINE_TEST_SEEDS = (470,)
CERTIFICATION_SEEDS = tuple(range(471, 481))
DEVELOPMENT_PILOT_SEEDS = (9971, 9972)
REQUIRED_LOADS = ("low", "mid", "high")
LOAD_CONFIGS = {
    "low": "Config/world_model_config_PP_48_low.json",
    "mid": "Config/world_model_config_PP_48_mid.json",
    "high": "Config/world_model_config_PP_48_high.json",
}

TICKS = 1500
TOP_M_METADATA_ONLY = 10
BASELINE_HORIZON = 3
CANDIDATE_HORIZON = 10
TRAJECTORY_BIN_EDGES = (0, 250, 500, 1000, 1500)
DECISION_TRACE_MAX_RECORDS = 200000
DECISION_TRACE_SCHEMA_VERSION = "phase_c_round1_candidate_decision_trace_v1"
PER_SEED_REPORT_SCHEMA_VERSION = "phase_c_round1_paired_seed_v1"
BOOTSTRAP_REPEATS = 5000
RANDOM_SEED = 20260725

# Formal outcome contract.  Relative comparisons are candidate minus baseline.
PRIMARY_METRIC = "completion_fraction"
SECONDARY_METRICS = (
    "completed_tasks_relative",
    "avg_excess_delay_relative",
    "open_order_fraction",
    "pending_order_fraction",
    "wait_or_stall",
    "deadlock_ratio_max",
)
SECONDARY_REQUIRED = 5

GUARDRAILS = {
    "avg_excess_delay_relative_ci95_upper": 0.05,
    "completed_flow_time_relative_ci95_upper": 0.10,
    "open_order_fraction_ci95_upper": 0.02,
    "pending_order_fraction_ci95_upper": 0.02,
    "open_order_age_fraction_ci95_upper": 0.05,
    "pending_order_age_fraction_ci95_upper": 0.05,
    "deadlock_ratio_max_absolute_ci95_upper": 0.02,
    "handoff_ratio_mean_absolute_ci95_upper": 0.01,
    "handoff_ratio_max_absolute_ci95_upper": 0.05,
}

# Canonical preregistration hash.  Changing any formal outcome, threshold,
# policy semantic, seed, or trajectory rule requires an explicit new version.
EXPECTED_FORMAL_PROTOCOL_SHA256 = (
    "external-fingerprint-omitted"
)


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
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "claim": (
            "the frozen Phase-C Round-1 World Model improves paired normal-"
            "arrival closed-loop scheduling over the original Stage-1 World "
            "Model without external assignment logic"
        ),
        "prerequisites": {
            "training_protocol_sha256": EXPECTED_TRAINING_PROTOCOL_SHA256,
            "baseline_checkpoint_sha256": EXPECTED_BASELINE_SHA256,
            "candidate_checkpoint_sha256": EXPECTED_CANDIDATE_SHA256,
            "candidate_selected_before_offline_test": True,
            "candidate_selection_rule": (
                "minimum validation top1 regret on seeds 468--469"
            ),
            "selected_checkpoint": "best_regret_world_model.pt",
            "selected_epoch": 2,
            "offline_test_seed": 470,
            "development_pilot_seeds": list(DEVELOPMENT_PILOT_SEEDS),
        },
        "policy_semantics": {
            "baseline": {
                "checkpoint_horizon": BASELINE_HORIZON,
                "action_space": "all currently idle robots only",
                "native_no_assign": False,
            },
            "candidate": {
                "checkpoint_horizon": CANDIDATE_HORIZON,
                "action_space": (
                    "all currently idle robots plus checkpoint-audited native "
                    "NO_ASSIGN"
                ),
                "native_no_assign": True,
                "no_assign_hard_threshold": False,
            },
            "top_m_is_online_limit": False,
            "fixed_context_assignment": True,
            "future_unknown_orders_in_model_rollout": False,
            "continuation_scheduler_in_model_rollout": False,
            "paired_arrivals": (
                "Stage-1 generates the realised normal-arrival manifest and "
                "Phase-C replays it exactly"
            ),
            "future_arrivals_visible_to_policy": False,
        },
        "formal_test": {
            "loads": list(REQUIRED_LOADS),
            "seeds": list(CERTIFICATION_SEEDS),
            "ticks": TICKS,
            "paired_arms": ["Stage1WorldModel", "PhaseCWorldModel"],
            "top_m_metadata_only": TOP_M_METADATA_ONLY,
            "bootstrap_unit": "simulation_seed_paired_across_loads",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "random_seed": RANDOM_SEED,
            "trajectory_bin_edges": list(TRAJECTORY_BIN_EDGES),
            "decision_trace_max_records": DECISION_TRACE_MAX_RECORDS,
            "decision_trace_schema_version": DECISION_TRACE_SCHEMA_VERSION,
            "one_shot_no_seed_or_load_dropping": True,
            "whole_seed_cluster_preserved_across_loads": True,
        },
        "acceptance": {
            "primary": {
                "metric": PRIMARY_METRIC,
                "definition": (
                    "completed_orders / paired_order_arrival_count"
                ),
                "direction": "higher_is_better",
                "overall_paired_seed_cluster_ci95_lower_above_zero": True,
                "each_load_point_estimate_above_zero": True,
            },
            "secondary": {
                "metrics": {
                    "completed_tasks_relative": "higher_is_better",
                    "avg_excess_delay_relative": "lower_is_better",
                    "open_order_fraction": "lower_is_better",
                    "pending_order_fraction": "lower_is_better",
                    "wait_or_stall": "lower_is_better",
                    "deadlock_ratio_max": "lower_is_better",
                },
                "minimum_favourable_point_estimates": SECONDARY_REQUIRED,
            },
            "noninferiority_guardrails": {
                **GUARDRAILS,
                "relative_definition": (
                    "(candidate-baseline)/max(abs(baseline),1)"
                ),
                "order_fraction_definition": (
                    "arm_order_count/paired_order_arrival_count"
                ),
                "age_fraction_definition": "order_age_p95/test_ticks",
                "ci_comparison": "candidate_minus_baseline",
            },
            "load_robustness": {
                "each_load_completion_fraction_positive": True,
                "each_load_avg_excess_delay_relative_at_most": 0.05,
                "each_load_open_order_fraction_at_most": 0.02,
                "each_load_deadlock_ratio_max_delta_at_most": 0.02,
            },
            "station_pressure": {
                "standalone_increase_is_not_failure": True,
                "conditional_high_load_rule": (
                    "if high-load station pressure increases, high-load "
                    "completion fraction must improve and excess delay, "
                    "bottleneck CVaR, and deadlock max must not worsen"
                ),
                "comparison_level": "paired_high_load_point_estimates",
            },
            "no_assign_trajectory": {
                "bin_edges": list(TRAJECTORY_BIN_EDGES),
                "report_selection_rate_by_load_seed_and_bin": True,
                "trace_count_must_match_online_stats": True,
                "terminal_bin_must_contain_robot_assignment_when_contexts_exist": True,
                "fixed_rate_or_online_gate": False,
            },
            "implementation": {
                "fallback_greedy_calls": 0,
                "all_idle_scope_required": True,
                "exact_paired_order_manifest_required": True,
                "baseline_native_no_assign_forbidden": True,
                "candidate_native_no_assign_required": True,
                "decision_trace_read_only": True,
            },
        },
        "diagnostic_only": {
            "unified_risk": (
                "reported but not gated because high-load pilots saturated at 1"
            ),
            "runtime": "reported but not a policy-quality acceptance metric",
            "station_pressure": (
                "interpreted jointly with throughput/delay/bottleneck/deadlock"
            ),
        },
        "forbidden": {
            "greedy_or_hungarian_assignment": True,
            "td_target_or_head": True,
            "work_drift_or_residual_head": True,
            "lyapunov_online_score": True,
            "future_demand_predictor": True,
            "hard_gap_or_load_gate": True,
            "post_test_checkpoint_selection": True,
            "post_test_no_assign_threshold_tuning": True,
            "post_test_seed_or_load_dropping": True,
        },
        "scope_limit": (
            "certifies the tested 48-robot map/configuration family, normal "
            "arrival generators, and all-idle online candidate sets observed "
            "on seeds 471--480; not arbitrary maps or fleets"
        ),
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    expected = EXPECTED_FORMAL_PROTOCOL_SHA256
    if expected != "__TO_BE_FROZEN__" and protocol["protocol_sha256"] != expected:
        raise RuntimeError(
            "Phase-C formal protocol changed without an explicit version/hash update"
        )
    return protocol


def validate_exact_seed_set(values: Sequence[int]) -> None:
    actual = tuple(sorted(int(value) for value in values))
    if actual != CERTIFICATION_SEEDS:
        raise ValueError("Phase-C certification requires exactly seeds 471--480")


__all__ = [
    "BASELINE_CHECKPOINT",
    "BASELINE_HORIZON",
    "BASE_ROOT",
    "BOOTSTRAP_REPEATS",
    "CANDIDATE_CHECKPOINT",
    "CANDIDATE_HORIZON",
    "CERTIFICATION_SEEDS",
    "CERT_REPORT_SCHEMA_VERSION",
    "DECISION_TRACE_MAX_RECORDS",
    "DECISION_TRACE_SCHEMA_VERSION",
    "EXPECTED_BASELINE_SHA256",
    "EXPECTED_CANDIDATE_SHA256",
    "EXPECTED_FORMAL_PROTOCOL_SHA256",
    "EXPECTED_TRAINING_PROTOCOL_SHA256",
    "FROZEN_BUNDLE_SCHEMA_VERSION",
    "GUARDRAILS",
    "LOAD_CONFIGS",
    "OFFLINE_BASELINE_REPORT",
    "OFFLINE_CANDIDATE_REPORT",
    "ONLINE_REPORT_SCHEMA_VERSION",
    "PER_SEED_REPORT_SCHEMA_VERSION",
    "PILOT_REPORTS",
    "RANDOM_SEED",
    "REQUIRED_LOADS",
    "SCHEMA_VERSION",
    "SECONDARY_METRICS",
    "SECONDARY_REQUIRED",
    "TICKS",
    "TOP_M_METADATA_ONLY",
    "TRAJECTORY_BIN_EDGES",
    "TRAINING_PROTOCOL",
    "TRAINING_SUMMARY",
    "canonical_sha256",
    "formal_protocol",
    "sha256_file",
    "validate_exact_seed_set",
]
