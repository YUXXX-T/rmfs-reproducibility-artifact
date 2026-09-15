"""Frozen post-failure protocol for analytic-A Layer-5 localisation.

This protocol does not replace the failed learned-head Layer-5 certificate.
It opens a new, independent diagnostic branch with exactly two online arms::

    pure World Model
    World Model + H=5 parameter-free analytic work drift

No trainable work/TD/context head, Greedy fallback, load gate or hard score gap
is part of the branch.  Decision snapshots are captured for fixed-context
H=5 isolated replay so a closed-loop failure can be assigned to the analytic
signal, its fusion with the World Model, World-Model OOD, or longer feedback.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = "analytic_work_layer5_localisation_protocol_v1"
FROZEN_BUNDLE_SCHEMA_VERSION = "analytic_work_layer5_frozen_bundle_v1"

BASE_WORLD_MODEL = (
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
    "stage1_world_model.pt"
)
LYAPUNOV_CONFIG = "Config/lyapunov_l1_config.json"
HORIZON_DEVELOPMENT_REPORT = (
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
    "lyapunov_td_impl/analytic_work_relief_residual_v1/"
    "horizon_models_dev_451_456/analytic_work_horizon_dev_451_456.json"
)
EFFICIENCY_DECOMPOSITION_REPORT = (
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
    "lyapunov_td_impl/analytic_work_relief_residual_v1/"
    "efficiency_decomposition_dev_451_456_v1/"
    "analytic_work_efficiency_decomposition_451_456.json"
)
EXPECTED_BASE_WORLD_MODEL_SHA256 = (
    "external-fingerprint-omitted"
)
EXPECTED_HORIZON_DEVELOPMENT_REPORT_SHA256 = (
    "external-fingerprint-omitted"
)
EXPECTED_EFFICIENCY_DECOMPOSITION_REPORT_SHA256 = (
    "external-fingerprint-omitted"
)

LOAD_CONFIGS = {
    "low": "Config/world_model_config_PP_48_low.json",
    "mid": "Config/world_model_config_PP_48_mid.json",
    "high": "Config/world_model_config_PP_48_high.json",
}
REQUIRED_LOADS = tuple(LOAD_CONFIGS)
DEVELOPMENT_SEEDS = (461, 462, 463)
POSTCERT_REPRODUCTION_SEEDS = (432, 437, 439)

ANALYTIC_HORIZON = 5
WM_SYSTEM_HORIZON = 10
FROZEN_LAMBDA = 0.25
TICKS = 1000
POSTCERT_REPRODUCTION_TICKS = 1500
TOP_M_METADATA_ONLY = 10
RESERVATION_WINDOW = 10
SNAPSHOT_INTERVAL = 25


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


def protocol() -> dict:
    value = {
        "schema_version": SCHEMA_VERSION,
        "role": "POST_LAYER5_FAILURE_CAUSAL_LOCALISATION_AND_PHASEC_CAPTURE",
        "claim": (
            "test whether parameter-free H=5 analytic work drift can improve "
            "the frozen World Model and localise any remaining closed-loop "
            "failure without adding another learned head"
        ),
        "arms": [
            "WorldModel",
            "WorldModel+AnalyticWorkH5",
        ],
        "integration": {
            "formula": (
                "group_range(wm_raw_cost) + 0.25 * "
                "group_range(analytic_H5_delta_L_work)"
            ),
            "group_range": "(x-group_mean)/(group_max-group_min)",
            "exact_tie_value": 0.0,
            "lambda": FROZEN_LAMBDA,
            "hard_gap_gate": False,
            "load_gate": False,
            "online_recalibration": False,
        },
        "analytic_work": {
            "horizon": ANALYTIC_HORIZON,
            "endpoint": "virtual_post_action_free_flow_work_ledger",
            "potential": "fixed_quadratic_L_work",
            "learned_parameters": False,
            "shared_efficiency_eta_in_decision": False,
            "candidate_residual_head": False,
            "legacy_work_drift_head": False,
            "development_evidence": {
                "horizon_report_sha256": (
                    EXPECTED_HORIZON_DEVELOPMENT_REPORT_SHA256
                ),
                "efficiency_decomposition_report_sha256": (
                    EXPECTED_EFFICIENCY_DECOMPOSITION_REPORT_SHA256
                ),
            },
        },
        "world_model_semantics": {
            "fixed_order_pod_station_context": True,
            "online_robot_candidates": "all currently available idle robots",
            "top_m_is_online_limit": False,
            "current_known_demand_only": True,
            "unknown_future_orders_in_rollout": False,
            "continuation_policy_in_rollout": False,
            "td_risk_v_head": False,
            "greedy_or_external_assignment_policy": False,
        },
        "paired_online_test": {
            "loads": list(REQUIRED_LOADS),
            "seeds": list(DEVELOPMENT_SEEDS),
            "ticks": TICKS,
            "top_m_metadata_only": TOP_M_METADATA_ONLY,
            "reservation_window": RESERVATION_WINDOW,
            "normal_arrivals": True,
            "order_stream": (
                "baseline-generated manifest replayed exactly in analytic arm"
            ),
            "post_failure_baseline_reuse": (
                "non-formal localisation may reuse a completed pure-WM arm "
                "only when its per-seed metrics and realised order-manifest "
                "hash close exactly"
            ),
        },
        "postcert_failure_reproduction": {
            "role": "DIAGNOSTIC_ONLY_NOT_CERTIFICATION",
            "seeds": list(POSTCERT_REPRODUCTION_SEEDS),
            "ticks": POSTCERT_REPRODUCTION_TICKS,
            "selection": (
                "post-hoc representatives of completion/deadlock, open-order, "
                "and excess-delay guardrail failures in the frozen 431--440 run"
            ),
            "baseline": (
                "reuse the immutable pure-WM metrics and realised order "
                "manifests from the failed Layer-5 run; execute only the new "
                "analytic arm"
            ),
            "may_support_new_claim": False,
        },
        "phase_c_localisation": {
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "capture_policy": "all_modified_plus_interval_controls",
            "snapshot_candidate_scope": "all_idle_online",
            "attach_exact_online_decision_trace": True,
            "fixed_context_analytic_truth_horizon": ANALYTIC_HORIZON,
            "fixed_context_system_truth_horizon": WM_SYSTEM_HORIZON,
            "fixed_context_replay_continuation": "isolated",
            "phase_c_priority_signals": [
                "candidate_count_above_training_top_m",
                "analytic_modified_selection",
                "harmful_combined_selection_under_isolated_truth",
                "wm_ranking_error_under_isolated_truth",
                "long_run_tail_state",
            ],
        },
        "forbidden": {
            "legacy_work_drift_head": True,
            "candidate_residual_head": True,
            "context_efficiency_head": True,
            "td_risk_v_head": True,
            "future_order_predictor": True,
            "continuation_policy_in_wm_rollout": True,
            "greedy_or_external_assignment_policy": True,
            "hard_gap_or_load_gate": True,
        },
        "certification_status": (
            "DEVELOPMENT_LOCALISATION_ONLY_NOT_LAYER5_CERTIFICATION"
        ),
    }
    value["protocol_sha256"] = canonical_sha256(value)
    return value


def validate_exact_seed_set(seeds: Sequence[int]) -> None:
    if tuple(int(value) for value in seeds) != DEVELOPMENT_SEEDS:
        raise ValueError(
            f"frozen development seeds must be {list(DEVELOPMENT_SEEDS)}"
        )


__all__ = [
    "ANALYTIC_HORIZON",
    "BASE_WORLD_MODEL",
    "DEVELOPMENT_SEEDS",
    "EFFICIENCY_DECOMPOSITION_REPORT",
    "EXPECTED_EFFICIENCY_DECOMPOSITION_REPORT_SHA256",
    "EXPECTED_BASE_WORLD_MODEL_SHA256",
    "EXPECTED_HORIZON_DEVELOPMENT_REPORT_SHA256",
    "FROZEN_BUNDLE_SCHEMA_VERSION",
    "FROZEN_LAMBDA",
    "HORIZON_DEVELOPMENT_REPORT",
    "LOAD_CONFIGS",
    "LYAPUNOV_CONFIG",
    "POSTCERT_REPRODUCTION_SEEDS",
    "POSTCERT_REPRODUCTION_TICKS",
    "REQUIRED_LOADS",
    "RESERVATION_WINDOW",
    "SNAPSHOT_INTERVAL",
    "TICKS",
    "TOP_M_METADATA_ONLY",
    "WM_SYSTEM_HORIZON",
    "protocol",
    "sha256_file",
    "validate_exact_seed_set",
]
