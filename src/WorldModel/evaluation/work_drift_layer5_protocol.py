"""Frozen protocol for Layer-5 normal-arrival closed-loop certification.

The protocol consumes the already certified Layer-3 work-group-range signal
and Layer-4 ``WorkDriftHead``.  It freezes one dimensionally valid integration
rule before seeds 431--440 are observed::

    score(a) = group_range(WM_cost(a))
               + lambda * group_range(work_drift_hat(a))

Both inputs use the same exact no-gap context transform.  This preserves the
pure-WM candidate ordering while giving the work auxiliary a dimensionless,
preregistered conservative weight.  The weight is not fitted to Layer-5 online
outcomes, and no online/load/gap gate or post-certification tuning is part of
the contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = "work_drift_layer5_protocol_v1"
FROZEN_BUNDLE_SCHEMA_VERSION = "work_drift_layer5_frozen_bundle_v1"

BASE_WORLD_MODEL = (
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
    "stage1_world_model.pt"
)
HEAD_CHECKPOINT = (
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
    "lyapunov_td_impl/layer4_work_drift_group_range_v1/"
    "work_drift_group_range_v1_seed20260716.pt"
)
LAYER3_REPORT = (
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
    "lyapunov_td_impl/lyapunov_oracle_isolated_v3/"
    "cert_work_group_range_421_430/"
    "five_layer_l1_l3_work_group_range_v2_cert_421_430.json"
)
LAYER4_REPORT = (
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
    "lyapunov_td_impl/layer4_work_drift_group_range_v1/"
    "work_drift_group_range_v1_test_421_430.json"
)
FIVE_LAYER_L4_REPORT = (
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
    "lyapunov_td_impl/layer4_work_drift_group_range_v1/"
    "five_layer_l1_l4_work_group_range_v1_421_430.json"
)
LYAPUNOV_CONFIG = "Config/lyapunov_l1_config.json"

EXPECTED_BASE_WORLD_MODEL_SHA256 = (
    "external-fingerprint-omitted"
)
EXPECTED_HEAD_SHA256 = (
    "external-fingerprint-omitted"
)
EXPECTED_LAYER3_PROTOCOL_SHA256 = (
    "external-fingerprint-omitted"
)
EXPECTED_LAYER4_PROTOCOL_SHA256 = (
    "external-fingerprint-omitted"
)

DEVELOPMENT_SEEDS = tuple(range(421, 431))
CERTIFICATION_SEEDS = tuple(range(431, 441))
REQUIRED_LOADS = ("low", "mid", "high")
LOAD_CONFIGS = {
    "low": "Config/world_model_config_PP_48_low.json",
    "mid": "Config/world_model_config_PP_48_mid.json",
    "high": "Config/world_model_config_PP_48_high.json",
}

ROLLOUT_HORIZON = 10
RESERVATION_WINDOW = 10
TICKS = 1500
TOP_M_METADATA_ONLY = 10
BOOTSTRAP_REPEATS = 5000
RANDOM_SEED = 20260716
BURN_IN = 250

# Conservative first-use weight.  It is deliberately smaller than the WM
# coefficient after both context signals are mapped to comparable group-range
# units.  This is a preregistered integration choice, not a statistical optimum
# selected from normal-arrival Layer-5 outcomes.
FROZEN_LAMBDA = 0.25
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


def _primary_layer3_outcome(report: Mapping) -> Mapping:
    layer3 = report.get("layer3_incremental_information") or {}
    if layer3.get("candidate_component") != "work_group_range":
        raise ValueError("Layer-3 report is not the work_group_range certificate")
    formal = layer3.get("formal_candidate_contract") or {}
    if not bool(layer3.get("supported")) or not bool(formal.get("passed")):
        raise ValueError("Layer-3 work_group_range certificate did not pass")
    protocol_hash = (
        (formal.get("protocol") or {}).get("protocol_sha256")
    )
    if protocol_hash != EXPECTED_LAYER3_PROTOCOL_SHA256:
        raise ValueError("unexpected frozen Layer-3 protocol hash")
    primary = layer3.get("primary_outcome")
    if primary != "realized_cost":
        raise ValueError("Layer-5 lambda is frozen to Layer-3 realized_cost")
    outcome = (layer3.get("independent_outcomes") or {}).get(primary)
    if not isinstance(outcome, Mapping):
        raise ValueError("Layer-3 primary outcome is missing")
    return outcome


def derive_frozen_lambda(layer3_report: Mapping) -> dict:
    """Audit Layer-3 support and return the preregistered conservative lambda.

    Layer-3 fold coefficients are checked for stable positive direction, but
    they are not converted into an online weight.  Their regression scaling is
    not the same object as a first-use closed-loop policy coefficient.
    """

    outcome = _primary_layer3_outcome(layer3_report)
    folds = outcome.get("wm_plus_candidate_fold_diagnostics") or []
    if len(folds) != len(DEVELOPMENT_SEEDS):
        raise ValueError("expected exactly ten Layer-3 seed folds")
    work_coefficients = []
    held_out = []
    for fold in folds:
        coefficients = fold.get("standardised_coefficients") or {}
        work_beta = float(coefficients["candidate:work_group_range"])
        if work_beta <= 0.0:
            raise ValueError("Layer-3 work direction is not positive in every fold")
        work_coefficients.append(work_beta)
        held_out.extend(str(value) for value in fold.get("held_out_clusters", ()))
    expected_clusters = [f"seed={seed}" for seed in DEVELOPMENT_SEEDS]
    if sorted(held_out) != sorted(expected_clusters):
        raise ValueError("Layer-3 lambda folds do not cover exactly seeds 421--430")
    return {
        "value": FROZEN_LAMBDA,
        "selection": "preregistered_conservative_dimensionless_weight",
        "fold_work_coefficients_direction_audit": [
            float(item) for item in work_coefficients
        ],
        "development_seed_clusters": expected_clusters,
        "outcome": "realized_cost",
        "uses_layer3_coefficient_magnitude": False,
        "uses_layer5_online_outcomes": False,
    }


def formal_protocol() -> dict:
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "claim": (
            "a frozen predicted analytic work-drift auxiliary improves or "
            "preserves pure World-Model scheduling under normal arrivals"
        ),
        "prerequisites": {
            "layer3_protocol_sha256": EXPECTED_LAYER3_PROTOCOL_SHA256,
            "layer4_protocol_sha256": EXPECTED_LAYER4_PROTOCOL_SHA256,
            "base_world_model_sha256": EXPECTED_BASE_WORLD_MODEL_SHA256,
            "work_drift_head_sha256": EXPECTED_HEAD_SHA256,
        },
        "integration": {
            "formula": (
                "group_range(wm_raw_cost) + lambda * "
                "group_range(predicted_raw_delta_L_work)"
            ),
            "group_range": "(x-group_mean)/(group_max-group_min)",
            "exact_tie_value": 0.0,
            "hard_raw_gap_gate": False,
            "load_gate": False,
            "lambda": FROZEN_LAMBDA,
            "lambda_provenance": (
                "preregistered conservative first-use auxiliary weight; "
                "Layer-3 fold magnitudes are not treated as an online lambda"
            ),
            "online_recalibration": False,
        },
        "world_model_semantics": {
            "fixed_order_pod_station_context": True,
            "online_robot_candidates": "all currently available idle robots",
            "top_m_is_online_limit": False,
            "wm_native_cost_horizon_unchanged": True,
            "work_endpoint_horizon": ROLLOUT_HORIZON,
            "feature_reservation_window": RESERVATION_WINDOW,
            "current_known_demand_only": True,
            "unknown_future_orders_in_rollout": False,
            "continuation_policy_in_rollout": False,
            "normal_arrivals_resume_only_in_real_simulator": True,
            "paired_exogenous_arrivals": (
                "the pure-WM arm generates the normal seeded arrival stream; "
                "its complete realised order manifest is frozen and replayed "
                "at identical ticks in the WorkDrift arm"
            ),
            "arrival_manifest_reference_policy": "pure WorldModel baseline",
            "future_arrivals_visible_to_policy": False,
            "multi_context_semantics": (
                "sequential robot exclusion over one pre-assignment state; "
                "the head estimates each fixed context as a marginal action, "
                "not a learned joint multi-action rollout"
            ),
            "stream_graph_frame_stride": (
                "ticks_plus_one; per-tick Lyapunov/risk scalars remain unthinned"
            ),
        },
        "formal_test": {
            "loads": list(REQUIRED_LOADS),
            "seeds": list(CERTIFICATION_SEEDS),
            "ticks": TICKS,
            "paired_arms": ["WorldModel", "WorldModel+WorkDrift"],
            "bootstrap_unit": "simulation_seed_paired_across_loads",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "random_seed": RANDOM_SEED,
            "burn_in": BURN_IN,
            "one_shot_no_seed_or_load_dropping": True,
            "interrupted_run_resume": (
                "an incomplete load may be rerun only with unchanged frozen "
                "inputs; any existing order manifest must match exactly"
            ),
        },
        "acceptance": {
            "primary": {
                "metric": "wm_label_cost",
                "direction": "lower_is_better",
                "paired_seed_cluster_ci95_upper_below_zero": True,
                "each_load_point_estimate_nonpositive": True,
            },
            "secondary": {
                "metrics": [
                    "completed_orders",
                    "avg_excess_delay",
                    "open_order_count",
                    "wait_or_stall",
                ],
                "minimum_favourable_point_estimates": 3,
            },
            "noninferiority_guardrails": {
                "completed_orders_relative_ci95_lower": -0.02,
                "avg_excess_delay_relative_ci95_upper": 0.05,
                "open_order_count_relative_ci95_upper": 0.05,
                "unified_risk_absolute_ci95_upper": 0.02,
                "deadlock_ratio_max_absolute_ci95_upper": 0.01,
            },
            "trajectory": {
                "work_arm_must_pass_closed_loop_validator": True,
                "required_load_groups": list(REQUIRED_LOADS),
            },
            "implementation": {
                "fallback_greedy_calls": 0,
                "all_idle_scope_required": True,
                "exact_paired_order_manifest_required": True,
                "candidate_superset_above_10_required": True,
                "at_least_one_auxiliary_modified_decision": True,
            },
        },
        "forbidden": {
            "td_risk_v_head": True,
            "long_risk_or_risk_gate_in_score": True,
            "greedy_or_external_assignment_policy": True,
            "future_demand_predictor": True,
            "hard_gap_gate": True,
            "load_gate": True,
            "post_test_lambda_tuning": True,
            "post_test_seed_or_load_dropping": True,
        },
        "scope_limit": (
            "this certifies the tested 48-robot map/configuration family and "
            "all-idle candidate supersets observed on seeds 431--440; it is "
            "not a proof for arbitrary maps, fleets, or arrival processes"
        ),
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    if protocol["protocol_sha256"] != EXPECTED_FORMAL_PROTOCOL_SHA256:
        raise RuntimeError(
            "Layer-5 formal protocol changed without an explicit version/hash update"
        )
    return protocol


def validate_exact_seed_set(values: Sequence[int]) -> None:
    if tuple(sorted(int(value) for value in values)) != CERTIFICATION_SEEDS:
        raise ValueError("Layer-5 certification requires exactly seeds 431--440")


__all__ = [
    "BASE_WORLD_MODEL",
    "BOOTSTRAP_REPEATS",
    "BURN_IN",
    "CERTIFICATION_SEEDS",
    "EXPECTED_BASE_WORLD_MODEL_SHA256",
    "EXPECTED_HEAD_SHA256",
    "EXPECTED_FORMAL_PROTOCOL_SHA256",
    "FIVE_LAYER_L4_REPORT",
    "FROZEN_BUNDLE_SCHEMA_VERSION",
    "FROZEN_LAMBDA",
    "HEAD_CHECKPOINT",
    "LAYER3_REPORT",
    "LAYER4_REPORT",
    "LOAD_CONFIGS",
    "LYAPUNOV_CONFIG",
    "RANDOM_SEED",
    "REQUIRED_LOADS",
    "RESERVATION_WINDOW",
    "ROLLOUT_HORIZON",
    "SCHEMA_VERSION",
    "TICKS",
    "TOP_M_METADATA_ONLY",
    "derive_frozen_lambda",
    "formal_protocol",
    "sha256_file",
    "validate_exact_seed_set",
]
