"""Frozen preregistration for the first work-drift Layer-4 evaluation."""

from __future__ import annotations

import hashlib
import json


SCHEMA_VERSION = "work_drift_group_range_layer4_protocol_v1"
PREDICTION_SEMANTICS = "analytic_current_work_plus_frozen_wm_endpoint_residual"
GROUP_RANGE_SEMANTICS = "(raw_delta_L_work-group_mean)/group_raw_range"
FORMAL_RANDOM_SEED = 20260716
FORMAL_BOOTSTRAP_REPEATS = 5000
FORMAL_TRAIN_SEEDS = tuple(range(411, 419))
FORMAL_DEVELOPMENT_SEEDS = (419, 420)
FORMAL_TEST_SEEDS = tuple(range(421, 431))
FORMAL_LOADS = ("low", "mid", "high")


def formal_layer4_protocol() -> dict:
    """Return the immutable machine-readable protocol and canonical hash.

    Seeds 421--430 already certified the analytic Layer-3 transform but remain
    untouched by this learned estimator.  If estimator choices are changed
    after their Layer-4 predictions are inspected, a later formal run must use
    new untouched seeds (recommended 431--440).
    """

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "claim": (
            "frozen World Model endpoint latent estimates isolated H=10 "
            "analytic work drift on held-out candidate contexts"
        ),
        "base_world_model": (
            "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered/"
            "stage1_world_model.pt"
        ),
        "collection_schema": "lyapunov_l1_collection_v3",
        "rollout_continuation_mode": "isolated",
        "horizon": 10,
        "required_loads": list(FORMAL_LOADS),
        "train_seeds": list(FORMAL_TRAIN_SEEDS),
        "development_seeds": list(FORMAL_DEVELOPMENT_SEEDS),
        "held_out_test_seeds": list(FORMAL_TEST_SEEDS),
        "head": {
            "prediction": PREDICTION_SEMANTICS,
            "hidden_dim": 128,
            "residual_limit": 4.0,
            "current_demand_only": True,
            "direct_action_embedding_to_head": False,
            "endpoint_station_work_then_fixed_L_work": True,
        },
        "training": {
            "seed": FORMAL_RANDOM_SEED,
            "epochs": 50,
            "patience": 8,
            "group_batch_size": 16,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "loss_weights": {
                "group_range": 1.0,
                "raw_drift": 0.25,
                "endpoint": 0.25,
                "all_non_tie_pairwise": 0.25,
            },
            "pairwise_temperature": 0.25,
            "scale_floor": 1e-3,
            "grad_clip": 1.0,
            "checkpoint_selection": (
                "min development [group_range_nrmse "
                "- 0.25*(pairwise-0.5) - 0.10*(top1-random_top1)]"
            ),
            "hard_raw_gap_gate": False,
            "load_gate": False,
        },
        "evaluation": {
            "candidate_transform": GROUP_RANGE_SEMANTICS,
            "candidate_group_weighting": "equal",
            "bootstrap_unit": "simulation_seed_paired_across_loads",
            "bootstrap_repeats": FORMAL_BOOTSTRAP_REPEATS,
            "random_seed": FORMAL_RANDOM_SEED,
            "primary_checks": [
                "normalised_rmse_ci95_upper_below_zero_predictor_1",
                "seed_cluster_spearman_ci95_lower_above_0",
                "pairwise_concordance_ci95_lower_above_random_0.5",
                "normalised_regret_improvement_ci95_lower_above_0",
            ],
            "robustness_checks": [
                "top1 improvement over tie-aware random is positive",
                "each low/mid/high load has pairwise > 0.5 and positive regret improvement",
                "at least 8 of 10 seed clusters have pairwise > 0.5 and positive regret improvement",
            ],
            "raw_drift_absolute_calibration": "diagnostic_only",
            "raw_range_strata": "diagnostic_only_not_a_gap_gate",
            "by_load": "robustness_check_not_an_online_gate",
            "candidate_count_strata": "diagnostic_only",
        },
        "forbidden": {
            "unknown_future_orders": True,
            "future_demand_predictor": True,
            "continuation_policy": True,
            "td_risk_v_head": True,
            "greedy_or_external_assignment_policy": True,
            "hard_gap_or_load_gate": True,
            "post_test_seed_or_load_dropping": True,
        },
        "scope_limit": (
            "online all-idle candidate supersets above collected top-m=10 "
            "remain unverified until Layer 5"
        ),
    }
    canonical = json.dumps(
        protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    protocol["protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    return protocol


__all__ = [
    "FORMAL_BOOTSTRAP_REPEATS",
    "FORMAL_DEVELOPMENT_SEEDS",
    "FORMAL_LOADS",
    "FORMAL_RANDOM_SEED",
    "FORMAL_TEST_SEEDS",
    "FORMAL_TRAIN_SEEDS",
    "GROUP_RANGE_SEMANTICS",
    "PREDICTION_SEMANTICS",
    "SCHEMA_VERSION",
    "formal_layer4_protocol",
]
