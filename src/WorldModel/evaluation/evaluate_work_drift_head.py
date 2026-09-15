"""Evaluate the finite-horizon work-group-range Layer-4 estimator.

This command scores held-out isolated H=10 candidate groups with the frozen
base World Model and a separately trained :class:`WorkDriftHead`.  It verifies
checkpoint/data provenance, seed disjointness and the no-future-demand/no-TD
contract before deciding whether the learned endpoint estimator beats the
within-group zero predictor.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from WorldModel.core.work_drift_head import WorkDriftHead
from WorldModel.evaluation.work_drift_layer4_protocol import (
    FORMAL_BOOTSTRAP_REPEATS,
    FORMAL_DEVELOPMENT_SEEDS,
    FORMAL_LOADS,
    FORMAL_RANDOM_SEED,
    FORMAL_TEST_SEEDS,
    FORMAL_TRAIN_SEEDS,
    GROUP_RANGE_SEMANTICS,
    PREDICTION_SEMANTICS,
    formal_layer4_protocol,
)
from WorldModel.training.train_td_risk_v import load_frozen_world_model
from WorldModel.training.train_work_drift_head import (
    _expand_paths,
    _load_samples,
    _precompute_paths,
    _sha256_file,
    _work_schema,
    evaluate_rows,
)


EVALUATION_SCHEMA_VERSION = "work_drift_group_range_evaluation_v1"


def _checkpoint_contract(payload: Mapping) -> dict:
    protocol = formal_layer4_protocol()
    manifest = payload.get("data_manifest") or {}
    train = manifest.get("train") or {}
    validation = manifest.get("validation") or {}
    config = payload.get("training_config") or {}
    semantics = payload.get("semantics") or {}
    head_config = payload.get("head_config") or {}
    recorded_protocol = payload.get("formal_protocol") or {}
    actual_loss = config.get("loss_weights") or {}
    expected_loss = protocol["training"]["loss_weights"]
    checks = {
        "exact_train_seeds_411_418": train.get("simulation_seeds")
        == list(FORMAL_TRAIN_SEEDS),
        "exact_development_seeds_419_420": validation.get("simulation_seeds")
        == list(FORMAL_DEVELOPMENT_SEEDS),
        "seed_disjoint": bool(manifest.get("seed_disjoint")),
        "recorded_protocol_sha256_matches": (
            recorded_protocol.get("protocol_sha256")
            == protocol["protocol_sha256"]
        ),
        "isolated_horizon_10": (
            semantics.get("rollout_continuation_mode") == "isolated"
            and int(semantics.get("horizon", -1)) == 10
        ),
        "current_demand_only": bool(semantics.get("current_demand_only")),
        "no_future_demand_predictor": semantics.get("future_demand_predictor") is False,
        "no_continuation_policy": semantics.get("continuation_policy") is False,
        "no_td_tail": semantics.get("td_tail") is False,
        "no_greedy_or_external_policy": semantics.get("greedy_or_external_policy") is False,
        "no_hard_gap": semantics.get("hard_raw_gap_gate") is False,
        "no_load_gate": semantics.get("load_gate") is False,
        "prediction_semantics": semantics.get("prediction") == PREDICTION_SEMANTICS,
        "no_direct_action_embedding_to_head": (
            semantics.get("direct_action_embedding_to_head") is False
        ),
        "hidden_dim": int(head_config.get("hidden_dim", -1)) == 128,
        "residual_limit": float(head_config.get("residual_limit", -1.0)) == 4.0,
        "training_seed": int(config.get("seed", -1)) == FORMAL_RANDOM_SEED,
        "epochs_requested": int(config.get("epochs_requested", -1)) == 50,
        "patience": int(config.get("patience", -1)) == 8,
        "learning_rate": float(config.get("lr", -1.0)) == 3e-4,
        "weight_decay": float(config.get("weight_decay", -1.0)) == 1e-4,
        "group_batch_size": int(config.get("group_batch_size", -1)) == 16,
        "loss_weights": all(
            float(actual_loss.get(name, -1.0)) == float(value)
            for name, value in expected_loss.items()
        ),
        "pairwise_temperature": float(config.get("pairwise_temperature", -1.0)) == 0.25,
        "scale_floor": float(config.get("scale_floor", -1.0)) == 1e-3,
        "grad_clip": float(config.get("grad_clip", -1.0)) == 1.0,
        "checkpoint_selection": (
            config.get("checkpoint_selection")
            == protocol["training"]["checkpoint_selection"]
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "protocol": protocol,
    }


def _metric_contract(metrics: Mapping) -> dict:
    continuous = metrics.get("continuous_group_range") or {}
    nrmse_ci = continuous.get(
        "normalised_rmse_ci95_seed_cluster_bootstrap"
    )
    spearman_ci = continuous.get("spearman_ci95_seed_cluster_bootstrap")
    pairwise_ci = continuous.get(
        "all_non_tie_pair_concordance_ci95_seed_cluster_bootstrap"
    )
    regret_ci = continuous.get(
        "normalised_selection_regret_improvement_over_uniform_random_"
        "ci95_seed_cluster_bootstrap"
    )
    top1_ci = continuous.get(
        "top1_improvement_over_tie_aware_random_ci95_seed_cluster_bootstrap"
    )
    regret_summary = continuous.get(
        "normalised_selection_regret_improvement_over_uniform_random"
    ) or {}
    regret_point = regret_summary.get("mean")
    top1 = continuous.get("top1_min_drift_accuracy")
    random_top1 = continuous.get("random_tie_aware_top1_baseline")
    by_load = metrics.get("by_load") or {}
    load_checks = {}
    for load in FORMAL_LOADS:
        row = by_load.get(load) or {}
        load_regret = (
            row.get(
                "normalised_selection_regret_improvement_over_uniform_random"
            ) or {}
        ).get("mean")
        load_pairwise = row.get("all_non_tie_pair_concordance")
        load_checks[load] = {
            "pairwise_concordance": load_pairwise,
            "normalised_regret_improvement": load_regret,
            "passed": bool(
                load_pairwise is not None and float(load_pairwise) > 0.5
                and load_regret is not None and float(load_regret) > 0.0
            ),
        }
    by_seed = metrics.get("by_seed") or {}
    robust_seed_count = 0
    seed_checks = {}
    for seed, row in by_seed.items():
        seed_regret = (
            row.get(
                "normalised_selection_regret_improvement_over_uniform_random"
            ) or {}
        ).get("mean")
        seed_pairwise = row.get("all_non_tie_pair_concordance")
        passed = bool(
            seed_pairwise is not None and float(seed_pairwise) > 0.5
            and seed_regret is not None and float(seed_regret) > 0.0
        )
        robust_seed_count += int(passed)
        seed_checks[str(seed)] = {
            "pairwise_concordance": seed_pairwise,
            "normalised_regret_improvement": seed_regret,
            "passed": passed,
        }
    checks = {
        "normalised_rmse_point_below_zero_predictor": (
            continuous.get("normalised_rmse") is not None
            and float(continuous["normalised_rmse"]) < 1.0
        ),
        "normalised_rmse_ci95_upper_below_zero_predictor": (
            isinstance(nrmse_ci, Sequence)
            and len(nrmse_ci) == 2
            and float(nrmse_ci[1]) < 1.0
        ),
        "spearman_positive": (
            continuous.get("spearman") is not None
            and float(continuous["spearman"]) > 0.0
        ),
        "seed_cluster_spearman_ci95_lower_above_zero": (
            isinstance(spearman_ci, Sequence)
            and len(spearman_ci) == 2
            and float(spearman_ci[0]) > 0.0
        ),
        "pairwise_concordance_above_random": (
            continuous.get("all_non_tie_pair_concordance") is not None
            and float(continuous["all_non_tie_pair_concordance"]) > 0.5
        ),
        "pairwise_ci95_lower_above_random": (
            isinstance(pairwise_ci, Sequence)
            and len(pairwise_ci) == 2
            and float(pairwise_ci[0]) > 0.5
        ),
        "normalised_regret_improvement_positive": (
            regret_point is not None and float(regret_point) > 0.0
        ),
        "normalised_regret_improvement_ci95_lower_above_zero": (
            isinstance(regret_ci, Sequence)
            and len(regret_ci) == 2
            and float(regret_ci[0]) > 0.0
        ),
        "top1_above_tie_aware_random": (
            top1 is not None and random_top1 is not None
            and float(top1) > float(random_top1)
        ),
        "all_required_loads_directionally_supported": all(
            row["passed"] for row in load_checks.values()
        ),
        "at_least_eight_of_ten_seed_clusters_supported": (
            robust_seed_count >= 8
        ),
    }
    primary_names = (
        "normalised_rmse_ci95_upper_below_zero_predictor",
        "seed_cluster_spearman_ci95_lower_above_zero",
        "pairwise_ci95_lower_above_random",
        "normalised_regret_improvement_ci95_lower_above_zero",
    )
    return {
        "checks": checks,
        "primary_checks": list(primary_names),
        "passed": all(checks[name] for name in primary_names),
        "guardrails_passed": all(checks.values()),
        "load_checks": load_checks,
        "seed_checks": seed_checks,
        "supported_seed_clusters": robust_seed_count,
        "top1_improvement_ci95_seed_cluster_bootstrap": top1_ci,
    }


def _schema_matches(checkpoint: Mapping, observed: Mapping) -> bool:
    expected = checkpoint.get("work_schema") or {}
    return (
        expected.get("station_ids") == observed.get("station_ids")
        and expected.get("work_capacity") == observed.get("work_capacity")
        and float(expected.get("work_weight", -1.0))
        == float(observed.get("work_weight", -2.0))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--world-model-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=FORMAL_RANDOM_SEED)
    parser.add_argument("--formal-421-430", action="store_true")
    parser.add_argument("--fail-unless-supported", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_repeats <= 0:
        raise SystemExit("--bootstrap-repeats must be positive")
    if args.formal_421_430 and (
        args.bootstrap_repeats != FORMAL_BOOTSTRAP_REPEATS
        or args.random_seed != FORMAL_RANDOM_SEED
    ):
        raise SystemExit(
            "formal Layer 4 freezes --bootstrap-repeats/--random-seed to "
            f"{(FORMAL_BOOTSTRAP_REPEATS, FORMAL_RANDOM_SEED)}"
        )
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing report: {output}")

    head, checkpoint = WorkDriftHead.from_checkpoint(args.head_checkpoint)
    checkpoint_contract = _checkpoint_contract(checkpoint)
    horizon = int((checkpoint.get("semantics") or {}).get("horizon", 10))
    data_paths = _expand_paths(args.data)
    first_samples, _ = _load_samples(
        [data_paths[0]], required_horizon=horizon
    )
    observed_schema = _work_schema(first_samples)
    train_seeds = set(
        ((checkpoint.get("data_manifest") or {}).get("train") or {})
        .get("simulation_seeds", [])
    )
    development_seeds = set(
        ((checkpoint.get("data_manifest") or {}).get("validation") or {})
        .get("simulation_seeds", [])
    )
    recorded_wm = checkpoint.get("base_world_model") or {}
    actual_wm_sha256 = _sha256_file(args.world_model_checkpoint)
    base_checkpoint_matches = actual_wm_sha256 == recorded_wm.get("sha256")
    if not base_checkpoint_matches:
        raise SystemExit(
            "base World Model checkpoint SHA256 mismatch; Layer-4 endpoint "
            "latents must use the exact training checkpoint"
        )
    if not _schema_matches(checkpoint, observed_schema):
        raise SystemExit("held-out work/station schema differs from the head checkpoint")
    model, model_config = load_frozen_world_model(
        args.world_model_checkpoint,
        first_samples[0],
        len(observed_schema["station_ids"]),
        args.device,
    )
    head.to(args.device).eval()
    rows, manifest, data_audit = _precompute_paths(
        model,
        data_paths,
        expected_schema=observed_schema,
        device=args.device,
        horizon=horizon,
    )
    del first_samples, model
    test_seeds = set(manifest["simulation_seeds"])
    split_disjoint = not bool(test_seeds & (train_seeds | development_seeds))
    metrics = evaluate_rows(
        head,
        rows,
        device=args.device,
        bootstrap_repeats=args.bootstrap_repeats,
        random_seed=args.random_seed,
    )
    metric_contract = _metric_contract(metrics)
    seed_load_runs = {}
    for row in rows:
        key = (int(row["seed"]), str(row["load"]))
        seed_load_runs.setdefault(key, set()).add(str(row["run_id"]))
    expected_seed_loads = {
        (seed, load)
        for seed in FORMAL_TEST_SEEDS
        for load in FORMAL_LOADS
    }
    data_checks = {
        "test_seed_disjoint_from_train_and_development": split_disjoint,
        "at_least_ten_independent_test_seed_clusters": (
            len(test_seeds) >= 10
        ),
        "exact_formal_test_seeds_421_430": (
            sorted(test_seeds) == list(FORMAL_TEST_SEEDS)
        ),
        "exact_paired_low_mid_high_arms": (
            set(seed_load_runs) == expected_seed_loads
            and all(len(run_ids) == 1 for run_ids in seed_load_runs.values())
        ),
        "exact_thirty_source_files": len(data_paths) == 30,
        "work_schema_matches_checkpoint": True,
        "base_world_model_sha256_matches": base_checkpoint_matches,
        "strict_full_horizon_isolated_data": bool(
            data_audit.get("strict_full_horizon")
        ),
    }
    formal_data_passed = all(data_checks.values())
    formal_passed = bool(
        args.formal_421_430
        and checkpoint_contract["passed"]
        and formal_data_passed
        and metric_contract["passed"]
        and metric_contract["guardrails_passed"]
    )
    if formal_passed:
        status = "WORLD_MODEL_WORK_GROUP_RANGE_ESTIMATION_SUPPORTED"
    elif metric_contract["passed"]:
        status = "WORLD_MODEL_WORK_GROUP_RANGE_ESTIMATION_PROMISING_PROTOCOL_NOT_CLOSED"
    else:
        continuous = metrics["continuous_group_range"]
        nrmse_ci = continuous.get(
            "normalised_rmse_ci95_seed_cluster_bootstrap"
        ) or [None, None]
        spearman_ci = continuous.get("spearman_ci95_seed_cluster_bootstrap") or [None, None]
        pairwise_ci = continuous.get(
            "all_non_tie_pair_concordance_ci95_seed_cluster_bootstrap"
        ) or [None, None]
        regret_ci = continuous.get(
            "normalised_selection_regret_improvement_over_uniform_random_"
            "ci95_seed_cluster_bootstrap"
        ) or [None, None]
        clearly_failed = (
            nrmse_ci[0] is not None and float(nrmse_ci[0]) >= 1.0
        ) or (
            spearman_ci[1] is not None and float(spearman_ci[1]) <= 0.0
        ) or (
            pairwise_ci[1] is not None and float(pairwise_ci[1]) <= 0.5
        ) or (
            regret_ci[1] is not None and float(regret_ci[1]) <= 0.0
        )
        status = (
            "WORLD_MODEL_WORK_GROUP_RANGE_ESTIMATION_FAILED"
            if clearly_failed
            else "WORLD_MODEL_WORK_GROUP_RANGE_ESTIMATION_INCONCLUSIVE"
        )
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "layer": 4,
        "name": "world_model_work_group_range_drift_estimation",
        "status": status,
        "supported": formal_passed,
        "passed": formal_passed,
        "online_ready": False,
        "online_blocker": "layer5_normal_arrival_closed_loop_not_certified",
        "semantics": {
            "prediction": PREDICTION_SEMANTICS,
            "target": "isolated H=10 raw analytic Delta L_work",
            "candidate_transform": GROUP_RANGE_SEMANTICS,
            "raw_absolute_calibration": "diagnostic_only",
            "future_unknown_orders": False,
            "continuation_policy": False,
            "td_risk_v_head": False,
            "hard_raw_gap_gate": False,
            "load_gate": False,
            "candidate_supersets_above_collected_top_m_10": "NOT_EVALUATED",
        },
        "formal_protocol": checkpoint_contract["protocol"],
        "checkpoint_contract": checkpoint_contract,
        "data_contract": {
            "checks": data_checks,
            "passed": formal_data_passed,
            "manifest": manifest,
            "audit": data_audit,
        },
        "metric_contract": metric_contract,
        "validation": metrics,
        "base_world_model": {
            "requested_path": os.path.abspath(args.world_model_checkpoint),
            "sha256": actual_wm_sha256,
            "model_config": model_config,
        },
        "head_checkpoint": {
            "path": os.path.abspath(args.head_checkpoint),
            "sha256": _sha256_file(args.head_checkpoint),
            "schema_version": checkpoint.get("schema_version"),
        },
        "parameters": {
            "formal_421_430": bool(args.formal_421_430),
            "bootstrap_repeats": args.bootstrap_repeats,
            "random_seed": args.random_seed,
            "device": args.device,
        },
        "next_step": (
            "Run Layer 5 under normal arrivals with all currently idle robots "
            "only if this untouched report is supported."
            if formal_passed else
            "Do not add the estimator online; diagnose development data or "
            "architecture, then use new untouched seeds 431-440 after changes."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    continuous = metrics["continuous_group_range"]
    print(f"saved: {output}")
    print("Layer 4 status =", status)
    print("formal passed =", formal_passed)
    print("protocol sha256 =", checkpoint_contract["protocol"]["protocol_sha256"])
    print("test seed clusters =", metrics["independent_seed_clusters"])
    print("group-range NRMSE =", continuous.get("normalised_rmse"))
    print(
        "NRMSE CI95 =",
        continuous.get("normalised_rmse_ci95_seed_cluster_bootstrap"),
    )
    print("Spearman =", continuous.get("spearman"))
    print(
        "Spearman CI95 =",
        continuous.get("spearman_ci95_seed_cluster_bootstrap"),
    )
    print("pairwise =", continuous.get("all_non_tie_pair_concordance"))
    print(
        "pairwise CI95 =",
        continuous.get(
            "all_non_tie_pair_concordance_ci95_seed_cluster_bootstrap"
        ),
    )
    print("top1 =", continuous.get("top1_min_drift_accuracy"))
    print(
        "normalised regret improvement =",
        continuous.get(
            "normalised_selection_regret_improvement_over_uniform_random"
        ),
    )
    print(
        "normalised regret improvement CI95 =",
        continuous.get(
            "normalised_selection_regret_improvement_over_uniform_random_"
            "ci95_seed_cluster_bootstrap"
        ),
    )
    print(
        "supported seed clusters =",
        metric_contract.get("supported_seed_clusters"),
    )
    print("by_load =", {
        load: row.get("normalised_rmse")
        for load, row in metrics["by_load"].items()
    })
    print("note: Layer 5 and online all-idle supersets remain unverified")
    if args.fail_unless_supported and not formal_passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
