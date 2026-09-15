"""Diagnose shared work-relief efficiency versus candidate-level residuals.

This is a development-only attribution tool for the analytic work-relief
experiments.  It does not train or promote an online policy.  For one frozen
horizon it compares:

``A``
    Parameter-free analytic free-flow relief.
``A_eta_global``
    One positive efficiency fitted on training groups only.
``A_eta_context_oracle``
    One label-visible efficiency shared by every candidate in a validation
    context.  This is an oracle diagnostic, never an online estimator.
``C_shared_projection``
    The existing C head projected onto one efficiency shared by all candidates
    in a context, using only C's predictions and the analytic nominal relief.
``C_full``
    Analytic relief plus the existing candidate-level learned residual.

The comparison separates scale calibration from candidate-specific action
information.  Future orders, continuation policies, TD tails and external
assignment heuristics remain outside the experiment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from WorldModel.core.analytic_work_residual_head import AnalyticWorkResidualHead
from WorldModel.core.work_drift_head import WorkDriftHead, group_range_normalise
from WorldModel.training.train_analytic_work_residual_head import (
    _legacy_rows,
    _load_samples_at_horizon,
    _precompute_paths,
)
from WorldModel.training.train_td_risk_v import load_frozen_world_model
from WorldModel.training.train_work_drift_head import (
    _aggregate_group_records,
    _bootstrap_metrics,
    _groups,
    _pearson,
    _quantiles,
    _sha256_file,
    _spearman,
    _work_schema,
    evaluate_rows,
)


SCHEMA_VERSION = "analytic_work_efficiency_residual_decomposition_v1"

ABC_DEFINITIONS = {
    "A_parameter_free_analytic": (
        "Fixed analytic free-flow relief from the post-action pipeline-chain "
        "ledger; no fitted parameters and no neural residual."
    ),
    "B_legacy_endpoint_head": (
        "Legacy WorkDriftHead that directly predicts the H-step endpoint work "
        "and drift from frozen-World-Model features."
    ),
    "C_analytic_plus_candidate_residual": (
        "Fixed analytic nominal relief plus a learned candidate-specific, "
        "station-wise signed residual."
    ),
    "phase_c_distinction": (
        "The A/B/C ablation label C is not Phase C.  Phase C is the later "
        "online decision-snapshot data and World-Model retraining stage."
    ),
}


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if not math.isfinite(float(denominator)) or abs(float(denominator)) <= 1e-15:
        return None
    return float(numerator) / float(denominator)


def _group_equal_eta(
    groups: Sequence[Sequence[Mapping]],
    target_getter: Callable[[Sequence[Mapping]], torch.Tensor],
    *,
    fallback: float = 1.0,
) -> dict:
    """Fit one eta with equal total weight for every decision context."""

    numerator = 0.0
    denominator = 0.0
    informative = 0
    for members in groups:
        nominal = torch.stack([
            torch.as_tensor(row["nominal_station_relief"], dtype=torch.float64)
            for row in members
        ])
        target = target_getter(members).to(dtype=torch.float64, device="cpu")
        if target.shape != nominal.shape:
            raise ValueError("shared-efficiency target and nominal shapes differ")
        group_denominator = float(torch.square(nominal).mean())
        if group_denominator <= 1e-15:
            continue
        numerator += float((nominal * target).mean())
        denominator += group_denominator
        informative += 1
    unconstrained = (
        numerator / denominator if denominator > 1e-15 else float(fallback)
    )
    clipped = float(np.clip(unconstrained, 0.0, 1.0))
    return {
        "eta": clipped,
        "unconstrained_eta": float(unconstrained),
        "clipped": bool(abs(clipped - unconstrained) > 1e-12),
        "informative_groups": int(informative),
        "total_groups": int(len(groups)),
        "weighting": "candidate_group_equal_then_candidate_station_equal",
        "fit_split": "training_only",
    }


def _eta_for_group(
    members: Sequence[Mapping],
    target: torch.Tensor,
    *,
    fallback: float,
) -> tuple[float, float, bool]:
    nominal = torch.stack([
        torch.as_tensor(row["nominal_station_relief"], dtype=torch.float64)
        for row in members
    ])
    target = target.to(dtype=torch.float64, device="cpu")
    denominator = float(torch.square(nominal).sum())
    if denominator <= 1e-15:
        return float(fallback), float(fallback), False
    unconstrained = float((nominal * target).sum()) / denominator
    clipped = float(np.clip(unconstrained, 0.0, 1.0))
    return clipped, unconstrained, bool(abs(clipped - unconstrained) > 1e-12)


def _work_potential(
    station_work: torch.Tensor,
    *,
    capacity: torch.Tensor,
    work_weight: float,
) -> torch.Tensor:
    return 0.5 * float(work_weight) * torch.square(
        station_work / capacity
    ).sum(dim=-1)


def _group_record(
    members: Sequence[Mapping],
    predicted_relief: torch.Tensor,
    *,
    capacity: torch.Tensor,
    work_weight: float,
) -> dict:
    predicted_relief = predicted_relief.to(dtype=torch.float64, device="cpu")
    start = torch.stack([
        torch.as_tensor(row["current_station_work"], dtype=torch.float64)
        for row in members
    ])
    post = torch.stack([
        torch.as_tensor(row["post_action_station_work"], dtype=torch.float64)
        for row in members
    ])
    endpoint = torch.stack([
        torch.as_tensor(row["endpoint_station_work"], dtype=torch.float64)
        for row in members
    ])
    if predicted_relief.shape != endpoint.shape:
        raise ValueError("predicted relief has an invalid candidate/station shape")
    predicted_endpoint = torch.clamp_min(post - predicted_relief, 0.0)
    predicted_raw = _work_potential(
        predicted_endpoint, capacity=capacity, work_weight=work_weight
    ) - _work_potential(start, capacity=capacity, work_weight=work_weight)
    target_raw = torch.tensor([
        float(row["target_raw_work_drift"]) for row in members
    ], dtype=torch.float64)
    return {
        "target_raw": target_raw.numpy(),
        "predicted_raw": predicted_raw.numpy(),
        "target_range": group_range_normalise(target_raw).numpy(),
        "predicted_range": group_range_normalise(predicted_raw).numpy(),
        "endpoint_absolute_error": (
            predicted_endpoint - endpoint
        ).abs().numpy(),
        "seed": int(members[0]["seed"]),
        "load": str(members[0]["load"]),
        "group_key": members[0]["group_key"],
        "candidate_count": len(members),
    }


def _evaluate_records(
    records: Sequence[Mapping],
    *,
    bootstrap_repeats: int,
    random_seed: int,
) -> dict:
    result = _aggregate_group_records(records)
    result["continuous_group_range"].update(_bootstrap_metrics(
        records,
        repeats=bootstrap_repeats,
        random_seed=random_seed,
    ))
    by_load = {}
    for load in sorted({str(row["load"]) for row in records}):
        by_load[load] = _aggregate_group_records([
            row for row in records if str(row["load"]) == load
        ])["continuous_group_range"]
    by_seed = {}
    for seed in sorted({int(row["seed"]) for row in records}):
        by_seed[str(seed)] = _aggregate_group_records([
            row for row in records if int(row["seed"]) == seed
        ])["continuous_group_range"]
    by_candidate_count = {}
    for count in sorted({int(row["candidate_count"]) for row in records}):
        by_candidate_count[str(count)] = _aggregate_group_records([
            row for row in records if int(row["candidate_count"]) == count
        ])["continuous_group_range"]
    result["by_load"] = by_load
    result["by_seed"] = by_seed
    result["by_candidate_count"] = by_candidate_count

    nonzero_ranges = np.asarray([
        float(np.max(row["target_raw"]) - np.min(row["target_raw"]))
        for row in records
        if float(np.max(row["target_raw"]) - np.min(row["target_raw"])) > 0.0
    ], dtype=float)
    boundaries = (
        np.quantile(nonzero_ranges, [0.25, 0.50, 0.75]).tolist()
        if nonzero_ranges.size else [0.0, 0.0, 0.0]
    )

    def range_label(record: Mapping) -> str:
        value = float(
            np.max(record["target_raw"]) - np.min(record["target_raw"])
        )
        if value == 0.0:
            return "zero"
        if value <= boundaries[0]:
            return "nonzero_q1"
        if value <= boundaries[1]:
            return "nonzero_q2"
        if value <= boundaries[2]:
            return "nonzero_q3"
        return "nonzero_q4"

    strata = {}
    for label in (
        "zero", "nonzero_q1", "nonzero_q2", "nonzero_q3", "nonzero_q4"
    ):
        selected = [row for row in records if range_label(row) == label]
        if selected:
            strata[label] = _aggregate_group_records(selected)[
                "continuous_group_range"
            ]
    result["raw_range_strata"] = {
        "role": "diagnostic_only_not_a_gap_gate",
        "nonzero_quartile_boundaries": boundaries,
        "strata": strata,
    }
    result["independent_seed_clusters"] = len(by_seed)
    continuous = result["continuous_group_range"]
    nrmse = continuous.get("normalised_rmse")
    pairwise = continuous.get("all_non_tie_pair_concordance")
    top1 = continuous.get("top1_min_drift_accuracy")
    random_top1 = continuous.get("random_tie_aware_top1_baseline")
    result["selection_score"] = (
        float(nrmse if nrmse is not None else 10.0)
        - 0.25 * ((0.5 if pairwise is None else float(pairwise)) - 0.5)
        - 0.10 * (
            (0.0 if top1 is None else float(top1))
            - (0.0 if random_top1 is None else float(random_top1))
        )
    )
    return result


def _c_prediction(
    head: AnalyticWorkResidualHead,
    members: Sequence[Mapping],
    *,
    device: str,
) -> torch.Tensor:
    global_context = torch.stack([
        torch.as_tensor(row["global_context"]) for row in members
    ]).to(device)
    station_context = torch.stack([
        torch.as_tensor(row["station_context"]) for row in members
    ]).to(device)
    start = torch.stack([
        torch.as_tensor(row["current_station_work"]) for row in members
    ]).to(device)
    with torch.no_grad():
        prediction = head.predict_from_features(
            global_context, station_context, start
        )
    return prediction.predicted_station_relief.detach().cpu().double()


def _pairwise_effect_concordance(
    true_groups: Sequence[np.ndarray],
    predicted_groups: Sequence[np.ndarray],
) -> dict:
    per_group = []
    pair_count = 0
    for true, predicted in zip(true_groups, predicted_groups):
        correct = []
        for left in range(len(true)):
            for right in range(left + 1, len(true)):
                true_gap = float(true[left] - true[right])
                if true_gap == 0.0:
                    continue
                predicted_gap = float(predicted[left] - predicted[right])
                correct.append(
                    0.5 if predicted_gap == 0.0
                    else float(np.sign(predicted_gap) == np.sign(true_gap))
                )
                pair_count += 1
        if correct:
            per_group.append(float(np.mean(correct)))
    return {
        "candidate_group_equal_concordance": (
            float(np.mean(per_group)) if per_group else None
        ),
        "non_tie_pairs": int(pair_count),
        "groups_with_non_tie_effects": int(len(per_group)),
    }


def _selection_switch_audit(
    groups: Sequence[Mapping],
    *,
    baseline_key: str,
    candidate_key: str,
) -> dict:
    same = beneficial = harmful = neutral = 0
    changes = []
    for row in groups:
        truth = np.asarray(row["target_raw"], dtype=float)
        baseline = np.asarray(row[baseline_key], dtype=float)
        candidate = np.asarray(row[candidate_key], dtype=float)
        baseline_choice = int(np.argmin(baseline))
        candidate_choice = int(np.argmin(candidate))
        if baseline_choice == candidate_choice:
            same += 1
        true_min = float(truth.min())
        raw_range = float(truth.max() - true_min)
        baseline_regret = float(truth[baseline_choice] - true_min)
        candidate_regret = float(truth[candidate_choice] - true_min)
        delta = candidate_regret - baseline_regret
        if delta < 0.0:
            beneficial += 1
        elif delta > 0.0:
            harmful += 1
        else:
            neutral += 1
        if raw_range > 0.0:
            changes.append(delta / raw_range)
    total = len(groups)
    return {
        "groups": int(total),
        "selection_preservation_rate": _safe_ratio(same, total),
        "beneficial_switch_rate": _safe_ratio(beneficial, total),
        "harmful_switch_rate": _safe_ratio(harmful, total),
        "neutral_rate": _safe_ratio(neutral, total),
        "normalised_regret_change_candidate_minus_baseline": _quantiles(changes),
        "negative_regret_change_is_better": True,
    }


def _eta_distribution(rows: Sequence[Mapping], key: str) -> dict:
    values = [float(row[key]) for row in rows]
    result = _quantiles(values)
    by_load = {}
    for load in sorted({str(row["load"]) for row in rows}):
        by_load[load] = _quantiles([
            float(row[key]) for row in rows if str(row["load"]) == load
        ])
    by_seed = {}
    for seed in sorted({int(row["seed"]) for row in rows}):
        by_seed[str(seed)] = _quantiles([
            float(row[key]) for row in rows if int(row["seed"]) == seed
        ])
    result["by_load"] = by_load
    result["by_seed"] = by_seed
    return result


def _oracle_eta_rows(
    groups: Sequence[Sequence[Mapping]],
    *,
    fallback: float,
) -> list[dict]:
    result = []
    for members in groups:
        true = torch.stack([
            torch.as_tensor(row["true_station_relief"], dtype=torch.float64)
            for row in members
        ])
        eta, unconstrained, clipped = _eta_for_group(
            members, true, fallback=fallback
        )
        result.append({
            "seed": int(members[0]["seed"]),
            "load": str(members[0]["load"]),
            "eta_context_oracle": eta,
            "eta_context_oracle_unconstrained": unconstrained,
            "eta_context_oracle_clipped": clipped,
        })
    return result


def _effect_alignment(rows: Sequence[Mapping]) -> dict:
    true_station = np.concatenate([
        np.asarray(row["true_candidate_station_residual"], dtype=float).reshape(-1)
        for row in rows
    ])
    predicted_station = np.concatenate([
        np.asarray(row["predicted_candidate_station_residual"], dtype=float).reshape(-1)
        for row in rows
    ])
    true_groups = []
    predicted_groups = []
    true_centered = []
    predicted_centered = []
    for row in rows:
        true = np.asarray(row["true_candidate_raw_effect"], dtype=float)
        predicted = np.asarray(row["predicted_candidate_raw_effect"], dtype=float)
        true_groups.append(true)
        predicted_groups.append(predicted)
        true_centered.extend((true - true.mean()).tolist())
        predicted_centered.extend((predicted - predicted.mean()).tolist())
    result = {
        "station_residual_pearson": _pearson(true_station, predicted_station),
        "station_residual_spearman": _spearman(true_station, predicted_station),
        "group_centered_raw_effect_pearson": _pearson(
            true_centered, predicted_centered
        ),
        "group_centered_raw_effect_spearman": _spearman(
            true_centered, predicted_centered
        ),
        "true_group_centered_raw_effect": _quantiles(true_centered),
        "predicted_group_centered_raw_effect": _quantiles(predicted_centered),
    }
    result.update(_pairwise_effect_concordance(true_groups, predicted_groups))
    return result


def _action_metric_delta(candidate: Mapping, baseline: Mapping) -> dict:
    candidate_c = candidate["continuous_group_range"]
    baseline_c = baseline["continuous_group_range"]

    def value(row: Mapping, key: str) -> Optional[float]:
        raw = row.get(key)
        return None if raw is None else float(raw)

    candidate_regret = (
        candidate_c.get("normalised_selection_regret") or {}
    ).get("mean")
    baseline_regret = (
        baseline_c.get("normalised_selection_regret") or {}
    ).get("mean")
    deltas = {
        "pairwise": None,
        "top1": None,
        "normalised_regret": None,
        "normalised_rmse": None,
    }
    for key in ("all_non_tie_pair_concordance", "top1_min_drift_accuracy"):
        left = value(candidate_c, key)
        right = value(baseline_c, key)
        output_key = "pairwise" if key.startswith("all_non") else "top1"
        if left is not None and right is not None:
            deltas[output_key] = left - right
    if candidate_regret is not None and baseline_regret is not None:
        deltas["normalised_regret"] = (
            float(candidate_regret) - float(baseline_regret)
        )
    candidate_nrmse = value(candidate_c, "normalised_rmse")
    baseline_nrmse = value(baseline_c, "normalised_rmse")
    if candidate_nrmse is not None and baseline_nrmse is not None:
        deltas["normalised_rmse"] = candidate_nrmse - baseline_nrmse
    checks = {
        "pairwise_non_harmful": (
            deltas["pairwise"] is not None and deltas["pairwise"] >= 0.0
        ),
        "top1_non_harmful": (
            deltas["top1"] is not None and deltas["top1"] >= 0.0
        ),
        "normalised_regret_non_harmful": (
            deltas["normalised_regret"] is not None
            and deltas["normalised_regret"] <= 0.0
        ),
    }
    strict_improvement = bool(
        (deltas["pairwise"] is not None and deltas["pairwise"] > 0.0)
        or (deltas["top1"] is not None and deltas["top1"] > 0.0)
        or (
            deltas["normalised_regret"] is not None
            and deltas["normalised_regret"] < 0.0
        )
    )
    return {
        "candidate_minus_baseline": deltas,
        "action_guardrail_checks": checks,
        "action_guardrails_passed": all(checks.values()),
        "at_least_one_action_metric_strictly_improves": strict_improvement,
        "zero_is_non_harmful_no_minimum_gap_is_used": True,
    }


def diagnose(
    *,
    train_rows: Sequence[Mapping],
    val_rows: Sequence[Mapping],
    head: AnalyticWorkResidualHead,
    schema: Mapping,
    device: str,
    bootstrap_repeats: int,
    random_seed: int,
) -> dict:
    train_groups = _groups(train_rows)
    val_groups = _groups(val_rows)
    global_fit = _group_equal_eta(
        train_groups,
        lambda members: torch.stack([
            torch.as_tensor(row["true_station_relief"], dtype=torch.float64)
            for row in members
        ]),
    )
    global_eta = float(global_fit["eta"])
    train_oracle_eta_rows = _oracle_eta_rows(
        train_groups, fallback=global_eta
    )
    capacity = torch.as_tensor(
        schema["work_capacity"], dtype=torch.float64
    ).flatten()
    work_weight = float(schema["work_weight"])

    records = defaultdict(list)
    decomposition_rows = []
    sse = defaultdict(list)
    oracle_clips = c_projection_clips = 0
    for members in val_groups:
        nominal = torch.stack([
            torch.as_tensor(row["nominal_station_relief"], dtype=torch.float64)
            for row in members
        ])
        true = torch.stack([
            torch.as_tensor(row["true_station_relief"], dtype=torch.float64)
            for row in members
        ])
        c_full = _c_prediction(head, members, device=device)
        eta_oracle, eta_oracle_raw, oracle_clipped = _eta_for_group(
            members, true, fallback=global_eta
        )
        eta_c, eta_c_raw, c_clipped = _eta_for_group(
            members, c_full, fallback=global_eta
        )
        oracle_clips += int(oracle_clipped)
        c_projection_clips += int(c_clipped)

        reliefs = {
            "A_parameter_free_analytic": nominal,
            "A_eta_global_train_only": global_eta * nominal,
            "A_eta_context_oracle_diagnostic": eta_oracle * nominal,
            "C_shared_efficiency_projection": eta_c * nominal,
            "C_full_candidate_residual": c_full,
        }
        group_records = {}
        for name, relief in reliefs.items():
            record = _group_record(
                members,
                relief,
                capacity=capacity,
                work_weight=work_weight,
            )
            records[name].append(record)
            group_records[name] = record

        global_prediction = reliefs["A_eta_global_train_only"]
        oracle_prediction = reliefs["A_eta_context_oracle_diagnostic"]
        c_shared = reliefs["C_shared_efficiency_projection"]
        sse["global_shared_efficiency"].append(float(torch.square(
            true - global_prediction
        ).mean()))
        sse["oracle_context_shared_efficiency"].append(float(torch.square(
            true - oracle_prediction
        ).mean()))
        sse["C_shared_efficiency_projection"].append(float(torch.square(
            true - c_shared
        ).mean()))
        sse["C_full_candidate_residual"].append(float(torch.square(
            true - c_full
        ).mean()))

        decomposition_rows.append({
            "seed": int(members[0]["seed"]),
            "load": str(members[0]["load"]),
            "group_key": members[0]["group_key"],
            "candidate_count": len(members),
            "eta_context_oracle": eta_oracle,
            "eta_context_oracle_unconstrained": eta_oracle_raw,
            "eta_C_shared_projection": eta_c,
            "eta_C_shared_projection_unconstrained": eta_c_raw,
            "true_candidate_station_residual": (
                true - oracle_prediction
            ).numpy(),
            "predicted_candidate_station_residual": (
                c_full - c_shared
            ).numpy(),
            "target_raw": group_records[
                "A_parameter_free_analytic"
            ]["target_raw"],
            "global_raw": group_records[
                "A_eta_global_train_only"
            ]["predicted_raw"],
            "oracle_context_raw": group_records[
                "A_eta_context_oracle_diagnostic"
            ]["predicted_raw"],
            "C_shared_raw": group_records[
                "C_shared_efficiency_projection"
            ]["predicted_raw"],
            "C_full_raw": group_records[
                "C_full_candidate_residual"
            ]["predicted_raw"],
            "true_candidate_raw_effect": (
                group_records["A_parameter_free_analytic"]["target_raw"]
                - group_records[
                    "A_eta_context_oracle_diagnostic"
                ]["predicted_raw"]
            ),
            "predicted_candidate_raw_effect": (
                group_records["C_full_candidate_residual"]["predicted_raw"]
                - group_records[
                    "C_shared_efficiency_projection"
                ]["predicted_raw"]
            ),
        })

    metrics = {
        name: _evaluate_records(
            rows,
            bootstrap_repeats=bootstrap_repeats,
            random_seed=random_seed,
        )
        for name, rows in records.items()
    }
    sse_mean = {name: float(np.mean(values)) for name, values in sse.items()}
    global_sse = sse_mean["global_shared_efficiency"]
    oracle_sse = sse_mean["oracle_context_shared_efficiency"]
    c_shared_sse = sse_mean["C_shared_efficiency_projection"]
    c_full_sse = sse_mean["C_full_candidate_residual"]

    alignment = _effect_alignment(decomposition_rows)
    alignment["by_load"] = {
        load: _effect_alignment([
            row for row in decomposition_rows if row["load"] == load
        ])
        for load in sorted({row["load"] for row in decomposition_rows})
    }
    alignment["by_seed"] = {
        str(seed): _effect_alignment([
            row for row in decomposition_rows if int(row["seed"]) == seed
        ])
        for seed in sorted({int(row["seed"]) for row in decomposition_rows})
    }

    c_vs_shared = _action_metric_delta(
        metrics["C_full_candidate_residual"],
        metrics["C_shared_efficiency_projection"],
    )
    c_seed_checks = {}
    for seed in sorted(metrics["C_full_candidate_residual"]["by_seed"]):
        full = {"continuous_group_range": metrics[
            "C_full_candidate_residual"
        ]["by_seed"][seed]}
        shared = {"continuous_group_range": metrics[
            "C_shared_efficiency_projection"
        ]["by_seed"][seed]}
        c_seed_checks[seed] = _action_metric_delta(full, shared)
    candidate_supported = bool(
        c_vs_shared["action_guardrails_passed"]
        and c_vs_shared["at_least_one_action_metric_strictly_improves"]
        and c_seed_checks
        and all(row["action_guardrails_passed"] for row in c_seed_checks.values())
    )

    return {
        "global_efficiency_fit": global_fit,
        "context_efficiency": {
            "oracle_role": (
                "label_visible diagnostic upper bound; never an online gate or "
                "deployable estimator"
            ),
            "C_projection_role": (
                "candidate-set-transductive projection fitted to C predictions "
                "inside each evaluated group; it is not an independently "
                "trained context-efficiency head and may change when the "
                "candidate superset changes"
            ),
            "train_oracle_eta_distribution": _eta_distribution(
                train_oracle_eta_rows, "eta_context_oracle"
            ),
            "validation_oracle_eta_distribution": _eta_distribution(
                decomposition_rows, "eta_context_oracle"
            ),
            "C_projection_eta_distribution": _eta_distribution(
                decomposition_rows, "eta_C_shared_projection"
            ),
            "oracle_eta_clip_rate": _safe_ratio(
                oracle_clips, len(decomposition_rows)
            ),
            "C_projection_eta_clip_rate": _safe_ratio(
                c_projection_clips, len(decomposition_rows)
            ),
        },
        "predictor_metrics": metrics,
        "station_relief_sse_decomposition": {
            "group_equal_mean_sse": sse_mean,
            "oracle_context_gain_fraction_over_global": _safe_ratio(
                global_sse - oracle_sse, global_sse
            ),
            "C_candidate_gain_fraction_over_C_shared": _safe_ratio(
                c_shared_sse - c_full_sse, c_shared_sse
            ),
            "positive_gain_means_lower_station_relief_error": True,
        },
        "candidate_residual_alignment": alignment,
        "selection_switch_audit": {
            "C_full_vs_C_shared": _selection_switch_audit(
                decomposition_rows,
                baseline_key="C_shared_raw",
                candidate_key="C_full_raw",
            ),
            "oracle_context_vs_global": _selection_switch_audit(
                decomposition_rows,
                baseline_key="global_raw",
                candidate_key="oracle_context_raw",
            ),
        },
        "action_comparisons": {
            "A_eta_global_vs_A": _action_metric_delta(
                metrics["A_eta_global_train_only"],
                metrics["A_parameter_free_analytic"],
            ),
            "oracle_context_vs_global_eta": _action_metric_delta(
                metrics["A_eta_context_oracle_diagnostic"],
                metrics["A_eta_global_train_only"],
            ),
            "C_shared_vs_global_eta": _action_metric_delta(
                metrics["C_shared_efficiency_projection"],
                metrics["A_eta_global_train_only"],
            ),
            "C_full_vs_C_shared": c_vs_shared,
            "C_full_vs_C_shared_by_seed": c_seed_checks,
        },
        "development_decision_support": {
            "current_C_candidate_residual_action_supported": candidate_supported,
            "support_contract": (
                "C_full must be non-harmful versus C_shared projection on "
                "pairwise, top1 and normalised regret overall and in every "
                "validation seed cluster, with at least one strict overall "
                "action-metric improvement; no minimum gap is imposed"
            ),
            "phase_c_ood_cause_evaluated": False,
            "phase_c_status": (
                "NOT_JUSTIFIED_BY_THIS_OFFLINE_DIAGNOSTIC_ALONE"
            ),
            "recommended_next_branch": (
                "RETAIN_CANDIDATE_RESIDUAL_FOR_NEW_HELD_OUT_TEST"
                if candidate_supported
                else "SHARED_EFFICIENCY_FIRST_THEN_OOD_CAUSAL_ATTRIBUTION"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-data", nargs="+", required=True)
    parser.add_argument("--val-data", nargs="+", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--legacy-head", default=None)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260719)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.horizon <= 0 or args.bootstrap_repeats < 0:
        raise SystemExit("horizon must be positive and bootstrap repeats non-negative")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite diagnostic output: {output}"
        )

    first_samples, _ = _load_samples_at_horizon(
        [args.train_data[0]], required_horizon=args.horizon
    )
    schema = _work_schema(first_samples)
    model, model_config = load_frozen_world_model(
        args.world_model,
        first_samples[0],
        len(schema["station_ids"]),
        args.device,
    )
    transition_steps = int(model.transition.step_embed.num_embeddings)
    if args.horizon > transition_steps:
        raise SystemExit(
            f"H={args.horizon} exceeds the frozen World Model's "
            f"{transition_steps} transition-step embeddings"
        )
    print(
        "precomputing frozen-WM features for efficiency decomposition: "
        f"H={args.horizon}"
    )
    train_rows, train_manifest, train_audit = _precompute_paths(
        model,
        args.train_data,
        expected_schema=schema,
        device=args.device,
        horizon=args.horizon,
    )
    val_rows, val_manifest, val_audit = _precompute_paths(
        model,
        args.val_data,
        expected_schema=schema,
        device=args.device,
        horizon=args.horizon,
    )
    overlap = sorted(
        set(train_manifest["simulation_seeds"])
        & set(val_manifest["simulation_seeds"])
    )
    if overlap:
        raise SystemExit(f"train/validation simulation_seed leakage: {overlap}")
    del first_samples, model

    head, head_payload = AnalyticWorkResidualHead.from_checkpoint(
        args.head, map_location=args.device
    )
    head_horizon = int((head_payload.get("semantics") or {}).get("horizon", -1))
    if head_horizon != args.horizon:
        raise SystemExit(
            f"head horizon H={head_horizon} differs from requested H={args.horizon}"
        )
    if dict(head_payload.get("work_schema") or {}) != dict(schema):
        raise SystemExit("head work schema differs from supplied data")
    world_model_hash = _sha256_file(args.world_model)
    expected_hash = (head_payload.get("base_world_model") or {}).get("sha256")
    if expected_hash is not None and expected_hash != world_model_hash:
        raise SystemExit("head and supplied World Model checkpoint differ")

    result = diagnose(
        train_rows=train_rows,
        val_rows=val_rows,
        head=head,
        schema=schema,
        device=args.device,
        bootstrap_repeats=args.bootstrap_repeats,
        random_seed=args.random_seed,
    )
    legacy = None
    if args.legacy_head:
        legacy_head, legacy_payload = WorkDriftHead.from_checkpoint(
            args.legacy_head, map_location=args.device
        )
        legacy_expected = (
            legacy_payload.get("base_world_model") or {}
        ).get("sha256")
        if legacy_expected is not None and legacy_expected != world_model_hash:
            raise SystemExit("legacy head and supplied World Model checkpoint differ")
        legacy = {
            "path": os.path.abspath(args.legacy_head),
            "sha256": _sha256_file(args.legacy_head),
            "schema_version": legacy_payload.get("schema_version"),
            "metrics": evaluate_rows(
                legacy_head,
                _legacy_rows(val_rows),
                device=args.device,
                bootstrap_repeats=args.bootstrap_repeats,
                random_seed=args.random_seed,
            ),
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "role": "DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_FORMAL_CERTIFICATION",
        "online_ready": False,
        "abc_definitions": ABC_DEFINITIONS,
        "semantics": {
            "fixed_potential": "quadratic analytic L_work",
            "horizon": args.horizon,
            "rollout_continuation_mode": "isolated",
            "current_demand_only": True,
            "unknown_future_orders": False,
            "continuation_policy": False,
            "td_tail": False,
            "greedy_or_external_policy": False,
            "hard_gap_gate": False,
            "load_gate": False,
            "context_oracle_is_label_visible": True,
            "C_shared_projection_is_candidate_set_transductive": True,
            "candidate_supersets_above_collected_top_m_10_verified": False,
            "phase_c_is_not_model_C": True,
        },
        "world_model": {
            "path": os.path.abspath(args.world_model),
            "sha256": world_model_hash,
            "model_config": model_config,
            "transition_steps": transition_steps,
        },
        "head": {
            "path": os.path.abspath(args.head),
            "sha256": _sha256_file(args.head),
            "schema_version": head_payload.get("schema_version"),
            "training_seed": (
                head_payload.get("training_config") or {}
            ).get("seed"),
        },
        "legacy_B": legacy or {
            "status": "NOT_EVALUATED_NO_LEGACY_HEAD_SUPPLIED"
        },
        "work_schema": schema,
        "data_manifest": {
            "train": train_manifest,
            "validation": val_manifest,
            "train_audit": train_audit,
            "validation_audit": val_audit,
            "seed_disjoint": True,
        },
        "diagnostic": result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {output}")
    print("role: DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_FORMAL_CERTIFICATION")
    print("global eta =", result["global_efficiency_fit"]["eta"])
    for name in (
        "A_parameter_free_analytic",
        "A_eta_global_train_only",
        "A_eta_context_oracle_diagnostic",
        "C_shared_efficiency_projection",
        "C_full_candidate_residual",
    ):
        metric = result["predictor_metrics"][name]["continuous_group_range"]
        print(
            name,
            "NRMSE=", metric.get("normalised_rmse"),
            "pairwise=", metric.get("all_non_tie_pair_concordance"),
            "top1=", metric.get("top1_min_drift_accuracy"),
            "regret=", (
                metric.get("normalised_selection_regret") or {}
            ).get("mean"),
        )
    decision = result["development_decision_support"]
    print(
        "candidate residual action supported =",
        decision["current_C_candidate_residual_action_supported"],
    )
    print("next branch =", decision["recommended_next_branch"])
    print("note: this diagnostic cannot by itself justify Phase C")


if __name__ == "__main__":
    main()
