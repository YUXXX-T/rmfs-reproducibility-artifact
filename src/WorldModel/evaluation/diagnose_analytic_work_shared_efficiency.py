"""Diagnose shared work-relief efficiency versus candidate-specific residuals.

This is a development-only analysis for the analytic work-relief experiment.
It does not train or replace the Lyapunov potential.  The fixed quadratic
``L_work`` and the post-action analytic work ledger remain unchanged.

For every fixed-context candidate group it decomposes the true relief into::

    true relief = eta_context * nominal relief + candidate residual

where ``eta_context`` is one scalar shared by every candidate in the group.
The script then asks three separate questions:

1. Is one train-only global efficiency scalar already sufficient?
2. Can a context-shared scalar be predicted without using endpoint labels?
3. After removing that shared scalar, is the within-group residual stable and
   predictable from current-known previews, frozen-WM features, or only from
   oracle rollout interaction trajectories?

The final probe is deliberately diagnostic.  Its raw residual regressors do
not constitute a physically closed endpoint head and are never marked online
ready.  Future orders, continuation policies, TD targets and greedy policies
are outside this script's contract.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from WorldModel.core.analytic_work_residual_head import (
    AnalyticWorkResidualHead,
)
from WorldModel.training.train_analytic_work_residual_head import (
    _load_samples_at_horizon,
    _precompute,
)
from WorldModel.training.train_td_risk_v import load_frozen_world_model
from WorldModel.training.train_work_drift_head import (
    _aggregate_group_records,
    _bootstrap_metrics,
    _expand_paths,
    _groups,
    _manifest_from_compact_rows,
    _sha256_file,
    _work_schema,
)


SCHEMA_VERSION = "analytic_work_shared_efficiency_residual_diagnostic_v1"
PREVIEW_NAMES = (
    "eta",
    "eta_bin",
    "arrival_delta_preview",
    "route_conflict_preview",
    "route_length_preview",
)
DEFAULT_RIDGE_ALPHAS = (
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
)


def _array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)


def _normalise_group(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.size == 0:
        raise ValueError("cannot normalise an empty candidate group")
    value_range = float(result.max() - result.min())
    if value_range == 0.0:
        return np.zeros_like(result)
    return (result - result.mean()) / value_range


def _quantiles(values: Sequence[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or x.std() <= 0.0 or y.std() <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def _finite_scalar(mapping: Mapping, name: str) -> tuple[float, float]:
    value = mapping.get(name)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0, 0.0
    if not math.isfinite(result):
        return 0.0, 0.0
    return result, 1.0


def _attach_diagnostic_fields(
    rows: Sequence[dict],
    samples: Sequence[Mapping],
    *,
    horizon: int,
) -> dict:
    if len(rows) != len(samples):
        raise ValueError("compact rows and raw samples do not align")
    preview_present = defaultdict(int)
    traffic_schemas = set()
    traffic_available = 0
    for row, sample in zip(rows, samples):
        candidate = sample.get("candidate_info")
        candidate = candidate if isinstance(candidate, Mapping) else {}
        values = []
        masks = []
        for name in PREVIEW_NAMES:
            value, present = _finite_scalar(candidate, name)
            values.append(value)
            masks.append(present)
            preview_present[name] += int(present > 0.5)
        row["candidate_preview_values"] = torch.tensor(
            values, dtype=torch.float32
        )
        row["candidate_preview_masks"] = torch.tensor(
            masks, dtype=torch.float32
        )

        names = tuple(str(name) for name in (
            sample.get("lyapunov_l0_traffic_trajectory_names") or ()
        ))
        trajectory = sample.get("lyapunov_l0_traffic_trajectory")
        if trajectory is None or not names:
            row["oracle_traffic_prefix"] = None
            row["oracle_traffic_names"] = ()
            continue
        tensor = torch.as_tensor(trajectory, dtype=torch.float32)
        if tensor.ndim != 2 or tensor.size(1) != len(names):
            raise ValueError("invalid Lyapunov traffic trajectory shape")
        if tensor.size(0) < horizon:
            raise ValueError("traffic trajectory is shorter than requested horizon")
        row["oracle_traffic_prefix"] = tensor[:horizon].clone()
        row["oracle_traffic_names"] = names
        traffic_schemas.add(names)
        traffic_available += 1
    if len(traffic_schemas) > 1:
        raise ValueError("mixed traffic trajectory schemas")
    return {
        "preview_fields": list(PREVIEW_NAMES),
        "preview_present_samples": dict(preview_present),
        "traffic_available_samples": int(traffic_available),
        "traffic_names": list(next(iter(traffic_schemas), ())),
        "traffic_prefix_horizon": int(horizon),
    }


def _precompute_diagnostic_paths(
    model,
    paths: Sequence[str],
    *,
    expected_schema: Mapping,
    device: str,
    horizon: int,
) -> tuple[list[dict], dict, dict]:
    rows: list[dict] = []
    source_counts = {}
    collection_horizons = set()
    field_audits = []
    expanded = _expand_paths(paths)
    for file_index, path in enumerate(expanded, 1):
        samples, audit = _load_samples_at_horizon(
            [path], required_horizon=horizon
        )
        observed_schema = _work_schema(samples)
        if observed_schema != dict(expected_schema):
            raise ValueError(f"{path}: work schema differs from diagnostic schema")
        compact = _precompute(
            model,
            samples,
            station_ids=expected_schema["station_ids"],
            work_capacity=expected_schema["work_capacity"],
            work_weight=expected_schema["work_weight"],
            device=device,
            horizon=horizon,
        )
        field_audits.append(_attach_diagnostic_fields(
            compact, samples, horizon=horizon
        ))
        rows.extend(compact)
        source_counts.update(audit["source_counts"])
        collection_horizons.update(audit["collection_horizons"])
        print(
            f"  diagnostic precompute {file_index}/{len(expanded)}: "
            f"{os.path.basename(os.path.dirname(path))} "
            f"candidates={len(compact)}"
        )
        del samples, compact
    if any(len(group) < 2 for group in _groups(rows)):
        raise ValueError("diagnostic precompute produced an incomplete group")
    traffic_names = {
        tuple(audit["traffic_names"])
        for audit in field_audits if audit["traffic_names"]
    }
    if len(traffic_names) > 1:
        raise ValueError("source files have different traffic schemas")
    return rows, _manifest_from_compact_rows(rows), {
        "source_counts": source_counts,
        "source_files_processed_sequentially": len(expanded),
        "required_horizon": int(horizon),
        "collection_horizons": sorted(collection_horizons),
        "candidate_groups": len(_groups(rows)),
        "preview_fields": list(PREVIEW_NAMES),
        "traffic_names": list(next(iter(traffic_names), ())),
        "traffic_available_samples": int(sum(
            audit["traffic_available_samples"] for audit in field_audits
        )),
    }


def _group_relief_arrays(group: Sequence[Mapping]) -> tuple[np.ndarray, np.ndarray]:
    nominal = np.stack([
        _array(row["nominal_station_relief"]) for row in group
    ])
    true = np.stack([
        _array(row["true_station_relief"]) for row in group
    ])
    return nominal, true


def _least_squares_eta(
    nominal: np.ndarray,
    true: np.ndarray,
    *,
    clip: bool = True,
) -> tuple[Optional[float], float]:
    denominator = float(np.square(nominal).mean())
    if denominator <= 1e-15:
        return None, denominator
    eta = float((nominal * true).mean() / denominator)
    if clip:
        eta = float(np.clip(eta, 0.0, 1.0))
    return eta, denominator


def fit_global_efficiency(groups: Sequence[Sequence[Mapping]]) -> dict:
    """Fit one clipped scalar with equal total weight per candidate group."""

    numerators = []
    denominators = []
    mass_ratios = []
    defined = 0
    for group in groups:
        nominal, true = _group_relief_arrays(group)
        denominator = float(np.square(nominal).mean())
        if denominator <= 1e-15:
            continue
        numerators.append(float((nominal * true).mean()))
        denominators.append(denominator)
        nominal_mass = float(nominal.sum())
        if abs(nominal_mass) > 1e-15:
            mass_ratios.append(float(true.sum() / nominal_mass))
        defined += 1
    if not denominators or sum(denominators) <= 1e-15:
        raise ValueError("global efficiency is undefined: no nominal relief")
    eta_unclipped = float(sum(numerators) / sum(denominators))
    return {
        "eta": float(np.clip(eta_unclipped, 0.0, 1.0)),
        "eta_unclipped": eta_unclipped,
        "clipped": bool(eta_unclipped < 0.0 or eta_unclipped > 1.0),
        "defined_groups": int(defined),
        "total_groups": int(len(groups)),
        "group_equal_weighting": True,
        "group_mass_ratio": _quantiles(mass_ratios),
    }


def _group_eta_targets(groups: Sequence[Sequence[Mapping]]) -> tuple[np.ndarray, np.ndarray]:
    eta = []
    defined = []
    for group in groups:
        nominal, true = _group_relief_arrays(group)
        value, _ = _least_squares_eta(nominal, true)
        eta.append(np.nan if value is None else value)
        defined.append(value is not None)
    return np.asarray(eta, dtype=np.float64), np.asarray(defined, dtype=bool)


def _efficiency_summary(
    groups: Sequence[Sequence[Mapping]],
    eta: np.ndarray,
    defined: np.ndarray,
) -> dict:
    values = eta[defined]
    by_load = {}
    by_seed = {}
    for load in sorted({str(group[0]["load"]) for group in groups}):
        selected = [
            eta[index] for index, group in enumerate(groups)
            if defined[index] and str(group[0]["load"]) == load
        ]
        by_load[load] = _quantiles(selected)
    for seed in sorted({int(group[0]["seed"]) for group in groups}):
        selected = [
            eta[index] for index, group in enumerate(groups)
            if defined[index] and int(group[0]["seed"]) == seed
        ]
        by_seed[str(seed)] = _quantiles(selected)
    return {
        "defined_groups": int(defined.sum()),
        "undefined_zero_nominal_groups": int((~defined).sum()),
        "eta_group_oracle": _quantiles(values),
        "by_load": by_load,
        "by_seed": by_seed,
        "role": "ORACLE_DECOMPOSITION_TARGET_NOT_AN_ONLINE_INPUT",
    }


def _eta_prediction(
    group: Sequence[Mapping],
    eta: float,
    *,
    capacity: np.ndarray,
    work_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    raw = []
    endpoints = []
    for row in group:
        start = _array(row["current_station_work"])
        post = _array(row["post_action_station_work"])
        nominal = _array(row["nominal_station_relief"])
        endpoint = np.maximum(post - float(eta) * nominal, 0.0)
        drift = 0.5 * float(work_weight) * float(
            np.square(endpoint / capacity).sum()
            - np.square(start / capacity).sum()
        )
        raw.append(drift)
        endpoints.append(endpoint)
    return np.asarray(raw, dtype=np.float64), np.stack(endpoints)


def _prediction_records(
    groups: Sequence[Sequence[Mapping]],
    predictions: Sequence[tuple[np.ndarray, Optional[np.ndarray]]],
) -> list[dict]:
    if len(groups) != len(predictions):
        raise ValueError("group and prediction counts differ")
    records = []
    for group, (predicted_raw, predicted_endpoint) in zip(groups, predictions):
        target_raw = np.asarray([
            float(row["target_raw_work_drift"]) for row in group
        ], dtype=np.float64)
        predicted_raw = np.asarray(predicted_raw, dtype=np.float64)
        if target_raw.shape != predicted_raw.shape:
            raise ValueError("candidate prediction shape differs from truth")
        if predicted_endpoint is None:
            endpoint_error = np.empty((0,), dtype=np.float64)
        else:
            endpoint = np.stack([
                _array(row["endpoint_station_work"]) for row in group
            ])
            endpoint_error = np.abs(
                np.asarray(predicted_endpoint, dtype=np.float64) - endpoint
            )
        records.append({
            "target_raw": target_raw,
            "predicted_raw": predicted_raw,
            "target_range": _normalise_group(target_raw),
            "predicted_range": _normalise_group(predicted_raw),
            "endpoint_absolute_error": endpoint_error,
            "seed": int(group[0]["seed"]),
            "load": str(group[0]["load"]),
            "group_key": group[0]["group_key"],
            "candidate_count": len(group),
        })
    return records


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
    result["by_load"] = {
        load: _aggregate_group_records([
            row for row in records if str(row["load"]) == load
        ])["continuous_group_range"]
        for load in sorted({str(row["load"]) for row in records})
    }
    result["by_seed"] = {
        str(seed): _aggregate_group_records([
            row for row in records if int(row["seed"]) == seed
        ])["continuous_group_range"]
        for seed in sorted({int(row["seed"]) for row in records})
    }
    result["by_candidate_count"] = {
        str(count): _aggregate_group_records([
            row for row in records if int(row["candidate_count"]) == count
        ])["continuous_group_range"]
        for count in sorted({int(row["candidate_count"]) for row in records})
    }
    return result


def _group_action_delta(reference: Mapping, candidate: Mapping) -> dict:
    truth = np.asarray(reference["target_raw"], dtype=np.float64)
    ref = np.asarray(reference["predicted_raw"], dtype=np.float64)
    cand = np.asarray(candidate["predicted_raw"], dtype=np.float64)
    true_min = float(truth.min())
    true_best = set(np.flatnonzero(truth == true_min).tolist())

    def action_metrics(prediction: np.ndarray) -> tuple[float, float, Optional[float]]:
        chosen = int(np.flatnonzero(prediction == prediction.min())[0])
        top1 = float(chosen in true_best)
        raw_range = float(truth.max() - truth.min())
        regret = (
            float((truth[chosen] - true_min) / raw_range)
            if raw_range > 0.0 else None
        )
        pair = []
        for left in range(len(truth)):
            for right in range(left + 1, len(truth)):
                true_gap = truth[left] - truth[right]
                if true_gap == 0.0:
                    continue
                pred_gap = prediction[left] - prediction[right]
                pair.append(0.5 if pred_gap == 0.0 else float(
                    np.sign(pred_gap) == np.sign(true_gap)
                ))
        return top1, float(np.mean(pair)) if pair else np.nan, regret

    ref_top1, ref_pair, ref_regret = action_metrics(ref)
    cand_top1, cand_pair, cand_regret = action_metrics(cand)
    return {
        "top1_delta": cand_top1 - ref_top1,
        "pairwise_delta": cand_pair - ref_pair,
        "normalised_regret_improvement": (
            ref_regret - cand_regret
            if ref_regret is not None and cand_regret is not None else np.nan
        ),
        "seed": int(reference["seed"]),
    }


def _paired_action_comparison(
    reference: Sequence[Mapping],
    candidate: Sequence[Mapping],
    *,
    bootstrap_repeats: int,
    random_seed: int,
) -> dict:
    if len(reference) != len(candidate):
        raise ValueError("paired action records are not aligned")
    rows = [
        _group_action_delta(ref, cand)
        for ref, cand in zip(reference, candidate)
    ]
    result = {
        name: _quantiles([row[name] for row in rows])
        for name in (
            "top1_delta",
            "pairwise_delta",
            "normalised_regret_improvement",
        )
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    if bootstrap_repeats > 0 and len(seeds) >= 2:
        by_seed = {
            seed: [row for row in rows if int(row["seed"]) == seed]
            for seed in seeds
        }
        rng = np.random.default_rng(random_seed)
        bootstrap = defaultdict(list)
        for _ in range(bootstrap_repeats):
            sampled = rng.choice(seeds, size=len(seeds), replace=True)
            selected = [row for seed in sampled for row in by_seed[int(seed)]]
            for name in (
                "top1_delta",
                "pairwise_delta",
                "normalised_regret_improvement",
            ):
                values = np.asarray([
                    row[name] for row in selected
                ], dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size:
                    bootstrap[name].append(float(values.mean()))
        result["ci95_seed_cluster_bootstrap"] = {
            name: [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ] if values else None
            for name, values in bootstrap.items()
        }
    result["positive_means_favour_candidate"] = True
    return result


def _within_group_direction_explained(
    records: Sequence[Mapping],
) -> dict:
    squared_truth = []
    squared_error = []
    range_ratios = []
    for row in records:
        truth = np.asarray(row["target_raw"], dtype=np.float64)
        predicted = np.asarray(row["predicted_raw"], dtype=np.float64)
        truth = truth - truth.mean()
        predicted = predicted - predicted.mean()
        squared_truth.append(float(np.square(truth).mean()))
        squared_error.append(float(np.square(truth - predicted).mean()))
        true_range = float(truth.max() - truth.min())
        if true_range > 0.0:
            range_ratios.append(float(
                (predicted.max() - predicted.min()) / true_range
            ))
    denominator = float(np.mean(squared_truth)) if squared_truth else 0.0
    error = float(np.mean(squared_error)) if squared_error else 0.0
    return {
        "candidate_group_equal_weighted_direction_r2": (
            float(1.0 - error / denominator) if denominator > 0.0 else None
        ),
        "candidate_group_equal_weighted_mse": error,
        "true_within_group_variance": denominator,
        "predicted_to_true_range_ratio": _quantiles(range_ratios),
    }


def _weighted_standardise(
    features: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalised = weights / weights.sum()
    mean = (features * normalised[:, None]).sum(axis=0)
    variance = (
        np.square(features - mean) * normalised[:, None]
    ).sum(axis=0)
    scale = np.sqrt(variance)
    scale[scale < 1e-10] = 1.0
    return (features - mean) / scale, mean, scale


def _fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
) -> dict:
    if alpha <= 0.0:
        raise ValueError("ridge alpha must be positive")
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if features.ndim != 2 or target.shape != (features.shape[0],):
        raise ValueError("invalid ridge feature/target shape")
    if weights.shape != target.shape or bool((weights <= 0.0).any()):
        raise ValueError("ridge weights must be positive and aligned")
    standardised, mean, scale = _weighted_standardise(features, weights)
    normalised = weights / weights.sum()
    target_mean = float(np.dot(normalised, target))
    root = np.sqrt(normalised)
    design = standardised * root[:, None]
    response = (target - target_mean) * root
    gram = design.T @ design
    projection = design.T @ response
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    coefficient = eigenvectors @ (
        (eigenvectors.T @ projection) / (eigenvalues + float(alpha))
    )
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "target_mean": target_mean,
        "coefficient": coefficient,
        "alpha": float(alpha),
    }


def _predict_ridge(model: Mapping, features: np.ndarray) -> np.ndarray:
    standardised = (
        np.asarray(features, dtype=np.float64) - model["feature_mean"]
    ) / model["feature_scale"]
    return (
        float(model["target_mean"])
        + standardised @ model["coefficient"]
    )


def _select_ridge_by_seed(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    seeds: np.ndarray,
    *,
    alphas: Sequence[float],
) -> tuple[dict, dict]:
    unique = sorted({int(seed) for seed in seeds})
    if len(unique) < 2:
        raise ValueError("ridge development requires at least two train seeds")
    rows = []
    for alpha in alphas:
        squared_error = []
        squared_weight = []
        folds = []
        for held_out in unique:
            train = seeds != held_out
            validation = seeds == held_out
            model = _fit_ridge(
                features[train], target[train], weights[train], alpha=alpha
            )
            predicted = _predict_ridge(model, features[validation])
            fold_weights = weights[validation]
            rmse = float(np.sqrt(np.average(
                np.square(predicted - target[validation]),
                weights=fold_weights,
            )))
            folds.append({"held_out_seed": held_out, "rmse": rmse})
            squared_error.extend(np.square(predicted - target[validation]))
            squared_weight.extend(fold_weights)
        score = float(np.sqrt(np.average(
            np.asarray(squared_error), weights=np.asarray(squared_weight)
        )))
        rows.append({"alpha": float(alpha), "seed_cv_rmse": score, "folds": folds})
    selected = min(rows, key=lambda row: (row["seed_cv_rmse"], row["alpha"]))
    final = _fit_ridge(
        features, target, weights, alpha=float(selected["alpha"])
    )
    return final, {
        "selection": "minimum leave-one-training-seed-out weighted RMSE",
        "selected_alpha": float(selected["alpha"]),
        "selected_seed_cv_rmse": float(selected["seed_cv_rmse"]),
        "alpha_path": rows,
        "validation_seeds_not_used_for_selection": True,
    }


def _latent_dim(group: Sequence[Mapping]) -> int:
    size = int(_array(group[0]["global_context"]).size)
    if size % 5 != 0:
        raise ValueError("frozen-WM global context does not have dimension 5D")
    return size // 5


def _state_only_group_features(
    group: Sequence[Mapping],
    *,
    capacity: np.ndarray,
    latent_dim: int,
) -> tuple[np.ndarray, float]:
    global_context = np.stack([_array(row["global_context"]) for row in group])
    # [mean(z_start), max(z_start), mean(z_end), max(z_end), demand_embedding]
    invariant = np.concatenate((
        global_context[:, : 2 * latent_dim],
        global_context[:, 4 * latent_dim : 5 * latent_dim],
    ), axis=1)
    invariant_error = float(np.abs(invariant - invariant.mean(axis=0)).max())
    start = np.stack([
        _array(row["current_station_work"]) / capacity for row in group
    ])
    invariant_error = max(
        invariant_error,
        float(np.abs(start - start.mean(axis=0)).max()),
    )
    return np.concatenate((invariant.mean(axis=0), start.mean(axis=0))), invariant_error


def _candidate_set_group_features(
    group: Sequence[Mapping],
    *,
    capacity: np.ndarray,
    latent_dim: int,
) -> tuple[np.ndarray, float]:
    state, invariant_error = _state_only_group_features(
        group, capacity=capacity, latent_dim=latent_dim
    )
    global_context = np.stack([_array(row["global_context"]) for row in group])
    endpoint = global_context[:, 2 * latent_dim : 4 * latent_dim]
    physical = []
    for name in (
        "post_action_station_work",
        "nominal_station_relief",
        "available_station_relief",
    ):
        values = np.stack([_array(row[name]) / capacity for row in group])
        physical.extend((values.mean(axis=0), values.std(axis=0)))
    return np.concatenate((
        state,
        endpoint.mean(axis=0),
        endpoint.std(axis=0),
        *physical,
        np.asarray([math.log1p(len(group))], dtype=np.float64),
    )), invariant_error


def _build_group_feature_matrix(
    groups: Sequence[Sequence[Mapping]],
    *,
    capacity: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict]:
    latent_dim = _latent_dim(groups[0])
    features = []
    invariant_errors = []
    for group in groups:
        if mode == "state_only":
            feature, error = _state_only_group_features(
                group, capacity=capacity, latent_dim=latent_dim
            )
        elif mode == "candidate_set":
            feature, error = _candidate_set_group_features(
                group, capacity=capacity, latent_dim=latent_dim
            )
        else:
            raise ValueError(f"unknown shared-efficiency feature mode: {mode}")
        features.append(feature)
        invariant_errors.append(error)
    return np.stack(features), {
        "mode": mode,
        "feature_dimension": int(len(features[0])),
        "latent_dim": int(latent_dim),
        "candidate_invariant_max_abs_error": float(max(invariant_errors)),
        "uses_true_endpoint": False,
        "uses_future_orders": False,
        "uses_load_label": False,
        "candidate_set_aggregate": mode == "candidate_set",
    }


def _context_efficiency_model(
    train_groups: Sequence[Sequence[Mapping]],
    val_groups: Sequence[Sequence[Mapping]],
    train_eta: np.ndarray,
    train_defined: np.ndarray,
    val_eta: np.ndarray,
    val_defined: np.ndarray,
    *,
    capacity: np.ndarray,
    mode: str,
    alphas: Sequence[float],
) -> tuple[np.ndarray, dict]:
    train_features, feature_audit = _build_group_feature_matrix(
        train_groups, capacity=capacity, mode=mode
    )
    val_features, val_feature_audit = _build_group_feature_matrix(
        val_groups, capacity=capacity, mode=mode
    )
    seeds = np.asarray([
        int(group[0]["seed"]) for group in train_groups
    ], dtype=np.int64)
    weights = np.ones(int(train_defined.sum()), dtype=np.float64)
    model, selection = _select_ridge_by_seed(
        train_features[train_defined],
        train_eta[train_defined],
        weights,
        seeds[train_defined],
        alphas=alphas,
    )
    predicted = np.clip(_predict_ridge(model, val_features), 0.0, 1.0)
    target = val_eta[val_defined]
    observed = predicted[val_defined]
    return predicted, {
        "role": "DEVELOPMENT_CONTEXT_SHARED_EFFICIENCY_PROBE",
        "feature_audit": feature_audit,
        "validation_feature_audit": val_feature_audit,
        "ridge_selection": selection,
        "validation_defined_groups": int(val_defined.sum()),
        "eta_prediction": _quantiles(predicted),
        "eta_target": _quantiles(target),
        "eta_rmse": (
            float(np.sqrt(np.mean(np.square(observed - target))))
            if target.size else None
        ),
        "eta_mae": (
            float(np.mean(np.abs(observed - target))) if target.size else None
        ),
        "eta_pearson": _pearson(observed, target),
        "eta_spearman": _spearman(observed, target),
        "validation_labels_not_used_for_fit_or_alpha_selection": True,
    }


def _candidate_feature_vector(
    row: Mapping,
    *,
    latent_dim: int,
    mode: str,
) -> np.ndarray:
    preview = np.concatenate((
        _array(row["candidate_preview_values"]),
        _array(row["candidate_preview_masks"]),
    ))
    global_context = _array(row["global_context"])
    station_context = _array(row["station_context"])
    wm = np.concatenate((
        global_context[2 * latent_dim : 4 * latent_dim],
        station_context[:, latent_dim : 2 * latent_dim].reshape(-1),
        station_context[:, -3:].reshape(-1),
    ))
    if mode == "preview":
        return preview
    if mode == "frozen_wm":
        return wm
    if mode == "frozen_wm_plus_preview":
        return np.concatenate((wm, preview))
    if mode == "oracle_traffic":
        trajectory = row.get("oracle_traffic_prefix")
        if trajectory is None:
            raise ValueError("oracle traffic trajectory is unavailable")
        values = _array(trajectory)
        return np.concatenate((
            values.mean(axis=0),
            values.max(axis=0),
            values[-1],
            values.sum(axis=0),
        ))
    raise ValueError(f"unknown candidate residual feature mode: {mode}")


def _oracle_shared_candidate_target(
    groups: Sequence[Sequence[Mapping]],
    eta: np.ndarray,
    defined: np.ndarray,
    *,
    capacity: np.ndarray,
    work_weight: float,
) -> tuple[list[np.ndarray], dict]:
    targets = []
    ranges = []
    variance = []
    undefined = 0
    for index, group in enumerate(groups):
        value = float(eta[index]) if defined[index] else 0.0
        if not defined[index]:
            undefined += 1
        shared_raw, _ = _eta_prediction(
            group, value, capacity=capacity, work_weight=work_weight
        )
        true_raw = np.asarray([
            float(row["target_raw_work_drift"]) for row in group
        ], dtype=np.float64)
        residual = (
            (true_raw - true_raw.mean())
            - (shared_raw - shared_raw.mean())
        )
        # Enforce the decomposition numerically.  This target contains only
        # within-group action variation and cannot learn a cross-context bias.
        residual -= residual.mean()
        targets.append(residual)
        ranges.append(float(residual.max() - residual.min()))
        variance.append(float(np.square(residual).mean()))
    return targets, {
        "definition": (
            "center(true raw drift) - center(raw drift from oracle group eta)"
        ),
        "group_mean_forced_to_zero": True,
        "undefined_eta_groups_using_zero_relief_fallback": int(undefined),
        "candidate_residual_range": _quantiles(ranges),
        "candidate_residual_within_group_variance": _quantiles(variance),
    }


def _candidate_probe_dataset(
    groups: Sequence[Sequence[Mapping]],
    targets: Sequence[np.ndarray],
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[slice], dict]:
    latent_dim = _latent_dim(groups[0])
    features = []
    values = []
    weights = []
    seeds = []
    slices = []
    offset = 0
    for group, target in zip(groups, targets):
        matrix = np.stack([
            _candidate_feature_vector(row, latent_dim=latent_dim, mode=mode)
            for row in group
        ])
        matrix -= matrix.mean(axis=0, keepdims=True)
        features.append(matrix)
        values.append(np.asarray(target, dtype=np.float64))
        weights.append(np.full(len(group), 1.0 / len(group), dtype=np.float64))
        seeds.append(np.full(len(group), int(group[0]["seed"]), dtype=np.int64))
        slices.append(slice(offset, offset + len(group)))
        offset += len(group)
    return (
        np.concatenate(features),
        np.concatenate(values),
        np.concatenate(weights),
        np.concatenate(seeds),
        slices,
        {
            "mode": mode,
            "feature_dimension": int(features[0].shape[1]),
            "features_centered_within_candidate_group": True,
            "target_centered_within_candidate_group": True,
            "uses_true_endpoint_as_input": False,
            "uses_future_orders": False,
            "oracle_future_trajectory": mode == "oracle_traffic",
        },
    )


def _residual_regression_metrics(
    true_groups: Sequence[np.ndarray],
    predicted_groups: Sequence[np.ndarray],
) -> dict:
    true = []
    predicted = []
    weights = []
    group_nrmse = []
    pairwise = []
    for target, estimate in zip(true_groups, predicted_groups):
        target = np.asarray(target, dtype=np.float64)
        estimate = np.asarray(estimate, dtype=np.float64)
        if target.shape != estimate.shape:
            raise ValueError("candidate residual prediction is misaligned")
        true.extend(target)
        predicted.extend(estimate)
        weights.extend(np.full(len(target), 1.0 / len(target)))
        scale = float(np.sqrt(np.square(target).mean()))
        if scale > 0.0:
            group_nrmse.append(float(
                np.sqrt(np.square(estimate - target).mean()) / scale
            ))
        correct = []
        for left in range(len(target)):
            for right in range(left + 1, len(target)):
                gap = target[left] - target[right]
                if gap == 0.0:
                    continue
                predicted_gap = estimate[left] - estimate[right]
                correct.append(0.5 if predicted_gap == 0.0 else float(
                    np.sign(predicted_gap) == np.sign(gap)
                ))
        if correct:
            pairwise.append(float(np.mean(correct)))
    true_array = np.asarray(true, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    weight_array /= weight_array.sum()
    truth_scale = float(np.sqrt(np.dot(weight_array, np.square(true_array))))
    rmse = float(np.sqrt(np.dot(
        weight_array, np.square(predicted_array - true_array)
    )))
    return {
        "candidate_group_equal_weighted_nrmse": (
            rmse / truth_scale if truth_scale > 0.0 else None
        ),
        "candidate_group_nrmse": _quantiles(group_nrmse),
        "pearson": _pearson(true_array, predicted_array),
        "spearman": _spearman(true_array, predicted_array),
        "pairwise_concordance": (
            float(np.mean(pairwise)) if pairwise else None
        ),
        "true_residual": _quantiles(true_array),
        "predicted_residual": _quantiles(predicted_array),
    }


def _fit_candidate_probe(
    train_groups: Sequence[Sequence[Mapping]],
    val_groups: Sequence[Sequence[Mapping]],
    train_targets: Sequence[np.ndarray],
    val_targets: Sequence[np.ndarray],
    *,
    mode: str,
    alphas: Sequence[float],
) -> tuple[list[np.ndarray], dict]:
    train = _candidate_probe_dataset(train_groups, train_targets, mode=mode)
    val = _candidate_probe_dataset(val_groups, val_targets, mode=mode)
    train_x, train_y, train_w, train_seed, _, train_audit = train
    val_x, _, _, _, val_slices, val_audit = val
    model, selection = _select_ridge_by_seed(
        train_x, train_y, train_w, train_seed, alphas=alphas
    )
    flat = _predict_ridge(model, val_x)
    predicted = []
    for group_slice in val_slices:
        values = np.asarray(flat[group_slice], dtype=np.float64)
        predicted.append(values - values.mean())
    return predicted, {
        "role": "LINEAR_REPRESENTATION_PROBE_DEVELOPMENT_ONLY",
        "train_feature_audit": train_audit,
        "validation_feature_audit": val_audit,
        "ridge_selection": selection,
        "validation": _residual_regression_metrics(val_targets, predicted),
        "not_a_physical_endpoint_head": True,
        "online_ready": False,
    }


def _c_predictions(
    head: AnalyticWorkResidualHead,
    groups: Sequence[Sequence[Mapping]],
    *,
    device: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    result = []
    head.eval()
    with torch.no_grad():
        for group in groups:
            global_context = torch.stack([
                row["global_context"] for row in group
            ]).to(device)
            station_context = torch.stack([
                row["station_context"] for row in group
            ]).to(device)
            start = torch.stack([
                row["current_station_work"] for row in group
            ]).to(device)
            prediction = head.predict_from_features(
                global_context, station_context, start
            )
            result.append((
                _array(prediction.raw_work_drift),
                _array(prediction.endpoint_station_work),
            ))
    return result


def _candidate_component_from_raw(
    raw_predictions: Sequence[np.ndarray],
    oracle_shared_predictions: Sequence[tuple[np.ndarray, Optional[np.ndarray]]],
) -> list[np.ndarray]:
    result = []
    for predicted, (shared, _) in zip(raw_predictions, oracle_shared_predictions):
        predicted = np.asarray(predicted, dtype=np.float64)
        shared = np.asarray(shared, dtype=np.float64)
        component = (
            (predicted - predicted.mean()) - (shared - shared.mean())
        )
        result.append(component - component.mean())
    return result


def _combine_shared_and_candidate(
    shared: Sequence[tuple[np.ndarray, np.ndarray]],
    candidate: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, None]]:
    return [
        (np.asarray(raw) + np.asarray(residual), None)
        for (raw, _), residual in zip(shared, candidate)
    ]


def _input_provenance(paths: Sequence[str]) -> list[dict]:
    return [
        {
            "path": os.path.abspath(path),
            "sha256": _sha256_file(path),
        }
        for path in _expand_paths(paths)
    ]


def _parse_alphas(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("ridge alphas must be positive")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-data", nargs="+", required=True)
    parser.add_argument("--val-data", nargs="+", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--residual-head", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--ridge-alphas",
        type=_parse_alphas,
        default=DEFAULT_RIDGE_ALPHAS,
        help="comma-separated positive train-seed-CV ridge alphas",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.horizon <= 0 or args.bootstrap_repeats < 0:
        raise SystemExit("horizon must be positive and repeats non-negative")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing diagnostic: {output}")

    train_paths = _expand_paths(args.train_data)
    val_paths = _expand_paths(args.val_data)
    first_samples, _ = _load_samples_at_horizon(
        [train_paths[0]], required_horizon=args.horizon
    )
    if not first_samples:
        raise SystemExit("training data are empty")
    schema = _work_schema(first_samples)
    model, model_config = load_frozen_world_model(
        args.world_model,
        first_samples[0],
        len(schema["station_ids"]),
        args.device,
    )
    print(
        "precomputing shared-efficiency diagnostic features: "
        f"train_files={len(train_paths)} val_files={len(val_paths)} "
        f"H={args.horizon}"
    )
    train_rows, train_manifest, train_audit = _precompute_diagnostic_paths(
        model,
        train_paths,
        expected_schema=schema,
        device=args.device,
        horizon=args.horizon,
    )
    val_rows, val_manifest, val_audit = _precompute_diagnostic_paths(
        model,
        val_paths,
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

    residual_head, residual_payload = AnalyticWorkResidualHead.from_checkpoint(
        args.residual_head, map_location=args.device
    )
    expected_world_model = (
        residual_payload.get("base_world_model") or {}
    ).get("sha256")
    actual_world_model = _sha256_file(args.world_model)
    if expected_world_model and expected_world_model != actual_world_model:
        raise SystemExit("residual head and supplied World Model checkpoint differ")
    head_horizon = (residual_payload.get("semantics") or {}).get("horizon")
    if head_horizon is not None and int(head_horizon) != args.horizon:
        raise SystemExit(
            f"residual head H={head_horizon} differs from requested H={args.horizon}"
        )
    if dict(residual_payload.get("work_schema") or {}) != dict(schema):
        raise SystemExit("residual head work schema differs from input data")

    train_groups = _groups(train_rows)
    val_groups = _groups(val_rows)
    capacity = np.asarray(schema["work_capacity"], dtype=np.float64)
    work_weight = float(schema["work_weight"])

    global_fit = fit_global_efficiency(train_groups)
    train_eta, train_defined = _group_eta_targets(train_groups)
    val_eta, val_defined = _group_eta_targets(val_groups)
    state_eta, state_report = _context_efficiency_model(
        train_groups,
        val_groups,
        train_eta,
        train_defined,
        val_eta,
        val_defined,
        capacity=capacity,
        mode="state_only",
        alphas=args.ridge_alphas,
    )
    candidate_set_eta, candidate_set_report = _context_efficiency_model(
        train_groups,
        val_groups,
        train_eta,
        train_defined,
        val_eta,
        val_defined,
        capacity=capacity,
        mode="candidate_set",
        alphas=args.ridge_alphas,
    )

    analytic_predictions = [
        (
            np.asarray([row["nominal_raw_work_drift"] for row in group]),
            np.stack([_array(row["nominal_endpoint_station_work"]) for row in group]),
        )
        for group in val_groups
    ]
    global_predictions = [
        _eta_prediction(
            group,
            global_fit["eta"],
            capacity=capacity,
            work_weight=work_weight,
        )
        for group in val_groups
    ]
    state_predictions = [
        _eta_prediction(
            group,
            float(eta),
            capacity=capacity,
            work_weight=work_weight,
        )
        for group, eta in zip(val_groups, state_eta)
    ]
    candidate_set_predictions = [
        _eta_prediction(
            group,
            float(eta),
            capacity=capacity,
            work_weight=work_weight,
        )
        for group, eta in zip(val_groups, candidate_set_eta)
    ]
    oracle_predictions = [
        _eta_prediction(
            group,
            float(val_eta[index]) if val_defined[index] else 0.0,
            capacity=capacity,
            work_weight=work_weight,
        )
        for index, group in enumerate(val_groups)
    ]
    c_predictions = _c_predictions(
        residual_head, val_groups, device=args.device
    )

    prediction_sets = {
        "A_parameter_free_analytic": analytic_predictions,
        "A_eta_global_train_only": global_predictions,
        "A_eta_context_state_only": state_predictions,
        "A_eta_context_candidate_set": candidate_set_predictions,
        "A_eta_oracle_group_upper_bound": oracle_predictions,
        "C_analytic_plus_residual": c_predictions,
    }
    records = {
        name: _prediction_records(val_groups, prediction)
        for name, prediction in prediction_sets.items()
    }
    action_estimators = {
        name: _evaluate_records(
            rows,
            bootstrap_repeats=args.bootstrap_repeats,
            random_seed=args.seed + index,
        )
        for index, (name, rows) in enumerate(records.items())
    }
    direction = {
        name: _within_group_direction_explained(rows)
        for name, rows in records.items()
    }
    paired_to_a = {
        name: _paired_action_comparison(
            records["A_parameter_free_analytic"],
            rows,
            bootstrap_repeats=args.bootstrap_repeats,
            random_seed=args.seed + 100 + index,
        )
        for index, (name, rows) in enumerate(records.items())
        if name != "A_parameter_free_analytic"
    }

    train_targets, train_target_report = _oracle_shared_candidate_target(
        train_groups,
        train_eta,
        train_defined,
        capacity=capacity,
        work_weight=work_weight,
    )
    val_targets, val_target_report = _oracle_shared_candidate_target(
        val_groups,
        val_eta,
        val_defined,
        capacity=capacity,
        work_weight=work_weight,
    )
    probe_modes = ["preview", "frozen_wm", "frozen_wm_plus_preview"]
    if val_audit["traffic_available_samples"] == len(val_rows) and (
        train_audit["traffic_available_samples"] == len(train_rows)
    ):
        probe_modes.append("oracle_traffic")
    probes = {}
    probe_records = {}
    for index, mode in enumerate(probe_modes):
        prediction, probe_report = _fit_candidate_probe(
            train_groups,
            val_groups,
            train_targets,
            val_targets,
            mode=mode,
            alphas=args.ridge_alphas,
        )
        combined = _combine_shared_and_candidate(state_predictions, prediction)
        combined_records = _prediction_records(val_groups, combined)
        probe_report["action_score_on_state_shared_eta"] = _evaluate_records(
            combined_records,
            bootstrap_repeats=args.bootstrap_repeats,
            random_seed=args.seed + 200 + index,
        )
        probe_report["paired_to_state_shared_eta"] = _paired_action_comparison(
            records["A_eta_context_state_only"],
            combined_records,
            bootstrap_repeats=args.bootstrap_repeats,
            random_seed=args.seed + 300 + index,
        )
        probes[mode] = probe_report
        probe_records[mode] = combined_records

    c_raw = [prediction[0] for prediction in c_predictions]
    c_candidate_component = _candidate_component_from_raw(
        c_raw, oracle_predictions
    )
    c_component_report = _residual_regression_metrics(
        val_targets, c_candidate_component
    )

    def regret_gain(name: str) -> Optional[float]:
        row = probes.get(name, {}).get("paired_to_state_shared_eta", {})
        value = row.get("normalised_regret_improvement", {}).get("mean")
        return None if value is None else float(value)

    wm_gain = regret_gain("frozen_wm_plus_preview")
    oracle_gain = regret_gain("oracle_traffic")
    if oracle_gain is not None and oracle_gain > 0.0 and (
        wm_gain is None or wm_gain <= 0.0
    ):
        phase_c = "PHASE_C_REPRESENTATION_HYPOTHESIS_SUPPORTED_DEVELOPMENT_ONLY"
    elif wm_gain is not None and wm_gain > 0.0:
        phase_c = "FROZEN_WM_RESIDUAL_SIGNAL_PRESENT_ARCHITECTURE_OR_TRAINING_NEXT"
    elif oracle_gain is not None and oracle_gain <= 0.0:
        phase_c = "CANDIDATE_RESIDUAL_NOT_USEFULLY_PREDICTABLE_IN_THIS_DEVELOPMENT_SPLIT"
    else:
        phase_c = "INSUFFICIENT_DIAGNOSTIC_EVIDENCE"

    report = {
        "schema_version": SCHEMA_VERSION,
        "role": "DEVELOPMENT_ONLY_NOT_FORMAL_CERTIFICATION",
        "online_ready": False,
        "semantics": {
            "fixed_potential": "quadratic analytic L_work",
            "horizon": int(args.horizon),
            "decomposition": (
                "true relief = context-shared eta * analytic nominal relief "
                "+ within-group candidate residual"
            ),
            "unknown_future_orders": False,
            "continuation_policy": False,
            "td_target_or_tail": False,
            "greedy_or_external_policy": False,
            "oracle_group_eta_is_label_only": True,
            "oracle_traffic_probe_is_not_deployable": True,
        },
        "abc_definitions": {
            "A_parameter_free_analytic": (
                "fixed post-action chain-ledger free-flow relief; no fitted parameters"
            ),
            "B_legacy_endpoint_head": (
                "legacy WorkDriftHead v1 that directly learns endpoint station work"
            ),
            "C_analytic_plus_residual": (
                "analytic nominal relief plus learned signed station-wise residual"
            ),
            "eta_variants_are_diagnostics_of_A_not_a_new_ABC_label": True,
        },
        "provenance": {
            "world_model": {
                "path": os.path.abspath(args.world_model),
                "sha256": actual_world_model,
                "model_config": model_config,
            },
            "residual_head": {
                "path": os.path.abspath(args.residual_head),
                "sha256": _sha256_file(args.residual_head),
                "training_seed": (
                    residual_payload.get("training_config") or {}
                ).get("seed"),
            },
            "train_inputs": _input_provenance(train_paths),
            "validation_inputs": _input_provenance(val_paths),
        },
        "data_manifest": {
            "train": train_manifest,
            "validation": val_manifest,
            "train_audit": train_audit,
            "validation_audit": val_audit,
            "seed_disjoint": True,
        },
        "work_schema": schema,
        "global_efficiency_train_only": global_fit,
        "oracle_context_efficiency": {
            "train": _efficiency_summary(
                train_groups, train_eta, train_defined
            ),
            "validation": _efficiency_summary(
                val_groups, val_eta, val_defined
            ),
        },
        "context_shared_efficiency_predictors": {
            "state_only": state_report,
            "candidate_set": candidate_set_report,
        },
        "action_estimators": action_estimators,
        "within_group_direction_explained": direction,
        "paired_action_comparison_to_A": paired_to_a,
        "candidate_residual_target": {
            "train": train_target_report,
            "validation": val_target_report,
        },
        "candidate_residual_representation_probes": probes,
        "C_candidate_component_against_oracle_decomposition": c_component_report,
        "phase_c_implication": {
            "classification": phase_c,
            "development_only": True,
            "frozen_wm_plus_preview_regret_gain_over_state_shared": wm_gain,
            "oracle_traffic_regret_gain_over_state_shared": oracle_gain,
            "rule": (
                "oracle positive with frozen-WM non-positive supports a Phase-C "
                "representation hypothesis; neither is a formal gate"
            ),
            "validation_seed_clusters": len(
                val_manifest["simulation_seeds"]
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("saved:", output)
    print("role:", report["role"])
    print("H =", args.horizon)
    print("global eta =", global_fit["eta"])
    print(
        "state-shared eta RMSE =",
        state_report["eta_rmse"],
        "candidate-set eta RMSE =",
        candidate_set_report["eta_rmse"],
    )
    for name in (
        "A_parameter_free_analytic",
        "A_eta_global_train_only",
        "A_eta_context_state_only",
        "A_eta_context_candidate_set",
        "A_eta_oracle_group_upper_bound",
        "C_analytic_plus_residual",
    ):
        metric = action_estimators[name]["continuous_group_range"]
        print(
            name,
            "NRMSE=", metric.get("normalised_rmse"),
            "pairwise=", metric.get("all_non_tie_pair_concordance"),
            "top1=", metric.get("top1_min_drift_accuracy"),
            "regret_gain=", (
                metric.get(
                    "normalised_selection_regret_improvement_over_uniform_random"
                ) or {}
            ).get("mean"),
        )
    print("Phase-C implication =", phase_c)
    print("note: this command does not certify Layer 4, Layer 5, or Phase C")


if __name__ == "__main__":
    main()
