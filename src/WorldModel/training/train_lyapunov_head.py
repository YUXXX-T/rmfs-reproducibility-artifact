"""Train analytic-anchor residual L0 and endpoint-demand heads.

Input files are counterfactual sample lists produced with
``WorldModelDataCollector(record_lyapunov_l0=True)``.  The base world model is
frozen.  This script trains only:

* ``EndpointDemandPredictor`` for ``demand_context(t+H)-demand_context(t)``;
* ``LyapunovComponentHead`` for endpoint-minus-current physical components.

The current state is always the simulator's analytic ledger.  The scalar L0
value is always reconstructed by ``FixedLyapunovFunctional``.
Train/validation files are explicit to avoid adjacent-tick or run leakage.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import asdict
import glob
import json
import math
import os
import random
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from WorldModel.core.lyapunov_head import (
    EndpointDemandPredictor,
    FixedLyapunovFunctional,
    LyapunovComponentHead,
    LyapunovLossWeights,
    LyapunovPrediction,
    compute_lyapunov_training_loss,
    target_from_analytic_snapshot,
)
from WorldModel.core.lyapunov import LyapunovL0Config
from WorldModel.training.train_td_risk_v import load_frozen_world_model


SCHEMA_VERSION = "lyapunov_component_residual_head_v2"


def _expand_paths(values: Iterable[str]) -> List[str]:
    paths = []
    for value in values:
        matches = sorted(glob.glob(value))
        paths.extend(matches or [value])
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"missing data files: {missing}")
    return paths


def _load_samples(paths: Sequence[str], max_samples=None) -> List[dict]:
    rows = []
    for path in _expand_paths(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "samples" in payload:
            payload = payload["samples"]
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected a list of samples")
        for sample in payload:
            required = (
                "lyapunov_l0_start", "lyapunov_l0_end",
                "lyapunov_l0_progress", "future_demand_context",
                "station_node_ids", "node_history", "edge_index",
                "edge_features", "demand_context", "action_node",
                "action_global", "future_mask",
            )
            if (all(sample.get(key) is not None for key in required)
                    and int(sample["future_mask"].sum().item()) > 0):
                sample = dict(sample)
                sample["_source_path"] = os.path.abspath(path)
                rows.append(sample)
                if max_samples is not None and len(rows) >= max_samples:
                    return rows
    return rows


def _split_unit(sample: dict):
    """Return the strongest available trajectory-level leakage unit."""
    seed = sample.get("simulation_seed")
    if seed is not None:
        return ("seed", str(seed))
    run_id = sample.get("run_id")
    if run_id:
        return ("run", str(run_id))
    return ("source_file", str(sample.get("_source_path", "unknown")))


def _load_label(sample: Mapping) -> str:
    """Resolve low/mid/high without trusting one collector-specific field."""
    for key in ("load", "load_label", "arm_label", "run_id", "_source_path"):
        value = str(sample.get(key) or "")
        match = re.search(
            r"(?:^|[_\\/.-])(low|mid|high)(?:$|[_\\/.-])",
            value,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).lower()
    return "unknown"


def _validate_disjoint_splits(
    train_samples: Sequence[dict],
    val_samples: Sequence[dict],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Reject seed/run leakage between the explicitly supplied splits."""
    train_units = sorted({_split_unit(sample) for sample in train_samples})
    val_units = sorted({_split_unit(sample) for sample in val_samples})
    overlap = sorted(set(train_units) & set(val_units))
    if overlap:
        raise ValueError(
            "train/validation trajectory units overlap: "
            f"{overlap}; split by complete seed/run"
        )
    return train_units, val_units


def _station_value(mapping: dict, station_id: int, default=0.0):
    return mapping.get(station_id, mapping.get(str(station_id), default))


def _functional_from_sample(sample: dict, device: str):
    snapshot = sample["lyapunov_l0_start"]
    station_ids = sorted(int(value) for value in snapshot["station_work"])
    work_capacity = [float(snapshot["work_capacity"])] * len(station_ids)
    arrival_capacity = [
        list(_station_value(snapshot["arrival_capacity"], station_id, ()))
        for station_id in station_ids
    ]
    config = dict(sample.get("lyapunov_l0_config") or {})
    functional = FixedLyapunovFunctional(
        work_capacity=work_capacity,
        arrival_capacity=arrival_capacity,
        work_weight=float(config.get("work_weight", 1.0)),
        station_weight=float(config.get("station_weight", 1.0)),
        traffic_weight=float(config.get("traffic_weight", 1.0)),
        stall_weight=float(config.get("stall_weight", 1.0)),
        plan_fail_weight=float(config.get("plan_fail_weight", 0.5)),
        arrival_weight=float(config.get("arrival_weight", 1.0)),
        station_safe_ratio=float(config.get("station_safe_ratio", 0.70)),
    ).to(device)
    return functional, station_ids, len(arrival_capacity[0])


def _resolved_l0_config(sample: dict) -> dict:
    """Return a canonical config so implicit and explicit defaults compare equal."""
    return asdict(LyapunovL0Config(**dict(
        sample.get("lyapunov_l0_config") or {}
    )))


def _l0_collection_signature(sample: Mapping) -> tuple[str, str, str, str]:
    start = sample.get("lyapunov_l0_start") or {}
    return (
        str(start.get("schema_version", "legacy_or_missing")),
        str(sample.get(
            "lyapunov_l0_collection_schema_version", "legacy_or_missing"
        )),
        str(sample.get("rollout_continuation_mode", "isolated_legacy")),
        str(sample.get(
            "rollout_continuation_policy",
            sample.get("continuation_policy", "unknown"),
        )),
    )


def _validate_l0_schema(
    samples: Sequence[dict],
    station_ids: Sequence[int],
    reference_config: dict,
    reference_work_capacity: Sequence[float],
    reference_arrival_capacity: Sequence[Sequence[float]],
    split_name: str,
    reference_collection_signature: Optional[
        tuple[str, str, str, str]
    ] = None,
) -> None:
    """Fail fast instead of silently mixing incompatible analytic functionals.

    The component head may share data across runs/policies, but its physical
    labels must retain one frozen meaning.  In particular, the scalar/value
    and drift losses cannot be reconstructed with the first sample's
    capacities when later samples used a different normalisation.
    """
    expected_station_ids = tuple(int(value) for value in station_ids)
    expected_work = tuple(float(value) for value in reference_work_capacity)
    expected_arrival = tuple(
        tuple(float(value) for value in row)
        for row in reference_arrival_capacity
    )
    if reference_collection_signature is None:
        reference_collection_signature = _l0_collection_signature(samples[0])

    for index, sample in enumerate(samples):
        signature = _l0_collection_signature(sample)
        if signature != reference_collection_signature:
            raise ValueError(
                f"{split_name}[{index}] uses L0 collection signature "
                f"{signature}, expected {reference_collection_signature}; "
                "do not fuse legacy/v2 labels or different continuation "
                "policies into one head"
            )
        config = _resolved_l0_config(sample)
        if config != reference_config:
            raise ValueError(
                f"{split_name}[{index}] uses a different LyapunovL0Config; "
                "train separate heads or recollect with one frozen config"
            )

        for endpoint in ("lyapunov_l0_start", "lyapunov_l0_end"):
            snapshot = sample[endpoint]
            if str(snapshot.get(
                    "schema_version", "legacy_or_missing"
                    )) != reference_collection_signature[0]:
                raise ValueError(
                    f"{split_name}[{index}] {endpoint} snapshot schema changed"
                )
            actual_station_ids = tuple(sorted(
                int(value) for value in snapshot["station_work"]
            ))
            if actual_station_ids != expected_station_ids:
                raise ValueError(
                    f"{split_name}[{index}] {endpoint} station ids "
                    f"{actual_station_ids} do not match {expected_station_ids}"
                )

            work_capacity = tuple(
                float(snapshot["work_capacity"])
                for _ in expected_station_ids
            )
            if work_capacity != expected_work:
                raise ValueError(
                    f"{split_name}[{index}] {endpoint} work_capacity changed; "
                    "use a fixed scale or train a separate head"
                )

            arrival_capacity = tuple(
                tuple(float(value) for value in _station_value(
                    snapshot["arrival_capacity"], station_id, ()
                ))
                for station_id in expected_station_ids
            )
            if arrival_capacity != expected_arrival:
                raise ValueError(
                    f"{split_name}[{index}] {endpoint} arrival_capacity/schema "
                    "changed; train a separate head"
                )


def _balance_target(progress: dict) -> float:
    arrivals = sum(float(value) for value in progress[
        "arrivals_by_station"
    ].values())
    productive = sum(float(value) for value in progress[
        "productive_by_station"
    ].values())
    reverse = sum(float(value) for value in progress[
        "reverse_by_station"
    ].values())
    replan = sum(float(value) for value in progress.get(
        "replan_residual_by_station", {}
    ).values())
    return arrivals - productive + reverse + replan


def _precompute(
    model,
    samples: Sequence[dict],
    station_ids: Sequence[int],
    device: str,
) -> List[dict]:
    rows = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            node_history = sample["node_history"].to(device)
            edge_index = sample["edge_index"].long().to(device)
            edge_features = sample["edge_features"].to(device)
            demand = sample["demand_context"].to(device)
            z0, e0, edge_attr = model.encode_state(
                node_history, edge_index, edge_features, demand
            )
            # The analytic endpoint and future demand target are captured at
            # the last *valid* counterfactual step.  A rollout that failed
            # early is zero-padded to the nominal horizon, so ``numel()``
            # would train against a latent endpoint later than its labels.
            horizon = int(sample["future_mask"].sum().item())
            station_node_ids = sample["station_node_ids"].long().flatten()
            if station_node_ids.numel() != len(station_ids):
                raise ValueError(
                    "station_node_ids and analytic station labels disagree"
                )
            *_, z_start, z_end = model.rollout(
                z0,
                e0,
                edge_attr,
                sample["action_node"].to(device),
                sample["action_global"].to(device),
                edge_index,
                station_node_ids.tolist(),
                K=horizon,
            )
            bottleneck = node_history[-1, :, 8].detach().cpu()
            current_target = target_from_analytic_snapshot(
                sample["lyapunov_l0_start"], station_ids
            )
            future_target = target_from_analytic_snapshot(
                sample["lyapunov_l0_end"], station_ids
            )
            rows.append({
                "z0": z_start.detach().cpu(),
                "zH": z_end.detach().cpu(),
                "e0": e0.detach().cpu(),
                "d0_raw": demand.detach().cpu(),
                "bottleneck": bottleneck,
                "current_target": current_target,
                "future_target": future_target,
                "future_demand": sample["future_demand_context"].float(),
                # These are graph node indices.  They are deliberately kept
                # separate from the simulator station ids used to order the
                # analytic labels (e.g. station ids 1..4 may map to graph
                # nodes [4, 15, 384, 395]).
                "station_node_ids": station_node_ids.cpu(),
                "group_key": (
                    str(sample.get("run_id") or sample["_source_path"]),
                    str(sample.get("candidate_group_id") or ""),
                ) if sample.get("candidate_group_id") else None,
                "seed": (
                    int(sample["simulation_seed"])
                    if sample.get("simulation_seed") is not None else None
                ),
                "load": _load_label(sample),
                "run_id": str(sample.get("run_id") or "unknown"),
                "balance_target": _balance_target(
                    sample["lyapunov_l0_progress"]
                ),
            })
    return rows


def _to_device_prediction(prediction, device):
    return type(prediction)(*(value.to(device) for value in prediction))


def _component_residual_scales(
    rows: Sequence[dict],
    floor: float = 1e-3,
) -> dict:
    """Robust scalar scale per physical field from train endpoint deltas."""
    if floor <= 0.0:
        raise ValueError("component scale floor must be positive")
    values = {name: [] for name in LyapunovPrediction._fields}
    for row in rows:
        for name, current, future in zip(
                LyapunovPrediction._fields,
                row["current_target"], row["future_target"]):
            values[name].extend((future - current).abs().flatten().tolist())
    result = {}
    for name, field_values in values.items():
        tensor = torch.as_tensor(field_values, dtype=torch.float32)
        scale = (
            float(torch.quantile(tensor, 0.75))
            if tensor.numel() else 0.0
        )
        result[name] = max(scale, float(floor))
    return result


def _component_training_support(
    rows: Sequence[dict],
    tolerance: float = 1e-8,
) -> dict:
    """Hard-mask fields that have no train endpoint-delta supervision."""
    if tolerance < 0.0:
        raise ValueError("component support tolerance must be non-negative")
    counts = {name: 0 for name in LyapunovPrediction._fields}
    for row in rows:
        for name, current, future in zip(
                LyapunovPrediction._fields,
                row["current_target"], row["future_target"]):
            counts[name] += int(((future - current).abs() > tolerance).sum())
    return {
        name: {
            "active": bool(count > 0),
            "nonzero_delta_values": int(count),
            "tolerance": float(tolerance),
        }
        for name, count in counts.items()
    }


def _demand_residual_scales(
    rows: Sequence[dict],
    floor: float = 1e-3,
) -> torch.Tensor:
    """Per-demand-field robust scales for ``d_H - d_0``."""
    if not rows:
        raise ValueError("cannot compute demand scales from an empty dataset")
    deltas = torch.stack([
        (row["future_demand"] - row["d0_raw"]).abs()
        for row in rows
    ])
    scales = torch.quantile(deltas, 0.75, dim=0)
    return torch.clamp(scales, min=float(floor))


def _ledger_loss_scales(
    rows: Sequence[dict],
    functional: FixedLyapunovFunctional,
    device: str,
    floor: float = 1e-3,
) -> dict:
    """Robust normalizers for scalar-L, drift and balance objectives."""
    drifts = []
    signed_drifts = []
    balances = []
    with torch.no_grad():
        for row in rows:
            current = _to_device_prediction(row["current_target"], device)
            future = _to_device_prediction(row["future_target"], device)
            drift = functional(future)["total"] - functional(current)["total"]
            drifts.append(abs(float(drift)))
            signed_drifts.append(float(drift))
            balances.append(abs(float(row["balance_target"])))

    def robust(values):
        tensor = torch.as_tensor(values, dtype=torch.float32)
        value = float(torch.quantile(tensor, 0.75)) if tensor.numel() else 0.0
        return max(value, float(floor))

    drift_scale = robust(drifts)
    return {
        # With an exact current anchor, endpoint scalar error and drift error
        # have the same physical units and use the same registered scale.
        "value": drift_scale,
        "drift": drift_scale,
        "balance": robust(balances),
        "train_mean_drift": (
            float(np.mean(signed_drifts)) if signed_drifts else 0.0
        ),
    }


def _normalized_demand_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    scales = scales.to(dtype=prediction.dtype, device=prediction.device)
    return F.smooth_l1_loss(
        (prediction - target) / scales,
        torch.zeros_like(prediction),
    )


def _pairwise_rank_loss(
    predicted_drifts: Sequence[torch.Tensor],
    target_drifts: Sequence[float],
    *,
    target_margin: float,
    prediction_margin: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Within-context drift ordering loss above a practical target gap."""
    if target_margin < 0.0 or prediction_margin < 0.0:
        raise ValueError("pairwise margins must be non-negative")
    if len(predicted_drifts) != len(target_drifts):
        raise ValueError("predicted and target drifts must align")
    if not predicted_drifts:
        return torch.tensor(0.0), 0
    losses = []
    for left in range(len(predicted_drifts)):
        for right in range(left + 1, len(predicted_drifts)):
            target_gap = float(target_drifts[left] - target_drifts[right])
            if abs(target_gap) <= max(target_margin, 1e-12):
                continue
            direction = 1.0 if target_gap > 0.0 else -1.0
            predicted_gap = predicted_drifts[left] - predicted_drifts[right]
            losses.append(F.relu(
                predicted_gap.new_tensor(prediction_margin)
                - direction * predicted_gap
            ))
    if not losses:
        return predicted_drifts[0].new_zeros(()), 0
    return torch.stack(losses).mean(), len(losses)


def _training_groups(rows: Sequence[dict]) -> List[List[dict]]:
    groups: Dict[object, List[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = row.get("group_key")
        groups[key if key is not None else ("row", index)].append(row)
    return list(groups.values())


def _pairwise_drift_accuracy(group_drifts: dict, target_margin: float = 0.0):
    """Tie-safe within-context ranking accuracy for lower/higher drift."""
    if target_margin < 0.0:
        raise ValueError("target_margin must be non-negative")
    pairwise_ok = []
    for members in group_drifts.values():
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                predicted_gap = members[left][0] - members[right][0]
                target_gap = members[left][1] - members[right][1]
                if abs(target_gap) <= max(1e-9, target_margin):
                    continue
                pairwise_ok.append(predicted_gap * target_gap > 0.0)
    return (
        float(np.mean(pairwise_ok)) if pairwise_ok else None,
        len(pairwise_ok),
    )


def _pairwise_gap_slices(
    group_drifts: dict,
    thresholds: Sequence[float] = (0.0, 0.1, 0.5),
) -> dict:
    result = {}
    for threshold in sorted({float(value) for value in thresholds}):
        if threshold < 0.0:
            raise ValueError("pairwise thresholds must be non-negative")
        accuracy, pairs = _pairwise_drift_accuracy(
            group_drifts, target_margin=threshold
        )
        result[f"{threshold:g}"] = {
            "target_gap_threshold": threshold,
            "accuracy": accuracy,
            "pairs": pairs,
        }
    return result


def _group_equal_weights(rows: Sequence[dict]) -> List[float]:
    """Give every candidate context equal total validation weight."""
    keys = []
    counts: Dict[object, int] = defaultdict(int)
    for index, row in enumerate(rows):
        key = row.get("group_key")
        key = key if key is not None else ("row", index)
        keys.append(key)
        counts[key] += 1
    return [1.0 / counts[key] for key in keys]


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights):
        raise ValueError("values and weights must align")
    if not values:
        return math.inf
    denominator = float(np.sum(weights))
    if denominator <= 0.0:
        raise ValueError("validation weights must have positive mass")
    return float(np.dot(values, weights) / denominator)


def _continuous_group_centered_drift_metrics(
    records: Sequence[dict],
) -> dict:
    """Scale-free learned-drift metrics over all within-context variation.

    A fixed raw target gap cannot certify a Lyapunov approximation because a
    positive rescaling of the analytic functional rescales every gap without
    changing its control meaning.  These metrics therefore centre prediction
    and target inside each candidate group and use every non-tied pair.
    """
    if not records:
        return {"samples": 0, "groups": 0}
    grouped: Dict[object, List[dict]] = defaultdict(list)
    for row in records:
        grouped[row["group_key"]].append(row)

    predicted = []
    target = []
    weights = []
    group_concordance = []
    pair_count = 0
    for members in grouped.values():
        pred_values = np.asarray([
            float(row["predicted_drift"]) for row in members
        ], dtype=np.float64)
        target_values = np.asarray([
            float(row["target_drift"]) for row in members
        ], dtype=np.float64)
        pred_centered = pred_values - pred_values.mean()
        target_centered = target_values - target_values.mean()
        member_weight = 1.0 / max(len(members), 1)
        predicted.extend(pred_centered.tolist())
        target.extend(target_centered.tolist())
        weights.extend([member_weight] * len(members))

        correct = []
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                target_gap = target_values[left] - target_values[right]
                if abs(target_gap) <= 1e-9:
                    continue
                predicted_gap = pred_values[left] - pred_values[right]
                correct.append(float(predicted_gap * target_gap > 0.0))
        if correct:
            group_concordance.append(float(np.mean(correct)))
            pair_count += len(correct)

    predicted_array = np.asarray(predicted, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    weight_array /= max(float(weight_array.sum()), 1e-12)
    error = predicted_array - target_array
    rmse = float(np.sqrt(np.dot(weight_array, error ** 2)))
    target_variance = float(np.dot(weight_array, target_array ** 2))
    target_std = float(np.sqrt(target_variance))
    predicted_std = float(np.sqrt(np.dot(
        weight_array, predicted_array ** 2
    )))
    covariance = float(np.dot(
        weight_array, predicted_array * target_array
    ))
    pearson = (
        float(covariance / (predicted_std * target_std))
        if predicted_std > 1e-12 and target_std > 1e-12 else None
    )
    return {
        "samples": len(records),
        "groups": len(grouped),
        "centering": "candidate_group_mean",
        "weighting": "candidate_group_equal",
        "raw_gap_gate": False,
        "rmse": rmse,
        "target_std": target_std,
        "normalized_rmse": (
            float(rmse / target_std) if target_std > 1e-12 else None
        ),
        "pearson": pearson,
        "all_non_tie_pair_concordance": (
            float(np.mean(group_concordance))
            if group_concordance else None
        ),
        "all_non_tie_pairs": int(pair_count),
    }


def _summarize_validation_records(
    records: Sequence[dict],
    pairwise_thresholds: Sequence[float] = (0.0, 0.1, 0.5),
) -> dict:
    if not records:
        return {"samples": 0, "groups": 0}
    weights = [float(row["weight"]) for row in records]

    def mean(name):
        return _weighted_mean(
            [float(row[name]) for row in records], weights
        )

    sign_rows = [row for row in records if row["sign_correct"] is not None]
    if sign_rows:
        sign_accuracy = _weighted_mean(
            [float(row["sign_correct"]) for row in sign_rows],
            [float(row["weight"]) for row in sign_rows],
        )
        positive_weight = sum(
            float(row["weight"])
            for row in sign_rows if row["target_sign"] > 0
        )
        negative_weight = sum(
            float(row["weight"])
            for row in sign_rows if row["target_sign"] < 0
        )
        majority_accuracy = max(positive_weight, negative_weight) / (
            positive_weight + negative_weight
        )
    else:
        sign_accuracy = None
        majority_accuracy = None

    group_drifts: Dict[object, List[tuple[float, float]]] = defaultdict(list)
    for row in records:
        group_drifts[row["group_key"]].append((
            float(row["predicted_drift"]), float(row["target_drift"])
        ))

    return {
        "samples": len(records),
        "groups": len({row["group_key"] for row in records}),
        "demand_mae": mean("demand_error"),
        "persistence_demand_mae": mean("persistence_demand_error"),
        "drift_mae": mean("drift_error"),
        "zero_drift_mae": mean("zero_drift_error"),
        "train_mean_drift_mae": mean("train_mean_drift_error"),
        "scalar_l_mae": mean("scalar_l_error"),
        "constant_l_mae": mean("constant_l_error"),
        "drift_sign_accuracy": sign_accuracy,
        "majority_drift_sign_accuracy": majority_accuracy,
        "continuous_group_centered": (
            _continuous_group_centered_drift_metrics(records)
        ),
        "pairwise_gap_slices_role": (
            "descriptive error-scale diagnostics only; not the learned "
            "Lyapunov validity gate"
        ),
        "pairwise_gap_slices": _pairwise_gap_slices(
            group_drifts, thresholds=pairwise_thresholds
        ),
    }


def _evaluate(
    head,
    demand_head,
    model,
    functional,
    rows,
    device,
    weights,
    demand_weight,
    component_scales: Mapping[str, float],
    demand_scales: torch.Tensor,
    ledger_loss_scales: Mapping[str, float],
    *,
    pairwise_target_margin: float = 0.1,
    gate_min_absolute_improvement: float = 0.0,
    gate_min_relative_improvement: float = 0.0,
    min_component_support: int = 10,
    min_pairwise_pairs: int = 10,
    apply_demand_gate: bool = False,
):
    head.eval()
    demand_head.eval()
    metric_weights = _group_equal_weights(rows)
    totals = []
    demand_mae = []
    persistence_demand_mae = []
    sign_ok = []
    target_signs = []
    sign_weights = []
    scalar_l_abs_error = []
    persistence_scalar_l_abs_error = []
    drift_abs_error = []
    persistence_drift_abs_error = []
    train_mean_drift_abs_error = []
    component_abs_error = {
        name: [] for name in LyapunovPrediction._fields
    }
    persistence_component_abs_error = {
        name: [] for name in LyapunovPrediction._fields
    }
    component_nonzero_delta = {
        name: 0 for name in LyapunovPrediction._fields
    }
    group_drifts = {}
    validation_records = []
    train_mean_drift = float(ledger_loss_scales.get("train_mean_drift", 0.0))
    with torch.no_grad():
        for row_index, (row, row_weight) in enumerate(zip(rows, metric_weights)):
            zH, e0 = (row[key].to(device) for key in ("zH", "e0"))
            d0_raw = row["d0_raw"].to(device)
            scores = row["bottleneck"].to(device)
            current_target = _to_device_prediction(row["current_target"], device)
            future_target = _to_device_prediction(row["future_target"], device)
            station_node_ids = row["station_node_ids"].tolist()
            raw_future, e_future = demand_head.predict_embedding(
                zH, e0, d0_raw, model.demand_encoder,
                apply_gate=apply_demand_gate,
            )
            # The current analytic ledger is exact and is never reconstructed
            # by the learned endpoint head.
            current = current_target
            future = head(
                zH, e_future, station_node_ids, scores,
                anchor=current_target,
                # The head buffer is the train-support mask during fitting
                # and the held-out deployment mask after checkpoint selection.
                apply_component_gates=True,
            )
            losses = compute_lyapunov_training_loss(
                current,
                future,
                current_target,
                future_target,
                functional,
                balance_target=torch.tensor(
                    row["balance_target"], device=device
                ),
                weights=weights,
                component_scales=component_scales,
                value_scale=ledger_loss_scales["value"],
                drift_scale=ledger_loss_scales["drift"],
                balance_scale=ledger_loss_scales["balance"],
            )
            future_demand = row["future_demand"].to(device)
            dloss = _normalized_demand_loss(
                raw_future, future_demand, demand_scales
            )
            totals.append(float(losses["total"] + demand_weight * dloss))
            demand_error = float((
                raw_future - future_demand
            ).abs().mean())
            persistence_demand_error = float((
                d0_raw - future_demand
            ).abs().mean())
            demand_mae.append(demand_error)
            persistence_demand_mae.append(persistence_demand_error)
            target_drift = float(losses["target_drift"])
            predicted_drift = float(losses["predicted_drift"])
            if abs(target_drift) > 1e-9:
                target_signs.append(1 if target_drift > 0.0 else -1)
                sign_weights.append(float(row_weight))
            if row["group_key"] is not None:
                group_drifts.setdefault(row["group_key"], []).append((
                    predicted_drift, target_drift
                ))
            future_l = functional(future)["total"]
            current_l = functional(current_target)["total"]
            target_future_l = functional(future_target)["total"]
            scalar_l_error = float(abs(future_l - target_future_l))
            constant_l_error = float(abs(current_l - target_future_l))
            drift_error = abs(predicted_drift - target_drift)
            zero_drift_error = abs(target_drift)
            train_mean_error = abs(train_mean_drift - target_drift)
            scalar_l_abs_error.append(scalar_l_error)
            persistence_scalar_l_abs_error.append(constant_l_error)
            drift_abs_error.append(drift_error)
            persistence_drift_abs_error.append(zero_drift_error)
            train_mean_drift_abs_error.append(train_mean_error)
            for name, predicted, expected in zip(
                LyapunovPrediction._fields, future, future_target
            ):
                component_abs_error[name].append(float(
                    (predicted - expected).abs().mean()
                ))
            for name, current_value, expected in zip(
                LyapunovPrediction._fields, current_target, future_target
            ):
                delta = (expected - current_value).abs()
                persistence_component_abs_error[name].append(float(
                    delta.mean()
                ))
                component_nonzero_delta[name] += int(
                    (delta > 1e-8).sum().item()
                )
            if abs(target_drift) > 1e-9:
                sign_correct = ((target_drift > 0) == (predicted_drift > 0))
                sign_ok.append(sign_correct)
            else:
                sign_correct = None
            validation_records.append({
                "weight": float(row_weight),
                "group_key": (
                    row["group_key"]
                    if row["group_key"] is not None else ("row", row_index)
                ),
                "seed": row.get("seed"),
                "load": str(row.get("load") or "unknown"),
                "run_id": str(row.get("run_id") or "unknown"),
                "demand_error": demand_error,
                "persistence_demand_error": persistence_demand_error,
                "drift_error": drift_error,
                "zero_drift_error": zero_drift_error,
                "train_mean_drift_error": train_mean_error,
                "scalar_l_error": scalar_l_error,
                "constant_l_error": constant_l_error,
                "sign_correct": sign_correct,
                "target_sign": (
                    1 if target_drift > 0.0 else -1
                    if target_drift < 0.0 else 0
                ),
                "predicted_drift": predicted_drift,
                "target_drift": target_drift,
            })

    # The primary ranking metric uses every analytically non-tied pair.  Raw
    # target-margin slices remain descriptive only because positive rescaling
    # of L changes every numeric margin without changing Lyapunov semantics.
    pairwise_accuracy, pairwise_pairs = _pairwise_drift_accuracy(
        group_drifts, 0.0
    )
    margin_pairwise_accuracy, margin_pairwise_pairs = (
        _pairwise_drift_accuracy(group_drifts, pairwise_target_margin)
    )
    pairwise_slices = _pairwise_gap_slices(
        group_drifts,
        thresholds=(0.0, 0.1, 0.5, pairwise_target_margin),
    )
    continuous_group_centered = _continuous_group_centered_drift_metrics(
        validation_records
    )
    component_mae = {
        name: _weighted_mean(values, metric_weights)
        if values else math.inf
        for name, values in component_abs_error.items()
    }
    persistence_component_mae = {
        name: _weighted_mean(values, metric_weights)
        if values else math.inf
        for name, values in persistence_component_abs_error.items()
    }
    model_demand_mae = _weighted_mean(demand_mae, metric_weights)
    baseline_demand_mae = (
        _weighted_mean(persistence_demand_mae, metric_weights)
        if persistence_demand_mae else math.inf
    )
    model_drift_mae = _weighted_mean(drift_abs_error, metric_weights)
    baseline_drift_mae = (
        _weighted_mean(persistence_drift_abs_error, metric_weights)
        if persistence_drift_abs_error else math.inf
    )
    train_mean_drift_mae = (
        _weighted_mean(train_mean_drift_abs_error, metric_weights)
        if train_mean_drift_abs_error else math.inf
    )
    strongest_drift_baseline_mae = min(
        baseline_drift_mae, train_mean_drift_mae
    )
    model_scalar_l_mae = (
        _weighted_mean(scalar_l_abs_error, metric_weights)
        if scalar_l_abs_error else math.inf
    )
    baseline_scalar_l_mae = (
        _weighted_mean(persistence_scalar_l_abs_error, metric_weights)
        if persistence_scalar_l_abs_error else math.inf
    )
    sign_accuracy = (
        _weighted_mean([float(value) for value in sign_ok], sign_weights)
        if sign_ok else None
    )
    if target_signs:
        positive = sum(
            weight for value, weight in zip(target_signs, sign_weights)
            if value > 0
        )
        negative = sum(sign_weights) - positive
        majority_sign_accuracy = max(positive, negative) / sum(sign_weights)
    else:
        majority_sign_accuracy = None

    def improved(model_value, baseline_value):
        required = max(
            float(gate_min_absolute_improvement),
            float(gate_min_relative_improvement) * baseline_value,
        )
        return bool(
            np.isfinite(model_value)
            and np.isfinite(baseline_value)
            and baseline_value - model_value > required
        )

    component_gates = {}
    for name in LyapunovPrediction._fields:
        supported = component_nonzero_delta[name] >= min_component_support
        component_gates[name] = {
            "enabled": bool(supported and improved(
                component_mae[name], persistence_component_mae[name]
            )),
            "supported": bool(supported),
            "nonzero_delta_values": int(component_nonzero_delta[name]),
            "model_mae": component_mae[name],
            "persistence_mae": persistence_component_mae[name],
        }

    by_seed: Dict[str, List[dict]] = defaultdict(list)
    by_load: Dict[str, List[dict]] = defaultdict(list)
    for record in validation_records:
        seed_key = (
            str(record["seed"])
            if record["seed"] is not None
            else f"run:{record['run_id']}"
        )
        by_seed[seed_key].append(record)
        by_load[record["load"]].append(record)
    pairwise_thresholds = (0.0, 0.1, 0.5, pairwise_target_margin)
    by_seed_summary = {
        key: _summarize_validation_records(
            value, pairwise_thresholds=pairwise_thresholds
        )
        for key, value in sorted(by_seed.items())
    }
    by_load_summary = {
        key: _summarize_validation_records(
            value, pairwise_thresholds=pairwise_thresholds
        )
        for key, value in sorted(by_load.items())
    }

    aggregate_gates = {
        "demand": improved(model_demand_mae, baseline_demand_mae),
        "scalar_l": improved(model_scalar_l_mae, baseline_scalar_l_mae),
        "drift": improved(model_drift_mae, strongest_drift_baseline_mae),
        "drift_sign": bool(
            sign_accuracy is not None
            and majority_sign_accuracy is not None
            and sign_accuracy > majority_sign_accuracy
        ),
        "pairwise": bool(
            pairwise_accuracy is not None
            and pairwise_pairs >= min_pairwise_pairs
            and pairwise_accuracy > 0.5
        ),
    }
    load_gates = {}
    for load, summary in by_load_summary.items():
        continuous_row = summary["continuous_group_centered"]
        strongest_load_drift = min(
            summary["zero_drift_mae"], summary["train_mean_drift_mae"]
        )
        load_gates[load] = {
            "demand": improved(
                summary["demand_mae"], summary["persistence_demand_mae"]
            ),
            "scalar_l": improved(
                summary["scalar_l_mae"], summary["constant_l_mae"]
            ),
            "drift": improved(summary["drift_mae"], strongest_load_drift),
            "drift_sign": bool(
                summary["drift_sign_accuracy"] is not None
                and summary["majority_drift_sign_accuracy"] is not None
                and summary["drift_sign_accuracy"]
                > summary["majority_drift_sign_accuracy"]
            ),
            "pairwise": bool(
                continuous_row["all_non_tie_pair_concordance"] is not None
                and continuous_row["all_non_tie_pairs"] >= min_pairwise_pairs
                and continuous_row["all_non_tie_pair_concordance"] > 0.5
            ),
            "strongest_drift_baseline_mae": strongest_load_drift,
            "pairwise_protocol": "all_analytic_non_ties_no_raw_gap_gate",
        }
    gates = {
        name: bool(
            aggregate_gates[name]
            and load_gates
            and all(row[name] for row in load_gates.values())
        )
        for name in aggregate_gates
    }
    gate_count = sum(int(row["enabled"]) for row in component_gates.values())
    gate_count += sum(int(value) for value in gates.values())
    ratios = []
    for name, row in component_gates.items():
        if row["supported"] and row["persistence_mae"] > 1e-12:
            ratios.append(row["model_mae"] / row["persistence_mae"])
    for model_value, baseline_value in (
        (model_demand_mae, baseline_demand_mae),
        (model_scalar_l_mae, baseline_scalar_l_mae),
        (model_drift_mae, strongest_drift_baseline_mae),
    ):
        if baseline_value > 1e-12:
            ratios.append(model_value / baseline_value)
    ratio_mean = float(np.mean(ratios)) if ratios else math.inf

    return {
        "loss": _weighted_mean(totals, metric_weights)
        if totals else math.inf,
        "validation_weighting": "candidate_group_equal",
        "demand_mae": model_demand_mae,
        "persistence_demand_mae": baseline_demand_mae,
        "drift_sign_accuracy": sign_accuracy,
        "majority_drift_sign_accuracy": majority_sign_accuracy,
        "drift_pairwise_accuracy": pairwise_accuracy,
        "drift_pairwise_pairs": pairwise_pairs,
        "continuous_group_centered": continuous_group_centered,
        "pairwise_protocol": "all_analytic_non_ties_no_raw_gap_gate",
        "pairwise_target_margin": float(pairwise_target_margin),
        "pairwise_target_margin_role": "descriptive_only",
        "margin_pairwise_accuracy": margin_pairwise_accuracy,
        "margin_pairwise_pairs": margin_pairwise_pairs,
        "pairwise_gap_slices": pairwise_slices,
        "scalar_l_mae": model_scalar_l_mae,
        "persistence_scalar_l_mae": baseline_scalar_l_mae,
        "constant_l_mae": baseline_scalar_l_mae,
        "drift_mae": model_drift_mae,
        "persistence_drift_mae": baseline_drift_mae,
        "zero_drift_mae": baseline_drift_mae,
        "train_mean_drift": train_mean_drift,
        "train_mean_drift_mae": train_mean_drift_mae,
        "strongest_drift_baseline_mae": strongest_drift_baseline_mae,
        "component_mae": component_mae,
        "persistence_component_mae": persistence_component_mae,
        "component_coverage": {
            name: {
                "nonzero_delta_values": int(component_nonzero_delta[name]),
                "supported": bool(component_gates[name]["supported"]),
            }
            for name in LyapunovPrediction._fields
        },
        "component_gates": component_gates,
        "aggregate_gates": aggregate_gates,
        "load_gates": load_gates,
        "gates": gates,
        "selection_gate_count": int(gate_count),
        "selection_ratio_mean": ratio_mean,
        # Gate count is the primary checkpoint criterion; normalized error
        # only breaks ties.  More physical gates therefore always wins.
        "selection_score": -float(gate_count) + 0.01 * ratio_mean,
        "validation_breakdowns": {
            "by_seed": by_seed_summary,
            "by_load": by_load_summary,
        },
        "samples": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-data", nargs="+", required=True)
    parser.add_argument("--val-data", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--demand-weight", type=float, default=1.0)
    parser.add_argument("--component-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--drift-weight", type=float, default=1.0)
    parser.add_argument("--balance-weight", type=float, default=1.0)
    parser.add_argument("--sign-weight", type=float, default=0.25)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--pairwise-target-margin", type=float, default=0.1)
    parser.add_argument("--pairwise-prediction-margin", type=float, default=0.0)
    parser.add_argument("--residual-limit", type=float, default=4.0)
    parser.add_argument("--scale-floor", type=float, default=1e-3)
    parser.add_argument("--gate-min-absolute-improvement", type=float, default=0.0)
    parser.add_argument("--gate-min-relative-improvement", type=float, default=0.0)
    parser.add_argument("--min-component-support", type=int, default=10)
    parser.add_argument("--min-pairwise-pairs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if (args.pairwise_weight < 0.0
            or args.pairwise_target_margin < 0.0
            or args.pairwise_prediction_margin < 0.0):
        raise SystemExit("pairwise weight/margins must be non-negative")
    if args.residual_limit <= 0.0 or args.scale_floor <= 0.0:
        raise SystemExit("residual-limit and scale-floor must be positive")
    if (args.gate_min_absolute_improvement < 0.0
            or args.gate_min_relative_improvement < 0.0):
        raise SystemExit("gate improvement thresholds must be non-negative")
    if args.min_component_support <= 0 or args.min_pairwise_pairs <= 0:
        raise SystemExit("gate support thresholds must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_samples = _load_samples(args.train_data, args.max_train_samples)
    val_samples = _load_samples(args.val_data, args.max_val_samples)
    if not train_samples or not val_samples:
        raise SystemExit(
            "no L0-labelled samples; collect with record_lyapunov_l0=True"
        )
    train_units, val_units = _validate_disjoint_splits(
        train_samples, val_samples
    )

    functional, station_ids, num_bins = _functional_from_sample(
        train_samples[0], args.device
    )
    resolved_l0_config = _resolved_l0_config(train_samples[0])
    collection_signature = _l0_collection_signature(train_samples[0])
    reference_work_capacity = functional.work_capacity.detach().cpu().tolist()
    reference_arrival_capacity = (
        functional.arrival_capacity.detach().cpu().tolist()
    )
    _validate_l0_schema(
        train_samples,
        station_ids,
        resolved_l0_config,
        reference_work_capacity,
        reference_arrival_capacity,
        "train",
        reference_collection_signature=collection_signature,
    )
    _validate_l0_schema(
        val_samples,
        station_ids,
        resolved_l0_config,
        reference_work_capacity,
        reference_arrival_capacity,
        "val",
        reference_collection_signature=collection_signature,
    )
    data_manifest = {
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_units": [list(unit) for unit in train_units],
        "val_units": [list(unit) for unit in val_units],
        "train_sources": sorted({
            sample["_source_path"] for sample in train_samples
        }),
        "val_sources": sorted({
            sample["_source_path"] for sample in val_samples
        }),
        "continuation_policies": sorted({
            str(sample.get("continuation_policy", "unknown"))
            for sample in (*train_samples, *val_samples)
        }),
        "l0_collection_signature": list(collection_signature),
    }
    example = train_samples[0]
    model, model_config = load_frozen_world_model(
        args.checkpoint, example, len(station_ids), args.device
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    latent_dim = int(model_config.get("hidden_dim", 64))
    raw_demand_dim = int(example["demand_context"].numel())

    print(f"precomputing train={len(train_samples)} val={len(val_samples)}")
    train_rows = _precompute(model, train_samples, station_ids, args.device)
    val_rows = _precompute(model, val_samples, station_ids, args.device)
    del train_samples, val_samples

    component_scales = _component_residual_scales(
        train_rows, floor=args.scale_floor
    )
    training_component_support = _component_training_support(train_rows)
    demand_scales = _demand_residual_scales(
        train_rows, floor=args.scale_floor
    )
    ledger_loss_scales = _ledger_loss_scales(
        train_rows, functional, args.device, floor=args.scale_floor
    )
    head = LyapunovComponentHead(
        latent_dim=latent_dim,
        demand_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        num_stations=len(station_ids),
        num_arrival_bins=num_bins,
        residual_scales=component_scales,
        residual_limit=args.residual_limit,
    ).to(args.device)
    head.set_component_gates({
        name: row["active"]
        for name, row in training_component_support.items()
    })
    demand_head = EndpointDemandPredictor(
        latent_dim=latent_dim,
        encoded_demand_dim=latent_dim,
        raw_demand_dim=raw_demand_dim,
        hidden_dim=args.hidden_dim,
        residual_scale=demand_scales,
        residual_limit=args.residual_limit,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        list(head.parameters()) + list(demand_head.parameters()), lr=args.lr
    )
    weights = LyapunovLossWeights(
        component=args.component_weight,
        value=args.value_weight,
        drift=args.drift_weight,
        balance=args.balance_weight,
        sign=args.sign_weight,
    )

    evaluation_kwargs = {
        "pairwise_target_margin": args.pairwise_target_margin,
        "gate_min_absolute_improvement": (
            args.gate_min_absolute_improvement
        ),
        "gate_min_relative_improvement": (
            args.gate_min_relative_improvement
        ),
        "min_component_support": args.min_component_support,
        "min_pairwise_pairs": args.min_pairwise_pairs,
    }
    baseline_metrics = _evaluate(
        head, demand_head, model, functional, val_rows,
        args.device, weights, args.demand_weight,
        component_scales, demand_scales, ledger_loss_scales,
        **evaluation_kwargs,
    )
    best = baseline_metrics["selection_score"]
    best_state = (
        copy.deepcopy(head.state_dict()),
        copy.deepcopy(demand_head.state_dict()),
        baseline_metrics,
    )
    patience = 0
    history = [{
        "epoch": 0,
        "train_loss": None,
        "train_pairwise_loss": None,
        "train_pairwise_pairs": 0,
        **{f"val_{key}": value for key, value in baseline_metrics.items()},
    }]
    train_groups = _training_groups(train_rows)
    for epoch in range(1, args.epochs + 1):
        head.train()
        demand_head.train()
        random.shuffle(train_groups)
        epoch_losses = []
        epoch_pairwise = []
        epoch_pair_count = 0
        for group in train_groups:
            member_losses = []
            predicted_drifts = []
            target_drifts = []
            for row in group:
                zH, e0 = (row[key].to(args.device)
                           for key in ("zH", "e0"))
                d0_raw = row["d0_raw"].to(args.device)
                scores = row["bottleneck"].to(args.device)
                current_target = _to_device_prediction(
                    row["current_target"], args.device
                )
                future_target = _to_device_prediction(
                    row["future_target"], args.device
                )
                raw_future, e_future = demand_head.predict_embedding(
                    zH, e0, d0_raw, model.demand_encoder,
                    apply_gate=False,
                )
                station_node_ids = row["station_node_ids"].tolist()
                future = head(
                    zH, e_future, station_node_ids, scores,
                    anchor=current_target,
                    # Fields with no train support (currently traffic) are
                    # hard-frozen at the analytic persistence anchor.
                    apply_component_gates=True,
                )
                losses = compute_lyapunov_training_loss(
                    current_target,
                    future,
                    current_target,
                    future_target,
                    functional,
                    balance_target=torch.tensor(
                        row["balance_target"], device=args.device
                    ),
                    weights=weights,
                    component_scales=component_scales,
                    value_scale=ledger_loss_scales["value"],
                    drift_scale=ledger_loss_scales["drift"],
                    balance_scale=ledger_loss_scales["balance"],
                )
                demand_loss = _normalized_demand_loss(
                    raw_future,
                    row["future_demand"].to(args.device),
                    demand_scales,
                )
                member_losses.append(
                    losses["total"] + args.demand_weight * demand_loss
                )
                predicted_drifts.append(losses["predicted_drift"])
                target_drifts.append(float(losses["target_drift"].detach()))
            pairwise_loss, pair_count = _pairwise_rank_loss(
                predicted_drifts,
                target_drifts,
                target_margin=args.pairwise_target_margin,
                prediction_margin=args.pairwise_prediction_margin,
            )
            loss = (
                torch.stack(member_losses).mean()
                + args.pairwise_weight * pairwise_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(head.parameters()) + list(demand_head.parameters()),
                args.grad_clip,
            )
            optimizer.step()
            epoch_losses.append(float(loss))
            epoch_pairwise.append(float(pairwise_loss))
            epoch_pair_count += pair_count

        metrics = _evaluate(
            head, demand_head, model, functional, val_rows,
            args.device, weights, args.demand_weight,
            component_scales, demand_scales, ledger_loss_scales,
            **evaluation_kwargs,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "train_pairwise_loss": float(np.mean(epoch_pairwise)),
            "train_pairwise_pairs": int(epoch_pair_count),
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} train={row['train_loss']:.6f} "
            f"val={metrics['loss']:.6f} "
            f"sign={metrics['drift_sign_accuracy']} "
            f"pair={metrics['drift_pairwise_accuracy']}"
        )
        if metrics["selection_score"] < best - 1e-7:
            best = metrics["selection_score"]
            best_state = (
                copy.deepcopy(head.state_dict()),
                copy.deepcopy(demand_head.state_dict()),
                metrics,
            )
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is None:
        raise SystemExit("training produced no finite checkpoint")
    head_state, demand_state, _ = best_state
    head.load_state_dict(head_state)
    demand_head.load_state_dict(demand_state)
    raw_metrics = _evaluate(
        head, demand_head, model, functional, val_rows,
        args.device, weights, args.demand_weight,
        component_scales, demand_scales, ledger_loss_scales,
        **evaluation_kwargs,
    )
    proposed_deployment_component_gates = {
        name: bool(
            training_component_support[name]["active"]
            and raw_metrics["component_gates"][name]["enabled"]
        )
        for name in LyapunovPrediction._fields
    }
    head.set_component_gates(proposed_deployment_component_gates)
    demand_head.set_enabled(raw_metrics["gates"]["demand"])
    deployed_metrics = _evaluate(
        head, demand_head, model, functional, val_rows,
        args.device, weights, args.demand_weight,
        component_scales, demand_scales, ledger_loss_scales,
        apply_demand_gate=True,
        **evaluation_kwargs,
    )
    required_physical_gates = ("scalar_l", "drift", "drift_sign", "pairwise")
    deployment_physical_passed = all(
        bool(deployed_metrics["gates"][name])
        for name in required_physical_gates
    )
    if not deployment_physical_passed:
        # Component-wise fallback can alter the combined scalar/drift.  Never
        # save that partially gated mixture unless the mixture itself passes
        # all held-out physical direction gates.
        deployment_component_gates = {
            name: False for name in LyapunovPrediction._fields
        }
        head.set_component_gates(deployment_component_gates)
        deployed_metrics = _evaluate(
            head, demand_head, model, functional, val_rows,
            args.device, weights, args.demand_weight,
            component_scales, demand_scales, ledger_loss_scales,
            apply_demand_gate=True,
            **evaluation_kwargs,
        )
        deployment_mode = "analytic_component_fallback"
    else:
        deployment_component_gates = proposed_deployment_component_gates
        deployment_mode = "validated_component_residuals"
    head_state = copy.deepcopy(head.state_dict())
    demand_state = copy.deepcopy(demand_head.state_dict())
    payload = {
        "schema_version": SCHEMA_VERSION,
        # Training this additive head closes the component/demand signature,
        # but it does not by itself certify rollout-latent consistency, OOD
        # coverage, or the online drift margins required by the design.
        "online_ready": False,
        "online_blocker": "offline_drift_and_ood_gates_not_passed",
        "lyapunov_component_head": head_state,
        "endpoint_demand_predictor": demand_state,
        "config": {
            "latent_dim": latent_dim,
            "raw_demand_dim": raw_demand_dim,
            "hidden_dim": args.hidden_dim,
            "num_stations": len(station_ids),
            "num_arrival_bins": num_bins,
            "station_ids": station_ids,
            "prediction_semantics": (
                "analytic_current_anchor_plus_endpoint_component_residual"
            ),
            "endpoint_demand_semantics": (
                "initial_raw_demand_plus_endpoint_residual"
            ),
            "component_residual_scales": component_scales,
            "demand_residual_scales": demand_scales.tolist(),
            "ledger_loss_scales": ledger_loss_scales,
            "residual_limit": args.residual_limit,
            "lyapunov_l0_config": resolved_l0_config,
            "base_checkpoint": os.path.abspath(args.checkpoint),
            "loss_weights": vars(weights),
            "demand_weight": args.demand_weight,
            "pairwise_weight": args.pairwise_weight,
            "pairwise_target_margin": args.pairwise_target_margin,
            "pairwise_prediction_margin": args.pairwise_prediction_margin,
            "checkpoint_selection": (
                "physical_gate_count_then_normalized_baseline_ratio"
            ),
            "training_component_support": training_component_support,
            # Keep the optimistic residual candidate separate from the
            # actually saved fallback configuration.  Otherwise a report can
            # be misread as deployable even when component fallback destroys
            # the combined drift/ranking guarantees.
            "raw_candidate_validation_gates": raw_metrics["gates"],
            "raw_candidate_component_gates": raw_metrics["component_gates"],
            "proposed_deployment_component_gates": (
                proposed_deployment_component_gates
            ),
            "deployment_component_gates": deployment_component_gates,
            "deployment_validation_gates": deployed_metrics["gates"],
            "deployment_mode": deployment_mode,
            "deployment_required_physical_gates": list(
                required_physical_gates
            ),
            "online_ready": False,
            "online_blocker": "offline_drift_and_ood_gates_not_passed",
            "intended_use": "offline_component_and_drift_diagnostics",
        },
        "data_manifest": data_manifest,
        "functional": {
            "work_capacity": functional.work_capacity.cpu().tolist(),
            "arrival_capacity": functional.arrival_capacity.cpu().tolist(),
            "work_weight": functional.work_weight,
            "station_weight": functional.station_weight,
            "traffic_weight": functional.traffic_weight,
            "stall_weight": functional.stall_weight,
            "plan_fail_weight": functional.plan_fail_weight,
            "arrival_weight": functional.arrival_weight,
            "station_safe_ratio": functional.station_safe_ratio,
        },
        "validation_raw_candidate": raw_metrics,
        "validation": deployed_metrics,
        "history": history,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(payload, args.output)
    report = os.path.splitext(args.output)[0] + "_report.json"
    with open(report, "w", encoding="utf-8") as handle:
        json.dump({
            key: value for key, value in payload.items()
            if key not in ("lyapunov_component_head", "endpoint_demand_predictor")
        }, handle, indent=2, ensure_ascii=False)
    print(f"saved: {args.output}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
