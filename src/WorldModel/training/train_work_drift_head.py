"""Train the finite-horizon work-only Layer-4 drift estimator.

The frozen base World Model performs an isolated H-step rollout under the
current candidate action and current known demand.  This trainer does not use
future demand, later scheduler decisions, a continuation policy or a TD tail.
It predicts endpoint station work, reconstructs raw analytic ``Delta L_work``,
and trains the within-candidate-group range coordinate used by Layer 3.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from WorldModel.core.lyapunov import LYAPUNOV_COLLECTION_SCHEMA_VERSION
from WorldModel.core.work_drift_head import (
    SCHEMA_VERSION as HEAD_SCHEMA_VERSION,
    WorkDriftHead,
    all_non_tie_pairwise_logistic_loss,
    group_range_normalise,
)
from WorldModel.evaluation.work_drift_layer4_protocol import (
    GROUP_RANGE_SEMANTICS,
    PREDICTION_SEMANTICS,
    formal_layer4_protocol,
)
from WorldModel.training.train_td_risk_v import load_frozen_world_model


TRAINING_REPORT_SCHEMA_VERSION = "work_drift_group_range_training_report_v1"


def _sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_paths(values: Iterable[str]) -> list[str]:
    paths: list[str] = []
    seen = set()
    for value in values:
        matches = sorted(glob.glob(value))
        for path in matches or [value]:
            absolute = os.path.abspath(path)
            if absolute not in seen:
                seen.add(absolute)
                paths.append(absolute)
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"missing data files: {missing}")
    return paths


def _load_label(sample: Mapping) -> str:
    for key in ("load_level", "load", "load_label", "run_id", "_source_path"):
        value = str(sample.get(key) or "")
        match = re.search(
            r"(?:^|[_\\/.-])(low|mid|high)(?:$|[_\\/.-])",
            value,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).lower()
    return "unknown"


def _sample_seed(sample: Mapping) -> int:
    value = sample.get("simulation_seed")
    if value is None:
        raise ValueError("Layer-4 data requires simulation_seed")
    return int(value)


def _group_key(sample: Mapping) -> tuple[str, str]:
    group_id = sample.get("candidate_group_id")
    if not group_id:
        raise ValueError("Layer-4 data requires candidate_group_id")
    run_id = str(sample.get("run_id") or sample.get("_source_path") or "unknown")
    return run_id, str(group_id)


def _load_samples(
    paths: Sequence[str],
    *,
    required_horizon: int = 10,
    max_samples: Optional[int] = None,
) -> tuple[list[dict], dict]:
    """Load strict isolated endpoint labels without silently changing H."""

    if required_horizon <= 0:
        raise ValueError("required_horizon must be positive")
    rows: list[dict] = []
    source_counts: dict[str, int] = {}
    required = (
        "node_history",
        "edge_index",
        "edge_features",
        "demand_context",
        "action_node",
        "action_global",
        "station_node_ids",
        "future_mask",
        "lyapunov_l0_start",
        "lyapunov_l0_end",
        "candidate_group_id",
        "simulation_seed",
    )
    for path in _expand_paths(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "samples" in payload:
            payload = payload["samples"]
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected list or payload['samples']")
        source_count = 0
        for index, original in enumerate(payload):
            if not isinstance(original, Mapping):
                raise ValueError(f"{path}[{index}]: sample must be a mapping")
            missing = [name for name in required if original.get(name) is None]
            if missing:
                raise ValueError(f"{path}[{index}]: missing fields {missing}")
            schema = str(original.get("lyapunov_l0_collection_schema_version"))
            if schema != LYAPUNOV_COLLECTION_SCHEMA_VERSION:
                raise ValueError(
                    f"{path}[{index}]: expected {LYAPUNOV_COLLECTION_SCHEMA_VERSION}, "
                    f"got {schema!r}"
                )
            continuation = str(original.get("rollout_continuation_mode"))
            if continuation != "isolated":
                raise ValueError(
                    f"{path}[{index}]: Layer 4 requires isolated continuation, "
                    f"got {continuation!r}"
                )
            if original.get("lyapunov_l0_valid") is False:
                raise ValueError(f"{path}[{index}]: invalid Lyapunov endpoint label")
            mask = torch.as_tensor(original["future_mask"]).bool().flatten()
            if mask.numel() != required_horizon or not bool(mask.all()):
                raise ValueError(
                    f"{path}[{index}]: Layer 4 freezes a fully valid H="
                    f"{required_horizon} endpoint; mask={mask.tolist()}"
                )
            sample = dict(original)
            sample["_source_path"] = os.path.abspath(path)
            _sample_seed(sample)
            _group_key(sample)
            rows.append(sample)
            source_count += 1
        source_counts[os.path.abspath(path)] = source_count
    truncated = False
    if max_samples is not None and len(rows) > max_samples:
        # Debug-only limiting must never cut a candidate group in half.
        grouped_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
        group_order: list[tuple[str, str]] = []
        for sample in rows:
            key = _group_key(sample)
            if key not in grouped_rows:
                group_order.append(key)
            grouped_rows[key].append(sample)
        selected: list[dict] = []
        for key in group_order:
            members = grouped_rows[key]
            if selected and len(selected) + len(members) > max_samples:
                break
            selected.extend(members)
        rows = selected
        truncated = True
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in rows:
        groups[_group_key(sample)].append(sample)
    singleton = [key for key, members in groups.items() if len(members) < 2]
    if singleton:
        raise ValueError(
            "candidate groups must remain complete with at least two actions; "
            f"singletons={singleton[:10]}"
        )
    duplicate_candidates = []
    for key, members in groups.items():
        identities = [
            str(
                sample.get("candidate_key")
                or sample.get("candidate_robot_id")
                or sample.get("action_identity")
                or ""
            )
            for sample in members
        ]
        nonempty = [value for value in identities if value]
        if nonempty and len(nonempty) != len(set(nonempty)):
            duplicate_candidates.append(key)
    if duplicate_candidates:
        raise ValueError(
            "duplicate candidate identity within groups: "
            f"{duplicate_candidates[:10]}"
        )
    return rows, {
        "source_counts": source_counts,
        "required_horizon": required_horizon,
        "strict_full_horizon": True,
        "candidate_groups": len(groups),
        "complete_group_truncation_applied": truncated,
    }


def _split_manifest(samples: Sequence[Mapping]) -> dict:
    return {
        "samples": len(samples),
        "candidate_groups": len({_group_key(sample) for sample in samples}),
        "simulation_seeds": sorted({_sample_seed(sample) for sample in samples}),
        "loads": sorted({_load_label(sample) for sample in samples}),
        "run_ids": sorted({str(sample.get("run_id") or "unknown") for sample in samples}),
        "source_files": sorted({str(sample.get("_source_path")) for sample in samples}),
    }


def _validate_disjoint_splits(
    train_samples: Sequence[Mapping],
    val_samples: Sequence[Mapping],
) -> None:
    train_seeds = {_sample_seed(sample) for sample in train_samples}
    val_seeds = {_sample_seed(sample) for sample in val_samples}
    overlap = sorted(train_seeds & val_seeds)
    if overlap:
        raise ValueError(f"train/validation simulation_seed leakage: {overlap}")


def _station_value(mapping: Mapping, station_id: int) -> float:
    if station_id in mapping:
        return float(mapping[station_id])
    if str(station_id) in mapping:
        return float(mapping[str(station_id)])
    raise KeyError(f"station {station_id} missing from analytic snapshot")


def _work_schema(samples: Sequence[Mapping], tolerance: float = 1e-5) -> dict:
    if not samples:
        raise ValueError("cannot resolve work schema from empty data")
    first = samples[0]
    start = first["lyapunov_l0_start"]
    station_ids = sorted(int(value) for value in start["station_work"])
    if not station_ids:
        raise ValueError("analytic station-work ledger is empty")
    capacity = float(start["work_capacity"])
    config = dict(first.get("lyapunov_l0_config") or {})
    work_weight = float(config.get("work_weight", 1.0))
    for index, sample in enumerate(samples):
        sample_config = dict(sample.get("lyapunov_l0_config") or {})
        if abs(float(sample_config.get("work_weight", 1.0)) - work_weight) > tolerance:
            raise ValueError(f"sample {index}: work_weight changed")
        graph_station_ids = torch.as_tensor(sample["station_node_ids"]).flatten()
        if graph_station_ids.numel() != len(station_ids):
            raise ValueError(f"sample {index}: station_node_ids length changed")
        for endpoint in ("lyapunov_l0_start", "lyapunov_l0_end"):
            snapshot = sample[endpoint]
            ids = sorted(int(value) for value in snapshot["station_work"])
            if ids != station_ids:
                raise ValueError(f"sample {index}: station ids changed at {endpoint}")
            if abs(float(snapshot["work_capacity"]) - capacity) > tolerance:
                raise ValueError(f"sample {index}: work capacity changed at {endpoint}")
            work = np.asarray([
                _station_value(snapshot["station_work"], station_id)
                for station_id in station_ids
            ], dtype=float)
            recomputed = 0.5 * work_weight * float(np.square(work / capacity).sum())
            stored = float(snapshot["components"]["work"])
            if abs(recomputed - stored) > tolerance * max(1.0, abs(stored)):
                raise ValueError(
                    f"sample {index}: stored L_work mismatch at {endpoint}: "
                    f"stored={stored} recomputed={recomputed}"
                )
    return {
        "station_ids": station_ids,
        "work_capacity": [capacity] * len(station_ids),
        "work_weight": work_weight,
    }


def _mean_max(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value.mean(dim=0), value.amax(dim=0)), dim=-1)


def _precompute(
    model,
    samples: Sequence[Mapping],
    *,
    station_ids: Sequence[int],
    work_capacity: Sequence[float],
    device: str,
    horizon: int,
) -> list[dict]:
    """Cache compact frozen-WM features, not full graph latent tensors."""

    rows: list[dict] = []
    capacity = torch.as_tensor(work_capacity, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for sample in samples:
            node_history = sample["node_history"].to(device)
            edge_index = sample["edge_index"].long().to(device)
            edge_features = sample["edge_features"].to(device)
            demand = sample["demand_context"].to(device)
            action_node = sample["action_node"].to(device)
            action_global = sample["action_global"].to(device)
            z0, e0, edge_attr = model.encode_state(
                node_history, edge_index, edge_features, demand
            )
            graph_station_ids = torch.as_tensor(
                sample["station_node_ids"], dtype=torch.long, device=device
            ).flatten()
            *_, z_start, z_end = model.rollout(
                z0,
                e0,
                edge_attr,
                action_node,
                action_global,
                edge_index,
                graph_station_ids.tolist(),
                K=horizon,
            )
            start_snapshot = sample["lyapunov_l0_start"]
            end_snapshot = sample["lyapunov_l0_end"]
            current_work = torch.tensor([
                _station_value(start_snapshot["station_work"], station_id)
                for station_id in station_ids
            ], dtype=torch.float32, device=device)
            endpoint_work = torch.tensor([
                _station_value(end_snapshot["station_work"], station_id)
                for station_id in station_ids
            ], dtype=torch.float32)
            current_ratio = current_work / capacity.to(device)
            global_context = torch.cat((
                _mean_max(z_start),
                _mean_max(z_end),
                e0,
            ), dim=-1)
            station_context = torch.cat((
                z_start.index_select(0, graph_station_ids),
                z_end.index_select(0, graph_station_ids),
                current_ratio.unsqueeze(-1),
            ), dim=-1)
            rows.append({
                "global_context": global_context.detach().cpu(),
                "station_context": station_context.detach().cpu(),
                "current_station_work": current_work.detach().cpu(),
                "endpoint_station_work": endpoint_work,
                "target_raw_work_drift": float(
                    end_snapshot["components"]["work"]
                    - start_snapshot["components"]["work"]
                ),
                "group_key": _group_key(sample),
                "seed": _sample_seed(sample),
                "load": _load_label(sample),
                "run_id": str(sample.get("run_id") or "unknown"),
                "candidate_key": str(
                    sample.get("candidate_key")
                    or sample.get("candidate_robot_id")
                    or len(rows)
                ),
                "source_path": str(sample.get("_source_path")),
            })
    return rows


def _manifest_from_compact_rows(rows: Sequence[Mapping]) -> dict:
    return {
        "samples": len(rows),
        "candidate_groups": len({row["group_key"] for row in rows}),
        "simulation_seeds": sorted({int(row["seed"]) for row in rows}),
        "loads": sorted({str(row["load"]) for row in rows}),
        "run_ids": sorted({str(row["run_id"]) for row in rows}),
        "source_files": sorted({str(row["source_path"]) for row in rows}),
    }


def _truncate_compact_rows_by_group(
    rows: Sequence[dict], max_samples: Optional[int]
) -> tuple[list[dict], bool]:
    if max_samples is None or len(rows) <= max_samples:
        return list(rows), False
    grouped: dict[object, list[dict]] = defaultdict(list)
    order = []
    for row in rows:
        key = row["group_key"]
        if key not in grouped:
            order.append(key)
        grouped[key].append(row)
    selected = []
    for key in order:
        members = grouped[key]
        if selected and len(selected) + len(members) > max_samples:
            break
        selected.extend(members)
    return selected, True


def _precompute_paths(
    model,
    paths: Sequence[str],
    *,
    expected_schema: Mapping,
    device: str,
    horizon: int,
    max_samples: Optional[int] = None,
) -> tuple[list[dict], dict, dict]:
    """Stream raw arm files and retain only compact frozen-WM features.

    Keeping all counterfactual samples resident would retain graph histories,
    action fields and H-step labels for roughly twelve thousand candidates.
    Processing one arm at a time bounds raw-data memory while the compact
    station/global features remain small enough for head training.
    """

    rows: list[dict] = []
    source_counts = {}
    expanded = _expand_paths(paths)
    for file_index, path in enumerate(expanded, 1):
        samples, audit = _load_samples([path], required_horizon=horizon)
        observed_schema = _work_schema(samples)
        if observed_schema != dict(expected_schema):
            raise ValueError(
                f"{path}: work schema differs from the frozen training schema"
            )
        compact = _precompute(
            model,
            samples,
            station_ids=expected_schema["station_ids"],
            work_capacity=expected_schema["work_capacity"],
            device=device,
            horizon=horizon,
        )
        rows.extend(compact)
        source_counts.update(audit["source_counts"])
        print(
            f"  precomputed arm {file_index}/{len(expanded)}: "
            f"{os.path.basename(os.path.dirname(path))} "
            f"candidates={len(compact)}"
        )
        del samples, compact
    rows, truncated = _truncate_compact_rows_by_group(rows, max_samples)
    groups = _groups(rows)
    if any(len(group) < 2 for group in groups):
        raise ValueError("streamed precompute produced an incomplete candidate group")
    manifest = _manifest_from_compact_rows(rows)
    audit = {
        "source_counts": source_counts,
        "source_files_processed_sequentially": len(expanded),
        "required_horizon": horizon,
        "strict_full_horizon": True,
        "candidate_groups": len(groups),
        "complete_group_truncation_applied": truncated,
        "raw_samples_retained_after_precompute": False,
    }
    return rows, manifest, audit


def _groups(rows: Sequence[Mapping]) -> list[list[Mapping]]:
    grouped: dict[object, list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[row["group_key"]].append(row)
    return list(grouped.values())


def _robust_scales(rows: Sequence[Mapping], floor: float) -> tuple[list[float], float]:
    if floor <= 0.0:
        raise ValueError("scale floor must be positive")
    endpoint_delta = torch.stack([
        (row["endpoint_station_work"] - row["current_station_work"]).abs()
        for row in rows
    ])
    station = torch.quantile(endpoint_delta, 0.75, dim=0).clamp_min(floor)
    drift = torch.tensor([
        abs(float(row["target_raw_work_drift"])) for row in rows
    ], dtype=torch.float32)
    drift_scale = max(float(torch.quantile(drift, 0.75)), floor)
    return station.tolist(), drift_scale


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.size < 2 or x.std() <= 0.0 or y.std() <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.size < 2:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def _quantiles(values: Sequence[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    if not array.size:
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


def _score_group(
    head: WorkDriftHead,
    members: Sequence[Mapping],
    device: str,
) -> dict:
    global_context = torch.stack([row["global_context"] for row in members]).to(device)
    station_context = torch.stack([row["station_context"] for row in members]).to(device)
    current = torch.stack([row["current_station_work"] for row in members]).to(device)
    endpoint = torch.stack([row["endpoint_station_work"] for row in members]).to(device)
    # Preserve the simulator ledger's float64 ordering before the bounded
    # group transform.  Casting a very small but non-zero analytic difference
    # to float32 here can manufacture a tie that did not exist in Layer 3.
    target_raw = torch.tensor([
        float(row["target_raw_work_drift"]) for row in members
    ], dtype=torch.float64)
    with torch.no_grad():
        prediction = head.predict_from_features(global_context, station_context, current)
        predicted_raw = prediction.raw_work_drift
        predicted_endpoint = prediction.endpoint_station_work
        target_range = group_range_normalise(target_raw)
        predicted_range = group_range_normalise(
            predicted_raw.detach().to(dtype=torch.float64, device="cpu")
        )
    return {
        "target_raw": target_raw.numpy(),
        "predicted_raw": predicted_raw.detach().cpu().double().numpy(),
        "target_range": target_range.numpy(),
        "predicted_range": predicted_range.numpy(),
        "endpoint_absolute_error": (
            predicted_endpoint - endpoint
        ).abs().cpu().numpy(),
        "seed": int(members[0]["seed"]),
        "load": str(members[0]["load"]),
        "group_key": members[0]["group_key"],
        "candidate_count": len(members),
    }


def _aggregate_group_records(records: Sequence[Mapping]) -> dict:
    if not records:
        return {"groups": 0, "samples": 0}
    truth = np.concatenate([row["target_range"] for row in records])
    predicted = np.concatenate([row["predicted_range"] for row in records])
    # Every fixed-context candidate group receives equal total mass.  Without
    # this weighting, contexts with ten candidates count five times as much as
    # two-candidate contexts even though the independent decision unit is the
    # context, not an individual candidate row.
    sample_weights = np.concatenate([
        np.full(len(row["target_range"]), 1.0 / len(row["target_range"]))
        for row in records
    ]).astype(np.float64)
    sample_weights /= float(sample_weights.sum())
    raw_truth = np.concatenate([row["target_raw"] for row in records])
    raw_predicted = np.concatenate([row["predicted_raw"] for row in records])
    endpoint_error = np.concatenate([
        row["endpoint_absolute_error"].reshape(-1) for row in records
    ])
    truth_mean = float(np.dot(sample_weights, truth))
    predicted_mean = float(np.dot(sample_weights, predicted))
    truth_centered = truth - truth_mean
    predicted_centered = predicted - predicted_mean
    scale = float(np.sqrt(np.dot(sample_weights, np.square(truth_centered))))
    predicted_scale = float(np.sqrt(np.dot(
        sample_weights, np.square(predicted_centered)
    )))
    weighted_covariance = float(np.dot(
        sample_weights, truth_centered * predicted_centered
    ))
    weighted_pearson = (
        weighted_covariance / (scale * predicted_scale)
        if scale > 0.0 and predicted_scale > 0.0 else None
    )
    raw_scale = float(raw_truth.std())
    group_pairwise = []
    pair_total = 0
    top1 = 0.0
    random_top1 = []
    regret = []
    normalised_regret = []
    uniform_random_normalised_regret = []
    normalised_regret_improvement = []
    group_raw_ranges = []
    for row in records:
        true = np.asarray(row["target_raw"], dtype=float)
        pred = np.asarray(row["predicted_raw"], dtype=float)
        true_min = float(true.min())
        predicted_min = float(pred.min())
        true_best = set(np.flatnonzero(true == true_min).tolist())
        predicted_best = np.flatnonzero(pred == predicted_min)
        chosen = int(predicted_best[0])
        top1 += float(chosen in true_best)
        random_top1.append(len(true_best) / len(true))
        chosen_regret = float(true[chosen] - true_min)
        raw_range = float(true.max() - true.min())
        group_raw_ranges.append(raw_range)
        regret.append(chosen_regret)
        if raw_range > 0.0:
            model_normalised_regret = chosen_regret / raw_range
            random_normalised_regret = float(
                np.mean((true - true_min) / raw_range)
            )
            normalised_regret.append(model_normalised_regret)
            uniform_random_normalised_regret.append(
                random_normalised_regret
            )
            normalised_regret_improvement.append(
                random_normalised_regret - model_normalised_regret
            )
        correct = []
        for left in range(len(true)):
            for right in range(left + 1, len(true)):
                true_gap = true[left] - true[right]
                if true_gap == 0.0:
                    continue
                pred_gap = pred[left] - pred[right]
                correct.append(0.5 if pred_gap == 0.0 else float(
                    np.sign(pred_gap) == np.sign(true_gap)
                ))
                pair_total += 1
        if correct:
            group_pairwise.append(float(np.mean(correct)))
    weighted_rmse = float(np.sqrt(np.dot(
        sample_weights, np.square(predicted - truth)
    )))
    nrmse = (
        float(weighted_rmse / scale)
        if scale > 0.0 else None
    )
    raw_nrmse = (
        float(np.sqrt(np.mean(np.square(raw_predicted - raw_truth))) / raw_scale)
        if raw_scale > 0.0 else None
    )
    return {
        "groups": len(records),
        "samples": int(len(truth)),
        "continuous_group_range": {
            "normalised_rmse": nrmse,
            "zero_predictor_normalised_rmse": 1.0 if scale > 0.0 else None,
            "rmse": weighted_rmse,
            "truth_std": scale,
            "pearson": weighted_pearson,
            "spearman": _spearman(truth, predicted),
            "all_non_tie_pair_concordance": (
                float(np.mean(group_pairwise)) if group_pairwise else None
            ),
            "all_non_tie_pairs": int(pair_total),
            "pairwise_weighting": "candidate_group_equal",
            "sample_weighting": "candidate_group_equal",
            "top1_min_drift_accuracy": float(top1 / len(records)),
            "random_tie_aware_top1_baseline": float(np.mean(random_top1)),
            "selection_regret": _quantiles(regret),
            "normalised_selection_regret": _quantiles(normalised_regret),
            "uniform_random_normalised_selection_regret": _quantiles(
                uniform_random_normalised_regret
            ),
            "normalised_selection_regret_improvement_over_uniform_random": (
                _quantiles(normalised_regret_improvement)
            ),
        },
        "raw_work_drift": {
            "normalised_rmse": raw_nrmse,
            "bias_predicted_minus_true": float(np.mean(raw_predicted - raw_truth)),
            "pearson": _pearson(raw_truth, raw_predicted),
            "spearman": _spearman(raw_truth, raw_predicted),
            "truth": _quantiles(raw_truth),
            "predicted": _quantiles(raw_predicted),
            "action_selection_primary_evidence": False,
        },
        "endpoint_station_work": {
            "absolute_error": _quantiles(endpoint_error),
        },
        "candidate_group_raw_range": _quantiles(group_raw_ranges),
    }


def _bootstrap_metrics(
    records: Sequence[Mapping],
    *,
    repeats: int,
    random_seed: int,
) -> dict:
    if repeats <= 0:
        return {}
    seeds = sorted({int(row["seed"]) for row in records})
    if len(seeds) < 2:
        return {}
    by_seed = {
        seed: [row for row in records if int(row["seed"]) == seed]
        for seed in seeds
    }
    sufficient = {
        seed: _aggregate_group_records(seed_records)[
            "continuous_group_range"
        ]
        for seed, seed_records in by_seed.items()
    }
    rng = np.random.default_rng(random_seed)
    values = defaultdict(list)
    for _ in range(repeats):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        rows = [sufficient[int(seed)] for seed in sampled]
        for name in (
            "normalised_rmse",
            "spearman",
            "all_non_tie_pair_concordance",
            "top1_min_drift_accuracy",
        ):
            finite = [
                float(row[name]) for row in rows
                if row.get(name) is not None
                and math.isfinite(float(row[name]))
            ]
            if finite:
                values[name].append(float(np.mean(finite)))
        regret_improvements = [
            row.get(
                "normalised_selection_regret_improvement_over_uniform_random",
                {},
            ).get("mean")
            for row in rows
        ]
        regret_improvements = [
            float(value) for value in regret_improvements
            if value is not None and math.isfinite(float(value))
        ]
        if regret_improvements:
            values[
                "normalised_selection_regret_improvement_over_uniform_random"
            ].append(float(np.mean(regret_improvements)))
        top1_improvements = [
            float(row["top1_min_drift_accuracy"])
            - float(row["random_tie_aware_top1_baseline"])
            for row in rows
            if row.get("top1_min_drift_accuracy") is not None
            and row.get("random_tie_aware_top1_baseline") is not None
        ]
        if top1_improvements:
            values["top1_improvement_over_tie_aware_random"].append(
                float(np.mean(top1_improvements))
            )
    return {
        f"{name}_ci95_seed_cluster_bootstrap": (
            [float(np.quantile(rows, 0.025)), float(np.quantile(rows, 0.975))]
            if rows else None
        )
        for name, rows in values.items()
    }


def evaluate_rows(
    head: WorkDriftHead,
    rows: Sequence[Mapping],
    *,
    device: str,
    bootstrap_repeats: int = 0,
    random_seed: int = 20260716,
) -> dict:
    head.eval()
    records = [_score_group(head, group, device) for group in _groups(rows)]
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

    def _range_label(record: Mapping) -> str:
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

    raw_range_strata = {}
    for label in (
        "zero", "nonzero_q1", "nonzero_q2", "nonzero_q3", "nonzero_q4"
    ):
        selected = [row for row in records if _range_label(row) == label]
        if selected:
            raw_range_strata[label] = _aggregate_group_records(selected)[
                "continuous_group_range"
            ]
    result["raw_range_strata"] = {
        "role": "diagnostic_only_not_a_gap_gate",
        "nonzero_quartile_boundaries": boundaries,
        "strata": raw_range_strata,
    }
    result["independent_seed_clusters"] = len(by_seed)
    continuous = result["continuous_group_range"]
    nrmse = continuous.get("normalised_rmse")
    pairwise = continuous.get("all_non_tie_pair_concordance")
    top1 = continuous.get("top1_min_drift_accuracy")
    random_top1 = continuous.get("random_tie_aware_top1_baseline")
    pairwise_value = 0.5 if pairwise is None else float(pairwise)
    top1_value = 0.0 if top1 is None else float(top1)
    random_top1_value = 0.0 if random_top1 is None else float(random_top1)
    result["selection_score"] = (
        float(nrmse if nrmse is not None else 10.0)
        - 0.25 * (pairwise_value - 0.5)
        - 0.10 * (top1_value - random_top1_value)
    )
    return result


def _group_training_loss(
    head: WorkDriftHead,
    members: Sequence[Mapping],
    *,
    device: str,
    station_scales: torch.Tensor,
    raw_drift_scale: float,
    group_range_weight: float,
    raw_drift_weight: float,
    endpoint_weight: float,
    pairwise_weight: float,
    pairwise_temperature: float,
) -> tuple[torch.Tensor, dict]:
    global_context = torch.stack([row["global_context"] for row in members]).to(device)
    station_context = torch.stack([row["station_context"] for row in members]).to(device)
    current = torch.stack([row["current_station_work"] for row in members]).to(device)
    endpoint = torch.stack([row["endpoint_station_work"] for row in members]).to(device)
    target_raw_values = [
        float(row["target_raw_work_drift"]) for row in members
    ]
    target_raw = torch.tensor(
        target_raw_values, dtype=torch.float32, device=device
    )
    prediction = head.predict_from_features(global_context, station_context, current)
    predicted_range = group_range_normalise(prediction.raw_work_drift)
    target_range = group_range_normalise(torch.tensor(
        target_raw_values, dtype=torch.float64
    )).to(device=device, dtype=predicted_range.dtype)
    range_loss = F.smooth_l1_loss(predicted_range, target_range)
    raw_loss = F.smooth_l1_loss(
        prediction.raw_work_drift / float(raw_drift_scale),
        target_raw / float(raw_drift_scale),
    )
    scales = station_scales.to(device=device, dtype=endpoint.dtype)
    endpoint_loss = F.smooth_l1_loss(
        (prediction.endpoint_station_work - endpoint) / scales,
        torch.zeros_like(endpoint),
    )
    pairwise_loss, pair_count = all_non_tie_pairwise_logistic_loss(
        predicted_range,
        target_range,
        temperature=pairwise_temperature,
    )
    total = (
        group_range_weight * range_loss
        + raw_drift_weight * raw_loss
        + endpoint_weight * endpoint_loss
        + pairwise_weight * pairwise_loss
    )
    return total, {
        "group_range": float(range_loss.detach()),
        "raw_drift": float(raw_loss.detach()),
        "endpoint": float(endpoint_loss.detach()),
        "pairwise": float(pairwise_loss.detach()),
        "pairs": pair_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-data", nargs="+", required=True)
    parser.add_argument("--val-data", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--group-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--group-range-weight", type=float, default=1.0)
    parser.add_argument("--raw-drift-weight", type=float, default=0.25)
    parser.add_argument("--endpoint-weight", type=float, default=0.25)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--pairwise-temperature", type=float, default=0.25)
    parser.add_argument("--residual-limit", type=float, default=4.0)
    parser.add_argument("--scale-floor", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    non_negative = (
        args.group_range_weight,
        args.raw_drift_weight,
        args.endpoint_weight,
        args.pairwise_weight,
        args.weight_decay,
    )
    if any(value < 0.0 for value in non_negative):
        raise SystemExit("loss weights and weight decay must be non-negative")
    if min(
        args.horizon,
        args.hidden_dim,
        args.epochs,
        args.patience,
        args.group_batch_size,
    ) <= 0:
        raise SystemExit("horizon/dim/epochs/patience/batch size must be positive")
    if min(
        args.lr,
        args.pairwise_temperature,
        args.residual_limit,
        args.scale_floor,
        args.grad_clip,
    ) <= 0.0:
        raise SystemExit("learning rate/temperature/scales/clip must be positive")
    output = Path(args.output)
    report_path = Path(args.report) if args.report else output.with_name(
        output.stem + "_report.json"
    )
    if not args.overwrite:
        existing = [str(path) for path in (output, report_path) if path.exists()]
        if existing:
            raise SystemExit(f"refusing to overwrite existing artifacts: {existing}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_paths = _expand_paths(args.train_data)
    val_paths = _expand_paths(args.val_data)
    first_samples, _ = _load_samples(
        [train_paths[0]], required_horizon=args.horizon
    )
    if not first_samples:
        raise SystemExit("training data are empty")
    schema = _work_schema(first_samples)
    station_ids = schema["station_ids"]

    model, model_config = load_frozen_world_model(
        args.checkpoint,
        first_samples[0],
        len(station_ids),
        args.device,
    )
    latent_dim = int(model_config.get("hidden_dim", 64))
    print(
        "precomputing frozen-WM compact features one arm at a time: "
        f"train_files={len(train_paths)} val_files={len(val_paths)} "
        f"H={args.horizon}"
    )
    train_rows, train_manifest, train_audit = _precompute_paths(
        model,
        train_paths,
        expected_schema=schema,
        device=args.device,
        horizon=args.horizon,
        max_samples=args.max_train_samples,
    )
    val_rows, val_manifest, val_audit = _precompute_paths(
        model,
        val_paths,
        expected_schema=schema,
        device=args.device,
        horizon=args.horizon,
        max_samples=args.max_val_samples,
    )
    if not train_rows or not val_rows:
        raise SystemExit("train and validation data must both be non-empty")
    overlap = sorted(
        set(train_manifest["simulation_seeds"])
        & set(val_manifest["simulation_seeds"])
    )
    if overlap:
        raise SystemExit(f"train/validation simulation_seed leakage: {overlap}")
    del first_samples, model

    station_scales, raw_drift_scale = _robust_scales(
        train_rows, args.scale_floor
    )
    head = WorkDriftHead(
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        num_stations=len(station_ids),
        work_capacity=schema["work_capacity"],
        work_weight=schema["work_weight"],
        residual_scale=station_scales,
        residual_limit=args.residual_limit,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_groups = _groups(train_rows)

    baseline = evaluate_rows(head, val_rows, device=args.device)
    best_score = float(baseline["selection_score"])
    best_epoch = 0
    best_state = copy.deepcopy(head.state_dict())
    best_metrics = baseline
    patience = 0
    history = [{"epoch": 0, "train_loss": None, "validation": baseline}]

    station_scale_tensor = torch.as_tensor(station_scales, dtype=torch.float32)
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_groups)
        head.train()
        loss_rows = []
        component_rows = defaultdict(list)
        for offset in range(0, len(train_groups), args.group_batch_size):
            batch = train_groups[offset:offset + args.group_batch_size]
            losses = []
            stats = []
            for group in batch:
                loss, row = _group_training_loss(
                    head,
                    group,
                    device=args.device,
                    station_scales=station_scale_tensor,
                    raw_drift_scale=raw_drift_scale,
                    group_range_weight=args.group_range_weight,
                    raw_drift_weight=args.raw_drift_weight,
                    endpoint_weight=args.endpoint_weight,
                    pairwise_weight=args.pairwise_weight,
                    pairwise_temperature=args.pairwise_temperature,
                )
                losses.append(loss)
                stats.append(row)
            batch_loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()
            loss_rows.append(float(batch_loss.detach()))
            for row in stats:
                for name in ("group_range", "raw_drift", "endpoint", "pairwise"):
                    component_rows[name].append(row[name])
                component_rows["pairs"].append(row["pairs"])

        metrics = evaluate_rows(head, val_rows, device=args.device)
        score = float(metrics["selection_score"])
        history_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(loss_rows)),
            "train_components": {
                name: float(np.mean(values)) if values else None
                for name, values in component_rows.items()
            },
            "validation": metrics,
        }
        history.append(history_row)
        continuous = metrics["continuous_group_range"]
        print(
            f"epoch {epoch:03d} train={history_row['train_loss']:.6f} "
            f"val_nrmse={continuous.get('normalised_rmse')} "
            f"pair={continuous.get('all_non_tie_pair_concordance')} "
            f"top1={continuous.get('top1_min_drift_accuracy')}"
        )
        if score < best_score - 1e-7:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            best_metrics = metrics
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    head.load_state_dict(best_state)
    best_metrics = evaluate_rows(
        head,
        val_rows,
        device=args.device,
        bootstrap_repeats=1000,
        random_seed=args.seed + 17,
    )
    checkpoint_hash = _sha256_file(args.checkpoint)
    semantics = {
        "layer": 4,
        "question": "Can the frozen World Model estimate isolated H-step analytic work drift?",
        "prediction": PREDICTION_SEMANTICS,
        "target": "raw analytic Delta L_work from endpoint station-work ledger",
        "candidate_transform": GROUP_RANGE_SEMANTICS,
        "current_demand_only": True,
        "unknown_future_orders": False,
        "future_demand_predictor": False,
        "continuation_policy": False,
        "td_tail": False,
        "greedy_or_external_policy": False,
        "direct_action_embedding_to_head": False,
        "hard_raw_gap_gate": False,
        "load_gate": False,
        "rollout_continuation_mode": "isolated",
        "horizon": args.horizon,
        "online_candidate_supersets_above_training_top_m": "NOT_EVALUATED",
    }
    payload = {
        "schema_version": HEAD_SCHEMA_VERSION,
        "online_ready": False,
        "online_blocker": "layer4_held_out_and_layer5_closed_loop_not_yet_certified",
        "head_config": head.checkpoint_config(),
        "head_state_dict": copy.deepcopy(head.state_dict()),
        "base_world_model": {
            "path": os.path.abspath(args.checkpoint),
            "sha256": checkpoint_hash,
            "model_config": model_config,
        },
        "work_schema": schema,
        "formal_protocol": formal_layer4_protocol(),
        "semantics": semantics,
        "training_config": {
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "epochs_ran": len(history) - 1,
            "best_epoch": best_epoch,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "group_batch_size": args.group_batch_size,
            "loss_weights": {
                "group_range": args.group_range_weight,
                "raw_drift": args.raw_drift_weight,
                "endpoint": args.endpoint_weight,
                "all_non_tie_pairwise": args.pairwise_weight,
            },
            "pairwise_temperature": args.pairwise_temperature,
            "scale_floor": args.scale_floor,
            "raw_drift_scale": raw_drift_scale,
            "station_residual_scales": station_scales,
            "residual_limit": args.residual_limit,
            "grad_clip": args.grad_clip,
            "checkpoint_selection": (
                "min development [group_range_nrmse "
                "- 0.25*(pairwise-0.5) - 0.10*(top1-random_top1)]"
            ),
        },
        "data_manifest": {
            "train": train_manifest,
            "validation": val_manifest,
            "train_audit": train_audit,
            "validation_audit": val_audit,
            "seed_disjoint": True,
        },
        "validation": best_metrics,
        "history": history,
    }
    report = {
        "schema_version": TRAINING_REPORT_SCHEMA_VERSION,
        "head_schema_version": HEAD_SCHEMA_VERSION,
        "checkpoint_output": str(output),
        "online_ready": False,
        "semantics": semantics,
        "base_world_model": payload["base_world_model"],
        "work_schema": schema,
        "formal_protocol": payload["formal_protocol"],
        "training_config": payload["training_config"],
        "data_manifest": payload["data_manifest"],
        "validation": best_metrics,
        "history": history,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved checkpoint: {output}")
    print(f"saved report: {report_path}")
    print(f"best epoch: {best_epoch}")
    print(
        "development validation group-range NRMSE:",
        best_metrics["continuous_group_range"]["normalised_rmse"],
    )
    print("note: this training command does not certify held-out Layer 4 or Layer 5")


if __name__ == "__main__":
    main()
