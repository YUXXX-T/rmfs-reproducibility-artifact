"""Train analytic nominal work relief plus a learned interaction residual.

This is a development-only Layer-4 experiment.  It preserves the fixed
quadratic ``L_work`` functional and compares three estimators on the same
isolated endpoint labels:

``A`` parameter-free analytic free-flow relief,
``B`` the legacy endpoint-only WorkDriftHead (when supplied), and
``C`` analytic relief plus a learned station-wise residual.

The residual target is built from the post-action work ledger.  Future
orders, later scheduler decisions, continuation policies and TD tails are
never used.  Existing Layer-4/5 checkpoints and reports are not overwritten.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import json
import os
from pathlib import Path
import random
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from WorldModel.core.analytic_work_residual_head import (
    SCHEMA_VERSION as HEAD_SCHEMA_VERSION,
    AnalyticWorkResidualHead,
)
from WorldModel.core.analytic_work_relief import (
    compute_snapshot_nominal_work_relief,
    extract_work_horizon_endpoint,
)
from WorldModel.core.work_drift_head import (
    WorkDriftHead,
    all_non_tie_pairwise_logistic_loss,
    group_range_normalise,
)
from WorldModel.training.train_td_risk_v import load_frozen_world_model
from WorldModel.training.train_work_drift_head import (
    _expand_paths,
    _group_key,
    _groups,
    _load_label,
    _load_samples,
    _manifest_from_compact_rows,
    _mean_max,
    _sample_seed,
    _sha256_file,
    _station_value,
    _truncate_compact_rows_by_group,
    _work_schema,
    evaluate_rows,
)


TRAINING_REPORT_SCHEMA_VERSION = (
    "analytic_work_relief_residual_training_report_v1"
)


def _snapshot_station_work(
    snapshot: Mapping,
    station_ids: Sequence[int],
    *,
    name: str,
) -> torch.Tensor:
    mapping = snapshot.get("station_work")
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name}.station_work is missing")
    values = torch.tensor([
        _station_value(mapping, int(station_id)) for station_id in station_ids
    ], dtype=torch.float32)
    if bool((values < 0.0).any()):
        raise ValueError(f"{name}.station_work must be non-negative")
    return values


def _load_samples_at_horizon(
    paths: Sequence[str],
    *,
    required_horizon: int,
) -> tuple[list[dict], dict]:
    """Load a true endpoint at ``H`` from a full-H label or station trajectory.

    A newly collected H=20 isolated trajectory can therefore support the
    pre-registered H=5/10/15/20 comparison without rerunning four simulators.
    Legacy H=10 files remain valid at H=10 only.  We intentionally retain only
    samples valid for their complete collected horizon so every horizon uses
    the same candidate groups and right-censoring cannot bias the comparison.
    """

    if int(required_horizon) <= 0:
        raise ValueError("required_horizon must be positive")
    expanded = _expand_paths(paths)
    rows: list[dict] = []
    combined_counts = {}
    collection_horizons = set()
    derived = 0
    for path in expanded:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        raw = payload.get("samples") if isinstance(payload, Mapping) else payload
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{path}: expected a non-empty sample list")
        first_mask = torch.as_tensor(raw[0].get("future_mask")).bool().flatten()
        collection_horizon = int(first_mask.numel())
        if collection_horizon <= 0:
            raise ValueError(f"{path}: empty future_mask")
        loaded, audit = _load_samples(
            [path], required_horizon=collection_horizon
        )
        collection_horizons.add(collection_horizon)
        combined_counts.update(audit["source_counts"])
        if required_horizon > collection_horizon:
            raise ValueError(
                f"{path}: requested H={required_horizon} exceeds collected "
                f"H={collection_horizon}; missing labels are never fabricated"
            )
        for sample in loaded:
            mask = torch.as_tensor(sample["future_mask"]).bool().flatten()
            if not bool(mask[:required_horizon].all()):
                raise ValueError(
                    f"{path}: H={required_horizon} endpoint is right-censored"
                )
            if required_horizon == collection_horizon:
                rows.append(sample)
                continue
            endpoint = extract_work_horizon_endpoint(
                sample, required_horizon
            )
            adapted = dict(sample)
            adapted["future_mask"] = mask[:required_horizon].clone()
            adapted["lyapunov_l0_end"] = endpoint
            adapted["analytic_work_endpoint_source"] = (
                "lyapunov_l0_station_work_trajectory"
            )
            adapted["analytic_work_requested_horizon"] = int(required_horizon)
            rows.append(adapted)
            derived += 1
    # groups = _groups(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)
    groups = list(grouped.values())
    if any(len(group) < 2 for group in groups):
        raise ValueError("horizon adaptation produced an incomplete candidate group")
    return rows, {
        "source_counts": combined_counts,
        "required_horizon": int(required_horizon),
        "collection_horizons": sorted(collection_horizons),
        "strict_full_collection_horizon_validity": True,
        "candidate_groups": len(groups),
        "station_trajectory_derived_samples": int(derived),
    }


def _pipeline_relief_from_post_action_snapshot(
    snapshot: Mapping,
    station_ids: Sequence[int],
    *,
    horizon: Optional[int],
) -> torch.Tensor:
    requested = 1 if horizon is None else int(horizon)
    forecast = compute_snapshot_nominal_work_relief(
        snapshot,
        horizon=requested,
        station_ids=station_ids,
    )
    values = (
        forecast.available_station_relief
        if horizon is None
        else forecast.nominal_station_relief
    )
    return torch.tensor(values, dtype=torch.float32)


def nominal_relief_from_post_action_snapshot(
    snapshot: Mapping,
    station_ids: Sequence[int],
    *,
    horizon: int,
) -> torch.Tensor:
    """Compute free-flow H-step relief for every active pipeline chain.

    A chain contributes ``mass * min(H, remaining_work) / initial_work`` to
    its station.  Pending chains are intentionally excluded: no robot has
    entered their physical PICK/DELIVER/RETURN pipeline, so free-flow motion
    cannot dissipate their unresolved mass.
    """

    return _pipeline_relief_from_post_action_snapshot(
        snapshot, station_ids, horizon=horizon
    )


def available_pipeline_relief_from_post_action_snapshot(
    snapshot: Mapping,
    station_ids: Sequence[int],
) -> torch.Tensor:
    """Return the maximum work mass that active pipelines can dissipate."""

    return _pipeline_relief_from_post_action_snapshot(
        snapshot, station_ids, horizon=None
    )


def _validate_post_action_snapshot(
    sample: Mapping,
    *,
    station_ids: Sequence[int],
    capacity: Sequence[float],
) -> Mapping:
    snapshot = sample.get("lyapunov_l0_post_action")
    if not isinstance(snapshot, Mapping):
        raise ValueError("analytic residual training requires lyapunov_l0_post_action")
    ids = sorted(int(value) for value in snapshot.get("station_work", {}))
    if ids != list(station_ids):
        raise ValueError("post-action station ids differ from the work schema")
    observed_capacity = float(snapshot.get("work_capacity", -1.0))
    expected = tuple(float(value) for value in capacity)
    if not expected or any(abs(value - expected[0]) > 1e-8 for value in expected):
        raise ValueError("current work schema requires a shared station capacity")
    if abs(observed_capacity - expected[0]) > 1e-5:
        raise ValueError("post-action work capacity differs from the work schema")
    return snapshot


def _precompute(
    model,
    samples: Sequence[Mapping],
    *,
    station_ids: Sequence[int],
    work_capacity: Sequence[float],
    work_weight: float,
    device: str,
    horizon: int,
) -> list[dict]:
    rows: list[dict] = []
    capacity_cpu = torch.as_tensor(work_capacity, dtype=torch.float32)
    capacity_device = capacity_cpu.to(device)
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
            post_snapshot = _validate_post_action_snapshot(
                sample,
                station_ids=station_ids,
                capacity=work_capacity,
            )
            end_snapshot = sample["lyapunov_l0_end"]
            start = _snapshot_station_work(
                start_snapshot, station_ids, name="lyapunov_l0_start"
            )
            post = _snapshot_station_work(
                post_snapshot, station_ids, name="lyapunov_l0_post_action"
            )
            endpoint = _snapshot_station_work(
                end_snapshot, station_ids, name="lyapunov_l0_end"
            )
            nominal = nominal_relief_from_post_action_snapshot(
                post_snapshot, station_ids, horizon=horizon
            )
            available = available_pipeline_relief_from_post_action_snapshot(
                post_snapshot, station_ids
            )
            true_relief = post - endpoint
            target_residual = true_relief - nominal
            nominal_endpoint = torch.clamp_min(post - nominal, 0.0)

            start_device = start.to(device)
            post_device = post.to(device)
            nominal_device = nominal.to(device)
            available_device = available.to(device)
            global_context = torch.cat((
                _mean_max(z_start),
                _mean_max(z_end),
                e0,
            ), dim=-1)
            physical = torch.stack((
                start_device / capacity_device,
                post_device / capacity_device,
                nominal_device / capacity_device,
                available_device / capacity_device,
            ), dim=-1)
            station_context = torch.cat((
                z_start.index_select(0, graph_station_ids),
                z_end.index_select(0, graph_station_ids),
                physical,
            ), dim=-1)
            legacy_station_context = torch.cat((
                z_start.index_select(0, graph_station_ids),
                z_end.index_select(0, graph_station_ids),
                (start_device / capacity_device).unsqueeze(-1),
            ), dim=-1)
            nominal_raw = 0.5 * float(work_weight) * float(
                torch.square(nominal_endpoint / capacity_cpu).sum()
                - torch.square(start / capacity_cpu).sum()
            )
            rows.append({
                "global_context": global_context.detach().cpu(),
                "station_context": station_context.detach().cpu(),
                "legacy_station_context": legacy_station_context.detach().cpu(),
                "current_station_work": start,
                "post_action_station_work": post,
                "nominal_station_relief": nominal,
                "available_station_relief": available,
                "nominal_endpoint_station_work": nominal_endpoint,
                "true_station_relief": true_relief,
                "target_station_relief_residual": target_residual,
                "endpoint_station_work": endpoint,
                "nominal_raw_work_drift": nominal_raw,
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


def _precompute_paths(
    model,
    paths: Sequence[str],
    *,
    expected_schema: Mapping,
    device: str,
    horizon: int,
    max_samples: Optional[int] = None,
) -> tuple[list[dict], dict, dict]:
    rows: list[dict] = []
    source_counts = {}
    collection_horizons = set()
    expanded = _expand_paths(paths)
    for file_index, path in enumerate(expanded, 1):
        samples, audit = _load_samples_at_horizon(
            [path], required_horizon=horizon
        )
        observed_schema = _work_schema(samples)
        if observed_schema != dict(expected_schema):
            raise ValueError(f"{path}: work schema differs from training schema")
        compact = _precompute(
            model,
            samples,
            station_ids=expected_schema["station_ids"],
            work_capacity=expected_schema["work_capacity"],
            work_weight=expected_schema["work_weight"],
            device=device,
            horizon=horizon,
        )
        rows.extend(compact)
        source_counts.update(audit["source_counts"])
        collection_horizons.update(audit["collection_horizons"])
        print(
            f"  precomputed arm {file_index}/{len(expanded)}: "
            f"{os.path.basename(os.path.dirname(path))} candidates={len(compact)}"
        )
        del samples, compact
    rows, truncated = _truncate_compact_rows_by_group(rows, max_samples)
    if any(len(group) < 2 for group in _groups(rows)):
        raise ValueError("precompute produced an incomplete candidate group")
    return rows, _manifest_from_compact_rows(rows), {
        "source_counts": source_counts,
        "source_files_processed_sequentially": len(expanded),
        "required_horizon": horizon,
        "strict_full_horizon": True,
        "collection_horizons": sorted(collection_horizons),
        "candidate_groups": len(_groups(rows)),
        "complete_group_truncation_applied": truncated,
        "raw_samples_retained_after_precompute": False,
        "nominal_relief_source": "lyapunov_l0_post_action.chains",
    }


def _robust_scales(
    rows: Sequence[Mapping], floor: float
) -> tuple[list[float], float]:
    if floor <= 0.0:
        raise ValueError("scale floor must be positive")
    residual = torch.stack([
        torch.as_tensor(row["target_station_relief_residual"]).abs()
        for row in rows
    ])
    station = torch.quantile(residual, 0.75, dim=0).clamp_min(floor)
    drift = torch.tensor([
        abs(float(row["target_raw_work_drift"])) for row in rows
    ], dtype=torch.float32)
    return station.tolist(), max(float(torch.quantile(drift, 0.75)), floor)


def _residual_diagnostics(rows: Sequence[Mapping]) -> dict:
    target = torch.stack([
        torch.as_tensor(row["target_station_relief_residual"], dtype=torch.float64)
        for row in rows
    ])
    nominal = torch.stack([
        torch.as_tensor(row["nominal_station_relief"], dtype=torch.float64)
        for row in rows
    ])
    true = torch.stack([
        torch.as_tensor(row["true_station_relief"], dtype=torch.float64)
        for row in rows
    ])
    available = torch.stack([
        torch.as_tensor(row["available_station_relief"], dtype=torch.float64)
        for row in rows
    ])

    def summary(values: torch.Tensor) -> dict:
        flat = values.flatten()
        return {
            "n": int(flat.numel()),
            "mean": float(flat.mean()),
            "std": float(flat.std(unbiased=False)),
            "p05": float(torch.quantile(flat, 0.05)),
            "median": float(torch.quantile(flat, 0.50)),
            "p95": float(torch.quantile(flat, 0.95)),
            "min": float(flat.min()),
            "max": float(flat.max()),
        }

    return {
        "nominal_station_relief": summary(nominal),
        "true_station_relief": summary(true),
        "available_pipeline_relief": summary(available),
        "target_residual_true_minus_nominal": summary(target),
        "negative_residual_rate": float((target < 0.0).double().mean()),
        "positive_residual_rate": float((target > 0.0).double().mean()),
        "negative_true_relief_rate": float((true < 0.0).double().mean()),
        "true_relief_above_available_rate": float(
            (true > available + 1e-8).double().mean()
        ),
        "analytic_bound_violation_rate": float(
            ((true < -1e-8) | (true > available + 1e-8)).double().mean()
        ),
    }


def _group_training_loss(
    head: AnalyticWorkResidualHead,
    members: Sequence[Mapping],
    *,
    device: str,
    station_scales: torch.Tensor,
    raw_drift_scale: float,
    group_range_weight: float,
    raw_drift_weight: float,
    endpoint_weight: float,
    residual_weight: float,
    pairwise_weight: float,
    pairwise_temperature: float,
) -> tuple[torch.Tensor, dict]:
    global_context = torch.stack([
        row["global_context"] for row in members
    ]).to(device)
    station_context = torch.stack([
        row["station_context"] for row in members
    ]).to(device)
    start = torch.stack([
        row["current_station_work"] for row in members
    ]).to(device)
    endpoint = torch.stack([
        row["endpoint_station_work"] for row in members
    ]).to(device)
    target_residual = torch.stack([
        row["target_station_relief_residual"] for row in members
    ]).to(device)
    target_values = [float(row["target_raw_work_drift"]) for row in members]
    target_raw = torch.tensor(target_values, dtype=torch.float32, device=device)
    prediction = head.predict_from_features(
        global_context, station_context, start
    )
    predicted_range = group_range_normalise(prediction.raw_work_drift)
    target_range = group_range_normalise(torch.tensor(
        target_values, dtype=torch.float64
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
    residual_loss = F.smooth_l1_loss(
        prediction.residual_station_relief / scales,
        target_residual / scales,
    )
    pairwise_loss, pair_count = all_non_tie_pairwise_logistic_loss(
        predicted_range, target_range, temperature=pairwise_temperature
    )
    total = (
        group_range_weight * range_loss
        + raw_drift_weight * raw_loss
        + endpoint_weight * endpoint_loss
        + residual_weight * residual_loss
        + pairwise_weight * pairwise_loss
    )
    return total, {
        "group_range": float(range_loss.detach()),
        "raw_drift": float(raw_loss.detach()),
        "endpoint": float(endpoint_loss.detach()),
        "relief_residual": float(residual_loss.detach()),
        "pairwise": float(pairwise_loss.detach()),
        "pairs": pair_count,
    }


def _legacy_rows(rows: Sequence[Mapping]) -> list[dict]:
    result = []
    for row in rows:
        copy_row = dict(row)
        copy_row["station_context"] = row["legacy_station_context"]
        result.append(copy_row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-data", nargs="+", required=True)
    parser.add_argument("--val-data", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--legacy-head", default=None)
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
    parser.add_argument("--residual-weight", type=float, default=0.25)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--pairwise-temperature", type=float, default=0.25)
    parser.add_argument("--residual-limit", type=float, default=4.0)
    parser.add_argument("--scale-floor", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if min(
        args.horizon,
        args.hidden_dim,
        args.epochs,
        args.patience,
        args.group_batch_size,
    ) <= 0:
        raise SystemExit("horizon/dim/epochs/patience/batch size must be positive")
    non_negative = (
        args.group_range_weight,
        args.raw_drift_weight,
        args.endpoint_weight,
        args.residual_weight,
        args.pairwise_weight,
        args.weight_decay,
    )
    if any(value < 0.0 for value in non_negative):
        raise SystemExit("loss weights and weight decay must be non-negative")
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
    first_samples, _ = _load_samples_at_horizon(
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
    transition_steps = int(model.transition.step_embed.num_embeddings)
    if args.horizon > transition_steps:
        raise SystemExit(
            f"H={args.horizon} exceeds the frozen World Model's "
            f"{transition_steps} distinct transition-step embeddings; "
            "H>20 residual learning requires an expanded/retrained World Model"
        )
    latent_dim = int(model_config.get("hidden_dim", 64))
    print(
        "precomputing analytic nominal + frozen-WM compact features: "
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
    head = AnalyticWorkResidualHead(
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        num_stations=len(station_ids),
        work_capacity=schema["work_capacity"],
        work_weight=schema["work_weight"],
        residual_scale=station_scales,
        residual_limit=args.residual_limit,
    ).to(args.device)
    analytic_baseline = evaluate_rows(head, val_rows, device=args.device)
    legacy = None
    legacy_metrics = None
    legacy_provenance = None
    legacy = None
    if args.legacy_head:
        legacy, legacy_payload = WorkDriftHead.from_checkpoint(
            args.legacy_head, map_location=args.device
        )
        expected_hash = (legacy_payload.get("base_world_model") or {}).get("sha256")
        actual_hash = _sha256_file(args.checkpoint)
        if expected_hash is not None and expected_hash != actual_hash:
            raise SystemExit("legacy head and supplied World Model checkpoint differ")
        legacy_metrics = evaluate_rows(
            legacy, _legacy_rows(val_rows), device=args.device
        )
        legacy_provenance = {
            "path": os.path.abspath(args.legacy_head),
            "sha256": _sha256_file(args.legacy_head),
            "schema_version": legacy_payload.get("schema_version"),
        }

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_groups = _groups(train_rows)
    best_score = float(analytic_baseline["selection_score"])
    best_epoch = 0
    best_state = copy.deepcopy(head.state_dict())
    patience = 0
    history = [{
        "epoch": 0,
        "train_loss": None,
        "validation": analytic_baseline,
        "role": "parameter_free_analytic_baseline",
    }]
    scale_tensor = torch.as_tensor(station_scales, dtype=torch.float32)

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
                    station_scales=scale_tensor,
                    raw_drift_scale=raw_drift_scale,
                    group_range_weight=args.group_range_weight,
                    raw_drift_weight=args.raw_drift_weight,
                    endpoint_weight=args.endpoint_weight,
                    residual_weight=args.residual_weight,
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
                for name, value in row.items():
                    component_rows[name].append(value)

        metrics = evaluate_rows(head, val_rows, device=args.device)
        score = float(metrics["selection_score"])
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(loss_rows)),
            "train_components": {
                name: float(np.mean(values)) if values else None
                for name, values in component_rows.items()
            },
            "validation": metrics,
        })
        continuous = metrics["continuous_group_range"]
        print(
            f"epoch {epoch:03d} train={history[-1]['train_loss']:.6f} "
            f"val_nrmse={continuous.get('normalised_rmse')} "
            f"pair={continuous.get('all_non_tie_pair_concordance')} "
            f"top1={continuous.get('top1_min_drift_accuracy')}"
        )
        if score < best_score - 1e-7:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    head.load_state_dict(best_state)
    residual_metrics = evaluate_rows(
        head,
        val_rows,
        device=args.device,
        bootstrap_repeats=1000,
        random_seed=args.seed + 17,
    )
    analytic_head = AnalyticWorkResidualHead(
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        num_stations=len(station_ids),
        work_capacity=schema["work_capacity"],
        work_weight=schema["work_weight"],
        residual_scale=station_scales,
        residual_limit=args.residual_limit,
    ).to(args.device)
    analytic_baseline = evaluate_rows(
        analytic_head,
        val_rows,
        device=args.device,
        bootstrap_repeats=1000,
        random_seed=args.seed + 17,
    )
    if legacy is not None:
        legacy_metrics = evaluate_rows(
            legacy,
            _legacy_rows(val_rows),
            device=args.device,
            bootstrap_repeats=1000,
            random_seed=args.seed + 17,
        )
    checkpoint_hash = _sha256_file(args.checkpoint)
    semantics = {
        "development_only": True,
        "layer": 4,
        "question": (
            "Can a frozen World Model estimate the residual around analytic "
            "free-flow work relief for an isolated H-step endpoint?"
        ),
        "fixed_potential": "quadratic analytic L_work",
        "nominal_model": (
            "post-action pipeline-chain mass * min(H, remaining_work) / "
            "initial_work"
        ),
        "learned_quantity": "signed station-wise relief residual only",
        "current_demand_only": True,
        "unknown_future_orders": False,
        "future_demand_predictor": False,
        "continuation_policy": False,
        "td_tail": False,
        "greedy_or_external_policy": False,
        "direct_action_embedding_to_head": False,
        "rollout_continuation_mode": "isolated",
        "horizon": args.horizon,
        "world_model_distinct_transition_steps": transition_steps,
        "long_horizon_claim": False,
        "online_ready": False,
    }
    training_config = {
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
            "relief_residual": args.residual_weight,
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
    }
    data_manifest = {
        "train": train_manifest,
        "validation": val_manifest,
        "train_audit": train_audit,
        "validation_audit": val_audit,
        "seed_disjoint": True,
    }
    abc = {
        "A_parameter_free_analytic": analytic_baseline,
        "B_legacy_endpoint_head": legacy_metrics or {
            "status": "NOT_EVALUATED_NO_LEGACY_HEAD_SUPPLIED"
        },
        "C_analytic_plus_residual": residual_metrics,
        "comparison_role": "DEVELOPMENT_ONLY_NOT_FORMAL_CERTIFICATION",
    }
    payload = {
        "schema_version": HEAD_SCHEMA_VERSION,
        "online_ready": False,
        "online_blocker": (
            "development A/B/C and held-out horizon/closed-loop tests pending"
        ),
        "head_config": head.checkpoint_config(),
        "head_state_dict": copy.deepcopy(head.state_dict()),
        "base_world_model": {
            "path": os.path.abspath(args.checkpoint),
            "sha256": checkpoint_hash,
            "model_config": model_config,
        },
        "legacy_head": legacy_provenance,
        "work_schema": schema,
        "semantics": semantics,
        "training_config": training_config,
        "data_manifest": data_manifest,
        "residual_target_diagnostics": {
            "train": _residual_diagnostics(train_rows),
            "validation": _residual_diagnostics(val_rows),
        },
        "abc_validation": abc,
        "validation": residual_metrics,
        "history": history,
    }
    report = {
        "schema_version": TRAINING_REPORT_SCHEMA_VERSION,
        "head_schema_version": HEAD_SCHEMA_VERSION,
        "checkpoint_output": str(output),
        "online_ready": False,
        "semantics": semantics,
        "base_world_model": payload["base_world_model"],
        "legacy_head": legacy_provenance,
        "work_schema": schema,
        "training_config": training_config,
        "data_manifest": data_manifest,
        "residual_target_diagnostics": payload["residual_target_diagnostics"],
        "abc_validation": abc,
        "validation": residual_metrics,
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
    print("A/B/C role: DEVELOPMENT_ONLY_NOT_FORMAL_CERTIFICATION")
    print(
        "note: H beyond the true labels in the supplied data is refused; "
        "this command does not certify Layer 4 or Layer 5"
    )


if __name__ == "__main__":
    main()
