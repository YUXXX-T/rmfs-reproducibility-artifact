"""Multi-horizon isolated system-dynamics evaluation for a frozen World Model.

This evaluator is intentionally separate from validate_dynamics.py so the
existing H=10 reports remain reproducible.  It evaluates one free-running
latent rollout at max(horizons), then compares exact endpoints from the same
prediction prefix and the same H-max simulator-label prefix.

Important contracts
-------------------
1. Every requested horizon uses the same encoded start state, action, demand,
   and graph.
2. A single H-max target tensor supplies every endpoint.
3. average_excess_delay is compared in physical tick units.  The label scale
   used by the data generator and the scale used to train the checkpoint are
   explicit CLI arguments.
4. Learned absolute step-embedding capacity/clamping is audited.  The script
   never invents untrained embedding rows for an old checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from WorldModel.data.dataset import WorldModelDataset
from WorldModel.evaluation.evaluate import _load_model, roc_auc, spearman


SCHEMA_VERSION = "wm_system_dynamics_horizon_eval_v1"

SYSTEM_CHANNELS = (
    "total_wait_time",
    "average_excess_delay",
    "station_queue_delta",
    "station_load_imbalance",
    "bottleneck_CVaR",
    "completed_orders_delta",
    "deadlock_or_severe_congestion_risk",
)

CHANNEL_NOTES = {
    "total_wait_time": (
        "Historical channel name.  The current label extractor stores the "
        "fraction of non-idle robots classified as stalled/blocked."
    ),
    "average_excess_delay": (
        "Reported by this evaluator in physical ticks after undoing the "
        "model/data normalization scales."
    ),
    "station_queue_delta": (
        "Average positive per-station physical queue-occupancy-ratio increase."
    ),
    "station_load_imbalance": "Current label-space value (raw imbalance / 5).",
    "bottleneck_CVaR": "Current label-space bottleneck tail-risk value.",
    "completed_orders_delta": (
        "Cumulative completed-order increment from the rollout window start."
    ),
    "deadlock_or_severe_congestion_risk": (
        "Current unified severe-congestion/deadlock risk label."
    ),
}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mu = _mean(values)
    return math.sqrt(math.fsum((x - mu) ** 2 for x in values) / len(values))


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 3 or len(left) != len(right):
        return 0.0
    ml = _mean(left)
    mr = _mean(right)
    num = math.fsum((a - ml) * (b - mr) for a, b in zip(left, right))
    dl = math.fsum((a - ml) ** 2 for a in left)
    dr = math.fsum((b - mr) ** 2 for b in right)
    denom = math.sqrt(dl * dr)
    return num / denom if denom > 1e-12 else 0.0


def _metric_bundle(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    include_auc: bool = False,
) -> dict:
    finite_pairs = [
        (float(pred), float(target))
        for pred, target in zip(predictions, targets)
        if math.isfinite(float(pred)) and math.isfinite(float(target))
    ]
    if not finite_pairs:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "pearson": None,
            "spearman": None,
            "pred_mean": None,
            "pred_std": None,
            "target_mean": None,
            "target_std": None,
        }

    pred = [pair[0] for pair in finite_pairs]
    target = [pair[1] for pair in finite_pairs]
    errors = [a - b for a, b in finite_pairs]
    abs_errors = [abs(value) for value in errors]
    sq_errors = [value * value for value in errors]
    target_std = _std(target)

    result = {
        "n": len(pred),
        "mae": _mean(abs_errors),
        "rmse": math.sqrt(_mean(sq_errors)),
        "bias": _mean(errors),
        "pearson": _pearson(pred, target),
        "spearman": spearman(pred, target),
        "pred_mean": _mean(pred),
        "pred_std": _std(pred),
        "target_mean": _mean(target),
        "target_std": target_std,
        "mae_over_target_std": (
            _mean(abs_errors) / target_std if target_std > 1e-12 else None
        ),
    }
    if include_auc:
        result["auc_at_target_ge_0_5"] = roc_auc(target, pred)
        result["positive_targets"] = sum(value >= 0.5 for value in target)
        result["negative_targets"] = sum(value < 0.5 for value in target)
    return result


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _normalise_horizons(values: Iterable[int]) -> List[int]:
    horizons = sorted(set(int(value) for value in values))
    if not horizons or horizons[0] <= 0:
        raise ValueError("horizons must be positive integers")
    return horizons


def _select_samples(
    full_dataset: WorldModelDataset,
    *,
    split: str,
    splits_path: Optional[str],
    train_ratio: float,
    val_ratio: float,
    split_seed: int,
) -> List[dict]:
    if split == "all":
        return list(full_dataset.samples)

    if splits_path:
        with open(splits_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        allowed = set(str(value) for value in payload[split])
        selected = [
            sample
            for sample in full_dataset.samples
            if str(sample.get("candidate_group_id", "")) in allowed
        ]
        return selected

    train_ds, val_ds, test_ds = full_dataset.split_by_group(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=split_seed,
    )
    return {
        "train": list(train_ds.samples),
        "val": list(val_ds.samples),
        "test": list(test_ds.samples),
    }[split]


def _station_node_ids(sample: dict):
    value = sample.get("station_node_ids")
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _rollout_system(
    model,
    sample: dict,
    device: torch.device,
    horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    node_history = sample["node_history"].to(device)
    edge_index = sample["edge_index"].to(device)
    edge_features = sample["edge_features"].to(device)
    demand_context = sample["demand_context"].to(device)
    action_node = sample["action_node"].to(device)
    action_global = sample["action_global"].to(device)

    z_start, demand_embedding, edge_embedding = model.encode_state(
        node_history,
        edge_index,
        edge_features,
        demand_context,
    )
    outputs = model.rollout(
        z_start,
        demand_embedding,
        edge_embedding,
        action_node,
        action_global,
        edge_index,
        _station_node_ids(sample),
        K=int(horizon),
    )
    system_predictions = outputs[1]
    returned_z_start = outputs[4]
    return (
        system_predictions.detach().cpu(),
        returned_z_start.detach().cpu(),
    )


def _step_embedding_audit(model, horizons: Sequence[int]) -> dict:
    training_horizon = int(getattr(model, "rollout_horizon", 0))
    transition = getattr(model, "transition", None)
    step_embedding = getattr(transition, "step_embed", None)
    capacity = (
        int(step_embedding.num_embeddings)
        if step_embedding is not None
        else None
    )

    by_horizon = {}
    for horizon in horizons:
        if capacity is None:
            clamped = None
            distinct = None
        else:
            clamped = max(0, int(horizon) - capacity)
            distinct = min(int(horizon), capacity)
        by_horizon[str(horizon)] = {
            "transitions": int(horizon),
            "inside_checkpoint_training_horizon": min(
                int(horizon), training_horizon
            ),
            "outside_checkpoint_training_horizon": max(
                0, int(horizon) - training_horizon
            ),
            "distinct_learned_step_indices_available": distinct,
            "tail_transitions_using_last_embedding_index": clamped,
        }

    max_horizon = max(horizons)
    return {
        "embedding_type": (
            type(step_embedding).__name__ if step_embedding is not None else None
        ),
        "checkpoint_training_horizon": training_horizon,
        "learned_step_embedding_capacity": capacity,
        "first_horizon_outside_training_contract": training_horizon + 1,
        "first_horizon_using_clamped_last_embedding": (
            capacity + 1 if capacity is not None else None
        ),
        "requested_max_horizon": max_horizon,
        "step_clamp_present": (
            capacity is not None and max_horizon > capacity
        ),
        "by_horizon": by_horizon,
        "interpretation": (
            "The evaluator reports the frozen checkpoint's native transition "
            "behavior. It does not invent untrained absolute-step embedding "
            "rows. A checkpoint with a learned finite table requires a "
            "retrained time-homogeneous or functional step encoding to remove "
            "this limitation."
        ),
    }


def _validate_data_contract(samples: Sequence[dict], max_horizon: int) -> dict:
    failures = []
    valid_counts = {str(horizon): 0 for horizon in range(1, max_horizon + 1)}
    group_ids = set()

    for index, sample in enumerate(samples):
        target = sample.get("future_system_labels")
        if target is None:
            failures.append(f"sample[{index}] missing future_system_labels")
            continue
        if target.ndim != 2 or target.shape[1] != len(SYSTEM_CHANNELS):
            failures.append(
                f"sample[{index}] future_system_labels shape={tuple(target.shape)}"
            )
            continue
        if target.shape[0] < max_horizon:
            failures.append(
                f"sample[{index}] has H={target.shape[0]} < {max_horizon}"
            )
            continue

        mask = sample.get("future_mask")
        if mask is not None and mask.shape[0] < max_horizon:
            failures.append(
                f"sample[{index}] future_mask H={mask.shape[0]} < {max_horizon}"
            )
            continue

        group_ids.add(str(sample.get("candidate_group_id", "")))
        for horizon in range(1, max_horizon + 1):
            if mask is None or float(mask[horizon - 1].item()) >= 0.5:
                valid_counts[str(horizon)] += 1

    return {
        "passed": not failures,
        "failures": failures[:50],
        "failure_count": len(failures),
        "samples": len(samples),
        "candidate_groups": len(group_ids),
        "max_required_horizon": max_horizon,
        "valid_sample_counts_by_step": valid_counts,
        "single_truth_prefix_reused_for_all_report_horizons": True,
    }


def _generation_contract_audit(
    data_path: str,
    explicit_path: Optional[str],
    max_horizon: int,
) -> dict:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.append(Path(data_path).with_name("gen_config.json"))

    config_path = next((path for path in candidates if path.is_file()), None)
    if config_path is None:
        return {
            "available": False,
            "passed": None,
            "path": None,
            "checks": {},
            "warning": (
                "No gen_config.json was found; isolated continuation semantics "
                "cannot be machine-audited from the evaluation file alone."
            ),
        }

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    configured_horizon = int(payload.get("horizon", 0))
    checks = {
        "horizon_covers_requested_endpoints": configured_horizon >= max_horizon,
        "isolated_rollout": (
            payload.get("rollout_continuation_mode") == "isolated"
        ),
        "no_future_unknown_orders": not bool(
            payload.get("future_unknown_orders_in_rollout", False)
        ),
        "no_continuation_scheduler": not bool(
            payload.get("continuation_scheduler_in_rollout", False)
        ),
    }
    return {
        "available": True,
        "passed": all(checks.values()),
        "path": str(config_path),
        "sha256": _sha256_file(str(config_path)),
        "configured_horizon": configured_horizon,
        "checks": checks,
        "payload": payload,
    }


def _prefix_consistency_audit(
    model,
    samples: Sequence[dict],
    device: torch.device,
    horizons: Sequence[int],
    full_predictions: Sequence[torch.Tensor],
    full_z_starts: Sequence[torch.Tensor],
    *,
    sample_count: int,
    tolerance: float,
) -> dict:
    checked = min(
        max(0, int(sample_count)),
        len(samples),
        len(full_predictions),
        len(full_z_starts),
    )
    records = []
    max_prediction_error = 0.0
    max_z_start_error = 0.0

    with torch.inference_mode():
        for sample_index in range(checked):
            sample = samples[sample_index]
            full_prediction = full_predictions[sample_index]
            full_z_start = full_z_starts[sample_index]
            for horizon in horizons:
                if horizon == max(horizons):
                    continue
                short_prediction, short_z_start = _rollout_system(
                    model, sample, device, horizon
                )
                prediction_error = float(
                    (short_prediction - full_prediction[:horizon])
                    .abs()
                    .max()
                    .item()
                )
                z_start_error = float(
                    (short_z_start - full_z_start).abs().max().item()
                )
                max_prediction_error = max(
                    max_prediction_error, prediction_error
                )
                max_z_start_error = max(max_z_start_error, z_start_error)
                records.append({
                    "sample_index": sample_index,
                    "horizon": int(horizon),
                    "prediction_prefix_max_abs_error": prediction_error,
                    "z_start_max_abs_error": z_start_error,
                    "passed": (
                        prediction_error <= tolerance
                        and z_start_error <= tolerance
                    ),
                })

    return {
        "passed": all(record["passed"] for record in records),
        "sample_count": checked,
        "comparisons": len(records),
        "tolerance": tolerance,
        "max_prediction_prefix_abs_error": max_prediction_error,
        "max_z_start_abs_error": max_z_start_error,
        "records": records,
    }


def _prediction_metadata(sample: dict, sample_index: int) -> dict:
    keys = (
        "candidate_key",
        "candidate_group_id",
        "seed",
        "load_level",
        "load",
        "robot_id",
        "candidate_action_type",
        "action_type",
        "tick",
        "order_id",
        "pod_id",
        "station_id",
    )
    result = {"sample_index": sample_index}
    for key in keys:
        value = sample.get(key)
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else value.tolist()
        if value is not None:
            result[key] = value
    return result


def run_evaluation(
    *,
    data_path: str,
    checkpoint_path: str,
    horizons: Sequence[int],
    data_delay_scale: float,
    model_delay_scale: float,
    split: str,
    splits_path: Optional[str],
    train_ratio: float,
    val_ratio: float,
    split_seed: int,
    max_samples: int,
    device_name: str,
    gen_config_path: Optional[str],
    prefix_audit_samples: int,
    prefix_tolerance: float,
    fail_on_step_clamp: bool,
    save_predictions: Optional[str],
    skip_sha256: bool,
) -> dict:
    if data_delay_scale <= 0.0 or model_delay_scale <= 0.0:
        raise ValueError("delay scales must be positive")

    horizons = _normalise_horizons(horizons)
    max_horizon = max(horizons)
    device = _resolve_device(device_name)

    full_dataset = WorldModelDataset.from_file(data_path)
    samples = _select_samples(
        full_dataset,
        split=split,
        splits_path=splits_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise RuntimeError("selected dataset split is empty")

    data_audit = _validate_data_contract(samples, max_horizon)
    if not data_audit["passed"]:
        raise RuntimeError(
            "data contract failed: " + "; ".join(data_audit["failures"][:5])
        )
    missing_valid = [
        horizon
        for horizon in horizons
        if data_audit["valid_sample_counts_by_step"][str(horizon)] <= 0
    ]
    if missing_valid:
        raise RuntimeError(
            "no valid future labels at requested horizons: "
            + ", ".join(str(value) for value in missing_valid)
        )
    generation_audit = _generation_contract_audit(
        data_path,
        gen_config_path,
        max_horizon,
    )
    if generation_audit["available"] and not generation_audit["passed"]:
        failed = [
            name
            for name, passed in generation_audit["checks"].items()
            if not passed
        ]
        raise RuntimeError(
            "data generation contract failed: " + ", ".join(failed)
        )

    model, label_schema = _load_model(checkpoint_path)
    model = model.to(device)
    model.eval()

    step_audit = _step_embedding_audit(model, horizons)
    if fail_on_step_clamp and step_audit["step_clamp_present"]:
        raise RuntimeError(
            "requested horizon exceeds learned step-embedding capacity; "
            "checkpoint-native rollout would clamp to the last embedding"
        )

    values: Dict[int, Dict[int, Dict[str, List[float]]]] = {
        horizon: {
            channel: {"pred": [], "target": []}
            for channel in range(len(SYSTEM_CHANNELS))
        }
        for horizon in horizons
    }
    valid_counts = {horizon: 0 for horizon in horizons}
    full_predictions: List[torch.Tensor] = []
    full_z_starts: List[torch.Tensor] = []
    prediction_records = []

    started = time.time()
    with torch.inference_mode():
        for sample_index, sample in enumerate(samples):
            predictions, z_start = _rollout_system(
                model, sample, device, max_horizon
            )
            if predictions.shape != (max_horizon, len(SYSTEM_CHANNELS)):
                raise RuntimeError(
                    f"sample[{sample_index}] prediction shape "
                    f"{tuple(predictions.shape)} != "
                    f"({max_horizon}, {len(SYSTEM_CHANNELS)})"
                )

            targets = sample["future_system_labels"].detach().cpu()
            mask = sample.get("future_mask")
            if mask is not None:
                mask = mask.detach().cpu()

            if sample_index < max(0, prefix_audit_samples):
                full_predictions.append(predictions)
                full_z_starts.append(z_start)
            record = _prediction_metadata(sample, sample_index)
            record["endpoints"] = {}

            for horizon in horizons:
                endpoint = horizon - 1
                if mask is not None and float(mask[endpoint].item()) < 0.5:
                    continue
                valid_counts[horizon] += 1
                pred_vector = predictions[endpoint]
                target_vector = targets[endpoint]

                endpoint_pred = []
                endpoint_target = []
                for channel in range(len(SYSTEM_CHANNELS)):
                    pred_value = float(pred_vector[channel].item())
                    target_value = float(target_vector[channel].item())

                    # Decode average_excess_delay into physical tick units.
                    if channel == 1:
                        pred_value *= model_delay_scale
                        target_value *= data_delay_scale

                    values[horizon][channel]["pred"].append(pred_value)
                    values[horizon][channel]["target"].append(target_value)
                    endpoint_pred.append(pred_value)
                    endpoint_target.append(target_value)

                record["endpoints"][str(horizon)] = {
                    "pred": endpoint_pred,
                    "target": endpoint_target,
                }

            if save_predictions:
                prediction_records.append(record)

            if (sample_index + 1) % 50 == 0:
                print(
                    f"  inference {sample_index + 1}/{len(samples)} "
                    f"elapsed={time.time() - started:.1f}s"
                )

    prefix_audit = _prefix_consistency_audit(
        model,
        samples,
        device,
        horizons,
        full_predictions,
        full_z_starts,
        sample_count=prefix_audit_samples,
        tolerance=prefix_tolerance,
    )
    if not prefix_audit["passed"]:
        raise RuntimeError(
            "same-start/prediction-prefix consistency audit failed"
        )

    metrics_by_horizon = {}
    for horizon in horizons:
        channel_metrics = {}
        for channel, name in enumerate(SYSTEM_CHANNELS):
            bundle = _metric_bundle(
                values[horizon][channel]["pred"],
                values[horizon][channel]["target"],
                include_auc=(channel == 6),
            )
            bundle["unit"] = "ticks" if channel == 1 else "label_space"
            bundle["label_note"] = CHANNEL_NOTES[name]
            channel_metrics[name] = bundle
        metrics_by_horizon[str(horizon)] = {
            "valid_samples": valid_counts[horizon],
            "channels": channel_metrics,
            "positive_spearman_channels": sum(
                metric["spearman"] is not None and metric["spearman"] > 0.0
                for metric in channel_metrics.values()
            ),
        }

    warnings = []
    if max_horizon > int(getattr(model, "rollout_horizon", 0)):
        warnings.append(
            "Requested horizons extend beyond the checkpoint training horizon; "
            "these results are extrapolation diagnostics."
        )
    if step_audit["step_clamp_present"]:
        warnings.append(
            "The frozen checkpoint has a finite learned absolute-step table; "
            "native transitions beyond its capacity reuse the last embedding. "
            "This is reported, not silently treated as arbitrary-step support."
        )
    if data_delay_scale != model_delay_scale:
        warnings.append(
            "Data/model delay normalizations differ; average_excess_delay was "
            "de-normalized to physical ticks before metric computation."
        )
    if not generation_audit["available"]:
        warnings.append(generation_audit["warning"])

    result = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_name": (
            "frozen_h10_world_model_isolated_system_dynamics_extrapolation"
        ),
        "data": str(Path(data_path)),
        "checkpoint": str(Path(checkpoint_path)),
        "data_sha256": None if skip_sha256 else _sha256_file(data_path),
        "checkpoint_sha256": (
            None if skip_sha256 else _sha256_file(checkpoint_path)
        ),
        "checkpoint_label_schema": label_schema,
        "device": str(device),
        "samples": len(samples),
        "split": split,
        "horizons": list(horizons),
        "max_horizon": max_horizon,
        "data_delay_scale": float(data_delay_scale),
        "model_delay_scale": float(model_delay_scale),
        "average_excess_delay_metric_unit": "physical_ticks",
        "data_contract": data_audit,
        "data_generation_contract": generation_audit,
        "step_embedding_audit": step_audit,
        "same_start_and_prefix_audit": prefix_audit,
        "metrics_by_horizon": metrics_by_horizon,
        "warnings": warnings,
        "runtime_seconds": time.time() - started,
        "interpretation": {
            "primary_quantity": (
                "free-running latent transition plus system decoder accuracy"
            ),
            "continuation_contract": (
                "Targets must come from one isolated forced-first-action "
                "simulator rollout with no future order generation or "
                "continuation scheduler."
            ),
            "not_decoder_only": True,
            "not_closed_loop_policy_forecast": True,
            "same_start_required": True,
        },
    }

    if save_predictions:
        output = Path(save_predictions)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema_version": SCHEMA_VERSION,
            "channels": list(SYSTEM_CHANNELS),
            "horizons": list(horizons),
            "records": prediction_records,
        }, output)

    return result


def print_summary(result: dict) -> None:
    def shown(value) -> str:
        return "       n/a" if value is None else f"{float(value):>10.4f}"

    print("=" * 100)
    print("Frozen World Model - Isolated Multi-Horizon System Dynamics Evaluation")
    print("=" * 100)
    print(f"checkpoint : {result['checkpoint']}")
    print(f"data       : {result['data']}")
    print(f"samples    : {result['samples']}")
    print(f"horizons   : {result['horizons']}")
    print(
        "step table : capacity="
        f"{result['step_embedding_audit']['learned_step_embedding_capacity']} "
        "training_horizon="
        f"{result['step_embedding_audit']['checkpoint_training_horizon']} "
        "clamp_present="
        f"{result['step_embedding_audit']['step_clamp_present']}"
    )
    print(
        "prefix audit: "
        f"passed={result['same_start_and_prefix_audit']['passed']} "
        "max_pred_error="
        f"{result['same_start_and_prefix_audit']['max_prediction_prefix_abs_error']:.3e} "
        "max_z0_error="
        f"{result['same_start_and_prefix_audit']['max_z_start_abs_error']:.3e}"
    )

    for horizon in result["horizons"]:
        section = result["metrics_by_horizon"][str(horizon)]
        print()
        print(f"H={horizon} valid_samples={section['valid_samples']}")
        print(
            f"{'channel':<38} {'MAE':>10} {'RMSE':>10} "
            f"{'bias':>10} {'Spearman':>10} {'Pearson':>10}"
        )
        for channel in SYSTEM_CHANNELS:
            metric = section["channels"][channel]
            print(
                f"{channel:<38} "
                f"{shown(metric['mae'])} "
                f"{shown(metric['rmse'])} "
                f"{shown(metric['bias'])} "
                f"{shown(metric['spearman'])} "
                f"{shown(metric['pearson'])}"
            )

    if result["warnings"]:
        print()
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[10, 19, 20, 21, 50, 80, 100],
    )
    parser.add_argument(
        "--data-delay-scale",
        type=float,
        required=True,
        help=(
            "Scale used when future_system_labels[:,1] was generated. "
            "Use 10 for a corrected fixed-scale H100 collector; use 100 for "
            "the current generic builder's default H100 behavior."
        ),
    )
    parser.add_argument(
        "--model-delay-scale",
        type=float,
        default=10.0,
        help="Scale used by the checkpoint's H=10 training labels.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
    )
    parser.add_argument("--splits", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--gen-config",
        default=None,
        help=(
            "Optional generation config. If omitted, gen_config.json next "
            "to --data is used when available."
        ),
    )
    parser.add_argument("--prefix-audit-samples", type=int, default=3)
    parser.add_argument("--prefix-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--fail-on-step-clamp",
        action="store_true",
        help=(
            "Abort instead of evaluating if max(horizons) exceeds the "
            "checkpoint's learned step-embedding capacity."
        ),
    )
    parser.add_argument("--save-json", required=True)
    parser.add_argument("--save-predictions", default=None)
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()

    result = run_evaluation(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        horizons=args.horizons,
        data_delay_scale=args.data_delay_scale,
        model_delay_scale=args.model_delay_scale,
        split=args.split,
        splits_path=args.splits,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        max_samples=args.max_samples,
        device_name=args.device,
        gen_config_path=args.gen_config,
        prefix_audit_samples=args.prefix_audit_samples,
        prefix_tolerance=args.prefix_tolerance,
        fail_on_step_clamp=args.fail_on_step_clamp,
        save_predictions=args.save_predictions,
        skip_sha256=args.skip_sha256,
    )
    print_summary(result)

    output = Path(args.save_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"\nJSON saved: {output}")


if __name__ == "__main__":
    main()
