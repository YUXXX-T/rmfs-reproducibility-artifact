"""Train the frozen-encoder station predictor used by J1 dispatch.

This is a dispatch-side auxiliary model, not RMFSWorldModel's station decoder
and not its jointly trained LongRiskHead.  Only the additive shared station
predictor is trained.  The Phase-C encoder is not loaded or updated here
because the dataset builder has already frozen and materialised its station
representations (station-node or region-pooled).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from WorldModel.core.station_congestion_head import (
    CHANNEL_NAMES,
    HEAD_SCHEMA_VERSION,
    StationCongestionHead,
    verify_scale_contract,
)
from WorldModel.data.build_station_congestion_head_dataset import (
    ARM_NAMES,
    DATASET_SCHEMA_VERSION,
    DEFAULT_OUTPUT_ROOT as DEFAULT_DATASET_ROOT,
    LOAD_NAMES,
    sha256_file,
)


TRAINING_SCHEMA_VERSION = "station_congestion_head_training_v1"
VALIDATION_SCHEMA_VERSION = "station_congestion_state_validation_v1"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATASET_ROOT / "linear_head_v1"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _json_safe(payload), indent=2, ensure_ascii=False
    ) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or right.size != left.size:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator <= 1e-15:
        return float("nan")
    return float((left @ right) / denominator)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(_rankdata(left), _rankdata(right))


def _one_hot(values: np.ndarray) -> list[np.ndarray]:
    unique = sorted(set(int(value) for value in values.tolist()))
    return [
        (values == category).astype(np.float64)
        for category in unique[1:]
    ]


def _partial_spearman(
    prediction: np.ndarray,
    target: np.ndarray,
    split: Mapping[str, torch.Tensor],
) -> float:
    columns = [np.ones(prediction.size, dtype=np.float64)]
    for key in (
        "tick_fraction",
        "global_open_order_count",
        "global_active_robot_ratio",
    ):
        values = split[key].cpu().numpy().astype(np.float64)
        ranks = _rankdata(values)
        scale = float(ranks.std())
        columns.append((ranks - ranks.mean()) / (scale if scale > 0 else 1.0))
    for key in ("load_code", "arm_code", "station_id"):
        columns.extend(_one_hot(split[key].cpu().numpy()))
    design = np.column_stack(columns)
    q, _ = np.linalg.qr(design, mode="reduced")
    left = _rankdata(prediction)
    right = _rankdata(target)
    left = left - q @ (q.T @ left)
    right = right - q @ (q.T @ right)
    return _pearson(left, right)


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    repeats: int = 2000,
    seed: int = 20260804,
) -> list[float]:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return [float("nan"), float("nan")]
    if finite.size == 1:
        return [float(finite[0]), float(finite[0])]
    rng = np.random.default_rng(seed)
    draws = rng.choice(finite, size=(int(repeats), finite.size), replace=True)
    return [float(value) for value in np.quantile(draws.mean(axis=1), [0.025, 0.975])]


def _group_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    correlations = []
    top1 = []
    order = np.argsort(groups, kind="mergesort")
    sorted_groups = groups[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_groups[1:] != sorted_groups[:-1], True]
    )
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        indices = order[start:end]
        if indices.size < 2:
            continue
        value = _spearman(prediction[indices], target[indices])
        if math.isfinite(value):
            correlations.append(value)
        predicted = int(indices[np.argmax(prediction[indices])])
        target_max = float(np.max(target[indices]))
        top1.append(bool(target[predicted] >= target_max - 1e-12))
    return {
        "groups": len(top1),
        "spearman_mean": (
            float(np.mean(correlations)) if correlations else float("nan")
        ),
        "spearman_median": (
            float(np.median(correlations)) if correlations else float("nan")
        ),
        "positive_fraction": (
            float(np.mean(np.asarray(correlations) > 0.0))
            if correlations else float("nan")
        ),
        "top1_hit_rate": float(np.mean(top1)) if top1 else float("nan"),
    }


def _channel_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    split: Mapping[str, torch.Tensor],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    residual = prediction - target
    run_codes = split["run_index"].cpu().numpy()
    per_run = []
    for code in np.unique(run_codes):
        indices = np.flatnonzero(run_codes == code)
        value = _spearman(prediction[indices], target[indices])
        if math.isfinite(value):
            per_run.append(value)
    by_arm = {}
    arm_codes = split["arm_code"].cpu().numpy()
    for code, name in enumerate(ARM_NAMES):
        indices = np.flatnonzero(arm_codes == code)
        by_arm[name] = (
            _spearman(prediction[indices], target[indices])
            if indices.size >= 2 else float("nan")
        )
    by_load = {}
    load_codes = split["load_code"].cpu().numpy()
    for code, name in enumerate(LOAD_NAMES):
        indices = np.flatnonzero(load_codes == code)
        by_load[name] = (
            _spearman(prediction[indices], target[indices])
            if indices.size >= 2 else float("nan")
        )
    return {
        "samples": int(prediction.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(math.sqrt(np.mean(residual ** 2))),
        "bias": float(np.mean(residual)),
        "prediction_std": float(np.std(prediction)),
        "target_std": float(np.std(target)),
        "pooled_spearman": _spearman(prediction, target),
        "partial_spearman": _partial_spearman(prediction, target, split),
        "per_run_spearman_mean": (
            float(np.mean(per_run)) if per_run else float("nan")
        ),
        "per_run_cluster_ci95": _bootstrap_mean_ci(
            per_run, seed=bootstrap_seed
        ),
        "run_sign_consistency": (
            float(np.mean(np.asarray(per_run) > 0.0))
            if per_run else float("nan")
        ),
        "within_tick_station": _group_metrics(
            prediction,
            target,
            split["frame_group"].cpu().numpy(),
        ),
        "by_arm_spearman": by_arm,
        "by_load_spearman": by_load,
    }


def evaluate_head(
    head: StationCongestionHead,
    split: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    head.eval()
    with torch.no_grad():
        prediction = head.forward_station_latents(
            split["latents"].to(device)
        ).cpu().numpy()
    target = split["targets"].cpu().numpy()
    channels = {}
    for index, name in enumerate(CHANNEL_NAMES):
        channels[name] = _channel_metrics(
            prediction[:, index],
            target[:, index],
            split,
            bootstrap_seed=20260804 + index,
        )
    return {
        "channels": channels,
        "mean_mse": float(np.mean((prediction - target) ** 2)),
    }


def _quick_validation(
    head: StationCongestionHead,
    split: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    """Cheap epoch-level validation; full clustered metrics run once at end."""

    head.eval()
    with torch.no_grad():
        prediction = head.forward_station_latents(
            split["latents"].to(device)
        ).cpu().numpy()
    target = split["targets"].cpu().numpy()
    return {
        "mean_mse": float(np.mean((prediction - target) ** 2)),
        "spearman": {
            name: _spearman(prediction[:, index], target[:, index])
            for index, name in enumerate(CHANNEL_NAMES)
        },
    }


def _load_dataset(path: Path) -> dict[str, Any]:
    dataset = torch.load(path, map_location="cpu", weights_only=False)
    if dataset.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("wrong station congestion latent dataset schema")
    if tuple(dataset.get("channel_names", ())) != CHANNEL_NAMES:
        raise ValueError("station congestion channel schema mismatch")
    verify_scale_contract(dataset["scale_contract"])
    splits = dataset.get("splits") or {}
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("station congestion dataset must have train/val/test")
    return dataset


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    *,
    dataset_path: Path,
    output_root: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
    formal: bool = False,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(
            f"training output already exists; choose a new path: {output_root}"
        )
    _seed_everything(seed)
    dataset = _load_dataset(dataset_path)
    if formal and dataset.get("development_only") is not False:
        raise ValueError(
            "formal station-head training requires a dataset explicitly "
            "marked development_only=False"
        )
    latent_dim = int(dataset["latent_dim"])
    representation = dataset.get("representation") or {
        "name": "station_node",
        "primary_region_hops": 3,
        "station_ids_required": True,
        "features": ["station_node"],
        "legacy_inferred": True,
    }
    splits = dataset["splits"]
    train_split = splits["train"]
    loader = DataLoader(
        TensorDataset(train_split["latents"], train_split["targets"]),
        batch_size=int(batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=False,
    )
    head = StationCongestionHead(latent_dim).to(device)
    optimiser = torch.optim.AdamW(
        head.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    loss_fn = torch.nn.MSELoss()
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    logs = []
    for epoch in range(1, int(epochs) + 1):
        head.train()
        total_loss = 0.0
        total_samples = 0
        for latents, targets in loader:
            latents = latents.to(device)
            targets = targets.to(device)
            optimiser.zero_grad(set_to_none=True)
            prediction = head.forward_station_latents(latents)
            loss = loss_fn(prediction, targets)
            loss.backward()
            optimiser.step()
            count = int(latents.size(0))
            total_loss += float(loss.item()) * count
            total_samples += count
        train_loss = total_loss / max(total_samples, 1)
        val_metrics = _quick_validation(head, splits["val"], device)
        val_loss = float(val_metrics["mean_mse"])
        row = {
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "val_traffic_spearman": val_metrics["spearman"]["traffic"],
            "val_service_spearman": val_metrics["spearman"]["service"],
        }
        logs.append(row)
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f} "
            f"traffic_rho={row['val_traffic_spearman']:.4f} "
            f"service_rho={row['val_service_spearman']:.4f}",
            flush=True,
        )
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                print(f"[early-stop] epoch={epoch} patience={patience}")
                break
    if best_state is None:
        raise RuntimeError("station congestion head training produced no checkpoint")
    head.load_state_dict(best_state)
    validation = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "development_only": not bool(formal),
        "performance_gate_frozen": False,
        "best_epoch": best_epoch,
        "best_val_mse": best_loss,
        "splits": {
            name: evaluate_head(head, split, device)
            for name, split in splits.items()
        },
        "interpretation": {
            "state_level_only": True,
            "rollout_endpoint_tested": False,
            "delta_psi_tested": False,
            "online_policy_changed": False,
            "representation": representation,
        },
    }

    staging = output_root.parent / f".{output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    checkpoint_payload = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "head_schema_version": HEAD_SCHEMA_VERSION,
        "state_dict": best_state,
        "latent_dim": latent_dim,
        "encoder_latent_dim": int(dataset.get("encoder_latent_dim", latent_dim)),
        "representation": representation,
        "channel_names": list(CHANNEL_NAMES),
        "scale_contract": dataset["scale_contract"],
        "scale_contract_sha256": dataset["scale_contract"]["contract_sha256"],
        "source_dataset": dataset_path.as_posix(),
        "source_dataset_sha256": sha256_file(dataset_path),
        "source_encoder_checkpoint": dataset["source_checkpoint"],
        "source_encoder_checkpoint_sha256": dataset["source_checkpoint_sha256"],
        "split_seeds": dataset["split_seeds"],
        "best_epoch": best_epoch,
        "best_val_mse": best_loss,
        "training": {
            "optimizer": "AdamW",
            "loss": "equal_channel_mse",
            "epochs_requested": int(epochs),
            "epochs_completed": len(logs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "patience": int(patience),
            "seed": int(seed),
        },
        "audit": {
            "passed": True,
            "formal_training_contract": bool(formal),
            "encoder_frozen": True,
            "station_ids_required": True,
            "fixed_station_output_dimension": False,
            "traffic_service_separate": True,
            "locked_501_510_used": False,
            "representation_frozen": True,
        },
    }
    checkpoint_path = staging / "best_station_congestion_head.pt"
    torch.save(checkpoint_payload, checkpoint_path)
    validation_path = staging / "state_level_validation.json"
    _atomic_json(validation_path, validation)
    log_path = staging / "train_log.jsonl"
    log_path.write_text(
        "".join(json.dumps(_json_safe(row)) + "\n" for row in logs),
        encoding="utf-8",
    )
    summary = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "development_only": not bool(formal),
        "best_epoch": best_epoch,
        "best_val_mse": best_loss,
        "representation": representation,
        "checkpoint": checkpoint_path.name,
        "validation": validation_path.name,
        "source_dataset_sha256": sha256_file(dataset_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_sha256": sha256_file(validation_path),
        "audit": checkpoint_payload["audit"],
    }
    _atomic_json(staging / "train_summary.json", summary)
    manifest = staging / "trained_outputs.sha256"
    manifest.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != manifest.name
        ) + "\n",
        encoding="utf-8",
    )
    staging.rename(output_root)
    print(f"[complete] station congestion head: {output_root}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "station_congestion_latents.pt",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(max(int(args.torch_threads), 1))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")
    train(
        dataset_path=args.dataset,
        output_root=args.output_root,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        formal=args.formal,
    )


if __name__ == "__main__":
    main()
