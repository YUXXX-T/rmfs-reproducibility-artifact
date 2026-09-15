"""Train and evaluate the frozen behavior-aligned ``psi_pre`` probe.

The encoder and transition are never updated.  Five whole-seed folds produce
out-of-fold endpoint predictions, which are compared with the frozen decoder
at H=10 and the H=1 persistence baseline.  The output is diagnostic only;
there is intentionally no deployed checkpoint or Q-score integration here.
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

from WorldModel.core.psi_pre_head import (
    PSI_PRE_HEAD_SCHEMA_VERSION,
    PsiPreHead,
)
from WorldModel.evaluation.phase_c_psi_pre_protocol import (
    DATASET_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    SERVICE_CHANNELS,
    TARGET_CHANNELS,
    TRAFFIC_CHANNELS,
    TRAINING_SCHEMA_VERSION,
    canonical_sha256,
    fold_seed_splits,
    sha256_file,
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n"
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


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        torch.save(dict(payload), tmp_name)
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
        return value.item()
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
    if left.size < 2 or right.size != left.size:
        return float("nan")
    left = left.astype(np.float64) - float(np.mean(left))
    right = right.astype(np.float64) - float(np.mean(right))
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator <= 1e-15:
        return float("nan")
    return float((left @ right) / denominator)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(_rankdata(np.asarray(left)), _rankdata(np.asarray(right)))


def _bootstrap_mean_ci(values: Sequence[float], seed: int, repeats: int = 2000) -> list[float]:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return [float("nan"), float("nan")]
    if finite.size == 1:
        return [float(finite[0]), float(finite[0])]
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(finite, size=(int(repeats), finite.size), replace=True)
    return [float(value) for value in np.quantile(draws.mean(axis=1), [0.025, 0.975])]


def _group_rank_metrics(prediction: np.ndarray, target: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    values = []
    top1 = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if indices.size < 2:
            continue
        rho = _spearman(prediction[indices], target[indices])
        if math.isfinite(rho):
            values.append(rho)
        predicted_best = indices[int(np.argmax(prediction[indices]))]
        target_best = float(np.max(target[indices]))
        top1.append(bool(target[predicted_best] >= target_best - 1e-12))
    return {
        "groups": int(len(top1)),
        "spearman_mean": float(np.mean(values)) if values else float("nan"),
        "spearman_median": float(np.median(values)) if values else float("nan"),
        "positive_fraction": float(np.mean(np.asarray(values) > 0.0)) if values else float("nan"),
        "top1_hit_rate": float(np.mean(top1)) if top1 else float("nan"),
    }


def _channel_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    tensors: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    indices_np = indices.cpu().numpy()
    pred = np.asarray(prediction, dtype=np.float64)[indices_np]
    true = np.asarray(target, dtype=np.float64)[indices_np]
    residual = pred - true
    run_codes = tensors["run_index"].cpu().numpy()[indices_np]
    per_run = []
    for run in np.unique(run_codes):
        run_idx = np.flatnonzero(run_codes == run)
        rho = _spearman(pred[run_idx], true[run_idx])
        if math.isfinite(rho):
            per_run.append(rho)
    load_codes = tensors["load_code"].cpu().numpy()[indices_np]
    by_load = {}
    for load_code, load_name in enumerate(("low", "mid", "high")):
        load_idx = np.flatnonzero(load_codes == load_code)
        by_load[load_name] = (
            _spearman(pred[load_idx], true[load_idx])
            if load_idx.size >= 2 else float("nan")
        )
    return {
        "samples": int(pred.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(math.sqrt(np.mean(residual ** 2))),
        "bias": float(np.mean(residual)),
        "prediction_std": float(np.std(pred)),
        "target_std": float(np.std(true)),
        "pooled_spearman": _spearman(pred, true),
        "per_run_spearman_mean": float(np.mean(per_run)) if per_run else float("nan"),
        "per_run_cluster_ci95": _bootstrap_mean_ci(per_run, seed=seed),
        "run_sign_consistency": float(np.mean(np.asarray(per_run) > 0.0)) if per_run else float("nan"),
        "within_frame_station": _group_rank_metrics(
            pred, true, tensors["frame_group"].cpu().numpy()[indices_np]
        ),
        "by_load_spearman": by_load,
    }


def _weighted_mse(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    error = (prediction - target).square()
    weights = weights.reshape(-1, 1).to(dtype=error.dtype)
    return (error * weights).sum() / (weights.sum() * error.size(1)).clamp_min(1e-12)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _load_dataset(path: Path) -> dict[str, Any]:
    dataset = torch.load(path, map_location="cpu", weights_only=False)
    if dataset.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"wrong psi_pre dataset schema: {path}")
    if tuple(dataset.get("target_channels", ())) != TARGET_CHANNELS:
        raise ValueError("psi_pre target channel schema mismatch")
    if set(dataset.get("folds", {})) != {str(i) for i in range(5)}:
        raise ValueError("psi_pre dataset must contain five frozen folds")
    required = {
        "rollout_latents", "targets", "baseline_state", "baseline_h10",
        "baseline_h1", "run_index", "seed", "load_code", "frame_group",
    }
    missing = sorted(required - set(dataset.get("tensors", {})))
    if missing:
        raise ValueError(f"psi_pre dataset missing tensors: {missing}")
    return dataset


def _predict(head: PsiPreHead, latents: torch.Tensor, device: torch.device) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        return head(latents.to(device)).cpu().numpy()


def _weighted_mean(target: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    selected_target = target.index_select(0, indices)
    selected_weight = weights.index_select(0, indices).reshape(-1, 1)
    return (selected_target * selected_weight).sum(dim=0) / selected_weight.sum().clamp_min(1e-12)


def _train_fold(
    *,
    dataset: Mapping[str, Any],
    fold: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
) -> tuple[PsiPreHead, dict[str, Any]]:
    tensors = dataset["tensors"]
    indices = dataset["folds"][str(fold["fold"])]
    train_idx = indices["train"].long()
    val_idx = indices["val"].long()
    latent = tensors["rollout_latents"].float()
    target = tensors["targets"].float()
    weights = tensors["row_weight"].float()
    _seed_everything(seed)
    head = PsiPreHead(int(dataset["latent_dim"]), len(TARGET_CHANNELS)).to(device)
    loader = DataLoader(
        TensorDataset(
            latent.index_select(0, train_idx),
            target.index_select(0, train_idx),
            weights.index_select(0, train_idx),
        ),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    optimiser = torch.optim.AdamW(
        head.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    log_rows = []
    val_latent = latent.index_select(0, val_idx)
    val_target = target.index_select(0, val_idx)
    val_weight = weights.index_select(0, val_idx)
    for epoch in range(1, int(epochs) + 1):
        head.train()
        total = 0.0
        count = 0
        for batch_latent, batch_target, batch_weight in loader:
            batch_latent = batch_latent.to(device)
            batch_target = batch_target.to(device)
            batch_weight = batch_weight.to(device)
            optimiser.zero_grad(set_to_none=True)
            prediction = head(batch_latent)
            loss = _weighted_mse(prediction, batch_target, batch_weight)
            loss.backward()
            optimiser.step()
            total += float(loss.item()) * int(batch_latent.size(0))
            count += int(batch_latent.size(0))
        head.eval()
        with torch.no_grad():
            val_prediction = head(val_latent.to(device))
            val_loss = float(_weighted_mse(val_prediction, val_target.to(device), val_weight.to(device)).item())
        row = {
            "epoch": epoch,
            "train_mse": total / max(count, 1),
            "val_mse": val_loss,
        }
        log_rows.append(row)
        if val_loss < best_loss - 1e-9:
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
                break
    if best_state is None:
        raise RuntimeError("psi_pre fold did not produce a checkpoint")
    head.load_state_dict(best_state)
    fold_root = output_root / "folds" / f"fold_{int(fold['fold']) + 1:02d}"
    fold_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_root / "best_psi_pre_head.pt"
    _atomic_torch_save(checkpoint_path, {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "head_schema_version": PSI_PRE_HEAD_SCHEMA_VERSION,
        "state_dict": best_state,
        "latent_dim": int(dataset["latent_dim"]),
        "output_dim": len(TARGET_CHANNELS),
        "target_channels": list(TARGET_CHANNELS),
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_mse": best_loss,
        "training": {
            "optimizer": "AdamW",
            "loss": "run_and_group_weighted_mse",
            "seed": int(seed),
            "epochs_requested": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "patience": int(patience),
        },
        "audit": {
            "encoder_frozen": True,
            "transition_frozen": True,
            "online_connection_allowed": False,
            "locked_501_510_used": False,
        },
    })
    (fold_root / "train_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in log_rows),
        encoding="utf-8",
        newline="\n",
    )
    return head, {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_mse": best_loss,
        "checkpoint": checkpoint_path.as_posix(),
        "epochs_completed": len(log_rows),
    }


def _compare_metrics(
    prediction: np.ndarray,
    baseline_state: np.ndarray,
    baseline_h10: np.ndarray,
    baseline_h1: np.ndarray,
    target: np.ndarray,
    tensors: Mapping[str, torch.Tensor],
    test_idx: torch.Tensor,
) -> dict[str, Any]:
    channels = {}
    for index, name in enumerate(TARGET_CHANNELS):
        channels[name] = {
            "psi_pre": _channel_metrics(
                prediction[:, index], target[:, index], tensors, test_idx,
                seed=20260806 + index,
            ),
            "decoder_state_persistence": _channel_metrics(
                baseline_state[:, index], target[:, index], tensors, test_idx,
                seed=20260856 + index,
            ),
            "decoder_h10": _channel_metrics(
                baseline_h10[:, index], target[:, index], tensors, test_idx,
                seed=20260906 + index,
            ),
            "decoder_h1_persistence": _channel_metrics(
                baseline_h1[:, index], target[:, index], tensors, test_idx,
                seed=20261006 + index,
            ),
        }
    return channels


def _gate(report_channels: Mapping[str, Any]) -> dict[str, Any]:
    service = [report_channels[name]["psi_pre"] for name in SERVICE_CHANNELS]
    traffic = [report_channels[name]["psi_pre"] for name in TRAFFIC_CHANNELS]
    service_pass = all(
        float(row["pooled_spearman"]) >= 0.50
        and float(row["per_run_cluster_ci95"][0]) > 0.30
        and float(row["within_frame_station"]["spearman_mean"]) >= 0.50
        for row in service
    )
    traffic_count = sum(
        float(row["pooled_spearman"]) >= 0.40
        and float(row["per_run_cluster_ci95"][0]) > 0.20
        for row in traffic
    )
    beats_decoder = sum(
        float(report_channels[name]["psi_pre"]["mae"])
        < float(report_channels[name]["decoder_h10"]["mae"])
        for name in TARGET_CHANNELS
    )
    return {
        "service_channels_passed": bool(service_pass),
        "traffic_channels_passed": int(traffic_count) >= 3,
        "traffic_channel_count": int(traffic_count),
        "beats_frozen_decoder_channel_count": int(beats_decoder),
        "beats_frozen_decoder_passed": int(beats_decoder) >= 6,
        "psi_pre_kill_gate_passed": bool(
            service_pass and traffic_count >= 3 and beats_decoder >= 6
        ),
        "online_connection_allowed": False,
    }


def train_and_validate(
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
    torch_threads: int,
) -> dict[str, Any]:
    dataset = _load_dataset(dataset_path)
    try:
        torch.set_num_threads(max(1, int(torch_threads)))
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    output_root.mkdir(parents=True, exist_ok=True)
    tensors = dataset["tensors"]
    target = tensors["targets"].float().numpy()
    baseline_state = tensors["baseline_state"].float().numpy()
    baseline_h10 = tensors["baseline_h10"].float().numpy()
    baseline_h1 = tensors["baseline_h1"].float().numpy()
    oof_prediction = np.full_like(target, np.nan, dtype=np.float32)
    fold_reports = []
    for fold in fold_seed_splits():
        fold_key = str(fold["fold"])
        test_idx = dataset["folds"][fold_key]["test"].long()
        head, fold_report = _train_fold(
            dataset=dataset,
            fold=fold,
            output_root=output_root,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            seed=seed + int(fold["fold"]),
        )
        oof_prediction[test_idx.numpy()] = _predict(
            head, tensors["rollout_latents"].index_select(0, test_idx), device
        )
        fold_report["test_channels"] = _compare_metrics(
            oof_prediction,
            baseline_state,
            baseline_h10,
            baseline_h1,
            target,
            tensors,
            test_idx,
        )
        fold_reports.append(fold_report)
        print(
            f"[fold {int(fold['fold']) + 1}/5] test_rows={test_idx.numel()} "
            f"best_val_mse={fold_report['best_val_mse']:.6f}", flush=True
        )
    if not np.isfinite(oof_prediction).all():
        raise RuntimeError("OOF predictions are incomplete")

    all_indices = torch.arange(target.shape[0], dtype=torch.long)
    channels = _compare_metrics(
        oof_prediction,
        baseline_state,
        baseline_h10,
        baseline_h1,
        target,
        tensors,
        all_indices,
    )
    gate = _gate(channels)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "development_only": True,
        "protocol_sha256": dataset["protocol_sha256"],
        "dataset": dataset_path.as_posix(),
        "dataset_sha256": sha256_file(dataset_path),
        "source_checkpoint": dataset["source_checkpoint"],
        "source_checkpoint_sha256": dataset["source_checkpoint_sha256"],
        "horizon": int(dataset["horizon"]),
        "row_count": int(target.shape[0]),
        "folds": fold_reports,
        "channels": channels,
        "kill_gate": gate,
        "interpretation": {
            "psi_pre_definition": "phi_roll(z_H_pred), endpoint-local probe only",
            "z_H_is_predicted": True,
            "target_is_behavior_continuation_endpoint": True,
            "target_space_is_behavior_decoder_label_space": True,
            "final_physical_phi_state_not_claimed": True,
            "phi_state_trained": False,
            "delta_psi_tested": False,
            "station_count_fixed_in_head": False,
            "cross_channel_cancellation": False,
            "online_policy_changed": False,
            "q_score_changed": False,
            "locked_501_510_used": False,
        },
    }
    prediction_path = output_root / "oof_predictions.pt"
    _atomic_torch_save(prediction_path, {
        "schema_version": REPORT_SCHEMA_VERSION,
        "target_channels": list(TARGET_CHANNELS),
        "predictions": torch.from_numpy(oof_prediction),
        "targets": torch.from_numpy(target.astype(np.float32)),
        "baseline_state": torch.from_numpy(baseline_state.astype(np.float32)),
        "baseline_h10": torch.from_numpy(baseline_h10.astype(np.float32)),
        "baseline_h1": torch.from_numpy(baseline_h1.astype(np.float32)),
        "run_index": tensors["run_index"],
        "seed": tensors["seed"],
        "load_code": tensors["load_code"],
        "frame_group": tensors["frame_group"],
        "station_index": tensors["station_index"],
    })
    report_path = output_root / "psi_pre_validation.json"
    _atomic_json(report_path, report)
    manifest = "\n".join([
        f"{sha256_file(report_path)}  {report_path.name}",
        f"{sha256_file(prediction_path)}  {prediction_path.name}",
        f"{sha256_file(dataset_path)}  {dataset_path.name}",
    ]) + "\n"
    (output_root / "validated_outputs.sha256").write_text(
        manifest, encoding="utf-8", newline="\n"
    )
    print(
        f"[complete] psi_pre kill gate={gate['psi_pre_kill_gate_passed']} "
        f"report={report_path}", flush=True
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    device = torch.device(args.device)
    train_and_validate(
        dataset_path=args.dataset,
        output_root=args.output_root,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        torch_threads=args.torch_threads,
    )


if __name__ == "__main__":
    main()
