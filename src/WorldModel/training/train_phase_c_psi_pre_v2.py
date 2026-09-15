"""Train/evaluate the corrected, channel-reduced ``psi_pre`` v2 probe.

The frozen encoder, transition and behavior labels are not changed.  This
module only fits a shared station-region linear readout on the projected v2
dataset and reports out-of-fold metrics.  ``region_blocked_max`` is retained
as a diagnostic output, but is deliberately absent from the primary channel
gate and from any scalar penalty.
"""

from __future__ import annotations

import argparse
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

from WorldModel.core.psi_pre_head import PsiPreHead
from WorldModel.evaluation.phase_c_psi_pre_protocol import fold_seed_splits
from WorldModel.evaluation.phase_c_psi_pre_v2_protocol import (
    DATASET_SCHEMA_VERSION,
    PRIMARY_CHANNELS,
    REPORT_SCHEMA_VERSION,
    SERVICE_CHANNELS,
    TARGET_CHANNELS,
    TRAFFIC_DIAGNOSTIC_CHANNELS,
    TRAFFIC_PRIMARY_CHANNELS,
)
from WorldModel.training.train_phase_c_psi_pre import (
    _bootstrap_mean_ci,
    _channel_metrics,
    _json_safe,
    _seed_everything,
    _weighted_mse,
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


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_dataset(path: Path) -> dict[str, Any]:
    dataset = torch.load(path, map_location="cpu", weights_only=False)
    if dataset.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"wrong psi_pre v2 dataset schema: {path}")
    if tuple(dataset.get("target_channels", ())) != TARGET_CHANNELS:
        raise ValueError("psi_pre v2 target channel schema mismatch")
    if not bool(dataset.get("baseline_node_bce_sigmoid", False)):
        raise ValueError("v2 dataset does not carry corrected decoder baselines")
    if set(dataset.get("folds", {})) != {str(i) for i in range(5)}:
        raise ValueError("psi_pre v2 dataset must contain five frozen folds")
    required = {
        "rollout_latents", "targets", "baseline_state", "baseline_h10",
        "baseline_h1", "run_index", "seed", "load_code", "frame_group",
        "row_weight",
    }
    missing = sorted(required - set(dataset.get("tensors", {})))
    if missing:
        raise ValueError(f"psi_pre v2 dataset missing tensors: {missing}")
    return dataset


def _predict(head: PsiPreHead, latents: torch.Tensor) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        return head(latents).cpu().numpy()


def _train_fold(
    *,
    dataset: Mapping[str, Any],
    fold: Mapping[str, Any],
    output_root: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
) -> tuple[PsiPreHead, dict[str, Any]]:
    tensors = dataset["tensors"]
    split = dataset["folds"][str(fold["fold"])]
    train_idx = split["train"].long()
    val_idx = split["val"].long()
    latent = tensors["rollout_latents"].float()
    target = tensors["targets"].float()
    weights = tensors["row_weight"].float()

    _seed_everything(seed)
    head = PsiPreHead(int(dataset["latent_dim"]), len(TARGET_CHANNELS))
    loader = DataLoader(
        TensorDataset(
            latent.index_select(0, train_idx),
            target.index_select(0, train_idx),
            weights.index_select(0, train_idx),
        ),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    optimiser = torch.optim.AdamW(
        head.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    val_latent = latent.index_select(0, val_idx)
    val_target = target.index_select(0, val_idx)
    val_weight = weights.index_select(0, val_idx)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    log_rows = []
    for epoch in range(1, int(epochs) + 1):
        head.train()
        total = 0.0
        count = 0
        for batch_latent, batch_target, batch_weight in loader:
            optimiser.zero_grad(set_to_none=True)
            prediction = head(batch_latent)
            loss = _weighted_mse(prediction, batch_target, batch_weight)
            loss.backward()
            optimiser.step()
            total += float(loss.item()) * int(batch_latent.size(0))
            count += int(batch_latent.size(0))
        head.eval()
        with torch.no_grad():
            val_loss = _weighted_mse(head(val_latent), val_target, val_weight)
        row = {
            "epoch": epoch,
            "train_mse": total / max(count, 1),
            "val_mse": float(val_loss.item()),
        }
        log_rows.append(row)
        if row["val_mse"] < best_loss - 1e-10:
            best_loss = row["val_mse"]
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
        raise RuntimeError("v2 fold never produced a validation checkpoint")
    head.load_state_dict(best_state)
    fold_root = output_root / "folds" / f"fold_{int(fold['fold']) + 1:02d}"
    fold_root.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_root / "best_psi_pre_v2_head.pt"
    torch.save({
        "schema_version": "psi_pre_head_v2",
        "target_channels": list(TARGET_CHANNELS),
        "primary_channels": list(PRIMARY_CHANNELS),
        "diagnostic_channels": list(TRAFFIC_DIAGNOSTIC_CHANNELS),
        "latent_dim": int(dataset["latent_dim"]),
        "state_dict": best_state,
        "fold": fold,
    }, checkpoint)
    (fold_root / "train_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in log_rows),
        encoding="utf-8",
        newline="\n",
    )
    return head, {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_mse": best_loss,
        "checkpoint": checkpoint.as_posix(),
        "epochs_completed": len(log_rows),
    }


def _compare_metrics(
    prediction: np.ndarray,
    baselines: Mapping[str, np.ndarray],
    target: np.ndarray,
    tensors: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
) -> dict[str, Any]:
    channels = {}
    for index, name in enumerate(TARGET_CHANNELS):
        channel = {
            "role": (
                "service" if name in SERVICE_CHANNELS
                else "traffic_primary" if name in TRAFFIC_PRIMARY_CHANNELS
                else "traffic_diagnostic"
            ),
            "psi_pre_v2": _channel_metrics(
                prediction[:, index], target[:, index], tensors, indices,
                seed=20260807 + index,
            ),
        }
        for baseline_name, values in baselines.items():
            channel[baseline_name] = _channel_metrics(
                values[:, index], target[:, index], tensors, indices,
                seed=20260900 + index,
            )
        channels[name] = channel
    return channels


def _finite_ge(value: Any, threshold: float) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) >= float(threshold)
    except (TypeError, ValueError):
        return False


def _gate(channels: Mapping[str, Any]) -> dict[str, Any]:
    service_rows = [channels[name]["psi_pre_v2"] for name in SERVICE_CHANNELS]
    traffic_rows = [channels[name]["psi_pre_v2"] for name in TRAFFIC_PRIMARY_CHANNELS]
    service_pass = all(
        _finite_ge(row["pooled_spearman"], 0.50)
        and _finite_ge(row["per_run_cluster_ci95"][0], 0.30)
        and _finite_ge(row["within_frame_station"]["spearman_mean"], 0.50)
        for row in service_rows
    )
    traffic_count = sum(
        _finite_ge(row["pooled_spearman"], 0.40)
        and _finite_ge(row["per_run_cluster_ci95"][0], 0.20)
        for row in traffic_rows
    )
    primary_beats = sum(
        float(channels[name]["psi_pre_v2"]["mae"])
        < float(channels[name]["decoder_h10"]["mae"])
        for name in PRIMARY_CHANNELS
    )
    return {
        "service_channels_passed": bool(service_pass),
        "traffic_primary_channel_count": int(traffic_count),
        "traffic_primary_channels_passed": int(traffic_count) == len(traffic_rows),
        "primary_channels_beating_corrected_decoder_count": int(primary_beats),
        "blocked_max_is_diagnostic_only": True,
        "psi_pre_v2_diagnostic_gate_passed": bool(
            service_pass and traffic_count == len(traffic_rows)
        ),
        "online_connection_allowed": False,
    }


def train_and_validate(
    *,
    dataset_path: Path,
    output_root: Path,
    epochs: int = 200,
    batch_size: int = 4096,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    patience: int = 25,
    seed: int = 20260807,
    torch_threads: int = 8,
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
    baselines = {
        "decoder_state_persistence": tensors["baseline_state"].float().numpy(),
        "decoder_h10": tensors["baseline_h10"].float().numpy(),
        "decoder_h1_persistence": tensors["baseline_h1"].float().numpy(),
    }
    oof = np.full_like(target, np.nan, dtype=np.float32)
    fold_reports = []
    for fold in fold_seed_splits():
        test_idx = dataset["folds"][str(fold["fold"])]["test"].long()
        head, fold_report = _train_fold(
            dataset=dataset,
            fold=fold,
            output_root=output_root,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            seed=seed + int(fold["fold"]),
        )
        oof[test_idx.numpy()] = _predict(
            head, tensors["rollout_latents"].index_select(0, test_idx)
        )
        fold_report["test_channels"] = _compare_metrics(
            oof, baselines, target, tensors, test_idx
        )
        fold_reports.append(fold_report)
        print(
            f"[fold {int(fold['fold']) + 1}/5] test_rows={test_idx.numel()} "
            f"best_val_mse={fold_report['best_val_mse']:.6f}", flush=True
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("v2 OOF predictions are incomplete")
    all_indices = torch.arange(target.shape[0], dtype=torch.long)
    channels = _compare_metrics(oof, baselines, target, tensors, all_indices)
    gate = _gate(channels)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "development_only": True,
        "protocol_sha256": dataset["protocol_sha256"],
        "dataset": dataset_path.as_posix(),
        "dataset_sha256": sha256_file(dataset_path),
        "source_full_dataset_sha256": dataset["source_full_dataset_sha256"],
        "source_checkpoint": dataset.get("source_checkpoint"),
        "source_checkpoint_sha256": dataset.get("source_checkpoint_sha256"),
        "horizon": int(dataset["horizon"]),
        "row_count": int(target.shape[0]),
        "target_channels": list(TARGET_CHANNELS),
        "primary_channels": list(PRIMARY_CHANNELS),
        "diagnostic_channels": list(TRAFFIC_DIAGNOSTIC_CHANNELS),
        "baseline_contract": dataset.get(
            "baseline_contract",
            "BCE-logit channels sigmoid before region mean/max",
        ),
        "folds": fold_reports,
        "channels": channels,
        "diagnostic_gate": gate,
        "interpretation": {
            "psi_pre_definition": "phi_roll(z_H_pred), endpoint-local station-region probe",
            "same_psi_family_as_v1": True,
            "v1_changed": False,
            "z_H_is_predicted": True,
            "target_is_behavior_continuation_endpoint": True,
            "blocked_max_role": "diagnostic_only",
            "scalar_psi_aggregation": False,
            "delta_psi_tested": False,
            "online_policy_changed": False,
            "q_score_changed": False,
            "locked_501_510_used": False,
        },
    }
    prediction_path = output_root / "oof_predictions.pt"
    _atomic_torch_save(prediction_path, {
        "schema_version": REPORT_SCHEMA_VERSION,
        "target_channels": list(TARGET_CHANNELS),
        "primary_channels": list(PRIMARY_CHANNELS),
        "diagnostic_channels": list(TRAFFIC_DIAGNOSTIC_CHANNELS),
        "predictions": torch.from_numpy(oof),
        "targets": torch.from_numpy(target.astype(np.float32)),
        "baseline_state": torch.from_numpy(baselines["decoder_state_persistence"].astype(np.float32)),
        "baseline_h10": torch.from_numpy(baselines["decoder_h10"].astype(np.float32)),
        "baseline_h1": torch.from_numpy(baselines["decoder_h1_persistence"].astype(np.float32)),
        "run_index": tensors["run_index"],
        "seed": tensors["seed"],
        "load_code": tensors["load_code"],
        "frame_group": tensors["frame_group"],
        "station_index": tensors["station_index"],
    })
    report_path = output_root / "psi_pre_v2_validation.json"
    _atomic_json(report_path, report)
    manifest = "\n".join([
        f"{sha256_file(report_path)}  {report_path.name}",
        f"{sha256_file(prediction_path)}  {prediction_path.name}",
        f"{sha256_file(dataset_path)}  {dataset_path.name}",
    ]) + "\n"
    (output_root / "validated_outputs_v2.sha256").write_text(
        manifest, encoding="utf-8", newline="\n"
    )
    print(
        f"[complete] psi_pre v2 diagnostic gate={gate['psi_pre_v2_diagnostic_gate_passed']} "
        f"report={report_path}", flush=True
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    train_and_validate(
        dataset_path=args.dataset,
        output_root=args.output_root,
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
