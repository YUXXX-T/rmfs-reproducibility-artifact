"""Legacy checkpoint-repair utility; not the canonical training entry point.

Canonical from-scratch training jointly optimizes the model-owned
``LongRiskHead`` in :mod:`WorldModel.training.run_train_v6`.  This module is
retained only so the historically evaluated frozen checkpoints remain
auditable and reproducible.

Train only LongRiskHead on frozen latent endpoints.

The output checkpoint is a copy of the source checkpoint with exactly the six
``long_risk_head`` parameter tensors replaced.  Every other tensor is audited
with ``torch.equal`` before the checkpoint is accepted.

The default audit is the original Phase-C lineage audit.  The opt-in
``--planner-specific-source`` mode is for a previously trained PIBT-specific
checkpoint: its existing LongRiskHead is allowed to differ from the Stage-1
initialization, while all non-head tensors must remain frozen during this
repair.  Existing jobs keep the original contract by default.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from WorldModel.core.long_risk_schema import (
    LONG_RISK_OUTPUT_INDEX,
    LONG_RISK_OUTPUT_NAMES,
    LONG_RISK_QUANTILE_COMBO_WEIGHTS,
    LONG_RISK_SCHEMA_VERSION,
)
from WorldModel.core.model import RMFSWorldModel
from WorldModel.data.dataset import WorldModelDataset
from WorldModel.training.run_train_v6 import _load_splits
from WorldModel.training.train import _sample_to_device


TRAINING_SCHEMA_VERSION = "phase_c_long_risk_head_only_training_v1"
FEATURE_CACHE_SCHEMA_VERSION = "phase_c_long_risk_feature_cache_v1"
TENSOR_AUDIT_SCHEMA_VERSION = "phase_c_long_risk_tensor_audit_v1"
HEAD_PREFIX = "long_risk_head."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_checkpoint(path: Path) -> tuple[dict, RMFSWorldModel]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"unsupported World-Model checkpoint: {path}")
    config = dict(payload.get("model_config") or {})
    if not config:
        raise ValueError(f"checkpoint lacks model_config: {path}")
    model = RMFSWorldModel(**config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model._checkpoint_action_schema = dict(payload.get("action_schema") or {})
    model.eval()
    return payload, model


def vectorized_long_risk_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Exact batched equivalent of the repository's scalar head loss."""

    if prediction.ndim != 2 or prediction.shape[1] != len(
        LONG_RISK_OUTPUT_NAMES
    ):
        raise ValueError(f"unexpected prediction shape {tuple(prediction.shape)}")
    if target.shape != prediction.shape:
        raise ValueError(
            f"target shape {tuple(target.shape)} != {tuple(prediction.shape)}"
        )

    def pinball(column: int, tau: float) -> torch.Tensor:
        delta = target[:, column] - prediction[:, column]
        return torch.where(delta >= 0, tau * delta, (tau - 1.0) * delta).mean()

    index = LONG_RISK_OUTPUT_INDEX
    event_column = index["event_logit"]
    event_target = target[:, event_column]
    event_logit = prediction[:, event_column]
    bce = F.binary_cross_entropy_with_logits(
        event_logit, event_target, reduction="none"
    )
    probability_true = (
        torch.sigmoid(event_logit) * event_target
        + (1.0 - torch.sigmoid(event_logit)) * (1.0 - event_target)
    )
    focal = (0.75 * (1.0 - probability_true).pow(2.0) * bce).mean()
    monotonic = F.relu(
        prediction[:, index["peak_q90"]]
        - prediction[:, index["peak_q95"]]
    ).pow(2.0).mean()
    return (
        pinball(index["peak_q90"], 0.90)
        + pinball(index["peak_q95"], 0.95)
        + pinball(index["cvar_q90"], 0.90)
        + pinball(index["terminal_q90"], 0.90)
        + pinball(index["delta_group_q90"], 0.90)
        + focal
        + monotonic
    )


def _label_tensor(labels: Mapping, *, dtype: torch.dtype) -> torch.Tensor:
    def number(name: str) -> float:
        value = labels[name]
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    peak = number("risk_peak")
    return torch.tensor(
        [
            peak,
            peak,
            number("risk_cvar"),
            number("risk_terminal"),
            number("risk_delta_group"),
            number("risk_event"),
        ],
        dtype=dtype,
    )


def _head_input(z_k: torch.Tensor, z_0: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            z_k.mean(dim=0),
            z_k.max(dim=0).values,
            z_0.mean(dim=0),
            z_0.max(dim=0).values,
        ],
        dim=-1,
    )


def _precompute_split(
    model: RMFSWorldModel,
    dataset: WorldModelDataset,
    device: torch.device,
    split: str,
) -> dict:
    features = []
    targets = []
    group_ids: list[str] = []
    source_cells: list[str] = []
    missing = 0
    model.eval()
    with torch.no_grad():
        for offset, raw_sample in enumerate(dataset.samples, start=1):
            labels = raw_sample.get("long_risk_labels")
            if labels is None:
                missing += 1
                continue
            sample = _sample_to_device(raw_sample, device)
            station_ids = sample.get("station_node_ids")
            if isinstance(station_ids, torch.Tensor):
                station_ids = station_ids.tolist()
            z, demand, edge_attr = model.encode_state(
                sample["node_history"],
                sample["edge_index"],
                sample["edge_features"],
                sample["demand_context"],
            )
            _, _, _, _, z_0, z_k = model.rollout(
                z,
                demand,
                edge_attr,
                sample["action_node"],
                sample["action_global"],
                sample["edge_index"],
                station_ids,
            )
            feature = _head_input(z_k, z_0).detach().cpu()
            features.append(feature)
            targets.append(_label_tensor(labels, dtype=feature.dtype))
            group_id = str(raw_sample.get("candidate_group_id") or "")
            if not group_id:
                raise ValueError("long-risk sample lacks candidate_group_id")
            group_ids.append(group_id)
            load = str(
                raw_sample.get("source_load_level")
                or raw_sample.get("load_level")
                or "unknown"
            )
            seed = str(
                raw_sample.get("source_seed")
                or raw_sample.get("simulation_seed")
                or "unknown"
            )
            source_cells.append(f"{load}:seed{seed}")
            if offset % 1000 == 0:
                print(
                    f"[features] split={split} scanned={offset}/{len(dataset)} "
                    f"kept={len(features)}",
                    flush=True,
                )
    if not features:
        raise ValueError(f"split {split} has no long-risk supervision")
    feature_tensor = torch.stack(features)
    target_tensor = torch.stack(targets)
    if not torch.isfinite(feature_tensor).all():
        raise ValueError(f"non-finite frozen features in {split}")
    if not torch.isfinite(target_tensor).all():
        raise ValueError(f"non-finite long-risk targets in {split}")
    group_sizes = Counter(group_ids)
    eligible_groups = sum(value >= 2 for value in group_sizes.values())
    if eligible_groups <= 0:
        raise ValueError(f"split {split} has no multi-candidate groups")
    return {
        "features": feature_tensor,
        "targets": target_tensor,
        "group_ids": group_ids,
        "source_cells": source_cells,
        "missing_labels": missing,
        "eligible_groups": eligible_groups,
    }


def _predict(
    head: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    head.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            rows.append(
                head.net(features[start : start + batch_size].to(device)).cpu()
            )
    return torch.cat(rows, dim=0)


def _rank_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    group_ids: list[str],
) -> dict:
    index = LONG_RISK_OUTPUT_INDEX
    weights = LONG_RISK_QUANTILE_COMBO_WEIGHTS
    predicted = {
        "peak": prediction[:, index["peak_q95"]],
        "cvar": prediction[:, index["cvar_q90"]],
        "terminal": prediction[:, index["terminal_q90"]],
        "delta": prediction[:, index["delta_group_q90"]],
        "combo": (
            weights["peak_q95"] * prediction[:, index["peak_q95"]]
            + weights["cvar_q90"] * prediction[:, index["cvar_q90"]]
            + weights["terminal_q90"]
            * prediction[:, index["terminal_q90"]]
        ),
    }
    truth = {
        "peak": target[:, index["peak_q95"]],
        "cvar": target[:, index["cvar_q90"]],
        "terminal": target[:, index["terminal_q90"]],
        "delta": target[:, index["delta_group_q90"]],
        "combo": (
            weights["peak_q95"] * target[:, index["peak_q95"]]
            + weights["cvar_q90"] * target[:, index["cvar_q90"]]
            + weights["terminal_q90"] * target[:, index["terminal_q90"]]
        ),
    }
    groups: dict[str, list[int]] = defaultdict(list)
    for row, group_id in enumerate(group_ids):
        groups[group_id].append(row)
    result: dict[str, float | int] = {
        "samples": int(len(group_ids)),
        "eligible_groups": int(sum(len(rows) >= 2 for rows in groups.values())),
        "loss": float(vectorized_long_risk_loss(prediction, target).item()),
    }
    for name in predicted:
        correct = 0
        total = 0
        regrets = []
        hits = 0
        for rows in groups.values():
            if len(rows) < 2:
                continue
            pred_values = predicted[name][rows]
            true_values = truth[name][rows]
            for left in range(len(rows)):
                for right in range(left + 1, len(rows)):
                    true_delta = float(true_values[left] - true_values[right])
                    if abs(true_delta) < 1e-6:
                        continue
                    total += 1
                    pred_delta = float(pred_values[left] - pred_values[right])
                    if pred_delta * true_delta > 0:
                        correct += 1
            model_pick = int(torch.argmin(pred_values).item())
            oracle_pick = int(torch.argmin(true_values).item())
            regrets.append(
                float(true_values[model_pick] - true_values[oracle_pick])
            )
            hits += int(model_pick == oracle_pick)
        suffix = name
        result[f"pairwise_accuracy_{suffix}"] = correct / max(total, 1)
        result[f"pairwise_total_{suffix}"] = total
        result[f"top1_regret_{suffix}"] = sum(regrets) / max(len(regrets), 1)
        result[f"best_hit_rate_{suffix}"] = hits / max(len(regrets), 1)

    coverage_specs = (
        ("peak_q90", 0.90),
        ("peak_q95", 0.95),
        ("cvar_q90", 0.90),
        ("terminal_q90", 0.90),
        ("delta_group_q90", 0.90),
    )
    for name, nominal in coverage_specs:
        column = index[name]
        coverage = float((target[:, column] <= prediction[:, column]).float().mean())
        result[f"coverage_{name}"] = coverage
        result[f"coverage_error_{name}"] = abs(coverage - nominal)

    try:
        from scipy import stats as scipy_stats

        target_by_name = {
            "peak_q90": target[:, index["peak_q90"]],
            "peak_q95": target[:, index["peak_q95"]],
            "cvar_q90": target[:, index["cvar_q90"]],
            "terminal_q90": target[:, index["terminal_q90"]],
            "delta_group_q90": target[:, index["delta_group_q90"]],
        }
        for name, values in target_by_name.items():
            rho, _ = scipy_stats.spearmanr(
                prediction[:, index[name]].numpy(), values.numpy()
            )
            result[f"spearman_{name}"] = (
                None if not math.isfinite(float(rho)) else float(rho)
            )
    except ImportError:
        result["spearman_unavailable"] = True

    event_true = target[:, index["event_logit"]].numpy()
    event_score = prediction[:, index["event_logit"]].numpy()
    if len(set(float(value) for value in event_true)) >= 2:
        try:
            from sklearn.metrics import roc_auc_score

            result["event_auc"] = float(roc_auc_score(event_true, event_score))
        except ImportError:
            result["event_auc_unavailable"] = True
    else:
        result["event_single_class"] = True
    return result


def _selection_key(metrics: Mapping) -> tuple[float, float, float]:
    return (
        float(metrics["top1_regret_combo"]),
        -float(metrics["pairwise_accuracy_combo"]),
        float(metrics["loss"]),
    )


def _tensor_audit(
    *,
    stage1_payload: Mapping,
    source_payload: Mapping,
    target_payload: Mapping,
    planner_specific_source: bool = False,
) -> dict:
    stage1 = stage1_payload["state_dict"]
    source = source_payload["state_dict"]
    target = target_payload["state_dict"]
    if set(source) != set(target):
        raise ValueError("source/target state_dict keys differ")
    head_keys = sorted(key for key in source if key.startswith(HEAD_PREFIX))
    non_head_keys = sorted(key for key in source if not key.startswith(HEAD_PREFIX))
    if len(head_keys) != 6:
        raise ValueError(f"expected six LongRiskHead tensors, got {head_keys}")
    source_stage1_head_equal = {
        key: bool(key in stage1 and torch.equal(source[key], stage1[key]))
        for key in head_keys
    }
    non_head_equal = {
        key: bool(torch.equal(source[key], target[key])) for key in non_head_keys
    }
    head_changed = {
        key: bool(not torch.equal(source[key], target[key])) for key in head_keys
    }
    output_weight_key = "long_risk_head.net.4.weight"
    output_bias_key = "long_risk_head.net.4.bias"
    output_channel_changed = {}
    if output_weight_key in source and output_bias_key in source:
        output_channel_changed = {
            str(index): bool(
                not torch.equal(
                    source[output_weight_key][index], target[output_weight_key][index]
                )
                or not torch.equal(
                    source[output_bias_key][index], target[output_bias_key][index]
                )
            )
            for index in range(int(source[output_weight_key].shape[0]))
        }
    max_abs_delta = {
        key: float((source[key] - target[key]).abs().max().item())
        for key in head_keys
    }
    stage1_non_head_changed = [
        key
        for key in non_head_keys
        if key in stage1 and not torch.equal(source[key], stage1[key])
    ]
    cost_keys = [key for key in non_head_keys if key.startswith("cost_head.")]
    source_head_matches_stage1 = all(source_stage1_head_equal.values())
    checks = {
        # Preserve the historical check and its literal meaning.  The
        # planner-specific mode records it as false when appropriate, but
        # explicitly removes it from the set of acceptance requirements.
        "source_phasec_head_equals_stage1_initialization": (
            source_head_matches_stage1
        ),
        "source_phasec_non_head_training_occurred": bool(stage1_non_head_changed),
        "all_non_long_risk_tensors_bitwise_equal": all(non_head_equal.values()),
        "long_risk_head_updated": any(head_changed.values()),
        "cost_head_bitwise_equal": all(non_head_equal[key] for key in cost_keys),
        "model_config_equal": (
            source_payload.get("model_config") == target_payload.get("model_config")
        ),
        "action_schema_equal": (
            source_payload.get("action_schema") == target_payload.get("action_schema")
        ),
    }
    required_check_names = list(checks)
    if planner_specific_source:
        required_check_names.remove(
            "source_phasec_head_equals_stage1_initialization"
        )
    return {
        "schema_version": TENSOR_AUDIT_SCHEMA_VERSION,
        "passed": all(checks[name] for name in required_check_names),
        "checks": checks,
        "required_checks": required_check_names,
        "planner_specific_source": bool(planner_specific_source),
        "source_head_matches_stage1": source_head_matches_stage1,
        "head_keys": head_keys,
        "source_stage1_head_equal": source_stage1_head_equal,
        "head_changed": head_changed,
        "output_channel_changed": output_channel_changed,
        "head_max_abs_delta": max_abs_delta,
        "non_head_tensor_count": len(non_head_keys),
        "non_head_changed_keys": [
            key for key, equal in non_head_equal.items() if not equal
        ],
        "source_vs_stage1_changed_non_head_tensor_count": len(
            stage1_non_head_changed
        ),
        "cost_head_keys": cost_keys,
    }


def train(args: argparse.Namespace) -> None:
    source_path = Path(args.checkpoint)
    stage1_path = Path(args.stage1_reference)
    data_path = Path(args.data)
    splits_path = Path(args.splits)
    output_root = Path(args.output_root)
    for path in (source_path, stage1_path, data_path, splits_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_root.exists():
        raise FileExistsError(
            f"output exists; preserve it or select a new path: {output_root}"
        )
    staging = output_root.parent / f".{output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(int(args.torch_threads))
    device = torch.device(args.device)
    print(f"[device] {device} torch_threads={torch.get_num_threads()}")

    source_payload, model = _load_checkpoint(source_path)
    stage1_payload, _ = _load_checkpoint(stage1_path)
    model.to(device)
    full_dataset = WorldModelDataset.from_file(str(data_path))
    train_ds, val_ds, test_ds = _load_splits(
        str(splits_path), full_dataset.samples
    )
    caches = {
        "train": _precompute_split(model, train_ds, device, "train"),
        "val": _precompute_split(model, val_ds, device, "val"),
        "test": _precompute_split(model, test_ds, device, "test"),
    }
    cache_payload = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "source_checkpoint": source_path.as_posix(),
        "source_checkpoint_sha256": sha256_file(source_path),
        "source_dataset": data_path.as_posix(),
        "source_dataset_sha256": sha256_file(data_path),
        "splits": splits_path.as_posix(),
        "splits_sha256": sha256_file(splits_path),
        "splits_data": caches,
    }
    cache_path = staging / "frozen_long_risk_features.pt"
    _atomic_torch_save(cache_path, cache_payload)

    head = copy.deepcopy(model.long_risk_head).to(device)
    initial_head_state = {
        key: value.detach().cpu().clone()
        for key, value in head.state_dict().items()
    }
    baseline = {
        split: _rank_metrics(
            _predict(
                head,
                cache["features"],
                device,
                int(args.eval_batch_size),
            ),
            cache["targets"],
            cache["group_ids"],
        )
        for split, cache in caches.items()
    }

    train_cache = caches["train"]
    cell_counts = Counter(train_cache["source_cells"])
    weights = torch.tensor(
        [1.0 / cell_counts[cell] for cell in train_cache["source_cells"]],
        dtype=torch.double,
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        TensorDataset(train_cache["features"], train_cache["targets"]),
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    best_state = None
    best_metrics = None
    best_epoch = -1
    stale = 0
    log_rows = []
    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        head.train()
        total_loss = 0.0
        total_rows = 0
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = head.net(features)
            loss = vectorized_long_risk_loss(prediction, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(features)
            total_rows += len(features)
        val_prediction = _predict(
            head,
            caches["val"]["features"],
            device,
            int(args.eval_batch_size),
        )
        val_metrics = _rank_metrics(
            val_prediction,
            caches["val"]["targets"],
            caches["val"]["group_ids"],
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_rows, 1),
            "val": val_metrics,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        log_rows.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_combo_acc={val_metrics['pairwise_accuracy_combo']:.4f} "
            f"val_combo_regret={val_metrics['top1_regret_combo']:.6f}",
            flush=True,
        )
        if best_metrics is None or _selection_key(val_metrics) < _selection_key(
            best_metrics
        ):
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            best_metrics = val_metrics
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if int(args.patience) > 0 and stale >= int(args.patience):
            print(f"[early-stop] epoch={epoch} stale={stale}")
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("LongRiskHead training produced no selected state")
    head.load_state_dict(best_state, strict=True)

    trained = {
        split: _rank_metrics(
            _predict(
                head,
                cache["features"],
                device,
                int(args.eval_batch_size),
            ),
            cache["targets"],
            cache["group_ids"],
        )
        for split, cache in caches.items()
    }
    target_payload = copy.deepcopy(source_payload)
    target_state = target_payload["state_dict"]
    for key, value in best_state.items():
        target_state[f"{HEAD_PREFIX}{key}"] = value.detach().cpu().clone()
    target_payload["long_risk_training"] = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "source_checkpoint": source_path.as_posix(),
        "source_checkpoint_sha256": sha256_file(source_path),
        "stage1_reference": stage1_path.as_posix(),
        "stage1_reference_sha256": sha256_file(stage1_path),
        "source_lineage_mode": (
            "planner_specific_on_policy_repair"
            if args.planner_specific_source
            else "phase_c_stage1_lineage_repair"
        ),
        "source_dataset": data_path.as_posix(),
        "source_dataset_sha256": sha256_file(data_path),
        "splits": splits_path.as_posix(),
        "splits_sha256": sha256_file(splits_path),
        "trainable_prefixes": [HEAD_PREFIX],
        "best_epoch": best_epoch,
        "selection_key": list(_selection_key(best_metrics)),
        "long_risk_schema_version": LONG_RISK_SCHEMA_VERSION,
    }
    checkpoint_path = staging / "best_long_risk_world_model.pt"
    _atomic_torch_save(checkpoint_path, target_payload)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    audit = _tensor_audit(
        stage1_payload=stage1_payload,
        source_payload=source_payload,
        target_payload=reloaded,
        planner_specific_source=bool(args.planner_specific_source),
    )
    if not audit["passed"]:
        failed = [
            name
            for name in audit.get("required_checks", audit["checks"])
            if not audit["checks"][name]
        ]
        raise RuntimeError("tensor audit failed: " + ", ".join(failed))
    _atomic_json(staging / "tensor_audit.json", audit)

    # Strict-load the emitted artifact before accepting it.
    _, strict_model = _load_checkpoint(checkpoint_path)
    strict_state = strict_model.long_risk_head.state_dict()
    if any(not torch.equal(strict_state[key], best_state[key]) for key in best_state):
        raise RuntimeError("strict reload changed LongRiskHead tensors")

    log_path = staging / "train_log.jsonl"
    log_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in log_rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "source_checkpoint": {
            "path": source_path.as_posix(),
            "sha256": sha256_file(source_path),
        },
        "stage1_reference": {
            "path": stage1_path.as_posix(),
            "sha256": sha256_file(stage1_path),
        },
        "dataset": {
            "path": data_path.as_posix(),
            "sha256": sha256_file(data_path),
            "splits_path": splits_path.as_posix(),
            "splits_sha256": sha256_file(splits_path),
            "split_samples": {
                split: int(len(cache["features"]))
                for split, cache in caches.items()
            },
            "split_eligible_groups": {
                split: int(cache["eligible_groups"])
                for split, cache in caches.items()
            },
            "split_missing_labels": {
                split: int(cache["missing_labels"])
                for split, cache in caches.items()
            },
        },
        "training": {
            "only_long_risk_head_trainable": True,
            "planner_specific_source": bool(args.planner_specific_source),
            "epochs_requested": int(args.epochs),
            "epochs_actual": len(log_rows),
            "best_epoch": best_epoch,
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "seed": int(args.seed),
            "balanced_by_source_load_seed_cell": True,
            "initial_head_tensor_count": len(initial_head_state),
        },
        "baseline_untrained_head": baseline,
        "trained_head": trained,
        "tensor_audit": audit,
        "output_checkpoint": checkpoint_path.name,
        "output_checkpoint_sha256": sha256_file(checkpoint_path),
        "feature_cache": cache_path.name,
        "feature_cache_sha256": sha256_file(cache_path),
    }
    _atomic_json(staging / "train_summary.json", summary)
    files = sorted(path for path in staging.iterdir() if path.is_file())
    (staging / "trained_outputs.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    staging.rename(output_root)
    print(f"[complete] LongRiskHead-only checkpoint: {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage1-reference", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--planner-specific-source",
        action="store_true",
        help=(
            "allow a previously trained planner-specific LongRiskHead source; "
            "all non-head tensors must still remain bitwise frozen"
        ),
    )
    args = parser.parse_args()
    if min(
        args.epochs,
        args.batch_size,
        args.eval_batch_size,
        args.torch_threads,
    ) <= 0:
        parser.error("epoch, batch, and thread counts must be positive")
    train(args)


if __name__ == "__main__":
    main()
