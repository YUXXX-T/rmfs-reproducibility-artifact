"""
V6 Two-Stage Joint World-Model Training
=======================================
Stage 1: short-horizon dynamics warm-up (no ranking or long-risk loss).
Stage 2: joint dynamics, ranking, and long-risk optimization.  LongRiskHead is
part of RMFSWorldModel and is saved in the same checkpoint; it is not a
separate canonical training stage.

Does NOT overwrite run_train_v5.py or train.py. Reuses compute_loss,
compute_ranking_loss, evaluate_ranking, evaluate_top1_regret from train.py.

Usage:
    python -m WorldModel.training.run_train_v6 \
      --data DataGen/wm_data/fused_v1/wm_train_data_fused.pt \
      --splits DataGen/wm_data/fused_v1/splits.json \
      --save-dir artifacts/generated/checkpoints/world_model_v6 \
      --stage1-epochs 15 --stage1-horizon 3 --stage1-lr 1e-3 \
      --stage2-epochs 30 --stage2-horizon 10 --stage2-lr 3e-4 \
      --stage2-alpha-rank 0.1 --stage2-freeze-epochs 5 \
      --balanced-sampler --device auto
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from WorldModel.model import RMFSWorldModel
from WorldModel.dataset import WorldModelDataset
from WorldModel.core.costs import CONGESTION_LAMBDAS
from WorldModel.train import (
    compute_loss,
    compute_ranking_loss,
    evaluate_ranking,
    evaluate_top1_regret,
    evaluate_long_risk,
    _sample_to_device,
    _compute_pos_weights_from_dataset,
    _extract_model_config,
    _infer_action_schema,
    _save_checkpoint,
    _build_data_quality_snapshot,
    NODE_CHANNEL_NAMES,
    DEFAULT_NODE_CHANNEL_WEIGHTS,
    SYSTEM_LABEL_NAMES,
)

_RANKING_KEYS = frozenset({
    "node_history", "edge_index", "edge_features", "demand_context",
    "action_node", "action_global", "station_node_ids",
})


def _pair_ranking_to_device(pair: dict, device: torch.device) -> dict:
    """Move only ranking-necessary fields to device (no future labels)."""
    def _sel(sample):
        out = {}
        for k, v in sample.items():
            if k not in _RANKING_KEYS:
                continue
            out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        return out
    return {
        "sample_i": _sel(pair["sample_i"]),
        "sample_j": _sel(pair["sample_j"]),
    }


# ------------------------------------------------------------------
# HorizonWrapper
# ------------------------------------------------------------------

class HorizonWrapper(Dataset):
    """Wrap a WorldModelDataset and truncate future labels to given horizon."""

    def __init__(self, dataset: WorldModelDataset, horizon: int):
        self.dataset = dataset
        self.horizon = horizon

    @property
    def samples(self):
        return self.dataset.samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        H = self.horizon
        out = dict(sample)
        for key in ("future_node_labels", "future_system_labels",
                    "future_station_labels", "future_mask"):
            if key in out and out[key].shape[0] > H:
                out[key] = out[key][:H]
        return out


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _collate_single(batch):
    return batch[0]


def _count_long_risk_samples(dataset: Optional[WorldModelDataset]) -> int:
    if dataset is None:
        return 0
    return sum(1 for s in dataset.samples if "long_risk_labels" in s)


def _has_long_risk_labels(dataset: Optional[WorldModelDataset]) -> bool:
    return _count_long_risk_samples(dataset) > 0


def _load_splits(splits_path: str, all_samples: List[dict]):
    """Load splits.json and partition samples by candidate_group_id."""
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_gids = set(splits["train"])
    val_gids = set(splits["val"])
    test_gids = set(splits["test"])

    train_s, val_s, test_s = [], [], []
    unassigned = 0
    for s in all_samples:
        gid = s.get("candidate_group_id", "")
        if gid in train_gids:
            train_s.append(s)
        elif gid in val_gids:
            val_s.append(s)
        elif gid in test_gids:
            test_s.append(s)
        else:
            unassigned += 1

    if unassigned > 0:
        print(f"  WARNING: {unassigned} samples not in any split (orphan group_ids)")

    return (
        WorldModelDataset(train_s),
        WorldModelDataset(val_s),
        WorldModelDataset(test_s),
    )


def _build_balanced_sampler(dataset: WorldModelDataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler balanced by source_run_id."""
    run_counts: Dict[str, int] = defaultdict(int)
    for s in dataset.samples:
        run_counts[s.get("source_run_id", "unknown")] += 1

    weights = []
    for s in dataset.samples:
        rid = s.get("source_run_id", "unknown")
        weights.append(1.0 / run_counts[rid])

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _balance_pairs(pairs: List[dict], field: str, seed: int = 42) -> List[dict]:
    """Resample pairwise pairs with balanced weights by a source field."""
    group_counts: Dict[str, int] = defaultdict(int)
    for p in pairs:
        val = p["sample_i"].get(field, "unknown")
        group_counts[val] += 1
    weights = [1.0 / group_counts[p["sample_i"].get(field, "unknown")] for p in pairs]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]
    rng = random.Random(seed)
    indices = rng.choices(range(len(pairs)), weights=probs, k=len(pairs))
    return [pairs[i] for i in indices]


def _build_model_from_sample(s0: dict, horizon: int, num_stations: int) -> RMFSWorldModel:
    """Build model with architecture inferred from a sample."""
    return RMFSWorldModel(
        node_feat_dim=s0["node_history"].shape[-1],
        edge_feat_dim=s0["edge_features"].shape[-1],
        hidden_dim=64,
        demand_dim=s0["demand_context"].shape[0],
        action_node_dim=s0["action_node"].shape[-1],
        action_global_dim=s0["action_global"].shape[0],
        num_stations=num_stations,
        rollout_horizon=horizon,
    )


def _infer_num_stations(s0: dict) -> int:
    sta = s0.get("station_node_ids")
    if sta is not None:
        return len(sta)
    return 4


def _run_epoch(
    model, loader, optimizer, device,
    alpha_system, alpha_station,
    node_channel_weights, wait_pos_weight, blocked_pos_weight,
    pairwise_data=None, alpha_rank=0.0, margin=1.0,
    pair_step_offset=0,
    alpha_long_risk=0.0,
    b3_tail_dims=None, b3_tail_tau=0.90, b3_residual_alpha=0.0,
):
    """Run one training epoch. Returns (metrics_dict, pair_step_offset)."""
    model.train()
    epoch_loss = 0.0
    epoch_node_channels = {name: 0.0 for name in NODE_CHANNEL_NAMES}
    epoch_system = 0.0
    epoch_station = 0.0
    epoch_rank = 0.0
    epoch_b3 = 0.0
    epoch_sys_channels = {name: 0.0 for name in SYSTEM_LABEL_NAMES}
    n = 0

    for sample in loader:
        sample = _sample_to_device(sample, device)
        optimizer.zero_grad()
        losses = compute_loss(
            model, sample,
            alpha_system=alpha_system,
            alpha_station=alpha_station,
            node_channel_weights=node_channel_weights,
            wait_pos_weight=wait_pos_weight,
            blocked_pos_weight=blocked_pos_weight,
            alpha_long_risk=alpha_long_risk,
            b3_tail_dims=b3_tail_dims,
            b3_tail_tau=b3_tail_tau,
            b3_residual_alpha=b3_residual_alpha,
        )
        total = losses["total"]

        if pairwise_data and alpha_rank > 0:
            pair_idx = (pair_step_offset + n) % len(pairwise_data)
            pair = pairwise_data[pair_idx]
            pair_dev = _pair_ranking_to_device(pair, device)
            rank_loss = compute_ranking_loss(model, pair_dev, margin=margin)
            total = total + alpha_rank * rank_loss
            epoch_rank += rank_loss.item()

        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()

        epoch_loss += total.item()
        for ch_name in NODE_CHANNEL_NAMES:
            epoch_node_channels[ch_name] += losses[ch_name]
        epoch_system += losses["system"]
        epoch_station += losses["station"]
        epoch_b3 += losses["b3_residual"]
        for ch_name, ch_val in losses["system_channels"].items():
            epoch_sys_channels[ch_name] += ch_val
        n += 1

    avg = lambda x: x / max(n, 1)
    record = {
        "loss": round(avg(epoch_loss), 6),
        "system": round(avg(epoch_system), 6),
        "station": round(avg(epoch_station), 6),
        "rank": round(avg(epoch_rank), 6),
        "b3_residual": round(avg(epoch_b3), 6),
        "grad_norm": round(grad_norm if n > 0 else 0.0, 4),
    }
    for ch_name in NODE_CHANNEL_NAMES:
        record[ch_name] = round(avg(epoch_node_channels[ch_name]), 6)
    for ch_name in SYSTEM_LABEL_NAMES:
        record[f"sys_{ch_name}"] = round(avg(epoch_sys_channels[ch_name]), 6)

    new_offset = pair_step_offset + n
    return record, new_offset


def _eval_val_dynamics(model, val_ds, device, horizon,
                       node_channel_weights, wait_pos_weight, blocked_pos_weight,
                       alpha_long_risk=0.0,
                       b3_tail_dims=None, b3_tail_tau=0.90, b3_residual_alpha=0.0):
    """Evaluate dynamics loss on val set without gradient."""
    model.eval()
    wrapped = HorizonWrapper(val_ds, horizon)
    loader = DataLoader(wrapped, batch_size=1, shuffle=False, collate_fn=_collate_single)
    total_loss = 0.0
    total_sys = 0.0
    total_sta = 0.0
    total_lr = 0.0
    total_b3 = 0.0
    n = 0
    with torch.no_grad():
        for sample in loader:
            sample = _sample_to_device(sample, device)
            losses = compute_loss(
                model, sample,
                alpha_system=1.0,
                alpha_station=0.5,
                node_channel_weights=node_channel_weights,
                wait_pos_weight=wait_pos_weight,
                blocked_pos_weight=blocked_pos_weight,
                alpha_long_risk=alpha_long_risk,
                b3_tail_dims=b3_tail_dims,
                b3_tail_tau=b3_tail_tau,
                b3_residual_alpha=b3_residual_alpha,
            )
            total_loss += losses["total"].item()
            total_sys += losses["system"]
            total_sta += losses["station"]
            total_lr += losses["long_risk"]
            total_b3 += losses["b3_residual"]
            n += 1
    avg = lambda x: x / max(n, 1)
    result = {
        "val_loss": round(avg(total_loss), 6),
        "val_system": round(avg(total_sys), 6),
        "val_station": round(avg(total_sta), 6),
    }
    if alpha_long_risk > 0:
        result["val_long_risk"] = round(avg(total_lr), 6)
    if b3_residual_alpha > 0:
        result["val_b3_residual"] = round(avg(total_b3), 6)
    return result


# ------------------------------------------------------------------
# Stage 1
# ------------------------------------------------------------------

def run_stage1(
    model, train_ds, val_ds, device, args,
    node_channel_weights, wait_pos_weight, blocked_pos_weight,
    save_dir,
    b3_tail_dims=None,
):
    """Stage 1: short-horizon dynamics warm-up, no ranking."""
    print(f"\n{'=' * 60}")
    print(f"  Stage 1: Dynamics Warm-up")
    print(f"    horizon={args.stage1_horizon}, epochs={args.stage1_epochs}, lr={args.stage1_lr}")
    print(f"    alpha_rank=0 (no ranking)")
    print(f"{'=' * 60}")

    wrapped = HorizonWrapper(train_ds, args.stage1_horizon)

    sampler = None
    shuffle = True
    if args.balanced_sampler:
        sampler = _build_balanced_sampler(train_ds)
        shuffle = False
    loader = DataLoader(wrapped, batch_size=1, shuffle=shuffle,
                        sampler=sampler, collate_fn=_collate_single)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.stage1_lr)

    log_path = os.path.join(save_dir, "train_log_stage1.jsonl")
    log_file = open(log_path, "w", encoding="utf-8")

    best_loss = float("inf")
    best_epoch = -1
    records = []

    for epoch in range(args.stage1_epochs):
        t0 = time.time()
        record, _ = _run_epoch(
            model, loader, optimizer, device,
            alpha_system=1.0, alpha_station=0.5,
            node_channel_weights=node_channel_weights,
            wait_pos_weight=wait_pos_weight,
            blocked_pos_weight=blocked_pos_weight,
            pairwise_data=None, alpha_rank=0.0,
            alpha_long_risk=0.0,  # Stage 1: no long-risk training
            b3_tail_dims=b3_tail_dims,
            b3_tail_tau=args.b3_system_tail_tau,
            b3_residual_alpha=args.b3_system_residual_alpha,
        )
        elapsed = time.time() - t0
        record["epoch"] = epoch + 1
        record["elapsed_seconds"] = round(elapsed, 1)

        if val_ds is not None and len(val_ds) > 0:
            val_metrics = _eval_val_dynamics(
                model, val_ds, device, args.stage1_horizon,
                node_channel_weights, wait_pos_weight, blocked_pos_weight,
                b3_tail_dims=b3_tail_dims,
                b3_tail_tau=args.b3_system_tail_tau,
                b3_residual_alpha=args.b3_system_residual_alpha,
            )
            record.update(val_metrics)

        if record["loss"] < best_loss:
            best_loss = record["loss"]
            best_epoch = epoch + 1

        msg = (f"  S1 Epoch {epoch+1:3d}/{args.stage1_epochs}  "
               f"loss={record['loss']:.4f}  "
               f"system={record['system']:.4f}  "
               f"station={record['station']:.4f}  "
               f"|g|={record['grad_norm']:.2f}")
        if "val_loss" in record:
            msg += f"  val={record['val_loss']:.4f}"
        msg += f"  {elapsed:.0f}s"
        print(msg)

        records.append(record)
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()

    log_file.close()
    ckpt_path = _save_checkpoint(model, save_dir, "stage1_world_model.pt")
    print(f"\n  Stage 1 done. best_epoch={best_epoch}, best_loss={best_loss:.4f}")
    print(f"  Checkpoint: {ckpt_path}")

    return {
        "epochs": args.stage1_epochs,
        "horizon": args.stage1_horizon,
        "lr": args.stage1_lr,
        "best_epoch_by_loss": best_epoch,
        "best_loss": round(best_loss, 6),
        "final_losses": records[-1] if records else {},
    }


# ------------------------------------------------------------------
# Stage 2
# ------------------------------------------------------------------

def run_stage2(
    model, train_ds, val_ds, device, args,
    node_channel_weights, wait_pos_weight, blocked_pos_weight,
    save_dir,
    cost_lambdas=None,
    b3_tail_dims=None,
):
    """Stage 2: jointly optimize full-horizon, ranking, and long-risk losses."""
    print(f"\n{'=' * 60}")
    print(f"  Stage 2: Full Horizon + Ranking + Long Risk")
    print(f"    horizon={args.stage2_horizon}, epochs={args.stage2_epochs}, "
          f"lr={args.stage2_lr}")
    print(f"    alpha_rank={args.stage2_alpha_rank}, "
          f"freeze_epochs={args.stage2_freeze_epochs}")
    print(f"    alpha_long_risk={args.alpha_long_risk}")
    print(f"    train_cost_head={args.train_cost_head}")
    print(f"    unfreeze_lr_factor={args.stage2_unfreeze_lr_factor}, "
          f"early_stopping={args.early_stopping_patience} "
          f"(monitor={args.early_stopping_monitor})")
    print(f"{'=' * 60}")

    # --- Build pairwise data (stay on CPU, move per-step) ---
    print(f"\n  Building pairwise data ...")
    train_pairs = train_ds.build_pairwise_data(
        max_pairs=args.max_train_pairs, cost_lambdas=cost_lambdas)
    val_pairs = []
    if val_ds is not None and len(val_ds) > 0:
        val_pairs = val_ds.build_pairwise_data(
            max_pairs=args.max_val_pairs, cost_lambdas=cost_lambdas)
    if args.pair_balance_by and train_pairs:
        train_pairs = _balance_pairs(train_pairs, args.pair_balance_by)
        print(f"    train pairs resampled by {args.pair_balance_by}: {len(train_pairs)}")
    print(f"    train pairs: {len(train_pairs)}, val pairs: {len(val_pairs)}")
    val_has_long_risk = _has_long_risk_labels(val_ds)
    if val_has_long_risk:
        print(f"    val long-risk samples: {_count_long_risk_samples(val_ds)}")

    # --- Cost head freeze ---
    if args.congestion_cost:
        lam = torch.tensor(
            CONGESTION_LAMBDAS,
            dtype=model.cost_head.lambdas.dtype,
            device=model.cost_head.lambdas.device,
        )
        model.cost_head.lambdas.data.copy_(lam)
        model.cost_head.lambdas.requires_grad_(False)
        print(f"  cost_head.lambdas = CONGESTION (ch5=0, frozen)")
    elif not args.train_cost_head:
        model.cost_head.lambdas.requires_grad_(False)
        print(f"  cost_head.lambdas frozen (default)")
    else:
        print(f"  cost_head.lambdas trainable (--train-cost-head)")

    # --- Freeze strategy ---
    freeze_modules = ["state_encoder", "demand_encoder",
                      "node_decoder", "system_decoder", "station_decoder"]
    train_modules = ["action_encoder", "transition", "long_risk_head"]
    if args.train_cost_head:
        train_modules.append("cost_head")
    decoder_recalibration_modules = [
        "node_decoder", "system_decoder", "station_decoder", "long_risk_head"
    ]
    if args.train_cost_head:
        decoder_recalibration_modules.append("cost_head")

    def _set_freeze(freeze: bool):
        for name in freeze_modules:
            mod = getattr(model, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = not freeze

    def _set_trainable_modules(module_names):
        allowed = set(module_names)
        for name, module in model.named_children():
            trainable = name in allowed
            for p in module.parameters():
                p.requires_grad = trainable
        if not args.train_cost_head and hasattr(model, "cost_head"):
            model.cost_head.lambdas.requires_grad_(False)

    # --- DataLoader ---
    sampler = None
    shuffle = True
    if args.balanced_sampler:
        sampler = _build_balanced_sampler(train_ds)
        shuffle = False
    loader = DataLoader(train_ds, batch_size=1, shuffle=shuffle,
                        sampler=sampler, collate_fn=_collate_single)

    log_path = os.path.join(save_dir, "train_log_stage2.jsonl")
    log_file = open(log_path, "w", encoding="utf-8")

    best_val_rank_acc = -1.0
    best_val_epoch = -1
    best_val_regret = float("inf")
    best_regret_epoch = -1
    best_loss = float("inf")
    best_loss_epoch = -1
    patience_counter = 0
    records = []
    pair_offset = 0

    for epoch in range(args.stage2_epochs):
        t0 = time.time()
        in_freeze = epoch < args.stage2_freeze_epochs and args.stage2_freeze_epochs > 0

        if epoch == 0:
            if args.stage2_freeze_epochs > 0:
                _set_freeze(True)
                current_lr = args.stage2_lr
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_lr,
                )
                print(f"\n  Frozen: {freeze_modules}")
                print(f"  Training: {train_modules}")
            else:
                current_lr = args.stage2_lr
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_lr,
                )
                print(f"\n  No freeze (freeze_epochs=0). Training all modules.")

        if args.stage2_freeze_epochs > 0 and epoch == args.stage2_freeze_epochs:
            if args.stage2_unfreeze_mode == "all":
                _set_freeze(False)
                active_modules = "all modules"
            elif args.stage2_unfreeze_mode == "decoder":
                _set_trainable_modules(decoder_recalibration_modules)
                active_modules = decoder_recalibration_modules
            else:
                raise ValueError(f"unknown stage2_unfreeze_mode: {args.stage2_unfreeze_mode}")
            current_lr = args.stage2_lr * args.stage2_unfreeze_lr_factor
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=current_lr,
            )
            print(f"\n  Stage2 unfreeze mode={args.stage2_unfreeze_mode}. "
                  f"Training {active_modules}. lr -> {current_lr}")

        record, pair_offset = _run_epoch(
            model, loader, optimizer, device,
            alpha_system=1.0, alpha_station=0.5,
            node_channel_weights=node_channel_weights,
            wait_pos_weight=wait_pos_weight,
            blocked_pos_weight=blocked_pos_weight,
            pairwise_data=train_pairs,
            alpha_rank=args.stage2_alpha_rank,
            margin=args.margin,
            pair_step_offset=pair_offset,
            alpha_long_risk=args.alpha_long_risk,  # Stage 2: enable long-risk
            b3_tail_dims=b3_tail_dims,
            b3_tail_tau=args.b3_system_tail_tau,
            b3_residual_alpha=args.b3_system_residual_alpha,
        )
        elapsed = time.time() - t0
        record["epoch"] = epoch + 1
        record["elapsed_seconds"] = round(elapsed, 1)
        record["frozen"] = in_freeze
        record["lr"] = current_lr

        if record["loss"] < best_loss:
            best_loss = record["loss"]
            best_loss_epoch = epoch + 1

        if (val_ds is not None and len(val_ds) > 0
                and args.stage2_val_dynamics_interval > 0
                and ((epoch + 1) % args.stage2_val_dynamics_interval == 0
                     or epoch == 0
                     or epoch + 1 == args.stage2_freeze_epochs
                     or epoch + 1 == args.stage2_epochs)):
            val_dyn = _eval_val_dynamics(
                model, val_ds, device, args.stage2_horizon,
                node_channel_weights, wait_pos_weight, blocked_pos_weight,
                alpha_long_risk=args.alpha_long_risk if val_has_long_risk else 0.0,
                b3_tail_dims=b3_tail_dims,
                b3_tail_tau=args.b3_system_tail_tau,
                b3_residual_alpha=args.b3_system_residual_alpha,
            )
            record.update(val_dyn)

        msg = (f"  S2 Epoch {epoch+1:3d}/{args.stage2_epochs}  "
               f"loss={record['loss']:.4f}  "
               f"system={record['system']:.4f}  "
               f"rank={record['rank']:.4f}  "
               f"|g|={record['grad_norm']:.2f}")
        if "val_loss" in record:
            msg += f"  dyn_val={record['val_loss']:.4f}"

        # --- Validation ---
        rank_improved = False
        if val_pairs:
            val_on_dev = [_pair_ranking_to_device(p, device) for p in val_pairs]
            val_rank = evaluate_ranking(model, val_on_dev, device=device)
            del val_on_dev
            record["val_rank_accuracy"] = val_rank["pairwise_rank_accuracy"]
            record["val_cost_std"] = val_rank["predicted_cost_std"]

            if val_ds is not None:
                val_regret = evaluate_top1_regret(
                    model, val_ds, device=device, cost_lambdas=cost_lambdas)
                record["val_top1_regret_mean"] = val_regret["top1_regret_mean"]
                record["val_top1_regret_max"] = val_regret["top1_regret_max"]

                # B2 long-risk validation (only if labels are present in samples)
                if val_has_long_risk:
                    lr_metrics = evaluate_long_risk(model, val_ds, device=device)
                    record["val_lr_pairwise_acc"] = lr_metrics["long_risk_pairwise_acc"]
                    record["val_lr_top1_regret"] = lr_metrics["long_risk_top1_regret"]
                    record["val_lr_hit_rate"] = lr_metrics["group_best_risk_hit_rate"]
                    for suffix in ("peak", "cvar", "terminal", "delta", "combo"):
                        record[f"val_lr_pairwise_acc_{suffix}"] = lr_metrics.get(
                            f"long_risk_pairwise_acc_{suffix}", 0.0)
                        record[f"val_lr_top1_regret_{suffix}"] = lr_metrics.get(
                            f"long_risk_top1_regret_{suffix}", 0.0)
                        record[f"val_lr_hit_rate_{suffix}"] = lr_metrics.get(
                            f"group_best_risk_hit_rate_{suffix}", 0.0)

            if val_rank["pairwise_rank_accuracy"] > best_val_rank_acc:
                best_val_rank_acc = val_rank["pairwise_rank_accuracy"]
                best_val_epoch = epoch + 1
                _save_checkpoint(model, save_dir, "best_rank_world_model.pt")
                rank_improved = True

            if "val_top1_regret_mean" in record:
                if record["val_top1_regret_mean"] < best_val_regret:
                    best_val_regret = record["val_top1_regret_mean"]
                    best_regret_epoch = epoch + 1
                    _save_checkpoint(model, save_dir, "best_regret_world_model.pt")

            msg += f"  val_acc={record['val_rank_accuracy']:.4f}"
            if "val_top1_regret_mean" in record:
                msg += f"  regret={record['val_top1_regret_mean']:.4f}"
            if "val_lr_pairwise_acc" in record:
                msg += f"  lr_acc={record['val_lr_pairwise_acc']:.3f}"
                if "val_lr_pairwise_acc_terminal" in record:
                    msg += f"  lr_term={record['val_lr_pairwise_acc_terminal']:.3f}"

        freeze_tag = " [frozen]" if in_freeze else ""
        msg += f"  {elapsed:.0f}s{freeze_tag}"
        print(msg)

        records.append(record)
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()

        # --- Early stopping ---
        if args.early_stopping_patience > 0 and val_pairs:
            if args.early_stopping_monitor == "val_top1_regret_mean":
                monitor_improved = (best_regret_epoch == epoch + 1)
            else:
                monitor_improved = rank_improved
            if monitor_improved:
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"\n  Early stopping at epoch {epoch + 1} "
                      f"(no {args.early_stopping_monitor} improvement for "
                      f"{patience_counter} epochs)")
                break

    log_file.close()

    # --- Save final checkpoint ---
    ckpt_path = _save_checkpoint(model, save_dir, "world_model.pt")
    print(f"\n  Stage 2 done.")
    if best_val_epoch > 0:
        print(f"  Best rank checkpoint: epoch {best_val_epoch}, "
              f"val_rank_accuracy={best_val_rank_acc:.4f}")
    if best_regret_epoch > 0:
        print(f"  Best regret checkpoint: epoch {best_regret_epoch}, "
              f"val_top1_regret_mean={best_val_regret:.4f}")
    print(f"  Final checkpoint: {ckpt_path}")

    # --- Final eval ---
    val_eval = {}
    val_regret_eval = {}
    if val_pairs:
        val_on_dev = [_pair_ranking_to_device(p, device) for p in val_pairs]
        val_eval = evaluate_ranking(model, val_on_dev, device=device)
    if val_ds is not None and len(val_ds) > 0:
        val_regret_eval = evaluate_top1_regret(
            model, val_ds, device=device, cost_lambdas=cost_lambdas)

    val_lr_eval = {}
    if val_ds is not None and len(val_ds) > 0 and val_has_long_risk:
        val_lr_eval = evaluate_long_risk(model, val_ds, device=device)

    return {
        "epochs": args.stage2_epochs,
        "epochs_actual": len(records),
        "horizon": args.stage2_horizon,
        "lr": args.stage2_lr,
        "unfreeze_lr_factor": args.stage2_unfreeze_lr_factor,
        "alpha_rank": args.stage2_alpha_rank,
        "freeze_epochs": args.stage2_freeze_epochs,
        "unfreeze_mode": args.stage2_unfreeze_mode,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_monitor": args.early_stopping_monitor,
        "best_epoch_by_rank": best_val_epoch,
        "best_val_rank_accuracy": round(best_val_rank_acc, 4) if best_val_rank_acc >= 0 else None,
        "best_epoch_by_regret": best_regret_epoch,
        "best_val_regret": round(best_val_regret, 4) if best_val_regret < float("inf") else None,
        "best_epoch_by_loss": best_loss_epoch,
        "best_loss": round(best_loss, 6),
        "final_losses": records[-1] if records else {},
        "val_ranking_eval": val_eval,
        "val_regret_eval": val_regret_eval,
        "val_long_risk_eval": val_lr_eval,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V6 two-stage joint world-model and long-risk training"
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--splits", type=str, default=None,
                        help="Path to splits.json from fuse_and_split")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="artifacts/generated/checkpoints/world_model_v6",
    )
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--stage1-epochs", type=int, default=15)
    parser.add_argument("--stage1-horizon", type=int, default=3)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)

    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--stage2-horizon", type=int, default=10)
    parser.add_argument("--stage2-lr", type=float, default=3e-4)
    parser.add_argument("--stage2-alpha-rank", type=float, default=0.1)
    parser.add_argument("--stage2-freeze-epochs", type=int, default=5)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--stage2-unfreeze-lr-factor", type=float, default=0.3,
                        help="LR multiplier after unfreezing (default: 0.3)")
    parser.add_argument("--stage2-unfreeze-mode", type=str, default="all",
                        choices=["all", "decoder"],
                        help=("Stage 2 post-freeze training mode. 'all' keeps "
                              "the existing joint fine-tune behavior. 'decoder' "
                              "freezes encoder/action/transition and trains "
                              "decoders + long_risk_head."))
    parser.add_argument("--stage2-val-dynamics-interval", type=int, default=1,
                        help=("Evaluate Stage 2 validation dynamics every N epochs "
                              "(0 disables; default: 1)"))
    parser.add_argument("--early-stopping-patience", type=int, default=8,
                        help="Stop if monitored metric no improve for N epochs (0=disable)")
    parser.add_argument("--early-stopping-monitor", type=str, default="val_rank_accuracy",
                        choices=["val_rank_accuracy", "val_top1_regret_mean"],
                        help="Metric to monitor for early stopping")

    parser.add_argument("--max-train-pairs", type=int, default=2000)
    parser.add_argument("--max-val-pairs", type=int, default=500)

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--balanced-sampler", action="store_true",
                        help="Balance training samples by source_run_id")
    parser.add_argument("--pair-balance-by", type=str, default=None,
                        choices=["source_run_id", "source_load_level"],
                        help="Balance pairwise pairs by this field")
    parser.add_argument("--train-cost-head", action="store_true",
                        help="Allow cost_head lambdas to be trainable")
    parser.add_argument("--congestion-cost", action="store_true",
                        help="Use congestion-only lambdas (ch5=0) for ranking/regret")

    parser.add_argument("--skip-stage1", action="store_true",
                        help="Skip Stage 1, load from --stage1-checkpoint directly")
    parser.add_argument("--stage1-checkpoint", type=str, default=None,
                        help="Path to existing stage1 checkpoint for --skip-stage1")

    parser.add_argument("--max-samples", type=int, default=0)

    parser.add_argument("--long-risk-path", type=str, default=None,
                        help=("Optional path to long_risk_labels.pt to merge at load time. "
                              "If omitted, embedded long_risk_labels in the dataset are used."))
    parser.add_argument("--alpha-long-risk", type=float, default=1.0,
                        help="Weight for long-risk loss (default: 1.0)")
    parser.add_argument("--b3-system-tail-dims", type=str, default="",
                        help="Comma-separated system dims for B3 tail loss (e.g. '4')")
    parser.add_argument("--b3-system-tail-tau", type=float, default=0.90,
                        help="Pinball quantile for B3 tail channels (default: 0.90)")
    parser.add_argument("--b3-system-residual-alpha", type=float, default=0.0,
                        help="Weight for B3 residual aux loss (0=disabled, start at 0.2)")
    args = parser.parse_args()

    # --- Mutual exclusion checks ---
    if args.congestion_cost and args.train_cost_head:
        parser.error("--congestion-cost requires frozen cost_head; "
                     "do not use --train-cost-head")
    if args.skip_stage1:
        if not args.stage1_checkpoint:
            parser.error("--skip-stage1 requires --stage1-checkpoint")
        if not os.path.exists(args.stage1_checkpoint):
            parser.error(f"--stage1-checkpoint not found: "
                         f"{args.stage1_checkpoint}")

    # --- Device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # --- B3 tail dims parsing ---
    b3_tail_dims = [int(d) for d in args.b3_system_tail_dims.split(",") if d.strip()] or None

    print("=" * 60)
    print("  V6 Two-Stage Joint World-Model Training")
    print(f"  Device: {device}")
    print("=" * 60)

    # --- Load data ---
    print(f"\n  Loading data from {args.data} ...")
    full_ds = WorldModelDataset.from_file(args.data,
                                          long_risk_path=args.long_risk_path)
    print(f"  Total samples: {len(full_ds)}")

    if args.max_samples > 0 and len(full_ds) > args.max_samples:
        full_ds = WorldModelDataset(full_ds.samples[:args.max_samples])
        print(f"  Truncated to {len(full_ds)} samples")

    # --- Split ---
    if args.splits:
        print(f"\n  Loading splits from {args.splits} ...")
        train_ds, val_ds, test_ds = _load_splits(args.splits, full_ds.samples)
    else:
        print(f"\n  No --splits provided, falling back to split_by_group() ...")
        train_ds, val_ds, test_ds = full_ds.split_by_group(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.split_seed,
        )

    print(f"  Train: {train_ds.summary()}")
    print(f"  Val:   {val_ds.summary()}")
    print(f"  Test:  {test_ds.summary()}")
    print(f"  Long-risk labels: "
          f"train={_count_long_risk_samples(train_ds)}, "
          f"val={_count_long_risk_samples(val_ds)}, "
          f"test={_count_long_risk_samples(test_ds)}")
    if args.alpha_long_risk > 0 and not _has_long_risk_labels(train_ds):
        raise ValueError(
            "--alpha-long-risk is positive, but the training split contains "
            "no long_risk_labels; provide embedded labels or --long-risk-path"
        )

    action_schema = _infer_action_schema(train_ds.samples)
    print(
        "  Action schema: "
        f"{action_schema['schema_version']} "
        f"(NO_ASSIGN={action_schema['supports_no_assign_candidate']}, "
        f"rows={action_schema['no_assign_samples']})"
    )

    if len(train_ds) == 0:
        print("  ERROR: No training samples. Exiting.")
        return

    # --- Infer model config ---
    s0 = train_ds[0]
    num_stations = _infer_num_stations(s0)
    data_horizon = s0["future_node_labels"].shape[0]

    stage2_horizon = min(args.stage2_horizon, data_horizon)
    stage1_horizon = min(args.stage1_horizon, stage2_horizon)

    print(f"\n  Data horizon: {data_horizon}")
    print(f"  Stage 1 horizon: {stage1_horizon}")
    print(f"  Stage 2 horizon: {stage2_horizon}")
    print(f"  Stations: {num_stations}")

    os.makedirs(args.save_dir, exist_ok=True)

    # --- Compute pos_weight ---
    print(f"\n  Computing pos_weight from train split ...")
    try:
        wait_pw, blocked_pw = _compute_pos_weights_from_dataset(train_ds)
        print(f"  pos_weight: wait={wait_pw:.2f}, blocked={blocked_pw:.2f}")
    except Exception:
        wait_pw, blocked_pw = 5.0, 10.0
        print(f"  pos_weight scan failed, using fallback: wait={wait_pw}, blocked={blocked_pw}")

    node_channel_weights = list(DEFAULT_NODE_CHANNEL_WEIGHTS)

    # ===== STAGE 1 =====
    if args.skip_stage1:
        print(f"\n  Skipping Stage 1, loading from {args.stage1_checkpoint}")
        stage1_summary = {"skipped": True, "checkpoint": args.stage1_checkpoint}
    else:
        model = _build_model_from_sample(s0, stage1_horizon, num_stations)
        model._checkpoint_action_schema = dict(action_schema)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Model parameters: {n_params:,}")

        stage1_summary = run_stage1(
            model, train_ds, val_ds, device, args,
            node_channel_weights, wait_pw, blocked_pw,
            args.save_dir,
            b3_tail_dims=b3_tail_dims,
        )

    # ===== STAGE 2 =====
    model2 = _build_model_from_sample(s0, stage2_horizon, num_stations)
    model2._checkpoint_action_schema = dict(action_schema)
    if args.skip_stage1:
        stage1_ckpt_path = args.stage1_checkpoint
    else:
        stage1_ckpt_path = os.path.join(args.save_dir, "stage1_world_model.pt")
    stage1_ckpt = torch.load(stage1_ckpt_path, weights_only=False)
    model2.load_state_dict(stage1_ckpt["state_dict"], strict=False)
    model2.rollout_horizon = stage2_horizon
    model2 = model2.to(device)
    print(f"\n  Loaded stage1 weights from {stage1_ckpt_path}, "
          f"rollout_horizon={model2.rollout_horizon}")

    if args.congestion_cost:
        lam = torch.tensor(
            CONGESTION_LAMBDAS,
            dtype=model2.cost_head.lambdas.dtype,
            device=model2.cost_head.lambdas.device,
        )
        model2.cost_head.lambdas.data.copy_(lam)
        model2.cost_head.lambdas.requires_grad_(False)

    cost_lambdas = CONGESTION_LAMBDAS if args.congestion_cost else None
    stage2_summary = run_stage2(
        model2, train_ds, val_ds, device, args,
        node_channel_weights, wait_pw, blocked_pw,
        args.save_dir,
        cost_lambdas=cost_lambdas,
        b3_tail_dims=b3_tail_dims,
    )

    # --- Save train_summary.json ---
    data_snapshot = _build_data_quality_snapshot(train_ds)
    model_config = _extract_model_config(model2)

    summary = {
        "stage1": stage1_summary,
        "stage2": stage2_summary,
        "model_config": model_config,
        "data_quality_snapshot": data_snapshot,
        "label_schema_version": "wm_v4_local_pressure",
        "action_schema": dict(action_schema),
    }
    summary_path = os.path.join(args.save_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Summary saved to {summary_path}")
    print(f"\n  Output files:")
    print(f"    {os.path.join(args.save_dir, 'stage1_world_model.pt')}")
    print(f"    {os.path.join(args.save_dir, 'world_model.pt')}")
    if stage2_summary.get("best_epoch_by_rank", -1) > 0:
        print(f"    {os.path.join(args.save_dir, 'best_rank_world_model.pt')}")
    if stage2_summary.get("best_epoch_by_regret", -1) > 0:
        print(f"    {os.path.join(args.save_dir, 'best_regret_world_model.pt')}")
    print(f"    {os.path.join(args.save_dir, 'train_log_stage1.jsonl')}")
    print(f"    {os.path.join(args.save_dir, 'train_log_stage2.jsonl')}")
    print(f"    {summary_path}")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
