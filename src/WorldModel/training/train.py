"""
World Model Training Loop 
==============================
Multi-task training for the ST-GNN RMFS world model.

V4 additions:
  - Per-channel system loss breakdown, grad norm, timing
  - train_log.jsonl + train_summary.json
  - Post-training sanity check (pairwise rank accuracy)

V5 additions:
  - Validation ranking accuracy + top1 regret each epoch
  - Best checkpoint saved by val_rank_accuracy
  - risk_weight parameter for deadlock_risk ablation
"""

import json
import os
import time
from collections import Counter
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from WorldModel.model import RMFSWorldModel
from WorldModel.dataset import WorldModelDataset, stable_candidate_group_key
from WorldModel.core.costs import compute_realized_cost
from WorldModel.core.long_risk_schema import (
    LONG_RISK_OUTPUT_DIM,
    LONG_RISK_OUTPUT_INDEX,
    LONG_RISK_OUTPUT_NAMES,
    LONG_RISK_QUANTILE_COMBO_WEIGHTS,
    decode_long_risk_predictions,
    long_risk_quantile_combo,
)


def _sample_to_device(sample: dict, device: torch.device) -> dict:
    """Move all tensors in a sample dict to the given device (recursive)."""
    out = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _sample_to_device(v, device)
        else:
            out[k] = v
    return out


def _pair_to_device(pair: dict, device: torch.device) -> dict:
    """Move a pairwise sample (with nested sample_i/sample_j) to device."""
    return {
        "sample_i": _sample_to_device(pair["sample_i"], device),
        "sample_j": _sample_to_device(pair["sample_j"], device),
    }


SYSTEM_LABEL_NAMES = [
    "total_wait_time",
    "average_excess_delay",
    "station_queue_delta",
    "station_load_imbalance",
    "bottleneck_CVaR",
    "completed_orders_delta",
    "deadlock_or_severe_congestion_risk",
]


def _collate_single(batch):
    return batch[0]


NODE_CHANNEL_NAMES = [
    "node_self_occupancy", "node_local_density", "node_local_wait_pressure",
    "node_local_blocked_pressure", "node_reservation_pressure", "node_congestion_score",
]

DEFAULT_NODE_CHANNEL_WEIGHTS = [1.0, 1.0, 2.0, 2.0, 1.0, 0.3]


# ---------------------------------------------------------------------------
# Phase B2/B3 loss utilities
# ---------------------------------------------------------------------------

def pinball_loss(pred: torch.Tensor, target: torch.Tensor,
                 tau: float = 0.90) -> torch.Tensor:
    """Quantile (pinball) loss for tail-sensitive prediction."""
    delta = target - pred
    return torch.where(delta >= 0, tau * delta, (tau - 1.0) * delta).mean()


def focal_bce_loss(logit: torch.Tensor, target: torch.Tensor,
                   gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    """Focal BCE for rare event detection."""
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    p_t = torch.sigmoid(logit) * target + (1 - torch.sigmoid(logit)) * (1 - target)
    focal_weight = alpha * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def long_risk_loss(pred: torch.Tensor, labels: dict) -> torch.Tensor:
    """Compute loss for LongRiskHead predictions.

    Parameters
    ----------
    pred : (6,) — [peak_q90, peak_q95, cvar_q90, terminal_q90,
                    delta_group_q90, event_logit]
    labels : dict with keys risk_peak, risk_cvar,
             risk_terminal, risk_delta_group, risk_event
    """
    if pred.shape[-1] != LONG_RISK_OUTPUT_DIM:
        raise ValueError(
            "LongRiskHead training contract mismatch: expected "
            f"{LONG_RISK_OUTPUT_DIM} channels {LONG_RISK_OUTPUT_NAMES}, "
            f"got shape {tuple(pred.shape)}"
        )
    index = LONG_RISK_OUTPUT_INDEX
    loss = (
        pinball_loss(pred[index["peak_q90"]], labels['risk_peak'], tau=0.90)
        + pinball_loss(pred[index["peak_q95"]], labels['risk_peak'], tau=0.95)
        + pinball_loss(pred[index["cvar_q90"]], labels['risk_cvar'], tau=0.90)
        + pinball_loss(
            pred[index["terminal_q90"]], labels['risk_terminal'], tau=0.90
        )
        + pinball_loss(
            pred[index["delta_group_q90"]],
            labels['risk_delta_group'],
            tau=0.90,
        )
        + focal_bce_loss(
            pred[index["event_logit"]], labels['risk_event'], gamma=2.0
        )
        + F.relu(
            pred[index["peak_q90"]] - pred[index["peak_q95"]]
        ).pow(2)  # monotonic: q95 >= q90
    )
    return loss


def _compute_pos_weights_from_dataset(
    dataset,
    wait_ch: int = 2,
    blocked_ch: int = 3,
    max_wait_pw: float = 20.0,
    max_blocked_pw: float = 50.0,
):
    """Scan train split to compute fixed pos_weight for BCE channels.

    Only uses samples where future_mask > 0.5 (valid rollout steps).
    Returns (wait_pos_weight, blocked_pos_weight).
    """
    wait_pos = 0.0
    wait_neg = 0.0
    blocked_pos = 0.0
    blocked_neg = 0.0

    for i in range(len(dataset)):
        sample = dataset[i]
        node_labels = sample["future_node_labels"]
        mask = sample.get("future_mask")
        K = node_labels.shape[0]
        for k in range(K):
            if mask is not None and mask[k].item() < 0.5:
                continue
            w_target = node_labels[k, :, wait_ch].clamp(0, 1).flatten()
            b_target = node_labels[k, :, blocked_ch].clamp(0, 1).flatten()
            wait_pos += w_target.sum().item()
            wait_neg += (1.0 - w_target).sum().item()
            blocked_pos += b_target.sum().item()
            blocked_neg += (1.0 - b_target).sum().item()

    eps = 1e-6
    wait_pw = min(max_wait_pw, max(1.0, wait_neg / (wait_pos + eps)))
    blocked_pw = min(max_blocked_pw, max(1.0, blocked_neg / (blocked_pos + eps)))
    return wait_pw, blocked_pw


def compute_loss(
    model: RMFSWorldModel,
    sample: dict,
    alpha_system: float = 1.0,
    alpha_station: float = 0.5,
    node_channel_weights: Optional[List[float]] = None,
    wait_pos_weight: float = 5.0,
    blocked_pos_weight: float = 10.0,
    alpha_long_risk: float = 1.0,
    b3_tail_dims: Optional[List[int]] = None,
    b3_tail_tau: float = 0.90,
    b3_residual_alpha: float = 0.0,
) -> dict:
    """Compute multi-task loss for a single sample."""
    if node_channel_weights is None:
        node_channel_weights = DEFAULT_NODE_CHANNEL_WEIGHTS

    station_nids = sample.get("station_node_ids")
    if station_nids is not None:
        station_nids = station_nids.tolist()

    node_preds, system_preds, station_preds, uncertainties, z_0, z_K = model(
        node_history=sample["node_history"],
        edge_index=sample["edge_index"],
        edge_features=sample["edge_features"],
        demand_context=sample["demand_context"],
        action_node=sample["action_node"],
        action_global=sample["action_global"],
        station_node_ids=station_nids,
    )

    future_node = sample["future_node_labels"]
    future_sys = sample["future_system_labels"]
    future_mask = sample.get("future_mask")

    K = min(len(node_preds), future_node.shape[0])

    dev = system_preds.device
    wait_pw = torch.tensor([wait_pos_weight], device=dev)
    blocked_pw = torch.tensor([blocked_pos_weight], device=dev)

    channel_losses = [0.0] * 6
    valid_steps = 0
    for k in range(K):
        w = future_mask[k].item() if future_mask is not None else 1.0
        if w < 0.5:
            continue
        valid_steps += 1
        pred = node_preds[k]
        target = future_node[k]

        channel_losses[0] += F.binary_cross_entropy_with_logits(
            pred[:, 0], target[:, 0].clamp(0, 1))
        channel_losses[1] += F.mse_loss(pred[:, 1], target[:, 1])
        channel_losses[2] += F.binary_cross_entropy_with_logits(
            pred[:, 2], target[:, 2].clamp(0, 1), pos_weight=wait_pw)
        channel_losses[3] += F.binary_cross_entropy_with_logits(
            pred[:, 3], target[:, 3].clamp(0, 1), pos_weight=blocked_pw)
        channel_losses[4] += F.mse_loss(pred[:, 4], target[:, 4])
        channel_losses[5] += F.mse_loss(pred[:, 5], target[:, 5])

    if valid_steps > 0:
        channel_losses = [cl / valid_steps for cl in channel_losses]

    loss_node = sum(
        node_channel_weights[i] * channel_losses[i] for i in range(6)
    )

    if future_mask is not None:
        mask_2d = future_mask[:K].unsqueeze(-1).expand_as(system_preds[:K])
        # 对于宏观的 7 个系统指标（总延迟、系统吞吐量、死锁风险等），使用 F.huber_loss (Huber 损失)，它在误差较小时表现为均方误差，在误差较大时表现为绝对误差。
        # 这有助于在训练过程中对异常值（如极端的系统状态）具有一定的鲁棒性，同时仍然鼓励模型尽可能准确地预测系统指标。
        loss_system = F.huber_loss(
            system_preds[:K] * mask_2d, future_sys[:K] * mask_2d,
        )
    else:
        loss_system = F.huber_loss(system_preds[:K], future_sys[:K])

    system_channel_losses = {}
    with torch.no_grad():
        # 单独把 7 个系统指标（等待时间、CVaR 等）的误差分别算了一遍，并打包成字典返回。
        for dim in range(min(system_preds.shape[-1], future_sys.shape[-1])):
            sp = system_preds[:K, dim]
            ft = future_sys[:K, dim]
            if future_mask is not None:
                m = future_mask[:K]
                ch_loss = F.huber_loss(sp * m, ft * m).item()
            else:
                ch_loss = F.huber_loss(sp, ft).item()
            name = SYSTEM_LABEL_NAMES[dim] if dim < len(SYSTEM_LABEL_NAMES) else f"dim{dim}"
            system_channel_losses[name] = ch_loss

    loss_station = torch.tensor(0.0, device=system_preds.device)
    future_sta = sample.get("future_station_labels")
    if future_sta is not None:
        K_sta = min(station_preds.shape[0], future_sta.shape[0])
        if future_mask is not None:
            mask_sta = future_mask[:K_sta].unsqueeze(-1).unsqueeze(-1)
            mask_sta = mask_sta.expand_as(station_preds[:K_sta])
            loss_station = F.mse_loss(
                station_preds[:K_sta] * mask_sta,
                future_sta[:K_sta] * mask_sta,
            )
        else:
            loss_station = F.mse_loss(station_preds[:K_sta], future_sta[:K_sta])

    # --- B3: tail-sensitive residual auxiliary pinball loss ---
    loss_b3_residual = torch.tensor(0.0, device=system_preds.device)
    baseline = sample.get("future_system_group_baseline")
    if b3_tail_dims and b3_residual_alpha > 0 and baseline is not None:
        for dim in b3_tail_dims:
            if dim < 0 or dim >= system_preds.shape[-1] or dim >= future_sys.shape[-1]:
                continue
            Kb = min(K, baseline.shape[0])
            if Kb <= 0:
                continue
            pred_d = system_preds[:Kb, dim]
            target_d = future_sys[:Kb, dim]
            base_d = baseline[:Kb, dim].to(device=pred_d.device, dtype=pred_d.dtype)
            pred_res = pred_d - base_d
            target_res = target_d - base_d
            delta = target_res - pred_res
            pb = torch.where(
                delta >= 0,
                b3_tail_tau * delta,
                (b3_tail_tau - 1.0) * delta,
            )
            if future_mask is not None:
                m = future_mask[:Kb]
                loss_dim = (pb * m).sum() / m.sum().clamp_min(1.0)
            else:
                loss_dim = pb.mean()
            loss_b3_residual = loss_b3_residual + loss_dim

    # --- Long-risk head loss (Phase B2, optional) ---
    loss_long_risk = torch.tensor(0.0, device=system_preds.device)
    lr_labels = sample.get("long_risk_labels")
    if lr_labels is not None and alpha_long_risk > 0:
        lr_preds = model.long_risk_head(z_K, z_0)
        loss_long_risk = long_risk_loss(lr_preds, lr_labels)

    total = (loss_node
             + alpha_system * (loss_system + b3_residual_alpha * loss_b3_residual)
             + alpha_station * loss_station
             + alpha_long_risk * loss_long_risk)

    node_channel_loss_dict = {}
    for i, name in enumerate(NODE_CHANNEL_NAMES):
        val = channel_losses[i]
        node_channel_loss_dict[name] = val.item() if isinstance(val, torch.Tensor) else val

    return {
        "total": total,
        **node_channel_loss_dict,
        "system": loss_system.item(),
        "station": loss_station.item(),
        "long_risk": loss_long_risk.item(),
        "b3_residual": loss_b3_residual.item(),
        "system_channels": system_channel_losses,
    }


def compute_ranking_loss(
    model: RMFSWorldModel,
    pair: dict,
    margin: float = 1.0,
) -> torch.Tensor:
    """ Value Alignment -- Pairwise margin ranking loss: max(0, margin + J(better) - J(worse)).  
        J(x) 是 cost, cost越小越好。如果模型判断好动作 cost_better 比坏动作 cost_worse 还大（即模型认为更差的方案成本更低），就会产生一个正的损失，推动模型调整参数来纠正这个错误。
        margin 是安全阈值, 起码对于好动作和坏动作的 cost 差距应该有多大，才不会被认为是模型的判断错误。通过调整 margin 的大小，可以控制训练过程中对排名错误的容忍度。
    """
    sample_better = pair["sample_i"]
    sample_worse = pair["sample_j"]

    snb = sample_better.get("station_node_ids")
    if snb is not None:
        snb = snb.tolist()
    snw = sample_worse.get("station_node_ids")
    if snw is not None:
        snw = snw.tolist()

    z_b, e_b, ea_b = model.encode_state(
        sample_better["node_history"], sample_better["edge_index"],
        sample_better["edge_features"], sample_better["demand_context"],
    )
    cost_better = model.predict_cost(
        z_b, e_b, ea_b,
        sample_better["action_node"], sample_better["action_global"],
        sample_better["edge_index"], snb,
    )

    z_w, e_w, ea_w = model.encode_state(
        sample_worse["node_history"], sample_worse["edge_index"],
        sample_worse["edge_features"], sample_worse["demand_context"],
    )
    cost_worse = model.predict_cost(
        z_w, e_w, ea_w,
        sample_worse["action_node"], sample_worse["action_global"],
        sample_worse["edge_index"], snw,
    )

    return F.relu(margin + cost_better - cost_worse)


def evaluate_ranking(
    model: RMFSWorldModel,
    pairwise_data: List[dict],
    device: Optional[torch.device] = None,
) -> dict:
    """
    Evaluate pairwise ranking accuracy and cost statistics. 在验证集 (Validation Set) 或测试集上, 评估模型的 pairwise ranking accuracy 和预测成本的统计信息。
    这个函数会遍历所有的成对样本, 使用模型预测每个样本的成本, 然后比较模型的预测与实际的排名 (哪个样本更好) 是否一致。
    最终返回一个字典, 包含 pairwise_rank_accuracy (模型正确判断更好样本的比例), num_eval_pairs (评估的成对样本数量), predicted_cost_mean (模型预测成本的平均值), predicted_cost_std (模型预测成本的标准差)。
    """
    model.eval()
    correct = 0
    total = 0
    predicted_costs = []

    with torch.no_grad():
        for pair in pairwise_data:
            s_better = pair["sample_i"]
            s_worse = pair["sample_j"]

            snb = s_better.get("station_node_ids")
            if snb is not None:
                snb = snb.tolist()
            snw = s_worse.get("station_node_ids")
            if snw is not None:
                snw = snw.tolist()

            z_b, e_b, ea_b = model.encode_state(
                s_better["node_history"], s_better["edge_index"],
                s_better["edge_features"], s_better["demand_context"],
            )
            cost_b = model.predict_cost(
                z_b, e_b, ea_b,
                s_better["action_node"], s_better["action_global"],
                s_better["edge_index"], snb,
            ).item()

            z_w, e_w, ea_w = model.encode_state(
                s_worse["node_history"], s_worse["edge_index"],
                s_worse["edge_features"], s_worse["demand_context"],
            )
            cost_w = model.predict_cost(
                z_w, e_w, ea_w,
                s_worse["action_node"], s_worse["action_global"],
                s_worse["edge_index"], snw,
            ).item()

            if cost_b < cost_w:
                correct += 1
            total += 1
            predicted_costs.extend([cost_b, cost_w])

    accuracy = correct / max(total, 1)
    pred_std = 0.0
    if len(predicted_costs) >= 2:
        mean_c = sum(predicted_costs) / len(predicted_costs)
        pred_std = (sum((c - mean_c) ** 2 for c in predicted_costs) / len(predicted_costs)) ** 0.5

    return {
        "pairwise_rank_accuracy": round(accuracy, 4),
        "num_eval_pairs": total,
        "predicted_cost_mean": round(sum(predicted_costs) / max(len(predicted_costs), 1), 4),
        "predicted_cost_std": round(pred_std, 4),
    }


def evaluate_top1_regret(
    model: RMFSWorldModel,
    dataset: "WorldModelDataset",
    device: Optional[torch.device] = None,
    cost_lambdas: Optional[list] = None,
) -> dict:
    """ 在测试集中
        Top-1 regret: for each candidate group, how much worse is the model's pick vs the oracle (lowest realized_cost)?
        Top-1 遗憾值：针对每个候选组，模型的选择相较于实际成本最低的最优选择的性能劣化程度。

        regret_i = realized_cost(model_pick) - realized_cost(oracle_pick) >= 0
    """
    model.eval()
    groups = dataset._group_samples()

    regrets = []
    with torch.no_grad():
        for members in groups.values():
            if len(members) < 2:
                continue

            costs_pred = []
            costs_real = []
            for s in members:
                if cost_lambdas is not None:
                    fsl = s.get("future_system_labels")
                    if fsl is not None:
                        real_cost = compute_realized_cost(
                            fsl.detach().cpu(), lambdas=cost_lambdas)
                    else:
                        real_cost = s["realized_cost"]
                else:
                    real_cost = s["realized_cost"]

                if device is not None:
                    s = _sample_to_device(s, device)
                sn = s.get("station_node_ids")
                if sn is not None:
                    sn = sn.tolist()
                z, e, ea = model.encode_state(
                    s["node_history"], s["edge_index"],
                    s["edge_features"], s["demand_context"],
                )
                cost = model.predict_cost(
                    z, e, ea,
                    s["action_node"], s["action_global"],
                    s["edge_index"], sn,
                ).item()
                costs_pred.append(cost)
                costs_real.append(real_cost)
            
            # 模型挑出预测代价最小的动作索引
            model_pick = min(range(len(costs_pred)), key=lambda i: costs_pred[i])
            # 实际代价最小的动作索引（oracle_pick）是通过比较 realized_cost 来确定的，代表了在这个候选组中实际表现最好的选择。
            oracle_pick = min(range(len(costs_real)), key=lambda i: costs_real[i])
            regrets.append(costs_real[model_pick] - costs_real[oracle_pick])

    # top1_regret_mean (平均遗憾值)：证明了模型在 99% 的日常调度中，决策质量非常逼近"最优解（Oracle）"，保障了系统的基础吞吐量。
    # top1_regret_max (最大遗憾值)：揭示了在极端情况下，模型的决策可能会比最优解差多少，这对于理解模型在边缘案例中的风险非常重要。如果 top1_regret_max 保持在一个很低的水平，这意味着模型拥有极强的"避坑/兜底"能力，它或许偶尔会选错，但绝不会做出导致系统崩溃的灾难性决策。
    if not regrets:
        return {"top1_regret_mean": 0.0, "top1_regret_max": 0.0, "num_groups": 0}

    return {
        "top1_regret_mean": round(sum(regrets) / len(regrets), 4),
        "top1_regret_max": round(max(regrets), 4),
        "num_groups": len(regrets),
    }


def evaluate_long_risk(
    model: RMFSWorldModel,
    dataset: "WorldModelDataset",
    device: Optional[torch.device] = None,
) -> dict:
    """Evaluate LongRiskHead on dataset samples that have long_risk_labels.

    Metrics:
      - long_risk_pairwise_acc: primary combo-risk pairwise ranking accuracy
      - long_risk_top1_regret: primary combo-risk top-1 regret
      - group_best_risk_hit_rate: primary combo-risk oracle hit rate
      - *_peak/cvar/terminal/delta/combo: per-target ranking metrics
      - spearman_per_dim: Spearman correlation per output dimension
    """
    model.eval()
    groups = dataset._group_samples()

    # Filter to groups where at least 2 members have long_risk_labels
    eligible_groups = {}
    for gid, members in groups.items():
        lr_members = [s for s in members if "long_risk_labels" in s]
        if len(lr_members) >= 2:
            eligible_groups[gid] = lr_members

    if not eligible_groups:
        return {
            "long_risk_pairwise_acc": 0.0,
            "long_risk_top1_regret": 0.0,
            "group_best_risk_hit_rate": 0.0,
            "num_eligible_groups": 0,
        }

    target_names = ("peak", "cvar", "terminal", "delta_group", "combo")
    combo_weights = {
        "peak": LONG_RISK_QUANTILE_COMBO_WEIGHTS["peak_q95"],
        "cvar": LONG_RISK_QUANTILE_COMBO_WEIGHTS["cvar_q90"],
        "terminal": LONG_RISK_QUANTILE_COMBO_WEIGHTS["terminal_q90"],
    }
    pairwise_correct = {name: 0 for name in target_names}
    pairwise_total = {name: 0 for name in target_names}
    regrets = {name: [] for name in target_names}
    hits = {name: 0 for name in target_names}

    # Per-dim collection for Spearman (each dim against its proper target)
    pred_all = {d: [] for d in range(6)}
    true_targets = {
        "peak": [],       # dims 0,1
        "cvar": [],        # dim 2
        "terminal": [],    # dim 3
        "delta_group": [], # dim 4
        "event": [],       # dim 5
    }

    with torch.no_grad():
        for gid, members in eligible_groups.items():
            # Get predictions and true labels for each candidate
            pred_peaks = []  # head's peak_q95 (dim 1) — used for ranking
            true_peaks = []  # true risk_peak
            pred_scores = {name: [] for name in target_names}
            true_scores = {name: [] for name in target_names}

            for s in members:
                if device is not None:
                    s = _sample_to_device(s, device)
                sn = s.get("station_node_ids")
                if sn is not None:
                    sn = sn.tolist()
                z, e, ea = model.encode_state(
                    s["node_history"], s["edge_index"],
                    s["edge_features"], s["demand_context"],
                )
                _, _, _, _, z_0, z_K = model.rollout(
                    z, e, ea, s["action_node"], s["action_global"],
                    s["edge_index"], sn,
                )
                lr_pred = model.long_risk_head(z_K, z_0)
                lr_components = decode_long_risk_predictions(
                    lr_pred,
                    scalar=lambda value: float(value.item()),
                )

                lr_lab = s["long_risk_labels"]
                pred_peak = lr_components["long_risk_peak_q95"]
                pred_cvar = lr_components["long_risk_cvar_q90"]
                pred_terminal = lr_components["long_risk_terminal_q90"]
                pred_delta = lr_components["long_risk_delta_group_q90"]
                true_peak = lr_lab["risk_peak"].item()
                true_cvar = lr_lab["risk_cvar"].item()
                true_terminal = lr_lab["risk_terminal"].item()
                true_delta = lr_lab["risk_delta_group"].item()
                pred_combo = lr_components["long_risk_quantile_combo"]
                true_combo = long_risk_quantile_combo(
                    true_peak,
                    true_cvar,
                    true_terminal,
                )

                pred_peaks.append(pred_peak)
                true_peaks.append(true_peak)
                pred_scores["peak"].append(pred_peak)
                pred_scores["cvar"].append(pred_cvar)
                pred_scores["terminal"].append(pred_terminal)
                pred_scores["delta_group"].append(pred_delta)
                pred_scores["combo"].append(pred_combo)
                true_scores["peak"].append(true_peak)
                true_scores["cvar"].append(true_cvar)
                true_scores["terminal"].append(true_terminal)
                true_scores["delta_group"].append(true_delta)
                true_scores["combo"].append(true_combo)

                for d in range(6):
                    pred_all[d].append(lr_pred[d].item())
                true_targets["peak"].append(true_peak)
                true_targets["cvar"].append(true_cvar)
                true_targets["terminal"].append(true_terminal)
                true_targets["delta_group"].append(true_delta)
                true_targets["event"].append(lr_lab["risk_event"].item())

            for name in target_names:
                preds = pred_scores[name]
                trues = true_scores[name]
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        if abs(trues[i] - trues[j]) < 1e-6:
                            continue
                        pairwise_total[name] += 1
                        # Concordant: same ordering direction.
                        if (preds[i] - preds[j]) * (trues[i] - trues[j]) > 0:
                            pairwise_correct[name] += 1

                model_pick = min(range(len(preds)), key=lambda i: preds[i])
                oracle_pick = min(range(len(trues)), key=lambda i: trues[i])
                regrets[name].append(trues[model_pick] - trues[oracle_pick])
                if model_pick == oracle_pick:
                    hits[name] += 1

    n_groups = len(eligible_groups)
    result = {
        # Primary B2 validation follows the aggregate terminal-risk score.
        "long_risk_pairwise_acc": round(
            pairwise_correct["combo"] / max(pairwise_total["combo"], 1), 4),
        "long_risk_top1_regret": round(
            sum(regrets["combo"]) / max(len(regrets["combo"]), 1), 4),
        "group_best_risk_hit_rate": round(hits["combo"] / max(n_groups, 1), 4),
        "num_eligible_groups": n_groups,
        "pairwise_total": pairwise_total["combo"],
        "long_risk_combo_weights": combo_weights,
    }
    for name in target_names:
        suffix = "delta" if name == "delta_group" else name
        result[f"long_risk_pairwise_acc_{suffix}"] = round(
            pairwise_correct[name] / max(pairwise_total[name], 1), 4)
        result[f"long_risk_top1_regret_{suffix}"] = round(
            sum(regrets[name]) / max(len(regrets[name]), 1), 4)
        result[f"group_best_risk_hit_rate_{suffix}"] = round(
            hits[name] / max(n_groups, 1), 4)
        result[f"pairwise_total_{suffix}"] = pairwise_total[name]

    # Spearman per dim (each against its proper true target)
    n_samples = len(true_targets["peak"])
    if n_samples >= 5:
        from scipy import stats as sp_stats
        # dims 0,1 (peak_q90, peak_q95) → true risk_peak
        # dim 2 (cvar_q90) → true risk_cvar
        # dim 3 (terminal_q90) → true risk_terminal
        # dim 4 (delta_group_q90) → true risk_delta_group
        # dim 5 (event_logit) → AUC against binary risk_event
        dim_target_map = [
            ("peak_q90",       "peak"),
            ("peak_q95",       "peak"),
            ("cvar_q90",       "cvar"),
            ("terminal_q90",   "terminal"),
            ("delta_group_q90", "delta_group"),
        ]
        for dim_name, target_key in dim_target_map:
            d = LONG_RISK_OUTPUT_INDEX[dim_name]
            rho, _ = sp_stats.spearmanr(pred_all[d], true_targets[target_key])
            result[f"spearman_{dim_name}"] = round(float(rho), 4)

        # Event logit: AUC if both classes present, else Spearman fallback
        event_true = true_targets["event"]
        event_pred = pred_all[LONG_RISK_OUTPUT_INDEX["event_logit"]]
        if len(set(event_true)) >= 2:
            try:
                from sklearn.metrics import roc_auc_score
                result["event_auc"] = round(
                    float(roc_auc_score(event_true, event_pred)), 4)
            except ImportError:
                rho, _ = sp_stats.spearmanr(event_pred, event_true)
                result["spearman_event_logit"] = round(float(rho), 4)
        else:
            rho, _ = sp_stats.spearmanr(event_pred, event_true)
            result["spearman_event_logit"] = round(float(rho), 4)

    return result


def _extract_model_config(model: RMFSWorldModel) -> dict:
    """Extract model architecture config from a trained model."""
    cfg = {}
    if hasattr(model, 'state_encoder'):
        enc = model.state_encoder
        cfg["node_feat_dim"] = enc.spatial.layers[0].lin_self.in_features
        cfg["edge_feat_dim"] = enc.edge_encoder.in_features
        cfg["hidden_dim"] = enc.temporal.gru.hidden_size
    if hasattr(model, 'demand_encoder'):
        cfg["demand_dim"] = model.demand_encoder.net[0].in_features
    if hasattr(model, 'action_encoder'):
        cfg["action_node_dim"] = model.action_encoder.node_net[0].in_features
        cfg["action_global_dim"] = model.action_encoder.global_net[0].in_features
    if hasattr(model, 'station_decoder'):
        cfg["num_stations"] = model.station_decoder.num_stations
    if hasattr(model, 'rollout_horizon'):
        cfg["rollout_horizon"] = model.rollout_horizon
    return cfg


def _infer_action_schema(samples) -> dict:
    """Infer whether training data fully covers native NO_ASSIGN.

    Online inference is allowed to expose the all-zero no-op only when every
    training candidate group contains an explicit no-op row and those rows
    actually use the frozen zero-tensor encoding.  Merely loading a newer
    trainer must never make an old robot-only checkpoint no-op capable.
    """
    from WorldModel.candidate_generator import (
        NO_ASSIGN_ACTION_SCHEMA_VERSION,
        NO_ASSIGN_ACTION_TYPE,
        NO_ASSIGN_ENCODING,
    )

    rows = list(samples or ())
    groups = {}
    no_assign_rows = []
    assign_rows = []
    for sample in rows:
        # Candidate group counters restart in every simulation process.
        # Preserve run/seed provenance so fused Phase-C files do not make ten
        # correct one-no-op groups look like one malformed ten-no-op group.
        group_id = stable_candidate_group_key(sample)
        groups.setdefault(group_id, []).append(sample)
        if sample.get("action_type") == NO_ASSIGN_ACTION_TYPE:
            no_assign_rows.append(sample)
        else:
            assign_rows.append(sample)

    complete_group_coverage = bool(groups) and all(
        sum(
            member.get("action_type") == NO_ASSIGN_ACTION_TYPE
            for member in members
        ) == 1
        and any(
            member.get("action_type") != NO_ASSIGN_ACTION_TYPE
            for member in members
        )
        for members in groups.values()
    )

    def zero_encoded(sample: dict) -> bool:
        for key in ("action_node", "action_global", "action_edge"):
            value = sample.get(key)
            if value is None:
                return False
            tensor = torch.as_tensor(value)
            if bool(torch.count_nonzero(tensor).item()):
                return False
        return sample.get("action_encoding") == NO_ASSIGN_ENCODING

    zero_encoding_verified = bool(no_assign_rows) and all(
        zero_encoded(sample) for sample in no_assign_rows
    )
    supported = bool(
        no_assign_rows
        and assign_rows
        and complete_group_coverage
        and zero_encoding_verified
    )
    return {
        "schema_version": (
            NO_ASSIGN_ACTION_SCHEMA_VERSION
            if supported else "wm_assign_robot_only_action_v1"
        ),
        "supports_no_assign_candidate": supported,
        "no_assign_encoding": (
            NO_ASSIGN_ENCODING if supported else None
        ),
        "complete_group_coverage": complete_group_coverage,
        "zero_encoding_verified": zero_encoding_verified,
        "training_samples": len(rows),
        "assign_robot_samples": len(assign_rows),
        "no_assign_samples": len(no_assign_rows),
        "candidate_groups": len(groups),
    }


def _save_checkpoint(
    model: RMFSWorldModel,
    save_dir: str,
    save_name: str,
) -> str:
    """Save a model checkpoint and return the path."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, save_name)
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    payload = {
        "state_dict": cpu_state,
        "model_config": _extract_model_config(model),
        "label_schema_version": "wm_v4_local_pressure",
        "action_schema": dict(getattr(
            model,
            "_checkpoint_action_schema",
            {
                "schema_version": "wm_assign_robot_only_action_v1",
                "supports_no_assign_candidate": False,
            },
        )),
    }
    torch.save(payload, path)
    return path


def _build_data_quality_snapshot(dataset: WorldModelDataset) -> dict:
    """Build a compact data quality snapshot for the training log.  对数据回访池进行数据质量分析，构建一个紧凑的数据质量快照，用于训练日志记录。"""
    samples = dataset.samples
    groups = {}
    for s in samples:
        gid = stable_candidate_group_key(s)
        groups.setdefault(gid, []).append(s)

    sizes = [len(m) for m in groups.values()]
    size_counter = dict(Counter(sizes))
    mean_size = sum(sizes) / max(len(sizes), 1)

    pair_count = 0
    no_assign_non_tie_pairs = 0
    no_assign_better_pairs = 0
    no_assign_worse_pairs = 0
    no_assign_tie_pairs = 0
    for members in groups.values():
        costs = [m["realized_cost"] for m in members]
        for i in range(len(costs)):
            for j in range(i + 1, len(costs)):
                difference = float(costs[i]) - float(costs[j])
                is_no_assign_pair = (
                    (members[i].get("action_type") == "no_assign")
                    != (members[j].get("action_type") == "no_assign")
                )
                if abs(difference) > 0.01:
                    pair_count += 1
                    if is_no_assign_pair:
                        no_assign_non_tie_pairs += 1
                        no_assign_index = (
                            i
                            if members[i].get("action_type") == "no_assign"
                            else j
                        )
                        other_index = j if no_assign_index == i else i
                        if costs[no_assign_index] < costs[other_index]:
                            no_assign_better_pairs += 1
                        else:
                            no_assign_worse_pairs += 1
                elif is_no_assign_pair:
                    no_assign_tie_pairs += 1

    sys_nonzero = {}
    n_samples = len(samples)
    if n_samples > 0:
        check_n = min(n_samples, 500)
        for dim, name in enumerate(SYSTEM_LABEL_NAMES):
            nz = 0
            total = 0
            for s in samples[:check_n]:
                fsl = s.get("future_system_labels")
                if fsl is not None:
                    ch = fsl[..., dim]
                    nz += (ch.abs() > 1e-8).sum().item()
                    total += ch.numel()
            sys_nonzero[name] = round(nz / max(total, 1), 4)

    return {
        "total_samples": len(samples),
        "candidate_group_count": len(groups),
        "mean_group_size": round(mean_size, 2),
        "group_size_histogram": {str(k): v for k, v in sorted(size_counter.items())},
        "pairwise_pairs": pair_count,
        "action_schema": _infer_action_schema(samples),
        "no_assign_pairwise": {
            "non_tie_pairs": no_assign_non_tie_pairs,
            "no_assign_better_pairs": no_assign_better_pairs,
            "no_assign_worse_pairs": no_assign_worse_pairs,
            "tie_pairs": no_assign_tie_pairs,
        },
        "system_nonzero_ratio": sys_nonzero,
    }


def train(
    model: RMFSWorldModel,
    dataset: WorldModelDataset,
    epochs: int = 50,
    lr: float = 1e-3,
    save_dir: str = "DataGen/wm_checkpoints",
    save_name: str = "world_model.pt",
    verbose: bool = True,
    pairwise_data: Optional[List[dict]] = None,
    alpha_rank: float = 0.1,
    alpha_system: float = 1.0,
    alpha_station: float = 0.5,
    margin: float = 1.0,
    val_dataset: Optional[WorldModelDataset] = None,
    val_pairwise_data: Optional[List[dict]] = None,
    risk_weight: Optional[float] = None,
    device: Optional[torch.device] = None,
    node_channel_weights: Optional[List[float]] = None,
    wait_pos_weight: float = 5.0,
    blocked_pos_weight: float = 10.0,
) -> RMFSWorldModel:
    """Train the world model on collected samples."""
    if device is None:
        device = torch.device("cpu")
    model._checkpoint_action_schema = _infer_action_schema(dataset.samples)
    model = model.to(device)
    if verbose:
        print(f"  Device: {device}")

    if pairwise_data:
        pairwise_data = [_pair_to_device(p, device) for p in pairwise_data]
    if val_pairwise_data:
        val_pairwise_data = [_pair_to_device(p, device) for p in val_pairwise_data]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_collate_single)

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train_log.jsonl")
    log_file = open(log_path, "w", encoding="utf-8")

    if node_channel_weights is None:
        node_channel_weights = list(DEFAULT_NODE_CHANNEL_WEIGHTS)

    try:
        wait_pw, blocked_pw = _compute_pos_weights_from_dataset(dataset)
        wait_pos_weight = wait_pw
        blocked_pos_weight = blocked_pw
        if verbose:
            print(f"  pos_weight from train split: wait={wait_pos_weight:.2f}, blocked={blocked_pos_weight:.2f}")
    except Exception:
        if verbose:
            print(f"  pos_weight scan failed, using fallback: wait={wait_pos_weight:.2f}, blocked={blocked_pos_weight:.2f}")

    training_config = {
        "epochs": epochs, "lr": lr, "alpha_rank": alpha_rank,
        "alpha_system": alpha_system,
        "alpha_station": alpha_station, "margin": margin,
        "num_samples": len(dataset),
        "num_pairwise": len(pairwise_data) if pairwise_data else 0,
        "num_val_samples": len(val_dataset) if val_dataset else 0,
        "num_val_pairwise": len(val_pairwise_data) if val_pairwise_data else 0,
        "risk_weight": risk_weight,
        "label_schema_version": "wm_v4_local_pressure",
        "node_channel_weights": node_channel_weights,
        "wait_pos_weight": wait_pos_weight,
        "blocked_pos_weight": blocked_pos_weight,
        "action_schema": dict(model._checkpoint_action_schema),
    }

    best_loss = float("inf")
    best_epoch = -1
    best_val_rank_acc = -1.0
    best_val_epoch = -1
    epoch_records = []

    for epoch in range(epochs):
        model.train()
        t_epoch = time.time()

        epoch_loss = 0.0
        epoch_node_channels = {name: 0.0 for name in NODE_CHANNEL_NAMES}
        epoch_system = 0.0
        epoch_station = 0.0
        epoch_rank = 0.0
        epoch_sys_channels = {name: 0.0 for name in SYSTEM_LABEL_NAMES}
        n = 0

        for sample in loader:
            sample = _sample_to_device(sample, device)
            optimizer.zero_grad()
            losses = compute_loss(model, sample,
                                  alpha_system=alpha_system,
                                  alpha_station=alpha_station,
                                  node_channel_weights=node_channel_weights,
                                  wait_pos_weight=wait_pos_weight,
                                  blocked_pos_weight=blocked_pos_weight)
            total = losses["total"]

            if pairwise_data:
                pair = pairwise_data[n % len(pairwise_data)]
                rank_loss = compute_ranking_loss(model, pair, margin=margin)
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
            for ch_name, ch_val in losses["system_channels"].items():
                epoch_sys_channels[ch_name] += ch_val
            n += 1

        elapsed = time.time() - t_epoch
        avg = lambda x: x / max(n, 1)

        record = {
            "epoch": epoch + 1,
            "loss": round(avg(epoch_loss), 6),
            "system": round(avg(epoch_system), 6),
            "station": round(avg(epoch_station), 6),
            "rank": round(avg(epoch_rank), 6),
            "grad_norm": round(grad_norm, 4),
            "elapsed_seconds": round(elapsed, 1),
        }
        for ch_name in NODE_CHANNEL_NAMES:
            record[ch_name] = round(avg(epoch_node_channels[ch_name]), 6)
        for ch_name in SYSTEM_LABEL_NAMES:
            record[f"sys_{ch_name}"] = round(avg(epoch_sys_channels[ch_name]), 6)

        if record["loss"] < best_loss:
            best_loss = record["loss"]
            best_epoch = epoch + 1

        if verbose and n > 0:
            msg = (
                f"  Epoch {epoch+1:3d}/{epochs}  "
                f"loss={record['loss']:.4f}  "
                f"system={record['system']:.4f}  "
                f"station={record['station']:.4f}"
            )
            if pairwise_data:
                msg += f"  rank={record['rank']:.4f}"
            msg += f"  |g|={record['grad_norm']:.2f}  {elapsed:.0f}s"

        # --- Validation ---
        if val_pairwise_data:
            val_rank = evaluate_ranking(model, val_pairwise_data, device=device)
            record["val_rank_accuracy"] = val_rank["pairwise_rank_accuracy"]
            record["val_cost_std"] = val_rank["predicted_cost_std"]

            if val_dataset is not None:
                val_regret = evaluate_top1_regret(model, val_dataset, device=device)
                record["val_top1_regret_mean"] = val_regret["top1_regret_mean"]
                record["val_top1_regret_max"] = val_regret["top1_regret_max"]

            if val_rank["pairwise_rank_accuracy"] > best_val_rank_acc:
                best_val_rank_acc = val_rank["pairwise_rank_accuracy"]
                best_val_epoch = epoch + 1
                _save_checkpoint(model, save_dir, "best_" + save_name)

            if verbose and n > 0:
                msg += f"  val_acc={record['val_rank_accuracy']:.4f}"
                if "val_top1_regret_mean" in record:
                    msg += f"  regret={record['val_top1_regret_mean']:.4f}"

        if verbose and n > 0:
            print(msg, flush=True)

        epoch_records.append(record)
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()

    log_file.close()

    # --- Build model config for checkpoints ---
    model_config = _extract_model_config(model)

    # --- Save last checkpoint ---
    ckpt_path = os.path.join(save_dir, save_name)
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({
        "state_dict": cpu_state,
        "model_config": model_config,
        "label_schema_version": "wm_v4_local_pressure",
        "action_schema": dict(model._checkpoint_action_schema),
    }, ckpt_path)

    # --- Post-training ranking sanity check (on training data) ---
    rank_eval = {}
    if pairwise_data:
        eval_subset = pairwise_data[:min(500, len(pairwise_data))]
        rank_eval = evaluate_ranking(model, eval_subset, device=device)
        if verbose:
            print(f"\n  Train ranking (on {rank_eval['num_eval_pairs']} pairs):")
            print(f"    pairwise_rank_accuracy = {rank_eval['pairwise_rank_accuracy']:.4f}")
            print(f"    predicted_cost_std     = {rank_eval['predicted_cost_std']:.4f}")

    # --- Final validation summary ---
    val_eval = {}
    val_regret_eval = {}
    if val_pairwise_data:
        val_eval = evaluate_ranking(model, val_pairwise_data, device=device)
        if verbose:
            print(f"\n  Val ranking (on {val_eval['num_eval_pairs']} pairs):")
            print(f"    pairwise_rank_accuracy = {val_eval['pairwise_rank_accuracy']:.4f}")
            print(f"    predicted_cost_std     = {val_eval['predicted_cost_std']:.4f}")
    if val_dataset is not None:
        val_regret_eval = evaluate_top1_regret(model, val_dataset, device=device)
        if verbose and val_regret_eval["num_groups"] > 0:
            print(f"    top1_regret_mean       = {val_regret_eval['top1_regret_mean']:.4f}")
            print(f"    top1_regret_max        = {val_regret_eval['top1_regret_max']:.4f}")

    if verbose and best_val_epoch > 0:
        print(f"\n  Best val checkpoint: epoch {best_val_epoch}"
              f"  val_rank_accuracy={best_val_rank_acc:.4f}")

    # --- Save summary ---
    data_snapshot = _build_data_quality_snapshot(dataset)
    summary = {
        "training_config": training_config,
        "data_quality_snapshot": data_snapshot,
        "best_epoch_by_loss": best_epoch,
        "best_loss": round(best_loss, 6),
        "best_epoch_by_val": best_val_epoch,
        "best_val_rank_accuracy": round(best_val_rank_acc, 4) if best_val_rank_acc >= 0 else None,
        "final_epoch": epochs,
        "final_losses": epoch_records[-1] if epoch_records else {},
        "train_ranking_eval": rank_eval,
        "val_ranking_eval": val_eval,
        "val_regret_eval": val_regret_eval,
        "model_config": model_config,
        "label_schema_version": "wm_v4_local_pressure",
        "action_schema": dict(model._checkpoint_action_schema),
        "checkpoint_path": ckpt_path,
        "best_checkpoint_path": os.path.join(save_dir, "best_" + save_name) if best_val_epoch > 0 else None,
    }
    summary_path = os.path.join(save_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n  Last checkpoint saved to {ckpt_path}")
        if best_val_epoch > 0:
            print(f"  Best checkpoint saved to {os.path.join(save_dir, 'best_' + save_name)}")
        print(f"  Train log saved to {log_path}")
        print(f"  Summary saved to {summary_path}")

    return model
