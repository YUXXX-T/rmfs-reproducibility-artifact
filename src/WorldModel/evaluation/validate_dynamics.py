"""
World Model Dynamics Validation
===============================
Offline validation of latent dynamics: future-state prediction,
horizon-wise rollout, ranking, and action sensitivity.
潜在动力学的离线验证：未来状态预测、按时间范围展开、排序及动作敏感性分析。

Usage:
    python -m WorldModel.validate_dynamics \
      --data DataGen/wm_data/v6_formal/wm_train_data_v6.pt \
      --checkpoint DataGen/wm_checkpoints/v6_formal/best_world_model.pt \
      --split test \
      --save-json DataGen/wm_checkpoints/v6_formal/dynamics_eval_test.json
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from WorldModel.evaluate import _load_model, spearman, roc_auc
from WorldModel.dataset import WorldModelDataset
from WorldModel.costs import compute_realized_cost, CONGESTION_LAMBDAS

# =====================================================================
# Label schema constants
# =====================================================================

SYSTEM_CHANNELS = [
    "total_wait_time",
    "average_excess_delay",
    "station_queue_delta",
    "station_load_imbalance",
    "bottleneck_CVaR",
    "completed_orders_delta",
    "deadlock_or_severe_congestion_risk",
]

NODE_CHANNELS = [
    "self_occupancy",
    "local_density",
    "local_wait_pressure",
    "local_blocked_pressure",
    "reservation_pressure",
    "congestion_score",
]

STATION_CHANNELS = [
    "station_queue",
    "assigned_load",
]


# =====================================================================
# Pure-Python metric helpers
# =====================================================================

def pearson(x: list, y: list) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(max(sum((a - mx) ** 2 for a in x), 1e-12))
    sy = math.sqrt(max(sum((b - my) ** 2 for b in y), 1e-12))
    return cov / (sx * sy)


def huber_loss(pred: list, target: list, delta: float = 1.0) -> float:
    total = 0.0
    for p, t in zip(pred, target):
        diff = abs(p - t)
        if diff <= delta:
            total += 0.5 * diff ** 2
        else:
            total += delta * (diff - 0.5 * delta)
    return total / max(len(pred), 1)


def percentile(values: list, q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = q / 100.0 * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


# =====================================================================
# Inference helper
# =====================================================================

def _predict_sample(model, sample, device):
    """Run model forward on a single sample. Returns (node_preds, sys_preds, sta_preds, cost)."""
    station_nids = sample.get("station_node_ids")
    if station_nids is not None:
        station_nids = station_nids.tolist()

    nh = sample["node_history"].to(device)
    ei = sample["edge_index"].to(device)
    ef = sample["edge_features"].to(device)
    dc = sample["demand_context"].to(device)
    an = sample["action_node"].to(device)
    ag = sample["action_global"].to(device)

    z, e_demand, edge_attr = model.encode_state(nh, ei, ef, dc)
    rollout_outputs = model.rollout(
        z, e_demand, edge_attr, an, ag, ei, station_nids,
    )
    node_preds, sys_preds, sta_preds = rollout_outputs[:3]
    cost = model.cost_head(sys_preds)

    node_preds_cpu = [p.cpu() for p in node_preds]
    sys_preds_cpu = sys_preds.cpu()
    sta_preds_cpu = sta_preds.cpu()
    cost_cpu = cost.cpu().item()

    return node_preds_cpu, sys_preds_cpu, sta_preds_cpu, cost_cpu


# =====================================================================
# 1. Future System Metrics
# =====================================================================

def compute_future_system_metrics(
    all_sys_preds: List[torch.Tensor],
    all_sys_targets: List[torch.Tensor],
    all_masks: List[torch.Tensor],
) -> dict:
    per_channel: Dict[str, Dict[str, list]] = {
        ch: {"pred": [], "target": []} for ch in SYSTEM_CHANNELS
    }

    for sp, st, mask in zip(all_sys_preds, all_sys_targets, all_masks):
        K = min(sp.shape[0], st.shape[0])
        for k in range(K):
            if mask is not None and mask[k].item() < 0.5:
                continue
            for d, ch_name in enumerate(SYSTEM_CHANNELS):
                per_channel[ch_name]["pred"].append(sp[k, d].item())
                per_channel[ch_name]["target"].append(st[k, d].item())

    metrics = {}
    statistics = {}
    for ch_name, data in per_channel.items():
        p, t = data["pred"], data["target"]
        if len(p) < 2:
            metrics[ch_name] = {"mae": 0, "rmse": 0, "note": "insufficient_data"}
            statistics[ch_name] = {}
            continue
        errors = [abs(a - b) for a, b in zip(p, t)]
        sq_errors = [e ** 2 for e in errors]
        mae = sum(errors) / len(errors)
        rmse = math.sqrt(sum(sq_errors) / len(sq_errors))

        p_mean = sum(p) / len(p)
        t_mean = sum(t) / len(t)
        p_std = math.sqrt(sum((v - p_mean) ** 2 for v in p) / len(p))
        t_std = math.sqrt(sum((v - t_mean) ** 2 for v in t) / len(t))
        t_nz = sum(1 for v in t if abs(v) > 1e-6) / len(t)

        metrics[ch_name] = {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "huber": round(huber_loss(p, t), 6),
            "pearson": round(pearson(p, t), 4),
            "spearman": round(spearman(p, t), 4),
            "n": len(p),
        }
        statistics[ch_name] = {
            "pred_mean": round(p_mean, 6),
            "target_mean": round(t_mean, 6),
            "pred_std": round(p_std, 6),
            "target_std": round(t_std, 6),
            "target_nonzero_ratio": round(t_nz, 4),
        }
    return metrics, statistics


# =====================================================================
# 2. Horizon-wise Rollout Error
# =====================================================================

def compute_horizon_wise_metrics(
    all_sys_preds: List[torch.Tensor],
    all_sys_targets: List[torch.Tensor],
    all_node_preds: List[List[torch.Tensor]],
    all_node_targets: List[torch.Tensor],
    all_sta_preds: List[torch.Tensor],
    all_sta_targets: List[torch.Tensor],
    all_masks: List[torch.Tensor],
    congestion_threshold: float = 0.2,
) -> dict:
    K = all_sys_preds[0].shape[0] if all_sys_preds else 10

    sys_mae_by_h = [[] for _ in range(K)]
    node_density_sp_by_h = [[] for _ in range(K)]
    wait_auc_by_h_true = [[] for _ in range(K)]
    wait_auc_by_h_score = [[] for _ in range(K)]
    cong_auc_by_h_true = [[] for _ in range(K)]
    cong_auc_by_h_score = [[] for _ in range(K)]
    sta_mae_by_h = [[] for _ in range(K)]

    for idx in range(len(all_sys_preds)):
        sp = all_sys_preds[idx]
        st = all_sys_targets[idx]
        np_ = all_node_preds[idx]
        nt = all_node_targets[idx]
        stap = all_sta_preds[idx]
        stat = all_sta_targets[idx]
        mask = all_masks[idx]

        H = min(sp.shape[0], st.shape[0], len(np_), nt.shape[0])
        for k in range(min(H, K)):
            if mask is not None and mask[k].item() < 0.5:
                continue

            sys_err = (sp[k] - st[k]).abs().mean().item()
            sys_mae_by_h[k].append(sys_err)

            sp_val = spearman(np_[k][:, 1].tolist(), nt[k, :, 1].tolist())
            node_density_sp_by_h[k].append(sp_val)

            wait_auc_by_h_score[k].extend(torch.sigmoid(np_[k][:, 2]).tolist())
            wait_auc_by_h_true[k].extend(nt[k, :, 2].clamp(0, 1).tolist())

            cong_auc_by_h_score[k].extend(np_[k][:, 5].tolist())
            cong_auc_by_h_true[k].extend(
                [1.0 if v > congestion_threshold else 0.0 for v in nt[k, :, 5].tolist()]
            )

            Hs = min(stap.shape[0], stat.shape[0])
            if k < Hs:
                sta_err = (stap[k] - stat[k]).abs().mean().item()
                sta_mae_by_h[k].append(sta_err)

    def _mean(lst):
        return round(sum(lst) / max(len(lst), 1), 6) if lst else None

    return {
        "system_mae": [_mean(h) for h in sys_mae_by_h],
        "node_density_spearman": [_mean(h) for h in node_density_sp_by_h],
        "wait_auc": [
            round(roc_auc(wait_auc_by_h_true[k], wait_auc_by_h_score[k]), 4)
            if wait_auc_by_h_true[k] else None
            for k in range(K)
        ],
        "congestion_auc": [
            round(roc_auc(cong_auc_by_h_true[k], cong_auc_by_h_score[k]), 4)
            if cong_auc_by_h_true[k] else None
            for k in range(K)
        ],
        "station_mae": [_mean(h) for h in sta_mae_by_h],
    }


# =====================================================================
# 3. Node-level Metrics
# =====================================================================

def compute_node_metrics(
    all_node_preds: List[List[torch.Tensor]],
    all_node_targets: List[torch.Tensor],
    all_masks: List[torch.Tensor],
    congestion_threshold: float = 0.2,
    top_k: int = 20,
) -> dict:
    density_sp_vals = []
    wait_true, wait_score = [], []
    blocked_true, blocked_score = [], []
    reservation_errors = []
    cong_true, cong_score = [], []
    cong_sp_vals = []
    hit_rates = []

    for np_, nt, mask in zip(all_node_preds, all_node_targets, all_masks):
        K = min(len(np_), nt.shape[0])
        for k in range(K):
            if mask is not None and mask[k].item() < 0.5:
                continue
            pred = np_[k]
            tgt = nt[k]
            N = pred.shape[0]

            density_sp_vals.append(spearman(pred[:, 1].tolist(), tgt[:, 1].tolist()))

            wait_score.extend(torch.sigmoid(pred[:, 2]).tolist())
            wait_true.extend(tgt[:, 2].clamp(0, 1).tolist())

            blocked_score.extend(pred[:, 3].tolist())
            blocked_true.extend([1.0 if v > 0.1 else 0.0 for v in tgt[:, 3].tolist()])

            reservation_errors.extend((pred[:, 4] - tgt[:, 4]).abs().tolist())

            cong_score.extend(pred[:, 5].tolist())
            cong_true.extend(
                [1.0 if v > congestion_threshold else 0.0 for v in tgt[:, 5].tolist()]
            )
            cong_sp_vals.append(spearman(pred[:, 5].tolist(), tgt[:, 5].tolist()))

            actual_k = min(top_k, N)
            if actual_k > 0:
                pred_top = set(pred[:, 5].topk(actual_k).indices.tolist())
                tgt_top = set(tgt[:, 5].topk(actual_k).indices.tolist())
                hit_rates.append(len(pred_top & tgt_top) / actual_k)

    def _mean(lst):
        return round(sum(lst) / max(len(lst), 1), 6) if lst else 0.0

    wait_pos = sum(1 for v in wait_true if v >= 0.5)
    wait_neg = len(wait_true) - wait_pos
    wait_positive_ratio = round(wait_pos / max(len(wait_true), 1), 4)

    wait_auc_val = round(roc_auc(wait_true, wait_score), 4)
    if wait_pos < 10 or wait_neg < 10:
        wait_auc_status = "not_applicable_insufficient_positive"
    elif wait_auc_val > 0.5:
        wait_auc_status = "pass"
    else:
        wait_auc_status = "below_random"

    return {
        "density_spearman": _mean(density_sp_vals),
        "wait_auc": wait_auc_val,
        "wait_positive_ratio": wait_positive_ratio,
        "wait_auc_status": wait_auc_status,
        "blocked_auc": round(roc_auc(blocked_true, blocked_score), 4),
        "reservation_mae": _mean(reservation_errors),
        "congestion_auc": round(roc_auc(cong_true, cong_score), 4),
        "congestion_spearman": _mean(cong_sp_vals),
        "top_k_congested_hit_rate": _mean(hit_rates),
        "n_snapshots": len(density_sp_vals),
    }


# =====================================================================
# 4. Station-level Metrics
# =====================================================================

def compute_station_naive_baseline(train_ds: WorldModelDataset) -> dict:
    """Compute per-channel mean from train split as naive baseline."""
    queue_vals, load_vals = [], []
    for s in train_ds.samples:
        st = s.get("future_station_labels")
        mask = s.get("future_mask")
        if st is None:
            continue
        K = st.shape[0]
        for k in range(K):
            if mask is not None and mask[k].item() < 0.5:
                continue
            queue_vals.extend(st[k, :, 0].tolist())
            load_vals.extend(st[k, :, 1].tolist())
    queue_mean = sum(queue_vals) / max(len(queue_vals), 1)
    load_mean = sum(load_vals) / max(len(load_vals), 1)
    return {"queue_mean": queue_mean, "load_mean": load_mean}


def compute_station_metrics(
    all_sta_preds: List[torch.Tensor],
    all_sta_targets: List[torch.Tensor],
    all_masks: List[torch.Tensor],
    naive_baseline: Optional[dict] = None,
) -> dict:
    queue_pred, queue_target = [], []
    load_pred, load_target = [], []
    imb_pred, imb_target = [], []

    for sp, st, mask in zip(all_sta_preds, all_sta_targets, all_masks):
        K = min(sp.shape[0], st.shape[0])
        for k in range(K):
            if mask is not None and mask[k].item() < 0.5:
                continue
            queue_pred.extend(sp[k, :, 0].tolist())
            queue_target.extend(st[k, :, 0].tolist())
            load_pred.extend(sp[k, :, 1].tolist())
            load_target.extend(st[k, :, 1].tolist())

            if sp.shape[1] >= 2:
                imb_pred.append(sp[k, :, 0].std().item())
                imb_target.append(st[k, :, 0].std().item())

    def _mae(p, t):
        if not p:
            return 0.0
        return sum(abs(a - b) for a, b in zip(p, t)) / len(p)

    def _rmse(p, t):
        if not p:
            return 0.0
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, t)) / len(p))

    q_mae = _mae(queue_pred, queue_target)
    l_mae = _mae(load_pred, load_target)

    result = {
        "queue_mae": round(q_mae, 6),
        "queue_rmse": round(_rmse(queue_pred, queue_target), 6),
        "queue_spearman": round(spearman(queue_pred, queue_target), 4),
        "assigned_load_mae": round(l_mae, 6),
        "assigned_load_rmse": round(_rmse(load_pred, load_target), 6),
        "station_imbalance_correlation": round(pearson(imb_pred, imb_target), 4),
        "n": len(queue_pred),
    }

    if naive_baseline and queue_target:
        nq = naive_baseline["queue_mean"]
        nl = naive_baseline["load_mean"]
        q_naive_mae = sum(abs(nq - t) for t in queue_target) / len(queue_target)
        l_naive_mae = sum(abs(nl - t) for t in load_target) / len(load_target)
        result["queue_naive_mae"] = round(q_naive_mae, 6)
        result["queue_improvement_over_naive"] = round(
            1 - q_mae / max(q_naive_mae, 1e-8), 4
        )
        result["assigned_load_naive_mae"] = round(l_naive_mae, 6)
        result["assigned_load_improvement_over_naive"] = round(
            1 - l_mae / max(l_naive_mae, 1e-8), 4
        )

    return result


# =====================================================================
# 5. Ranking Metrics
# =====================================================================

def compute_ranking_metrics(
    model,
    dataset: WorldModelDataset,
    all_predicted_costs: List[float],
    risk_weight: Optional[float],
    device,
    cost_lambdas=None,
) -> dict:
    pairs = dataset.build_pairwise_data(risk_weight=risk_weight, cost_lambdas=cost_lambdas)

    if not pairs:
        return {
            "pairwise_rank_accuracy": 0.0,
            "num_pairs": 0,
            "note": "no_pairs",
        }

    correct = 0
    model.eval()
    with torch.no_grad():
        for pair in pairs:
            sb, sw = pair["sample_i"], pair["sample_j"]
            _, _, _, cb = _predict_sample(model, sb, device)
            _, _, _, cw = _predict_sample(model, sw, device)
            if cb < cw:
                correct += 1

    pairwise_acc = correct / len(pairs)

    groups = dataset._group_samples()
    top1_correct = 0
    top1_regrets = []
    group_spearman_vals = []
    total_groups = 0

    for gid, members in groups.items():
        if len(members) < 2:
            continue
        total_groups += 1

        pred_costs = []
        real_costs = []
        for s in members:
            _, _, _, pc = _predict_sample(model, s, device)
            rc = dataset._get_cost(s, risk_weight, cost_lambdas)
            pred_costs.append(pc)
            real_costs.append(rc)

        best_pred_idx = min(range(len(pred_costs)), key=lambda i: pred_costs[i])
        best_real_idx = min(range(len(real_costs)), key=lambda i: real_costs[i])

        if best_pred_idx == best_real_idx:
            top1_correct += 1

        regret = real_costs[best_pred_idx] - real_costs[best_real_idx]
        oracle_range = max(real_costs) - min(real_costs)
        norm_regret = regret / max(oracle_range, 1e-6)
        top1_regrets.append(norm_regret)

        if len(pred_costs) >= 3:
            group_spearman_vals.append(spearman(pred_costs, real_costs))

    cost_std = 0.0
    if all_predicted_costs:
        mean_c = sum(all_predicted_costs) / len(all_predicted_costs)
        cost_std = math.sqrt(
            sum((c - mean_c) ** 2 for c in all_predicted_costs) / len(all_predicted_costs)
        )

    return {
        "pairwise_rank_accuracy": round(pairwise_acc, 4),
        "num_pairs": len(pairs),
        "top1_accuracy": round(top1_correct / max(total_groups, 1), 4),
        "top1_regret_mean": round(sum(top1_regrets) / max(len(top1_regrets), 1), 4),
        "top1_regret_p90": round(percentile(top1_regrets, 90), 4),
        "top1_regret_max": round(max(top1_regrets) if top1_regrets else 0.0, 4),
        "predicted_cost_std": round(cost_std, 4),
        "cost_spearman": round(
            sum(group_spearman_vals) / max(len(group_spearman_vals), 1), 4
        ),
        "num_groups": total_groups,
    }


# =====================================================================
# 6. Action Sensitivity
# =====================================================================

def compute_action_sensitivity(
    model,
    dataset: WorldModelDataset,
    risk_weight: Optional[float],
    device,
    cost_lambdas=None,
) -> dict:
    groups = dataset._group_samples()

    cost_stds = []
    sys_stds = []
    real_cost_stds = []
    group_spearman_vals = []

    model.eval()
    with torch.no_grad():
        for gid, members in groups.items():
            if len(members) < 2:
                continue

            pred_costs = []
            real_costs = []
            sys_means = []

            for s in members:
                _, sp, _, pc = _predict_sample(model, s, device)
                rc = dataset._get_cost(s, risk_weight, cost_lambdas)
                pred_costs.append(pc)
                real_costs.append(rc)
                sys_means.append(sp.mean().item())

            if len(pred_costs) < 2:
                continue

            pc_mean = sum(pred_costs) / len(pred_costs)
            cost_stds.append(
                math.sqrt(sum((c - pc_mean) ** 2 for c in pred_costs) / len(pred_costs))
            )

            sm_mean = sum(sys_means) / len(sys_means)
            sys_stds.append(
                math.sqrt(sum((s - sm_mean) ** 2 for s in sys_means) / len(sys_means))
            )

            rc_mean = sum(real_costs) / len(real_costs)
            real_cost_stds.append(
                math.sqrt(sum((c - rc_mean) ** 2 for c in real_costs) / len(real_costs))
            )

            if len(pred_costs) >= 3:
                group_spearman_vals.append(spearman(pred_costs, real_costs))

    def _mean(lst):
        return round(sum(lst) / max(len(lst), 1), 6) if lst else 0.0

    def _median(lst):
        if not lst:
            return 0.0
        s = sorted(lst)
        mid = len(s) // 2
        return round(s[mid], 6) if len(s) % 2 else round((s[mid - 1] + s[mid]) / 2, 6)

    return {
        "predicted_cost_group_std_mean": _mean(cost_stds),
        "predicted_cost_group_std_median": _median(cost_stds),
        "predicted_system_group_std_mean": _mean(sys_stds),
        "realized_cost_group_std_mean": _mean(real_cost_stds),
        "predicted_vs_realized_group_spearman_mean": _mean(group_spearman_vals),
        "predicted_vs_realized_group_spearman_median": _median(group_spearman_vals),
        "num_groups": len(cost_stds),
    }


# =====================================================================
# 7. Prediction Examples
# =====================================================================

def build_prediction_examples(
    samples: List[dict],
    all_sys_preds: List[torch.Tensor],
    all_sta_preds: List[torch.Tensor],
    all_predicted_costs: List[float],
    risk_weight: Optional[float],
    n_examples: int = 5,
    cost_lambdas=None,
) -> list:
    indices = list(range(min(n_examples, len(samples))))
    examples = []
    for i in indices:
        s = samples[i]
        sp = all_sys_preds[i]
        stap = all_sta_preds[i]
        st = s["future_system_labels"]
        stat = s.get("future_station_labels")

        K = min(sp.shape[0], st.shape[0])
        horizon_data = []
        for k in range(K):
            p = sp[k].tolist()
            t = st[k].tolist()
            err = [round(abs(a - b), 6) for a, b in zip(p, t)]
            horizon_data.append({
                "k": k + 1,
                "pred": [round(v, 4) for v in p],
                "target": [round(v, 4) for v in t],
                "abs_error": err,
            })

        if cost_lambdas is not None:
            rc = compute_realized_cost(st, lambdas=cost_lambdas)
        else:
            rc = compute_realized_cost(st, risk_weight=risk_weight)
        ex = {
            "sample_index": i,
            "candidate_group_id": s.get("candidate_group_id", ""),
            "decision_tick": s.get("decision_tick", -1),
            "predicted_cost": round(all_predicted_costs[i], 4),
            "realized_cost": round(rc, 4),
            "future_system_by_horizon": horizon_data,
        }

        if stat is not None and stap is not None:
            Ks = min(stap.shape[0], stat.shape[0])
            sta_data = []
            for k in range(min(3, Ks)):
                sta_data.append({
                    "k": k + 1,
                    "pred": [round(v, 4) for v in stap[k].flatten().tolist()],
                    "target": [round(v, 4) for v in stat[k].flatten().tolist()],
                })
            ex["future_station_by_horizon"] = sta_data

        examples.append(ex)
    return examples


# =====================================================================
# 8. Pass Criteria
# =====================================================================

def evaluate_pass_criteria(
    sys_metrics: dict,
    sys_stats: dict,
    node_metrics: dict,
    station_metrics: dict,
    ranking_metrics: dict,
    sensitivity_metrics: dict,
) -> dict:
    sys_not_const = any(
        sys_stats.get(ch, {}).get("pred_std", 0) > 0.001
        for ch in SYSTEM_CHANNELS
    )
    sys_spearman_positive = sum(
        1 for ch in SYSTEM_CHANNELS
        if sys_metrics.get(ch, {}).get("spearman", 0) > 0
    )

    wait_status = node_metrics.get("wait_auc_status", "")
    if wait_status == "not_applicable_insufficient_positive":
        wait_auc_pass = True
    else:
        wait_auc_pass = node_metrics.get("wait_auc", 0) > 0.5

    station_below_naive = (
        station_metrics.get("queue_improvement_over_naive", 0) > 0
        or station_metrics.get("assigned_load_improvement_over_naive", 0) > 0
    )
    pairwise_good = ranking_metrics.get("pairwise_rank_accuracy", 0) >= 0.8
    action_spearman_positive = (
        sensitivity_metrics.get("predicted_vs_realized_group_spearman_mean", 0) > 0
    )

    basic = {
        "system_not_constant": sys_not_const,
        "system_spearman_mostly_positive": sys_spearman_positive >= 4,
        "density_spearman_positive": node_metrics.get("density_spearman", 0) > 0,
        "wait_auc_above_0.5": wait_auc_pass,
        "congestion_auc_above_0.5": node_metrics.get("congestion_auc", 0) > 0.5,
        "station_mae_below_naive": station_below_naive,
        "action_sensitive": sensitivity_metrics.get("predicted_cost_group_std_mean", 0) > 0.01,
        "action_spearman_positive": action_spearman_positive,
        "pairwise_rank_accuracy_ge_0.8": pairwise_good,
    }

    strong = {
        "density_spearman_ge_0.5": node_metrics.get("density_spearman", 0) >= 0.5,
        "wait_auc_ge_0.7": node_metrics.get("wait_auc", 0) >= 0.7,
        "congestion_auc_ge_0.7": node_metrics.get("congestion_auc", 0) >= 0.7,
        "system_spearman_significant": sys_spearman_positive >= 5,
        "pairwise_rank_accuracy_ge_0.9": ranking_metrics.get("pairwise_rank_accuracy", 0) >= 0.9,
        "action_spearman_ge_0.5": (
            sensitivity_metrics.get("predicted_vs_realized_group_spearman_mean", 0) >= 0.5
        ),
        "top1_regret_le_0.1": ranking_metrics.get("top1_regret_mean", 1) <= 0.1,
    }

    return {"basic": basic, "strong": strong}


# =====================================================================
# Label schema for JSON
# =====================================================================

def build_prediction_targets_schema() -> dict:
    return {
        "future_system": {
            "pred_tensor": "system_preds",
            "target_tensor": "future_system_labels",
            "shape": ["H", 7],
            "channels": [
                {"dim": d, "name": ch}
                for d, ch in enumerate(SYSTEM_CHANNELS)
            ],
        },
        "future_node": {
            "pred_tensor": "node_preds",
            "target_tensor": "future_node_labels",
            "shape": ["H", "N", 6],
            "channels": [
                {"dim": d, "name": ch}
                for d, ch in enumerate(NODE_CHANNELS)
            ],
        },
        "future_station": {
            "pred_tensor": "station_preds",
            "target_tensor": "future_station_labels",
            "shape": ["H", "S", 2],
            "channels": [
                {"dim": d, "name": ch}
                for d, ch in enumerate(STATION_CHANNELS)
            ],
        },
    }


# =====================================================================
# Main validation runner
# =====================================================================

def run_validation(
    data_path: str,
    checkpoint_path: str,
    split: str = "test",
    splits_path: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    split_seed: int = 42,
    risk_weight: Optional[float] = None,
    max_samples: int = 0,
    device_str: str = "auto",
    congestion_threshold: float = 0.2,
    save_predictions: Optional[str] = None,
    save_node_predictions: bool = False,
    cost_lambdas=None,
) -> dict:
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"  Device: {device}")
    print(f"  Loading data from {data_path} ...")
    full_ds = WorldModelDataset.from_file(data_path)
    print(f"  Total samples: {len(full_ds)}")

    if splits_path:
        print(f"  Loading splits from {splits_path} ...")
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        train_gids = set(splits_data["train"])
        val_gids = set(splits_data["val"])
        test_gids = set(splits_data["test"])
        train_s, val_s, test_s = [], [], []
        for s in full_ds.samples:
            gid = s.get("candidate_group_id", "")
            if gid in train_gids:
                train_s.append(s)
            elif gid in val_gids:
                val_s.append(s)
            elif gid in test_gids:
                test_s.append(s)
        train_ds = WorldModelDataset(train_s)
        val_ds = WorldModelDataset(val_s)
        test_ds = WorldModelDataset(test_s)
    else:
        train_ds, val_ds, test_ds = full_ds.split_by_group(
            train_ratio=train_ratio, val_ratio=val_ratio, seed=split_seed,
        )
    split_map = {"train": train_ds, "val": val_ds, "test": test_ds, "all": full_ds}
    ds = split_map.get(split, test_ds)
    print(f"  Split '{split}': {ds.summary()}")

    if max_samples > 0 and len(ds) > max_samples:
        ds = WorldModelDataset(ds.samples[:max_samples])
        print(f"  Truncated to {len(ds)} samples")

    print(f"  Loading model from {checkpoint_path} ...")
    model, schema = _load_model(checkpoint_path)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded ({n_params:,} params, schema={schema})")

    # ---- Forward pass on all samples ----
    print(f"\n  Running inference on {len(ds)} samples ...")
    t0 = time.time()
    all_sys_preds = []
    all_node_preds = []
    all_sta_preds = []
    all_predicted_costs = []
    all_sys_targets = []
    all_node_targets = []
    all_sta_targets = []
    all_masks = []
    prediction_records = []

    with torch.no_grad():
        for i, sample in enumerate(ds.samples):
            np_, sp, stap, cost = _predict_sample(model, sample, device)
            all_node_preds.append(np_)
            all_sys_preds.append(sp)
            all_sta_preds.append(stap)
            all_predicted_costs.append(cost)

            all_sys_targets.append(sample["future_system_labels"])
            all_node_targets.append(sample["future_node_labels"])
            sta_t = sample.get("future_station_labels")
            if sta_t is None:
                sta_t = torch.zeros_like(stap)
            all_sta_targets.append(sta_t)
            all_masks.append(sample.get("future_mask"))

            if save_predictions:
                rec = {
                    "sample_index": i,
                    "candidate_group_id": sample.get("candidate_group_id", ""),
                    "decision_tick": sample.get("decision_tick", -1),
                    "candidate_info": sample.get("candidate_info", {}),
                    "predicted_cost": cost,
                    "realized_cost": sample.get("realized_cost", 0.0),
                    "system_preds": sp,
                    "system_targets": sample["future_system_labels"],
                    "station_preds": stap,
                    "station_targets": sta_t,
                }
                if save_node_predictions:
                    rec["node_preds"] = np_
                    rec["node_targets"] = sample["future_node_labels"]
                prediction_records.append(rec)

            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(ds)} ...")

    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s ({elapsed / max(len(ds), 1) * 1000:.1f} ms/sample)")

    # ---- Compute metrics ----
    print("\n  Computing future system metrics ...")
    sys_metrics, sys_stats = compute_future_system_metrics(
        all_sys_preds, all_sys_targets, all_masks,
    )

    print("  Computing horizon-wise metrics ...")
    horizon_metrics = compute_horizon_wise_metrics(
        all_sys_preds, all_sys_targets,
        all_node_preds, all_node_targets,
        all_sta_preds, all_sta_targets,
        all_masks, congestion_threshold,
    )

    print("  Computing node-level metrics ...")
    node_metrics = compute_node_metrics(
        all_node_preds, all_node_targets, all_masks,
        congestion_threshold,
    )

    print("  Computing station naive baseline from train split ...")
    naive_baseline = compute_station_naive_baseline(train_ds)

    print("  Computing station-level metrics ...")
    station_metrics = compute_station_metrics(
        all_sta_preds, all_sta_targets, all_masks,
        naive_baseline=naive_baseline,
    )

    print("  Computing ranking metrics ...")
    ranking_metrics = compute_ranking_metrics(
        model, ds, all_predicted_costs, risk_weight, device,
        cost_lambdas=cost_lambdas,
    )

    print("  Computing action sensitivity ...")
    sensitivity_metrics = compute_action_sensitivity(
        model, ds, risk_weight, device,
        cost_lambdas=cost_lambdas,
    )

    print("  Building prediction examples ...")
    pred_examples = build_prediction_examples(
        ds.samples, all_sys_preds, all_sta_preds,
        all_predicted_costs, risk_weight,
        cost_lambdas=cost_lambdas,
    )

    # ---- Build prediction_statistics ----
    node_stat_data = {ch: [] for ch in NODE_CHANNELS}
    node_target_data = {ch: [] for ch in NODE_CHANNELS}
    for np_, nt, mask in zip(all_node_preds, all_node_targets, all_masks):
        K = min(len(np_), nt.shape[0])
        for k in range(K):
            if mask is not None and mask[k].item() < 0.5:
                continue
            for c, ch in enumerate(NODE_CHANNELS):
                node_stat_data[ch].extend(np_[k][:, c].tolist())
                node_target_data[ch].extend(nt[k, :, c].tolist())

    node_stats = {}
    for ch in NODE_CHANNELS:
        p, t = node_stat_data[ch], node_target_data[ch]
        if p:
            node_stats[ch] = {
                "pred_mean": round(sum(p) / len(p), 6),
                "target_mean": round(sum(t) / len(t), 6),
                "target_nonzero_ratio": round(
                    sum(1 for v in t if abs(v) > 1e-6) / len(t), 4
                ),
            }

    sta_stats = {}
    for c, ch in enumerate(STATION_CHANNELS):
        vals_p, vals_t = [], []
        for sp, st, mask in zip(all_sta_preds, all_sta_targets, all_masks):
            K = min(sp.shape[0], st.shape[0])
            for k in range(K):
                if mask is not None and mask[k].item() < 0.5:
                    continue
                vals_p.extend(sp[k, :, c].tolist())
                vals_t.extend(st[k, :, c].tolist())
        if vals_p:
            sta_stats[ch] = {
                "pred_mean": round(sum(vals_p) / len(vals_p), 6),
                "target_mean": round(sum(vals_t) / len(vals_t), 6),
            }

    prediction_statistics = {
        "future_system": sys_stats,
        "future_node": node_stats,
        "future_station": sta_stats,
    }

    print("  Evaluating pass criteria ...")
    pass_criteria = evaluate_pass_criteria(
        sys_metrics, sys_stats, node_metrics, station_metrics,
        ranking_metrics, sensitivity_metrics,
    )

    # ---- Save predictions .pt ----
    if save_predictions and prediction_records:
        os.makedirs(os.path.dirname(save_predictions) or ".", exist_ok=True)
        torch.save(prediction_records, save_predictions)
        print(f"\n  Predictions saved: {save_predictions}")

    # ---- Build result ----
    result = {
        "meta": {
            "checkpoint": checkpoint_path,
            "data": data_path,
            "split": split,
            "splits_path": splits_path,
            "num_samples": len(ds),
            "num_groups": ds.summary()["num_groups"],
            "risk_weight": risk_weight,
            "cost_mode": "congestion" if cost_lambdas is not None else "default",
            "device": str(device),
            "label_schema_version": schema,
            "inference_time_s": round(elapsed, 2),
        },
        "prediction_targets": build_prediction_targets_schema(),
        "aggregate_metrics": {
            "future_system": sys_metrics,
            "future_node": node_metrics,
            "future_station": station_metrics,
        },
        "prediction_statistics": prediction_statistics,
        "horizon_wise": horizon_metrics,
        "ranking": ranking_metrics,
        "action_sensitivity": sensitivity_metrics,
        "prediction_examples": pred_examples,
        "pass_criteria": pass_criteria,
    }

    return result


# =====================================================================
# Pretty-print summary
# =====================================================================

def print_summary(result: dict):
    print("\n" + "=" * 62)
    print("  Dynamics Validation Summary")
    print("=" * 62)

    meta = result["meta"]
    print(f"\n  Checkpoint: {meta['checkpoint']}")
    print(f"  Split: {meta['split']}  |  Samples: {meta['num_samples']}  |  "
          f"Groups: {meta['num_groups']}")
    if meta.get("risk_weight") is not None:
        print(f"  Risk weight: {meta['risk_weight']}")
    if meta.get("cost_mode") == "congestion":
        print(f"  Cost mode: congestion (ch5=0)")

    sys_m = result["aggregate_metrics"]["future_system"]
    sys_s = result.get("prediction_statistics", {}).get("future_system", {})
    print(f"\n  --- Future System Metrics ---")
    print(f"  {'Channel':<36s} {'MAE':>8s} {'RMSE':>8s} {'Spearman':>9s} {'pred_std':>9s}")
    for ch in SYSTEM_CHANNELS:
        m = sys_m.get(ch, {})
        s = sys_s.get(ch, {})
        print(f"  {ch:<36s} {m.get('mae', 0):>8.4f} {m.get('rmse', 0):>8.4f} "
              f"{m.get('spearman', 0):>+9.4f} {s.get('pred_std', 0):>9.4f}")

    node_m = result["aggregate_metrics"]["future_node"]
    print(f"\n  --- Node-level Metrics ---")
    print(f"  density_spearman      : {node_m.get('density_spearman', 0):+.4f}")
    wait_status = node_m.get('wait_auc_status', '')
    print(f"  wait_auc              : {node_m.get('wait_auc', 0):.4f}  ({wait_status})")
    print(f"  blocked_auc           : {node_m.get('blocked_auc', 0):.4f}")
    print(f"  reservation_mae       : {node_m.get('reservation_mae', 0):.6f}")
    print(f"  congestion_auc        : {node_m.get('congestion_auc', 0):.4f}")
    print(f"  congestion_spearman   : {node_m.get('congestion_spearman', 0):+.4f}")
    print(f"  top_k_hit_rate        : {node_m.get('top_k_congested_hit_rate', 0):.4f}")

    sta_m = result["aggregate_metrics"]["future_station"]
    print(f"\n  --- Station-level Metrics ---")
    print(f"  queue_mae             : {sta_m.get('queue_mae', 0):.6f}")
    if "queue_naive_mae" in sta_m:
        print(f"  queue_naive_mae       : {sta_m['queue_naive_mae']:.6f}")
        print(f"  queue_improvement     : {sta_m.get('queue_improvement_over_naive', 0):+.2%}")
    print(f"  queue_rmse            : {sta_m.get('queue_rmse', 0):.6f}")
    print(f"  queue_spearman        : {sta_m.get('queue_spearman', 0):+.4f}")
    print(f"  assigned_load_mae     : {sta_m.get('assigned_load_mae', 0):.6f}")
    if "assigned_load_naive_mae" in sta_m:
        print(f"  load_naive_mae        : {sta_m['assigned_load_naive_mae']:.6f}")
        print(f"  load_improvement      : {sta_m.get('assigned_load_improvement_over_naive', 0):+.2%}")
    print(f"  assigned_load_rmse    : {sta_m.get('assigned_load_rmse', 0):.6f}")
    print(f"  imbalance_corr        : {sta_m.get('station_imbalance_correlation', 0):+.4f}")

    rk = result["ranking"]
    print(f"\n  --- Ranking Metrics ---")
    print(f"  pairwise_rank_accuracy: {rk.get('pairwise_rank_accuracy', 0):.4f}  "
          f"({rk.get('num_pairs', 0)} pairs)")
    print(f"  top1_accuracy         : {rk.get('top1_accuracy', 0):.4f}")
    print(f"  top1_regret_mean      : {rk.get('top1_regret_mean', 0):.4f}")
    print(f"  top1_regret_p90       : {rk.get('top1_regret_p90', 0):.4f}")
    print(f"  cost_spearman         : {rk.get('cost_spearman', 0):+.4f}")

    sens = result["action_sensitivity"]
    print(f"\n  --- Action Sensitivity ---")
    print(f"  pred_cost_group_std   : {sens.get('predicted_cost_group_std_mean', 0):.4f}")
    print(f"  pred_sys_group_std    : {sens.get('predicted_system_group_std_mean', 0):.6f}")
    print(f"  pred_vs_real_spearman : {sens.get('predicted_vs_realized_group_spearman_mean', 0):+.4f}")

    pc = result["pass_criteria"]
    print(f"\n  --- Pass Criteria ---")
    print(f"  Basic:")
    for k, v in pc["basic"].items():
        if isinstance(v, bool):
            tag = "PASS" if v else "FAIL"
        else:
            tag = str(v).upper()
        print(f"    [{tag}] {k}")
    print(f"  Strong:")
    for k, v in pc["strong"].items():
        if isinstance(v, bool):
            tag = "PASS" if v else "FAIL"
        else:
            tag = str(v).upper()
        print(f"    [{tag}] {k}")

    all_basic = all(v for v in pc["basic"].values() if isinstance(v, bool))
    all_strong = all(v for v in pc["strong"].values() if isinstance(v, bool))
    print()
    if all_strong:
        print("  >>> ALL STRONG CRITERIA PASSED")
    elif all_basic:
        print("  >>> ALL BASIC CRITERIA PASSED (some strong criteria pending)")
    else:
        print("  >>> SOME BASIC CRITERIA FAILED")
    print("=" * 62)


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="World Model Dynamics Validation",
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--splits", type=str, default=None,
                        help="Path to splits.json from fuse_and_split")
    parser.add_argument("--risk-weight", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--save-predictions", type=str, default=None)
    parser.add_argument("--save-node-predictions", action="store_true")
    parser.add_argument("--congestion-threshold", type=float, default=0.2)
    parser.add_argument("--congestion-cost", action="store_true",
                        help="Use congestion-only lambdas (ch5=0) for ranking/regret metrics")

    args = parser.parse_args()

    cost_lambdas = CONGESTION_LAMBDAS if args.congestion_cost else None

    print("=" * 62)
    print("  RMFS World Model — Dynamics Validation")
    print("=" * 62)
    if args.congestion_cost:
        print("  Cost mode: CONGESTION (ch5=0, lambdas=[1,0.5,0.5,0.5,1,0,2])")

    result = run_validation(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        split=args.split,
        splits_path=args.splits,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        risk_weight=args.risk_weight,
        max_samples=args.max_samples,
        device_str=args.device,
        congestion_threshold=args.congestion_threshold,
        save_predictions=args.save_predictions,
        save_node_predictions=args.save_node_predictions,
        cost_lambdas=cost_lambdas,
    )

    print_summary(result)

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  JSON saved: {args.save_json}")


if __name__ == "__main__":
    main()
