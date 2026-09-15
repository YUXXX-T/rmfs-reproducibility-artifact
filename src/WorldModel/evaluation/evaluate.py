"""
World Model Go/No-Go Evaluation (v3 — counterfactual)
======================================================
Computes the metrics from represent_world_model.md §10:

  1. Node density Spearman correlation  >= 0.60
  2. Wait prediction AUC               >= 0.70
  3. Congestion prediction AUC          >= 0.70
  4. Pairwise ranking accuracy          >= 65%
  5. Throughput improvement vs Greedy   >= 10%

Usage:
    python -m WorldModel.evaluate
"""

import os
import sys
import random
import inspect
from typing import List

import torch
import torch.nn.functional as F


# =====================================================================
# Metric helpers (pure Python, no sklearn/scipy dependency)
# =====================================================================

def _rank_array(values):
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for rank, idx in enumerate(indexed):
        ranks[idx] = float(rank)
    return ranks


def spearman(pred: list, target: list) -> float:
    n = len(pred)
    if n < 3:
        return 0.0
    rp = _rank_array(pred)
    rt = _rank_array(target)
    d_sq = sum((a - b) ** 2 for a, b in zip(rp, rt))
    return 1.0 - 6.0 * d_sq / (n * (n * n - 1))


def roc_auc(y_true: list, y_score: list) -> float:
    pos = sum(1 for y in y_true if y >= 0.5)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5
    pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    tp = 0
    auc_sum = 0.0
    for _, label in pairs:
        if label >= 0.5:
            tp += 1
        else:
            auc_sum += tp
    return auc_sum / (pos * neg)


# =====================================================================
# Engine builder (shared across phases)
# =====================================================================

def _build_engine(config, task_assigner=None):
    from Engine.simulation_engine import SimulationEngine
    import Policies  # noqa — trigger auto-registration
    from Policies.policy_registry import get_policy

    og_name, og_params = config.policies.order_generator
    ta_name, ta_params = config.policies.task_assigner
    pp_name, pp_params = config.policies.path_planner
    rp_name, rp_params = config.policies.pod_return_planner
    pr_name, pr_params = config.policies.pod_retriever

    OG = get_policy("order_generator", og_name)
    PP = get_policy("path_planner", pp_name)
    RP = get_policy("pod_return_planner", rp_name)
    PR = get_policy("pod_retriever", pr_name)

    order_gen = OG(
        order_interval=config.simulation.order_interval,
        max_items_per_order=config.simulation.max_items_per_order,
        fixed_order_size=config.simulation.fixed_order_size,
        max_items_per_sku=config.simulation.max_items_per_sku,
        **og_params,
    )
    if config.simulation.seed is not None and "seed" not in pp_params:
        if "seed" in inspect.signature(PP).parameters:
            pp_params = {**pp_params, "seed": config.simulation.seed}
    path_planner = PP(**pp_params)
    pod_return = RP(**rp_params)
    pod_retriever = PR(**pr_params)

    if task_assigner is None:
        TA = get_policy("task_assigner", ta_name)
        task_assigner = TA(**ta_params)

    task_assigner.pod_return_planner = pod_return
    task_assigner.pod_retriever = pod_retriever
    task_assigner.path_planner = path_planner

    return SimulationEngine(
        config=config,
        order_generator=order_gen,
        task_assigner=task_assigner,
        path_planner=path_planner,
        visualizer=None,
    )


# =====================================================================
# Model loader (handles both v2 raw state_dict and v3 checkpoint)
# =====================================================================

def _load_model(checkpoint_path, fallback_demand_dim=9, fallback_num_stations=4):
    from WorldModel.model import RMFSWorldModel

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_config" in ckpt:
        cfg = ckpt["model_config"]
        model = RMFSWorldModel(
            node_feat_dim=cfg.get("node_feat_dim", 10),
            edge_feat_dim=cfg.get("edge_feat_dim", 6),
            demand_dim=cfg.get("demand_dim", fallback_demand_dim),
            action_node_dim=cfg.get("action_node_dim", 8),
            action_global_dim=cfg.get("action_global_dim", 6),
            hidden_dim=cfg.get("hidden_dim", 64),
            num_spatial_layers=3,
            rollout_horizon=cfg.get("rollout_horizon", 10),
            num_stations=cfg.get("num_stations", fallback_num_stations),
        )
        model.load_state_dict(ckpt["state_dict"])
        schema = ckpt.get("label_schema_version", "unknown")
    else:
        model = RMFSWorldModel(
            node_feat_dim=10, edge_feat_dim=6,
            demand_dim=fallback_demand_dim,
            hidden_dim=64, num_spatial_layers=3,
            rollout_horizon=10, num_stations=fallback_num_stations,
        )
        model.load_state_dict(ckpt)
        schema = "legacy"

    return model, schema


# =====================================================================
# Phase A — prediction quality (Spearman, AUC)
# =====================================================================

CONGESTION_THRESHOLD = 0.2


def evaluate_predictions(model, samples):
    model.eval()
    spearman_vals = []
    wait_true_all, wait_score_all = [], []
    cong_true_all, cong_score_all = [], []

    with torch.no_grad():
        for sample in samples:
            station_nids = sample.get("station_node_ids")
            if station_nids is not None:
                station_nids = station_nids.tolist()

            node_preds, sys_preds, sta_preds, _ = model(
                node_history=sample["node_history"],
                edge_index=sample["edge_index"],
                edge_features=sample["edge_features"],
                demand_context=sample["demand_context"],
                action_node=sample["action_node"],
                action_global=sample["action_global"],
                station_node_ids=station_nids,
            )
            future_node = sample["future_node_labels"]
            mask = sample.get("future_mask")
            K = min(len(node_preds), future_node.shape[0])

            for k in range(K):
                if mask is not None and mask[k].item() < 0.5:
                    continue
                pred = node_preds[k]
                tgt = future_node[k]

                sp = spearman(pred[:, 1].tolist(), tgt[:, 1].tolist())
                spearman_vals.append(sp)

                wait_score_all.extend(torch.sigmoid(pred[:, 2]).tolist())
                wait_true_all.extend(tgt[:, 2].clamp(0, 1).tolist())

                cong_score_all.extend(pred[:, 5].tolist())
                cong_true_all.extend(
                    [1.0 if v > CONGESTION_THRESHOLD else 0.0
                     for v in tgt[:, 5].tolist()]
                )

    return {
        "density_spearman": sum(spearman_vals) / max(len(spearman_vals), 1),
        "wait_auc": roc_auc(wait_true_all, wait_score_all),
        "congestion_auc": roc_auc(cong_true_all, cong_score_all),
        "n_snapshots": len(spearman_vals),
    }


# =====================================================================
# Phase B — pairwise ranking accuracy
# =====================================================================

def evaluate_ranking(model, pairs):
    if not pairs:
        return {"ranking_accuracy": 0.0, "num_pairs": 0}

    model.eval()
    correct = 0

    with torch.no_grad():
        for pair in pairs:
            sb = pair["sample_i"]
            sw = pair["sample_j"]

            snb = sb.get("station_node_ids")
            if snb is not None:
                snb = snb.tolist()
            snw = sw.get("station_node_ids")
            if snw is not None:
                snw = snw.tolist()

            zb, eb, eab = model.encode_state(
                sb["node_history"], sb["edge_index"],
                sb["edge_features"], sb["demand_context"],
            )
            cb = model.predict_cost(
                zb, eb, eab,
                sb["action_node"], sb["action_global"], sb["edge_index"],
                snb,
            )

            zw, ew, eaw = model.encode_state(
                sw["node_history"], sw["edge_index"],
                sw["edge_features"], sw["demand_context"],
            )
            cw = model.predict_cost(
                zw, ew, eaw,
                sw["action_node"], sw["action_global"], sw["edge_index"],
                snw,
            )

            if cb.item() < cw.item():
                correct += 1

    return {"ranking_accuracy": correct / len(pairs), "num_pairs": len(pairs)}


# =====================================================================
# Phase C — Greedy vs WorldModel throughput comparison
# =====================================================================

def _sim_metrics(engine):
    from WorldState.task_state import TaskStatus
    world = engine.world
    completed_orders = world.order_state.total_completed

    done = [t for t in world.task_state.tasks.values()
            if t.status == TaskStatus.COMPLETED
            and t.created_at is not None
            and t.completed_at is not None]

    if done:
        durations = [t.completed_at - t.created_at for t in done]
        avg_duration = sum(durations) / len(durations)
        excess = []
        for t in done:
            if t.free_flow_time and t.free_flow_time > 0:
                excess.append(max(0, (t.completed_at - t.created_at) - t.free_flow_time))
        avg_excess = sum(excess) / len(excess) if excess else 0.0
    else:
        avg_duration = avg_excess = 0.0

    return {
        "completed_orders": completed_orders,
        "completed_tasks": len(done),
        "avg_task_duration": avg_duration,
        "avg_excess_delay": avg_excess,
    }


def compare_throughput(config_path, checkpoint_path, seed, max_ticks):
    from Config.config_loader import load_config
    from Policies.TaskAssigner import GreedyTaskAssigner, WorldModelTaskAssigner

    results = {}
    for name, ta in [
        ("Greedy", GreedyTaskAssigner()),
        ("WorldModel", WorldModelTaskAssigner(
            checkpoint_path=checkpoint_path, top_m=3)),
    ]:
        cfg = load_config(config_path)
        cfg.simulation.max_ticks = max_ticks
        random.seed(seed)
        engine = _build_engine(cfg, task_assigner=ta)
        engine.run()
        m = _sim_metrics(engine)
        results[name] = m
        print(f"    {name:12s}  orders={m['completed_orders']:3d}  "
              f"tasks={m['completed_tasks']:4d}  "
              f"avg_dur={m['avg_task_duration']:.1f}  "
              f"avg_excess={m['avg_excess_delay']:.1f}")

    g, w = results["Greedy"], results["WorldModel"]
    tp_imp = ((w["completed_orders"] - g["completed_orders"])
              / max(g["completed_orders"], 1))
    dl_imp = ((g["avg_excess_delay"] - w["avg_excess_delay"])
              / max(g["avg_excess_delay"], 0.01))

    results["throughput_improvement"] = tp_imp
    results["delay_improvement"] = dl_imp
    return results


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 62)
    print("  RMFS World Model v3 — Go / No-Go Evaluation")
    print("=" * 62)

    from Config.config_loader import load_config
    from WorldModel.data_collector import WorldModelDataCollector
    from WorldModel.model import RMFSWorldModel
    from WorldModel.dataset import WorldModelDataset
    from WorldModel.train import train as train_model

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Config", "world_model_config.json",
    )
    config = load_config(config_path)
    checkpoint = "DataGen/wm_checkpoints/world_model.pt"

    # ---- 1. train if no checkpoint ----
    if not os.path.exists(checkpoint):
        print("\n[1/6] No checkpoint — training model first ...")
        random.seed(config.simulation.seed)
        engine = _build_engine(config)
        coll = WorldModelDataCollector(
            output_dir="DataGen/wm_data", history_len=4,
            rollout_horizon=10, sample_interval=5, top_m_candidates=5,
            min_group_size=2)
        engine.pre_assignment_callbacks.append(coll.on_pre_assignment)
        engine.on_tick_callbacks.append(coll.on_post_tick)
        engine.run()

        dp = coll.save()

        ds = WorldModelDataset.from_file(dp)
        s0 = ds[0]
        ns = len(engine.world.map_state.station_positions)
        mdl = RMFSWorldModel(
            node_feat_dim=10, edge_feat_dim=6,
            demand_dim=s0["demand_context"].shape[0],
            hidden_dim=64, num_spatial_layers=3,
            rollout_horizon=10, num_stations=ns)
        pw = coll.build_pairwise_data(epsilon=0.01)
        train_model(mdl, ds, epochs=20, lr=1e-3,
                     pairwise_data=pw if pw else None, verbose=True)
        print(f"  Checkpoint saved: {checkpoint}")
    else:
        print(f"\n[1/6] Checkpoint found: {checkpoint}")

    # ---- 2. collect test data (seed=99) ----
    print("\n[2/6] Collecting test data (seed=99, 200 ticks) ...")
    test_seed = 99
    random.seed(test_seed)
    config.simulation.max_ticks = 200
    engine = _build_engine(config)
    coll = WorldModelDataCollector(
        output_dir="DataGen/wm_data", history_len=4,
        rollout_horizon=10, sample_interval=5, top_m_candidates=5,
        min_group_size=2)
    engine.pre_assignment_callbacks.append(coll.on_pre_assignment)
    engine.on_tick_callbacks.append(coll.on_post_tick)
    engine.run()

    test_samples = coll._finalized_samples
    pairwise = coll.build_pairwise_data(epsilon=0.01)
    print(f"  Test samples: {len(test_samples)},  "
          f"Pairwise pairs: {len(pairwise)}")

    if not test_samples:
        print("  ERROR: no test samples. Aborting.")
        sys.exit(1)

    # ---- 3. load model ----
    print("\n[3/6] Loading model ...")
    ns = len(engine.world.map_state.station_positions)
    dd = test_samples[0]["demand_context"].shape[0]
    model, schema = _load_model(checkpoint, fallback_demand_dim=dd,
                                 fallback_num_stations=ns)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded ({n_params:,} params, schema={schema})")

    # ---- 4. prediction quality ----
    print("\n[4/6] Prediction quality ...")
    pq = evaluate_predictions(model, test_samples)
    print(f"  Density Spearman : {pq['density_spearman']:.4f}  "
          f"({pq['n_snapshots']} snapshots)")
    print(f"  Wait AUC         : {pq['wait_auc']:.4f}")
    print(f"  Congestion AUC   : {pq['congestion_auc']:.4f}")

    # ---- 5. ranking accuracy ----
    print("\n[5/6] Ranking accuracy ...")
    rk = evaluate_ranking(model, pairwise)
    print(f"  Accuracy         : {rk['ranking_accuracy']:.4f}  "
          f"({rk['num_pairs']} pairs)")

    # ---- 6. throughput comparison ----
    print("\n[6/6] Greedy vs WorldModel (seed=99, 200 ticks) ...")
    cmp = compare_throughput(config_path, checkpoint, seed=test_seed,
                             max_ticks=200)

    # ================================================================
    # Go / No-Go Table
    # ================================================================
    print("\n" + "=" * 62)
    print("  Go / No-Go Checklist")
    print("=" * 62)

    checks = [
        ("Density Spearman  >= 0.60", pq["density_spearman"], 0.60),
        ("Wait AUC          >= 0.70", pq["wait_auc"],         0.70),
        ("Congestion AUC    >= 0.70", pq["congestion_auc"],   0.70),
        ("Ranking accuracy  >= 0.65", rk["ranking_accuracy"], 0.65),
        ("Throughput impr.  >= 0.10", cmp["throughput_improvement"], 0.10),
    ]

    passed_all = True
    for label, val, thr in checks:
        ok = val >= thr
        tag = "PASS" if ok else "FAIL"
        if not ok:
            passed_all = False
        print(f"  [{tag}]  {label}   actual = {val:+.4f}")

    print(f"\n  Delay improvement: {cmp['delay_improvement']:+.2%}")

    # ---- Verdict ----
    print()
    if passed_all:
        print("  >>> ALL CHECKS PASSED")
    else:
        print("  >>> SOME CHECKS FAILED — suggestions:")
        if pq["density_spearman"] < 0.60:
            print("      - Increase epochs or simulation length (more training data)")
        if pq["wait_auc"] < 0.70:
            print("      - Increase congestion: more robots or shorter order interval")
        if pq["congestion_auc"] < 0.70:
            print("      - Train longer; congestion signal is harder to learn")
        if rk["ranking_accuracy"] < 0.65:
            print("      - Raise alpha_rank (e.g. 0.3) or lower ranking margin")
        if cmp["throughput_improvement"] < 0.10:
            print("      - Use max_ticks>=500 for more meaningful comparison")
            print("      - Train on higher-load scenarios (order_interval=1)")

    print("=" * 62)


if __name__ == "__main__":
    main()
