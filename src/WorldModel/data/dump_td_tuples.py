"""
TD Tuple Dumper (6.2 V0/V2 shared tool)
========================================
Dumps per-candidate TD tuples from decision-point snapshots:

  - risk_seq: per-tick unified_risk over the W-tick closed-loop rollout
    (same source/semantics as the MC long-risk statistics);
  - frames[K]: observation frame at t+K (node_history 4-frame window,
    edge_features, demand_context) rebuilt with the data-collection
    feature builders, so TD residual diagnostics (V0) and TDValueHead
    training (V3) run without further simulation.

Registered spec: td_bootstrap_6p2_plan.md section 4 (V0/V2), D2-D5.

Index alignment (documented, used by V0/V3):
  risk_seq[k] = unified_risk after the (k+1)-th world step post-decision
  (k=0 is the forced-apply tick completion), i.e. c_{t+k}.
  frames[K] is captured right after risk_seq[K-1] is recorded, i.e. the
  state s_{t+K} whose value continues the tail sum_{k>=K} gamma^{k-K} c.
  => y = sum_{k<K} gamma^k * risk_seq[k] + gamma^K * V(frames[K]).

Continuation policy mirrors generate_long_risk_labels.py:
  greedy (default) -> tuples train/evaluate V_td^{G} (diagnostic /
  ablation arm, plan D3). The main arm V_td^{WM} consumes the Phase C
  round-1 online stream; this tool only covers snapshot-based dumps
  (wm option available for offline approximation, cold-start semantics
  as registered for D0).

Cold-start caveats (registered):
  - flow / edge-flow counters restart at rollout t=0 (decay 0.85 =>
    effective memory ~20 ticks). Frames at K >= ~25 are warm.
  - FeatureHistory needs 4 pushes; frames at K >= 4 are fully populated.

Usage:
  python -m WorldModel.data.dump_td_tuples \
    --snapshot-dir DataGen/wm_data/phaseB_b0_aligned/mid/seed43/snapshots \
    --output DataGen/wm_data/td_tuples/mid_seed43_W200.pt \
    --W 200 --frame-ticks 25,50,100
"""

import argparse
import copy
import os
import pickle
import random
import re
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from WorldState.task_state import Task
from WorldState.order_state import Order
from WorldState.risk import compute_unified_risk
from WorldModel.data.counterfactual_rollout import (
    force_apply_candidate, step_world,
)
from WorldModel.data.generate_long_risk_labels import (
    _continuation_tick,
    _make_continuation_assigner,
    select_snapshots,
)

SCHEMA_VERSION = "td_tuples_v1"


# ------------------------------------------------------------------
# Shared helpers (imported by analyze_td_residual_v0 / train_td_value)
# ------------------------------------------------------------------

def load_td_tuples(path: str):
    """Load a td_tuples.pt file -> (header, tuples)."""
    data = torch.load(path, weights_only=False)
    header = data.get("header", {})
    if header.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unexpected td tuples schema: {header.get('schema_version')}"
        )
    return header, data["tuples"]


def build_sample_index(samples: List[dict], run_filter: Optional[str] = None):
    """Index decision-point training samples by their raw candidate_key.

    Fused datasets carry namespaced keys plus the original key in
    `source_candidate_key` (fuse_and_split.namespace_samples); raw per-run
    datasets carry `candidate_key` directly. `run_filter` is a substring
    matched against source_run_id (e.g. "mid_seed43") to disambiguate
    raw-key collisions across loads/seeds in fused data.
    """
    index: Dict[str, dict] = {}
    duplicates = 0
    for s in samples:
        if run_filter is not None:
            if run_filter not in str(s.get("source_run_id", "")):
                continue
        key = s.get("source_candidate_key") or s.get("candidate_key")
        if not key:
            continue
        if key in index:
            duplicates += 1
            continue
        index[key] = s
    return index, duplicates


def match_tuples_to_samples(tuples: List[dict], index: Dict[str, dict]):
    """Join tuples to samples by raw candidate_key."""
    matched, missing = [], 0
    for t in tuples:
        s = index.get(t["candidate_key"])
        if s is None:
            missing += 1
            continue
        matched.append((t, s))
    return matched, missing


def load_frozen_world_model(checkpoint_path: str, sample: dict,
                            device: str = "cpu"):
    """Build + load the frozen base model (mirrors assigner._init)."""
    from WorldModel.model import RMFSWorldModel

    model_config = {
        "node_feat_dim": int(sample["node_history"].shape[-1]),
        "edge_feat_dim": int(sample["edge_features"].shape[-1]),
        "demand_dim": int(sample["demand_context"].shape[0]),
        "action_node_dim": int(sample["action_node"].shape[-1]),
        "action_global_dim": int(sample["action_global"].shape[0]),
        "hidden_dim": 64,
        "num_spatial_layers": 3,
        "rollout_horizon": 10,
        "num_stations": int(len(sample["station_node_ids"])),
    }
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_config" in ckpt:
        saved_cfg = ckpt["model_config"]
        for k in ("node_feat_dim", "edge_feat_dim", "demand_dim",
                  "action_node_dim", "action_global_dim", "hidden_dim",
                  "num_stations", "rollout_horizon"):
            if k in saved_cfg:
                model_config[k] = saved_cfg[k]
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model = RMFSWorldModel(**model_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, model_config


# ------------------------------------------------------------------
# Rollout with frame capture
# ------------------------------------------------------------------

def _rollout_candidate_capture(
    snapshot: dict,
    candidate: dict,
    fixed_context: dict,
    W: int,
    frame_ticks: List[int],
    static_graph,
    continuation_policy: str = "greedy",
    wm_checkpoint: Optional[str] = None,
    reservation_window: int = 1,
) -> Optional[Dict]:
    """Clone world, force-apply candidate, roll W ticks; capture risk_seq
    and observation frames at the requested tick offsets."""
    from WorldModel.graph_builder import (
        extract_node_features, extract_edge_features,
        extract_demand_context, FeatureHistory,
    )

    (edge_index, node_map, inv_node_map, local_capacity,
     bottleneck_score, node_type_arr, adj) = static_graph

    # Restore global state (identical to generate_long_risk_labels)
    Task._next_id = snapshot["task_next_id"]
    Order._next_id = snapshot["order_next_id"]
    random.setstate(snapshot["python_rng_state"])
    np.random.set_state(snapshot["numpy_rng_state"])
    torch.random.set_rng_state(snapshot["torch_rng_state"])

    world = copy.deepcopy(snapshot["world_snapshot"])
    config = snapshot["config"]
    path_planner = copy.deepcopy(snapshot["path_planner_state"])
    order_generator = copy.deepcopy(snapshot["order_generator_state"])

    r_at_decision = compute_unified_risk(world)["unified_risk"]

    ok = force_apply_candidate(world, candidate, fixed_context, config)
    if not ok:
        return None

    # Feature maintenance state (mirrors WorldModelDataCollector.on_post_tick;
    # counters start cold at rollout t=0 — registered approximation)
    fh = FeatureHistory(len(node_map), feat_dim=10,
                        history_len=4)
    flow_counter: Dict[int, float] = {}
    edge_flow_counter: Dict[tuple, float] = {}
    last_edge_tick = [-1]

    def _post_step():
        for nid in flow_counter:
            flow_counter[nid] *= 0.85
        for agent in world.agents:
            nid = node_map.get(agent.position)
            if nid is not None:
                flow_counter[nid] = flow_counter.get(nid, 0.0) + 1.0
        if world.tick != last_edge_tick[0]:
            last_edge_tick[0] = world.tick
            for key in edge_flow_counter:
                edge_flow_counter[key] *= 0.85
            for agent in world.agents:
                if agent.moved_this_tick and agent.previous_position != agent.position:
                    a = node_map.get(agent.previous_position)
                    b = node_map.get(agent.position)
                    if a is not None and b is not None:
                        edge_flow_counter[(a, b)] = (
                            edge_flow_counter.get((a, b), 0.0) + 1.0
                        )
        nf = extract_node_features(
            world, node_map, local_capacity, bottleneck_score,
            node_type_arr, adj, flow_counter,
            reservation_window=reservation_window,
        )
        fh.push(nf)

    def _capture():
        return {
            "node_history": fh.get_history().clone(),
            "edge_features": extract_edge_features(
                edge_index, node_map, inv_node_map, local_capacity, world,
                adj=adj, edge_flow_counter=edge_flow_counter,
                reservation_window=reservation_window,
            ),
            "demand_context": extract_demand_context(world),
        }

    frame_set = set(frame_ticks)
    frames: Dict[int, dict] = {}
    risk_series: List[float] = []

    # t=0: remainder of current tick WITHOUT order generation
    step_world(world, path_planner, config)
    risk_series.append(compute_unified_risk(world)["unified_risk"])
    _post_step()
    if len(risk_series) in frame_set:
        frames[len(risk_series)] = _capture()

    # t=1..W: closed-loop continuation
    assigner = _make_continuation_assigner(
        continuation_policy, path_planner, world, wm_checkpoint=wm_checkpoint,
    )
    for _ in range(W - 1):
        _continuation_tick(world, order_generator, assigner, path_planner, config)
        risk_series.append(compute_unified_risk(world)["unified_risk"])
        _post_step()
        if len(risk_series) in frame_set:
            frames[len(risk_series)] = _capture()

    out = {
        "risk_seq": torch.tensor(risk_series, dtype=torch.float32),
        "frames": frames,
        "r_at_decision": float(r_at_decision),
        "continuation_policy": continuation_policy,
    }
    if continuation_policy == "wm":
        s = assigner.stats
        out["wm_rollout_stats"] = {
            k: s.get(k, 0) for k in (
                "assign_calls", "model_assign_calls", "fallback_greedy_calls",
                "energy_conv_contexts", "energy_conv_warmup_contexts",
                "energy_conv_active_contexts", "energy_conv_modified_decisions",
            )
        }
    return out


def process_snapshot_tuples(
    snap_path: str,
    W: int,
    frame_ticks: List[int],
    continuation_policy: str = "greedy",
    wm_checkpoint: Optional[str] = None,
    reservation_window: int = 1,
):
    """One snapshot file -> (list of tuples, static edge_index)."""
    from WorldModel.graph_builder import build_static_graph

    with open(snap_path, "rb") as f:
        snapshot = pickle.load(f)

    static_graph = build_static_graph(snapshot["world_snapshot"].map_state)
    edge_index = static_graph[0]

    group_id = snapshot["candidate_group_id"]
    tick = snapshot["decision_tick"]
    fc = snapshot["fixed_context"]

    results = []
    for cand in snapshot["candidates"]:
        robot_id = cand["robot_id"]
        candidate_key = (
            f"{group_id}"
            f"_r{robot_id}"
            f"_o{fc['order_id']}"
            f"_p{fc['pod_id']}"
            f"_s{fc['station_id']}"
            f"_t{tick}"
        )
        r = _rollout_candidate_capture(
            snapshot, cand, fc, W, frame_ticks, static_graph,
            continuation_policy=continuation_policy,
            wm_checkpoint=wm_checkpoint,
            reservation_window=reservation_window,
        )
        if r is None:
            continue
        results.append({
            "candidate_key": candidate_key,
            "candidate_group_id": group_id,
            "robot_id": robot_id,
            "decision_tick": tick,
            "run_id": snapshot.get("run_id", ""),
            **r,
        })
    return results, edge_index


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dump TD tuples (risk_seq + t+K frames) from snapshots")
    parser.add_argument("--snapshot-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="td_tuples.pt")
    parser.add_argument("--W", type=int, default=200,
                        help="Rollout horizon (default: 200)")
    parser.add_argument("--frame-ticks", type=str, default="25,50,100",
                        help="Comma-separated t+K frame capture offsets "
                             "(default: 25,50,100; K=50 is the D4 default)")
    parser.add_argument("--continuation-policy", type=str, default="greedy",
                        choices=["greedy", "wm"])
    parser.add_argument("--wm-checkpoint", type=str, default=None)
    parser.add_argument("--reservation-window", type=int, default=1,
                        help="Must match data-collection setting (default: 1)")
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--snapshot-sampling", type=str, default="first",
                        choices=["first", "uniform", "random", "stratified"])
    parser.add_argument("--snapshot-seed", type=int, default=42)
    parser.add_argument("--num-strata", type=int, default=3)
    args = parser.parse_args()

    frame_ticks = sorted({int(x) for x in args.frame_ticks.split(",") if x})
    dropped = [k for k in frame_ticks if k > args.W or k < 4]
    frame_ticks = [k for k in frame_ticks if 4 <= k <= args.W]
    if dropped:
        print(f"  WARNING: frame ticks {dropped} outside [4, W]; dropped")
    if not frame_ticks:
        print("ERROR: no valid frame ticks")
        sys.exit(1)

    def _snap_sort_key(fname):
        m = re.search(r"tick(\d+)", fname)
        return int(m.group(1)) if m else 0

    snap_files = sorted(
        [f for f in os.listdir(args.snapshot_dir) if f.endswith(".pkl")],
        key=_snap_sort_key,
    )
    snap_files = select_snapshots(
        snap_files, args.max_snapshots,
        mode=args.snapshot_sampling, seed=args.snapshot_seed,
        num_strata=args.num_strata,
    )

    print("TD Tuple Dumper")
    print(f"  snapshot_dir      : {args.snapshot_dir}")
    print(f"  snapshots selected: {len(snap_files)}")
    print(f"  W                 : {args.W}")
    print(f"  frame_ticks       : {frame_ticks}")
    print(f"  continuation      : {args.continuation_policy}")
    print()

    all_tuples = []
    edge_index = None
    t0 = time.time()
    for i, snap_file in enumerate(snap_files):
        t_snap = time.time()
        tuples, ei = process_snapshot_tuples(
            os.path.join(args.snapshot_dir, snap_file),
            args.W, frame_ticks,
            continuation_policy=args.continuation_policy,
            wm_checkpoint=args.wm_checkpoint,
            reservation_window=args.reservation_window,
        )
        if edge_index is None:
            edge_index = ei
        elif edge_index.shape != ei.shape:
            print(f"\nERROR: edge_index shape mismatch in {snap_file}")
            sys.exit(1)
        all_tuples.extend(tuples)
        elapsed = time.time() - t_snap
        total = time.time() - t0
        eta = total / (i + 1) * (len(snap_files) - i - 1)
        sys.stdout.write(
            f"\r  [{i+1}/{len(snap_files)}] {snap_file}  "
            f"cands={len(tuples)}  snap_time={elapsed:.1f}s  "
            f"total={total:.0f}s  ETA={eta:.0f}s"
        )
        sys.stdout.flush()

    print(f"\n\nDone. Total tuples: {len(all_tuples)}")
    if all_tuples:
        rs = torch.stack([t["risk_seq"] for t in all_tuples])
        print(f"  risk_seq: shape={tuple(rs.shape)}  "
              f"mean={rs.mean():.4f}  max={rs.max():.4f}")
        f0 = all_tuples[0]["frames"][frame_ticks[0]]
        print(f"  frame[{frame_ticks[0]}]: node_history="
              f"{tuple(f0['node_history'].shape)}  "
              f"edge_features={tuple(f0['edge_features'].shape)}  "
              f"demand={tuple(f0['demand_context'].shape)}")

    header = {
        "schema_version": SCHEMA_VERSION,
        "W": args.W,
        "frame_ticks": frame_ticks,
        "continuation_policy": args.continuation_policy,
        "reservation_window": args.reservation_window,
        "snapshot_dir": args.snapshot_dir,
        "edge_index": edge_index,
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save({"header": header, "tuples": all_tuples}, args.output)
    print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()
