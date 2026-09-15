"""
Long-Risk Label Generator (Phase B2)
=====================================
Generates W-step closed-loop long-risk labels from decision-point snapshots.

Input:  snapshot pickle files from data collection (B0.7)
Output: long_risk_labels.pt — per-candidate long-horizon risk statistics

Usage:
  python -m WorldModel.data.generate_long_risk_labels \
    --snapshot-dir DataGen/wm_data/phaseB_snapshots \
    --output long_risk_labels.pt \
    --W 100 --terminal-window 50

D0 dual-continuation diagnostic (td_bootstrap_6p2_plan.md, section D0):
  --continuation-policy wm swaps the t=1..W closed-loop continuation from
  Greedy to the frozen A1 arm (GATED_C3_L025). Run both policies on the
  same snapshots (two output files) to quantify the pi_ref gap:
  label_G = Q^{Greedy}(s,a) vs label_WM = Q^{pi_WM}(s,a).
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
    _validate_no_assign_candidate as validate_no_assign_candidate,
    force_apply_candidate,
    step_world,
)
from WorldModel.data.candidate_generator import is_no_assign_candidate


# ------------------------------------------------------------------
# Closed-loop continuation tick (order gen + assign + step)
# ------------------------------------------------------------------

def _continuation_tick(world, order_generator, assigner, path_planner, config):
    """One full tick with order generation + task assignment + step_world.

    Mirrors SimulationEngine._tick() for closed-loop rollout, including
    both on_pre_assignment and on_completed backlog refill modes.
    Works with any assigner exposing .assign(world) (Greedy or WM arm).
    """
    tick = world.tick

    # Order generation
    new_orders = order_generator.generate(world)
    for order in new_orders:
        world.order_state.add_order(order)

    # Backlog refill (pre-assignment)
    mode = config.simulation.backlog_refill_mode
    if mode == "on_pre_assignment":
        _refill_backlog(world, config, order_generator, tick)

    # Task assignment via continuation policy
    new_tasks = assigner.assign(world)
    for task in new_tasks:
        if task.created_at is None:
            task.created_at = tick
        if task.agent_id is not None and task.assigned_at is None:
            task.assigned_at = tick

    # Step world (station cascade, plan, move, actions, completion check, etc.)
    step_world(world, path_planner, config)

    # Backlog refill (on_completed) — tick counter already advanced by step_world
    if mode == "on_completed":
        _refill_backlog(world, config, order_generator, tick)


def _refill_backlog(world, config, order_generator, tick):
    """Replicate SimulationEngine._refill_order_backlog."""
    floor = config.simulation.backlog_floor
    if floor is None or floor <= 0:
        return
    pending = sum(1 for o in world.order_state.orders.values()
                  if o.status.name == "PENDING")
    while pending < floor:
        if hasattr(order_generator, "generate_one"):
            order = order_generator.generate_one(world, created_at=tick)
        else:
            orders = order_generator.generate(world)
            order = orders[0] if orders else None
        if order is None:
            break
        world.order_state.add_order(order)
        pending += 1


# ------------------------------------------------------------------
# Continuation assigner factory (D0 dual-continuation diagnostic)
# ------------------------------------------------------------------

# Frozen A1 arm (GATED_C3_L025) construction params. Must match the S4 run
# meta (energy_conversion_s4/online_2000_mid_s4_gated.json) and Record6_1.md
# frozen CLI exactly. Deliberately NOT exposed as CLI knobs: D0 measures the
# gap against the frozen deployed arm, not a tunable variant.
_A1_CHECKPOINT = (
    "WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered_decoder/"
    "best_regret_world_model.pt"
)
_A1_PARAMS = dict(
    top_m=10,
    long_risk_beta={"cvar": 0.2, "terminal": 0.5},
    energy_scoring_mode="conversion",
    energy_potential_form="endpoint",
    energy_discount=0.95,
    energy_drift_signal="combo",
    energy_gate_mode="sigmoid",
    energy_gate_window=200,
    energy_gate_warmup=50,
    energy_gate_quantile=0.80,
    energy_conv_lambda=0.25,
)


def _make_continuation_assigner(policy, path_planner, world, wm_checkpoint=None):
    """Build a fresh continuation assigner for one candidate rollout.

    greedy: original pipeline, unchanged.
    wm: frozen A1 conversion arm. Construction + lazy _init are wrapped in a
        full RNG save/restore because RMFSWorldModel.__init__ randomly
        initializes weights (consuming torch RNG) before load_state_dict
        overwrites them — without the guard the WM arm's world-evolution RNG
        stream would diverge from the Greedy arm for non-policy reasons.
        Each rollout starts cold by design (FeatureHistory needs 4 frames ->
        greedy fallback for the first ~4 assign calls; gate warmup 50
        contexts -> conversion inactive early). Registered D0 semantics.
    """
    from Policies.TaskAssigner import GreedyTaskAssigner

    if policy == "greedy":
        ta = GreedyTaskAssigner()
        ta.path_planner = path_planner
        return ta

    if policy == "wm":
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            from Policies.TaskAssigner import WorldModelTaskAssigner

            ta = WorldModelTaskAssigner(
                checkpoint_path=wm_checkpoint or _A1_CHECKPOINT,
                **_A1_PARAMS,
            )
            ta.path_planner = path_planner
            ta._init(world)  # trigger lazy init under the RNG guard
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
        return ta

    raise ValueError(f"unknown continuation policy: {policy}")


# ------------------------------------------------------------------
# Per-candidate rollout
# ------------------------------------------------------------------

def _rollout_candidate(
    snapshot: dict,
    candidate: dict,
    fixed_context: dict,
    W: int,
    terminal_window: int,
    continuation_policy: str = "greedy",
    wm_checkpoint: Optional[str] = None,
) -> Optional[Dict]:
    """Clone world, apply/defer one candidate, and rollout W ticks.

    t=0: apply the robot candidate (or validate NO_ASSIGN) + remainder of
         the current tick (no order generation)
    t=1..W: closed-loop continuation under continuation_policy

    Returns dict of risk statistics, or None if candidate is invalid.
    """
    # Restore global state
    Task._next_id = snapshot["task_next_id"]
    Order._next_id = snapshot["order_next_id"]
    random.setstate(snapshot["python_rng_state"])
    np.random.set_state(snapshot["numpy_rng_state"])
    torch.random.set_rng_state(snapshot["torch_rng_state"])

    world = copy.deepcopy(snapshot["world_snapshot"])
    config = snapshot["config"]
    path_planner = copy.deepcopy(snapshot["path_planner_state"])
    order_generator = copy.deepcopy(snapshot["order_generator_state"])

    # Compute risk at decision time (BEFORE apply/defer; same for all
    # candidates in a group).
    r_at_decision = compute_unified_risk(world)["unified_risk"]

    # Native NO_ASSIGN means that this context is deliberately deferred for
    # the current tick.  It must be validated without creating a task chain:
    # ``force_apply_candidate`` expects a real robot id and would otherwise
    # call ``world.get_agent(None)``.  The normal continuation loop below then
    # gives the policy a chance to reconsider the context on the next tick.
    if is_no_assign_candidate(candidate):
        ok = validate_no_assign_candidate(world, fixed_context)
    else:
        ok = force_apply_candidate(world, candidate, fixed_context, config)
    if not ok:
        return None

    risk_series = []

    # t=0: remainder of current tick WITHOUT order generation
    step_world(world, path_planner, config)
    risk_series.append(compute_unified_risk(world)["unified_risk"])

    # t=1..W: closed-loop continuation
    assigner = _make_continuation_assigner(
        continuation_policy, path_planner, world, wm_checkpoint=wm_checkpoint,
    )

    for _ in range(W - 1):
        _continuation_tick(world, order_generator, assigner, path_planner, config)
        risk_series.append(compute_unified_risk(world)["unified_risk"])

    # Per-rollout WM-arm stats: quantify how much of the continuation actually
    # ran under model scoring / active conversion (cold-start caveat: gate
    # warmup=50 contexts may exceed the contexts accrued within one W-tick
    # rollout, in which case conversion never fires and the WM continuation
    # degenerates to the base WM arm).
    wm_rollout_stats = None
    if continuation_policy == "wm":
        s = assigner.stats
        wm_rollout_stats = {
            k: s.get(k, 0) for k in (
                "assign_calls", "model_assign_calls", "fallback_greedy_calls",
                "energy_conv_contexts", "energy_conv_warmup_contexts",
                "energy_conv_active_contexts", "energy_conv_modified_decisions",
            )
        }

    # Compute labels
    risk_arr = np.array(risk_series)
    risk_peak = float(risk_arr.max())
    top_k = max(1, len(risk_arr) // 10)
    risk_cvar = float(np.sort(risk_arr)[-top_k:].mean())

    tw = min(terminal_window, len(risk_arr))
    risk_terminal = float(risk_arr[-tw:].mean()) if tw > 0 else 0.0

    # Event: peak >= 1.0 OR 5+ consecutive ticks with risk >= 0.7
    consecutive = 0
    max_consecutive = 0
    for r in risk_series:
        if r >= 0.7:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    risk_event = 1.0 if (risk_peak >= 1.0 or max_consecutive >= 5) else 0.0

    risk_delta_current = risk_peak - r_at_decision

    return {
        "risk_peak": risk_peak,
        "risk_cvar": risk_cvar,
        "risk_terminal": risk_terminal,
        "risk_event": risk_event,
        "risk_delta_current": risk_delta_current,
        "r_at_decision": r_at_decision,
        **({"wm_rollout_stats": wm_rollout_stats}
           if wm_rollout_stats is not None else {}),
    }


# ------------------------------------------------------------------
# Process one snapshot (one candidate group)
# ------------------------------------------------------------------

def process_snapshot(
    snap_path: str,
    W: int,
    terminal_window: int,
    continuation_policy: str = "greedy",
    wm_checkpoint: Optional[str] = None,
) -> List[Dict]:
    """Process one snapshot file → list of candidate labels."""
    with open(snap_path, "rb") as f:
        snapshot = pickle.load(f)

    group_id = snapshot["candidate_group_id"]
    tick = snapshot["decision_tick"]
    candidates = snapshot["candidates"]
    fc = snapshot["fixed_context"]

    results = []
    for cand in candidates:
        robot_id = cand["robot_id"]
        robot_token = (
            "NO_ASSIGN" if is_no_assign_candidate(cand) else str(robot_id)
        )
        candidate_key = (
            f"{group_id}"
            f"_r{robot_token}"
            f"_o{fc['order_id']}"
            f"_p{fc['pod_id']}"
            f"_s{fc['station_id']}"
            f"_t{tick}"
        )

        r = _rollout_candidate(
            snapshot, cand, fc, W, terminal_window,
            continuation_policy=continuation_policy,
            wm_checkpoint=wm_checkpoint,
        )
        if r is None:
            continue

        results.append({
            "candidate_group_id": group_id,
            "candidate_key": candidate_key,
            "robot_id": robot_id,
            "decision_tick": tick,
            "continuation_policy": continuation_policy,
            "long_risk_horizon": int(W),
            "long_risk_terminal_window": int(terminal_window),
            "source_path_planner": snapshot.get("path_planner_override"),
            "source_path_planner_params": snapshot.get(
                "path_planner_params_override"
            ),
            **r,
        })

    # Post-hoc: compute risk_delta_group
    if len(results) >= 2:
        peaks = [r["risk_peak"] for r in results]
        median_peak = float(np.median(peaks))
        for r in results:
            r["risk_delta_group"] = r["risk_peak"] - median_peak
    else:
        for r in results:
            r["risk_delta_group"] = 0.0

    return results


# ------------------------------------------------------------------
# Snapshot sampling
# ------------------------------------------------------------------

def select_snapshots(snap_files, max_snapshots, mode="first", seed=42, num_strata=3):
    """Select a subset of snapshot files for pilot or full generation."""
    if max_snapshots is None or max_snapshots >= len(snap_files):
        return snap_files

    n = max_snapshots
    if mode == "first":
        return snap_files[:n]

    if mode == "uniform":
        idx = np.linspace(0, len(snap_files) - 1, n, dtype=int)
        return [snap_files[i] for i in idx]

    if mode == "random":
        rng = random.Random(seed)
        chosen = set(rng.sample(snap_files, n))
        return [f for f in snap_files if f in chosen]

    if mode == "stratified":
        selected = []
        strata = np.array_split(snap_files, num_strata)
        base = n // num_strata
        rem = n % num_strata
        rng = random.Random(seed)
        for si, bucket in enumerate(strata):
            k = base + (1 if si < rem else 0)
            bucket = list(bucket)
            if len(bucket) <= k:
                selected.extend(bucket)
            else:
                selected.extend(rng.sample(bucket, k))
        chosen = set(selected)
        return [f for f in snap_files if f in chosen]

    raise ValueError(f"unknown snapshot sampling mode: {mode}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate long-risk labels from decision-point snapshots"
    )
    parser.add_argument("--snapshot-dir", type=str, required=True,
                        help="Directory containing snapshot .pkl files")
    parser.add_argument("--output", type=str, default="long_risk_labels.pt",
                        help="Output file path")
    parser.add_argument("--W", type=int, default=100,
                        help="Rollout horizon W (default: 100 for pilot)")
    parser.add_argument("--terminal-window", type=int, default=50,
                        help="Terminal window for risk_terminal (default: 50)")
    parser.add_argument("--max-snapshots", type=int, default=None,
                        help="Max snapshots to process (for pilot)")
    parser.add_argument("--snapshot-sampling", type=str, default="first",
                        choices=["first", "uniform", "random", "stratified"],
                        help="Snapshot selection strategy (default: first)")
    parser.add_argument("--snapshot-seed", type=int, default=42,
                        help="Seed for random/stratified sampling (default: 42)")
    parser.add_argument("--num-strata", type=int, default=3,
                        help="Number of strata for stratified sampling (default: 3)")
    parser.add_argument("--continuation-policy", type=str, default="greedy",
                        choices=["greedy", "wm"],
                        help="t=1..W closed-loop continuation policy: greedy "
                             "(original pipeline) or wm (frozen A1 arm, D0 "
                             "dual-continuation diagnostic)")
    parser.add_argument("--wm-checkpoint", type=str, default=None,
                        help="Checkpoint for --continuation-policy wm "
                             f"(default: {_A1_CHECKPOINT})")
    args = parser.parse_args()

    if args.continuation_policy == "wm":
        ckpt = args.wm_checkpoint or _A1_CHECKPOINT
        if not os.path.exists(ckpt):
            print(f"ERROR: wm checkpoint not found: {ckpt}")
            sys.exit(1)

    def _snap_sort_key(fname):
        m = re.search(r'tick(\d+)', fname)
        return int(m.group(1)) if m else 0

    snap_files = sorted(
        [f for f in os.listdir(args.snapshot_dir) if f.endswith(".pkl")],
        key=_snap_sort_key,
    )
    snap_files = select_snapshots(
        snap_files, args.max_snapshots,
        mode=args.snapshot_sampling,
        seed=args.snapshot_seed,
        num_strata=args.num_strata,
    )

    print(f"Long-Risk Label Generator")
    print(f"  snapshot_dir     : {args.snapshot_dir}")
    print(f"  snapshots selected: {len(snap_files)}")
    print(f"  sampling         : {args.snapshot_sampling}")
    print(f"  W                : {args.W}")
    print(f"  terminal_window  : {args.terminal_window}")
    print(f"  continuation     : {args.continuation_policy}")
    if args.continuation_policy == "wm":
        print(f"  wm_checkpoint    : {args.wm_checkpoint or _A1_CHECKPOINT}")
    print()

    all_labels = []
    t0 = time.time()

    for i, snap_file in enumerate(snap_files):
        snap_path = os.path.join(args.snapshot_dir, snap_file)
        t_snap = time.time()

        labels = process_snapshot(
            snap_path, args.W, args.terminal_window,
            continuation_policy=args.continuation_policy,
            wm_checkpoint=args.wm_checkpoint,
        )
        all_labels.extend(labels)

        elapsed = time.time() - t_snap
        total_elapsed = time.time() - t0
        eta = total_elapsed / (i + 1) * (len(snap_files) - i - 1)
        sys.stdout.write(
            f"\r  [{i+1}/{len(snap_files)}] "
            f"{snap_file}  cands={len(labels)}  "
            f"snap_time={elapsed:.1f}s  "
            f"total={total_elapsed:.0f}s  ETA={eta:.0f}s"
        )
        sys.stdout.flush()

    print(f"\n\nDone. Total labels: {len(all_labels)}")

    # Print pilot diagnostics
    if all_labels:
        peaks = [l["risk_peak"] for l in all_labels]
        events = [l["risk_event"] for l in all_labels]
        deltas = [l["risk_delta_group"] for l in all_labels]

        print(f"\n  === Pilot Diagnostics ===")
        print(f"  total candidates     : {len(all_labels)}")
        print(f"  risk_peak   mean={np.mean(peaks):.4f}  "
              f"std={np.std(peaks):.4f}  "
              f"min={np.min(peaks):.4f}  max={np.max(peaks):.4f}")
        print(f"  risk_event  rate={np.mean(events):.4f}")
        print(f"  risk_delta_group  "
              f"mean={np.mean(deltas):.4f}  std={np.std(deltas):.4f}")
        print(f"  non_zero_peak       : {sum(1 for p in peaks if p > 0.01)}/{len(peaks)}")

        # Per-group intra-group std
        from collections import defaultdict
        groups = defaultdict(list)
        for l in all_labels:
            groups[l["candidate_group_id"]].append(l["risk_peak"])
        intra_stds = [np.std(v) for v in groups.values() if len(v) >= 2]
        if intra_stds:
            print(f"  intra-group peak std: "
                  f"mean={np.mean(intra_stds):.4f}  "
                  f"median={np.median(intra_stds):.4f}  "
                  f">{0.02:.0%}: {sum(1 for s in intra_stds if s > 0.02)}/{len(intra_stds)}")

    # Save
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(all_labels, args.output)
    print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()
