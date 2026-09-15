"""
World Model Data Generation (reproducible)
============================================
Generates counterfactual training data and saves to DataGen/wm_data/
with full metadata for exact reproduction.

Usage:
    conda activate multi_robot
    python -m WorldModel.generate_data                      # default: 200 ticks, seed=42
    python -m WorldModel.generate_data --ticks 500 --seed 42
    python -m WorldModel.generate_data --ticks 1000 --seed 42 --interval 3

Output files (in DataGen/wm_data/):
    wm_train_data.pt         — training samples
    wm_train_meta.json       — generation parameters + data quality stats (for reproduction)
"""

import argparse
import json
import os
import sys
import random
import time
import inspect
from collections import Counter

import numpy as np

import torch


def main():
    parser = argparse.ArgumentParser(description="Generate world model training data")
    parser.add_argument("--ticks", type=int, default=200, help="Simulation ticks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--interval", type=int, default=None,
                        help="Order interval (overrides config)")
    parser.add_argument("--sample-interval", type=int, default=5,
                        help="Ticks between sampling opportunities")
    parser.add_argument(
        "--reservation-window",
        type=int,
        default=1,
        help="Future path-reservation window used by graph/candidate features.",
    )
    parser.add_argument("--horizon", type=int, default=10,
                        help="Rollout horizon H")
    parser.add_argument("--top-m", type=int, default=5,
                        help="Max candidates per group")
    parser.add_argument("--min-group", type=int, default=2,
                        help="Min candidates to keep a group")
    parser.add_argument("--output-dir", type=str, default="DataGen/wm_data",
                        help="Output directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Config path (default: Config/world_model_config_PP.json)")
    parser.add_argument("--num-robots", type=int, default=None,
                        help="Override number of robots")
    parser.add_argument("--initial-order-pool-size", type=int, default=None,
                        help="Initial order pool size for backlog")
    parser.add_argument("--backlog-floor", type=int, default=None,
                        help="Backlog floor for order refill")
    parser.add_argument("--backlog-refill-mode", type=str, default=None,
                        choices=["none", "on_pre_assignment", "on_completed"],
                        help="Backlog refill mode")
    parser.add_argument("--output-name", type=str, default="wm_train_data.pt",
                        help="Output filename")
    parser.add_argument("--gen-config-name", type=str, default="gen_config.json",
                        help="Generation config filename (saved to output-dir)")
    parser.add_argument("--meta-name", type=str, default=None,
                        help=("Metadata filename. Default: wm_train_meta.json for "
                              "wm_train_data.pt, otherwise derived from --output-name"))
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Abort if output file already exists")
    parser.add_argument("--action-path-mode", type=int, default=0,
                        choices=[0, 1],
                        help="Action path mode: 0=BFS (default), 1=path_planner preview")
    parser.add_argument("--delay-scale", type=float, default=None,
                        help="Normalization scale for avg_excess_delay (default: rollout_horizon)")
    parser.add_argument("--station-queue-delta-scale", type=float, default=None,
                        help="Normalization scale for station_queue_delta (default: num_robots)")
    parser.add_argument("--stalled-ratio-threshold", type=float, default=0.3,
                        help="Threshold for global stall risk (default: 0.3)")
    parser.add_argument("--risk-duration", type=int, default=8,
                        help="Duration for forced robot risk normalization (default: 8)")
    parser.add_argument("--station-queue-scale", type=float, default=None,
                        help="Normalization scale for station queue labels (default: num_robots)")
    parser.add_argument("--station-load-scale", type=float, default=None,
                        help="Normalization scale for station load labels (default: num_robots)")
    parser.add_argument("--assignment-mode", type=str, default="parallel",
                        choices=["parallel", "serial"],
                        help="Task execution mode (default: parallel)")
    parser.add_argument("--max-groups-per-tick", type=int, default=None,
                        help="Max candidate groups per sampling tick (default: all dispatchable)")
    parser.add_argument("--save-snapshots", action="store_true",
                        help="Save decision-point world snapshots for long-risk label generation")
    parser.add_argument("--snapshot-dir", type=str,
                        default="DataGen/wm_data/phaseB_snapshots",
                        help="Directory for snapshot pickle files")
    parser.add_argument("--run-id", type=str, default="",
                        help="Run identifier for snapshot filenames (e.g. seed101)")
    parser.add_argument(
        "--load-level",
        type=str,
        default=None,
        choices=["low", "mid", "high"],
        help="Explicit load arm stored in metadata for balanced fusion/splits.",
    )
    parser.add_argument(
        "--record-lyapunov-l0",
        action="store_true",
        help=("Record analytic L0 start/end/trajectory, productive progress, "
              "and rollout-end demand context for component-head training."),
    )
    parser.add_argument(
        "--lyapunov-config",
        type=str,
        default=None,
        help="Optional JSON object with LyapunovL0Config fields.",
    )
    parser.add_argument(
        "--rollout-continuation",
        type=str,
        default="isolated",
        choices=["isolated", "behavior"],
        help=(
            "Counterfactual continuation after the forced candidate: "
            "isolated keeps the legacy no-new-order/no-assignment rollout; "
            "behavior deep-copies the run's order generator and task assigner "
            "and executes a closed-loop continuation."
        ),
    )
    parser.add_argument(
        "--candidate-robot-mode",
        type=str,
        default="nearest",
        choices=["nearest", "eta_stratified", "stratified"],
        help=(
            "Robot candidate coverage. 'stratified' keeps ETA-bin and "
            "route-conflict extremes in addition to nearest robots."
        ),
    )
    parser.add_argument(
        "--include-no-assign",
        action="store_true",
        help=(
            "Append native NO_ASSIGN after the top-m robot candidates. "
            "The no-op uses all-zero action tensors and is valid only with "
            "isolated rollout."
        ),
    )
    args = parser.parse_args()

    if args.assignment_mode == "serial":
        print("ERROR: order_pod_robot data collection requires --assignment-mode parallel")
        sys.exit(1)
    if args.include_no_assign and args.rollout_continuation != "isolated":
        print("ERROR: --include-no-assign requires --rollout-continuation isolated")
        sys.exit(1)

    print("=" * 60)
    print("World Model Data Generation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Config", "world_model_config_PP.json",
        )

    from Config.config_loader import load_config
    config = load_config(config_path)

    config.simulation.max_ticks = args.ticks
    config.simulation.seed = args.seed
    if args.interval is not None:
        config.simulation.order_interval = args.interval
    if args.num_robots is not None:
        config.robots.num_robots = args.num_robots
    if args.initial_order_pool_size is not None:
        config.simulation.initial_order_pool_size = args.initial_order_pool_size
    if args.backlog_floor is not None:
        config.simulation.backlog_floor = args.backlog_floor
    if args.backlog_refill_mode is not None:
        config.simulation.backlog_refill_mode = args.backlog_refill_mode
    config.simulation.task_execution_mode = args.assignment_mode

    out_path = os.path.join(args.output_dir, args.output_name)
    if args.no_overwrite and os.path.exists(out_path):
        print(f"ERROR: {out_path} already exists (--no-overwrite)")
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n  Config         : {config_path}")
    print(f"  Seed           : {args.seed}")
    print(f"  Ticks          : {args.ticks}")
    print(f"  Robots         : {config.robots.num_robots}")
    print(f"  Order interval : {config.simulation.order_interval}")
    print(f"  Sample interval: {args.sample_interval}")
    print(f"  Reserv. window : {args.reservation_window}")
    print(f"  Horizon H      : {args.horizon}")
    print(f"  Top-m          : {args.top_m}")
    print(f"  Min group size : {args.min_group}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Output name    : {args.output_name}")
    if args.initial_order_pool_size is not None:
        print(f"  Init order pool: {args.initial_order_pool_size}")
    if args.backlog_floor is not None:
        print(f"  Backlog floor  : {args.backlog_floor}")
    if args.backlog_refill_mode is not None:
        print(f"  Backlog mode   : {args.backlog_refill_mode}")
    print(f"  Assign mode    : {args.assignment_mode}")
    if args.max_groups_per_tick is not None:
        print(f"  Max grps/tick  : {args.max_groups_per_tick}")
    if args.load_level is not None:
        print(f"  Load level     : {args.load_level}")
    if args.action_path_mode != 0:
        print(f"  Action path    : {args.action_path_mode} (planner preview)")
    if args.delay_scale is not None:
        print(f"  Delay scale    : {args.delay_scale}")
    if args.station_queue_delta_scale is not None:
        print(f"  SQ delta scale : {args.station_queue_delta_scale}")
    if args.stalled_ratio_threshold != 0.3:
        print(f"  Stall threshold: {args.stalled_ratio_threshold}")
    if args.risk_duration != 8:
        print(f"  Risk duration  : {args.risk_duration}")
    if args.station_queue_scale is not None:
        print(f"  SQ label scale : {args.station_queue_scale}")
    if args.station_load_scale is not None:
        print(f"  SL label scale : {args.station_load_scale}")
    print(f"  Rollout cont.  : {args.rollout_continuation}")
    print(f"  Candidate mode : {args.candidate_robot_mode}")
    print(f"  Native no-op   : {args.include_no_assign}")

    # ------------------------------------------------------------------
    # Build engine
    # ------------------------------------------------------------------
    from Engine.simulation_engine import SimulationEngine
    import Policies  # noqa: F401
    from Policies.policy_registry import get_policy
    from WorldModel.data_collector import (
        WorldModelDataCollector,
        summarize_lyapunov_collection,
    )
    from WorldModel.core.lyapunov import LYAPUNOV_COLLECTION_SCHEMA_VERSION
    from WorldModel.candidate_generator import (
        NO_ASSIGN_ACTION_SCHEMA_VERSION,
        NO_ASSIGN_ACTION_TYPE,
        NO_ASSIGN_ENCODING,
    )

    og_name, og_params = config.policies.order_generator
    ta_name, ta_params = config.policies.task_assigner
    pp_name, pp_params = config.policies.path_planner
    rp_name, rp_params = config.policies.pod_return_planner
    pr_name, pr_params = config.policies.pod_retriever

    OrderGeneratorCls = get_policy("order_generator", og_name)
    TaskAssignerCls = get_policy("task_assigner", ta_name)
    PathPlannerCls = get_policy("path_planner", pp_name)
    PodReturnPlannerCls = get_policy("pod_return_planner", rp_name)
    PodRetrieverCls = get_policy("pod_retriever", pr_name)

    order_generator = OrderGeneratorCls(
        order_interval=config.simulation.order_interval,
        max_items_per_order=config.simulation.max_items_per_order,
        fixed_order_size=config.simulation.fixed_order_size,
        max_items_per_sku=config.simulation.max_items_per_sku,
        **og_params,
    )
    task_assigner = TaskAssignerCls(**ta_params)
    if (config.simulation.seed is not None
            and "seed" not in pp_params
            and "seed" in inspect.signature(PathPlannerCls).parameters):
        pp_params = {**pp_params, "seed": config.simulation.seed}
    path_planner = PathPlannerCls(**pp_params)
    pod_return_planner = PodReturnPlannerCls(**rp_params)
    pod_retriever = PodRetrieverCls(**pr_params)

    task_assigner.pod_return_planner = pod_return_planner
    task_assigner.pod_retriever = pod_retriever
    task_assigner.path_planner = path_planner

    engine = SimulationEngine(
        config=config,
        order_generator=order_generator,
        task_assigner=task_assigner,
        path_planner=path_planner,
        visualizer=None,
    )

    # ------------------------------------------------------------------
    # Attach collector
    # ------------------------------------------------------------------
    lyapunov_l0_config = {}
    if args.lyapunov_config:
        with open(args.lyapunov_config, "r", encoding="utf-8") as handle:
            lyapunov_l0_config = json.load(handle)
        if not isinstance(lyapunov_l0_config, dict):
            raise ValueError("--lyapunov-config must contain a JSON object")

    record_lyapunov_l0 = bool(
        args.record_lyapunov_l0 or args.lyapunov_config
    )
    lyapunov_collection_schema = (
        LYAPUNOV_COLLECTION_SCHEMA_VERSION if record_lyapunov_l0 else None
    )

    collector = WorldModelDataCollector(
        output_dir=args.output_dir,
        history_len=4,
        rollout_horizon=args.horizon,
        sample_interval=args.sample_interval,
        reservation_window=args.reservation_window,
        top_m_candidates=args.top_m,
        min_group_size=args.min_group,
        action_path_mode=args.action_path_mode,
        delay_scale=args.delay_scale,
        station_queue_delta_scale=args.station_queue_delta_scale,
        stalled_ratio_threshold=args.stalled_ratio_threshold,
        risk_duration=args.risk_duration,
        station_queue_scale=args.station_queue_scale,
        station_load_scale=args.station_load_scale,
        max_groups_per_tick=args.max_groups_per_tick,
        save_snapshots=args.save_snapshots,
        snapshot_dir=args.snapshot_dir,
        run_id=args.run_id,
        record_lyapunov_l0=record_lyapunov_l0,
        lyapunov_l0_config=lyapunov_l0_config,
        rollout_continuation_mode=args.rollout_continuation,
        candidate_robot_mode=args.candidate_robot_mode,
        include_no_assign_candidate=args.include_no_assign,
    )
    engine.pre_assignment_callbacks.append(collector.on_pre_assignment)
    engine.on_tick_callbacks.append(collector.on_post_tick)

    _spinner_chars = "|/-\\"
    _gen_t0 = [time.time()]
    _bar_width = 30

    def _progress_cb(eng):
        tick = eng.world.tick
        if tick <= 0:
            return
        total = args.ticks
        pct = tick / total
        elapsed = time.time() - _gen_t0[0]
        eta = elapsed / tick * (total - tick)
        filled = int(_bar_width * pct)
        bar = "#" * filled + "-" * (_bar_width - filled)
        spinner = _spinner_chars[tick % len(_spinner_chars)]
        line = (f"\r  {spinner} [{bar}] {pct:6.1%}  "
                f"tick {tick}/{total}  "
                f"samples={collector.num_samples}  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
        sys.stdout.write(line)
        sys.stdout.flush()

    engine.on_tick_callbacks.append(_progress_cb)

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------
    print(f"\n  Running {args.ticks}-tick simulation...", flush=True)
    t0 = time.time()
    engine.run()
    elapsed = time.time() - t0
    sys.stdout.write("\r" + " " * 120 + "\r")
    sys.stdout.flush()
    print(f"  Simulation done in {elapsed:.1f}s  ({collector.num_samples} samples)")

    # ------------------------------------------------------------------
    # Save data
    # ------------------------------------------------------------------
    data_path = collector.save(filename=args.output_name)
    print(f"\n  Data saved to: {data_path}")
    print(f"  Total samples: {collector.num_samples}")

    # ------------------------------------------------------------------
    # Build & print pairwise stats
    # ------------------------------------------------------------------
    pairwise = collector.build_pairwise_data(epsilon=0.01)
    print(f"  Pairwise pairs: {len(pairwise)}")

    # ------------------------------------------------------------------
    # Save metadata for reproduction
    # ------------------------------------------------------------------
    samples = collector._finalized_samples
    groups = {}
    for s in samples:
        gid = s.get("candidate_group_id", "")
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(s)

    sizes = [len(m) for m in groups.values()]
    size_counter = dict(Counter(sizes))
    cost_stds = []
    for members in groups.values():
        costs = [m["realized_cost"] for m in members]
        if len(costs) >= 2:
            mc = sum(costs) / len(costs)
            cost_stds.append((sum((c - mc) ** 2 for c in costs) / len(costs)) ** 0.5)

    lyapunov_quality = summarize_lyapunov_collection(samples)

    meta = {
        "generation_params": {
            "config_path": os.path.abspath(config_path),
            "seed": args.seed,
            "max_ticks": args.ticks,
            "num_robots": config.robots.num_robots,
            "order_interval": config.simulation.order_interval,
            "sample_interval": args.sample_interval,
            "reservation_window": args.reservation_window,
            "rollout_horizon": args.horizon,
            "top_m_candidates": args.top_m,
            "min_group_size": args.min_group,
            "map_size": f"{config.map.rows}x{config.map.cols}",
            "path_planner": pp_name,
            "task_assigner": ta_name,
            "order_generator": og_name,
            "initial_order_pool_size": config.simulation.initial_order_pool_size,
            "backlog_floor": config.simulation.backlog_floor,
            "backlog_refill_mode": config.simulation.backlog_refill_mode,
            "output_name": args.output_name,
            "run_id": args.run_id,
            "action_path_mode": args.action_path_mode,
            "delay_scale": args.delay_scale,
            "station_queue_delta_scale": args.station_queue_delta_scale,
            "stalled_ratio_threshold": args.stalled_ratio_threshold,
            "risk_duration": args.risk_duration,
            "station_queue_scale": args.station_queue_scale,
            "station_load_scale": args.station_load_scale,
            "assignment_mode": args.assignment_mode,
            # The collector varies robots inside one already-proposed
            # (order, pod, station) context.  Older metadata called this
            # ``order_pod_robot`` even though order/pod were fixed before the
            # candidate group reached the World Model.  Keep the actual scope
            # explicit so these samples cannot certify a pure full-action
            # online policy.
            "action_scope": (
                "robot_or_no_assign_given_proposed_order_pod_context_v1"
                if args.include_no_assign
                else "robot_given_proposed_order_pod_context_v1"
            ),
            "action_schema_version": (
                NO_ASSIGN_ACTION_SCHEMA_VERSION
                if args.include_no_assign else None
            ),
            "supports_no_assign_candidate": bool(args.include_no_assign),
            "no_assign_encoding": (
                NO_ASSIGN_ENCODING if args.include_no_assign else None
            ),
            "candidate_source": "task_assigner_proposed_dispatch_contexts",
            "context_provider_policy": ta_name,
            "max_groups_per_tick": args.max_groups_per_tick,
            "load_level": args.load_level,
            "record_lyapunov_l0": record_lyapunov_l0,
            "lyapunov_l0_collection_schema_version": (
                lyapunov_collection_schema
            ),
            "analytic_work_relief_trajectory_schema_version": (
                "analytic_work_relief_trajectory_v1"
                if record_lyapunov_l0 else None
            ),
            "lyapunov_l0_config": lyapunov_l0_config,
            "lyapunov_config_path": (
                os.path.abspath(args.lyapunov_config)
                if args.lyapunov_config else None
            ),
            "rollout_continuation_mode": args.rollout_continuation,
            "candidate_robot_mode": args.candidate_robot_mode,
            "include_no_assign_candidate": bool(args.include_no_assign),
            "rollout_continuation_policy": (
                f"{TaskAssignerCls.__module__}.{TaskAssignerCls.__name__}"
                if args.rollout_continuation == "behavior"
                else (
                    "isolated_robot_or_native_no_assign"
                    if args.include_no_assign
                    else "isolated_forced_candidate"
                )
            ),
        },
        "data_quality": {
            "total_samples": len(samples),
            "candidate_group_count": len(groups),
            "mean_group_size": round(sum(sizes) / max(len(sizes), 1), 2),
            "group_size_histogram": {str(k): v for k, v in sorted(size_counter.items())},
            "pairwise_pair_count": len(pairwise),
            "realized_cost_std_mean": round(
                sum(cost_stds) / max(len(cost_stds), 1), 4),
            "rollout_vertex_conflicts": sum(
                s.get("rollout_vertex_conflicts", 0) for s in samples),
            "rollout_swap_conflicts": sum(
                s.get("rollout_swap_conflicts", 0) for s in samples),
            "rollout_blocked_moves": sum(
                s.get("rollout_blocked_moves", 0) for s in samples),
            "rollout_generated_orders": sum(
                s.get("rollout_generated_orders", 0) for s in samples),
            "rollout_assigned_tasks": sum(
                s.get("rollout_assigned_tasks", 0) for s in samples),
            "native_no_assign_samples": sum(
                1 for s in samples
                if s.get("action_type") == NO_ASSIGN_ACTION_TYPE
            ),
            "native_no_assign_groups": len({
                s.get("candidate_group_id") for s in samples
                if s.get("action_type") == NO_ASSIGN_ACTION_TYPE
            }),
            "samples_with_conflicts": sum(
                1 for s in samples
                if s.get("rollout_vertex_conflicts", 0) > 0
                or s.get("rollout_swap_conflicts", 0) > 0),
            "lyapunov_collection": lyapunov_quality,
        },
        "simulation_result": {
            "completed_orders": engine.world.order_state.total_completed,
            "total_tasks": len(engine.world.task_state.tasks),
            "elapsed_seconds": round(elapsed, 1),
        },
        "schema_version": (
            "wm_v4_counterfactual_behavior_continuation"
            if args.rollout_continuation == "behavior"
            else "wm_v3_counterfactual"
        ),
        "reproduction_command": (
            f"python -m WorldModel.generate_data"
            f" --config {config_path}"
            f" --ticks {args.ticks}"
            f" --seed {args.seed}"
            f" --interval {config.simulation.order_interval}"
            f" --sample-interval {args.sample_interval}"
            f" --reservation-window {args.reservation_window}"
            f" --horizon {args.horizon}"
            f" --top-m {args.top_m}"
            f" --min-group {args.min_group}"
            f" --num-robots {config.robots.num_robots}"
            f" --output-dir {args.output_dir}"
            f" --output-name {args.output_name}"
            + (f" --initial-order-pool-size {config.simulation.initial_order_pool_size}"
               if config.simulation.initial_order_pool_size else "")
            + (f" --backlog-floor {config.simulation.backlog_floor}"
               if config.simulation.backlog_floor else "")
            + (f" --backlog-refill-mode {config.simulation.backlog_refill_mode}"
               if config.simulation.backlog_refill_mode != "none" else "")
            + (f" --action-path-mode {args.action_path_mode}"
               if args.action_path_mode != 0 else "")
            + (f" --delay-scale {args.delay_scale}"
               if args.delay_scale is not None else "")
            + (f" --station-queue-delta-scale {args.station_queue_delta_scale}"
               if args.station_queue_delta_scale is not None else "")
            + (f" --stalled-ratio-threshold {args.stalled_ratio_threshold}"
               if args.stalled_ratio_threshold != 0.3 else "")
            + (f" --risk-duration {args.risk_duration}"
               if args.risk_duration != 8 else "")
            + (f" --station-queue-scale {args.station_queue_scale}"
               if args.station_queue_scale is not None else "")
            + (f" --station-load-scale {args.station_load_scale}"
               if args.station_load_scale is not None else "")
            + f" --assignment-mode {args.assignment_mode}"
            + (f" --max-groups-per-tick {args.max_groups_per_tick}"
               if args.max_groups_per_tick is not None else "")
            + (f" --load-level {args.load_level}"
               if args.load_level is not None else "")
            + (" --record-lyapunov-l0"
               if args.record_lyapunov_l0 and not args.lyapunov_config else "")
            + (f" --lyapunov-config {args.lyapunov_config}"
               if args.lyapunov_config else "")
            + (f" --rollout-continuation {args.rollout_continuation}"
               if args.rollout_continuation != "isolated" else "")
            + (f" --candidate-robot-mode {args.candidate_robot_mode}"
               if args.candidate_robot_mode != "nearest" else "")
            + (" --include-no-assign" if args.include_no_assign else "")
        ),
    }

    if args.meta_name is not None:
        meta_name = args.meta_name
    elif args.output_name == "wm_train_data.pt":
        meta_name = "wm_train_meta.json"
    else:
        stem, _ = os.path.splitext(args.output_name)
        if stem.startswith("wm_train_data"):
            suffix = stem[len("wm_train_data"):]
            meta_name = f"wm_train_meta{suffix}.json"
        else:
            meta_name = f"{stem}_meta.json"
    meta_path = os.path.join(args.output_dir, meta_name)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved to: {meta_path}")

    # ------------------------------------------------------------------
    # Save generation config (compact, for quick reference)
    # ------------------------------------------------------------------
    gen_config = {
        "seed": args.seed,
        "run_id": args.run_id,
        "ticks": args.ticks,
        "num_robots": config.robots.num_robots,
        "order_interval": config.simulation.order_interval,
        "sample_interval": args.sample_interval,
        "reservation_window": args.reservation_window,
        "horizon": args.horizon,
        "top_m": args.top_m,
        "min_group": args.min_group,
        "assignment_mode": args.assignment_mode,
        "action_scope": (
            "robot_or_no_assign_given_proposed_order_pod_context_v1"
            if args.include_no_assign
            else "robot_given_proposed_order_pod_context_v1"
        ),
        "action_schema_version": (
            NO_ASSIGN_ACTION_SCHEMA_VERSION
            if args.include_no_assign else None
        ),
        "supports_no_assign_candidate": bool(args.include_no_assign),
        "no_assign_encoding": (
            NO_ASSIGN_ENCODING if args.include_no_assign else None
        ),
        "candidate_source": "task_assigner_proposed_dispatch_contexts",
        "context_provider_policy": ta_name,
        "max_groups_per_tick": args.max_groups_per_tick,
        "load_level": args.load_level,
        "action_path_mode": args.action_path_mode,
        "map_size": f"{config.map.rows}x{config.map.cols}",
        "config_path": os.path.abspath(config_path),
        "path_planner": pp_name,
        "task_assigner": ta_name,
        "order_generator": og_name,
        "initial_order_pool_size": config.simulation.initial_order_pool_size,
        "backlog_floor": config.simulation.backlog_floor,
        "backlog_refill_mode": config.simulation.backlog_refill_mode,
        "delay_scale": args.delay_scale,
        "station_queue_delta_scale": args.station_queue_delta_scale,
        "stalled_ratio_threshold": args.stalled_ratio_threshold,
        "risk_duration": args.risk_duration,
        "station_queue_scale": args.station_queue_scale,
        "station_load_scale": args.station_load_scale,
        "rollout_continuation_mode": args.rollout_continuation,
        "candidate_robot_mode": args.candidate_robot_mode,
        "include_no_assign_candidate": bool(args.include_no_assign),
        "record_lyapunov_l0": record_lyapunov_l0,
        "lyapunov_l0_collection_schema_version": lyapunov_collection_schema,
        "analytic_work_relief_trajectory_schema_version": (
            "analytic_work_relief_trajectory_v1"
            if record_lyapunov_l0 else None
        ),
    }
    gen_config_path = os.path.join(args.output_dir, args.gen_config_name)
    with open(gen_config_path, "w", encoding="utf-8") as f:
        json.dump(gen_config, f, indent=2, ensure_ascii=False)
    print(f"  Gen config saved to: {gen_config_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  复现命令:")
    print(f"  {meta['reproduction_command']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
