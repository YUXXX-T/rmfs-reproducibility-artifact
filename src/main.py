"""
MAS-RMFS：多智能体机器人移动履行系统仿真
======================================================================
仿真程序入口。

用法：
    python main.py                          # 使用默认配置
    python main.py --config path/to/cfg.json  # 使用自定义配置
    python main.py --visualize              # 启用终端可视化
    python main.py --mpl                    # 启用 matplotlib 仪表盘
"""

import argparse
import os
import sys

from Config.config_loader import load_config, SimulationConfig
from Engine.simulation_engine import SimulationEngine
import Policies  # noqa: F401 — 触发算法自动注册
from Policies.policy_registry import get_policy
from Visualization.visualizer import TerminalVisualizer, MatplotlibVisualizer
from Metrics import SnapshotCollector
from Debug.logger import SimLogger


def _run_benchmark(config, logger):
    """Run MovingAI MAPF benchmark using config.benchmark settings."""
    import random
    from Benchmarks.movingai_loader import MovingAILoader
    from Benchmarks.mapf_runner import MAPFRunner

    bm = config.benchmark
    if not bm.map_path:
        logger.error("benchmark.map_path is required. Set it in the config JSON.")
        sys.exit(1)

    loader = MovingAILoader()
    map_state = loader.load_map(bm.map_path)
    logger.info(f"Loaded map: {bm.map_path} ({map_state.rows}x{map_state.cols})")

    # Build agent start/goal list
    if bm.scen_path:
        scenarios = loader.load_scenario(bm.scen_path)
        agents_with_goals = scenarios[:bm.num_agents]
        logger.info(f"Loaded scenario: {bm.scen_path} ({len(agents_with_goals)} agents)")
    else:
        rng = random.Random(bm.random_seed)
        free_cells = [
            (r, c)
            for r in range(map_state.rows)
            for c in range(map_state.cols)
            if map_state.is_walkable(r, c)
        ]
        if len(free_cells) < bm.num_agents * 2:
            logger.error(
                f"Not enough free cells ({len(free_cells)}) for "
                f"{bm.num_agents} agents (need {bm.num_agents * 2} for start+goal)."
            )
            sys.exit(1)
        sampled = rng.sample(free_cells, bm.num_agents * 2)
        agents_with_goals = [
            {"start": sampled[i], "goal": sampled[bm.num_agents + i]}
            for i in range(bm.num_agents)
        ]
        logger.info(f"Generated {bm.num_agents} random agents (seed={bm.random_seed})")

    # Instantiate path planner from policies config
    pp_name, pp_params = config.policies.path_planner
    PathPlannerCls = get_policy("path_planner", pp_name)
    if (config.simulation.seed is not None
            and "seed" not in pp_params
            and "seed" in __import__("inspect").signature(PathPlannerCls).parameters):
        pp_params = {**pp_params, "seed": config.simulation.seed}
    path_planner = PathPlannerCls(**pp_params)
    logger.info(f"Path planner: {pp_name}")

    # Run
    runner = MAPFRunner(map_state, agents_with_goals, path_planner, max_ticks=bm.max_ticks)

    snapshot_collector = None
    if config.snapshot.enabled:
        map_label = os.path.splitext(os.path.basename(bm.map_path))[0]
        episode_id = SnapshotCollector.resolve_episode_id(
            config.snapshot.episode_id,
            {
                "map": map_label,
                "planner": pp_name,
                "assigner": "none",
                "robots": str(bm.num_agents),
                "agents": str(bm.num_agents),
            },
        )
        snapshot_collector = SnapshotCollector(
            output_dir=config.snapshot.output_dir,
            agent_goals_override=runner.goals,
        )
        snapshot_collector.start_episode(episode_id)
        snapshot_collector.write_header(
            runner.world,
            extra={
                "mode": "mapf_benchmark",
                "map_path": bm.map_path,
                "scen_path": bm.scen_path,
                "planner": {"name": pp_name, "params": pp_params},
                "num_agents": bm.num_agents,
                "max_ticks": bm.max_ticks,
                "random_seed": bm.random_seed,
            },
        )
        runner.snapshot_collector = snapshot_collector
        logger.info(f"Snapshot collection enabled → {config.snapshot.output_dir}/{episode_id}.jsonl")

    result = runner.run()

    if snapshot_collector is not None:
        snapshot_collector.end_episode()

    # Print results
    logger.info("=" * 60)
    logger.info("MAPF BENCHMARK RESULTS")
    logger.info(f"  Map:              {bm.map_path}")
    logger.info(f"  Planner:          {pp_name}")
    logger.info(f"  Agents:           {result['total_agents']}")
    logger.info(f"  Completed:        {result['completed_agents']}/{result['total_agents']}")
    logger.info(f"  Success rate:     {result['success_rate']:.2%}")
    logger.info(f"  Makespan:         {result['makespan']} ticks")
    logger.info(f"  Total conflicts:  {result['total_conflicts']}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="MAS-RMFS：多智能体机器人移动履行系统仿真"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "Config", "default_config.json"),
        help="JSON 配置文件路径。",
    )

    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument(
        "--visualize",
        action="store_true",
        help="每个 tick 启用基于终端的 ASCII 可视化。",
    )
    viz_group.add_argument(
        "--mpl",
        action="store_true",
        help="启用 matplotlib 2×2 动态仪表盘。",
    )
    viz_group.add_argument(
        "--p3d",
        action="store_true",
        help="启用 Panda3D 2D 正交可视化。",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="运行 MovingAI MAPF benchmark 模式（纯路径规划，无订单/Pod）。",
    )

    args = parser.parse_args()

    # --- 加载配置 ---
    logger = SimLogger("Main")
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # --- 全局随机种子（影响 ZipfOrderGenerator / RandomOrderGenerator /
    #     DefaultPodInitializer / benchmark 随机 start-goal 等所有用 random.* / np.random.* 的模块）---
    if config.simulation.seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(config.simulation.seed)
        _np.random.seed(config.simulation.seed)
        logger.info(f"Global seed set: {config.simulation.seed}")

    # --- Benchmark 模式 ---
    if args.benchmark:
        _run_benchmark(config, logger)
        return

    # --- 从配置实例化策略 ---
    og_name, og_params = config.policies.order_generator
    ta_name, ta_params = config.policies.task_assigner
    pp_name, pp_params = config.policies.path_planner
    rp_name, rp_params = config.policies.pod_return_planner
    pr_name, pr_params = config.policies.pod_retriever

    # 当 use_recorded_orders=true 时，使用 RecordedOrderGenerator 回放预录制订单
    if config.simulation.use_recorded_orders:
        og_name = "RecordedOrderGenerator"
        og_params = {
            "recorded_orders_path": config.simulation.recorded_orders_path,
            "immediate_dispatch": config.simulation.immediate_dispatch,
        }
        # immediate_dispatch 强制使用 serial 模式
        if config.simulation.immediate_dispatch:
            config.simulation.task_execution_mode = "serial"
            logger.info("immediate_dispatch=true → forced task_execution_mode='serial'")

    OrderGeneratorCls = get_policy("order_generator", og_name)
    TaskAssignerCls = get_policy("task_assigner", ta_name)
    PathPlannerCls = get_policy("path_planner", pp_name)
    PodReturnPlannerCls = get_policy("pod_return_planner", rp_name)
    PodRetrieverCls = get_policy("pod_retriever", pr_name)

    logger.info(f"Policies: order_generator={og_name}, "
                f"task_assigner={ta_name}, "
                f"path_planner={pp_name}, "
                f"pod_return_planner={rp_name}, "
                f"pod_retriever={pr_name}")

    order_generator = OrderGeneratorCls(
        order_interval=config.simulation.order_interval,
        max_items_per_order=config.simulation.max_items_per_order,
        fixed_order_size=config.simulation.fixed_order_size,
        max_items_per_sku=config.simulation.max_items_per_sku,
        **og_params,
    )
    task_assigner = TaskAssignerCls(**ta_params)
    # 如果 planner 构造函数接受 `seed` 且配置里没显式给，注入全局 seed
    if (config.simulation.seed is not None
            and "seed" not in pp_params
            and "seed" in __import__("inspect").signature(PathPlannerCls).parameters):
        pp_params = {**pp_params, "seed": config.simulation.seed}
    path_planner = PathPlannerCls(**pp_params)
    pod_return_planner = PodReturnPlannerCls(**rp_params)
    pod_retriever = PodRetrieverCls(**pr_params)

    # 将归还规划器和 Pod 检索器注入任务分配器
    task_assigner.pod_return_planner = pod_return_planner
    task_assigner.pod_retriever = pod_retriever
    task_assigner.path_planner = path_planner

    # --- 可选的可视化器 ---
    if args.mpl:
        visualizer = MatplotlibVisualizer(
            night_mode=config.simulation.night_mode,
        )
    elif args.p3d:
        from Visualization.panda3d_visualizer import Panda3DVisualizer
        visualizer = Panda3DVisualizer(
            view_mode=config.simulation.p3d_view_mode,
            use_gpu=config.simulation.p3d_use_gpu,
            night_mode=config.simulation.night_mode,
            robot_label_scale=config.simulation.robot_label_scale,
            robot_model_cfg=config.robot_model,
        )
    elif args.visualize:
        visualizer = TerminalVisualizer()
    else:
        visualizer = None

    # --- 创建引擎 ---
    engine = SimulationEngine(
        config=config,
        order_generator=order_generator,
        task_assigner=task_assigner,
        path_planner=path_planner,
        visualizer=visualizer,
    )

    # --- 快照采集（可选） ---
    snapshot_collector = None
    if config.snapshot.enabled:
        episode_id = SnapshotCollector.resolve_episode_id(
            config.snapshot.episode_id,
            {
                "map": f"{config.map.rows}x{config.map.cols}",
                "planner": pp_name,
                "assigner": ta_name,
                "robots": str(config.robots.num_robots),
            },
        )
        snapshot_collector = SnapshotCollector(output_dir=config.snapshot.output_dir)
        engine.on_tick_callbacks.append(snapshot_collector.on_tick)
        snapshot_collector.start_episode(episode_id)
        snapshot_collector.write_header(
            engine.world,
            extra={
                "mode": "rmfs",
                "planner": {"name": pp_name, "params": pp_params},
                "task_assigner": {"name": ta_name, "params": ta_params},
                "order_generator": {"name": og_name, "params": og_params},
                "pod_return_planner": rp_name,
                "pod_retriever": pr_name,
                "num_robots": config.robots.num_robots,
                "order_interval": config.simulation.order_interval,
                "max_items_per_order": config.simulation.max_items_per_order,
                "use_recorded_orders": config.simulation.use_recorded_orders,
                "task_execution_mode": config.simulation.task_execution_mode,
            },
        )
        logger.info(f"Snapshot collection enabled → {config.snapshot.output_dir}/{episode_id}.jsonl")

    # --- 运行 ---
    try:
        if args.p3d and visualizer is not None:
            # Qt UI 驱动循环（替代 engine.run）
            from Visualization.ui import SimulationUI
            ui = SimulationUI(
                engine=engine,
                visualizer=visualizer,
                night_mode=config.simulation.night_mode,
            )
            ui.run()
        else:
            engine.run()
    finally:
        if snapshot_collector is not None:
            snapshot_collector.end_episode()


if __name__ == "__main__":
    main()
