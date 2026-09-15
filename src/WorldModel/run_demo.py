"""
World Model End-to-End Demo (v3 — counterfactual)
===================================================
Runs the full pipeline:
  1. Simulate with pre-assignment counterfactual data collection
  2. Build pairwise ranking data from candidate groups
  3. Train the ST-GNN world model with multi-task + ranking loss
  4. Inference: rank candidate assignments by predicted cost
  5. Print results

Usage:
    conda activate multi_robot
    python -m WorldModel.run_demo
"""

import os
import sys
import random

import torch


def main():
    print("=" * 60)
    print("RMFS World Model v3 — Counterfactual Demo")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: Collect training data via simulation
    # ------------------------------------------------------------------
    print("\n[Phase 1] Collecting counterfactual training data...")

    from Config.config_loader import load_config
    from Engine.simulation_engine import SimulationEngine
    import Policies  # noqa: F401
    from Policies.policy_registry import get_policy
    from WorldModel.data_collector import WorldModelDataCollector

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Config", "world_model_config.json",
    )
    config = load_config(config_path)

    if config.simulation.seed is not None:
        random.seed(config.simulation.seed)

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
            and "seed" in __import__("inspect").signature(PathPlannerCls).parameters):
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

    collector = WorldModelDataCollector(
        output_dir="DataGen/wm_data",
        history_len=4,
        rollout_horizon=10,
        sample_interval=5,
        top_m_candidates=5,
        min_group_size=2,
    )
    engine.pre_assignment_callbacks.append(collector.on_pre_assignment)
    engine.on_tick_callbacks.append(collector.on_post_tick)

    print(f"  Running {config.simulation.max_ticks}-tick simulation "
          f"with {config.robots.num_robots} robots...")
    engine.run()

    data_path = collector.save()
    print(f"  Collected {collector.num_samples} training samples -> {data_path}")

    if collector.num_samples == 0:
        print("  ERROR: No samples collected. Check simulation configuration.")
        sys.exit(1)

    pairwise_data = collector.build_pairwise_data(epsilon=0.01)
    print(f"  Built {len(pairwise_data)} pairwise ranking pairs")

    # ------------------------------------------------------------------
    # Phase 2: Train the model
    # ------------------------------------------------------------------
    print(f"\n[Phase 2] Training ST-GNN world model...")

    from WorldModel.model import RMFSWorldModel
    from WorldModel.dataset import WorldModelDataset
    from WorldModel.train import train

    dataset = WorldModelDataset.from_file(data_path)
    sample0 = dataset[0]
    demand_dim = sample0["demand_context"].shape[0]
    num_stations = len(engine.world.map_state.station_positions)

    model = RMFSWorldModel(
        node_feat_dim=10,
        edge_feat_dim=6,
        demand_dim=demand_dim,
        action_node_dim=8,
        action_global_dim=6,
        hidden_dim=64,
        num_spatial_layers=3,
        rollout_horizon=10,
        num_stations=num_stations,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Training on {len(dataset)} samples for 10 epochs...")

    model = train(
        model, dataset, epochs=10, lr=1e-3,
        save_dir="DataGen/wm_checkpoints",
        verbose=True,
        pairwise_data=pairwise_data if pairwise_data else None,
        alpha_rank=0.1,
    )

    # ------------------------------------------------------------------
    # Phase 3: Inference — rank candidate assignments
    # ------------------------------------------------------------------
    print(f"\n[Phase 3] Inference — ranking candidate assignments...")

    model.eval()
    world = engine.world

    from WorldModel.graph_builder import (
        build_static_graph, extract_node_features, extract_edge_features,
        extract_demand_context, build_action_field, FeatureHistory,
    )
    from WorldModel.candidate_generator import generate_candidates

    edge_index, node_map, inv_node_map, local_cap, btk_score, ntype, adj = \
        build_static_graph(world.map_state)

    flow_counter = {}
    for agent in world.agents:
        nid = node_map.get(agent.position)
        if nid is not None:
            flow_counter[nid] = flow_counter.get(nid, 0.0) + 1.0

    fh = FeatureHistory(len(node_map), 10, 4)
    for _ in range(4):
        nf = extract_node_features(
            world, node_map, local_cap, btk_score, ntype, adj, flow_counter,
            reservation_window=1,
        )
        fh.push(nf)

    node_hist = fh.get_history()
    edge_feat = extract_edge_features(
        edge_index, node_map, inv_node_map, local_cap, world,
        adj=adj, edge_flow_counter={}, reservation_window=1,
    )
    demand = extract_demand_context(world)

    station_node_ids = []
    for sid, spos in world.map_state.station_positions.items():
        nid = node_map.get(spos)
        if nid is not None:
            station_node_ids.append(nid)

    groups = generate_candidates(world, top_m=5)
    if groups:
        print(f"  Generated {len(groups)} candidate groups from live simulation state")
        group = groups[0]
        fc = group["fixed_context"]
        candidates = group["candidates"]
        print(f"  Group: {group['group_id']} — "
              f"pod@{fc['pod_location']}, station@{fc['station_location']}")
    else:
        print("  No candidate groups from live state, using dummy candidates")
        stations = list(world.map_state.station_positions.items())
        candidates = []
        for i in range(2):
            pos = inv_node_map.get(i * 10, (0, 0))
            candidates.append({
                "robot_start": inv_node_map.get(0, (0, 0)),
                "pod_location": pos,
                "station_location": stations[0][1] if stations else (0, 4),
                "return_location": pos,
                "order_size": 1,
            })
        fc = None

    with torch.no_grad():
        z, e_demand, edge_attr = model.encode_state(
            node_hist, edge_index, edge_feat, demand,
        )

        costs = []
        for i, cand in enumerate(candidates):
            if fc is not None:
                from WorldModel.candidate_generator import build_candidate_assignment
                assignment = build_candidate_assignment(cand, fc)
            else:
                assignment = cand

            action_node, action_global = build_action_field(
                assignment, world, node_map, inv_node_map, local_cap,
                node_features=node_hist[-1],
            )
            cost = model.predict_cost(
                z, e_demand, edge_attr,
                action_node, action_global, edge_index, station_node_ids,
            )
            costs.append(cost.item())

            robot_pos = cand.get("robot_start", assignment.get("robot_start", "?"))
            print(f"  Candidate {i+1}: cost = {cost.item():.4f}  "
                  f"(robot@{robot_pos})")

    best = costs.index(min(costs))
    print(f"\n  Selected: Candidate {best + 1} (lowest predicted cost = {costs[best]:.4f})")

    print("\n" + "=" * 60)
    print("World Model v3 demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
