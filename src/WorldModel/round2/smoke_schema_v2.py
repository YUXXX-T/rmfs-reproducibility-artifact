"""Real-state, non-deployable smoke test for the Round-2 action schema.

This module deliberately lives outside every frozen Gate-1 runtime path.  It
loads the audited Round-1 checkpoint, rebuilds only its action encoder for the
schema-v2 input dimension, and exercises one assignment plus one
context-conditioned defer action against a real ``WorldState``.

The migrated action encoder has not been trained.  Consequently this smoke
only validates interfaces, tensor semantics, exact checkpoint inheritance,
and finite action-conditioned execution.  It never saves a checkpoint and its
score ordering must not be interpreted as policy evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import Policies  # noqa: F401  # Register policies used by the real config.
from Config.config_loader import load_config
from Policies.TaskAssigner.base_task_assigner import AssignmentContext
from Policies.TaskAssigner.context_assignment import (
    materialize_fixed_order_pod_contexts,
    propose_fixed_assignment_contexts,
)
from Policies.policy_registry import get_policy
from WorldModel.core.dispatch_potential import (
    DispatchDebtLedger,
    DispatchServiceDurations,
    StaticShortestPathDistance,
    context_key,
)
from WorldModel.data.candidate_generator import generate_robot_candidates
from WorldModel.graph.graph_builder import (
    FeatureHistory,
    build_static_graph,
    compute_preview_legs,
    extract_demand_context,
    extract_edge_features,
    extract_node_features,
)
from WorldModel.round2.action_schema_v2 import (
    ACTION_EDGE_DIM_V2,
    ACTION_GLOBAL_DIM_V2,
    ACTION_NODE_DIM_V2,
    CONTEXT_NODE_CHANNELS,
    ROBOT_NODE_CHANNEL,
    ROUTE_NODE_CHANNELS,
    build_action_edge_field_v2,
    build_action_field_v2,
    build_candidate_assignment_v2,
    action_sample_metadata_v2,
    infer_defer_context_action_schema_v2,
    make_defer_context_candidate_v2,
    schema_contract_v2,
)
from WorldModel.round2.checkpoint_migration_v2 import (
    migrate_round1_checkpoint_to_v2_model,
)
from WorldState.world import WorldState


SMOKE_SCHEMA_VERSION = "wm_defer_context_schema_v2_real_smoke_v1"
DEFAULT_CHECKPOINT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "model_round1_v1/best_regret_world_model.pt"
)
DEFAULT_CONFIG = Path("Config/world_model_config_PP_48_low.json")
FEATURE_HISTORY_LENGTH = 4
LEDGER_WARMUP_TICKS = 3


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(f"schema-v2 real smoke failed: {message}")


def _path_text(path: Path) -> str:
    return path.resolve().as_posix()


def _policy_entry(config, attribute: str) -> tuple[str, dict[str, Any]]:
    name, params = getattr(config.policies, attribute)
    return str(name), dict(params or {})


def _build_real_world(config, seed: int) -> tuple[WorldState, object, object]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    world = WorldState(config)
    generator_name, generator_params = _policy_entry(
        config, "order_generator"
    )
    generator_kwargs = dict(generator_params)
    generator_kwargs.setdefault(
        "order_interval", int(config.simulation.order_interval)
    )
    generator_kwargs.setdefault(
        "max_items_per_order", int(config.simulation.max_items_per_order)
    )
    generator_kwargs.setdefault(
        "fixed_order_size", bool(config.simulation.fixed_order_size)
    )
    generator_kwargs.setdefault(
        "max_items_per_sku", int(config.simulation.max_items_per_sku)
    )
    generator_cls = get_policy("order_generator", generator_name)
    generator = generator_cls(**generator_kwargs)
    _require(
        hasattr(generator, "generate_one"),
        f"{generator_name} does not expose generate_one()",
    )
    order = generator.generate_one(world, created_at=0)
    _require(order is not None, "the configured generator produced no order")
    world.order_state.add_order(order)

    retriever_name, retriever_params = _policy_entry(config, "pod_retriever")
    retriever_cls = get_policy("pod_retriever", retriever_name)
    retriever = retriever_cls(**retriever_params)

    return_name, return_params = _policy_entry(config, "pod_return_planner")
    return_cls = get_policy("pod_return_planner", return_name)
    return_planner = return_cls(**return_params)

    assigner_name, assigner_params = _policy_entry(config, "task_assigner")
    assigner_cls = get_policy("task_assigner", assigner_name)
    assigner = assigner_cls(**assigner_params)
    assigner.pod_retriever = retriever
    assigner.pod_return_planner = return_planner

    materialize_fixed_order_pod_contexts(world, retriever)
    contexts = propose_fixed_assignment_contexts(
        assigner, world, max_contexts=1
    )
    _require(contexts, "no real fixed assignment context was materialized")
    context = contexts[0]
    _require(
        isinstance(context, AssignmentContext),
        "context is not an AssignmentContext",
    )
    return world, context, order


def _all_finite(tensors: Sequence[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all().item()) for value in tensors)


def _models_equal(
    left: torch.nn.Module, right: torch.nn.Module
) -> tuple[bool, list[str]]:
    left_state = left.state_dict()
    right_state = right.state_dict()
    failures = []
    if set(left_state) != set(right_state):
        failures.append("parameter_key_set")
        return False, failures
    for name, value in left_state.items():
        if not torch.equal(value.cpu(), right_state[name].cpu()):
            failures.append(name)
    return not failures, failures


def _compact_migration_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": audit.get("schema_version"),
        "source_checkpoint": dict(audit.get("source_checkpoint") or {}),
        "target": dict(audit.get("target") or {}),
        "policy": dict(audit.get("policy") or {}),
        "inherited_parameter_count": int(
            audit.get("inherited_parameter_count", 0)
        ),
        "rebuilt_parameter_count": int(audit.get("rebuilt_parameter_count", 0)),
        "overridden_parameter_count": int(
            audit.get("overridden_parameter_count", 0)
        ),
        "exact_inheritance_verified": bool(
            audit.get("exact_inheritance_verified", False)
        ),
        "congestion_only_cost_verified": bool(
            audit.get("congestion_only_cost_verified", False)
        ),
        "passed": bool(audit.get("passed", False)),
    }


def run_smoke(
    *,
    checkpoint_path: Path,
    config_path: Path,
    seed: int,
    action_encoder_init_seed: int,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    config_path = config_path.resolve()
    _require(checkpoint_path.is_file(), f"checkpoint not found: {checkpoint_path}")
    _require(config_path.is_file(), f"config not found: {config_path}")

    config = load_config(str(config_path))
    world, context, order = _build_real_world(config, seed)
    _require(isinstance(world, WorldState), "world is not a real WorldState")

    (
        edge_index,
        node_map,
        inv_node_map,
        local_capacity,
        bottleneck_score,
        node_type_arr,
        adjacency,
    ) = build_static_graph(world.map_state)
    _require(len(node_map) > 0, "real warehouse graph is empty")
    _require(edge_index.ndim == 2 and edge_index.shape[0] == 2, "bad graph")

    distances = StaticShortestPathDistance(node_map, adjacency)
    durations = DispatchServiceDurations.from_world(world)
    ledger = DispatchDebtLedger()
    contexts = [context]
    for _ in range(LEDGER_WARMUP_TICKS):
        snapshot = ledger.snapshot(world, contexts, distances, durations)
        ledger.finalize(snapshot, assigned_keys=())
        world.advance_tick()
    snapshot = ledger.snapshot(world, contexts, distances, durations)
    terms = snapshot.terms(context)
    key = context_key(context)
    _require(terms.eligible, "real context has no eligible robot/free-flow path")
    _require(
        terms.free_flow_time is not None and terms.free_flow_time > 0.0,
        "real context has invalid free-flow time",
    )
    context_age = float(snapshot.order_age[key])
    free_flow_time = float(terms.free_flow_time)
    dispatch_debt = float(terms.debt_before)
    _require(
        all(
            math.isfinite(value)
            for value in (context_age, free_flow_time, dispatch_debt)
        ),
        "dispatch context contains non-finite values",
    )
    _require(
        context_age > 0.0 and dispatch_debt > 0.0,
        "ledger warmup did not exercise age/debt channels",
    )

    groups = generate_robot_candidates(
        contexts, world, top_m=2, candidate_mode="nearest"
    )
    _require(len(groups) == 1, "expected one real candidate group")
    group = groups[0]
    candidates = list(group.get("candidates") or ())
    _require(
        len(candidates) >= 2,
        "real context did not expose at least two robot candidates",
    )
    fixed_context = dict(group["fixed_context"])
    dispatch = {
        "context_age": context_age,
        "free_flow_time": free_flow_time,
        "dispatch_debt": dispatch_debt,
    }
    assignment = build_candidate_assignment_v2(
        candidates[0], fixed_context, **dispatch
    )
    second_assignment = build_candidate_assignment_v2(
        candidates[1], fixed_context, **dispatch
    )
    defer_candidate = make_defer_context_candidate_v2(
        fixed_context, **dispatch
    )
    defer = build_candidate_assignment_v2(
        defer_candidate, fixed_context, **dispatch
    )

    node_features = extract_node_features(
        world,
        node_map,
        local_capacity,
        bottleneck_score,
        node_type_arr,
        adjacency,
        {},
        reservation_window=10,
    )
    history = FeatureHistory(
        len(node_map), feat_dim=int(node_features.shape[1]),
        history_len=FEATURE_HISTORY_LENGTH,
    )
    for _ in range(FEATURE_HISTORY_LENGTH):
        history.push(node_features)
    node_history = history.get_history()
    edge_features = extract_edge_features(
        edge_index,
        node_map,
        inv_node_map,
        local_capacity,
        world,
        adj=adjacency,
        edge_flow_counter={},
        reservation_window=10,
    )
    demand_context = extract_demand_context(world)
    station_node_ids = [
        node_map[world.map_state.station_positions[station_id]]
        for station_id in sorted(world.map_state.station_positions)
    ]

    assignment_legs = compute_preview_legs(
        assignment, world.map_state, node_map
    )
    assign_node, assign_global = build_action_field_v2(
        assignment,
        world,
        node_map,
        inv_node_map,
        local_capacity,
        node_features=node_features,
        precomputed_legs=assignment_legs,
    )
    assign_edge = build_action_edge_field_v2(
        assignment,
        edge_index,
        node_map,
        world.map_state,
        precomputed_legs=assignment_legs,
    )
    second_assignment_legs = compute_preview_legs(
        second_assignment, world.map_state, node_map
    )
    second_assign_node, second_assign_global = build_action_field_v2(
        second_assignment,
        world,
        node_map,
        inv_node_map,
        local_capacity,
        node_features=node_features,
        precomputed_legs=second_assignment_legs,
    )
    second_assign_edge = build_action_edge_field_v2(
        second_assignment,
        edge_index,
        node_map,
        world.map_state,
        precomputed_legs=second_assignment_legs,
    )
    defer_node, defer_global = build_action_field_v2(
        defer,
        world,
        node_map,
        inv_node_map,
        local_capacity,
        node_features=node_features,
    )
    defer_edge = build_action_edge_field_v2(
        defer, edge_index, node_map, world.map_state
    )

    schema_samples = []
    for candidate_key, action, action_node, action_global, action_edge in (
        (
            f"{group['group_id']}_r{assignment['robot_id']}",
            assignment,
            assign_node,
            assign_global,
            assign_edge,
        ),
        (
            f"{group['group_id']}_r{second_assignment['robot_id']}",
            second_assignment,
            second_assign_node,
            second_assign_global,
            second_assign_edge,
        ),
        (
            f"{group['group_id']}_defer",
            defer,
            defer_node,
            defer_global,
            defer_edge,
        ),
    ):
        schema_samples.append({
            "run_id": f"round2_schema_smoke_seed{seed}",
            "simulation_seed": int(seed),
            "candidate_group_id": str(group["group_id"]),
            "candidate_key": candidate_key,
            "candidate_info": {"robot_id": action.get("robot_id")},
            **action_sample_metadata_v2(action),
            "action_node": action_node,
            "action_global": action_global,
            "action_edge": action_edge,
        })
    action_schema_audit = infer_defer_context_action_schema_v2(schema_samples)

    expected_node_shape = (len(node_map), ACTION_NODE_DIM_V2)
    expected_global_shape = (ACTION_GLOBAL_DIM_V2,)
    expected_edge_shape = (int(edge_index.shape[1]), ACTION_EDGE_DIM_V2)
    checks = {
        "real_world_state": isinstance(world, WorldState),
        "real_assignment_context": isinstance(context, AssignmentContext),
        "real_order_and_candidates": (
            order is not None and len(candidates) >= 2
        ),
        "dispatch_channels_exercised": (
            context_age > 0.0
            and free_flow_time > 0.0
            and dispatch_debt > 0.0
        ),
        "assignment_action_tensor_contract": (
            tuple(assign_node.shape) == expected_node_shape
            and tuple(assign_global.shape) == expected_global_shape
            and tuple(assign_edge.shape) == expected_edge_shape
        ),
        "defer_action_tensor_contract": (
            tuple(defer_node.shape) == expected_node_shape
            and tuple(defer_global.shape) == expected_global_shape
            and tuple(defer_edge.shape) == expected_edge_shape
        ),
        "action_tensors_finite": _all_finite(
            (
                assign_node,
                assign_global,
                assign_edge,
                defer_node,
                defer_global,
                defer_edge,
            )
        ),
        "assignment_and_defer_tensors_differ": (
            not torch.equal(assign_node, defer_node)
            and not torch.equal(assign_global, defer_global)
        ),
        "defer_robot_node_zero": (
            int(torch.count_nonzero(defer_node[:, ROBOT_NODE_CHANNEL]).item())
            == 0
        ),
        "defer_route_node_zero": (
            int(torch.count_nonzero(defer_node[:, ROUTE_NODE_CHANNELS]).item())
            == 0
        ),
        "defer_route_edge_zero": int(torch.count_nonzero(defer_edge).item())
        == 0,
        "defer_context_markers_present": all(
            int(torch.count_nonzero(defer_node[:, channel]).item()) == 1
            for channel in CONTEXT_NODE_CHANNELS
        ),
        "complete_real_group_schema_audit": bool(
            action_schema_audit.get("passed")
        ),
    }
    for name, passed in checks.items():
        detail = name
        if name == "complete_real_group_schema_audit" and not passed:
            detail += ": " + ", ".join(
                action_schema_audit.get("failures", ())[:10]
            )
        _require(passed, detail)

    model, migration_audit = migrate_round1_checkpoint_to_v2_model(
        checkpoint_path,
        action_encoder_init_seed=int(action_encoder_init_seed),
    )
    repeated_model, repeated_audit = migrate_round1_checkpoint_to_v2_model(
        checkpoint_path,
        action_encoder_init_seed=int(action_encoder_init_seed),
    )
    deterministic_migration, deterministic_failures = _models_equal(
        model, repeated_model
    )
    checks.update({
        "migration_audit_passed": bool(migration_audit.get("passed")),
        "repeated_migration_audit_passed": bool(repeated_audit.get("passed")),
        "exact_non_action_inheritance": bool(
            migration_audit.get("exact_inheritance_verified")
        ),
        "congestion_only_cost_contract": bool(
            migration_audit.get("congestion_only_cost_verified")
        ),
        "deterministic_migration": deterministic_migration,
    })
    for name in (
        "migration_audit_passed",
        "repeated_migration_audit_passed",
        "exact_non_action_inheritance",
        "congestion_only_cost_contract",
        "deterministic_migration",
    ):
        _require(checks[name], name)

    target_config = dict(migration_audit["target"]["model_config"])
    checks["state_tensor_contract_matches_checkpoint"] = (
        int(node_history.shape[2]) == int(target_config["node_feat_dim"])
        and int(edge_features.shape[1]) == int(target_config["edge_feat_dim"])
        and int(demand_context.shape[0]) == int(target_config["demand_dim"])
        and len(station_node_ids) == int(target_config["num_stations"])
    )
    _require(
        checks["state_tensor_contract_matches_checkpoint"],
        "state_tensor_contract_matches_checkpoint",
    )

    model.eval()
    with torch.no_grad():
        z, encoded_demand, encoded_edges = model.encode_state(
            node_history, edge_index, edge_features, demand_context
        )
        assign_embedding = model.action_encoder(
            assign_node, assign_global, len(node_map)
        )
        defer_embedding = model.action_encoder(
            defer_node, defer_global, len(node_map)
        )
        assign_cost, assign_details = model.predict_cost(
            z,
            encoded_demand,
            encoded_edges,
            assign_node,
            assign_global,
            edge_index,
            station_node_ids,
            return_details=True,
        )
        defer_cost, defer_details = model.predict_cost(
            z,
            encoded_demand,
            encoded_edges,
            defer_node,
            defer_global,
            edge_index,
            station_node_ids,
            return_details=True,
        )

    forward_tensors = (
        z,
        encoded_demand,
        encoded_edges,
        assign_embedding,
        defer_embedding,
        assign_cost.reshape(1),
        defer_cost.reshape(1),
        assign_details["system_preds"],
        assign_details["long_risk_preds"],
        assign_details["z_endpoint"],
        defer_details["system_preds"],
        defer_details["long_risk_preds"],
        defer_details["z_endpoint"],
    )
    checks.update({
        "finite_model_forward": _all_finite(forward_tensors),
        "action_embeddings_different": not torch.equal(
            assign_embedding, defer_embedding
        ),
    })
    _require(checks["finite_model_forward"], "finite_model_forward")
    _require(
        checks["action_embeddings_different"],
        "action_embeddings_different",
    )

    assign_cost_value = float(assign_cost.item())
    defer_cost_value = float(defer_cost.item())
    all_checks_passed = all(bool(value) for value in checks.values())
    _require(all_checks_passed, "not all hard checks passed")

    return {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "role": "DEVELOPMENT_INTERFACE_SMOKE_ONLY",
        "passed": True,
        "checks": checks,
        "declarations": {
            "action_edge_consumed_by_model": False,
            "deployment_ready": False,
            "score_ordering_interpretable": False,
            "checkpoint_written": False,
            "requires_schema_v2_training_before_online_use": True,
            "history_construction": "same_real_state_frame_repeated_4_v1",
            "ledger_warmup": "three_control_only_ticks_v1",
        },
        # Duplicate the critical declarations at top level for fail-closed
        # consumers that do not inspect nested metadata.
        "action_edge_consumed_by_model": False,
        "deployment_ready": False,
        "score_ordering_interpretable": False,
        "inputs": {
            "checkpoint": _path_text(checkpoint_path),
            "checkpoint_sha256": migration_audit["source_checkpoint"][
                "sha256"
            ],
            "config": _path_text(config_path),
            "effective_seed": int(seed),
            "config_declared_seed": config.simulation.seed,
            "action_encoder_init_seed": int(action_encoder_init_seed),
        },
        "real_world": {
            "tick": int(world.tick),
            "nodes": len(node_map),
            "edges": int(edge_index.shape[1]),
            "robots": len(world.agents),
            "pods": int(world.pod_state.total_pods),
            "stations": len(station_node_ids),
            "order_id": int(order.order_id),
        },
        "real_context": {
            "group_id": str(group["group_id"]),
            "order_id": int(context.order_id),
            "pod_id": int(context.pod_id),
            "station_id": int(context.station_id),
            "robot_candidate_count": len(candidates),
            "context_age": context_age,
            "free_flow_time": free_flow_time,
            "dispatch_debt": dispatch_debt,
        },
        "state_tensors": {
            "node_history": list(node_history.shape),
            "edge_index": list(edge_index.shape),
            "edge_features": list(edge_features.shape),
            "demand_context": list(demand_context.shape),
            "station_node_ids": len(station_node_ids),
        },
        "action_tensors": {
            "schema_contract": schema_contract_v2(),
            "complete_group_audit": action_schema_audit,
            "assignment": {
                "action_node": list(assign_node.shape),
                "action_global": list(assign_global.shape),
                "action_edge": list(assign_edge.shape),
                "route_node_nonzero": int(
                    torch.count_nonzero(assign_node[:, ROUTE_NODE_CHANNELS]).item()
                ),
                "route_edge_nonzero": int(torch.count_nonzero(assign_edge).item()),
            },
            "defer_context": {
                "action_node": list(defer_node.shape),
                "action_global": list(defer_global.shape),
                "action_edge": list(defer_edge.shape),
                "robot_node_nonzero": int(
                    torch.count_nonzero(
                        defer_node[:, ROBOT_NODE_CHANNEL]
                    ).item()
                ),
                "route_node_nonzero": int(
                    torch.count_nonzero(
                        defer_node[:, ROUTE_NODE_CHANNELS]
                    ).item()
                ),
                "route_edge_nonzero": int(torch.count_nonzero(defer_edge).item()),
            },
        },
        "model_forward": {
            "assignment_cost": assign_cost_value,
            "defer_context_cost": defer_cost_value,
            "absolute_cost_difference": abs(
                assign_cost_value - defer_cost_value
            ),
            "costs_different_observation": (
                assign_cost_value != defer_cost_value
            ),
            "action_embedding_shape": list(assign_embedding.shape),
            "action_embeddings_different": checks[
                "action_embeddings_different"
            ],
        },
        "migration": {
            **_compact_migration_audit(migration_audit),
            "deterministic_repeat_verified": deterministic_migration,
            "deterministic_repeat_failures": deterministic_failures,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the isolated real-state Phase-C Round-2 schema-v2 smoke. "
            "Only the requested audit JSON is written; no checkpoint is saved."
        )
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=491)
    parser.add_argument(
        "--action-encoder-init-seed", type=int, default=20260728
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = args.output.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config_path = args.config.resolve()
    _require(
        output_path not in (checkpoint_path, config_path),
        "output must not overwrite an input",
    )
    audit = run_smoke(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        seed=int(args.seed),
        action_encoder_init_seed=int(args.action_encoder_init_seed),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("Phase-C Round-2 schema-v2 real smoke PASS")
    print(f"audit = {output_path.as_posix()}")
    print("checkpoint_written = false")
    print("deployment_ready = false")


if __name__ == "__main__":
    main()
