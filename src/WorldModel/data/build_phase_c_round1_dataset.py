"""Build Phase C round-1 training samples from pure-WM decision snapshots.

Every candidate is labelled by an isolated fixed-context rollout: no future
unknown orders and no continuation scheduler.  The builder rejects external
baseline snapshots and requires one native NO_ASSIGN row in every candidate
group.  The output is compatible with the existing WorldModelDataset and V6
training pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from WorldModel.data.candidate_generator import (
    ASSIGN_ROBOT_ACTION_TYPE,
    NO_ASSIGN_ACTION_SCHEMA_VERSION,
    NO_ASSIGN_ACTION_TYPE,
    NO_ASSIGN_ENCODING,
    build_candidate_assignment,
    compute_heuristic_cost,
    is_no_assign_candidate,
)
from WorldModel.data.counterfactual_rollout import evaluate_candidate_rollout
from WorldModel.evaluation.decision_snapshot_probe import (
    PHASE_C_DATA_CONTRACT,
    PHASE_C_SNAPINDEX_SCHEMA_VERSION,
    PHASE_C_SNAPSHOT_SCHEMA_VERSION,
)
from WorldModel.graph.graph_builder import (
    build_action_edge_field,
    build_action_field,
    build_static_graph,
    compute_preview_legs,
)
from WorldState.order_state import Order
from WorldState.task_state import Task


SCHEMA_VERSION = "phase_c_round1_dataset_build_v1"
FORMAL_HORIZON = 10
ACTION_PATH_MODE = 0
ACTION_ROUTE_ENCODING = "canonical_bfs_service_cell_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _snapshot_entries(snapshot_dir: Path) -> tuple[dict, list[tuple[Path, dict]]]:
    indexes = sorted(snapshot_dir.glob("snapindex_*.json"))
    if len(indexes) != 1:
        raise RuntimeError(
            f"expected exactly one snapindex in {snapshot_dir}, got {len(indexes)}"
        )
    index_path = indexes[0]
    index = _read_json(index_path)
    if index.get("schema_version") != PHASE_C_SNAPINDEX_SCHEMA_VERSION:
        raise ValueError(
            f"{index_path}: expected {PHASE_C_SNAPINDEX_SCHEMA_VERSION}"
        )
    if index.get("phase_c_data_contract") != PHASE_C_DATA_CONTRACT:
        raise ValueError(f"{index_path}: unexpected Phase C data contract")
    if index.get("training_source_policy") != "world_model_on_policy":
        raise ValueError(f"{index_path}: not a World-Model on-policy stream")
    if bool(index.get("external_baseline_training_samples")):
        raise ValueError(f"{index_path}: external baseline samples are forbidden")
    if bool(index.get("td_target_enabled")) or bool(
        index.get("td_value_head_enabled")
    ):
        raise ValueError(f"{index_path}: TD targets/heads are forbidden")
    if not bool(index.get("include_no_assign_candidate")):
        raise ValueError(f"{index_path}: native NO_ASSIGN was not captured")

    entries = []
    for row in index.get("decisions", ()):
        path = snapshot_dir / str(row["file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append((path, row))
    entries.sort(key=lambda item: (
        int(item[1].get("tick", -1)),
        str(item[1].get("group_id", "")),
    ))
    return {
        "path": str(index_path),
        "sha256": _sha256_file(index_path),
        "payload": index,
    }, entries


def _uniform_select(
    entries: list[tuple[Path, dict]], max_snapshots: int | None
) -> list[tuple[Path, dict]]:
    if max_snapshots is None or max_snapshots <= 0 or len(entries) <= max_snapshots:
        return entries
    if max_snapshots == 1:
        return [entries[len(entries) // 2]]
    selected = []
    for index in range(max_snapshots):
        position = round(index * (len(entries) - 1) / (max_snapshots - 1))
        selected.append(entries[position])
    return selected


def _fixed_context_copy(fixed_context: dict) -> dict:
    return {
        "order_id": fixed_context["order_id"],
        "pod_id": fixed_context["pod_id"],
        "pod_location": fixed_context["pod_location"],
        "station_id": fixed_context["station_id"],
        "station_location": fixed_context["station_location"],
        "entry_position": fixed_context.get("entry_position"),
        "exit_position": fixed_context.get("exit_position"),
        "return_location": fixed_context["return_location"],
    }


def _candidate_info(candidate: dict, action_type: str) -> dict:
    return {
        "action_type": action_type,
        "robot_id": candidate.get("robot_id"),
        "robot_start": candidate.get("robot_start"),
        "candidate_policy": candidate.get("candidate_policy"),
        "candidate_selection_mode": candidate.get(
            "candidate_selection_mode", "nearest"
        ),
        "eta": candidate.get("eta"),
        "eta_bin": candidate.get("eta_bin"),
        "arrival_delta_preview": candidate.get("arrival_delta_preview"),
        "route_conflict_preview": candidate.get("route_conflict_preview"),
        "route_length_preview": candidate.get("route_length_preview"),
    }


def _attach_lyapunov(sample: dict, result: dict) -> None:
    from WorldModel.core.lyapunov import LYAPUNOV_COLLECTION_SCHEMA_VERSION

    sample.update({
        "lyapunov_l0_collection_schema_version": (
            result.get(
                "lyapunov_l0_collection_schema_version",
                LYAPUNOV_COLLECTION_SCHEMA_VERSION,
            )
        ),
        "lyapunov_l0_valid": bool(result.get("lyapunov_l0_valid", False)),
    })
    if not result.get("lyapunov_l0_valid"):
        return
    for key in (
        "lyapunov_l0_start",
        "lyapunov_l0_post_action",
        "lyapunov_l0_end",
        "lyapunov_l0_immediate_delta",
        "lyapunov_l0_delta",
        "lyapunov_l0_trajectory",
        "analytic_work_relief_trajectory_schema_version",
        "lyapunov_l0_station_ids",
        "lyapunov_l0_station_work_trajectory",
        "lyapunov_l0_progress",
        "lyapunov_l0_traffic_trajectory_names",
        "lyapunov_l0_traffic_trajectory",
        "lyapunov_l0_config",
    ):
        sample[key] = result[key]


def _process_snapshot(
    path: Path,
    *,
    horizon: int,
    lyapunov_config: dict,
) -> list[dict]:
    with path.open("rb") as handle:
        snapshot = pickle.load(handle)
    if snapshot.get("schema_version") != PHASE_C_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"{path}: unexpected snapshot schema")
    if snapshot.get("phase_c_data_contract") != PHASE_C_DATA_CONTRACT:
        raise ValueError(f"{path}: unexpected Phase C contract")
    if snapshot.get("training_source_policy") != "world_model_on_policy":
        raise ValueError(f"{path}: external/non-WM policy snapshot")
    if bool(snapshot.get("external_baseline_training_samples")):
        raise ValueError(f"{path}: external baseline data are forbidden")
    arm_label = str(snapshot.get("arm_label") or "")
    if any(token in arm_label.lower() for token in ("greedy", "hungarian")):
        raise ValueError(f"{path}: external baseline arm is forbidden")
    contract = snapshot.get("rollout_label_contract") or {}
    if (
        contract.get("continuation_mode") != "isolated"
        or bool(contract.get("future_order_generation"))
        or bool(contract.get("continuation_scheduler"))
    ):
        raise ValueError(f"{path}: invalid rollout label contract")

    world = snapshot["world_snapshot"]
    config = snapshot["config"]
    path_planner = snapshot["path_planner_state"]
    fixed_context = snapshot["fixed_context"]
    candidates = snapshot["candidates"]
    no_assign_count = sum(is_no_assign_candidate(row) for row in candidates)
    robot_count = sum(not is_no_assign_candidate(row) for row in candidates)
    if no_assign_count != 1 or robot_count < 2:
        raise ValueError(
            f"{path}: expected >=2 robots and exactly one NO_ASSIGN, got "
            f"robots={robot_count}, no_assign={no_assign_count}"
        )

    (
        edge_index,
        node_map,
        inv_node_map,
        local_capacity,
        bottleneck_score,
        _node_type_arr,
        adj,
    ) = build_static_graph(world.map_state)
    stored_edge_index = torch.as_tensor(snapshot["edge_index"])
    if not torch.equal(edge_index.cpu(), stored_edge_index.cpu()):
        raise ValueError(f"{path}: static graph edge index changed")
    node_history = torch.as_tensor(snapshot["node_history"]).clone()
    node_latest = node_history[-1]
    edge_features = torch.as_tensor(snapshot["edge_features"]).clone()
    demand_context = torch.as_tensor(snapshot["demand_context"]).clone()
    station_node_ids = torch.as_tensor(
        snapshot["station_node_ids"], dtype=torch.long
    ).clone()

    outer_task_id = Task._next_id
    outer_order_id = Order._next_id
    outer_python = random.getstate()
    outer_numpy = np.random.get_state()
    outer_torch = torch.random.get_rng_state()
    samples = []
    try:
        for candidate in candidates:
            Task._next_id = snapshot["task_next_id"]
            Order._next_id = snapshot["order_next_id"]
            random.setstate(snapshot["python_rng_state"])
            np.random.set_state(snapshot["numpy_rng_state"])
            torch.random.set_rng_state(snapshot["torch_rng_state"])

            no_assign = is_no_assign_candidate(candidate)
            action_type = (
                NO_ASSIGN_ACTION_TYPE
                if no_assign else ASSIGN_ROBOT_ACTION_TYPE
            )
            assignment = build_candidate_assignment(candidate, fixed_context)
            legs = compute_preview_legs(
                assignment,
                world.map_state,
                node_map,
                path_planner=None,
                world=world,
            )
            action_node, action_global = build_action_field(
                assignment,
                world,
                node_map,
                inv_node_map,
                local_capacity,
                node_features=node_latest,
                precomputed_legs=legs,
            )
            action_edge = build_action_edge_field(
                assignment,
                edge_index,
                node_map,
                world.map_state,
                precomputed_legs=legs,
            )
            result = evaluate_candidate_rollout(
                world=world,
                candidate=candidate,
                fixed_context=fixed_context,
                config=config,
                path_planner=path_planner,
                horizon=horizon,
                node_map=node_map,
                local_capacity=local_capacity,
                bottleneck_score=bottleneck_score,
                adj=adj,
                reservation_window=1,
                record_lyapunov_l0=True,
                lyapunov_l0_config=lyapunov_config,
                rollout_continuation_mode="isolated",
            )

            robot_token = (
                "NO_ASSIGN" if no_assign else str(candidate["robot_id"])
            )
            sample = {
                "phase_c_sample_schema_version": (
                    "phase_c_round1_counterfactual_sample_v1"
                ),
                "phase_c_data_contract": PHASE_C_DATA_CONTRACT,
                "phase_c_round": snapshot.get("phase_c_round"),
                "training_source_policy": "world_model_on_policy",
                "external_baseline_training_samples": False,
                "td_target_enabled": False,
                "td_value_head_enabled": False,
                "run_id": snapshot["run_id"],
                "simulation_seed": snapshot.get("seed"),
                "source_load_level": snapshot.get("load"),
                "source_snapshot_file": path.name,
                "source_state_diagnostics": snapshot.get(
                    "phase_c_state_diagnostics"
                ),
                "continuation_policy": (
                    "isolated_native_no_assign"
                    if no_assign else "isolated_forced_candidate"
                ),
                "rollout_continuation_mode": "isolated",
                "node_history": node_history.clone(),
                "edge_index": stored_edge_index.clone(),
                "edge_features": edge_features.clone(),
                "demand_context": demand_context.clone(),
                "action_node": action_node,
                "action_global": action_global,
                "action_edge": action_edge,
                "candidate_group_id": snapshot["candidate_group_id"],
                "decision_tick": snapshot["decision_tick"],
                "action_type": action_type,
                "action_schema_version": NO_ASSIGN_ACTION_SCHEMA_VERSION,
                "action_encoding": (
                    NO_ASSIGN_ENCODING if no_assign else "route_fields_v1"
                ),
                "action_path_mode": ACTION_PATH_MODE,
                "action_route_encoding": ACTION_ROUTE_ENCODING,
                "candidate_key": (
                    f"{snapshot['candidate_group_id']}_r{robot_token}"
                    f"_o{fixed_context['order_id']}"
                    f"_p{fixed_context['pod_id']}"
                    f"_s{fixed_context['station_id']}"
                    f"_t{snapshot['decision_tick']}"
                ),
                "candidate_info": _candidate_info(candidate, action_type),
                "fixed_context": _fixed_context_copy(fixed_context),
                "station_node_ids": station_node_ids.clone(),
                "future_node_labels": result["future_node_labels"],
                "future_system_labels": result["future_system_labels"],
                "future_station_labels": result["future_station_labels"],
                "future_mask": result["future_mask"],
                "realized_cost": result["realized_cost"],
                "heuristic_cost": (
                    None
                    if no_assign
                    else compute_heuristic_cost(
                        candidate, fixed_context, world
                    )
                ),
                "heuristic_cost_valid": not no_assign,
                "rollout_vertex_conflicts": result[
                    "rollout_vertex_conflicts"
                ],
                "rollout_swap_conflicts": result[
                    "rollout_swap_conflicts"
                ],
                "rollout_blocked_moves": result["rollout_blocked_moves"],
                "rollout_generated_orders": result.get(
                    "rollout_generated_orders", 0
                ),
                "rollout_assigned_tasks": result.get(
                    "rollout_assigned_tasks", 0
                ),
                "future_demand_context": result.get(
                    "future_demand_context"
                ),
                "no_assign_applied": bool(
                    result.get("no_assign_applied", False)
                ),
                "no_assign_audit": result.get("no_assign_audit"),
            }
            _attach_lyapunov(sample, result)
            samples.append(sample)
    finally:
        Task._next_id = outer_task_id
        Order._next_id = outer_order_id
        random.setstate(outer_python)
        np.random.set_state(outer_numpy)
        torch.random.set_rng_state(outer_torch)
    return samples


def _audit(samples: list[dict], expected_groups: int, horizon: int) -> dict:
    groups = defaultdict(list)
    for sample in samples:
        groups[str(sample["candidate_group_id"])].append(sample)
    no_assign_rows = [
        sample for sample in samples
        if sample["action_type"] == NO_ASSIGN_ACTION_TYPE
    ]
    checks = {
        "nonempty": bool(samples),
        "expected_group_count": len(groups) == expected_groups,
        "one_no_assign_per_group": all(
            sum(row["action_type"] == NO_ASSIGN_ACTION_TYPE for row in rows)
            == 1
            for rows in groups.values()
        ),
        "at_least_two_robot_candidates_per_group": all(
            sum(row["action_type"] != NO_ASSIGN_ACTION_TYPE for row in rows)
            >= 2
            for rows in groups.values()
        ),
        "isolated_rollout_only": all(
            row["rollout_continuation_mode"] == "isolated"
            and int(row.get("rollout_generated_orders", 0)) == 0
            and int(row.get("rollout_assigned_tasks", 0)) == 0
            for row in samples
        ),
        "full_horizon_labels": all(
            int(torch.as_tensor(row["future_mask"]).sum().item()) == horizon
            for row in samples
        ),
        "no_external_baseline_samples": all(
            row.get("training_source_policy") == "world_model_on_policy"
            and not bool(row.get("external_baseline_training_samples"))
            for row in samples
        ),
        "no_td_targets_or_heads": all(
            not bool(row.get("td_target_enabled"))
            and not bool(row.get("td_value_head_enabled"))
            for row in samples
        ),
        "no_assign_audit": all(
            bool((row.get("no_assign_audit") or {}).get(
                "immediate_context_unchanged", False
            ))
            and not (row.get("no_assign_audit") or {}).get(
                "tasks_added_during_isolated_rollout", []
            )
            for row in no_assign_rows
        ),
    }
    pairwise_pairs = 0
    practical_pairs = 0
    for rows in groups.values():
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                pairwise_pairs += 1
                if abs(
                    float(rows[left]["realized_cost"])
                    - float(rows[right]["realized_cost"])
                ) > 0.01:
                    practical_pairs += 1
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "samples": len(samples),
        "groups": len(groups),
        "no_assign_samples": len(no_assign_rows),
        "pairwise_pairs": pairwise_pairs,
        "practical_pairs": practical_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--lyapunov-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="data.pt")
    parser.add_argument("--horizon", type=int, default=FORMAL_HORIZON)
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    if args.horizon <= 0:
        raise SystemExit("--horizon must be positive")
    if args.max_snapshots is not None and args.max_snapshots <= 0:
        raise SystemExit("--max-snapshots must be positive")
    if args.formal:
        if args.horizon != FORMAL_HORIZON:
            raise SystemExit("formal Phase C round 1 freezes --horizon=10")
        if args.max_snapshots is not None:
            raise SystemExit(
                "formal Phase C round 1 labels every captured snapshot"
            )

    snapshot_dir = Path(args.snapshot_dir)
    lyapunov_path = Path(args.lyapunov_config)
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(snapshot_dir)
    if not lyapunov_path.is_file():
        raise FileNotFoundError(lyapunov_path)
    lyapunov_config = _read_json(lyapunov_path)

    index_info, entries = _snapshot_entries(snapshot_dir)
    selected = _uniform_select(entries, args.max_snapshots)
    if not selected:
        raise RuntimeError("no Phase C snapshots selected")

    samples = []
    for number, (path, row) in enumerate(selected, start=1):
        print(
            f"[{number}/{len(selected)}] tick={row.get('tick')} "
            f"group={row.get('group_id')} {path.name}"
        )
        samples.extend(_process_snapshot(
            path,
            horizon=args.horizon,
            lyapunov_config=lyapunov_config,
        ))

    audit = _audit(samples, len(selected), args.horizon)
    if not audit["passed"]:
        failed = [
            name for name, passed in audit["checks"].items() if not passed
        ]
        raise RuntimeError(f"Phase C dataset audit failed: {failed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Phase C data: {output_path}")
    torch.save(samples, output_path)

    index = index_info["payload"]
    gen_config = {
        "schema_version": SCHEMA_VERSION,
        "run_id": index["run_id"],
        "seed": index.get("seed"),
        "load_level": index.get("load"),
        "phase_c_round": index.get("phase_c_round"),
        "path_planner_override": index.get("path_planner_override"),
        "path_planner_params_override": index.get(
            "path_planner_params_override"
        ),
        "phase_c_data_contract": PHASE_C_DATA_CONTRACT,
        "training_source_policy": "world_model_on_policy",
        "external_baseline_training_samples": False,
        "td_target_enabled": False,
        "td_value_head_enabled": False,
        "rollout_continuation_mode": "isolated",
        "future_unknown_orders_in_rollout": False,
        "continuation_scheduler_in_rollout": False,
        "horizon": args.horizon,
        "source_snapshot_index": index_info["path"],
        "source_snapshot_index_sha256": index_info["sha256"],
        "lyapunov_config": str(lyapunov_path),
        "lyapunov_config_sha256": _sha256_file(lyapunov_path),
        "action_schema_version": NO_ASSIGN_ACTION_SCHEMA_VERSION,
        "supports_no_assign_candidate": True,
        "action_path_mode": ACTION_PATH_MODE,
        "action_route_encoding": ACTION_ROUTE_ENCODING,
        "action_path_planner_preview_used": False,
    }
    (output_dir / "gen_config.json").write_text(
        json.dumps(gen_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    meta = {
        "schema_version": SCHEMA_VERSION,
        "data_file": str(output_path),
        "data_sha256": _sha256_file(output_path),
        "selected_snapshots": len(selected),
        "available_snapshots": len(entries),
        "audit": audit,
    }
    (output_dir / "data_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nsaved Phase C data: {output_path}")
    print(f"samples={audit['samples']} groups={audit['groups']}")
    print(f"practical_pairs={audit['practical_pairs']}")
    print("audit PASS")


if __name__ == "__main__":
    main()
