"""Replay one collection shard and audit H=10 station-potential transport."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from WorldModel.data.candidate_generator import (
    build_candidate_assignment,
    is_no_assign_candidate,
)
from WorldModel.data.counterfactual_rollout import evaluate_candidate_rollout
from WorldModel.evaluation.collect_phase_c_station_congestion_endpoint import (
    _load_bundle,
    _read_json,
    _verify_artifact,
    _verify_hashes,
)
from WorldModel.evaluation.decision_snapshot_probe import (
    PHASE_C_DATA_CONTRACT,
    PHASE_C_SNAPINDEX_SCHEMA_VERSION,
    PHASE_C_SNAPSHOT_SCHEMA_VERSION,
)
from WorldModel.evaluation.evaluate import _load_model
from WorldModel.evaluation.freeze_phase_c_station_congestion_endpoint_h10 import (
    FROZEN_FILENAME,
)
from WorldModel.evaluation.phase_c_station_congestion_endpoint_protocol import (
    COLLECTION_SCHEMA_VERSION,
    HORIZON,
    LOADS,
    OUTPUT_ROOT,
    REPLAY_SCHEMA_VERSION,
    REPLAY_SHARDS,
    SEEDS,
    sha256_file,
)
from WorldModel.evaluation.station_congestion_endpoint import (
    ENDPOINT_OBSERVATION_SCHEMA_VERSION,
    ENDPOINT_RECORD_SCHEMA_VERSION,
    EndpointObservationObserver,
    build_station_layout,
    evaluate_station_psi,
    load_frozen_station_head,
)
from WorldModel.graph.graph_builder import (
    build_action_field,
    build_static_graph,
    compute_preview_legs,
)
from WorldState.order_state import Order
from WorldState.task_state import Task


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _write_hashes(root: Path) -> Path:
    output = root / "shard_outputs.sha256"
    rows = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name in {
            output.name,
            "replay_summary.json",
        }:
            continue
        rows.append(f"{sha256_file(path)}  {path.name}")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return output


def _resume(
    shard_dir: Path,
    *,
    protocol_sha256: str,
    load: str,
    seed: int,
    shard_index: int,
    shard_count: int,
) -> bool:
    summary_path = shard_dir / "replay_summary.json"
    if not summary_path.is_file():
        return False
    summary = _read_json(summary_path)
    manifest = shard_dir / "shard_outputs.sha256"
    checks = {
        "schema": summary.get("schema_version") == REPLAY_SCHEMA_VERSION,
        "protocol": summary.get("protocol_sha256") == protocol_sha256,
        "load": summary.get("load") == load,
        "seed": int(summary.get("seed", -1)) == int(seed),
        "shard_index": int(summary.get("shard_index", -1)) == shard_index,
        "shard_count": int(summary.get("shard_count", -1)) == shard_count,
        "audit": bool((summary.get("audit") or {}).get("passed")),
        "manifest_hash": (
            manifest.is_file()
            and sha256_file(manifest) == summary.get("shard_outputs_sha256")
        ),
        "outputs": _verify_hashes(shard_dir, manifest),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"cannot resume endpoint replay {shard_dir}: {failed}")
    print(
        f"[resume] endpoint replay {load} seed={seed} "
        f"shard={shard_index}/{shard_count}"
    )
    return True


def _collection_source(
    output_root: Path,
    *,
    protocol_sha256: str,
    load: str,
    seed: int,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    run_id = f"phasec_s1_{load}_seed{seed}"
    run_dir = output_root / "collections" / run_id
    summary = _read_json(run_dir / "collection_summary.json")
    manifest = run_dir / "run_outputs.sha256"
    checks = {
        "schema": summary.get("schema_version") == COLLECTION_SCHEMA_VERSION,
        "protocol": summary.get("protocol_sha256") == protocol_sha256,
        "load": summary.get("load") == load,
        "seed": int(summary.get("seed", -1)) == int(seed),
        "audit": bool((summary.get("audit") or {}).get("passed")),
        "manifest_hash": (
            manifest.is_file()
            and sha256_file(manifest) == summary.get("run_outputs_sha256")
        ),
        "outputs": _verify_hashes(run_dir, manifest),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"endpoint collection source failed audit: {failed}")
    snapshot_dir = run_dir / str(summary["outputs"]["snapshot_dir"])
    index_path = run_dir / str(summary["outputs"]["snapshot_index"])
    index = _read_json(index_path)
    if index.get("schema_version") != PHASE_C_SNAPINDEX_SCHEMA_VERSION:
        raise ValueError("wrong endpoint snapshot index schema")
    return run_dir, summary, snapshot_dir, index


def _channel_dict(values: torch.Tensor) -> dict[str, float]:
    row = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if row.numel() != 2:
        raise ValueError("station potential must have two channels")
    return {"traffic": float(row[0]), "service": float(row[1])}


def _process_snapshot(
    path: Path,
    *,
    model,
    head,
    head_payload: Mapping[str, Any],
    device: torch.device,
    horizon: int,
) -> tuple[list[dict[str, Any]], int, int]:
    with path.open("rb") as handle:
        snapshot = pickle.load(handle)
    if snapshot.get("schema_version") != PHASE_C_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"unexpected snapshot schema: {path}")
    if snapshot.get("phase_c_data_contract") != PHASE_C_DATA_CONTRACT:
        raise ValueError(f"unexpected snapshot data contract: {path}")
    if bool(snapshot.get("snapshot_includes_no_assign")):
        raise ValueError("endpoint H=10 v1 is assignment-candidate only")

    world = snapshot["world_snapshot"]
    config = snapshot["config"]
    path_planner = snapshot["path_planner_state"]
    fixed_context = snapshot["fixed_context"]
    candidates = list(snapshot["candidates"])
    if len(candidates) < 2 or any(
        is_no_assign_candidate(candidate) for candidate in candidates
    ):
        raise ValueError(f"invalid assignment candidate set: {path}")
    (
        edge_index,
        node_map,
        inv_node_map,
        local_capacity,
        bottleneck_score,
        node_type_arr,
        adj,
    ) = build_static_graph(world.map_state)
    stored_edge_index = torch.as_tensor(snapshot["edge_index"], dtype=torch.long)
    if not torch.equal(edge_index.cpu(), stored_edge_index.cpu()):
        raise ValueError(f"static graph changed for snapshot: {path}")
    layout = build_station_layout(world, node_map, edge_index)
    stored_station_ids = tuple(
        int(value) for value in torch.as_tensor(
            snapshot["station_node_ids"], dtype=torch.long
        ).tolist()
    )
    if stored_station_ids != layout.station_node_ids:
        raise ValueError("snapshot station-node alignment changed")

    node_history = torch.as_tensor(
        snapshot["node_history"], dtype=torch.float32
    ).clone()
    edge_features = torch.as_tensor(
        snapshot["edge_features"], dtype=torch.float32
    ).clone()
    demand_context = torch.as_tensor(
        snapshot["demand_context"], dtype=torch.float32
    ).clone()
    with torch.no_grad():
        z_start, e_demand, edge_attr = model.encode_state(
            node_history.to(device),
            edge_index.to(device),
            edge_features.to(device),
            demand_context.to(device),
        )
        psi_start = evaluate_station_psi(
            z_start,
            head=head,
            head_payload=head_payload,
            layout=layout,
        ).detach().cpu()

    context_station_id = int(fixed_context["station_id"])
    if context_station_id not in layout.station_ids:
        raise ValueError("fixed context references an unknown station")
    outer_task_id = Task._next_id
    outer_order_id = Order._next_id
    outer_python = random.getstate()
    outer_numpy = np.random.get_state()
    outer_torch = torch.random.get_rng_state()
    records: list[dict[str, Any]] = []
    action_count = 0
    try:
        for candidate_offset, candidate in enumerate(candidates):
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
                node_features=node_history[-1],
                precomputed_legs=legs,
            )
            with torch.no_grad():
                _, _, _, _, _, z_predicted = model.rollout(
                    z_start,
                    e_demand,
                    edge_attr,
                    action_node.to(device),
                    action_global.to(device),
                    edge_index.to(device),
                    torch.tensor(
                        layout.station_node_ids,
                        dtype=torch.long,
                        device=device,
                    ),
                    K=horizon,
                )
                psi_predicted = evaluate_station_psi(
                    z_predicted,
                    head=head,
                    head_payload=head_payload,
                    layout=layout,
                ).detach().cpu()

            Task._next_id = snapshot["task_next_id"]
            Order._next_id = snapshot["order_next_id"]
            random.setstate(snapshot["python_rng_state"])
            np.random.set_state(snapshot["numpy_rng_state"])
            torch.random.set_rng_state(snapshot["torch_rng_state"])
            observer = EndpointObservationObserver(
                node_history=node_history,
                edge_index=edge_index,
                node_map=node_map,
                inv_node_map=inv_node_map,
                local_capacity=local_capacity,
                bottleneck_score=bottleneck_score,
                node_type_arr=node_type_arr,
                adj=adj,
                flow_counter=snapshot.get("flow_counter") or {},
                edge_flow_counter=snapshot.get("edge_flow_counter") or {},
                layout=layout,
                scale_contract=head_payload["scale_contract"],
                horizon=horizon,
                reservation_window=1,
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
                rollout_continuation_mode="isolated",
                rollout_observer=observer,
            )
            if int(torch.as_tensor(result["future_mask"]).sum().item()) != horizon:
                raise RuntimeError(f"incomplete real endpoint rollout: {path}")
            endpoint = result.get("rollout_observer_output")
            if not isinstance(endpoint, Mapping):
                raise RuntimeError("real endpoint observation was not captured")
            if endpoint.get("schema_version") != ENDPOINT_OBSERVATION_SCHEMA_VERSION:
                raise ValueError("wrong real endpoint observation schema")
            with torch.no_grad():
                z_real, _, _ = model.encode_state(
                    endpoint["node_history"].to(device),
                    edge_index.to(device),
                    endpoint["edge_features"].to(device),
                    endpoint["demand_context"].to(device),
                )
                psi_real = evaluate_station_psi(
                    z_real,
                    head=head,
                    head_payload=head_payload,
                    layout=layout,
                ).detach().cpu()
            physical = torch.as_tensor(
                endpoint["physical_targets"], dtype=torch.float32
            )
            if physical.shape != psi_real.shape:
                raise ValueError("physical/head station target shape mismatch")
            action_id = (
                f"{snapshot['run_id']}|{path.name}|"
                f"r{int(candidate['robot_id'])}"
            )
            candidate_group = (
                f"{snapshot['run_id']}|{path.name}|s{context_station_id}"
            )
            for station_offset, station_id in enumerate(layout.station_ids):
                delta_predicted = (
                    psi_predicted[station_offset] - psi_start[station_offset]
                )
                delta_real = (
                    psi_real[station_offset] - psi_start[station_offset]
                )
                records.append({
                    "schema_version": ENDPOINT_RECORD_SCHEMA_VERSION,
                    "run_id": str(snapshot["run_id"]),
                    "load": str(snapshot.get("load")),
                    "seed": int(snapshot.get("seed")),
                    "decision_tick": int(snapshot["decision_tick"]),
                    "source_snapshot": path.name,
                    "candidate_group_id": str(snapshot["candidate_group_id"]),
                    "action_id": action_id,
                    "candidate_rank_group": candidate_group,
                    "candidate_offset": int(candidate_offset),
                    "robot_id": int(candidate["robot_id"]),
                    "context_station_id": context_station_id,
                    "station_id": int(station_id),
                    "is_context_station": int(station_id) == context_station_id,
                    "horizon": int(horizon),
                    "endpoint_tick": int(endpoint["endpoint_tick"]),
                    "psi_start": _channel_dict(psi_start[station_offset]),
                    "psi_predicted_endpoint": _channel_dict(
                        psi_predicted[station_offset]
                    ),
                    "psi_real_endpoint": _channel_dict(
                        psi_real[station_offset]
                    ),
                    "physical_target_endpoint": _channel_dict(
                        physical[station_offset]
                    ),
                    "delta_psi_predicted": _channel_dict(delta_predicted),
                    "delta_psi_real": _channel_dict(delta_real),
                    "physical_target_components": endpoint[
                        "physical_components"
                    ][station_offset],
                    "realized_cost": float(result["realized_cost"]),
                    "rollout_vertex_conflicts": int(
                        result["rollout_vertex_conflicts"]
                    ),
                    "rollout_swap_conflicts": int(
                        result["rollout_swap_conflicts"]
                    ),
                    "rollout_blocked_moves": int(
                        result["rollout_blocked_moves"]
                    ),
                })
            action_count += 1
    finally:
        Task._next_id = outer_task_id
        Order._next_id = outer_order_id
        random.setstate(outer_python)
        np.random.set_state(outer_numpy)
        torch.random.set_rng_state(outer_torch)
    return records, action_count, len(layout.station_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", choices=LOADS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=REPLAY_SHARDS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--frozen-bundle", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(int(args.torch_threads), 1))
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("require 0 <= shard-index < shard-count")
    if not args.development:
        if int(args.seed) not in SEEDS:
            raise SystemExit(f"frozen replay seed must be one of {list(SEEDS)}")
        if int(args.shard_count) != REPLAY_SHARDS:
            raise SystemExit(
                f"frozen replay requires --shard-count={REPLAY_SHARDS}"
            )
    bundle_path = args.frozen_bundle or args.output_root / FROZEN_FILENAME
    bundle = _load_bundle(bundle_path)
    protocol_sha = str(bundle["protocol_sha256"])
    protocol = bundle["protocol"]
    model_path = Path(str(protocol["inputs"]["model_checkpoint"]))
    head_path = Path(str(protocol["inputs"]["station_head_checkpoint"]))
    scale_path = Path(str(protocol["inputs"]["station_scale_contract"]))
    for path in (model_path, head_path, scale_path):
        _verify_artifact(bundle, path)
    run_dir, collection, snapshot_dir, index = _collection_source(
        args.output_root,
        protocol_sha256=protocol_sha,
        load=args.load,
        seed=args.seed,
    )
    entries = sorted(
        index.get("decisions") or [],
        key=lambda row: (int(row.get("tick", -1)), str(row.get("group_id"))),
    )
    selected = [
        row for offset, row in enumerate(entries)
        if offset % int(args.shard_count) == int(args.shard_index)
    ]
    if not selected:
        raise RuntimeError("endpoint replay shard selected zero snapshots")

    run_id = f"phasec_s1_{args.load}_seed{args.seed}"
    shard_name = (
        f"shard_{int(args.shard_index):03d}_of_{int(args.shard_count):03d}"
    )
    shard_dir = args.output_root / "replays" / run_id / shard_name
    if _resume(
        shard_dir,
        protocol_sha256=protocol_sha,
        load=args.load,
        seed=args.seed,
        shard_index=int(args.shard_index),
        shard_count=int(args.shard_count),
    ):
        return
    if shard_dir.exists():
        raise FileExistsError(
            f"partial endpoint replay exists without summary: {shard_dir}"
        )
    staging = shard_dir.parent / f".{shard_name}.tmp.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)

    device = torch.device(args.device)
    model, label_schema = _load_model(str(model_path))
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head, head_payload = load_frozen_station_head(
        head_path,
        model_checkpoint_path=model_path,
        device=device,
    )
    records_path = staging / "endpoint_records.jsonl"
    snapshot_count = 0
    action_count = 0
    record_count = 0
    expected_record_count = 0
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for offset, row in enumerate(selected, start=1):
            path = snapshot_dir / str(row["file"])
            print(
                f"[{offset}/{len(selected)}] {run_id} "
                f"tick={row.get('tick')} shard={args.shard_index}",
                flush=True,
            )
            (
                snapshot_records,
                snapshot_actions,
                snapshot_station_count,
            ) = _process_snapshot(
                path,
                model=model,
                head=head,
                head_payload=head_payload,
                device=device,
                horizon=HORIZON,
            )
            for record in snapshot_records:
                handle.write(json.dumps(
                    record, ensure_ascii=False, separators=(",", ":")
                ) + "\n")
            snapshot_count += 1
            action_count += snapshot_actions
            record_count += len(snapshot_records)
            expected_record_count += (
                snapshot_actions * snapshot_station_count
            )

    checks = {
        "snapshots_match_selection": snapshot_count == len(selected),
        "all_snapshots_have_actions": action_count >= snapshot_count * 2,
        "all_station_rows_per_action": (
            record_count == expected_record_count
        ),
        "records_nonempty": record_count > 0,
        "horizon_fixed_10": HORIZON == 10,
        "assignment_only": not bool(index.get("include_no_assign_candidate")),
        "source_collection_passed": bool(
            (collection.get("audit") or {}).get("passed")
        ),
    }
    audit = {"passed": all(checks.values()), "checks": checks}
    if not audit["passed"]:
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError(f"endpoint replay audit failed: {failed}")
    output_manifest = _write_hashes(staging)
    summary = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "development_only": True,
        "protocol_sha256": protocol_sha,
        "load": args.load,
        "seed": int(args.seed),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "horizon": HORIZON,
        "snapshots": snapshot_count,
        "actions": action_count,
        "station_records": record_count,
        "checkpoint_label_schema": label_schema,
        "source": {
            "collection_dir": run_dir.as_posix(),
            "collection_outputs_sha256": collection["run_outputs_sha256"],
            "model_checkpoint_sha256": sha256_file(model_path),
            "station_head_checkpoint_sha256": sha256_file(head_path),
            "scale_contract_sha256": head_payload["scale_contract_sha256"],
        },
        "outputs": {"records": records_path.name},
        "audit": audit,
        "shard_outputs_sha256": sha256_file(output_manifest),
    }
    _atomic_json(staging / "replay_summary.json", summary)
    shard_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(shard_dir)
    print(f"[complete] endpoint replay: {shard_dir}")


if __name__ == "__main__":
    main()
