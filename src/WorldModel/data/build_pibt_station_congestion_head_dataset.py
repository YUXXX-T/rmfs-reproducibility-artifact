"""Build a PIBT-on-policy station-head dataset from Phase-C snapshots.

The builder is deliberately isolated from the historical PP station-head
pipeline.  It re-encodes the decision-state observations collected under
PIBT with the PIBT-adapted World Model, while keeping the published J1 target
definition and scale contract unchanged.  Multiple fixed contexts captured
at the same decision tick are verified to contain the same state and then
deduplicated before station rows are emitted.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from WorldModel.core.station_congestion_head import (
    CHANNEL_NAMES,
    COMPONENT_NAMES,
    build_station_targets,
    verify_scale_contract,
)
from WorldModel.data.build_station_congestion_head_dataset import (
    ARM_NAMES,
    DATASET_SCHEMA_VERSION,
    LOAD_NAMES,
    REPRESENTATION_STATION_REGION_MEAN_MAX,
    _finalise_split,
    _new_split_buffer,
    _target_stats,
    sha256_file,
    split_for_seed,
)
from WorldModel.evaluation.evaluate import _load_model
from WorldModel.evaluation.station_congestion_endpoint import (
    build_endpoint_station_rows,
    build_station_layout,
    build_station_representations,
)
from WorldModel.graph.graph_builder import build_static_graph


SUMMARY_SCHEMA_VERSION = "pibt_station_congestion_dataset_summary_v1"
PLANNER_NAME = "PIBTPlanner"
PLANNER_PARAMS = {"strict_validation": True}
DEFAULT_SPLITS = {
    "train": tuple(range(461, 468)),
    "val": (468, 469),
    "test": (470,),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value)).upper()


def _tensor_equal(left: Any, right: Any) -> bool:
    return torch.equal(torch.as_tensor(left).cpu(), torch.as_tensor(right).cpu())


def _validate_training_bundle(path: Path) -> tuple[dict[str, Any], str]:
    payload = _read_json(path)
    protocol = payload.get("protocol") or {}
    if protocol.get("campaign") != "pibt_on_policy_round1":
        raise ValueError("station-head source is not the PIBT training campaign")
    planner = protocol.get("planner") or {}
    if planner.get("name") != PLANNER_NAME or planner.get("params") != PLANNER_PARAMS:
        raise ValueError("PIBT training bundle planner contract changed")
    protocol_sha = str(payload.get("protocol_sha256") or "")
    if not protocol_sha:
        raise ValueError("PIBT training bundle has no protocol hash")
    return payload, protocol_sha


def discover_snapshot_cells(
    snapshot_root: Path,
    splits: Mapping[str, Sequence[int]],
) -> list[tuple[str, int, str, list[Path]]]:
    selected = {int(seed) for values in splits.values() for seed in values}
    cells: list[tuple[str, int, str, list[Path]]] = []
    for load in LOAD_NAMES:
        for seed in sorted(selected):
            directory = snapshot_root / load / f"seed{seed}" / "snapshots"
            paths = sorted(directory.glob("*.pkl")) if directory.is_dir() else []
            if not paths:
                raise FileNotFoundError(
                    f"no PIBT decision snapshots for load={load} seed={seed}: "
                    f"{directory}"
                )
            cells.append((load, seed, split_for_seed(seed, splits), paths))
    return cells


def group_snapshot_paths(paths: Sequence[Path]) -> list[tuple[int, list[Path]]]:
    grouped: dict[int, list[Path]] = {}
    for path in paths:
        with path.open("rb") as handle:
            snapshot = pickle.load(handle)
        tick = int(snapshot.get("decision_tick", -1))
        if tick < 0:
            raise ValueError(f"snapshot has no valid decision tick: {path}")
        grouped.setdefault(tick, []).append(path)
    return [(tick, grouped[tick]) for tick in sorted(grouped)]


def load_verified_tick_snapshot(
    paths: Sequence[Path],
    *,
    expected_load: str,
    expected_seed: int,
) -> tuple[dict[str, Any], int]:
    if not paths:
        raise ValueError("snapshot tick group cannot be empty")
    snapshots = []
    for path in paths:
        with path.open("rb") as handle:
            snapshot = pickle.load(handle)
        checks = {
            "training_source_policy": snapshot.get("training_source_policy")
            == "world_model_on_policy",
            "external_baseline_forbidden": not bool(
                snapshot.get("external_baseline_training_samples")
            ),
            "load": snapshot.get("load") == expected_load,
            "seed": int(snapshot.get("seed", -1)) == int(expected_seed),
            "planner_name": snapshot.get("path_planner_override") == PLANNER_NAME,
            "planner_params": snapshot.get("path_planner_params_override")
            == PLANNER_PARAMS,
        }
        planner = snapshot.get("path_planner_state")
        checks.update({
            "planner_type": type(planner).__name__ == PLANNER_NAME,
            "planner_batch": bool(getattr(planner, "supports_batch_planning", False)),
            "planner_single_step": bool(
                getattr(planner, "is_single_step_planner", False)
            ),
            "planner_strict": bool(getattr(planner, "strict_validation", False)),
        })
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise ValueError(f"PIBT snapshot contract failed ({path}): {failed}")
        if int(snapshot["world_snapshot"].tick) != int(snapshot["decision_tick"]):
            raise ValueError(f"snapshot world/decision tick mismatch: {path}")
        snapshots.append(snapshot)

    reference = snapshots[0]
    invariant_keys = (
        "node_history",
        "edge_index",
        "edge_features",
        "demand_context",
        "station_node_ids",
    )
    for offset, snapshot in enumerate(snapshots[1:], start=1):
        if int(snapshot["decision_tick"]) != int(reference["decision_tick"]):
            raise ValueError("mixed decision ticks in one snapshot group")
        for key in invariant_keys:
            if not _tensor_equal(reference[key], snapshot[key]):
                raise ValueError(
                    f"same-tick context snapshots disagree on {key}: "
                    f"{paths[0]} vs {paths[offset]}"
                )
        if (
            int(snapshot["task_next_id"]) != int(reference["task_next_id"])
            or int(snapshot["order_next_id"]) != int(reference["order_next_id"])
        ):
            raise ValueError("same-tick context snapshots disagree on ID state")
    return reference, len(snapshots) - 1


def build_dataset(
    *,
    snapshot_root: Path,
    checkpoint: Path,
    scale_contract_path: Path,
    training_bundle_path: Path,
    output_root: Path,
    splits: Mapping[str, Sequence[int]],
    device: torch.device,
    ticks: int,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(
            f"output already exists; preserve it or choose a new path: {output_root}"
        )
    for path in (checkpoint, scale_contract_path, training_bundle_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    _, protocol_sha = _validate_training_bundle(training_bundle_path)
    scale_contract = _read_json(scale_contract_path)
    verify_scale_contract(scale_contract)

    model, checkpoint_label_schema = _load_model(str(checkpoint))
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoder_latent_dim = int(model.state_encoder.temporal.gru.hidden_size)
    latent_dim = encoder_latent_dim * 3

    buffers = {name: _new_split_buffer() for name in splits}
    run_ids: list[str] = []
    global_frame_group = 0
    run_index = 0
    snapshot_files = 0
    unique_ticks = 0
    duplicate_context_snapshots = 0

    cells = discover_snapshot_cells(snapshot_root, splits)
    with torch.no_grad():
        for load, seed, split_name, paths in cells:
            run_ids.append(f"pibt_phaseC_r1_{load}_seed{seed}")
            tick_groups = group_snapshot_paths(paths)
            if not tick_groups:
                raise ValueError(f"empty snapshot cell: {load} seed={seed}")
            for _, tick_paths in tick_groups:
                snapshot, duplicate_count = load_verified_tick_snapshot(
                    tick_paths,
                    expected_load=load,
                    expected_seed=seed,
                )
                snapshot_files += len(tick_paths)
                unique_ticks += 1
                duplicate_context_snapshots += duplicate_count
                tick = int(snapshot["decision_tick"])
                if tick < 0 or tick >= int(ticks):
                    raise ValueError(f"decision tick outside frozen horizon: {tick}")
                world = snapshot["world_snapshot"]
                (
                    edge_index,
                    node_map,
                    _inv_node_map,
                    _local_capacity,
                    _bottleneck_score,
                    _node_type_arr,
                    _adj,
                ) = build_static_graph(world.map_state)
                if not _tensor_equal(edge_index, snapshot["edge_index"]):
                    raise ValueError("snapshot static graph differs from rebuilt graph")
                layout = build_station_layout(world, node_map, edge_index)
                if tuple(layout.station_node_ids) != tuple(
                    int(value)
                    for value in torch.as_tensor(
                        snapshot["station_node_ids"], dtype=torch.long
                    ).tolist()
                ):
                    raise ValueError("snapshot station node ordering changed")

                node_history = torch.as_tensor(
                    snapshot["node_history"], dtype=torch.float32
                )
                edge_features = torch.as_tensor(
                    snapshot["edge_features"], dtype=torch.float32
                )
                demand_context = torch.as_tensor(
                    snapshot["demand_context"], dtype=torch.float32
                )
                z, _, _ = model.encode_state(
                    node_history.to(device),
                    edge_index.to(device),
                    edge_features.to(device),
                    demand_context.to(device),
                )
                station_latents = build_station_representations(
                    z,
                    layout.station_node_ids,
                    representation=REPRESENTATION_STATION_REGION_MEAN_MAX,
                    station_region_node_ids=layout.station_region_node_ids,
                ).detach().cpu()
                station_rows = build_endpoint_station_rows(
                    world, node_history[-1], layout, node_map
                )
                if station_latents.size(0) != len(station_rows):
                    raise AssertionError("station latent/target count mismatch")

                open_orders = sum(
                    _enum_name(order.status) not in {"COMPLETED", "CANCELLED"}
                    for order in world.order_state.orders.values()
                )
                active_ratio = sum(
                    _enum_name(agent.status) != "IDLE" for agent in world.agents
                ) / max(len(world.agents), 1)
                buffer = buffers[split_name]
                for station_offset, station_row in enumerate(station_rows):
                    target = build_station_targets(station_row, scale_contract)
                    raw = target["raw_components"]
                    normalised = target["normalised_components"]
                    channels = target["channels"]
                    buffer["latents"].append(station_latents[station_offset])
                    buffer["targets"].append(torch.tensor(
                        [channels[name] for name in CHANNEL_NAMES],
                        dtype=torch.float32,
                    ))
                    buffer["raw_components"].append(torch.tensor(
                        [raw[name] for name in COMPONENT_NAMES],
                        dtype=torch.float32,
                    ))
                    buffer["normalised_components"].append(torch.tensor(
                        [normalised[name] for name in COMPONENT_NAMES],
                        dtype=torch.float32,
                    ))
                    buffer["frame_group"].append(global_frame_group)
                    buffer["run_index"].append(run_index)
                    buffer["station_id"].append(int(station_row["station_id"]))
                    buffer["seed"].append(seed)
                    buffer["tick"].append(tick)
                    buffer["tick_fraction"].append(tick / max(int(ticks), 1))
                    buffer["arm_code"].append(ARM_NAMES.index("phasec"))
                    buffer["load_code"].append(LOAD_NAMES.index(load))
                    buffer["global_open_order_count"].append(float(open_orders))
                    buffer["global_active_robot_ratio"].append(float(active_ratio))
                    buffer["region_node_count"].append(
                        len(layout.station_region_node_ids[station_offset])
                    )
                global_frame_group += 1
            print(
                f"[encode] PIBT load={load} seed={seed} "
                f"split={split_name} ticks={len(tick_groups)}",
                flush=True,
            )
            run_index += 1

    final_splits = {
        name: _finalise_split(buffer, latent_dim)
        for name, buffer in buffers.items()
    }
    dataset = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "development_only": False,
        "source_protocol_sha256": protocol_sha,
        "source_checkpoint": checkpoint.as_posix(),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_label_schema": checkpoint_label_schema,
        "latent_dim": latent_dim,
        "encoder_latent_dim": encoder_latent_dim,
        "representation": {
            "name": REPRESENTATION_STATION_REGION_MEAN_MAX,
            "primary_region_hops": 3,
            "station_ids_required": True,
            "features": ["station_node", "region_mean", "region_max"],
        },
        "channel_names": list(CHANNEL_NAMES),
        "component_names": list(COMPONENT_NAMES),
        "scale_contract": scale_contract,
        "split_seeds": {
            name: [int(value) for value in values]
            for name, values in splits.items()
        },
        "arm_names": list(ARM_NAMES),
        "load_names": list(LOAD_NAMES),
        "run_ids": run_ids,
        "splits": final_splits,
        "planner_contract": {
            "name": PLANNER_NAME,
            "params": dict(PLANNER_PARAMS),
            "joint_one_step_batch": True,
        },
    }

    staging = output_root.parent / f".{output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    dataset_path = staging / "station_congestion_latents.pt"
    torch.save(dataset, dataset_path)
    copied_scale = staging / "station_congestion_scale_contract.json"
    shutil.copyfile(scale_contract_path, copied_scale)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "source": {
            "snapshot_root": snapshot_root.as_posix(),
            "training_bundle": training_bundle_path.as_posix(),
            "training_protocol_sha256": protocol_sha,
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint),
            "scale_contract": scale_contract_path.as_posix(),
            "scale_contract_file_sha256": sha256_file(scale_contract_path),
            "scale_contract_sha256": scale_contract["contract_sha256"],
        },
        "planner": dataset["planner_contract"],
        "split_seeds": dataset["split_seeds"],
        "snapshot_files": snapshot_files,
        "unique_decision_ticks": unique_ticks,
        "same_tick_context_snapshots_deduplicated": duplicate_context_snapshots,
        "runs": len(run_ids),
        "latent_dim": latent_dim,
        "encoder_latent_dim": encoder_latent_dim,
        "representation": dataset["representation"],
        "samples": {
            name: int(value["targets"].size(0))
            for name, value in final_splits.items()
        },
        "target_stats": {
            name: _target_stats(value)
            for name, value in final_splits.items()
        },
        "audit": {
            "passed": True,
            "pibt_snapshot_provenance_verified": True,
            "same_tick_state_equality_verified": True,
            "same_tick_contexts_deduplicated": True,
            "encoder_checkpoint_bound": True,
            "pp_j1_scale_contract_reused_without_refit": True,
            "station_target_definition_unchanged": True,
        },
    }
    summary_path = staging / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path = staging / "dataset_outputs.sha256"
    manifest_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in sorted(staging.iterdir())
            if path.is_file() and path != manifest_path
        ),
        encoding="utf-8",
    )
    staging.rename(output_root)
    print(f"[complete] PIBT station-head dataset: {output_root}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scale-contract", type=Path, required=True)
    parser.add_argument("--training-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--train-seeds", nargs="+", type=int, default=DEFAULT_SPLITS["train"]
    )
    parser.add_argument(
        "--val-seeds", nargs="+", type=int, default=DEFAULT_SPLITS["val"]
    )
    parser.add_argument(
        "--test-seeds", nargs="+", type=int, default=DEFAULT_SPLITS["test"]
    )
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(max(int(args.torch_threads), 1))
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    splits = {
        "train": tuple(args.train_seeds),
        "val": tuple(args.val_seeds),
        "test": tuple(args.test_seeds),
    }
    all_seeds = [seed for values in splits.values() for seed in values]
    if len(all_seeds) != len(set(all_seeds)):
        raise SystemExit("train/val/test seed splits must be disjoint")
    build_dataset(
        snapshot_root=args.snapshot_root,
        checkpoint=args.checkpoint,
        scale_contract_path=args.scale_contract,
        training_bundle_path=args.training_bundle,
        output_root=args.output_root,
        splits=splits,
        device=device,
        ticks=int(args.ticks),
    )


if __name__ == "__main__":
    main()
