"""Build frozen-encoder station-congestion probe data from 511--520 runs.

The expensive closed-loop simulations are reused read-only.  This builder
aligns each saved latent input frame with the same-tick station trace, fits
component scales on training seeds only, runs the frozen Phase-C encoder, and
writes a compact station-latent dataset for head training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from WorldModel.core.station_congestion_head import (
    CHANNEL_NAMES,
    COMPONENT_NAMES,
    build_station_targets,
    extract_station_components,
    fit_scale_contract,
)
from WorldModel.evaluation.evaluate import _load_model


DATASET_SCHEMA_VERSION = "station_congestion_latent_dataset_v1"
SUMMARY_SCHEMA_VERSION = "station_congestion_dataset_summary_v1"

DEFAULT_INPUT_ROOT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "station_congestion_correlation_dev_511_520_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "station_congestion_head_dev_511_520_v1"
)

REPRESENTATION_STATION_NODE = "station_node"
REPRESENTATION_STATION_REGION_MEAN_MAX = "station_region_mean_max"
REPRESENTATION_CHOICES = (
    REPRESENTATION_STATION_NODE,
    REPRESENTATION_STATION_REGION_MEAN_MAX,
)
DEFAULT_CHECKPOINT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "model_round1_v1/best_regret_world_model.pt"
)

DEFAULT_SPLITS = {
    "train": tuple(range(511, 517)),
    "val": (517, 518),
    "test": (519, 520),
}
ARM_NAMES = ("greedy", "hungarian", "phasec", "phasec_s1")
LOAD_NAMES = ("low", "mid", "high")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
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


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        torch.save(dict(payload), tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_trace(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            tick = int(row["tick"])
            if tick in result:
                raise ValueError(f"duplicate station trace tick {tick}: {path}")
            result[tick] = row
    return result


def _verify_hash_manifest(root: Path, manifest: Path) -> None:
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(None, 1)
        path = root / relative.strip()
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected:
            raise ValueError(f"source run output hash mismatch: {path}")


@dataclass(frozen=True)
class RunSource:
    run_id: str
    arm: str
    load: str
    seed: int
    ticks: int
    run_dir: Path
    trace_path: Path
    td_path: Path


def discover_runs(input_root: Path) -> list[RunSource]:
    runs_root = input_root / "runs"
    if not runs_root.is_dir():
        raise FileNotFoundError(runs_root)
    result = []
    for summary_path in sorted(runs_root.glob("*/station_congestion_summary.json")):
        summary = _read_json(summary_path)
        if not bool((summary.get("audit") or {}).get("passed")):
            raise ValueError(f"source run audit failed: {summary_path}")
        run_dir = summary_path.parent
        manifest = run_dir / "run_outputs.sha256"
        if sha256_file(manifest) != summary.get("run_outputs_sha256"):
            raise ValueError(f"source run manifest hash mismatch: {run_dir}")
        _verify_hash_manifest(run_dir, manifest)
        outputs = summary.get("outputs") or {}
        result.append(RunSource(
            run_id=str(summary["run_id"] if "run_id" in summary else run_dir.name),
            arm=str(summary["arm"]),
            load=str(summary["load"]),
            seed=int(summary["seed"]),
            ticks=int(summary["ticks"]),
            run_dir=run_dir,
            trace_path=run_dir / str(outputs["trace"]),
            td_path=run_dir / str(outputs["td_stream"]),
        ))
    if not result:
        raise ValueError(f"no complete station congestion runs under {runs_root}")
    return result


def split_for_seed(seed: int, splits: Mapping[str, Sequence[int]]) -> str:
    matches = [name for name, seeds in splits.items() if int(seed) in set(seeds)]
    if len(matches) != 1:
        raise ValueError(f"seed {seed} belongs to {len(matches)} splits")
    return matches[0]


def align_station_node_ids(
    station_rows: Sequence[Mapping[str, Any]],
    payload_station_node_ids: Sequence[int],
) -> list[int]:
    """Match trace station order to TD payload station service-node ids.

    Each trace row stores service/entry/exit seed nodes.  The TD payload stores
    only service-node ids, so membership gives a run-local, map-independent
    alignment without assuming dictionary insertion order.
    """

    payload_ids = [int(value) for value in payload_station_node_ids]
    if len(payload_ids) != len(station_rows):
        raise ValueError("station row count differs from station node id count")
    if len(set(payload_ids)) != len(payload_ids):
        raise ValueError("TD payload station node ids are not unique")
    unused = set(payload_ids)
    aligned = []
    for row in station_rows:
        seed_nodes = {int(value) for value in row.get("station_seed_nodes", ())}
        matches = sorted(unused.intersection(seed_nodes))
        if len(matches) != 1:
            raise ValueError(
                "cannot uniquely align station to service node: "
                f"station={row.get('station_id')} matches={matches}"
            )
        aligned.append(matches[0])
        unused.remove(matches[0])
    if unused:
        raise ValueError(f"unaligned station node ids remain: {sorted(unused)}")
    return aligned


def build_station_region_node_ids(
    station_rows: Sequence[Mapping[str, Any]],
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    hops: int = 3,
) -> list[list[int]]:
    """Return deterministic graph-hop regions from trace station seed nodes."""

    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if int(num_nodes) <= 0 or int(hops) < 0:
        raise ValueError("num_nodes must be positive and hops non-negative")
    adjacency = [set() for _ in range(int(num_nodes))]
    for left, right in edge_index.detach().cpu().t().tolist():
        left = int(left)
        right = int(right)
        if not 0 <= left < num_nodes or not 0 <= right < num_nodes:
            raise ValueError("edge_index contains an invalid node id")
        adjacency[left].add(right)
        adjacency[right].add(left)

    regions = []
    for row in station_rows:
        seeds = {
            int(value) for value in row.get("station_seed_nodes", ())
        }
        if not seeds or any(value < 0 or value >= num_nodes for value in seeds):
            raise ValueError(
                f"station {row.get('station_id')} has invalid region seeds"
            )
        visited = set(seeds)
        frontier = set(seeds)
        for _ in range(int(hops)):
            following = set()
            for node_id in frontier:
                following.update(adjacency[node_id])
            following.difference_update(visited)
            if not following:
                break
            visited.update(following)
            frontier = following
        regions.append(sorted(visited))
    return regions


def build_station_representations(
    z: torch.Tensor,
    station_node_ids: Sequence[int],
    *,
    representation: str,
    station_region_node_ids: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """Build the frozen station representation without training parameters."""

    if z.ndim != 2:
        raise ValueError("z must have shape (N, D)")
    ids = torch.as_tensor(
        station_node_ids, dtype=torch.long, device=z.device
    ).reshape(-1)
    if ids.numel() == 0:
        raise ValueError("station_node_ids cannot be empty")
    station_z = z.index_select(0, ids)
    if representation == REPRESENTATION_STATION_NODE:
        return station_z
    if representation != REPRESENTATION_STATION_REGION_MEAN_MAX:
        raise ValueError(f"unknown station representation: {representation}")
    if station_region_node_ids is None:
        raise ValueError("station region node ids are required for region pooling")
    if len(station_region_node_ids) != station_z.size(0):
        raise ValueError("station region count differs from station count")
    rows = []
    for offset, region in enumerate(station_region_node_ids):
        region_ids = torch.as_tensor(
            list(region), dtype=torch.long, device=z.device
        ).reshape(-1)
        if region_ids.numel() == 0:
            raise ValueError("station region cannot be empty")
        if int(region_ids.min().item()) < 0 or int(region_ids.max().item()) >= z.size(0):
            raise ValueError("station region contains an invalid node id")
        region_z = z.index_select(0, region_ids)
        rows.append(torch.cat([
            station_z[offset],
            region_z.mean(dim=0),
            region_z.max(dim=0).values,
        ], dim=0))
    return torch.stack(rows, dim=0)


def _load_td(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "td_stream_v1":
        raise ValueError(f"wrong TD stream schema: {path}")
    frames = payload.get("frames_by_tick")
    if not isinstance(frames, Mapping) or not frames:
        raise ValueError(f"TD stream has no frames: {path}")
    return payload


def _station_rows(trace_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = trace_row.get("stations")
    if not isinstance(rows, list) or not rows:
        raise ValueError("station trace row has no station rows")
    return sorted((dict(row) for row in rows), key=lambda row: int(row["station_id"]))


def collect_training_components(
    runs: Sequence[RunSource],
    train_seeds: Sequence[int],
) -> tuple[list[dict[str, float]], int]:
    train_set = {int(value) for value in train_seeds}
    components: list[dict[str, float]] = []
    frame_count = 0
    for source in runs:
        if source.seed not in train_set:
            continue
        payload = _load_td(source.td_path)
        trace = _read_trace(source.trace_path)
        for tick in sorted(int(value) for value in payload["frames_by_tick"]):
            row = trace.get(tick)
            if row is None:
                raise ValueError(f"missing trace tick {tick}: {source.run_id}")
            frame_count += 1
            components.extend(
                extract_station_components(station_row)
                for station_row in _station_rows(row)
            )
    return components, frame_count


def _new_split_buffer() -> dict[str, list[Any]]:
    return {
        "latents": [],
        "targets": [],
        "raw_components": [],
        "normalised_components": [],
        "frame_group": [],
        "run_index": [],
        "station_id": [],
        "seed": [],
        "tick": [],
        "tick_fraction": [],
        "arm_code": [],
        "load_code": [],
        "global_open_order_count": [],
        "global_active_robot_ratio": [],
        "region_node_count": [],
    }


def _finalise_split(buffer: Mapping[str, list[Any]], latent_dim: int) -> dict[str, Any]:
    count = len(buffer["latents"])
    if count == 0:
        raise ValueError("station congestion dataset split is empty")
    return {
        "latents": torch.stack(buffer["latents"]).reshape(count, latent_dim),
        "targets": torch.stack(buffer["targets"]).reshape(count, len(CHANNEL_NAMES)),
        "raw_components": torch.stack(buffer["raw_components"]).reshape(
            count, len(COMPONENT_NAMES)
        ),
        "normalised_components": torch.stack(
            buffer["normalised_components"]
        ).reshape(count, len(COMPONENT_NAMES)),
        "frame_group": torch.tensor(buffer["frame_group"], dtype=torch.int64),
        "run_index": torch.tensor(buffer["run_index"], dtype=torch.int64),
        "station_id": torch.tensor(buffer["station_id"], dtype=torch.int64),
        "seed": torch.tensor(buffer["seed"], dtype=torch.int64),
        "tick": torch.tensor(buffer["tick"], dtype=torch.int64),
        "tick_fraction": torch.tensor(
            buffer["tick_fraction"], dtype=torch.float32
        ),
        "arm_code": torch.tensor(buffer["arm_code"], dtype=torch.int64),
        "load_code": torch.tensor(buffer["load_code"], dtype=torch.int64),
        "global_open_order_count": torch.tensor(
            buffer["global_open_order_count"], dtype=torch.float32
        ),
        "global_active_robot_ratio": torch.tensor(
            buffer["global_active_robot_ratio"], dtype=torch.float32
        ),
        "region_node_count": torch.tensor(
            buffer["region_node_count"], dtype=torch.int64
        ),
    }


def _target_stats(split: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    targets = split["targets"].to(torch.float64)
    result = {}
    for index, name in enumerate(CHANNEL_NAMES):
        values = targets[:, index]
        result[name] = {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }
    return result


def build_dataset(
    *,
    input_root: Path,
    checkpoint: Path,
    output_root: Path,
    splits: Mapping[str, Sequence[int]],
    device: torch.device,
    lower_quantile: float,
    upper_quantile: float,
    representation: str = REPRESENTATION_STATION_NODE,
    max_runs: int | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(
            f"output already exists; preserve it or choose a new path: {output_root}"
        )
    if representation not in REPRESENTATION_CHOICES:
        raise ValueError(f"unsupported station representation: {representation}")
    frozen_bundle_path = input_root / "phase_c_station_congestion_frozen_protocol.json"
    frozen_bundle = _read_json(frozen_bundle_path)
    protocol = frozen_bundle.get("protocol") or {}
    protocol_sha = str(protocol.get("protocol_sha256", ""))
    if not protocol_sha:
        raise ValueError("source station congestion protocol lacks sha256")

    runs = discover_runs(input_root)
    selected_seeds = {int(value) for values in splits.values() for value in values}
    runs = [source for source in runs if source.seed in selected_seeds]
    if max_runs is not None:
        runs = runs[: int(max_runs)]
    if not runs:
        raise ValueError("no source runs remain after seed selection")
    for source in runs:
        if source.arm not in ARM_NAMES or source.load not in LOAD_NAMES:
            raise ValueError(f"unsupported source run: {source}")
        split_for_seed(source.seed, splits)

    components, scale_frame_count = collect_training_components(
        runs, splits["train"]
    )
    scale_contract = fit_scale_contract(
        components,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        fitted_seeds=splits["train"],
        source_protocol_sha256=protocol_sha,
    )

    model, checkpoint_label_schema = _load_model(str(checkpoint))
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoder_latent_dim = int(model.state_encoder.temporal.gru.hidden_size)
    representation_dim = (
        encoder_latent_dim
        if representation == REPRESENTATION_STATION_NODE
        else encoder_latent_dim * 3
    )

    buffers = {name: _new_split_buffer() for name in splits}
    run_ids = []
    global_frame_group = 0
    with torch.no_grad():
        for run_index, source in enumerate(runs):
            run_ids.append(source.run_id)
            split_name = split_for_seed(source.seed, splits)
            buffer = buffers[split_name]
            payload = _load_td(source.td_path)
            trace = _read_trace(source.trace_path)
            frames = payload["frames_by_tick"]
            payload_station_ids = payload.get("station_node_ids") or []
            first_tick = min(int(value) for value in frames)
            first_trace_row = trace.get(first_tick)
            if first_trace_row is None:
                raise ValueError(
                    f"missing first trace tick {first_tick}: {source.run_id}"
                )
            template_station_rows = _station_rows(first_trace_row)
            aligned_node_ids = align_station_node_ids(
                template_station_rows, payload_station_ids
            )
            station_regions = build_station_region_node_ids(
                template_station_rows,
                payload["edge_index"],
                num_nodes=int(first_trace_row["system"].get(
                    "node_count", frames[first_tick]["node_history"].size(1)
                )),
                hops=3,
            )
            for tick in sorted(int(value) for value in frames):
                trace_row = trace.get(tick)
                if trace_row is None:
                    raise ValueError(f"missing trace tick {tick}: {source.run_id}")
                station_rows = _station_rows(trace_row)
                if [int(row["station_id"]) for row in station_rows] != [
                    int(row["station_id"]) for row in template_station_rows
                ]:
                    raise ValueError("station order changed within a source run")
                frame = frames[tick]
                z, _, _ = model.encode_state(
                    frame["node_history"].to(device),
                    payload["edge_index"].to(device),
                    frame["edge_features"].to(device),
                    frame["demand_context"].to(device),
                )
                station_latents = build_station_representations(
                    z,
                    aligned_node_ids,
                    representation=representation,
                    station_region_node_ids=station_regions,
                ).detach().cpu()
                if station_latents.size(0) != len(station_rows):
                    raise AssertionError("station latent/label count mismatch")
                system = trace_row.get("system") or {}
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
                    buffer["seed"].append(source.seed)
                    buffer["tick"].append(tick)
                    buffer["tick_fraction"].append(tick / max(source.ticks, 1))
                    buffer["arm_code"].append(ARM_NAMES.index(source.arm))
                    buffer["load_code"].append(LOAD_NAMES.index(source.load))
                    buffer["global_open_order_count"].append(
                        float(system.get("open_order_count", 0.0))
                    )
                    buffer["global_active_robot_ratio"].append(
                        float(system.get("active_robot_ratio", 0.0))
                    )
                    buffer["region_node_count"].append(
                        len(station_regions[station_offset])
                        if representation == REPRESENTATION_STATION_REGION_MEAN_MAX
                        else 1
                    )
                global_frame_group += 1
            print(
                f"[encode] {run_index + 1}/{len(runs)} {source.run_id} "
                f"split={split_name} frames={len(frames)}",
                flush=True,
            )

    final_splits = {
        name: _finalise_split(buffer, representation_dim)
        for name, buffer in buffers.items()
    }
    dataset = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "development_only": True,
        "source_protocol_sha256": protocol_sha,
        "source_checkpoint": checkpoint.as_posix(),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_label_schema": checkpoint_label_schema,
        "latent_dim": representation_dim,
        "encoder_latent_dim": encoder_latent_dim,
        "representation": {
            "name": representation,
            "primary_region_hops": 3,
            "station_ids_required": True,
            "features": (
                ["station_node"]
                if representation == REPRESENTATION_STATION_NODE
                else ["station_node", "region_mean", "region_max"]
            ),
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
    }

    staging = output_root.parent / f".{output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    dataset_path = staging / "station_congestion_latents.pt"
    scale_path = staging / "station_congestion_scale_contract.json"
    _atomic_torch_save(dataset_path, dataset)
    _atomic_json(scale_path, scale_contract)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "development_only": True,
        "source": {
            "input_root": input_root.as_posix(),
            "source_protocol_sha256": protocol_sha,
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint),
        },
        "split_seeds": dataset["split_seeds"],
        "runs": len(runs),
        "scale_fit_frames": scale_frame_count,
        "scale_fit_station_rows": len(components),
        "latent_dim": representation_dim,
        "encoder_latent_dim": encoder_latent_dim,
        "representation": dataset["representation"],
        "channels": list(CHANNEL_NAMES),
        "components": list(COMPONENT_NAMES),
        "samples": {
            name: int(value["targets"].size(0))
            for name, value in final_splits.items()
        },
        "target_stats": {
            name: _target_stats(value)
            for name, value in final_splits.items()
        },
        "scale_contract_sha256": scale_contract["contract_sha256"],
        "dataset_sha256": sha256_file(dataset_path),
        "audit": {
            "passed": True,
            "encoder_frozen": True,
            "station_ids_required": True,
            "fixed_station_output_dimension": False,
            "locked_501_510_used": False,
            "train_scale_only": True,
            "cross_channel_cancellation": False,
        },
    }
    summary_path = staging / "dataset_summary.json"
    _atomic_json(summary_path, summary)
    manifest_path = staging / "dataset_outputs.sha256"
    rows = []
    for path in sorted(staging.iterdir()):
        if path.is_file() and path.name != manifest_path.name:
            rows.append(f"{sha256_file(path)}  {path.name}")
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    staging.rename(output_root)
    print(f"[complete] station congestion latent dataset: {output_root}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=DEFAULT_SPLITS["train"])
    parser.add_argument("--val-seeds", nargs="+", type=int, default=DEFAULT_SPLITS["val"])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=DEFAULT_SPLITS["test"])
    parser.add_argument("--lower-quantile", type=float, default=0.05)
    parser.add_argument("--upper-quantile", type=float, default=0.95)
    parser.add_argument(
        "--representation",
        choices=REPRESENTATION_CHOICES,
        default=REPRESENTATION_STATION_NODE,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-runs", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(max(int(args.torch_threads), 1))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    splits = {
        "train": tuple(args.train_seeds),
        "val": tuple(args.val_seeds),
        "test": tuple(args.test_seeds),
    }
    all_seeds = [seed for values in splits.values() for seed in values]
    if len(all_seeds) != len(set(all_seeds)):
        raise SystemExit("train/val/test seed splits must be disjoint")
    print(f"[device] {device}")
    build_dataset(
        input_root=args.input_root,
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        splits=splits,
        device=device,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        representation=args.representation,
        max_runs=args.max_runs,
    )


if __name__ == "__main__":
    main()
