"""Materialise and fuse the behavior-aligned ``psi_pre`` latent dataset.

The expensive operation in this module is the frozen Phase-C encoder and
H=10 latent transition.  It is performed once per behavior sample and saved
as compact per-run shards.  Head training and all baselines then reuse those
shards without touching the simulator or the frozen model again.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from WorldModel.core.psi_pre_head import PsiPreHead
from WorldModel.data.dataset import WorldModelDataset
from WorldModel.evaluation.evaluate import _load_model
from WorldModel.evaluation.phase_c_psi_pre_protocol import (
    BEHAVIOR_ROOT,
    DATASET_SCHEMA_VERSION,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    HORIZON,
    LOADS,
    MODEL_CHECKPOINT,
    OUTPUT_ROOT,
    REGION_HOPS,
    SEEDS,
    SHARD_SCHEMA_VERSION,
    TARGET_CHANNELS,
    canonical_sha256,
    expected_run_files,
    fold_seed_splits,
    parse_run_name,
    sha256_file,
)


NODE_TARGET_COLUMNS = {
    "region_density": 1,
    "region_wait": 2,
    "region_blocked": 3,
    "region_congestion": 5,
}
# NodeDecoder channels 0, 2 and 3 are trained with BCE-with-logits.  The
# psi_pre region targets use channels 1, 2, 3 and 5, so only 2 and 3 need to
# be transformed when a frozen decoder prediction is used as a [0, 1]
# baseline.  The default remains False to preserve the already materialised
# v1 artifacts byte-for-byte.
NODE_BCE_LOGIT_COLUMNS = (0, 2, 3)
LOAD_CODE = {name: index for index, name in enumerate(LOADS)}


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


def _resolve(path: str | Path, repo_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _load_bundle(path: Path) -> dict[str, Any]:
    bundle = _read_json(path)
    # v2 uses a distinct bundle schema while deliberately reusing this
    # materialiser.  Keep the historical v1 schema accepted so old runs are
    # untouched and auditable.
    if bundle.get("schema_version") not in {
        FROZEN_BUNDLE_SCHEMA_VERSION,
        "phase_c_psi_pre_bundle_v2",
    }:
        raise ValueError(f"wrong psi_pre frozen bundle schema: {path}")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("psi_pre bundle lacks protocol")
    if canonical_sha256(protocol) != bundle.get("protocol_sha256"):
        raise ValueError("psi_pre protocol hash mismatch")
    return bundle


def _verify_source_rows(
    bundle: Mapping[str, Any],
    repo_root: Path,
    *,
    verify_hashes: bool = True,
) -> list[dict[str, Any]]:
    artifacts = bundle.get("artifacts") or {}
    rows = artifacts.get("behavior_runs") or []
    if len(rows) != len(LOADS) * len(SEEDS):
        raise ValueError("psi_pre source artifact coverage is incomplete")
    for row in rows:
        data_path = _resolve(row["data_path"], repo_root)
        meta_path = _resolve(row["meta_path"], repo_root)
        if not data_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"missing source artifact: {data_path} / {meta_path}")
        if verify_hashes:
            if sha256_file(data_path) != row["data_sha256"]:
                raise ValueError(f"source data hash mismatch: {data_path}")
            if sha256_file(meta_path) != row["meta_sha256"]:
                raise ValueError(f"source metadata hash mismatch: {meta_path}")
    checkpoint = artifacts.get("model_checkpoint") or {}
    checkpoint_path = _resolve(checkpoint["path"], repo_root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if verify_hashes and sha256_file(checkpoint_path) != checkpoint["sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")
    return [dict(row) for row in rows]


def _build_regions(
    edge_index: torch.Tensor,
    station_node_ids: Sequence[int] | torch.Tensor,
    num_nodes: int,
    hops: int,
) -> list[list[int]]:
    edge_index = torch.as_tensor(edge_index, dtype=torch.long).cpu()
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape (2, E)")
    adjacency = [set() for _ in range(int(num_nodes))]
    for left, right in edge_index.t().tolist():
        if not (0 <= int(left) < num_nodes and 0 <= int(right) < num_nodes):
            raise ValueError("edge_index contains an invalid node")
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    result = []
    for station in torch.as_tensor(station_node_ids, dtype=torch.long).reshape(-1).tolist():
        station = int(station)
        if station < 0 or station >= num_nodes:
            raise ValueError(f"station node id outside graph: {station}")
        visited = {station}
        frontier = {station}
        for _ in range(int(hops)):
            following = set()
            for node in frontier:
                following.update(adjacency[node])
            following.difference_update(visited)
            visited.update(following)
            frontier = following
            if not frontier:
                break
        result.append(sorted(visited))
    return result


def _station_region_latents(
    z: torch.Tensor,
    station_node_ids: Sequence[int] | torch.Tensor,
    regions: Sequence[Sequence[int]],
) -> torch.Tensor:
    return PsiPreHead.concat_station_region(z, station_node_ids, regions)


def _aggregate_node_values(
    node_values: torch.Tensor,
    regions: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Return eight mean/max channels in the frozen target order."""

    if node_values.ndim != 2 or node_values.size(1) < 6:
        raise ValueError("node values must have shape (N, >=6)")
    rows = []
    for region in regions:
        ids = torch.as_tensor(list(region), dtype=torch.long, device=node_values.device)
        values = node_values.index_select(0, ids)
        output = []
        for column in NODE_TARGET_COLUMNS.values():
            selected = values[:, int(column)]
            output.extend([selected.mean(), selected.max()])
        rows.append(torch.stack(output))
    return torch.stack(rows, dim=0)


def _endpoint_target(
    sample: Mapping[str, Any],
    regions: Sequence[Sequence[int]],
    horizon: int,
) -> torch.Tensor:
    mask = torch.as_tensor(sample.get("future_mask", torch.ones(horizon)))
    if mask.numel() < horizon or float(mask[horizon - 1].item()) < 0.5:
        raise ValueError("sample does not contain a complete endpoint label")
    station = torch.as_tensor(sample["future_station_labels"])[horizon - 1].float()
    if station.ndim != 2 or station.size(1) < 2:
        raise ValueError("future_station_labels has an invalid shape")
    node = torch.as_tensor(sample["future_node_labels"])[horizon - 1].float()
    node_channels = _aggregate_node_values(node, regions)
    return torch.cat([station[:, :2], node_channels], dim=1)


def _baseline_channels(
    node_prediction: torch.Tensor,
    station_prediction: torch.Tensor,
    regions: Sequence[Sequence[int]],
    *,
    apply_node_bce_sigmoid: bool = False,
) -> torch.Tensor:
    station_prediction = torch.as_tensor(station_prediction).float()
    if station_prediction.ndim != 2 or station_prediction.size(1) < 2:
        raise ValueError("station prediction has an invalid shape")
    node_prediction = node_prediction.float()
    if apply_node_bce_sigmoid:
        if node_prediction.ndim != 2 or node_prediction.size(1) <= max(NODE_BCE_LOGIT_COLUMNS):
            raise ValueError("node prediction lacks BCE-logit channels")
        # Do this before region mean/max aggregation.  Applying sigmoid after
        # aggregation is not equivalent for the mean channel and would make
        # the decoder baseline mis-scaled.
        node_prediction = node_prediction.clone()
        for column in NODE_BCE_LOGIT_COLUMNS:
            node_prediction[:, int(column)] = torch.sigmoid(
                node_prediction[:, int(column)]
            )
    node_channels = _aggregate_node_values(node_prediction, regions)
    return torch.cat([station_prediction[:, :2], node_channels], dim=1)


def _run_output_paths(output_root: Path, run_id: str) -> tuple[Path, Path]:
    shard_dir = output_root / "shards"
    return shard_dir / f"{run_id}.pt", shard_dir / f"{run_id}.json"


def materialize_run(
    *,
    data_path: Path,
    output_root: Path,
    checkpoint: Path,
    protocol_sha256: str,
    horizon: int = HORIZON,
    region_hops: int = REGION_HOPS,
    torch_threads: int = 4,
    max_samples: int = 0,
    apply_node_bce_sigmoid: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    run_id = data_path.parent.name
    load, seed = parse_run_name(run_id)
    shard_path, meta_path = _run_output_paths(output_root, run_id)
    if resume and shard_path.is_file() and meta_path.is_file():
        meta = _read_json(meta_path)
        if (
            meta.get("schema_version") == SHARD_SCHEMA_VERSION
            and meta.get("source_sha256") == sha256_file(data_path)
            and meta.get("protocol_sha256") == protocol_sha256
            and meta.get("max_samples") == int(max_samples)
            and bool(meta.get("apply_node_bce_sigmoid", False))
            == bool(apply_node_bce_sigmoid)
        ):
            print(f"[resume] {run_id} shard is valid", flush=True)
            return meta

    try:
        torch.set_num_threads(max(1, int(torch_threads)))
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    dataset = WorldModelDataset.from_file(str(data_path))
    samples = list(dataset.samples)
    if max_samples > 0:
        samples = samples[: int(max_samples)]
    model, checkpoint_schema = _load_model(str(checkpoint))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    buffers: dict[str, list[Any]] = {
        "state_latents": [],
        "rollout_latents": [],
        "targets": [],
        "baseline_state": [],
        "baseline_h10": [],
        "baseline_h1": [],
        "frame_group": [],
        "candidate_group": [],
        "station_index": [],
        "station_node_id": [],
        "decision_tick": [],
    }
    complete_samples = 0
    graph_cache: dict[tuple[int, ...], list[list[int]]] = {}
    group_ids: dict[str, int] = {}
    with torch.no_grad():
        for sample_index, sample in enumerate(samples):
            station_ids = torch.as_tensor(sample["station_node_ids"], dtype=torch.long).reshape(-1)
            key = tuple(
                [int(sample["node_history"].size(1))]
                + station_ids.tolist()
                + torch.as_tensor(sample["edge_index"], dtype=torch.long).reshape(-1).tolist()
            )
            regions = graph_cache.get(key)
            if regions is None:
                regions = _build_regions(
                    sample["edge_index"], station_ids,
                    int(sample["node_history"].size(1)), region_hops
                )
                graph_cache[key] = regions
            try:
                target = _endpoint_target(sample, regions, horizon)
                z, e_demand, edge_attr = model.encode_state(
                    sample["node_history"],
                    sample["edge_index"],
                    sample["edge_features"],
                    sample["demand_context"],
                )
                rollout = model.rollout(
                    z,
                    e_demand,
                    edge_attr,
                    sample["action_node"],
                    sample["action_global"],
                    sample["edge_index"],
                    station_ids.tolist(),
                    K=horizon,
                )
                node_preds, _, station_preds, _, z0, zH = rollout
                state_repr = _station_region_latents(z0, station_ids, regions)
                endpoint_repr = _station_region_latents(zH, station_ids, regions)
                state_node_prediction, _ = model.node_decoder(z0)
                state_station_prediction = model.station_decoder(
                    z0, station_ids.tolist()
                )
                baseline_state = _baseline_channels(
                    state_node_prediction, state_station_prediction, regions,
                    apply_node_bce_sigmoid=apply_node_bce_sigmoid,
                )
                baseline_h10 = _baseline_channels(
                    node_preds[horizon - 1], station_preds[horizon - 1], regions,
                    apply_node_bce_sigmoid=apply_node_bce_sigmoid,
                )
                baseline_h1 = _baseline_channels(
                    node_preds[0], station_preds[0], regions,
                    apply_node_bce_sigmoid=apply_node_bce_sigmoid,
                )
            except (KeyError, IndexError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"invalid psi_pre sample {run_id}[{sample_index}]: {exc}"
                ) from exc
            group_key = str(sample.get("candidate_group_id") or sample_index)
            if group_key not in group_ids:
                group_ids[group_key] = len(group_ids)
            frame_group = int(sample_index)
            stations = int(target.size(0))
            if not (
                state_repr.size(0) == endpoint_repr.size(0) == target.size(0)
                == baseline_state.size(0)
                == baseline_h10.size(0) == baseline_h1.size(0)
            ):
                raise RuntimeError(f"station row mismatch in {run_id}[{sample_index}]")
            for station_offset in range(stations):
                buffers["state_latents"].append(state_repr[station_offset].cpu())
                buffers["rollout_latents"].append(endpoint_repr[station_offset].cpu())
                buffers["targets"].append(target[station_offset].cpu())
                buffers["baseline_state"].append(baseline_state[station_offset].cpu())
                buffers["baseline_h10"].append(baseline_h10[station_offset].cpu())
                buffers["baseline_h1"].append(baseline_h1[station_offset].cpu())
                buffers["frame_group"].append(frame_group)
                buffers["candidate_group"].append(group_ids[group_key])
                buffers["station_index"].append(station_offset)
                buffers["station_node_id"].append(int(station_ids[station_offset]))
                buffers["decision_tick"].append(int(sample.get("decision_tick", -1)))
            complete_samples += 1

    if not buffers["targets"]:
        raise RuntimeError(f"no valid psi_pre rows in {run_id}")
    payload = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "protocol_sha256": protocol_sha256,
        "source_path": data_path.as_posix(),
        "source_sha256": sha256_file(data_path),
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_schema": checkpoint_schema,
        "run_id": run_id,
        "load": load,
        "seed": int(seed),
        "horizon": int(horizon),
        "region_hops": int(region_hops),
        "target_channels": list(TARGET_CHANNELS),
        "representation": {
            "name": "station_region_mean_max",
            "features": ["station_node", "region_mean", "region_max"],
            "latent_dim": int(buffers["rollout_latents"][0].numel()),
        },
        "state_latents": torch.stack(buffers["state_latents"]).float(),
        "rollout_latents": torch.stack(buffers["rollout_latents"]).float(),
        "targets": torch.stack(buffers["targets"]).float(),
        "baseline_state": torch.stack(buffers["baseline_state"]).float(),
        "baseline_h10": torch.stack(buffers["baseline_h10"]).float(),
        "baseline_h1": torch.stack(buffers["baseline_h1"]).float(),
        "frame_group": torch.tensor(buffers["frame_group"], dtype=torch.int64),
        "candidate_group": torch.tensor(buffers["candidate_group"], dtype=torch.int64),
        "station_index": torch.tensor(buffers["station_index"], dtype=torch.int64),
        "station_node_id": torch.tensor(buffers["station_node_id"], dtype=torch.int64),
        "decision_tick": torch.tensor(buffers["decision_tick"], dtype=torch.int64),
    }
    if apply_node_bce_sigmoid:
        # Keep the historical v1 payload shape unchanged when the optional
        # correction is not requested.
        payload["baseline_node_bce_sigmoid"] = True
    meta = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "protocol_sha256": protocol_sha256,
        "run_id": run_id,
        "load": load,
        "seed": int(seed),
        "source_path": data_path.as_posix(),
        "source_sha256": payload["source_sha256"],
        "samples_seen": len(samples),
        "complete_samples": complete_samples,
        "rows": int(payload["targets"].size(0)),
        "latent_dim": int(payload["rollout_latents"].size(1)),
        "horizon": int(horizon),
        "region_hops": int(region_hops),
        "max_samples": int(max_samples),
        "station_count_variable": True,
        "online_connection_allowed": False,
    }
    if apply_node_bce_sigmoid:
        meta["apply_node_bce_sigmoid"] = True
    _atomic_torch_save(shard_path, payload)
    meta["shard_sha256"] = sha256_file(shard_path)
    _atomic_json(meta_path, meta)
    print(f"[materialize] {run_id}: samples={complete_samples} rows={meta['rows']}", flush=True)
    return meta


def _materialize_task(task: tuple[Any, ...]) -> dict[str, Any]:
    return materialize_run(
        data_path=Path(task[0]),
        output_root=Path(task[1]),
        checkpoint=Path(task[2]),
        protocol_sha256=str(task[3]),
        horizon=int(task[4]),
        region_hops=int(task[5]),
        torch_threads=int(task[6]),
        max_samples=int(task[7]),
        apply_node_bce_sigmoid=bool(task[8]),
        resume=bool(task[9]),
    )


def materialize_all(
    *,
    bundle_path: Path,
    repo_root: Path,
    output_root: Path,
    checkpoint: Path,
    workers: int,
    torch_threads: int,
    max_samples: int,
    apply_node_bce_sigmoid: bool = False,
    resume: bool = True,
    verify_hashes: bool = True,
) -> list[dict[str, Any]]:
    bundle = _load_bundle(bundle_path)
    source_rows = _verify_source_rows(
        bundle, repo_root, verify_hashes=bool(verify_hashes)
    )
    protocol_sha = str(bundle["protocol_sha256"])
    frozen_checkpoint = _resolve(
        (bundle.get("artifacts") or {}).get("model_checkpoint", {}).get("path", ""),
        repo_root,
    )
    if checkpoint.resolve() != frozen_checkpoint.resolve():
        if not checkpoint.is_file() or sha256_file(checkpoint) != (
            bundle.get("artifacts") or {}
        ).get("model_checkpoint", {}).get("sha256"):
            raise ValueError("CLI checkpoint does not match frozen psi_pre checkpoint")
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            str(_resolve(row["data_path"], repo_root)),
            str(output_root),
            str(checkpoint),
            protocol_sha,
            HORIZON,
            REGION_HOPS,
            torch_threads,
            max_samples,
            apply_node_bce_sigmoid,
            resume,
        )
        for row in source_rows
    ]
    results = []
    if int(workers) <= 1:
        for task in tasks:
            results.append(_materialize_task(task))
    else:
        context = __import__("multiprocessing").get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(workers), mp_context=context
        ) as executor:
            futures = [executor.submit(_materialize_task, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda row: row["run_id"])
    _atomic_json(output_root / "materialization_summary.json", {
        "schema_version": "phase_c_psi_pre_materialization_summary_v1",
        "protocol_sha256": protocol_sha,
        "expected_runs": len(tasks),
        "completed_runs": len(results),
        "runs": results,
    })
    if len(results) != len(tasks):
        raise RuntimeError("psi_pre materialization is incomplete")
    return results


def _load_shard(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SHARD_SCHEMA_VERSION:
        raise ValueError(f"wrong psi_pre shard schema: {path}")
    if tuple(payload.get("target_channels", ())) != TARGET_CHANNELS:
        raise ValueError(f"psi_pre target channel mismatch: {path}")
    required = {
        "state_latents", "rollout_latents", "targets", "baseline_state",
        "baseline_h10", "baseline_h1", "frame_group", "candidate_group",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"psi_pre shard missing fields {missing}: {path}")
    return payload


def fuse_shards(
    *,
    bundle_path: Path,
    repo_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    bundle = _load_bundle(bundle_path)
    # Materialisation already verified every source hash.  Fusion still checks
    # that all paths exist and that each shard metadata carries the frozen
    # source digest, avoiding a second multi-hundred-MB scan.
    source_rows = _verify_source_rows(bundle, repo_root, verify_hashes=False)
    protocol_sha = str(bundle["protocol_sha256"])
    by_run = {str(row["run_id"]): row for row in source_rows}
    shard_payloads = []
    for run_id in sorted(by_run):
        shard_path, meta_path = _run_output_paths(output_root, run_id)
        if not shard_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"missing psi_pre shard for {run_id}")
        meta = _read_json(meta_path)
        if meta.get("protocol_sha256") != protocol_sha:
            raise ValueError(f"protocol mismatch in shard metadata: {run_id}")
        if meta.get("shard_sha256") != sha256_file(shard_path):
            raise ValueError(f"shard hash mismatch: {run_id}")
        if meta.get("source_sha256") != by_run[run_id]["data_sha256"]:
            raise ValueError(f"source hash mismatch in shard: {run_id}")
        shard_payloads.append(_load_shard(shard_path))

    baseline_contracts = {
        bool(payload.get("baseline_node_bce_sigmoid", False))
        for payload in shard_payloads
    }
    if len(baseline_contracts) != 1:
        raise ValueError("mixed decoder baseline activation contracts across shards")
    baseline_node_bce_sigmoid = baseline_contracts.pop()

    keys = (
        "state_latents", "rollout_latents", "targets", "baseline_state", "baseline_h10",
        "baseline_h1", "frame_group", "candidate_group", "station_index",
        "station_node_id", "decision_tick",
    )
    run_indices = []
    seed_values = []
    load_values = []
    run_names = []
    global_candidate_groups = []
    candidate_group_offset = 0
    global_frame_groups = []
    frame_group_offset = 0
    fused: dict[str, torch.Tensor] = {}
    offset = 0
    for run_index, payload in enumerate(shard_payloads):
        rows = int(payload["targets"].size(0))
        for key in keys:
            value = payload[key]
            fused.setdefault(key, [])
            fused[key].append(value)
        run_indices.extend([run_index] * rows)
        seed_values.extend([int(payload["seed"])] * rows)
        load_values.extend([LOADS.index(str(payload["load"]))] * rows)
        run_names.append(str(payload["run_id"]))
        local_groups = payload["candidate_group"].long()
        global_candidate_groups.extend(
            (local_groups + int(candidate_group_offset)).tolist()
        )
        local_group_count = (
            int(local_groups.max().item()) + 1 if local_groups.numel() else 0
        )
        candidate_group_offset += local_group_count
        local_frames = payload["frame_group"].long()
        global_frame_groups.extend(
            (local_frames + int(frame_group_offset)).tolist()
        )
        local_frame_count = (
            int(local_frames.max().item()) + 1 if local_frames.numel() else 0
        )
        frame_group_offset += local_frame_count
        offset += rows
    fused = {key: torch.cat(values, dim=0) for key, values in fused.items()}
    fused["run_index"] = torch.tensor(run_indices, dtype=torch.int64)
    fused["seed"] = torch.tensor(seed_values, dtype=torch.int64)
    fused["load_code"] = torch.tensor(load_values, dtype=torch.int64)

    fused["candidate_group"] = torch.tensor(
        global_candidate_groups, dtype=torch.int64
    )
    fused["frame_group"] = torch.tensor(
        global_frame_groups, dtype=torch.int64
    )
    run_row_counts = torch.bincount(
        fused["run_index"], minlength=len(run_names)
    )
    group_counts = torch.bincount(fused["candidate_group"]).float()
    groups_per_run = []
    for run_index in range(len(run_names)):
        groups_per_run.append(
            int(torch.unique(
                fused["candidate_group"][fused["run_index"] == run_index]
            ).numel())
        )
    groups_per_run_tensor = torch.tensor(groups_per_run, dtype=torch.float32)
    row_weights = 1.0 / (
        group_counts.index_select(0, fused["candidate_group"])
        * groups_per_run_tensor.index_select(0, fused["run_index"])
    )
    fused["row_weight"] = row_weights

    folds = {}
    for fold in fold_seed_splits():
        entries = {}
        for split_name, seeds in (
            ("train", fold["train_seeds"]),
            ("val", fold["val_seeds"]),
            ("test", fold["test_seeds"]),
        ):
            mask = torch.zeros_like(fused["seed"], dtype=torch.bool)
            for seed in seeds:
                mask |= fused["seed"] == int(seed)
            entries[split_name] = torch.nonzero(
                mask, as_tuple=False
            ).reshape(-1)
        folds[str(fold["fold"])] = entries

    dataset = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "development_only": True,
        "protocol_sha256": protocol_sha,
        "source_checkpoint": bundle["artifacts"]["model_checkpoint"]["path"],
        "source_checkpoint_sha256": bundle["artifacts"]["model_checkpoint"]["sha256"],
        "horizon": HORIZON,
        "region_hops": REGION_HOPS,
        "target_channels": list(TARGET_CHANNELS),
        "service_channels": [0, 1],
        "traffic_channels": list(range(2, len(TARGET_CHANNELS))),
        "representation": {
            "name": "station_region_mean_max",
            "features": ["station_node", "region_mean", "region_max"],
            "station_count_variable": True,
            "station_id_embedding": False,
        },
        "latent_dim": int(fused["rollout_latents"].size(1)),
        "row_count": int(fused["targets"].size(0)),
        "run_names": run_names,
        "run_keys": [f"{payload['load']}|{payload['seed']}" for payload in shard_payloads],
        "split_seeds": fold_seed_splits(),
        "folds": folds,
        "weight_contract": "each source run has total weight one; groups are not pair-count weighted",
        "target_scale_contract": "source behavior H=10 labels, no test-fold fitting",
        "baseline_names": [
            "decoder_state_persistence",
            "decoder_h10",
            "decoder_h1_persistence",
        ],
        "tensors": fused,
    }
    if baseline_node_bce_sigmoid:
        dataset["baseline_node_bce_sigmoid"] = True
    dataset_path = output_root / "psi_pre_latents.pt"
    _atomic_torch_save(dataset_path, dataset)
    summary = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "development_only": True,
        "protocol_sha256": protocol_sha,
        "dataset": dataset_path.name,
        "dataset_sha256": sha256_file(dataset_path),
        "row_count": int(fused["targets"].size(0)),
        "latent_dim": int(fused["rollout_latents"].size(1)),
        "runs": len(run_names),
        "samples_by_run": {
            run_names[index]: int(run_row_counts[index].item())
            for index in range(len(run_names))
        },
        "rows_by_load": {
            load: int((fused["load_code"] == code).sum().item())
            for code, load in enumerate(LOADS)
        },
        "rows_by_seed": {
            str(seed): int((fused["seed"] == seed).sum().item())
            for seed in SEEDS
        },
        "target_channels": list(TARGET_CHANNELS),
        "folds": fold_seed_splits(),
        "online_connection_allowed": False,
        "state_latents_retained_for_future_delta_test": True,
    }
    summary_path = output_root / "dataset_summary.json"
    _atomic_json(summary_path, summary)
    manifest = "\n".join([
        f"{sha256_file(dataset_path)}  {dataset_path.name}",
        f"{sha256_file(summary_path)}  {summary_path.name}",
        f"{sha256_file(bundle_path)}  {bundle_path.name}",
    ]) + "\n"
    (output_root / "dataset_outputs.sha256").write_text(
        manifest, encoding="utf-8", newline="\n"
    )
    print(f"[fuse] rows={summary['row_count']} runs={summary['runs']} dataset={dataset_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("materialize-run", "materialize-all", "fuse"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_CHECKPOINT)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--node-bce-sigmoid",
        action="store_true",
        help=(
            "apply sigmoid to NodeDecoder BCE-logit channels before region "
            "aggregation; omitted for the historical v1 contract"
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--skip-source-hash",
        action="store_true",
        help="trust the already-verified frozen_inputs.sha256 manifest",
    )
    args = parser.parse_args()
    if args.mode == "materialize-run":
        if args.data_path is None:
            raise SystemExit("--data-path is required for materialize-run")
        bundle = _load_bundle(args.bundle)
        materialize_run(
            data_path=args.data_path,
            output_root=args.output_root,
            checkpoint=args.checkpoint,
            protocol_sha256=str(bundle["protocol_sha256"]),
            torch_threads=args.torch_threads,
            max_samples=args.max_samples,
            apply_node_bce_sigmoid=args.node_bce_sigmoid,
            resume=not args.no_resume,
        )
    elif args.mode == "materialize-all":
        materialize_all(
            bundle_path=args.bundle,
            repo_root=args.repo_root,
            output_root=args.output_root,
            checkpoint=args.checkpoint,
            workers=args.workers,
            torch_threads=args.torch_threads,
            max_samples=args.max_samples,
            apply_node_bce_sigmoid=args.node_bce_sigmoid,
            resume=not args.no_resume,
            verify_hashes=not args.skip_source_hash,
        )
    else:
        fuse_shards(
            bundle_path=args.bundle,
            repo_root=args.repo_root,
            output_root=args.output_root,
        )


if __name__ == "__main__":
    main()
