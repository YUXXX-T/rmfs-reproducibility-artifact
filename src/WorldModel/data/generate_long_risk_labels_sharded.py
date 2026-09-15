"""Deterministic process-level sharding for W-step long-risk labels.

This module deliberately delegates candidate rollout semantics to
``generate_long_risk_labels.process_snapshot``.  It only partitions snapshot
files and merges the resulting lists, allowing one seed/load cell to use more
than one CPU process without changing the label definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Iterable, Mapping

import torch

from WorldModel.data.generate_long_risk_labels import process_snapshot


SCHEMA_VERSION = "long_risk_label_shard_v1"
MERGE_SCHEMA_VERSION = "long_risk_label_shard_merge_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"tick(\d+)", path.name)
    return (int(match.group(1)) if match else -1, path.name)


def snapshot_files(snapshot_dir: Path) -> list[Path]:
    files = sorted(snapshot_dir.glob("*.pkl"), key=_snapshot_sort_key)
    if not files:
        raise ValueError(f"no snapshot pickle files under {snapshot_dir}")
    return files


def partition_names(names: Iterable[str], num_shards: int) -> list[list[str]]:
    if int(num_shards) <= 0:
        raise ValueError("num_shards must be positive")
    result = [[] for _ in range(int(num_shards))]
    for index, name in enumerate(names):
        result[index % int(num_shards)].append(str(name))
    return result


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def generate_shard(
    *,
    snapshot_dir: Path,
    output: Path,
    horizon: int,
    terminal_window: int,
    shard_index: int,
    num_shards: int,
) -> dict:
    if horizon <= 0 or terminal_window <= 0:
        raise ValueError("horizon and terminal_window must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("require 0 <= shard_index < num_shards")
    files = snapshot_files(snapshot_dir)
    partitions = partition_names((path.name for path in files), num_shards)
    chosen_names = partitions[shard_index]
    if not chosen_names:
        raise ValueError(
            f"empty shard {shard_index}/{num_shards} for {snapshot_dir}"
        )
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_dir": snapshot_dir.as_posix(),
        "snapshot_names_sha256": _canonical_sha256(
            [path.name for path in files]
        ),
        "snapshot_count": len(files),
        "selected_snapshot_names": chosen_names,
        "selected_snapshot_names_sha256": _canonical_sha256(chosen_names),
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "long_risk_horizon": int(horizon),
        "terminal_window": int(terminal_window),
        "continuation_policy": "greedy",
    }
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if output.is_file() or meta_path.is_file():
        if not output.is_file() or not meta_path.is_file():
            raise FileExistsError(f"partial shard output: {output}")
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing.get("protocol") != protocol:
            raise FileExistsError(f"incompatible existing shard: {output}")
        if existing.get("output_sha256") != sha256_file(output):
            raise ValueError(f"existing shard hash mismatch: {output}")
        print(f"[resume] verified {output}")
        return existing

    labels: list[dict] = []
    started = time.time()
    for offset, name in enumerate(chosen_names, start=1):
        rows = process_snapshot(
            str(snapshot_dir / name),
            int(horizon),
            int(terminal_window),
            continuation_policy="greedy",
        )
        labels.extend(rows)
        print(
            f"[{offset}/{len(chosen_names)}] {name} candidates={len(rows)}",
            flush=True,
        )
    keys = [str(row.get("candidate_key", "")) for row in labels]
    if not labels or any(not key for key in keys):
        raise ValueError("long-risk shard produced empty or malformed labels")
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate candidate keys within long-risk shard")

    _atomic_torch_save(output, labels)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "label_count": len(labels),
        "candidate_key_sha256": _canonical_sha256(sorted(keys)),
        "elapsed_seconds": round(time.time() - started, 3),
        "output": output.as_posix(),
        "output_sha256": sha256_file(output),
    }
    _atomic_json(meta_path, payload)
    print(f"[complete] {output} labels={len(labels)}")
    return payload


def merge_shards(
    *,
    snapshot_dir: Path,
    inputs: list[Path],
    output: Path,
) -> dict:
    if len(inputs) < 2:
        raise ValueError("merge requires at least two shard inputs")
    all_snapshot_names = [path.name for path in snapshot_files(snapshot_dir)]
    expected_snapshot_hash = _canonical_sha256(all_snapshot_names)
    metas = []
    labels: list[dict] = []
    selected_names: list[str] = []
    for path in inputs:
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if not path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(path if not path.is_file() else meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("output_sha256") != sha256_file(path):
            raise ValueError(f"shard hash mismatch: {path}")
        protocol = meta.get("protocol") or {}
        if protocol.get("snapshot_names_sha256") != expected_snapshot_hash:
            raise ValueError(f"snapshot set changed for shard: {path}")
        metas.append(meta)
        selected_names.extend(protocol.get("selected_snapshot_names") or [])
        rows = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(rows, list):
            raise TypeError(f"expected label list: {path}")
        labels.extend(rows)

    num_shards = {int(meta["protocol"]["num_shards"]) for meta in metas}
    shard_indices = {int(meta["protocol"]["shard_index"]) for meta in metas}
    horizons = {
        int(meta["protocol"]["long_risk_horizon"]) for meta in metas
    }
    windows = {int(meta["protocol"]["terminal_window"]) for meta in metas}
    if len(num_shards) != 1 or next(iter(num_shards)) != len(inputs):
        raise ValueError("incomplete or inconsistent shard count")
    if shard_indices != set(range(len(inputs))):
        raise ValueError("shard indices are incomplete")
    if len(horizons) != 1 or len(windows) != 1:
        raise ValueError("long-risk shard contracts differ")
    if len(selected_names) != len(set(selected_names)):
        raise ValueError("snapshot overlap between shards")
    if sorted(selected_names, key=lambda name: _snapshot_sort_key(Path(name))) != (
        all_snapshot_names
    ):
        raise ValueError("shards do not exactly cover the snapshot directory")

    labels.sort(
        key=lambda row: (
            int(row.get("decision_tick", -1)),
            str(row.get("candidate_group_id", "")),
            str(row.get("candidate_key", "")),
        )
    )
    keys = [str(row.get("candidate_key", "")) for row in labels]
    if not labels or any(not key for key in keys):
        raise ValueError("merged long-risk labels are empty or malformed")
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate candidate keys across shards")
    horizon = next(iter(horizons))
    window = next(iter(windows))
    if any(
        int(row.get("long_risk_horizon", -1)) != horizon
        or int(row.get("long_risk_terminal_window", -1)) != window
        or row.get("continuation_policy") != "greedy"
        for row in labels
    ):
        raise ValueError("merged label rows violate the frozen contract")

    merge_protocol = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "snapshot_dir": snapshot_dir.as_posix(),
        "snapshot_names_sha256": expected_snapshot_hash,
        "snapshot_count": len(all_snapshot_names),
        "shards": [path.as_posix() for path in inputs],
        "shard_hashes": [sha256_file(path) for path in inputs],
        "long_risk_horizon": horizon,
        "terminal_window": window,
        "continuation_policy": "greedy",
    }
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if output.is_file() or meta_path.is_file():
        if not output.is_file() or not meta_path.is_file():
            raise FileExistsError(f"partial merged output: {output}")
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing.get("protocol") != merge_protocol:
            raise FileExistsError(f"incompatible merged output: {output}")
        if existing.get("output_sha256") != sha256_file(output):
            raise ValueError(f"merged output hash mismatch: {output}")
        print(f"[resume] verified {output}")
        return existing

    _atomic_torch_save(output, labels)
    payload = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "protocol": merge_protocol,
        "label_count": len(labels),
        "candidate_key_sha256": _canonical_sha256(keys),
        "output": output.as_posix(),
        "output_sha256": sha256_file(output),
    }
    _atomic_json(meta_path, payload)
    print(f"[complete] merged {output} labels={len(labels)}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "merge"), required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--W", type=int, default=200)
    parser.add_argument("--terminal-window", type=int, default=50)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--inputs", nargs="*", type=Path, default=[])
    args = parser.parse_args()

    if args.mode == "generate":
        if args.shard_index is None or args.num_shards is None:
            parser.error("generate requires --shard-index and --num-shards")
        if args.inputs:
            parser.error("generate does not accept --inputs")
        generate_shard(
            snapshot_dir=args.snapshot_dir,
            output=args.output,
            horizon=int(args.W),
            terminal_window=int(args.terminal_window),
            shard_index=int(args.shard_index),
            num_shards=int(args.num_shards),
        )
        return
    if args.shard_index is not None or args.num_shards is not None:
        parser.error("merge does not accept shard index arguments")
    merge_shards(
        snapshot_dir=args.snapshot_dir,
        inputs=list(args.inputs),
        output=args.output,
    )


if __name__ == "__main__":
    main()
