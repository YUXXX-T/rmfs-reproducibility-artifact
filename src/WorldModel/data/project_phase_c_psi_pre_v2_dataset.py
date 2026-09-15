"""Project corrected full psi_pre shards onto the frozen v2 channel contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from WorldModel.data.build_phase_c_psi_pre_dataset import _atomic_torch_save
from WorldModel.evaluation.phase_c_psi_pre_protocol import (
    DATASET_SCHEMA_VERSION as V1_DATASET_SCHEMA_VERSION,
    TARGET_CHANNELS as V1_TARGET_CHANNELS,
)
from WorldModel.evaluation.phase_c_psi_pre_v2_protocol import (
    DATASET_SCHEMA_VERSION,
    PRIMARY_CHANNELS,
    SERVICE_CHANNELS,
    TARGET_CHANNELS,
    TARGET_SOURCE_INDICES,
    TRAFFIC_DIAGNOSTIC_CHANNELS,
    TRAFFIC_PRIMARY_CHANNELS,
    fold_seed_splits,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
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


def project_dataset(*, source: Path, output: Path) -> dict[str, Any]:
    dataset = torch.load(source, map_location="cpu", weights_only=False)
    if dataset.get("schema_version") != V1_DATASET_SCHEMA_VERSION:
        raise ValueError(f"expected corrected full v1 dataset, got {source}")
    if tuple(dataset.get("target_channels", ())) != V1_TARGET_CHANNELS:
        raise ValueError("full dataset target channel order mismatch")
    if not bool(dataset.get("baseline_node_bce_sigmoid", False)):
        raise ValueError(
            "refusing to project an uncorrected decoder baseline; rerun materialisation "
            "with --node-bce-sigmoid"
        )
    if tuple(V1_TARGET_CHANNELS[index] for index in TARGET_SOURCE_INDICES) != TARGET_CHANNELS:
        raise AssertionError("v2 source index contract is inconsistent")

    tensors = dict(dataset["tensors"])
    for key in ("targets", "baseline_state", "baseline_h10", "baseline_h1"):
        value = tensors[key]
        tensors[key] = value[:, list(TARGET_SOURCE_INDICES)].contiguous()

    projected = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "development_only": True,
        "protocol_sha256": dataset["protocol_sha256"],
        "source_full_dataset": source.as_posix(),
        "source_full_dataset_sha256": sha256_file(source),
        "source_checkpoint": dataset.get("source_checkpoint"),
        "source_checkpoint_sha256": dataset.get("source_checkpoint_sha256"),
        "horizon": int(dataset["horizon"]),
        "region_hops": int(dataset["region_hops"]),
        "target_channels": list(TARGET_CHANNELS),
        "service_channels": list(SERVICE_CHANNELS),
        "traffic_primary_channels": list(TRAFFIC_PRIMARY_CHANNELS),
        "traffic_diagnostic_channels": list(TRAFFIC_DIAGNOSTIC_CHANNELS),
        "primary_channels": list(PRIMARY_CHANNELS),
        "source_target_indices": list(TARGET_SOURCE_INDICES),
        "baseline_node_bce_sigmoid": True,
        "baseline_contract": "BCE-logit channels sigmoid before region mean/max",
        "representation": dataset.get("representation", {}),
        "latent_dim": int(dataset["latent_dim"]),
        "row_count": int(tensors["targets"].size(0)),
        "run_names": list(dataset["run_names"]),
        "run_keys": list(dataset["run_keys"]),
        "split_seeds": fold_seed_splits(),
        "folds": dataset["folds"],
        "weight_contract": dataset.get("weight_contract"),
        "target_scale_contract": dataset.get("target_scale_contract"),
        "baseline_names": list(dataset.get("baseline_names", [])),
        "tensors": tensors,
    }
    _atomic_torch_save(output, projected)
    summary = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset": output.name,
        "dataset_sha256": sha256_file(output),
        "source_full_dataset": source.as_posix(),
        "source_full_dataset_sha256": projected["source_full_dataset_sha256"],
        "row_count": projected["row_count"],
        "latent_dim": projected["latent_dim"],
        "target_channels": list(TARGET_CHANNELS),
        "primary_channels": list(PRIMARY_CHANNELS),
        "diagnostic_channels": list(TRAFFIC_DIAGNOSTIC_CHANNELS),
        "baseline_node_bce_sigmoid": True,
        "online_connection_allowed": False,
    }
    summary_path = output.parent / "dataset_v2_summary.json"
    _atomic_json(summary_path, summary)
    manifest = "\n".join([
        f"{sha256_file(output)}  {output.name}",
        f"{sha256_file(summary_path)}  {summary_path.name}",
        f"{projected['source_full_dataset_sha256']}  {source.as_posix()}",
    ]) + "\n"
    (output.parent / "dataset_v2_outputs.sha256").write_text(
        manifest, encoding="utf-8", newline="\n"
    )
    print(f"[project] rows={summary['row_count']} channels={summary['target_channels']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_dataset(source=args.source, output=args.output)


if __name__ == "__main__":
    main()
