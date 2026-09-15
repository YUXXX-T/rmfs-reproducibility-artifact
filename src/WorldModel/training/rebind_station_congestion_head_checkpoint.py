"""Rebind a frozen static-J station head to a head-only WM checkpoint.

The operation is allowed only when every World-Model tensor outside
``long_risk_head`` is bitwise identical.  Station-head weights and the scale
contract are copied unchanged; only the source-checkpoint binding metadata is
updated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import torch


SCHEMA_VERSION = "station_congestion_head_tensor_equivalent_rebind_v1"
HEAD_PREFIX = "long_risk_head."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _world_model_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("state_dict"), Mapping
    ):
        raise ValueError(f"invalid World-Model checkpoint: {path}")
    return payload


def rebind(
    *,
    source_head: Path,
    source_world_model: Path,
    target_world_model: Path,
    output_root: Path,
) -> dict:
    for path in (source_head, source_world_model, target_world_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_root.exists():
        raise FileExistsError(output_root)
    source_head_payload = torch.load(
        source_head, map_location="cpu", weights_only=False
    )
    if not isinstance(source_head_payload, dict):
        raise ValueError("station head checkpoint must contain a dictionary")
    source_hash = sha256_file(source_world_model)
    target_hash = sha256_file(target_world_model)
    if (
        str(source_head_payload.get("source_encoder_checkpoint_sha256") or "")
        != source_hash
    ):
        raise ValueError(
            "station head is not bound to the declared source World Model"
        )

    source_payload = _world_model_payload(source_world_model)
    target_payload = _world_model_payload(target_world_model)
    source_state = source_payload["state_dict"]
    target_state = target_payload["state_dict"]
    if set(source_state) != set(target_state):
        raise ValueError("source and target World-Model state keys differ")
    head_keys = sorted(
        key for key in source_state if key.startswith(HEAD_PREFIX)
    )
    non_head_keys = sorted(
        key for key in source_state if not key.startswith(HEAD_PREFIX)
    )
    if len(head_keys) != 6:
        raise ValueError(f"expected six LongRiskHead tensors, got {head_keys}")
    changed_non_head = [
        key
        for key in non_head_keys
        if not torch.equal(source_state[key], target_state[key])
    ]
    changed_head = [
        key
        for key in head_keys
        if not torch.equal(source_state[key], target_state[key])
    ]
    if changed_non_head:
        raise ValueError(
            "cannot rebind station head; non-long-risk tensors changed: "
            + ", ".join(changed_non_head[:10])
        )
    if not changed_head:
        raise ValueError("target did not update LongRiskHead")
    if source_payload.get("model_config") != target_payload.get("model_config"):
        raise ValueError("source and target model_config differ")
    if source_payload.get("action_schema") != target_payload.get("action_schema"):
        raise ValueError("source and target action_schema differ")

    audit = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "source_world_model": source_world_model.as_posix(),
        "source_world_model_sha256": source_hash,
        "target_world_model": target_world_model.as_posix(),
        "target_world_model_sha256": target_hash,
        "non_long_risk_tensor_count": len(non_head_keys),
        "changed_non_long_risk_tensors": changed_non_head,
        "long_risk_tensor_count": len(head_keys),
        "changed_long_risk_tensors": changed_head,
        "station_head_weights_changed": False,
        "scale_contract_changed": False,
        "justification": (
            "the static-J head consumes an encoder representation that is "
            "bitwise identical in source and target checkpoints"
        ),
    }
    rebound = copy.deepcopy(source_head_payload)
    rebound["source_encoder_checkpoint"] = target_world_model.as_posix()
    rebound["source_encoder_checkpoint_sha256"] = target_hash
    rebound["tensor_equivalent_rebind"] = {
        **audit,
        "source_station_head": source_head.as_posix(),
        "source_station_head_sha256": sha256_file(source_head),
    }

    staging = output_root.parent / f".{output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    output = staging / "best_station_congestion_head.pt"
    _atomic_torch_save(output, rebound)
    reloaded = torch.load(output, map_location="cpu", weights_only=False)
    if reloaded.get("source_encoder_checkpoint_sha256") != target_hash:
        raise RuntimeError("rebound station head lost its target binding")
    if set(reloaded.get("state_dict") or {}) != set(
        source_head_payload.get("state_dict") or {}
    ):
        raise RuntimeError("station head state keys changed during rebinding")
    if any(
        not torch.equal(
            reloaded["state_dict"][key], source_head_payload["state_dict"][key]
        )
        for key in source_head_payload.get("state_dict") or {}
    ):
        raise RuntimeError("station head weights changed during rebinding")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "audit": audit,
        "source_station_head": {
            "path": source_head.as_posix(),
            "sha256": sha256_file(source_head),
        },
        "output": output.name,
        "output_sha256": sha256_file(output),
    }
    _atomic_json(staging / "rebind_summary.json", summary)
    files = sorted(path for path in staging.iterdir() if path.is_file())
    (staging / "rebound_outputs.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    staging.rename(output_root)
    print(f"[complete] tensor-equivalent station-head rebind: {output_root}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-head", type=Path, required=True)
    parser.add_argument("--source-world-model", type=Path, required=True)
    parser.add_argument("--target-world-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rebind(
        source_head=args.source_head,
        source_world_model=args.source_world_model,
        target_world_model=args.target_world_model,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
