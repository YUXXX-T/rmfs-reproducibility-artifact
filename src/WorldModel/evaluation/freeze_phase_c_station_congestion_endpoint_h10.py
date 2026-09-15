"""Freeze the fresh-seed H=10 station congestion endpoint audit."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from WorldModel.core.station_congestion_head import verify_scale_contract
from WorldModel.evaluation.phase_c_station_congestion_endpoint_protocol import (
    FROZEN_BUNDLE_SCHEMA_VERSION,
    HEAD_CHECKPOINT,
    LOAD_CONFIGS,
    MODEL_CHECKPOINT,
    OUTPUT_ROOT,
    SCALE_CONTRACT,
    SEEDS,
    canonical_sha256,
    formal_protocol,
    sha256_file,
)


FROZEN_FILENAME = "station_congestion_endpoint_h10_frozen_protocol.json"

CODE_INPUTS = (
    Path("Engine/simulation_engine.py"),
    Path("Env/rmfs_env.py"),
    Path("Policies/TaskAssigner/WorldModelTaskAssigner/world_model_task_assigner.py"),
    Path("Policies/TaskAssigner/context_assignment.py"),
    Path("WorldModel/core/model.py"),
    Path("WorldModel/core/station_congestion_head.py"),
    Path("WorldModel/data/build_station_congestion_head_dataset.py"),
    Path("WorldModel/data/candidate_generator.py"),
    Path("WorldModel/data/counterfactual_rollout.py"),
    Path("WorldModel/evaluation/decision_snapshot_probe.py"),
    Path("WorldModel/evaluation/evaluate_online_v6.py"),
    Path("WorldModel/evaluation/station_congestion_endpoint.py"),
    Path("WorldModel/evaluation/phase_c_station_congestion_endpoint_protocol.py"),
    Path("WorldModel/evaluation/freeze_phase_c_station_congestion_endpoint_h10.py"),
    Path("WorldModel/evaluation/collect_phase_c_station_congestion_endpoint.py"),
    Path("WorldModel/evaluation/replay_phase_c_station_congestion_endpoint.py"),
    Path("WorldModel/evaluation/analyze_phase_c_station_congestion_endpoint.py"),
    Path("WorldModel/evaluation/run_phase_c_station_congestion_endpoint_h10_cpu.slurm"),
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite changed freeze: {path}")
        return
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


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _verify_head_contract() -> dict[str, Any]:
    payload = torch.load(
        HEAD_CHECKPOINT, map_location="cpu", weights_only=False
    )
    scale = json.loads(SCALE_CONTRACT.read_text(encoding="utf-8"))
    verify_scale_contract(scale)
    checks = {
        "source_encoder_matches_model": (
            payload.get("source_encoder_checkpoint_sha256")
            == sha256_file(MODEL_CHECKPOINT)
        ),
        "scale_hash_matches_external_contract": (
            payload.get("scale_contract_sha256")
            == scale.get("contract_sha256")
        ),
        "scale_payload_matches_external_contract": (
            payload.get("scale_contract") == scale
        ),
        "region_representation": (
            (payload.get("representation") or {}).get("name")
            == "station_region_mean_max"
        ),
        "encoder_latent_dim": int(payload.get("encoder_latent_dim", -1)) == 64,
        "head_latent_dim": int(payload.get("latent_dim", -1)) == 192,
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"station head contract audit failed: {failed}")
    return {
        "passed": True,
        "checks": checks,
        "head_checkpoint_sha256": sha256_file(HEAD_CHECKPOINT),
        "model_checkpoint_sha256": sha256_file(MODEL_CHECKPOINT),
        "scale_contract_sha256": scale["contract_sha256"],
    }


def freeze(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if set(SEEDS).intersection(range(501, 511)):
        raise ValueError("endpoint audit must not use locked seeds 501--510")
    protocol = formal_protocol()
    protocol_sha = canonical_sha256(protocol)
    paths = [
        MODEL_CHECKPOINT,
        HEAD_CHECKPOINT,
        SCALE_CONTRACT,
        *LOAD_CONFIGS.values(),
        *CODE_INPUTS,
    ]
    artifacts = {
        path.as_posix(): _artifact(path)
        for path in paths
    }
    bundle = {
        "schema_version": FROZEN_BUNDLE_SCHEMA_VERSION,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "artifacts": artifacts,
        "head_contract_audit": _verify_head_contract(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    frozen_path = output_root / FROZEN_FILENAME
    _atomic_json(frozen_path, bundle)
    manifest_path = output_root / "frozen_inputs.sha256"
    manifest_text = "".join(
        f"{artifacts[path.as_posix()]['sha256']}  {path.as_posix()}\n"
        for path in paths
    )
    if manifest_path.is_file():
        if manifest_path.read_text(encoding="utf-8") != manifest_text:
            raise FileExistsError(
                f"refusing to overwrite changed manifest: {manifest_path}"
            )
    else:
        manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")
    print(f"[freeze] protocol sha256={protocol_sha}")
    print(f"[freeze] wrote {frozen_path}")
    print(f"[freeze] wrote {manifest_path}")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    freeze(args.output_root)


if __name__ == "__main__":
    main()
