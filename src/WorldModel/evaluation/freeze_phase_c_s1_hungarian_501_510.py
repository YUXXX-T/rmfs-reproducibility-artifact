"""Freeze or audit the isolated Phase-C/S1 seed replication protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path

from WorldModel.evaluation.phase_c_s1_hungarian_replication_501_510 import (
    OUTPUT_ROOT,
    SEEDS,
    install,
)


protocol = install()

BUNDLE_NAME = "phase_c_s1_hungarian_frozen_protocol.json"
HASH_NAME = "frozen_inputs.sha256"

INPUTS = {
    "candidate_checkpoint": protocol.CANDIDATE_CHECKPOINT,
    "config_low": protocol.LOAD_CONFIGS["low"],
    "config_mid": protocol.LOAD_CONFIGS["mid"],
    "config_high": protocol.LOAD_CONFIGS["high"],
    "shared_protocol": (
        "WorldModel/evaluation/phase_c_s1_hungarian_protocol.py"
    ),
    "replication_protocol": (
        "WorldModel/evaluation/phase_c_s1_hungarian_replication_501_510.py"
    ),
    "shared_runner": (
        "WorldModel/evaluation/run_phase_c_s1_hungarian_arm.py"
    ),
    "replication_runner": (
        "WorldModel/evaluation/run_phase_c_s1_hungarian_501_510_arm.py"
    ),
    "shared_validator": (
        "WorldModel/evaluation/validate_phase_c_s1_hungarian_491_500.py"
    ),
    "replication_validator": (
        "WorldModel/evaluation/validate_phase_c_s1_hungarian_501_510.py"
    ),
    "freeze_source": (
        "WorldModel/evaluation/freeze_phase_c_s1_hungarian_501_510.py"
    ),
    "collector_slurm": (
        "WorldModel/evaluation/collect_phase_c_s1_hungarian_501_510_cpu.slurm"
    ),
    "evaluation_source": "WorldModel/evaluation/evaluate_online_v6.py",
    "world_model_assigner": (
        "Policies/TaskAssigner/WorldModelTaskAssigner/"
        "world_model_task_assigner.py"
    ),
    "hungarian_assigner": (
        "Policies/TaskAssigner/HungarianTaskAssigner/"
        "hungarian_task_assigner.py"
    ),
    "greedy_assigner": (
        "Policies/TaskAssigner/GreedyTaskAssigner/greedy_task_assigner.py"
    ),
    "context_assignment": "Policies/TaskAssigner/context_assignment.py",
    "simulation_engine": "Engine/simulation_engine.py",
    "world_model_core": "WorldModel/core/model.py",
    "cost_definition": "WorldModel/core/costs.py",
}


def _write_exact(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"frozen file differs: {path}")
        print(f"[audit] unchanged {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[freeze] wrote {path}")


def main() -> None:
    root = Path(os.environ.get("PHASEC_S1_CERT", OUTPUT_ROOT))
    bundle_path = root / BUNDLE_NAME
    hash_path = root / HASH_NAME
    result_dirs = (root / "order_manifests", root / "per_arm")
    if not bundle_path.is_file() and any(
        path.exists() and any(path.rglob("*")) for path in result_dirs
    ):
        raise RuntimeError(
            "result artifacts exist without a frozen protocol; refusing to mix runs"
        )

    artifacts = {}
    for key, raw_path in INPUTS.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts[key] = {
            "path": path.as_posix(),
            "sha256": protocol.sha256_file(path),
        }

    formal = protocol.formal_protocol()
    protocol_sha256 = protocol.canonical_sha256(formal)
    bundle = {
        "schema_version": protocol.FROZEN_BUNDLE_SCHEMA_VERSION,
        "protocol_sha256": protocol_sha256,
        "protocol": formal,
        "artifacts": artifacts,
    }
    bundle_text = json.dumps(bundle, indent=2, ensure_ascii=False) + "\n"
    hashes_text = "".join(
        f"{value['sha256']}  {value['path']}\n"
        for _, value in sorted(artifacts.items())
    )
    _write_exact(bundle_path, bundle_text)
    _write_exact(hash_path, hashes_text)
    print(f"protocol sha256 = {protocol_sha256}")
    print(f"formal replication seeds = {list(SEEDS)}")
    print(f"loads = {list(protocol.LOADS)}, ticks = {protocol.TICKS}")
    print("seed change only = true; manifest source remains Greedy")
    print(f"[complete] frozen at {root}")


if __name__ == "__main__":
    main()

