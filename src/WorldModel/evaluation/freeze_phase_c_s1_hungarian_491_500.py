"""Freeze or audit the Phase-C/S1/Hungarian seeds 491--500 protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path

from WorldModel.evaluation.phase_c_s1_hungarian_protocol import (
    CANDIDATE_CHECKPOINT,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOAD_CONFIGS,
    OUTPUT_ROOT,
    canonical_sha256,
    formal_protocol,
    sha256_file,
)


BUNDLE_NAME = "phase_c_s1_hungarian_frozen_protocol.json"
HASH_NAME = "frozen_inputs.sha256"

INPUTS = {
    "candidate_checkpoint": CANDIDATE_CHECKPOINT,
    "config_low": LOAD_CONFIGS["low"],
    "config_mid": LOAD_CONFIGS["mid"],
    "config_high": LOAD_CONFIGS["high"],
    "protocol_source": (
        "WorldModel/evaluation/phase_c_s1_hungarian_protocol.py"
    ),
    "runner_source": (
        "WorldModel/evaluation/run_phase_c_s1_hungarian_arm.py"
    ),
    "validator_source": (
        "WorldModel/evaluation/validate_phase_c_s1_hungarian_491_500.py"
    ),
    "manifest_slurm": (
        "WorldModel/evaluation/run_phase_c_s1_hungarian_manifest_cpu.slurm"
    ),
    "hungarian_slurm": (
        "WorldModel/evaluation/run_phase_c_s1_hungarian_baseline_cpu.slurm"
    ),
    "model_slurm": (
        "WorldModel/evaluation/run_phase_c_s1_hungarian_models_cpu.slurm"
    ),
    "single_node_collector": (
        "WorldModel/evaluation/collect_phase_c_s1_hungarian_491_500_cpu.slurm"
    ),
    "smoke_source": "WorldModel/evaluation/smoke_phase_c_s1_hungarian.sh",
    "protocol_test": (
        "WorldModel/tests/test_phase_c_s1_hungarian_protocol.py"
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
        existing = path.read_text(encoding="utf-8")
        if existing != text:
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
            "sha256": sha256_file(path),
        }

    protocol = formal_protocol()
    protocol_sha256 = canonical_sha256(protocol)
    bundle = {
        "schema_version": FROZEN_BUNDLE_SCHEMA_VERSION,
        "protocol_sha256": protocol_sha256,
        "protocol": protocol,
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
    print("formal seeds = [491, 492, 493, 494, 495, 496, 497, 498, 499, 500]")
    print("loads = [low, mid, high], ticks = 1500")
    print(f"[complete] frozen at {root}")


if __name__ == "__main__":
    main()
