"""Freeze the isolated 571--590 multi-context behavior-label bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from WorldModel.evaluation.context_j_behavior_h10_protocol import (
    BASE_ROOT,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOAD_CONFIGS,
    MODEL_CHECKPOINT,
    OUTPUT_ROOT,
    canonical_sha256,
    formal_protocol,
    sha256_file,
)


FROZEN_FILENAME = "context_j_behavior_h10_frozen_protocol.json"
INPUTS_FILENAME = "frozen_inputs.sha256"


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": path.as_posix(), "sha256": sha256_file(path)}


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to change frozen file: {path}")
        print(f"[audit] unchanged: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(encoded, encoding="utf-8", newline="\n")
    partial.replace(path)
    print(f"[freeze] wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--model", type=Path, default=MODEL_CHECKPOINT)
    parser.add_argument("--low-config", type=Path, default=LOAD_CONFIGS["low"])
    parser.add_argument("--mid-config", type=Path, default=LOAD_CONFIGS["mid"])
    parser.add_argument("--high-config", type=Path, default=LOAD_CONFIGS["high"])
    args = parser.parse_args()

    protocol = formal_protocol()
    protocol_sha = canonical_sha256(protocol)
    artifacts = {
        "model_checkpoint": _artifact(args.model),
        "load_configs": {
            "low": _artifact(args.low_config),
            "mid": _artifact(args.mid_config),
            "high": _artifact(args.high_config),
        },
    }
    bundle = {
        "schema_version": FROZEN_BUNDLE_SCHEMA_VERSION,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "artifacts": artifacts,
        "audit": {
            "fresh_seed_block": [571, 590],
            "training_split": [571, 582],
            "validation_split": [583, 586],
            "test_split": [587, 590],
            "seed_501_510_preserved": True,
            "seed_551_570_preserved": True,
            "checkpoint_retrained": False,
            "psi_pre_connected": False,
            "q_score_changed": False,
            "e_demand_schema_changed": False,
            "backlog_overridden": False,
            "behavior_policy_is_external_to_tested_model": True,
        },
    }

    output_root = args.output_root
    bundle_path = output_root / FROZEN_FILENAME
    _write_immutable(bundle_path, bundle)

    rows = [
        f"{sha256_file(bundle_path)}  {bundle_path.as_posix()}",
        f"{artifacts['model_checkpoint']['sha256']}  "
        f"{artifacts['model_checkpoint']['path']}",
    ]
    for load in ("low", "mid", "high"):
        artifact = artifacts["load_configs"][load]
        rows.append(f"{artifact['sha256']}  {artifact['path']}")
    inputs_path = output_root / INPUTS_FILENAME
    inputs_text = "\n".join(rows) + "\n"
    if inputs_path.exists():
        if inputs_path.read_text(encoding="utf-8") != inputs_text:
            raise RuntimeError(f"refusing to change frozen file: {inputs_path}")
        print(f"[audit] unchanged: {inputs_path}")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        inputs_path.write_text(inputs_text, encoding="utf-8", newline="\n")
        print(f"[freeze] wrote {inputs_path}")

    print(f"context-J behavior H=10 protocol sha256 = {protocol_sha}")
    print(f"seeds = {protocol['inputs']['seeds']}")
    print(f"loads = {list(protocol['inputs']['loads'])}")
    print(f"sample_interval = {protocol['collection']['sample_interval']}")
    print(f"max_groups_per_tick = {protocol['collection']['max_groups_per_tick']}")
    print(f"output_root = {output_root}")


if __name__ == "__main__":
    main()
