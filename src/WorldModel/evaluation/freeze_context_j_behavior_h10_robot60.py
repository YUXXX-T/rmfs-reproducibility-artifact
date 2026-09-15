"""Derive 60-robot configs and freeze the 611--630 context-J bundle."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from WorldModel.evaluation.context_j_behavior_h10_robot60_protocol import (
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOADS,
    MODEL_CHECKPOINT,
    OUTPUT_ROOT,
    SOURCE_LOAD_CONFIGS,
    SOURCE_ROBOT_COUNT,
    TARGET_ROBOT_COUNT,
    canonical_sha256,
    derived_config_path,
    formal_protocol,
    sha256_file,
)


FROZEN_FILENAME = "context_j_behavior_h10_robot60_frozen_protocol.json"
INPUTS_FILENAME = "frozen_inputs.sha256"
RUNNER_PATH = Path(__file__)
PROTOCOL_PATH = Path(
    "WorldModel/evaluation/context_j_behavior_h10_robot60_protocol.py"
)
COLLECTION_WRAPPER_PATH = Path(
    "WorldModel/evaluation/generate_data_fifo_v2.py"
)
GENERATOR_PATH = Path("WorldModel/generate_data.py")
COLLECTOR_PATH = Path("WorldModel/data/data_collector.py")
ROLLOUT_PATH = Path("WorldModel/data/counterfactual_rollout.py")
STATION_STATE_PATH = Path("WorldState/station_state.py")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _changed_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                result.append(child)
            else:
                result.extend(_changed_paths(left[key], right[key], child))
        return result
    if isinstance(left, list) and isinstance(right, list):
        result = []
        if len(left) != len(right):
            result.append(f"{prefix}.length")
        for index, (lvalue, rvalue) in enumerate(zip(left, right)):
            result.extend(
                _changed_paths(lvalue, rvalue, f"{prefix}[{index}]")
            )
        return result
    return [] if left == right else [prefix]


def _write_immutable_text(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"refusing to change frozen file: {path}")
        print(f"[audit] unchanged: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8", newline="\n")
    partial.replace(path)
    print(f"[freeze] wrote {path}")


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_immutable_text(
        path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": path.as_posix(), "sha256": sha256_file(path)}


def _derive_configs(output_root: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for load in LOADS:
        source_path = SOURCE_LOAD_CONFIGS[load]
        source = _read_json(source_path)
        if int(source.get("robots", {}).get("num_robots", -1)) != SOURCE_ROBOT_COUNT:
            raise ValueError(f"source config is not 48-robot: {source_path}")
        derived = copy.deepcopy(source)
        derived["robots"]["num_robots"] = TARGET_ROBOT_COUNT
        changed = _changed_paths(source, derived)
        if changed != ["robots.num_robots"]:
            raise AssertionError(
                f"derived {load} config changed unexpected fields: {changed}"
            )
        output_path = derived_config_path(output_root, load)
        _write_immutable_json(output_path, derived)
        entries[load] = {
            "source": _artifact(source_path),
            "derived": _artifact(output_path),
            "changed_paths": changed,
            "source_robot_count": SOURCE_ROBOT_COUNT,
            "target_robot_count": TARGET_ROBOT_COUNT,
        }
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--model", type=Path, default=MODEL_CHECKPOINT)
    args = parser.parse_args()

    output_root = args.output_root
    config_entries = _derive_configs(output_root)
    derived_configs = {
        load: Path(config_entries[load]["derived"]["path"]) for load in LOADS
    }
    protocol = formal_protocol(
        output_root=output_root,
        load_configs=derived_configs,
    )
    protocol_sha = canonical_sha256(protocol)
    code_paths = {
        "freeze_runner": RUNNER_PATH,
        "protocol": PROTOCOL_PATH,
        "collection_wrapper": COLLECTION_WRAPPER_PATH,
        "generator": GENERATOR_PATH,
        "collector": COLLECTOR_PATH,
        "counterfactual_rollout": ROLLOUT_PATH,
        "station_state": STATION_STATE_PATH,
    }
    artifacts = {
        "model_checkpoint": _artifact(args.model),
        # Keep the generic behavior-H10 validator's artifact contract: the
        # active load configs live directly under ``load_configs``.
        "load_configs": {
            load: config_entries[load]["derived"] for load in LOADS
        },
        "source_load_configs": {
            load: config_entries[load]["source"] for load in LOADS
        },
        "config_derivation": {
            load: {
                "changed_paths": config_entries[load]["changed_paths"],
                "source_robot_count": SOURCE_ROBOT_COUNT,
                "target_robot_count": TARGET_ROBOT_COUNT,
            }
            for load in LOADS
        },
        "code": {name: _artifact(path) for name, path in code_paths.items()},
    }
    bundle = {
        "schema_version": FROZEN_BUNDLE_SCHEMA_VERSION,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "artifacts": artifacts,
        "audit": {
            "fresh_seed_block": [611, 630],
            "training_split": [611, 622],
            "validation_split": [623, 626],
            "test_split": [627, 630],
            "pressure_seed_601_610_preserved": True,
            "source_robot_count": SOURCE_ROBOT_COUNT,
            "target_robot_count": TARGET_ROBOT_COUNT,
            "config_changed_only_robot_count": True,
            "checkpoint_retrained": False,
            "station_admission": "committed_capacity_fifo_v2",
            "behavior_policy_is_external_to_tested_model": True,
        },
    }

    bundle_path = output_root / FROZEN_FILENAME
    _write_immutable_json(bundle_path, bundle)

    rows = [
        f"{sha256_file(bundle_path)}  {bundle_path.as_posix()}",
        (
            f"{artifacts['model_checkpoint']['sha256']}  "
            f"{artifacts['model_checkpoint']['path']}"
        ),
    ]
    for load in LOADS:
        for kind in ("source", "derived"):
            artifact = config_entries[load][kind]
            rows.append(f"{artifact['sha256']}  {artifact['path']}")
    for artifact in artifacts["code"].values():
        rows.append(f"{artifact['sha256']}  {artifact['path']}")
    _write_immutable_text(output_root / INPUTS_FILENAME, "\n".join(rows) + "\n")

    print(f"context-J robot60 behavior H=10 protocol sha256 = {protocol_sha}")
    print("seeds =", protocol["inputs"]["seeds"])
    print("split_seeds =", protocol["inputs"]["split_seeds"])
    print("target_robot_count =", TARGET_ROBOT_COUNT)
    print("station_admission =", protocol["collection"]["station_admission"])
    print("output_root =", output_root)


if __name__ == "__main__":
    main()
