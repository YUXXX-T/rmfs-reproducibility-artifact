"""Freeze the independent behavior-aligned ``psi_pre`` protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from WorldModel.evaluation.phase_c_psi_pre_protocol import (
    BEHAVIOR_ROOT,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    MODEL_CHECKPOINT,
    OUTPUT_ROOT,
    canonical_sha256,
    expected_run_files,
    formal_protocol,
    relative_to_repo,
    sha256_file,
)


FROZEN_FILENAME = "phase_c_psi_pre_frozen_protocol.json"
INPUTS_FILENAME = "frozen_inputs.sha256"


def _write_immutable(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to change frozen file: {path}")
        print(f"[audit] unchanged: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(encoded, encoding="utf-8", newline="\n")
    partial.replace(path)
    print(f"[freeze] wrote {path}")


def _write_text_immutable(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"refusing to change frozen file: {path}")
        print(f"[audit] unchanged: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(text, encoding="utf-8", newline="\n")
    partial.replace(path)
    print(f"[freeze] wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-root", type=Path, default=BEHAVIOR_ROOT)
    parser.add_argument("--model", type=Path, default=MODEL_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root used only to make protocol paths portable",
    )
    args = parser.parse_args()

    for path in (args.behavior_root, args.model):
        if not path.exists():
            raise FileNotFoundError(path)

    protocol = formal_protocol(
        behavior_root=args.behavior_root,
        model_checkpoint=args.model,
    )
    protocol_sha = canonical_sha256(protocol)
    source_rows = expected_run_files(args.behavior_root)
    artifacts = {
        "model_checkpoint": {
            "path": relative_to_repo(args.model, args.repo_root),
            "sha256": sha256_file(args.model),
        },
        "behavior_runs": [
            {
                "run_id": row["run_id"],
                "load": row["load"],
                "seed": row["seed"],
                "data_path": relative_to_repo(row["data_path"], args.repo_root),
                "data_sha256": row["data_sha256"],
                "meta_path": relative_to_repo(row["meta_path"], args.repo_root),
                "meta_sha256": row["meta_sha256"],
            }
            for row in source_rows
        ],
    }
    bundle = {
        "schema_version": FROZEN_BUNDLE_SCHEMA_VERSION,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "artifacts": artifacts,
        "audit": {
            "fresh_seed_block": [531, 540],
            "seed_501_510_preserved": True,
            "checkpoint_retrained": False,
            "psi_pre_connected": False,
            "delta_psi_tested": False,
            "q_score_changed": False,
            "source_run_count": len(source_rows),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    bundle_path = args.output_root / FROZEN_FILENAME
    _write_immutable(bundle_path, bundle)

    rows = [
        f"{sha256_file(bundle_path)}  {bundle_path.as_posix()}",
        f"{artifacts['model_checkpoint']['sha256']}  {artifacts['model_checkpoint']['path']}",
    ]
    for row in artifacts["behavior_runs"]:
        rows.append(f"{row['data_sha256']}  {row['data_path']}")
        rows.append(f"{row['meta_sha256']}  {row['meta_path']}")
    inputs_text = "\n".join(rows) + "\n"
    _write_text_immutable(args.output_root / INPUTS_FILENAME, inputs_text)

    print(f"psi_pre protocol sha256 = {protocol_sha}")
    print(f"source runs = {len(source_rows)}")
    print(f"output root = {args.output_root}")


if __name__ == "__main__":
    main()
