"""Freeze inputs and source hashes for the 541--550 phi-context ablation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from WorldModel.evaluation.phase_c_phi_context_ablation_protocol import (
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOAD_CONFIGS,
    MODEL_CHECKPOINT,
    OUTPUT_ROOT,
    PHI_HEAD_CHECKPOINT,
    PHI_SCALE_CONTRACT,
    SOURCE_FILES,
    SOURCE_POLICY_BUNDLE,
    protocol_payload,
    sha256_file,
)


BUNDLE_NAME = "phase_c_phi_context_frozen_protocol.json"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite changed file: {path}")
        return
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


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def freeze(output_root: Path = OUTPUT_ROOT) -> Path:
    artifacts: dict[str, Mapping[str, Any]] = {
        "source_policy_bundle": _artifact(SOURCE_POLICY_BUNDLE),
        "model_checkpoint": _artifact(MODEL_CHECKPOINT),
        "phi_head_checkpoint": _artifact(PHI_HEAD_CHECKPOINT),
        "phi_scale_contract": _artifact(PHI_SCALE_CONTRACT),
    }
    for load, path in LOAD_CONFIGS.items():
        artifacts[f"config_{load}"] = _artifact(path)
    for path_value in SOURCE_FILES:
        path = Path(path_value)
        artifacts[f"source:{path.as_posix()}"] = _artifact(path)

    hashes = {
        key: str(value["sha256"])
        for key, value in artifacts.items()
        if not key.startswith("source:")
    }
    protocol = protocol_payload(hashes)
    bundle = {
        "schema_version": FROZEN_BUNDLE_SCHEMA_VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol": protocol,
        "artifacts": artifacts,
    }
    bundle_path = output_root / BUNDLE_NAME
    _atomic_write(
        bundle_path,
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
    )

    manifest_rows = []
    for key, value in sorted(artifacts.items()):
        # Keep this a standard ``sha256sum -c`` manifest.  The JSON bundle
        # carries the human-readable artifact keys; comments here would be
        # interpreted as part of a filename by GNU coreutils.
        manifest_rows.append(f"{value['sha256']}  {value['path']}")
    manifest_rows.append(
        f"{sha256_file(bundle_path)}  {bundle_path.as_posix()}"
    )
    _atomic_write(
        output_root / "frozen_inputs.sha256",
        "\n".join(manifest_rows) + "\n",
    )
    return bundle_path


def main() -> None:
    path = freeze()
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    print(f"[freeze] wrote {path}")
    print(f"protocol sha256 = {protocol['protocol_sha256']}")
    print(f"seeds = {protocol['development']['seeds']}")
    print(f"ticks = {protocol['development']['ticks']}")


if __name__ == "__main__":
    main()
