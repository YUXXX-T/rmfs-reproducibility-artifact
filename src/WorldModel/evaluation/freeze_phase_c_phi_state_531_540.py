"""Freeze the fresh 531--540 ``phi_state`` validation inputs."""

from __future__ import annotations

import json
from pathlib import Path

from WorldModel.evaluation.phase_c_phi_state_531_540_protocol import (
    FROZEN_BUNDLE_SCHEMA_VERSION,
    HEAD_CHECKPOINT,
    OUTPUT_ROOT,
    SCALE_CONTRACT,
    SOURCE_BUNDLE,
    SOURCE_FILES,
    canonical_sha256,
    protocol_payload,
    sha256_file,
)


BUNDLE_NAME = "phase_c_phi_state_frozen_protocol.json"
HASH_NAME = "frozen_inputs.sha256"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_exact(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"frozen file differs: {path}")
        print(f"[audit] unchanged {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"[freeze] wrote {path}")


def _artifact(path: Path, expected: str | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected is not None and actual != str(expected):
        raise ValueError(f"source frozen artifact changed: {path}")
    return {"path": path.as_posix(), "sha256": actual}


def main() -> None:
    if not SOURCE_BUNDLE.is_file():
        raise FileNotFoundError(SOURCE_BUNDLE)
    if not HEAD_CHECKPOINT.is_file():
        raise FileNotFoundError(HEAD_CHECKPOINT)
    if not SCALE_CONTRACT.is_file():
        raise FileNotFoundError(SCALE_CONTRACT)

    source_bundle = _read_json(SOURCE_BUNDLE)
    source_artifacts = source_bundle.get("artifacts") or {}
    artifacts: dict[str, dict] = {
        "source_frozen_bundle": _artifact(SOURCE_BUNDLE),
        "station_head_checkpoint": _artifact(HEAD_CHECKPOINT),
        "station_scale_contract": _artifact(SCALE_CONTRACT),
    }
    for key in ("candidate_checkpoint", "config_low", "config_mid", "config_high"):
        value = source_artifacts.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"source bundle lacks artifact {key!r}")
        artifacts[f"source_{key}"] = _artifact(
            Path(str(value.get("path", ""))), str(value.get("sha256", ""))
        )

    for raw_path in SOURCE_FILES:
        path = Path(raw_path)
        key = "source_" + path.stem
        suffix = 2
        while key in artifacts:
            key = f"source_{path.stem}_{suffix}"
            suffix += 1
        artifacts[key] = _artifact(path)

    protocol = protocol_payload(
        sha256_file(SOURCE_BUNDLE),
        sha256_file(HEAD_CHECKPOINT),
        sha256_file(SCALE_CONTRACT),
    )
    bundle = {
        "schema_version": FROZEN_BUNDLE_SCHEMA_VERSION,
        "protocol": protocol,
        "artifacts": artifacts,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    bundle_path = OUTPUT_ROOT / BUNDLE_NAME
    _write_exact(bundle_path, json.dumps(bundle, indent=2, ensure_ascii=False) + "\n")
    rows = [
        f"{entry['sha256']}  {entry['path']}"
        for _, entry in sorted(artifacts.items())
    ]
    rows.append(f"{sha256_file(bundle_path)}  {bundle_path.as_posix()}")
    _write_exact(OUTPUT_ROOT / HASH_NAME, "\n".join(rows) + "\n")
    print(f"protocol sha256 = {protocol['protocol_sha256']}")
    print(f"validation seeds = {list(protocol['development']['seeds'])}")
    print(f"head sha256 = {protocol['frozen_inputs']['head_checkpoint_sha256']}")
    print(f"scale sha256 = {protocol['frozen_inputs']['scale_contract_sha256']}")
    print(f"[complete] phi_state protocol frozen at {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
