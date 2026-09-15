"""Freeze the isolated 511--520 station congestion correlation protocol."""

from __future__ import annotations

import json
from pathlib import Path

from WorldModel.evaluation.phase_c_station_congestion_correlation_protocol import (
    FROZEN_BUNDLE_SCHEMA_VERSION,
    OUTPUT_ROOT,
    SEEDS,
    SOURCE_BUNDLE,
    SOURCE_FILES,
    protocol_payload,
    sha256_file,
)


BUNDLE_NAME = "phase_c_station_congestion_frozen_protocol.json"
HASH_NAME = "frozen_inputs.sha256"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
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
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
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
    source_bundle = _read_json(SOURCE_BUNDLE)
    source_artifacts = source_bundle.get("artifacts") or {}

    artifacts = {
        "source_frozen_bundle": _artifact(SOURCE_BUNDLE),
    }
    for key in ("candidate_checkpoint", "config_low", "config_mid", "config_high"):
        value = source_artifacts.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"source bundle lacks artifact {key!r}")
        artifacts[f"source_{key}"] = _artifact(
            Path(str(value.get("path", ""))),
            str(value.get("sha256", "")),
        )

    for raw_path in SOURCE_FILES:
        path = Path(raw_path)
        key = "source_" + path.stem
        suffix = 2
        while key in artifacts:
            key = f"source_{path.stem}_{suffix}"
            suffix += 1
        artifacts[key] = _artifact(path)

    protocol = protocol_payload(sha256_file(SOURCE_BUNDLE))
    bundle = {
        "schema_version": FROZEN_BUNDLE_SCHEMA_VERSION,
        "protocol": protocol,
        "artifacts": artifacts,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    bundle_path = OUTPUT_ROOT / BUNDLE_NAME
    _write_exact(
        bundle_path,
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
    )
    rows = [
        f"{entry['sha256']}  {entry['path']}"
        for _, entry in sorted(artifacts.items())
    ]
    rows.append(f"{sha256_file(bundle_path)}  {bundle_path.as_posix()}")
    _write_exact(OUTPUT_ROOT / HASH_NAME, "\n".join(rows) + "\n")
    print(f"protocol sha256 = {protocol['protocol_sha256']}")
    print(f"development seeds = {list(SEEDS)}")
    print(f"[complete] station congestion correlation frozen at {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()

