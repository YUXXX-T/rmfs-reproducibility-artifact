"""Freeze the targeted Phase-C failure diagnostic without touching old bundles."""

from __future__ import annotations

import json
from pathlib import Path

from WorldModel.evaluation.phase_c_failure_diagnostic_protocol import (
    ARMS,
    CASES,
    DIAGNOSTIC_SOURCE_FILES,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    OUTPUT_ROOT,
    SOURCE_BUNDLE,
    SOURCE_ROOT,
    protocol_payload,
    sha256_file,
)


BUNDLE_NAME = "phase_c_failure_diagnostic_protocol.json"
HASH_NAME = "diagnostic_inputs.sha256"


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


def _artifact(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": path.as_posix(), "sha256": sha256_file(path)}


def main() -> None:
    if not SOURCE_BUNDLE.is_file():
        raise FileNotFoundError(SOURCE_BUNDLE)
    source_bundle_hash = sha256_file(SOURCE_BUNDLE)

    artifacts = {"source_frozen_bundle": _artifact(SOURCE_BUNDLE)}
    for case in CASES:
        load = str(case["load"])
        seed = int(case["seed"])
        manifest = SOURCE_ROOT / "order_manifests" / (
            f"orders_{load}_seed{seed}.json"
        )
        artifacts[f"manifest_{load}_seed{seed}"] = _artifact(manifest)
        for arm in ARMS:
            reference = SOURCE_ROOT / "per_arm" / arm / (
                f"{load}_seed{seed}.json"
            )
            artifacts[f"reference_{arm}_{load}_seed{seed}"] = _artifact(
                reference
            )

    for source in DIAGNOSTIC_SOURCE_FILES:
        key = "source_" + Path(source).stem
        if key in artifacts:
            raise RuntimeError(f"duplicate frozen artifact key: {key}")
        artifacts[key] = _artifact(Path(source))

    protocol = protocol_payload(source_bundle_hash)
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

    hash_rows = [
        f"{entry['sha256']}  {entry['path']}"
        for entry in artifacts.values()
    ]
    hash_rows.append(f"{sha256_file(bundle_path)}  {bundle_path.as_posix()}")
    _write_exact(OUTPUT_ROOT / HASH_NAME, "\n".join(hash_rows) + "\n")

    print(f"protocol sha256 = {protocol['protocol_sha256']}")
    print(f"cases = {[(c['load'], c['seed']) for c in CASES]}")
    print(f"[complete] diagnostic frozen at {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
