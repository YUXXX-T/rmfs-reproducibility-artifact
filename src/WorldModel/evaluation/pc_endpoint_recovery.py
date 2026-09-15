"""Prepare a read-only 26-run exploratory recovery from underfilled H=10 data.

The original frozen collection root is never modified.  Completed collection
directories and non-zero post-run staging directories are exposed through
symlinks in a separate recovery root, with an explicit integrity-only audit.
Zero-snapshot runs remain excluded and are listed in the recovery manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from WorldModel.evaluation.collect_phase_c_station_congestion_endpoint import (
    _read_json,
    _verify_hashes,
)
from WorldModel.evaluation.decision_snapshot_probe import (
    PHASE_C_SNAPINDEX_SCHEMA_VERSION,
)
from WorldModel.evaluation.freeze_phase_c_station_congestion_endpoint_h10 import (
    FROZEN_FILENAME,
)
from WorldModel.evaluation.phase_c_station_congestion_endpoint_protocol import (
    COLLECTION_SCHEMA_VERSION,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOADS,
    MIN_SNAPSHOTS_PER_RUN,
    OUTPUT_ROOT as SOURCE_ROOT,
    SCHEMA_VERSION,
    SEEDS,
    canonical_sha256,
    sha256_file,
)


RECOVERY_SCHEMA_VERSION = (
    "station_congestion_endpoint_exploratory_recovery_v1"
)
DEFAULT_RECOVERY_ROOT = SOURCE_ROOT.parent / (
    "pc_endpoint_h10_recovery_521_530_v1"
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite changed recovery: {path}")
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


def _parse_run_id(run_id: str) -> tuple[str, int]:
    prefix = "phasec_s1_"
    if not run_id.startswith(prefix) or "_seed" not in run_id:
        raise ValueError(f"unexpected endpoint run id: {run_id}")
    load, seed_text = run_id[len(prefix):].rsplit("_seed", 1)
    if load not in LOADS:
        raise ValueError(f"unexpected endpoint load in {run_id}")
    seed = int(seed_text)
    if seed not in SEEDS:
        raise ValueError(f"unexpected endpoint seed in {run_id}")
    return load, seed


def _load_frozen_bundle(source_root: Path) -> tuple[Path, dict[str, Any]]:
    path = source_root / FROZEN_FILENAME
    bundle = _read_json(path)
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError("wrong source endpoint frozen bundle schema")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("source endpoint bundle lacks protocol")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong source endpoint protocol schema")
    if canonical_sha256(protocol) != bundle.get("protocol_sha256"):
        raise ValueError("source endpoint protocol hash mismatch")
    return path, bundle


def _validate_snapshot_source(
    source_dir: Path,
    *,
    expected_run_id: str,
) -> dict[str, Any]:
    snapshot_dir = source_dir / "snapshots"
    indexes = sorted(snapshot_dir.glob("snapindex_*.json"))
    if len(indexes) != 1:
        raise ValueError(
            f"expected one snapshot index in {source_dir}, got {len(indexes)}"
        )
    index_path = indexes[0]
    index = _read_json(index_path)
    if index.get("schema_version") != PHASE_C_SNAPINDEX_SCHEMA_VERSION:
        raise ValueError(f"wrong snapshot index schema: {index_path}")
    if str(index.get("run_id")) != expected_run_id:
        raise ValueError(f"snapshot run id mismatch: {index_path}")
    decisions = list(index.get("decisions") or [])
    snapshot_count = int(index.get("n_snapshots", -1))
    if snapshot_count != len(decisions):
        raise ValueError(f"snapshot count/index mismatch: {index_path}")
    files = []
    for row in decisions:
        path = snapshot_dir / str(row["file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path)
    order_manifest = source_dir / "order_manifest.json"
    if not order_manifest.is_file():
        raise FileNotFoundError(order_manifest)
    return {
        "snapshot_dir": snapshot_dir,
        "snapshot_index": index_path,
        "snapshot_index_payload": index,
        "snapshot_files": files,
        "snapshot_count": snapshot_count,
        "order_manifest": order_manifest,
    }


def _discover_finalized(
    source_root: Path,
    *,
    protocol_sha256: str,
) -> dict[str, dict[str, Any]]:
    result = {}
    collections = source_root / "collections"
    for summary_path in sorted(collections.glob(
        "phasec_s1_*/collection_summary.json"
    )):
        source_dir = summary_path.parent
        summary = _read_json(summary_path)
        run_id = source_dir.name
        load, seed = _parse_run_id(run_id)
        manifest = source_dir / "run_outputs.sha256"
        checks = {
            "schema": summary.get("schema_version") == COLLECTION_SCHEMA_VERSION,
            "protocol": summary.get("protocol_sha256") == protocol_sha256,
            "load": summary.get("load") == load,
            "seed": int(summary.get("seed", -1)) == seed,
            "audit": bool((summary.get("audit") or {}).get("passed")),
            "manifest_hash": (
                manifest.is_file()
                and sha256_file(manifest) == summary.get("run_outputs_sha256")
            ),
            "outputs": _verify_hashes(source_dir, manifest),
        }
        if not all(checks.values()):
            failed = [key for key, passed in checks.items() if not passed]
            raise ValueError(f"invalid finalized source {source_dir}: {failed}")
        source = _validate_snapshot_source(
            source_dir, expected_run_id=run_id
        )
        source.update({
            "run_id": run_id,
            "load": load,
            "seed": seed,
            "source_dir": source_dir,
            "source_kind": "finalized_collection",
            "source_summary": summary_path,
        })
        result[run_id] = source
    return result


def _discover_staging(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    collections = source_root / "collections"
    for source_dir in sorted(collections.glob(".phasec_s1_*.tmp.*")):
        if not source_dir.is_dir():
            continue
        name = source_dir.name[1:]
        run_id = name.split(".tmp.", 1)[0]
        load, seed = _parse_run_id(run_id)
        source = _validate_snapshot_source(
            source_dir, expected_run_id=run_id
        )
        source.update({
            "run_id": run_id,
            "load": load,
            "seed": seed,
            "source_dir": source_dir,
            "source_kind": "postrun_underfilled_staging",
            "source_summary": None,
        })
        result.setdefault(run_id, []).append(source)
    return result


def _select_sources(
    finalized: Mapping[str, dict[str, Any]],
    staging: Mapping[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    included = []
    excluded = []
    for load in LOADS:
        for seed in SEEDS:
            run_id = f"phasec_s1_{load}_seed{seed}"
            if run_id in finalized:
                included.append(dict(finalized[run_id]))
                continue
            candidates = list(staging.get(run_id, ()))
            if not candidates:
                excluded.append({
                    "run_id": run_id,
                    "load": load,
                    "seed": seed,
                    "reason": "missing_collection_and_staging",
                    "snapshots": 0,
                })
                continue
            signatures = {
                (
                    int(row["snapshot_count"]),
                    sha256_file(row["snapshot_index"]),
                )
                for row in candidates
            }
            if len(signatures) > 1:
                raise ValueError(
                    f"multiple non-identical staging attempts for {run_id}: "
                    f"{sorted(signatures)}"
                )
            source = max(
                candidates,
                key=lambda row: (
                    int(row["snapshot_count"]),
                    str(row["source_dir"]),
                ),
            )
            if int(source["snapshot_count"]) <= 0:
                excluded.append({
                    "run_id": run_id,
                    "load": load,
                    "seed": seed,
                    "reason": "zero_snapshots",
                    "snapshots": 0,
                    "source_dir": source["source_dir"].as_posix(),
                })
                continue
            included.append(dict(source))
    return included, excluded


def _link(source: Path, destination: Path, *, copy_mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        expected = source.resolve()
        if destination.is_symlink() and destination.resolve() == expected:
            return
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "symlink":
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    elif copy_mode == "copy":
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    else:
        raise ValueError(copy_mode)


def _write_recovery_collection(
    recovery_root: Path,
    source: Mapping[str, Any],
    *,
    protocol_sha256: str,
    copy_mode: str,
) -> dict[str, Any]:
    run_id = str(source["run_id"])
    target = recovery_root / "collections" / run_id
    target.mkdir(parents=True, exist_ok=True)
    _link(
        Path(source["snapshot_dir"]),
        target / "snapshots",
        copy_mode=copy_mode,
    )
    _link(
        Path(source["order_manifest"]),
        target / "order_manifest.json",
        copy_mode=copy_mode,
    )
    index_name = Path(source["snapshot_index"]).name
    relative_files = [Path("order_manifest.json"), Path("snapshots") / index_name]
    relative_files.extend(
        Path("snapshots") / Path(path).name
        for path in source["snapshot_files"]
    )
    output_manifest = target / "run_outputs.sha256"
    manifest_text = "".join(
        f"{sha256_file(target / relative)}  {relative.as_posix()}\n"
        for relative in relative_files
    )
    if output_manifest.is_file():
        if output_manifest.read_text(encoding="utf-8") != manifest_text:
            raise FileExistsError(
                f"changed recovery output manifest: {output_manifest}"
            )
    else:
        output_manifest.write_text(
            manifest_text, encoding="utf-8", newline="\n"
        )
    snapshot_count = int(source["snapshot_count"])
    formal_minimum_met = snapshot_count >= MIN_SNAPSHOTS_PER_RUN
    summary = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "development_only": True,
        "exploratory_underfilled": not formal_minimum_met,
        "formal_protocol_passed": False,
        "protocol_sha256": protocol_sha256,
        "load": str(source["load"]),
        "seed": int(source["seed"]),
        "ticks": 1500,
        "snapshot_interval": 100,
        "snapshot_top_m": 4,
        "snapshots": snapshot_count,
        "source": {
            "recovery_kind": str(source["source_kind"]),
            "original_directory": Path(source["source_dir"]).as_posix(),
            "original_summary": (
                Path(source["source_summary"]).as_posix()
                if source.get("source_summary") is not None else None
            ),
        },
        "outputs": {
            "snapshot_dir": "snapshots",
            "snapshot_index": f"snapshots/{index_name}",
            "order_manifest": "order_manifest.json",
        },
        "metrics": {},
        "audit": {
            "passed": True,
            "scope": "exploratory_recovery_integrity_only",
            "checks": {
                "source_snapshot_count_positive": snapshot_count > 0,
                "snapshot_files_complete": (
                    len(source["snapshot_files"]) == snapshot_count
                ),
                "formal_minimum_snapshots_met": formal_minimum_met,
                "formal_protocol_passed": False,
                "original_source_read_only": True,
            },
        },
        "run_outputs_sha256": sha256_file(output_manifest),
    }
    _atomic_json(target / "collection_summary.json", summary)
    return {
        "run_id": run_id,
        "load": str(source["load"]),
        "seed": int(source["seed"]),
        "snapshots": snapshot_count,
        "source_kind": str(source["source_kind"]),
        "formal_minimum_met": formal_minimum_met,
        "recovery_collection": target.as_posix(),
    }


def prepare(
    *,
    source_root: Path,
    recovery_root: Path,
    copy_mode: str,
) -> dict[str, Any]:
    frozen_path, bundle = _load_frozen_bundle(source_root)
    protocol_sha = str(bundle["protocol_sha256"])
    finalized = _discover_finalized(
        source_root, protocol_sha256=protocol_sha
    )
    staging = _discover_staging(source_root)
    included_sources, excluded = _select_sources(finalized, staging)
    recovery_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(frozen_path, recovery_root / FROZEN_FILENAME)
    frozen_manifest = source_root / "frozen_inputs.sha256"
    if frozen_manifest.is_file():
        shutil.copy2(
            frozen_manifest, recovery_root / "source_frozen_inputs.sha256"
        )
    included = [
        _write_recovery_collection(
            recovery_root,
            source,
            protocol_sha256=protocol_sha,
            copy_mode=copy_mode,
        )
        for source in included_sources
    ]
    tasks_path = recovery_root / "recovery_tasks.tsv"
    tasks_text = "".join(
        f"{row['load']}\t{row['seed']}\t{row['run_id']}\n"
        for row in included
    )
    if tasks_path.is_file():
        if tasks_path.read_text(encoding="utf-8") != tasks_text:
            raise FileExistsError(f"changed recovery task list: {tasks_path}")
    else:
        tasks_path.write_text(tasks_text, encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "development_only": True,
        "exploratory_underfilled": True,
        "formal_protocol_passed": False,
        "source_protocol_sha256": protocol_sha,
        "source_root": source_root.as_posix(),
        "recovery_root": recovery_root.as_posix(),
        "copy_mode": copy_mode,
        "included_runs": included,
        "excluded_runs": excluded,
        "counts": {
            "included_runs": len(included),
            "excluded_runs": len(excluded),
            "included_by_load": dict(Counter(
                row["load"] for row in included
            )),
            "snapshots_by_load": dict(Counter({
                load: sum(
                    int(row["snapshots"])
                    for row in included if row["load"] == load
                )
                for load in LOADS
            })),
            "total_snapshots": sum(
                int(row["snapshots"]) for row in included
            ),
            "underfilled_included_runs": sum(
                not bool(row["formal_minimum_met"]) for row in included
            ),
        },
        "analysis_contract": {
            "primary_weighting": "run-level macro and run-cluster CI",
            "pooled_metrics_are_secondary": True,
            "high_load_coverage_is_underfilled": True,
            "may_decide_whether_a_v2_rerun_is_worthwhile": True,
            "may_not_be_reported_as_formal_protocol_pass": True,
        },
        "outputs": {
            "tasks": tasks_path.name,
            "frozen_bundle": FROZEN_FILENAME,
        },
    }
    _atomic_json(recovery_root / "exploratory_recovery_manifest.json", manifest)
    print(
        f"[complete] exploratory recovery included={len(included)} "
        f"excluded={len(excluded)} snapshots={manifest['counts']['total_snapshots']}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    parser.add_argument(
        "--copy-mode",
        choices=("symlink", "copy"),
        default="symlink",
    )
    args = parser.parse_args()
    prepare(
        source_root=args.source_root,
        recovery_root=args.recovery_root,
        copy_mode=args.copy_mode,
    )


if __name__ == "__main__":
    main()
