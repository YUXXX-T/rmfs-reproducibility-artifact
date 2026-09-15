"""Collect fresh Phase-C/S1 decision snapshots for the frozen H=10 audit."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from Policies.TaskAssigner import WorldModelTaskAssigner
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
from WorldModel.evaluation.freeze_phase_c_station_congestion_endpoint_h10 import (
    FROZEN_FILENAME,
)
from WorldModel.evaluation.phase_c_station_congestion_endpoint_protocol import (
    BEHAVIOUR_TOP_M,
    COLLECTION_SCHEMA_VERSION,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOADS,
    MIN_SNAPSHOTS_PER_RUN,
    OUTPUT_ROOT,
    SCHEMA_VERSION,
    SEEDS,
    SNAPSHOT_CANDIDATE_ROBOT_MODE,
    SNAPSHOT_INTERVAL,
    SNAPSHOT_MAX_CONTEXTS_PER_TICK,
    SNAPSHOT_TOP_M,
    TICKS,
    canonical_sha256,
    sha256_file,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _load_bundle(path: Path) -> dict[str, Any]:
    bundle = _read_json(path)
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError("wrong endpoint frozen bundle schema")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("endpoint frozen bundle lacks protocol")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong endpoint protocol schema")
    if canonical_sha256(protocol) != bundle.get("protocol_sha256"):
        raise ValueError("endpoint frozen protocol hash mismatch")
    return bundle


def _verify_artifact(bundle: Mapping[str, Any], path: Path) -> None:
    row = (bundle.get("artifacts") or {}).get(path.as_posix())
    if not isinstance(row, Mapping):
        raise ValueError(f"frozen bundle lacks artifact {path}")
    if not path.is_file() or sha256_file(path) != row.get("sha256"):
        raise ValueError(f"frozen endpoint artifact changed: {path}")


def _write_hashes(root: Path) -> Path:
    output = root / "run_outputs.sha256"
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {
            output.name,
            "collection_summary.json",
        }:
            continue
        rows.append(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        )
    output.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return output


def _verify_hashes(root: Path, manifest: Path) -> bool:
    if not manifest.is_file():
        return False
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(None, 1)
        path = root / relative.strip()
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return True


def _resume(
    run_dir: Path,
    *,
    protocol_sha256: str,
    load: str,
    seed: int,
) -> bool:
    summary_path = run_dir / "collection_summary.json"
    if not summary_path.is_file():
        return False
    summary = _read_json(summary_path)
    manifest = run_dir / "run_outputs.sha256"
    checks = {
        "schema": summary.get("schema_version") == COLLECTION_SCHEMA_VERSION,
        "protocol": summary.get("protocol_sha256") == protocol_sha256,
        "load": summary.get("load") == load,
        "seed": int(summary.get("seed", -1)) == int(seed),
        "audit": bool((summary.get("audit") or {}).get("passed")),
        "manifest_hash": (
            manifest.is_file()
            and sha256_file(manifest) == summary.get("run_outputs_sha256")
        ),
        "outputs": _verify_hashes(run_dir, manifest),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"cannot resume endpoint collection {run_dir}: {failed}")
    print(f"[resume] endpoint collection load={load} seed={seed}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", choices=LOADS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--frozen-bundle", type=Path, default=None)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--snapshot-interval", type=int, default=SNAPSHOT_INTERVAL)
    parser.add_argument("--snapshot-top-m", type=int, default=SNAPSHOT_TOP_M)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(int(args.torch_threads), 1))
    bundle_path = args.frozen_bundle or args.output_root / FROZEN_FILENAME
    bundle = _load_bundle(bundle_path)
    protocol = bundle["protocol"]
    protocol_sha = str(bundle["protocol_sha256"])
    if not args.development:
        if int(args.seed) not in SEEDS:
            raise SystemExit(f"formal development seed must be one of {list(SEEDS)}")
        if int(args.ticks) != TICKS:
            raise SystemExit(f"frozen collection requires --ticks={TICKS}")
        if int(args.snapshot_interval) != SNAPSHOT_INTERVAL:
            raise SystemExit(
                f"frozen collection requires --snapshot-interval={SNAPSHOT_INTERVAL}"
            )
        if int(args.snapshot_top_m) != SNAPSHOT_TOP_M:
            raise SystemExit(
                f"frozen collection requires --snapshot-top-m={SNAPSHOT_TOP_M}"
            )
    if int(args.ticks) <= 0 or int(args.snapshot_interval) <= 0:
        raise SystemExit("ticks and snapshot interval must be positive")
    if int(args.snapshot_top_m) < 2:
        raise SystemExit("snapshot top-m must be at least two")

    inputs = protocol["inputs"]
    model_path = Path(str(inputs["model_checkpoint"]))
    config_path = Path(str(inputs["loads"][args.load]))
    _verify_artifact(bundle, model_path)
    _verify_artifact(bundle, config_path)

    run_id = f"phasec_s1_{args.load}_seed{args.seed}"
    run_dir = args.output_root / "collections" / run_id
    if _resume(
        run_dir,
        protocol_sha256=protocol_sha,
        load=args.load,
        seed=args.seed,
    ):
        return
    if run_dir.exists():
        raise FileExistsError(
            f"partial endpoint collection exists without summary: {run_dir}"
        )
    staging = run_dir.parent / f".{run_id}.tmp.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    snapshot_dir = staging / "snapshots"
    order_manifest = staging / "order_manifest.json"

    behaviour = protocol["state_collection"]
    assigner_config = dict(behaviour["behaviour_config"])
    assigner = WorldModelTaskAssigner(
        checkpoint_path=str(model_path),
        top_m=int(behaviour["behaviour_top_m"]),
        energy_conv_random_flip_seed=0,
        **assigner_config,
    )
    metrics = _run_one_assigner(
        str(config_path),
        assigner,
        int(args.seed),
        int(args.ticks),
        trace_label="StationCongestionEndpointPhaseCS1",
        decision_snapshot_dir=str(snapshot_dir),
        decision_snapshot_interval=int(args.snapshot_interval),
        decision_snapshot_top_m=int(args.snapshot_top_m),
        decision_snapshot_run_id=run_id,
        decision_snapshot_meta={
            "arm_label": "StationCongestionEndpointPhaseCS1",
            "load": args.load,
            "seed": int(args.seed),
            "config": config_path.as_posix(),
            "checkpoint_path": model_path.as_posix(),
            "endpoint_protocol_sha256": protocol_sha,
        },
        decision_snapshot_candidate_scope="top_m_snapshot",
        decision_snapshot_attach_trace=False,
        decision_snapshot_capture_policy="interval_all",
        decision_snapshot_candidate_robot_mode=(
            SNAPSHOT_CANDIDATE_ROBOT_MODE
        ),
        decision_snapshot_include_no_assign=False,
        decision_snapshot_max_contexts_per_tick=(
            SNAPSHOT_MAX_CONTEXTS_PER_TICK
        ),
        decision_snapshot_phase_c_round="station_congestion_endpoint_h10_v1",
        decision_snapshot_exclude_external_baselines=True,
        save_order_manifest=str(order_manifest),
    )
    indexes = list(snapshot_dir.glob("snapindex_*.json"))
    if len(indexes) != 1:
        raise RuntimeError(f"expected one snapshot index, got {len(indexes)}")
    index = _read_json(indexes[0])
    snapshot_count = int(index.get("n_snapshots", -1))
    minimum = 1 if args.development else MIN_SNAPSHOTS_PER_RUN
    checks = {
        "snapshots_complete": snapshot_count >= minimum,
        "metrics_count_matches_index": int(
            metrics.get("decision_snapshots_saved", -1)
        ) == snapshot_count,
        "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
        "no_greedy_fallback": int(
            metrics.get("fallback_greedy_calls", 0)
        ) == 0,
        "native_no_assign_disabled": not bool(
            metrics.get("native_no_assign_enabled", False)
        ),
        "s1_observed": int(metrics.get("energy_conv_contexts", 0)) > 0,
        "assignment_candidates_only": not bool(
            index.get("include_no_assign_candidate", True)
        ),
        "one_context_per_tick": int(
            index.get("max_contexts_per_tick", -1)
        ) == SNAPSHOT_MAX_CONTEXTS_PER_TICK,
        "stratified_candidates": (
            index.get("candidate_robot_mode")
            == SNAPSHOT_CANDIDATE_ROBOT_MODE
        ),
        "order_manifest_saved": order_manifest.is_file(),
    }
    audit = {"passed": all(checks.values()), "checks": checks}
    if not audit["passed"]:
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError(f"endpoint collection audit failed: {failed}")

    output_manifest = _write_hashes(staging)
    summary = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "development_only": True,
        "protocol_sha256": protocol_sha,
        "load": args.load,
        "seed": int(args.seed),
        "ticks": int(args.ticks),
        "snapshot_interval": int(args.snapshot_interval),
        "snapshot_top_m": int(args.snapshot_top_m),
        "snapshots": snapshot_count,
        "source": {
            "frozen_bundle": bundle_path.as_posix(),
            "model_checkpoint": model_path.as_posix(),
            "config": config_path.as_posix(),
        },
        "outputs": {
            "snapshot_dir": "snapshots",
            "snapshot_index": indexes[0].relative_to(staging).as_posix(),
            "order_manifest": order_manifest.name,
        },
        "metrics": metrics,
        "audit": audit,
        "run_outputs_sha256": sha256_file(output_manifest),
    }
    _atomic_json(staging / "collection_summary.json", summary)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(run_dir)
    print(f"[complete] endpoint collection: {run_dir}")


if __name__ == "__main__":
    main()
