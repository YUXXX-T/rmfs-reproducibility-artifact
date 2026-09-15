"""Collect one fresh 531--540 arm with same-tick station instrumentation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from WorldModel.evaluation.phase_c_phi_state_531_540_protocol import (
    ARMS,
    FRAME_STRIDE,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    HEAD_CHECKPOINT,
    LOADS,
    OUTPUT_ROOT,
    RUN_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SEEDS,
    SOURCE_BUNDLE,
    TICKS,
    canonical_sha256,
    run_id,
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_s1_hungarian_arm import (
    ARM_KEYS,
    _artifact,
    _load_bundle,
    _make_assigner,
    _read_json,
)
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
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


def _load_frozen_bundle(path: Path) -> tuple[dict, dict]:
    bundle = _read_json(path)
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"wrong phi_state bundle schema: {path}")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("phi_state bundle lacks protocol")
    protocol = dict(protocol)
    claimed = str(protocol.pop("protocol_sha256", ""))
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong phi_state protocol schema")
    if canonical_sha256(protocol) != claimed:
        raise ValueError("phi_state protocol hash mismatch")
    protocol["protocol_sha256"] = claimed
    return bundle, protocol


def _verify_bundle_artifact(bundle: Mapping[str, Any], key: str) -> Path:
    value = (bundle.get("artifacts") or {}).get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"phi_state bundle lacks artifact {key!r}")
    path = Path(str(value.get("path", "")))
    expected = str(value.get("sha256", ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected:
        raise ValueError(f"frozen artifact changed: {path}")
    return path


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _write_output_hashes(root: Path) -> Path:
    output = root / "run_outputs.sha256"
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"phi_state_run_summary.json", output.name}
    )
    output.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
            for path in files
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


def _verify_output_hashes(root: Path, manifest: Path) -> bool:
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
    arm: str,
    load: str,
    seed: int,
) -> bool:
    summary_path = run_dir / "phi_state_run_summary.json"
    if not summary_path.is_file():
        return False
    summary = _read_json(summary_path)
    checks = {
        "schema": summary.get("schema_version") == RUN_SCHEMA_VERSION,
        "protocol": summary.get("protocol_sha256") == protocol_sha256,
        "arm": summary.get("arm") == arm,
        "load": summary.get("load") == load,
        "seed": int(summary.get("seed", -1)) == int(seed),
        "audit": bool((summary.get("audit") or {}).get("passed")),
    }
    manifest = run_dir / "run_outputs.sha256"
    checks["output_manifest"] = (
        manifest.is_file()
        and sha256_file(manifest) == summary.get("run_outputs_sha256")
    )
    checks["outputs"] = _verify_output_hashes(run_dir, manifest)
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"cannot resume incompatible run {run_dir}: {failed}")
    print(f"[resume] {arm} {load} seed={seed}: {run_dir}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--load", choices=LOADS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--frozen-bundle",
        default=str(OUTPUT_ROOT / "phase_c_phi_state_frozen_protocol.json"),
    )
    parser.add_argument("--source-bundle", default=str(SOURCE_BUNDLE))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    if int(args.seed) not in SEEDS:
        raise SystemExit(f"seed must be one of {list(SEEDS)}")
    rid = run_id(args.arm, args.load, args.seed)
    bundle_path = Path(args.frozen_bundle)
    frozen_bundle, protocol = _load_frozen_bundle(bundle_path)
    protocol_sha = str(protocol["protocol_sha256"])
    source_bundle_path = Path(args.source_bundle)
    expected_source_bundle = _verify_bundle_artifact(
        frozen_bundle, "source_frozen_bundle"
    )
    if source_bundle_path != expected_source_bundle:
        raise ValueError("--source-bundle differs from frozen protocol input")
    frozen_head = _verify_bundle_artifact(frozen_bundle, "station_head_checkpoint")
    if frozen_head != HEAD_CHECKPOINT:
        raise ValueError("frozen head path differs from protocol")
    source_bundle = _load_bundle(source_bundle_path)
    config_path = Path(str(_artifact(source_bundle, f"config_{args.load}")["path"]))

    output_root = Path(args.output_root)
    run_dir = output_root / "runs" / rid
    if _resume(
        run_dir,
        protocol_sha256=protocol_sha,
        arm=args.arm,
        load=args.load,
        seed=args.seed,
    ):
        return
    manifest_path = output_root / "order_manifests" / (
        f"orders_{args.load}_seed{args.seed}.json"
    )
    if args.arm != "greedy" and not manifest_path.is_file():
        raise FileNotFoundError(
            f"Greedy manifest must finish before this arm: {manifest_path}"
        )
    if run_dir.exists():
        raise FileExistsError(
            f"partial output exists without resumable summary: {run_dir}"
        )
    staging = output_root / "runs" / f".{rid}.tmp.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)

    # _make_assigner is imported from the already-frozen 491--500 arm module;
    # it uses the same relative checkpoint and S1 configuration recorded in
    # the source bundle.  No new policy parameters are introduced here.
    assigner = _make_assigner(args.arm)
    td_meta = {
        "diagnostic_schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol_sha,
        "arm": args.arm,
        "arm_label": ARM_KEYS[args.arm],
        "load": args.load,
        "seed": int(args.seed),
        "ticks": TICKS,
        "config": config_path.as_posix(),
        "source_bundle": source_bundle_path.as_posix(),
        "order_manifest": manifest_path.as_posix(),
        "head_checkpoint": frozen_head.as_posix(),
        "same_tick_trace": True,
    }
    run_kwargs: dict[str, Any] = {}
    if args.arm == "greedy":
        if manifest_path.exists():
            raise FileExistsError(
                f"manifest exists without resumable Greedy run: {manifest_path}"
            )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        run_kwargs["save_order_manifest"] = str(manifest_path)
    else:
        run_kwargs["recorded_orders_path"] = str(manifest_path)

    import WorldModel.evaluation.td_stream_probe as td_module
    from WorldModel.evaluation.station_congestion_trace_probe import (
        StationCongestionTraceProbe,
    )

    original_probe = td_module.TDStreamProbe
    td_module.TDStreamProbe = StationCongestionTraceProbe
    try:
        metrics = _run_one_assigner(
            str(config_path),
            assigner,
            int(args.seed),
            TICKS,
            trace_label=ARM_KEYS[args.arm],
            td_stream_dir=str(staging),
            td_stream_frame_stride=FRAME_STRIDE,
            td_stream_run_id=rid,
            td_stream_meta=td_meta,
            **run_kwargs,
        )
    finally:
        td_module.TDStreamProbe = original_probe

    manifest_payload = _read_json(manifest_path)
    trace_path = staging / f"station_congestion_trace_{rid}.jsonl"
    td_path = staging / f"tdstream_{rid}.pt"
    checks = {
        "fresh_seed": int(args.seed) in SEEDS,
        "trace_complete": _line_count(trace_path) == TICKS,
        "td_stream_exists": td_path.is_file(),
        "td_frames_saved": int(metrics.get("td_stream_saved_frames", 0)) > 0,
        "manifest_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest_payload.get("manifest_sha256")
        ),
        "manifest_count_matches": int(metrics.get("order_arrival_count", -1))
        == int(manifest_payload.get("total_orders", -2)),
        "manifest_semantics": (
            not bool(metrics.get("order_arrival_replayed"))
            if args.arm == "greedy"
            else bool(metrics.get("order_arrival_replayed"))
        ),
    }
    if args.arm in {"phasec", "phasec_s1"}:
        checks.update({
            "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
            "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
            "native_no_assign_disabled": not bool(
                metrics.get("native_no_assign_enabled", False)
            ),
        })
    audit = {"passed": all(checks.values()), "checks": checks}
    if not audit["passed"]:
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError(f"phi_state run audit failed: {failed}")

    output_hash_path = _write_output_hashes(staging)
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "protocol_sha256": protocol_sha,
        "arm": args.arm,
        "arm_label": ARM_KEYS[args.arm],
        "load": args.load,
        "seed": int(args.seed),
        "ticks": TICKS,
        "frame_stride": FRAME_STRIDE,
        "source": {
            "frozen_bundle": bundle_path.as_posix(),
            "source_bundle": source_bundle_path.as_posix(),
            "config": config_path.as_posix(),
            "order_manifest": manifest_path.as_posix(),
            "order_manifest_sha256": manifest_payload.get("manifest_sha256"),
            "head_checkpoint": frozen_head.as_posix(),
        },
        "outputs": {
            "trace": trace_path.name,
            "trace_rows": TICKS,
            "td_stream": td_path.name,
            "saved_frames": int(metrics.get("td_stream_saved_frames", 0)),
        },
        "metrics": metrics,
        "audit": audit,
        "run_outputs_manifest": output_hash_path.name,
        "run_outputs_sha256": sha256_file(output_hash_path),
    }
    _atomic_write_json(staging / "phi_state_run_summary.json", summary)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(run_dir)
    print(f"[complete] {rid}: {run_dir}")


if __name__ == "__main__":
    main()
