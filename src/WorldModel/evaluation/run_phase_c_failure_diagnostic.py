"""Run one exact-replay arm for the targeted Phase-C failure diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from WorldModel.evaluation.phase_c_s1_hungarian_replication_501_510 import (
    install as install_replication_protocol,
)

install_replication_protocol()

from WorldModel.evaluation.phase_c_failure_diagnostic_probe import (  # noqa: E402
    FailureDecisionSnapshotProbe,
    FailureTrajectoryProbe,
)
from WorldModel.evaluation.phase_c_failure_diagnostic_protocol import (  # noqa: E402
    ARMS,
    DETERMINISTIC_METRIC_KEYS,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    FRAME_STRIDE,
    OUTPUT_ROOT,
    RUN_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SOURCE_BUNDLE,
    SOURCE_ROOT,
    TICKS,
    TRACE_HORIZONS,
    WM_DETERMINISTIC_METRIC_KEYS,
    canonical_sha256,
    case_for,
    run_id,
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_s1_hungarian_arm import (  # noqa: E402
    ARM_KEYS,
    _artifact,
    _load_bundle,
    _make_assigner,
    _read_json,
)
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner  # noqa: E402


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _verify_artifact(bundle: Mapping[str, Any], key: str) -> Path:
    value = (bundle.get("artifacts") or {}).get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"diagnostic frozen bundle lacks artifact {key!r}")
    path = Path(str(value.get("path", "")))
    expected = str(value.get("sha256", ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected:
        raise ValueError(f"diagnostic frozen artifact changed: {path}")
    return path


def _load_diagnostic_bundle(path: Path) -> tuple[dict, dict]:
    bundle = _read_json(path)
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"wrong diagnostic bundle schema: {path}")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("diagnostic bundle lacks protocol")
    protocol = dict(protocol)
    claimed = str(protocol.pop("protocol_sha256", ""))
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong diagnostic protocol schema")
    if canonical_sha256(protocol) != claimed:
        raise ValueError("diagnostic protocol hash mismatch")
    protocol["protocol_sha256"] = claimed
    return bundle, protocol


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _metric_audit(
    metrics: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    wm_arm: bool,
) -> dict[str, Any]:
    keys = list(DETERMINISTIC_METRIC_KEYS)
    if wm_arm:
        keys.extend(WM_DETERMINISTIC_METRIC_KEYS)
    comparisons = {}
    for key in keys:
        expected = reference.get(key)
        actual = metrics.get(key)
        comparisons[key] = {
            "expected": expected,
            "actual": actual,
            "passed": actual == expected,
        }
    return {
        "passed": all(row["passed"] for row in comparisons.values()),
        "comparisons": comparisons,
    }


def _write_output_hashes(root: Path) -> Path:
    output = root / "run_outputs.sha256"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"diagnostic_summary.json", output.name}
    )
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in files
    ]
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(rows) + "\n")
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
    summary_path = run_dir / "diagnostic_summary.json"
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
    checks["manifest_hash"] = (
        manifest.is_file()
        and sha256_file(manifest) == summary.get("run_outputs_sha256")
    )
    checks["outputs"] = _verify_output_hashes(run_dir, manifest)
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"cannot resume incompatible diagnostic {run_dir}: {failed}")
    print(f"[resume] {arm} {load} seed={seed}: {run_dir}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--load", choices=("low", "mid", "high"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--diagnostic-bundle",
        default=str(OUTPUT_ROOT / "phase_c_failure_diagnostic_protocol.json"),
    )
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--source-bundle", default=str(SOURCE_BUNDLE))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--capture-snapshots", action="store_true")
    args = parser.parse_args()

    case = case_for(args.load, args.seed)
    expected_snapshot_arm = str(case["snapshot_arm"])
    if bool(args.capture_snapshots) != (args.arm == expected_snapshot_arm):
        raise SystemExit(
            "snapshot capture must be enabled exactly for the frozen snapshot "
            f"arm {expected_snapshot_arm!r} in {args.load} seed={args.seed}"
        )

    diagnostic_bundle_path = Path(args.diagnostic_bundle)
    diagnostic_bundle, protocol = _load_diagnostic_bundle(
        diagnostic_bundle_path
    )
    protocol_sha = str(protocol["protocol_sha256"])
    source_root = Path(args.source_root)
    source_bundle_path = Path(args.source_bundle)
    output_root = Path(args.output_root)
    rid = run_id(args.arm, args.load, args.seed)
    run_dir = output_root / "runs" / rid
    if _resume(
        run_dir,
        protocol_sha256=protocol_sha,
        arm=args.arm,
        load=args.load,
        seed=args.seed,
    ):
        return

    _verify_artifact(diagnostic_bundle, "source_frozen_bundle")
    manifest_key = f"manifest_{args.load}_seed{args.seed}"
    reference_key = f"reference_{args.arm}_{args.load}_seed{args.seed}"
    manifest_path = _verify_artifact(diagnostic_bundle, manifest_key)
    reference_path = _verify_artifact(diagnostic_bundle, reference_key)
    if source_bundle_path != Path(
        str(
            (diagnostic_bundle.get("artifacts") or {})[
                "source_frozen_bundle"
            ]["path"]
        )
    ):
        raise ValueError("--source-bundle differs from the frozen diagnostic input")
    if source_root != Path(str(protocol["source_comparison"]["root"])):
        raise ValueError("--source-root differs from the frozen diagnostic input")

    source_bundle = _load_bundle(source_bundle_path)
    config_path = Path(str(_artifact(source_bundle, f"config_{args.load}")["path"]))
    manifest_payload = _read_json(manifest_path)
    reference_payload = _read_json(reference_path)
    reference_metrics = reference_payload.get("metrics") or {}

    if run_dir.exists():
        raise FileExistsError(
            f"partial diagnostic directory exists without resumable output: {run_dir}"
        )
    staging = output_root / "runs" / f".{rid}.tmp.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)

    assigner = _make_assigner(args.arm)
    wm_arm = args.arm in ("phasec", "phasec_s1")
    if wm_arm:
        assigner.decision_trace_enabled = True

    trace_path = staging / f"decision_trace_{rid}.jsonl"
    td_meta = {
        "diagnostic_schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol_sha,
        "arm": args.arm,
        "arm_label": ARM_KEYS[args.arm],
        "load": args.load,
        "seed": int(args.seed),
        "config": config_path.as_posix(),
        "checkpoint_path": (
            str(getattr(assigner, "checkpoint_path", "")) if wm_arm else None
        ),
        "order_manifest_path": manifest_path.as_posix(),
        "order_manifest_sha256": manifest_payload.get("manifest_sha256"),
    }

    import WorldModel.evaluation.decision_snapshot_probe as snapshot_module
    import WorldModel.evaluation.td_stream_probe as td_module

    original_snapshot_probe = snapshot_module.DecisionSnapshotProbe
    original_td_probe = td_module.TDStreamProbe
    snapshot_module.DecisionSnapshotProbe = FailureDecisionSnapshotProbe
    td_module.TDStreamProbe = FailureTrajectoryProbe
    try:
        metrics = _run_one_assigner(
            str(config_path),
            assigner,
            int(args.seed),
            TICKS,
            trace_label=ARM_KEYS[args.arm],
            decision_trace_jsonl=(str(trace_path) if wm_arm else None),
            decision_trace_horizons=list(TRACE_HORIZONS),
            td_stream_dir=str(staging),
            td_stream_frame_stride=FRAME_STRIDE,
            td_stream_run_id=rid,
            td_stream_meta=td_meta,
            decision_snapshot_dir=(
                str(staging / "snapshots")
                if args.capture_snapshots
                else None
            ),
            decision_snapshot_interval=1,
            decision_snapshot_top_m=10,
            decision_snapshot_run_id=rid,
            decision_snapshot_meta=td_meta,
            decision_snapshot_candidate_scope="all_idle_online",
            decision_snapshot_attach_trace=args.capture_snapshots,
            decision_snapshot_require_trace_alignment=args.capture_snapshots,
            decision_snapshot_capture_policy="interval_all",
            decision_snapshot_candidate_robot_mode="nearest",
            decision_snapshot_include_no_assign=False,
            decision_snapshot_max_contexts_per_tick=None,
            decision_snapshot_phase_c_round="failure_diagnostic_v1",
            recorded_orders_path=str(manifest_path),
        )
    finally:
        snapshot_module.DecisionSnapshotProbe = original_snapshot_probe
        td_module.TDStreamProbe = original_td_probe

    metric_audit = _metric_audit(
        metrics,
        reference_metrics,
        wm_arm=wm_arm,
    )
    trajectory_path = staging / f"trajectory_{rid}.jsonl"
    td_path = staging / f"tdstream_{rid}.pt"
    snapshot_dir = staging / "snapshots"
    snapindex_path = snapshot_dir / f"snapindex_{rid}.json"
    snapshot_files = (
        sorted(snapshot_dir.glob("*.pkl")) if snapshot_dir.is_dir() else []
    )
    trace_count = _line_count(trace_path) if wm_arm else 0
    trajectory_count = _line_count(trajectory_path)
    output_checks = {
        "metric_non_perturbation": metric_audit["passed"],
        "trajectory_complete": trajectory_count == TICKS,
        "td_stream_exists": td_path.is_file(),
        "wm_trace_exists": (trace_count > 0 if wm_arm else not trace_path.exists()),
        "snapshot_scope_matches": (
            snapindex_path.is_file() and len(snapshot_files) > 0
            if args.capture_snapshots
            else not snapshot_files and not snapindex_path.exists()
        ),
        "manifest_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest_payload.get("manifest_sha256")
        ),
        "manifest_count_matches": int(metrics.get("order_arrival_count", -1))
        == int(manifest_payload.get("total_orders", -2)),
    }
    audit = {
        "passed": all(output_checks.values()),
        "checks": output_checks,
        "metric_comparison": metric_audit["comparisons"],
    }
    if not audit["passed"]:
        failed = [key for key, passed in output_checks.items() if not passed]
        raise RuntimeError(
            f"diagnostic audit failed for {rid}: {', '.join(failed)}"
        )

    output_hash_path = _write_output_hashes(staging)
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "protocol_sha256": protocol_sha,
        "arm": args.arm,
        "arm_label": ARM_KEYS[args.arm],
        "load": args.load,
        "seed": int(args.seed),
        "ticks": TICKS,
        "capture_snapshots": bool(args.capture_snapshots),
        "question": case["question"],
        "source": {
            "diagnostic_bundle": diagnostic_bundle_path.as_posix(),
            "source_bundle": source_bundle_path.as_posix(),
            "reference_result": reference_path.as_posix(),
            "order_manifest": manifest_path.as_posix(),
            "order_manifest_sha256": manifest_payload.get("manifest_sha256"),
        },
        "outputs": {
            "decision_trace": (
                trace_path.name if trace_path.is_file() else None
            ),
            "decision_trace_rows": trace_count,
            "trajectory": trajectory_path.name,
            "trajectory_rows": trajectory_count,
            "td_stream": td_path.name,
            "snapshot_index": (
                f"snapshots/{snapindex_path.name}"
                if snapindex_path.is_file()
                else None
            ),
            "snapshot_count": len(snapshot_files),
        },
        "metrics": metrics,
        "audit": audit,
        "run_outputs_manifest": output_hash_path.name,
        "run_outputs_sha256": sha256_file(output_hash_path),
    }
    _atomic_write_json(staging / "diagnostic_summary.json", summary)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(run_dir)
    print(f"[complete] {rid}: {run_dir}")


if __name__ == "__main__":
    main()
