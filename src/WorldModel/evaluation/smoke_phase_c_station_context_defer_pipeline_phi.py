"""Run a short paired-manifest wiring smoke for station-context defer V2.

This script is not an outcome experiment and must not be used for tuning.  It
creates short reference stubs solely to exercise the production V2 runner's
manifest, risk-mode, admission, liveness, and output-schema audits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_station_context_defer_pipeline_phi import (
    ARM_KEY,
    BASELINE_ARM_KEY,
    READY_MAX_V1_ARM_KEY,
    SCHEMA_VERSION,
    main as run_v2_main,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _manifest_content_sha256(
    schema_version: str,
    orders: list[dict[str, Any]],
) -> str:
    canonical = {
        "schema_version": schema_version,
        "orders": orders,
    }
    canonical = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _reference_stub(
    arm: str,
    load: str,
    seed: int,
    ticks: int,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "meta": {"arm_key": arm, "load": load, "seed": seed, "ticks": ticks},
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "metrics": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", choices=("low", "mid", "high"), default="low")
    parser.add_argument("--seed", type=int, default=491)
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--frozen-bundle", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    if args.work_root.exists():
        raise SystemExit(f"refusing to overwrite smoke root: {args.work_root}")
    manifest_path = args.work_root / "manifest" / args.manifest.name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    smoke_orders = [
        order
        for order in source_manifest.get("orders", [])
        if int(order.get("tick", 0)) < int(args.ticks)
    ]
    manifest = {
        "schema_version": source_manifest.get("schema_version"),
        "orders": smoke_orders,
        "total_orders": len(smoke_orders),
        "manifest_sha256": _manifest_content_sha256(
            str(source_manifest.get("schema_version")), smoke_orders
        ),
    }
    _write_json(manifest_path, manifest)
    reference_root = args.work_root / "reference"
    v1_reference_root = args.work_root / "v1_reference"
    output_root = args.work_root / "output"
    baseline_path = (
        reference_root / "per_arm" / BASELINE_ARM_KEY
        / f"{args.load}_seed{args.seed}.json"
    )
    v1_path = (
        v1_reference_root / "per_arm" / READY_MAX_V1_ARM_KEY
        / f"{args.load}_seed{args.seed}.json"
    )
    _write_json(
        baseline_path,
        _reference_stub(
            BASELINE_ARM_KEY,
            args.load,
            args.seed,
            args.ticks,
            manifest_path,
            manifest,
        ),
    )
    _write_json(
        v1_path,
        _reference_stub(
            READY_MAX_V1_ARM_KEY,
            args.load,
            args.seed,
            args.ticks,
            manifest_path,
            manifest,
        ),
    )

    import sys

    original_argv = sys.argv
    sys.argv = [
        "run_phase_c_station_context_defer_pipeline_phi",
        "--load", args.load,
        "--seed", str(args.seed),
        "--ticks", str(args.ticks),
        "--source-root", str(args.source_root),
        "--reference-root", str(reference_root),
        "--v1-reference-root", str(v1_reference_root),
        "--frozen-bundle", str(args.frozen_bundle),
        "--output-root", str(output_root),
        "--trace-max-records", "100",
    ]
    try:
        run_v2_main()
    finally:
        sys.argv = original_argv

    result_path = (
        output_root / "per_arm" / ARM_KEY
        / f"{args.load}_seed{args.seed}.json"
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    checks = {
        "schema": payload.get("schema_version") == SCHEMA_VERSION,
        "audit": bool((payload.get("audit") or {}).get("passed")),
        "ready_diagnostic_only": not bool(
            metrics.get(
                "station_context_defer_ready_contention_in_station_risk",
                True,
            )
        ),
        "admission": bool(
            (payload.get("station_admission_audit") or {}).get("passed")
        ),
        "liveness": int(
            metrics.get("station_context_defer_liveness_bound_violations", -1)
        ) == 0,
        "defer_evaluated": int(
            metrics.get("station_context_defer_evaluations", 0)
        ) > 0,
    }
    summary = {
        "schema_version": "phase_c_station_context_defer_pipeline_phi_smoke_v1",
        "outcome_not_for_tuning": True,
        "source_manifest_sha256": sha256_file(args.manifest),
        "smoke_manifest_sha256": sha256_file(manifest_path),
        "manifest_content_sha256": manifest.get("manifest_sha256"),
        "smoke_manifest_orders": manifest.get("total_orders"),
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": {
            "completed_orders": metrics.get("completed_orders"),
            "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
            "station_context_defer_rate": metrics.get(
                "station_context_defer_rate"
            ),
            "station_context_defer_all_remaining_batches": metrics.get(
                "station_context_defer_all_remaining_batches"
            ),
            "station_context_defer_ready_would_dominate_evaluations": (
                metrics.get(
                    "station_context_defer_ready_would_dominate_evaluations"
                )
            ),
        },
        "result": result_path.as_posix(),
    }
    report_path = args.work_root / "smoke_report.json"
    _write_json(report_path, summary)
    if not summary["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("V2 wiring smoke failed: " + ", ".join(failed))
    # Reference stubs have served their only purpose.  Keep the auditable V2
    # output and report, while avoiding confusion with real reference results.
    shutil.rmtree(reference_root)
    shutil.rmtree(v1_reference_root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
