"""Generate paired manifests and run Dynamic-J ETA-overbooking stress seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import LOADS


BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_SOURCE_ROOT = BASE_ROOT / "psi_dispatch_dynamic_eta_overbooking_stress_source_551_560_v1"
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "psi_dispatch_dynamic_eta_overbooking_551_560_v3"
DEFAULT_BUNDLE = (
    BASE_ROOT
    / "psi_dispatch_ablation_551_560_v1"
    / "phase_c_psi_dispatch_frozen_protocol.json"
)
ARM_KEY = "s1_psi_dynamic_eta_overbooking_v3"


def _run(command: list[str], log_path: Path, *, append: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[exec]", " ".join(command), flush=True)
    with log_path.open("a" if append else "w", encoding="utf-8") as handle:
        if append:
            handle.write("\n[resume-validation]\n")
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; see {log_path}"
        )


def _summary(output_root: Path, load: str, seeds: list[int]) -> dict:
    rows = []
    for seed in seeds:
        path = output_root / "per_arm" / ARM_KEY / f"{load}_seed{seed}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics") or {}
        admission = payload.get("station_admission_audit") or {}
        station_rows = admission.get("station_metrics_final") or []
        rows.append({
            "seed": seed,
            "completed_orders": metrics.get("completed_orders"),
            "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
            "stall_ratio_mean": metrics.get("stall_ratio_mean"),
            "capacity_rejections": sum(
                int(row.get("rejected_capacity", 0)) for row in station_rows
            ),
            "overbooking_grants": sum(
                int((row.get("dynamic_eta") or {}).get("overbooking_grants", 0))
                for row in station_rows
            ),
            "max_committed_load": max(
                [int(value) for value in admission.get("max_committed_load", {}).values()]
                or [0]
            ),
            "max_occupancy": max(
                [int(value) for value in admission.get("max_occupancy", {}).values()]
                or [0]
            ),
            "physical_capacity_violations": int(
                admission.get("physical_capacity_violation_count", 0)
            ),
            "hard_limit_violations": int(
                admission.get("hard_limit_violation_count", 0)
            ),
            "token_mismatches": int(admission.get("token_mismatch_count", 0)),
            "path": path.as_posix(),
        })
    if not rows:
        return {"load": load, "runs": []}

    def mean(key: str) -> float:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return sum(values) / max(len(values), 1)

    return {
        "load": load,
        "run_count": len(rows),
        "completed_orders_mean": round(mean("completed_orders"), 6),
        "deadlock_ratio_mean": round(mean("deadlock_ratio_mean"), 6),
        "stall_ratio_mean": round(mean("stall_ratio_mean"), 6),
        "capacity_rejections_sum": sum(row["capacity_rejections"] for row in rows),
        "overbooking_grants_sum": sum(row["overbooking_grants"] for row in rows),
        "physical_capacity_violations_sum": sum(
            row["physical_capacity_violations"] for row in rows
        ),
        "hard_limit_violations_sum": sum(
            row["hard_limit_violations"] for row in rows
        ),
        "token_mismatches_sum": sum(row["token_mismatches"] for row in rows),
        "max_committed_load": max(row["max_committed_load"] for row in rows),
        "max_occupancy": max(row["max_occupancy"] for row in rows),
        "runs": rows,
    }


def _canonical_manifest_sha(payload: dict) -> str:
    canonical = {
        "schema_version": payload.get("schema_version"),
        "orders": payload.get("orders"),
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("orders")
    if (
        payload.get("schema_version") != "layer5_order_arrival_manifest_v1"
        or not isinstance(rows, list)
        or int(payload.get("total_orders", -1)) != len(rows)
        or _canonical_manifest_sha(payload) != payload.get("manifest_sha256")
    ):
        raise ValueError(f"invalid manifest: {path}")
    return payload


def _recorded_manifest_contract(
    reference_root: Path, load: str, seed: int
) -> tuple[str, int] | None:
    candidates = [
        reference_root
        / "per_arm"
        / "s1_psi_dynamic_committed_admission_v1"
        / f"{load}_seed{seed}.json",
        BASE_ROOT
        / "psi_dispatch_dynamic_probe_551_560_h1500_v1"
        / "per_arm"
        / "s1_psi_dynamic_probe"
        / f"{load}_seed{seed}.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = payload.get("manifest") or {}
        content_sha = manifest.get("content_sha256")
        total_orders = manifest.get("total_orders")
        if content_sha and total_orders is not None:
            return str(content_sha), int(total_orders)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", choices=LOADS, default="high")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(551, 561)))
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frozen-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--trace-max-records", type=int, default=100)
    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument(
        "--manifest-reference-root",
        type=Path,
        default=BASE_ROOT / "psi_dispatch_dynamic_admission_551_560_v1",
    )
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()

    args.source_root.mkdir(parents=True, exist_ok=True)
    logs = args.output_root / "logs"
    python = sys.executable
    for seed in args.seeds:
        manifest = (
            args.source_root
            / "order_manifests"
            / f"orders_{args.load}_seed{seed}.json"
        )
        if not args.skip_manifest and not manifest.is_file():
            manifest_command = [
                python,
                "-m",
                "WorldModel.evaluation.run_phase_c_psi_dispatch_arm",
                "--arm",
                "greedy_manifest",
                "--load",
                args.load,
                "--seed",
                str(seed),
                "--frozen-bundle",
                str(args.frozen_bundle),
                "--output-root",
                str(args.source_root),
                "--ticks",
                str(args.ticks),
            ]
            if int(args.ticks) != 1500:
                manifest_command.append("--development")
            _run(
                manifest_command,
                logs / f"manifest_{args.load}_seed{seed}.log",
            )

        manifest_payload = _validate_manifest(manifest)
        recorded = _recorded_manifest_contract(
            args.manifest_reference_root, args.load, seed
        )
        if recorded is not None:
            expected_sha, expected_count = recorded
            actual_sha = str(manifest_payload["manifest_sha256"])
            actual_count = int(manifest_payload["total_orders"])
            if actual_sha != expected_sha or actual_count != expected_count:
                raise RuntimeError(
                    f"rebuilt manifest differs from recorded contract "
                    f"{args.load} seed={seed}: "
                    f"sha {actual_sha} != {expected_sha} or "
                    f"count {actual_count} != {expected_count}"
                )
            print(
                f"[verified] historical manifest content {args.load} "
                f"seed={seed} sha={actual_sha} orders={actual_count}",
                flush=True,
            )

        output = (
            args.output_root
            / "per_arm"
            / ARM_KEY
            / f"{args.load}_seed{seed}.json"
        )
        _run([
            python,
            "-m",
            "WorldModel.evaluation.run_phase_c_psi_dynamic_eta_overbooking",
            "--load",
            args.load,
            "--seed",
            str(seed),
            "--ticks",
            str(args.ticks),
            "--trace-max-records",
            str(args.trace_max_records),
            "--source-root",
            str(args.source_root),
            "--frozen-bundle",
            str(args.frozen_bundle),
            "--output-root",
            str(args.output_root),
        ], logs / f"dynamic_eta_{args.load}_seed{seed}.log", append=output.is_file())

    summary = _summary(args.output_root, args.load, args.seeds)
    if args.no_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return
    summary_path = args.output_root / "validation" / f"stress_{args.load}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
