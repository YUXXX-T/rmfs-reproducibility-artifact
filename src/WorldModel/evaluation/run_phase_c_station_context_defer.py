"""Run Dynamic J + station-local defer + S1 + committed admission V1.

The arm is additive and writes only to a new output root.  It replays the
exact manifests used by the existing Dynamic-J committed-admission block, so
the primary comparison changes only the scheduling-layer context defer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from Policies.TaskAssigner.WorldModelTaskAssigner.station_context_defer_assigner import (
    StationContextDeferDynamicPsiAssigner,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    LOADS,
    S1_CONFIG,
    TOP_M,
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_psi_dynamic_admission import (
    StationAdmissionAuditProbe,
)
from WorldModel.evaluation.run_phase_c_psi_dynamic_probe import (
    _artifact,
    _atomic_json,
    _load_bundle,
    _read_json,
)
from WorldState.station_state import STATION_ADMISSION_COMMITTED_V1


SCHEMA_VERSION = "phase_c_station_context_defer_arm_v1"
ARM_KEY = "s1_psi_dynamic_station_context_defer_admission_v1"
ARM_LABEL = "PhaseCS1PsiDynamicStationContextDeferAdmissionV1"
BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_SOURCE_ROOT = BASE_ROOT / "psi_dispatch_ablation_551_560_v1"
DEFAULT_REFERENCE_ROOT = BASE_ROOT / "psi_dispatch_dynamic_admission_551_560_v1"
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "station_context_defer_551_560_v1"


def _output_path(root: Path, load: str, seed: int) -> Path:
    return root / "per_arm" / ARM_KEY / f"{load}_seed{seed}.json"


def _reference_path(root: Path, load: str, seed: int) -> Path:
    return (
        root
        / "per_arm"
        / "s1_psi_dynamic_committed_admission_v1"
        / f"{load}_seed{seed}.json"
    )


def _manifest_from_reference(reference: Mapping[str, Any]) -> Path:
    manifest = reference.get("manifest") or {}
    path = Path(str(manifest.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(
            "paired manifest is not local; sync the exact order_manifests "
            f"listed by the reference result: {path}"
        )
    expected = manifest.get("file_sha256")
    if expected and sha256_file(path) != str(expected):
        raise ValueError(f"paired manifest changed: {path}")
    return path


def _metric_subset(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "completed_orders",
        "completed_tasks",
        "deadlock_ratio_mean",
        "deadlock_ratio_max",
        "stall_ratio_mean",
        "stall_ratio_max",
        "pending_order_count",
        "open_order_count",
    )
    return {key: metrics.get(key) for key in keys}


def _audit(
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    admission: Mapping[str, Any],
    ticks: int,
) -> dict[str, Any]:
    checks = {
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "manifest_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_count_matches": (
            int(metrics.get("order_arrival_count", -1))
            == int(manifest.get("total_orders", -2))
        ),
        "ticks_match": int(metrics.get("ticks", -1)) == int(ticks),
        "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
        "s1_used": int(metrics.get("energy_conv_contexts", 0)) > 0,
        "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
        "dynamic_batches_seen": int(metrics.get("dynamic_probe_batches", 0)) > 0,
        "dynamic_j_interleaved": (
            metrics.get("psi_dispatch_dynamic_update")
            == "virtual_pending_and_idle_candidates"
        ),
        "station_context_defer_enabled": bool(
            metrics.get("station_context_defer_enabled", False)
        ),
        "station_context_defer_evaluated": int(
            metrics.get("station_context_defer_evaluations", 0)
        ) > 0,
        "context_conditioned": bool(
            metrics.get("station_context_defer_context_conditioned", False)
        ),
        "continues_other_stations": bool(
            metrics.get(
                "station_context_defer_continues_other_stations", False
            )
        ),
        "no_exact_eta": not bool(
            metrics.get("station_context_defer_exact_eta_used", True)
        ),
        "no_new_trainable_parameters": int(
            metrics.get("station_context_defer_new_trainable_parameters", -1)
        ) == 0,
        "no_global_no_assign": not bool(
            metrics.get("psi_dispatch_no_assign_added", True)
        ),
        "e_demand_unchanged": not bool(
            metrics.get("psi_dispatch_e_demand_modified", True)
        ),
        "committed_admission_v1": (
            admission.get("mode") == STATION_ADMISSION_COMMITTED_V1
        ),
        "committed_admission_invariant": bool(admission.get("passed")),
        "no_liveness_bound_violation": int(
            metrics.get(
                "station_context_defer_liveness_bound_violations", -1
            )
        ) == 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", choices=LOADS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT
    )
    parser.add_argument("--frozen-bundle", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--trace-max-records", type=int, default=500)
    args = parser.parse_args()
    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive")

    bundle_path = args.frozen_bundle or (
        args.source_root / "phase_c_psi_dispatch_frozen_protocol.json"
    )
    bundle, protocol = _load_bundle(bundle_path)
    model_checkpoint = _artifact(bundle, "model_checkpoint")
    psi_head_checkpoint = _artifact(bundle, "psi_head_checkpoint")
    psi_scale_contract = _artifact(bundle, "psi_scale_contract")
    config_path = _artifact(bundle, f"config_{args.load}")

    reference_path = _reference_path(
        args.reference_root, args.load, args.seed
    )
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    reference = _read_json(reference_path)
    reference_meta = reference.get("meta") or {}
    if int(reference_meta.get("ticks", -1)) != int(args.ticks):
        raise ValueError("reference tick budget differs from requested run")
    manifest_path = _manifest_from_reference(reference)
    manifest = _read_json(manifest_path)

    output_path = _output_path(args.output_root, args.load, args.seed)
    if output_path.is_file():
        existing = _read_json(output_path)
        meta = existing.get("meta") or {}
        if (
            existing.get("schema_version") == SCHEMA_VERSION
            and meta.get("load") == args.load
            and int(meta.get("seed", -1)) == int(args.seed)
            and int(meta.get("ticks", -1)) == int(args.ticks)
            and bool((existing.get("audit") or {}).get("passed"))
        ):
            print(f"[resume] {output_path}")
            return
        raise FileExistsError(f"incompatible existing output: {output_path}")

    assigner = StationContextDeferDynamicPsiAssigner(
        psi_head_checkpoint=str(psi_head_checkpoint),
        psi_scale_contract=str(psi_scale_contract),
        dynamic_trace_enabled=True,
        dynamic_trace_max_records=int(args.trace_max_records),
        dynamic_interleaved=True,
        checkpoint_path=str(model_checkpoint),
        top_m=TOP_M,
        energy_conv_random_flip_seed=0,
        **S1_CONFIG,
    )

    import WorldModel.evaluation.evaluate_online_v6 as eval_module

    original_builder = eval_module._build_engine
    holder: dict[str, Any] = {}

    def build_with_committed_admission(cfg, task_assigner=None):
        engine = original_builder(cfg, task_assigner=task_assigner)
        engine.world.station_state.set_admission_mode(
            STATION_ADMISSION_COMMITTED_V1
        )
        probe = StationAdmissionAuditProbe(engine)
        engine.on_tick_callbacks.append(probe.on_tick)
        holder["engine"] = engine
        holder["probe"] = probe
        return engine

    eval_module._build_engine = build_with_committed_admission
    try:
        print(
            f"[run] station-context defer load={args.load} "
            f"seed={args.seed} ticks={args.ticks}"
        )
        metrics = eval_module._run_one_assigner(
            str(config_path),
            assigner,
            int(args.seed),
            int(args.ticks),
            trace_label=ARM_LABEL,
            recorded_orders_path=str(manifest_path),
        )
    finally:
        eval_module._build_engine = original_builder

    probe = holder.get("probe")
    if probe is None:
        raise RuntimeError("station admission audit probe was not attached")
    admission = probe.summary()
    metrics.update(assigner.dynamic_probe_metrics())
    audit = _audit(metrics, manifest, admission, args.ticks)
    if not audit["passed"]:
        failed = [key for key, passed in audit["checks"].items() if not passed]
        raise RuntimeError(
            f"station-context defer audit failed {args.load} "
            f"seed={args.seed}: {failed}"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": str(protocol["protocol_sha256"]),
            "frozen_bundle": bundle_path.as_posix(),
            "frozen_bundle_sha256": sha256_file(bundle_path),
            "arm_key": ARM_KEY,
            "arm_label": ARM_LABEL,
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "development_only": True,
            "station_admission": STATION_ADMISSION_COMMITTED_V1,
            "fifo_v2_used": False,
            "selection_order": (
                "dynamic J -> station-conditioned context defer -> S1 -> "
                "committed admission V1"
            ),
            "model_checkpoint": model_checkpoint.as_posix(),
            "psi_head_checkpoint": psi_head_checkpoint.as_posix(),
            "psi_scale_contract": psi_scale_contract.as_posix(),
            "reference_result": reference_path.as_posix(),
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "reference": {
            "arm": "s1_psi_dynamic_committed_admission_v1",
            "metrics": _metric_subset(reference.get("metrics") or {}),
        },
        "audit": audit,
        "station_admission_audit": admission,
        "metrics": metrics,
        "dynamic_trace": list(assigner.dynamic_probe_trace_records),
    }
    _atomic_json(output_path, payload)
    print(f"[done] {output_path}")
    print(json.dumps({
        "completed_orders": metrics.get("completed_orders"),
        "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
        "stall_ratio_mean": metrics.get("stall_ratio_mean"),
        "context_defer_rate": metrics.get("station_context_defer_rate"),
        "all_remaining_deferred_batches": metrics.get(
            "station_context_defer_all_remaining_batches"
        ),
        "max_eligible_defer_ticks": metrics.get(
            "station_context_defer_eligible_ticks_max"
        ),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
