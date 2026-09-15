"""Run one missing FIFO-V2 arm for the six-arm Phase-C comparison.

Existing S1+FIFO and Dynamic-J+FIFO outputs are not touched.  This runner only
adds Greedy, Hungarian, pure Phase-C, or Static-J under the same FIFO station
admission and the same recorded order manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from Policies.TaskAssigner import (
    GreedyTaskAssigner,
    HungarianTaskAssigner,
    WorldModelTaskAssigner,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_context_assigner import (
    PsiDispatchContextWorldModelTaskAssigner,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    LOADS,
    S1_CONFIG,
    TOP_M,
    sha256_file,
)
from WorldModel.evaluation.phase_c_s1_hungarian_protocol import PHASEC_CONFIG
from WorldModel.evaluation.run_phase_c_psi_dynamic_probe import (
    _artifact,
    _atomic_json,
    _load_bundle,
    _read_json,
)
from WorldModel.evaluation.run_phase_c_s1_fifo_pair import (
    _fifo_engine,
    _manifest_path,
    _validate_manifest,
)
from WorldState.station_state import STATION_ADMISSION_COMMITTED_FIFO_V2


SCHEMA_VERSION = "phase_c_fifo_missing_arm_v1"
ARM_LABELS = {
    "greedy": "GreedyFifoV2",
    "hungarian": "HungarianFifoV2",
    "phasec": "PhaseCPureFifoV2",
    "static_j": "PhaseCStaticJFifoV2",
}
ARM_KEYS = {
    "greedy": "greedy_fifo_v2",
    "hungarian": "hungarian_fifo_v2",
    "phasec": "phasec_fifo_v2",
    "static_j": "static_j_fifo_v2",
}


def _output_path(root: Path, arm: str, load: str, seed: int) -> Path:
    return root / "per_arm" / ARM_KEYS[arm] / f"{load}_seed{seed}.json"


def _make_assigner(
    arm: str,
    *,
    model_checkpoint: Path,
    psi_head_checkpoint: Path,
    psi_scale_contract: Path,
    trace_max_records: int,
):
    if arm == "greedy":
        return GreedyTaskAssigner()
    if arm == "hungarian":
        return HungarianTaskAssigner()
    common = {
        "checkpoint_path": str(model_checkpoint),
        "top_m": TOP_M,
        "energy_conv_random_flip_seed": 0,
    }
    if arm == "phasec":
        return WorldModelTaskAssigner(**common, **PHASEC_CONFIG)
    if arm == "static_j":
        return PsiDispatchContextWorldModelTaskAssigner(
            psi_head_checkpoint=str(psi_head_checkpoint),
            psi_scale_contract=str(psi_scale_contract),
            psi_context_mode="j_ascending",
            psi_trace_enabled=True,
            psi_trace_max_records=int(trace_max_records),
            **common,
            **S1_CONFIG,
        )
    raise ValueError(arm)


def _audit(
    arm: str,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "manifest_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_count_matches": int(metrics.get("order_arrival_count", -1))
        == int(manifest.get("total_orders", -2)),
        "fifo_admission_invariant": bool(admission.get("passed")),
        "fifo_mode": (
            admission.get("mode") == STATION_ADMISSION_COMMITTED_FIFO_V2
        ),
    }
    if arm in {"phasec", "static_j"}:
        checks.update({
            "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
            "no_greedy_fallback": int(
                metrics.get("fallback_greedy_calls", 0)
            ) == 0,
            "native_no_assign_disabled": not bool(
                metrics.get("native_no_assign_enabled", False)
            ),
        })
    if arm == "phasec":
        checks["s1_disabled"] = int(
            metrics.get("energy_conv_contexts", 0)
        ) == 0
    if arm == "static_j":
        checks.update({
            "s1_used": int(metrics.get("energy_conv_contexts", 0)) > 0,
            "psi_head_loaded": bool(
                metrics.get("psi_dispatch_head_loaded", False)
            ),
            "psi_evaluated": int(
                metrics.get("psi_dispatch_eval_calls", 0)
            ) > 0,
            "static_j_mode": metrics.get("psi_dispatch_mode") == "j_ascending",
            "no_assign_not_added": not bool(
                metrics.get("psi_dispatch_no_assign_added", True)
            ),
            "hard_gate_not_added": not bool(
                metrics.get("psi_dispatch_hard_gate_added", True)
            ),
        })
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARM_LABELS), required=True)
    parser.add_argument("--load", choices=LOADS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-bundle", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--trace-max-records", type=int, default=500)
    args = parser.parse_args()
    if args.seed <= 0 or args.ticks <= 0:
        raise SystemExit("seed and ticks must be positive")

    bundle, protocol = _load_bundle(args.frozen_bundle)
    config_path = _artifact(bundle, f"config_{args.load}")
    model_checkpoint = _artifact(bundle, "model_checkpoint")
    psi_head_checkpoint = _artifact(bundle, "psi_head_checkpoint")
    psi_scale_contract = _artifact(bundle, "psi_scale_contract")
    manifest_path = _manifest_path(args.source_root, args.load, args.seed)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _validate_manifest(manifest_path)

    output = _output_path(args.output_root, args.arm, args.load, args.seed)
    runner_hash = sha256_file(Path(__file__))
    bundle_hash = sha256_file(args.frozen_bundle)
    if output.is_file():
        old = _read_json(output)
        meta = old.get("meta") or {}
        if (
            old.get("schema_version") != SCHEMA_VERSION
            or meta.get("runner_sha256") != runner_hash
            or meta.get("frozen_bundle_sha256") != bundle_hash
            or meta.get("arm_key") != ARM_KEYS[args.arm]
            or meta.get("load") != args.load
            or int(meta.get("seed", -1)) != int(args.seed)
            or int(meta.get("ticks", -1)) != int(args.ticks)
            or not bool((old.get("audit") or {}).get("passed"))
        ):
            raise FileExistsError(f"incompatible existing output: {output}")
        print(f"[resume] {output}")
        return

    assigner = _make_assigner(
        args.arm,
        model_checkpoint=model_checkpoint,
        psi_head_checkpoint=psi_head_checkpoint,
        psi_scale_contract=psi_scale_contract,
        trace_max_records=args.trace_max_records,
    )
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    with _fifo_engine(args.trace_max_records) as holder:
        metrics = _run_one_assigner(
            str(config_path),
            assigner,
            int(args.seed),
            int(args.ticks),
            trace_label=ARM_LABELS[args.arm],
            recorded_orders_path=str(manifest_path),
        )
    probe = holder.get("probe")
    if probe is None:
        raise RuntimeError("FIFO audit probe was not attached")
    admission = probe.summary()
    if isinstance(assigner, PsiDispatchContextWorldModelTaskAssigner):
        metrics.update(assigner.psi_dispatch_metrics())
    metrics.update({
        "station_admission_mode": STATION_ADMISSION_COMMITTED_FIFO_V2,
        "waiting_assigned_agent_ticks": admission.get("waiting_agent_ticks"),
        "waiting_assigned_ratio": admission.get("waiting_assigned_ratio"),
        "waiting_promotions": admission.get("waiting_promotions"),
        "waiting_duration_p95_ticks": admission.get(
            "waiting_duration_p95_ticks"
        ),
        "unresolved_waiter_count_final": admission.get(
            "unresolved_waiter_count_final"
        ),
    })
    audit = _audit(args.arm, metrics, manifest, admission)
    if not audit["passed"]:
        failed = [key for key, value in audit["checks"].items() if not value]
        raise RuntimeError(
            f"{args.arm} FIFO audit failed {args.load} seed={args.seed}: {failed}"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "arm_key": ARM_KEYS[args.arm],
            "arm_label": ARM_LABELS[args.arm],
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "station_admission": STATION_ADMISSION_COMMITTED_FIFO_V2,
            "source_protocol_sha256": protocol.get("protocol_sha256"),
            "frozen_bundle": args.frozen_bundle.as_posix(),
            "frozen_bundle_sha256": bundle_hash,
            "runner": Path(__file__).as_posix(),
            "runner_sha256": runner_hash,
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "audit": audit,
        "admission_audit": admission,
        "metrics": metrics,
        "psi_dispatch_trace": (
            list(assigner.psi_dispatch_trace_records)
            if isinstance(assigner, PsiDispatchContextWorldModelTaskAssigner)
            else []
        ),
    }
    _atomic_json(output, payload)
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
