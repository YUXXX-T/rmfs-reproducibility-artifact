"""Run the S1+FIFO side of the paired Dynamic-J validation.

This is the only missing arm in the existing codebase: Dynamic-J+FIFO already
uses ``run_phase_c_psi_dynamic_admission_wait.py``.  The ``manifest`` mode is
used only for the fresh 561--570 block before either paired arm starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from Policies.TaskAssigner import GreedyTaskAssigner, WorldModelTaskAssigner
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    LOAD_CONFIGS,
    LOADS,
    S1_CONFIG,
    TOP_M,
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_psi_dynamic_probe import (
    _artifact,
    _atomic_json,
    _load_bundle,
    _read_json,
)
from WorldModel.evaluation.run_phase_c_psi_dynamic_admission_wait import (
    FifoWaitingAuditProbe,
)
from WorldState.station_state import STATION_ADMISSION_COMMITTED_FIFO_V2


SCHEMA_VERSION = "phase_c_s1_fifo_pair_arm_v1"
ARM_KEY = "s1_fifo_v2"
ARM_LABEL = "PhaseCS1FifoV2"


def _manifest_path(root: Path, load: str, seed: int) -> Path:
    return root / "order_manifests" / f"orders_{load}_seed{seed}.json"


def _output_path(root: Path, load: str, seed: int) -> Path:
    return root / "per_arm" / ARM_KEY / f"{load}_seed{seed}.json"


def _validate_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    rows = payload.get("orders")
    if (
        payload.get("schema_version") != "layer5_order_arrival_manifest_v1"
        or not isinstance(rows, list)
        or int(payload.get("total_orders", -1)) != len(rows)
    ):
        raise ValueError(f"invalid order manifest: {path}")
    canonical = {
        "schema_version": payload["schema_version"],
        "orders": rows,
    }
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != payload.get("manifest_sha256"):
        raise ValueError(f"manifest hash mismatch: {path}")
    return payload


@contextmanager
def _fifo_engine(trace_max_records: int) -> Iterator[dict[str, Any]]:
    import WorldModel.evaluation.evaluate_online_v6 as eval_module

    original_builder = eval_module._build_engine
    holder: dict[str, Any] = {}

    def build_fifo(cfg, task_assigner=None):
        engine = original_builder(cfg, task_assigner=task_assigner)
        engine.world.station_state.set_admission_mode(
            STATION_ADMISSION_COMMITTED_FIFO_V2
        )
        probe = FifoWaitingAuditProbe(
            engine, trace_max_records=int(trace_max_records)
        )
        engine.on_tick_callbacks.append(probe.on_tick)
        holder["probe"] = probe
        return engine

    eval_module._build_engine = build_fifo
    try:
        yield holder
    finally:
        eval_module._build_engine = original_builder


def _prepare_manifest(args: argparse.Namespace) -> None:
    path = _manifest_path(args.source_root, args.load, args.seed)
    if path.is_file():
        _validate_manifest(path)
        print(f"[resume] valid manifest {path}")
        return

    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    path.parent.mkdir(parents=True, exist_ok=True)
    with _fifo_engine(trace_max_records=0) as holder:
        _run_one_assigner(
            str(LOAD_CONFIGS[args.load]),
            GreedyTaskAssigner(),
            int(args.seed),
            int(args.ticks),
            trace_label="GreedyFifoManifestSource",
            save_order_manifest=str(path),
        )
    probe = holder.get("probe")
    if probe is None or not bool(probe.summary().get("passed")):
        raise RuntimeError(f"FIFO manifest-source audit failed: {path}")
    _validate_manifest(path)
    print(f"[done] manifest {path}")


def _audit(
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "manifest_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_count_matches": int(metrics.get("order_arrival_count", -1))
        == int(manifest.get("total_orders", -2)),
        "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
        "s1_used": int(metrics.get("energy_conv_contexts", 0)) > 0,
        "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
        "native_no_assign_disabled": not bool(
            metrics.get("native_no_assign_enabled", False)
        ),
        "fifo_admission_invariant": bool(admission.get("passed")),
        "fifo_mode": (
            admission.get("mode") == STATION_ADMISSION_COMMITTED_FIFO_V2
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _run_s1(args: argparse.Namespace) -> None:
    bundle_path = args.frozen_bundle or (
        args.source_root / "phase_c_psi_dispatch_frozen_protocol.json"
    )
    bundle, protocol = _load_bundle(bundle_path)
    model_checkpoint = _artifact(bundle, "model_checkpoint")
    config_path = _artifact(bundle, f"config_{args.load}")
    manifest_path = _manifest_path(args.source_root, args.load, args.seed)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _validate_manifest(manifest_path)

    output_root = args.output_root or args.source_root
    output = _output_path(output_root, args.load, args.seed)
    runner_hash = sha256_file(Path(__file__))
    bundle_hash = sha256_file(bundle_path)
    if output.is_file():
        old = _read_json(output)
        meta = old.get("meta") or {}
        if (
            old.get("schema_version") != SCHEMA_VERSION
            or meta.get("runner_sha256") != runner_hash
            or meta.get("frozen_bundle_sha256") != bundle_hash
            or meta.get("load") != args.load
            or int(meta.get("seed", -1)) != int(args.seed)
            or int(meta.get("ticks", -1)) != int(args.ticks)
            or not bool((old.get("audit") or {}).get("passed"))
        ):
            raise FileExistsError(f"incompatible existing output: {output}")
        print(f"[resume] {output}")
        return

    assigner = WorldModelTaskAssigner(
        checkpoint_path=str(model_checkpoint),
        top_m=TOP_M,
        energy_conv_random_flip_seed=0,
        **S1_CONFIG,
    )
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    with _fifo_engine(args.trace_max_records) as holder:
        metrics = _run_one_assigner(
            str(config_path),
            assigner,
            int(args.seed),
            int(args.ticks),
            trace_label=ARM_LABEL,
            recorded_orders_path=str(manifest_path),
        )
    probe = holder.get("probe")
    if probe is None:
        raise RuntimeError("FIFO audit probe was not attached")
    admission = probe.summary()
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
    audit = _audit(metrics, manifest, admission)
    if not audit["passed"]:
        failed = [key for key, value in audit["checks"].items() if not value]
        raise RuntimeError(
            f"S1 FIFO audit failed {args.load} seed={args.seed}: {failed}"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "arm_key": ARM_KEY,
            "arm_label": ARM_LABEL,
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "station_admission": STATION_ADMISSION_COMMITTED_FIFO_V2,
            "source_protocol_sha256": protocol.get("protocol_sha256"),
            "frozen_bundle": bundle_path.as_posix(),
            "frozen_bundle_sha256": bundle_hash,
            "model_checkpoint": model_checkpoint.as_posix(),
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
    }
    _atomic_json(output, payload)
    print(f"[done] {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("manifest", "s1"), required=True)
    parser.add_argument("--load", choices=LOADS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--frozen-bundle", type=Path, default=None)
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--trace-max-records", type=int, default=500)
    args = parser.parse_args()
    if args.seed <= 0 or args.ticks <= 0:
        raise SystemExit("seed and ticks must be positive")
    if args.mode == "manifest":
        _prepare_manifest(args)
    else:
        _run_s1(args)


if __name__ == "__main__":
    main()
