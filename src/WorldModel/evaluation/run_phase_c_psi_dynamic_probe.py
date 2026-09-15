"""Run one isolated dynamic-J online probe arm.

The probe replays an existing order-arrival manifest and writes to a new
output root.  It never overwrites the frozen static psi-dispatch arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_dynamic_probe_assigner import (
    DynamicPsiDispatchProbeAssigner,
)
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOADS,
    S1_CONFIG,
    TOP_M,
    canonical_sha256,
    sha256_file,
)


SCHEMA_VERSION = "phase_c_psi_dynamic_probe_arm_v1"
DYNAMIC_ARM = "s1_psi_dynamic_probe"
DYNAMIC_LABEL = "PhaseCS1PsiDynamicBacklogProbe"
BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_SOURCE_ROOT = BASE_ROOT / "psi_dispatch_ablation_551_560_v1"
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "psi_dispatch_dynamic_probe_551_560_v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite changed output: {path}")
        return
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_bundle(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = _read_json(path)
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"wrong frozen bundle: {path}")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("frozen bundle lacks protocol")
    protocol = dict(protocol)
    claimed = str(protocol.pop("protocol_sha256", ""))
    if canonical_sha256(protocol) != claimed:
        raise ValueError("frozen bundle protocol hash mismatch")
    if str(bundle.get("protocol_sha256", "")) != claimed:
        raise ValueError("frozen bundle/protocol hash mismatch")
    protocol["protocol_sha256"] = claimed
    return bundle, protocol


def _artifact(bundle: Mapping[str, Any], key: str) -> Path:
    value = (bundle.get("artifacts") or {}).get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"frozen bundle lacks artifact {key!r}")
    path = Path(str(value.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != str(value.get("sha256", "")):
        raise ValueError(f"frozen artifact changed: {path}")
    return path


def _output_path(root: Path, load: str, seed: int) -> Path:
    return root / "per_arm" / DYNAMIC_ARM / f"{load}_seed{seed}.json"


def _manifest_path(root: Path, load: str, seed: int) -> Path:
    return root / "order_manifests" / f"orders_{load}_seed{seed}.json"


def _audit(
    metrics: Mapping[str, Any],
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "manifest_exists": manifest_path.is_file(),
        "manifest_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest_payload.get("manifest_sha256")
        ),
        "manifest_count_matches": int(metrics.get("order_arrival_count", -1))
        == int(manifest_payload.get("total_orders", -2)),
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
        "s1_used": int(metrics.get("energy_conv_contexts", 0)) > 0,
        "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
        "dynamic_batches_seen": int(metrics.get("dynamic_probe_batches", 0)) > 0,
        "dynamic_choices_reindexed": bool(
            metrics.get("dynamic_probe_choices_reindexed", False)
        ),
        "dynamic_head_loaded": bool(metrics.get("psi_dispatch_head_loaded")),
        "parent_robot_scorer_preserved": (
            metrics.get("psi_dispatch_robot_scorer")
            == "WorldModelTaskAssigner.select_robots_unmodified"
        ),
        "no_assign_added": not bool(
            metrics.get("psi_dispatch_no_assign_added", True)
        ),
        "hard_gate_added": not bool(
            metrics.get("psi_dispatch_hard_gate_added", True)
        ),
        "e_demand_modified": not bool(
            metrics.get("psi_dispatch_e_demand_modified", True)
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _reference_metrics(source_root: Path, load: str, seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ("s1_psi_shadow", "s1_psi_dispatch"):
        path = source_root / "per_arm" / arm / f"{load}_seed{seed}.json"
        if path.is_file():
            payload = _read_json(path)
            metrics = payload.get("metrics") or {}
            result[arm] = {
                "path": path.as_posix(),
                "completed_orders": metrics.get("completed_orders"),
                "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
                "pending_order_count": metrics.get("pending_order_count"),
                "stall_ratio_mean": metrics.get("stall_ratio_mean"),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", choices=LOADS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--frozen-bundle", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--trace-max-records", type=int, default=500)
    args = parser.parse_args()

    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive")
    source_root = args.source_root
    bundle_path = args.frozen_bundle or (
        source_root / "phase_c_psi_dispatch_frozen_protocol.json"
    )
    bundle, protocol = _load_bundle(bundle_path)
    protocol_sha = str(protocol["protocol_sha256"])
    bundle_sha = sha256_file(bundle_path)
    model_checkpoint = _artifact(bundle, "model_checkpoint")
    psi_head_checkpoint = _artifact(bundle, "psi_head_checkpoint")
    psi_scale_contract = _artifact(bundle, "psi_scale_contract")
    config_path = _artifact(bundle, f"config_{args.load}")
    dynamic_code_path = Path(
        "Policies/TaskAssigner/WorldModelTaskAssigner/"
        "psi_dispatch_dynamic_probe_assigner.py"
    )
    dynamic_code_sha = sha256_file(dynamic_code_path)
    manifest_path = _manifest_path(source_root, args.load, args.seed)
    output_path = _output_path(args.output_root, args.load, args.seed)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing source manifest (run Greedy source first): {manifest_path}"
        )
    if output_path.is_file():
        existing = _read_json(output_path)
        if (
            existing.get("schema_version") != SCHEMA_VERSION
            or existing.get("meta", {}).get("protocol_sha256") != protocol_sha
            or int(existing.get("meta", {}).get("seed", -1)) != args.seed
            or existing.get("meta", {}).get("load") != args.load
            or int(existing.get("meta", {}).get("ticks", -1)) != args.ticks
        ):
            raise FileExistsError(f"incompatible existing output: {output_path}")
        print(f"[resume] {output_path}")
        return

    common = {
        "checkpoint_path": str(model_checkpoint),
        "top_m": TOP_M,
        "energy_conv_random_flip_seed": 0,
        **S1_CONFIG,
    }
    assigner = DynamicPsiDispatchProbeAssigner(
        psi_head_checkpoint=str(psi_head_checkpoint),
        psi_scale_contract=str(psi_scale_contract),
        dynamic_trace_enabled=True,
        dynamic_trace_max_records=args.trace_max_records,
        **common,
    )
    print(
        f"[run] dynamic probe load={args.load} seed={args.seed} "
        f"ticks={args.ticks}"
    )
    metrics = _run_one_assigner(
        str(config_path),
        assigner,
        int(args.seed),
        int(args.ticks),
        trace_label=DYNAMIC_LABEL,
        recorded_orders_path=str(manifest_path),
    )
    metrics.update(assigner.dynamic_probe_metrics())
    manifest_payload = _read_json(manifest_path)
    audit = _audit(metrics, manifest_path, manifest_payload)
    if not audit["passed"]:
        failed = [
            key for key, passed in audit["checks"].items() if not passed
        ]
        raise RuntimeError(
            f"dynamic probe audit failed {args.load} seed={args.seed}: {failed}"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": protocol_sha,
            "frozen_bundle": bundle_path.as_posix(),
            "frozen_bundle_sha256": bundle_sha,
            "arm_key": DYNAMIC_ARM,
            "arm_label": DYNAMIC_LABEL,
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "formal_development": False,
            "dynamic_update_mode": (
                "virtual_pending_and_idle_candidates_interleaved"
            ),
            "model_checkpoint": model_checkpoint.as_posix(),
            "psi_head_checkpoint": psi_head_checkpoint.as_posix(),
            "psi_scale_contract": psi_scale_contract.as_posix(),
            "source_manifest_root": source_root.as_posix(),
            "dynamic_probe_code": dynamic_code_path.as_posix(),
            "dynamic_probe_code_sha256": dynamic_code_sha,
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest_payload.get("manifest_sha256"),
            "total_orders": manifest_payload.get("total_orders"),
        },
        "reference_metrics": _reference_metrics(
            source_root, args.load, args.seed
        ),
        "audit": audit,
        "metrics": metrics,
        "dynamic_probe_trace": assigner.dynamic_probe_trace_records,
    }
    _atomic_json(output_path, payload)
    print(f"[done] {output_path}")
    print(json.dumps({
        "completed_orders": metrics.get("completed_orders"),
        "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
        "dynamic_probe_batches": metrics.get("dynamic_probe_batches"),
        "dynamic_probe_steps": metrics.get("dynamic_probe_steps"),
        "dynamic_probe_order_changed_batches": metrics.get(
            "dynamic_probe_order_changed_batches"
        ),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
