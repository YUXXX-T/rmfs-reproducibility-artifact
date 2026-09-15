"""Run one resumable arm of the 551--560 psi-dispatch ablation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from Policies.TaskAssigner import GreedyTaskAssigner, WorldModelTaskAssigner
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_context_assigner import (
    PsiDispatchContextWorldModelTaskAssigner,
)
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    ARM_KEYS,
    ARM_LABELS,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    GREEDY_ARM,
    LOADS,
    OUTPUT_ROOT,
    PER_ARM_SCHEMA_VERSION,
    PSI_ARM,
    PSI_HEAD_CHECKPOINT,
    PSI_SCALE_CONTRACT,
    PSI_SHADOW_ARM,
    S1_ARM,
    SCHEMA_VERSION,
    SEEDS,
    TICKS,
    TOP_M,
    canonical_sha256,
    run_id,
    sha256_file,
)
from WorldModel.evaluation.phase_c_s1_hungarian_protocol import S1_CONFIG


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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
        raise ValueError(f"wrong psi-dispatch frozen bundle: {path}")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("psi-dispatch bundle lacks protocol")
    protocol = dict(protocol)
    claimed = str(protocol.pop("protocol_sha256", ""))
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong psi-dispatch protocol schema")
    if canonical_sha256(protocol) != claimed:
        raise ValueError("psi-dispatch protocol hash mismatch")
    if str(bundle.get("protocol_sha256", "")) != claimed:
        raise ValueError("bundle/protocol hash mismatch")
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


def _paths(root: Path, arm: str, load: str, seed: int) -> tuple[Path, Path]:
    manifest = root / "order_manifests" / f"orders_{load}_seed{seed}.json"
    output = root / "per_arm" / arm / f"{load}_seed{seed}.json"
    return manifest, output


def _resume(
    path: Path,
    *,
    protocol_sha256: str,
    bundle_sha256: str,
    arm: str,
    load: str,
    seed: int,
    ticks: int,
) -> bool:
    if not path.is_file():
        return False
    payload = _read_json(path)
    meta = payload.get("meta") or {}
    checks = {
        "schema": payload.get("schema_version") == PER_ARM_SCHEMA_VERSION,
        "protocol": meta.get("protocol_sha256") == protocol_sha256,
        "bundle": meta.get("frozen_bundle_sha256") == bundle_sha256,
        "arm": meta.get("arm_key") == arm,
        "load": meta.get("load") == load,
        "seed": int(meta.get("seed", -1)) == int(seed),
        "ticks": int(meta.get("ticks", -1)) == int(ticks),
        "audit": bool((payload.get("audit") or {}).get("passed")),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"cannot resume incompatible output {path}: {failed}")
    print(f"[resume] {arm} {load} seed={seed}: {path}")
    return True


def _make_assigner(
    arm: str,
    *,
    model_checkpoint: Path,
    psi_head_checkpoint: Path,
    psi_scale_contract: Path,
):
    if arm == GREEDY_ARM:
        return GreedyTaskAssigner()
    common = {
        "checkpoint_path": str(model_checkpoint),
        "top_m": TOP_M,
        "energy_conv_random_flip_seed": 0,
        **S1_CONFIG,
    }
    if arm == S1_ARM:
        return WorldModelTaskAssigner(**common)
    if arm in (PSI_SHADOW_ARM, PSI_ARM):
        mode = "shadow" if arm == PSI_SHADOW_ARM else "j_ascending"
        return PsiDispatchContextWorldModelTaskAssigner(
            psi_head_checkpoint=str(psi_head_checkpoint),
            psi_scale_contract=str(psi_scale_contract),
            psi_context_mode=mode,
            psi_trace_enabled=True,
            psi_trace_max_records=TICKS,
            **common,
        )
    raise ValueError(arm)


def _audit(
    arm: str,
    metrics: Mapping[str, Any],
    *,
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
    }
    if arm == GREEDY_ARM:
        checks["manifest_is_source"] = not bool(
            metrics.get("order_arrival_replayed")
        )
    else:
        checks.update({
            "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
            "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
            "s1_used": int(metrics.get("energy_conv_contexts", 0)) > 0,
            "no_greedy_fallback": int(
                metrics.get("fallback_greedy_calls", 0)
            ) == 0,
            "native_no_assign_disabled": not bool(
                metrics.get("native_no_assign_enabled", False)
            ),
        })
    if arm in (PSI_SHADOW_ARM, PSI_ARM):
        checks.update({
            "psi_head_loaded": bool(
                metrics.get("psi_dispatch_head_loaded", False)
            ),
            "psi_evaluated": int(
                metrics.get("psi_dispatch_eval_calls", 0)
            ) > 0,
            "j_contexts_seen": int(
                metrics.get("psi_dispatch_contexts_seen", 0)
            ) > 0,
            "service_debt_finite": all(
                isinstance(metrics.get(name), (int, float))
                and float(metrics.get(name)) == float(metrics.get(name))
                for name in (
                    "psi_dispatch_service_debt_mean",
                    "psi_dispatch_j_mean",
                )
            ),
            "parent_robot_scorer_preserved": metrics.get(
                "psi_dispatch_robot_scorer"
            ) == "WorldModelTaskAssigner.select_robots_unmodified",
            "no_hard_gate": not bool(
                metrics.get("psi_dispatch_hard_gate_added", True)
            ),
            "no_assign_not_added": not bool(
                metrics.get("psi_dispatch_no_assign_added", True)
            ),
            "e_demand_unmodified": not bool(
                metrics.get("psi_dispatch_e_demand_modified", True)
            ),
        })
        if arm == PSI_ARM:
            checks["execution_order_applied"] = (
                metrics.get("psi_dispatch_mode") == "j_ascending"
            )
        else:
            checks["execution_order_shadow"] = (
                metrics.get("psi_dispatch_mode") == "shadow"
            )
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARM_KEYS, required=True)
    parser.add_argument("--load", choices=LOADS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frozen-bundle", required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument(
        "--development",
        action="store_true",
        help="Allow a smaller tick count for a mechanism smoke.",
    )
    args = parser.parse_args()

    if int(args.ticks) <= 0:
        raise SystemExit("ticks must be positive")
    if int(args.seed) not in SEEDS:
        raise SystemExit(f"seed must be one of {list(SEEDS)}")
    if not args.development and int(args.ticks) != TICKS:
        raise SystemExit(f"formal development run freezes --ticks={TICKS}")

    bundle_path = Path(args.frozen_bundle)
    bundle, protocol = _load_bundle(bundle_path)
    protocol_sha = str(protocol["protocol_sha256"])
    bundle_sha = sha256_file(bundle_path)
    model_checkpoint = _artifact(bundle, "model_checkpoint")
    psi_head_checkpoint = _artifact(bundle, "psi_head_checkpoint")
    psi_scale_contract = _artifact(bundle, "psi_scale_contract")
    config_path = _artifact(bundle, f"config_{args.load}")

    root = Path(args.output_root)
    manifest_path, output_path = _paths(
        root, args.arm, args.load, int(args.seed)
    )
    if _resume(
        output_path,
        protocol_sha256=protocol_sha,
        bundle_sha256=bundle_sha,
        arm=args.arm,
        load=args.load,
        seed=int(args.seed),
        ticks=int(args.ticks),
    ):
        return
    if args.arm != GREEDY_ARM and not manifest_path.is_file():
        raise FileNotFoundError(
            f"Greedy manifest must finish first: {manifest_path}"
        )

    assigner = _make_assigner(
        args.arm,
        model_checkpoint=model_checkpoint,
        psi_head_checkpoint=psi_head_checkpoint,
        psi_scale_contract=psi_scale_contract,
    )
    run_kwargs: dict[str, Any] = {}
    if args.arm == GREEDY_ARM:
        if manifest_path.exists() and not output_path.exists():
            raise FileExistsError(
                "manifest exists without a resumable Greedy output: "
                f"{manifest_path}"
            )
        run_kwargs["save_order_manifest"] = str(manifest_path)
    else:
        run_kwargs["recorded_orders_path"] = str(manifest_path)

    print(
        f"[run] arm={args.arm} load={args.load} seed={args.seed} "
        f"ticks={args.ticks}"
    )
    metrics = _run_one_assigner(
        str(config_path),
        assigner,
        int(args.seed),
        int(args.ticks),
        trace_label=ARM_LABELS[args.arm],
        **run_kwargs,
    )
    if isinstance(assigner, PsiDispatchContextWorldModelTaskAssigner):
        metrics.update(assigner.psi_dispatch_metrics())

    manifest_payload = _read_json(manifest_path)
    audit = _audit(
        args.arm,
        metrics,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
    )
    if not audit["passed"]:
        failed = [key for key, passed in audit["checks"].items() if not passed]
        raise RuntimeError(
            f"psi-dispatch arm audit failed {args.arm} {args.load} "
            f"seed={args.seed}: {failed}"
        )

    payload = {
        "schema_version": PER_ARM_SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": protocol_sha,
            "frozen_bundle": bundle_path.as_posix(),
            "frozen_bundle_sha256": bundle_sha,
            "arm_key": args.arm,
            "arm_label": ARM_LABELS[args.arm],
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "formal_development": not bool(args.development),
            "config": config_path.as_posix(),
            "model_checkpoint": (
                model_checkpoint.as_posix()
                if args.arm != GREEDY_ARM else None
            ),
            "psi_head_checkpoint": (
                psi_head_checkpoint.as_posix()
                if args.arm in (PSI_SHADOW_ARM, PSI_ARM) else None
            ),
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest_payload.get("manifest_sha256"),
            "total_orders": manifest_payload.get("total_orders"),
        },
        "audit": audit,
        "metrics": metrics,
        "psi_dispatch_trace": (
            assigner.psi_dispatch_trace_records
            if isinstance(assigner, PsiDispatchContextWorldModelTaskAssigner)
            else []
        ),
    }
    _atomic_json(output_path, payload)
    print(f"[done] {run_id(args.arm, args.load, args.seed)}: {output_path}")


if __name__ == "__main__":
    main()
