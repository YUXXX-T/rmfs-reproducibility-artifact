"""Run one frozen arm for one Phase-C/S1/Hungarian load/seed pair.

The file is intentionally arm-granular so a CPU Slurm array can keep 64--128
cores busy.  Greedy first writes the realised order manifest.  Every other arm
replays that exact manifest and writes one resumable JSON artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from Policies.TaskAssigner import (
    GreedyTaskAssigner,
    HungarianTaskAssigner,
    WorldModelTaskAssigner,
)
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
from WorldModel.evaluation.phase_c_s1_hungarian_protocol import (
    ARMS,
    CANDIDATE_CHECKPOINT,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    GREEDY_LABEL,
    HUNGARIAN_LABEL,
    LOAD_CONFIGS,
    LOADS,
    OUTPUT_ROOT,
    PER_SEED_SCHEMA_VERSION,
    PHASEC_CONFIG,
    PHASEC_LABEL,
    PHASEC_S1_LABEL,
    S1_CONFIG,
    SEEDS,
    TICKS,
    TOP_M,
    canonical_sha256,
    sha256_file,
)


ARM_KEYS = {
    "greedy": GREEDY_LABEL,
    "hungarian": HUNGARIAN_LABEL,
    "phasec": PHASEC_LABEL,
    "phasec_s1": PHASEC_S1_LABEL,
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing != encoded:
            raise FileExistsError(f"refusing to overwrite changed output: {path}")
        return
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _artifact(bundle: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    artifacts = bundle.get("artifacts") or {}
    value = artifacts.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"frozen bundle lacks artifact {key!r}")
    path = Path(str(value.get("path", "")))
    expected = str(value.get("sha256", ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"frozen artifact changed: {path}")
    return value


def _load_bundle(path: Path) -> dict[str, Any]:
    bundle = _read_json(path)
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"wrong frozen bundle schema: {path}")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("frozen bundle lacks protocol")
    expected = str(bundle.get("protocol_sha256", ""))
    if canonical_sha256(protocol) != expected:
        raise ValueError("frozen protocol hash mismatch")
    _artifact(bundle, "candidate_checkpoint")
    return bundle


def _paths(root: Path, load: str, seed: int, arm_key: str) -> tuple[Path, Path]:
    manifest = root / "order_manifests" / f"orders_{load}_seed{seed}.json"
    output = root / "per_arm" / arm_key / f"{load}_seed{seed}.json"
    return manifest, output


def _resume(
    path: Path,
    *,
    protocol_sha256: str,
    load: str,
    seed: int,
    arm_label: str,
    ticks: int,
    bundle_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    payload = _read_json(path)
    meta = payload.get("meta") or {}
    checks = {
        "schema": payload.get("schema_version") == PER_SEED_SCHEMA_VERSION,
        "protocol": meta.get("protocol_sha256") == protocol_sha256,
        "bundle": meta.get("frozen_bundle_sha256") == bundle_sha256,
        "load": meta.get("load") == load,
        "seed": int(meta.get("seed", -1)) == int(seed),
        "arm": meta.get("arm") == arm_label,
        "ticks": int(meta.get("ticks", -1)) == int(ticks),
        "audit": bool((payload.get("audit") or {}).get("passed")),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"cannot resume incompatible {path}: {failed}")
    print(f"[resume] {arm_label} load={load} seed={seed}: {path}")
    return True


def _make_assigner(arm_key: str):
    if arm_key == "greedy":
        return GreedyTaskAssigner()
    if arm_key == "hungarian":
        return HungarianTaskAssigner()
    if arm_key == "phasec":
        return WorldModelTaskAssigner(
            checkpoint_path=CANDIDATE_CHECKPOINT,
            top_m=TOP_M,
            **PHASEC_CONFIG,
        )
    if arm_key == "phasec_s1":
        return WorldModelTaskAssigner(
            checkpoint_path=CANDIDATE_CHECKPOINT,
            top_m=TOP_M,
            energy_conv_random_flip_seed=0,
            **S1_CONFIG,
        )
    raise ValueError(arm_key)


def _audit_metrics(
    arm_key: str,
    metrics: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "manifest_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest_payload.get("manifest_sha256")
        ),
        "manifest_count_matches": int(metrics.get("order_arrival_count", -1))
        == int(manifest_payload.get("total_orders", -2)),
        "manifest_file_exists": manifest_path.is_file(),
    }
    if arm_key == "greedy":
        checks["manifest_is_source"] = not bool(
            metrics.get("order_arrival_replayed")
        )
    else:
        checks["manifest_was_replayed"] = bool(
            metrics.get("order_arrival_replayed")
        )
    if arm_key in ("phasec", "phasec_s1"):
        checks.update({
            "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
            "no_greedy_fallback": int(
                metrics.get("fallback_greedy_calls", 0)
            ) == 0,
            "native_no_assign_disabled": not bool(
                metrics.get("native_no_assign_enabled", False)
            ),
        })
    if arm_key == "phasec":
        checks["s1_disabled"] = int(
            metrics.get("energy_conv_contexts", 0)
        ) == 0
    if arm_key == "phasec_s1":
        checks.update({
            "s1_contexts_observed": int(
                metrics.get("energy_conv_contexts", 0)
            ) > 0,
            "s1_no_random_flip": float(
                S1_CONFIG["energy_conv_random_flip_rate"]
            ) == 0.0,
        })
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(ARM_KEYS), required=True)
    parser.add_argument("--load", choices=LOADS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frozen-bundle", required=True)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument(
        "--development",
        action="store_true",
        help="Allow a non-formal seed/tick count for mechanism smoke only.",
    )
    args = parser.parse_args()

    if int(args.ticks) <= 0:
        raise SystemExit("ticks must be positive")
    if not args.development:
        if int(args.seed) not in SEEDS:
            raise SystemExit(f"formal seed must be one of {list(SEEDS)}")
        if int(args.ticks) != TICKS:
            raise SystemExit(f"formal comparison freezes --ticks={TICKS}")

    bundle_path = Path(args.frozen_bundle)
    bundle = _load_bundle(bundle_path)
    protocol_sha256 = str(bundle["protocol_sha256"])
    bundle_sha256 = sha256_file(bundle_path)
    arm_label = ARM_KEYS[args.arm]
    if arm_label not in ARMS:
        raise ValueError(f"arm not frozen: {arm_label}")

    config_artifact = _artifact(bundle, f"config_{args.load}")
    config_path = Path(str(config_artifact["path"]))
    root = Path(args.output_root)
    manifest_path, output_path = _paths(root, args.load, args.seed, args.arm)
    if _resume(
        output_path,
        protocol_sha256=protocol_sha256,
        load=args.load,
        seed=args.seed,
        arm_label=arm_label,
        ticks=int(args.ticks),
        bundle_sha256=bundle_sha256,
    ):
        return

    if args.arm != "greedy" and not manifest_path.is_file():
        raise FileNotFoundError(
            f"Greedy manifest must finish before {arm_label}: {manifest_path}"
        )

    assigner = _make_assigner(args.arm)
    print(
        f"[run] arm={arm_label} load={args.load} seed={args.seed} "
        f"ticks={args.ticks} formal={not args.development}"
    )
    run_kwargs: dict[str, Any] = {}
    if args.arm == "greedy":
        run_kwargs["save_order_manifest"] = str(manifest_path)
    else:
        run_kwargs["recorded_orders_path"] = str(manifest_path)
    metrics = _run_one_assigner(
        str(config_path),
        assigner,
        int(args.seed),
        int(args.ticks),
        trace_label=arm_label,
        **run_kwargs,
    )
    manifest_payload = _read_json(manifest_path)
    audit = _audit_metrics(
        args.arm,
        metrics,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
    )
    if not audit["passed"]:
        failed = [
            key for key, passed in audit["checks"].items() if not passed
        ]
        raise RuntimeError(
            f"arm audit failed for {arm_label} {args.load} seed={args.seed}: "
            + ", ".join(failed)
        )

    payload = {
        "schema_version": PER_SEED_SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": protocol_sha256,
            "frozen_bundle": bundle_path.as_posix(),
            "frozen_bundle_sha256": bundle_sha256,
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "formal": not args.development,
            "top_m": TOP_M,
            "arm_key": args.arm,
            "arm": arm_label,
            "config": config_path.as_posix(),
            "checkpoint": (
                CANDIDATE_CHECKPOINT
                if args.arm in ("phasec", "phasec_s1") else None
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
    }
    _write_json_atomic(output_path, payload)
    print(f"[done] {arm_label} load={args.load} seed={args.seed}: {output_path}")


if __name__ == "__main__":
    main()
