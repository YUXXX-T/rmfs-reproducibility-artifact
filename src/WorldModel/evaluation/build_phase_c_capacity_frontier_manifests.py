"""Build paired offered-load manifests by scaling one long base order stream.

For an offered-load multiplier ``m``, orders from the same base stream with
``base_tick < target_ticks * m`` are selected and replayed at
``floor(base_tick / m)``.  Therefore higher multipliers contain a longer
prefix of exactly the same underlying orders and compress that prefix into the
same target horizon.  This produces a genuinely paired load sweep instead of
comparing unrelated low/mid/high manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "phase_c_capacity_frontier_manifest_contract_v1"
MANIFEST_SCHEMA_VERSION = "layer5_order_arrival_manifest_v1"
MANIFEST_SOURCE_KEY = "greedy_manifest"


def _canonical_manifest_sha(payload: Mapping[str, Any]) -> str:
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


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        current = _read_json(path)
        if current == payload:
            print(f"[resume] {path}")
            return
        raise FileExistsError(
            f"existing file differs from deterministic capacity-frontier output: {path}"
        )
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    orders = payload.get("orders")
    checks = {
        "schema": payload.get("schema_version") == MANIFEST_SCHEMA_VERSION,
        "orders": isinstance(orders, list),
        "count": isinstance(orders, list)
        and int(payload.get("total_orders", -1)) == len(orders),
        "hash": payload.get("manifest_sha256") == _canonical_manifest_sha(payload),
    }
    if isinstance(orders, list):
        last = (-1, -1)
        ordered = True
        for row in orders:
            if not isinstance(row, dict):
                ordered = False
                break
            key = (int(row.get("tick", -1)), int(row.get("order_id", -1)))
            if key < last:
                ordered = False
                break
            last = key
        checks["ordered"] = ordered
    else:
        checks["ordered"] = False
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"invalid manifest {path}: {failed}")
    return payload


def _parse_multipliers(values: Iterable[str]) -> list[Decimal]:
    parsed: list[Decimal] = []
    for raw in values:
        try:
            value = Decimal(str(raw))
        except InvalidOperation as exc:
            raise SystemExit(f"invalid multiplier: {raw}") from exc
        if value <= 0:
            raise SystemExit(f"multipliers must be positive: {raw}")
        if value * 100 != (value * 100).to_integral_value():
            raise SystemExit(
                f"multiplier must have at most two decimal places for stable tags: {raw}"
            )
        parsed.append(value)
    unique = sorted(set(parsed))
    if len(unique) != len(parsed):
        raise SystemExit("multipliers must be unique")
    return unique


def multiplier_tag(multiplier: Decimal) -> str:
    return f"m{int(multiplier * 100):03d}"


def _scaled_manifest(
    base: Mapping[str, Any], *, multiplier: Decimal, target_ticks: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    cutoff = Decimal(target_ticks) * multiplier
    selected = [
        row for row in (base.get("orders") or [])
        if Decimal(int(row["tick"])) < cutoff
    ]
    scaled_orders = []
    selected_base_keys = []
    for row in selected:
        base_tick = int(row["tick"])
        scaled_tick = int(math.floor(Decimal(base_tick) / multiplier))
        if not 0 <= scaled_tick < target_ticks:
            raise ValueError(
                f"scaled tick outside target horizon: {base_tick}/{multiplier}={scaled_tick}"
            )
        scaled = dict(row)
        scaled["tick"] = scaled_tick
        scaled_orders.append(scaled)
        selected_base_keys.append([base_tick, int(row["order_id"])])
    scaled_orders.sort(key=lambda row: (int(row["tick"]), int(row["order_id"])))
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "orders": scaled_orders,
        "total_orders": len(scaled_orders),
    }
    payload["manifest_sha256"] = _canonical_manifest_sha(payload)
    selection_contract = {
        "cutoff_base_tick_exclusive": str(cutoff),
        "selected_order_count": len(selected_base_keys),
        "selected_base_keys_sha256": _canonical_sha(
            {"base_tick_order_id": selected_base_keys}
        ),
        "scaled_min_tick": (
            min(int(row["tick"]) for row in scaled_orders)
            if scaled_orders else None
        ),
        "scaled_max_tick": (
            max(int(row["tick"]) for row in scaled_orders)
            if scaled_orders else None
        ),
    }
    return payload, selection_contract


def _manifest_path(root: Path, load: str, seed: int) -> Path:
    return root / "order_manifests" / f"orders_{load}_seed{seed}.json"


def _manifest_source_path(root: Path, load: str, seed: int) -> Path:
    return root / "per_arm" / MANIFEST_SOURCE_KEY / f"{load}_seed{seed}.json"


def _validate_base_source_contract(
    root: Path,
    *,
    load: str,
    seed: int,
    base_ticks: int,
    manifest: Mapping[str, Any],
    required: bool,
) -> dict[str, Any]:
    path = _manifest_source_path(root, load, seed)
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"base manifest source contract is required but missing: {path}"
            )
        return {"checked": False, "path": path.as_posix()}

    payload = _read_json(path)
    meta = payload.get("meta") or {}
    source_manifest = payload.get("manifest") or {}
    checks = {
        "load": meta.get("load") == load,
        "seed": int(meta.get("seed", -1)) == int(seed),
        "ticks": int(meta.get("ticks", -1)) == int(base_ticks),
        "manifest_sha256": (
            source_manifest.get("content_sha256") == manifest.get("manifest_sha256")
        ),
        "manifest_count": (
            int(source_manifest.get("total_orders", -1))
            == int(manifest.get("total_orders", -1))
        ),
    }
    audit = payload.get("audit")
    if isinstance(audit, Mapping) and "passed" in audit:
        checks["source_audit"] = bool(audit.get("passed"))
    station_audit = payload.get("station_admission_audit")
    if isinstance(station_audit, Mapping) and "passed" in station_audit:
        checks["station_audit"] = bool(station_audit.get("passed"))
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"invalid base manifest source contract {path}: {failed}")
    return {
        "checked": True,
        "passed": True,
        "path": path.as_posix(),
        "schema_version": payload.get("schema_version"),
        "ticks": int(meta["ticks"]),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--load", choices=("low", "mid", "high"), default="high")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--base-ticks", type=int, required=True)
    parser.add_argument("--target-ticks", type=int, default=1500)
    parser.add_argument(
        "--multipliers", nargs="+", default=("0.8", "1.0", "1.2", "1.4", "1.6")
    )
    parser.add_argument("--unit-reference-root", type=Path, default=None)
    parser.add_argument("--require-unit-reference-match", action="store_true")
    parser.add_argument(
        "--require-base-source-contract",
        action="store_true",
        help=(
            "require each long base manifest to have a matching manifest-source "
            "result whose recorded tick horizon equals --base-ticks"
        ),
    )
    args = parser.parse_args()

    if args.base_ticks <= 0 or args.target_ticks <= 0:
        raise SystemExit("--base-ticks and --target-ticks must be positive")
    multipliers = _parse_multipliers(args.multipliers)
    required_base_ticks = int(
        math.ceil(Decimal(args.target_ticks) * max(multipliers))
    )
    if args.base_ticks < required_base_ticks:
        raise SystemExit(
            f"base horizon {args.base_ticks} is too short; need at least {required_base_ticks}"
        )
    if args.require_unit_reference_match and args.unit_reference_root is None:
        raise SystemExit(
            "--require-unit-reference-match requires --unit-reference-root"
        )

    per_seed: dict[str, Any] = {}
    unit_multiplier = Decimal("1.0")
    for seed in args.seeds:
        base_path = _manifest_path(args.base_root, args.load, int(seed))
        if not base_path.is_file():
            raise FileNotFoundError(base_path)
        base = _validate_manifest(base_path)
        base_orders = base["orders"]
        if base_orders and max(int(row["tick"]) for row in base_orders) >= args.base_ticks:
            raise ValueError(
                f"base manifest contains tick >= base horizon {args.base_ticks}: {base_path}"
            )
        seed_contract: dict[str, Any] = {
            "base_manifest": base_path.as_posix(),
            "base_manifest_sha256": base["manifest_sha256"],
            "base_total_orders": int(base["total_orders"]),
            "base_max_tick": (
                max(int(row["tick"]) for row in base_orders)
                if base_orders else None
            ),
            "base_source_contract": _validate_base_source_contract(
                args.base_root,
                load=args.load,
                seed=int(seed),
                base_ticks=int(args.base_ticks),
                manifest=base,
                required=bool(args.require_base_source_contract),
            ),
            "scaled": {},
        }
        for multiplier in multipliers:
            tag = multiplier_tag(multiplier)
            scaled, selection = _scaled_manifest(
                base, multiplier=multiplier, target_ticks=args.target_ticks
            )
            output_path = _manifest_path(
                args.output_root / tag, args.load, int(seed)
            )
            _atomic_json(output_path, scaled)
            validated = _validate_manifest(output_path)
            reference_check: dict[str, Any] = {"checked": False}
            if multiplier == unit_multiplier and args.unit_reference_root is not None:
                reference_path = _manifest_path(
                    args.unit_reference_root, args.load, int(seed)
                )
                if reference_path.is_file():
                    reference = _validate_manifest(reference_path)
                    passed = (
                        validated["manifest_sha256"]
                        == reference["manifest_sha256"]
                        and int(validated["total_orders"])
                        == int(reference["total_orders"])
                    )
                    reference_check = {
                        "checked": True,
                        "passed": passed,
                        "path": reference_path.as_posix(),
                        "content_sha256": reference["manifest_sha256"],
                        "total_orders": int(reference["total_orders"]),
                    }
                    if args.require_unit_reference_match and not passed:
                        raise ValueError(
                            f"1.0x scaled manifest does not match reference: seed={seed}"
                        )
                elif args.require_unit_reference_match:
                    raise FileNotFoundError(reference_path)
            seed_contract["scaled"][tag] = {
                "multiplier": float(multiplier),
                "manifest": output_path.as_posix(),
                "manifest_sha256": validated["manifest_sha256"],
                "total_orders": int(validated["total_orders"]),
                "selection": selection,
                "unit_reference": reference_check,
            }
        per_seed[str(seed)] = seed_contract

    nested_prefix_checks = []
    ordered_tags = [multiplier_tag(value) for value in multipliers]
    for seed in args.seeds:
        base = _validate_manifest(_manifest_path(args.base_root, args.load, int(seed)))
        previous_keys: set[tuple[int, int]] = set()
        for multiplier, tag in zip(multipliers, ordered_tags):
            cutoff = Decimal(args.target_ticks) * multiplier
            keys = {
                (int(row["tick"]), int(row["order_id"]))
                for row in base["orders"]
                if Decimal(int(row["tick"])) < cutoff
            }
            passed = previous_keys.issubset(keys)
            nested_prefix_checks.append({
                "seed": int(seed),
                "multiplier": float(multiplier),
                "tag": tag,
                "previous_prefix_is_subset": passed,
                "selected_order_count": len(keys),
            })
            if not passed:
                raise RuntimeError(
                    f"scaled manifest prefixes are not nested: seed={seed} tag={tag}"
                )
            previous_keys = keys

    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "load": args.load,
        "seeds": [int(seed) for seed in args.seeds],
        "base_root": args.base_root.as_posix(),
        "output_root": args.output_root.as_posix(),
        "base_ticks": int(args.base_ticks),
        "target_ticks": int(args.target_ticks),
        "scaling_formula": {
            "selection": "base_tick < target_ticks * multiplier",
            "scaled_tick": "floor(base_tick / multiplier)",
            "interpretation": (
                "higher multipliers select a longer nested prefix of the same "
                "base stream and compress it into the same target horizon"
            ),
        },
        "multipliers": [
            {"value": float(value), "tag": multiplier_tag(value)}
            for value in multipliers
        ],
        "unit_reference_root": (
            args.unit_reference_root.as_posix()
            if args.unit_reference_root is not None else None
        ),
        "unit_reference_required": bool(args.require_unit_reference_match),
        "base_source_contract_required": bool(args.require_base_source_contract),
        "per_seed": per_seed,
        "nested_prefix_checks": nested_prefix_checks,
    }
    contract["contract_sha256"] = _canonical_sha(contract)
    contract_path = args.output_root / "capacity_frontier_manifest_contract.json"
    _atomic_json(contract_path, contract)
    print(json.dumps({
        "contract": contract_path.as_posix(),
        "contract_sha256": contract["contract_sha256"],
        "multipliers": contract["multipliers"],
        "seeds": contract["seeds"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
