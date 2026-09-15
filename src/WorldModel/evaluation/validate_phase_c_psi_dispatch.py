"""Validate paired outcomes from the 551--560 psi-dispatch development run."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    ARM_KEYS,
    ARM_LABELS,
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOADS,
    PER_ARM_SCHEMA_VERSION,
    PSI_ARM,
    PSI_SHADOW_ARM,
    REPORT_SCHEMA_VERSION,
    S1_ARM,
    SCHEMA_VERSION,
    SEEDS,
    canonical_sha256,
    sha256_file,
)


METRIC_KEYS = (
    "completed_orders",
    "completed_tasks",
    "avg_task_duration",
    "avg_excess_delay",
    "open_order_count",
    "pending_order_count",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "stall_ratio_mean",
    "stall_ratio_max",
    "assignment_time_ms_mean",
    "wall_time_s",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite changed output: {path}")
        return
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
    temporary.replace(path)


def _metric(payload: Mapping[str, Any], key: str) -> float | None:
    value = (payload.get("metrics") or {}).get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _bootstrap_ci(
    values_by_seed: Mapping[int, list[float]],
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | None]:
    if not values_by_seed:
        return {"mean": None, "ci95_lower": None, "ci95_upper": None}
    seeds = sorted(values_by_seed)
    seed_means = [mean(values_by_seed[s]) for s in seeds]
    point = mean(seed_means)
    rng = random.Random(seed)
    draws = []
    for _ in range(max(1, int(repeats))):
        sample = [seed_means[rng.randrange(len(seed_means))] for _ in seeds]
        draws.append(mean(sample))
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return {
        "mean": float(point),
        "ci95_lower": float(lo),
        "ci95_upper": float(hi),
    }


def _load_rows(root: Path, protocol_sha: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm in ARM_KEYS:
        for load in LOADS:
            for seed in SEEDS:
                path = root / "per_arm" / arm / f"{load}_seed{seed}.json"
                if not path.is_file():
                    missing.append(str(path))
                    continue
                payload = _read_json(path)
                meta = payload.get("meta") or {}
                if payload.get("schema_version") != PER_ARM_SCHEMA_VERSION:
                    raise ValueError(f"wrong arm schema: {path}")
                if meta.get("protocol_sha256") != protocol_sha:
                    raise ValueError(f"protocol mismatch: {path}")
                if not bool((payload.get("audit") or {}).get("passed")):
                    raise ValueError(f"arm audit failed: {path}")
                metrics = {
                    key: _metric(payload, key) for key in METRIC_KEYS
                }
                rows.append({
                    "arm": arm,
                    "arm_label": ARM_LABELS[arm],
                    "load": load,
                    "seed": int(seed),
                    "path": path.as_posix(),
                    "metrics": metrics,
                    "mechanism": {
                        key: (payload.get("metrics") or {}).get(key)
                        for key in (
                            "psi_dispatch_mode",
                            "psi_dispatch_head_loaded",
                            "psi_dispatch_eval_calls",
                            "psi_dispatch_contexts_seen",
                            "psi_dispatch_reordered_calls",
                            "psi_dispatch_replacement_count",
                            "psi_dispatch_service_debt_mean",
                            "psi_dispatch_j_mean",
                            "psi_dispatch_robot_scorer",
                        )
                    },
                })
    return rows, missing


def _paired_deltas(
    rows: list[dict[str, Any]],
    arm: str,
    baseline: str = S1_ARM,
) -> list[dict[str, Any]]:
    index = {
        (row["arm"], row["load"], row["seed"]): row
        for row in rows
    }
    result = []
    for load in LOADS:
        for seed in SEEDS:
            left = index.get((arm, load, seed))
            right = index.get((baseline, load, seed))
            if left is None or right is None:
                continue
            delta = {
                key: (
                    None
                    if left["metrics"].get(key) is None
                    or right["metrics"].get(key) is None
                    else left["metrics"][key] - right["metrics"][key]
                )
                for key in METRIC_KEYS
            }
            result.append({
                "load": load,
                "seed": int(seed),
                "delta_arm": arm,
                "delta_baseline": baseline,
                "metrics": delta,
            })
    return result


def _summarise_deltas(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in METRIC_KEYS:
        by_seed: dict[int, list[float]] = {}
        by_load: dict[str, list[float]] = {load: [] for load in LOADS}
        for row in deltas:
            value = row["metrics"].get(key)
            if value is None:
                continue
            by_seed.setdefault(int(row["seed"]), []).append(float(value))
            by_load[row["load"]].append(float(value))
        summary = _bootstrap_ci(by_seed)
        summary["per_load_mean"] = {
            load: (float(mean(values)) if values else None)
            for load, values in by_load.items()
        }
        result[key] = summary
    return result


def validate(input_root: Path, frozen_bundle: Path, output_dir: Path) -> Path:
    bundle = _read_json(frozen_bundle)
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError("wrong psi-dispatch frozen bundle schema")
    protocol = bundle.get("protocol") or {}
    claimed = str(protocol.get("protocol_sha256", ""))
    unsigned = dict(protocol)
    unsigned.pop("protocol_sha256", None)
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong psi-dispatch protocol schema")
    if canonical_sha256(unsigned) != claimed:
        raise ValueError("psi-dispatch protocol hash mismatch")
    rows, missing = _load_rows(input_root, claimed)
    complete = not missing and len(rows) == len(ARM_KEYS) * len(LOADS) * len(SEEDS)

    deltas = {
        arm: _paired_deltas(rows, arm)
        for arm in (PSI_SHADOW_ARM, PSI_ARM)
    }
    mechanism_rows = [
        row for row in rows if row["arm"] in (PSI_SHADOW_ARM, PSI_ARM)
    ]
    mechanism = {
        "model_runs": len(mechanism_rows),
        "head_loaded_all": all(
            bool(row["mechanism"].get("psi_dispatch_head_loaded"))
            for row in mechanism_rows
        ) if mechanism_rows else False,
        "j_evaluated_all": all(
            int(row["mechanism"].get("psi_dispatch_eval_calls") or 0) > 0
            and int(row["mechanism"].get("psi_dispatch_contexts_seen") or 0) > 0
            for row in mechanism_rows
        ) if mechanism_rows else False,
        "service_debt_finite_all": all(
            row["mechanism"].get("psi_dispatch_service_debt_mean") is not None
            and float(row["mechanism"]["psi_dispatch_service_debt_mean"])
            == float(row["mechanism"]["psi_dispatch_service_debt_mean"])
            for row in mechanism_rows
        ) if mechanism_rows else False,
        "parent_scorer_all": all(
            row["mechanism"].get("psi_dispatch_robot_scorer")
            == "WorldModelTaskAssigner.select_robots_unmodified"
            for row in mechanism_rows
        ) if mechanism_rows else False,
        "applied_reordered_total": sum(
            int(row["mechanism"].get("psi_dispatch_reordered_calls") or 0)
            for row in mechanism_rows
            if row["arm"] == PSI_ARM
        ),
        "applied_replacement_total": sum(
            int(row["mechanism"].get("psi_dispatch_replacement_count") or 0)
            for row in mechanism_rows
            if row["arm"] == PSI_ARM
        ),
    }
    mechanism_contract = all(
        bool(mechanism[key])
        for key in (
            "model_runs",
            "head_loaded_all",
            "j_evaluated_all",
            "service_debt_finite_all",
            "parent_scorer_all",
        )
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_sha256": claimed,
        "frozen_bundle": frozen_bundle.as_posix(),
        "frozen_bundle_sha256": sha256_file(frozen_bundle),
        "input_root": input_root.as_posix(),
        "complete": complete,
        "missing_outputs": missing,
        "counts": {
            "expected": len(ARM_KEYS) * len(LOADS) * len(SEEDS),
            "observed": len(rows),
        },
        "mechanism": mechanism,
        "paired_deltas": deltas,
        "paired_summary": {
            arm: _summarise_deltas(values)
            for arm, values in deltas.items()
        },
        "pass_criteria": {
            "collection_complete": complete,
            "mechanism_contract": mechanism_contract,
            "outcome_noninferiority_not_adjudicated": True,
        },
        "passed": bool(complete and mechanism_contract),
    }
    output_path = output_dir / "phase_c_psi_dispatch_validation.json"
    _atomic_write(output_path, report)
    _atomic_write(
        output_dir / "validation.sha256.json",
        {
            "schema_version": "phase_c_psi_dispatch_validation_hash_v1",
            "validation": output_path.as_posix(),
            "sha256": sha256_file(output_path),
        },
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--frozen-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    path = validate(args.input_root, args.frozen_bundle, args.output_dir)
    report = _read_json(path)
    print(f"[validate] wrote {path}")
    print(f"[validate] complete={report['complete']} passed={report['passed']}")


if __name__ == "__main__":
    main()
