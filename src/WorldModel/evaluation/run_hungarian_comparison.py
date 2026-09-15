"""Run an independent Hungarian baseline on frozen World-Model order streams.

The Hungarian assigner is never called from the World Model policy.  This
runner reuses an already-completed pure-WM arm and replays its exact realised
order manifest into a separate Manhattan Hungarian simulation arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable

from Policies.TaskAssigner.HungarianTaskAssigner import HungarianTaskAssigner
from WorldModel.evaluation.evaluate_online_v6 import (
    _run_one_assigner,
    aggregate_seeds,
)


SCHEMA_VERSION = "worldmodel_hungarian_paired_comparison_v2"
BASELINE_LABEL = "WorldModel"
HUNGARIAN_LABEL = "Hungarian(Manhattan)"
FORMAL_SEEDS = tuple(range(431, 441))
FORMAL_TICKS = 1500
LOAD_CONFIGS = {
    "low": "Config/world_model_config_PP_48_low.json",
    "mid": "Config/world_model_config_PP_48_mid.json",
    "high": "Config/world_model_config_PP_48_high.json",
}

# Direction is from the user's system objective, not from either policy's
# internal scoring rule.  The report stores both raw Hungarian-WM deltas and
# direction-normalised improvement, where positive always favours Hungarian.
FOCUS_METRICS = {
    "completed_orders": "higher",
    "completed_tasks": "higher",
    "avg_task_duration": "lower",
    "avg_excess_delay": "lower",
    "open_order_count": "lower",
    "pending_order_count": "lower",
    "open_order_age_p95": "lower",
    "pending_order_age_p95": "lower",
    "completed_order_flow_time_p95": "lower",
    "wm_label_cost": "lower",
    "wait_or_stall": "lower",
    "station_pressure": "lower",
    "bottleneck_CVaR": "lower",
    "unified_risk": "lower",
    "risk_rate_per_100": "lower",
    "stall_ratio_mean": "lower",
    "deadlock_ratio_mean": "lower",
    "deadlock_ratio_max": "lower",
}


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: dict) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def _bootstrap_ci95(
    values: list[float],
    *,
    seed: int = 20260720,
    trials: int = 10000,
) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = random.Random(seed)
    n = len(values)
    boot = []
    for _ in range(trials):
        boot.append(_mean(values[rng.randrange(n)] for _ in range(n)))
    boot.sort()
    lower = boot[int(math.floor(0.025 * (trials - 1)))]
    upper = boot[int(math.ceil(0.975 * (trials - 1)))]
    return [float(lower), float(upper)]


def _paired_summary(per_seed: dict[int, dict[str, dict]]) -> dict:
    summary = {}
    for metric, direction in FOCUS_METRICS.items():
        rows = []
        for seed in sorted(per_seed):
            baseline = per_seed[seed][BASELINE_LABEL].get(metric)
            hungarian = per_seed[seed][HUNGARIAN_LABEL].get(metric)
            if not isinstance(baseline, (int, float)) or not isinstance(
                hungarian, (int, float)
            ):
                continue
            raw_delta = float(hungarian) - float(baseline)
            favourable = raw_delta if direction == "higher" else -raw_delta
            rows.append({
                "seed": int(seed),
                "world_model": float(baseline),
                "hungarian": float(hungarian),
                "hungarian_minus_world_model": raw_delta,
                "favourable_improvement": favourable,
            })
        if not rows:
            continue
        raw = [row["hungarian_minus_world_model"] for row in rows]
        favourable = [row["favourable_improvement"] for row in rows]
        summary[metric] = {
            "direction": direction,
            "paired_seeds": len(rows),
            "hungarian_minus_world_model_mean": _mean(raw),
            "hungarian_minus_world_model_ci95_seed_bootstrap": (
                _bootstrap_ci95(raw)
            ),
            "favourable_improvement_mean": _mean(favourable),
            "favourable_improvement_ci95_seed_bootstrap": (
                _bootstrap_ci95(favourable)
            ),
            "hungarian_favourable_seed_count": sum(
                row["favourable_improvement"] > 0.0 for row in rows
            ),
            "world_model_favourable_seed_count": sum(
                row["favourable_improvement"] < 0.0 for row in rows
            ),
            "tie_seed_count": sum(
                row["favourable_improvement"] == 0.0 for row in rows
            ),
            "per_seed": rows,
        }
    return summary


def _validate_baseline(
    baseline: dict,
    *,
    load: str,
    config: str,
    seeds: list[int],
    ticks: int,
) -> None:
    meta = baseline.get("meta") or {}
    if str(meta.get("load")) != load:
        raise ValueError(
            f"baseline load mismatch: {meta.get('load')} != {load}"
        )
    if Path(str(meta.get("config"))).as_posix() != Path(config).as_posix():
        raise ValueError("baseline config differs from requested config")
    if int(meta.get("ticks", -1)) != int(ticks):
        raise ValueError("baseline tick count differs from requested ticks")
    if [int(value) for value in meta.get("seeds", ())] != seeds:
        raise ValueError("baseline seeds differ from requested paired seeds")
    per_seed = baseline.get("per_seed") or {}
    for seed in seeds:
        arms = per_seed.get(str(seed)) or {}
        if BASELINE_LABEL not in arms:
            raise ValueError(
                f"baseline result lacks pure WorldModel arm for seed {seed}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--manifests-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--load", choices=tuple(LOAD_CONFIGS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--ticks", type=int, default=FORMAL_TICKS)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--formal",
        action="store_true",
        help="Require seeds 431--440, 1500 ticks and the frozen load config.",
    )
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds]
    if len(seeds) != len(set(seeds)):
        raise SystemExit("duplicate seeds are not allowed")
    if args.formal:
        if tuple(seeds) != FORMAL_SEEDS:
            raise SystemExit("formal Hungarian comparison freezes seeds 431--440")
        if int(args.ticks) != FORMAL_TICKS:
            raise SystemExit("formal Hungarian comparison freezes --ticks=1500")
        if Path(args.config).as_posix() != Path(
            LOAD_CONFIGS[args.load]
        ).as_posix():
            raise SystemExit(
                f"formal {args.load} comparison requires "
                f"{LOAD_CONFIGS[args.load]}"
            )

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite Hungarian result: {output}")
    baseline_path = Path(args.baseline_results)
    baseline = _read_json(baseline_path)
    _validate_baseline(
        baseline,
        load=args.load,
        config=args.config,
        seeds=seeds,
        ticks=args.ticks,
    )
    manifests_dir = Path(args.manifests_dir)
    baseline_per_seed = baseline["per_seed"]
    repo_root = Path(__file__).resolve().parents[2]
    hungarian_source = repo_root / (
        "Policies/TaskAssigner/HungarianTaskAssigner/"
        "hungarian_task_assigner.py"
    )
    commit_source = repo_root / (
        "Policies/TaskAssigner/context_assignment.py"
    )

    protocol = {
        "schema_version": "hungarian_comparison_protocol_v2",
        "formal": bool(args.formal),
        "load": args.load,
        "config": Path(args.config).as_posix(),
        "seeds": seeds,
        "ticks": int(args.ticks),
        "baseline_arm": BASELINE_LABEL,
        "comparison_arm": HUNGARIAN_LABEL,
        "order_stream": "replay_exact_worldmodel_manifest",
        "hungarian_objective": "global_minimum_robot_to_pod_manhattan_distance",
        "hungarian_implementation_version": (
            "manhattan_global_shared_fixed_context_commit_v2"
        ),
        "task_chain_commit": "shared_fixed_context_commit",
        "station_queue_contract": True,
        "return_source": "station_exit_or_service_fallback",
        "hungarian_source_sha256": _sha256_file(hungarian_source),
        "shared_commit_source_sha256": _sha256_file(commit_source),
        "world_model_loaded_by_hungarian_arm": False,
        "hungarian_called_inside_world_model": False,
        "future_order_prediction": False,
    }
    protocol["protocol_sha256"] = _canonical_sha256(protocol)

    per_seed: dict[int, dict[str, dict]] = {}
    manifest_audit = {}
    for seed in seeds:
        baseline_metrics = dict(
            baseline_per_seed[str(seed)][BASELINE_LABEL]
        )
        manifest_path = manifests_dir / (
            f"layer5_{args.load}_seed{seed}_orders.json"
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = _read_json(manifest_path)
        manifest_sha = str(manifest.get("manifest_sha256", ""))
        expected_sha = str(
            baseline_metrics.get("order_arrival_manifest_sha256", "")
        )
        expected_count = int(
            baseline_metrics.get("order_arrival_count", -1)
        )
        if manifest_sha != expected_sha:
            raise RuntimeError(
                f"seed {seed}: manifest content hash differs from WM baseline"
            )
        if int(manifest.get("total_orders", -1)) != expected_count:
            raise RuntimeError(
                f"seed {seed}: manifest order count differs from WM baseline"
            )

        print(f"\n--- {args.load} seed={seed}: independent Hungarian arm ---")
        hungarian = HungarianTaskAssigner()
        hungarian_metrics = _run_one_assigner(
            args.config,
            hungarian,
            seed,
            args.ticks,
            trace_label=HUNGARIAN_LABEL,
            recorded_orders_path=str(manifest_path),
        )
        if (
            hungarian_metrics.get("order_arrival_manifest_sha256")
            != expected_sha
            or int(hungarian_metrics.get("order_arrival_count", -1))
            != expected_count
        ):
            raise RuntimeError(
                f"seed {seed}: Hungarian did not receive the exact WM order stream"
            )
        per_seed[seed] = {
            BASELINE_LABEL: baseline_metrics,
            HUNGARIAN_LABEL: hungarian_metrics,
        }
        manifest_audit[str(seed)] = {
            "path": str(manifest_path),
            "file_sha256": _sha256_file(manifest_path),
            "manifest_sha256": manifest_sha,
            "order_count": expected_count,
            "paired_hash_match": True,
            "paired_count_match": True,
        }
        print(
            f"[done] {args.load} seed={seed} "
            f"orders {baseline_metrics.get('completed_orders')} -> "
            f"{hungarian_metrics.get('completed_orders')}; "
            f"cost {baseline_metrics.get('wm_label_cost')} -> "
            f"{hungarian_metrics.get('wm_label_cost')}"
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "meta": {
            "load": args.load,
            "config": args.config,
            "seeds": seeds,
            "ticks": int(args.ticks),
            "baseline_results": str(baseline_path),
            "baseline_results_sha256": _sha256_file(baseline_path),
            "manifests_dir": str(manifests_dir),
            "paired_arms": [BASELINE_LABEL, HUNGARIAN_LABEL],
            "pure_world_model_reused": True,
            "world_model_rerun": False,
            "hungarian_is_independent_baseline": True,
        },
        "manifest_audit": manifest_audit,
        "per_seed": {str(seed): rows for seed, rows in per_seed.items()},
        "aggregate": aggregate_seeds(per_seed),
        "paired_comparison": _paired_summary(per_seed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nsaved paired WorldModel/Hungarian result: {output}")
    for metric in (
        "completed_orders",
        "avg_excess_delay",
        "wm_label_cost",
        "unified_risk",
        "deadlock_ratio_max",
    ):
        row = result["paired_comparison"].get(metric)
        if row:
            print(
                f"{metric}: favourable improvement="
                f"{row['favourable_improvement_mean']:.6g}, "
                f"CI95={row['favourable_improvement_ci95_seed_bootstrap']}"
            )


if __name__ == "__main__":
    main()
