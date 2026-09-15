"""Aggregate the frozen H=10 station congestion endpoint transport audit."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from WorldModel.evaluation.collect_phase_c_station_congestion_endpoint import (
    _load_bundle,
    _read_json,
    _verify_hashes,
)
from WorldModel.evaluation.freeze_phase_c_station_congestion_endpoint_h10 import (
    FROZEN_FILENAME,
)
from WorldModel.evaluation.phase_c_station_congestion_endpoint_protocol import (
    ANALYSIS_SCHEMA_VERSION,
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    HORIZON,
    LOADS,
    OUTPUT_ROOT,
    REPLAY_SCHEMA_VERSION,
    REPLAY_SHARDS,
    SEEDS,
    TICKS,
    sha256_file,
)
from WorldModel.evaluation.station_congestion_endpoint import (
    ENDPOINT_RECORD_SCHEMA_VERSION,
)


CHANNELS = ("traffic", "service")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        _json_safe(payload), indent=2, ensure_ascii=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite changed analysis: {path}")
        return
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or right.size != left.size:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator <= 1e-15:
        return float("nan")
    return float((left @ right) / denominator)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(_rankdata(left), _rankdata(right))


def _one_hot(values: Sequence[Any]) -> list[np.ndarray]:
    values = np.asarray(values)
    unique = sorted(set(values.tolist()), key=str)
    return [(values == category).astype(np.float64) for category in unique[1:]]


def _partial_spearman(
    prediction: np.ndarray,
    target: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> float:
    columns = [np.ones(len(rows), dtype=np.float64)]
    ticks = np.asarray(
        [float(row["decision_tick"]) / max(TICKS, 1) for row in rows]
    )
    columns.append(ticks - ticks.mean())
    columns.extend(_one_hot([row["load"] for row in rows]))
    columns.extend(_one_hot([int(row["station_id"]) for row in rows]))
    design = np.column_stack(columns)
    q, _ = np.linalg.qr(design, mode="reduced")
    left = _rankdata(prediction)
    right = _rankdata(target)
    left = left - q @ (q.T @ left)
    right = right - q @ (q.T @ right)
    return _pearson(left, right)


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    repeats: int,
    seed: int,
) -> list[float]:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return [float("nan"), float("nan")]
    if finite.size == 1:
        return [float(finite[0]), float(finite[0])]
    rng = np.random.default_rng(seed)
    draws = rng.choice(finite, size=(int(repeats), finite.size), replace=True)
    return [
        float(value)
        for value in np.quantile(draws.mean(axis=1), [0.025, 0.975])
    ]


def _pairwise_accuracy(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int]:
    correct = 0
    total = 0
    for left in range(len(prediction)):
        for right in range(left + 1, len(prediction)):
            target_delta = float(target[left] - target[right])
            if abs(target_delta) <= 1e-12:
                continue
            pred_delta = float(prediction[left] - prediction[right])
            if abs(pred_delta) <= 1e-12:
                continue
            total += 1
            correct += int(pred_delta * target_delta > 0.0)
    return correct, total


def _group_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    groups: Sequence[str],
    *,
    best: str,
) -> dict[str, Any]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[str(group)].append(index)
    correlations = []
    top1 = []
    pair_correct = 0
    pair_total = 0
    for indices_list in by_group.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        if indices.size < 2:
            continue
        correlation = _spearman(prediction[indices], target[indices])
        if math.isfinite(correlation):
            correlations.append(correlation)
        if best == "max":
            chosen = indices[int(np.argmax(prediction[indices]))]
            optimum = float(np.max(target[indices]))
            top1.append(float(target[chosen]) >= optimum - 1e-12)
        elif best == "min":
            chosen = indices[int(np.argmin(prediction[indices]))]
            optimum = float(np.min(target[indices]))
            top1.append(float(target[chosen]) <= optimum + 1e-12)
        else:
            raise ValueError(best)
        correct, total = _pairwise_accuracy(
            prediction[indices], target[indices]
        )
        pair_correct += correct
        pair_total += total
    return {
        "groups": len(top1),
        "spearman_mean": (
            float(np.mean(correlations)) if correlations else float("nan")
        ),
        "spearman_median": (
            float(np.median(correlations)) if correlations else float("nan")
        ),
        "positive_spearman_fraction": (
            float(np.mean(np.asarray(correlations) > 0.0))
            if correlations else float("nan")
        ),
        "top1_hit_rate": float(np.mean(top1)) if top1 else float("nan"),
        "pairwise_accuracy": (
            float(pair_correct / pair_total) if pair_total else float("nan")
        ),
        "pairwise_pairs": int(pair_total),
    }


def _sign_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    def accuracy(indices: np.ndarray) -> float:
        if indices.size == 0:
            return float("nan")
        return float(np.mean(
            np.sign(prediction[indices]) == np.sign(target[indices])
        ))

    all_indices = np.arange(target.size)
    nontrivial = np.flatnonzero(np.abs(target) >= 0.01)
    return {
        "all_sign_accuracy": accuracy(all_indices),
        "nontrivial_abs_real_ge_0p01_samples": int(nontrivial.size),
        "nontrivial_abs_real_ge_0p01_sign_accuracy": accuracy(nontrivial),
        "real_near_zero_abs_lt_0p01_fraction": float(
            np.mean(np.abs(target) < 0.01)
        ),
    }


def _channel_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    prediction_key: str,
    target_key: str,
    channel: str,
    include_sign: bool,
    bootstrap_seed: int,
) -> dict[str, Any]:
    prediction = np.asarray([
        float(row[prediction_key][channel]) for row in rows
    ], dtype=np.float64)
    target = np.asarray([
        float(row[target_key][channel]) for row in rows
    ], dtype=np.float64)
    residual = prediction - target
    per_run = []
    per_run_map = {}
    run_ids = np.asarray([str(row["run_id"]) for row in rows])
    for run_id in sorted(set(run_ids.tolist())):
        indices = np.flatnonzero(run_ids == run_id)
        value = _spearman(prediction[indices], target[indices])
        if math.isfinite(value):
            per_run.append(value)
            per_run_map[run_id] = value
    by_load = {}
    for load in LOADS:
        indices = np.asarray([
            offset for offset, row in enumerate(rows) if row["load"] == load
        ], dtype=np.int64)
        by_load[load] = (
            _spearman(prediction[indices], target[indices])
            if indices.size >= 2 else float("nan")
        )
    metrics = {
        "samples": int(prediction.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(math.sqrt(np.mean(residual ** 2))),
        "bias": float(np.mean(residual)),
        "prediction_std": float(np.std(prediction)),
        "target_std": float(np.std(target)),
        "prediction_quantiles": {
            "q05": float(np.quantile(prediction, 0.05)),
            "q50": float(np.quantile(prediction, 0.50)),
            "q95": float(np.quantile(prediction, 0.95)),
        },
        "target_quantiles": {
            "q05": float(np.quantile(target, 0.05)),
            "q50": float(np.quantile(target, 0.50)),
            "q95": float(np.quantile(target, 0.95)),
        },
        "pooled_spearman": _spearman(prediction, target),
        "partial_spearman": _partial_spearman(prediction, target, rows),
        "per_run_spearman": per_run_map,
        "per_run_spearman_mean": (
            float(np.mean(per_run)) if per_run else float("nan")
        ),
        "per_run_cluster_ci95": _bootstrap_mean_ci(
            per_run,
            repeats=BOOTSTRAP_REPEATS,
            seed=bootstrap_seed,
        ),
        "run_sign_consistency": (
            float(np.mean(np.asarray(per_run) > 0.0))
            if per_run else float("nan")
        ),
        "by_load_spearman": by_load,
        "station_ranking_within_action": _group_metrics(
            prediction,
            target,
            [str(row["action_id"]) for row in rows],
            best="max",
        ),
    }
    context_indices = np.asarray([
        offset for offset, row in enumerate(rows)
        if bool(row["is_context_station"])
    ], dtype=np.int64)
    metrics["candidate_ranking_at_context_station"] = _group_metrics(
        prediction[context_indices],
        target[context_indices],
        [str(rows[offset]["candidate_rank_group"]) for offset in context_indices],
        best="min",
    )
    if include_sign:
        metrics["sign"] = _sign_metrics(prediction, target)
    return metrics


def _layer_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    prediction_key: str,
    target_key: str,
    include_sign: bool,
    seed_offset: int,
) -> dict[str, Any]:
    subsets = {
        "all_stations": list(rows),
        "context_station": [
            row for row in rows if bool(row["is_context_station"])
        ],
    }
    return {
        subset_name: {
            channel: _channel_metrics(
                subset_rows,
                prediction_key=prediction_key,
                target_key=target_key,
                channel=channel,
                include_sign=include_sign,
                bootstrap_seed=BOOTSTRAP_SEED + seed_offset + channel_offset,
            )
            for channel_offset, channel in enumerate(CHANNELS)
        }
        for subset_name, subset_rows in subsets.items()
    }


def _load_records(
    output_root: Path,
    *,
    protocol_sha256: str,
    require_complete: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    records = []
    seen_shards = set()
    for summary_path in sorted(output_root.glob(
        "replays/*/shard_*/replay_summary.json"
    )):
        summary = _read_json(summary_path)
        shard_dir = summary_path.parent
        manifest = shard_dir / "shard_outputs.sha256"
        checks = {
            "schema": summary.get("schema_version") == REPLAY_SCHEMA_VERSION,
            "protocol": summary.get("protocol_sha256") == protocol_sha256,
            "audit": bool((summary.get("audit") or {}).get("passed")),
            "manifest_hash": (
                manifest.is_file()
                and sha256_file(manifest) == summary.get("shard_outputs_sha256")
            ),
            "outputs": _verify_hashes(shard_dir, manifest),
        }
        if not all(checks.values()):
            failed = [key for key, passed in checks.items() if not passed]
            raise ValueError(f"invalid endpoint replay shard {shard_dir}: {failed}")
        key = (
            str(summary["load"]),
            int(summary["seed"]),
            int(summary["shard_index"]),
        )
        if key in seen_shards:
            raise ValueError(f"duplicate endpoint replay shard: {key}")
        seen_shards.add(key)
        records_path = shard_dir / str(summary["outputs"]["records"])
        with records_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if row.get("schema_version") != ENDPOINT_RECORD_SCHEMA_VERSION:
                    raise ValueError(f"wrong endpoint record schema: {records_path}")
                records.append(row)
        summaries.append(summary)
    expected = {
        (load, seed, shard)
        for load in LOADS
        for seed in SEEDS
        for shard in range(REPLAY_SHARDS)
    }
    if require_complete and seen_shards != expected:
        missing = sorted(expected - seen_shards)
        extra = sorted(seen_shards - expected)
        raise RuntimeError(
            f"endpoint replay shards incomplete: missing={missing} extra={extra}"
        )
    if not records:
        raise RuntimeError("no endpoint replay records found")
    return records, summaries


def analyze(output_root: Path, *, require_complete: bool = True) -> dict[str, Any]:
    bundle = _load_bundle(output_root / FROZEN_FILENAME)
    protocol_sha = str(bundle["protocol_sha256"])
    rows, summaries = _load_records(
        output_root,
        protocol_sha256=protocol_sha,
        require_complete=require_complete,
    )
    unique_record_keys = {
        (str(row["action_id"]), int(row["station_id"])) for row in rows
    }
    checks = {
        "records_unique": len(unique_record_keys) == len(rows),
        "horizon_fixed_10": all(int(row["horizon"]) == HORIZON for row in rows),
        "fresh_seeds_only": (
            all(int(row["seed"]) in SEEDS for row in rows)
            if require_complete
            else all(int(row["seed"]) > 520 for row in rows)
        ),
        "locked_501_510_absent": all(
            int(row["seed"]) not in range(501, 511) for row in rows
        ),
        "all_loads_present": (
            set(str(row["load"]) for row in rows) == set(LOADS)
            if require_complete
            else set(str(row["load"]) for row in rows).issubset(set(LOADS))
        ),
        "context_station_present_once_per_action": all(
            count == 1
            for count in (
                sum(
                    bool(row["is_context_station"])
                    for row in rows if row["action_id"] == action_id
                )
                for action_id in set(str(row["action_id"]) for row in rows)
            )
        ),
        "all_shard_audits_passed": all(
            bool((summary.get("audit") or {}).get("passed"))
            for summary in summaries
        ),
    }
    audit = {"passed": all(checks.values()), "checks": checks}
    if not audit["passed"]:
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError(f"endpoint analysis audit failed: {failed}")

    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "development_only": True,
        "performance_gate_frozen": False,
        "protocol_sha256": protocol_sha,
        "horizon": HORIZON,
        "counts": {
            "replay_shards": len(summaries),
            "runs": len(set(str(row["run_id"]) for row in rows)),
            "snapshots": len(set(
                (str(row["run_id"]), str(row["source_snapshot"]))
                for row in rows
            )),
            "actions": len(set(str(row["action_id"]) for row in rows)),
            "station_records": len(rows),
            "context_station_records": sum(
                bool(row["is_context_station"]) for row in rows
            ),
        },
        "layers": {
            "A_real_endpoint_decoding": _layer_metrics(
                rows,
                prediction_key="psi_real_endpoint",
                target_key="physical_target_endpoint",
                include_sign=False,
                seed_offset=0,
            ),
            "B_endpoint_latent_transport": _layer_metrics(
                rows,
                prediction_key="psi_predicted_endpoint",
                target_key="psi_real_endpoint",
                include_sign=False,
                seed_offset=100,
            ),
            "C_delta_transport": _layer_metrics(
                rows,
                prediction_key="delta_psi_predicted",
                target_key="delta_psi_real",
                include_sign=True,
                seed_offset=200,
            ),
        },
        "audit": audit,
        "interpretation_contract": {
            "A_failure": "station head does not decode real endpoint state",
            "B_failure_with_A_pass": (
                "frozen World-Model transition does not transport the latent "
                "to the correct H=10 endpoint potential"
            ),
            "C_failure_with_A_and_B_pass": (
                "endpoint levels are usable but action-induced potential "
                "changes are too small or poorly ordered for Q(c,r)"
            ),
            "online_policy_changed": False,
            "q_score_changed": False,
            "defer_context_tested": False,
        },
    }
    output_path = output_root / "station_congestion_endpoint_h10_analysis.json"
    _atomic_json(output_path, analysis)
    manifest = output_root / "analyzed_outputs.sha256"
    manifest_text = f"{sha256_file(output_path)}  {output_path.name}\n"
    if manifest.is_file():
        if manifest.read_text(encoding="utf-8") != manifest_text:
            raise FileExistsError(f"refusing to overwrite changed manifest: {manifest}")
    else:
        manifest.write_text(manifest_text, encoding="utf-8", newline="\n")
    print("[complete] H=10 station congestion endpoint analysis")
    print(json.dumps(_json_safe(analysis["counts"]), ensure_ascii=False))
    for layer_name, layer in analysis["layers"].items():
        traffic = layer["context_station"]["traffic"]
        service = layer["context_station"]["service"]
        print(
            f"{layer_name}: context rho traffic="
            f"{traffic['pooled_spearman']:.4f} service="
            f"{service['pooled_spearman']:.4f}"
        )
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Development smoke only; formal analysis requires all shards.",
    )
    args = parser.parse_args()
    analyze(args.output_root, require_complete=not args.allow_incomplete)


if __name__ == "__main__":
    main()
