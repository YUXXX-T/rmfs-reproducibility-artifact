"""Rank same-tick physical indicators of station and system congestion."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from WorldModel.evaluation.phase_c_station_congestion_correlation_protocol import (
    ARMS,
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    FRAME_STRIDE,
    LOADS,
    OUTPUT_ROOT,
    PRIMARY_REGION_HOPS,
    REGION_HOPS,
    REPORT_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    SEEDS,
    TICKS,
    TRACE_SCHEMA_VERSION,
    sha256_file,
)


STATION_TARGETS = (
    f"h{PRIMARY_REGION_HOPS}_mobility_impairment_ratio",
    f"h{PRIMARY_REGION_HOPS}_deadlocked_active_ratio",
    f"h{PRIMARY_REGION_HOPS}_conflict_participant_active_ratio",
    "recent_mobility_impairment",
)
SYSTEM_TARGETS = (
    "risk_unified",
    "risk_stall_ratio",
    "risk_deadlock_ratio",
    "mobility_impairment_ratio",
)

STATION_BASE_FEATURES = (
    "queue_occupancy_ratio",
    "assigned_agent_capacity_ratio",
    "assigned_agent_work_ratio",
    "pending_pressure",
    "in_progress_pressure",
    "open_order_pressure",
    "active_task_pressure",
    "recent_completed_orders_per_tick",
    "recent_conflict_events_per_tick",
)
REGION_FEATURES = (
    "traffic_blocked_active_ratio",
    "stuck_active_ratio",
    "plan_failed_active_ratio",
    "deadlocked_active_ratio",
    "mobility_impairment_ratio",
    "moved_active_ratio",
    "stationary_ticks_mean",
    "stationary_ticks_max",
    "conflict_events",
    "conflict_participant_active_ratio",
    "node_density_mean",
    "node_density_max",
    "node_density_cvar90",
    "node_wait_mean",
    "node_wait_max",
    "node_wait_cvar90",
    "node_blocked_mean",
    "node_blocked_max",
    "node_blocked_cvar90",
    "node_reservation_mean",
    "node_reservation_max",
    "node_reservation_cvar90",
    "node_recent_flow_mean",
    "node_recent_flow_max",
    "node_recent_flow_cvar90",
    "node_bottleneck_mean",
    "node_bottleneck_max",
    "node_bottleneck_cvar90",
    "node_bottleneck_weighted_density",
)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 3 or right.size != left.size:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _finite_pair(
    rows: Sequence[Mapping[str, Any]], feature: str, target: str
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    left = []
    right = []
    indices = []
    for index, row in enumerate(rows):
        try:
            x = float(row[feature])
            y = float(row[target])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        left.append(x)
        right.append(y)
        indices.append(index)
    return (
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        indices,
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3 or right.size != left.size:
        return float("nan")
    if np.all(left == left[0]) or np.all(right == right[0]):
        return float("nan")
    return _pearson(_rankdata(left), _rankdata(right))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _rankdata(np.asarray(scores, dtype=np.float64))
    rank_sum = float(ranks[labels].sum())
    return float(
        (rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _category_columns(values: Sequence[Any]) -> list[np.ndarray]:
    categories = sorted({str(value) for value in values})
    return [
        np.asarray([float(str(value) == category) for value in values])
        for category in categories[1:]
    ]


def _partial_spearman(
    rows: Sequence[Mapping[str, Any]],
    feature: str,
    target: str,
    *,
    station_scope: bool,
) -> float:
    left, right, indices = _finite_pair(rows, feature, target)
    if left.size < 10:
        return float("nan")
    selected = [rows[index] for index in indices]
    columns = [np.ones(left.size, dtype=np.float64)]
    for key in ("tick_fraction", "global_open_order_count", "global_active_robot_ratio"):
        values = np.asarray([float(row[key]) for row in selected], dtype=np.float64)
        scale = float(values.std())
        columns.append((values - values.mean()) / (scale if scale > 0 else 1.0))
    columns.extend(_category_columns([row["load"] for row in selected]))
    columns.extend(_category_columns([row["arm"] for row in selected]))
    if station_scope:
        columns.extend(_category_columns([
            row["station_id"] for row in selected
        ]))
    design = np.column_stack(columns)
    rank_left = _rankdata(left)
    rank_right = _rankdata(right)
    beta_left, *_ = np.linalg.lstsq(design, rank_left, rcond=None)
    beta_right, *_ = np.linalg.lstsq(design, rank_right, rcond=None)
    residual_left = rank_left - design @ beta_left
    residual_right = rank_right - design @ beta_right
    # A feature or target can be explained exactly by the frozen controls.
    # Correlating two floating-point round-off vectors would otherwise return
    # a misleading +/-1 instead of an undefined partial correlation.
    if (
        float(np.std(residual_left)) <= 1e-10
        or float(np.std(residual_right)) <= 1e-10
    ):
        return float("nan")
    return _pearson(residual_left, residual_right)


def _bootstrap_mean_ci(
    values: Sequence[float], repeats: int, seed: int
) -> tuple[float, float]:
    finite = np.asarray([
        float(value) for value in values if math.isfinite(float(value))
    ], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), float("nan")
    if finite.size == 1:
        return float(finite[0]), float(finite[0])
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(finite, size=(int(repeats), finite.size), replace=True)
    means = draws.mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def _within_tick_station_correlation(
    rows: Sequence[Mapping[str, Any]], feature: str, target: str
) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["run_id"]), int(row["tick"]))].append(row)
    values = []
    for group in groups.values():
        left, right, _ = _finite_pair(group, feature, target)
        value = _spearman(left, right)
        if math.isfinite(value):
            values.append(value)
    return {
        "groups": len(values),
        "mean": float(np.mean(values)) if values else float("nan"),
        "median": float(np.median(values)) if values else float("nan"),
        "positive_fraction": (
            float(np.mean(np.asarray(values) > 0.0)) if values else float("nan")
        ),
    }


def _feature_role(feature: str) -> str:
    if "completed" in feature or "recent_flow" in feature or "moved_active" in feature:
        return "flow_or_relief"
    if any(token in feature for token in (
        "blocked", "stuck", "deadlock", "impairment", "conflict", "plan_failed"
    )):
        return "direct_congestion_component"
    if any(token in feature for token in (
        "queue", "assigned", "pending", "pressure", "density", "reservation", "bottleneck"
    )):
        return "pressure_or_topology"
    return "state_context"


def _rowwise_rankdata(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks for a small station axis, vectorised over ticks."""
    values = np.asarray(values, dtype=np.float64)
    greater = values[:, :, None] > values[:, None, :]
    equal = values[:, :, None] == values[:, None, :]
    return 1.0 + greater.sum(axis=2) + 0.5 * (equal.sum(axis=2) - 1.0)


class _PreparedScope:
    """Cache arrays, control projection and grouping for many correlations."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        keys: Sequence[str],
        *,
        station_scope: bool,
    ) -> None:
        self.rows = rows
        self.station_scope = bool(station_scope)
        self.values = {
            key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
            for key in keys
        }
        for key, values in self.values.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"non-finite values in analysis column {key}")
        self.run_groups = self._groups("run_id")
        self.load_groups = self._groups("load")
        self.arm_groups = self._groups("arm")
        self.tick_station_groups = self._tick_station_groups() if station_scope else None
        self._rank_cache: dict[str, np.ndarray] = {}
        self._residual_cache: dict[str, np.ndarray | None] = {}
        self._control_q = self._control_projection()

    def _groups(self, key: str) -> dict[str, np.ndarray]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            grouped[str(row[key])].append(index)
        return {
            value: np.asarray(indices, dtype=np.int64)
            for value, indices in grouped.items()
        }

    def _tick_station_groups(self) -> np.ndarray:
        grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            grouped[(str(row["run_id"]), int(row["tick"]))].append(index)
        sizes = {len(indices) for indices in grouped.values()}
        if len(sizes) != 1 or not sizes or min(sizes) < 2:
            raise ValueError(
                f"station ranking groups have inconsistent sizes: {sorted(sizes)}"
            )
        return np.asarray(list(grouped.values()), dtype=np.int64)

    def _control_projection(self) -> np.ndarray:
        columns = [np.ones(len(self.rows), dtype=np.float64)]
        for key in (
            "tick_fraction",
            "global_open_order_count",
            "global_active_robot_ratio",
        ):
            values = _rankdata(np.asarray([
                float(row[key]) for row in self.rows
            ], dtype=np.float64))
            scale = float(values.std())
            columns.append(
                (values - values.mean()) / (scale if scale > 0 else 1.0)
            )
        for key in ("load", "arm"):
            columns.extend(_category_columns([row[key] for row in self.rows]))
        if self.station_scope:
            columns.extend(_category_columns([
                row["station_id"] for row in self.rows
            ]))
        design = np.column_stack(columns)
        q, _ = np.linalg.qr(design, mode="reduced")
        return q

    def ranks(self, key: str) -> np.ndarray:
        cached = self._rank_cache.get(key)
        if cached is None:
            cached = _rankdata(self.values[key])
            self._rank_cache[key] = cached
        return cached

    def residual_ranks(self, key: str) -> np.ndarray | None:
        if key in self._residual_cache:
            return self._residual_cache[key]
        ranks = self.ranks(key)
        residual = ranks - self._control_q @ (self._control_q.T @ ranks)
        value = residual if float(np.std(residual)) > 1e-10 else None
        self._residual_cache[key] = value
        return value


def _within_tick_prepared(
    prepared: _PreparedScope, feature: str, target: str
) -> dict[str, Any]:
    indices = prepared.tick_station_groups
    if indices is None:
        return {}
    left = prepared.values[feature][indices]
    right = prepared.values[target][indices]
    rank_left = _rowwise_rankdata(left)
    rank_right = _rowwise_rankdata(right)
    rank_left -= rank_left.mean(axis=1, keepdims=True)
    rank_right -= rank_right.mean(axis=1, keepdims=True)
    numerator = (rank_left * rank_right).sum(axis=1)
    denominator = np.sqrt(
        (rank_left * rank_left).sum(axis=1)
        * (rank_right * rank_right).sum(axis=1)
    )
    valid = denominator > 0.0
    values = numerator[valid] / denominator[valid]
    return {
        "groups": int(values.size),
        "mean": float(values.mean()) if values.size else float("nan"),
        "median": float(np.median(values)) if values.size else float("nan"),
        "positive_fraction": (
            float(np.mean(values > 0.0)) if values.size else float("nan")
        ),
    }


def _correlation_summary_prepared(
    prepared: _PreparedScope,
    feature: str,
    target: str,
    *,
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    left = prepared.values[feature]
    right = prepared.values[target]
    pooled = _pearson(prepared.ranks(feature), prepared.ranks(target))
    residual_left = prepared.residual_ranks(feature)
    residual_right = prepared.residual_ranks(target)
    partial = (
        _pearson(residual_left, residual_right)
        if residual_left is not None and residual_right is not None
        else float("nan")
    )
    per_run = []
    for indices in prepared.run_groups.values():
        value = _spearman(left[indices], right[indices])
        if math.isfinite(value):
            per_run.append(value)
    ci_low, ci_high = _bootstrap_mean_ci(
        per_run, bootstrap_repeats, bootstrap_seed
    )
    pooled_sign = 0 if not math.isfinite(pooled) or pooled == 0 else (1 if pooled > 0 else -1)
    sign_consistency = (
        float(np.mean([
            (value > 0) if pooled_sign > 0 else (value < 0)
            for value in per_run
        ]))
        if per_run and pooled_sign != 0 else float("nan")
    )
    by_load = {
        load: _spearman(left[indices], right[indices])
        for load, indices in prepared.load_groups.items()
    }
    by_arm = {
        arm: _spearman(left[indices], right[indices])
        for arm, indices in prepared.arm_groups.items()
    }
    result = {
        "scope": "station" if prepared.station_scope else "system",
        "target": target,
        "feature": feature,
        "feature_role": _feature_role(feature),
        "n": int(left.size),
        "runs": len(per_run),
        "pooled_spearman": pooled,
        "partial_spearman": partial,
        "per_run_spearman_mean": (
            float(np.mean(per_run)) if per_run else float("nan")
        ),
        "per_run_spearman_median": (
            float(np.median(per_run)) if per_run else float("nan")
        ),
        "cluster_bootstrap_ci95": [ci_low, ci_high],
        "run_sign_consistency": sign_consistency,
        "positive_state_auc": _auc(left, right > 0.0),
        "by_load_spearman": by_load,
        "by_arm_spearman": by_arm,
    }
    if prepared.station_scope:
        result["within_tick_station_ranking"] = _within_tick_prepared(
            prepared, feature, target
        )
    return result


def _correlation_summary(
    rows: Sequence[Mapping[str, Any]],
    feature: str,
    target: str,
    *,
    station_scope: bool,
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    left, right, _ = _finite_pair(rows, feature, target)
    pooled = _spearman(left, right)
    per_run = []
    run_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        run_groups[str(row["run_id"])].append(row)
    for run_rows in run_groups.values():
        run_left, run_right, _ = _finite_pair(run_rows, feature, target)
        value = _spearman(run_left, run_right)
        if math.isfinite(value):
            per_run.append(value)
    ci_low, ci_high = _bootstrap_mean_ci(
        per_run, bootstrap_repeats, bootstrap_seed
    )
    pooled_sign = 0 if not math.isfinite(pooled) or pooled == 0 else (1 if pooled > 0 else -1)
    sign_consistency = (
        float(np.mean([
            (value > 0) if pooled_sign > 0 else (value < 0)
            for value in per_run
        ]))
        if per_run and pooled_sign != 0 else float("nan")
    )

    by_load = {}
    for load in LOADS:
        selected = [row for row in rows if row["load"] == load]
        x, y, _ = _finite_pair(selected, feature, target)
        by_load[load] = _spearman(x, y)
    by_arm = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        x, y, _ = _finite_pair(selected, feature, target)
        by_arm[arm] = _spearman(x, y)

    result = {
        "scope": "station" if station_scope else "system",
        "target": target,
        "feature": feature,
        "feature_role": _feature_role(feature),
        "n": int(left.size),
        "runs": len(per_run),
        "pooled_spearman": pooled,
        "partial_spearman": _partial_spearman(
            rows, feature, target, station_scope=station_scope
        ),
        "per_run_spearman_mean": (
            float(np.mean(per_run)) if per_run else float("nan")
        ),
        "per_run_spearman_median": (
            float(np.median(per_run)) if per_run else float("nan")
        ),
        "cluster_bootstrap_ci95": [ci_low, ci_high],
        "run_sign_consistency": sign_consistency,
        "positive_state_auc": _auc(left, right > 0.0),
        "by_load_spearman": by_load,
        "by_arm_spearman": by_arm,
    }
    if station_scope:
        result["within_tick_station_ranking"] = (
            _within_tick_station_correlation(rows, feature, target)
        )
    return result


def _flatten_station(
    meta: Mapping[str, Any], tick: int, system: Mapping[str, Any], station: Mapping[str, Any]
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": meta["run_id"],
        "arm": meta["arm"],
        "load": meta["load"],
        "seed": int(meta["seed"]),
        "tick": int(tick),
        "tick_fraction": float(tick) / max(TICKS - 1, 1),
        "station_id": int(station["station_id"]),
        "global_open_order_count": float(system["open_order_count"]),
        "global_active_robot_ratio": float(system["active_robot_ratio"]),
    }
    for key, value in station.items():
        if key in {"station_id", "station_seed_nodes", "regions"}:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            row[key] = float(value)
    for region_name, region in (station.get("regions") or {}).items():
        for key, value in region.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[f"{region_name}_{key}"] = float(value)
    return row


def _flatten_system(
    meta: Mapping[str, Any], tick: int, system: Mapping[str, Any]
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": meta["run_id"],
        "arm": meta["arm"],
        "load": meta["load"],
        "seed": int(meta["seed"]),
        "tick": int(tick),
        "tick_fraction": float(tick) / max(TICKS - 1, 1),
        "global_open_order_count": float(system["open_order_count"]),
        "global_active_robot_ratio": float(system["active_robot_ratio"]),
    }
    for key, value in system.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            row[key] = float(value)
    return row


def _station_features(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    available = set.intersection(*(set(row) for row in rows[:100])) if rows else set()
    requested = list(STATION_BASE_FEATURES)
    for hops in REGION_HOPS:
        requested.extend(f"h{hops}_{name}" for name in REGION_FEATURES)
    return sorted(
        feature for feature in requested
        if feature in available and feature not in STATION_TARGETS
    )


def _system_features(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return []
    available = set.intersection(*(set(row) for row in rows[:100]))
    excluded = {
        "seed", "tick", "tick_fraction", "global_open_order_count",
        "global_active_robot_ratio", *SYSTEM_TARGETS,
    }
    features = []
    for key in sorted(available - excluded):
        if key in {"run_id", "arm", "load"}:
            continue
        if any(token in key for token in (
            "ratio", "pressure", "conflict", "queue", "assigned",
            "node_", "bottleneck_", "impairment", "completed", "order_count",
            "active_task", "stationary", "stuck", "blocked", "deadlock",
        )):
            features.append(key)
    return features


def _redundancy_rows(
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    scope: str,
    *,
    threshold: float = 0.90,
    max_rows: int = 50000,
) -> list[dict[str, Any]]:
    if not rows or len(features) < 2:
        return []
    if len(rows) > max_rows:
        indices = np.linspace(0, len(rows) - 1, max_rows, dtype=int)
        selected = [rows[int(index)] for index in indices]
    else:
        selected = list(rows)
    valid_features = []
    columns = []
    for feature in features:
        values = np.asarray([
            float(row.get(feature, float("nan"))) for row in selected
        ], dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.all(values == values[0]):
            continue
        valid_features.append(feature)
        columns.append(_rankdata(values))
    if len(columns) < 2:
        return []
    matrix = np.corrcoef(np.column_stack(columns), rowvar=False)
    result = []
    for left in range(len(valid_features)):
        for right in range(left + 1, len(valid_features)):
            value = float(matrix[left, right])
            if abs(value) >= threshold:
                result.append({
                    "scope": scope,
                    "feature_a": valid_features[left],
                    "feature_b": valid_features[right],
                    "spearman": value,
                    "abs_spearman": abs(value),
                })
    return sorted(result, key=lambda row: row["abs_spearman"], reverse=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tabular_result(result: Mapping[str, Any]) -> dict[str, Any]:
    within = result.get("within_tick_station_ranking") or {}
    ci = result["cluster_bootstrap_ci95"]
    return {
        "scope": result["scope"],
        "target": result["target"],
        "feature": result["feature"],
        "feature_role": result["feature_role"],
        "n": result["n"],
        "runs": result["runs"],
        "pooled_spearman": result["pooled_spearman"],
        "partial_spearman": result["partial_spearman"],
        "per_run_spearman_mean": result["per_run_spearman_mean"],
        "ci95_low": ci[0],
        "ci95_high": ci[1],
        "run_sign_consistency": result["run_sign_consistency"],
        "positive_state_auc": result["positive_state_auc"],
        "within_tick_rank_mean": within.get("mean"),
        "within_tick_rank_groups": within.get("groups"),
    }


def _write_output_hashes(root: Path, paths: Sequence[Path]) -> Path:
    manifest = root / "correlation_outputs.sha256"
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(paths)
    ]
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    expected = {
        f"{arm}_{load}_seed{seed}"
        for arm in ARMS for load in LOADS for seed in SEEDS
    }
    summaries = sorted((root / "runs").glob("*/station_congestion_summary.json"))
    found = {path.parent.name for path in summaries}
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing and not args.allow_incomplete:
        raise RuntimeError(f"missing {len(missing)} frozen runs; first={missing[:5]}")
    if extra:
        raise RuntimeError(f"unexpected run directories: {extra[:5]}")

    station_rows: list[dict[str, Any]] = []
    system_rows: list[dict[str, Any]] = []
    run_audits = []
    for summary_path in summaries:
        summary = _read_json(summary_path)
        if summary.get("schema_version") != RUN_SCHEMA_VERSION:
            raise ValueError(f"wrong run schema: {summary_path}")
        if not bool((summary.get("audit") or {}).get("passed")):
            raise ValueError(f"failed run audit: {summary_path}")
        meta = {
            "run_id": summary_path.parent.name,
            "arm": summary["arm"],
            "load": summary["load"],
            "seed": int(summary["seed"]),
        }
        if meta["seed"] not in SEEDS:
            raise ValueError(f"non-development seed entered analysis: {meta}")
        trace_path = summary_path.parent / summary["outputs"]["trace"]
        trace_rows = 0
        selected_rows = 0
        with trace_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                trace_rows += 1
                payload = json.loads(raw)
                if payload.get("schema_version") != TRACE_SCHEMA_VERSION:
                    raise ValueError(f"wrong trace schema: {trace_path}")
                tick = int(payload["tick"])
                if tick % FRAME_STRIDE != 0:
                    continue
                selected_rows += 1
                system = payload["system"]
                system_rows.append(_flatten_system(meta, tick, system))
                for station in payload["stations"]:
                    station_rows.append(_flatten_station(
                        meta, tick, system, station
                    ))
        if trace_rows != TICKS:
            raise ValueError(
                f"trace row count {trace_rows} != {TICKS}: {trace_path}"
            )
        run_audits.append({
            **meta,
            "trace_rows": trace_rows,
            "analysis_rows": selected_rows,
            "trace_sha256": sha256_file(trace_path),
        })

    station_features = _station_features(station_rows)
    system_features = _system_features(system_rows)
    station_prepared = _PreparedScope(
        station_rows,
        [*station_features, *STATION_TARGETS],
        station_scope=True,
    )
    system_prepared = _PreparedScope(
        system_rows,
        [*system_features, *SYSTEM_TARGETS],
        station_scope=False,
    )
    all_results = []
    for target_index, target in enumerate(STATION_TARGETS):
        for feature_index, feature in enumerate(station_features):
            all_results.append(_correlation_summary_prepared(
                station_prepared,
                feature,
                target,
                bootstrap_repeats=BOOTSTRAP_REPEATS,
                bootstrap_seed=(
                    BOOTSTRAP_SEED + 10000 * target_index + feature_index
                ),
            ))
    station_result_count = len(all_results)
    for target_index, target in enumerate(SYSTEM_TARGETS):
        for feature_index, feature in enumerate(system_features):
            all_results.append(_correlation_summary_prepared(
                system_prepared,
                feature,
                target,
                bootstrap_repeats=BOOTSTRAP_REPEATS,
                bootstrap_seed=(
                    BOOTSTRAP_SEED + 500000 + 10000 * target_index + feature_index
                ),
            ))

    def rank_key(result: Mapping[str, Any]) -> tuple:
        partial = result["partial_spearman"]
        pooled = result["pooled_spearman"]
        score = abs(partial) if math.isfinite(partial) else abs(pooled)
        return (-score, result["feature"])

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in all_results:
        grouped[(result["scope"], result["target"])].append(result)
    rankings = {
        f"{scope}:{target}": [
            _json_safe(result)
            for result in sorted(results, key=rank_key)[:20]
        ]
        for (scope, target), results in grouped.items()
    }
    redundancy = [
        *_redundancy_rows(station_rows, station_features, "station"),
        *_redundancy_rows(system_rows, system_features, "system"),
    ]

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "development_only": True,
        "locked_reference_501_510_used": False,
        "data": {
            "expected_runs": len(expected),
            "analysed_runs": len(summaries),
            "missing_runs": missing,
            "station_rows": len(station_rows),
            "system_rows": len(system_rows),
            "analysis_stride": FRAME_STRIDE,
            "station_features": station_features,
            "system_features": system_features,
            "station_targets": list(STATION_TARGETS),
            "system_targets": list(SYSTEM_TARGETS),
        },
        "interpretation": {
            "same_tick_only": True,
            "future_prediction_target": False,
            "direct_components_are_marked": True,
            "correlation_is_not_action_causality": True,
            "automatic_feature_selection": False,
        },
        "top_rankings": rankings,
        "run_audits": run_audits,
        "redundant_feature_pairs_abs_spearman_ge_0_90": _json_safe(redundancy),
    }
    report_path = root / "station_congestion_correlation_report.json"
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    station_csv = root / "station_metric_correlations.csv"
    system_csv = root / "system_metric_correlations.csv"
    redundancy_csv = root / "metric_redundancy.csv"
    _write_csv(
        station_csv,
        [_json_safe(_tabular_result(result)) for result in all_results[:station_result_count]],
    )
    _write_csv(
        system_csv,
        [_json_safe(_tabular_result(result)) for result in all_results[station_result_count:]],
    )
    _write_csv(redundancy_csv, [_json_safe(row) for row in redundancy])
    manifest = _write_output_hashes(
        root, [report_path, station_csv, system_csv, redundancy_csv]
    )
    print(f"[complete] station/system congestion correlation audit: {root}")
    print(f"report={report_path}")
    print(f"outputs={manifest}")


if __name__ == "__main__":
    main()
