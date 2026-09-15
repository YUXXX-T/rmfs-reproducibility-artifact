"""Diagnose cross-seed scale shift in the analytic ``Delta L_work`` signal.

This module is deliberately diagnostic-only.  It never changes the frozen
Layer-3 certificate, excludes a seed, or promotes an alternative transform to
an online score.  Its job is to reproduce a failed held-out seed, locate the
responsible candidate groups, and compare scale-invariant hypotheses before a
new protocol is designed on untouched seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from WorldModel.evaluation.evaluate_lyapunov_validity import (
    _attach_centred_component_deltas,
    _build_folds,
    _centred_rows,
    _component_delta,
    _cross_validated_linear_diagnostics,
    _expand_paths,
    _field,
    _groups,
    _incremental_outcome_report,
    _infer_load,
    _load_samples,
    _quantiles,
    _rankdata,
    _score_with_world_model,
)


SCHEMA_VERSION = "lyapunov_work_scale_shift_diagnostic_v1"
TRANSFORM_FIELDS = {
    "raw_centered": "_work_transform.raw_centered",
    "relative_start": "_work_transform.relative_start",
    "group_range": "_work_transform.group_range",
    "group_rank": "_work_transform.group_rank",
    "robust_tanh": "_work_transform.robust_tanh",
}


def _finite(value) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(values: Sequence[float]) -> dict:
    array = np.asarray([
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ], dtype=float)
    if not array.size:
        return {
            **_quantiles(()),
            "p01": None,
            "p99": None,
            "abs_p95": None,
            "abs_p99": None,
            "abs_max": None,
        }
    absolute = np.abs(array)
    return {
        **_quantiles(array.tolist()),
        "p01": float(np.quantile(array, 0.01)),
        "p99": float(np.quantile(array, 0.99)),
        "abs_p95": float(np.quantile(absolute, 0.95)),
        "abs_p99": float(np.quantile(absolute, 0.99)),
        "abs_max": float(absolute.max()),
    }


def _centre(values: Sequence[Optional[float]]) -> list[Optional[float]]:
    finite = [float(value) for value in values if value is not None]
    mean = float(np.mean(finite)) if finite else None
    return [
        float(value - mean) if value is not None and mean is not None else None
        for value in values
    ]


def _attach_work_transforms(rows: Sequence[dict]) -> dict[str, str]:
    component_fields = _attach_centred_component_deltas(rows)
    raw_centred_field = component_fields["work"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)

    for members in grouped.values():
        raw = [_component_delta(row["sample"], "work") for row in members]
        start = [
            _field(row["sample"], "lyapunov_l0_start.components.work")
            for row in members
        ]
        end = [
            _field(row["sample"], "lyapunov_l0_end.components.work")
            for row in members
        ]
        centred = [row.get(raw_centred_field) for row in members]

        relative = []
        for value, base in zip(raw, start):
            if value is None or base is None:
                relative.append(None)
            else:
                relative.append(float(value / max(abs(base), 1e-12)))
        relative = _centre(relative)

        finite_raw = np.asarray([
            value for value in raw if value is not None
        ], dtype=float)
        raw_range = (
            float(finite_raw.max() - finite_raw.min())
            if finite_raw.size else 0.0
        )
        group_range = [
            float(value / raw_range)
            if value is not None and raw_range > 1e-15 else 0.0
            if value is not None else None
            for value in centred
        ]

        rank_values: list[Optional[float]] = [None] * len(raw)
        finite_indices = [index for index, value in enumerate(raw) if value is not None]
        if len(finite_indices) >= 2:
            ranks = _rankdata(np.asarray([raw[index] for index in finite_indices]))
            denominator = float(len(finite_indices) - 1)
            for index, rank in zip(finite_indices, ranks):
                rank_values[index] = float(
                    2.0 * (rank - 1.0) / denominator - 1.0
                )
        elif finite_indices:
            rank_values[finite_indices[0]] = 0.0
        rank_values = _centre(rank_values)

        robust_values: list[Optional[float]] = [None] * len(raw)
        if finite_raw.size:
            median = float(np.median(finite_raw))
            mad = float(np.median(np.abs(finite_raw - median)))
            q25, q75 = np.quantile(finite_raw, (0.25, 0.75))
            robust_scale = max(
                1.4826 * mad,
                float((q75 - q25) / 1.349),
            )
            if robust_scale <= 1e-15 and raw_range > 1e-15:
                robust_scale = 0.5 * raw_range
            for index, value in enumerate(raw):
                if value is not None:
                    robust_values[index] = (
                        float(np.tanh((value - median) / robust_scale))
                        if robust_scale > 1e-15 else 0.0
                    )
        robust_values = _centre(robust_values)

        for index, row in enumerate(members):
            row[TRANSFORM_FIELDS["raw_centered"]] = centred[index]
            row[TRANSFORM_FIELDS["relative_start"]] = relative[index]
            row[TRANSFORM_FIELDS["group_range"]] = group_range[index]
            row[TRANSFORM_FIELDS["group_rank"]] = rank_values[index]
            row[TRANSFORM_FIELDS["robust_tanh"]] = robust_values[index]
            row["_work_raw_delta"] = raw[index]
            row["_work_start"] = start[index]
            row["_work_end"] = end[index]
    return dict(TRANSFORM_FIELDS)


def _seed(row: Mapping) -> Optional[int]:
    value = row.get("sample", {}).get("simulation_seed")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scale_summary(rows: Sequence[Mapping]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    group_ranges = []
    group_sizes = []
    start_spreads = []
    for members in grouped.values():
        raw = [row.get("_work_raw_delta") for row in members]
        raw = [value for value in raw if value is not None]
        if raw:
            group_ranges.append(float(max(raw) - min(raw)))
        group_sizes.append(len(members))
        starts = [row.get("_work_start") for row in members]
        starts = [value for value in starts if value is not None]
        if starts:
            start_spreads.append(float(max(starts) - min(starts)))
    return {
        "samples": len(rows),
        "groups": len(grouped),
        "group_size": _distribution(group_sizes),
        "raw_work_delta": _distribution([
            row.get("_work_raw_delta") for row in rows
        ]),
        "centred_work_delta": _distribution([
            row.get(TRANSFORM_FIELDS["raw_centered"]) for row in rows
        ]),
        "start_work_potential": _distribution([
            row.get("_work_start") for row in rows
        ]),
        "end_work_potential": _distribution([
            row.get("_work_end") for row in rows
        ]),
        "relative_start": _distribution([
            row.get(TRANSFORM_FIELDS["relative_start"]) for row in rows
        ]),
        "within_group_raw_range": _distribution(group_ranges),
        "within_group_start_work_spread": _distribution(start_spreads),
    }


def _focus_reference_comparison(
    rows: Sequence[Mapping],
    *,
    focus_seed: int,
) -> dict:
    def comparison(focus: Sequence[Mapping], reference: Sequence[Mapping]) -> dict:
        focus_values = np.asarray([
            row[TRANSFORM_FIELDS["raw_centered"]] for row in focus
            if row.get(TRANSFORM_FIELDS["raw_centered"]) is not None
        ], dtype=float)
        reference_values = np.asarray([
            row[TRANSFORM_FIELDS["raw_centered"]] for row in reference
            if row.get(TRANSFORM_FIELDS["raw_centered"]) is not None
        ], dtype=float)
        if not focus_values.size or not reference_values.size:
            return {"status": "INSUFFICIENT_ROWS"}
        reference_min = float(reference_values.min())
        reference_max = float(reference_values.max())
        reference_abs = max(abs(reference_min), abs(reference_max))
        focus_abs = float(np.max(np.abs(focus_values)))
        reference_std = float(reference_values.std())
        outside = (focus_values < reference_min) | (focus_values > reference_max)
        return {
            "status": "COMPARED",
            "focus": _distribution(focus_values.tolist()),
            "reference": _distribution(reference_values.tolist()),
            "focus_to_reference_abs_range_ratio": (
                focus_abs / reference_abs if reference_abs > 0.0 else None
            ),
            "focus_max_abs_in_reference_standard_deviations": (
                focus_abs / reference_std if reference_std > 0.0 else None
            ),
            "focus_rows_outside_reference_range": int(outside.sum()),
            "focus_rows_outside_reference_rate": float(outside.mean()),
        }

    focus_rows = [row for row in rows if _seed(row) == focus_seed]
    reference_rows = [row for row in rows if _seed(row) != focus_seed]
    loads = sorted({str(row["load"]) for row in rows})
    return {
        "focus_seed": int(focus_seed),
        "overall": comparison(focus_rows, reference_rows),
        "by_load": {
            load: comparison(
                [row for row in focus_rows if str(row["load"]) == load],
                [row for row in reference_rows if str(row["load"]) == load],
            )
            for load in loads
        },
    }


def _candidate_record(
    row: Mapping,
    transform_fields: Mapping[str, str],
    *,
    wm_field: str,
) -> dict:
    sample = row["sample"]
    return {
        "candidate_key": str(sample.get("candidate_key", "unknown")),
        "candidate_info": _jsonable(sample.get("candidate_info")),
        "work_delta_raw": row.get("_work_raw_delta"),
        "work_delta_centred": row.get(transform_fields["raw_centered"]),
        "work_start": row.get("_work_start"),
        "work_end": row.get("_work_end"),
        "all_component_deltas": {
            name: _component_delta(sample, name)
            for name in ("work", "station", "traffic", "stall", "arrival")
        },
        "work_ledger_audit": _work_ledger_audit(sample),
        "productive_progress": _jsonable(sample.get("lyapunov_l0_progress")),
        "wm_score_centred": row.get(wm_field),
        "realized_cost_centred": row.get("realized_cost"),
        "transforms": {
            name: row.get(field) for name, field in transform_fields.items()
        },
    }


def _work_ledger_audit(sample: Mapping) -> dict:
    start = sample.get("lyapunov_l0_start")
    end = sample.get("lyapunov_l0_end")
    if not isinstance(start, Mapping) or not isinstance(end, Mapping):
        return {"status": "MISSING_ENDPOINT_SNAPSHOT"}
    start_work = start.get("station_work")
    end_work = end.get("station_work")
    if not isinstance(start_work, Mapping) or not isinstance(end_work, Mapping):
        return {"status": "MISSING_STATION_WORK"}
    capacity = _finite(start.get("work_capacity"))
    end_capacity = _finite(end.get("work_capacity"))
    config = sample.get("lyapunov_l0_config")
    weight = (
        _finite(config.get("work_weight"))
        if isinstance(config, Mapping) else 1.0
    )
    if weight is None:
        weight = 1.0
    if capacity is None or capacity <= 0.0:
        return {"status": "INVALID_WORK_CAPACITY", "capacity": capacity}
    stations = sorted({str(key) for key in start_work} | {
        str(key) for key in end_work
    })

    def lookup(mapping: Mapping, key: str) -> float:
        for candidate in (key, int(key) if key.lstrip("-").isdigit() else key):
            if candidate in mapping:
                return float(mapping[candidate])
        return 0.0

    contributions = {}
    reconstructed = 0.0
    for station in stations:
        before = lookup(start_work, station)
        after = lookup(end_work, station)
        delta = after - before
        contribution = float(
            weight / (capacity ** 2)
            * (before * delta + 0.5 * delta * delta)
        )
        reconstructed += contribution
        contributions[station] = {
            "start_work": before,
            "end_work": after,
            "delta_work": delta,
            "delta_L_work_contribution": contribution,
        }
    stored = _component_delta(sample, "work")
    return {
        "status": "AUDITED",
        "work_weight": weight,
        "start_capacity": capacity,
        "end_capacity": end_capacity,
        "capacity_unchanged": (
            end_capacity is not None and abs(end_capacity - capacity) <= 1e-12
        ),
        "stored_delta_L_work": stored,
        "reconstructed_delta_L_work": reconstructed,
        "closure_error": (
            reconstructed - stored if stored is not None else None
        ),
        "station_contributions": contributions,
    }


def _prediction_error_attribution(
    rows: Sequence[Mapping],
    *,
    wm_field: str,
    outcome_field: str,
    predictor_field: str,
    transform_name: str,
    focus_seed: int,
    limit: int,
) -> dict:
    usable = [
        row for row in rows
        if row.get(wm_field) is not None
        and row.get(outcome_field) is not None
        and row.get(predictor_field) is not None
    ]
    if len(usable) < 6:
        return {"status": "INSUFFICIENT_ROWS", "n": len(usable)}
    wm = np.asarray([row[wm_field] for row in usable], dtype=float)
    outcome = np.asarray([row[outcome_field] for row in usable], dtype=float)
    predictor = np.asarray([row[predictor_field] for row in usable], dtype=float)
    folds = _build_folds(usable)
    base_fit = _cross_validated_linear_diagnostics(
        wm[:, None],
        outcome,
        folds,
        feature_names=["wm"],
        rows=usable,
    )
    augmented_fit = _cross_validated_linear_diagnostics(
        np.column_stack((wm, predictor)),
        outcome,
        folds,
        feature_names=["wm", f"candidate:{transform_name}"],
        rows=usable,
    )
    if base_fit is None or augmented_fit is None:
        return {"status": "CROSS_VALIDATION_FAILED", "n": len(usable)}
    base_prediction = base_fit[0]
    augmented_prediction = augmented_fit[0]
    base_squared = (outcome - base_prediction) ** 2
    augmented_squared = (outcome - augmented_prediction) ** 2
    excess = augmented_squared - base_squared

    by_seed = defaultdict(list)
    by_load = defaultdict(list)
    by_seed_load = defaultdict(list)
    grouped = defaultdict(list)
    for index, row in enumerate(usable):
        cluster = str(row["cluster"])
        load = str(row["load"])
        by_seed[cluster].append(index)
        by_load[load].append(index)
        by_seed_load[(cluster, load)].append(index)
        grouped[row["group"]].append(index)

    def aggregate(indices: Sequence[int]) -> dict:
        index = np.asarray(indices, dtype=int)
        return {
            "n": int(index.size),
            "wm_only_sse": float(base_squared[index].sum()),
            "wm_plus_transform_sse": float(augmented_squared[index].sum()),
            "excess_sse_positive_is_harm": float(excess[index].sum()),
            "wm_only_rmse": float(np.sqrt(base_squared[index].mean())),
            "wm_plus_transform_rmse": float(
                np.sqrt(augmented_squared[index].mean())
            ),
        }

    group_records = []
    for key, indices in grouped.items():
        first = usable[indices[0]]
        sample = first["sample"]
        record = {
            "seed": _seed(first),
            "load": str(first["load"]),
            "run_id": str(sample.get("run_id", "unknown")),
            "candidate_group_id": str(
                sample.get("candidate_group_id", key)
            ),
            "decision_tick": sample.get("decision_tick"),
            "fixed_context": _jsonable(sample.get("fixed_context")),
            **aggregate(indices),
            "candidates": [],
        }
        for index in indices:
            row = usable[index]
            record["candidates"].append({
                "candidate_key": str(
                    row["sample"].get("candidate_key", "unknown")
                ),
                "candidate_info": _jsonable(
                    row["sample"].get("candidate_info")
                ),
                "target_centred": float(outcome[index]),
                "wm_prediction": float(base_prediction[index]),
                "wm_plus_transform_prediction": float(
                    augmented_prediction[index]
                ),
                "wm_squared_error": float(base_squared[index]),
                "wm_plus_transform_squared_error": float(
                    augmented_squared[index]
                ),
                "excess_squared_error_positive_is_harm": float(excess[index]),
                "predictor": float(predictor[index]),
                "raw_work_delta": row.get("_work_raw_delta"),
                "centred_work_delta": row.get(
                    TRANSFORM_FIELDS["raw_centered"]
                ),
            })
        group_records.append(record)
    harmful = sorted(
        group_records,
        key=lambda row: row["excess_sse_positive_is_harm"],
        reverse=True,
    )
    helpful = sorted(
        group_records,
        key=lambda row: row["excess_sse_positive_is_harm"],
    )
    total = aggregate(range(len(usable)))
    total_excess = total["excess_sse_positive_is_harm"]
    focus_key = f"seed={focus_seed}"
    focus = aggregate(by_seed.get(focus_key, ())) if focus_key in by_seed else None
    return {
        "status": "ATTRIBUTED",
        "interpretation": (
            "positive excess SSE means the transform worsened held-out WM error"
        ),
        "total": total,
        "focus_seed": focus,
        "focus_share_of_total_excess_sse": (
            focus["excess_sse_positive_is_harm"] / total_excess
            if focus is not None and total_excess > 0.0 else None
        ),
        "by_seed": {
            seed: aggregate(indices) for seed, indices in sorted(by_seed.items())
        },
        "by_load": {
            load: aggregate(indices) for load, indices in sorted(by_load.items())
        },
        "by_seed_and_load": {
            seed: {
                load: aggregate(by_seed_load[(seed, load)])
                for load in sorted({
                    pair_load for pair_seed, pair_load in by_seed_load
                    if pair_seed == seed
                })
            }
            for seed in sorted(by_seed)
        },
        "top_harmful_groups": harmful[:limit],
        "top_helpful_groups": helpful[:limit],
    }


def _extreme_groups(
    rows: Sequence[Mapping],
    *,
    transform_fields: Mapping[str, str],
    wm_field: str,
    focus_seed: int,
    limit: int,
) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    records = []
    for key, members in grouped.items():
        centred = [
            row.get(transform_fields["raw_centered"]) for row in members
            if row.get(transform_fields["raw_centered"]) is not None
        ]
        raw = [
            row.get("_work_raw_delta") for row in members
            if row.get("_work_raw_delta") is not None
        ]
        if not centred or not raw:
            continue
        first = members[0]
        sample = first["sample"]
        records.append({
            "seed": _seed(first),
            "load": str(first["load"]),
            "run_id": str(sample.get("run_id", "unknown")),
            "source_path": str(sample.get("_source_path", "unknown")),
            "candidate_group_id": str(sample.get("candidate_group_id", key)),
            "decision_tick": sample.get("decision_tick"),
            "fixed_context": _jsonable(sample.get("fixed_context")),
            "candidate_count": len(members),
            "work_raw_range": float(max(raw) - min(raw)),
            "max_abs_centred_work": float(max(abs(value) for value in centred)),
            "start_work_spread": float(
                max(row.get("_work_start") for row in members)
                - min(row.get("_work_start") for row in members)
            ) if all(row.get("_work_start") is not None for row in members) else None,
            "candidates": sorted(
                [
                    _candidate_record(
                        row,
                        transform_fields,
                        wm_field=wm_field,
                    )
                    for row in members
                ],
                key=lambda item: (
                    item["work_delta_centred"] is None,
                    item["work_delta_centred"] or 0.0,
                ),
            ),
        })
    records.sort(key=lambda item: item["max_abs_centred_work"], reverse=True)
    focus = [item for item in records if item["seed"] == int(focus_seed)]
    reference = [item for item in records if item["seed"] != int(focus_seed)]
    by_load = {}
    for load in sorted({item["load"] for item in focus}):
        by_load[load] = [
            item for item in focus if item["load"] == load
        ][:limit]
    return {
        "focus_seed_top_groups": focus[:limit],
        "focus_seed_top_groups_by_load": by_load,
        "reference_top_groups": reference[:limit],
    }


def _compact_transform_summary(report: Mapping, focus_seed: int) -> dict:
    by_seed = report.get("by_seed", {})
    focus_key = f"seed={focus_seed}"
    seed_improvements = {
        str(seed): row.get("normalised_rmse_improvement")
        for seed, row in by_seed.items()
    }
    finite_seed = {
        seed: value for seed, value in seed_improvements.items()
        if value is not None
    }
    worst_seed = (
        min(finite_seed, key=finite_seed.get) if finite_seed else None
    )
    ranking = report.get("ranking", {})
    base = ranking.get("wm_only", {})
    augmented = ranking.get("wm_plus_predictor", {})
    fold = next((
        row for row in report.get("wm_plus_candidate_fold_diagnostics", [])
        if focus_key in row.get("held_out_clusters", [])
    ), None)
    return {
        "status": report.get("status"),
        "normalised_rmse_improvement": report.get(
            "normalised_rmse_improvement"
        ),
        "ci95": report.get(
            "normalised_rmse_improvement_ci95_cluster_bootstrap"
        ),
        "focus_seed_improvement": seed_improvements.get(focus_key),
        "worst_seed": worst_seed,
        "worst_seed_improvement": finite_seed.get(worst_seed),
        "by_seed": seed_improvements,
        "by_load": {
            load: row.get("normalised_rmse_improvement")
            for load, row in report.get("by_load", {}).items()
        },
        "top1_accuracy_improvement": ranking.get(
            "top1_accuracy_improvement"
        ),
        "pairwise_concordance_improvement": ranking.get(
            "pairwise_concordance_improvement"
        ),
        "normalised_regret": {
            "wm_only": base.get("normalised_selection_regret", {}).get("mean"),
            "wm_plus_transform": augmented.get(
                "normalised_selection_regret", {}
            ).get("mean"),
        },
        "focus_fold": fold,
    }


def _formal_reproduction_audit(
    formal_report: Optional[Mapping],
    raw_report: Mapping,
    rows: Sequence[Mapping],
    *,
    focus_seed: int,
    tolerance: float = 1e-6,
) -> Optional[dict]:
    if not isinstance(formal_report, Mapping):
        return None
    expected = (
        formal_report.get("layer3_incremental_information", {})
        .get("independent_outcomes", {})
        .get("realized_cost", {})
    )
    expected_work = (
        formal_report.get("layer2_action_controllable_drift", {})
        .get("within_group_component_attribution", {})
        .get("work", {})
        .get("within_group_delta", {})
    )
    observed_values = [
        row.get(TRANSFORM_FIELDS["raw_centered"]) for row in rows
        if row.get(TRANSFORM_FIELDS["raw_centered"]) is not None
    ]
    observed_work = _distribution(observed_values)
    observed_focus = raw_report.get("by_seed", {}).get(
        f"seed={focus_seed}", {}
    ).get("normalised_rmse_improvement")
    checks = {
        "samples": len(rows) == int(formal_report.get("samples", -1)),
        "groups": len({row["group"] for row in rows}) == int(
            formal_report.get("candidate_groups", -1)
        ),
        "raw_overall_improvement": (
            expected.get("normalised_rmse_improvement") is not None
            and raw_report.get("normalised_rmse_improvement") is not None
            and abs(
                raw_report["normalised_rmse_improvement"]
                - expected["normalised_rmse_improvement"]
            ) <= tolerance
        ),
        "focus_seed_improvement": (
            observed_focus is not None
            and expected.get("by_seed", {}).get(
                f"seed={focus_seed}", {}
            ).get("normalised_rmse_improvement") is not None
            and abs(
                observed_focus
                - expected["by_seed"][f"seed={focus_seed}"]
                ["normalised_rmse_improvement"]
            ) <= tolerance
        ),
        "work_min": (
            expected_work.get("min") is not None
            and abs(observed_work["min"] - expected_work["min"]) <= tolerance
        ),
        "work_max": (
            expected_work.get("max") is not None
            and abs(observed_work["max"] - expected_work["max"]) <= tolerance
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "formal_protocol_sha256": (
            formal_report.get("layer3_incremental_information", {})
            .get("formal_candidate_contract", {})
            .get("protocol", {})
            .get("protocol_sha256")
        ),
        "expected_raw_improvement": expected.get(
            "normalised_rmse_improvement"
        ),
        "observed_raw_improvement": raw_report.get(
            "normalised_rmse_improvement"
        ),
        "expected_focus_seed_improvement": expected.get("by_seed", {}).get(
            f"seed={focus_seed}", {}
        ).get("normalised_rmse_improvement"),
        "observed_focus_seed_improvement": observed_focus,
    }


def diagnose_work_scale_shift(
    samples: Sequence[Mapping],
    *,
    wm_score_field: str,
    focus_seed: int = 416,
    bootstrap_repeats: int = 1000,
    placebo_repeats: int = 200,
    random_seed: int = 20260715,
    top_groups: int = 20,
    formal_report: Optional[Mapping] = None,
) -> dict:
    groups = _groups(samples)
    fields = [wm_score_field, "lyapunov_l0_delta", "realized_cost"]
    rows, missing = _centred_rows(groups, fields)
    transform_fields = _attach_work_transforms(rows)
    seeds = sorted({seed for seed in map(_seed, rows) if seed is not None})
    loads = sorted({str(row["load"]) for row in rows})
    if int(focus_seed) not in seeds:
        raise ValueError(f"focus seed {focus_seed} is absent; available={seeds}")

    transform_reports = {}
    error_attribution = {}
    for index, (name, field) in enumerate(transform_fields.items()):
        report = _incremental_outcome_report(
            rows,
            wm_field=wm_score_field,
            outcome_field="realized_cost",
            predictor_field=field,
            candidate_label=name,
            minimum_independent_clusters=2,
            bootstrap_repeats=bootstrap_repeats,
            placebo_repeats=placebo_repeats,
            random_seed=random_seed + index * 100003,
        )
        report["role"] = "DEVELOPMENT_DIAGNOSTIC_ONLY"
        transform_reports[name] = report
        error_attribution[name] = _prediction_error_attribution(
            rows,
            wm_field=wm_score_field,
            outcome_field="realized_cost",
            predictor_field=field,
            transform_name=name,
            focus_seed=int(focus_seed),
            limit=int(top_groups),
        )

    by_seed_load = {}
    for seed in seeds:
        by_seed_load[str(seed)] = {
            load: _scale_summary([
                row for row in rows
                if _seed(row) == seed and str(row["load"]) == load
            ])
            for load in loads
        }
    focus_rows = [row for row in rows if _seed(row) == int(focus_seed)]
    reference_rows = [row for row in rows if _seed(row) != int(focus_seed)]
    compact = {
        name: _compact_transform_summary(report, int(focus_seed))
        for name, report in transform_reports.items()
    }
    raw = transform_reports["raw_centered"]
    reproduction = _formal_reproduction_audit(
        formal_report,
        raw,
        rows,
        focus_seed=int(focus_seed),
    )
    raw_focus = compact["raw_centered"]
    other_seed_values = [
        value for seed, value in raw_focus["by_seed"].items()
        if seed != f"seed={focus_seed}" and value is not None
    ]
    scale_comparison = _focus_reference_comparison(
        rows,
        focus_seed=int(focus_seed),
    )
    raw_failure_reproduced = bool(
        raw_focus.get("focus_seed_improvement") is not None
        and raw_focus["focus_seed_improvement"] < 0.0
        and other_seed_values
        and float(np.median(other_seed_values)) > 0.0
        and scale_comparison["overall"].get(
            "focus_rows_outside_reference_range", 0
        ) > 0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "role": "POST_CERTIFICATION_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "semantics": {
            "changes_frozen_certificate": False,
            "excludes_focus_seed": False,
            "certifies_alternative_transform": False,
            "permits_online_use": False,
            "purpose": (
                "reproduce and localise cross-seed work-drift scale shift "
                "before freezing a new signal on untouched seeds"
            ),
        },
        "parameters": {
            "wm_score_field": wm_score_field,
            "focus_seed": int(focus_seed),
            "bootstrap_repeats": int(bootstrap_repeats),
            "placebo_repeats": int(placebo_repeats),
            "random_seed": int(random_seed),
            "top_groups": int(top_groups),
        },
        "samples": len(rows),
        "groups": len(groups),
        "seeds": seeds,
        "loads": loads,
        "missing_values": missing,
        "raw_scale_failure_reproduced": raw_failure_reproduced,
        "overall_scale": _scale_summary(rows),
        "focus_scale": _scale_summary(focus_rows),
        "reference_scale": _scale_summary(reference_rows),
        "focus_vs_reference": scale_comparison,
        "scale_by_seed_and_load": by_seed_load,
        "transform_summary": compact,
        "transform_evaluation": transform_reports,
        "prediction_error_attribution": error_attribution,
        "extreme_groups": _extreme_groups(
            rows,
            transform_fields=transform_fields,
            wm_field=wm_score_field,
            focus_seed=int(focus_seed),
            limit=int(top_groups),
        ),
        "formal_reproduction_audit": reproduction,
        "next_decision_boundary": {
            "if_ledger_or_endpoint_violation_is_found": (
                "fix collection/functional code; keep this certificate failed "
                "and use new untouched seeds after the fix"
            ),
            "if_focus_scale_shift_is_physically_valid": (
                "freeze one bounded or scale-invariant work transform; treat "
                "411-420 as development and certify on untouched seeds"
            ),
            "do_not": [
                "delete seed 416 post hoc",
                "lower the frozen Layer-3 acceptance contract",
                "promote the best transform in this report without new seeds",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--world-model-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--focus-seed", type=int, default=416)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--placebo-repeats", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260715)
    parser.add_argument("--top-groups", type=int, default=20)
    parser.add_argument("--formal-report", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.bootstrap_repeats <= 0 or args.placebo_repeats <= 0:
        raise SystemExit("bootstrap/placebo repeats must be positive")
    if args.top_groups <= 0:
        raise SystemExit("--top-groups must be positive")
    output = Path(args.output)
    if output.exists():
        raise SystemExit(
            f"output already exists: {output}; keep it unchanged or choose a new name"
        )

    paths = _expand_paths(args.data)
    samples = _load_samples(paths)
    samples, wm_scoring = _score_with_world_model(
        samples,
        args.world_model_checkpoint,
        device=args.device,
    )
    formal_report = None
    if args.formal_report:
        formal_report = json.loads(
            Path(args.formal_report).read_text(encoding="utf-8")
        )
    report = diagnose_work_scale_shift(
        samples,
        wm_score_field="_validation_wm_score",
        focus_seed=args.focus_seed,
        bootstrap_repeats=args.bootstrap_repeats,
        placebo_repeats=args.placebo_repeats,
        random_seed=args.random_seed,
        top_groups=args.top_groups,
        formal_report=formal_report,
    )
    report["world_model_scoring"] = wm_scoring
    report["source_files"] = paths
    report["formal_report"] = (
        str(Path(args.formal_report).resolve()) if args.formal_report else None
    )
    report["diagnostic_implementation"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256_file(__file__),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"saved: {output}")
    print("role:", report["role"])
    print("raw scale failure reproduced:", report["raw_scale_failure_reproduced"])
    audit = report.get("formal_reproduction_audit")
    print("formal reproduction audit:", audit.get("passed") if audit else None)
    for name, summary in report["transform_summary"].items():
        regret = summary["normalised_regret"]
        print(
            f"{name}: overall={summary['normalised_rmse_improvement']} "
            f"focus={summary['focus_seed_improvement']} "
            f"top1={summary['top1_accuracy_improvement']} "
            f"pairwise={summary['pairwise_concordance_improvement']} "
            f"regret={regret['wm_only']}->{regret['wm_plus_transform']}"
        )


if __name__ == "__main__":
    main()
