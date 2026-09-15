"""Validate a Lyapunov functional with a scale-invariant five-layer protocol.

The protocol deliberately separates five questions which were conflated by
the historical gap-based oracle report:

1. Does ``L(s)`` satisfy its state-potential and bookkeeping contract?
2. Do actions create measurable *within-context* differences in drift?
3. Does analytic drift add information after a real World Model score?
4. Can the World Model estimate the action-dependent drift?
5. Does a frozen policy remain stable under normal stochastic arrivals?

No raw ``Delta L`` magnitude is used as a validity gate.  All action tests
are based on group-centred ranks, normalised error, held-out prediction, or
cluster-bootstrap uncertainty.  ``P_prod`` is reported only as ledger-coupled
evidence because it shares the unfinished-work ledger with ``L_work``; it is
never the primary evidence for online auxiliary value.

Layer 5 is produced by ``validate_lyapunov_closed_loop.py`` and can be attached
with ``--closed-loop-report``.  Keeping the closed-loop experiment separate
prevents an evaluated policy failure from being mislabelled as a functional
failure.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from WorldModel.evaluation.evaluate_lyapunov_oracle import (
    CURRENT_L0_COLLECTION_SCHEMA,
    L0_COMPONENT_NAMES,
    audit_analytic_invariants,
    audit_isolated_semantics,
)


SCHEMA_VERSION = "lyapunov_five_layer_validation_v2"
LAYER3_DIAGNOSTIC_SCHEMA_VERSION = (
    "lyapunov_layer3_load_component_diagnostic_v2"
)
LAYER3_FORMAL_CANDIDATE_SCHEMA_VERSION = (
    "lyapunov_layer3_formal_candidate_v1"
)
LAYER3_FORMAL_WORK_GROUP_RANGE_SCHEMA_VERSION = (
    "lyapunov_layer3_formal_work_group_range_v2"
)
WORK_GROUP_RANGE_FIELD = "_work_transform.group_range"
WORK_GROUP_RANGE_RAW_DELTA_FIELD = "_work_transform.raw_delta"
WORK_GROUP_RANGE_RAW_RANGE_FIELD = "_work_transform.raw_range"
FORMAL_WORK_GROUP_RANGE_CERTIFICATION_SEEDS = tuple(range(421, 431))
FORMAL_WORK_ONLY_REQUIRED_SECONDARY_OUTCOMES = (
    "_validation_outcomes.wait_or_stall_mean",
    "_validation_outcomes.average_excess_delay_mean",
    "_validation_outcomes.deadlock_risk_max",
)
FORMAL_WORK_ONLY_GUARDRAIL_OUTCOMES = (
    "_validation_outcomes.bottleneck_cvar_mean",
    "_validation_outcomes.completed_orders_cost",
    "rollout_blocked_moves",
)
FORMAL_WORK_ONLY_PRIMARY_OUTCOME = "realized_cost"
FORMAL_WORK_ONLY_REQUIRED_LOADS = ("low", "mid", "high")
FORMAL_WORK_ONLY_MINIMUM_SEED_CLUSTERS = 10
FORMAL_WORK_ONLY_MINIMUM_SUPPORTIVE_SECONDARY_OUTCOMES = 2
FORMAL_WORK_ONLY_BOOTSTRAP_REPEATS = 5000
FORMAL_WORK_ONLY_PLACEBO_REPEATS = 2000
FORMAL_WORK_ONLY_RANDOM_SEED = 20260715
# These are coverage/resolution guards for outcome labels, not a Delta-L
# action gap.  The simulator labels are stored as float32, so sub-resolution
# differences must not be amplified into enormous NRMSE values.
LAYER3_FLOAT32_RESOLUTION_MULTIPLIER = 8.0
LAYER3_MIN_EFFECTIVE_OUTCOME_GROUPS = 5
DEFAULT_OUTCOME_FIELDS = (
    "realized_cost",
    "_validation_outcomes.wait_or_stall_mean",
    "_validation_outcomes.average_excess_delay_mean",
    "_validation_outcomes.station_queue_delta_mean",
    "_validation_outcomes.station_load_imbalance_mean",
    "_validation_outcomes.bottleneck_cvar_mean",
    "_validation_outcomes.completed_orders_cost",
    "_validation_outcomes.deadlock_risk_max",
    "rollout_blocked_moves",
    "rollout_vertex_conflicts",
    "rollout_swap_conflicts",
)
DEPENDENT_LEDGER_FIELD = "lyapunov_l0_progress.productive_total"


def _formal_work_only_protocol() -> dict:
    """Return the immutable, machine-readable Layer-3 preregistration."""
    specification = {
        "schema_version": LAYER3_FORMAL_CANDIDATE_SCHEMA_VERSION,
        "frozen": True,
        "candidate_component": "work",
        "candidate_predictor": "within_candidate_group_centred_Delta_L_work",
        "state_potential_remains": (
            "L_work + L_station + L_arrival; selecting work here does not "
            "redefine the Layer-1 state functional"
        ),
        "primary_outcome": FORMAL_WORK_ONLY_PRIMARY_OUTCOME,
        "required_secondary_outcomes": list(
            FORMAL_WORK_ONLY_REQUIRED_SECONDARY_OUTCOMES
        ),
        "guardrail_outcomes": list(FORMAL_WORK_ONLY_GUARDRAIL_OUTCOMES),
        "required_loads": list(FORMAL_WORK_ONLY_REQUIRED_LOADS),
        "minimum_independent_seed_clusters": (
            FORMAL_WORK_ONLY_MINIMUM_SEED_CLUSTERS
        ),
        "minimum_supportive_secondary_outcomes": (
            FORMAL_WORK_ONLY_MINIMUM_SUPPORTIVE_SECONDARY_OUTCOMES
        ),
        "resampling": {
            "bootstrap_repeats": FORMAL_WORK_ONLY_BOOTSTRAP_REPEATS,
            "placebo_repeats": FORMAL_WORK_ONLY_PLACEBO_REPEATS,
            "random_seed": FORMAL_WORK_ONLY_RANDOM_SEED,
        },
        "validation_split": (
            "leave-one-simulator-seed-out; the same seed across all loads "
            "remains in one fold"
        ),
        "world_model_score_contract": {
            "required_source": (
                "checkpoint-scored inside evaluate_lyapunov_validity"
            ),
            "heuristic_cost_is_forbidden": True,
            "checkpoint_must_not_train_on_certification_seeds": True,
            "checkpoint_hash_is_frozen_by_collection_manifest": True,
        },
        "acceptance_contract": {
            "data": [
                "at least ten independent seed clusters",
                "every evaluated seed contains paired low/mid/high arms",
                "every seed/load pair has exactly one run arm and explicit simulation_seed provenance",
                "Delta L_work has effective within-group variation in every load",
            ],
            "primary": [
                "overall NRMSE improvement is positive with seed-cluster CI95 lower bound above zero",
                "observed improvement exceeds within-group permutation-placebo p95",
                "each required load has positive improvement with seed-cluster CI95 lower bound above zero",
                "each held-out seed has positive point improvement",
                "every fold has a positive standardised work coefficient",
                "Top-1 accuracy and pairwise concordance improve",
                "mean normalised selection regret decreases",
            ],
            "secondary": [
                "all preregistered outcomes are present with effective variation",
                "at least two of wait/stall, excess delay and deadlock risk satisfy the overall CI/placebo support rule",
                "none has statistically supported overall harm",
            ],
            "guardrails": [
                "missing or near-constant guardrails are reported as unavailable, not fabricated evidence",
                "no evaluable guardrail may show statistically supported harm overall or within a required load",
            ],
        },
        "forbidden_adaptation_after_certification_data": [
            "switching the primary predictor to total, station or arrival drift",
            "tuning a raw Delta-L action gap",
            "changing the primary outcome or required loads",
            "learning a load gate from the certification seeds",
        ],
    }
    canonical = json.dumps(
        specification,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **specification,
        "protocol_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _formal_work_group_range_protocol() -> dict:
    """Return the frozen v2 protocol for range-normalised work drift.

    Seeds 411--420 are development data because they were used to select this
    transform.  Only the untouched 421--430 seed set may satisfy this formal
    protocol.
    """
    specification = {
        "schema_version": LAYER3_FORMAL_WORK_GROUP_RANGE_SCHEMA_VERSION,
        "frozen": True,
        "candidate_component": "work_group_range",
        "candidate_predictor": (
            "within_candidate_group_centered_Delta_L_work_divided_by_"
            "within_group_raw_Delta_L_work_range"
        ),
        "formula": (
            "(Delta_L_work(a)-mean_a Delta_L_work(a))/"
            "(max_a Delta_L_work(a)-min_a Delta_L_work(a)); "
            "zero when the denominator is zero"
        ),
        "state_potential_remains": (
            "L_work + L_station + L_arrival; the range transform is an "
            "action-comparison coordinate and does not redefine Layer-1 L"
        ),
        "development_seed_range": [411, 420],
        "required_simulation_seeds": list(
            FORMAL_WORK_GROUP_RANGE_CERTIFICATION_SEEDS
        ),
        "primary_outcome": FORMAL_WORK_ONLY_PRIMARY_OUTCOME,
        "required_secondary_outcomes": list(
            FORMAL_WORK_ONLY_REQUIRED_SECONDARY_OUTCOMES
        ),
        "guardrail_outcomes": list(FORMAL_WORK_ONLY_GUARDRAIL_OUTCOMES),
        "required_loads": list(FORMAL_WORK_ONLY_REQUIRED_LOADS),
        "minimum_independent_seed_clusters": (
            FORMAL_WORK_ONLY_MINIMUM_SEED_CLUSTERS
        ),
        "minimum_supportive_secondary_outcomes": (
            FORMAL_WORK_ONLY_MINIMUM_SUPPORTIVE_SECONDARY_OUTCOMES
        ),
        "resampling": {
            "bootstrap_repeats": FORMAL_WORK_ONLY_BOOTSTRAP_REPEATS,
            "placebo_repeats": FORMAL_WORK_ONLY_PLACEBO_REPEATS,
            "random_seed": FORMAL_WORK_ONLY_RANDOM_SEED,
        },
        "validation_split": (
            "leave-one-simulator-seed-out; the same seed across all loads "
            "remains in one fold"
        ),
        "world_model_score_contract": {
            "required_source": (
                "checkpoint-scored inside evaluate_lyapunov_validity"
            ),
            "heuristic_cost_is_forbidden": True,
            "checkpoint_must_not_train_on_certification_seeds": True,
            "checkpoint_hash_is_frozen_by_collection_manifest": True,
        },
        "candidate_set_contract": {
            "collection_candidate_limit": 10,
            "online_candidate_scope": "all_currently_idle_robots",
            "raw_gap_gate": False,
            "load_gate": False,
            "development_preflight": [
                "raw_range_quantile_stratification",
                "observed_candidate_count_stratification",
                "leave_one_candidate_out_combined_score_stability",
            ],
            "scope_limit": (
                "development subset stability covers candidate removal up to "
                "the observed top-m=10 collection; candidate supersets above "
                "10 remain a Layer-4/online validation question"
            ),
        },
        "acceptance_contract": {
            "data": [
                "exactly the untouched simulator seeds 421 through 430",
                "every evaluated seed contains paired low/mid/high arms",
                "every seed/load pair has exactly one run arm and explicit simulation_seed provenance",
                "work_group_range has effective within-group variation in every load",
            ],
            "primary": [
                "overall NRMSE improvement is positive with seed-cluster CI95 lower bound above zero",
                "observed improvement exceeds within-group permutation-placebo p95",
                "each required load has positive improvement with seed-cluster CI95 lower bound above zero",
                "each held-out seed has positive point improvement",
                "every fold has a positive standardised work_group_range coefficient",
                "Top-1 accuracy and pairwise concordance improve",
                "mean normalised selection regret decreases",
            ],
            "secondary": [
                "all preregistered outcomes are present with effective variation",
                "at least two of wait/stall, excess delay and deadlock risk satisfy the overall CI/placebo support rule",
                "none has statistically supported overall harm",
            ],
            "guardrails": [
                "missing or near-constant guardrails are reported as unavailable, not fabricated evidence",
                "no evaluable guardrail may show statistically supported harm overall or within a required load",
            ],
        },
        "forbidden_adaptation_after_certification_data": [
            "switching back to raw work, total, station or arrival drift",
            "tuning a raw Delta-L action gap",
            "changing the primary outcome or required loads",
            "learning a load gate from the certification seeds",
            "changing the range transform after inspecting seeds 421-430",
        ],
    }
    canonical = json.dumps(
        specification,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **specification,
        "protocol_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _parse_csv(text: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in text.split(",") if value.strip())


def _expand_paths(values: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for value in values:
        matches = sorted(glob.glob(value)) if any(ch in value for ch in "*?") else []
        for path in matches or [value]:
            resolved = str(Path(path).expanduser().resolve())
            if resolved not in paths:
                paths.append(resolved)
    return paths


def _load_samples(paths: Sequence[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected a list of counterfactual samples")
        for sample in payload:
            if not isinstance(sample, Mapping):
                raise ValueError(f"{path}: sample is not a mapping")
            row = dict(sample)
            row["_source_path"] = str(path)
            rows.append(row)
    return rows


def _future_system_outcomes(sample: Mapping) -> dict:
    """Extract horizon-normalised simulator outcomes, all in cost direction.

    These channels are produced by the simulator rollout and do not reuse the
    analytic unfinished-work ledger.  ``completed_orders_cost`` negates the
    completion channel so that a larger value is consistently worse for all
    outcomes in the continuous association tests.
    """
    labels = sample.get("future_system_labels")
    if labels is None:
        return {}
    tensor = torch.as_tensor(labels).detach().cpu().to(dtype=torch.float64)
    if tensor.ndim != 2 or tensor.size(1) < 7 or tensor.size(0) == 0:
        return {}
    mask = sample.get("future_mask")
    if mask is not None:
        valid = torch.as_tensor(mask).detach().cpu().flatten() > 0.5
        valid = valid[:tensor.size(0)]
        tensor = tensor[:valid.numel()][valid]
    if tensor.size(0) == 0 or not bool(torch.isfinite(tensor).all()):
        return {}
    means = tensor[:, :6].mean(dim=0)
    return {
        "wait_or_stall_mean": float(means[0]),
        "average_excess_delay_mean": float(means[1]),
        "station_queue_delta_mean": float(means[2]),
        "station_load_imbalance_mean": float(means[3]),
        "bottleneck_cvar_mean": float(means[4]),
        "completed_orders_cost": -float(means[5]),
        "deadlock_risk_max": float(tensor[:, 6].max()),
    }


def _prepare_samples(samples: Sequence[Mapping]) -> list[dict]:
    prepared = []
    for sample in samples:
        row = dict(sample)
        row["_validation_outcomes"] = _future_system_outcomes(sample)
        prepared.append(row)
    return prepared


def _score_with_world_model(
    samples: Sequence[Mapping],
    checkpoint: str,
    *,
    device: str = "cpu",
) -> tuple[list[dict], dict]:
    """Attach the actual checkpoint score used by the World Model."""
    from WorldModel.evaluation.evaluate import _load_model

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"requested device {device!r}, but CUDA is unavailable")
    model, checkpoint_schema = _load_model(checkpoint)
    model = model.to(device)
    model.eval()
    prepared = _prepare_samples(samples)
    state_cache = {}
    scored = 0
    with torch.no_grad():
        for row in prepared:
            required = (
                "node_history", "edge_index", "edge_features",
                "demand_context", "action_node", "action_global",
            )
            missing = [name for name in required if row.get(name) is None]
            if missing:
                raise ValueError(
                    f"sample {_group_key(row)} lacks WM inputs {missing}"
                )
            key = _group_key(row)
            edge_index = torch.as_tensor(row["edge_index"]).to(device)
            if key not in state_cache:
                z, demand, edge_attr = model.encode_state(
                    torch.as_tensor(row["node_history"]).to(device),
                    edge_index,
                    torch.as_tensor(row["edge_features"]).to(device),
                    torch.as_tensor(row["demand_context"]).to(device),
                )
                state_cache[key] = (z, demand, edge_attr)
            else:
                z, demand, edge_attr = state_cache[key]
            station_node_ids = row.get("station_node_ids")
            if station_node_ids is not None:
                station_node_ids = torch.as_tensor(
                    station_node_ids
                ).detach().cpu().tolist()
            score = model.predict_cost(
                z,
                demand,
                edge_attr,
                torch.as_tensor(row["action_node"]).to(device),
                torch.as_tensor(row["action_global"]).to(device),
                edge_index,
                station_node_ids,
            )
            row["_validation_wm_score"] = float(score.detach().cpu().item())
            scored += 1
    return prepared, {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "checkpoint_schema": checkpoint_schema,
        "device": str(device),
        "score_field": "_validation_wm_score",
        "scored_samples": scored,
        "encoded_candidate_groups": len(state_cache),
    }


def _finite(value) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _field(sample: Mapping, name: str) -> Optional[float]:
    """Read a finite scalar from a dotted field path."""
    value = sample
    for part in str(name).split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return _finite(value)


def _quantiles(values: Sequence[float]) -> dict:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not array.size:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, implemented without scipy."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    cursor = 0
    while cursor < order.size:
        end = cursor + 1
        while end < order.size and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + end - 1) + 1.0
        cursor = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if left.size < 3:
        return None
    left_scale = float(np.max(np.abs(left)))
    right_scale = float(np.max(np.abs(right)))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    left = left / left_scale
    right = right / right_scale
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return None
    return float(np.dot(left, right) / denominator)


def _spearman(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if left.size < 3:
        return None
    return _pearson(_rankdata(left), _rankdata(right))


def _ci(values: Sequence[float], alpha: float = 0.05) -> Optional[list[float]]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not array.size:
        return None
    return [
        float(np.quantile(array, alpha / 2.0)),
        float(np.quantile(array, 1.0 - alpha / 2.0)),
    ]


def _group_key(sample: Mapping) -> tuple[str, str]:
    return (
        str(sample.get("run_id") or sample.get("_source_path") or "unknown"),
        str(sample.get("candidate_group_id", "unknown")),
    )


def _cluster_key(sample: Mapping) -> str:
    seed = sample.get("simulation_seed")
    if seed is not None:
        # The same numeric simulator seed is commonly reused for low/mid/high
        # paired runs.  Those runs can share random structure and must stay in
        # one bootstrap/CV cluster; including load here would create false
        # independence and leak the same seed into Layer-3 training folds.
        return f"seed={seed}"
    return str(sample.get("run_id") or sample.get("_source_path") or "unknown")


def _infer_load(sample: Mapping) -> str:
    value = sample.get("load") or sample.get("load_level")
    if value is not None:
        return str(value).lower()
    text = " ".join((
        str(sample.get("run_id", "")),
        str(sample.get("_source_path", "")),
    )).lower()
    for load in ("low", "mid", "high"):
        if load in text:
            return load
    return "unknown"


def _groups(samples: Sequence[Mapping]) -> dict[tuple[str, str], list[Mapping]]:
    grouped: dict[tuple[str, str], list[Mapping]] = defaultdict(list)
    for sample in samples:
        if sample.get("lyapunov_l0_valid") is False:
            continue
        if _field(sample, "lyapunov_l0_delta") is None:
            continue
        grouped[_group_key(sample)].append(sample)
    return {key: rows for key, rows in grouped.items() if len(rows) >= 2}


def _bootstrap_clusters(
    clusters: np.ndarray,
    statistic: Callable[[np.ndarray], Optional[float]],
    *,
    repeats: int,
    seed: int,
) -> Optional[list[float]]:
    """Percentile CI after resampling independent run/seed clusters."""
    clusters = np.asarray(clusters, dtype=object)
    unique = np.unique(clusters)
    if not unique.size or repeats <= 0:
        return None
    rng = np.random.default_rng(seed)
    results: list[float] = []
    for _ in range(int(repeats)):
        selected = rng.choice(unique, size=unique.size, replace=True)
        pieces = [np.flatnonzero(clusters == value) for value in selected]
        if not pieces:
            continue
        value = statistic(np.concatenate(pieces))
        if value is not None and math.isfinite(float(value)):
            results.append(float(value))
    return _ci(results)


def _component_delta(sample: Mapping, name: str) -> Optional[float]:
    explicit = sample.get("lyapunov_l0_component_delta")
    if isinstance(explicit, Mapping) and name in explicit:
        return _finite(explicit.get(name))
    start = sample.get("lyapunov_l0_start")
    end = sample.get("lyapunov_l0_end")
    if not isinstance(start, Mapping) or not isinstance(end, Mapping):
        return None
    start_components = start.get("components")
    end_components = end.get("components")
    if not isinstance(start_components, Mapping) or not isinstance(end_components, Mapping):
        return None
    before = _finite(start_components.get(name))
    after = _finite(end_components.get(name))
    if before is None or after is None:
        return None
    return float(after - before)


def _start_total(sample: Mapping) -> Optional[float]:
    return _field(sample, "lyapunov_l0_start.total")


def _normalise_tree(value):
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _normalise_tree(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_normalise_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def _manifest(samples: Sequence[Mapping]) -> dict:
    loads = defaultdict(lambda: {"samples": 0, "groups": set(), "seeds": set()})
    for sample in samples:
        load = _infer_load(sample)
        row = loads[load]
        row["samples"] += 1
        row["groups"].add(_group_key(sample))
        if sample.get("simulation_seed") is not None:
            row["seeds"].add(int(sample["simulation_seed"]))
    return {
        load: {
            "samples": int(row["samples"]),
            "groups": int(len(row["groups"])),
            "seeds": sorted(row["seeds"]),
        }
        for load, row in sorted(loads.items())
    }


def _work_formula(station_work: Sequence[float], weight: float, capacity: float) -> float:
    capacity = max(float(capacity), 1e-12)
    return float(0.5 * float(weight) * sum((float(value) / capacity) ** 2 for value in station_work))


def _station_formula(queue_ratios: Sequence[float], weight: float, safe_ratio: float) -> float:
    return float(0.5 * float(weight) * sum(max(0.0, float(value) - float(safe_ratio)) ** 2 for value in queue_ratios))


def _arrival_formula(
    arrival_bins: Mapping,
    arrival_capacity: Mapping,
    weight: float,
) -> Optional[float]:
    total = 0.0
    terms = 0
    for station_id, values in arrival_bins.items():
        capacities = arrival_capacity.get(station_id)
        if capacities is None:
            capacities = arrival_capacity.get(str(station_id))
        if capacities is None or len(capacities) != len(values):
            return None
        cumulative_load = 0.0
        cumulative_capacity = 0.0
        for value, capacity in zip(values, capacities):
            cumulative_load += float(value)
            cumulative_capacity += float(capacity)
            excess = max(0.0, cumulative_load - cumulative_capacity)
            total += (excess / max(cumulative_capacity, 1.0)) ** 2
            terms += 1
    return float(0.5 * float(weight) * total / max(terms, 1))


def _formula_audit(samples: Sequence[Mapping], tolerance: float) -> dict:
    checks = {
        "work": {"tested": 0, "failures": 0, "max_abs_error": 0.0},
        "station": {"tested": 0, "failures": 0, "max_abs_error": 0.0},
        "arrival": {"tested": 0, "failures": 0, "max_abs_error": 0.0},
    }
    for sample in samples:
        config = sample.get("lyapunov_l0_config")
        if not isinstance(config, Mapping):
            continue
        for endpoint in ("start", "post_action", "end"):
            snapshot = sample.get(f"lyapunov_l0_{endpoint}")
            if not isinstance(snapshot, Mapping):
                continue
            components = snapshot.get("components")
            if not isinstance(components, Mapping):
                continue

            station_work = snapshot.get("station_work")
            capacity = _finite(snapshot.get("work_capacity"))
            observed = _finite(components.get("work"))
            if isinstance(station_work, Mapping) and capacity is not None and observed is not None:
                expected = _work_formula(
                    [float(value) for value in station_work.values()],
                    float(config.get("work_weight", 1.0)),
                    capacity,
                )
                error = abs(observed - expected)
                row = checks["work"]
                row["tested"] += 1
                row["failures"] += int(error > tolerance * max(1.0, abs(observed), abs(expected)))
                row["max_abs_error"] = max(row["max_abs_error"], error)

            queue_ratios = snapshot.get("station_queue_ratio")
            observed = _finite(components.get("station"))
            if isinstance(queue_ratios, Mapping) and observed is not None:
                expected = _station_formula(
                    [float(value) for value in queue_ratios.values()],
                    float(config.get("station_weight", 1.0)),
                    float(config.get("station_safe_ratio", 0.70)),
                )
                error = abs(observed - expected)
                row = checks["station"]
                row["tested"] += 1
                row["failures"] += int(error > tolerance * max(1.0, abs(observed), abs(expected)))
                row["max_abs_error"] = max(row["max_abs_error"], error)

            arrival_bins = snapshot.get("arrival_bins")
            arrival_capacity = snapshot.get("arrival_capacity")
            observed = _finite(components.get("arrival"))
            if isinstance(arrival_bins, Mapping) and isinstance(arrival_capacity, Mapping) and observed is not None:
                expected = _arrival_formula(
                    arrival_bins,
                    arrival_capacity,
                    float(config.get("arrival_weight", 1.0)),
                )
                if expected is not None:
                    error = abs(observed - expected)
                    row = checks["arrival"]
                    row["tested"] += 1
                    row["failures"] += int(error > tolerance * max(1.0, abs(observed), abs(expected)))
                    row["max_abs_error"] = max(row["max_abs_error"], error)
    for row in checks.values():
        row["passed"] = bool(row["tested"] > 0 and row["failures"] == 0)
    return {
        "checks": checks,
        "tested_components": [name for name, row in checks.items() if row["tested"] > 0],
        "passed": all(row["failures"] == 0 for row in checks.values() if row["tested"] > 0),
    }


def _analytic_perturbation_contract(samples: Sequence[Mapping]) -> dict:
    configs = []
    signatures = set()
    for sample in samples:
        config = sample.get("lyapunov_l0_config")
        if isinstance(config, Mapping):
            signature = _normalise_tree(config)
            if signature not in signatures:
                signatures.add(signature)
                configs.append(config)
    if not configs:
        return {
            "status": "NOT_EVALUATED_MISSING_CONFIG",
            "unique_configs": 0,
            "checks": {},
            "passed": False,
        }

    per_config = []
    for config in configs:
        work_weight = float(config.get("work_weight", 1.0))
        station_weight = float(config.get("station_weight", 1.0))
        arrival_weight = float(config.get("arrival_weight", 1.0))
        safe_ratio = float(config.get("station_safe_ratio", 0.70))
        capacity = float(config.get("work_capacity", 0.0))
        capacity = capacity if capacity > 0.0 else 1.0

        balanced = _work_formula((1.0, 1.0), work_weight, capacity)
        concentrated = _work_formula((2.0, 0.0), work_weight, capacity)
        before_service = _work_formula((2.0, 1.0), work_weight, capacity)
        after_service = _work_formula((1.5, 1.0), work_weight, capacity)
        before_addition = _work_formula((1.0, 1.0), work_weight, capacity)
        after_addition = _work_formula((1.25, 1.0), work_weight, capacity)
        station_healthy = _station_formula((safe_ratio, safe_ratio), station_weight, safe_ratio)
        station_overloaded = _station_formula((safe_ratio + 0.25, safe_ratio), station_weight, safe_ratio)
        arrival_capacity = {1: (1.0, 1.0, 1.0)}
        arrival_later = _arrival_formula({1: (0.0, 0.0, 3.0)}, arrival_capacity, arrival_weight)
        arrival_earlier = _arrival_formula({1: (3.0, 0.0, 0.0)}, arrival_capacity, arrival_weight)

        checks = {
            "nonnegative_weights": all(
                float(config.get(name, 0.0)) >= 0.0
                for name in (
                    "work_weight", "station_weight", "traffic_weight",
                    "stall_weight", "plan_fail_weight", "arrival_weight",
                )
            ),
            # A non-negative but constant L == 0 satisfies weak monotonicity
            # vacuously.  The core unfinished-work term must therefore react
            # *strictly* to physical work addition/service.  This is an
            # algebraic sensitivity check, not a raw empirical gap threshold.
            "positive_core_work_weight": work_weight > 0.0,
            "adding_unfinished_work_is_monotone": after_addition >= before_addition - 1e-12,
            "concentrating_equal_work_is_not_better": concentrated >= balanced - 1e-12,
            "physical_service_does_not_raise_work_potential": after_service <= before_service + 1e-12,
            "station_overload_is_monotone": station_overloaded >= station_healthy - 1e-12,
            "earlier_crowded_arrival_is_not_better": (
                arrival_earlier is not None and arrival_later is not None
                and arrival_earlier >= arrival_later - 1e-12
            ),
            "adding_unfinished_work_strictly_increases_core_potential": (
                after_addition > before_addition
            ),
            "concentrating_equal_work_strictly_increases_core_potential": (
                concentrated > balanced
            ),
            "physical_service_strictly_decreases_core_potential": (
                after_service < before_service
            ),
            "enabled_station_term_is_strictly_sensitive": (
                station_weight <= 0.0 or station_overloaded > station_healthy
            ),
            "enabled_arrival_term_is_strictly_sensitive": (
                arrival_weight <= 0.0
                or (
                    arrival_earlier is not None
                    and arrival_later is not None
                    and arrival_earlier > arrival_later
                )
            ),
        }
        per_config.append({
            "config": dict(config),
            "checks": checks,
            "passed": all(checks.values()),
        })
    return {
        "status": "SUPPORTED" if all(row["passed"] for row in per_config) else "FAILED",
        "unique_configs": len(configs),
        "frozen_config": len(configs) == 1,
        "configs": per_config,
        "passed": bool(per_config) and all(row["passed"] for row in per_config),
    }


def _observed_transition_contract(samples: Sequence[Mapping], tolerance: float) -> dict:
    """Check service and replan semantics on transitions that expose them."""
    service_tested = 0
    service_failures = 0
    replan_tested = 0
    replan_failures = 0
    for sample in samples:
        start = sample.get("lyapunov_l0_start")
        end = sample.get("lyapunov_l0_end")
        progress = sample.get("lyapunov_l0_progress")
        if not isinstance(start, Mapping) or not isinstance(end, Mapping) or not isinstance(progress, Mapping):
            continue
        start_components = start.get("components")
        end_components = end.get("components")
        if not isinstance(start_components, Mapping) or not isinstance(end_components, Mapping):
            continue
        start_work = _finite(start_components.get("work"))
        end_work = _finite(end_components.get("work"))
        if start_work is None or end_work is None:
            continue
        productive = _finite(progress.get("productive_total")) or 0.0
        reverse = _finite(progress.get("reverse_total")) or 0.0
        arrival = _finite(progress.get("arrival_total")) or 0.0
        residual = _finite(progress.get("replan_residual_total")) or 0.0
        churn = _finite(progress.get("route_plan_churn_total")) or 0.0
        quiet_exogenous = max(abs(reverse), abs(arrival), abs(residual)) <= tolerance
        if productive > tolerance and quiet_exogenous:
            service_tested += 1
            service_failures += int(end_work > start_work + tolerance * max(1.0, abs(start_work)))
        if churn > tolerance and quiet_exogenous and abs(productive) <= tolerance:
            replan_tested += 1
            replan_failures += int(abs(end_work - start_work) > tolerance * max(1.0, abs(start_work)))
    checks = {
        "physical_service_work_potential": {
            "tested": service_tested,
            "failures": service_failures,
            "passed": service_tested > 0 and service_failures == 0,
        },
        "route_replan_cannot_fake_work_dissipation": {
            "tested": replan_tested,
            "failures": replan_failures,
            "passed": replan_tested > 0 and replan_failures == 0,
        },
    }
    violated = [
        name for name, row in checks.items()
        if row["tested"] > 0 and row["failures"] > 0
    ]
    observed = [name for name, row in checks.items() if row["tested"] > 0]
    if violated:
        status = "FAILED_ON_OBSERVED_TRANSITIONS"
    elif observed:
        status = "SUPPORTED_ON_OBSERVED_TRANSITIONS"
    else:
        status = "NOT_OBSERVED"
    return {
        **checks,
        "status": status,
        "observed_checks": observed,
        "violated_checks": violated,
        # Lack of coverage is not evidence of validity, but an actually
        # observed violation must block Layer 1.
        "passed": not violated,
        "note": "Zero tested transitions means not observed, not failed; algebraic and ledger audits remain separate.",
    }


def evaluate_state_potential(
    samples: Sequence[Mapping],
    *,
    invariant_tolerance: float = 1e-6,
    strict_isolated: bool = False,
    required_l0_schema: str = CURRENT_L0_COLLECTION_SCHEMA,
) -> dict:
    """Layer 1: validate state semantics without asking L to choose actions."""
    invariant = audit_analytic_invariants(samples, tolerance=invariant_tolerance)
    semantic = audit_isolated_semantics(
        samples,
        required_schema=required_l0_schema,
        tolerance=min(invariant_tolerance, 1e-9),
    )
    formula = _formula_audit(samples, invariant_tolerance)
    perturbations = _analytic_perturbation_contract(samples)
    observed_transitions = _observed_transition_contract(samples, invariant_tolerance)
    if not invariant["passed"]:
        status = "INVALID_STATE_LEDGER"
    elif strict_isolated and not semantic["passed"]:
        status = "INVALID_ISOLATED_SEMANTICS"
    elif not perturbations["passed"]:
        status = "STATE_FUNCTIONAL_CONTRACT_FAILED"
    elif not observed_transitions["passed"]:
        status = "STATE_TRANSITION_CONTRACT_FAILED"
    elif not formula["tested_components"]:
        status = "SUPPORTED_LEDGER_FORMULA_NOT_RECOMPUTABLE"
    elif formula["passed"]:
        status = "STATE_POTENTIAL_SUPPORTED"
    else:
        status = "STORED_FUNCTIONAL_FORMULA_MISMATCH"
    return {
        "layer": 1,
        "name": "state_potential",
        "question": "Does L(s) express a non-negative, conserved and monotone state pressure?",
        "semantics": {
            "action_selection_claim": False,
            "productive_progress_role": "ledger_closure_only",
        },
        "isolated_semantics_audit": semantic,
        "analytic_invariant_audit": invariant,
        "stored_formula_audit": formula,
        "analytic_state_perturbations": perturbations,
        "observed_transition_contract": observed_transitions,
        "status": status,
        "supported": status in {
            "STATE_POTENTIAL_SUPPORTED",
            "SUPPORTED_LEDGER_FORMULA_NOT_RECOMPUTABLE",
        },
    }


def _centred_rows(
    groups: Mapping[tuple[str, str], Sequence[Mapping]],
    fields: Sequence[str],
) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    missing = {field: 0 for field in fields}
    for key, members in groups.items():
        raw = {field: [_field(sample, field) for sample in members] for field in fields}
        means = {}
        for field, values in raw.items():
            finite = [value for value in values if value is not None]
            means[field] = float(np.mean(finite)) if finite else None
        for index, sample in enumerate(members):
            row = {
                "sample": sample,
                "group": key,
                "cluster": _cluster_key(sample),
                "load": _infer_load(sample),
            }
            for field in fields:
                value = raw[field][index]
                if value is None or means[field] is None:
                    row[field] = None
                    missing[field] += 1
                else:
                    row[field] = float(value - means[field])
            rows.append(row)
    return rows, missing


def _association(
    rows: Sequence[Mapping],
    left_field: str,
    right_field: str,
    *,
    repeats: int,
    seed: int,
) -> dict:
    usable = [row for row in rows if row.get(left_field) is not None and row.get(right_field) is not None]
    if len(usable) < 3:
        return {"n": len(usable), "spearman": None, "pearson": None, "spearman_ci95_cluster_bootstrap": None}
    left = np.asarray([row[left_field] for row in usable], dtype=float)
    right = np.asarray([row[right_field] for row in usable], dtype=float)
    clusters = np.asarray([row["cluster"] for row in usable], dtype=object)
    spearman = _spearman(left, right)
    ci = _bootstrap_clusters(
        clusters,
        lambda index: _spearman(left[index], right[index]),
        repeats=repeats,
        seed=seed,
    )
    concordance = _within_group_pair_concordance(
        usable,
        left_field,
        right_field,
        repeats=repeats,
        seed=seed + 17,
    )
    return {
        "n": len(usable),
        "spearman": spearman,
        "pearson": _pearson(left, right),
        "spearman_ci95_cluster_bootstrap": ci,
        "all_non_tie_pair_concordance": concordance,
    }


def _within_group_pair_concordance(
    rows: Sequence[Mapping],
    left_field: str,
    right_field: str,
    *,
    repeats: int,
    seed: int,
) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        if row.get(left_field) is not None and row.get(right_field) is not None:
            grouped[row["group"]].append(row)
    group_rows = []
    total_pairs = 0
    for members in grouped.values():
        correct = []
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                left_gap = float(
                    members[left][left_field] - members[right][left_field]
                )
                right_gap = float(
                    members[left][right_field] - members[right][right_field]
                )
                tolerance = np.finfo(float).eps * max(
                    1.0, abs(left_gap), abs(right_gap)
                ) * 16.0
                if abs(left_gap) <= tolerance or abs(right_gap) <= tolerance:
                    continue
                correct.append(float(left_gap * right_gap > 0.0))
        if correct:
            group_rows.append({
                "cluster": members[0]["cluster"],
                "accuracy": float(np.mean(correct)),
                "pairs": len(correct),
            })
            total_pairs += len(correct)
    if not group_rows:
        return {
            "groups": 0,
            "pairs": 0,
            "group_equal_mean": None,
            "group_equal_mean_ci95_cluster_bootstrap": None,
            "raw_gap_gate": False,
        }
    accuracies = np.asarray([row["accuracy"] for row in group_rows], dtype=float)
    clusters = np.asarray([row["cluster"] for row in group_rows], dtype=object)
    ci = _bootstrap_clusters(
        clusters,
        lambda index: float(accuracies[index].mean()),
        repeats=repeats,
        seed=seed,
    )
    return {
        "groups": len(group_rows),
        "pairs": int(total_pairs),
        "group_equal_mean": float(accuracies.mean()),
        "group_equal_mean_ci95_cluster_bootstrap": ci,
        "raw_gap_gate": False,
    }


def evaluate_action_controllability(
    samples: Sequence[Mapping],
    *,
    outcome_fields: Sequence[str] = DEFAULT_OUTCOME_FIELDS,
    bootstrap_repeats: int = 1000,
    random_seed: int = 2026,
) -> dict:
    """Layer 2: measure continuous within-context action-dependent drift."""
    groups = _groups(samples)
    if not groups:
        return {
            "layer": 2,
            "name": "action_controllable_drift",
            "status": "INSUFFICIENT_CANDIDATE_GROUPS",
            "supported": False,
            "groups": 0,
        }

    ranges: list[float] = []
    normalised_ranges: list[float] = []
    advantages: list[float] = []
    normalised_advantages: list[float] = []
    group_records = []
    centred_deltas = []
    for members in groups.values():
        values = np.asarray([
            _field(sample, "lyapunov_l0_delta") for sample in members
        ], dtype=float)
        centred_deltas.extend((values - values.mean()).tolist())
    # Normalise by action-dependent variation only.  Between-context drift
    # offsets are irrelevant to candidate controllability and can otherwise
    # make an action signal look artificially small.
    observed_scale = float(np.std(centred_deltas))
    global_scale = observed_scale if observed_scale > 0.0 else 1.0
    varying_groups = 0
    for key, members in groups.items():
        values = np.asarray([_field(sample, "lyapunov_l0_delta") for sample in members], dtype=float)
        spread = float(values.max() - values.min())
        advantage = float(values.mean() - values.min())
        within_scale = float(values.std())
        relative_epsilon = (
            np.finfo(float).eps * float(np.max(np.abs(values))) * 16.0
        )
        varying = spread > relative_epsilon
        varying_groups += int(varying)
        ranges.append(spread)
        normalised_ranges.append(spread / global_scale)
        advantages.append(advantage)
        normalised_advantages.append(advantage / global_scale)
        start_values = [_start_total(sample) for sample in members]
        start_l = next((value for value in start_values if value is not None), None)
        group_records.append({
            "key": key,
            "cluster": _cluster_key(members[0]),
            "load": _infer_load(members[0]),
            "start_L": start_l,
            "range": spread,
            "normalised_range": spread / global_scale,
            "controllability_advantage": advantage,
            "normalised_controllability_advantage": advantage / global_scale,
            "within_std": within_scale,
            "varying": varying,
        })

    group_clusters = np.asarray([row["cluster"] for row in group_records], dtype=object)
    normalised_range_array = np.asarray(normalised_ranges, dtype=float)
    range_ci = _bootstrap_clusters(
        group_clusters,
        lambda index: float(normalised_range_array[index].mean()),
        repeats=bootstrap_repeats,
        seed=random_seed + 101,
    )

    all_fields = ["lyapunov_l0_delta", *outcome_fields, DEPENDENT_LEDGER_FIELD]
    centred, missing = _centred_rows(groups, all_fields)
    independent_associations = {
        field: _association(
            centred,
            "lyapunov_l0_delta",
            field,
            repeats=bootstrap_repeats,
            seed=random_seed + 1009 * (index + 1),
        )
        for index, field in enumerate(outcome_fields)
        if missing[field] < len(centred)
    }
    progress_association = _association(
        centred,
        "lyapunov_l0_delta",
        DEPENDENT_LEDGER_FIELD,
        repeats=bootstrap_repeats,
        seed=random_seed + 7919,
    )

    component_rows = []
    for row in centred:
        sample = row["sample"]
        component_rows.append({
            **row,
            **{name: _component_delta(sample, name) for name in L0_COMPONENT_NAMES},
        })
    component_report = {}
    for name in L0_COMPONENT_NAMES:
        # Centre component deltas within the same candidate group.
        values_by_group = defaultdict(list)
        for row in component_rows:
            if row[name] is not None:
                values_by_group[row["group"]].append(float(row[name]))
        means = {key: float(np.mean(values)) for key, values in values_by_group.items()}
        paired = [
            (float(row[name]) - means[row["group"]], float(row["lyapunov_l0_delta"]))
            for row in component_rows
            if row[name] is not None and row["lyapunov_l0_delta"] is not None
            and row["group"] in means
        ]
        if paired:
            component = np.asarray([value[0] for value in paired], dtype=float)
            total = np.asarray([value[1] for value in paired], dtype=float)
            attribution_scale = float(np.max(np.abs(total)))
            if attribution_scale > 0.0:
                scaled_component = component / attribution_scale
                scaled_total = total / attribution_scale
                total_variance = float(np.var(scaled_total))
                covariance_share = (
                    float(
                        np.mean(
                            (scaled_component - scaled_component.mean())
                            * (scaled_total - scaled_total.mean())
                        ) / total_variance
                    )
                    if total_variance > 0.0 else None
                )
            else:
                covariance_share = None
            component_report[name] = {
                "n": len(paired),
                "within_group_delta": _quantiles(component.tolist()),
                "spearman_with_total_centred_drift": _spearman(component, total),
                "covariance_share_of_total_within_group_variance": covariance_share,
            }
        else:
            component_report[name] = {"n": 0}

    covariance_shares = [
        row.get("covariance_share_of_total_within_group_variance")
        for row in component_report.values()
        if row.get("covariance_share_of_total_within_group_variance") is not None
    ]
    component_reconstruction = {
        "covariance_share_sum": float(sum(covariance_shares)) if covariance_shares else None,
        "expected_sum_when_endpoints_are_complete": 1.0,
        "note": (
            "Shares use covariance(component, total)/variance(total), so they "
            "attribute candidate-group variation rather than absolute drift."
        ),
    }

    high_state_rows = [row for row in group_records if row["start_L"] is not None]
    high_state_relation = {
        "n": len(high_state_rows),
        "spearman_start_L_vs_normalised_controllability": _spearman(
            np.asarray([row["start_L"] for row in high_state_rows], dtype=float),
            np.asarray([row["normalised_controllability_advantage"] for row in high_state_rows], dtype=float),
        ) if len(high_state_rows) >= 3 else None,
    }

    by_load = {}
    for load in sorted({row["load"] for row in group_records}):
        load_rows = [row for row in group_records if row["load"] == load]
        by_load[load] = {
            "groups": len(load_rows),
            "varying_groups": int(sum(row["varying"] for row in load_rows)),
            "varying_group_rate": float(np.mean([row["varying"] for row in load_rows])),
            "normalised_within_group_drift_range": _quantiles([
                row["normalised_range"] for row in load_rows
            ]),
            "normalised_controllability_advantage": _quantiles([
                row["normalised_controllability_advantage"] for row in load_rows
            ]),
        }

    independent_cluster_count = int(
        len(set(row["cluster"] for row in group_records))
    )
    if varying_groups == 0:
        status = "ACTION_INSENSITIVE_STATE_MONITOR_ONLY"
    elif independent_cluster_count < 2:
        status = "INSUFFICIENT_INDEPENDENT_CONTEXT_CLUSTERS"
    elif range_ci is not None and range_ci[0] > 0.0:
        status = "ACTION_CONTROLLABLE_DRIFT_SUPPORTED"
    else:
        status = "ACTION_CONTROLLABILITY_INCONCLUSIVE"
    return {
        "layer": 2,
        "name": "action_controllable_drift",
        "question": "Within a fixed context, do candidate actions continuously change drift?",
        "semantics": {
            "group_centring": "subtract candidate-group mean",
            "raw_gap_gate": False,
            "scale_invariance": "normalise by the observed drift standard deviation; use ranks and covariance shares",
            "cluster_unit": "simulation_seed across all loads; run/source only when seed is absent",
        },
        "groups": len(groups),
        "independent_clusters": independent_cluster_count,
        "samples": int(sum(len(rows) for rows in groups.values())),
        "varying_groups": varying_groups,
        "varying_group_rate": float(varying_groups / len(groups)),
        "within_group_drift_range": _quantiles(ranges),
        "normalised_within_group_drift_range": _quantiles(normalised_ranges),
        "normalised_range_mean_ci95_cluster_bootstrap": range_ci,
        "controllability_advantage_mean_minus_min": _quantiles(advantages),
        "normalised_controllability_advantage": _quantiles(normalised_advantages),
        "high_state_controllability_relation": high_state_relation,
        "within_group_component_attribution": component_report,
        "component_attribution_closure": component_reconstruction,
        "by_load": by_load,
        "independent_outcome_associations": independent_associations,
        "ledger_coupled_evidence": {
            "field": DEPENDENT_LEDGER_FIELD,
            "association": progress_association,
            "primary_validity_evidence": False,
            "reason": "P_prod and L_work share the unfinished-work ledger.",
        },
        "missing_values": missing,
        "status": status,
        "supported": status == "ACTION_CONTROLLABLE_DRIFT_SUPPORTED",
    }


def _build_folds(rows: Sequence[Mapping]) -> list[np.ndarray]:
    clusters = np.asarray([row["cluster"] for row in rows], dtype=object)
    unique = np.unique(clusters)
    if unique.size >= 2:
        return [np.flatnonzero(clusters == cluster) for cluster in unique]
    groups = np.asarray([str(row["group"]) for row in rows], dtype=object)
    unique_groups = np.unique(groups)
    fold_count = min(5, unique_groups.size)
    if fold_count < 2:
        return []
    return [
        np.flatnonzero(np.isin(groups, unique_groups[index::fold_count]))
        for index in range(fold_count)
    ]


def _fit_linear(train_x: np.ndarray, train_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0.0] = 1.0
    design = np.column_stack((np.ones(train_x.shape[0]), (train_x - mean) / scale))
    coefficients = np.linalg.lstsq(design, train_y, rcond=None)[0]
    return coefficients, mean, scale


def _predict_linear(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    coefficients, mean, scale = model
    design = np.column_stack((np.ones(x.shape[0]), (x - mean) / scale))
    return design @ coefficients


def _cross_validated_predictions(x: np.ndarray, y: np.ndarray, folds: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    predictions = np.full(y.shape, np.nan, dtype=float)
    all_indices = np.arange(y.size)
    for test in folds:
        train = np.setdiff1d(all_indices, test, assume_unique=False)
        if train.size <= x.shape[1] + 1 or not test.size:
            continue
        model = _fit_linear(x[train], y[train])
        predictions[test] = _predict_linear(x[test], model)
    return predictions if np.isfinite(predictions).all() else None


def _prediction_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    residual = y - prediction
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    scale = float(y.std())
    nrmse = rmse / scale if scale > 0.0 else None
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / sst) if sst > 0.0 else None
    return {
        "rmse": rmse,
        "normalised_rmse": nrmse,
        "r2": r2,
        "spearman": _spearman(y, prediction),
    }


def _outcome_variation_diagnostics(
    rows: Sequence[Mapping],
    values: np.ndarray,
    *,
    min_effective_groups: int = LAYER3_MIN_EFFECTIVE_OUTCOME_GROUPS,
    reference_scale: Optional[float] = None,
) -> dict:
    """Audit effective label variation at float32 simulator resolution."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "status": "INSUFFICIENT_ROWS",
            "passed": False,
            "n": 0,
            "groups": 0,
            "effective_non_tie_groups": 0,
        }
    max_abs = float(np.max(np.abs(values)))
    resolution = float(
        np.finfo(np.float32).eps
        * LAYER3_FLOAT32_RESOLUTION_MULTIPLIER
        * max(1.0, max_abs)
    )
    scale = float(values.std())
    scale_floor = max(
        resolution,
        float(reference_scale or 0.0) * 1e-6,
    )
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["group"]].append(index)
    spreads = []
    effective = 0
    for indices in grouped.values():
        local = values[np.asarray(indices, dtype=int)]
        spread = float(local.max() - local.min())
        spreads.append(spread)
        effective += int(spread > resolution)
    median = float(np.median(values))
    iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    mad = float(np.median(np.abs(values - median)))
    if values.size < 3:
        status = "INSUFFICIENT_ROWS"
    elif scale <= scale_floor:
        status = "NEAR_ZERO_VARIATION"
    elif effective < int(min_effective_groups):
        status = "INSUFFICIENT_EFFECTIVE_GROUPS"
    else:
        status = "EFFECTIVE_VARIATION"
    return {
        "status": status,
        "passed": status == "EFFECTIVE_VARIATION",
        "n": int(values.size),
        "groups": len(grouped),
        "effective_non_tie_groups": int(effective),
        "effective_group_rate": float(effective / len(grouped)) if grouped else 0.0,
        "standard_deviation": scale,
        "iqr": iqr,
        "mad": mad,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "float32_resolution_floor": resolution,
        "scale_floor": scale_floor,
        "group_spread": _quantiles(spreads),
        "minimum_effective_groups": int(min_effective_groups),
        "note": (
            "This is an outcome-label resolution/coverage guard, not a raw "
            "Delta-L action gap."
        ),
    }


def _cross_validated_linear_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[np.ndarray],
    *,
    feature_names: Sequence[str],
    rows: Sequence[Mapping],
) -> Optional[tuple[np.ndarray, list[dict]]]:
    """Return OOF predictions plus fold coefficients/range diagnostics."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("invalid Layer-3 design matrix")
    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names do not match Layer-3 design matrix")
    predictions = np.full(y.shape, np.nan, dtype=float)
    all_indices = np.arange(y.size)
    reports = []
    for fold_index, test in enumerate(folds):
        train = np.setdiff1d(all_indices, test, assume_unique=False)
        if train.size <= x.shape[1] + 1 or not test.size:
            continue
        model = _fit_linear(x[train], y[train])
        predictions[test] = _predict_linear(x[test], model)
        coefficients, mean, scale = model
        standardised = (x[train] - mean) / scale
        design = np.column_stack((np.ones(train.size), standardised))
        condition = float(np.linalg.cond(design))
        ranges = {}
        for feature_index, name in enumerate(feature_names):
            train_min = float(x[train, feature_index].min())
            train_max = float(x[train, feature_index].max())
            test_min = float(x[test, feature_index].min())
            test_max = float(x[test, feature_index].max())
            ranges[name] = {
                "train": [train_min, train_max],
                "test": [test_min, test_max],
                "test_outside_train_range": bool(
                    test_min < train_min or test_max > train_max
                ),
                "train_standard_deviation": float(scale[feature_index]),
            }
        reports.append({
            "fold": int(fold_index),
            "held_out_clusters": sorted({str(rows[i]["cluster"]) for i in test}),
            "train_rows": int(train.size),
            "test_rows": int(test.size),
            "train_loads": sorted({str(rows[i]["load"]) for i in train}),
            "test_loads": sorted({str(rows[i]["load"]) for i in test}),
            "intercept": float(coefficients[0]),
            "standardised_coefficients": {
                name: float(coefficients[index + 1])
                for index, name in enumerate(feature_names)
            },
            "design_condition_number": condition if math.isfinite(condition) else None,
            "feature_ranges": ranges,
        })
    if not bool(np.isfinite(predictions).all()):
        return None
    return predictions, reports


def _ranking_metrics_from_predictions(
    rows: Sequence[Mapping],
    outcome: np.ndarray,
    prediction: np.ndarray,
    *,
    resolution: float,
) -> dict:
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["group"]].append(index)
    regrets = []
    normalised_regrets = []
    top1 = 0
    pairs = 0
    correct = 0.0
    used_groups = 0
    for indices in grouped.values():
        index = np.asarray(indices, dtype=int)
        truth = outcome[index]
        predicted = prediction[index]
        spread = float(truth.max() - truth.min())
        if spread <= resolution:
            continue
        true_best = int(np.argmin(truth))
        predicted_best = int(np.argmin(predicted))
        regret = float(truth[predicted_best] - truth[true_best])
        regrets.append(regret)
        normalised_regrets.append(regret / spread)
        top1 += int(predicted_best == true_best)
        used_groups += 1
        for left in range(index.size):
            for right in range(left + 1, index.size):
                truth_gap = float(truth[left] - truth[right])
                if abs(truth_gap) <= resolution:
                    continue
                predicted_gap = float(predicted[left] - predicted[right])
                if abs(predicted_gap) <= resolution:
                    correct += 0.5
                else:
                    correct += float(truth_gap * predicted_gap > 0.0)
                pairs += 1
    return {
        "effective_groups": used_groups,
        "top1_accuracy": float(top1 / used_groups) if used_groups else None,
        "non_tie_pairs": int(pairs),
        "pairwise_concordance": float(correct / pairs) if pairs else None,
        "selection_regret": _quantiles(regrets),
        "normalised_selection_regret": _quantiles(normalised_regrets),
    }


def _prediction_slice_report(
    rows: Sequence[Mapping],
    outcome: np.ndarray,
    base_prediction: np.ndarray,
    augmented_prediction: np.ndarray,
    indices: np.ndarray,
    *,
    reference_scale: float,
    min_effective_groups: int,
) -> dict:
    indices = np.asarray(indices, dtype=int)
    local_rows = [rows[index] for index in indices]
    local_outcome = outcome[indices]
    variation = _outcome_variation_diagnostics(
        local_rows,
        local_outcome,
        min_effective_groups=min_effective_groups,
        reference_scale=reference_scale,
    )
    base_rmse = float(np.sqrt(np.mean(
        (local_outcome - base_prediction[indices]) ** 2
    )))
    augmented_rmse = float(np.sqrt(np.mean(
        (local_outcome - augmented_prediction[indices]) ** 2
    )))
    local_scale = float(variation.get("standard_deviation") or 0.0)
    normalised = (
        float((base_rmse - augmented_rmse) / local_scale)
        if variation["passed"] and local_scale > 0.0 else None
    )
    return {
        "status": variation["status"],
        "n": int(indices.size),
        "target_variation": variation,
        "wm_only_rmse": base_rmse,
        "wm_plus_predictor_rmse": augmented_rmse,
        "wm_only_normalised_rmse": (
            base_rmse / local_scale if normalised is not None else None
        ),
        "wm_plus_predictor_normalised_rmse": (
            augmented_rmse / local_scale if normalised is not None else None
        ),
        "normalised_rmse_improvement": normalised,
        "global_scale_normalised_rmse_improvement": (
            float((base_rmse - augmented_rmse) / reference_scale)
            if reference_scale > 0.0 else None
        ),
    }


def _model_comparison_diagnostics(
    rows: Sequence[Mapping],
    outcome: np.ndarray,
    base_prediction: np.ndarray,
    augmented_prediction: np.ndarray,
    *,
    bootstrap_repeats: int = 500,
    random_seed: int = 20260715,
) -> dict:
    """Compare two already-held-out prediction streams without raw gaps."""
    outcome = np.asarray(outcome, dtype=float)
    reference_scale = float(outcome.std())
    all_indices = np.arange(outcome.size)
    overall = _prediction_slice_report(
        rows,
        outcome,
        base_prediction,
        augmented_prediction,
        all_indices,
        reference_scale=reference_scale,
        min_effective_groups=LAYER3_MIN_EFFECTIVE_OUTCOME_GROUPS,
    )
    by_load = {}
    for load in sorted({str(row["load"]) for row in rows}):
        index = np.asarray([
            position for position, row in enumerate(rows)
            if str(row["load"]) == load
        ], dtype=int)
        report = _prediction_slice_report(
            rows,
            outcome,
            base_prediction,
            augmented_prediction,
            index,
            reference_scale=reference_scale,
            min_effective_groups=3,
        )
        if report["target_variation"]["passed"]:
            local_outcome = outcome[index]
            local_base = base_prediction[index]
            local_augmented = augmented_prediction[index]
            local_scale = float(local_outcome.std())
            local_clusters = np.asarray([rows[i]["cluster"] for i in index], dtype=object)
            report["normalised_rmse_improvement_ci95_seed_cluster_bootstrap"] = (
                _bootstrap_clusters(
                    local_clusters,
                    lambda selected: float(
                        (
                            np.sqrt(np.mean(
                                (local_outcome[selected] - local_base[selected]) ** 2
                            ))
                            - np.sqrt(np.mean(
                                (local_outcome[selected] - local_augmented[selected]) ** 2
                            ))
                        ) / local_scale
                    ),
                    repeats=bootstrap_repeats,
                    seed=random_seed + len(by_load) * 1009,
                )
            )
            report["ci_scope"] = (
                "resample fixed OOF errors by seed; no bootstrap refit"
            )
        else:
            report["normalised_rmse_improvement_ci95_seed_cluster_bootstrap"] = None
        by_load[load] = report

    by_seed = {}
    for cluster in sorted({str(row["cluster"]) for row in rows}):
        index = np.asarray([
            position for position, row in enumerate(rows)
            if str(row["cluster"]) == cluster
        ], dtype=int)
        by_seed[cluster] = _prediction_slice_report(
            rows,
            outcome,
            base_prediction,
            augmented_prediction,
            index,
            reference_scale=reference_scale,
            min_effective_groups=2,
        )

    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["group"]].append(index)
    resolution = float(overall["target_variation"].get(
        "float32_resolution_floor", 0.0
    ))
    group_improvements = []
    for indices in grouped.values():
        index = np.asarray(indices, dtype=int)
        local = outcome[index]
        if float(local.max() - local.min()) <= resolution:
            continue
        base_rmse = float(np.sqrt(np.mean((local - base_prediction[index]) ** 2)))
        augmented_rmse = float(np.sqrt(np.mean(
            (local - augmented_prediction[index]) ** 2
        )))
        if reference_scale > 0.0:
            group_improvements.append(
                (base_rmse - augmented_rmse) / reference_scale
            )
    valid_load_improvements = [
        row["normalised_rmse_improvement"]
        for row in by_load.values()
        if row.get("normalised_rmse_improvement") is not None
    ]
    signs = {int(np.sign(value)) for value in valid_load_improvements}
    if -1 in signs and 1 in signs:
        classification = "LOAD_HETEROGENEOUS_POINT_ESTIMATES"
    elif signs == {1}:
        classification = "ALL_INFORMATIVE_LOAD_POINT_ESTIMATES_POSITIVE"
    elif valid_load_improvements:
        classification = "NO_POSITIVE_LOAD_POINT_ESTIMATE"
    else:
        classification = "INSUFFICIENT_LOAD_VARIATION"
    base_ranking = _ranking_metrics_from_predictions(
        rows, outcome, base_prediction, resolution=resolution
    )
    augmented_ranking = _ranking_metrics_from_predictions(
        rows, outcome, augmented_prediction, resolution=resolution
    )
    return {
        "overall": overall,
        "by_load": by_load,
        "by_seed": by_seed,
        "group_equal_normalised_rmse_improvement": _quantiles(
            group_improvements
        ),
        "load_equal_mean_normalised_rmse_improvement": (
            float(np.mean(valid_load_improvements))
            if valid_load_improvements else None
        ),
        "load_heterogeneity_classification": classification,
        "bootstrap_repeats": int(bootstrap_repeats),
        "ranking": {
            "wm_only": base_ranking,
            "wm_plus_predictor": augmented_ranking,
            "top1_accuracy_improvement": (
                augmented_ranking["top1_accuracy"] - base_ranking["top1_accuracy"]
                if augmented_ranking["top1_accuracy"] is not None
                and base_ranking["top1_accuracy"] is not None else None
            ),
            "pairwise_concordance_improvement": (
                augmented_ranking["pairwise_concordance"]
                - base_ranking["pairwise_concordance"]
                if augmented_ranking["pairwise_concordance"] is not None
                and base_ranking["pairwise_concordance"] is not None else None
            ),
        },
    }


def _attach_centred_component_deltas(rows: Sequence[dict]) -> dict[str, str]:
    fields = {
        name: f"_validation_component_delta.{name}"
        for name in L0_COMPONENT_NAMES
    }
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    for name, field in fields.items():
        for members in grouped.values():
            values = [_component_delta(row["sample"], name) for row in members]
            finite = [value for value in values if value is not None]
            mean = float(np.mean(finite)) if finite else None
            for row, value in zip(members, values):
                row[field] = (
                    float(value - mean)
                    if value is not None and mean is not None else None
                )
    return fields


def _attach_work_group_range(rows: Sequence[dict]) -> str:
    """Attach a bounded, within-context work-drift coordinate.

    The raw physical ``Delta L_work`` is retained on every row.  Only the
    action-comparison predictor is range-normalised; this does not alter the
    stored Lyapunov state functional or its absolute drift ledger.
    """
    component_fields = _attach_centred_component_deltas(rows)
    centred_field = component_fields["work"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)

    for members in grouped.values():
        raw = [_component_delta(row["sample"], "work") for row in members]
        finite = [float(value) for value in raw if value is not None]
        raw_range = (
            float(max(finite) - min(finite)) if finite else None
        )
        for row, value in zip(members, raw):
            centred = row.get(centred_field)
            row[WORK_GROUP_RANGE_RAW_DELTA_FIELD] = value
            row[WORK_GROUP_RANGE_RAW_RANGE_FIELD] = raw_range
            if centred is None or raw_range is None:
                row[WORK_GROUP_RANGE_FIELD] = None
            elif raw_range == 0.0:
                row[WORK_GROUP_RANGE_FIELD] = 0.0
            else:
                row[WORK_GROUP_RANGE_FIELD] = float(centred / raw_range)
    return WORK_GROUP_RANGE_FIELD


def _load_interaction_design(
    rows: Sequence[Mapping],
    wm: np.ndarray,
    predictor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    loads = sorted({str(row["load"]) for row in rows})
    one_hot = np.asarray([
        [float(str(row["load"]) == load) for load in loads]
        for row in rows
    ], dtype=float)
    wm_by_load = one_hot * wm[:, None]
    predictor_by_load = one_hot * predictor[:, None]
    base_names = [f"wm:{load}" for load in loads]
    predictor_names = [f"drift:{load}" for load in loads]
    return (
        wm_by_load,
        np.column_stack((wm_by_load, predictor_by_load)),
        base_names,
        [*base_names, *predictor_names],
        loads,
    )


def _layer3_load_component_diagnostics(
    rows: Sequence[dict],
    *,
    wm_field: str,
    outcome_fields: Sequence[str],
    bootstrap_repeats: int = 500,
    random_seed: int = 20260715,
    formal_candidate_component: str = "total",
) -> dict:
    """Development diagnostic separating load mismatch and L components."""
    component_fields = _attach_centred_component_deltas(rows)
    diagnostic_bootstrap_repeats = min(int(bootstrap_repeats), 500)
    by_outcome = {}
    for outcome_index, outcome_field in enumerate(outcome_fields):
        comparison_seed = random_seed + 100003 * (outcome_index + 1)
        usable = [
            row for row in rows
            if row.get(wm_field) is not None
            and row.get("lyapunov_l0_delta") is not None
            and row.get(outcome_field) is not None
        ]
        if len(usable) < 6:
            by_outcome[outcome_field] = {
                "status": "INSUFFICIENT_ROWS",
                "n": len(usable),
            }
            continue
        wm = np.asarray([row[wm_field] for row in usable], dtype=float)
        drift = np.asarray([row["lyapunov_l0_delta"] for row in usable], dtype=float)
        outcome = np.asarray([row[outcome_field] for row in usable], dtype=float)
        variation = _outcome_variation_diagnostics(usable, outcome)
        if not variation["passed"]:
            by_outcome[outcome_field] = {
                "status": "INSUFFICIENT_EFFECTIVE_VARIATION",
                "n": len(usable),
                "target_variation": variation,
            }
            continue
        folds = _build_folds(usable)
        if not folds:
            by_outcome[outcome_field] = {
                "status": "INSUFFICIENT_INDEPENDENT_FOLDS",
                "n": len(usable),
                "target_variation": variation,
            }
            continue
        base_fit = _cross_validated_linear_diagnostics(
            wm[:, None], outcome, folds,
            feature_names=["wm"], rows=usable,
        )
        shared_fit = _cross_validated_linear_diagnostics(
            np.column_stack((wm, drift)), outcome, folds,
            feature_names=["wm", "total_drift"], rows=usable,
        )
        if base_fit is None or shared_fit is None:
            by_outcome[outcome_field] = {
                "status": "CROSS_VALIDATION_FAILED",
                "n": len(usable),
                "target_variation": variation,
            }
            continue
        base_prediction, base_folds = base_fit
        shared_prediction, shared_folds = shared_fit
        shared_comparison = _model_comparison_diagnostics(
            usable,
            outcome,
            base_prediction,
            shared_prediction,
            bootstrap_repeats=(
                bootstrap_repeats
                if formal_candidate_component == "total"
                else diagnostic_bootstrap_repeats
            ),
            random_seed=comparison_seed + 101,
        )

        load_base_x, load_augmented_x, load_base_names, load_augmented_names, loads = (
            _load_interaction_design(usable, wm, drift)
        )
        load_base_fit = _cross_validated_linear_diagnostics(
            load_base_x, outcome, folds,
            feature_names=load_base_names, rows=usable,
        )
        load_augmented_fit = _cross_validated_linear_diagnostics(
            load_augmented_x, outcome, folds,
            feature_names=load_augmented_names, rows=usable,
        )
        if load_base_fit is not None and load_augmented_fit is not None:
            load_base_prediction, load_base_folds = load_base_fit
            load_augmented_prediction, load_augmented_folds = load_augmented_fit
            load_comparison = _model_comparison_diagnostics(
                usable,
                outcome,
                load_base_prediction,
                load_augmented_prediction,
                bootstrap_repeats=diagnostic_bootstrap_repeats,
                random_seed=comparison_seed + 211,
            )
            load_conditioned = {
                "status": "DIAGNOSTIC_ONLY_NOT_AN_ONLINE_GATE",
                "loads": loads,
                "comparison": load_comparison,
                "wm_only_fold_diagnostics": load_base_folds,
                "wm_plus_drift_fold_diagnostics": load_augmented_folds,
            }
        else:
            load_conditioned = {"status": "CROSS_VALIDATION_FAILED"}

        residual = outcome - base_prediction
        residual_diagnostics = {}
        for load in loads:
            index = np.asarray([
                position for position, row in enumerate(usable)
                if str(row["load"]) == load
            ], dtype=int)
            residual_diagnostics[load] = {
                "n": int(index.size),
                "spearman_wm_residual_vs_total_drift": _spearman(
                    residual[index], drift[index]
                ),
                "pearson_wm_residual_vs_total_drift": _pearson(
                    residual[index], drift[index]
                ),
                "spearman_wm_vs_total_drift": _spearman(
                    wm[index], drift[index]
                ),
                "pearson_wm_vs_total_drift": _pearson(
                    wm[index], drift[index]
                ),
            }

        component_ablation = {}
        active_component_predictions = []
        active_component_names = []
        active_component_fields = []
        for name, field in component_fields.items():
            component_rows = [row for row in usable if row.get(field) is not None]
            if len(component_rows) != len(usable):
                component_ablation[name] = {
                    "status": "MISSING_COMPONENT_ENDPOINTS",
                    "n": len(component_rows),
                }
                continue
            component = np.asarray([row[field] for row in usable], dtype=float)
            predictor_variation = _outcome_variation_diagnostics(
                usable,
                component,
                min_effective_groups=3,
            )
            if not predictor_variation["passed"]:
                component_ablation[name] = {
                    "status": "PREDICTOR_HAS_INSUFFICIENT_VARIATION",
                    "predictor_variation": predictor_variation,
                }
                continue
            fit = _cross_validated_linear_diagnostics(
                np.column_stack((wm, component)),
                outcome,
                folds,
                feature_names=["wm", f"component:{name}"],
                rows=usable,
            )
            if fit is None:
                component_ablation[name] = {"status": "CROSS_VALIDATION_FAILED"}
                continue
            prediction, fold_reports = fit
            component_ablation[name] = {
                "status": "DIAGNOSTIC_ONLY",
                "predictor_variation": predictor_variation,
                "comparison": _model_comparison_diagnostics(
                    usable,
                    outcome,
                    base_prediction,
                    prediction,
                    bootstrap_repeats=(
                        bootstrap_repeats
                        if name == formal_candidate_component
                        else diagnostic_bootstrap_repeats
                    ),
                    random_seed=comparison_seed + 1009 * (
                        len(active_component_predictions) + 1
                    ),
                ),
                "fold_diagnostics": fold_reports,
            }
            active_component_predictions.append(component)
            active_component_names.append(f"component:{name}")
            active_component_fields.append(name)
        if active_component_predictions:
            all_component_x = np.column_stack(
                (wm, *active_component_predictions)
            )
            all_fit = _cross_validated_linear_diagnostics(
                all_component_x,
                outcome,
                folds,
                feature_names=["wm", *active_component_names],
                rows=usable,
            )
            if all_fit is not None:
                all_prediction, all_folds = all_fit
                drop_one = {}
                if len(active_component_predictions) >= 2:
                    for omitted_index, omitted_name in enumerate(
                        active_component_fields
                    ):
                        kept_arrays = [
                            values for index, values in enumerate(
                                active_component_predictions
                            )
                            if index != omitted_index
                        ]
                        kept_names = [
                            name for index, name in enumerate(
                                active_component_names
                            )
                            if index != omitted_index
                        ]
                        without_fit = _cross_validated_linear_diagnostics(
                            np.column_stack((wm, *kept_arrays)),
                            outcome,
                            folds,
                            feature_names=["wm", *kept_names],
                            rows=usable,
                        )
                        if without_fit is None:
                            drop_one[omitted_name] = {
                                "status": "CROSS_VALIDATION_FAILED"
                            }
                            continue
                        without_prediction, without_folds = without_fit
                        drop_one[omitted_name] = {
                            "status": "DIAGNOSTIC_ONLY",
                            "without_component_vs_wm": (
                                _model_comparison_diagnostics(
                                    usable,
                                    outcome,
                                    base_prediction,
                                    without_prediction,
                                    bootstrap_repeats=diagnostic_bootstrap_repeats,
                                    random_seed=(
                                        comparison_seed
                                        + 50021
                                        + omitted_index * 101
                                    ),
                                )
                            ),
                            "increment_from_adding_omitted_component": (
                                _model_comparison_diagnostics(
                                    usable,
                                    outcome,
                                    without_prediction,
                                    all_prediction,
                                    bootstrap_repeats=diagnostic_bootstrap_repeats,
                                    random_seed=(
                                        comparison_seed
                                        + 60013
                                        + omitted_index * 101
                                    ),
                                )
                            ),
                            "without_component_fold_diagnostics": without_folds,
                        }
                component_ablation["all_active_components_free_coefficients"] = {
                    "status": "DIAGNOSTIC_ONLY_NOT_A_REWEIGHTED_L_CERTIFICATE",
                    "components": active_component_fields,
                    "comparison": _model_comparison_diagnostics(
                        usable,
                        outcome,
                        base_prediction,
                        all_prediction,
                        bootstrap_repeats=diagnostic_bootstrap_repeats,
                        random_seed=comparison_seed + 70001,
                    ),
                    "fold_diagnostics": all_folds,
                    "drop_one": drop_one,
                }

        by_outcome[outcome_field] = {
            "status": "DIAGNOSTIC_COMPLETE",
            "n": len(usable),
            "folds": len(folds),
            "target_variation": variation,
            "shared_coefficient_model": {
                "contract": "M0=WM versus M1=WM+total_DeltaL",
                "comparison": shared_comparison,
                "wm_only_fold_diagnostics": base_folds,
                "wm_plus_drift_fold_diagnostics": shared_folds,
            },
            "load_conditioned_model": {
                "contract": (
                    "M0_load=WM*load versus M2_load=(WM+total_DeltaL)*load"
                ),
                **load_conditioned,
            },
            "wm_residual_diagnostics_by_load": residual_diagnostics,
            "component_ablation": component_ablation,
        }
    return {
        "schema_version": LAYER3_DIAGNOSTIC_SCHEMA_VERSION,
        "status": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "semantics": {
            "changes_state_function_L": False,
            "enables_soft_gate": False,
            "raw_delta_L_gap_gate": False,
            "load_interaction_role": (
                "diagnose coefficient mismatch before defining any gate"
            ),
            "component_free_coefficients_role": (
                "ablation only; cannot certify reweighted Lyapunov potential"
            ),
            "station_load_imbalance_is_L_station": False,
            "station_load_imbalance_note": (
                "simulator outcome label: standard deviation of assigned/in-progress "
                "DELIVER task counts across stations; distinct from L_station queue pressure"
            ),
        },
        "bootstrap_repeats": int(bootstrap_repeats),
        "diagnostic_bootstrap_repeats": int(
            diagnostic_bootstrap_repeats
        ),
        "formal_candidate_component": formal_candidate_component,
        "random_seed": int(random_seed),
        "by_outcome": by_outcome,
    }


def _incremental_outcome_report(
    rows: Sequence[Mapping],
    *,
    wm_field: str,
    outcome_field: str,
    predictor_field: str = "lyapunov_l0_delta",
    candidate_label: str = "total",
    minimum_independent_clusters: int = 5,
    bootstrap_repeats: int,
    placebo_repeats: int,
    random_seed: int,
) -> dict:
    usable = [
        row for row in rows
        if row.get(wm_field) is not None
        and row.get(predictor_field) is not None
        and row.get(outcome_field) is not None
    ]
    if len(usable) < 6:
        return {"status": "INSUFFICIENT_ROWS", "n": len(usable)}
    wm = np.asarray([row[wm_field] for row in usable], dtype=float)
    predictor = np.asarray([row[predictor_field] for row in usable], dtype=float)
    outcome = np.asarray([row[outcome_field] for row in usable], dtype=float)
    outcome_variation = _outcome_variation_diagnostics(usable, outcome)
    if not outcome_variation["passed"]:
        return {
            "status": "INSUFFICIENT_EFFECTIVE_VARIATION",
            "n": len(usable),
            "outcome_variation": outcome_variation,
        }
    predictor_variation = _outcome_variation_diagnostics(
        usable,
        predictor,
        min_effective_groups=3,
    )
    predictor_variation["note"] = (
        "Resolution/coverage guard for the centred candidate predictor; "
        "this is not a raw Delta-L action-gap threshold."
    )
    if not predictor_variation["passed"]:
        return {
            "status": "PREDICTOR_HAS_INSUFFICIENT_VARIATION",
            "n": len(usable),
            "outcome_variation": outcome_variation,
            "predictor_variation": predictor_variation,
        }
    folds = _build_folds(usable)
    if not folds:
        return {"status": "INSUFFICIENT_INDEPENDENT_FOLDS", "n": len(usable)}
    candidate_feature = f"candidate:{candidate_label}"
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
        feature_names=["wm", candidate_feature],
        rows=usable,
    )
    if base_fit is None or augmented_fit is None:
        return {"status": "CROSS_VALIDATION_FAILED", "n": len(usable)}
    base_prediction, base_fold_diagnostics = base_fit
    augmented_prediction, augmented_fold_diagnostics = augmented_fit
    base = _prediction_metrics(outcome, base_prediction)
    augmented = _prediction_metrics(outcome, augmented_prediction)
    outcome_scale = float(outcome.std())
    if outcome_scale == 0.0:
        return {"status": "NO_OUTCOME_VARIATION", "n": len(usable)}
    improvement = float(
        (np.sqrt(np.mean((outcome - base_prediction) ** 2))
         - np.sqrt(np.mean((outcome - augmented_prediction) ** 2)))
        / outcome_scale
    )
    clusters = np.asarray([row["cluster"] for row in usable], dtype=object)
    improvement_ci = _bootstrap_clusters(
        clusters,
        lambda index: float(
            (np.sqrt(np.mean((outcome[index] - base_prediction[index]) ** 2))
             - np.sqrt(np.mean((outcome[index] - augmented_prediction[index]) ** 2)))
            / outcome_scale
        ),
        repeats=bootstrap_repeats,
        seed=random_seed + 37,
    )
    comparison = _model_comparison_diagnostics(
        usable,
        outcome,
        base_prediction,
        augmented_prediction,
        bootstrap_repeats=bootstrap_repeats,
        random_seed=random_seed + 53,
    )

    # Full-data standardised coefficient is descriptive; held-out improvement
    # and its cluster CI carry the inferential claim.
    full_model = _fit_linear(np.column_stack((wm, predictor)), outcome)
    beta_candidate = float(full_model[0][2])

    by_load = {}
    for load, sliced in comparison["by_load"].items():
        by_load[load] = {
            **sliced,
            "wm_plus_drift_rmse": sliced["wm_plus_predictor_rmse"],
            "wm_plus_drift_normalised_rmse": (
                sliced["wm_plus_predictor_normalised_rmse"]
            ),
        }

    rng = np.random.default_rng(random_seed + 73)
    groups = defaultdict(list)
    for index, row in enumerate(usable):
        groups[row["group"]].append(index)
    placebo_improvements = []
    for _ in range(int(placebo_repeats)):
        placebo = predictor.copy()
        for indices in groups.values():
            shuffled = np.asarray(indices, dtype=int)
            placebo[shuffled] = rng.permutation(placebo[shuffled])
        prediction = _cross_validated_predictions(np.column_stack((wm, placebo)), outcome, folds)
        if prediction is None:
            continue
        placebo_improvements.append(float(
            (np.sqrt(np.mean((outcome - base_prediction) ** 2))
             - np.sqrt(np.mean((outcome - prediction) ** 2))) / outcome_scale
        ))
    placebo_p95 = float(np.quantile(placebo_improvements, 0.95)) if placebo_improvements else None
    independent_clusters = int(np.unique(clusters).size)
    enough_clusters_for_certification = (
        independent_clusters >= int(minimum_independent_clusters)
    )
    if (
        enough_clusters_for_certification
        and improvement_ci is not None and improvement_ci[0] > 0.0 and (
        placebo_p95 is None or improvement > placebo_p95
        )
    ):
        status = "INCREMENTAL_INFORMATION_SUPPORTED"
    elif improvement_ci is not None and improvement_ci[1] <= 0.0:
        status = "NO_STABLE_INCREMENTAL_INFORMATION"
    else:
        status = "INCREMENTAL_INFORMATION_INCONCLUSIVE"
    return {
        "status": status,
        "candidate_component": candidate_label,
        "candidate_predictor_field": predictor_field,
        "n": len(usable),
        "folds": len(folds),
        "independent_seed_clusters": independent_clusters,
        "minimum_seed_clusters_for_certification": int(
            minimum_independent_clusters
        ),
        "enough_seed_clusters_for_certification": enough_clusters_for_certification,
        "outcome_variation": outcome_variation,
        "predictor_variation": predictor_variation,
        "wm_only": base,
        "wm_plus_candidate_drift": augmented,
        "wm_plus_analytic_drift": augmented,
        "normalised_rmse_improvement": improvement,
        "normalised_rmse_improvement_ci95_cluster_bootstrap": improvement_ci,
        "cluster_bootstrap_scope": (
            "resamples fixed leave-seed-out prediction errors; does not refit "
            "the model inside each bootstrap replicate"
        ),
        "candidate_coefficient_per_one_predictor_sd_full_data": beta_candidate,
        "drift_coefficient_per_one_predictor_sd_full_data": beta_candidate,
        "by_load": by_load,
        "by_seed": comparison["by_seed"],
        "ranking": comparison["ranking"],
        "group_equal_normalised_rmse_improvement": (
            comparison["group_equal_normalised_rmse_improvement"]
        ),
        "load_equal_mean_normalised_rmse_improvement": (
            comparison["load_equal_mean_normalised_rmse_improvement"]
        ),
        "load_heterogeneity_classification": (
            comparison["load_heterogeneity_classification"]
        ),
        "wm_only_fold_diagnostics": base_fold_diagnostics,
        "wm_plus_candidate_fold_diagnostics": augmented_fold_diagnostics,
        "candidate_standardised_coefficients_by_fold": [
            fold["standardised_coefficients"].get(candidate_feature)
            for fold in augmented_fold_diagnostics
        ],
        "within_group_permutation_placebo": {
            "repeats": len(placebo_improvements),
            "normalised_rmse_improvement": _quantiles(placebo_improvements),
            "p95": placebo_p95,
            "observed_beats_placebo_p95": (
                improvement > placebo_p95 if placebo_p95 is not None else None
            ),
        },
    }


def _formal_work_only_candidate_contract(
    independent: Mapping[str, Mapping],
    rows: Sequence[Mapping],
    *,
    primary_outcome: Optional[str],
    candidate_predictor_field: str,
    minimum_independent_clusters: int,
    wm_score_provenance_verified: bool,
    bootstrap_repeats: int,
    placebo_repeats: int,
    random_seed: int,
    protocol_override: Optional[Mapping] = None,
    candidate_component_label: str = "work",
    candidate_predictor_label: str = "candidate-group-centred Delta L_work",
    status_namespace: str = "FORMAL_WORK_ONLY",
    schema_version: str = LAYER3_FORMAL_CANDIDATE_SCHEMA_VERSION,
    semantic_candidate_check: str = "candidate_is_frozen_work_component",
    required_simulation_seeds: Optional[Sequence[int]] = None,
) -> dict:
    """Apply frozen Layer-3 gates to a work-derived candidate predictor."""
    protocol = dict(protocol_override or _formal_work_only_protocol())
    required_load_order = tuple(protocol["required_loads"])
    required_loads = set(required_load_order)

    clusters = sorted({str(row["cluster"]) for row in rows})
    cluster_loads: dict[str, set[str]] = defaultdict(set)
    cluster_load_runs: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    explicit_seed_rows = 0
    explicit_seed_values = set()
    for row in rows:
        cluster = str(row["cluster"])
        load = str(row["load"])
        sample = row.get("sample", {})
        run = str(
            sample.get("run_id")
            or sample.get("_source_path")
            or "unknown"
        )
        cluster_loads[cluster].add(load)
        cluster_load_runs[cluster][load].add(run)
        simulation_seed = sample.get("simulation_seed")
        explicit_seed_rows += int(simulation_seed is not None)
        if simulation_seed is not None:
            explicit_seed_values.add(int(simulation_seed))
    paired_clusters = [
        cluster for cluster, loads in cluster_loads.items()
        if loads == required_loads
    ]
    predictor_variation_by_load = {}
    for load in required_load_order:
        load_rows = [
            row for row in rows
            if str(row["load"]) == load
            and row.get(candidate_predictor_field) is not None
        ]
        values = np.asarray([
            row[candidate_predictor_field] for row in load_rows
        ], dtype=float)
        predictor_variation_by_load[load] = (
            _outcome_variation_diagnostics(
                load_rows,
                values,
                min_effective_groups=3,
            )
            if values.size
            else {
                "status": "NO_PREDICTOR_ROWS",
                "passed": False,
                "n": 0,
            }
        )
    data_checks = {
        "minimum_independent_seed_clusters": (
            len(clusters) >= int(minimum_independent_clusters)
        ),
        "every_seed_has_exact_paired_low_mid_high_arms": (
            bool(clusters) and len(paired_clusters) == len(clusters)
        ),
        "every_seed_load_has_exactly_one_run_arm": bool(clusters) and all(
            set(by_load) == required_loads
            and all(len(runs) == 1 for runs in by_load.values())
            for by_load in cluster_load_runs.values()
        ),
        "all_rows_have_explicit_simulation_seed": (
            bool(rows) and explicit_seed_rows == len(rows)
        ),
        "only_frozen_required_loads_are_present": (
            set().union(*cluster_loads.values()) == required_loads
            if cluster_loads else False
        ),
        "work_predictor_has_effective_variation_in_every_load": all(
            report.get("passed", False)
            for report in predictor_variation_by_load.values()
        ),
    }
    if required_simulation_seeds is not None:
        required_seed_set = {int(seed) for seed in required_simulation_seeds}
        data_checks["exact_frozen_simulation_seed_set"] = (
            explicit_seed_values == required_seed_set
        )
    data_contract = {
        "checks": data_checks,
        "passed": all(data_checks.values()),
        "independent_seed_clusters": len(clusters),
        "minimum_independent_seed_clusters": int(
            minimum_independent_clusters
        ),
        "paired_seed_clusters": len(paired_clusters),
        "required_loads": list(required_load_order),
        "loads_by_seed_cluster": {
            cluster: sorted(loads)
            for cluster, loads in sorted(cluster_loads.items())
        },
        "run_arms_by_seed_cluster_and_load": {
            cluster: {
                load: sorted(runs)
                for load, runs in sorted(by_load.items())
            }
            for cluster, by_load in sorted(cluster_load_runs.items())
        },
        "work_predictor_variation_by_load": predictor_variation_by_load,
    }
    if required_simulation_seeds is not None:
        data_contract["required_simulation_seeds"] = sorted(
            int(seed) for seed in required_simulation_seeds
        )
        data_contract["observed_simulation_seeds"] = sorted(
            explicit_seed_values
        )
    registered_resampling = protocol["resampling"]
    observed_resampling = {
        "bootstrap_repeats": int(bootstrap_repeats),
        "placebo_repeats": int(placebo_repeats),
        "random_seed": int(random_seed),
    }
    resampling_checks = {
        key: observed_resampling[key] == int(expected)
        for key, expected in registered_resampling.items()
    }
    resampling_contract = {
        "registered": dict(registered_resampling),
        "observed": observed_resampling,
        "checks": resampling_checks,
        "passed": all(resampling_checks.values()),
        "note": (
            "Formal certification freezes Monte-Carlo repetition counts and "
            "the RNG seed. Lower-cost runs remain diagnostics only."
        ),
    }

    def _positive_ci(value) -> bool:
        return bool(
            isinstance(value, (list, tuple))
            and len(value) == 2
            and value[0] is not None
            and float(value[0]) > 0.0
        )

    def _supported_harm(value) -> bool:
        return bool(
            isinstance(value, (list, tuple))
            and len(value) == 2
            and value[1] is not None
            and float(value[1]) <= 0.0
        )

    def evidence(field: str) -> dict:
        report = independent.get(field, {})
        folds = report.get("wm_plus_candidate_fold_diagnostics", [])
        by_load = report.get("by_load", {})
        informative_loads = {
            load: row for load, row in by_load.items()
            if row.get("normalised_rmse_improvement") is not None
        }
        required_loads_present = set(informative_loads) == required_loads
        load_ci_pass = required_loads_present and all(
            row["normalised_rmse_improvement"] > 0.0
            and _positive_ci(row.get(
                "normalised_rmse_improvement_ci95_seed_cluster_bootstrap"
            ))
            for row in informative_loads.values()
        )
        coefficients = report.get(
            "candidate_standardised_coefficients_by_fold", []
        )
        positive_coefficients = bool(coefficients) and all(
            value is not None and float(value) > 0.0 for value in coefficients
        )
        paired_load_folds = bool(folds) and all(
            set(map(str, fold.get("test_loads", ()))) == required_loads
            for fold in folds
        )
        by_seed = report.get("by_seed", {})
        seed_point_improvements = {
            str(seed): row.get("normalised_rmse_improvement")
            for seed, row in by_seed.items()
        }
        all_seed_points_positive = bool(seed_point_improvements) and all(
            value is not None and float(value) > 0.0
            for value in seed_point_improvements.values()
        )
        ranking = report.get("ranking", {})
        base_ranking = ranking.get("wm_only", {})
        candidate_ranking = ranking.get("wm_plus_predictor", {})
        base_regret = (
            base_ranking.get("normalised_selection_regret", {}).get("mean")
        )
        candidate_regret = (
            candidate_ranking.get("normalised_selection_regret", {}).get("mean")
        )
        ranking_pass = bool(
            (ranking.get("top1_accuracy_improvement") or 0.0) > 0.0
            and (ranking.get("pairwise_concordance_improvement") or 0.0) > 0.0
            and base_regret is not None
            and candidate_regret is not None
            and candidate_regret < base_regret
        )
        placebo_pass = bool(
            report.get("within_group_permutation_placebo", {}).get(
                "observed_beats_placebo_p95", False
            )
        )
        enough_clusters = bool(
            int(report.get("independent_seed_clusters", 0))
            >= int(minimum_independent_clusters)
        )
        overall_ci = report.get(
            "normalised_rmse_improvement_ci95_cluster_bootstrap"
        )
        checks = {
            "effective_outcome_variation": (
                report.get("outcome_variation", {}).get("passed", False)
            ),
            "enough_independent_seed_clusters": enough_clusters,
            "overall_improvement_positive": (
                report.get("normalised_rmse_improvement") is not None
                and report["normalised_rmse_improvement"] > 0.0
            ),
            "overall_seed_cluster_ci_lower_bound_positive": _positive_ci(
                overall_ci
            ),
            "beats_within_group_placebo_p95": placebo_pass,
            "all_required_loads_present": required_loads_present,
            "all_informative_load_ci_lower_bounds_positive": load_ci_pass,
            "each_held_out_seed_contains_all_required_loads": paired_load_folds,
            "each_held_out_seed_point_improvement_positive": (
                all_seed_points_positive
            ),
            "all_leave_seed_out_work_coefficients_positive": positive_coefficients,
            "candidate_ranking_and_regret_improve": ranking_pass,
        }
        return {
            "field": field,
            "available": bool(report),
            "checks": checks,
            "passed": bool(report and all(checks.values())),
            "independent_seed_clusters": report.get("independent_seed_clusters"),
            "normalised_rmse_improvement": report.get(
                "normalised_rmse_improvement"
            ),
            "normalised_rmse_improvement_ci95": report.get(
                "normalised_rmse_improvement_ci95_cluster_bootstrap"
            ),
            "seed_point_improvements": seed_point_improvements,
            "work_coefficients_by_fold": coefficients,
            "load_evidence": {
                load: {
                    "normalised_rmse_improvement": row.get(
                        "normalised_rmse_improvement"
                    ),
                    "ci95": row.get(
                        "normalised_rmse_improvement_ci95_seed_cluster_bootstrap"
                    ),
                }
                for load, row in informative_loads.items()
            },
            "ranking": ranking,
        }

    def secondary_evidence(field: str) -> dict:
        report = independent.get(field, {})
        available = bool(
            report
            and report.get("outcome_variation", {}).get("passed", False)
        )
        ci = report.get(
            "normalised_rmse_improvement_ci95_cluster_bootstrap"
        )
        supportive = bool(
            available
            and report.get("status") == "INCREMENTAL_INFORMATION_SUPPORTED"
            and report.get("within_group_permutation_placebo", {}).get(
                "observed_beats_placebo_p95", False
            )
        )
        supported_harm = bool(available and _supported_harm(ci))
        return {
            "field": field,
            "available_with_effective_variation": available,
            "supportive": supportive,
            "statistically_supported_overall_harm": supported_harm,
            "normalised_rmse_improvement": report.get(
                "normalised_rmse_improvement"
            ),
            "normalised_rmse_improvement_ci95": ci,
            "status": report.get("status", "NOT_EVALUATED"),
        }

    def guardrail_evidence(field: str) -> dict:
        report = independent.get(field, {})
        evaluable = bool(
            report
            and report.get("outcome_variation", {}).get("passed", False)
        )
        overall_ci = report.get(
            "normalised_rmse_improvement_ci95_cluster_bootstrap"
        )
        overall_harm = bool(evaluable and _supported_harm(overall_ci))
        harmful_loads = []
        if evaluable:
            for load, load_report in report.get("by_load", {}).items():
                if _supported_harm(load_report.get(
                    "normalised_rmse_improvement_ci95_seed_cluster_bootstrap"
                )):
                    harmful_loads.append(str(load))
        return {
            "field": field,
            "evaluable": evaluable,
            "unavailable_is_not_support": not evaluable,
            "statistically_supported_overall_harm": overall_harm,
            "loads_with_statistically_supported_harm": sorted(harmful_loads),
            "passed_no_supported_harm": not overall_harm and not harmful_loads,
            "note": (
                "Predictive Layer-3 precheck only; causal policy safety remains "
                "a frozen Layer-5 closed-loop question."
            ),
        }

    primary = evidence(primary_outcome) if primary_outcome else {
        "field": None,
        "available": False,
        "checks": {},
        "passed": False,
    }
    secondary = {
        field: secondary_evidence(field)
        for field in FORMAL_WORK_ONLY_REQUIRED_SECONDARY_OUTCOMES
    }
    supportive_secondary_outcomes = sum(
        int(row["supportive"]) for row in secondary.values()
    )
    secondary_checks = {
        "all_required_secondary_outcomes_have_effective_variation": all(
            row["available_with_effective_variation"]
            for row in secondary.values()
        ),
        "minimum_supportive_secondary_outcomes": (
            supportive_secondary_outcomes
            >= FORMAL_WORK_ONLY_MINIMUM_SUPPORTIVE_SECONDARY_OUTCOMES
        ),
        "no_secondary_outcome_has_supported_overall_harm": not any(
            row["statistically_supported_overall_harm"]
            for row in secondary.values()
        ),
    }
    all_secondary_passed = all(secondary_checks.values())
    guardrails = {
        field: guardrail_evidence(field)
        for field in FORMAL_WORK_ONLY_GUARDRAIL_OUTCOMES
    }
    guardrail_checks = {
        field: row["passed_no_supported_harm"]
        for field, row in guardrails.items()
    }
    guardrails_passed = all(guardrail_checks.values())
    primary_field_passed = primary_outcome == "realized_cost"
    semantic_checks = {
        "uses_checkpoint_scored_world_model_value": bool(
            wm_score_provenance_verified
        ),
        semantic_candidate_check: True,
        "state_function_remains_unchanged": True,
        "online_gate_is_not_learned_from_certificate_seeds": True,
    }
    passed = bool(
        data_contract["passed"]
        and resampling_contract["passed"]
        and all(semantic_checks.values())
        and primary_field_passed
        and primary["passed"]
        and all_secondary_passed
        and guardrails_passed
    )
    if passed:
        status = f"{status_namespace}_CANDIDATE_SUPPORTED"
    elif not all(semantic_checks.values()):
        status = f"{status_namespace}_SEMANTIC_CONTRACT_FAILED"
    elif not data_contract["passed"]:
        status = f"{status_namespace}_DATA_CONTRACT_FAILED"
    elif not resampling_contract["passed"]:
        status = f"{status_namespace}_RESAMPLING_CONTRACT_FAILED"
    elif not primary.get("available", False):
        status = f"{status_namespace}_PRIMARY_EVIDENCE_MISSING"
    elif not all(
        row.get("available_with_effective_variation", False)
        for row in secondary.values()
    ):
        status = f"{status_namespace}_SECONDARY_EVIDENCE_MISSING"
    elif not guardrails_passed:
        status = f"{status_namespace}_GUARDRAIL_HARM_DETECTED"
    else:
        status = f"{status_namespace}_CERTIFICATION_INCONCLUSIVE"
    return {
        "schema_version": schema_version,
        "frozen": True,
        "candidate_component": candidate_component_label,
        "candidate_predictor": candidate_predictor_label,
        "protocol": protocol,
        "data_contract": data_contract,
        "resampling_contract": resampling_contract,
        "semantic_checks": semantic_checks,
        "minimum_independent_seed_clusters": int(minimum_independent_clusters),
        "required_loads": list(required_load_order),
        "primary_outcome": primary_outcome,
        "required_primary_outcome": "realized_cost",
        "primary_outcome_matches_frozen_contract": primary_field_passed,
        "required_secondary_outcomes": list(
            FORMAL_WORK_ONLY_REQUIRED_SECONDARY_OUTCOMES
        ),
        "guardrail_outcomes_for_layer5": list(
            FORMAL_WORK_ONLY_GUARDRAIL_OUTCOMES
        ),
        "primary_evidence": primary,
        "secondary_evidence": secondary,
        "secondary_checks": secondary_checks,
        "supportive_secondary_outcomes": supportive_secondary_outcomes,
        "all_required_secondary_outcomes_passed": all_secondary_passed,
        "guardrail_evidence": guardrails,
        "guardrail_checks": guardrail_checks,
        "guardrails_passed": guardrails_passed,
        "status": status,
        "passed": passed,
        "semantics": {
            "changes_state_function_L": False,
            "online_soft_gate_enabled": False,
            "arrival_and_station_in_online_score": False,
            "arrival_and_station_remain_in_state_potential": True,
            "new_seed_results_may_not_change_candidate_component": True,
            "productive_progress_primary_evidence": False,
        },
    }


def _formal_work_group_range_candidate_contract(
    independent: Mapping[str, Mapping],
    rows: Sequence[Mapping],
    *,
    primary_outcome: Optional[str],
    candidate_predictor_field: str,
    minimum_independent_clusters: int,
    wm_score_provenance_verified: bool,
    bootstrap_repeats: int,
    placebo_repeats: int,
    random_seed: int,
) -> dict:
    """Apply the untouched-seed v2 contract to group-range work drift."""
    return _formal_work_only_candidate_contract(
        independent,
        rows,
        primary_outcome=primary_outcome,
        candidate_predictor_field=candidate_predictor_field,
        minimum_independent_clusters=minimum_independent_clusters,
        wm_score_provenance_verified=wm_score_provenance_verified,
        bootstrap_repeats=bootstrap_repeats,
        placebo_repeats=placebo_repeats,
        random_seed=random_seed,
        protocol_override=_formal_work_group_range_protocol(),
        candidate_component_label="work_group_range",
        candidate_predictor_label=(
            "candidate-group-centred Delta L_work divided by the candidate-"
            "group raw Delta L_work range"
        ),
        status_namespace="FORMAL_WORK_GROUP_RANGE",
        schema_version=LAYER3_FORMAL_WORK_GROUP_RANGE_SCHEMA_VERSION,
        semantic_candidate_check=(
            "candidate_is_frozen_work_group_range_transform"
        ),
        required_simulation_seeds=(
            FORMAL_WORK_GROUP_RANGE_CERTIFICATION_SEEDS
        ),
    )


def _group_range_slice_report(
    rows: Sequence[Mapping],
    outcome: np.ndarray,
    base_prediction: np.ndarray,
    augmented_prediction: np.ndarray,
    indices: Sequence[int],
    *,
    resolution: float,
    global_outcome_scale: float,
) -> dict:
    index = np.asarray(list(indices), dtype=int)
    if index.size < 3:
        return {"status": "INSUFFICIENT_ROWS", "rows": int(index.size)}
    local_rows = [rows[int(position)] for position in index]
    local_outcome = outcome[index]
    local_base = base_prediction[index]
    local_augmented = augmented_prediction[index]
    base_metrics = _prediction_metrics(local_outcome, local_base)
    augmented_metrics = _prediction_metrics(local_outcome, local_augmented)
    local_improvement = None
    if (
        base_metrics.get("normalised_rmse") is not None
        and augmented_metrics.get("normalised_rmse") is not None
    ):
        local_improvement = float(
            base_metrics["normalised_rmse"]
            - augmented_metrics["normalised_rmse"]
        )
    global_scale_improvement = None
    if global_outcome_scale > 0.0:
        global_scale_improvement = float(
            (base_metrics["rmse"] - augmented_metrics["rmse"])
            / global_outcome_scale
        )
    base_ranking = _ranking_metrics_from_predictions(
        local_rows,
        local_outcome,
        local_base,
        resolution=resolution,
    )
    augmented_ranking = _ranking_metrics_from_predictions(
        local_rows,
        local_outcome,
        local_augmented,
        resolution=resolution,
    )
    return {
        "status": "DIAGNOSTIC_COMPLETE",
        "rows": int(index.size),
        "groups": len({str(row["group"]) for row in local_rows}),
        "wm_only": base_metrics,
        "wm_plus_work_group_range": augmented_metrics,
        "normalised_rmse_improvement": local_improvement,
        "global_outcome_scale_normalised_rmse_improvement": (
            global_scale_improvement
        ),
        "ranking": {
            "wm_only": base_ranking,
            "wm_plus_work_group_range": augmented_ranking,
            "top1_accuracy_improvement": (
                augmented_ranking["top1_accuracy"]
                - base_ranking["top1_accuracy"]
                if augmented_ranking.get("top1_accuracy") is not None
                and base_ranking.get("top1_accuracy") is not None
                else None
            ),
            "pairwise_concordance_improvement": (
                augmented_ranking["pairwise_concordance"]
                - base_ranking["pairwise_concordance"]
                if augmented_ranking.get("pairwise_concordance") is not None
                and base_ranking.get("pairwise_concordance") is not None
                else None
            ),
        },
    }


def _summarise_subset_trials(trials: Sequence[Mapping]) -> dict:
    if not trials:
        return {
            "trials": 0,
            "selection_preservation_rate": None,
            "harmful_switch_rate": None,
            "beneficial_switch_rate": None,
            "mean_normalised_regret_change": None,
            "pairwise_score_concordance": None,
        }
    preserved = [bool(row["selection_preserved"]) for row in trials]
    harmful = [bool(row["harmful_switch"]) for row in trials]
    beneficial = [bool(row["beneficial_switch"]) for row in trials]
    regret_change = [
        float(row["normalised_regret_change"])
        for row in trials
        if row.get("normalised_regret_change") is not None
    ]
    pair_correct = sum(float(row["pair_correct"]) for row in trials)
    pair_total = sum(int(row["pair_total"]) for row in trials)
    return {
        "trials": len(trials),
        "selection_preservation_rate": float(np.mean(preserved)),
        "harmful_switch_rate": float(np.mean(harmful)),
        "beneficial_switch_rate": float(np.mean(beneficial)),
        "mean_normalised_regret_change": (
            float(np.mean(regret_change)) if regret_change else None
        ),
        "normalised_regret_change": _quantiles(regret_change),
        "pairwise_score_concordance": (
            float(pair_correct / pair_total) if pair_total else None
        ),
    }


def _work_group_range_development_diagnostics(
    rows: Sequence[Mapping],
    *,
    wm_field: str,
    outcome_field: str,
    predictor_field: str,
) -> dict:
    """Diagnose scale and candidate-set dependence without defining a gap."""
    usable = [
        row for row in rows
        if row.get(wm_field) is not None
        and row.get(outcome_field) is not None
        and row.get(predictor_field) is not None
        and row.get(WORK_GROUP_RANGE_RAW_DELTA_FIELD) is not None
        and row.get(WORK_GROUP_RANGE_RAW_RANGE_FIELD) is not None
    ]
    if len(usable) < 6:
        return {
            "schema_version": "lyapunov_work_group_range_preflight_v1",
            "status": "INSUFFICIENT_ROWS",
            "rows": len(usable),
        }

    wm = np.asarray([row[wm_field] for row in usable], dtype=float)
    predictor = np.asarray([row[predictor_field] for row in usable], dtype=float)
    outcome = np.asarray([row[outcome_field] for row in usable], dtype=float)
    folds = _build_folds(usable)
    if not folds:
        return {
            "schema_version": "lyapunov_work_group_range_preflight_v1",
            "status": "INSUFFICIENT_INDEPENDENT_FOLDS",
            "rows": len(usable),
        }
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
        feature_names=["wm", "work_group_range"],
        rows=usable,
    )
    if base_fit is None or augmented_fit is None:
        return {
            "schema_version": "lyapunov_work_group_range_preflight_v1",
            "status": "CROSS_VALIDATION_FAILED",
            "rows": len(usable),
        }
    base_prediction, _ = base_fit
    augmented_prediction, fold_reports = augmented_fit
    variation = _outcome_variation_diagnostics(usable, outcome)
    resolution = float(variation.get("float32_resolution_floor") or 0.0)
    global_scale = float(outcome.std())

    group_indices: dict[object, list[int]] = defaultdict(list)
    for index, row in enumerate(usable):
        group_indices[row["group"]].append(index)
    group_records = []
    for group, indices in group_indices.items():
        first = usable[indices[0]]
        group_records.append({
            "group": group,
            "indices": indices,
            "raw_range": float(first[WORK_GROUP_RANGE_RAW_RANGE_FIELD]),
            "candidate_count": len(indices),
            "load": str(first["load"]),
        })

    zero_range = [row for row in group_records if row["raw_range"] == 0.0]
    nonzero = sorted(
        (row for row in group_records if row["raw_range"] > 0.0),
        key=lambda row: (row["raw_range"], str(row["group"])),
    )
    range_strata: dict[str, list[dict]] = defaultdict(list)
    if zero_range:
        range_strata["zero"].extend(zero_range)
    for rank, row in enumerate(nonzero):
        quartile = min(3, int(4 * rank / max(len(nonzero), 1))) + 1
        range_strata[f"nonzero_q{quartile}"].append(row)
    range_reports = {}
    for name, records in range_strata.items():
        indices = [index for row in records for index in row["indices"]]
        report = _group_range_slice_report(
            usable,
            outcome,
            base_prediction,
            augmented_prediction,
            indices,
            resolution=resolution,
            global_outcome_scale=global_scale,
        )
        report["raw_range"] = _quantiles([
            row["raw_range"] for row in records
        ])
        range_reports[name] = report

    count_strata: dict[int, list[dict]] = defaultdict(list)
    for row in group_records:
        count_strata[int(row["candidate_count"])].append(row)
    count_reports = {}
    for count, records in sorted(count_strata.items()):
        indices = [index for row in records for index in row["indices"]]
        count_reports[str(count)] = _group_range_slice_report(
            usable,
            outcome,
            base_prediction,
            augmented_prediction,
            indices,
            resolution=resolution,
            global_outcome_scale=global_scale,
        )

    all_indices = np.arange(len(usable))
    subset_trials = []
    for test in folds:
        train = np.setdiff1d(all_indices, test, assume_unique=False)
        if train.size <= 3:
            continue
        model = _fit_linear(
            np.column_stack((wm[train], predictor[train])),
            outcome[train],
        )
        test_set = set(int(value) for value in test)
        test_groups: dict[object, list[int]] = defaultdict(list)
        for index in test:
            test_groups[usable[int(index)]["group"]].append(int(index))
        for group, indices in test_groups.items():
            if len(indices) < 3 or not all(index in test_set for index in indices):
                continue
            full_x = np.column_stack((wm[indices], predictor[indices]))
            full_prediction = _predict_linear(full_x, model)
            raw = np.asarray([
                usable[index][WORK_GROUP_RANGE_RAW_DELTA_FIELD]
                for index in indices
            ], dtype=float)
            truth = outcome[indices]
            spread = float(truth.max() - truth.min())
            for removed in range(len(indices)):
                keep = np.asarray([
                    index for index in range(len(indices)) if index != removed
                ], dtype=int)
                subset_raw = raw[keep]
                subset_range = float(subset_raw.max() - subset_raw.min())
                subset_predictor = (
                    (subset_raw - subset_raw.mean()) / subset_range
                    if subset_range > 0.0
                    else np.zeros(subset_raw.shape, dtype=float)
                )
                subset_x = np.column_stack((
                    wm[np.asarray(indices, dtype=int)[keep]],
                    subset_predictor,
                ))
                recomputed_prediction = _predict_linear(subset_x, model)
                reference_prediction = full_prediction[keep]
                reference_choice = int(np.argmin(reference_prediction))
                recomputed_choice = int(np.argmin(recomputed_prediction))
                subset_truth = truth[keep]
                regret_change = float(
                    subset_truth[recomputed_choice]
                    - subset_truth[reference_choice]
                )
                normalised_regret_change = (
                    regret_change / spread if spread > resolution else None
                )
                pair_correct = 0.0
                pair_total = 0
                for left in range(keep.size):
                    for right in range(left + 1, keep.size):
                        reference_sign = np.sign(
                            reference_prediction[left]
                            - reference_prediction[right]
                        )
                        recomputed_sign = np.sign(
                            recomputed_prediction[left]
                            - recomputed_prediction[right]
                        )
                        pair_correct += (
                            0.5
                            if reference_sign == 0.0 or recomputed_sign == 0.0
                            else float(reference_sign == recomputed_sign)
                        )
                        pair_total += 1
                subset_trials.append({
                    "load": str(usable[indices[0]]["load"]),
                    "original_candidate_count": len(indices),
                    "selection_preserved": (
                        reference_choice == recomputed_choice
                    ),
                    "harmful_switch": regret_change > resolution,
                    "beneficial_switch": regret_change < -resolution,
                    "normalised_regret_change": normalised_regret_change,
                    "pair_correct": pair_correct,
                    "pair_total": pair_total,
                })

    by_load = {}
    for load in sorted({str(row["load"]) for row in subset_trials}):
        by_load[load] = _summarise_subset_trials([
            row for row in subset_trials if str(row["load"]) == load
        ])
    by_count = {}
    for count in sorted({int(row["original_candidate_count"]) for row in subset_trials}):
        by_count[str(count)] = _summarise_subset_trials([
            row for row in subset_trials
            if int(row["original_candidate_count"]) == count
        ])

    return {
        "schema_version": "lyapunov_work_group_range_preflight_v1",
        "status": "DEVELOPMENT_DIAGNOSTIC_COMPLETE",
        "role": "NOT_A_CERTIFICATION_AND_NOT_AN_ONLINE_GATE",
        "semantics": {
            "changes_state_function_L": False,
            "raw_gap_gate": False,
            "load_gate": False,
            "range_strata_are_descriptive": True,
            "candidate_subset_test": (
                "leave one observed candidate out, recompute group-range, and "
                "reuse the held-out-seed model fitted on complete training groups"
            ),
            "candidate_supersets_above_observed_max": "NOT_EVALUATED",
        },
        "rows": len(usable),
        "groups": len(group_records),
        "observed_candidate_count": _quantiles([
            float(row["candidate_count"]) for row in group_records
        ]),
        "raw_range": {
            "all_groups": _quantiles([
                row["raw_range"] for row in group_records
            ]),
            "nonzero_groups": _quantiles([
                row["raw_range"] for row in nonzero
            ]),
            "zero_range_groups": len(zero_range),
            "strata": range_reports,
        },
        "candidate_count_strata": count_reports,
        "leave_one_candidate_out_stability": {
            "overall": _summarise_subset_trials(subset_trials),
            "by_load": by_load,
            "by_original_candidate_count": by_count,
            "excluded_groups_with_fewer_than_three_candidates": sum(
                int(row["candidate_count"] < 3) for row in group_records
            ),
            "observed_max_candidate_count": max(
                (int(row["candidate_count"]) for row in group_records),
                default=0,
            ),
            "online_all_idle_above_observed_max": "NOT_EVALUATED",
        },
        "fold_diagnostics": fold_reports,
    }


def evaluate_incremental_information(
    samples: Sequence[Mapping],
    *,
    wm_score_field: Optional[str],
    wm_score_provenance_verified: bool = False,
    candidate_component: str = "total",
    minimum_independent_clusters: Optional[int] = None,
    outcome_fields: Sequence[str] = DEFAULT_OUTCOME_FIELDS,
    bootstrap_repeats: int = 1000,
    placebo_repeats: int = 200,
    random_seed: int = 2026,
) -> dict:
    """Layer 3: test whether analytic drift adds held-out value beyond WM."""
    candidate_component = str(candidate_component).strip().lower()
    allowed_components = ("total", "work", "work_group_range")
    if candidate_component not in allowed_components:
        raise ValueError(
            f"candidate_component must be one of {allowed_components}, got "
            f"{candidate_component!r}"
        )
    formal_work_candidates = {"work", "work_group_range"}
    if (
        candidate_component in formal_work_candidates
        and minimum_independent_clusters is not None
        and int(minimum_independent_clusters)
        < FORMAL_WORK_ONLY_MINIMUM_SEED_CLUSTERS
    ):
        raise ValueError(
            "formal work-only Layer 3 requires at least "
            f"{FORMAL_WORK_ONLY_MINIMUM_SEED_CLUSTERS} independent seed "
            "clusters; the frozen threshold cannot be lowered"
        )
    if minimum_independent_clusters is None:
        minimum_independent_clusters = (
            FORMAL_WORK_ONLY_MINIMUM_SEED_CLUSTERS
            if candidate_component in formal_work_candidates else 5
        )
    if int(minimum_independent_clusters) < 2:
        raise ValueError("minimum_independent_clusters must be at least 2")
    if candidate_component in formal_work_candidates:
        frozen_fields = {
            FORMAL_WORK_ONLY_PRIMARY_OUTCOME,
            *FORMAL_WORK_ONLY_REQUIRED_SECONDARY_OUTCOMES,
            *FORMAL_WORK_ONLY_GUARDRAIL_OUTCOMES,
        }
        omitted = sorted(frozen_fields.difference(outcome_fields))
        if omitted:
            raise ValueError(
                "formal work-only Layer 3 cannot omit frozen outcome fields: "
                + ", ".join(omitted)
            )
    if not wm_score_field:
        return {
            "layer": 3,
            "name": "incremental_information_over_world_model",
            "status": "NOT_EVALUATED_MISSING_TRUE_WM_SCORE_FIELD",
            "supported": False,
            "note": "heuristic_cost is not silently accepted as a World Model score",
        }
    if wm_score_field == "heuristic_cost":
        proxy_warning = True
    else:
        proxy_warning = False
    groups = _groups(samples)
    fields = [wm_score_field, "lyapunov_l0_delta", *outcome_fields, DEPENDENT_LEDGER_FIELD]
    centred, missing = _centred_rows(groups, fields)
    component_fields = _attach_centred_component_deltas(centred)
    if candidate_component == "total":
        candidate_predictor_field = "lyapunov_l0_delta"
    elif candidate_component == "work_group_range":
        candidate_predictor_field = _attach_work_group_range(centred)
    else:
        candidate_predictor_field = component_fields[candidate_component]
    if missing[wm_score_field] == len(centred):
        return {
            "layer": 3,
            "name": "incremental_information_over_world_model",
            "wm_score_field": wm_score_field,
            "status": "NOT_EVALUATED_WM_SCORE_ABSENT_FROM_DATA",
            "supported": False,
        }
    candidate_values = [
        row.get(candidate_predictor_field) for row in centred
        if row.get(candidate_predictor_field) is not None
    ]
    if not candidate_values:
        return {
            "layer": 3,
            "name": "incremental_information_over_world_model",
            "wm_score_field": wm_score_field,
            "candidate_component": candidate_component,
            "candidate_predictor_field": candidate_predictor_field,
            "status": "NOT_EVALUATED_CANDIDATE_COMPONENT_ABSENT",
            "supported": False,
        }
    independent = {}
    formal_work_fields = {
        "realized_cost",
        *FORMAL_WORK_ONLY_REQUIRED_SECONDARY_OUTCOMES,
    }
    for index, field in enumerate(outcome_fields):
        if missing[field] == len(centred):
            continue
        independent[field] = _incremental_outcome_report(
            centred,
            wm_field=wm_score_field,
            outcome_field=field,
            predictor_field=candidate_predictor_field,
            candidate_label=candidate_component,
            minimum_independent_clusters=int(minimum_independent_clusters),
            bootstrap_repeats=bootstrap_repeats,
            placebo_repeats=(
                placebo_repeats
                if candidate_component not in formal_work_candidates
                or field in formal_work_fields
                else min(placebo_repeats, 200)
            ),
            random_seed=random_seed + 10007 * (index + 1),
        )
    primary = "realized_cost" if "realized_cost" in independent else next(iter(independent), None)
    primary_status = independent.get(primary, {}).get("status") if primary else None
    diagnostics_v2 = _layer3_load_component_diagnostics(
        centred,
        wm_field=wm_score_field,
        outcome_fields=tuple(independent),
        bootstrap_repeats=bootstrap_repeats,
        random_seed=random_seed + 700001,
        formal_candidate_component=candidate_component,
    )
    primary_diagnostic = (
        diagnostics_v2.get("by_outcome", {}).get(primary, {})
        if primary else {}
    )
    if candidate_component == "total":
        candidate_comparison = (
            primary_diagnostic.get("shared_coefficient_model", {})
            .get("comparison", {})
        )
    elif candidate_component == "work_group_range":
        candidate_comparison = independent.get(primary, {}) if primary else {}
    else:
        candidate_comparison = (
            primary_diagnostic.get("component_ablation", {})
            .get(candidate_component, {})
            .get("comparison", {})
        )
    informative_load_rows = [
        row for row in candidate_comparison.get("by_load", {}).values()
        if row.get("normalised_rmse_improvement") is not None
    ]
    load_robustness_passed = bool(informative_load_rows) and all(
        row["normalised_rmse_improvement"] > 0.0
        and isinstance(
            row.get("normalised_rmse_improvement_ci95_seed_cluster_bootstrap"),
            list,
        )
        and row["normalised_rmse_improvement_ci95_seed_cluster_bootstrap"][0] > 0.0
        for row in informative_load_rows
    )
    load_robustness = {
        "informative_loads": len(informative_load_rows),
        "all_informative_loads_positive_with_seed_cluster_ci": (
            load_robustness_passed
        ),
        "classification": candidate_comparison.get(
            "load_heterogeneity_classification"
        ),
        "required_for_universal_online_auxiliary": True,
    }
    formal_candidate_contract = None
    if candidate_component == "work":
        formal_candidate_contract = _formal_work_only_candidate_contract(
            independent,
            centred,
            primary_outcome=primary,
            candidate_predictor_field=candidate_predictor_field,
            minimum_independent_clusters=int(minimum_independent_clusters),
            wm_score_provenance_verified=bool(
                wm_score_provenance_verified and not proxy_warning
            ),
            bootstrap_repeats=bootstrap_repeats,
            placebo_repeats=placebo_repeats,
            random_seed=random_seed,
        )
    elif candidate_component == "work_group_range":
        formal_candidate_contract = (
            _formal_work_group_range_candidate_contract(
                independent,
                centred,
                primary_outcome=primary,
                candidate_predictor_field=candidate_predictor_field,
                minimum_independent_clusters=int(
                    minimum_independent_clusters
                ),
                wm_score_provenance_verified=bool(
                    wm_score_provenance_verified and not proxy_warning
                ),
                bootstrap_repeats=bootstrap_repeats,
                placebo_repeats=placebo_repeats,
                random_seed=random_seed,
            )
        )

    group_range_preflight = None
    if candidate_component == "work_group_range" and primary:
        group_range_preflight = _work_group_range_development_diagnostics(
            centred,
            wm_field=wm_score_field,
            outcome_field=primary,
            predictor_field=candidate_predictor_field,
        )

    if proxy_warning:
        status = "PROXY_ONLY_NOT_A_WORLD_MODEL_INCREMENTAL_TEST"
    elif (
        candidate_component in formal_work_candidates
        and formal_candidate_contract["passed"]
    ):
        status = "INCREMENTAL_INFORMATION_SUPPORTED"
    elif candidate_component in formal_work_candidates and primary_status == (
        "NO_STABLE_INCREMENTAL_INFORMATION"
    ):
        status = "NO_STABLE_INCREMENTAL_INFORMATION"
    elif candidate_component in formal_work_candidates:
        status = "INCREMENTAL_INFORMATION_INCONCLUSIVE"
    elif (
        primary_status == "INCREMENTAL_INFORMATION_SUPPORTED"
        and load_robustness_passed
    ):
        status = "INCREMENTAL_INFORMATION_SUPPORTED"
    elif primary_status == "INCREMENTAL_INFORMATION_SUPPORTED":
        status = "LOAD_ROBUSTNESS_INCONCLUSIVE"
    elif primary_status == "NO_STABLE_INCREMENTAL_INFORMATION":
        status = "NO_STABLE_INCREMENTAL_INFORMATION"
    elif independent:
        status = "INCREMENTAL_INFORMATION_INCONCLUSIVE"
    else:
        status = "NOT_EVALUATED_NO_INDEPENDENT_OUTCOME"

    progress = None
    if missing[DEPENDENT_LEDGER_FIELD] < len(centred):
        progress = _incremental_outcome_report(
            centred,
            wm_field=wm_score_field,
            outcome_field=DEPENDENT_LEDGER_FIELD,
            predictor_field=candidate_predictor_field,
            candidate_label=candidate_component,
            minimum_independent_clusters=int(minimum_independent_clusters),
            bootstrap_repeats=bootstrap_repeats,
            placebo_repeats=(
                placebo_repeats
                if candidate_component not in formal_work_candidates
                else min(placebo_repeats, 200)
            ),
            random_seed=random_seed + 99991,
        )
    return {
        "layer": 3,
        "name": "incremental_information_over_world_model",
        "question": "After observing the real WM score, does centred analytic drift improve held-out prediction?",
        "semantics": {
            "group_fixed_effect": "all variables are centred inside candidate group",
            "validation": "leave-seed-out across all loads when possible, otherwise leave-run/source or deterministic group folds",
            "raw_gap_gate": False,
            "soft_gate_status": (
                "not designed here; load/component interactions are diagnostic only"
            ),
            "formal_candidate_mode": (
                candidate_component in formal_work_candidates
            ),
            "candidate_component_frozen_for_new_seeds": (
                candidate_component in formal_work_candidates
            ),
            "state_function_is_frozen": True,
            "station_load_imbalance_distinct_from_L_station": True,
            "heuristic_proxy_used": proxy_warning,
            "wm_score_provenance_verified": bool(
                wm_score_provenance_verified and not proxy_warning
            ),
            "wm_score_provenance_requirement": (
                "Formal work-only certification requires checkpoint scoring "
                "inside the evaluator. The frozen checkpoint must not have "
                "trained on the evaluated seeds."
            ),
        },
        "wm_score_field": wm_score_field,
        "candidate_component": candidate_component,
        "candidate_predictor_field": candidate_predictor_field,
        "minimum_independent_clusters": int(minimum_independent_clusters),
        "primary_outcome": primary,
        "independent_outcomes": independent,
        "load_component_diagnostics_v2": diagnostics_v2,
        "primary_load_robustness": load_robustness,
        "formal_candidate_contract": formal_candidate_contract,
        "work_group_range_preflight": group_range_preflight,
        "ledger_coupled_evidence": {
            "field": DEPENDENT_LEDGER_FIELD,
            "result": progress,
            "primary_validity_evidence": False,
            "reason": "P_prod and L_work are computed from the same physical work ledger.",
        },
        "missing_values": missing,
        "status": status,
        "supported": status == "INCREMENTAL_INFORMATION_SUPPORTED",
    }


def _ranking_report(rows: Sequence[Mapping], true_field: str, predicted_field: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        if row.get(true_field) is not None and row.get(predicted_field) is not None:
            grouped[row["group"]].append(row)
    regrets = []
    normalised_regrets = []
    top1 = 0
    usable = 0
    pair_correct = 0.0
    pair_total = 0
    for members in grouped.values():
        if len(members) < 2:
            continue
        true = np.asarray([row[true_field] for row in members], dtype=float)
        predicted = np.asarray([row[predicted_field] for row in members], dtype=float)
        true_best = int(np.argmin(true))
        predicted_best = int(np.argmin(predicted))
        regret = float(true[predicted_best] - true[true_best])
        spread = float(true.max() - true.min())
        relative_epsilon = (
            np.finfo(float).eps * float(np.max(np.abs(true))) * 16.0
        )
        regrets.append(regret)
        if spread > relative_epsilon:
            normalised_regrets.append(regret / spread)
        top1 += int(predicted_best == true_best)
        usable += 1
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                true_sign = np.sign(true[left] - true[right])
                pred_sign = np.sign(predicted[left] - predicted[right])
                if true_sign == 0.0 or pred_sign == 0.0:
                    pair_correct += 0.5
                else:
                    pair_correct += float(true_sign == pred_sign)
                pair_total += 1
    return {
        "groups": usable,
        "top1_min_drift_accuracy": float(top1 / usable) if usable else None,
        "pairwise_concordance": float(pair_correct / pair_total) if pair_total else None,
        "pairs": pair_total,
        "selection_regret": _quantiles(regrets),
        "normalised_selection_regret_fraction_of_group_range": _quantiles(normalised_regrets),
    }


def evaluate_drift_prediction(
    samples: Sequence[Mapping],
    *,
    predicted_drift_field: Optional[str],
    learned_report: Optional[Mapping] = None,
    uncertainty_field: Optional[str] = None,
    predicted_component_fields: Optional[Mapping[str, str]] = None,
    bootstrap_repeats: int = 1000,
    random_seed: int = 2026,
) -> dict:
    """Layer 4: compare learned and analytic action-dependent drift."""
    if not predicted_drift_field:
        if isinstance(learned_report, Mapping):
            metrics = learned_report.get("validation")
            if not isinstance(metrics, Mapping):
                metrics = learned_report
            group_range = metrics.get("continuous_group_range")
            if isinstance(group_range, Mapping):
                source_schema = learned_report.get("schema_version")
                is_held_out_layer4 = (
                    source_schema
                    == "work_drift_group_range_evaluation_v1"
                )
                source_supported = bool(learned_report.get("supported", False))
                checkpoint_passed = bool(
                    (learned_report.get("checkpoint_contract") or {})
                    .get("passed", False)
                )
                data_passed = bool(
                    (learned_report.get("data_contract") or {})
                    .get("passed", False)
                )
                metric_passed = bool(
                    (learned_report.get("metric_contract") or {})
                    .get("passed", False)
                )
                supported = bool(
                    is_held_out_layer4
                    and source_supported
                    and checkpoint_passed
                    and data_passed
                    and metric_passed
                )
                source_status = str(learned_report.get("status", "UNKNOWN"))
                if supported:
                    status = "WORLD_MODEL_DRIFT_ESTIMATION_SUPPORTED"
                elif source_status.endswith("_FAILED"):
                    status = "WORLD_MODEL_DRIFT_ESTIMATION_FAILED"
                else:
                    status = "WORLD_MODEL_DRIFT_ESTIMATION_INCONCLUSIVE"
                return {
                    "layer": 4,
                    "name": "world_model_drift_estimation",
                    "source": (
                        "held_out_work_group_range_report"
                        if is_held_out_layer4
                        else "development_work_group_range_report"
                    ),
                    "source_schema_version": source_schema,
                    "source_status": source_status,
                    "continuous_group_range": dict(group_range),
                    "raw_work_drift": metrics.get("raw_work_drift"),
                    "endpoint_station_work": metrics.get(
                        "endpoint_station_work"
                    ),
                    "by_load": metrics.get("by_load"),
                    "by_candidate_count": metrics.get(
                        "by_candidate_count"
                    ),
                    "raw_range_strata": metrics.get("raw_range_strata"),
                    "source_contracts": {
                        "checkpoint_passed": checkpoint_passed,
                        "data_passed": data_passed,
                        "metric_passed": metric_passed,
                    },
                    "semantics": {
                        "target": (
                            "isolated H-step analytic Delta L_work, compared "
                            "as the within-candidate-group range coordinate"
                        ),
                        "raw_gap_gate": False,
                        "load_gate": False,
                        "unknown_future_orders": False,
                        "continuation_policy": False,
                        "td_tail": False,
                        "absolute_raw_drift": "diagnostic_only",
                    },
                    "status": status,
                    "supported": supported,
                }
            continuous = metrics.get("continuous_group_centered")
            if isinstance(continuous, Mapping):
                nrmse = _finite(continuous.get("normalized_rmse"))
                pearson = _finite(continuous.get("pearson"))
                concordance = _finite(
                    continuous.get("all_non_tie_pair_concordance")
                )
                if (
                    nrmse is not None and nrmse < 1.0
                    and pearson is not None and pearson > 0.0
                    and concordance is not None and concordance > 0.5
                ):
                    status = "WORLD_MODEL_DRIFT_ESTIMATION_SUPPORTED"
                elif nrmse is not None and nrmse >= 1.0:
                    status = "WORLD_MODEL_DRIFT_ESTIMATION_FAILED"
                else:
                    status = "WORLD_MODEL_DRIFT_ESTIMATION_INCONCLUSIVE"
                return {
                    "layer": 4,
                    "name": "world_model_drift_estimation",
                    "source": "learned_head_validation_report",
                    "source_schema_version": learned_report.get(
                        "schema_version"
                    ),
                    "continuous_group_centered": dict(continuous),
                    "semantics": {
                        "raw_gap_gate": False,
                        "historical_gap_slices": "descriptive_only",
                    },
                    "status": status,
                    "supported": status
                    == "WORLD_MODEL_DRIFT_ESTIMATION_SUPPORTED",
                }
        return {
            "layer": 4,
            "name": "world_model_drift_estimation",
            "status": "NOT_EVALUATED_MISSING_PREDICTED_DRIFT_FIELD",
            "supported": False,
        }
    groups = _groups(samples)
    fields = ["lyapunov_l0_delta", predicted_drift_field]
    centred, missing = _centred_rows(groups, fields)
    if uncertainty_field:
        missing[uncertainty_field] = 0
        for row in centred:
            # Uncertainty is a non-negative calibration scale, not a signed
            # action effect.  Keep it on its raw scale while drift and drift
            # prediction remain group-centred.
            value = _field(row["sample"], uncertainty_field)
            row[uncertainty_field] = value
            missing[uncertainty_field] += int(value is None)
    usable = [
        row for row in centred
        if row["lyapunov_l0_delta"] is not None and row[predicted_drift_field] is not None
    ]
    if len(usable) < 3:
        return {
            "layer": 4,
            "name": "world_model_drift_estimation",
            "predicted_drift_field": predicted_drift_field,
            "status": "NOT_EVALUATED_PREDICTION_ABSENT_OR_INSUFFICIENT",
            "supported": False,
            "usable_rows": len(usable),
        }
    truth = np.asarray([row["lyapunov_l0_delta"] for row in usable], dtype=float)
    predicted = np.asarray([row[predicted_drift_field] for row in usable], dtype=float)
    error = predicted - truth
    scale = float(truth.std())
    nrmse = float(np.sqrt(np.mean(error ** 2)) / scale) if scale > 0.0 else None
    clusters = np.asarray([row["cluster"] for row in usable], dtype=object)
    spearman = _spearman(truth, predicted)
    spearman_ci = _bootstrap_clusters(
        clusters,
        lambda index: _spearman(truth[index], predicted[index]),
        repeats=bootstrap_repeats,
        seed=random_seed + 127,
    )
    nrmse_ci = (
        _bootstrap_clusters(
            clusters,
            lambda index: float(
                np.sqrt(np.mean((predicted[index] - truth[index]) ** 2)) / scale
            ),
            repeats=bootstrap_repeats,
            seed=random_seed + 257,
        )
        if scale > 0.0 else None
    )
    ranking = _ranking_report(usable, "lyapunov_l0_delta", predicted_drift_field)

    raw_pairs = [
        (
            _field(row["sample"], "lyapunov_l0_delta"),
            _field(row["sample"], predicted_drift_field),
        )
        for row in usable
    ]
    raw_pairs = [row for row in raw_pairs if row[0] is not None and row[1] is not None]
    if raw_pairs:
        raw_truth = np.asarray([row[0] for row in raw_pairs], dtype=float)
        raw_predicted = np.asarray([row[1] for row in raw_pairs], dtype=float)
        raw_scale = float(raw_truth.std())
        absolute_calibration = {
            "n": len(raw_pairs),
            "bias_predicted_minus_true": float(np.mean(raw_predicted - raw_truth)),
            "normalised_rmse": (
                float(np.sqrt(np.mean((raw_predicted - raw_truth) ** 2)) / raw_scale)
                if raw_scale > 0.0 else None
            ),
            "spearman": _spearman(raw_truth, raw_predicted),
            "action_selection_primary_evidence": False,
            "note": "Common group offsets do not change action ranking but matter for absolute drift calibration.",
        }
    else:
        absolute_calibration = {"n": 0}

    uncertainty = None
    if uncertainty_field:
        uncertain_rows = [row for row in usable if row.get(uncertainty_field) is not None]
        if uncertain_rows:
            uncertainty_values = np.asarray([row[uncertainty_field] for row in uncertain_rows], dtype=float)
            absolute_error = np.asarray([
                abs(float(row[predicted_drift_field]) - float(row["lyapunov_l0_delta"]))
                for row in uncertain_rows
            ], dtype=float)
            uncertainty = {
                "n": len(uncertain_rows),
                "spearman_uncertainty_vs_absolute_error": _spearman(uncertainty_values, absolute_error),
                "uncertainty": _quantiles(uncertainty_values.tolist()),
                "absolute_error": _quantiles(absolute_error.tolist()),
            }

    component_reports = {}
    for name, field in (predicted_component_fields or {}).items():
        component_rows = []
        for members in groups.values():
            raw = [
                (_component_delta(sample, name), _field(sample, field))
                for sample in members
            ]
            usable_raw = [row for row in raw if row[0] is not None and row[1] is not None]
            if len(usable_raw) < 2:
                continue
            truth_mean = float(np.mean([row[0] for row in usable_raw]))
            predicted_mean = float(np.mean([row[1] for row in usable_raw]))
            component_rows.extend(
                (float(truth_value - truth_mean), float(predicted_value - predicted_mean))
                for truth_value, predicted_value in usable_raw
            )
        if component_rows:
            component_truth = np.asarray([row[0] for row in component_rows], dtype=float)
            component_predicted = np.asarray([row[1] for row in component_rows], dtype=float)
            component_scale = float(component_truth.std())
            component_reports[name] = {
                "n": len(component_rows),
                "normalised_rmse": (
                    float(np.sqrt(np.mean((component_predicted - component_truth) ** 2)) / component_scale)
                    if component_scale > 0.0 else None
                ),
                "spearman": _spearman(component_truth, component_predicted),
            }

    by_load = {}
    for load in sorted({row["load"] for row in usable}):
        load_rows = [row for row in usable if row["load"] == load]
        load_truth = np.asarray([row["lyapunov_l0_delta"] for row in load_rows], dtype=float)
        load_predicted = np.asarray([row[predicted_drift_field] for row in load_rows], dtype=float)
        load_scale = float(load_truth.std())
        by_load[load] = {
            "n": len(load_rows),
            "normalised_rmse": (
                float(np.sqrt(np.mean((load_predicted - load_truth) ** 2)) / load_scale)
                if load_scale > 0.0 else None
            ),
            "spearman": _spearman(load_truth, load_predicted),
        }

    if nrmse is None:
        status = "TRUE_DRIFT_HAS_NO_WITHIN_GROUP_VARIATION"
    elif spearman_ci is not None and spearman_ci[0] > 0.0 and nrmse < 1.0:
        status = "WORLD_MODEL_DRIFT_ESTIMATION_SUPPORTED"
    elif spearman_ci is not None and spearman_ci[1] <= 0.0:
        status = "WORLD_MODEL_DRIFT_ESTIMATION_FAILED"
    else:
        status = "WORLD_MODEL_DRIFT_ESTIMATION_INCONCLUSIVE"
    return {
        "layer": 4,
        "name": "world_model_drift_estimation",
        "question": "Can learned predictions reproduce centred analytic drift on unseen contexts?",
        "semantics": {
            "target": "candidate-group-centred analytic Delta L",
            "raw_gap_gate": False,
            "normalised_rmse_baseline": "1.0 is the group-mean predictor scale",
            "prediction_provenance_requirement": (
                "Predictions must be generated out of sample; the evaluator assumes the supplied data split is held out."
            ),
        },
        "predicted_drift_field": predicted_drift_field,
        "uncertainty_field": uncertainty_field,
        "usable_rows": len(usable),
        "normalised_rmse": nrmse,
        "normalised_rmse_ci95_cluster_bootstrap": nrmse_ci,
        "pearson": _pearson(truth, predicted),
        "spearman": spearman,
        "spearman_ci95_cluster_bootstrap": spearman_ci,
        "ranking": ranking,
        "absolute_drift_calibration": absolute_calibration,
        "uncertainty_calibration": uncertainty,
        "component_predictions": component_reports,
        "by_load": by_load,
        "missing_values": missing,
        "status": status,
        "supported": status == "WORLD_MODEL_DRIFT_ESTIMATION_SUPPORTED",
    }


def evaluate_closed_loop_attachment(closed_loop_report: Optional[Mapping]) -> dict:
    """Layer 5 adapter; statistical work remains in the trajectory validator."""
    if not isinstance(closed_loop_report, Mapping):
        return {
            "layer": 5,
            "name": "normal_arrival_closed_loop_stability",
            "status": "NOT_EVALUATED_MISSING_CLOSED_LOOP_REPORT",
            "supported": False,
        }
    semantics = closed_loop_report.get("semantics") or {}
    role = semantics.get("five_layer_role") if isinstance(semantics, Mapping) else None
    verdict = str(closed_loop_report.get("verdict", "UNKNOWN"))
    compatible = role == "layer5_normal_arrival_closed_loop_stability"
    source_passed = bool(closed_loop_report.get("passed", False))
    internally_consistent = (
        source_passed == (verdict == "PASS_CLOSED_LOOP_STABILITY")
    )
    supported = bool(
        compatible
        and internally_consistent
        and verdict == "PASS_CLOSED_LOOP_STABILITY"
    )
    if not compatible:
        status = "INCOMPATIBLE_CLOSED_LOOP_REPORT"
    elif not internally_consistent:
        status = "INCONSISTENT_CLOSED_LOOP_REPORT"
    else:
        status = verdict
    return {
        "layer": 5,
        "name": "normal_arrival_closed_loop_stability",
        "source_schema_version": closed_loop_report.get("schema_version"),
        "source_role": role,
        "compatible": compatible,
        "internally_consistent": internally_consistent,
        "source_verdict": verdict,
        "source_passed": source_passed,
        "status": status,
        "supported": supported,
        "parameters": closed_loop_report.get("parameters", {}),
        "policy_load_groups": closed_loop_report.get("groups", {}),
        "summary": {
            "failed_policy_groups": closed_loop_report.get("failed_policy_groups", []),
            "incomplete_groups": closed_loop_report.get("incomplete_groups", []),
            "undercovered_components": closed_loop_report.get("undercovered_components", []),
            "recommendations": closed_loop_report.get("recommendations", []),
        },
    }


def _role_decision(layers: Mapping[str, Mapping]) -> dict:
    layer1 = layers["layer1_state_potential"]
    layer2 = layers["layer2_action_controllable_drift"]
    layer3 = layers["layer3_incremental_information"]
    layer4 = layers["layer4_world_model_drift_estimation"]
    layer5 = layers["layer5_closed_loop_stability"]
    if not layer1.get("supported", False):
        role = "REDESIGN_STATE_FUNCTIONAL_OR_LEDGER"
        next_step = "Repair state semantics/formula before using L for monitoring or control."
    elif layer2.get("status") == "ACTION_INSENSITIVE_STATE_MONITOR_ONLY":
        role = "STATE_MONITOR_ONLY_ACTION_INSENSITIVE"
        next_step = "Keep L for monitoring; redesign action-sensitive components only if control use is required."
    elif not layer2.get("supported", False):
        role = "STATE_MONITOR_ACTION_CONTROLLABILITY_INCONCLUSIVE"
        next_step = (
            "Keep L out of the action score until independent fixed-context "
            "candidate groups establish action-dependent drift without a raw gap gate."
        )
    elif layer3.get("status") in {
        "NOT_EVALUATED_MISSING_TRUE_WM_SCORE_FIELD",
        "NOT_EVALUATED_WM_SCORE_ABSENT_FROM_DATA",
        "PROXY_ONLY_NOT_A_WORLD_MODEL_INCREMENTAL_TEST",
    }:
        role = "STATE_MONITOR_AND_ORACLE_CANDIDATE_NEEDS_TRUE_WM_TEST"
        next_step = "Join the real WM score to each candidate and run held-out incremental tests."
    elif layer3.get("status") == "NO_STABLE_INCREMENTAL_INFORMATION":
        role = "STATE_MONITOR_ONLY_WM_REDUNDANT"
        next_step = "Do not add L to the online action score; retain it for telemetry, OOD/stress detection and audit."
    elif not layer3.get("supported", False):
        role = "INCREMENTAL_VALUE_INCONCLUSIVE"
        next_step = "Increase independent seeds or improve independent outcomes without tuning a raw drift gap."
    elif layer4.get("status") in {
        "NOT_EVALUATED_MISSING_PREDICTED_DRIFT_FIELD",
        "NOT_EVALUATED_PREDICTION_ABSENT_OR_INSUFFICIENT",
    }:
        role = "ANALYTIC_AUXILIARY_PROMISING_ESTIMATOR_UNTESTED"
        next_step = "Evaluate the learned drift endpoint, or compute analytic drift online if feasible."
    elif not layer4.get("supported", False):
        role = "ANALYTIC_AUXILIARY_VALID_BUT_ESTIMATOR_NOT_READY"
        next_step = "Improve endpoint prediction/calibration; do not redesign L solely for model error."
    elif layer5.get("status") == "NOT_EVALUATED_MISSING_CLOSED_LOOP_REPORT":
        role = "READY_FOR_FROZEN_CLOSED_LOOP_VALIDATION"
        next_step = "Freeze the integration rule and run independent normal-arrival closed-loop seeds."
    elif layer5.get("supported", False):
        role = "ONLINE_ACTION_AUXILIARY_SUPPORTED"
        next_step = "Keep monitoring distribution shift and repeat on untouched seeds/load regimes."
    else:
        role = "CLOSED_LOOP_POLICY_OR_INTEGRATION_FAILURE"
        next_step = "Diagnose policy integration, uncertainty and load feasibility before changing L itself."
    return {
        "recommended_role": role,
        "next_step": next_step,
        "monitor_only_is_valid_outcome": True,
        "monitor_only_uses": [
            "state-pressure telemetry and incident diagnosis",
            "OOD/stress trigger or safety fallback activation",
            "dataset stratification and regression monitoring",
            "auxiliary representation target without direct action-score injection",
        ],
        "online_action_score_enabled_by_this_report": role == "ONLINE_ACTION_AUXILIARY_SUPPORTED",
    }


def evaluate_five_layer_validity(
    samples: Sequence[Mapping],
    *,
    wm_score_field: Optional[str] = None,
    wm_score_provenance_verified: bool = False,
    layer3_candidate_component: str = "total",
    layer3_min_independent_clusters: Optional[int] = None,
    predicted_drift_field: Optional[str] = None,
    learned_report: Optional[Mapping] = None,
    uncertainty_field: Optional[str] = None,
    predicted_component_fields: Optional[Mapping[str, str]] = None,
    outcome_fields: Sequence[str] = DEFAULT_OUTCOME_FIELDS,
    closed_loop_report: Optional[Mapping] = None,
    strict_isolated: bool = False,
    required_l0_schema: str = CURRENT_L0_COLLECTION_SCHEMA,
    invariant_tolerance: float = 1e-6,
    bootstrap_repeats: int = 1000,
    placebo_repeats: int = 200,
    random_seed: int = 2026,
) -> dict:
    loaded = _prepare_samples(samples)
    layers = {
        "layer1_state_potential": evaluate_state_potential(
            loaded,
            invariant_tolerance=invariant_tolerance,
            strict_isolated=strict_isolated,
            required_l0_schema=required_l0_schema,
        ),
        "layer2_action_controllable_drift": evaluate_action_controllability(
            loaded,
            outcome_fields=outcome_fields,
            bootstrap_repeats=bootstrap_repeats,
            random_seed=random_seed,
        ),
        "layer3_incremental_information": evaluate_incremental_information(
            loaded,
            wm_score_field=wm_score_field,
            wm_score_provenance_verified=wm_score_provenance_verified,
            candidate_component=layer3_candidate_component,
            minimum_independent_clusters=layer3_min_independent_clusters,
            outcome_fields=outcome_fields,
            bootstrap_repeats=bootstrap_repeats,
            placebo_repeats=placebo_repeats,
            random_seed=random_seed,
        ),
        "layer4_world_model_drift_estimation": evaluate_drift_prediction(
            loaded,
            predicted_drift_field=predicted_drift_field,
            learned_report=learned_report,
            uncertainty_field=uncertainty_field,
            predicted_component_fields=predicted_component_fields,
            bootstrap_repeats=bootstrap_repeats,
            random_seed=random_seed,
        ),
        "layer5_closed_loop_stability": evaluate_closed_loop_attachment(closed_loop_report),
    }
    decision = _role_decision(layers)
    return {
        "schema_version": SCHEMA_VERSION,
        "semantics": {
            "raw_delta_L_gap_is_validity_gate": False,
            "scale_invariant": True,
            "layers_are_sequential_claims": True,
            "productive_progress_primary_evidence": False,
            "productive_progress_reason": "P_prod shares the unfinished-work ledger with L_work.",
            "heuristic_cost_is_true_wm_score": False,
        },
        "parameters": {
            "wm_score_field": wm_score_field,
            "wm_score_provenance_verified": bool(
                wm_score_provenance_verified
            ),
            "layer3_candidate_component": layer3_candidate_component,
            "layer3_min_independent_clusters": (
                layer3_min_independent_clusters
            ),
            "predicted_drift_field": predicted_drift_field,
            "learned_report_attached": bool(
                isinstance(learned_report, Mapping)
            ),
            "uncertainty_field": uncertainty_field,
            "outcome_fields": list(outcome_fields),
            "strict_isolated": bool(strict_isolated),
            "required_l0_schema": str(required_l0_schema),
            "invariant_tolerance": float(invariant_tolerance),
            "bootstrap_repeats": int(bootstrap_repeats),
            "placebo_repeats": int(placebo_repeats),
            "random_seed": int(random_seed),
        },
        "samples": len(loaded),
        "candidate_groups": len(_groups(loaded)),
        "manifest": _manifest(loaded),
        **layers,
        "decision": decision,
        "verdict": decision["recommended_role"],
        "passed": decision["recommended_role"] == "ONLINE_ACTION_AUXILIARY_SUPPORTED",
    }


def _parse_component_fields(values: Sequence[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                "--predicted-component-field expects COMPONENT=FIELD"
            )
        component, field = value.split("=", 1)
        component = component.strip()
        field = field.strip()
        if component not in L0_COMPONENT_NAMES or not field:
            raise argparse.ArgumentTypeError(
                f"invalid component mapping {value!r}"
            )
        result[component] = field
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument(
        "--wm-score-field",
        default=None,
        help="Scalar field containing the real World Model action score.",
    )
    parser.add_argument(
        "--world-model-checkpoint",
        default=None,
        help=(
            "Optional RMFSWorldModel checkpoint. When supplied, candidate "
            "scores are computed directly and used by Layer 3."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--layer3-candidate-component",
        choices=("total", "work", "work_group_range"),
        default="total",
        help=(
            "Frozen analytic predictor used for the formal Layer-3 verdict. "
            "Use 'work' for the frozen raw-work v1 certificate or "
            "'work_group_range' for the untouched-seed v2 certificate. "
            "Station, arrival and disabled components remain diagnostic-only."
        ),
    )
    parser.add_argument(
        "--layer3-min-independent-clusters",
        type=int,
        default=None,
        help=(
            "Minimum independent seed clusters. Defaults to 10 for the "
            "formal work-derived candidates and 5 for legacy total drift."
        ),
    )
    parser.add_argument("--predicted-drift-field", default=None)
    parser.add_argument(
        "--learned-report",
        default=None,
        help=(
            "Optional held-out report from evaluate_work_drift_head.py. "
            "Layer 4 reads validation.continuous_group_range. Legacy "
            "train_lyapunov_head.py reports remain diagnostic-compatible."
        ),
    )
    parser.add_argument("--uncertainty-field", default=None)
    parser.add_argument(
        "--predicted-component-field",
        action="append",
        default=[],
        metavar="COMPONENT=FIELD",
    )
    parser.add_argument(
        "--outcome-fields",
        type=_parse_csv,
        default=DEFAULT_OUTCOME_FIELDS,
    )
    parser.add_argument(
        "--closed-loop-report",
        default=None,
        help="JSON report produced by validate_lyapunov_closed_loop.py.",
    )
    parser.add_argument("--strict-isolated", action="store_true")
    parser.add_argument(
        "--required-l0-schema",
        default=CURRENT_L0_COLLECTION_SCHEMA,
        help="Collection schema required by the isolated semantic audit.",
    )
    parser.add_argument("--invariant-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=1000,
        help=(
            "Cluster-bootstrap repeats. The official formal work-only CLI "
            f"freezes this to {FORMAL_WORK_ONLY_BOOTSTRAP_REPEATS}."
        ),
    )
    parser.add_argument(
        "--placebo-repeats",
        type=int,
        default=200,
        help=(
            "Within-group placebo repeats. The official formal work-only "
            f"CLI freezes this to {FORMAL_WORK_ONLY_PLACEBO_REPEATS}."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=2026,
        help=(
            "Evaluation RNG seed. The official formal work-only CLI freezes "
            f"this to {FORMAL_WORK_ONLY_RANDOM_SEED}."
        ),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-unless-online-supported", action="store_true")
    args = parser.parse_args()
    if args.invariant_tolerance <= 0.0:
        raise SystemExit("--invariant-tolerance must be positive")
    if args.bootstrap_repeats <= 0 or args.placebo_repeats <= 0:
        raise SystemExit("bootstrap/placebo repeats must be positive")
    if (
        args.layer3_min_independent_clusters is not None
        and args.layer3_min_independent_clusters < 2
    ):
        raise SystemExit("--layer3-min-independent-clusters must be >= 2")
    if (
        args.layer3_candidate_component in {"work", "work_group_range"}
        and args.layer3_min_independent_clusters is not None
        and args.layer3_min_independent_clusters
        < FORMAL_WORK_ONLY_MINIMUM_SEED_CLUSTERS
    ):
        raise SystemExit(
            "formal work-derived Layer 3 freezes "
            f"--layer3-min-independent-clusters >= "
            f"{FORMAL_WORK_ONLY_MINIMUM_SEED_CLUSTERS}; it cannot be lowered"
        )
    if args.layer3_candidate_component in {"work", "work_group_range"}:
        registered = (
            FORMAL_WORK_ONLY_BOOTSTRAP_REPEATS,
            FORMAL_WORK_ONLY_PLACEBO_REPEATS,
            FORMAL_WORK_ONLY_RANDOM_SEED,
        )
        actual = (
            args.bootstrap_repeats,
            args.placebo_repeats,
            args.random_seed,
        )
        if actual != registered:
            raise SystemExit(
                "formal work-derived Layer 3 freezes "
                "--bootstrap-repeats/--placebo-repeats/--random-seed to "
                f"{registered}; got {actual}"
            )
    paths = _expand_paths(args.data)
    samples = _load_samples(paths)
    wm_score_field = args.wm_score_field
    wm_scoring = None
    if args.world_model_checkpoint:
        samples, wm_scoring = _score_with_world_model(
            samples,
            args.world_model_checkpoint,
            device=args.device,
        )
        wm_score_field = "_validation_wm_score"
    closed_loop = None
    if args.closed_loop_report:
        closed_loop = json.loads(Path(args.closed_loop_report).read_text(encoding="utf-8"))
    learned_report = None
    if args.learned_report:
        learned_report = json.loads(
            Path(args.learned_report).read_text(encoding="utf-8")
        )
    report = evaluate_five_layer_validity(
        samples,
        wm_score_field=wm_score_field,
        wm_score_provenance_verified=bool(args.world_model_checkpoint),
        layer3_candidate_component=args.layer3_candidate_component,
        layer3_min_independent_clusters=(
            args.layer3_min_independent_clusters
        ),
        predicted_drift_field=args.predicted_drift_field,
        learned_report=learned_report,
        uncertainty_field=args.uncertainty_field,
        predicted_component_fields=_parse_component_fields(args.predicted_component_field),
        outcome_fields=args.outcome_fields,
        closed_loop_report=closed_loop,
        strict_isolated=args.strict_isolated,
        required_l0_schema=args.required_l0_schema,
        invariant_tolerance=args.invariant_tolerance,
        bootstrap_repeats=args.bootstrap_repeats,
        placebo_repeats=args.placebo_repeats,
        random_seed=args.random_seed,
    )
    report["world_model_scoring"] = wm_scoring
    report["source_files"] = paths
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"saved: {output}")
    else:
        print(text)
    print(f"five-layer Lyapunov role: {report['verdict']}")
    if args.fail_unless_online_supported and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
