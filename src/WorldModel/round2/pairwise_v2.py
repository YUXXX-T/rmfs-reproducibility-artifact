"""Action-aware, group-balanced ranking pairs for Phase-C Round-2.

DEFER_CONTEXT rows train short-horizon dynamics but never participate in the
H=10 pairwise objective.  Only assign-vs-assign pairs are constructed.  Each
retained candidate group has total pair weight exactly one, independent of its
robot candidate count or number of non-tie pairs.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Mapping, Optional, Sequence

import torch

from WorldModel.core.costs import CONGESTION_LAMBDAS, compute_realized_cost
from WorldModel.data.dataset import stable_candidate_group_key
from WorldModel.round2.action_schema_v2 import (
    ASSIGN_ROBOT_ACTION_TYPE_V2,
    DEFER_CONTEXT_ACTION_SCHEMA_VERSION,
    DEFER_CONTEXT_ACTION_TYPE,
    infer_defer_context_action_schema_v2,
)


ROUND2_PAIRWISE_SCHEMA_VERSION = "wm_round2_assign_pairwise_v1"
ROUND2_RANKING_COST_SCHEMA_VERSION = "wm_round2_h10_congestion_only_v1"
ROUND2_RANKING_HORIZON = 10


def _action_type(sample: Mapping) -> str:
    return str(sample.get("action_type", "")).lower()


def _member_sort_key(sample: Mapping) -> tuple:
    info = sample.get("candidate_info") or {}
    robot_id = info.get("robot_id", sample.get("robot_id"))
    return (
        str(sample.get("candidate_key", "")),
        int(robot_id) if robot_id is not None else 10**12,
        str(sample.get("action_type", "")),
    )


def _congestion_cost(sample: Mapping) -> float:
    labels = sample.get("future_system_labels")
    if labels is None:
        raise ValueError("Round-2 ranking sample lacks future_system_labels")
    labels = torch.as_tensor(labels, dtype=torch.float32)
    if tuple(labels.shape) != (ROUND2_RANKING_HORIZON, 7):
        raise ValueError(
            "Round-2 ranking requires complete H=10 seven-channel labels"
        )
    if not bool(torch.isfinite(labels).all()):
        raise ValueError("Round-2 ranking labels must be finite")
    mask = sample.get("future_mask")
    if mask is None:
        raise ValueError("Round-2 ranking sample lacks future_mask")
    mask = torch.as_tensor(mask, dtype=torch.float32)
    if tuple(mask.shape) != (ROUND2_RANKING_HORIZON,) or not bool(
        (mask > 0.5).all()
    ):
        raise ValueError("Round-2 ranking requires a complete H=10 mask")
    cost = compute_realized_cost(labels, lambdas=CONGESTION_LAMBDAS)
    if not math.isfinite(cost):
        raise ValueError("Round-2 congestion-only cost must be finite")
    return float(cost)


def _validate_group_contract(group_key, members: Sequence[Mapping]) -> None:
    if any(
        sample.get("action_schema_version")
        != DEFER_CONTEXT_ACTION_SCHEMA_VERSION
        for sample in members
    ):
        raise ValueError(f"group {group_key} mixes or lacks schema-v2 rows")
    action_types = [_action_type(sample) for sample in members]
    allowed = {ASSIGN_ROBOT_ACTION_TYPE_V2, DEFER_CONTEXT_ACTION_TYPE}
    unknown = sorted(set(action_types) - allowed)
    if unknown:
        raise ValueError(f"group {group_key} has unknown actions: {unknown}")
    if any(action_type == "no_assign" for action_type in action_types):
        raise ValueError(f"group {group_key} contains legacy NO_ASSIGN")
    if action_types.count(DEFER_CONTEXT_ACTION_TYPE) != 1:
        raise ValueError(
            f"group {group_key} must contain exactly one DEFER_CONTEXT"
        )
    if action_types.count(ASSIGN_ROBOT_ACTION_TYPE_V2) < 2:
        raise ValueError(f"group {group_key} needs at least two robot actions")
    contexts = [sample.get("action_context") for sample in members]
    if not contexts or any(context != contexts[0] for context in contexts):
        raise ValueError(f"group {group_key} has inconsistent action_context")
    candidate_keys = [str(sample.get("candidate_key", "")) for sample in members]
    if any(not key for key in candidate_keys) or len(set(candidate_keys)) != len(
        candidate_keys
    ):
        raise ValueError(f"group {group_key} has missing/duplicate candidates")


def _all_group_pairs(
    group_key,
    members: Sequence[Mapping],
    *,
    epsilon: float,
) -> list[dict]:
    _validate_group_contract(group_key, members)
    assignments = sorted(
        (
            sample
            for sample in members
            if _action_type(sample) == ASSIGN_ROBOT_ACTION_TYPE_V2
        ),
        key=_member_sort_key,
    )
    rows = []
    for left_index in range(len(assignments)):
        for right_index in range(left_index + 1, len(assignments)):
            left = assignments[left_index]
            right = assignments[right_index]
            left_cost = _congestion_cost(left)
            right_cost = _congestion_cost(right)
            if not math.isfinite(left_cost) or not math.isfinite(right_cost):
                raise ValueError("Round-2 pair cost must be finite")
            if abs(left_cost - right_cost) <= float(epsilon):
                continue
            better, worse = (
                (left, right) if left_cost < right_cost else (right, left)
            )
            rows.append({
                "sample_i": better,
                "sample_j": worse,
                "candidate_group_key": tuple(group_key),
                "pair_schema_version": ROUND2_PAIRWISE_SCHEMA_VERSION,
                "pair_action_types": [
                    ASSIGN_ROBOT_ACTION_TYPE_V2,
                    ASSIGN_ROBOT_ACTION_TYPE_V2,
                ],
                "ranking_cost_schema_version": (
                    ROUND2_RANKING_COST_SCHEMA_VERSION
                ),
                "ranking_cost_lambdas": list(CONGESTION_LAMBDAS),
                "ranking_horizon": ROUND2_RANKING_HORIZON,
                "ranking_epsilon": float(epsilon),
                "sample_i_ranking_cost": min(left_cost, right_cost),
                "sample_j_ranking_cost": max(left_cost, right_cost),
            })
    return rows


def _balanced_select(
    bundles: Sequence[tuple[tuple, list[dict]]],
    *,
    max_pairs: Optional[int],
    seed: int,
) -> tuple[list[tuple[tuple, list[dict]]], dict]:
    rng = random.Random(int(seed))
    prepared = []
    for group_key, pairs in bundles:
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        prepared.append((group_key, shuffled))
    rng.shuffle(prepared)

    total_available = sum(len(pairs) for _, pairs in prepared)
    if max_pairs is None or int(max_pairs) >= total_available:
        selected = [(key, list(pairs)) for key, pairs in prepared]
        return selected, {
            "max_pairs": max_pairs,
            "truncated": False,
            "available_pairs": total_available,
            "selected_pairs": total_available,
        }
    budget = max(int(max_pairs), 0)
    selected_by_group = {key: [] for key, _ in prepared}
    cursors = {key: 0 for key, _ in prepared}
    pair_map = {key: pairs for key, pairs in prepared}

    while budget > 0:
        progress = False
        for group_key, _ in prepared:
            cursor = cursors[group_key]
            group_pairs = pair_map[group_key]
            if cursor >= len(group_pairs):
                continue
            selected_by_group[group_key].append(group_pairs[cursor])
            cursors[group_key] = cursor + 1
            budget -= 1
            progress = True
            if budget == 0:
                break
        if not progress:
            break

    selected = [
        (group_key, selected_by_group[group_key])
        for group_key, _ in prepared
        if selected_by_group[group_key]
    ]
    selected_count = sum(len(pairs) for _, pairs in selected)
    return selected, {
        "max_pairs": int(max_pairs),
        "truncated": True,
        "available_pairs": total_available,
        "selected_pairs": selected_count,
    }


def build_round2_pairwise_data(
    samples: Sequence[Mapping],
    *,
    epsilon: float = 0.01,
    max_pairs: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """Build deterministic assign-only pairs with per-group total weight 1."""
    if not math.isfinite(float(epsilon)) or float(epsilon) < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    if max_pairs is not None and int(max_pairs) <= 0:
        raise ValueError("max_pairs must be positive when supplied")
    source_action_types = [_action_type(sample) for sample in samples]
    if "no_assign" in source_action_types:
        raise ValueError("Round-2 ranking rejects legacy NO_ASSIGN")
    unknown_action_types = sorted(
        set(source_action_types)
        - {ASSIGN_ROBOT_ACTION_TYPE_V2, DEFER_CONTEXT_ACTION_TYPE}
    )
    if unknown_action_types:
        raise ValueError(
            f"Round-2 ranking rejects unknown actions: {unknown_action_types}"
        )
    action_audit = infer_defer_context_action_schema_v2(samples)
    if not action_audit["passed"]:
        raise ValueError(
            "Round-2 pair construction requires a passed action-schema audit: "
            + ", ".join(action_audit["failures"][:10])
        )
    groups = defaultdict(list)
    for sample in samples:
        groups[stable_candidate_group_key(sample)].append(sample)

    bundles = []
    zero_pair_groups = 0
    for group_key in sorted(groups):
        pairs = _all_group_pairs(
            group_key,
            groups[group_key],
            epsilon=float(epsilon),
        )
        if not pairs:
            zero_pair_groups += 1
            continue
        bundles.append((group_key, pairs))

    selected, selection_audit = _balanced_select(
        bundles,
        max_pairs=max_pairs,
        seed=int(seed),
    )
    available_by_group = {
        group_key: len(pairs) for group_key, pairs in bundles
    }
    selected_pair_count = sum(len(pairs) for _, pairs in selected)
    selected_group_count = len(selected)
    if selected_pair_count <= 0 or selected_group_count <= 0:
        raise ValueError("Round-2 ranking produced no non-tie robot pairs")
    uniform_pair_scale = selected_pair_count / float(selected_group_count)
    result = []
    for group_key, pairs in selected:
        weight = 1.0 / float(len(pairs))
        original_count = available_by_group[group_key]
        for pair in pairs:
            result.append({
                **pair,
                "pair_weight": weight,
                "uniform_pair_sgd_weight": uniform_pair_scale * weight,
                "selected_pairs_in_group": len(pairs),
                "valid_pairs_before_balanced_limit": original_count,
            })

    rng = random.Random(int(seed) ^ 0x5A17)
    rng.shuffle(result)
    audit = audit_round2_pairwise_data(result)
    if not audit["passed"]:
        raise RuntimeError(
            "internal Round-2 pairwise contract failed: "
            + ", ".join(audit["failures"])
        )
    for pair in result:
        pair["pairwise_build_audit"] = {
            **selection_audit,
            "source_candidate_groups": len(groups),
            "valid_pair_groups": len(bundles),
            "zero_pair_groups": zero_pair_groups,
        }
    return result


def audit_round2_pairwise_data(pairs: Sequence[Mapping]) -> dict:
    rows = list(pairs or ())
    group_weights = defaultdict(float)
    group_counts = defaultdict(int)
    failures = []
    parsed_rows = []
    for index, pair in enumerate(rows):
        if pair.get("pair_schema_version") != ROUND2_PAIRWISE_SCHEMA_VERSION:
            failures.append(f"pair[{index}]:schema")
        left_type = _action_type(pair.get("sample_i") or {})
        right_type = _action_type(pair.get("sample_j") or {})
        if (left_type, right_type) != (
            ASSIGN_ROBOT_ACTION_TYPE_V2,
            ASSIGN_ROBOT_ACTION_TYPE_V2,
        ):
            failures.append(f"pair[{index}]:defer_or_invalid_action")
        key = tuple(pair.get("candidate_group_key") or ())
        if not key:
            failures.append(f"pair[{index}]:group_key")
        try:
            left_key = tuple(stable_candidate_group_key(pair["sample_i"]))
            right_key = tuple(stable_candidate_group_key(pair["sample_j"]))
        except (KeyError, TypeError, ValueError):
            left_key = right_key = ()
        if not key or left_key != key or right_key != key:
            failures.append(f"pair[{index}]:group_provenance")
        left_candidate = str(pair.get("sample_i", {}).get("candidate_key", ""))
        right_candidate = str(pair.get("sample_j", {}).get("candidate_key", ""))
        if not left_candidate or not right_candidate or left_candidate == right_candidate:
            failures.append(f"pair[{index}]:candidate_identity")
        if pair.get("sample_i", {}).get("action_context") != pair.get(
            "sample_j", {}
        ).get("action_context"):
            failures.append(f"pair[{index}]:context_mismatch")
        if (
            pair.get("ranking_cost_schema_version")
            != ROUND2_RANKING_COST_SCHEMA_VERSION
            or list(pair.get("ranking_cost_lambdas") or ())
            != list(CONGESTION_LAMBDAS)
            or int(pair.get("ranking_horizon", -1)) != ROUND2_RANKING_HORIZON
        ):
            failures.append(f"pair[{index}]:cost_schema")
        try:
            epsilon = float(pair.get("ranking_epsilon", float("nan")))
            left_cost = _congestion_cost(pair["sample_i"])
            right_cost = _congestion_cost(pair["sample_j"])
            declared_left = float(pair.get("sample_i_ranking_cost"))
            declared_right = float(pair.get("sample_j_ranking_cost"))
            if (
                not math.isfinite(epsilon)
                or epsilon < 0.0
                or not left_cost + epsilon < right_cost
                or not math.isclose(left_cost, declared_left, abs_tol=1e-6)
                or not math.isclose(right_cost, declared_right, abs_tol=1e-6)
            ):
                failures.append(f"pair[{index}]:cost_ordering")
        except (KeyError, TypeError, ValueError, OverflowError):
            failures.append(f"pair[{index}]:cost_ordering")
        try:
            weight = float(pair.get("pair_weight", -1.0))
            sgd_weight = float(pair.get("uniform_pair_sgd_weight", -1.0))
        except (TypeError, ValueError, OverflowError):
            weight = sgd_weight = -1.0
        if not math.isfinite(weight) or not (0.0 < weight <= 1.0):
            failures.append(f"pair[{index}]:weight")
        if not math.isfinite(sgd_weight) or sgd_weight <= 0.0:
            failures.append(f"pair[{index}]:sgd_weight")
        group_weights[key] += weight
        group_counts[key] += 1
        parsed_rows.append((index, key, weight, sgd_weight))
    for key, total in group_weights.items():
        if abs(total - 1.0) > 1e-9:
            failures.append(f"group[{key}]:weight_sum={total}")
    pair_count = len(rows)
    group_count = len(group_counts)
    if pair_count and group_count:
        scale = pair_count / float(group_count)
        for index, _, weight, sgd_weight in parsed_rows:
            if not math.isclose(
                sgd_weight, scale * weight, rel_tol=1e-9, abs_tol=1e-12
            ):
                failures.append(f"pair[{index}]:sgd_importance_scale")
    checks = {
        "nonempty": bool(rows),
        "no_defer_pairs": not any(
            "defer_or_invalid_action" in failure for failure in failures
        ),
        "per_group_weight_sum_one": not any(
            "weight_sum" in failure for failure in failures
        ),
        "positive_weights": not any(
            failure.endswith(":weight") for failure in failures
        ),
        "schema_complete": not any(
            failure.endswith(":schema") or failure.endswith(":group_key")
            for failure in failures
        ),
        "congestion_only_cost_contract": not any(
            ":cost_schema" in failure or ":cost_ordering" in failure
            for failure in failures
        ),
        "group_provenance": not any(
            ":group_provenance" in failure
            or ":candidate_identity" in failure
            or ":context_mismatch" in failure
            for failure in failures
        ),
        "uniform_pair_sgd_unbiased_scale": not any(
            ":sgd_weight" in failure or ":sgd_importance_scale" in failure
            for failure in failures
        ),
    }
    return {
        "schema_version": ROUND2_PAIRWISE_SCHEMA_VERSION,
        "pairs": len(rows),
        "groups": len(group_weights),
        "group_pair_counts": {
            repr(key): value for key, value in sorted(group_counts.items())
        },
        "group_weight_sums": {
            repr(key): value for key, value in sorted(group_weights.items())
        },
        "checks": checks,
        "failures": failures[:100],
        "passed": all(checks.values()),
    }


def group_balanced_pair_order(
    pairs: Sequence[Mapping],
    *,
    seed: int,
) -> list[Mapping]:
    """Interleave groups so any training prefix is group-balanced."""
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[tuple(pair["candidate_group_key"])].append(pair)
    rng = random.Random(int(seed))
    groups = sorted(grouped)
    rng.shuffle(groups)
    for rows in grouped.values():
        rng.shuffle(rows)
    result = []
    depth = 0
    while True:
        added = False
        for group_key in groups:
            rows = grouped[group_key]
            if depth < len(rows):
                result.append(rows[depth])
                added = True
        if not added:
            break
        depth += 1
    return result


__all__ = [
    "ROUND2_PAIRWISE_SCHEMA_VERSION",
    "ROUND2_RANKING_COST_SCHEMA_VERSION",
    "ROUND2_RANKING_HORIZON",
    "audit_round2_pairwise_data",
    "build_round2_pairwise_data",
    "group_balanced_pair_order",
]
