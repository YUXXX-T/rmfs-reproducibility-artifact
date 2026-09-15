"""Training-side consumers for Round-2 group-balanced ranking pairs."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Mapping, Sequence

import torch

from WorldModel.round2.pairwise_v2 import (
    ROUND2_PAIRWISE_SCHEMA_VERSION,
    audit_round2_pairwise_data,
)
from WorldModel.training.train import compute_ranking_loss


_RANKING_KEYS = frozenset({
    "node_history",
    "edge_index",
    "edge_features",
    "demand_context",
    "action_node",
    "action_global",
    "station_node_ids",
})


def pair_ranking_to_device_v2(
    pair: Mapping,
    device: torch.device,
) -> dict:
    """Move ranking tensors while preserving group and weight metadata."""
    if pair.get("pair_schema_version") != ROUND2_PAIRWISE_SCHEMA_VERSION:
        raise ValueError("ranking pair is not a Round-2 pair")

    def select(sample: Mapping) -> dict:
        result = {}
        for key, value in sample.items():
            if key not in _RANKING_KEYS:
                continue
            result[key] = (
                value.to(device) if isinstance(value, torch.Tensor) else value
            )
        return result

    weight = float(pair.get("pair_weight", -1.0))
    sgd_weight = float(pair.get("uniform_pair_sgd_weight", -1.0))
    if not (0.0 < weight <= 1.0):
        raise ValueError("Round-2 pair_weight must be in (0, 1]")
    if not (sgd_weight > 0.0):
        raise ValueError("Round-2 uniform_pair_sgd_weight must be positive")
    return {
        "sample_i": select(pair["sample_i"]),
        "sample_j": select(pair["sample_j"]),
        "pair_weight": weight,
        "uniform_pair_sgd_weight": sgd_weight,
        "candidate_group_key": tuple(pair["candidate_group_key"]),
        "pair_schema_version": ROUND2_PAIRWISE_SCHEMA_VERSION,
    }


def compute_weighted_ranking_loss_v2(
    model,
    pair: Mapping,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    """Unbiased one-pair SGD estimate of the group-balanced objective.

    Formal Round-2 training samples uniformly from the flattened pair set.
    The frozen importance factor is ``P / (G * n_g)``.  Its expectation is
    exactly the mean over groups of each group's mean pair loss.
    """
    if pair.get("pair_schema_version") != ROUND2_PAIRWISE_SCHEMA_VERSION:
        raise ValueError("ranking loss received a non-Round-2 pair")
    weight = float(pair.get("uniform_pair_sgd_weight", -1.0))
    if not (weight > 0.0):
        raise ValueError("invalid Round-2 uniform-pair SGD weight")
    raw_loss = compute_ranking_loss(model, pair, margin=float(margin))
    return raw_loss * weight


def compute_group_balanced_ranking_objective_v2(
    model,
    pairs: Sequence[Mapping],
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    """Return mean-over-groups of each group's mean pairwise loss.

    This is the exact Round-2 objective:

    ``(1/G) * sum_g (1/n_g) * sum_{p in g} loss_p``.
    """
    audit = audit_round2_pairwise_data(pairs)
    if not audit["passed"] or not pairs:
        raise ValueError("cannot train on an invalid/empty Round-2 pair set")
    weighted_losses = [
        compute_ranking_loss(model, pair, margin=float(margin))
        * float(pair["pair_weight"])
        for pair in pairs
    ]
    return torch.stack(weighted_losses).sum() / float(audit["groups"])


def uniform_pair_sgd_epoch_v2(
    pairs: Sequence[Mapping],
    *,
    seed: int,
) -> list[Mapping]:
    """Return one deterministic uniform-pair epoch for the formal runner.

    Every retained pair appears exactly once.  Combined with
    ``uniform_pair_sgd_weight``, the mean loss over the epoch equals the exact
    group-balanced objective while each shuffled step is an unbiased estimate.
    """
    audit = audit_round2_pairwise_data(pairs)
    if not audit["passed"]:
        raise ValueError("cannot sample an invalid Round-2 pair set")
    result = list(pairs)
    random.Random(int(seed)).shuffle(result)
    return result


def aggregate_group_balanced_epoch_v2(
    rows: Sequence[Mapping],
) -> dict:
    """Audit and aggregate detached per-pair losses using group weights."""
    groups = defaultdict(list)
    failures = []
    for index, row in enumerate(rows):
        group_key = tuple(row.get("candidate_group_key") or ())
        weight = float(row.get("pair_weight", -1.0))
        raw_loss = float(row.get("raw_ranking_loss", float("nan")))
        if not group_key or not (0.0 < weight <= 1.0):
            failures.append(f"row[{index}]:metadata")
            continue
        if not torch.isfinite(torch.tensor(raw_loss)):
            failures.append(f"row[{index}]:loss")
            continue
        groups[group_key].append((weight, raw_loss))

    per_group = {}
    for group_key, values in groups.items():
        total_weight = sum(weight for weight, _ in values)
        if abs(total_weight - 1.0) > 1e-9:
            failures.append(f"group[{group_key}]:weight_sum={total_weight}")
            continue
        per_group[repr(group_key)] = sum(
            weight * loss for weight, loss in values
        )
    mean = (
        sum(per_group.values()) / len(per_group)
        if per_group
        else None
    )
    return {
        "schema_version": "wm_round2_group_balanced_epoch_v1",
        "groups": len(per_group),
        "pairs": len(rows),
        "mean_group_ranking_loss": mean,
        "per_group_ranking_loss": per_group,
        "failures": failures,
        "passed": bool(per_group) and not failures,
    }


__all__ = [
    "aggregate_group_balanced_epoch_v2",
    "compute_group_balanced_ranking_objective_v2",
    "compute_weighted_ranking_loss_v2",
    "pair_ranking_to_device_v2",
    "uniform_pair_sgd_epoch_v2",
]
