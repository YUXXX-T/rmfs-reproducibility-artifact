"""Audit Phase-C NO_ASSIGN labels and their closed-loop semantics.

The audit distinguishes two questions that were previously conflated:

* Is the stored row a valid *single isolated H-step no-op* label?
* Does that row identify the value of repeatedly deferring online?

The first can pass while the second is structurally unsupported because the
snapshot state distribution was collected by an assign-only Stage-1 policy.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping

import torch

from WorldModel.core.analytic_work_relief import (
    compute_snapshot_nominal_work_relief,
    work_potential,
)


REPORT_SCHEMA_VERSION = "phase_c_no_assign_label_audit_v1"
VERDICT = "SINGLE_STEP_LABEL_VALID_CLOSED_LOOP_NO_ASSIGN_UNDERIDENTIFIED"
NO_ASSIGN = "no_assign"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _plain_tensor(value) -> torch.Tensor:
    return torch.as_tensor(value).detach().cpu().contiguous()


def _quantile(values: Iterable[float], probability: float) -> float | None:
    rows = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not rows:
        return None
    position = min(max(float(probability), 0.0), 1.0) * (len(rows) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return float(rows[left])
    weight = position - left
    return float(rows[left] * (1.0 - weight) + rows[right] * weight)


def _distribution(values: Iterable[float]) -> dict:
    rows = [float(value) for value in values if math.isfinite(float(value))]
    if not rows:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    mean = sum(rows) / len(rows)
    variance = sum((value - mean) ** 2 for value in rows) / len(rows)
    return {
        "n": len(rows),
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
        "p05": _quantile(rows, 0.05),
        "p50": _quantile(rows, 0.50),
        "p95": _quantile(rows, 0.95),
        "min": float(min(rows)),
        "max": float(max(rows)),
    }


def _mapping_value(mapping: Mapping, key: str, default=0.0) -> float:
    if key in mapping:
        return float(mapping[key])
    return float(default)


def _work_component(snapshot: Mapping) -> float:
    return _mapping_value(snapshot.get("components") or {}, "work")


def _actual_work_drift(sample: Mapping) -> float:
    start = sample.get("lyapunov_l0_start") or {}
    endpoint = sample.get("lyapunov_l0_end") or {}
    return float(_work_component(endpoint) - _work_component(start))


def _analytic_work_drift(sample: Mapping, horizon: int) -> float:
    post = sample.get("lyapunov_l0_post_action")
    if not isinstance(post, Mapping):
        raise ValueError("sample is missing lyapunov_l0_post_action")
    forecast = compute_snapshot_nominal_work_relief(
        post,
        horizon=int(horizon),
    )
    config = sample.get("lyapunov_l0_config") or {}
    work_weight = float(config.get("work_weight", 1.0))
    capacity = float(post.get("work_capacity", 0.0))
    if capacity <= 0.0:
        raise ValueError("post-action work capacity must be positive")
    current = work_potential(
        forecast.post_action_station_work,
        work_capacity=capacity,
        work_weight=work_weight,
    )
    endpoint = work_potential(
        forecast.nominal_endpoint_station_work,
        work_capacity=capacity,
        work_weight=work_weight,
    )
    return float(endpoint - current)


def _completed_orders(sample: Mapping) -> float:
    labels = _plain_tensor(sample["future_system_labels"])
    if labels.ndim != 2 or labels.shape[1] < 6:
        raise ValueError("future_system_labels must have completed-orders channel")
    return float(labels[:, 5].sum().item())


def _group_rows(samples: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for sample in samples:
        key = str(sample["candidate_group_id"])
        groups.setdefault(key, []).append(sample)
    return groups


def _tick_bin(tick: int, width: int = 250) -> str:
    start = max(0, int(tick) // int(width) * int(width))
    return f"{start}-{start + int(width)}"


def _normalise_load(value) -> str:
    text = str(value or "unknown").lower()
    return text[:-5] if text.endswith("_load") else text


def _analyse_group(
    group_id: str,
    rows: list[dict],
    *,
    horizon: int,
    tie_epsilon: float,
) -> dict:
    no_assign_rows = [row for row in rows if row.get("action_type") == NO_ASSIGN]
    robot_rows = [row for row in rows if row.get("action_type") != NO_ASSIGN]
    if len(no_assign_rows) != 1 or not robot_rows:
        raise ValueError(
            f"group {group_id} must contain one NO_ASSIGN and robot candidates"
        )
    no_assign = no_assign_rows[0]
    no_assign_cost = _scalar(no_assign["realized_cost"])
    best_robot = min(robot_rows, key=lambda row: _scalar(row["realized_cost"]))
    best_robot_cost = _scalar(best_robot["realized_cost"])
    costs = [_scalar(row["realized_cost"]) for row in rows]
    cost_range = max(costs) - min(costs)
    cost_margin = best_robot_cost - no_assign_cost
    if cost_margin > tie_epsilon:
        top1_class = "strict_better"
    elif cost_margin < -tie_epsilon:
        top1_class = "strict_worse"
    else:
        top1_class = "practical_tie"

    pair_better = 0
    pair_worse = 0
    pair_tie = 0
    for robot in robot_rows:
        difference = _scalar(robot["realized_cost"]) - no_assign_cost
        if difference > tie_epsilon:
            pair_better += 1
        elif difference < -tie_epsilon:
            pair_worse += 1
        else:
            pair_tie += 1

    no_assign_actual = _actual_work_drift(no_assign)
    robot_actual = _actual_work_drift(best_robot)
    no_assign_analytic = _analytic_work_drift(no_assign, horizon)
    robot_analytic = _analytic_work_drift(best_robot, horizon)
    no_assign_completed = _completed_orders(no_assign)
    robot_completed = _completed_orders(best_robot)
    meta = no_assign
    return {
        "group_id": group_id,
        "split": "unassigned",
        "load": _normalise_load(meta.get("source_load_level")),
        "seed": str(meta.get("source_seed", meta.get("simulation_seed", "unknown"))),
        "tick": int(meta.get("decision_tick", -1)),
        "candidate_count": len(rows),
        "robot_candidate_count": len(robot_rows),
        "no_assign_cost": no_assign_cost,
        "best_robot_cost": best_robot_cost,
        "cost_margin_best_robot_minus_no_assign": float(cost_margin),
        "normalised_cost_margin": float(
            cost_margin / max(cost_range, 1e-12)
        ),
        "top1_class": top1_class,
        "pair_better": pair_better,
        "pair_worse": pair_worse,
        "pair_tie": pair_tie,
        "no_assign_actual_work_drift": no_assign_actual,
        "best_robot_actual_work_drift": robot_actual,
        "actual_assign_advantage": float(no_assign_actual - robot_actual),
        "no_assign_analytic_work_drift": no_assign_analytic,
        "best_robot_analytic_work_drift": robot_analytic,
        "analytic_assign_advantage": float(no_assign_analytic - robot_analytic),
        "no_assign_completed_orders": no_assign_completed,
        "best_robot_completed_orders": robot_completed,
        "completed_orders_assign_advantage": float(
            robot_completed - no_assign_completed
        ),
        "cost_work_conflict": bool(
            top1_class == "strict_better"
            and no_assign_actual > robot_actual + 1e-12
        ),
        "cost_completion_conflict": bool(
            top1_class == "strict_better"
            and robot_completed > no_assign_completed + 1e-12
        ),
        "analytic_actual_direction_agreement": bool(
            (no_assign_analytic - robot_analytic)
            * (no_assign_actual - robot_actual)
            >= 0.0
        ),
    }


def _summary(rows: list[dict]) -> dict:
    groups = len(rows)
    pair_better = sum(int(row["pair_better"]) for row in rows)
    pair_worse = sum(int(row["pair_worse"]) for row in rows)
    pair_tie = sum(int(row["pair_tie"]) for row in rows)
    non_tie = pair_better + pair_worse
    return {
        "groups": groups,
        "no_assign_strict_top1": sum(
            row["top1_class"] == "strict_better" for row in rows
        ),
        "no_assign_strict_top1_rate": float(
            sum(row["top1_class"] == "strict_better" for row in rows)
            / max(groups, 1)
        ),
        "no_assign_practical_tie_rate": float(
            sum(row["top1_class"] == "practical_tie" for row in rows)
            / max(groups, 1)
        ),
        "no_assign_strict_worse_rate": float(
            sum(row["top1_class"] == "strict_worse" for row in rows)
            / max(groups, 1)
        ),
        "no_assign_pairwise": {
            "non_tie_pairs": non_tie,
            "better_pairs": pair_better,
            "worse_pairs": pair_worse,
            "tie_pairs": pair_tie,
            "better_rate_non_tie": float(pair_better / max(non_tie, 1)),
        },
        "normalised_cost_margin": _distribution(
            row["normalised_cost_margin"] for row in rows
        ),
        "actual_assign_advantage": _distribution(
            row["actual_assign_advantage"] for row in rows
        ),
        "analytic_assign_advantage": _distribution(
            row["analytic_assign_advantage"] for row in rows
        ),
        "completed_orders_assign_advantage": _distribution(
            row["completed_orders_assign_advantage"] for row in rows
        ),
        "cost_work_conflict_rate": float(
            sum(bool(row["cost_work_conflict"]) for row in rows)
            / max(groups, 1)
        ),
        "cost_completion_conflict_rate": float(
            sum(bool(row["cost_completion_conflict"]) for row in rows)
            / max(groups, 1)
        ),
        "analytic_actual_direction_agreement_rate": float(
            sum(bool(row["analytic_actual_direction_agreement"]) for row in rows)
            / max(groups, 1)
        ),
    }


def _assign_splits(group_rows: list[dict], splits: Mapping | None) -> None:
    if not splits:
        return
    lookup = {}
    for split in ("train", "val", "test"):
        for group_id in splits.get(split, ()):  # fused IDs are strings
            lookup[str(group_id)] = split
    for row in group_rows:
        row["split"] = lookup.get(row["group_id"], "unassigned")


def _tensor_digest(digest, value) -> None:
    tensor = _plain_tensor(value)
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())


def _state_signature(sample: Mapping) -> str:
    digest = hashlib.sha256()
    for key in (
        "node_history",
        "edge_index",
        "edge_features",
        "demand_context",
        "station_node_ids",
    ):
        _tensor_digest(digest, sample[key])
    return digest.hexdigest()


def _future_label_signature(sample: Mapping) -> str:
    digest = hashlib.sha256()
    for key in (
        "future_node_labels",
        "future_system_labels",
        "future_station_labels",
        "future_mask",
    ):
        _tensor_digest(digest, sample[key])
    digest.update(repr(_scalar(sample["realized_cost"])).encode("utf-8"))
    return digest.hexdigest()


def _zero_action(sample: Mapping) -> bool:
    for key in ("action_node", "action_global", "action_edge"):
        value = sample.get(key)
        if value is not None and bool(torch.count_nonzero(_plain_tensor(value)).item()):
            return False
    return sample.get("action_encoding") == "zero_action_tensors_v1"


def _duplicate_no_assign_audit(samples: list[dict]) -> dict:
    rows = [sample for sample in samples if sample.get("action_type") == NO_ASSIGN]
    by_state: dict[str, list[dict]] = defaultdict(list)
    for sample in rows:
        by_state[_state_signature(sample)].append(sample)
    duplicated = [members for members in by_state.values() if len(members) > 1]
    inconsistent = 0
    maximum_cost_spread = 0.0
    for members in duplicated:
        signatures = {_future_label_signature(sample) for sample in members}
        if len(signatures) != 1:
            inconsistent += 1
        costs = [_scalar(sample["realized_cost"]) for sample in members]
        maximum_cost_spread = max(maximum_cost_spread, max(costs) - min(costs))
    duplicate_rows = sum(len(members) - 1 for members in duplicated)
    return {
        "no_assign_rows": len(rows),
        "zero_action_encoding_verified": bool(rows) and all(
            _zero_action(sample) for sample in rows
        ),
        "unique_global_state_signatures": len(by_state),
        "duplicated_global_state_signatures": len(duplicated),
        "duplicate_rows_beyond_one_per_state": duplicate_rows,
        "duplicate_row_fraction": float(duplicate_rows / max(len(rows), 1)),
        "duplicated_state_label_inconsistencies": inconsistent,
        "maximum_duplicate_cost_spread": float(maximum_cost_spread),
        "interpretation": (
            "NO_ASSIGN is a global zero action. Multiple fixed contexts from the "
            "same state therefore duplicate one global no-op label and its "
            "supervision weight; identical labels are required."
        ),
    }


def _collector_assign_only_audit(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    explicit_false = 0
    explicit_true = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "include_no_assign_candidate":
                continue
            if isinstance(keyword.value, ast.Constant):
                if keyword.value.value is False:
                    explicit_false += 1
                elif keyword.value.value is True:
                    explicit_true += 1
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "explicit_include_no_assign_false_calls": explicit_false,
        "explicit_include_no_assign_true_calls": explicit_true,
        "assign_only_source_policy_verified": explicit_false > 0 and explicit_true == 0,
    }


def _pair_exposure(
    samples: list[dict],
    *,
    tie_epsilon: float,
    max_pairs: int,
    seed: int,
    balance_field: str | None,
) -> dict:
    groups = _group_rows(samples)
    pairs = []
    for members in groups.values():
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                left_cost = _scalar(members[left]["realized_cost"])
                right_cost = _scalar(members[right]["realized_cost"])
                if abs(left_cost - right_cost) <= tie_epsilon:
                    continue
                better = members[left] if left_cost < right_cost else members[right]
                pairs.append({
                    "contains_no_assign": (
                        (members[left].get("action_type") == NO_ASSIGN)
                        != (members[right].get("action_type") == NO_ASSIGN)
                    ),
                    "balance_value": str(
                        better.get(balance_field, "unknown")
                        if balance_field
                        else "unbalanced"
                    ),
                })
    rng = random.Random(int(seed))
    rng.shuffle(pairs)
    sampled = pairs[: int(max_pairs)] if max_pairs > 0 else pairs
    balanced = list(sampled)
    if balance_field and sampled:
        counts = defaultdict(int)
        for pair in sampled:
            counts[pair["balance_value"]] += 1
        weights = [1.0 / counts[pair["balance_value"]] for pair in sampled]
        balance_rng = random.Random(int(seed))
        indices = balance_rng.choices(
            range(len(sampled)),
            weights=weights,
            k=len(sampled),
        )
        balanced = [sampled[index] for index in indices]
    no_assign_samples = sum(
        sample.get("action_type") == NO_ASSIGN for sample in samples
    )
    sample_fraction = no_assign_samples / max(len(samples), 1)
    all_no_assign = sum(pair["contains_no_assign"] for pair in pairs)
    sampled_no_assign = sum(
        pair["contains_no_assign"] for pair in sampled
    )
    balanced_no_assign = sum(
        pair["contains_no_assign"] for pair in balanced
    )
    all_pair_fraction = all_no_assign / max(len(pairs), 1)
    sampled_pair_fraction = sampled_no_assign / max(len(sampled), 1)
    balanced_pair_fraction = balanced_no_assign / max(len(balanced), 1)
    return {
        "samples": len(samples),
        "no_assign_samples": no_assign_samples,
        "no_assign_sample_fraction": float(sample_fraction),
        "all_practical_pairs": len(pairs),
        "all_no_assign_pairs": int(all_no_assign),
        "all_no_assign_pair_fraction": float(all_pair_fraction),
        "sampled_pairs": len(sampled),
        "sampled_no_assign_pairs": int(sampled_no_assign),
        "sampled_no_assign_pair_fraction": float(sampled_pair_fraction),
        "balanced_pairs": len(balanced),
        "balanced_no_assign_pairs": int(balanced_no_assign),
        "balanced_no_assign_pair_fraction": float(balanced_pair_fraction),
        "pairwise_exposure_multiplier_all": float(
            all_pair_fraction / max(sample_fraction, 1e-12)
        ),
        "pairwise_exposure_multiplier_sampled": float(
            sampled_pair_fraction / max(sample_fraction, 1e-12)
        ),
        "pairwise_exposure_multiplier_after_balance": float(
            balanced_pair_fraction / max(sample_fraction, 1e-12)
        ),
        "pair_sampling_seed": int(seed),
        "pair_sampling_cap": int(max_pairs),
        "pair_balance_by": balance_field,
    }


def _balanced_sample_exposure(
    samples: list[dict],
    *,
    balance_field: str = "source_run_id",
) -> dict:
    by_group = defaultdict(list)
    for sample in samples:
        by_group[str(sample.get(balance_field, "unknown"))].append(sample)
    fractions = {
        key: float(
            sum(row.get("action_type") == NO_ASSIGN for row in rows)
            / max(len(rows), 1)
        )
        for key, rows in by_group.items()
    }
    raw_fraction = float(
        sum(sample.get("action_type") == NO_ASSIGN for sample in samples)
        / max(len(samples), 1)
    )
    expected = float(sum(fractions.values()) / max(len(fractions), 1))
    return {
        "balance_field": balance_field,
        "source_groups": len(by_group),
        "raw_no_assign_fraction": raw_fraction,
        "expected_no_assign_fraction_after_weighted_sampling": expected,
        "exposure_multiplier": float(expected / max(raw_fraction, 1e-12)),
        "by_source_group": dict(sorted(fractions.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--tie-epsilon", type=float, default=0.01)
    parser.add_argument("--max-train-pairs", type=int, default=20000)
    parser.add_argument("--pair-seed", type=int, default=42)
    parser.add_argument(
        "--pair-balance-by",
        default="source_load_level",
    )
    parser.add_argument(
        "--collector-source",
        default="WorldModel/evaluation/collect_phase_c_round1.py",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    splits_path = Path(args.splits) if args.splits else None
    output_path = Path(args.output)
    collector_path = Path(args.collector_source)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if splits_path is not None and not splits_path.is_file():
        raise FileNotFoundError(splits_path)
    if not collector_path.is_file():
        raise FileNotFoundError(collector_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if int(args.horizon) <= 0 or float(args.tie_epsilon) < 0.0:
        raise ValueError("horizon must be positive and tie epsilon non-negative")

    samples = torch.load(dataset_path, weights_only=False)
    if not isinstance(samples, list) or not samples:
        raise ValueError("dataset must be a non-empty list of samples")
    splits = None
    if splits_path is not None:
        with splits_path.open("r", encoding="utf-8") as handle:
            splits = json.load(handle)

    groups = _group_rows(samples)
    group_rows = [
        _analyse_group(
            group_id,
            rows,
            horizon=int(args.horizon),
            tie_epsilon=float(args.tie_epsilon),
        )
        for group_id, rows in groups.items()
    ]
    _assign_splits(group_rows, splits)

    by_split = defaultdict(list)
    by_load = defaultdict(list)
    by_seed = defaultdict(list)
    by_tick = defaultdict(list)
    for row in group_rows:
        by_split[row["split"]].append(row)
        by_load[row["load"]].append(row)
        by_seed[row["seed"]].append(row)
        by_tick[_tick_bin(row["tick"])].append(row)

    no_assign_rows = [
        sample for sample in samples if sample.get("action_type") == NO_ASSIGN
    ]
    single_step_checks = {
        "one_no_assign_per_group": all(
            sum(row.get("action_type") == NO_ASSIGN for row in members) == 1
            for members in groups.values()
        ),
        "isolated_rollout_only": all(
            sample.get("rollout_continuation_mode") == "isolated"
            for sample in no_assign_rows
        ),
        "no_future_orders": all(
            int(sample.get("rollout_generated_orders", 0)) == 0
            for sample in no_assign_rows
        ),
        "no_continuation_assignments": all(
            int(sample.get("rollout_assigned_tasks", 0)) == 0
            for sample in no_assign_rows
        ),
        "immediate_context_unchanged": all(
            bool((sample.get("no_assign_audit") or {}).get(
                "immediate_context_unchanged", False
            ))
            for sample in no_assign_rows
        ),
        "full_horizon": all(
            int(_plain_tensor(sample["future_mask"]).sum().item())
            == int(args.horizon)
            for sample in no_assign_rows
        ),
    }
    collector = _collector_assign_only_audit(collector_path)
    duplicate_audit = _duplicate_no_assign_audit(samples)
    single_step_checks["zero_action_encoding"] = duplicate_audit[
        "zero_action_encoding_verified"
    ]
    single_step_checks["duplicate_state_labels_consistent"] = (
        duplicate_audit["duplicated_state_label_inconsistencies"] == 0
    )

    train_samples = samples
    if splits:
        train_ids = {str(value) for value in splits.get("train", ())}
        train_samples = [
            sample
            for sample in samples
            if str(sample["candidate_group_id"]) in train_ids
        ]
    exposure = _pair_exposure(
        train_samples,
        tie_epsilon=float(args.tie_epsilon),
        max_pairs=int(args.max_train_pairs),
        seed=int(args.pair_seed),
        balance_field=(str(args.pair_balance_by) if args.pair_balance_by else None),
    )
    sample_exposure = _balanced_sample_exposure(train_samples)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": "POSTCERT_TRAINING_LABEL_SEMANTICS_AUDIT",
        "verdict": VERDICT,
        "inputs": {
            "dataset": {
                "path": dataset_path.as_posix(),
                "sha256": sha256_file(dataset_path),
            },
            "splits": (
                {
                    "path": splits_path.as_posix(),
                    "sha256": sha256_file(splits_path),
                }
                if splits_path is not None
                else None
            ),
            "collector": collector,
        },
        "contract": {
            "stage1_training_contains_native_no_assign": False,
            "phase_c_counterfactual_contains_native_no_assign": True,
            "phase_c_no_assign_rows": len(no_assign_rows),
            "phase_c_candidate_groups": len(groups),
            "label_horizon": int(args.horizon),
            "label_semantics": (
                "one global zero action followed by isolated physical evolution; "
                "no future order generation and no continuation scheduler"
            ),
            "online_semantics_not_covered": [
                "repeated NO_ASSIGN decisions",
                "defer opportunity cost beyond H",
                "states induced by a NO_ASSIGN streak",
                "long-term task starvation",
                "interaction with later per-context assignments",
            ],
        },
        "single_step_label_checks": {
            "checks": single_step_checks,
            "passed": all(single_step_checks.values()),
        },
        "closed_loop_coverage": {
            "source_assigner_supports_no_assign": False,
            "assign_only_source_policy_verified": collector[
                "assign_only_source_policy_verified"
            ],
            "observed_no_assign_streak_state_coverage": 0,
            "sequential_no_assign_return_or_continuation_labels": False,
            "passed": False,
            "reason": (
                "counterfactual NO_ASSIGN is injected only after an assign-only "
                "Stage-1 snapshot has been captured"
            ),
        },
        "duplicate_global_noop": duplicate_audit,
        "training_system_loss_exposure": sample_exposure,
        "training_pairwise_exposure": exposure,
        "overall": _summary(group_rows),
        "by_split": {
            key: _summary(rows) for key, rows in sorted(by_split.items())
        },
        "by_load": {
            key: _summary(rows) for key, rows in sorted(by_load.items())
        },
        "by_seed": {
            key: _summary(rows) for key, rows in sorted(by_seed.items())
        },
        "by_tick_bin": {
            key: _summary(rows) for key, rows in sorted(by_tick.items())
        },
        "interpretation": {
            "single_step_label": (
                "valid if the single-step checks pass; it answers whether waiting "
                "for H ticks is locally cheaper in the isolated frozen state"
            ),
            "closed_loop_value": (
                "not identified by this dataset; a policy may repeatedly select "
                "the locally cheap action and enter an unseen absorbing regime"
            ),
            "do_not_infer": (
                "a high NO_ASSIGN pairwise win rate does not certify that repeated "
                "online deferral is safe"
            ),
        },
    }
    if not report["single_step_label_checks"]["passed"]:
        report["verdict"] = "NO_ASSIGN_SINGLE_STEP_LABEL_INTEGRITY_FAILURE"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(f"saved: {output_path}")
    print("verdict =", report["verdict"])
    print("Phase-C NO_ASSIGN rows =", len(no_assign_rows))
    print(
        "train NO_ASSIGN sample fraction =",
        exposure["no_assign_sample_fraction"],
    )
    print(
        "train balanced pair fraction =",
        exposure["balanced_no_assign_pair_fraction"],
        "multiplier =",
        exposure["pairwise_exposure_multiplier_after_balance"],
    )
    print(
        "NO_ASSIGN strict top1 rate =",
        report["overall"]["no_assign_strict_top1_rate"],
    )
    print(
        "cost/work conflict rate =",
        report["overall"]["cost_work_conflict_rate"],
    )
    print("sequential NO_ASSIGN state coverage = 0")


if __name__ == "__main__":
    main()
