from collections import defaultdict
from unittest.mock import patch

import pytest
import torch

from WorldModel.round2.action_schema_v2 import (
    ASSIGN_CONTEXT_ENCODING,
    DEFER_CONTEXT_ACTION_SCHEMA_VERSION,
    DEFER_CONTEXT_ENCODING,
)
from WorldModel.round2.pairwise_v2 import (
    audit_round2_pairwise_data,
    build_round2_pairwise_data,
    group_balanced_pair_order,
)
from WorldModel.round2.training_v2 import (
    aggregate_group_balanced_epoch_v2,
    compute_group_balanced_ranking_objective_v2,
    compute_weighted_ranking_loss_v2,
    pair_ranking_to_device_v2,
    uniform_pair_sgd_epoch_v2,
)


def _group(group_id, costs, *, seed=491):
    age_ratio = 20.0 / 60.0
    debt_ratio = 0.5 / 1.5
    context = {
        "order_id": group_id,
        "pod_id": group_id + 10,
        "station_id": 1,
        "context_age": 20.0,
        "free_flow_time": 40.0,
        "dispatch_debt": 0.5,
        "age_over_age_plus_tff": age_ratio,
        "debt_over_one_plus_debt": debt_ratio,
    }
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    def labels(cost):
        result = torch.zeros(10, 7)
        result[0, 0] = float(cost)
        return result

    rows = []
    for robot_id, cost in enumerate(costs):
        action_node = torch.zeros(2, 8)
        action_node[robot_id % 2, 0] = 1.0
        action_node[0, 1] = 1.0
        action_node[1, 2] = 1.0
        action_node[0, 3] = 1.0
        action_node[:, 4:] = 1.0
        rows.append({
            "run_id": f"seed{seed}",
            "simulation_seed": seed,
            "candidate_group_id": f"g{group_id}",
            "candidate_key": f"g{group_id}_r{robot_id}",
            "candidate_info": {"robot_id": robot_id},
            "action_type": "assign_robot",
            "action_schema_version": DEFER_CONTEXT_ACTION_SCHEMA_VERSION,
            "action_encoding": ASSIGN_CONTEXT_ENCODING,
            "action_context": dict(context),
            "realized_cost": -1000.0 - float(cost),
            "future_system_labels": labels(cost),
            "future_mask": torch.ones(10),
            "action_node": action_node,
            "action_global": torch.tensor([
                0.1 + robot_id * 0.01,
                0.2,
                0.3,
                0.25,
                0.1,
                0.2,
                0.0,
                age_ratio,
                debt_ratio,
            ]),
            "action_edge": torch.ones(2, 4),
            "edge_index": edge_index.clone(),
            "edge_features": torch.zeros(2, 6),
            "node_history": torch.zeros(1, 2, 10),
            "demand_context": torch.zeros(9),
            "station_node_ids": torch.tensor([0]),
        })
    defer_node = torch.zeros(2, 8)
    defer_node[0, 1] = 1.0
    defer_node[1, 2] = 1.0
    defer_node[0, 3] = 1.0
    rows.append({
        "run_id": f"seed{seed}",
        "simulation_seed": seed,
        "candidate_group_id": f"g{group_id}",
        "candidate_key": f"g{group_id}_defer",
        "candidate_info": {"robot_id": None},
        "action_type": "defer_context",
        "action_schema_version": DEFER_CONTEXT_ACTION_SCHEMA_VERSION,
        "action_encoding": DEFER_CONTEXT_ENCODING,
        "action_context": dict(context),
        "realized_cost": -100.0,
        "future_system_labels": labels(-100.0),
        "future_mask": torch.ones(10),
        "action_node": defer_node,
        "action_global": torch.tensor([
            0.0,
            0.2,
            0.3,
            0.25,
            0.0,
            0.2,
            1.0,
            age_ratio,
            debt_ratio,
        ]),
        "action_edge": torch.zeros(2, 4),
        "edge_index": edge_index.clone(),
        "edge_features": torch.zeros(2, 6),
        "node_history": torch.zeros(1, 2, 10),
        "demand_context": torch.zeros(9),
        "station_node_ids": torch.tensor([0]),
    })
    return rows


def test_round2_pairs_exclude_defer_and_equalise_group_weight():
    samples = _group(1, [1.0, 2.0, 3.0]) + _group(2, [1.0, 4.0])
    pairs = build_round2_pairwise_data(samples, seed=17)
    report = audit_round2_pairwise_data(pairs)

    assert len(pairs) == 4
    assert report["passed"]
    assert report["groups"] == 2
    assert all(
        pair["sample_i"]["action_type"] == "assign_robot"
        and pair["sample_j"]["action_type"] == "assign_robot"
        for pair in pairs
    )
    sums = defaultdict(float)
    for pair in pairs:
        sums[tuple(pair["candidate_group_key"])] += pair["pair_weight"]
    assert list(sums.values()) == pytest.approx([1.0, 1.0])
    assert all(
        pair["sample_i_ranking_cost"] < pair["sample_j_ranking_cost"]
        for pair in pairs
    )
    assert all(
        pair["sample_i"]["realized_cost"]
        > pair["sample_j"]["realized_cost"]
        for pair in pairs
    )


def test_empty_pairwise_audit_fails_closed():
    report = audit_round2_pairwise_data([])
    assert not report["passed"]
    assert not report["checks"]["nonempty"]


def test_max_pairs_is_applied_by_group_balanced_round_robin():
    samples = _group(1, [1.0, 2.0, 3.0]) + _group(2, [1.0, 4.0])
    pairs = build_round2_pairwise_data(samples, seed=23, max_pairs=3)
    report = audit_round2_pairwise_data(pairs)

    assert len(pairs) == 3
    assert report["passed"]
    assert report["groups"] == 2
    assert all(
        pair["pairwise_build_audit"]["truncated"] for pair in pairs
    )


def test_round2_pair_construction_is_deterministic_under_input_reordering():
    samples = _group(1, [1.0, 2.0, 3.0]) + _group(2, [1.0, 4.0])
    forward = build_round2_pairwise_data(samples, seed=31, max_pairs=3)
    backward = build_round2_pairwise_data(
        list(reversed(samples)), seed=31, max_pairs=3
    )
    signature = lambda rows: [
        (
            pair["sample_i"]["candidate_key"],
            pair["sample_j"]["candidate_key"],
            pair["pair_weight"],
        )
        for pair in rows
    ]
    assert signature(forward) == signature(backward)


def test_legacy_no_assign_is_rejected_instead_of_silently_filtered():
    samples = _group(1, [1.0, 2.0])
    samples[-1]["action_type"] = "no_assign"
    with pytest.raises(ValueError, match="legacy NO_ASSIGN"):
        build_round2_pairwise_data(samples)


def test_unknown_action_and_nonfinite_labels_fail_closed():
    unknown = _group(1, [1.0, 2.0])
    unknown[-1]["action_type"] = "typo_action"
    with pytest.raises(ValueError, match="unknown actions"):
        build_round2_pairwise_data(unknown)

    nonfinite = _group(1, [1.0, 2.0])
    nonfinite[0]["future_system_labels"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_round2_pairwise_data(nonfinite)


def test_pair_audit_rejects_forged_cross_group_provenance():
    pair = dict(build_round2_pairwise_data(_group(1, [1.0, 2.0]))[0])
    pair["sample_j"] = dict(pair["sample_j"])
    pair["sample_j"]["candidate_group_id"] = "forged_other_group"
    report = audit_round2_pairwise_data([pair])
    assert not report["passed"]
    assert not report["checks"]["group_provenance"]


def test_training_consumes_pair_weight_and_aggregates_by_group():
    pairs = build_round2_pairwise_data(
        _group(1, [1.0, 2.0, 3.0]) + _group(2, [1.0, 4.0]),
        seed=11,
    )
    pair = pairs[0]
    moved = pair_ranking_to_device_v2(pair, torch.device("cpu"))
    assert moved["pair_weight"] == pair["pair_weight"]
    assert moved["uniform_pair_sgd_weight"] == pair[
        "uniform_pair_sgd_weight"
    ]
    assert moved["candidate_group_key"] == tuple(pair["candidate_group_key"])

    with patch(
        "WorldModel.round2.training_v2.compute_ranking_loss",
        return_value=torch.tensor(3.0),
    ):
        weighted = compute_weighted_ranking_loss_v2(object(), pair)
    assert weighted.item() == pytest.approx(
        3.0 * pair["uniform_pair_sgd_weight"]
    )

    loss_by_pair = {
        (
            row["sample_i"]["candidate_key"],
            row["sample_j"]["candidate_key"],
        ): float(index + 1)
        for index, row in enumerate(pairs)
    }

    def fake_loss(_model, row, *, margin):
        del margin
        return torch.tensor(loss_by_pair[(
            row["sample_i"]["candidate_key"],
            row["sample_j"]["candidate_key"],
        )])

    with patch(
        "WorldModel.round2.training_v2.compute_ranking_loss",
        side_effect=fake_loss,
    ):
        objective = compute_group_balanced_ranking_objective_v2(
            object(), pairs
        )
    expected = sum(
        loss_by_pair[(
            pair["sample_i"]["candidate_key"],
            pair["sample_j"]["candidate_key"],
        )] * pair["pair_weight"]
        for pair in pairs
    ) / 2.0
    assert objective.item() == pytest.approx(expected)

    with patch(
        "WorldModel.round2.training_v2.compute_ranking_loss",
        side_effect=fake_loss,
    ):
        sgd_epoch_mean = torch.stack([
            compute_weighted_ranking_loss_v2(object(), row)
            for row in uniform_pair_sgd_epoch_v2(pairs, seed=91)
        ]).mean()
    assert sgd_epoch_mean.item() == pytest.approx(objective.item())


def test_epoch_aggregation_requires_each_group_weight_sum_one():
    rows = [
        {
            "candidate_group_key": ("run", "491", "g1"),
            "pair_weight": 0.5,
            "raw_ranking_loss": 2.0,
        },
        {
            "candidate_group_key": ("run", "491", "g1"),
            "pair_weight": 0.5,
            "raw_ranking_loss": 4.0,
        },
        {
            "candidate_group_key": ("run", "491", "g2"),
            "pair_weight": 1.0,
            "raw_ranking_loss": 1.0,
        },
    ]
    report = aggregate_group_balanced_epoch_v2(rows)
    assert report["passed"]
    assert report["mean_group_ranking_loss"] == pytest.approx(2.0)
    ordered = group_balanced_pair_order(
        build_round2_pairwise_data(
            _group(1, [1.0, 2.0, 3.0]) + _group(2, [1.0, 4.0]),
            seed=7,
        ),
        seed=7,
    )
    assert len({tuple(row["candidate_group_key"]) for row in ordered[:2]}) == 2
