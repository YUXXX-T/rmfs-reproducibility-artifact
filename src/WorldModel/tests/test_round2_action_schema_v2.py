from types import SimpleNamespace

import pytest
import torch

from WorldModel.round2.action_schema_v2 import (
    ASSIGN_CONTEXT_ENCODING,
    ASSIGN_ROBOT_ACTION_TYPE_V2,
    DEFER_CONTEXT_ACTION_SCHEMA_VERSION,
    DEFER_CONTEXT_ACTION_TYPE,
    action_sample_metadata_v2,
    build_action_edge_field_v2,
    build_action_field_v2,
    build_candidate_assignment_v2,
    infer_defer_context_action_schema_v2,
    make_defer_context_candidate_v2,
    schema_contract_v2,
)


class _Map:
    rows = 5
    cols = 5
    station_positions = {1: (0, 2)}


class _OrderState:
    @staticmethod
    def get_in_progress_orders():
        return [SimpleNamespace(station_id=1)]


def _world():
    return SimpleNamespace(
        map_state=_Map(),
        order_state=_OrderState(),
        agents=[object(), object(), object(), object()],
        config=SimpleNamespace(
            simulation=SimpleNamespace(
                max_items_per_order=2,
                max_items_per_sku=3,
            )
        ),
    )


NODE_MAP = {
    (0, 0): 0,
    (0, 1): 1,
    (0, 2): 2,
    (0, 3): 3,
    (1, 0): 4,
}
INV_NODE_MAP = {value: key for key, value in NODE_MAP.items()}
LOCAL_CAPACITY = [2.0] * len(NODE_MAP)
EDGE_INDEX = torch.tensor(
    [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
    dtype=torch.long,
)
FIXED_CONTEXT = {
    "order_id": 11,
    "pod_id": 7,
    "pod_location": (0, 1),
    "station_id": 1,
    "station_location": (0, 2),
    "entry_position": (1, 2),
    "exit_position": (0, 3),
    "return_location": (0, 3),
    "order_size": 2,
}
DISPATCH = {
    "context_age": 44.0,
    "free_flow_time": 44.0,
    "dispatch_debt": 2.0,
}


def _candidate(robot_id=3, robot_start=(0, 0)):
    return {
        "action_type": ASSIGN_ROBOT_ACTION_TYPE_V2,
        "robot_id": robot_id,
        "robot_start": robot_start,
    }


def _tensors(assignment, legs):
    node, glob = build_action_field_v2(
        assignment,
        _world(),
        NODE_MAP,
        INV_NODE_MAP,
        LOCAL_CAPACITY,
        precomputed_legs=legs,
    )
    edge = build_action_edge_field_v2(
        assignment,
        EDGE_INDEX,
        NODE_MAP,
        _Map(),
        precomputed_legs=legs,
    )
    return node, glob, edge


def test_schema_v2_defer_retains_context_and_zeros_only_robot_route_fields():
    assign = build_candidate_assignment_v2(
        _candidate(), FIXED_CONTEXT, **DISPATCH
    )
    defer_candidate = make_defer_context_candidate_v2(
        FIXED_CONTEXT, **DISPATCH
    )
    defer = build_candidate_assignment_v2(
        defer_candidate, FIXED_CONTEXT, **DISPATCH
    )
    assign_node, assign_global, assign_edge = _tensors(
        assign, ([0, 1], [1, 2], [2, 3])
    )
    defer_node, defer_global, defer_edge = _tensors(defer, ([], [], []))

    assert assign_node.shape == defer_node.shape == (5, 8)
    assert assign_global.shape == defer_global.shape == (9,)
    assert assign_edge.shape == defer_edge.shape == (6, 4)
    assert assign["action_encoding"] == ASSIGN_CONTEXT_ENCODING
    assert assign_global[6].item() == 0.0
    assert defer_global[6].item() == 1.0
    assert assign_global[7:].tolist() == pytest.approx([0.5, 2.0 / 3.0])
    assert defer_global[7:].tolist() == pytest.approx([0.5, 2.0 / 3.0])

    assert torch.count_nonzero(defer_node[:, 0]).item() == 0
    assert torch.count_nonzero(defer_node[:, 4:]).item() == 0
    assert defer_node[NODE_MAP[(0, 1)], 1].item() == 1.0
    assert defer_node[NODE_MAP[(0, 2)], 2].item() == 1.0
    assert defer_node[NODE_MAP[(0, 3)], 3].item() == 1.0
    assert torch.count_nonzero(defer_edge).item() == 0

    assert defer_global[0].item() == 0.0
    assert defer_global[1].item() > 0.0
    assert defer_global[2].item() > 0.0
    assert defer_global[3].item() == pytest.approx(0.25)
    assert defer_global[4].item() == 0.0
    assert defer_global[5].item() == pytest.approx(2.0 / 6.0)


def test_schema_v2_requires_explicit_dispatch_context_fields():
    assignment = build_candidate_assignment_v2(
        _candidate(), FIXED_CONTEXT, **DISPATCH
    )
    assignment.pop("dispatch_debt")
    with pytest.raises(ValueError, match="dispatch context fields"):
        build_action_field_v2(
            assignment,
            _world(),
            NODE_MAP,
            INV_NODE_MAP,
            LOCAL_CAPACITY,
            precomputed_legs=([0, 1], [1, 2], [2, 3]),
        )


def test_schema_v2_candidate_join_rejects_unknown_action_and_context_swap():
    with pytest.raises(ValueError, match="explicitly declare"):
        build_candidate_assignment_v2(
            {"action_type": "typo_action", "robot_id": 3},
            FIXED_CONTEXT,
            **DISPATCH,
        )

    defer_candidate = make_defer_context_candidate_v2(
        FIXED_CONTEXT, **DISPATCH
    )
    defer_candidate["fixed_context"] = dict(defer_candidate["fixed_context"])
    defer_candidate["fixed_context"]["pod_id"] = 999
    with pytest.raises(ValueError, match="fixed_context mismatch"):
        build_candidate_assignment_v2(
            defer_candidate, FIXED_CONTEXT, **DISPATCH
        )


def test_complete_schema_v2_group_audit_passes_and_detects_context_leakage():
    assignments = [
        build_candidate_assignment_v2(
            _candidate(3, (0, 0)), FIXED_CONTEXT, **DISPATCH
        ),
        build_candidate_assignment_v2(
            _candidate(4, (1, 0)), FIXED_CONTEXT, **DISPATCH
        ),
        build_candidate_assignment_v2(
            make_defer_context_candidate_v2(FIXED_CONTEXT, **DISPATCH),
            FIXED_CONTEXT,
            **DISPATCH,
        ),
    ]
    samples = []
    legs = (
        ([0, 1], [1, 2], [2, 3]),
        ([4, 0, 1], [1, 2], [2, 3]),
        ([], [], []),
    )
    for index, (assignment, action_legs) in enumerate(zip(assignments, legs)):
        node, glob, edge = _tensors(assignment, action_legs)
        samples.append({
            "run_id": "round2_low_seed491",
            "simulation_seed": 491,
            "candidate_group_id": "g1",
            "candidate_key": f"g1_{index}",
            "candidate_info": {"robot_id": assignment.get("robot_id")},
            **action_sample_metadata_v2(assignment),
            "action_node": node,
            "action_global": glob,
            "action_edge": edge,
        })

    report = infer_defer_context_action_schema_v2(samples)
    assert report["passed"]
    assert report["defer_context_samples"] == 1
    assert report["assign_robot_samples"] == 2
    assert report["legacy_no_assign_samples"] == 0

    samples[0]["action_context"] = dict(samples[0]["action_context"])
    samples[0]["action_context"]["dispatch_debt"] = 99.0
    failed = infer_defer_context_action_schema_v2(samples)
    assert not failed["passed"]
    assert not failed["checks"]["context_consistent_within_group"]


def test_schema_v2_audit_reports_missing_tensor_without_raising():
    assignments = [
        build_candidate_assignment_v2(
            _candidate(3, (0, 0)), FIXED_CONTEXT, **DISPATCH
        ),
        build_candidate_assignment_v2(
            _candidate(4, (1, 0)), FIXED_CONTEXT, **DISPATCH
        ),
        build_candidate_assignment_v2(
            make_defer_context_candidate_v2(FIXED_CONTEXT, **DISPATCH),
            FIXED_CONTEXT,
            **DISPATCH,
        ),
    ]
    samples = []
    for index, assignment in enumerate(assignments):
        node, glob, edge = _tensors(
            assignment,
            ([], [], []) if index == 2 else ([0, 1], [1, 2], [2, 3]),
        )
        samples.append({
            "run_id": "round2_low_seed491",
            "simulation_seed": 491,
            "candidate_group_id": "g1",
            "candidate_key": f"g1_{index}",
            **action_sample_metadata_v2(assignment),
            "action_node": node,
            "action_global": glob,
            "action_edge": edge,
        })

    samples[0].pop("action_global")
    report = infer_defer_context_action_schema_v2(samples)
    assert not report["passed"]
    assert "row[0]:missing_action_global" in report["failures"]


def test_schema_v2_audit_rejects_zero_assignment_and_ratio_tampering():
    assignments = [
        build_candidate_assignment_v2(
            _candidate(3, (0, 0)), FIXED_CONTEXT, **DISPATCH
        ),
        build_candidate_assignment_v2(
            _candidate(4, (1, 0)), FIXED_CONTEXT, **DISPATCH
        ),
        build_candidate_assignment_v2(
            make_defer_context_candidate_v2(FIXED_CONTEXT, **DISPATCH),
            FIXED_CONTEXT,
            **DISPATCH,
        ),
    ]
    samples = []
    for index, assignment in enumerate(assignments):
        node, glob, edge = _tensors(
            assignment,
            ([], [], []) if index == 2 else ([0, 1], [1, 2], [2, 3]),
        )
        samples.append({
            "run_id": "round2_low_seed491",
            "simulation_seed": 491,
            "candidate_group_id": "g1",
            "candidate_key": f"g1_{index}",
            "candidate_info": {"robot_id": assignment.get("robot_id")},
            **action_sample_metadata_v2(assignment),
            "action_node": node,
            "action_global": glob,
            "action_edge": edge,
        })

    samples[0]["action_node"] = torch.zeros_like(samples[0]["action_node"])
    samples[0]["action_edge"] = torch.zeros_like(samples[0]["action_edge"])
    zero_report = infer_defer_context_action_schema_v2(samples)
    assert not zero_report["passed"]
    assert any(
        "assign_marker_contract" in failure
        for failure in zero_report["failures"]
    )

    samples[0]["action_node"], _, samples[0]["action_edge"] = _tensors(
        assignments[0], ([0, 1], [1, 2], [2, 3])
    )
    samples[0]["action_global"] = samples[0]["action_global"].clone()
    samples[0]["action_global"][7] = 0.9
    ratio_report = infer_defer_context_action_schema_v2(samples)
    assert not ratio_report["passed"]
    assert "row[0]:global_age_ratio" in ratio_report["failures"]


def test_schema_contract_declares_only_global_dimension_change():
    contract = schema_contract_v2()
    assert contract["schema_version"] == DEFER_CONTEXT_ACTION_SCHEMA_VERSION
    assert contract["dimensions"] == {
        "action_node": 8,
        "action_global": 9,
        "action_edge": 4,
    }
    assert DEFER_CONTEXT_ACTION_TYPE in contract["action_types"]
