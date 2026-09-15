"""Context-conditioned DEFER action contract for Phase-C Round-2.

The module is standalone and versioned so the frozen Gate-1 runtime does not
import it.  It reuses the six legacy assignment globals verbatim, then appends
three explicit context channels shared by assignment and defer:

``is_defer_context, age/(age+T_ff), debt/(1+debt)``.

DEFER keeps pod/station/return markers and context-level globals.  Only robot
and route-specific fields are zero.  This makes DEFER an action conditioned on
the declined chain instead of the Round-1 global all-zero constant.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence, Tuple

import torch

from WorldModel.graph.graph_builder import (
    build_action_edge_field as build_legacy_action_edge_field,
    build_action_field as build_legacy_action_field,
)


DEFER_CONTEXT_ACTION_SCHEMA_VERSION = "wm_defer_context_action_v2"
DEFER_CONTEXT_ENCODING = "context_markers_and_dispatch_state_v2"
ASSIGN_CONTEXT_ENCODING = "route_fields_and_dispatch_state_v2"
ASSIGN_ROBOT_ACTION_TYPE_V2 = "assign_robot"
DEFER_CONTEXT_ACTION_TYPE = "defer_context"

ACTION_NODE_DIM_V2 = 8
ACTION_GLOBAL_DIM_V2 = 9
ACTION_EDGE_DIM_V2 = 4

GLOBAL_IS_DEFER_INDEX = 6
GLOBAL_AGE_RATIO_INDEX = 7
GLOBAL_DEBT_RATIO_INDEX = 8

ROBOT_NODE_CHANNEL = 0
CONTEXT_NODE_CHANNELS = (1, 2, 3)
ROUTE_NODE_CHANNELS = (4, 5, 6, 7)


@dataclass(frozen=True)
class DispatchContextFeaturesV2:
    context_age: float
    free_flow_time: float
    dispatch_debt: float

    def __post_init__(self) -> None:
        values = (
            float(self.context_age),
            float(self.free_flow_time),
            float(self.dispatch_debt),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("dispatch context features must be finite")
        if self.context_age < 0.0:
            raise ValueError("context_age must be non-negative")
        if self.free_flow_time <= 0.0:
            raise ValueError("free_flow_time must be positive")
        if self.dispatch_debt < 0.0:
            raise ValueError("dispatch_debt must be non-negative")

    @property
    def age_ratio(self) -> float:
        age = float(self.context_age)
        return age / (age + float(self.free_flow_time))

    @property
    def debt_ratio(self) -> float:
        debt = float(self.dispatch_debt)
        return debt / (1.0 + debt)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "age_over_age_plus_tff": self.age_ratio,
            "debt_over_one_plus_debt": self.debt_ratio,
        }


def _action_type(value: Mapping) -> str:
    return str(value.get("action_type", "")).lower()


def is_defer_context_action(value: Optional[Mapping]) -> bool:
    return bool(value and _action_type(value) == DEFER_CONTEXT_ACTION_TYPE)


def _position(value, name: str) -> Tuple[int, int]:
    if value is None or len(value) != 2:
        raise ValueError(f"{name} must be a two-dimensional position")
    return int(value[0]), int(value[1])


def _context_features(assignment: Mapping) -> DispatchContextFeaturesV2:
    missing = [
        name
        for name in ("context_age", "free_flow_time", "dispatch_debt")
        if name not in assignment
    ]
    if missing:
        raise ValueError(
            "schema-v2 assignment lacks dispatch context fields: "
            + ", ".join(missing)
        )
    return DispatchContextFeaturesV2(
        context_age=float(assignment["context_age"]),
        free_flow_time=float(assignment["free_flow_time"]),
        dispatch_debt=float(assignment["dispatch_debt"]),
    )


def validate_action_assignment_v2(assignment: Mapping) -> None:
    action_type = _action_type(assignment)
    if action_type not in (
        ASSIGN_ROBOT_ACTION_TYPE_V2,
        DEFER_CONTEXT_ACTION_TYPE,
    ):
        raise ValueError(f"unsupported schema-v2 action_type: {action_type!r}")
    if assignment.get("action_schema_version") != (
        DEFER_CONTEXT_ACTION_SCHEMA_VERSION
    ):
        raise ValueError("assignment does not declare wm_defer_context_action_v2")
    for name in (
        "order_id",
        "pod_id",
        "station_id",
        "pod_location",
        "station_location",
        "return_location",
        "order_size",
    ):
        if name not in assignment:
            raise ValueError(f"schema-v2 assignment lacks {name}")
    for name in ("pod_location", "station_location", "return_location"):
        _position(assignment[name], name)
    _context_features(assignment)

    robot_id = assignment.get("robot_id")
    robot_start = assignment.get("robot_start")
    if action_type == ASSIGN_ROBOT_ACTION_TYPE_V2:
        if robot_id is None or robot_start is None:
            raise ValueError("assign_robot requires robot_id and robot_start")
        _position(robot_start, "robot_start")
        if assignment.get("action_encoding") != ASSIGN_CONTEXT_ENCODING:
            raise ValueError("assign_robot has an invalid v2 action encoding")
    else:
        if robot_id is not None or robot_start is not None:
            raise ValueError("defer_context forbids robot-specific identity")
        if assignment.get("action_encoding") != DEFER_CONTEXT_ENCODING:
            raise ValueError("defer_context has an invalid v2 action encoding")


def make_defer_context_candidate_v2(
    fixed_context: Mapping,
    *,
    context_age: float,
    free_flow_time: float,
    dispatch_debt: float,
) -> dict:
    """Create a context-bearing candidate without robot-specific fields."""
    features = DispatchContextFeaturesV2(
        context_age=context_age,
        free_flow_time=free_flow_time,
        dispatch_debt=dispatch_debt,
    )
    return {
        "action_type": DEFER_CONTEXT_ACTION_TYPE,
        "action_schema_version": DEFER_CONTEXT_ACTION_SCHEMA_VERSION,
        "action_encoding": DEFER_CONTEXT_ENCODING,
        "robot_id": None,
        "robot_start": None,
        "candidate_policy": "native_defer_context_v2",
        "candidate_selection_mode": "native_defer_context_v2",
        "chosen": False,
        "fixed_context": dict(fixed_context),
        **asdict(features),
    }


def build_candidate_assignment_v2(
    candidate: Mapping,
    fixed_context: Mapping,
    *,
    context_age: float,
    free_flow_time: float,
    dispatch_debt: float,
) -> dict:
    """Join one candidate with its complete context under schema v2."""
    features = DispatchContextFeaturesV2(
        context_age=context_age,
        free_flow_time=free_flow_time,
        dispatch_debt=dispatch_debt,
    )
    candidate_action_type = _action_type(candidate)
    if candidate_action_type not in (
        ASSIGN_ROBOT_ACTION_TYPE_V2,
        DEFER_CONTEXT_ACTION_TYPE,
    ):
        raise ValueError(
            "schema-v2 candidate must explicitly declare assign_robot or "
            "defer_context"
        )
    defer = candidate_action_type == DEFER_CONTEXT_ACTION_TYPE
    if defer:
        embedded_context = candidate.get("fixed_context")
        if not isinstance(embedded_context, Mapping):
            raise ValueError("defer_context requires its embedded fixed_context")
        comparable_fields = (
            "order_id",
            "pod_id",
            "station_id",
            "order_size",
            "pod_location",
            "station_location",
            "return_location",
        )
        for field in comparable_fields:
            embedded_value = embedded_context.get(field)
            joined_value = fixed_context.get(field)
            if field.endswith("_location"):
                embedded_value = _position(embedded_value, field)
                joined_value = _position(joined_value, field)
            if embedded_value != joined_value:
                raise ValueError(
                    f"defer_context fixed_context mismatch for {field}"
                )
    action_type = (
        DEFER_CONTEXT_ACTION_TYPE if defer else ASSIGN_ROBOT_ACTION_TYPE_V2
    )
    assignment = {
        "action_type": action_type,
        "action_schema_version": DEFER_CONTEXT_ACTION_SCHEMA_VERSION,
        "action_encoding": (
            DEFER_CONTEXT_ENCODING if defer else ASSIGN_CONTEXT_ENCODING
        ),
        "order_id": int(fixed_context["order_id"]),
        "pod_id": int(fixed_context["pod_id"]),
        "station_id": int(fixed_context["station_id"]),
        "pod_location": _position(
            fixed_context["pod_location"], "pod_location"
        ),
        "station_location": _position(
            fixed_context["station_location"], "station_location"
        ),
        "entry_position": (
            _position(fixed_context["entry_position"], "entry_position")
            if fixed_context.get("entry_position") is not None
            else None
        ),
        "exit_position": (
            _position(fixed_context["exit_position"], "exit_position")
            if fixed_context.get("exit_position") is not None
            else None
        ),
        "return_location": _position(
            fixed_context["return_location"], "return_location"
        ),
        "order_size": int(fixed_context.get("order_size", 1)),
        "robot_id": None if defer else int(candidate["robot_id"]),
        "robot_start": (
            None
            if defer
            else _position(candidate["robot_start"], "robot_start")
        ),
        **asdict(features),
    }
    validate_action_assignment_v2(assignment)
    return assignment


def _station_queue(world, station_id: int) -> int:
    return sum(
        int(getattr(order, "station_id", -1)) == int(station_id)
        for order in world.order_state.get_in_progress_orders()
    )


def _normalisation_scales(
    world,
    *,
    station_queue_scale: Optional[float],
    order_size_scale: Optional[float],
) -> tuple[float, float]:
    if station_queue_scale is None:
        station_queue_scale = max(len(world.agents), 1)
    if order_size_scale is None:
        sim_cfg = getattr(getattr(world, "config", None), "simulation", None)
        order_size_scale = (
            float(sim_cfg.max_items_per_order * sim_cfg.max_items_per_sku)
            if sim_cfg is not None
            else 10.0
        )
    return max(float(station_queue_scale), 1.0), max(
        float(order_size_scale), 1.0
    )


def _manhattan(a: Sequence[int], b: Sequence[int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def build_action_field_v2(
    assignment: Mapping,
    world,
    node_map,
    inv_node_map,
    local_capacity,
    *,
    path_planner=None,
    node_features: Optional[torch.Tensor] = None,
    station_queue_scale: Optional[float] = None,
    order_size_scale: Optional[float] = None,
    precomputed_legs=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(N, 8)`` node and ``(9,)`` global schema-v2 tensors."""
    validate_action_assignment_v2(assignment)
    defer = is_defer_context_action(assignment)
    features = _context_features(assignment)

    if not defer:
        legacy_assignment = {
            "action_type": ASSIGN_ROBOT_ACTION_TYPE_V2,
            "robot_id": assignment["robot_id"],
            "robot_start": assignment["robot_start"],
            "pod_id": assignment["pod_id"],
            "pod_location": assignment["pod_location"],
            "station_location": assignment["station_location"],
            "return_location": assignment["return_location"],
            "order_size": assignment["order_size"],
        }
        action_node, legacy_global = build_legacy_action_field(
            legacy_assignment,
            world,
            node_map,
            inv_node_map,
            local_capacity,
            path_planner=path_planner,
            node_features=node_features,
            station_queue_scale=station_queue_scale,
            order_size_scale=order_size_scale,
            precomputed_legs=precomputed_legs,
        )
    else:
        n_nodes = len(node_map)
        action_node = torch.zeros(
            (n_nodes, ACTION_NODE_DIM_V2), dtype=torch.float32
        )
        for channel, field in zip(
            CONTEXT_NODE_CHANNELS,
            ("pod_location", "station_location", "return_location"),
        ):
            node_id = node_map.get(tuple(assignment[field]))
            if node_id is not None:
                action_node[int(node_id), int(channel)] = 1.0

        queue_scale, size_scale = _normalisation_scales(
            world,
            station_queue_scale=station_queue_scale,
            order_size_scale=order_size_scale,
        )
        max_distance = max(
            int(world.map_state.rows) + int(world.map_state.cols), 1
        )
        station_queue = _station_queue(world, int(assignment["station_id"]))
        legacy_global = torch.tensor(
            [
                0.0,
                _manhattan(
                    assignment["pod_location"],
                    assignment["station_location"],
                ) / max_distance,
                _manhattan(
                    assignment["station_location"],
                    assignment["return_location"],
                ) / max_distance,
                station_queue / queue_scale,
                0.0,
                float(assignment["order_size"]) / size_scale,
            ],
            dtype=torch.float32,
        )

    context_global = torch.tensor(
        [
            1.0 if defer else 0.0,
            features.age_ratio,
            features.debt_ratio,
        ],
        dtype=legacy_global.dtype,
        device=legacy_global.device,
    )
    action_global = torch.cat((legacy_global, context_global), dim=0)
    if tuple(action_node.shape) != (len(node_map), ACTION_NODE_DIM_V2):
        raise RuntimeError("schema-v2 action_node shape contract failed")
    if tuple(action_global.shape) != (ACTION_GLOBAL_DIM_V2,):
        raise RuntimeError("schema-v2 action_global shape contract failed")
    return action_node, action_global


def build_action_edge_field_v2(
    assignment: Mapping,
    edge_index: torch.LongTensor,
    node_map,
    map_state,
    *,
    path_planner=None,
    world=None,
    precomputed_legs=None,
) -> torch.Tensor:
    """Build route-only edge fields; DEFER has no route by definition."""
    validate_action_assignment_v2(assignment)
    if is_defer_context_action(assignment):
        return torch.zeros(
            (int(edge_index.shape[1]), ACTION_EDGE_DIM_V2),
            dtype=torch.float32,
        )
    legacy_assignment = {
        "action_type": ASSIGN_ROBOT_ACTION_TYPE_V2,
        "robot_id": assignment["robot_id"],
        "robot_start": assignment["robot_start"],
        "pod_id": assignment["pod_id"],
        "pod_location": assignment["pod_location"],
        "station_location": assignment["station_location"],
        "return_location": assignment["return_location"],
    }
    result = build_legacy_action_edge_field(
        legacy_assignment,
        edge_index,
        node_map,
        map_state,
        path_planner=path_planner,
        world=world,
        precomputed_legs=precomputed_legs,
    )
    if tuple(result.shape) != (int(edge_index.shape[1]), ACTION_EDGE_DIM_V2):
        raise RuntimeError("schema-v2 action_edge shape contract failed")
    return result


def action_sample_metadata_v2(assignment: Mapping) -> dict:
    """Return the explicit metadata stored beside schema-v2 tensors."""
    validate_action_assignment_v2(assignment)
    features = _context_features(assignment)
    return {
        "action_type": _action_type(assignment),
        "action_schema_version": DEFER_CONTEXT_ACTION_SCHEMA_VERSION,
        "action_encoding": assignment["action_encoding"],
        "action_context": {
            "order_id": int(assignment["order_id"]),
            "pod_id": int(assignment["pod_id"]),
            "station_id": int(assignment["station_id"]),
            **features.to_dict(),
        },
    }


def infer_defer_context_action_schema_v2(samples: Sequence[Mapping]) -> dict:
    """Fail-closed audit of a complete schema-v2 candidate dataset."""
    from WorldModel.data.dataset import stable_candidate_group_key

    rows = list(samples or ())
    groups = {}
    failures = []
    parsed_tensors = {}
    parsed_contexts = {}
    assign_count = 0
    defer_count = 0
    legacy_no_assign_count = 0
    for index, sample in enumerate(rows):
        groups.setdefault(stable_candidate_group_key(sample), []).append(
            (index, sample)
        )
        action_type = _action_type(sample)
        if action_type == "no_assign":
            legacy_no_assign_count += 1
        elif action_type == DEFER_CONTEXT_ACTION_TYPE:
            defer_count += 1
        elif action_type == ASSIGN_ROBOT_ACTION_TYPE_V2:
            assign_count += 1
        else:
            failures.append(f"row[{index}]:invalid_action_type")
        if sample.get("action_schema_version") != (
            DEFER_CONTEXT_ACTION_SCHEMA_VERSION
        ):
            failures.append(f"row[{index}]:schema_version")
            continue
        tensors = {}
        for field in ("action_node", "action_global", "action_edge"):
            value = sample.get(field)
            if value is None:
                failures.append(f"row[{index}]:missing_{field}")
                tensors[field] = None
                continue
            try:
                tensors[field] = torch.as_tensor(value)
            except (TypeError, ValueError, RuntimeError):
                failures.append(f"row[{index}]:invalid_{field}")
                tensors[field] = None

        node = tensors["action_node"]
        glob = tensors["action_global"]
        edge = tensors["action_edge"]
        node_shape_valid = bool(
            node is not None
            and node.ndim == 2
            and node.shape[1] == ACTION_NODE_DIM_V2
        )
        global_shape_valid = bool(
            glob is not None and tuple(glob.shape) == (ACTION_GLOBAL_DIM_V2,)
        )
        edge_shape_valid = bool(
            edge is not None
            and edge.ndim == 2
            and edge.shape[1] == ACTION_EDGE_DIM_V2
        )
        if node is not None and not node_shape_valid:
            failures.append(f"row[{index}]:action_node_shape")
        if glob is not None and not global_shape_valid:
            failures.append(f"row[{index}]:action_global_shape")
        if edge is not None and not edge_shape_valid:
            failures.append(f"row[{index}]:action_edge_shape")
        for field, tensor in tensors.items():
            if tensor is not None and not bool(torch.isfinite(tensor).all()):
                failures.append(f"row[{index}]:nonfinite_{field}")
        parsed_tensors[index] = (node, glob, edge)

        context = sample.get("action_context")
        parsed_context = None
        if not isinstance(context, Mapping):
            failures.append(f"row[{index}]:missing_action_context")
        else:
            required_context = (
                "order_id",
                "pod_id",
                "station_id",
                "context_age",
                "free_flow_time",
                "dispatch_debt",
                "age_over_age_plus_tff",
                "debt_over_one_plus_debt",
            )
            missing_context = [
                field for field in required_context if field not in context
            ]
            if missing_context:
                failures.append(
                    f"row[{index}]:action_context_fields="
                    + ",".join(missing_context)
                )
            else:
                try:
                    context_features = DispatchContextFeaturesV2(
                        context_age=float(context["context_age"]),
                        free_flow_time=float(context["free_flow_time"]),
                        dispatch_debt=float(context["dispatch_debt"]),
                    )
                    declared_age_ratio = float(
                        context["age_over_age_plus_tff"]
                    )
                    declared_debt_ratio = float(
                        context["debt_over_one_plus_debt"]
                    )
                    if not math.isclose(
                        declared_age_ratio,
                        context_features.age_ratio,
                        rel_tol=1e-6,
                        abs_tol=1e-7,
                    ):
                        failures.append(f"row[{index}]:context_age_ratio")
                    if not math.isclose(
                        declared_debt_ratio,
                        context_features.debt_ratio,
                        rel_tol=1e-6,
                        abs_tol=1e-7,
                    ):
                        failures.append(f"row[{index}]:context_debt_ratio")
                    parsed_context = (
                        int(context["order_id"]),
                        int(context["pod_id"]),
                        int(context["station_id"]),
                        float(context_features.context_age),
                        float(context_features.free_flow_time),
                        float(context_features.dispatch_debt),
                    )
                    if global_shape_valid and not math.isclose(
                        float(glob[GLOBAL_AGE_RATIO_INDEX]),
                        context_features.age_ratio,
                        rel_tol=1e-6,
                        abs_tol=1e-7,
                    ):
                        failures.append(f"row[{index}]:global_age_ratio")
                    if global_shape_valid and not math.isclose(
                        float(glob[GLOBAL_DEBT_RATIO_INDEX]),
                        context_features.debt_ratio,
                        rel_tol=1e-6,
                        abs_tol=1e-7,
                    ):
                        failures.append(f"row[{index}]:global_debt_ratio")
                except (TypeError, ValueError, OverflowError):
                    failures.append(f"row[{index}]:invalid_action_context")
        parsed_contexts[index] = parsed_context

        candidate_info = sample.get("candidate_info")
        if not isinstance(candidate_info, Mapping):
            failures.append(f"row[{index}]:candidate_info")
            candidate_robot_id = None
        else:
            candidate_robot_id = candidate_info.get("robot_id")
        if action_type == DEFER_CONTEXT_ACTION_TYPE:
            if sample.get("action_encoding") != DEFER_CONTEXT_ENCODING:
                failures.append(f"row[{index}]:defer_encoding")
            if (
                global_shape_valid
                and float(glob[GLOBAL_IS_DEFER_INDEX]) != 1.0
            ):
                failures.append(f"row[{index}]:defer_flag")
            if node_shape_valid and (
                bool(torch.count_nonzero(node[:, ROBOT_NODE_CHANNEL]).item())
                or bool(torch.count_nonzero(node[:, ROUTE_NODE_CHANNELS]).item())
            ):
                failures.append(f"row[{index}]:defer_robot_or_route_nonzero")
            if edge_shape_valid and bool(torch.count_nonzero(edge).item()):
                failures.append(f"row[{index}]:defer_edge_nonzero")
            if node_shape_valid and any(
                int(torch.count_nonzero(node[:, channel]).item()) != 1
                for channel in CONTEXT_NODE_CHANNELS
            ):
                failures.append(f"row[{index}]:defer_context_marker_missing")
            if candidate_robot_id is not None:
                failures.append(f"row[{index}]:defer_robot_identity")
        elif action_type == ASSIGN_ROBOT_ACTION_TYPE_V2:
            if sample.get("action_encoding") != ASSIGN_CONTEXT_ENCODING:
                failures.append(f"row[{index}]:assign_encoding")
            if (
                global_shape_valid
                and float(glob[GLOBAL_IS_DEFER_INDEX]) != 0.0
            ):
                failures.append(f"row[{index}]:assign_flag")
            if candidate_robot_id is None:
                failures.append(f"row[{index}]:assign_robot_identity")
            if node_shape_valid:
                marker_channels = (ROBOT_NODE_CHANNEL, *CONTEXT_NODE_CHANNELS)
                if any(
                    int(torch.count_nonzero(node[:, channel]).item()) != 1
                    for channel in marker_channels
                ):
                    failures.append(f"row[{index}]:assign_marker_contract")
                if not bool(
                    torch.count_nonzero(node[:, ROUTE_NODE_CHANNELS]).item()
                ):
                    failures.append(f"row[{index}]:assign_route_missing")
            if edge_shape_valid and not bool(torch.count_nonzero(edge).item()):
                failures.append(f"row[{index}]:assign_edge_route_missing")

    complete_groups = True
    context_consistent_groups = True
    for group_key, indexed_members in groups.items():
        members = [sample for _, sample in indexed_members]
        defer_rows = [
            row for row in members if _action_type(row) == DEFER_CONTEXT_ACTION_TYPE
        ]
        assign_rows = [
            row for row in members if _action_type(row) == ASSIGN_ROBOT_ACTION_TYPE_V2
        ]
        if len(defer_rows) != 1 or len(assign_rows) < 2:
            complete_groups = False
            failures.append(f"group[{group_key}]:coverage")
        contexts = [parsed_contexts[index] for index, _ in indexed_members]
        if (
            not contexts
            or contexts[0] is None
            or any(context != contexts[0] for context in contexts)
        ):
            context_consistent_groups = False
            failures.append(f"group[{group_key}]:context_mismatch")
        candidate_keys = [str(row.get("candidate_key", "")) for row in members]
        if any(not key for key in candidate_keys) or len(set(candidate_keys)) != len(
            candidate_keys
        ):
            failures.append(f"group[{group_key}]:candidate_keys")

        reference_context_node = None
        reference_context_global = None
        for index, _ in indexed_members:
            node, glob, _ = parsed_tensors[index]
            if (
                node is None
                or node.ndim != 2
                or node.shape[1] != ACTION_NODE_DIM_V2
                or glob is None
                or tuple(glob.shape) != (ACTION_GLOBAL_DIM_V2,)
            ):
                continue
            context_node = node[:, CONTEXT_NODE_CHANNELS]
            context_global = glob[[1, 2, 3, 5, 7, 8]]
            if reference_context_node is None:
                reference_context_node = context_node
                reference_context_global = context_global
                continue
            if not torch.equal(context_node, reference_context_node):
                failures.append(f"group[{group_key}]:context_node_mismatch")
            if not torch.allclose(
                context_global,
                reference_context_global,
                rtol=1e-6,
                atol=1e-7,
            ):
                failures.append(f"group[{group_key}]:context_global_mismatch")

    checks = {
        "nonempty": bool(rows),
        "no_legacy_no_assign": legacy_no_assign_count == 0,
        "assign_and_defer_present": assign_count > 0 and defer_count > 0,
        "complete_group_coverage": bool(groups) and complete_groups,
        "context_consistent_within_group": bool(groups)
        and context_consistent_groups,
        "tensor_and_encoding_contract": not failures,
    }
    return {
        "schema_version": DEFER_CONTEXT_ACTION_SCHEMA_VERSION,
        "supports_defer_context": all(checks.values()),
        "action_node_dim": ACTION_NODE_DIM_V2,
        "action_global_dim": ACTION_GLOBAL_DIM_V2,
        "action_edge_dim": ACTION_EDGE_DIM_V2,
        "training_samples": len(rows),
        "assign_robot_samples": assign_count,
        "defer_context_samples": defer_count,
        "legacy_no_assign_samples": legacy_no_assign_count,
        "candidate_groups": len(groups),
        "checks": checks,
        "failures": failures[:100],
        "passed": all(checks.values()),
    }


def schema_contract_v2() -> dict:
    return {
        "schema_version": DEFER_CONTEXT_ACTION_SCHEMA_VERSION,
        "action_types": [
            ASSIGN_ROBOT_ACTION_TYPE_V2,
            DEFER_CONTEXT_ACTION_TYPE,
        ],
        "dimensions": {
            "action_node": ACTION_NODE_DIM_V2,
            "action_global": ACTION_GLOBAL_DIM_V2,
            "action_edge": ACTION_EDGE_DIM_V2,
        },
        "global_channels": [
            "robot_to_pod_distance",
            "pod_to_station_distance",
            "station_to_return_distance",
            "station_queue",
            "route_pressure",
            "order_size",
            "is_defer_context",
            "age_over_age_plus_tff",
            "debt_over_one_plus_debt",
        ],
        "defer_zero_fields": [
            "robot_marker",
            "robot_to_pod_distance",
            "route_pressure",
            "route_node_channels",
            "route_edge_channels",
        ],
        "defer_preserved_fields": [
            "pod_marker",
            "station_marker",
            "return_marker",
            "pod_to_station_distance",
            "station_to_return_distance",
            "station_queue",
            "order_size",
            "context_age_ratio",
            "dispatch_debt_ratio",
        ],
    }


__all__ = [
    "ACTION_EDGE_DIM_V2",
    "ACTION_GLOBAL_DIM_V2",
    "ACTION_NODE_DIM_V2",
    "ASSIGN_CONTEXT_ENCODING",
    "ASSIGN_ROBOT_ACTION_TYPE_V2",
    "DEFER_CONTEXT_ACTION_SCHEMA_VERSION",
    "DEFER_CONTEXT_ACTION_TYPE",
    "DEFER_CONTEXT_ENCODING",
    "DispatchContextFeaturesV2",
    "action_sample_metadata_v2",
    "build_action_edge_field_v2",
    "build_action_field_v2",
    "build_candidate_assignment_v2",
    "infer_defer_context_action_schema_v2",
    "is_defer_context_action",
    "make_defer_context_candidate_v2",
    "schema_contract_v2",
    "validate_action_assignment_v2",
]
