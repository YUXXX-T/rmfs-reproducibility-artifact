"""Isolated Phase-C Round-2 components.

This package is deliberately not imported by the current online or training
entry points.  Gate-1 can therefore remain frozen while schema-v2 components
are developed and audited here.
"""

from .action_schema_v2 import (
    ACTION_EDGE_DIM_V2,
    ACTION_GLOBAL_DIM_V2,
    ACTION_NODE_DIM_V2,
    ASSIGN_ROBOT_ACTION_TYPE_V2,
    DEFER_CONTEXT_ACTION_TYPE,
    DEFER_CONTEXT_ACTION_SCHEMA_VERSION,
    build_action_edge_field_v2,
    build_action_field_v2,
    build_candidate_assignment_v2,
    infer_defer_context_action_schema_v2,
    make_defer_context_candidate_v2,
)

__all__ = [
    "ACTION_EDGE_DIM_V2",
    "ACTION_GLOBAL_DIM_V2",
    "ACTION_NODE_DIM_V2",
    "ASSIGN_ROBOT_ACTION_TYPE_V2",
    "DEFER_CONTEXT_ACTION_TYPE",
    "DEFER_CONTEXT_ACTION_SCHEMA_VERSION",
    "build_action_edge_field_v2",
    "build_action_field_v2",
    "build_candidate_assignment_v2",
    "infer_defer_context_action_schema_v2",
    "make_defer_context_candidate_v2",
]
