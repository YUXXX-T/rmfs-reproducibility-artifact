"""Explicit Round-1 to schema-v2 World Model checkpoint migration.

Compatible state/demand/dynamics parameters are copied exactly.  The complete
action encoder is rebuilt because its global input contract changes from 6 to
9, while the cost lambdas are explicitly replaced by the frozen Round-2
congestion-only contract.  No padding, slicing, or silent shape adaptation is
permitted.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from WorldModel.core.costs import CONGESTION_LAMBDAS
from WorldModel.core.model import RMFSWorldModel
from WorldModel.round2.action_schema_v2 import (
    ACTION_GLOBAL_DIM_V2,
    ACTION_NODE_DIM_V2,
    DEFER_CONTEXT_ACTION_SCHEMA_VERSION,
    schema_contract_v2,
)
from WorldModel.round2.pairwise_v2 import (
    ROUND2_PAIRWISE_SCHEMA_VERSION,
    ROUND2_RANKING_COST_SCHEMA_VERSION,
)


ROUND1_ACTION_SCHEMA = "wm_native_no_assign_action_v1"
MIGRATION_SCHEMA_VERSION = "wm_round1_to_defer_context_v2_migration_v1"
ROUND2_COST_SCHEMA_VERSION = ROUND2_RANKING_COST_SCHEMA_VERSION
ROUND2_TRAINING_AUDIT_SCHEMA_VERSION = "wm_round2_training_audit_v1"
REBUILT_PREFIX = "action_encoder."
OVERRIDDEN_COST_KEY = "cost_head.lambdas"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def action_encoder_sha256_v2(model: RMFSWorldModel) -> str:
    return _tensor_state_sha256(model.action_encoder.state_dict())


def build_round2_training_audit_v2(
    model: RMFSWorldModel,
    *,
    migration_audit: Mapping,
    training_action_schema: Mapping,
    pairwise_audit: Mapping,
    optimizer_steps: int,
) -> dict:
    """Bind a trained model to the action, ranking, and migration contracts."""
    current_digest = action_encoder_sha256_v2(model)
    expected_cost = torch.tensor(
        CONGESTION_LAMBDAS,
        dtype=model.cost_head.lambdas.dtype,
        device=model.cost_head.lambdas.device,
    )
    checks = {
        "migration_passed": (
            migration_audit.get("schema_version") == MIGRATION_SCHEMA_VERSION
            and bool(migration_audit.get("passed"))
        ),
        "action_schema_passed": (
            training_action_schema.get("schema_version")
            == DEFER_CONTEXT_ACTION_SCHEMA_VERSION
            and bool(training_action_schema.get("passed"))
            and bool(training_action_schema.get("supports_defer_context"))
        ),
        "pairwise_schema_passed": (
            pairwise_audit.get("schema_version")
            == ROUND2_PAIRWISE_SCHEMA_VERSION
            and bool(pairwise_audit.get("passed"))
            and bool(
                (pairwise_audit.get("checks") or {}).get(
                    "congestion_only_cost_contract"
                )
            )
            and bool(
                (pairwise_audit.get("checks") or {}).get(
                    "uniform_pair_sgd_unbiased_scale"
                )
            )
        ),
        "optimizer_steps_positive": int(optimizer_steps) > 0,
        "action_encoder_changed": current_digest
        != migration_audit.get("initial_action_encoder_sha256"),
        "congestion_cost_exact": torch.equal(
            model.cost_head.lambdas.detach(), expected_cost
        ),
        "congestion_cost_frozen": not model.cost_head.lambdas.requires_grad,
        "schema_v2_encoder_dimensions": (
            model.action_encoder.node_net[0].in_features == ACTION_NODE_DIM_V2
            and model.action_encoder.global_net[0].in_features
            == ACTION_GLOBAL_DIM_V2
        ),
    }
    return {
        "schema_version": ROUND2_TRAINING_AUDIT_SCHEMA_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "optimizer_steps": int(optimizer_steps),
        "action_encoder_trained": bool(checks["action_encoder_changed"]),
        "trained_action_encoder_sha256": current_digest,
        "migration_initial_action_encoder_sha256": migration_audit.get(
            "initial_action_encoder_sha256"
        ),
        "action_schema_version": training_action_schema.get("schema_version"),
        "pairwise_schema_version": pairwise_audit.get("schema_version"),
        "ranking_cost_schema_version": ROUND2_RANKING_COST_SCHEMA_VERSION,
        "cost_lambdas": list(CONGESTION_LAMBDAS),
    }


def _load_payload(source) -> tuple[dict[str, Any], str | None, str | None]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload, path.as_posix(), _sha256_file(path)
    if not isinstance(source, Mapping):
        raise TypeError("source must be a checkpoint path or mapping")
    return dict(source), None, None


def _model_config(payload: Mapping) -> dict:
    source = dict(payload.get("model_config") or {})
    required = (
        "node_feat_dim",
        "edge_feat_dim",
        "demand_dim",
        "hidden_dim",
        "num_stations",
        "rollout_horizon",
    )
    missing = [name for name in required if name not in source]
    if missing:
        raise ValueError(
            "source checkpoint lacks model config fields: "
            + ", ".join(missing)
        )
    if int(source.get("action_node_dim", -1)) != ACTION_NODE_DIM_V2:
        raise ValueError("Round-1 source action_node_dim must be 8")
    if int(source.get("action_global_dim", -1)) != 6:
        raise ValueError("Round-1 source action_global_dim must be 6")
    return {
        "node_feat_dim": int(source["node_feat_dim"]),
        "edge_feat_dim": int(source["edge_feat_dim"]),
        "demand_dim": int(source["demand_dim"]),
        "action_node_dim": ACTION_NODE_DIM_V2,
        "action_global_dim": ACTION_GLOBAL_DIM_V2,
        "hidden_dim": int(source["hidden_dim"]),
        "num_spatial_layers": int(source.get("num_spatial_layers", 3)),
        "rollout_horizon": int(source["rollout_horizon"]),
        "num_stations": int(source["num_stations"]),
    }


def migrate_round1_checkpoint_to_v2_model(
    source,
    *,
    action_encoder_init_seed: int = 20260728,
) -> tuple[RMFSWorldModel, dict]:
    """Construct a v2 model and return a machine-checkable migration audit."""
    payload, source_path, source_sha256 = _load_payload(source)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("unsupported source checkpoint payload")
    source_schema = dict(payload.get("action_schema") or {})
    if (
        source_schema.get("schema_version") != ROUND1_ACTION_SCHEMA
        or not bool(source_schema.get("supports_no_assign_candidate"))
        or source_schema.get("no_assign_encoding")
        != "zero_action_tensors_v1"
        or not bool(source_schema.get("complete_group_coverage"))
        or not bool(source_schema.get("zero_encoding_verified"))
    ):
        raise ValueError("source is not the audited Phase-C Round-1 checkpoint")
    config = _model_config(payload)

    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(action_encoder_init_seed))
        model = RMFSWorldModel(**config)
    finally:
        torch.random.set_rng_state(rng_state)

    source_state = dict(payload["state_dict"])
    target_state = model.state_dict()
    source_keys = set(source_state)
    target_keys = set(target_state)
    if source_keys != target_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        raise ValueError(
            "source/target parameter names differ; "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )

    inherited = []
    rebuilt = []
    overridden = []
    for name, target_tensor in target_state.items():
        source_tensor = source_state[name]
        if name.startswith(REBUILT_PREFIX):
            rebuilt.append(name)
            continue
        if name == OVERRIDDEN_COST_KEY:
            congestion_lambdas = torch.tensor(
                CONGESTION_LAMBDAS,
                dtype=target_tensor.dtype,
                device=target_tensor.device,
            )
            if tuple(congestion_lambdas.shape) != tuple(target_tensor.shape):
                raise ValueError("congestion-only cost shape contract failed")
            target_state[name] = congestion_lambdas
            overridden.append(name)
            continue
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f"non-action parameter shape changed: {name} "
                f"{tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}"
            )
        target_state[name] = source_tensor.detach().clone()
        inherited.append(name)

    if not rebuilt or any(not name.startswith(REBUILT_PREFIX) for name in rebuilt):
        raise RuntimeError("migration did not isolate action encoder rebuild")
    model.load_state_dict(target_state, strict=True)
    model.cost_head.lambdas.requires_grad_(False)
    model._checkpoint_action_schema = {
        "schema_version": DEFER_CONTEXT_ACTION_SCHEMA_VERSION,
        "supports_defer_context": True,
        "requires_complete_group_coverage": True,
        "requires_schema_v2_training_before_online_use": True,
        "dimensions": schema_contract_v2()["dimensions"],
    }
    model._checkpoint_cost_schema = {
        "schema_version": ROUND2_COST_SCHEMA_VERSION,
        "lambdas": list(CONGESTION_LAMBDAS),
        "completed_orders_delta_weight": 0.0,
        "trainable": False,
    }

    equality_failures = []
    migrated_state = model.state_dict()
    for name in inherited:
        if not torch.equal(migrated_state[name].cpu(), source_state[name].cpu()):
            equality_failures.append(name)
    if equality_failures:
        raise RuntimeError(
            "compatible parameter inheritance was not exact: "
            + ", ".join(equality_failures[:10])
        )

    congestion_only_cost_verified = bool(
        overridden == [OVERRIDDEN_COST_KEY]
        and torch.equal(
            model.cost_head.lambdas.detach().cpu(),
            torch.tensor(CONGESTION_LAMBDAS),
        )
        and not model.cost_head.lambdas.requires_grad
    )
    if not congestion_only_cost_verified:
        raise RuntimeError("migration did not establish the Round-2 cost contract")

    audit = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "source_checkpoint": {
            "path": source_path,
            "sha256": source_sha256,
            "action_schema": source_schema,
            "action_node_dim": 8,
            "action_global_dim": 6,
        },
        "target": {
            "action_schema": dict(model._checkpoint_action_schema),
            "model_config": config,
            "action_node_dim": ACTION_NODE_DIM_V2,
            "action_global_dim": ACTION_GLOBAL_DIM_V2,
            "parameter_shapes": {
                name: list(tensor.shape)
                for name, tensor in migrated_state.items()
            },
            "cost_schema": dict(model._checkpoint_cost_schema),
        },
        "policy": {
            "silent_shape_adaptation": False,
            "rebuilt_module_prefixes": [REBUILT_PREFIX],
            "overridden_parameter_keys": list(overridden),
            "action_encoder_init_seed": int(action_encoder_init_seed),
        },
        "inherited_parameter_keys": inherited,
        "rebuilt_parameter_keys": rebuilt,
        "overridden_parameter_keys": overridden,
        "inherited_parameter_count": len(inherited),
        "rebuilt_parameter_count": len(rebuilt),
        "overridden_parameter_count": len(overridden),
        "exact_inheritance_verified": True,
        "congestion_only_cost_verified": congestion_only_cost_verified,
        "initial_action_encoder_sha256": action_encoder_sha256_v2(model),
        "passed": True,
    }
    return model, audit


def build_v2_checkpoint_payload(
    model: RMFSWorldModel,
    *,
    migration_audit: Mapping,
    training_action_schema: Mapping,
    training_audit: Mapping,
) -> dict:
    """Build, but do not write, a schema-v2 checkpoint payload."""
    if migration_audit.get("schema_version") != MIGRATION_SCHEMA_VERSION:
        raise ValueError("missing explicit schema-v2 migration audit")
    if not bool(migration_audit.get("passed")):
        raise ValueError("schema-v2 migration audit did not pass")
    if (
        training_action_schema.get("schema_version")
        != DEFER_CONTEXT_ACTION_SCHEMA_VERSION
        or not bool(training_action_schema.get("passed"))
        or not bool(training_action_schema.get("supports_defer_context"))
        or int(training_action_schema.get("action_node_dim", -1))
        != ACTION_NODE_DIM_V2
        or int(training_action_schema.get("action_global_dim", -1))
        != ACTION_GLOBAL_DIM_V2
        or int(training_action_schema.get("action_edge_dim", -1)) != 4
    ):
        raise ValueError("training data did not pass schema-v2 action audit")
    training_checks = training_audit.get("checks") or {}
    if (
        training_audit.get("schema_version")
        != ROUND2_TRAINING_AUDIT_SCHEMA_VERSION
        or not bool(training_audit.get("passed"))
        or not bool(training_audit.get("action_encoder_trained"))
        or training_audit.get("ranking_cost_schema_version")
        != ROUND2_RANKING_COST_SCHEMA_VERSION
        or not training_checks
        or not all(training_checks.values())
    ):
        raise ValueError("missing passed Round-2 training audit")
    config = dict(migration_audit["target"]["model_config"])
    state = model.state_dict()
    expected_shapes = {
        name: tuple(shape)
        for name, shape in migration_audit["target"][
            "parameter_shapes"
        ].items()
    }
    if set(state) != set(expected_shapes):
        raise ValueError("model parameter names do not match migration target")
    shape_failures = [
        name
        for name, tensor in state.items()
        if tuple(tensor.shape) != expected_shapes[name]
    ]
    if shape_failures:
        raise ValueError(
            "model parameter shapes do not match migration target: "
            + ", ".join(shape_failures[:10])
        )
    if (
        model.action_encoder.node_net[0].in_features != ACTION_NODE_DIM_V2
        or model.action_encoder.global_net[0].in_features
        != ACTION_GLOBAL_DIM_V2
    ):
        raise ValueError("model does not implement the schema-v2 action encoder")
    expected_cost = torch.tensor(
        CONGESTION_LAMBDAS,
        dtype=model.cost_head.lambdas.dtype,
        device=model.cost_head.lambdas.device,
    )
    if not torch.equal(model.cost_head.lambdas.detach(), expected_cost):
        raise ValueError("model violates the congestion-only cost contract")
    if model.cost_head.lambdas.requires_grad:
        raise ValueError("Round-2 congestion-only cost must remain frozen")
    action_encoder_sha256 = action_encoder_sha256_v2(model)
    if action_encoder_sha256 == migration_audit.get(
        "initial_action_encoder_sha256"
    ):
        raise ValueError("refusing to package an untrained action encoder")
    if training_audit.get("trained_action_encoder_sha256") != (
        action_encoder_sha256
    ):
        raise ValueError("training audit is not bound to this action encoder")
    return {
        "state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in state.items()
        },
        "model_config": config,
        "label_schema_version": "wm_v4_local_pressure",
        "action_schema": dict(training_action_schema),
        "cost_schema": dict(migration_audit["target"]["cost_schema"]),
        "checkpoint_migration": dict(migration_audit),
        "training_audit": dict(training_audit),
    }


__all__ = [
    "MIGRATION_SCHEMA_VERSION",
    "ROUND2_COST_SCHEMA_VERSION",
    "ROUND2_TRAINING_AUDIT_SCHEMA_VERSION",
    "action_encoder_sha256_v2",
    "build_round2_training_audit_v2",
    "build_v2_checkpoint_payload",
    "migrate_round1_checkpoint_to_v2_model",
]
