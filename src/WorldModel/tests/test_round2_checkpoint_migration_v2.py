import torch

from WorldModel.core.model import RMFSWorldModel
from WorldModel.round2.checkpoint_migration_v2 import (
    MIGRATION_SCHEMA_VERSION,
    ROUND2_TRAINING_AUDIT_SCHEMA_VERSION,
    action_encoder_sha256_v2,
    build_round2_training_audit_v2,
    build_v2_checkpoint_payload,
    migrate_round1_checkpoint_to_v2_model,
)


def _source_payload():
    config = {
        "node_feat_dim": 10,
        "edge_feat_dim": 6,
        "demand_dim": 7,
        "action_node_dim": 8,
        "action_global_dim": 6,
        "hidden_dim": 8,
        "num_spatial_layers": 3,
        "rollout_horizon": 3,
        "num_stations": 2,
    }
    model = RMFSWorldModel(**config)
    return {
        "state_dict": {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        },
        "model_config": config,
        "label_schema_version": "wm_v4_local_pressure",
        "action_schema": {
            "schema_version": "wm_native_no_assign_action_v1",
            "supports_no_assign_candidate": True,
            "no_assign_encoding": "zero_action_tensors_v1",
            "complete_group_coverage": True,
            "zero_encoding_verified": True,
        },
    }


def test_v2_migration_rebuilds_only_action_encoder_and_copies_rest_exactly():
    source = _source_payload()
    rng_before = torch.random.get_rng_state().clone()
    model, audit = migrate_round1_checkpoint_to_v2_model(
        source, action_encoder_init_seed=123
    )
    rng_after = torch.random.get_rng_state()

    assert torch.equal(rng_before, rng_after)
    assert model.action_encoder.node_net[0].in_features == 8
    assert model.action_encoder.global_net[0].in_features == 9
    assert audit["schema_version"] == MIGRATION_SCHEMA_VERSION
    assert audit["passed"]
    assert audit["exact_inheritance_verified"]
    assert audit["congestion_only_cost_verified"]
    assert audit["overridden_parameter_keys"] == ["cost_head.lambdas"]
    assert model.cost_head.lambdas.tolist() == [
        1.0, 0.5, 0.5, 0.5, 1.0, 0.0, 2.0
    ]
    assert not model.cost_head.lambdas.requires_grad
    assert audit["rebuilt_parameter_keys"]
    assert all(
        name.startswith("action_encoder.")
        for name in audit["rebuilt_parameter_keys"]
    )
    assert not any(
        name.startswith("action_encoder.")
        for name in audit["inherited_parameter_keys"]
    )
    migrated = model.state_dict()
    for name in audit["inherited_parameter_keys"]:
        assert torch.equal(migrated[name], source["state_dict"][name])


def test_v2_action_encoder_initialisation_is_reproducible():
    source = _source_payload()
    first, _ = migrate_round1_checkpoint_to_v2_model(
        source, action_encoder_init_seed=77
    )
    second, _ = migrate_round1_checkpoint_to_v2_model(
        source, action_encoder_init_seed=77
    )
    for name, value in first.action_encoder.state_dict().items():
        assert torch.equal(value, second.action_encoder.state_dict()[name])


def test_v2_checkpoint_payload_requires_passed_training_schema_audit():
    source = _source_payload()
    model, migration = migrate_round1_checkpoint_to_v2_model(source)
    action_schema = {
        "schema_version": "wm_defer_context_action_v2",
        "passed": True,
        "supports_defer_context": True,
        "action_node_dim": 8,
        "action_global_dim": 9,
        "action_edge_dim": 4,
    }
    with torch.no_grad():
        model.action_encoder.global_net[0].weight.add_(0.01)
    trained_digest = action_encoder_sha256_v2(model)
    pairwise_audit = {
        "schema_version": "wm_round2_assign_pairwise_v1",
        "passed": True,
        "checks": {
            "congestion_only_cost_contract": True,
            "uniform_pair_sgd_unbiased_scale": True,
        },
    }
    training_audit = build_round2_training_audit_v2(
        model,
        migration_audit=migration,
        training_action_schema=action_schema,
        pairwise_audit=pairwise_audit,
        optimizer_steps=3,
    )
    assert training_audit["trained_action_encoder_sha256"] == trained_digest
    payload = build_v2_checkpoint_payload(
        model,
        migration_audit=migration,
        training_action_schema=action_schema,
        training_audit=training_audit,
    )
    assert payload["model_config"]["action_global_dim"] == 9
    assert payload["action_schema"] == action_schema
    assert payload["checkpoint_migration"]["passed"]
    assert payload["training_audit"] == training_audit
    assert payload["cost_schema"]["completed_orders_delta_weight"] == 0.0


def test_v2_checkpoint_payload_rejects_untrained_or_wrong_shape_model():
    source = _source_payload()
    model, migration = migrate_round1_checkpoint_to_v2_model(source)
    action_schema = {
        "schema_version": "wm_defer_context_action_v2",
        "passed": True,
        "supports_defer_context": True,
        "action_node_dim": 8,
        "action_global_dim": 9,
        "action_edge_dim": 4,
    }
    audit = {
        "schema_version": ROUND2_TRAINING_AUDIT_SCHEMA_VERSION,
        "passed": True,
        "action_encoder_trained": True,
        "ranking_cost_schema_version": "wm_round2_h10_congestion_only_v1",
        "trained_action_encoder_sha256": action_encoder_sha256_v2(model),
        "checks": {"synthetic_training_complete": True},
    }
    try:
        build_v2_checkpoint_payload(
            model,
            migration_audit=migration,
            training_action_schema=action_schema,
            training_audit=audit,
        )
    except ValueError as exc:
        assert "untrained action encoder" in str(exc)
    else:
        raise AssertionError("payload accepted the random migrated encoder")

    wrong_model = RMFSWorldModel(**source["model_config"])
    with torch.no_grad():
        wrong_model.action_encoder.global_net[0].weight.add_(0.01)
    audit["trained_action_encoder_sha256"] = action_encoder_sha256_v2(
        wrong_model
    )
    try:
        build_v2_checkpoint_payload(
            wrong_model,
            migration_audit=migration,
            training_action_schema=action_schema,
            training_audit=audit,
        )
    except ValueError as exc:
        assert "shapes" in str(exc) or "schema-v2" in str(exc)
    else:
        raise AssertionError("payload accepted a six-global-channel model")


def test_v2_migration_refuses_implicit_dimension_adaptation():
    source = _source_payload()
    source["model_config"]["action_global_dim"] = 7
    try:
        migrate_round1_checkpoint_to_v2_model(source)
    except ValueError as exc:
        assert "must be 6" in str(exc)
    else:
        raise AssertionError("migration accepted an unknown source dimension")
