from __future__ import annotations

from pathlib import Path
import json

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_training_contracts_freeze_greedy_continuation() -> None:
    world_model = yaml.safe_load(
        (ROOT / "configs/training/world_model.yaml").read_text(encoding="utf-8")
    )
    j1_dispatch = yaml.safe_load(
        (ROOT / "configs/training/j1_dispatch.yaml").read_text(encoding="utf-8")
    )
    long_risk = world_model["long_risk"]
    assert long_risk["continuation_policy"] == "frozen_greedy"
    assert long_risk["simulator_continuation_horizon"] == 200
    assert long_risk["jointly_optimized_with_world_model"] is True
    assert long_risk["separate_head_only_training_stage"] is False
    assert long_risk["stage2_loss_weight"] > 0
    assert world_model["canonical_trainer"] == "WorldModel.training.run_train_v6"
    assert j1_dispatch["role"] == "dispatch_auxiliary"
    assert j1_dispatch["world_model_encoder"] == "frozen"
    assert j1_dispatch["runtime_score"] == (
        "0.5 * service + 0.5 * traffic - service_debt"
    )


def test_behavior_continuation_is_deepcopied_and_preserves_tick_order() -> None:
    source = (ROOT / "src/WorldModel/data/counterfactual_rollout.py").read_text(
        encoding="utf-8"
    )
    required = [
        "rollout_continuation_mode not in (\"isolated\", \"behavior\")",
        "copy.deepcopy(\n                    continuation_order_generator",
        "copy.deepcopy(\n                    continuation_task_assigner",
        "orders = order_generator.generate(world)",
        "tasks = task_assigner.assign(world)",
        "generate_orders=(k > 0)",
        "behavior continuation policies must be deepcopy-safe",
    ]
    assert all(fragment in source for fragment in required)
    assert source.index("orders = order_generator.generate(world)") < source.index(
        "tasks = task_assigner.assign(world)"
    )


def test_public_explanation_distinguishes_policy_from_physical_state() -> None:
    text = (ROOT / "docs/method/frozen_greedy_continuation.md").read_text(
        encoding="utf-8"
    )
    assert "policy and its parameters are fixed" in text
    assert "not that the physical state is frozen" in text


def test_long_risk_is_model_owned_and_canonical_cli_is_joint() -> None:
    model_source = (ROOT / "src/WorldModel/core/model.py").read_text(encoding="utf-8")
    trainer_source = (
        ROOT / "src/WorldModel/training/run_train_v6.py"
    ).read_text(encoding="utf-8")
    public_cli = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    assert "self.long_risk_head = LongRiskHead(hidden_dim)" in model_source
    assert 'train_modules = ["action_encoder", "transition", "long_risk_head"]' in trainer_source
    assert "alpha_long_risk=args.alpha_long_risk" in trainer_source
    assert '"world-model": "WorldModel.training.run_train_v6"' in public_cli
    assert '"long-risk"' not in public_cli
    assert "train_long_risk_head_only" not in public_cli


def test_training_scale_is_explicitly_stage_scoped() -> None:
    payload = json.loads(
        (ROOT / "artifacts/statistics/training_data_scale.json").read_text(
            encoding="utf-8"
        )
    )
    stages = payload["world_model_training_splits"]
    assert [stage["training_candidate_samples"] for stage in stages] == [7915, 10810]
    assert payload["j1_station_predictor_development_data"]["development_only"] is True
