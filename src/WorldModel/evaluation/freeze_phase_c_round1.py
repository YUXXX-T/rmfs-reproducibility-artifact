"""Freeze the Phase C round-1 data-aggregation protocol before collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from WorldModel.evaluation.collect_phase_c_round1 import (
    FINAL_CERTIFICATION_SEEDS,
    FORMAL_MAX_CONTEXTS_PER_TICK,
    FORMAL_ROLLOUT_HORIZON,
    FORMAL_SEEDS,
    FORMAL_SNAPSHOT_INTERVAL,
    FORMAL_SNAPSHOT_TOP_M,
    FORMAL_TICKS,
    LOAD_CONFIGS,
)


SCHEMA_VERSION = "phase_c_round1_frozen_protocol_v1"
TRAIN_SEEDS = tuple(range(461, 468))
VALIDATION_SEEDS = (468, 469)
OFFLINE_TEST_SEEDS = (470,)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: dict) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lyapunov-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    checkpoint = Path(args.checkpoint)
    lyapunov_config = Path(args.lyapunov_config)
    source_paths = {
        "phase_c_collector": repo_root / (
            "WorldModel/evaluation/collect_phase_c_round1.py"
        ),
        "collection_orchestration": repo_root / (
            "WorldModel/evaluation/collect_phase_c_round1_461_470.sh"
        ),
        "decision_snapshot_probe": repo_root / (
            "WorldModel/evaluation/decision_snapshot_probe.py"
        ),
        "online_runner": repo_root / (
            "WorldModel/evaluation/evaluate_online_v6.py"
        ),
        "dataset_builder": repo_root / (
            "WorldModel/data/build_phase_c_round1_dataset.py"
        ),
        "replay_orchestration": repo_root / (
            "WorldModel/evaluation/replay_phase_c_round1_461_470.sh"
        ),
        "counterfactual_rollout": repo_root / (
            "WorldModel/data/counterfactual_rollout.py"
        ),
        "candidate_generator": repo_root / (
            "WorldModel/data/candidate_generator.py"
        ),
        "data_fusion": repo_root / "WorldModel/data/fuse_and_split.py",
        "fusion_orchestration": repo_root / (
            "WorldModel/evaluation/fuse_phase_c_round1_461_470.sh"
        ),
        "training_entry": repo_root / (
            "WorldModel/evaluation/train_phase_c_round1.sh"
        ),
        "training_implementation": repo_root / (
            "WorldModel/training/run_train_v6.py"
        ),
        "world_model_assigner": repo_root / (
            "Policies/TaskAssigner/WorldModelTaskAssigner/"
            "world_model_task_assigner.py"
        ),
        "shared_context_commit": repo_root / (
            "Policies/TaskAssigner/context_assignment.py"
        ),
    }
    required = [checkpoint, lyapunov_config]
    required.extend(Path(path) for path in LOAD_CONFIGS.values())
    required.extend(source_paths.values())
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "round": "phase_c_round1",
        "mainline": (
            "pure_world_model_on_policy_state_aggregation_with_isolated_"
            "robot_or_native_no_assign_counterfactual_labels"
        ),
        "behavior_checkpoint": checkpoint.as_posix(),
        "behavior_checkpoint_sha256": _sha256_file(checkpoint),
        "loads": list(LOAD_CONFIGS),
        "load_configs": {
            load: {
                "path": Path(path).as_posix(),
                "sha256": _sha256_file(Path(path)),
            }
            for load, path in LOAD_CONFIGS.items()
        },
        "normal_arrivals": True,
        "collection_seeds": list(FORMAL_SEEDS),
        "train_seeds": list(TRAIN_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "offline_test_seeds": list(OFFLINE_TEST_SEEDS),
        "reserved_final_online_certification_seeds": list(
            FINAL_CERTIFICATION_SEEDS
        ),
        "whole_seed_split_required": True,
        "ticks": FORMAL_TICKS,
        "snapshot_interval": FORMAL_SNAPSHOT_INTERVAL,
        "snapshot_top_m_robot_candidates": FORMAL_SNAPSHOT_TOP_M,
        "snapshot_top_m_is_online_limit": False,
        "max_contexts_per_tick": FORMAL_MAX_CONTEXTS_PER_TICK,
        "candidate_robot_mode": "stratified",
        "candidate_actions": "sampled_idle_robots_plus_native_no_assign",
        "rollout_horizon": FORMAL_ROLLOUT_HORIZON,
        "rollout_continuation_mode": "isolated",
        "future_unknown_orders_in_rollout": False,
        "continuation_scheduler_in_rollout": False,
        "training_source_policy": "world_model_on_policy",
        "external_baseline_training_samples": False,
        "hungarian_training_samples": False,
        "greedy_training_samples": False,
        "hungarian_or_greedy_online_inference": False,
        "td_target_enabled": False,
        "td_value_head_enabled": False,
        "td_risk_v_head_enabled": False,
        "learned_work_drift_head_primary": False,
        "lyapunov_role": "analytic_interpretable_auxiliary_and_audit_label",
        "training": {
            "mode": "fine_tune_existing_world_model_only",
            "stage1_checkpoint": checkpoint.as_posix(),
            "horizon": 10,
            "epochs": 30,
            "learning_rate": 1e-4,
            "ranking_alpha": 0.1,
            "freeze_epochs": 0,
            "unfreeze_mode": "all",
            "train_cost_head": False,
            "balanced_sampler": "source_run_id",
            "pair_balance_by": "source_load_level",
            "early_stopping_monitor": "val_top1_regret_mean",
            "early_stopping_patience": 8,
        },
        "lyapunov_config": lyapunov_config.as_posix(),
        "lyapunov_config_sha256": _sha256_file(lyapunov_config),
        "source_sha256": {
            name: _sha256_file(path) for name, path in source_paths.items()
        },
    }
    protocol["protocol_sha256"] = _canonical_sha256(protocol)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(protocol, indent=2, ensure_ascii=False) + "\n"
    if output.exists():
        existing = output.read_text(encoding="utf-8")
        if existing != encoded:
            raise RuntimeError(
                f"refusing to overwrite changed frozen protocol: {output}"
            )
        print(f"[audit] frozen Phase C protocol unchanged: {output}")
    else:
        output.write_text(encoded, encoding="utf-8")
        print(f"[freeze] wrote {output}")
    print(f"Phase C round-1 protocol sha256 = {protocol['protocol_sha256']}")
    print(f"collection seeds = {list(FORMAL_SEEDS)}")
    print(f"final certification seeds = {list(FINAL_CERTIFICATION_SEEDS)}")


if __name__ == "__main__":
    main()
