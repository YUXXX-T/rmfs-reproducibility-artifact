from __future__ import annotations

import argparse
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from WorldModel.data.build_pibt_station_congestion_head_dataset import (
    load_verified_tick_snapshot,
)
from WorldModel.evaluation import run_phase_c_pibt_planner_study as study
from WorldModel.evaluation.phase_c_pibt_planner_study_protocol import (
    ADAPTED_NEW_ARMS,
    ACTION_PATH_MODE,
    ACTION_ROUTE_ENCODING,
    ALL_ARMS,
    EVAL_SEEDS,
    LOADS,
    PLANNER_NAME,
    PLANNER_PARAMS,
    TRAIN_SEEDS,
    TRAIN_SPLIT,
    ZERO_SHOT_ARMS,
    evaluation_protocol,
    training_protocol,
)
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_manifest(path: Path, load: str, seed: int) -> None:
    payload = {
        "schema_version": "layer5_order_arrival_manifest_v1",
        "orders": [
            {
                "tick": int(seed),
                "order_id": int(seed),
                "load": load,
            }
        ],
        "total_orders": 1,
    }
    payload["manifest_sha256"] = study.canonical_sha256({
        "schema_version": payload["schema_version"],
        "orders": payload["orders"],
    })
    _write_json(path, payload)


class PIBTPlanner:
    supports_batch_planning = True
    is_single_step_planner = True
    strict_validation = True


def _build_manifest_contract(root: Path) -> Path:
    high_root = root / "high_source"
    low_mid_root = root / "low_mid_source"
    for load in LOADS:
        source = high_root if load == "high" else low_mid_root
        for seed in EVAL_SEEDS:
            _write_manifest(
                source
                / "order_manifests"
                / f"orders_{load}_seed{seed}.json",
                load,
                seed,
            )
    output_root = root / "evaluation"
    args = argparse.Namespace(
        high_manifest_root=str(high_root),
        low_mid_manifest_root=str(low_mid_root),
        output_root=str(output_root),
        manifest_contract=None,
    )
    study._audit_manifests(args)
    return study._manifest_contract_path(output_root)


class PhaseCPIBTPlannerStudyTests(unittest.TestCase):
    def test_pibt_station_head_snapshot_dedup_verifies_same_tick_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = {
                "training_source_policy": "world_model_on_policy",
                "external_baseline_training_samples": False,
                "load": "high",
                "seed": 461,
                "decision_tick": 20,
                "path_planner_override": PLANNER_NAME,
                "path_planner_params_override": PLANNER_PARAMS,
                "path_planner_state": PIBTPlanner(),
                "world_snapshot": SimpleNamespace(tick=20),
                "task_next_id": 4,
                "order_next_id": 5,
                "node_history": torch.zeros(4, 3, 10),
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
                "edge_features": torch.zeros(2, 6),
                "demand_context": torch.zeros(7),
                "station_node_ids": torch.tensor([1]),
            }
            paths = []
            for offset in range(2):
                path = root / f"context_{offset}.pkl"
                with path.open("wb") as handle:
                    pickle.dump(dict(base, candidate_group_id=f"g{offset}"), handle)
                paths.append(path)
            snapshot, duplicates = load_verified_tick_snapshot(
                paths, expected_load="high", expected_seed=461
            )
            self.assertEqual(snapshot["decision_tick"], 20)
            self.assertEqual(duplicates, 1)

            corrupted = dict(base, edge_features=torch.ones(2, 6))
            with paths[1].open("wb") as handle:
                pickle.dump(corrupted, handle)
            with self.assertRaisesRegex(ValueError, "edge_features"):
                load_verified_tick_snapshot(
                    paths, expected_load="high", expected_seed=461
                )

    def test_protocol_separates_zero_shot_and_pibt_adaptation(self):
        self.assertTrue(set(ZERO_SHOT_ARMS).isdisjoint(ADAPTED_NEW_ARMS))
        self.assertEqual(ALL_ARMS, ZERO_SHOT_ARMS + ADAPTED_NEW_ARMS)
        self.assertEqual(len(ZERO_SHOT_ARMS), 4)
        self.assertEqual(len(ADAPTED_NEW_ARMS), 2)

        zero = evaluation_protocol(
            "zero_shot", Path(study.PP_TRAINED_CHECKPOINT)
        )
        adapted = evaluation_protocol(
            "adapted",
            study.PIBT_TRAINED_CHECKPOINT
            if study.PIBT_TRAINED_CHECKPOINT.is_file()
            else Path(study.PP_TRAINED_CHECKPOINT),
        )
        self.assertEqual(zero["planner"]["name"], PLANNER_NAME)
        self.assertTrue(zero["planner"]["joint_one_step_batch_required"])
        self.assertEqual(zero["new_simulation_count"], 120)
        self.assertEqual(adapted["new_simulation_count"], 60)
        self.assertEqual(
            zero["station_admission"]["mode"],
            STATION_ADMISSION_PHYSICAL_ONLY,
        )
        self.assertEqual(
            zero["action_route_encoding"]["action_path_mode"],
            ACTION_PATH_MODE,
        )
        self.assertEqual(
            zero["action_route_encoding"]["encoding"],
            ACTION_ROUTE_ENCODING,
        )

        training = training_protocol()
        self.assertEqual(training["planner"]["name"], PLANNER_NAME)
        self.assertEqual(training["planner"]["params"], PLANNER_PARAMS)
        self.assertEqual(training["snapshot_run_count"], 30)
        self.assertEqual(training["isolated_replay_run_count"], 30)
        self.assertEqual(
            training["split"],
            {key: list(values) for key, values in TRAIN_SPLIT.items()},
        )
        self.assertTrue(training["long_risk_labels"]["required"])
        self.assertTrue(training["station_context_head"]["planner_specific"])
        self.assertIn("planner-specific", training["interpretation_limit"])

    def test_manifest_contract_and_frozen_bundle_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract_path = _build_manifest_contract(root)
            contract = study._load_manifest_contract(contract_path)
            self.assertEqual(len(contract["cells"]), 30)

            checkpoint = root / "checkpoint.pt"
            psi_head = root / "psi.pt"
            psi_scale = root / "psi_scale.json"
            source = root / "source.py"
            for path, content in (
                (checkpoint, b"checkpoint"),
                (psi_head, b"psi"),
                (psi_scale, b"{}"),
                (source, b"source"),
            ):
                path.write_bytes(content)

            output_root = root / "frozen_eval"
            args = argparse.Namespace(
                output_root=str(output_root),
                checkpoint=str(checkpoint),
                campaign="zero_shot",
                source_zero_shot_root=str(root / "unused"),
                manifest_contract=str(contract_path),
            )
            protocol_stub = lambda campaign, checkpoint_path: {
                "campaign": campaign,
                "checkpoint": Path(checkpoint_path).as_posix(),
                "metric_semantics": {},
            }
            with (
                patch.object(study, "PP_TRAINED_CHECKPOINT", checkpoint),
                patch.object(
                    study,
                    "psi_artifacts_for_arm",
                    return_value=(psi_head, psi_scale),
                ),
                patch.object(study, "_validate_psi_artifacts", return_value={}),
                patch.object(study, "SOURCE_FILES", {"source": source}),
                patch.object(study, "evaluation_protocol", protocol_stub),
            ):
                study._freeze_evaluation(args)
                bundle_path = study._eval_bundle_path(output_root)
                bundle = study._verify_bundle(
                    bundle_path, expected_campaign="zero_shot"
                )
                self.assertEqual(
                    study._bundle_manifest_contract(bundle)["contract_sha256"],
                    contract["contract_sha256"],
                )
                source.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    study._verify_bundle(
                        bundle_path, expected_campaign="zero_shot"
                    )

    def test_runtime_audit_rejects_sequential_fallback_or_event_logit(self):
        manifest = {"manifest_sha256": "manifest", "total_orders": 100}
        station_audit = {
            "passed": True,
            "mode": STATION_ADMISSION_PHYSICAL_ONLY,
            "physical_capacity_violation_count": 0,
        }
        metrics = {
            "order_arrival_replayed": True,
            "order_arrival_manifest_sha256": "manifest",
            "order_arrival_count": 100,
            "path_planner_name": PLANNER_NAME,
            "path_planner_batch_interface": True,
            "path_planner_single_step": True,
            "pibt_strict_validation": True,
            "pibt_batch_calls": 10,
            "pibt_single_calls": 0,
            "pibt_planned_agents": 20,
            "pibt_move_decisions": 12,
            "pibt_wait_decisions": 8,
            "pibt_validation_failures": 0,
            "pibt_last_batch_audit": {"passed": True},
            "path_planner_engine_vertex_conflicts": 0,
            "path_planner_engine_swap_conflicts": 0,
            "completed_orders": 40,
            "avg_task_duration": 12.0,
            "deadlock_ratio_mean": 0.1,
            "model_assign_calls": 10,
            "fallback_greedy_calls": 0,
            "action_path_mode": ACTION_PATH_MODE,
            "action_route_encoding": ACTION_ROUTE_ENCODING,
            "world_model_path_planner_injected": True,
            "energy_drift_signal": "combo",
            "psi_dispatch_mode": "j_ascending",
            "psi_dispatch_head_loaded": True,
            "psi_dispatch_encoder_contract_verified": True,
            "psi_dispatch_source_encoder_checkpoint_sha256": study.sha256_file(
                study.PP_TRAINED_CHECKPOINT
            ),
        }
        accepted = study._run_audit(
            "combo_ppwm_pibt", metrics, manifest, station_audit
        )
        self.assertTrue(accepted["passed"])

        sequential = dict(metrics, pibt_single_calls=1)
        self.assertFalse(
            study._run_audit(
                "combo_ppwm_pibt", sequential, manifest, station_audit
            )["passed"]
        )
        event_logit = dict(metrics, energy_drift_signal="event_logit")
        self.assertFalse(
            study._run_audit(
                "combo_ppwm_pibt", event_logit, manifest, station_audit
            )["passed"]
        )

    def test_paired_report_clusters_the_three_loads_by_seed(self):
        rows = {}
        for load in LOADS:
            for seed in EVAL_SEEDS:
                rows[(load, seed)] = {
                    "baseline": {"completed_orders": float(seed)},
                    "candidate": {"completed_orders": float(seed + 2)},
                }
        report, csv_rows = study._paired_report(
            rows,
            "baseline",
            "candidate",
            "completed_orders",
            1234,
        )
        self.assertEqual(report["overall_seed_cluster"]["n"], 10)
        self.assertEqual(report["overall_seed_cluster"]["mean"], 2.0)
        self.assertEqual(len(csv_rows), 4)

    def test_complete_training_audit_checks_planner_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage1 = root / "stage1.pt"
            torch.save(
                {
                    "state_dict": {
                        "long_risk_head.net.0.weight": torch.zeros(1, 1),
                    }
                },
                stage1,
            )
            model_root = root / "model_round1_v1"
            trained = model_root / "best_regret_world_model.pt"

            training_spec = {
                "stage1_checkpoint": stage1.as_posix(),
                "stage2_horizon": 10,
                "stage2_epochs": 30,
                "stage2_lr": 1e-4,
                "stage2_alpha_rank": 0.1,
                "stage2_freeze_epochs": 0,
                "stage2_unfreeze_mode": "all",
                "early_stopping_patience": 8,
                "early_stopping_monitor": "val_top1_regret_mean",
            }
            protocol = {
                "campaign": "pibt_on_policy_round1",
                "training": training_spec,
            }
            bundle = {
                "schema_version": study.BUNDLE_SCHEMA_VERSION,
                "protocol": protocol,
                "protocol_sha256": study.canonical_sha256(protocol),
                "artifacts": {},
            }
            bundle_path = study._training_bundle_path(root)
            _write_json(bundle_path, bundle)

            snapshot_root = root / "snapshots_461_470"
            dataset_root = root / "datasets_461_470"
            long_risk_root = root / "long_risk_w200_461_470"
            (snapshot_root / "snapshot_collection_outputs.sha256").parent.mkdir(
                parents=True, exist_ok=True
            )
            (snapshot_root / "snapshot_collection_outputs.sha256").write_text(
                "", encoding="utf-8"
            )
            (dataset_root / "replayed_datasets.sha256").parent.mkdir(
                parents=True, exist_ok=True
            )
            (dataset_root / "replayed_datasets.sha256").write_text(
                "", encoding="utf-8"
            )
            (long_risk_root / "long_risk_outputs.sha256").parent.mkdir(
                parents=True, exist_ok=True
            )
            (long_risk_root / "long_risk_outputs.sha256").write_text(
                "", encoding="utf-8"
            )

            stage1_sha = study.sha256_file(stage1)
            for load in LOADS:
                collection_protocol = {
                    "protocol_sha256": f"collector-{load}",
                    "load": load,
                    "seeds": list(TRAIN_SEEDS),
                    "path_planner_override": PLANNER_NAME,
                    "path_planner_params_override": PLANNER_PARAMS,
                    "behavior_checkpoint_sha256": stage1_sha,
                }
                _write_json(
                    snapshot_root / f"collection_{load}.json",
                    {
                        "protocol": collection_protocol,
                        "load": load,
                        "per_seed": {str(seed): {} for seed in TRAIN_SEEDS},
                    },
                )
                for seed in TRAIN_SEEDS:
                    run_root = snapshot_root / load / f"seed{seed}"
                    metrics = {
                        "path_planner_name": PLANNER_NAME,
                        "path_planner_batch_interface": True,
                        "path_planner_single_step": True,
                        "pibt_strict_validation": True,
                        "pibt_batch_calls": 5,
                        "pibt_single_calls": 0,
                        "pibt_planned_agents": 8,
                        "pibt_move_decisions": 5,
                        "pibt_wait_decisions": 3,
                        "pibt_validation_failures": 0,
                        "pibt_last_batch_audit": {"passed": True},
                        "path_planner_engine_vertex_conflicts": 0,
                        "path_planner_engine_swap_conflicts": 0,
                    }
                    _write_json(
                        run_root / "metrics.json",
                        {
                            "protocol_sha256": f"collector-{load}",
                            "load": load,
                            "seed": seed,
                            "metrics": metrics,
                        },
                    )
                    _write_json(
                        run_root
                        / "snapshots"
                        / f"snapindex_{load}_{seed}.json",
                        {
                            "load": load,
                            "seed": seed,
                            "path_planner_override": PLANNER_NAME,
                            "path_planner_params_override": PLANNER_PARAMS,
                        },
                    )
                    data_root = dataset_root / load / f"seed{seed}"
                    data_root.mkdir(parents=True, exist_ok=True)
                    (data_root / "data.pt").write_bytes(b"data")
                    _write_json(
                        data_root / "gen_config.json",
                        {
                            "load_level": load,
                            "seed": seed,
                            "path_planner_override": PLANNER_NAME,
                            "path_planner_params_override": PLANNER_PARAMS,
                            "rollout_continuation_mode": "isolated",
                            "training_source_policy": "world_model_on_policy",
                            "external_baseline_training_samples": False,
                            "action_path_mode": ACTION_PATH_MODE,
                            "action_route_encoding": ACTION_ROUTE_ENCODING,
                            "action_path_planner_preview_used": False,
                        },
                    )
                    _write_json(
                        data_root / "data_meta.json",
                        {"audit": {"passed": True}},
                    )
                    long_risk_path = (
                        long_risk_root
                        / load
                        / f"seed{seed}"
                        / "long_risk_labels.pt"
                    )
                    long_risk_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        [
                            {
                                "candidate_key": (
                                    f"g_{load}_{seed}_r1_o1_p1_s1_t20"
                                ),
                                "long_risk_horizon": 200,
                                "long_risk_terminal_window": 50,
                                "continuation_policy": "greedy",
                                "source_path_planner": PLANNER_NAME,
                                "source_path_planner_params": PLANNER_PARAMS,
                            },
                            {
                                "candidate_key": (
                                    f"g_{load}_{seed}_rNO_ASSIGN_o1_p1_s1_t20"
                                ),
                                "long_risk_horizon": 200,
                                "long_risk_terminal_window": 50,
                                "continuation_policy": "greedy",
                                "source_path_planner": PLANNER_NAME,
                                "source_path_planner_params": PLANNER_PARAMS,
                            },
                        ],
                        long_risk_path,
                    )

            fused = root / "fused_seed_split_v1"
            fused.mkdir(parents=True, exist_ok=True)
            (fused / "phase_c_round1_fused.pt").write_bytes(b"fused")
            _write_json(
                fused / "splits.json",
                {
                    "split_unit": "source_seed",
                    "explicit_seeds": {
                        key: [str(seed) for seed in values]
                        for key, values in TRAIN_SPLIT.items()
                    },
                },
            )
            _write_json(fused / "manifest.json", {})
            _write_json(
                fused / "quality_report.json",
                {
                    "total_long_risk_samples": 60,
                    "total_long_risk_groups": 30,
                },
            )
            (fused / "validated_outputs.sha256").write_text("", encoding="utf-8")

            model_root.mkdir(parents=True, exist_ok=True)
            action_schema = {
                "supports_no_assign_candidate": True,
                "complete_group_coverage": True,
                "zero_encoding_verified": True,
            }
            torch.save(
                {"action_schema": action_schema}, model_root / "world_model.pt"
            )
            (model_root / "best_rank_world_model.pt").write_bytes(b"rank")
            torch.save(
                {
                    "state_dict": {
                        "long_risk_head.net.0.weight": torch.ones(1, 1),
                    }
                },
                model_root / "best_regret_world_model.pt",
            )
            _write_json(
                model_root / "train_summary.json",
                {
                    "stage1": {
                        "skipped": True,
                        "checkpoint": stage1.as_posix(),
                    },
                    "stage2": {
                        "horizon": 10,
                        "epochs": 30,
                        "lr": 1e-4,
                        "alpha_rank": 0.1,
                        "freeze_epochs": 0,
                        "unfreeze_mode": "all",
                        "early_stopping_patience": 8,
                        "early_stopping_monitor": "val_top1_regret_mean",
                        "val_long_risk_eval": {"risk_peak": {"mae": 0.1}},
                    },
                    "action_schema": action_schema,
                },
            )
            (model_root / "trained_outputs.sha256").write_text(
                "", encoding="utf-8"
            )

            station_root = (
                root / "station_congestion_head_region_pibt_461_470_v1"
            )
            station_root.mkdir(parents=True, exist_ok=True)
            (station_root / "station_congestion_latents.pt").write_bytes(b"data")
            pp_scale = root / "pp_station_congestion_scale_contract.json"
            _write_json(pp_scale, {"contract_sha256": "scale"})
            _write_json(
                station_root / "station_congestion_scale_contract.json",
                {"contract_sha256": "scale"},
            )
            _write_json(
                station_root / "dataset_summary.json",
                {
                    "audit": {
                        "passed": True,
                        "pibt_snapshot_provenance_verified": True,
                    }
                },
            )
            linear_root = station_root / "linear_head_v1"
            linear_root.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "source_encoder_checkpoint_sha256": study.sha256_file(
                        trained
                    ),
                    "scale_contract_sha256": "scale",
                    "scale_contract": {"contract_sha256": "scale"},
                    "audit": {"formal_training_contract": True},
                },
                linear_root / "best_station_congestion_head.pt",
            )
            _write_json(linear_root / "train_summary.json", {})

            args = argparse.Namespace(
                output_root=str(root),
                frozen_bundle=str(bundle_path),
                training_stage="complete",
            )
            with (
                patch.object(study, "PHASE_B_STAGE1_CHECKPOINT", stage1),
                patch.object(study, "PIBT_TRAINED_CHECKPOINT", trained),
                patch.object(study, "PSI_SCALE_CONTRACT", pp_scale),
            ):
                study._audit_training(args)
            report = json.loads(
                (root / "validation" / "training_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["snapshot_runs"], 30)
            self.assertEqual(report["dataset_runs"], 30)


if __name__ == "__main__":
    unittest.main()
