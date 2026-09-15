from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from WorldModel.data.generate_long_risk_labels_sharded import partition_names
from WorldModel.core.model import LongRiskHead
from WorldModel.evaluation.phase_c_long_risk_head_repair_protocol import (
    DATA_SEEDS,
    LONG_RISK_HORIZON,
    LONG_RISK_TERMINAL_WINDOW,
    OFFLINE_TEST_SEEDS,
    ONLINE_TEST_SEEDS,
    OUTPUT_ROOT,
    PHASE_B_STAGE1_CHECKPOINT,
    REPAIRED_CHECKPOINT,
    SOURCE_CHECKPOINT,
    SOURCE_PSI_HEAD,
    TRAIN_SEEDS,
    VAL_SEEDS,
    formal_protocol,
)
from WorldModel.training.rebind_station_congestion_head_checkpoint import rebind
from WorldModel.training.train import long_risk_loss
from WorldModel.training.train_long_risk_head_only import (
    _predict,
    _tensor_audit,
    vectorized_long_risk_loss,
)


class LongRiskHeadRepairProtocolTests(unittest.TestCase):
    def test_seed_roles_are_disjoint_and_complete(self):
        self.assertEqual(
            set(DATA_SEEDS),
            set(TRAIN_SEEDS) | set(VAL_SEEDS) | set(OFFLINE_TEST_SEEDS),
        )
        self.assertFalse(set(TRAIN_SEEDS) & set(VAL_SEEDS))
        self.assertFalse(set(DATA_SEEDS) & set(ONLINE_TEST_SEEDS))

    def test_protocol_freezes_head_only_w200_repair(self):
        protocol = formal_protocol()
        self.assertEqual(protocol["lineage"]["trainable_prefixes"], ["long_risk_head."])
        self.assertEqual(protocol["data"]["long_risk_horizon"], LONG_RISK_HORIZON)
        self.assertEqual(
            protocol["data"]["terminal_window"], LONG_RISK_TERMINAL_WINDOW
        )
        self.assertTrue(protocol["forbidden"]["joint_encoder_or_transition_finetuning"])
        self.assertNotIn("pibt", OUTPUT_ROOT.as_posix().lower())
        self.assertNotEqual(SOURCE_CHECKPOINT, REPAIRED_CHECKPOINT)

    def test_shards_exactly_partition_snapshot_names(self):
        names = [f"snapshot_tick{tick}.pkl" for tick in range(17)]
        shards = partition_names(names, 4)
        flattened = [name for shard in shards for name in shard]
        self.assertEqual(set(flattened), set(names))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(all(shard for shard in shards))

    def test_vectorized_loss_matches_scalar_contract(self):
        prediction = torch.tensor(
            [0.2, 0.4, 0.1, 0.3, -0.2, 0.7], requires_grad=True
        )
        labels = {
            "risk_peak": torch.tensor(0.5),
            "risk_cvar": torch.tensor(0.35),
            "risk_terminal": torch.tensor(0.25),
            "risk_delta_group": torch.tensor(-0.1),
            "risk_event": torch.tensor(1.0),
        }
        target = torch.tensor([[0.5, 0.5, 0.35, 0.25, -0.1, 1.0]])
        scalar = long_risk_loss(prediction, labels)
        vectorized = vectorized_long_risk_loss(prediction.unsqueeze(0), target)
        self.assertTrue(torch.allclose(scalar, vectorized, atol=1e-7, rtol=0.0))

    def test_prepooled_feature_path_calls_long_risk_network(self):
        head = LongRiskHead(latent_dim=64)
        features = torch.randn(5, 64 * 4)
        prediction = _predict(head, features, torch.device("cpu"), batch_size=2)
        self.assertEqual(tuple(prediction.shape), (5, 6))

    def test_planner_specific_audit_allows_trained_source_head_only_by_opt_in(self):
        head_state = {
            "long_risk_head.net.0.weight": torch.zeros(2, 2),
            "long_risk_head.net.0.bias": torch.zeros(2),
            "long_risk_head.net.2.weight": torch.zeros(2, 2),
            "long_risk_head.net.2.bias": torch.zeros(2),
            "long_risk_head.net.4.weight": torch.zeros(6, 2),
            "long_risk_head.net.4.bias": torch.zeros(6),
        }
        stage1 = {
            "state_dict": {
                **head_state,
                "encoder.weight": torch.zeros(2, 2),
                "cost_head.weight": torch.zeros(1, 2),
            },
            "model_config": {"latent_dim": 2},
            "action_schema": {"version": 1},
        }
        source = copy.deepcopy(stage1)
        source["state_dict"]["encoder.weight"] += 1.0
        for key in head_state:
            source["state_dict"][key] += 0.5
        target = copy.deepcopy(source)
        for key in head_state:
            target["state_dict"][key] += 0.25

        default_audit = _tensor_audit(
            stage1_payload=stage1,
            source_payload=source,
            target_payload=target,
        )
        self.assertFalse(default_audit["passed"])

        planner_audit = _tensor_audit(
            stage1_payload=stage1,
            source_payload=source,
            target_payload=target,
            planner_specific_source=True,
        )
        self.assertTrue(planner_audit["passed"])
        self.assertTrue(planner_audit["planner_specific_source"])
        self.assertFalse(planner_audit["source_head_matches_stage1"])
        self.assertTrue(
            planner_audit["checks"]["all_non_long_risk_tensors_bitwise_equal"]
        )

    @unittest.skipUnless(
        SOURCE_CHECKPOINT.is_file() and PHASE_B_STAGE1_CHECKPOINT.is_file(),
        "formal checkpoints are not present",
    )
    def test_repository_lineage_matches_reported_failure(self):
        stage1 = torch.load(
            PHASE_B_STAGE1_CHECKPOINT, map_location="cpu", weights_only=False
        )
        source = torch.load(
            SOURCE_CHECKPOINT, map_location="cpu", weights_only=False
        )
        target = copy.deepcopy(source)
        for key, value in target["state_dict"].items():
            if key.startswith("long_risk_head."):
                target["state_dict"][key] = value + 0.125
        audit = _tensor_audit(
            stage1_payload=stage1,
            source_payload=source,
            target_payload=target,
        )
        self.assertTrue(
            audit["checks"]["source_phasec_head_equals_stage1_initialization"]
        )
        self.assertTrue(
            audit["checks"]["source_phasec_non_head_training_occurred"]
        )
        self.assertTrue(audit["passed"])

    @unittest.skipUnless(
        SOURCE_CHECKPOINT.is_file() and SOURCE_PSI_HEAD.is_file(),
        "formal source artifacts are not present",
    )
    def test_station_head_rebind_requires_tensor_equivalence(self):
        source = torch.load(
            SOURCE_CHECKPOINT, map_location="cpu", weights_only=False
        )
        target = copy.deepcopy(source)
        for key, value in target["state_dict"].items():
            if key.startswith("long_risk_head."):
                target["state_dict"][key] = value + 0.125
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target_path = root / "target.pt"
            torch.save(target, target_path)
            output = root / "rebound"
            summary = rebind(
                source_head=SOURCE_PSI_HEAD,
                source_world_model=SOURCE_CHECKPOINT,
                target_world_model=target_path,
                output_root=output,
            )
            self.assertTrue(summary["audit"]["passed"])
            rebound = torch.load(
                output / "best_station_congestion_head.pt",
                map_location="cpu",
                weights_only=False,
            )
            original = torch.load(
                SOURCE_PSI_HEAD, map_location="cpu", weights_only=False
            )
            for key in original["state_dict"]:
                self.assertTrue(
                    torch.equal(
                        original["state_dict"][key], rebound["state_dict"][key]
                    )
                )


if __name__ == "__main__":
    unittest.main()
