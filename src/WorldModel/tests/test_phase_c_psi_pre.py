"""Unit tests for the isolated behavior-aligned psi_pre tooling."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from WorldModel.core.psi_pre_head import PsiPreHead
from WorldModel.data.build_phase_c_psi_pre_dataset import (
    _aggregate_node_values,
    _build_regions,
    _atomic_torch_save,
    fuse_shards,
)
from WorldModel.evaluation.phase_c_psi_pre_protocol import (
    FROZEN_BUNDLE_SCHEMA_VERSION,
    SHARD_SCHEMA_VERSION,
    TARGET_CHANNELS,
    canonical_sha256,
    fold_seed_splits,
)


class PsiPreHeadTests(unittest.TestCase):
    def test_variable_station_count_and_region_features(self):
        z = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 3, 4], [1, 2, 3, 4, 4, 5]], dtype=torch.long
        )
        regions = _build_regions(edge_index, [0, 5], 6, 1)
        representation = PsiPreHead.concat_station_region(z, [0, 5], regions)
        self.assertEqual(tuple(representation.shape), (2, 12))
        head = PsiPreHead(12, len(TARGET_CHANNELS))
        output = head(representation)
        self.assertEqual(tuple(output.shape), (2, len(TARGET_CHANNELS)))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(bool(((output >= 0.0) & (output <= 1.0)).all()))

    def test_region_aggregation_order_is_explicit(self):
        values = torch.zeros(4, 6)
        values[:, 1] = torch.tensor([0.1, 0.2, 0.3, 0.4])
        values[:, 2] = torch.tensor([0.5, 0.2, 0.1, 0.0])
        values[:, 3] = torch.tensor([0.0, 0.4, 0.2, 0.1])
        values[:, 5] = torch.tensor([0.7, 0.1, 0.3, 0.2])
        result = _aggregate_node_values(values, [[0, 1], [2, 3]])
        self.assertEqual(tuple(result.shape), (2, 8))
        self.assertAlmostEqual(float(result[0, 0]), 0.15, places=6)
        self.assertAlmostEqual(float(result[0, 1]), 0.20, places=6)
        self.assertAlmostEqual(float(result[0, 2]), 0.35, places=6)
        self.assertAlmostEqual(float(result[0, 3]), 0.50, places=6)
        self.assertAlmostEqual(float(result[0, 6]), 0.40, places=6)
        self.assertAlmostEqual(float(result[0, 7]), 0.70, places=6)

    def test_folds_are_whole_seed_and_cover_each_seed_once_as_test(self):
        folds = fold_seed_splits()
        self.assertEqual(len(folds), 5)
        test_seeds = [seed for fold in folds for seed in fold["test_seeds"]]
        self.assertEqual(sorted(test_seeds), list(range(531, 541)))
        for fold in folds:
            self.assertTrue(set(fold["train_seeds"]).isdisjoint(fold["val_seeds"]))
            self.assertTrue(set(fold["train_seeds"]).isdisjoint(fold["test_seeds"]))
            self.assertTrue(set(fold["val_seeds"]).isdisjoint(fold["test_seeds"]))
            self.assertNotIn(501, fold["train_seeds"] + fold["val_seeds"] + fold["test_seeds"])

    def test_fuse_globalises_frame_groups_and_balances_source_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out"
            model_path = root / "model.pt"
            model_path.write_bytes(b"model")
            protocol = {"name": "synthetic"}
            rows = []
            for run_index, seed in enumerate(range(531, 541)):
                for load_code, load in enumerate(("low", "mid", "high")):
                    run_id = f"{load}_seed{seed}"
                    source = root / f"{run_id}.source"
                    meta_source = root / f"{run_id}.json"
                    source.write_bytes(run_id.encode())
                    meta_source.write_text("{}", encoding="utf-8")
                    rows.append({
                        "run_id": run_id,
                        "load": load,
                        "seed": seed,
                        "data_path": source.name,
                        "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "meta_path": meta_source.name,
                        "meta_sha256": hashlib.sha256(meta_source.read_bytes()).hexdigest(),
                    })
                    shard = output / "shards" / f"{run_id}.pt"
                    payload = {
                        "schema_version": SHARD_SCHEMA_VERSION,
                        "protocol_sha256": canonical_sha256(protocol),
                        "source_sha256": rows[-1]["data_sha256"],
                        "run_id": run_id,
                        "load": load,
                        "seed": seed,
                        "target_channels": list(TARGET_CHANNELS),
                        "state_latents": torch.zeros(2, 6),
                        "rollout_latents": torch.zeros(2, 6),
                        "targets": torch.zeros(2, 10),
                        "baseline_state": torch.zeros(2, 10),
                        "baseline_h10": torch.zeros(2, 10),
                        "baseline_h1": torch.zeros(2, 10),
                        "frame_group": torch.tensor([0, 0]),
                        "candidate_group": torch.tensor([0, 0]),
                        "station_index": torch.tensor([0, 1]),
                        "station_node_id": torch.tensor([0, 1]),
                        "decision_tick": torch.tensor([25, 25]),
                    }
                    _atomic_torch_save(shard, payload)
                    shard_hash = hashlib.sha256(shard.read_bytes()).hexdigest()
                    (shard.with_suffix(".json")).write_text(json.dumps({
                        "schema_version": SHARD_SCHEMA_VERSION,
                        "protocol_sha256": canonical_sha256(protocol),
                        "source_sha256": rows[-1]["data_sha256"],
                        "shard_sha256": shard_hash,
                        "run_id": run_id,
                    }), encoding="utf-8")
            bundle = {
                "schema_version": FROZEN_BUNDLE_SCHEMA_VERSION,
                "protocol": protocol,
                "protocol_sha256": canonical_sha256(protocol),
                "artifacts": {
                    "model_checkpoint": {
                        "path": model_path.name,
                        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    },
                    "behavior_runs": rows,
                },
            }
            bundle_path = root / "bundle.json"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            summary = fuse_shards(
                bundle_path=bundle_path,
                repo_root=root,
                output_root=output,
            )
            dataset = torch.load(output / "psi_pre_latents.pt", map_location="cpu", weights_only=False)
            tensors = dataset["tensors"]
            self.assertEqual(int(summary["row_count"]), 60)
            self.assertEqual(int(torch.unique(tensors["frame_group"]).numel()), 30)
            for run in range(30):
                weight = tensors["row_weight"][tensors["run_index"] == run].sum()
                self.assertAlmostEqual(float(weight), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
