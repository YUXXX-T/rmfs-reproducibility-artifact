import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from WorldModel.core.station_congestion_head import (
    CHANNEL_NAMES,
    COMPONENT_NAMES,
    StationCongestionHead,
    build_station_targets,
    extract_station_components,
    fit_scale_contract,
    verify_scale_contract,
)
from WorldModel.core.model import RMFSWorldModel
from WorldModel.data.build_station_congestion_head_dataset import (
    DATASET_SCHEMA_VERSION,
    REPRESENTATION_STATION_REGION_MEAN_MAX,
    align_station_node_ids,
    build_dataset,
    build_station_region_node_ids,
    build_station_representations,
    sha256_file,
)
from WorldModel.training.train_station_congestion_head import train


def _station_row(**overrides):
    row = {
        "station_id": 7,
        "station_seed_nodes": [11, 12, 13],
        "assigned_agent_capacity_ratio": 0.6,
        "assigned_agent_work_ratio": 99.0,
        "in_progress_pressure": 0.4,
        "active_task_pressure": 88.0,
        "regions": {
            "h3": {
                "stationary_ticks_max": 12.0,
                "node_density_mean": 0.3,
                "node_density_cvar90": 0.8,
                "node_bottleneck_weighted_density": 0.65,
            }
        },
    }
    row.update(overrides)
    return row


class StationCongestionTargetTest(unittest.TestCase):
    def test_selected_components_and_bottleneck_excess(self):
        components = extract_station_components(_station_row())
        self.assertEqual(tuple(components), COMPONENT_NAMES)
        self.assertAlmostEqual(components["stationary_ticks_max"], 12.0)
        self.assertAlmostEqual(components["node_density_cvar90"], 0.8)
        self.assertAlmostEqual(components["bottleneck_density_excess"], 0.35)
        self.assertAlmostEqual(
            components["assigned_agent_capacity_ratio"], 0.6
        )
        self.assertAlmostEqual(components["in_progress_pressure"], 0.4)
        self.assertNotIn("assigned_agent_work_ratio", components)
        self.assertNotIn("active_task_pressure", components)

    def test_nonnegative_bottleneck_excess(self):
        row = _station_row()
        row["regions"]["h3"]["node_bottleneck_weighted_density"] = 0.1
        components = extract_station_components(row)
        self.assertEqual(components["bottleneck_density_excess"], 0.0)

    def test_scale_contract_and_two_channel_target(self):
        rows = []
        for index in range(20):
            row = _station_row()
            row["regions"]["h3"]["stationary_ticks_max"] = float(index)
            row["regions"]["h3"]["node_density_cvar90"] = index / 20.0
            row["regions"]["h3"]["node_bottleneck_weighted_density"] = (
                0.3 + index / 40.0
            )
            row["assigned_agent_capacity_ratio"] = index / 20.0
            row["in_progress_pressure"] = index / 40.0
            rows.append(extract_station_components(row))
        contract = fit_scale_contract(rows, fitted_seeds=[511, 512])
        verify_scale_contract(contract)
        target = build_station_targets(_station_row(), contract)
        self.assertEqual(tuple(target["channels"]), CHANNEL_NAMES)
        for value in target["channels"].values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

        corrupted = copy.deepcopy(contract)
        corrupted["scales"]["in_progress_pressure"]["upper"] += 1.0
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_scale_contract(corrupted)


class StationCongestionHeadTest(unittest.TestCase):
    def test_station_specific_dynamic_output(self):
        head = StationCongestionHead(latent_dim=3)
        with torch.no_grad():
            head.proj.weight.zero_()
            head.proj.bias.zero_()
            head.proj.weight[0, 0] = 1.0
            head.proj.weight[1, 1] = 1.0
        z = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 0.0],
            [5.0, 6.0, 0.0],
        ])
        output = head(z, [2, 0, 3])
        self.assertEqual(tuple(output.shape), (3, 2))
        self.assertTrue(torch.allclose(
            output[0], torch.sigmoid(torch.tensor([3.0, 4.0]))
        ))
        self.assertTrue(torch.allclose(
            output[1], torch.sigmoid(torch.tensor([0.0, 0.0]))
        ))

    def test_station_ids_are_required_and_unique(self):
        head = StationCongestionHead(latent_dim=2)
        z = torch.zeros(4, 2)
        with self.assertRaisesRegex(ValueError, "required"):
            head(z, None)
        with self.assertRaisesRegex(ValueError, "unique"):
            head(z, [1, 1])
        with self.assertRaisesRegex(ValueError, "invalid"):
            head(z, [4])


class StationAlignmentTest(unittest.TestCase):
    def test_aligns_trace_station_order_without_dict_order_assumption(self):
        rows = [
            {"station_id": 1, "station_seed_nodes": [20, 21, 22]},
            {"station_id": 2, "station_seed_nodes": [30, 31, 32]},
        ]
        self.assertEqual(align_station_node_ids(rows, [30, 20]), [20, 30])

    def test_rejects_ambiguous_alignment(self):
        rows = [
            {"station_id": 1, "station_seed_nodes": [20, 30]},
            {"station_id": 2, "station_seed_nodes": [30]},
        ]
        with self.assertRaisesRegex(ValueError, "uniquely align"):
            align_station_node_ids(rows, [20, 30])

    def test_builds_three_hop_regions_and_mean_max_representation(self):
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3],
        ], dtype=torch.long)
        rows = [
            {"station_id": 1, "station_seed_nodes": [0]},
            {"station_id": 2, "station_seed_nodes": [4]},
        ]
        regions = build_station_region_node_ids(
            rows, edge_index, num_nodes=5, hops=1
        )
        self.assertEqual(regions, [[0, 1], [3, 4]])
        z = torch.tensor([
            [0.0, 10.0],
            [2.0, 8.0],
            [4.0, 6.0],
            [6.0, 4.0],
            [8.0, 2.0],
        ])
        representation = build_station_representations(
            z,
            [0, 4],
            representation=REPRESENTATION_STATION_REGION_MEAN_MAX,
            station_region_node_ids=regions,
        )
        self.assertEqual(tuple(representation.shape), (2, 6))
        self.assertTrue(torch.equal(
            representation[0],
            torch.tensor([0.0, 10.0, 1.0, 9.0, 2.0, 10.0]),
        ))


class StationCongestionTrainingSmokeTest(unittest.TestCase):
    @staticmethod
    def _split(seed: int):
        generator = torch.Generator().manual_seed(seed)
        frames = 12
        stations = 4
        count = frames * stations
        latents = torch.randn(count, 3, generator=generator)
        targets = torch.sigmoid(latents[:, :2])
        frame_group = torch.arange(frames).repeat_interleave(stations)
        station_id = torch.arange(stations).repeat(frames)
        return {
            "latents": latents,
            "targets": targets,
            "frame_group": frame_group,
            "run_index": torch.arange(frames).repeat_interleave(stations) // 3,
            "station_id": station_id,
            "seed": torch.full((count,), seed, dtype=torch.int64),
            "tick": torch.arange(frames).repeat_interleave(stations) * 5,
            "tick_fraction": torch.linspace(0.0, 1.0, frames).repeat_interleave(stations),
            "arm_code": (torch.arange(frames).repeat_interleave(stations) % 4),
            "load_code": (torch.arange(frames).repeat_interleave(stations) % 3),
            "global_open_order_count": torch.linspace(1.0, 10.0, frames).repeat_interleave(stations),
            "global_active_robot_ratio": torch.linspace(0.2, 0.9, frames).repeat_interleave(stations),
        }

    def test_end_to_end_linear_head_training(self):
        scale_rows = []
        for index in range(20):
            scale_rows.append({name: float(index + offset) for offset, name in enumerate(COMPONENT_NAMES)})
        contract = fit_scale_contract(scale_rows, fitted_seeds=[511])
        dataset = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "latent_dim": 3,
            "channel_names": list(CHANNEL_NAMES),
            "scale_contract": contract,
            "source_checkpoint": "synthetic.pt",
            "source_checkpoint_sha256": "0" * 64,
            "split_seeds": {"train": [511], "val": [512], "test": [513]},
            "splits": {
                "train": self._split(511),
                "val": self._split(512),
                "test": self._split(513),
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.pt"
            torch.save(dataset, dataset_path)
            output = root / "trained"
            summary = train(
                dataset_path=dataset_path,
                output_root=output,
                device=torch.device("cpu"),
                epochs=20,
                batch_size=32,
                learning_rate=0.05,
                weight_decay=0.0,
                patience=8,
                seed=20260804,
            )
            self.assertTrue(summary["audit"]["passed"])
            self.assertTrue((output / "best_station_congestion_head.pt").is_file())
            self.assertTrue((output / "state_level_validation.json").is_file())


class StationCongestionDatasetBuilderSmokeTest(unittest.TestCase):
    @staticmethod
    def _write_run(root: Path, seed: int, checkpoint_model: RMFSWorldModel):
        run_id = f"greedy_low_seed{seed}"
        run_dir = root / "runs" / run_id
        run_dir.mkdir(parents=True)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
        ], dtype=torch.long)
        generator = torch.Generator().manual_seed(seed)
        frame = {
            "node_history": torch.rand(4, 6, 10, generator=generator),
            "edge_features": torch.rand(edge_index.size(1), 6, generator=generator),
            "demand_context": torch.rand(7, generator=generator),
        }
        td_path = run_dir / f"tdstream_{run_id}.pt"
        torch.save({
            "schema_version": "td_stream_v1",
            "edge_index": edge_index,
            "station_node_ids": [1, 4],
            "frames_by_tick": {5: frame},
        }, td_path)
        rows = []
        for station_id, node_id, offset in ((0, 1, 0.0), (1, 4, 0.2)):
            row = _station_row(
                station_id=station_id,
                station_seed_nodes=[node_id],
                assigned_agent_capacity_ratio=0.2 + offset,
                in_progress_pressure=0.1 + offset,
            )
            row["regions"]["h3"]["stationary_ticks_max"] = 2.0 + offset
            rows.append(row)
        trace_path = run_dir / f"station_congestion_trace_{run_id}.jsonl"
        trace_path.write_text(json.dumps({
            "tick": 5,
            "system": {
                "open_order_count": 3,
                "active_robot_ratio": 0.5,
            },
            "stations": rows,
        }) + "\n", encoding="utf-8")
        manifest = run_dir / "run_outputs.sha256"
        manifest.write_text(
            f"{sha256_file(td_path)}  {td_path.name}\n"
            f"{sha256_file(trace_path)}  {trace_path.name}\n",
            encoding="utf-8",
        )
        summary = {
            "arm": "greedy",
            "load": "low",
            "seed": seed,
            "ticks": 10,
            "outputs": {"trace": trace_path.name, "td_stream": td_path.name},
            "audit": {"passed": True},
            "run_outputs_sha256": sha256_file(manifest),
        }
        (run_dir / "station_congestion_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

    def test_builds_latents_from_frozen_encoder_and_same_tick_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            input_root.mkdir()
            (input_root / "phase_c_station_congestion_frozen_protocol.json").write_text(
                json.dumps({"protocol": {"protocol_sha256": "a" * 64}}),
                encoding="utf-8",
            )
            model = RMFSWorldModel(
                demand_dim=7,
                hidden_dim=4,
                num_stations=2,
            )
            checkpoint = root / "model.pt"
            torch.save({
                "model_config": {
                    "node_feat_dim": 10,
                    "edge_feat_dim": 6,
                    "demand_dim": 7,
                    "action_node_dim": 8,
                    "action_global_dim": 6,
                    "hidden_dim": 4,
                    "rollout_horizon": 10,
                    "num_stations": 2,
                },
                "state_dict": model.state_dict(),
                "label_schema_version": "synthetic",
            }, checkpoint)
            for seed in (511, 517, 519):
                self._write_run(input_root, seed, model)
            output = root / "dataset"
            summary = build_dataset(
                input_root=input_root,
                checkpoint=checkpoint,
                output_root=output,
                splits={"train": (511,), "val": (517,), "test": (519,)},
                device=torch.device("cpu"),
                lower_quantile=0.05,
                upper_quantile=0.95,
            )
            self.assertTrue(summary["audit"]["passed"])
            payload = torch.load(
                output / "station_congestion_latents.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(payload["schema_version"], DATASET_SCHEMA_VERSION)
            self.assertEqual(tuple(payload["splits"]["train"]["latents"].shape), (2, 4))

            region_output = root / "region_dataset"
            region_summary = build_dataset(
                input_root=input_root,
                checkpoint=checkpoint,
                output_root=region_output,
                splits={"train": (511,), "val": (517,), "test": (519,)},
                device=torch.device("cpu"),
                lower_quantile=0.05,
                upper_quantile=0.95,
                representation=REPRESENTATION_STATION_REGION_MEAN_MAX,
            )
            self.assertEqual(
                region_summary["representation"]["name"],
                REPRESENTATION_STATION_REGION_MEAN_MAX,
            )
            region_payload = torch.load(
                region_output / "station_congestion_latents.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                tuple(region_payload["splits"]["train"]["latents"].shape),
                (2, 12),
            )


if __name__ == "__main__":
    unittest.main()
