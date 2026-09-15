import tempfile
import unittest

import torch

from WorldModel.core.analytic_work_residual_head import (
    SCHEMA_VERSION,
    AnalyticWorkResidualHead,
)
from WorldModel.core.work_drift_head import (
    SCHEMA_VERSION as LEGACY_SCHEMA_VERSION,
    WorkDriftHead,
)
from WorldModel.training.train_analytic_work_residual_head import (
    nominal_relief_from_post_action_snapshot,
)


class AnalyticWorkResidualHeadTests(unittest.TestCase):
    def _head(self):
        return AnalyticWorkResidualHead(
            latent_dim=4,
            hidden_dim=8,
            num_stations=2,
            work_capacity=[2.0, 4.0],
            work_weight=1.0,
            residual_scale=[0.5, 0.75],
            residual_limit=2.0,
        )

    def test_zero_initialisation_is_exact_analytic_baseline(self):
        head = self._head()
        z_start = torch.randn(6, 4)
        z_endpoint = torch.randn(6, 4)
        demand = torch.randn(4)
        start = torch.tensor([2.0, 4.0])
        post = torch.tensor([2.5, 4.0])
        nominal_relief = torch.tensor([0.25, 1.0])
        available_relief = torch.tensor([1.0, 2.0])
        prediction = head(
            z_start,
            z_endpoint,
            demand,
            [1, 4],
            start,
            post,
            nominal_relief,
            available_relief,
        )
        expected_endpoint = torch.tensor([2.25, 3.0])
        expected_drift = (
            head.work_potential(expected_endpoint)
            - head.work_potential(start)
        )
        self.assertTrue(torch.equal(
            prediction.residual_station_relief,
            torch.zeros_like(nominal_relief),
        ))
        self.assertTrue(torch.allclose(
            prediction.endpoint_station_work, expected_endpoint
        ))
        self.assertTrue(torch.allclose(
            prediction.raw_work_drift, expected_drift
        ))
        self.assertTrue(torch.equal(
            prediction.raw_work_drift, prediction.nominal_raw_work_drift
        ))

    def test_signed_residual_changes_relief_not_the_potential_definition(self):
        head = self._head()
        with torch.no_grad():
            head.station_net[-1].bias.fill_(0.5)
        batch = 3
        start = torch.tensor([[2.0, 4.0]]).expand(batch, -1).clone()
        post = torch.tensor([[2.5, 4.0]]).expand(batch, -1).clone()
        nominal = torch.tensor([[0.25, 1.0]]).expand(batch, -1).clone()
        available = torch.tensor([[1.0, 2.0]]).expand(batch, -1).clone()
        capacity = head.work_capacity
        station_context = torch.zeros(batch, 2, head.station_dim)
        station_context[..., -4] = start / capacity
        station_context[..., -3] = post / capacity
        station_context[..., -2] = nominal / capacity
        station_context[..., -1] = available / capacity
        prediction = head.predict_from_features(
            torch.zeros(batch, head.global_dim), station_context, start
        )
        reconstructed = (
            head.work_potential(prediction.endpoint_station_work)
            - head.work_potential(start)
        )
        self.assertTrue(torch.all(prediction.residual_station_relief > 0.0))
        self.assertTrue(torch.allclose(prediction.raw_work_drift, reconstructed))

    def test_residual_cannot_dissipate_pending_work(self):
        head = self._head()
        with torch.no_grad():
            head.station_net[-1].bias.fill_(10.0)
        start = torch.tensor([[4.0, 4.0]])
        post = torch.tensor([[4.0, 4.0]])
        nominal = torch.tensor([[0.1, 0.2]])
        available = torch.tensor([[0.25, 0.5]])
        station_context = torch.zeros(1, 2, head.station_dim)
        station_context[..., -4] = start / head.work_capacity
        station_context[..., -3] = post / head.work_capacity
        station_context[..., -2] = nominal / head.work_capacity
        station_context[..., -1] = available / head.work_capacity
        prediction = head.predict_from_features(
            torch.zeros(1, head.global_dim), station_context, start
        )
        self.assertTrue(torch.allclose(
            prediction.predicted_station_relief, available
        ))
        self.assertTrue(torch.allclose(
            prediction.endpoint_station_work, post - available
        ))

    def test_signed_residual_can_represent_reverse_physical_progress(self):
        head = self._head()
        with torch.no_grad():
            head.station_net[-1].bias.fill_(-10.0)
        start = torch.tensor([[2.0, 4.0]])
        post = start.clone()
        nominal = torch.tensor([[0.1, 0.2]])
        available = torch.tensor([[0.25, 0.5]])
        station_context = torch.zeros(1, 2, head.station_dim)
        station_context[..., -4] = start / head.work_capacity
        station_context[..., -3] = post / head.work_capacity
        station_context[..., -2] = nominal / head.work_capacity
        station_context[..., -1] = available / head.work_capacity
        prediction = head.predict_from_features(
            torch.zeros(1, head.global_dim), station_context, start
        )
        self.assertTrue(torch.all(
            prediction.predicted_station_relief < 0.0
        ))
        self.assertTrue(torch.all(
            prediction.endpoint_station_work > post
        ))

    def test_mismatched_compact_physical_context_is_rejected(self):
        head = self._head()
        station_context = torch.zeros(1, 2, head.station_dim)
        station_context[..., -4] = torch.tensor([1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "encoded and explicit"):
            head.predict_from_features(
                torch.zeros(1, head.global_dim),
                station_context,
                torch.zeros(1, 2),
            )

    def test_checkpoint_round_trip_and_legacy_schema_is_untouched(self):
        head = self._head()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "head_config": head.checkpoint_config(),
            "head_state_dict": head.state_dict(),
        }
        legacy = WorkDriftHead(
            latent_dim=4,
            hidden_dim=8,
            num_stations=2,
            work_capacity=[2.0, 4.0],
        )
        legacy_payload = {
            "schema_version": LEGACY_SCHEMA_VERSION,
            "head_config": legacy.checkpoint_config(),
            "head_state_dict": legacy.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            current_path = f"{directory}/analytic_residual.pt"
            legacy_path = f"{directory}/legacy.pt"
            torch.save(payload, current_path)
            torch.save(legacy_payload, legacy_path)
            loaded, restored = AnalyticWorkResidualHead.from_checkpoint(
                current_path
            )
            legacy_loaded, legacy_restored = WorkDriftHead.from_checkpoint(
                legacy_path
            )
        self.assertEqual(restored["schema_version"], SCHEMA_VERSION)
        self.assertEqual(loaded.checkpoint_config(), head.checkpoint_config())
        self.assertEqual(legacy_restored["schema_version"], LEGACY_SCHEMA_VERSION)
        self.assertEqual(
            legacy_loaded.checkpoint_config(), legacy.checkpoint_config()
        )

    def test_snapshot_nominal_relief_excludes_pending_and_saturates(self):
        snapshot = {
            "station_work": {10: 2.5, 20: 1.0},
            "chains": {
                "1:1": {
                    "state": "pipeline",
                    "active_task_id": 1,
                    "station_id": 10,
                    "mass": 1.0,
                    "initial_work": 20.0,
                    "remaining_work": 12.0,
                },
                "2:2": {
                    "state": "pipeline",
                    "active_task_id": 2,
                    "station_id": 10,
                    "mass": 2.0,
                    "initial_work": 10.0,
                    "remaining_work": 3.0,
                },
                "3:3": {
                    "state": "pending",
                    "station_id": 20,
                    "mass": 1.0,
                    "initial_work": 1.0,
                    "remaining_work": 1.0,
                },
            },
        }
        h2 = nominal_relief_from_post_action_snapshot(
            snapshot, [10, 20], horizon=2
        )
        h20 = nominal_relief_from_post_action_snapshot(
            snapshot, [10, 20], horizon=20
        )
        self.assertTrue(torch.all(h20 >= h2))
        self.assertAlmostEqual(float(h2[0]), 0.5)
        self.assertAlmostEqual(float(h20[0]), 1.2)
        self.assertEqual(float(h2[1]), 0.0)
        self.assertEqual(float(h20[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
