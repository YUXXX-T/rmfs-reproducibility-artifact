import types
import unittest
from unittest import mock

from WorldModel.data.candidate_generator import generate_robot_candidates


class CollectionCandidateStratificationTests(unittest.TestCase):
    @staticmethod
    def _context():
        return types.SimpleNamespace(
            order_id=1,
            pod_id=7,
            pod_location=(0, 0),
            station_id=1,
            station_location=(0, 10),
            entry_position=(1, 10),
            exit_position=(0, 11),
            return_location=(5, 5),
            order_size=1,
        )

    @staticmethod
    def _world():
        agents = [
            types.SimpleNamespace(
                agent_id=index + 1,
                position=(0, distance),
                path=[],
                path_index=0,
                stationary_ticks=0,
                plan_failed_streak=0,
            )
            for index, distance in enumerate((0, 1, 2, 10, 30))
        ]
        station = types.SimpleNamespace(capacity=1, occupancy=lambda: 0)
        return types.SimpleNamespace(
            tick=0,
            agents=agents,
            get_idle_agents=lambda: list(agents),
            order_state=types.SimpleNamespace(orders={}),
            task_state=types.SimpleNamespace(tasks={}),
            station_state=types.SimpleNamespace(stations={1: station}),
            map_state=types.SimpleNamespace(station_positions={1: (0, 10)}),
            config=types.SimpleNamespace(
                simulation=types.SimpleNamespace(
                    pickup_duration=2,
                    station_process_duration=5,
                    dropoff_duration=2,
                )
            ),
        )

    def test_nearest_mode_is_backward_compatible(self):
        groups = generate_robot_candidates(
            [self._context()], self._world(), top_m=3,
        )
        candidates = groups[0]["candidates"]
        self.assertEqual([row["robot_id"] for row in candidates], [1, 2, 3])
        self.assertTrue(candidates[0]["chosen"])

    def test_stratified_mode_keeps_eta_and_traffic_extremes(self):
        feature_rows = {
            1: {"eta": 5.0, "eta_bin": 0, "arrival_delta_preview": 0.0,
                "route_conflict_preview": 2.0, "route_length_preview": 5},
            2: {"eta": 6.0, "eta_bin": 0, "arrival_delta_preview": 0.0,
                "route_conflict_preview": 0.1, "route_length_preview": 6},
            3: {"eta": 15.0, "eta_bin": 1, "arrival_delta_preview": 1.0,
                "route_conflict_preview": 0.5, "route_length_preview": 15},
            4: {"eta": 30.0, "eta_bin": 2, "arrival_delta_preview": 1.0,
                "route_conflict_preview": 5.0, "route_length_preview": 30},
            5: {"eta": 60.0, "eta_bin": 3, "arrival_delta_preview": 0.0,
                "route_conflict_preview": 1.0, "route_length_preview": 60},
        }

        def preview(_ctx, agent, _world, **_kwargs):
            return dict(feature_rows[agent.agent_id])

        with mock.patch(
            "WorldModel.data.candidate_generator._candidate_preview_features",
            side_effect=preview,
        ):
            groups = generate_robot_candidates(
                [self._context()],
                self._world(),
                top_m=5,
                candidate_mode="stratified",
            )

        candidates = groups[0]["candidates"]
        ids = [row["robot_id"] for row in candidates]
        self.assertEqual(ids[0], 1)
        self.assertIn(2, ids)  # low-conflict representative
        self.assertIn(4, ids)  # high-conflict representative
        self.assertGreaterEqual(len({row["eta_bin"] for row in candidates}), 3)
        self.assertEqual(groups[0]["fixed_context"]["entry_position"], (1, 10))
        self.assertEqual(groups[0]["fixed_context"]["exit_position"], (0, 11))


if __name__ == "__main__":
    unittest.main()
