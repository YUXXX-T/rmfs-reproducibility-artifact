import types
import unittest
import random
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Policies.TaskAssigner.WorldModelTaskAssigner.world_model_task_assigner import (
    WorldModelTaskAssigner,
)
from WorldModel.core.lyapunov import LyapunovL0Config


class CandidateCoverageTests(unittest.TestCase):
    @staticmethod
    def _assigner(mode):
        assigner = WorldModelTaskAssigner.__new__(WorldModelTaskAssigner)
        assigner.top_m = 3
        assigner.candidate_robot_mode = mode
        assigner.lyapunov_l0_config = LyapunovL0Config(
            eta_bin_edges=(15, 25, 50)
        )
        assigner.stats = {
            "all_idle_candidate_contexts": 0,
            "all_idle_candidates_scored": 0,
        }
        return assigner

    @staticmethod
    def _world():
        station = types.SimpleNamespace(capacity=1)
        return types.SimpleNamespace(
            config=types.SimpleNamespace(
                simulation=types.SimpleNamespace(
                    pickup_duration=2,
                    station_process_duration=5,
                    dropoff_duration=2,
                )
            ),
            station_state=types.SimpleNamespace(stations={1: station}),
        )

    @staticmethod
    def _context():
        return types.SimpleNamespace(
            station_id=1,
            pod_location=(0, 0),
            station_location=(0, 10),
        )

    @staticmethod
    def _agents():
        distances = (0, 1, 2, 10, 30)
        return [
            types.SimpleNamespace(agent_id=i + 1, position=(0, distance))
            for i, distance in enumerate(distances)
        ]

    def test_nearest_training_mode_does_not_truncate_online_candidates(self):
        assigner = self._assigner("nearest")
        selected = assigner._select_robot_candidates(
            self._world(), self._context(), self._agents()
        )
        self.assertEqual([agent.agent_id for agent in selected], [1, 2, 3, 4, 5])
        self.assertEqual(assigner.stats["all_idle_candidate_contexts"], 1)
        self.assertEqual(assigner.stats["all_idle_candidates_scored"], 5)

    def test_stratified_training_mode_also_scores_every_idle_robot_online(self):
        assigner = self._assigner("eta_stratified")
        snapshot = types.SimpleNamespace(arrival_bins={1: (0.0,) * 4})
        selected = assigner._select_robot_candidates(
            self._world(), self._context(), self._agents(), snapshot
        )
        self.assertEqual([agent.agent_id for agent in selected], [1, 2, 3, 4, 5])

    def test_context_stratification_is_not_prefix_only(self):
        assigner = WorldModelTaskAssigner.__new__(WorldModelTaskAssigner)
        assigner.candidate_context_mode = "stratified"
        assigner.lyapunov_l0_config = LyapunovL0Config(
            eta_bin_edges=(15, 25, 50)
        )
        assigner.reservation_window = 10
        assigner._candidate_explore_rng = random.Random(0)
        assigner._node_map = {}
        assigner._edge_flow_counter = {}
        assigner._bottleneck_score = None
        assigner.stats = {
            "context_stratified_calls": 0,
            "context_stratified_available": 0,
            "context_stratified_selected": 0,
            "context_pressure_representatives": 0,
            "context_eta_representatives": 0,
            "context_route_representatives": 0,
            "context_conflict_representatives": 0,
            "context_explore_representatives": 0,
        }
        idle = types.SimpleNamespace(agent_id=1, position=(0, 0), path=[])
        stations = {
            station_id: types.SimpleNamespace(capacity=1, occupancy=lambda: 0)
            for station_id in range(1, 5)
        }
        world = types.SimpleNamespace(
            tick=0,
            agents=[idle],
            get_idle_agents=lambda: [idle],
            order_state=types.SimpleNamespace(orders={}),
            task_state=types.SimpleNamespace(tasks={}),
            station_state=types.SimpleNamespace(stations=stations),
            map_state=types.SimpleNamespace(
                station_positions={
                    station_id: (0, station_id)
                    for station_id in range(1, 5)
                }
            ),
            config=types.SimpleNamespace(
                simulation=types.SimpleNamespace(
                    pickup_duration=2,
                    station_process_duration=5,
                    dropoff_duration=2,
                )
            ),
        )
        pod_distances = (30, 20, 1, 2)
        contexts = [
            types.SimpleNamespace(
                order_id=index,
                pod_id=index,
                station_id=index + 1,
                pod_location=(0, distance),
                station_location=(0, index + 1),
                entry_position=None,
                return_location=(0, 0),
                order_size=1,
            )
            for index, distance in enumerate(pod_distances)
        ]
        selected = assigner._select_context_representatives(
            world, contexts, budget=2
        )
        self.assertEqual(len(selected), 2)
        self.assertIn(contexts[2], selected)
        self.assertNotEqual(selected, contexts[:2])


if __name__ == "__main__":
    unittest.main()
