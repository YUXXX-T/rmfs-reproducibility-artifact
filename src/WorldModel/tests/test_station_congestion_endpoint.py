import types
import unittest
from unittest import mock

import numpy as np
import torch

from WorldModel.core.station_congestion_head import (
    StationCongestionHead,
    fit_scale_contract,
)
from WorldModel.data.counterfactual_rollout import evaluate_candidate_rollout
from WorldModel.evaluation.analyze_phase_c_station_congestion_endpoint import (
    _layer_metrics,
)
from WorldModel.evaluation.station_congestion_endpoint import (
    EndpointObservationObserver,
    StationLayout,
    build_endpoint_station_rows,
    evaluate_station_psi,
)
from WorldState.agent_state import AgentStatus


def _scale_contract():
    low = {
        "stationary_ticks_max": 0.0,
        "node_density_cvar90": 0.0,
        "bottleneck_density_excess": 0.0,
        "assigned_agent_capacity_ratio": 0.0,
        "in_progress_pressure": 0.0,
    }
    high = {key: 1.0 for key in low}
    return fit_scale_contract([low, high], fitted_seeds=(1,))


def _world():
    queue_1 = types.SimpleNamespace(
        capacity=2,
        _assigned_agents={1},
        occupancy=lambda: 1,
        entry_position=(0, 1),
        exit_position=(0, 2),
    )
    queue_2 = types.SimpleNamespace(
        capacity=2,
        _assigned_agents=set(),
        occupancy=lambda: 0,
        entry_position=(1, 1),
        exit_position=(1, 2),
    )
    queues = {1: queue_1, 2: queue_2}
    agents = [
        types.SimpleNamespace(
            agent_id=1,
            status=AgentStatus.MOVING,
            position=(0, 0),
            previous_position=(0, 1),
            moved_this_tick=True,
            stationary_ticks=5,
        ),
        types.SimpleNamespace(
            agent_id=2,
            status=AgentStatus.IDLE,
            position=(1, 0),
            previous_position=(1, 0),
            moved_this_tick=False,
            stationary_ticks=0,
        ),
    ]
    orders = {
        1: types.SimpleNamespace(status="IN_PROGRESS", station_id=1),
        2: types.SimpleNamespace(status="PENDING", station_id=2),
    }
    return types.SimpleNamespace(
        tick=12,
        agents=agents,
        map_state=types.SimpleNamespace(
            station_positions={1: (0, 0), 2: (1, 0)}
        ),
        station_state=types.SimpleNamespace(
            get_queue=lambda station_id: queues[station_id]
        ),
        order_state=types.SimpleNamespace(orders=orders, total_completed=0),
        get_agent=lambda _agent_id: types.SimpleNamespace(
            status=AgentStatus.IDLE, plan_failed_streak=0
        ),
    )


class StationCongestionEndpointTests(unittest.TestCase):
    def test_physical_endpoint_targets_remain_station_specific(self):
        world = _world()
        layout = StationLayout(
            station_ids=(1, 2),
            station_node_ids=(0, 2),
            station_seed_node_ids=((0,), (2,)),
            station_region_node_ids=((0, 1), (2, 3)),
        )
        node_map = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        features = torch.zeros(4, 10)
        features[:, 8] = 1.0
        features[0:2, 1] = torch.tensor([0.8, 0.6])
        features[2:4, 1] = torch.tensor([0.1, 0.2])

        rows = build_endpoint_station_rows(world, features, layout, node_map)

        self.assertEqual(len(rows), 2)
        self.assertGreater(
            rows[0]["regions"]["h3"]["node_density_cvar90"],
            rows[1]["regions"]["h3"]["node_density_cvar90"],
        )
        self.assertEqual(rows[0]["assigned_agent_capacity_ratio"], 0.5)
        self.assertEqual(rows[1]["assigned_agent_capacity_ratio"], 0.0)
        self.assertGreater(rows[0]["in_progress_pressure"], 0.0)
        self.assertEqual(rows[1]["in_progress_pressure"], 0.0)

    def test_endpoint_observer_rebuilds_full_history_and_targets(self):
        world = _world()
        layout = StationLayout(
            station_ids=(1, 2),
            station_node_ids=(0, 2),
            station_seed_node_ids=((0,), (2,)),
            station_region_node_ids=((0, 1), (2, 3)),
        )
        node_map = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        observer = EndpointObservationObserver(
            node_history=torch.zeros(4, 4, 10),
            edge_index=torch.tensor([[0, 1, 2], [1, 0, 3]]),
            node_map=node_map,
            inv_node_map={value: key for key, value in node_map.items()},
            local_capacity=[1.0] * 4,
            bottleneck_score=[1.0] * 4,
            node_type_arr=[0.0] * 4,
            adj={0: [1], 1: [0], 2: [3], 3: [2]},
            flow_counter={},
            edge_flow_counter={},
            layout=layout,
            scale_contract=_scale_contract(),
            horizon=2,
        )
        frames = [torch.full((4, 10), 0.2), torch.full((4, 10), 0.4)]
        for frame in frames:
            frame[:, 8] = 1.0

        with (
            mock.patch(
                "WorldModel.evaluation.station_congestion_endpoint.extract_node_features",
                side_effect=frames,
            ),
            mock.patch(
                "WorldModel.evaluation.station_congestion_endpoint.extract_edge_features",
                return_value=torch.zeros(3, 6),
            ),
            mock.patch(
                "WorldModel.evaluation.station_congestion_endpoint.extract_demand_context",
                return_value=torch.zeros(9),
            ),
        ):
            observer.on_post_step(world)
            observer.on_post_step(world)
            endpoint = observer.finalize(world)

        self.assertEqual(tuple(endpoint["node_history"].shape), (4, 4, 10))
        torch.testing.assert_close(endpoint["node_history"][-2], frames[0])
        torch.testing.assert_close(endpoint["node_history"][-1], frames[1])
        self.assertEqual(tuple(endpoint["physical_targets"].shape), (2, 2))

    def test_rollout_observer_is_opt_in_and_not_silently_ignored(self):
        class Observer:
            def __init__(self):
                self.steps = 0

            def on_post_step(self, _world):
                self.steps += 1

            def finalize(self, _world):
                return {"steps": self.steps}

        world = _world()

        def advance(value, _planner, _config):
            value.tick += 1
            return 0, 0, 0

        patches = (
            mock.patch(
                "WorldModel.data.counterfactual_rollout.force_apply_candidate",
                return_value=True,
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.step_world",
                side_effect=advance,
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.extract_node_labels",
                return_value=torch.zeros(1, 6),
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.extract_system_labels",
                return_value=torch.zeros(7),
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.extract_station_labels",
                return_value=torch.zeros(2, 2),
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.extract_demand_context",
                return_value=torch.zeros(9),
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.compute_realized_cost",
                return_value=0.0,
            ),
        )
        observer = Observer()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            result = evaluate_candidate_rollout(
                world=world,
                candidate={"robot_id": 1},
                fixed_context={},
                config=types.SimpleNamespace(simulation=types.SimpleNamespace()),
                path_planner=object(),
                horizon=2,
                node_map={(0, 0): 0},
                local_capacity=[1.0],
                bottleneck_score=[0.0],
                adj={},
                rollout_observer=observer,
            )
        self.assertEqual(result["rollout_observer_output"], {"steps": 2})

    def test_region_head_keeps_variable_station_count(self):
        head = StationCongestionHead(192)
        payload = {
            "representation": {"name": "station_region_mean_max"}
        }
        layout = StationLayout(
            station_ids=(10, 20),
            station_node_ids=(0, 3),
            station_seed_node_ids=((0,), (3,)),
            station_region_node_ids=((0, 1, 2), (3, 4)),
        )
        output = evaluate_station_psi(
            torch.randn(5, 64),
            head=head,
            head_payload=payload,
            layout=layout,
        )
        self.assertEqual(tuple(output.shape), (2, 2))

    def test_synthetic_delta_analysis_reports_perfect_transport(self):
        rows = []
        for seed in (521, 522):
            for action in (0, 1):
                for station in (1, 2):
                    value = 0.1 * action + 0.2 * station + 0.01 * seed
                    rows.append({
                        "run_id": f"low_seed{seed}",
                        "load": "low",
                        "seed": seed,
                        "decision_tick": 100,
                        "action_id": f"{seed}|{action}",
                        "candidate_rank_group": f"{seed}|s1",
                        "station_id": station,
                        "is_context_station": station == 1,
                        "delta_psi_predicted": {
                            "traffic": value,
                            "service": value * 0.5,
                        },
                        "delta_psi_real": {
                            "traffic": value,
                            "service": value * 0.5,
                        },
                    })
        metrics = _layer_metrics(
            rows,
            prediction_key="delta_psi_predicted",
            target_key="delta_psi_real",
            include_sign=True,
            seed_offset=0,
        )
        self.assertAlmostEqual(
            metrics["all_stations"]["traffic"]["pooled_spearman"], 1.0
        )
        self.assertAlmostEqual(
            metrics["context_station"]["service"]["sign"]["all_sign_accuracy"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
