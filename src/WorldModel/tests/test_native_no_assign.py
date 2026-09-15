import importlib
import tempfile
import types
import unittest
from unittest import mock

import torch

from Policies.TaskAssigner.WorldModelTaskAssigner import WorldModelTaskAssigner
from WorldModel.data.candidate_generator import (
    NO_ASSIGN_ACTION_SCHEMA_VERSION,
    NO_ASSIGN_ACTION_TYPE,
    NO_ASSIGN_ENCODING,
    build_candidate_assignment,
    generate_robot_candidates,
    make_no_assign_candidate,
)
from WorldModel.data.counterfactual_rollout import evaluate_candidate_rollout
from WorldModel.data.data_collector import WorldModelDataCollector
from WorldModel.data.dataset import WorldModelDataset
from WorldModel.data.fuse_and_split import compute_sample_fingerprint
from WorldModel.graph.graph_builder import (
    build_action_edge_field,
    build_action_field,
    compute_preview_legs,
)
from WorldModel.training.train import _infer_action_schema, _save_checkpoint
from WorldState.order_state import Order, OrderState, OrderStatus
from WorldState.task_state import TaskState


def _context(order_id=1):
    return types.SimpleNamespace(
        order_id=order_id,
        pod_id=7,
        pod_location=(0, 1),
        station_id=1,
        station_location=(0, 2),
        entry_position=(1, 2),
        exit_position=(0, 3),
        return_location=(0, 1),
        order_size=1,
    )


class NativeNoAssignTests(unittest.TestCase):
    def test_candidate_is_appended_after_top_m_robots(self):
        agents = [
            types.SimpleNamespace(agent_id=index, position=(0, index))
            for index in range(1, 5)
        ]
        world = types.SimpleNamespace(
            tick=4,
            get_idle_agents=lambda: list(agents),
            order_state=types.SimpleNamespace(orders={}),
        )

        group = generate_robot_candidates(
            [_context()],
            world,
            top_m=2,
            include_no_assign_candidate=True,
        )[0]

        self.assertEqual(group["robot_candidate_count"], 2)
        self.assertEqual(len(group["candidates"]), 3)
        self.assertEqual(
            group["candidates"][-1]["action_type"],
            NO_ASSIGN_ACTION_TYPE,
        )
        self.assertIsNone(group["candidates"][-1]["robot_id"])

    def test_zero_action_encoding_preserves_all_tensor_shapes(self):
        assignment = build_candidate_assignment(
            make_no_assign_candidate(),
            vars(_context()),
        )
        node_map = {(0, 0): 0, (0, 1): 1, (0, 2): 2}
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        world = types.SimpleNamespace(map_state=types.SimpleNamespace())

        legs = compute_preview_legs(
            assignment,
            world.map_state,
            node_map,
        )
        action_node, action_global = build_action_field(
            assignment,
            world,
            node_map,
            {value: key for key, value in node_map.items()},
            [1.0] * len(node_map),
        )
        action_edge = build_action_edge_field(
            assignment,
            edge_index,
            node_map,
            world.map_state,
        )

        self.assertEqual(legs, ([], [], []))
        self.assertEqual(tuple(action_node.shape), (3, 8))
        self.assertEqual(tuple(action_global.shape), (6,))
        self.assertEqual(tuple(action_edge.shape), (2, 4))
        self.assertEqual(int(torch.count_nonzero(action_node)), 0)
        self.assertEqual(int(torch.count_nonzero(action_global)), 0)
        self.assertEqual(int(torch.count_nonzero(action_edge)), 0)

    def test_isolated_rollout_does_not_create_a_task_for_no_assign(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [7]
        orders = OrderState()
        orders.add_order(order)
        tasks = TaskState()
        pod = types.SimpleNamespace(
            pod_id=7,
            is_carried=False,
            current_position=(0, 1),
        )

        class World:
            def __init__(self):
                self.tick = 5
                self.order_state = orders
                self.task_state = tasks
                self.pod_state = types.SimpleNamespace(get_pod=lambda _: pod)
                self.map_state = types.SimpleNamespace(
                    station_positions={1: (0, 2)}
                )
                self.station_state = types.SimpleNamespace(
                    get_queue=lambda _: None
                )

            def get_agent(self, _agent_id):
                return None

        config = types.SimpleNamespace(
            simulation=types.SimpleNamespace(
                pickup_duration=1,
                station_process_duration=1,
                dropoff_duration=1,
            )
        )

        def advance(world, _planner, _config):
            world.tick += 1
            return 0, 0, 0

        with (
            mock.patch(
                "WorldModel.data.counterfactual_rollout.force_apply_candidate"
            ) as force_apply,
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
                return_value=torch.zeros(1, 2),
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.extract_demand_context",
                return_value=torch.zeros(6),
            ),
            mock.patch(
                "WorldModel.data.counterfactual_rollout.compute_realized_cost",
                return_value=0.0,
            ),
        ):
            result = evaluate_candidate_rollout(
                world=World(),
                candidate=make_no_assign_candidate(),
                fixed_context=vars(_context(order.order_id)),
                config=config,
                path_planner=object(),
                horizon=2,
                node_map={(0, 0): 0},
                local_capacity=[1.0],
                bottleneck_score=[0.0],
                adj={},
                rollout_continuation_mode="isolated",
            )

        force_apply.assert_not_called()
        self.assertTrue(result["no_assign_applied"])
        self.assertEqual(result["candidate_action_type"], NO_ASSIGN_ACTION_TYPE)
        self.assertEqual(result["rollout_generated_orders"], 0)
        self.assertEqual(result["rollout_assigned_tasks"], 0)
        self.assertEqual(result["no_assign_audit"]["tasks_added_immediately"], [])
        self.assertEqual(
            result["no_assign_audit"]["tasks_added_during_isolated_rollout"],
            [],
        )
        self.assertEqual(float(result["future_mask"].sum()), 2.0)
        self.assertEqual(order.status, OrderStatus.PENDING)
        self.assertEqual(len(tasks.tasks), 0)

    def test_pairwise_ranking_keeps_no_assign_in_the_same_group(self):
        collector = WorldModelDataCollector(include_no_assign_candidate=True)
        collector._finalized_samples = [
            {
                "candidate_group_id": "g1",
                "action_type": "assign_robot",
                "realized_cost": 2.0,
            },
            {
                "candidate_group_id": "g1",
                "action_type": NO_ASSIGN_ACTION_TYPE,
                "realized_cost": 1.0,
            },
        ]

        pairs = collector.build_pairwise_data(epsilon=0.01)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(
            pairs[0]["sample_i"]["action_type"],
            NO_ASSIGN_ACTION_TYPE,
        )

    def test_fused_seeds_never_cross_pair_same_named_groups(self):
        samples = []
        for seed, robot_cost, no_assign_cost in (
            (11, 1.0, 2.0),
            (12, 100.0, 101.0),
        ):
            common = {
                "run_id": f"seed{seed}",
                "simulation_seed": seed,
                "candidate_group_id": "t5_o0_p7",
            }
            samples.extend([
                {
                    **common,
                    "action_type": "assign_robot",
                    "realized_cost": robot_cost,
                },
                {
                    **common,
                    "action_type": NO_ASSIGN_ACTION_TYPE,
                    "realized_cost": no_assign_cost,
                },
            ])

        dataset = WorldModelDataset(samples)
        pairs = dataset.build_pairwise_data(epsilon=0.01)

        self.assertEqual(dataset.summary()["num_groups"], 2)
        self.assertEqual(len(pairs), 2)
        for pair in pairs:
            self.assertEqual(
                pair["sample_i"]["simulation_seed"],
                pair["sample_j"]["simulation_seed"],
            )

    def test_fusion_fingerprint_accepts_no_numeric_heuristic(self):
        no_assign = {
            "candidate_group_id": "g1",
            "decision_tick": 5,
            "action_type": NO_ASSIGN_ACTION_TYPE,
            "candidate_info": {"action_type": NO_ASSIGN_ACTION_TYPE},
            "fixed_context": {"order_id": 1, "pod_id": 7},
            "realized_cost": 1.0,
            "heuristic_cost": None,
            "heuristic_cost_valid": False,
            "action_global": torch.zeros(6),
        }
        robot = {
            **no_assign,
            "action_type": "assign_robot",
            "candidate_info": {"action_type": "assign_robot", "robot_id": 3},
            "heuristic_cost": 4.0,
            "heuristic_cost_valid": True,
        }

        no_assign_fingerprint = compute_sample_fingerprint(no_assign)
        robot_fingerprint = compute_sample_fingerprint(robot)

        self.assertEqual(len(no_assign_fingerprint), 64)
        self.assertNotEqual(no_assign_fingerprint, robot_fingerprint)

    def test_checkpoint_schema_requires_complete_zero_encoded_groups(self):
        robot = {
            "candidate_group_id": "g1",
            "action_type": "assign_robot",
            "action_node": torch.ones(2, 8),
            "action_global": torch.ones(6),
            "action_edge": torch.ones(1, 4),
            "action_encoding": "route_fields_v1",
        }
        no_assign = {
            "candidate_group_id": "g1",
            "action_type": NO_ASSIGN_ACTION_TYPE,
            "action_node": torch.zeros(2, 8),
            "action_global": torch.zeros(6),
            "action_edge": torch.zeros(1, 4),
            "action_encoding": NO_ASSIGN_ENCODING,
        }

        schema = _infer_action_schema([robot, no_assign])
        robot_only = _infer_action_schema([robot])
        fused_rows = []
        for seed in (11, 12):
            for row in (robot, no_assign):
                fused_rows.append({
                    **row,
                    "run_id": f"seed{seed}",
                    "simulation_seed": seed,
                })
        fused_schema = _infer_action_schema(fused_rows)

        self.assertEqual(
            schema["schema_version"], NO_ASSIGN_ACTION_SCHEMA_VERSION
        )
        self.assertTrue(schema["supports_no_assign_candidate"])
        self.assertTrue(schema["complete_group_coverage"])
        self.assertTrue(schema["zero_encoding_verified"])
        self.assertFalse(robot_only["supports_no_assign_candidate"])
        self.assertTrue(fused_schema["supports_no_assign_candidate"])
        self.assertEqual(fused_schema["candidate_groups"], 2)

        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(1, 1)
            model._checkpoint_action_schema = schema
            path = _save_checkpoint(model, tmp, "model.pt")
            payload = torch.load(path, weights_only=False)
        self.assertEqual(payload["action_schema"], schema)

    def test_old_checkpoint_is_rejected_before_online_no_assign_scoring(self):
        assigner_module = importlib.import_module(
            "Policies.TaskAssigner.WorldModelTaskAssigner."
            "world_model_task_assigner"
        )
        assigner = WorldModelTaskAssigner(
            checkpoint_path="old_robot_only.pt",
            include_no_assign_candidate=True,
        )
        world = types.SimpleNamespace(
            map_state=types.SimpleNamespace(
                station_positions={1: (0, 0)}
            )
        )
        static_graph = (
            torch.zeros((2, 0), dtype=torch.long),
            {(0, 0): 0},
            {0: (0, 0)},
            [1.0],
            [0.0],
            [0],
            {0: []},
        )

        with (
            mock.patch.object(
                assigner_module.os.path,
                "isfile",
                return_value=True,
            ),
            mock.patch(
                "WorldModel.graph.graph_builder.build_static_graph",
                return_value=static_graph,
            ),
            mock.patch.object(
                assigner_module.torch,
                "load",
                return_value={"model_config": {}, "state_dict": {}},
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "was not trained on the complete native NO_ASSIGN",
            ):
                assigner._init(world)

    def test_online_native_action_set_can_select_no_assign(self):
        assigner = WorldModelTaskAssigner(
            checkpoint_path="schema_checked_elsewhere.pt",
            include_no_assign_candidate=True,
        )
        assigner._feature_history = types.SimpleNamespace(
            get_history=lambda: torch.zeros(4, 1, 10),
            get_latest=lambda: torch.zeros(1, 10),
        )
        assigner._edge_index = torch.zeros((2, 0), dtype=torch.long)
        assigner._node_map = {(0, 0): 0}
        assigner._inv_node_map = {0: (0, 0)}
        assigner._local_capacity = [1.0]
        assigner._bottleneck_score = [0.0]
        assigner._adj = {0: []}
        assigner._edge_flow_counter = {}
        assigner._station_node_ids = [0]

        class Model:
            @staticmethod
            def encode_state(*_args):
                return torch.zeros(1), torch.zeros(1), torch.zeros(0, 1)

            @staticmethod
            def predict_cost(
                _z,
                _demand,
                _edge_attr,
                action_node,
                _action_global,
                _edge_index,
                _station_ids,
                **_kwargs,
            ):
                is_no_assign = int(torch.count_nonzero(action_node)) == 0
                cost = 0.1 if is_no_assign else 1.0
                return torch.tensor(cost), {"risk_max": torch.tensor(0.0)}

        assigner._model = Model()
        agent = types.SimpleNamespace(agent_id=7, position=(0, 0))
        world = types.SimpleNamespace(
            get_idle_agents=lambda: [agent],
            map_state=types.SimpleNamespace(station_positions={1: (0, 0)}),
        )
        ctx = _context()

        def fake_action_field(assignment, *_args, **_kwargs):
            if assignment.get("action_type") == NO_ASSIGN_ACTION_TYPE:
                return torch.zeros(1, 8), torch.zeros(6)
            return torch.ones(1, 8), torch.ones(6)

        with (
            mock.patch(
                "WorldModel.graph.graph_builder.extract_edge_features",
                return_value=torch.zeros(0, 6),
            ),
            mock.patch(
                "WorldModel.graph.graph_builder.extract_demand_context",
                return_value=torch.zeros(6),
            ),
            mock.patch(
                "WorldModel.graph.graph_builder.compute_preview_legs",
                return_value=([], [], []),
            ),
            mock.patch(
                "WorldModel.graph.graph_builder.build_action_field",
                side_effect=fake_action_field,
            ),
        ):
            choices = assigner.select_robots(world, [ctx])

        self.assertEqual(choices, {})
        self.assertEqual(assigner.stats["native_no_assign_contexts"], 1)
        self.assertEqual(assigner.stats["native_no_assign_selected"], 1)
        self.assertEqual(assigner.stats["native_no_assign_scored"], 1)


if __name__ == "__main__":
    unittest.main()
