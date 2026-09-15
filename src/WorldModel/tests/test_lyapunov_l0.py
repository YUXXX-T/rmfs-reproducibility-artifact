from dataclasses import replace
import math
from pathlib import Path
import sys
import types
import unittest

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from WorldModel.core.lyapunov import (
        LyapunovL0Config,
        _arrival_potential,
        compute_lyapunov_snapshot,
        compute_productive_progress,
        preview_candidate_eta,
    )
except ModuleNotFoundError as exc:
    # The ledger itself is intentionally torch-free.  Let its unit tests run
    # in lightweight analysis environments where importing WorldModel's
    # package-level neural API fails because PyTorch is absent.
    if exc.name != "torch":
        raise
    import importlib.util
    module_path = Path(__file__).parents[1] / "core" / "lyapunov.py"
    spec = importlib.util.spec_from_file_location("_lyapunov_l0_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    LyapunovL0Config = module.LyapunovL0Config
    _arrival_potential = module._arrival_potential
    compute_lyapunov_snapshot = module.compute_lyapunov_snapshot
    compute_productive_progress = module.compute_productive_progress
    preview_candidate_eta = module.preview_candidate_eta
from WorldState.order_state import Order, OrderStatus
from WorldState.agent_state import AgentState, AgentStatus
from WorldState.task_state import Task, TaskStatus, TaskType


class _Station:
    def __init__(self, occupancy=0, capacity=2, entry_position=None,
                 exit_position=None):
        self._occupancy = occupancy
        self.capacity = capacity
        self.entry_position = entry_position
        self.exit_position = exit_position

    def occupancy(self):
        return self._occupancy


class _World:
    def __init__(self, order, agents, tasks=()):
        self.tick = 0
        self.agents = list(agents)
        self.order_state = types.SimpleNamespace(orders={order.order_id: order})
        self.task_state = types.SimpleNamespace(
            tasks={task.task_id: task for task in tasks}
        )
        self.station_state = types.SimpleNamespace(
            stations={order.station_id: _Station()}
        )
        self.map_state = types.SimpleNamespace(
            station_positions={order.station_id: (0, 10)}
        )
        self.config = types.SimpleNamespace(
            simulation=types.SimpleNamespace(
                pickup_duration=2,
                station_process_duration=5,
                dropoff_duration=2,
            )
        )

    def get_agent(self, agent_id):
        return next((a for a in self.agents if a.agent_id == agent_id), None)


def _agent(agent_id=1, position=(0, 0)):
    return types.SimpleNamespace(
        agent_id=agent_id,
        position=position,
        path=[],
        path_index=0,
        wait_ticks=0,
        stationary_ticks=0,
        plan_failed_streak=0,
        status=AgentStatus.MOVING_TO_POD,
    )


def _chain(order, pod_id, agent):
    pick = Task(TaskType.PICK, order.order_id, pod_id, (0, 0), (0, 4))
    deliver = Task(
        TaskType.DELIVER, order.order_id, pod_id, (0, 4), (0, 10)
    )
    ret = Task(TaskType.RETURN, order.order_id, pod_id, (0, 10), (0, 2))
    for task in (pick, deliver, ret):
        task.agent_id = agent.agent_id
        task.status = TaskStatus.ASSIGNED
    pick.status = TaskStatus.IN_PROGRESS
    pick.free_flow_time = 6
    deliver.free_flow_time = 11
    ret.free_flow_time = 10
    return pick, deliver, ret


class LyapunovL0Tests(unittest.TestCase):
    def test_json_style_eta_bin_list_is_normalised(self):
        config = LyapunovL0Config(eta_bin_edges=[5, 10])
        self.assertEqual(config.eta_bin_edges, (5, 10))

    def test_pending_to_assigned_is_continuous_for_two_pods(self):
        order = Order({"sku_a": 1, "sku_b": 1}, station_id=1)
        order.pod_ids = [10, 11]
        world = _World(order, [_agent(1), _agent(2)])
        pending = compute_lyapunov_snapshot(world)
        self.assertAlmostEqual(pending.station_work[1], 2.0)

        tasks = []
        for pod_id, agent in zip(order.pod_ids, world.agents):
            tasks.extend(_chain(order, pod_id, agent))
        order.status = OrderStatus.IN_PROGRESS
        world.task_state.tasks = {task.task_id: task for task in tasks}
        assigned = compute_lyapunov_snapshot(world)
        self.assertAlmostEqual(assigned.station_work[1], 2.0)

    def test_unresolved_materialisation_is_residual_not_external_arrival(self):
        order = Order({"sku_a": 1, "sku_b": 1}, station_id=1)
        agents = [_agent(1), _agent(2)]
        world = _World(order, agents)
        unresolved = compute_lyapunov_snapshot(world)
        self.assertAlmostEqual(unresolved.station_work[1], 1.0)

        order.pod_ids = [10, 11]
        order.status = OrderStatus.IN_PROGRESS
        tasks = []
        for pod_id, agent in zip(order.pod_ids, agents):
            tasks.extend(_chain(order, pod_id, agent))
        world.task_state.tasks = {task.task_id: task for task in tasks}
        world.tick = 1
        materialised = compute_lyapunov_snapshot(world)
        progress = compute_productive_progress(unresolved, materialised)

        self.assertEqual(progress.arrival_total, 0.0)
        self.assertEqual(progress.productive_total, 0.0)
        self.assertAlmostEqual(progress.replan_residual_total, 1.0)

    def test_new_order_id_is_external_arrival_not_replan_residual(self):
        first = Order({"sku": 1}, station_id=1)
        first.pod_ids = [10]
        world = _World(first, [_agent(1)])
        before = compute_lyapunov_snapshot(world)

        second = Order({"sku": 1}, station_id=1)
        second.pod_ids = [20]
        world.order_state.orders[second.order_id] = second
        world.tick = 1
        after = compute_lyapunov_snapshot(world)
        progress = compute_productive_progress(before, after)

        self.assertAlmostEqual(progress.arrival_total, 1.0)
        self.assertEqual(progress.replan_residual_total, 0.0)

    def test_physical_motion_produces_progress_but_no_motion_does_not(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        agent = _agent(position=(0, 0))
        tasks = _chain(order, 10, agent)
        order.status = OrderStatus.IN_PROGRESS
        world = _World(order, [agent], tasks)
        before = compute_lyapunov_snapshot(world)

        same = compute_lyapunov_snapshot(world)
        self.assertEqual(compute_productive_progress(before, same).productive_total, 0)

        agent.position = (0, 2)
        world.tick = 2
        after = compute_lyapunov_snapshot(world)
        progress = compute_productive_progress(before, after)
        self.assertGreater(progress.productive_total, 0.0)
        self.assertGreater(progress.mu_by_station[1], 0.0)

    def test_moving_away_is_reverse_residual_not_productive(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        agent = _agent(position=(0, 2))
        tasks = _chain(order, 10, agent)
        order.status = OrderStatus.IN_PROGRESS
        world = _World(order, [agent], tasks)
        before = compute_lyapunov_snapshot(world)
        agent.position = (0, 0)
        world.tick = 1
        after = compute_lyapunov_snapshot(world)
        progress = compute_productive_progress(before, after)
        self.assertEqual(progress.productive_total, 0.0)
        self.assertGreater(progress.reverse_total, 0.0)

    def test_route_replan_is_residual_not_productive_progress(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        agent = _agent(position=(0, 0))
        tasks = _chain(order, 10, agent)
        order.status = OrderStatus.IN_PROGRESS
        world = _World(order, [agent], tasks)

        # Same physical state, but the first route contains two extra steps.
        agent.path = [
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (0, 4),
        ]
        before = compute_lyapunov_snapshot(world)
        agent.path = [(0, 1), (0, 2), (0, 3), (0, 4)]
        after = compute_lyapunov_snapshot(world)

        progress = compute_productive_progress(before, after)
        self.assertEqual(progress.productive_total, 0.0)
        self.assertEqual(progress.reverse_total, 0.0)
        self.assertEqual(progress.replan_residual_total, 0.0)
        self.assertGreater(progress.route_plan_churn_total, 0.0)
        self.assertAlmostEqual(before.station_work[1], after.station_work[1])

    def test_planned_wait_consumption_is_not_a_replan(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        agent = _agent(position=(0, 0))
        tasks = _chain(order, 10, agent)
        order.status = OrderStatus.IN_PROGRESS
        world = _World(order, [agent], tasks)

        # The first time-indexed path cell is an intentional wait.  Consuming
        # that cell changes the plan countdown, but the plan itself was not
        # replaced and must not be labelled as replanning churn.
        agent.path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        before = compute_lyapunov_snapshot(world)
        agent.path_index = 1
        world.tick = 1
        after = compute_lyapunov_snapshot(world)

        progress = compute_productive_progress(before, after)
        self.assertEqual(progress.productive_total, 0.0)
        self.assertEqual(progress.replan_residual_total, 0.0)
        self.assertEqual(progress.route_plan_churn_total, 0.0)

    def test_deliver_exiting_does_not_restore_full_service_work(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        agent = _agent(position=(0, 10))
        agent.status = AgentStatus.EXITING
        tasks = _chain(order, 10, agent)
        tasks[0].status = TaskStatus.COMPLETED
        tasks[1].status = TaskStatus.IN_PROGRESS
        tasks[2].status = TaskStatus.ASSIGNED
        order.status = OrderStatus.IN_PROGRESS
        world = _World(order, [agent], tasks)
        snapshot = compute_lyapunov_snapshot(world)
        chain = snapshot.chains[(order.order_id, 10)]
        # Only the untouched RETURN leg remains.  The completed station service
        # duration must not be added again merely because DELIVER is EXITING.
        self.assertAlmostEqual(chain.remaining_work, 10.0)

    def test_cancelled_chain_does_not_receive_productive_credit(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        agent = _agent(position=(0, 0))
        tasks = _chain(order, 10, agent)
        order.status = OrderStatus.IN_PROGRESS
        world = _World(order, [agent], tasks)
        before = compute_lyapunov_snapshot(world)

        for task in tasks:
            task.status = TaskStatus.CANCELLED
        order.status = OrderStatus.PENDING
        order.pod_ids.clear()
        world.tick = 1
        after = compute_lyapunov_snapshot(world)
        progress = compute_productive_progress(before, after)
        self.assertEqual(progress.productive_total, 0.0)
        self.assertAlmostEqual(after.station_work[1], 1.0)

    def test_eta_uses_remaining_pick_wait_not_full_service_again(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        agent = _agent(position=(0, 4))
        agent.wait_ticks = 1
        tasks = _chain(order, 10, agent)
        tasks[0].status = TaskStatus.IN_PROGRESS
        order.status = OrderStatus.IN_PROGRESS
        world = _World(order, [agent], tasks)
        snapshot = compute_lyapunov_snapshot(
            world, LyapunovL0Config(eta_bin_edges=(7,))
        )
        self.assertEqual(snapshot.arrival_bins[1], (1.0, 0.0))

    def test_l0_is_non_negative_and_age_free(self):
        order = Order({"sku": 1}, station_id=1, created_at=-10000)
        order.pod_ids = [10]
        world = _World(order, [_agent()])
        a = compute_lyapunov_snapshot(world)
        order.created_at = 0
        b = compute_lyapunov_snapshot(world)
        self.assertGreaterEqual(a.total, 0.0)
        self.assertEqual(a.total, b.total)

    def test_unrelated_free_path_cannot_dilute_existing_traffic_conflict(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        a = _agent(1, (0, 0))
        b = _agent(2, (0, 1))
        a.path = [(0, 1)]
        b.path = [(0, 0)]
        world = _World(order, [a, b])
        config = LyapunovL0Config(traffic_weight=1.0)
        conflicted = compute_lyapunov_snapshot(world, config)

        c = _agent(3, (5, 5))
        c.path = [(5, 6)]
        world.agents.append(c)
        with_unrelated_path = compute_lyapunov_snapshot(world, config)
        self.assertEqual(
            conflicted.components["traffic"],
            with_unrelated_path.components["traffic"],
        )

    def test_same_future_vertex_is_a_traffic_reservation_conflict(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        a = _agent(1, (0, 0))
        b = _agent(2, (1, 1))
        a.path = [(0, 1)]
        b.path = [(0, 1)]
        snapshot = compute_lyapunov_snapshot(
            _World(order, [a, b]), LyapunovL0Config(traffic_weight=1.0)
        )
        self.assertGreater(
            snapshot.traffic_diagnostics["vertex_reservation_pressure"],
            0.0,
        )
        self.assertGreater(snapshot.components["traffic"], 0.0)

    def test_realised_blocked_move_has_nonzero_traffic_label(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        agent = _agent()
        agent.traffic_blocked_this_tick = True
        snapshot = compute_lyapunov_snapshot(
            _World(order, [agent]), LyapunovL0Config(traffic_weight=1.0)
        )
        self.assertEqual(
            snapshot.traffic_diagnostics["blocked_move_count"], 1.0
        )
        self.assertGreater(snapshot.components["traffic"], 0.0)

    def test_counterfactual_resolver_persists_blocked_traffic_signal(self):
        try:
            from WorldModel.data.counterfactual_rollout import _move_agents_step
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest("counterfactual integration requires torch")
            raise

        first = AgentState(1, (0, 0))
        second = AgentState(2, (1, 1))
        first.status = AgentStatus.MOVING
        second.status = AgentStatus.MOVING
        first.assign_path([(0, 1)])
        second.assign_path([(0, 1)])
        move_world = types.SimpleNamespace(
            agents=[first, second],
            get_agent=lambda agent_id: (
                first if agent_id == first.agent_id else second
            ),
            pod_state=types.SimpleNamespace(get_pod=lambda _pod_id: None),
        )
        _, blocked = _move_agents_step(move_world)
        self.assertEqual(blocked, 2)
        self.assertFalse(first.has_path)
        self.assertFalse(second.has_path)

        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        snapshot = compute_lyapunov_snapshot(
            _World(order, [first, second]),
            LyapunovL0Config(traffic_weight=1.0),
        )
        self.assertEqual(
            snapshot.traffic_diagnostics["blocked_move_count"], 2.0
        )
        self.assertGreater(snapshot.components["traffic"], 0.0)

    def test_realised_swap_survives_consumed_paths(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        world = _World(order, [_agent()])
        world.traffic_swap_conflicts_this_tick = 1
        snapshot = compute_lyapunov_snapshot(
            world, LyapunovL0Config(traffic_weight=1.0)
        )
        self.assertEqual(
            snapshot.traffic_diagnostics["realized_swap_conflict_count"],
            1.0,
        )
        self.assertGreater(snapshot.components["traffic"], 0.0)

    def test_intentional_wait_without_contention_is_not_traffic(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        agent = _agent(position=(0, 0))
        agent.path = [(0, 0), (0, 1)]
        snapshot = compute_lyapunov_snapshot(_World(order, [agent]))
        self.assertGreater(
            snapshot.traffic_diagnostics["planned_wait_count"], 0.0
        )
        self.assertEqual(snapshot.components["traffic"], 0.0)

    def test_arrived_station_robot_is_not_a_future_arrival(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        agent = _agent(position=(0, 9))
        tasks = _chain(order, 10, agent)
        tasks[0].status = TaskStatus.COMPLETED
        tasks[1].status = TaskStatus.IN_PROGRESS
        agent.status = AgentStatus.QUEUING
        order.status = OrderStatus.IN_PROGRESS
        snapshot = compute_lyapunov_snapshot(_World(order, [agent], tasks))
        self.assertEqual(snapshot.arrival_manifest, ())
        self.assertEqual(sum(snapshot.arrival_bins[1]), 0.0)

    def test_arrival_manifest_keeps_exact_path_aware_eta(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        agent = _agent(position=(0, 0))
        tasks = _chain(order, 10, agent)
        # One intentional wait plus four moves to the pod.  Manhattan-only
        # ETA would be 12; the exact active-path ETA is 13.
        agent.path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        order.status = OrderStatus.IN_PROGRESS
        snapshot = compute_lyapunov_snapshot(_World(order, [agent], tasks))
        self.assertEqual(len(snapshot.arrival_manifest), 1)
        record = snapshot.arrival_manifest[0]
        self.assertEqual(record["eta"], 13.0)
        self.assertIn("active_path", record["eta_source"])

    def test_arrival_eta_decreases_and_crosses_bin_with_path_progress(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        agent = _agent(position=(0, 0))
        agent.status = AgentStatus.CARRYING
        tasks = _chain(order, 10, agent)
        tasks[0].status = TaskStatus.COMPLETED
        tasks[1].status = TaskStatus.IN_PROGRESS
        order.status = OrderStatus.IN_PROGRESS
        agent.path = [(0, col) for col in range(1, 11)]
        world = _World(order, [agent], tasks)
        config = LyapunovL0Config(eta_bin_edges=(7,))

        before = compute_lyapunov_snapshot(world, config)
        agent.position = (0, 3)
        agent.path_index = 3
        world.tick = 3
        after = compute_lyapunov_snapshot(world, config)

        self.assertGreater(
            before.arrival_manifest[0]["eta"],
            after.arrival_manifest[0]["eta"],
        )
        self.assertEqual(before.arrival_manifest[0]["eta_bin"], 1)
        self.assertEqual(after.arrival_manifest[0]["eta_bin"], 0)

    def test_v3_snapshot_is_finite_and_carries_raw_manifests(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        snapshot = compute_lyapunov_snapshot(_World(order, [_agent()]))
        payload = snapshot.to_dict()
        self.assertEqual(payload["schema_version"], "lyapunov_l1_snapshot_v3")
        self.assertIn("arrival_manifest", payload)
        self.assertIn("traffic_diagnostics", payload)
        self.assertTrue(math.isfinite(snapshot.total))
        self.assertTrue(all(
            math.isfinite(float(value))
            for value in snapshot.traffic_diagnostics.values()
        ))

    def test_eta_preview_uses_arrival_barrier_not_distance_reward(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        world = _World(order, [_agent()])
        config = LyapunovL0Config(
            eta_bin_edges=(5, 10),
            arrival_capacity_scale=0.0,
        )
        snapshot = compute_lyapunov_snapshot(world, config)
        preview = preview_candidate_eta(
            world,
            station_id=1,
            robot_position=(0, 0),
            pod_position=(0, 2),
            station_position=(0, 5),
            config=config,
            snapshot=snapshot,
        )
        self.assertGreater(preview.delta_arrival_potential, 0.0)
        self.assertEqual(preview.arrival_after, preview.arrival_before + 1.0)

    def test_eta_shift_to_empty_bin_has_lower_barrier_delta(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        world = _World(order, [_agent()])
        config = LyapunovL0Config(
            eta_bin_edges=(5, 10),
            arrival_capacity_scale=0.0,
        )
        snapshot = replace(
            compute_lyapunov_snapshot(world, config),
            arrival_bins={1: (2.0, 0.0, 0.0)},
        )
        near = preview_candidate_eta(
            world, 1, (0, 0), (0, 0), (0, 3), config, snapshot
        )
        shifted = preview_candidate_eta(
            world, 1, (0, 0), (0, 4), (0, 8), config, snapshot
        )
        self.assertEqual(near.eta_bin, 0)
        self.assertEqual(shifted.eta_bin, 1)
        self.assertLess(
            shifted.delta_arrival_potential,
            near.delta_arrival_potential,
        )

    def test_arrival_barrier_uses_cumulative_deadline_pressure(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        world = _World(order, [_agent()])
        config = LyapunovL0Config(
            eta_bin_edges=(5,),
            arrival_capacity_scale=0.0,
        )
        early = _arrival_potential(world, {1: (2.0, 0.0)}, config)
        delayed = _arrival_potential(world, {1: (0.0, 2.0)}, config)
        self.assertGreater(early, delayed)
        self.assertAlmostEqual(early, 2.0)
        self.assertAlmostEqual(delayed, 1.0)

    def test_traffic_and_stall_are_diagnostics_not_default_potential(self):
        order = Order({"sku": 1}, station_id=1)
        order.pod_ids = [10]
        first = _agent(1, (0, 0))
        second = _agent(2, (1, 1))
        first.path = [(0, 1)]
        second.path = [(0, 1)]
        first.stationary_ticks = 100
        snapshot = compute_lyapunov_snapshot(_World(order, [first, second]))
        self.assertGreater(
            snapshot.traffic_diagnostics["vertex_reservation_pressure"], 0.0
        )
        self.assertGreater(snapshot.stationary_excess_rms, 0.0)
        self.assertEqual(snapshot.components["traffic"], 0.0)
        self.assertEqual(snapshot.components["stall"], 0.0)


if __name__ == "__main__":
    unittest.main()
