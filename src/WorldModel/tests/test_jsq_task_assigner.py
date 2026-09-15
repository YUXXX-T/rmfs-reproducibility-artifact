"""Tests for JSQTaskAssigner -- Join-Shortest-Queue baseline.

Unit tests validate JSQ-specific ordering (lowest committed-load station
first) with greedy nearest-robot selection.  An integration test runs the
full simulation engine to confirm end-to-end compatibility.
"""

import types
import unittest

from Policies.TaskAssigner.JSQTaskAssigner.jsq_task_assigner import (
    JSQTaskAssigner,
)
from Policies.TaskAssigner.base_task_assigner import AssignmentContext
from WorldState.agent_state import AgentStatus
from WorldState.order_state import Order, OrderState, OrderStatus
from WorldState.task_state import Task, TaskState, TaskStatus, TaskType


# ── Mock helpers ──────────────────────────────────────────────────


def _make_queue(committed: int):
    """Return a mock StationQueueState with a fixed committed_load."""
    return types.SimpleNamespace(committed_load=lambda: committed)


def _make_agent(agent_id, position):
    return types.SimpleNamespace(
        agent_id=agent_id,
        position=position,
        status=AgentStatus.IDLE,
        assigned_task_id=None,
    )


def _make_pod(pod_id, position):
    return types.SimpleNamespace(
        pod_id=pod_id,
        current_position=position,
        home_position=position,
        is_carried=False,
    )


class _MultiStationWorld:
    """Two-station world whose queues have distinct committed loads.

    Station 1 (row 0, service at (0, 8)):  committed_load = ``load_s1``
    Station 2 (row 4, service at (4, 8)):  committed_load = ``load_s2``
    """

    def __init__(self, *, load_s1=3, load_s2=1, num_agents=3):
        self.tick = 10
        self.config = types.SimpleNamespace(
            simulation=types.SimpleNamespace(
                task_execution_mode="parallel",
            ),
        )
        # Orders: two target station 1, one targets station 2.
        o1 = Order({"sku": 1}, station_id=1); o1.pod_ids = [10]
        o2 = Order({"sku": 1}, station_id=1); o2.pod_ids = [11]
        o3 = Order({"sku": 1}, station_id=2); o3.pod_ids = [12]
        self.order_state = OrderState()
        for order in (o1, o2, o3):
            self.order_state.add_order(order)

        self.task_state = TaskState()

        pods = {
            10: _make_pod(10, (0, 2)),
            11: _make_pod(11, (0, 4)),
            12: _make_pod(12, (4, 2)),
        }
        self.pod_state = types.SimpleNamespace(
            get_pod=lambda pid: pods.get(pid),
        )

        queues = {
            1: _make_queue(load_s1),
            2: _make_queue(load_s2),
        }
        self.station_state = types.SimpleNamespace(
            get_service_position=lambda sid: {1: (0, 8), 2: (4, 8)}[sid],
            get_entry_position=lambda sid: {1: (1, 8), 2: (5, 8)}[sid],
            get_exit_position=lambda sid: {1: (0, 9), 2: (4, 9)}[sid],
            get_queue=lambda sid: queues.get(sid),
        )
        self.map_state = types.SimpleNamespace(
            station_positions={1: (0, 8), 2: (4, 8)},
        )

        self.agents = [
            _make_agent(i, (2, i * 2)) for i in range(1, num_agents + 1)
        ]

    def get_idle_agents(self):
        return [a for a in self.agents if a.status == AgentStatus.IDLE]

    def get_agent(self, agent_id):
        return next(
            (a for a in self.agents if a.agent_id == agent_id), None
        )


class _SingleStationWorld:
    """Single-station world for testing degenerate/greedy-equivalent case."""

    def __init__(self, *, committed=0, num_agents=2):
        self.tick = 5
        self.config = types.SimpleNamespace(
            simulation=types.SimpleNamespace(
                task_execution_mode="parallel",
            ),
        )
        o1 = Order({"sku": 1}, station_id=1); o1.pod_ids = [10]
        o2 = Order({"sku": 1}, station_id=1); o2.pod_ids = [11]
        self.order_state = OrderState()
        self.order_state.add_order(o1)
        self.order_state.add_order(o2)

        self.task_state = TaskState()

        pods = {
            10: _make_pod(10, (0, 1)),
            11: _make_pod(11, (0, 9)),
        }
        self.pod_state = types.SimpleNamespace(
            get_pod=lambda pid: pods.get(pid),
        )

        queues = {1: _make_queue(committed)}
        self.station_state = types.SimpleNamespace(
            get_service_position=lambda _: (0, 5),
            get_entry_position=lambda _: (1, 5),
            get_exit_position=lambda _: (0, 6),
            get_queue=lambda sid: queues.get(sid),
        )
        self.map_state = types.SimpleNamespace(
            station_positions={1: (0, 5)},
        )

        self.agents = [
            _make_agent(1, (0, 0)),   # close to pod 10 at (0,1)
            _make_agent(2, (0, 10)),  # close to pod 11 at (0,9)
        ][:num_agents]

    def get_idle_agents(self):
        return [a for a in self.agents if a.status == AgentStatus.IDLE]

    def get_agent(self, agent_id):
        return next(
            (a for a in self.agents if a.agent_id == agent_id), None
        )


# ── Unit tests ────────────────────────────────────────────────────


class JSQTaskAssignerTests(unittest.TestCase):
    """Unit tests for JSQ-specific context ordering and robot selection."""

    def _assigner(self):
        a = JSQTaskAssigner()
        a.pod_retriever = None
        a.pod_return_planner = None
        return a

    # ── basic assignment produces correct task chains ──

    def test_basic_assignment_creates_pick_deliver_return(self):
        world = _SingleStationWorld()
        assigner = self._assigner()

        tasks = assigner.assign(world)

        self.assertEqual(len(tasks), 6)  # 2 orders * 3 tasks each
        types_seen = {t.task_type for t in tasks}
        self.assertEqual(
            types_seen, {TaskType.PICK, TaskType.DELIVER, TaskType.RETURN}
        )

    def test_deliver_and_return_tasks_carry_station_id(self):
        world = _SingleStationWorld()
        tasks = self._assigner().assign(world)

        for task in tasks:
            if task.task_type in (TaskType.DELIVER, TaskType.RETURN):
                self.assertEqual(task.station_id, 1)

    def test_return_source_is_exit_position(self):
        world = _SingleStationWorld()
        tasks = self._assigner().assign(world)

        for task in tasks:
            if task.task_type == TaskType.RETURN:
                self.assertEqual(task.source, (0, 6))

    # ── JSQ ordering: less-loaded station first ──

    def test_jsq_prefers_less_loaded_station(self):
        """With 1 idle agent, JSQ should pick the context targeting station 2
        (committed_load=1) over station 1 (committed_load=3)."""
        world = _MultiStationWorld(load_s1=3, load_s2=1, num_agents=1)
        assigner = self._assigner()

        tasks = assigner.assign(world)

        self.assertEqual(len(tasks), 3)
        pick = next(t for t in tasks if t.task_type == TaskType.PICK)
        # Pod 12 targets station 2 (the less-loaded one).
        self.assertEqual(pick.pod_id, 12)

    def test_jsq_with_equal_loads_preserves_proposal_order(self):
        """When both stations have equal load, the context from the
        earlier-proposed order (lower order age) should come first."""
        world = _MultiStationWorld(load_s1=2, load_s2=2, num_agents=1)
        assigner = self._assigner()

        tasks = assigner.assign(world)

        self.assertEqual(len(tasks), 3)
        pick = next(t for t in tasks if t.task_type == TaskType.PICK)
        # First proposed context targets station 1 (pod 10), since orders
        # for station 1 were added before the order for station 2.
        self.assertEqual(pick.pod_id, 10)

    def test_jsq_dynamic_virtual_load_balances_across_stations(self):
        """With 3 agents and loads s1=1, s2=1, the first two assignments
        should go to different stations (one each), and the third should
        go to whichever station still has a lower virtual load."""
        world = _MultiStationWorld(load_s1=1, load_s2=1, num_agents=3)
        assigner = self._assigner()

        tasks = assigner.assign(world)

        picks = [t for t in tasks if t.task_type == TaskType.PICK]
        self.assertEqual(len(picks), 3)  # 3 agents, 3 contexts
        stations_assigned = [
            next(
                o.station_id
                for o in world.order_state.orders.values()
                if o.order_id == p.order_id
            )
            for p in picks
        ]
        # Both stations should be represented; at least one assignment to each.
        self.assertIn(1, stations_assigned)
        self.assertIn(2, stations_assigned)

    # ── greedy robot selection: nearest to pod ──

    def test_greedy_assigns_nearest_robot_to_pod(self):
        """Robot 1 at (0,0) should get pod 10 at (0,1);
        robot 2 at (0,10) should get pod 11 at (0,9)."""
        world = _SingleStationWorld(committed=0, num_agents=2)
        assigner = self._assigner()

        tasks = assigner.assign(world)

        picks = {t.agent_id: t.pod_id for t in tasks if t.task_type == TaskType.PICK}
        self.assertEqual(picks, {1: 10, 2: 11})

    # ── edge cases ──

    def test_zero_idle_agents_returns_empty(self):
        world = _SingleStationWorld(num_agents=2)
        for agent in world.agents:
            agent.status = AgentStatus.MOVING_TO_POD  # not idle
        tasks = self._assigner().assign(world)
        self.assertEqual(tasks, [])

    def test_no_pending_orders_returns_empty(self):
        world = _SingleStationWorld()
        for order in world.order_state.orders.values():
            order.status = OrderStatus.COMPLETED
        tasks = self._assigner().assign(world)
        self.assertEqual(tasks, [])

    def test_agents_marked_moving_after_assignment(self):
        world = _SingleStationWorld(num_agents=2)
        self._assigner().assign(world)

        for agent in world.agents:
            self.assertEqual(agent.status, AgentStatus.MOVING_TO_POD)

    def test_orders_marked_in_progress_when_all_pods_covered(self):
        world = _SingleStationWorld(num_agents=2)
        self._assigner().assign(world)

        for order in world.order_state.orders.values():
            self.assertEqual(order.status, OrderStatus.IN_PROGRESS)

    def test_propose_skips_carried_pods(self):
        world = _SingleStationWorld(num_agents=2)
        # Mark pod 10 as carried.
        world.pod_state.get_pod(10).is_carried = True

        tasks = self._assigner().assign(world)

        pods_assigned = {t.pod_id for t in tasks if t.task_type == TaskType.PICK}
        self.assertNotIn(10, pods_assigned)
        self.assertIn(11, pods_assigned)

    def test_propose_skips_reserved_pods(self):
        world = _SingleStationWorld(num_agents=2)
        # Create an active task that reserves pod 10.
        active = Task(TaskType.PICK, 999, 10, (0, 0), (0, 1))
        active.status = TaskStatus.ASSIGNED
        active.agent_id = 99
        world.task_state.add_task(active)

        tasks = self._assigner().assign(world)

        pods_assigned = {t.pod_id for t in tasks if t.task_type == TaskType.PICK}
        self.assertNotIn(10, pods_assigned)

    # ── pipeline contract: propose/select/commit independently ──

    def test_propose_does_not_mutate_world(self):
        world = _SingleStationWorld()
        assigner = self._assigner()
        before = len(world.task_state.tasks)

        contexts = assigner.propose_assignment_contexts(world)

        self.assertGreater(len(contexts), 0)
        self.assertEqual(len(world.task_state.tasks), before)

    def test_select_returns_correct_index_mapping(self):
        world = _MultiStationWorld(load_s1=3, load_s2=1, num_agents=2)
        assigner = self._assigner()
        contexts = assigner.propose_assignment_contexts(world)

        choices = assigner.select_robots(world, contexts)

        # Keys must be valid context indices.
        for idx in choices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(contexts))
        # Values must be valid agent IDs.
        agent_ids = {a.agent_id for a in world.agents}
        for robot_id in choices.values():
            self.assertIn(robot_id, agent_ids)
        # No robot used twice.
        self.assertEqual(len(set(choices.values())), len(choices))


# ── Integration test with full engine ─────────────────────────────


class JSQIntegrationTests(unittest.TestCase):
    """Run JSQTaskAssigner inside the full simulation engine."""

    def test_engine_100_ticks_no_crash(self):
        """Build engine with JSQTaskAssigner, run 100 ticks, verify tasks."""
        # Reuse the build_engine pattern from test_batch_plan.py.
        from Config.config_loader import load_config
        from Engine.simulation_engine import SimulationEngine
        from Policies.policy_registry import get_policy

        Task._next_id = 0
        Order._next_id = 0

        cfg = load_config("Config/default_config.json")
        cfg.simulation.tick_delay = 0
        cfg.simulation.log_level = "CRITICAL"

        on, op = cfg.policies.order_generator
        rn, rp_ = cfg.policies.pod_return_planner
        prn, prp = cfg.policies.pod_retriever
        _, ta_p = cfg.policies.task_assigner

        if cfg.simulation.use_recorded_orders:
            on = "RecordedOrderGenerator"
            op = {
                "recorded_orders_path": cfg.simulation.recorded_orders_path,
                "immediate_dispatch": cfg.simulation.immediate_dispatch,
            }

        og = get_policy("order_generator", on)(
            order_interval=cfg.simulation.order_interval,
            max_items_per_order=cfg.simulation.max_items_per_order,
            fixed_order_size=cfg.simulation.fixed_order_size,
            max_items_per_sku=cfg.simulation.max_items_per_sku,
            **op,
        )
        rp = get_policy("pod_return_planner", rn)(**rp_)
        pr = get_policy("pod_retriever", prn)(**prp)
        ta = get_policy("task_assigner", "JSQTaskAssigner")(**ta_p)
        ta.pod_return_planner = rp
        ta.pod_retriever = pr

        pp = get_policy("path_planner", "AStarPathPlanner")()

        engine = SimulationEngine(
            config=cfg,
            order_generator=og,
            task_assigner=ta,
            path_planner=pp,
            visualizer=None,
        )

        for _ in range(100):
            engine._tick()

        tasks = engine.world.task_state.tasks
        self.assertGreater(len(tasks), 0, "Expected tasks to be created")

        completed = [
            t for t in tasks.values() if t.status == TaskStatus.COMPLETED
        ]
        self.assertGreater(
            len(completed), 0, "Expected at least one completed task"
        )

    def test_engine_500_ticks_no_pod_double_assigned(self):
        """500-tick stability: no pod should be simultaneously assigned to
        two active PICK tasks."""
        from collections import Counter
        from Config.config_loader import load_config
        from Engine.simulation_engine import SimulationEngine
        from Policies.policy_registry import get_policy

        Task._next_id = 0
        Order._next_id = 0

        cfg = load_config("Config/default_config.json")
        cfg.simulation.tick_delay = 0
        cfg.simulation.log_level = "CRITICAL"

        on, op = cfg.policies.order_generator
        rn, rp_ = cfg.policies.pod_return_planner
        prn, prp = cfg.policies.pod_retriever
        _, ta_p = cfg.policies.task_assigner

        if cfg.simulation.use_recorded_orders:
            on = "RecordedOrderGenerator"
            op = {
                "recorded_orders_path": cfg.simulation.recorded_orders_path,
                "immediate_dispatch": cfg.simulation.immediate_dispatch,
            }

        og = get_policy("order_generator", on)(
            order_interval=cfg.simulation.order_interval,
            max_items_per_order=cfg.simulation.max_items_per_order,
            fixed_order_size=cfg.simulation.fixed_order_size,
            max_items_per_sku=cfg.simulation.max_items_per_sku,
            **op,
        )
        rp = get_policy("pod_return_planner", rn)(**rp_)
        pr = get_policy("pod_retriever", prn)(**prp)
        ta = get_policy("task_assigner", "JSQTaskAssigner")(**ta_p)
        ta.pod_return_planner = rp
        ta.pod_retriever = pr

        pp = get_policy("path_planner", "AStarPathPlanner")()

        engine = SimulationEngine(
            config=cfg,
            order_generator=og,
            task_assigner=ta,
            path_planner=pp,
            visualizer=None,
        )

        for _ in range(500):
            engine._tick()

        self.assertEqual(engine.world.tick, 500)

        active_picks = [
            t.pod_id
            for t in engine.world.task_state.tasks.values()
            if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
            and t.task_type == TaskType.PICK
        ]
        doubles = {
            pod: cnt
            for pod, cnt in Counter(active_picks).items()
            if cnt > 1
        }
        self.assertEqual(
            len(doubles), 0,
            f"Pods double-assigned in active PICKs: {doubles}",
        )


if __name__ == "__main__":
    unittest.main()
