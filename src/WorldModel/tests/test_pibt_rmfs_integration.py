import types
import unittest

from Engine.simulation_engine import SimulationEngine
from Policies.PathPlanner.PIBTPlanner.pibt_path_planner import PIBTPlanner
from WorldState.agent_state import AgentState, AgentStatus
from WorldState.task_state import Task, TaskState, TaskStatus, TaskType
from WorldModel.data.counterfactual_rollout import _plan_and_activate_step


class _GridMap:
    def __init__(self, rows, cols, obstacles=()):
        self.rows = rows
        self.cols = cols
        self.obstacles = set(obstacles)

    def get_neighbors(self, row, col):
        result = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (row + dr, col + dc)
            if (
                0 <= nxt[0] < self.rows
                and 0 <= nxt[1] < self.cols
                and nxt not in self.obstacles
            ):
                result.append(nxt)
        return result


class _EmptyTaskState:
    def get_active_task_for_agent(self, agent_id):
        return None

    def get_next_task_for_agent(self, agent_id):
        return None


def _world(agents, rows=3, cols=4, task_state=None):
    return types.SimpleNamespace(
        tick=0,
        agents=agents,
        map_state=_GridMap(rows, cols),
        task_state=task_state or _EmptyTaskState(),
        pod_state=types.SimpleNamespace(pods={}),
    )


def _moving(agent_id, position, *, carrying=False):
    agent = AgentState(agent_id=agent_id, start_position=position)
    agent.status = AgentStatus.CARRYING if carrying else AgentStatus.MOVING
    if carrying:
        agent.carried_pod_id = 1000 + agent_id
    return agent


class PIBTRMFSIntegrationTests(unittest.TestCase):
    def test_batch_honours_extra_blocked_cells(self):
        agent = _moving(0, (1, 0))
        world = _world([agent], rows=3, cols=3)
        planner = PIBTPlanner(seed=7)

        path = planner.plan_batch(
            [(agent, (1, 2))],
            world,
            extra_blocked={(1, 1)},
        )[0]

        self.assertEqual(len(path), 1)
        self.assertNotEqual(path[0], (1, 1))
        self.assertTrue(planner.runtime_metrics()["pibt_last_batch_audit"]["passed"])

    def test_nonrequested_station_zone_robot_is_pinned(self):
        mover = _moving(0, (1, 0))
        station_robot = _moving(1, (1, 1))
        station_robot.status = AgentStatus.EXITING
        world = _world([mover, station_robot], rows=3, cols=3)
        planner = PIBTPlanner(seed=3)

        path = planner.plan_batch([(mover, (1, 2))], world)[0]

        self.assertNotEqual(path[0], station_robot.position)
        self.assertEqual(planner._decided[station_robot.agent_id], (1, 1))

    def test_explicit_batch_goals_override_deliver_service_destinations(self):
        first = _moving(0, (0, 0), carrying=True)
        second = _moving(1, (2, 2), carrying=True)
        task_state = TaskState()
        for agent, misleading_destination in (
            (first, (2, 0)),
            (second, (0, 2)),
        ):
            task = Task(
                TaskType.DELIVER,
                order_id=agent.agent_id,
                pod_id=agent.carried_pod_id,
                source=agent.position,
                destination=misleading_destination,
            )
            task.agent_id = agent.agent_id
            task.status = TaskStatus.IN_PROGRESS
            task_state.add_task(task)
        world = _world([first, second], rows=3, cols=3, task_state=task_state)
        planner = PIBTPlanner(seed=11)
        explicit_goals = {0: (0, 2), 1: (2, 0)}

        paths = planner.plan_batch(
            [(first, explicit_goals[0]), (second, explicit_goals[1])],
            world,
        )

        for agent in (first, second):
            before = abs(agent.position[0] - explicit_goals[agent.agent_id][0]) + abs(
                agent.position[1] - explicit_goals[agent.agent_id][1]
            )
            nxt = paths[agent.agent_id][0]
            after = abs(nxt[0] - explicit_goals[agent.agent_id][0]) + abs(
                nxt[1] - explicit_goals[agent.agent_id][1]
            )
            self.assertLess(after, before)

    def test_adjacent_opposite_goals_do_not_swap(self):
        left = _moving(0, (0, 0))
        right = _moving(1, (0, 1))
        world = _world([left, right], rows=1, cols=2)
        planner = PIBTPlanner(seed=5)

        paths = planner.plan_batch(
            [(left, right.position), (right, left.position)],
            world,
        )

        self.assertFalse(
            paths[0][0] == right.position and paths[1][0] == left.position
        )
        audit = planner.runtime_metrics()["pibt_last_batch_audit"]
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["swap_conflicts"], [])

    def test_priority_inheritance_pushes_a_three_robot_chain(self):
        agents = [_moving(i, (0, i)) for i in range(3)]
        world = _world(agents, rows=1, cols=4)
        planner = PIBTPlanner(seed=13)
        planner._priorities[0] = 100

        paths = planner.plan_batch(
            [(agents[0], (0, 3)), (agents[1], (0, 3)), (agents[2], (0, 3))],
            world,
        )

        self.assertEqual(paths[0], [(0, 1)])
        self.assertEqual(paths[1], [(0, 2)])
        self.assertEqual(paths[2], [(0, 3)])
        metrics = planner.runtime_metrics()
        self.assertGreaterEqual(metrics["pibt_priority_inheritance_calls"], 2)
        self.assertEqual(metrics["pibt_validation_failures"], 0)

    def test_engine_uses_batch_interface_without_touching_sequential_planners(self):
        agent = _moving(0, (1, 0))
        agent.status = AgentStatus.MOVING_TO_POD
        task_state = TaskState()
        pick = Task(
            TaskType.PICK,
            order_id=1,
            pod_id=2,
            source=agent.position,
            destination=(1, 1),
        )
        pick.agent_id = agent.agent_id
        pick.status = TaskStatus.ASSIGNED
        task_state.add_task(pick)

        world = types.SimpleNamespace(
            agents=[agent],
            task_state=task_state,
            station_state=types.SimpleNamespace(stations={}),
        )

        class _RecordingBatchPlanner:
            supports_batch_planning = True

            def __init__(self):
                self.calls = []

            def plan(self, *args, **kwargs):
                raise AssertionError("sequential plan() must not be used")

            def plan_batch(self, requests, world_state, extra_blocked=None):
                self.calls.append((list(requests), extra_blocked))
                return {
                    request_agent.agent_id: [(1, 1)]
                    for request_agent, _ in requests
                }

        planner = _RecordingBatchPlanner()
        engine = SimulationEngine.__new__(SimulationEngine)
        engine.world = world
        engine.path_planner = planner
        engine.logger = types.SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
        )
        engine._record_path_len = lambda *args, **kwargs: None

        engine._plan_and_activate(0)

        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(planner.calls[0][0][0][1], (1, 1))
        self.assertEqual(pick.status, TaskStatus.IN_PROGRESS)
        self.assertEqual(agent.path, [(1, 1)])

    def test_engine_preserves_sequential_interface_for_nonbatch_planners(self):
        agent = _moving(0, (1, 0))
        agent.status = AgentStatus.MOVING_TO_POD
        task_state = TaskState()
        pick = Task(
            TaskType.PICK,
            order_id=1,
            pod_id=2,
            source=agent.position,
            destination=(1, 1),
        )
        pick.agent_id = agent.agent_id
        pick.status = TaskStatus.ASSIGNED
        task_state.add_task(pick)
        world = types.SimpleNamespace(
            agents=[agent],
            task_state=task_state,
            station_state=types.SimpleNamespace(stations={}),
        )

        class _RecordingSequentialPlanner:
            def __init__(self):
                self.calls = []

            def plan(self, request_agent, goal, world_state, extra_blocked=None):
                self.calls.append((request_agent.agent_id, goal, extra_blocked))
                return [(1, 1)]

            def plan_batch(self, *args, **kwargs):
                raise AssertionError("nonbatch planner must stay on plan()")

        planner = _RecordingSequentialPlanner()
        engine = SimulationEngine.__new__(SimulationEngine)
        engine.world = world
        engine.path_planner = planner
        engine.logger = types.SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
        )
        engine._record_path_len = lambda *args, **kwargs: None

        engine._plan_and_activate(0)

        self.assertEqual(planner.calls, [(0, (1, 1), None)])
        self.assertEqual(agent.path, [(1, 1)])

    def test_valid_pibt_wait_is_stuck_but_not_a_planning_failure(self):
        agent = _moving(0, (1, 0))
        agent.status = AgentStatus.MOVING_TO_POD
        agent.plan_failed_streak = 4
        task_state = TaskState()
        pick = Task(
            TaskType.PICK,
            order_id=1,
            pod_id=2,
            source=agent.position,
            destination=(1, 1),
        )
        pick.agent_id = agent.agent_id
        pick.status = TaskStatus.ASSIGNED
        task_state.add_task(pick)
        world = types.SimpleNamespace(
            agents=[agent],
            task_state=task_state,
            station_state=types.SimpleNamespace(stations={}),
            pod_state=types.SimpleNamespace(get_pod=lambda pod_id: None),
        )

        planner = types.SimpleNamespace(
            supports_batch_planning=True,
            plan_batch=lambda requests, world_state, extra_blocked=None: {
                request_agent.agent_id: [request_agent.position]
                for request_agent, _ in requests
            },
        )
        engine = SimulationEngine.__new__(SimulationEngine)
        engine.world = world
        engine.path_planner = planner
        engine.logger = types.SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
        )
        engine._record_path_len = lambda *args, **kwargs: None

        engine._plan_and_activate(0)
        self.assertEqual(agent.plan_failed_streak, 0)
        self.assertEqual(agent.path, [agent.position])

        engine._move_agents(0)
        self.assertTrue(agent.stuck_this_tick)
        self.assertFalse(agent.moved_this_tick)
        self.assertEqual(agent.plan_failed_streak, 0)

    def test_counterfactual_rollout_uses_the_same_batch_contract(self):
        agent = _moving(0, (1, 0))
        agent.status = AgentStatus.MOVING_TO_POD
        task_state = TaskState()
        pick = Task(
            TaskType.PICK,
            order_id=1,
            pod_id=2,
            source=agent.position,
            destination=(1, 1),
        )
        pick.agent_id = agent.agent_id
        pick.status = TaskStatus.ASSIGNED
        task_state.add_task(pick)

        world = types.SimpleNamespace(
            agents=[agent],
            task_state=task_state,
            station_state=types.SimpleNamespace(stations={}),
        )
        planner = types.SimpleNamespace(
            supports_batch_planning=True,
            plan=lambda *args, **kwargs: self.fail(
                "counterfactual sequential plan() must not be used"
            ),
            plan_batch=lambda requests, world_state, extra_blocked=None: {
                request_agent.agent_id: [(1, 1)]
                for request_agent, _ in requests
            },
        )
        config = types.SimpleNamespace(
            simulation=types.SimpleNamespace(
                pickup_duration=2,
                station_process_duration=5,
                dropoff_duration=2,
            )
        )

        _plan_and_activate_step(world, planner, tick=0, config=config)

        self.assertEqual(pick.status, TaskStatus.IN_PROGRESS)
        self.assertEqual(agent.path, [(1, 1)])


if __name__ == "__main__":
    unittest.main()
