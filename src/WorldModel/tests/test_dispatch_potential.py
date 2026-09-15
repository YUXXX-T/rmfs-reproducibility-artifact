from types import SimpleNamespace

import pytest

from Policies.TaskAssigner.base_task_assigner import AssignmentContext
from Policies.TaskAssigner.context_assignment import (
    enumerate_pending_assignment_contexts,
    propose_fixed_assignment_contexts,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.world_model_task_assigner import (
    WorldModelTaskAssigner,
)
from WorldModel.core.dispatch_potential import (
    LIVENESS_CROSSING,
    DispatchDebtLedger,
    DispatchPotentialSnapshot,
    DispatchServiceDurations,
    StaticShortestPathDistance,
    context_key,
    exact_group_range,
)
from WorldState.task_state import TaskStatus, TaskType


class _OrderState:
    def __init__(self, orders):
        self.orders = {order.order_id: order for order in orders}

    def get_pending_orders(self):
        return list(self.orders.values())


class _PodState:
    def __init__(self, pods):
        self.pods = {pod.pod_id: pod for pod in pods}

    def get_pod(self, pod_id):
        return self.pods.get(int(pod_id))


class _StationState:
    def get_service_position(self, station_id):
        return (0, 2)

    def get_entry_position(self, station_id):
        return None

    def get_exit_position(self, station_id):
        return None


class _World:
    def __init__(self, orders, pods, agents, tasks=()):
        self.tick = 0
        self.order_state = _OrderState(orders)
        self.pod_state = _PodState(pods)
        self.station_state = _StationState()
        self.task_state = SimpleNamespace(
            tasks={index: task for index, task in enumerate(tasks)}
        )
        self.map_state = SimpleNamespace(station_positions={0: (0, 2)})
        self.agents = list(agents)

    def get_idle_agents(self):
        return list(self.agents)


def _order(order_id, pod_ids, created_at=0):
    return SimpleNamespace(
        order_id=order_id,
        pod_ids=list(pod_ids),
        station_id=0,
        sku_demands={"sku": 1},
        delivered_pod_ids=[],
        created_at=created_at,
    )


def _pod(pod_id=7, *, carried=False):
    return SimpleNamespace(
        pod_id=pod_id,
        current_position=(0, 1),
        home_position=(0, 3),
        is_carried=carried,
    )


def _agent(agent_id=0):
    return SimpleNamespace(agent_id=agent_id, position=(0, 0))


def _context(order_id=1, pod_id=7):
    return AssignmentContext(
        order_id=order_id,
        pod_id=pod_id,
        pod_location=(0, 1),
        station_id=0,
        station_location=(0, 2),
        entry_position=(0, 2),
        exit_position=None,
        return_location=(0, 3),
        order_size=1,
    )


def test_exact_group_range_has_unit_span_and_zero_tie():
    assert exact_group_range([4.0, 4.0]) == [0.0, 0.0]
    transformed = exact_group_range([-6.0, 2.0, 7.0])
    assert max(transformed) - min(transformed) == pytest.approx(1.0)
    assert sum(transformed) == pytest.approx(0.0)


def test_static_distance_uses_graph_not_manhattan():
    node_map = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
    adjacency = {0: [3], 3: [2], 2: [1], 1: [2]}
    distances = StaticShortestPathDistance(node_map, adjacency)
    assert distances.distance((0, 0), (0, 1)) == 3


def test_full_context_enumeration_preserves_distinct_shared_pod_debts():
    orders = [_order(1, [7]), _order(2, [7])]
    world = _World(orders, [_pod()], [_agent()])
    assigner = SimpleNamespace(pod_return_planner=None)

    legacy = propose_fixed_assignment_contexts(assigner, world)
    full = enumerate_pending_assignment_contexts(
        assigner,
        world,
        include_temporarily_unavailable=True,
        deduplicate_pods=False,
    )

    assert [(row.order_id, row.pod_id) for row in legacy] == [(1, 7)]
    assert [(row.order_id, row.pod_id) for row in full] == [(1, 7), (2, 7)]


def test_full_context_removes_own_pipeline_but_keeps_other_reserved_chain():
    orders = [_order(1, [7]), _order(2, [7])]
    active = SimpleNamespace(
        order_id=1,
        pod_id=7,
        status=TaskStatus.ASSIGNED,
        task_type=TaskType.PICK,
    )
    world = _World(orders, [_pod()], [_agent()], tasks=[active])
    assigner = SimpleNamespace(pod_return_planner=None)

    full = enumerate_pending_assignment_contexts(
        assigner,
        world,
        include_temporarily_unavailable=True,
        deduplicate_pods=False,
    )

    assert [(row.order_id, row.pod_id) for row in full] == [(2, 7)]
    assert propose_fixed_assignment_contexts(assigner, world) == []


def test_crossing_overcomes_worst_case_unit_wm_preference_for_defer():
    ctx = _context()
    key = context_key(ctx)
    free_flow = 44.0
    increment = 1.0 / free_flow
    snapshot = DispatchPotentialSnapshot(
        tick=10,
        contexts=(ctx,),
        pending_keys=frozenset({key}),
        pressure_keys=frozenset({key}),
        eligible_free_flow={key: free_flow},
        debt_before={key: LIVENESS_CROSSING - increment},
        continuous_eligible_mass_before={key: 0.0},
        continuous_eligible_ticks_before={key: 0},
        order_age={key: 10},
        pending_pressure={0: 0.0},
        station_capacity=12.0,
        unresolved_pending_orders=0,
    )
    terms = snapshot.terms(ctx)
    assign_gr, defer_gr = exact_group_range([1.0, 0.0])

    assert terms.crossing_reached
    assert assign_gr + terms.delta_l_assign <= defer_gr + terms.delta_l_defer


def test_debt_updates_once_per_tick_and_assignment_clears_it():
    order = _order(1, [7])
    world = _World([order], [_pod()], [_agent()])
    ctx = _context()
    graph = StaticShortestPathDistance(
        {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3},
        {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]},
    )
    durations = DispatchServiceDurations(2, 5, 2)
    ledger = DispatchDebtLedger()

    snapshot = ledger.snapshot(world, [ctx], graph, durations)
    free_flow = snapshot.eligible_free_flow[context_key(ctx)]
    first = ledger.finalize(snapshot, assigned_keys=[])

    assert ledger.debt[context_key(ctx)] == pytest.approx(1.0 / free_flow)
    assert first["eligible_debt_increments"] == 1
    with pytest.raises(RuntimeError, match="already finalized"):
        ledger.finalize(snapshot, assigned_keys=[])

    world.tick = 1
    next_snapshot = ledger.snapshot(world, [ctx], graph, durations)
    ledger.finalize(next_snapshot, assigned_keys=[context_key(ctx)])
    assert context_key(ctx) not in ledger.debt


@pytest.mark.parametrize(
    ("mode", "expected_defer"),
    [("diagnostic", True), ("bridge", False)],
)
def test_dispatch_diagnostic_is_nonperturbing_and_bridge_uses_composite(
    mode,
    expected_defer,
):
    ctx = _context()
    key = context_key(ctx)
    free_flow = 44.0
    assigner = WorldModelTaskAssigner(
        checkpoint_path="unused-in-unit-test.pt",
        include_no_assign_candidate=True,
        dispatch_potential_mode=mode,
    )
    assigner._dispatch_snapshot = DispatchPotentialSnapshot(
        tick=10,
        contexts=(ctx,),
        pending_keys=frozenset({key}),
        pressure_keys=frozenset({key}),
        eligible_free_flow={key: free_flow},
        debt_before={key: LIVENESS_CROSSING},
        continuous_eligible_mass_before={key: LIVENESS_CROSSING},
        continuous_eligible_ticks_before={key: 62},
        order_age={key: 100},
        pending_pressure={0: 1.0 / 12.0},
        station_capacity=12.0,
        unresolved_pending_orders=0,
    )
    robots = [
        {"score": 2.0, "best_robot": 1},
        {"score": 3.0, "best_robot": 2},
    ]
    defer = {"score": 0.0, "best_robot": None}

    decision = assigner._apply_dispatch_potential(ctx, robots, defer)

    assert decision["raw_defer"] is True
    assert decision["bridge_defer"] is False
    assert decision["actual_defer"] is expected_defer
    assert assigner.stats["dispatch_crossing_violations"] == 0
    assert max(
        row["dispatch_wm_group_range"] for row in robots + [defer]
    ) - min(
        row["dispatch_wm_group_range"] for row in robots + [defer]
    ) <= 1.0
