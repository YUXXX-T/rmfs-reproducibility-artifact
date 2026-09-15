"""Analytic dispatcher potential for the Phase-C Round-2 Gate-1 bridge.

The World Model remains a bounded, within-context short-horizon congestion
score.  This module supplies the separate liveness term described in
``phase_c_round2_dispatch_potential_plan_v1.md``.  It contains no learned
parameters and never mutates simulator state.

One pending order/pod chain carries one unit of dispatch work mass.  Service
debt is measured in free-flow chain-time units and is updated exactly once per
simulator tick by :class:`DispatchDebtLedger`.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


ContextKey = Tuple[int, int, int]
Position = Tuple[int, int]

DISPATCH_POTENTIAL_SCHEMA_VERSION = "phase_c_dispatch_potential_v1"
DISPATCH_TRACE_SCHEMA_VERSION = "phase_c_dispatch_bridge_trace_v1"
LIVENESS_CROSSING = math.sqrt(2.0)
DOMINANCE_DEBT = 1.5
MAX_ELIGIBLE_DEFER_MASS = 2.0 * LIVENESS_CROSSING


def context_key(context) -> ContextKey:
    """Return the stable order/pod/station identity for one context."""

    return (
        int(context.order_id),
        int(context.pod_id),
        int(context.station_id),
    )


def exact_group_range(values: Sequence[float]) -> list[float]:
    """Frozen no-gap group transform already used by Layers 3--5.

    The output span is exactly one for every non-constant group.  Constant
    groups map to zero and are resolved by the assignment-favouring tie-break.
    """

    if not values:
        return []
    numbers = [float(value) for value in values]
    mean = sum(numbers) / float(len(numbers))
    value_range = max(numbers) - min(numbers)
    if value_range == 0.0:
        return [0.0 for _ in numbers]
    return [(value - mean) / value_range for value in numbers]


@dataclass(frozen=True)
class DispatchServiceDurations:
    pickup: int
    station_process: int
    dropoff: int

    def __post_init__(self) -> None:
        if min(int(self.pickup), int(self.station_process), int(self.dropoff)) < 0:
            raise ValueError("dispatch service durations must be non-negative")

    @classmethod
    def from_world(cls, world) -> "DispatchServiceDurations":
        simulation = getattr(getattr(world, "config", None), "simulation", None)
        if simulation is None:
            raise ValueError("world.config.simulation is required")
        return cls(
            pickup=int(getattr(simulation, "pickup_duration")),
            station_process=int(getattr(simulation, "station_process_duration")),
            dropoff=int(getattr(simulation, "dropoff_duration")),
        )


class StaticShortestPathDistance:
    """Lazy all-destination BFS cache over the immutable warehouse graph."""

    def __init__(
        self,
        node_map: Mapping[Position, int],
        adjacency: Mapping[int, Sequence[int]],
    ) -> None:
        self.node_map = {
            (int(position[0]), int(position[1])): int(node_id)
            for position, node_id in node_map.items()
        }
        self.adjacency = {
            int(node_id): tuple(int(value) for value in neighbours)
            for node_id, neighbours in adjacency.items()
        }
        self._cache: Dict[int, Dict[int, int]] = {}

    def _distances_from(self, source: int) -> Dict[int, int]:
        cached = self._cache.get(int(source))
        if cached is not None:
            return cached
        distances = {int(source): 0}
        queue = deque([int(source)])
        while queue:
            current = queue.popleft()
            next_distance = distances[current] + 1
            for neighbour in self.adjacency.get(current, ()):
                if neighbour in distances:
                    continue
                distances[neighbour] = next_distance
                queue.append(neighbour)
        self._cache[int(source)] = distances
        return distances

    def distance(self, start: Position, end: Position) -> Optional[int]:
        start_id = self.node_map.get((int(start[0]), int(start[1])))
        end_id = self.node_map.get((int(end[0]), int(end[1])))
        if start_id is None or end_id is None:
            return None
        return self._distances_from(start_id).get(end_id)


@dataclass(frozen=True)
class DispatchCandidateTerms:
    key: ContextKey
    station_id: int
    eligible: bool
    debt_before: float
    free_flow_time: Optional[float]
    debt_increment: float
    pending_pressure_before: float
    station_capacity: float
    delta_pressure_assign: float
    delta_debt_assign: float
    delta_debt_defer: float
    delta_l_assign: float
    delta_l_defer: float
    relative_defer_minus_assign: float
    crossing_reached: bool
    potential_dominates_wm: bool

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["key"] = [int(value) for value in self.key]
        payload["schema_version"] = DISPATCH_POTENTIAL_SCHEMA_VERSION
        return payload


@dataclass(frozen=True)
class DispatchPotentialSnapshot:
    tick: int
    contexts: Tuple[object, ...]
    pending_keys: frozenset[ContextKey]
    pressure_keys: frozenset[ContextKey]
    eligible_free_flow: Mapping[ContextKey, float]
    debt_before: Mapping[ContextKey, float]
    continuous_eligible_mass_before: Mapping[ContextKey, float]
    continuous_eligible_ticks_before: Mapping[ContextKey, int]
    order_age: Mapping[ContextKey, int]
    pending_pressure: Mapping[int, float]
    station_capacity: float
    unresolved_pending_orders: int

    def terms(self, context) -> DispatchCandidateTerms:
        key = context_key(context)
        station_id = int(context.station_id)
        debt = float(self.debt_before.get(key, 0.0))
        free_flow = self.eligible_free_flow.get(key)
        eligible = free_flow is not None and float(free_flow) > 0.0
        increment = 1.0 / float(free_flow) if eligible else 0.0
        pressure = float(self.pending_pressure.get(station_id, 0.0))
        capacity = max(float(self.station_capacity), 1.0)
        pressure_after_assign = (
            max(pressure - 1.0 / capacity, 0.0)
            if key in self.pressure_keys
            else pressure
        )
        delta_pressure_assign = 0.5 * (
            pressure_after_assign * pressure_after_assign - pressure * pressure
        )
        delta_debt_assign = -0.5 * debt * debt
        delta_debt_defer = (
            0.5 * ((debt + increment) ** 2 - debt ** 2)
            if eligible
            else 0.0
        )
        delta_assign = delta_pressure_assign + delta_debt_assign
        delta_defer = delta_debt_defer
        return DispatchCandidateTerms(
            key=key,
            station_id=station_id,
            eligible=eligible,
            debt_before=debt,
            free_flow_time=float(free_flow) if eligible else None,
            debt_increment=increment,
            pending_pressure_before=pressure,
            station_capacity=capacity,
            delta_pressure_assign=delta_pressure_assign,
            delta_debt_assign=delta_debt_assign,
            delta_debt_defer=delta_debt_defer,
            delta_l_assign=delta_assign,
            delta_l_defer=delta_defer,
            relative_defer_minus_assign=delta_defer - delta_assign,
            crossing_reached=bool(eligible and debt + increment >= LIVENESS_CROSSING),
            potential_dominates_wm=bool(debt > DOMINANCE_DEBT),
        )


def pending_chain_keys(world) -> tuple[frozenset[ContextKey], int]:
    """Return every materialised pending order/pod chain identity.

    Pod reservation and robot reachability do not delete an existing debt.
    They only make the chain temporarily ineligible for a new increment.
    """

    from Policies.TaskAssigner.base_task_assigner import (
        fulfilled_pod_ids_for_order,
    )

    from WorldState.task_state import TaskStatus

    active_chain_keys = {
        (int(task.order_id), int(task.pod_id))
        for task in world.task_state.tasks.values()
        if task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
    }
    keys: set[ContextKey] = set()
    unresolved = 0
    for order in world.order_state.get_pending_orders():
        pod_ids = [int(value) for value in getattr(order, "pod_ids", ())]
        if not pod_ids:
            unresolved += 1
            continue
        fulfilled = fulfilled_pod_ids_for_order(world, order)
        for pod_id in pod_ids:
            if pod_id in fulfilled:
                continue
            if (int(order.order_id), int(pod_id)) in active_chain_keys:
                continue
            keys.add((
                int(order.order_id),
                int(pod_id),
                int(order.station_id),
            ))
    return frozenset(keys), int(unresolved)


def _best_free_flow_time(
    context,
    idle_agents: Iterable,
    distances: StaticShortestPathDistance,
    durations: DispatchServiceDurations,
) -> Optional[float]:
    pod = tuple(context.pod_location)
    # ``station_location`` is the non-walkable service cell in the simulator.
    # Physical robot paths terminate at the station entry; station-internal
    # service is represented by ``station_process`` below.
    station = tuple(context.entry_position or context.station_location)
    return_source = tuple(context.exit_position or context.station_location)
    return_location = tuple(context.return_location)

    pod_to_station = distances.distance(pod, station)
    station_to_return = distances.distance(return_source, return_location)
    if pod_to_station is None or station_to_return is None:
        return None

    best_robot_to_pod = None
    for agent in idle_agents:
        distance = distances.distance(tuple(agent.position), pod)
        if distance is None:
            continue
        if best_robot_to_pod is None or distance < best_robot_to_pod:
            best_robot_to_pod = int(distance)
    if best_robot_to_pod is None:
        return None

    return float(max(
        best_robot_to_pod
        + int(durations.pickup)
        + int(pod_to_station)
        + int(durations.station_process)
        + int(station_to_return)
        + int(durations.dropoff),
        1,
    ))


class DispatchDebtLedger:
    """Persistent, policy-local service debt with one update per world tick."""

    def __init__(self) -> None:
        self.debt: Dict[ContextKey, float] = {}
        self.continuous_eligible_mass: Dict[ContextKey, float] = {}
        self.continuous_eligible_ticks: Dict[ContextKey, int] = {}
        self.last_finalized_tick: Optional[int] = None
        self.max_debt = 0.0
        self.max_continuous_eligible_mass = 0.0
        self.max_continuous_eligible_ticks = 0

    def snapshot(
        self,
        world,
        contexts: Sequence[object],
        distances: StaticShortestPathDistance,
        durations: DispatchServiceDurations,
    ) -> DispatchPotentialSnapshot:
        tick = int(getattr(world, "tick", -1))
        pending_keys, unresolved = pending_chain_keys(world)
        idle_agents = tuple(world.get_idle_agents())
        eligible: Dict[ContextKey, float] = {}
        ages: Dict[ContextKey, int] = {}
        counts: Dict[int, int] = {}

        from WorldState.task_state import TaskStatus

        reserved_pods = {
            int(task.pod_id)
            for task in world.task_state.tasks.values()
            if task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        }

        unique_contexts: Dict[ContextKey, object] = {}
        pressure_keys: set[ContextKey] = set()
        for context in contexts:
            key = context_key(context)
            if key not in pending_keys:
                continue
            unique_contexts[key] = context
            order = world.order_state.orders.get(int(context.order_id))
            ages[key] = max(
                tick - int(getattr(order, "created_at", tick)),
                0,
            ) if order is not None else 0
            pod = world.pod_state.get_pod(int(context.pod_id))
            physically_available = bool(
                pod is not None
                and not bool(getattr(pod, "is_carried", False))
                and int(context.pod_id) not in reserved_pods
            )
            if physically_available:
                pressure_keys.add(key)
                counts[int(context.station_id)] = (
                    counts.get(int(context.station_id), 0) + 1
                )
            free_flow = (
                _best_free_flow_time(
                    context, idle_agents, distances, durations
                )
                if physically_available
                else None
            )
            if free_flow is not None:
                eligible[key] = float(free_flow)

        num_stations = max(
            len(getattr(world.map_state, "station_positions", {}) or {}),
            1,
        )
        capacity = max(float(len(getattr(world, "agents", ()))) / num_stations, 1.0)
        pressure = {
            int(station_id): float(count) / capacity
            for station_id, count in counts.items()
        }
        return DispatchPotentialSnapshot(
            tick=tick,
            contexts=tuple(unique_contexts.values()),
            pending_keys=pending_keys,
            pressure_keys=frozenset(pressure_keys),
            eligible_free_flow=eligible,
            debt_before={
                key: float(value)
                for key, value in self.debt.items()
                if key in pending_keys
            },
            continuous_eligible_mass_before={
                key: float(value)
                for key, value in self.continuous_eligible_mass.items()
                if key in pending_keys
            },
            continuous_eligible_ticks_before={
                key: int(value)
                for key, value in self.continuous_eligible_ticks.items()
                if key in pending_keys
            },
            order_age=ages,
            pending_pressure=pressure,
            station_capacity=capacity,
            unresolved_pending_orders=unresolved,
        )

    def finalize(
        self,
        snapshot: DispatchPotentialSnapshot,
        assigned_keys: Iterable[ContextKey],
    ) -> dict:
        tick = int(snapshot.tick)
        if self.last_finalized_tick == tick:
            raise RuntimeError(f"dispatch debt already finalized for tick {tick}")
        self.last_finalized_tick = tick
        assigned = {tuple(int(value) for value in key) for key in assigned_keys}

        for key in list(self.debt):
            if key not in snapshot.pending_keys or key in assigned:
                self.debt.pop(key, None)
                self.continuous_eligible_mass.pop(key, None)
                self.continuous_eligible_ticks.pop(key, None)

        increments = 0
        ineligible = 0
        for key in snapshot.pending_keys:
            if key in assigned:
                self.debt.pop(key, None)
                self.continuous_eligible_mass.pop(key, None)
                self.continuous_eligible_ticks.pop(key, None)
                continue
            free_flow = snapshot.eligible_free_flow.get(key)
            if free_flow is None or float(free_flow) <= 0.0:
                ineligible += 1
                self.continuous_eligible_mass[key] = 0.0
                self.continuous_eligible_ticks[key] = 0
                continue
            increment = 1.0 / float(free_flow)
            self.debt[key] = float(self.debt.get(key, 0.0)) + increment
            self.continuous_eligible_mass[key] = float(
                self.continuous_eligible_mass.get(key, 0.0)
            ) + increment
            self.continuous_eligible_ticks[key] = int(
                self.continuous_eligible_ticks.get(key, 0)
            ) + 1
            increments += 1
            self.max_debt = max(self.max_debt, self.debt[key])
            self.max_continuous_eligible_mass = max(
                self.max_continuous_eligible_mass,
                self.continuous_eligible_mass[key],
            )
            self.max_continuous_eligible_ticks = max(
                self.max_continuous_eligible_ticks,
                self.continuous_eligible_ticks[key],
            )

        return {
            "tick": tick,
            "assigned_contexts": len(assigned),
            "eligible_debt_increments": increments,
            "temporarily_ineligible_contexts": ineligible,
            "active_debt_contexts": len(self.debt),
            "max_debt": float(self.max_debt),
            "max_continuous_eligible_mass": float(
                self.max_continuous_eligible_mass
            ),
            "max_continuous_eligible_ticks": int(
                self.max_continuous_eligible_ticks
            ),
            "streak_bound_passed": bool(
                self.max_continuous_eligible_mass <= MAX_ELIGIBLE_DEFER_MASS
            ),
        }


def prioritise_contexts(
    snapshot: DispatchPotentialSnapshot,
    limit: int,
) -> list[object]:
    """Debt/age priority followed by stable IDs and pod de-duplication."""

    budget = max(int(limit), 0)
    if budget == 0:
        return []
    rows = [
        context
        for context in snapshot.contexts
        if context_key(context) in snapshot.eligible_free_flow
    ]
    rows.sort(key=lambda context: (
        -float(snapshot.debt_before.get(context_key(context), 0.0)),
        -int(snapshot.order_age.get(context_key(context), 0)),
        int(context.order_id),
        int(context.pod_id),
        int(context.station_id),
    ))
    selected = []
    used_pods = set()
    for context in rows:
        pod_id = int(context.pod_id)
        if pod_id in used_pods:
            continue
        selected.append(context)
        used_pods.add(pod_id)
        if len(selected) >= budget:
            break
    return selected


__all__ = [
    "ContextKey",
    "DISPATCH_POTENTIAL_SCHEMA_VERSION",
    "DISPATCH_TRACE_SCHEMA_VERSION",
    "DOMINANCE_DEBT",
    "DispatchCandidateTerms",
    "DispatchDebtLedger",
    "DispatchPotentialSnapshot",
    "DispatchServiceDurations",
    "LIVENESS_CROSSING",
    "MAX_ELIGIBLE_DEFER_MASS",
    "StaticShortestPathDistance",
    "context_key",
    "exact_group_range",
    "pending_chain_keys",
    "prioritise_contexts",
]
