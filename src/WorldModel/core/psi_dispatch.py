"""Frozen semantics for the Phase-C station/context dispatch potential.

This module deliberately contains no World Model parameters.  It combines the
two learned, station-indexed physical channels (``service`` and ``traffic``)
with an exactly observable, context-indexed service debt.  The latter is kept
outside ``e_demand`` so the frozen Round-1 encoder/checkpoint remains valid.

The controller uses lower dispatch cost as better:

    J(c) = 0.5 * service[s(c)] + 0.5 * traffic[s(c)] - service_debt[c]

``service_debt`` is the mean of two bounded, monotone terms:

* station pending-chain work, normalised by the station queue capacity (with
  agents/stations as a declared fallback); and
* context age, normalised by its free-flow chain time.

The physical terms are intentionally not averaged into one learned head.  The
separate fields make sign errors and starvation mechanisms auditable.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from Policies.TaskAssigner.base_task_assigner import (
    fulfilled_pod_ids_for_order,
)
from WorldState.task_state import TaskStatus


PSI_DISPATCH_SCHEMA_VERSION = "phase_c_psi_dispatch_v1"
PSI_DISPATCH_CHANNELS = ("service", "traffic", "service_debt")

ContextKey = Tuple[int, int, int]


class StaticGraphDistance:
    """Lazy shortest-path cache used only for the age normaliser."""

    def __init__(
        self,
        node_map: Mapping[Sequence[int], int],
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

    def distance(self, start: Sequence[int], end: Sequence[int]) -> int | None:
        start_id = self.node_map.get((int(start[0]), int(start[1])))
        end_id = self.node_map.get((int(end[0]), int(end[1])))
        if start_id is None or end_id is None:
            return None
        return self._distances_from(start_id).get(end_id)


def context_key(context) -> ContextKey:
    """Return the stable order/pod/station identity for one context."""

    return (
        int(context.order_id),
        int(context.pod_id),
        int(context.station_id),
    )


def _nonnegative(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def bounded_backlog_score(
    unserved_chain_count: float,
    station_capacity: float,
) -> float:
    """Map pending chain work to ``[0, 1)`` without a hand-tuned cap.

    ``station_capacity`` is the natural average number of agents available to
    one station.  The rational map preserves ordering and grows smoothly:
    ``x/(1+x)`` where ``x = chains/capacity``.
    """

    chains = _nonnegative(unserved_chain_count, "unserved_chain_count")
    capacity = max(_nonnegative(station_capacity, "station_capacity"), 1.0)
    ratio = chains / capacity
    return float(ratio / (1.0 + ratio))


def bounded_age_score(age_ticks: float, free_flow_time: float) -> float:
    """Map waiting age to ``[0, 1)`` using its physical free-flow time."""

    age = _nonnegative(age_ticks, "age_ticks")
    free_flow = _nonnegative(free_flow_time, "free_flow_time")
    if free_flow <= 0.0:
        raise ValueError("free_flow_time must be positive")
    ratio = age / free_flow
    return float(ratio / (1.0 + ratio))


@dataclass(frozen=True)
class DispatchServiceDurations:
    """Physical service durations used by the free-flow age normaliser."""

    pickup: int
    station_process: int
    dropoff: int

    @classmethod
    def from_world(cls, world) -> "DispatchServiceDurations":
        simulation = getattr(getattr(world, "config", None), "simulation", None)
        if simulation is None:
            raise ValueError("world.config.simulation is required")
        values = cls(
            pickup=int(getattr(simulation, "pickup_duration")),
            station_process=int(getattr(simulation, "station_process_duration")),
            dropoff=int(getattr(simulation, "dropoff_duration")),
        )
        if min(values.pickup, values.station_process, values.dropoff) < 0:
            raise ValueError("service durations must be non-negative")
        return values


def _manhattan(start: Sequence[int], end: Sequence[int]) -> int:
    return abs(int(start[0]) - int(end[0])) + abs(int(start[1]) - int(end[1]))


def estimate_free_flow_time(
    context,
    idle_agents: Iterable,
    distance,
    durations: DispatchServiceDurations,
) -> float:
    """Estimate the shortest physical chain time for one context.

    The static graph distance is preferred.  Manhattan distance is a declared
    conservative fallback for a missing graph endpoint; this keeps the debt
    feature observable rather than silently setting urgency to zero.
    """

    def dist(start, end) -> int:
        value = distance.distance(tuple(start), tuple(end))
        return int(value) if value is not None else _manhattan(start, end)

    station_entry = tuple(context.entry_position or context.station_location)
    station_exit = tuple(context.exit_position or context.station_location)
    pod = tuple(context.pod_location)
    return_location = tuple(context.return_location)
    robot_to_pod = [
        dist(agent.position, pod)
        for agent in idle_agents
    ]
    best_robot_to_pod = min(robot_to_pod) if robot_to_pod else 0
    total = (
        best_robot_to_pod
        + int(durations.pickup)
        + dist(pod, station_entry)
        + int(durations.station_process)
        + dist(station_exit, return_location)
        + int(durations.dropoff)
    )
    return float(max(total, 1))


def _pending_chain_counts_by_station(world) -> Dict[int, int]:
    """Count unserved pending order/pod chains without proposing contexts.

    This deliberately does not depend on idle-agent count or on the proposal
    prefix.  A pending order with no materialised pod list contributes one
    conservative unit; materialised pods contribute one unit per unresolved
    order/pod chain.  Chains belonging to an active task or an already
    delivered pod are excluded.
    """

    active_keys = {
        (int(task.order_id), int(task.pod_id))
        for task in getattr(world.task_state, "tasks", {}).values()
        if task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
    }
    counts: Dict[int, int] = {}
    for order in world.order_state.get_pending_orders():
        station_id = int(order.station_id)
        fulfilled = fulfilled_pod_ids_for_order(world, order)
        pod_ids = [int(value) for value in getattr(order, "pod_ids", ())]
        if not pod_ids:
            counts[station_id] = counts.get(station_id, 0) + 1
            continue
        unresolved = sum(
            1
            for pod_id in pod_ids
            if pod_id not in fulfilled
            and (int(order.order_id), pod_id) not in active_keys
        )
        if unresolved:
            counts[station_id] = counts.get(station_id, 0) + unresolved
    return counts


@dataclass(frozen=True)
class ContextDebtFeatures:
    """Auditable components of one context's service debt."""

    station_id: int
    unserved_chain_count: int
    station_capacity: float
    backlog_score: float
    order_age_ticks: int
    free_flow_time: float
    age_score: float
    service_debt: float


def build_context_debt_features(
    world,
    contexts: Sequence,
    idle_agents: Iterable,
    distance,
    durations: DispatchServiceDurations,
) -> Dict[ContextKey, ContextDebtFeatures]:
    """Build station-independent-of-proposal debt features for contexts."""

    idle_agents = tuple(idle_agents)
    station_count = max(
        len(getattr(world.map_state, "station_positions", {}) or {}),
        1,
    )
    fallback_capacity = max(
        float(len(getattr(world, "agents", ()))) / station_count,
        1.0,
    )
    pending_counts = _pending_chain_counts_by_station(world)
    tick = int(getattr(world, "tick", 0))
    result: Dict[ContextKey, ContextDebtFeatures] = {}
    for context in contexts:
        key = context_key(context)
        order = world.order_state.orders.get(int(context.order_id))
        raw_created = getattr(order, "created_at", tick) if order is not None else tick
        try:
            created = int(raw_created)
        except (TypeError, ValueError):
            created = tick
        age = max(tick - created, 0)
        queue = getattr(getattr(world, "station_state", None), "stations", {}).get(
            int(context.station_id)
        )
        station_capacity = max(
            float(getattr(queue, "capacity", fallback_capacity))
            if queue is not None else fallback_capacity,
            1.0,
        )
        free_flow = estimate_free_flow_time(
            context, idle_agents, distance, durations
        )
        backlog_score = bounded_backlog_score(
            pending_counts.get(int(context.station_id), 0),
            station_capacity,
        )
        age_score = bounded_age_score(age, free_flow)
        # Equal weights are frozen deliberately: both terms have the same
        # [0,1) semantics and neither is tuned on outcome seeds.
        service_debt = 0.5 * (backlog_score + age_score)
        result[key] = ContextDebtFeatures(
            station_id=int(context.station_id),
            unserved_chain_count=int(
                pending_counts.get(int(context.station_id), 0)
            ),
            station_capacity=float(station_capacity),
            backlog_score=float(backlog_score),
            order_age_ticks=int(age),
            free_flow_time=float(free_flow),
            age_score=float(age_score),
            service_debt=float(service_debt),
        )
    return result


def dispatch_cost(service: float, traffic: float, service_debt: float) -> float:
    """Return the frozen lower-is-better cross-context cost ``J(c)``."""

    values = {
        "service": _nonnegative(service, "service"),
        "traffic": _nonnegative(traffic, "traffic"),
        "service_debt": _nonnegative(service_debt, "service_debt"),
    }
    if values["service"] > 1.0 + 1e-6 or values["traffic"] > 1.0 + 1e-6:
        raise ValueError("service and traffic must be in [0,1]")
    if values["service_debt"] > 1.0 + 1e-6:
        raise ValueError("service_debt must be in [0,1]")
    return 0.5 * (values["service"] + values["traffic"]) - values[
        "service_debt"
    ]


__all__ = [
    "ContextDebtFeatures",
    "ContextKey",
    "DispatchServiceDurations",
    "StaticGraphDistance",
    "PSI_DISPATCH_CHANNELS",
    "PSI_DISPATCH_SCHEMA_VERSION",
    "bounded_age_score",
    "bounded_backlog_score",
    "build_context_debt_features",
    "context_key",
    "dispatch_cost",
    "estimate_free_flow_time",
]
