"""Pure contracts for context-conditioned adaptive station pre-admission.

The controller implemented by this module decides whether a dispatch context
may start its PICK chain.  It intentionally does *not* estimate a precise
robot arrival time.  Every already-launched station-bound chain consumes one
unit of pipeline mass because it will eventually require one station slot.

FIFO-V2 remains the physical safety fallback.  This layer acts earlier, while
the robot is still idle, so post-PICK ``WAITING_ASSIGNED`` is exceptional
rather than the normal way to regulate station load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from WorldState.agent_state import AgentStatus
from WorldState.task_state import TaskStatus, TaskType


ADAPTIVE_PRE_ADMISSION_SCHEMA_VERSION = (
    "phase_c_adaptive_pre_admission_v1"
)


@dataclass(frozen=True)
class StationPipelineSnapshot:
    """Mutually exclusive station-bound chain counts at one decision tick."""

    station_id: int
    capacity: int
    physical: int
    committed_in_transit: int
    moving_to_pod: int
    waiting_assigned: int

    @property
    def pipeline_mass(self) -> float:
        # V1 deliberately gives every launched chain unit mass.  This avoids
        # pretending that a static route length is a reliable arrival ETA.
        return float(
            self.physical
            + self.committed_in_transit
            + self.moving_to_pod
            + self.waiting_assigned
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "station_id": int(self.station_id),
            "capacity": int(self.capacity),
            "physical": int(self.physical),
            "committed_in_transit": int(self.committed_in_transit),
            "moving_to_pod": int(self.moving_to_pod),
            "waiting_assigned": int(self.waiting_assigned),
            "pipeline_mass": float(self.pipeline_mass),
        }


@dataclass(frozen=True)
class PreAdmissionDecision:
    """Auditable EXECUTE/DEFER result for one context."""

    execute: bool
    current_pipeline_mass: float
    pipeline_mass_after_execute: float
    capacity: float
    traffic: float
    local_waiting_ratio: float
    global_waiting_ratio: float
    base_headroom: float
    defer_credit: float
    effective_headroom: float
    effective_pipeline_limit: float
    base_execute: bool
    debt_override: bool
    excess_mass: float

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "execute": bool(self.execute),
            "current_pipeline_mass": float(self.current_pipeline_mass),
            "pipeline_mass_after_execute": float(
                self.pipeline_mass_after_execute
            ),
            "capacity": float(self.capacity),
            "traffic": float(self.traffic),
            "local_waiting_ratio": float(self.local_waiting_ratio),
            "global_waiting_ratio": float(self.global_waiting_ratio),
            "base_headroom": float(self.base_headroom),
            "defer_credit": float(self.defer_credit),
            "effective_headroom": float(self.effective_headroom),
            "effective_pipeline_limit": float(
                self.effective_pipeline_limit
            ),
            "base_execute": bool(self.base_execute),
            "debt_override": bool(self.debt_override),
            "excess_mass": float(self.excess_mass),
        }


def _unit_interval(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def adaptive_pre_admission_decision(
    snapshot: StationPipelineSnapshot,
    *,
    traffic: float,
    global_waiting_ratio: float,
    defer_mass: float,
    virtual_pipeline_additions: int = 0,
) -> PreAdmissionDecision:
    """Return the frozen V1 pre-admission decision.

    The station has one physical capacity window plus at most one adaptive
    prefetch window.  Low traffic and little waiting expose that second
    window.  Local/global waiting continuously remove it.  Context defer debt
    restores headroom after repeated eligible deferrals, preventing a
    permanent context-level absorbing state.

    ``defer_mass`` is dimensionless.  The online assigner increments it by
    ``1 / static_free_flow_time`` after each eligible rejected tick and clips
    its decision credit to ``[0, 1]``.  Static free flow is only a debt clock;
    it is not used as a predicted arrival time.
    """

    capacity = max(float(snapshot.capacity), 1.0)
    traffic_value = _unit_interval(traffic)
    global_wait = _unit_interval(global_waiting_ratio)
    local_wait = _unit_interval(snapshot.waiting_assigned / capacity)
    debt_credit = _unit_interval(defer_mass)

    # Exact pipeline occupancy already represents station service load.  The
    # learned service channel remains part of J(c), while traffic determines
    # whether speculative prefetch headroom is physically safe.
    base_headroom = (
        (1.0 - traffic_value)
        * (1.0 - local_wait)
        * (1.0 - global_wait)
    )
    effective_headroom = max(base_headroom, debt_credit)

    current_mass = (
        float(snapshot.pipeline_mass) + max(int(virtual_pipeline_additions), 0)
    )
    after_mass = current_mass + 1.0
    base_limit = capacity * (1.0 + base_headroom)
    effective_limit = capacity * (1.0 + effective_headroom)
    tolerance = 1e-9
    base_execute = after_mass <= base_limit + tolerance
    execute = after_mass <= effective_limit + tolerance
    return PreAdmissionDecision(
        execute=bool(execute),
        current_pipeline_mass=float(current_mass),
        pipeline_mass_after_execute=float(after_mass),
        capacity=float(capacity),
        traffic=float(traffic_value),
        local_waiting_ratio=float(local_wait),
        global_waiting_ratio=float(global_wait),
        base_headroom=float(base_headroom),
        defer_credit=float(debt_credit),
        effective_headroom=float(effective_headroom),
        effective_pipeline_limit=float(effective_limit),
        base_execute=bool(base_execute),
        debt_override=bool(execute and not base_execute),
        excess_mass=float(max(after_mass - effective_limit, 0.0)),
    )


def eligible_defer_increment(static_free_flow_time: float) -> float:
    """One eligible tick of context debt on its natural chain-time scale."""

    return 1.0 / max(float(static_free_flow_time), 1.0)


def build_station_pipeline_snapshots(world) -> Dict[int, StationPipelineSnapshot]:
    """Count all launched station-bound chains without an ETA model."""

    queues: Mapping[int, object] = world.station_state.stations
    rows: dict[int, dict[str, object]] = {}
    committed_agent_ids: set[int] = set()
    waiting_agent_ids: set[int] = set()
    for raw_station_id, queue in queues.items():
        station_id = int(raw_station_id)
        physical = {int(value) for value in queue.physical_agent_ids()}
        committed = {int(value) for value in queue.committed_agent_ids()}
        waiting = {int(value) for value in queue.waiting_agent_ids()}
        committed_agent_ids.update(committed)
        waiting_agent_ids.update(waiting)
        rows[station_id] = {
            "capacity": max(int(queue.capacity), 1),
            "physical": len(physical),
            "committed_in_transit": len(committed - physical),
            "moving_to_pod_agent_ids": set(),
            "waiting_assigned": len(waiting),
        }

    # A PICK chain has a paired ASSIGNED DELIVER task that carries station_id.
    # Count each robot once and exclude any restored/inconsistent chain that
    # already appears in the committed or waiting ledgers.
    tasks = tuple(world.task_state.tasks.values())
    deliver_station_by_chain: dict[tuple[int, int, int], int] = {}
    active = (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
    for task in tasks:
        if (
            task.task_type == TaskType.DELIVER
            and task.status in active
            and task.agent_id is not None
            and task.station_id is not None
        ):
            deliver_station_by_chain[(
                int(task.order_id),
                int(task.pod_id),
                int(task.agent_id),
            )] = int(task.station_id)

    for task in tasks:
        if (
            task.task_type != TaskType.PICK
            or task.status not in active
            or task.agent_id is None
        ):
            continue
        agent_id = int(task.agent_id)
        if agent_id in committed_agent_ids or agent_id in waiting_agent_ids:
            continue
        station_id = deliver_station_by_chain.get((
            int(task.order_id), int(task.pod_id), agent_id
        ))
        if station_id not in rows:
            continue
        agent = world.get_agent(agent_id)
        if agent is not None and agent.status == AgentStatus.WAITING_ASSIGNED:
            continue
        rows[station_id]["moving_to_pod_agent_ids"].add(agent_id)

    return {
        station_id: StationPipelineSnapshot(
            station_id=station_id,
            capacity=int(values["capacity"]),
            physical=int(values["physical"]),
            committed_in_transit=int(values["committed_in_transit"]),
            moving_to_pod=len(values["moving_to_pod_agent_ids"]),
            waiting_assigned=int(values["waiting_assigned"]),
        )
        for station_id, values in rows.items()
    }


__all__ = [
    "ADAPTIVE_PRE_ADMISSION_SCHEMA_VERSION",
    "PreAdmissionDecision",
    "StationPipelineSnapshot",
    "adaptive_pre_admission_decision",
    "build_station_pipeline_snapshots",
    "eligible_defer_increment",
]
