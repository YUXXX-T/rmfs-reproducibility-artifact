"""Station-conditioned context defer for Dynamic-J dispatch.

This module defines a small, training-free controller contract.  It does not
change the World Model, station head, demand encoder, S1 robot scorer, or the
station queue implementation.

The comparison is dimensionless and lower-level-threshold free.  The frozen
V1 risk is::

    station_risk_v1 = max(
        prospective pipeline excess,
        ready-but-unadmitted contention,
        station phi pressure while contention exists,
    )

The V2 ablation removes the duplicated ready term from the maximum while
retaining ready robots in pipeline mass and in the diagnostic trace::

    station_risk_v2 = max(
        prospective pipeline excess,
        station phi pressure while contention exists,
    )

    context_credit = service_debt(context) + eligible defer debt

``EXECUTE`` wins when context credit covers station risk.  Otherwise only the
current context is deferred; contexts for other stations remain available.
The physical committed-capacity admission invariant remains a separate final
safety layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from WorldState.agent_state import AgentStatus
from WorldState.task_state import TaskStatus, TaskType


STATION_CONTEXT_DEFER_SCHEMA_VERSION = (
    "phase_c_station_context_defer_v1"
)
STATION_CONTEXT_DEFER_PIPELINE_PHI_SCHEMA_VERSION = (
    "phase_c_station_context_defer_pipeline_phi_v2"
)
STATION_CONTEXT_DEFER_RISK_READY_MAX_V1 = "ready_max_v1"
STATION_CONTEXT_DEFER_RISK_PIPELINE_PHI_V2 = "pipeline_phi_v2"
STATION_CONTEXT_DEFER_RISK_MODES = (
    STATION_CONTEXT_DEFER_RISK_READY_MAX_V1,
    STATION_CONTEXT_DEFER_RISK_PIPELINE_PHI_V2,
)


def _unit_interval(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number < -1e-9 or number > 1.0 + 1e-9:
        raise ValueError(f"{name} must be in [0, 1]")
    return min(max(number, 0.0), 1.0)


@dataclass(frozen=True)
class StationDispatchSnapshot:
    """Mutually exclusive station-bound pipeline counts at one tick."""

    station_id: int
    capacity: int
    physical: int
    committed_in_transit: int
    pre_pick: int
    ready_unadmitted: int

    @property
    def committed_load(self) -> int:
        return int(self.physical + self.committed_in_transit)

    @property
    def upstream_pipeline(self) -> int:
        return int(self.pre_pick + self.ready_unadmitted)

    @property
    def pipeline_mass(self) -> int:
        return int(self.committed_load + self.upstream_pipeline)

    @property
    def admission_slack(self) -> int:
        return max(int(self.capacity) - self.committed_load, 0)

    def as_dict(self) -> dict[str, int]:
        return {
            "station_id": int(self.station_id),
            "capacity": int(self.capacity),
            "physical": int(self.physical),
            "committed_in_transit": int(self.committed_in_transit),
            "committed_load": int(self.committed_load),
            "pre_pick": int(self.pre_pick),
            "ready_unadmitted": int(self.ready_unadmitted),
            "upstream_pipeline": int(self.upstream_pipeline),
            "pipeline_mass": int(self.pipeline_mass),
            "admission_slack": int(self.admission_slack),
        }


@dataclass(frozen=True)
class StationContextDeferDecision:
    """Auditable EXECUTE/DEFER_CONTEXT comparison for one context."""

    execute: bool
    capacity: float
    current_pipeline_mass: float
    prospective_pipeline_mass: float
    pipeline_excess_mass: float
    pipeline_excess_score: float
    ready_contention_score: float
    ready_contention_in_station_risk: bool
    phi_pressure: float
    station_risk: float
    risk_mode: str
    service_debt: float
    defer_credit: float
    context_credit: float
    base_execute: bool
    debt_override: bool
    defer_margin: float
    dominant_risk: str
    diagnostic_dominant_risk: str
    ready_contention_would_dominate: bool
    worst_case_liveness_bound_ticks: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "execute": bool(self.execute),
            "capacity": float(self.capacity),
            "current_pipeline_mass": float(self.current_pipeline_mass),
            "prospective_pipeline_mass": float(
                self.prospective_pipeline_mass
            ),
            "pipeline_excess_mass": float(self.pipeline_excess_mass),
            "pipeline_excess_score": float(self.pipeline_excess_score),
            "ready_contention_score": float(
                self.ready_contention_score
            ),
            "ready_contention_in_station_risk": bool(
                self.ready_contention_in_station_risk
            ),
            "phi_pressure": float(self.phi_pressure),
            "station_risk": float(self.station_risk),
            "risk_mode": str(self.risk_mode),
            "service_debt": float(self.service_debt),
            "defer_credit": float(self.defer_credit),
            "context_credit": float(self.context_credit),
            "base_execute": bool(self.base_execute),
            "debt_override": bool(self.debt_override),
            "defer_margin": float(self.defer_margin),
            "dominant_risk": str(self.dominant_risk),
            "diagnostic_dominant_risk": str(
                self.diagnostic_dominant_risk
            ),
            "ready_contention_would_dominate": bool(
                self.ready_contention_would_dominate
            ),
            "worst_case_liveness_bound_ticks": int(
                self.worst_case_liveness_bound_ticks
            ),
        }


def eligible_defer_increment(free_flow_time: float) -> float:
    """Advance context liveness debt by one natural chain-time tick."""

    value = float(free_flow_time)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("free_flow_time must be positive and finite")
    return 1.0 / max(value, 1.0)


def _bounded_excess(excess: float, capacity: float) -> float:
    """Use the same rational normalization as dispatch service debt."""

    excess = max(float(excess), 0.0)
    capacity = max(float(capacity), 1.0)
    return excess / (capacity + excess) if excess > 0.0 else 0.0


def station_context_defer_decision(
    snapshot: StationDispatchSnapshot,
    *,
    service: float,
    traffic: float,
    service_debt: float,
    defer_mass: float,
    free_flow_time: float,
    virtual_pipeline_additions: int = 0,
    risk_mode: str = STATION_CONTEXT_DEFER_RISK_READY_MAX_V1,
) -> StationContextDeferDecision:
    """Compare launching one context with locally deferring it.

    No station is penalised merely for being normally occupied.  Station risk
    is exactly zero while the candidate still fits inside the physical
    capacity window and no picked pod is waiting for admission.

    Once prospective upstream work exceeds that window, the following
    interpretable signals are computed without cancellation:

    * bounded prospective pipeline excess;
    * direct ready-but-unadmitted contention (V1 risk; V2 diagnostic only);
    * the frozen station head's mean service/traffic pressure.

    In V2, ready work is not ignored: it is already part of
    ``snapshot.pipeline_mass`` and therefore contributes to prospective
    pipeline excess.  Only its second, independent max-risk path is removed.

    The context side uses the already frozen service debt plus a context-local
    defer ledger.  Because every signal is in ``[0, 1]`` and defer debt grows
    by ``1/T_ff``, defer debt alone reaches one after at most
    ``ceil(T_ff)`` rejected eligible ticks.  This is an unconditional bound
    even if the observable backlog component of service debt later decreases.
    """

    capacity = max(float(snapshot.capacity), 1.0)
    service_value = _unit_interval(service, "service")
    traffic_value = _unit_interval(traffic, "traffic")
    debt_value = _unit_interval(service_debt, "service_debt")
    defer_value = _unit_interval(defer_mass, "defer_mass")
    if int(virtual_pipeline_additions) < 0:
        raise ValueError("virtual_pipeline_additions must be non-negative")
    risk_mode = str(risk_mode)
    if risk_mode not in STATION_CONTEXT_DEFER_RISK_MODES:
        raise ValueError(
            "risk_mode must be one of: "
            + ", ".join(STATION_CONTEXT_DEFER_RISK_MODES)
        )
    free_flow = float(free_flow_time)
    if not math.isfinite(free_flow) or free_flow <= 0.0:
        raise ValueError("free_flow_time must be positive and finite")

    current_mass = float(
        snapshot.pipeline_mass + int(virtual_pipeline_additions)
    )
    prospective_mass = current_mass + 1.0
    excess = max(prospective_mass - capacity, 0.0)
    excess_score = _bounded_excess(excess, capacity)

    ready = max(float(snapshot.ready_unadmitted), 0.0)
    slack = max(float(snapshot.admission_slack), 0.0)
    ready_contention = (
        ready / (ready + slack) if ready > 0.0 else 0.0
    )

    contention_exists = excess > 0.0 or ready > 0.0
    phi_pressure = (
        0.5 * (service_value + traffic_value)
        if contention_exists else 0.0
    )
    diagnostic_components = {
        "pipeline_excess": float(excess_score),
        "ready_unadmitted": float(ready_contention),
        "station_phi": float(phi_pressure),
    }
    ready_in_risk = risk_mode == STATION_CONTEXT_DEFER_RISK_READY_MAX_V1
    risk_components = dict(diagnostic_components)
    if not ready_in_risk:
        risk_components.pop("ready_unadmitted")
    priority = {
        "ready_unadmitted": 2,
        "pipeline_excess": 1,
        "station_phi": 0,
    }
    diagnostic_dominant_risk = max(
        diagnostic_components,
        key=lambda name: (
            diagnostic_components[name], priority[name]
        ),
    )
    dominant_risk = max(
        risk_components,
        key=lambda name: (risk_components[name], priority[name]),
    )
    station_risk = max(risk_components.values())
    context_credit = min(debt_value + defer_value, 1.0)
    tolerance = 1e-9
    base_execute = station_risk <= debt_value + tolerance
    execute = station_risk <= context_credit + tolerance
    worst_case_bound = int(math.ceil(max(free_flow, 1.0)))
    return StationContextDeferDecision(
        execute=bool(execute),
        capacity=capacity,
        current_pipeline_mass=current_mass,
        prospective_pipeline_mass=prospective_mass,
        pipeline_excess_mass=excess,
        pipeline_excess_score=excess_score,
        ready_contention_score=ready_contention,
        ready_contention_in_station_risk=ready_in_risk,
        phi_pressure=phi_pressure,
        station_risk=station_risk,
        risk_mode=risk_mode,
        service_debt=debt_value,
        defer_credit=defer_value,
        context_credit=context_credit,
        base_execute=bool(base_execute),
        debt_override=bool(execute and not base_execute),
        defer_margin=max(station_risk - context_credit, 0.0),
        dominant_risk=dominant_risk,
        diagnostic_dominant_risk=diagnostic_dominant_risk,
        ready_contention_would_dominate=(
            diagnostic_dominant_risk == "ready_unadmitted"
        ),
        worst_case_liveness_bound_ticks=worst_case_bound,
    )


def build_station_dispatch_snapshots(world) -> Dict[int, StationDispatchSnapshot]:
    """Count station pipelines without using a precise arrival-time model.

    The categories are mutually exclusive:

    * ``physical`` and ``committed_in_transit`` come from admission tokens;
    * ``ready_unadmitted`` has already completed PICK and carries the pod;
    * ``pre_pick`` has an active PICK/DELIVER chain but has not picked the pod.

    Each agent contributes to exactly one category for its target station.
    """

    queues: Mapping[int, Any] = world.station_state.stations
    rows: dict[int, dict[str, Any]] = {}
    committed_agents: set[int] = set()
    for raw_station_id, queue in queues.items():
        station_id = int(raw_station_id)
        physical = {int(value) for value in queue.physical_agent_ids()}
        committed = {int(value) for value in queue.committed_agent_ids()}
        committed_agents.update(committed)
        rows[station_id] = {
            "capacity": max(int(queue.capacity), 1),
            "physical": physical,
            "committed_in_transit": committed - physical,
            "pre_pick": set(),
            "ready_unadmitted": set(),
        }

    tasks = tuple(world.task_state.tasks.values())
    active = (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
    pick_by_chain: dict[tuple[int, int, int], Any] = {}
    for task in tasks:
        if task.task_type != TaskType.PICK or task.agent_id is None:
            continue
        pick_by_chain[(
            int(task.order_id), int(task.pod_id), int(task.agent_id)
        )] = task

    classified: set[int] = set(committed_agents)
    for deliver in tasks:
        if (
            deliver.task_type != TaskType.DELIVER
            or deliver.status not in active
            or deliver.agent_id is None
            or deliver.station_id is None
        ):
            continue
        station_id = int(deliver.station_id)
        agent_id = int(deliver.agent_id)
        if station_id not in rows or agent_id in classified:
            continue
        agent = world.get_agent(agent_id)
        if agent is None:
            continue
        pick = pick_by_chain.get((
            int(deliver.order_id), int(deliver.pod_id), agent_id
        ))
        has_pod = (
            agent.carried_pod_id is not None
            and int(agent.carried_pod_id) == int(deliver.pod_id)
        )
        pick_completed = (
            pick is not None and pick.status == TaskStatus.COMPLETED
        )
        if (
            has_pod
            or pick_completed
            or agent.status == AgentStatus.WAITING_ASSIGNED
        ):
            rows[station_id]["ready_unadmitted"].add(agent_id)
            classified.add(agent_id)
            continue
        if pick is not None and pick.status in active:
            rows[station_id]["pre_pick"].add(agent_id)
            classified.add(agent_id)

    return {
        station_id: StationDispatchSnapshot(
            station_id=station_id,
            capacity=int(values["capacity"]),
            physical=len(values["physical"]),
            committed_in_transit=len(values["committed_in_transit"]),
            pre_pick=len(values["pre_pick"]),
            ready_unadmitted=len(values["ready_unadmitted"]),
        )
        for station_id, values in sorted(rows.items())
    }


__all__ = [
    "STATION_CONTEXT_DEFER_PIPELINE_PHI_SCHEMA_VERSION",
    "STATION_CONTEXT_DEFER_RISK_MODES",
    "STATION_CONTEXT_DEFER_RISK_PIPELINE_PHI_V2",
    "STATION_CONTEXT_DEFER_RISK_READY_MAX_V1",
    "STATION_CONTEXT_DEFER_SCHEMA_VERSION",
    "StationContextDeferDecision",
    "StationDispatchSnapshot",
    "build_station_dispatch_snapshots",
    "eligible_defer_increment",
    "station_context_defer_decision",
]
