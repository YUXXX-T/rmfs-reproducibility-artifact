"""Analytic Lyapunov work/potential ledger for RMFS scheduling.

This module deliberately contains no learned weights and no PyTorch
dependency.  It turns the simulator state into an auditable unfinished-work
ledger and non-negative barrier components.  Neural modules may predict the
same components at a rollout endpoint, but they must not redefine the
functional itself.

The L0 contract uses one unit of work mass per order-pod task chain
(``m_op = 1``) and excludes order age.  Assignment transfers one unit from
pending work to pipeline work; only physical task progress dissipates it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


Position = Tuple[int, int]
ChainKey = Tuple[int, int]
LYAPUNOV_SNAPSHOT_SCHEMA_VERSION = "lyapunov_l1_snapshot_v3"
LYAPUNOV_COLLECTION_SCHEMA_VERSION = "lyapunov_l1_collection_v3"
TRAFFIC_DIAGNOSTIC_NAMES = (
    "total_pressure",
    "reservation_pressure",
    "vertex_reservation_pressure",
    "opposite_edge_pressure",
    "blocked_move_count",
    "realized_vertex_conflict_count",
    "realized_swap_conflict_count",
    "planned_wait_count",
    "next_intent_blocked_count",
)
PHYSICAL_SUMMARY_NAMES = (
    "L_work",
    "L_station",
    "L_traffic",
    "L_stall",
    "L_arrival",
    "L_total",
    "station_work_sum",
    "station_work_max",
    "station_queue_ratio_mean",
    "station_queue_ratio_max",
    "arrival_bin_max",
    "reservation_conflicts",
)


@dataclass(frozen=True)
class LyapunovL0Config:
    """Frozen parameters for the current age-free analytic functional.

    The v3/L1 default potential contains unfinished work, station overload,
    and cumulative incoming-work pressure.  Traffic, stall, and plan-failure
    signals remain fully recorded for safety guards and dissipation losses,
    but their default potential weights are zero so transient route/timer
    events cannot dominate the stored system potential.
    """

    work_weight: float = 1.0
    station_weight: float = 1.0
    traffic_weight: float = 0.0
    stall_weight: float = 0.0
    plan_fail_weight: float = 0.0
    arrival_weight: float = 1.0

    # Healthy-region thresholds.  These are physical/configuration values,
    # not online rolling quantiles.
    station_safe_ratio: float = 0.70
    stall_threshold: int = 10
    plan_fail_threshold: int = 2
    reservation_window: int = 10
    opposite_reservation_weight: float = 1.0
    # A movement rejected by the simulator's conflict resolver is a realised
    # physical traffic event.  It is kept separate from intentional
    # space-time-plan waits and added to the traffic barrier with a frozen,
    # non-negative weight.
    movement_blocked_weight: float = 1.0
    realized_conflict_weight: float = 1.0

    # ETA bins are upper bounds.  The last bin is open ended and uses the
    # width of the final finite bin for its service-capacity estimate.
    eta_bin_edges: Tuple[int, ...] = (10, 25, 50)
    arrival_capacity_scale: float = 1.0

    # Work is normalised by fleet capacity per station unless explicitly
    # overridden.  A value <= 0 selects the automatic scale.
    work_capacity: float = 0.0
    unresolved_pending_mass: float = 1.0

    def __post_init__(self):
        object.__setattr__(
            self, "eta_bin_edges", tuple(int(edge) for edge in self.eta_bin_edges)
        )
        non_negative = (
            "work_weight", "station_weight", "traffic_weight",
            "stall_weight", "plan_fail_weight", "arrival_weight",
            "station_safe_ratio", "opposite_reservation_weight",
            "movement_blocked_weight", "realized_conflict_weight",
            "arrival_capacity_scale",
            "unresolved_pending_mass",
        )
        for name in non_negative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.stall_threshold <= 0 or self.plan_fail_threshold <= 0:
            raise ValueError("stall/plan-fail thresholds must be positive")
        if self.reservation_window <= 0:
            raise ValueError("reservation_window must be positive")
        if any(edge <= 0 for edge in self.eta_bin_edges):
            raise ValueError("ETA bin edges must be positive")
        if tuple(sorted(set(self.eta_bin_edges))) != self.eta_bin_edges:
            raise ValueError("ETA bin edges must be strictly increasing")


@dataclass(frozen=True)
class ChainWork:
    """Remaining work for one order-pod PICK/DELIVER/RETURN chain."""

    order_id: int
    pod_id: int
    station_id: int
    mass: float
    initial_work: float
    remaining_work: float
    remaining_fraction: float
    # Route-plan estimate is diagnostic only.  The physical ledger above is
    # deliberately based on realised position/service state, so replanning
    # cannot manufacture productive progress.  Keeping the plan estimate in
    # parallel lets ``compute_productive_progress`` expose plan-only churn as
    # ``D_plan`` without feeding it back into ``P_prod`` or ``L_work``.
    planned_remaining_work: float
    planned_remaining_fraction: float
    active_task_id: Optional[int]
    active_position: Optional[Position]
    active_path_remaining: Tuple[Position, ...]
    state: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LyapunovSnapshot:
    """Complete analytic state needed for L0 diagnostics and supervision."""

    tick: int
    total: float
    components: Mapping[str, float]
    station_work: Mapping[int, float]
    station_queue_ratio: Mapping[int, float]
    arrival_bins: Mapping[int, Tuple[float, ...]]
    chains: Mapping[ChainKey, ChainWork]
    # Open-order work manifest used to distinguish genuine exogenous order
    # arrivals from representation changes (for example an unresolved order
    # later materialising into several order-pod chains).
    order_work: Mapping[int, Tuple[int, float]] = field(default_factory=dict)
    # Lossless-enough raw records for rebuilding ETA bins after collection.
    # In particular, an exact per-chain ETA lets downstream code determine
    # whether a chain crosses a coarse bin boundary within rollout horizon H.
    arrival_manifest: Tuple[Mapping[str, object], ...] = field(
        default_factory=tuple
    )
    # Raw traffic primitives are persisted instead of relying only on the
    # scalar L_traffic label.  This keeps a later traffic-functional redesign
    # auditable without recollecting the same simulator trajectories.
    traffic_diagnostics: Mapping[str, float] = field(default_factory=dict)
    unresolved_pending_orders: int = 0
    reservation_conflicts: float = 0.0
    traffic_excess_rms: float = 0.0
    stationary_excess_rms: float = 0.0
    plan_fail_excess_rms: float = 0.0
    completed_chain_keys: Tuple[ChainKey, ...] = field(default_factory=tuple)
    cancelled_chain_keys: Tuple[ChainKey, ...] = field(default_factory=tuple)
    work_capacity: float = 1.0
    arrival_capacity: Mapping[int, Tuple[float, ...]] = field(default_factory=dict)

    def physical_summary(self) -> Tuple[float, ...]:
        """Fixed low-dimensional summary suitable for an optional head input."""
        work_values = list(self.station_work.values())
        queue_values = list(self.station_queue_ratio.values())
        arrivals = [value for row in self.arrival_bins.values() for value in row]
        return (
            float(self.components.get("work", 0.0)),
            float(self.components.get("station", 0.0)),
            float(self.components.get("traffic", 0.0)),
            float(self.components.get("stall", 0.0)),
            float(self.components.get("arrival", 0.0)),
            float(self.total),
            float(sum(work_values)),
            float(max(work_values, default=0.0)),
            float(sum(queue_values) / max(len(queue_values), 1)),
            float(max(queue_values, default=0.0)),
            float(max(arrivals, default=0.0)),
            float(self.reservation_conflicts),
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": LYAPUNOV_SNAPSHOT_SCHEMA_VERSION,
            "tick": int(self.tick),
            "total": float(self.total),
            "components": {k: float(v) for k, v in self.components.items()},
            "station_work": {int(k): float(v) for k, v in self.station_work.items()},
            "station_queue_ratio": {
                int(k): float(v) for k, v in self.station_queue_ratio.items()
            },
            "arrival_bins": {
                int(k): [float(v) for v in values]
                for k, values in self.arrival_bins.items()
            },
            "chains": {
                f"{order_id}:{pod_id}": chain.to_dict()
                for (order_id, pod_id), chain in self.chains.items()
            },
            "order_work": {
                int(order_id): {
                    "station_id": int(value[0]),
                    "work": float(value[1]),
                }
                for order_id, value in self.order_work.items()
            },
            "arrival_manifest": [dict(record) for record in self.arrival_manifest],
            "traffic_diagnostics": {
                str(key): float(value)
                for key, value in self.traffic_diagnostics.items()
            },
            "unresolved_pending_orders": int(self.unresolved_pending_orders),
            "reservation_conflicts": float(self.reservation_conflicts),
            "traffic_excess_rms": float(self.traffic_excess_rms),
            "stationary_excess_rms": float(self.stationary_excess_rms),
            "plan_fail_excess_rms": float(self.plan_fail_excess_rms),
            "completed_chain_keys": [
                [int(order_id), int(pod_id)]
                for order_id, pod_id in self.completed_chain_keys
            ],
            "cancelled_chain_keys": [
                [int(order_id), int(pod_id)]
                for order_id, pod_id in self.cancelled_chain_keys
            ],
            "work_capacity": float(self.work_capacity),
            "arrival_capacity": {
                int(station_id): [float(value) for value in values]
                for station_id, values in self.arrival_capacity.items()
            },
            "physical_summary": list(self.physical_summary()),
        }


@dataclass(frozen=True)
class ProductiveProgress:
    """Observed physical work dissipation between two analytic snapshots."""

    horizon: int
    productive_by_station: Mapping[int, float]
    reverse_by_station: Mapping[int, float]
    arrivals_by_station: Mapping[int, float]
    replan_residual_by_station: Mapping[int, float] = field(default_factory=dict)
    route_plan_churn_by_station: Mapping[int, float] = field(default_factory=dict)

    @property
    def mu_by_station(self) -> Dict[int, float]:
        h = max(int(self.horizon), 1)
        return {station: value / h
                for station, value in self.productive_by_station.items()}

    @property
    def productive_total(self) -> float:
        return float(sum(self.productive_by_station.values()))

    @property
    def reverse_total(self) -> float:
        return float(sum(self.reverse_by_station.values()))

    @property
    def arrival_total(self) -> float:
        return float(sum(self.arrivals_by_station.values()))

    @property
    def replan_residual_total(self) -> float:
        return float(sum(self.replan_residual_by_station.values()))

    @property
    def route_plan_churn_total(self) -> float:
        return float(sum(self.route_plan_churn_by_station.values()))

    def to_dict(self) -> dict:
        return {
            "schema_version": "lyapunov_l0_progress_v1",
            "horizon": int(self.horizon),
            "productive_by_station": dict(self.productive_by_station),
            "reverse_by_station": dict(self.reverse_by_station),
            "arrivals_by_station": dict(self.arrivals_by_station),
            "replan_residual_by_station": dict(self.replan_residual_by_station),
            "route_plan_churn_by_station": dict(
                self.route_plan_churn_by_station
            ),
            "mu_by_station": self.mu_by_station,
            "productive_total": self.productive_total,
            "reverse_total": self.reverse_total,
            "arrival_total": self.arrival_total,
            "replan_residual_total": self.replan_residual_total,
            "route_plan_churn_total": self.route_plan_churn_total,
        }


@dataclass(frozen=True)
class CandidateEtaPreview:
    """Immediate analytic arrival-barrier effect of one assignment candidate."""

    station_id: int
    eta: float
    eta_bin: int
    arrival_before: float
    arrival_after: float
    delta_arrival_potential: float

    @property
    def delta_total(self) -> float:
        return float(self.delta_arrival_potential)

    def to_dict(self) -> dict:
        return asdict(self)


def _enum_name(value) -> str:
    return str(getattr(value, "name", value)).upper()


def _manhattan(a: Position, b: Position) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _simulation_config(world):
    return getattr(getattr(world, "config", None), "simulation", None)


def _service_duration(world, task_type_name: str) -> int:
    sim = _simulation_config(world)
    if sim is None:
        return 0
    if task_type_name == "PICK":
        return int(getattr(sim, "pickup_duration", 0))
    if task_type_name == "DELIVER":
        return int(getattr(sim, "station_process_duration", 0))
    if task_type_name == "RETURN":
        return int(getattr(sim, "dropoff_duration", 0))
    return 0


def _task_initial_work(world, task) -> float:
    stored = getattr(task, "free_flow_time", None)
    if stored is not None and float(stored) > 0.0:
        return float(stored)
    source = tuple(getattr(task, "source", (0, 0)))
    destination = tuple(getattr(task, "destination", source))
    return float(
        _manhattan(source, destination)
        + _service_duration(world, _enum_name(getattr(task, "task_type", "")))
    )


def _agent_for_task(world, task):
    agent_id = getattr(task, "agent_id", None)
    if agent_id is None:
        return None
    getter = getattr(world, "get_agent", None)
    if callable(getter):
        return getter(agent_id)
    for agent in getattr(world, "agents", ()):
        if getattr(agent, "agent_id", None) == agent_id:
            return agent
    return None


def _task_remaining_work(world, task, is_first_incomplete: bool) -> float:
    status = _enum_name(getattr(task, "status", ""))
    if status in ("COMPLETED", "CANCELLED"):
        return 0.0
    initial = max(_task_initial_work(world, task), 0.0)
    if not is_first_incomplete:
        return initial

    agent = _agent_for_task(world, task)
    if agent is None:
        return initial
    task_type = _enum_name(getattr(task, "task_type", ""))
    agent_status = _enum_name(getattr(agent, "status", ""))
    destination = tuple(getattr(task, "destination", getattr(agent, "position", (0, 0))))
    position = tuple(getattr(agent, "position", destination))
    travel_destination = destination
    # DELIVER remains IN_PROGRESS while the station releases the robot to its
    # exit cell.  Service has already completed in this state; re-adding the
    # full station duration would create a false reverse-work spike.
    if task_type == "DELIVER" and agent_status == "EXITING":
        station_id = getattr(task, "station_id", None)
        queue = None
        station_state = getattr(world, "station_state", None)
        getter = getattr(station_state, "get_queue", None)
        if callable(getter) and station_id is not None:
            queue = getter(station_id)
        elif station_id is not None:
            queue = getattr(station_state, "stations", {}).get(station_id)
        exit_position = getattr(queue, "exit_position", None)
        if exit_position is not None:
            travel_destination = tuple(exit_position)
    travel = float(_manhattan(position, travel_destination))

    # wait_ticks is physical service already in progress.  A route replanning
    # operation cannot change either the current position or this countdown,
    # so it cannot manufacture productive progress in this ledger.
    wait_ticks = max(0.0, float(getattr(agent, "wait_ticks", 0.0)))
    if task_type == "DELIVER" and agent_status == "EXITING":
        service = 0.0
    elif wait_ticks > 0.0:
        service = wait_ticks
    elif position == destination and status == "IN_PROGRESS":
        service = float(_service_duration(
            world, task_type))
    else:
        service = float(_service_duration(
            world, task_type))
    return travel + service


def _task_planned_remaining_work(
    world,
    task,
    is_first_incomplete: bool,
    physical_remaining: Optional[float] = None,
) -> float:
    """Return the route-plan estimate used only for replan diagnostics.

    The active leg uses the number of remaining time-indexed path cells when
    the path actually terminates at the task's current physical target.  A
    missing/stale path falls back to the geometry-based physical estimate.
    Future legs retain their frozen free-flow work.  This value must never be
    used to compute L0 or productive service credit.
    """
    status = _enum_name(getattr(task, "status", ""))
    if status in ("COMPLETED", "CANCELLED"):
        return 0.0
    initial = max(_task_initial_work(world, task), 0.0)
    if not is_first_incomplete:
        return initial

    agent = _agent_for_task(world, task)
    if agent is None:
        return initial
    task_type = _enum_name(getattr(task, "task_type", ""))
    agent_status = _enum_name(getattr(agent, "status", ""))
    destination = tuple(
        getattr(task, "destination", getattr(agent, "position", (0, 0)))
    )
    travel_destination = destination
    if task_type == "DELIVER" and agent_status == "EXITING":
        station_id = getattr(task, "station_id", None)
        queue = None
        station_state = getattr(world, "station_state", None)
        getter = getattr(station_state, "get_queue", None)
        if callable(getter) and station_id is not None:
            queue = getter(station_id)
        elif station_id is not None:
            queue = getattr(station_state, "stations", {}).get(station_id)
        exit_position = getattr(queue, "exit_position", None)
        if exit_position is not None:
            travel_destination = tuple(exit_position)

    physical = (
        float(physical_remaining)
        if physical_remaining is not None
        else _task_remaining_work(world, task, is_first_incomplete=True)
    )
    position = tuple(getattr(agent, "position", travel_destination))
    physical_travel = float(_manhattan(position, travel_destination))
    service = max(0.0, physical - physical_travel)

    remaining_path = list(_remaining_path(agent, max(len(
        getattr(agent, "path", ()) or ()), 1)))
    if remaining_path and tuple(remaining_path[-1]) == travel_destination:
        planned_travel = float(len(remaining_path))
    else:
        planned_travel = physical_travel
    return planned_travel + service


def _chain_work(world, order, pod_id: int, tasks: Sequence) -> ChainWork:
    type_order = {"PICK": 0, "DELIVER": 1, "RETURN": 2}
    ordered = sorted(
        tasks,
        key=lambda task: (
            type_order.get(_enum_name(getattr(task, "task_type", "")), 99),
            int(getattr(task, "task_id", 0)),
        ),
    )
    initial = sum(max(_task_initial_work(world, task), 0.0) for task in ordered)
    initial = max(initial, 1.0)
    remaining = 0.0
    planned_remaining = 0.0
    first_incomplete_seen = False
    active_task_id: Optional[int] = None
    active_position: Optional[Position] = None
    active_path_remaining: Tuple[Position, ...] = ()
    nonterminal = 0
    for task in ordered:
        status = _enum_name(getattr(task, "status", ""))
        terminal = status in ("COMPLETED", "CANCELLED")
        if not terminal:
            nonterminal += 1
        is_first = not terminal and not first_incomplete_seen
        if is_first:
            first_incomplete_seen = True
            active_task_id = int(getattr(task, "task_id", -1))
            agent = _agent_for_task(world, task)
            if agent is not None:
                active_position = tuple(getattr(agent, "position", (0, 0)))
                path = getattr(agent, "path", ()) or ()
                active_path_remaining = tuple(
                    _remaining_path(agent, max(len(path), 1))
                )
        task_remaining = _task_remaining_work(world, task, is_first)
        remaining += task_remaining
        planned_remaining += _task_planned_remaining_work(
            world,
            task,
            is_first,
            physical_remaining=task_remaining,
        )
    fraction = max(0.0, remaining / initial)
    planned_fraction = max(0.0, planned_remaining / initial)
    return ChainWork(
        order_id=int(getattr(order, "order_id")),
        pod_id=int(pod_id),
        station_id=int(getattr(order, "station_id")),
        mass=1.0,
        initial_work=float(initial),
        remaining_work=float(remaining),
        remaining_fraction=float(fraction),
        planned_remaining_work=float(planned_remaining),
        planned_remaining_fraction=float(planned_fraction),
        active_task_id=active_task_id,
        active_position=active_position,
        active_path_remaining=active_path_remaining,
        state="pipeline" if nonterminal else "complete",
    )


def _station_ids(world) -> Sequence[int]:
    ids = set(getattr(getattr(world, "map_state", None),
                      "station_positions", {}).keys())
    ids.update(getattr(getattr(world, "station_state", None),
                       "stations", {}).keys())
    for order in getattr(getattr(world, "order_state", None), "orders", {}).values():
        ids.add(getattr(order, "station_id", -1))
    return sorted(int(value) for value in ids if int(value) >= 0)


def _build_work_ledger(world, config: LyapunovL0Config):
    orders = getattr(getattr(world, "order_state", None), "orders", {})
    tasks = getattr(getattr(world, "task_state", None), "tasks", {})
    tasks_by_chain: Dict[ChainKey, list] = {}
    for task in tasks.values():
        key = (int(getattr(task, "order_id")), int(getattr(task, "pod_id")))
        tasks_by_chain.setdefault(key, []).append(task)

    completed_keys = set()
    cancelled_keys = set()
    for key, chain_tasks in tasks_by_chain.items():
        statuses = {_enum_name(getattr(task, "status", "")) for task in chain_tasks}
        if statuses and statuses <= {"COMPLETED"}:
            completed_keys.add(key)
        elif statuses and statuses <= {"CANCELLED"}:
            cancelled_keys.add(key)

    chains: Dict[ChainKey, ChainWork] = {}
    station_work = {station_id: 0.0 for station_id in _station_ids(world)}
    order_work: Dict[int, Tuple[int, float]] = {}
    unresolved = 0

    for order in orders.values():
        if _enum_name(getattr(order, "status", "")) == "COMPLETED":
            continue
        order_id = int(getattr(order, "order_id"))
        station_id = int(getattr(order, "station_id"))
        delivered = set(int(value) for value in getattr(order, "delivered_pod_ids", ()))
        pod_ids = [int(value) for value in getattr(order, "pod_ids", ())]

        known_chain_pods = {
            pod_id
            for (oid, pod_id), chain_tasks in tasks_by_chain.items()
            if oid == order_id and any(
                _enum_name(getattr(task, "status", "")) != "CANCELLED"
                for task in chain_tasks
            )
        }
        all_pods = list(dict.fromkeys(pod_ids + sorted(known_chain_pods)))
        if not all_pods:
            # The retriever has not materialised the pod set yet.  L0 keeps an
            # explicit diagnostic rather than mutating the state from a label
            # function.  Normal online scoring materialises contexts first.
            unresolved += 1
            cancelled_mass = len({
                pod_id for (oid, pod_id) in cancelled_keys if oid == order_id
            })
            unresolved_mass = max(
                float(config.unresolved_pending_mass), float(cancelled_mass)
            )
            station_work.setdefault(station_id, 0.0)
            station_work[station_id] += unresolved_mass
            order_work[order_id] = (station_id, float(unresolved_mass))
            continue

        order_mass = 0.0
        for pod_id in all_pods:
            key = (order_id, pod_id)
            chain_tasks = tasks_by_chain.get(key, ())
            if chain_tasks:
                statuses = {
                    _enum_name(getattr(task, "status", ""))
                    for task in chain_tasks
                }
                if statuses and statuses <= {"CANCELLED"}:
                    chain = ChainWork(
                        order_id=order_id,
                        pod_id=pod_id,
                        station_id=station_id,
                        mass=1.0,
                        initial_work=1.0,
                        remaining_work=1.0,
                        remaining_fraction=1.0,
                        planned_remaining_work=1.0,
                        planned_remaining_fraction=1.0,
                        active_task_id=None,
                        active_position=None,
                        active_path_remaining=(),
                        state="cancelled_pending",
                    )
                else:
                    chain = _chain_work(world, order, pod_id, chain_tasks)
            elif pod_id in delivered:
                continue
            else:
                chain = ChainWork(
                    order_id=order_id,
                    pod_id=pod_id,
                    station_id=station_id,
                    mass=1.0,
                    initial_work=1.0,
                    remaining_work=1.0,
                    remaining_fraction=1.0,
                    planned_remaining_work=1.0,
                    planned_remaining_fraction=1.0,
                    active_task_id=None,
                    active_position=None,
                    active_path_remaining=(),
                    state="pending",
                )
            if chain.remaining_fraction <= 0.0:
                continue
            chains[key] = chain
            station_work.setdefault(station_id, 0.0)
            contribution = chain.mass * chain.remaining_fraction
            station_work[station_id] += contribution
            order_mass += contribution
        order_work[order_id] = (station_id, float(order_mass))
    return (
        station_work,
        chains,
        order_work,
        unresolved,
        tuple(sorted(completed_keys)),
        tuple(sorted(cancelled_keys)),
    )


def _station_queue_ratios(world) -> Dict[int, float]:
    ratios: Dict[int, float] = {}
    stations = getattr(getattr(world, "station_state", None), "stations", {})
    for station_id in _station_ids(world):
        station = stations.get(station_id)
        if station is None:
            ratios[station_id] = 0.0
            continue
        occupancy = getattr(station, "occupancy", None)
        value = float(occupancy() if callable(occupancy) else occupancy or 0.0)
        capacity = max(float(getattr(station, "capacity", 1.0)), 1.0)
        ratios[station_id] = value / capacity
    return ratios


def _remaining_path(agent, window: int) -> Sequence[Position]:
    path = list(getattr(agent, "path", ()) or ())
    index = max(0, int(getattr(agent, "path_index", 0)))
    return [tuple(position) for position in path[index:index + window]]


def _next_intent_blocked_count(world) -> int:
    """Re-evaluate the simulator's next-move blocking rule without mutation."""
    movable = []
    for agent in getattr(world, "agents", ()):
        status = _enum_name(getattr(agent, "status", ""))
        if status in ("QUEUING", "DELIVERING", "EXITING"):
            continue
        if bool(getattr(agent, "is_waiting", False)):
            continue
        if bool(getattr(agent, "has_path", False)):
            movable.append(agent)

    intended = {
        int(getattr(agent, "agent_id")): tuple(
            getattr(agent, "path")[int(getattr(agent, "path_index", 0))]
        )
        for agent in movable
    }
    occupied = {
        tuple(getattr(agent, "position", (0, 0)))
        for agent in getattr(world, "agents", ())
        if int(getattr(agent, "agent_id")) not in intended
    }
    blocked = set()
    changed = True
    while changed:
        changed = False
        target_counts: Dict[Position, list] = {}
        for agent_id, position in intended.items():
            if agent_id not in blocked:
                target_counts.setdefault(position, []).append(agent_id)
        for position, agent_ids in target_counts.items():
            if len(agent_ids) <= 1 and position not in occupied:
                continue
            for agent_id in agent_ids:
                if agent_id in blocked:
                    continue
                blocked.add(agent_id)
                agent = next(
                    candidate for candidate in movable
                    if int(getattr(candidate, "agent_id")) == agent_id
                )
                occupied.add(tuple(getattr(agent, "position", (0, 0))))
                changed = True
    return len(blocked)


def _traffic_conflict_diagnostics(
    world,
    config: LyapunovL0Config,
) -> Dict[str, float]:
    """Return auditable reservation and realised-block traffic primitives.

    The former implementation only inspected directed edge reservations and
    skipped wait actions.  It therefore missed the two cases that dominate a
    PrioritizedPathPlanner rollout: two paths reserving the same future
    vertex, and a stale path targeting a robot that has become stationary.
    Every robot is now represented in the future vertex table; after its
    current path ends, its last position remains reserved for the rest of the
    configured window, matching the planner's conservative occupancy model.
    """
    window = max(int(config.reservation_window), 1)
    vertex_reservations: Dict[Tuple[Position, int], float] = {}
    edge_reservations: Dict[Tuple[Position, Position, int], float] = {}
    planned_wait_count = 0.0

    for agent in getattr(world, "agents", ()):
        previous = tuple(getattr(agent, "position", (0, 0)))
        remaining = list(_remaining_path(agent, window))
        future_positions = list(remaining)
        final_position = future_positions[-1] if future_positions else previous
        if len(future_positions) < window:
            future_positions.extend(
                [final_position] * (window - len(future_positions))
            )

        for offset, position in enumerate(future_positions, start=1):
            position = tuple(position)
            vertex_key = (position, offset)
            vertex_reservations[vertex_key] = (
                vertex_reservations.get(vertex_key, 0.0) + 1.0
            )
            if offset <= len(remaining):
                if position == previous:
                    planned_wait_count += 1.0
                else:
                    edge_key = (previous, position, offset)
                    edge_reservations[edge_key] = (
                        edge_reservations.get(edge_key, 0.0) + 1.0
                    )
            previous = position

    vertex_pressure = sum(
        max(0.0, count - 1.0) ** 2
        for count in vertex_reservations.values()
    )

    visited = set()
    opposite_pressure = 0.0
    for source, target, offset in edge_reservations:
        pair = (min(source, target), max(source, target), offset)
        if pair in visited:
            continue
        visited.add(pair)
        forward = edge_reservations.get((source, target, offset), 0.0)
        reverse = edge_reservations.get((target, source, offset), 0.0)
        opposite_pressure += forward * reverse

    reservation_pressure = (
        float(vertex_pressure)
        + float(config.opposite_reservation_weight) * float(opposite_pressure)
    )
    blocked_move_count = float(sum(
        bool(getattr(agent, "traffic_blocked_this_tick", False))
        for agent in getattr(world, "agents", ())
    ))
    realized_vertex_conflict_count = float(getattr(
        world, "traffic_vertex_conflicts_this_tick", 0
    ) or 0)
    realized_swap_conflict_count = float(getattr(
        world, "traffic_swap_conflicts_this_tick", 0
    ) or 0)
    total_pressure = (
        reservation_pressure
        + float(config.movement_blocked_weight) * blocked_move_count
        + float(config.realized_conflict_weight) * (
            realized_vertex_conflict_count + realized_swap_conflict_count
        )
    )
    return {
        "total_pressure": float(total_pressure),
        "reservation_pressure": float(reservation_pressure),
        "vertex_reservation_pressure": float(vertex_pressure),
        "opposite_edge_pressure": float(opposite_pressure),
        "blocked_move_count": blocked_move_count,
        "realized_vertex_conflict_count": realized_vertex_conflict_count,
        "realized_swap_conflict_count": realized_swap_conflict_count,
        "planned_wait_count": float(planned_wait_count),
        "next_intent_blocked_count": float(_next_intent_blocked_count(world)),
    }


def _traffic_conflict_pressure(world, config: LyapunovL0Config) -> float:
    return _traffic_conflict_diagnostics(world, config)["total_pressure"]


def _planned_travel_to(
    agent,
    destination: Position,
) -> Tuple[float, str, float, float]:
    """Prefer the active time-indexed path (including waits) over Manhattan."""
    destination = tuple(destination)
    path = list(getattr(agent, "path", ()) or ())
    remaining = _remaining_path(agent, max(len(path), 1))
    if remaining and tuple(remaining[-1]) == destination:
        steps = float(len(remaining))
        return steps, "active_path", steps, 0.0
    fallback = float(_manhattan(tuple(agent.position), destination))
    return fallback, "manhattan", 0.0, fallback


def _arrival_record(world, order, chain_tasks: Sequence) -> Optional[dict]:
    """Return the exact analytic ETA record for one not-yet-arrived chain.

    QUEUING/DELIVERING/EXITING robots have already reached the station and
    must not remain in the future-arrival wave.  The old collector counted
    them until the DELIVER task was finally marked COMPLETED, which
    double-counted station occupancy as a new arrival and was especially
    damaging under high load.
    """
    by_type = {
        _enum_name(getattr(task, "task_type", "")): task for task in chain_tasks
    }
    deliver = by_type.get("DELIVER")
    if deliver is None or _enum_name(getattr(deliver, "status", "")) in (
            "COMPLETED", "CANCELLED"):
        return None
    pick = by_type.get("PICK")
    agent = _agent_for_task(world, deliver) or (
        _agent_for_task(world, pick) if pick is not None else None
    )
    if agent is None:
        return None

    agent_status = _enum_name(getattr(agent, "status", ""))
    if agent_status in ("QUEUING", "DELIVERING", "EXITING"):
        return None

    deliver_destination = tuple(getattr(deliver, "destination", agent.position))
    station_id = getattr(deliver, "station_id", getattr(order, "station_id", None))
    station_state = getattr(world, "station_state", None)
    getter = getattr(station_state, "get_queue", None)
    queue = getter(station_id) if callable(getter) and station_id is not None else None
    if queue is None and station_id is not None:
        queue = getattr(station_state, "stations", {}).get(station_id)
    entry_position = getattr(queue, "entry_position", None)
    if entry_position is not None:
        deliver_destination = tuple(entry_position)

    pick_done = pick is None or _enum_name(getattr(pick, "status", "")) in (
        "COMPLETED", "CANCELLED")
    if pick_done:
        # Reaching the queue entry is an arrival even if admission is delayed
        # by a full queue.  Do not duplicate it in L_arrival and L_station.
        if tuple(getattr(agent, "position", deliver_destination)) == deliver_destination:
            return None
        eta, eta_source, path_steps, fallback_steps = _planned_travel_to(
            agent, deliver_destination
        )
        service_steps = 0.0
        phase = "to_station"
    else:
        pick_destination = tuple(getattr(pick, "destination", agent.position))
        (pick_travel, pick_source,
         path_steps, pick_fallback_steps) = _planned_travel_to(
            agent, pick_destination
        )
        # Reuse the physical ledger's service countdown semantics, then swap
        # only its Manhattan travel part for the exact active-path duration.
        # This prevents a partially consumed pickup wait from being reset to
        # the full configured service duration in the arrival target.
        physical_pick_remaining = float(_task_remaining_work(
            world, pick, is_first_incomplete=True
        ))
        physical_pick_travel = float(_manhattan(
            tuple(getattr(agent, "position", pick_destination)),
            pick_destination,
        ))
        pick_service = max(
            0.0, physical_pick_remaining - physical_pick_travel
        )
        eta = (
            pick_travel
            + pick_service
            + float(_manhattan(pick_destination, deliver_destination))
        )
        station_fallback_steps = float(_manhattan(
            pick_destination, deliver_destination
        ))
        fallback_steps = pick_fallback_steps + station_fallback_steps
        service_steps = pick_service
        eta_source = f"{pick_source}+pickup+manhattan_to_station"
        phase = "pre_pick"

    return {
        "order_id": int(getattr(order, "order_id")),
        "pod_id": int(getattr(deliver, "pod_id")),
        "station_id": int(getattr(order, "station_id")),
        # L0 currently freezes m_op=1.  Persist it explicitly so a later
        # business-mass ablation cannot silently reinterpret old records.
        "mass": 1.0,
        "eta": float(eta),
        "phase": phase,
        "eta_source": eta_source,
        "path_steps_used": float(path_steps),
        "fallback_steps_used": float(fallback_steps),
        "service_steps": float(service_steps),
        "agent_id": int(getattr(agent, "agent_id")),
        "agent_status": agent_status,
        "pick_status": (
            _enum_name(getattr(pick, "status", "")) if pick is not None else "NONE"
        ),
        "deliver_status": _enum_name(getattr(deliver, "status", "")),
        "path_steps_remaining": int(len(_remaining_path(
            agent, max(len(getattr(agent, "path", ()) or ()), 1)
        ))),
        "wait_ticks": int(max(0, getattr(agent, "wait_ticks", 0))),
    }


def _task_eta_to_station(world, order, chain_tasks: Sequence) -> Optional[float]:
    record = _arrival_record(world, order, chain_tasks)
    return None if record is None else float(record["eta"])


def _eta_bin_index(eta: float, edges: Sequence[int]) -> int:
    for index, upper in enumerate(edges):
        if eta <= upper:
            return index
    return len(edges)


def _eta_bin_widths(edges: Sequence[int]) -> Tuple[int, ...]:
    previous = 0
    widths = []
    for upper in edges:
        widths.append(max(1, int(upper) - previous))
        previous = int(upper)
    widths.append(widths[-1] if widths else 1)
    return tuple(widths)


def _arrival_capacity(world, station_id: int, bin_index: int,
                      config: LyapunovL0Config) -> float:
    widths = _eta_bin_widths(config.eta_bin_edges)
    width = widths[min(bin_index, len(widths) - 1)]
    service = max(_service_duration(world, "DELIVER"), 1)
    service_slots = max(1.0, float(width) / float(service))
    # A station has one physical service slot in the current simulator.  Queue
    # and buffer slots absorb arrivals but do not process pods in parallel.
    return config.arrival_capacity_scale * service_slots


def _arrival_manifest(world, config: LyapunovL0Config) -> Tuple[dict, ...]:
    orders = getattr(getattr(world, "order_state", None), "orders", {})
    tasks = getattr(getattr(world, "task_state", None), "tasks", {})
    grouped: Dict[ChainKey, list] = {}
    for task in tasks.values():
        grouped.setdefault(
            (int(getattr(task, "order_id")), int(getattr(task, "pod_id"))),
            [],
        ).append(task)

    records = []
    for (order_id, _), chain_tasks in grouped.items():
        order = orders.get(order_id)
        if order is None:
            continue
        record = _arrival_record(world, order, chain_tasks)
        if record is None:
            continue
        record["eta_bin"] = int(_eta_bin_index(
            float(record["eta"]), config.eta_bin_edges
        ))
        records.append(record)

    bin_loads: Dict[Tuple[int, int], float] = {}
    for record in records:
        key = (int(record["station_id"]), int(record["eta_bin"]))
        bin_loads[key] = bin_loads.get(key, 0.0) + float(
            record.get("mass", 1.0)
        )
    for record in records:
        station_id = int(record["station_id"])
        bin_index = int(record["eta_bin"])
        capacity = float(_arrival_capacity(
            world, station_id, bin_index, config
        ))
        load = float(bin_loads[(station_id, bin_index)])
        record["bin_load"] = load
        record["bin_capacity"] = capacity
        record["bin_excess"] = max(0.0, load - capacity)

    return tuple(sorted(
        records,
        key=lambda row: (
            int(row["station_id"]), float(row["eta"]),
            int(row["order_id"]), int(row["pod_id"]),
        ),
    ))


def _arrival_bins(
    world,
    config: LyapunovL0Config,
    manifest: Optional[Sequence[Mapping[str, object]]] = None,
) -> Dict[int, Tuple[float, ...]]:
    station_ids = _station_ids(world)
    bins = {station_id: [0.0] * (len(config.eta_bin_edges) + 1)
            for station_id in station_ids}
    for record in manifest if manifest is not None else _arrival_manifest(world, config):
        station_id = int(record["station_id"])
        bins.setdefault(station_id, [0.0] * (len(config.eta_bin_edges) + 1))
        bin_index = int(record.get(
            "eta_bin",
            _eta_bin_index(float(record["eta"]), config.eta_bin_edges),
        ))
        bins[station_id][bin_index] += float(record.get("mass", 1.0))
    return {station: tuple(values) for station, values in bins.items()}


def _arrival_potential(world, arrival_bins: Mapping[int, Sequence[float]],
                       config: LyapunovL0Config) -> float:
    """Return cumulative, capacity-normalised incoming-work pressure.

    Disjoint-bin penalties can be gamed by moving one chain just across a bin
    boundary: the near bin loses one unit while an otherwise empty later bin
    gains it.  The L1 functional instead compares cumulative work due by every
    ETA boundary with cumulative station service capacity.  Thus delaying a
    chain only removes pressure from deadlines that genuinely precede its new
    ETA; all later deadlines still account for that work.

    Averaging over station/prefix terms and normalising by cumulative capacity
    keeps this component dimensionless and prevents the raw squared counts
    observed in L0 from overwhelming work and station pressure.
    """
    total = 0.0
    terms = 0
    for station_id, values in arrival_bins.items():
        cumulative_load = 0.0
        cumulative_capacity = 0.0
        for bin_index, value in enumerate(values):
            cumulative_load += float(value)
            cumulative_capacity += _arrival_capacity(
                world, station_id, bin_index, config
            )
            excess = max(0.0, cumulative_load - cumulative_capacity)
            capacity_scale = max(float(cumulative_capacity), 1.0)
            total += (excess / capacity_scale) ** 2
            terms += 1
    return 0.5 * config.arrival_weight * total / max(terms, 1)


def compute_lyapunov_snapshot(
    world,
    config: Optional[LyapunovL0Config] = None,
) -> LyapunovSnapshot:
    """Compute the age-free analytic L0 functional from a simulator state."""
    config = config or LyapunovL0Config()
    (station_work, chains, order_work, unresolved,
     completed_keys, cancelled_keys) = _build_work_ledger(world, config)
    num_stations = max(len(station_work), 1)
    num_agents = max(len(getattr(world, "agents", ())), 1)
    work_capacity = (
        config.work_capacity if config.work_capacity > 0.0
        else max(float(num_agents) / float(num_stations), 1.0)
    )
    work_component = 0.5 * config.work_weight * sum(
        (float(value) / work_capacity) ** 2 for value in station_work.values()
    )

    queue_ratios = _station_queue_ratios(world)
    station_component = 0.5 * config.station_weight * sum(
        max(0.0, ratio - config.station_safe_ratio) ** 2
        for ratio in queue_ratios.values()
    )

    traffic_diagnostics = _traffic_conflict_diagnostics(world, config)
    traffic_pressure = float(traffic_diagnostics["total_pressure"])
    traffic_component = 0.5 * config.traffic_weight * traffic_pressure

    stall_sum = 0.0
    fail_sum = 0.0
    for agent in getattr(world, "agents", ()):
        stall_excess = max(
            0.0,
            float(getattr(agent, "stationary_ticks", 0))
            / float(config.stall_threshold) - 1.0,
        )
        fail_excess = max(
            0.0,
            float(getattr(agent, "plan_failed_streak", 0))
            / float(config.plan_fail_threshold) - 1.0,
        )
        stall_sum += stall_excess * stall_excess
        fail_sum += fail_excess * fail_excess
    stall_component = (
        0.5 * config.stall_weight * stall_sum / num_agents
        + 0.5 * config.plan_fail_weight * fail_sum / num_agents
    )
    stationary_excess_rms = math.sqrt(stall_sum / num_agents)
    plan_fail_excess_rms = math.sqrt(fail_sum / num_agents)

    arrival_manifest = _arrival_manifest(world, config)
    arrival_bins = _arrival_bins(world, config, manifest=arrival_manifest)
    arrival_capacity = {
        station_id: tuple(
            _arrival_capacity(world, station_id, bin_index, config)
            for bin_index in range(len(values))
        )
        for station_id, values in arrival_bins.items()
    }
    arrival_component = _arrival_potential(world, arrival_bins, config)
    components = {
        "work": float(work_component),
        "station": float(station_component),
        "traffic": float(traffic_component),
        "stall": float(stall_component),
        "arrival": float(arrival_component),
    }
    total = float(sum(components.values()))
    return LyapunovSnapshot(
        tick=int(getattr(world, "tick", 0)),
        total=total,
        components=components,
        station_work=station_work,
        station_queue_ratio=queue_ratios,
        arrival_bins=arrival_bins,
        chains=chains,
        order_work=order_work,
        arrival_manifest=arrival_manifest,
        traffic_diagnostics=traffic_diagnostics,
        unresolved_pending_orders=unresolved,
        reservation_conflicts=float(
            traffic_diagnostics["reservation_pressure"]
        ),
        traffic_excess_rms=math.sqrt(max(float(traffic_pressure), 0.0)),
        stationary_excess_rms=float(stationary_excess_rms),
        plan_fail_excess_rms=float(plan_fail_excess_rms),
        completed_chain_keys=completed_keys,
        cancelled_chain_keys=cancelled_keys,
        work_capacity=float(work_capacity),
        arrival_capacity=arrival_capacity,
    )


def compute_productive_progress(
    previous: LyapunovSnapshot,
    current: LyapunovSnapshot,
    horizon: Optional[int] = None,
) -> ProductiveProgress:
    """Close the station work ledger without rewarding representation churn.

    ``arrivals`` are work carried by genuinely new order ids.  Signed
    ``replan_residual`` closes materialisation/cancellation changes for orders
    that already existed.  Route-plan length churn is reported separately and
    never enters ``P_prod`` or the work-balance target.
    """
    if horizon is None:
        horizon = max(int(current.tick) - int(previous.tick), 1)
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    productive: Dict[int, float] = {}
    reverse: Dict[int, float] = {}
    arrivals: Dict[int, float] = {}
    replan_residual: Dict[int, float] = {}
    route_plan_churn: Dict[int, float] = {}
    keys = set(previous.chains) | set(current.chains)
    for key in keys:
        before = previous.chains.get(key)
        after = current.chains.get(key)
        if before is None and after is not None:
            # This can be a pending->assigned representation change.  External
            # arrivals are reconstructed from the station-level conservation
            # equation below, so assignment cannot manufacture A_j.
            continue
        if before is None:
            continue
        before_value = before.mass * before.remaining_fraction
        if after is None:
            if key not in set(current.completed_chain_keys):
                # Cancellation/retrieval reset is not productive service.
                continue
            after_value = 0.0
        else:
            if after.state == "cancelled_pending":
                continue
            after_value = after.mass * after.remaining_fraction
        station_id = before.station_id
        if before_value >= after_value:
            productive[station_id] = (
                productive.get(station_id, 0.0) + before_value - after_value
            )
        else:
            reverse[station_id] = (
                reverse.get(station_id, 0.0) + after_value - before_value
            )

        # The analytic ledger itself ignores route-plan length.  When the
        # physical remaining-work state is unchanged, any change in the
        # parallel path estimate is therefore pure planning churn.  Record
        # its magnitude as D_plan, but never transfer it into productive or
        # reverse physical work.
        if after is not None and abs(before_value - after_value) <= 1e-9:
            before_plan = before.mass * before.planned_remaining_fraction
            after_plan = after.mass * after.planned_remaining_fraction
            plan_only_change = abs(before_plan - after_plan)
            before_path = tuple(before.active_path_remaining)
            after_path = tuple(after.active_path_remaining)
            same_task = before.active_task_id == after.active_task_id
            if after_path:
                consumed_only = (
                    same_task
                    and len(after_path) < len(before_path)
                    and before_path[-len(after_path):] == after_path
                )
            else:
                # Emptying an all-wait suffix is normal path consumption.
                # Clearing a route containing movement while staying at the
                # same physical state is a replan and remains diagnostic.
                consumed_only = (
                    same_task
                    and bool(before_path)
                    and before.active_position == after.active_position
                    and all(
                        position == before.active_position
                        for position in before_path
                    )
                )
            if plan_only_change > 1e-9 and same_task and not consumed_only:
                route_plan_churn[station_id] = (
                    route_plan_churn.get(station_id, 0.0)
                    + plan_only_change
                )

    # External work injection A_j is identified by genuinely new order ids.
    # A pre-existing unresolved order later expanding into several pod chains
    # is a representation/retrieval residual, not a new order arrival.
    previous_orders = previous.order_work
    current_orders = current.order_work
    for order_id, (station_id, work) in current_orders.items():
        if order_id not in previous_orders and float(work) > 1e-9:
            arrivals[int(station_id)] = (
                arrivals.get(int(station_id), 0.0) + float(work)
            )

    stations = set(previous.station_work) | set(current.station_work)
    for station_id in stations:
        delta_work = (
            float(current.station_work.get(station_id, 0.0))
            - float(previous.station_work.get(station_id, 0.0))
        )
        residual = (
            delta_work
            - arrivals.get(station_id, 0.0)
            + productive.get(station_id, 0.0)
            - reverse.get(station_id, 0.0)
        )
        if abs(residual) > 1e-9:
            replan_residual[station_id] = residual

    return ProductiveProgress(
        horizon=int(horizon),
        productive_by_station=productive,
        reverse_by_station=reverse,
        arrivals_by_station=arrivals,
        replan_residual_by_station=replan_residual,
        route_plan_churn_by_station=route_plan_churn,
    )


def preview_candidate_eta(
    world,
    station_id: int,
    robot_position: Position,
    pod_position: Position,
    station_position: Position,
    config: Optional[LyapunovL0Config] = None,
    snapshot: Optional[LyapunovSnapshot] = None,
) -> CandidateEtaPreview:
    """Compute the immediate ETA-bin barrier delta for one pod assignment."""
    config = config or LyapunovL0Config()
    snapshot = snapshot or compute_lyapunov_snapshot(world, config)
    eta = float(
        _manhattan(tuple(robot_position), tuple(pod_position))
        + _service_duration(world, "PICK")
        + _manhattan(tuple(pod_position), tuple(station_position))
    )
    bin_index = _eta_bin_index(eta, config.eta_bin_edges)
    existing = snapshot.arrival_bins.get(
        int(station_id), (0.0,) * (len(config.eta_bin_edges) + 1)
    )
    before = float(existing[bin_index])
    after = before + 1.0
    before_bins = {
        int(sid): tuple(float(value) for value in row)
        for sid, row in snapshot.arrival_bins.items()
    }
    after_bins = dict(before_bins)
    updated = list(after_bins.get(
        int(station_id),
        (0.0,) * (len(config.eta_bin_edges) + 1),
    ))
    updated[bin_index] += 1.0
    after_bins[int(station_id)] = tuple(updated)
    delta = (
        _arrival_potential(world, after_bins, config)
        - _arrival_potential(world, before_bins, config)
    )
    return CandidateEtaPreview(
        station_id=int(station_id),
        eta=eta,
        eta_bin=bin_index,
        arrival_before=before,
        arrival_after=after,
        delta_arrival_potential=float(delta),
    )


def preview_assignment_context(
    world,
    context,
    agent,
    config: Optional[LyapunovL0Config] = None,
    snapshot: Optional[LyapunovSnapshot] = None,
) -> CandidateEtaPreview:
    """Convenience adapter for ``AssignmentContext`` plus an idle agent."""
    entry_position = getattr(context, "entry_position", None)
    station_position = entry_position or getattr(
        context, "station_location", None
    )
    if station_position is None:
        raise ValueError(
            "assignment context must provide entry_position or "
            "station_location"
        )
    return preview_candidate_eta(
        world=world,
        station_id=int(context.station_id),
        robot_position=tuple(agent.position),
        pod_position=tuple(context.pod_location),
        station_position=tuple(station_position),
        config=config,
        snapshot=snapshot,
    )


__all__ = [
    "CandidateEtaPreview",
    "ChainWork",
    "LyapunovL0Config",
    "LyapunovSnapshot",
    "LYAPUNOV_COLLECTION_SCHEMA_VERSION",
    "LYAPUNOV_SNAPSHOT_SCHEMA_VERSION",
    "ProductiveProgress",
    "PHYSICAL_SUMMARY_NAMES",
    "TRAFFIC_DIAGNOSTIC_NAMES",
    "compute_lyapunov_snapshot",
    "compute_productive_progress",
    "preview_assignment_context",
    "preview_candidate_eta",
]
