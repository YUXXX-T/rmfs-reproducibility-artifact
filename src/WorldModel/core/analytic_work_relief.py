"""Analytic free-flow work relief for one fixed RMFS assignment context.

This module contains no learned parameters and intentionally does not inspect
future simulator state.  It uses only information available at the decision
point (robot/context geometry, the configured service durations and the
current analytic station-work ledger) to construct a nominal endpoint.

The work ledger in :mod:`WorldModel.core.lyapunov` assigns one unit of mass to
an order/pod PICK -> DELIVER -> RETURN chain.  Its remaining fraction is the
remaining physical free-flow work divided by the chain's initial free-flow
work.  Under the explicit nominal assumptions of one unit of physical
progress per tick and no blocking, the candidate therefore relieves

    min(H, chain_free_flow_work) / chain_free_flow_work

units of station work over horizon ``H``.  Congestion, reservations, queueing,
multi-robot interaction and path failures are *not* hidden inside this
formula; they are exactly the effects that an optional residual estimator is
allowed to learn.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence, Tuple


Position = Tuple[int, int]
WORK_HORIZON_LABEL_SCHEMA_VERSION = "analytic_work_horizon_labels_v1"
NOMINAL_RELIEF_SCHEMA_VERSION = "analytic_nominal_work_relief_v1"
SNAPSHOT_NOMINAL_RELIEF_SCHEMA_VERSION = (
    "analytic_snapshot_nominal_work_relief_v1"
)
VIRTUAL_CANDIDATE_WORK_DRIFT_SCHEMA_VERSION = (
    "analytic_virtual_candidate_work_drift_v1"
)
STATION_WORK_TRAJECTORY_SCHEMA_VERSION = (
    "analytic_work_relief_trajectory_v1"
)


def _position(value, name: str) -> Position:
    if value is None or len(value) != 2:
        raise ValueError(f"{name} must be a two-dimensional position")
    return int(value[0]), int(value[1])


def _manhattan(left: Position, right: Position) -> int:
    return abs(int(left[0]) - int(right[0])) + abs(int(left[1]) - int(right[1]))


def _mapping_value(mapping: Mapping, key: int) -> float:
    if key in mapping:
        return float(mapping[key])
    if str(key) in mapping:
        return float(mapping[str(key)])
    raise KeyError(f"station {key} is missing from the work ledger")


@dataclass(frozen=True)
class WorkServiceDurations:
    """Configured service time for the three physical task-chain legs."""

    pickup: int
    station_process: int
    dropoff: int

    def __post_init__(self) -> None:
        if min(int(self.pickup), int(self.station_process), int(self.dropoff)) < 0:
            raise ValueError("work service durations must be non-negative")

    @classmethod
    def from_mapping(cls, config: Mapping) -> "WorkServiceDurations":
        simulation = config.get("simulation", config)
        return cls(
            pickup=int(simulation["pickup_duration"]),
            station_process=int(simulation["station_process_duration"]),
            dropoff=int(simulation["dropoff_duration"]),
        )

    @classmethod
    def from_world(cls, world) -> "WorkServiceDurations":
        """Read the same service durations used by the live simulator."""

        simulation = getattr(getattr(world, "config", None), "simulation", None)
        if simulation is None:
            raise ValueError("world.config.simulation is required")
        return cls(
            pickup=int(getattr(simulation, "pickup_duration")),
            station_process=int(getattr(simulation, "station_process_duration")),
            dropoff=int(getattr(simulation, "dropoff_duration")),
        )


@dataclass(frozen=True)
class NominalWorkRelief:
    """Auditable analytic prediction for one candidate and one horizon."""

    horizon: int
    station_id: int
    station_index: int
    chain_mass: float
    pick_work: float
    deliver_work: float
    return_work: float
    chain_free_flow_work: float
    nominal_progress_work: float
    nominal_relief_fraction: float
    nominal_relief_mass: float
    applied_relief_mass: float
    current_station_work: Tuple[float, ...]
    endpoint_station_work: Tuple[float, ...]
    current_work_potential: float
    endpoint_work_potential: float
    raw_work_drift: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["schema_version"] = NOMINAL_RELIEF_SCHEMA_VERSION
        payload["current_station_work"] = list(self.current_station_work)
        payload["endpoint_station_work"] = list(self.endpoint_station_work)
        return payload


@dataclass(frozen=True)
class SnapshotNominalWorkRelief:
    """Free-flow relief for every active pipeline chain in one snapshot."""

    horizon: int
    station_ids: Tuple[int, ...]
    post_action_station_work: Tuple[float, ...]
    nominal_station_relief: Tuple[float, ...]
    available_station_relief: Tuple[float, ...]
    nominal_endpoint_station_work: Tuple[float, ...]
    active_pipeline_chains: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["schema_version"] = SNAPSHOT_NOMINAL_RELIEF_SCHEMA_VERSION
        for name in (
            "station_ids",
            "post_action_station_work",
            "nominal_station_relief",
            "available_station_relief",
            "nominal_endpoint_station_work",
        ):
            payload[name] = list(payload[name])
        return payload


@dataclass(frozen=True)
class VirtualCandidateWorkDrift:
    """Parameter-free H-step ``L_work`` drift for one online candidate.

    The current order/pod chain already contributes pending mass to the
    decision-state work ledger.  A virtual assignment changes only that
    chain's state from pending to pipeline, then applies the same free-flow
    snapshot relief used by the offline analytic-A experiment.  No simulator
    step, future order, continuation policy, latent endpoint or learned head
    is consulted.
    """

    horizon: int
    order_id: int
    pod_id: int
    station_id: int
    chain_mass: float
    chain_free_flow_work: float
    candidate_nominal_relief_mass: float
    active_pipeline_chains: int
    current_station_work: Tuple[float, ...]
    nominal_station_relief: Tuple[float, ...]
    endpoint_station_work: Tuple[float, ...]
    current_work_potential: float
    endpoint_work_potential: float
    raw_work_drift: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["schema_version"] = (
            VIRTUAL_CANDIDATE_WORK_DRIFT_SCHEMA_VERSION
        )
        for name in (
            "current_station_work",
            "nominal_station_relief",
            "endpoint_station_work",
        ):
            payload[name] = list(payload[name])
        return payload


def work_potential(
    station_work: Sequence[float],
    *,
    work_capacity: Sequence[float] | float,
    work_weight: float,
) -> float:
    """Evaluate the fixed analytic quadratic ``L_work`` functional."""

    values = tuple(float(value) for value in station_work)
    if isinstance(work_capacity, (int, float)):
        capacity = (float(work_capacity),) * len(values)
    else:
        capacity = tuple(float(value) for value in work_capacity)
    if len(capacity) != len(values) or any(value <= 0.0 for value in capacity):
        raise ValueError("work_capacity must be positive and match station_work")
    if float(work_weight) < 0.0:
        raise ValueError("work_weight must be non-negative")
    return 0.5 * float(work_weight) * sum(
        (value / scale) ** 2 for value, scale in zip(values, capacity)
    )


def compute_nominal_work_relief(
    *,
    candidate_info: Mapping,
    fixed_context: Mapping,
    station_ids: Sequence[int],
    current_station_work: Sequence[float] | Mapping,
    service_durations: WorkServiceDurations,
    horizon: int,
    work_capacity: Sequence[float] | float,
    work_weight: float,
    chain_mass: float = 1.0,
) -> NominalWorkRelief:
    """Return the no-blocking endpoint implied by the analytic work ledger.

    The function never reads ``lyapunov_l0_end``, a future order, a future
    demand embedding or a continuation policy.  It is therefore safe to use
    both as an online baseline and as the deterministic baseline of a learned
    residual target.
    """

    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    if float(chain_mass) <= 0.0:
        raise ValueError("chain_mass must be positive")
    ids = tuple(int(value) for value in station_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("station_ids must be non-empty and unique")
    station_id = int(fixed_context["station_id"])
    if station_id not in ids:
        raise ValueError("candidate station is absent from station_ids")
    station_index = ids.index(station_id)

    if isinstance(current_station_work, Mapping):
        current = tuple(
            _mapping_value(current_station_work, station) for station in ids
        )
    else:
        current = tuple(float(value) for value in current_station_work)
    if len(current) != len(ids) or any(value < 0.0 for value in current):
        raise ValueError("current_station_work must be non-negative length S")

    robot = _position(candidate_info.get("robot_start"), "robot_start")
    pod = _position(fixed_context.get("pod_location"), "pod_location")
    station = _position(
        fixed_context.get("station_location"), "station_location"
    )
    return_source = _position(
        fixed_context.get("exit_position") or station,
        "exit_position/station_location",
    )
    return_destination = _position(
        fixed_context.get("return_location"), "return_location"
    )

    pick_work = float(_manhattan(robot, pod) + service_durations.pickup)
    deliver_work = float(
        _manhattan(pod, station) + service_durations.station_process
    )
    return_work = float(
        _manhattan(return_source, return_destination)
        + service_durations.dropoff
    )
    chain_work = max(pick_work + deliver_work + return_work, 1.0)
    progress = min(float(horizon), chain_work)
    relief_fraction = progress / chain_work
    nominal_relief = float(chain_mass) * relief_fraction

    # A malformed/stale context must never create negative work.  Clamping to
    # the currently recorded station ledger also makes the analytic baseline
    # safe for partially materialised legacy snapshots.
    applied_relief = min(nominal_relief, current[station_index])
    endpoint = list(current)
    endpoint[station_index] = max(0.0, endpoint[station_index] - applied_relief)
    current_potential = work_potential(
        current, work_capacity=work_capacity, work_weight=work_weight
    )
    endpoint_potential = work_potential(
        endpoint, work_capacity=work_capacity, work_weight=work_weight
    )
    return NominalWorkRelief(
        horizon=int(horizon),
        station_id=station_id,
        station_index=station_index,
        chain_mass=float(chain_mass),
        pick_work=pick_work,
        deliver_work=deliver_work,
        return_work=return_work,
        chain_free_flow_work=chain_work,
        nominal_progress_work=progress,
        nominal_relief_fraction=relief_fraction,
        nominal_relief_mass=nominal_relief,
        applied_relief_mass=applied_relief,
        current_station_work=current,
        endpoint_station_work=tuple(endpoint),
        current_work_potential=current_potential,
        endpoint_work_potential=endpoint_potential,
        raw_work_drift=endpoint_potential - current_potential,
    )


def compute_snapshot_nominal_work_relief(
    snapshot: Mapping,
    *,
    horizon: int,
    station_ids: Optional[Sequence[int]] = None,
    tolerance: float = 1e-5,
) -> SnapshotNominalWorkRelief:
    """Aggregate analytic relief over all active post-action pipeline chains.

    This is the training/oracle counterpart of the online virtual-candidate
    helper above. It consumes only a decision-time post-action snapshot.
    A pipeline chain contributes

    mass * min(H, remaining_work) / initial_work

    nominal relief, while its entire current contribution is the maximum
    positive relief that any residual estimator may claim. Pending chains
    receive no relief because isolated rollout contains no later scheduler.
    """

    requested = int(horizon)
    if requested <= 0:
        raise ValueError("horizon must be positive")
    work_mapping = snapshot.get("station_work")
    if not isinstance(work_mapping, Mapping) or not work_mapping:
        raise ValueError("snapshot.station_work must be a non-empty mapping")
    if station_ids is None:
        ids = tuple(sorted(int(value) for value in work_mapping))
    else:
        ids = tuple(int(value) for value in station_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("station_ids must be non-empty and unique")
    index = {station_id: offset for offset, station_id in enumerate(ids)}
    post = tuple(_mapping_value(work_mapping, station_id) for station_id in ids)
    if any(value < 0.0 for value in post):
        raise ValueError("post-action station work must be non-negative")

    chains = snapshot.get("chains")
    if not isinstance(chains, Mapping):
        raise ValueError("post-action snapshot is missing the chain ledger")
    nominal = [0.0] * len(ids)
    available = [0.0] * len(ids)
    active = 0
    for key, chain in chains.items():
        if not isinstance(chain, Mapping):
            raise ValueError(f"chain {key!r} must be a mapping")
        if str(chain.get("state") or "").lower() != "pipeline":
            continue
        if chain.get("active_task_id") is None:
            raise ValueError(
                f"pipeline chain {key!r} has no active physical task"
            )
        station_id = int(chain["station_id"])
        if station_id not in index:
            raise ValueError(
                f"chain {key!r} references unknown station {station_id}"
            )
        mass = float(chain["mass"])
        initial = float(chain["initial_work"])
        remaining = float(chain["remaining_work"])
        if mass < 0.0 or initial <= 0.0 or remaining < 0.0:
            raise ValueError(f"chain {key!r} has an invalid physical ledger")
        stored_fraction = chain.get("remaining_fraction")
        if stored_fraction is not None:
            expected_fraction = remaining / initial
            if abs(float(stored_fraction) - expected_fraction) > (
                tolerance * max(1.0, abs(expected_fraction))
            ):
                raise ValueError(
                    f"chain {key!r} remaining_fraction does not close"
                )
        offset = index[station_id]
        available[offset] += mass * remaining / initial
        nominal[offset] += (
            mass * min(float(requested), remaining) / initial
        )
        active += 1

    for offset, station_id in enumerate(ids):
        allowed = tolerance * max(1.0, abs(post[offset]))
        if available[offset] > post[offset] + allowed:
            raise ValueError(
                "active pipeline work exceeds the post-action station ledger "
                f"at station {station_id}"
            )
        if nominal[offset] > available[offset] + allowed:
            raise ValueError(
                f"nominal relief exceeds available pipeline work at {station_id}"
            )
        available[offset] = min(available[offset], post[offset])
        nominal[offset] = min(nominal[offset], available[offset])

    endpoint = tuple(
        max(0.0, value - relief)
        for value, relief in zip(post, nominal)
    )
    return SnapshotNominalWorkRelief(
        horizon=requested,
        station_ids=ids,
        post_action_station_work=post,
        nominal_station_relief=tuple(nominal),
        available_station_relief=tuple(available),
        nominal_endpoint_station_work=endpoint,
        active_pipeline_chains=active,
    )


def compute_virtual_candidate_work_drift(
    snapshot,
    *,
    candidate_info: Mapping,
    fixed_context: Mapping,
    service_durations: WorkServiceDurations,
    horizon: int,
    work_weight: float,
) -> VirtualCandidateWorkDrift:
    """Evaluate analytic-A on a virtual post-assignment work ledger.

    ``snapshot`` may be a :class:`LyapunovSnapshot` or its serialized mapping.
    The function is deliberately strict: the fixed order/pod chain must be a
    pending chain already represented in the decision-state ledger.  Silently
    adding missing mass would make the online and collected post-action
    semantics differ.
    """

    if hasattr(snapshot, "to_dict"):
        payload = snapshot.to_dict()
    elif isinstance(snapshot, Mapping):
        payload = copy.deepcopy(dict(snapshot))
    else:
        raise TypeError("snapshot must be a LyapunovSnapshot or mapping")

    station_work = payload.get("station_work")
    chains = payload.get("chains")
    if not isinstance(station_work, Mapping) or not station_work:
        raise ValueError("snapshot.station_work must be a non-empty mapping")
    if not isinstance(chains, Mapping):
        raise ValueError("snapshot.chains must be a mapping")

    order_id = int(fixed_context["order_id"])
    pod_id = int(fixed_context["pod_id"])
    station_id = int(fixed_context["station_id"])
    chain_key = f"{order_id}:{pod_id}"
    chain = chains.get(chain_key)
    if not isinstance(chain, Mapping):
        raise ValueError(
            "fixed order/pod chain is absent from the decision-state ledger: "
            f"{chain_key}"
        )
    state = str(chain.get("state") or "").lower()
    if state != "pending":
        raise ValueError(
            f"virtual assignment requires a pending chain, got {state!r}"
        )
    if int(chain.get("station_id", -1)) != station_id:
        raise ValueError("fixed context station differs from pending chain")
    chain_mass = float(chain.get("mass", 0.0))
    remaining_fraction = float(chain.get("remaining_fraction", 0.0))
    if chain_mass <= 0.0 or abs(remaining_fraction - 1.0) > 1e-6:
        raise ValueError("pending chain must carry positive full unresolved mass")

    station_ids = tuple(sorted(int(value) for value in station_work))
    work_capacity = float(payload.get("work_capacity", 0.0))
    if work_capacity <= 0.0:
        raise ValueError("snapshot.work_capacity must be positive")

    candidate_chain = compute_nominal_work_relief(
        candidate_info=candidate_info,
        fixed_context=fixed_context,
        station_ids=station_ids,
        current_station_work=station_work,
        service_durations=service_durations,
        horizon=int(horizon),
        work_capacity=work_capacity,
        work_weight=float(work_weight),
        chain_mass=chain_mass,
    )

    virtual_chain = dict(chain)
    virtual_chain.update({
        "initial_work": float(candidate_chain.chain_free_flow_work),
        "remaining_work": float(candidate_chain.chain_free_flow_work),
        "remaining_fraction": 1.0,
        "planned_remaining_work": float(candidate_chain.chain_free_flow_work),
        "planned_remaining_fraction": 1.0,
        # A non-null sentinel is sufficient for the analytic ledger.  No live
        # Task is created and no global task id is consumed.
        "active_task_id": -1,
        "active_position": list(candidate_info["robot_start"]),
        "active_path_remaining": [],
        "state": "pipeline",
    })
    virtual_payload = copy.deepcopy(payload)
    virtual_payload["chains"][chain_key] = virtual_chain
    forecast = compute_snapshot_nominal_work_relief(
        virtual_payload,
        horizon=int(horizon),
        station_ids=station_ids,
    )

    current = tuple(
        _mapping_value(station_work, station_id_value)
        for station_id_value in station_ids
    )
    current_potential = work_potential(
        current,
        work_capacity=work_capacity,
        work_weight=float(work_weight),
    )
    recorded_work = (payload.get("components") or {}).get("work")
    if recorded_work is not None and abs(
        float(recorded_work) - current_potential
    ) > 1e-8 * max(1.0, abs(current_potential)):
        raise ValueError(
            "snapshot work component does not close with its station ledger"
        )
    endpoint_potential = work_potential(
        forecast.nominal_endpoint_station_work,
        work_capacity=work_capacity,
        work_weight=float(work_weight),
    )
    return VirtualCandidateWorkDrift(
        horizon=int(horizon),
        order_id=order_id,
        pod_id=pod_id,
        station_id=station_id,
        chain_mass=chain_mass,
        chain_free_flow_work=float(candidate_chain.chain_free_flow_work),
        candidate_nominal_relief_mass=float(
            candidate_chain.nominal_relief_mass
        ),
        active_pipeline_chains=int(forecast.active_pipeline_chains),
        current_station_work=current,
        nominal_station_relief=forecast.nominal_station_relief,
        endpoint_station_work=forecast.nominal_endpoint_station_work,
        current_work_potential=float(current_potential),
        endpoint_work_potential=float(endpoint_potential),
        raw_work_drift=float(endpoint_potential - current_potential),
    )


def _plain_list(value) -> list:
    if value is None:
        raise ValueError("required trajectory value is missing")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def extract_work_horizon_endpoint(sample: Mapping, horizon: int) -> Mapping:
    """Resolve a true isolated endpoint for legacy H-only or multi-H data."""

    requested = int(horizon)
    if requested <= 0:
        raise ValueError("horizon must be positive")
    labels = sample.get("lyapunov_work_horizon_labels") or {}
    if labels:
        if labels.get("schema_version") != WORK_HORIZON_LABEL_SCHEMA_VERSION:
            raise ValueError("unexpected analytic work-horizon label schema")
        if str(labels.get("rollout_continuation_mode")) != "isolated":
            raise ValueError("work-horizon labels must use isolated continuation")
        endpoint = (labels.get("endpoints") or {}).get(str(requested))
        if endpoint is not None:
            return endpoint

    mask = sample.get("future_mask")
    if mask is not None:
        mask_values = [bool(value) for value in _plain_list(mask)]
        if requested > len(mask_values):
            raise ValueError(
                f"requested H={requested} exceeds collected H={len(mask_values)}"
            )
        if not all(mask_values[:requested]):
            raise ValueError(f"H={requested} endpoint is right-censored")
    if mask is not None and len(mask) == requested:
        endpoint = sample.get("lyapunov_l0_end")
        if endpoint is not None:
            return endpoint
    if mask is not None and requested < len(mask):
        schema = str(sample.get(
            "analytic_work_relief_trajectory_schema_version", ""
        ))
        if schema != STATION_WORK_TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(
                "intermediate work endpoint requires "
                f"{STATION_WORK_TRAJECTORY_SCHEMA_VERSION}, got {schema!r}"
            )
        station_ids = tuple(int(value) for value in _plain_list(
            sample.get("lyapunov_l0_station_ids")
        ))
        trajectory = _plain_list(
            sample.get("lyapunov_l0_station_work_trajectory")
        )
        if len(trajectory) != len(mask):
            raise ValueError("station-work trajectory length differs from mask")
        row = tuple(float(value) for value in _plain_list(
            trajectory[requested - 1]
        ))
        if len(row) != len(station_ids) or any(value < 0.0 for value in row):
            raise ValueError("invalid station-work trajectory endpoint")
        start = sample.get("lyapunov_l0_start")
        full_endpoint = sample.get("lyapunov_l0_end")
        if not isinstance(start, Mapping) or not isinstance(full_endpoint, Mapping):
            raise ValueError("intermediate endpoint requires start/full snapshots")
        capacity = float(start["work_capacity"])
        work_weight = float(
            (sample.get("lyapunov_l0_config") or {}).get("work_weight", 1.0)
        )
        endpoint = copy.deepcopy(full_endpoint)
        endpoint["tick"] = int(start.get("tick", 0)) + requested
        endpoint["station_work"] = {
            station_id: value
            for station_id, value in zip(station_ids, row)
        }
        components = dict(endpoint.get("components") or {})
        components["work"] = work_potential(
            row,
            work_capacity=capacity,
            work_weight=work_weight,
        )
        endpoint["components"] = components
        endpoint["analytic_work_endpoint_source"] = (
            "lyapunov_l0_station_work_trajectory"
        )
        endpoint["analytic_work_requested_horizon"] = requested
        return endpoint
    raise ValueError(
        f"sample does not contain a true isolated H={requested} work endpoint"
    )


__all__ = [
    "NOMINAL_RELIEF_SCHEMA_VERSION",
    "SNAPSHOT_NOMINAL_RELIEF_SCHEMA_VERSION",
    "STATION_WORK_TRAJECTORY_SCHEMA_VERSION",
    "VIRTUAL_CANDIDATE_WORK_DRIFT_SCHEMA_VERSION",
    "WORK_HORIZON_LABEL_SCHEMA_VERSION",
    "NominalWorkRelief",
    "SnapshotNominalWorkRelief",
    "VirtualCandidateWorkDrift",
    "WorkServiceDurations",
    "compute_nominal_work_relief",
    "compute_snapshot_nominal_work_relief",
    "compute_virtual_candidate_work_drift",
    "extract_work_horizon_endpoint",
    "work_potential",
]
