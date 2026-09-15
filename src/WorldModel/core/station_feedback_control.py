"""Opt-in station feedback state for closed-loop context dispatch.

This module is deliberately training-free and simulator-read-only.  It does
not modify station admission, task materialisation, the order generator, or
path planning.  Instead it converts already observable station flow signals
into one per-station state that an experimental assigner may use to defer only
new contexts for that station.

The controller separates two cases that must not be conflated:

* ``BRAKE`` is a preventive, reversible drain-only state reached before a
  persistent exit blockage becomes structural;
* ``LOCKED`` is an absorbing diagnostic for a persistent exit blockage.  The
  current simulator has no system-level rescue action, so the dispatch layer
  can only contain this failure by avoiding additional upstream work.

Raw admission rejection counts are never interpreted as independent dispatch
decisions.  Rejection streaks advance by at most one per observed tick, using
counter deltas only to determine whether that tick contained rejection without
grant or release progress.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


STATION_FEEDBACK_SCHEMA_VERSION = "phase_c_station_feedback_closed_loop_v2"
STATION_FEEDBACK_RELEASE_AWARE_SCHEMA_VERSION = (
    "phase_c_station_feedback_closed_loop_v3"
)

STATION_FEEDBACK_RELEASE_SIGNAL_LEGACY = "legacy_high_occupancy_ticks"
STATION_FEEDBACK_RELEASE_SIGNAL_SERVICE_DUE = "service_due_overdue_v1"
STATION_FEEDBACK_RELEASE_SIGNALS = (
    STATION_FEEDBACK_RELEASE_SIGNAL_LEGACY,
    STATION_FEEDBACK_RELEASE_SIGNAL_SERVICE_DUE,
)

STATION_FEEDBACK_EARLY_RULE_MULTI_EVIDENCE = "multi_evidence_v2"
STATION_FEEDBACK_EARLY_RULE_SERVICE_DUE = "service_due_guarded_v3"
STATION_FEEDBACK_EARLY_RULES = (
    STATION_FEEDBACK_EARLY_RULE_MULTI_EVIDENCE,
    STATION_FEEDBACK_EARLY_RULE_SERVICE_DUE,
)

STATION_FEEDBACK_MODE_OFF = "off"
STATION_FEEDBACK_MODE_SHADOW = "shadow_v1"
STATION_FEEDBACK_MODE_ACTIVE = "active_v1"
# V2 is an explicit opt-in controller profile.  V1 mode names and behaviour
# remain available so the late-BRAKE baseline can be replayed unchanged.
STATION_FEEDBACK_MODE_SHADOW_V2 = "shadow_v2"
STATION_FEEDBACK_MODE_ACTIVE_V2 = "active_v2"
# V3 keeps the same station-local filter but replaces elapsed busy time with
# an explicit missed-release signal.  Existing V1/V2 profiles remain frozen.
STATION_FEEDBACK_MODE_SHADOW_V3 = "shadow_v3"
STATION_FEEDBACK_MODE_ACTIVE_V3 = "active_v3"
STATION_FEEDBACK_MODES = (
    STATION_FEEDBACK_MODE_OFF,
    STATION_FEEDBACK_MODE_SHADOW,
    STATION_FEEDBACK_MODE_ACTIVE,
)
STATION_FEEDBACK_SUPPORTED_MODES = (
    *STATION_FEEDBACK_MODES,
    STATION_FEEDBACK_MODE_SHADOW_V2,
    STATION_FEEDBACK_MODE_ACTIVE_V2,
    STATION_FEEDBACK_MODE_SHADOW_V3,
    STATION_FEEDBACK_MODE_ACTIVE_V3,
)
STATION_FEEDBACK_EARLY_BRAKE_EXPERIMENT_MODES = (
    STATION_FEEDBACK_MODE_OFF,
    STATION_FEEDBACK_MODE_SHADOW_V2,
    STATION_FEEDBACK_MODE_ACTIVE,
    STATION_FEEDBACK_MODE_ACTIVE_V2,
)
STATION_FEEDBACK_RELEASE_AWARE_EXPERIMENT_MODES = (
    STATION_FEEDBACK_MODE_OFF,
    STATION_FEEDBACK_MODE_SHADOW_V3,
    STATION_FEEDBACK_MODE_ACTIVE_V2,
    STATION_FEEDBACK_MODE_ACTIVE_V3,
)


def station_feedback_mode_is_active(mode: str) -> bool:
    return str(mode) in (
        STATION_FEEDBACK_MODE_ACTIVE,
        STATION_FEEDBACK_MODE_ACTIVE_V2,
        STATION_FEEDBACK_MODE_ACTIVE_V3,
    )


def station_feedback_mode_is_shadow(mode: str) -> bool:
    return str(mode) in (
        STATION_FEEDBACK_MODE_SHADOW,
        STATION_FEEDBACK_MODE_SHADOW_V2,
        STATION_FEEDBACK_MODE_SHADOW_V3,
    )


def station_feedback_mode_uses_early_brake(mode: str) -> bool:
    return str(mode) in (
        STATION_FEEDBACK_MODE_SHADOW_V2,
        STATION_FEEDBACK_MODE_ACTIVE_V2,
        STATION_FEEDBACK_MODE_SHADOW_V3,
        STATION_FEEDBACK_MODE_ACTIVE_V3,
    )


def station_feedback_mode_uses_release_aware_brake(mode: str) -> bool:
    return str(mode) in (
        STATION_FEEDBACK_MODE_SHADOW_V3,
        STATION_FEEDBACK_MODE_ACTIVE_V3,
    )

STATION_FEEDBACK_OPEN = "open"
STATION_FEEDBACK_CAUTION = "caution"
STATION_FEEDBACK_BRAKE = "brake"
STATION_FEEDBACK_RECOVERY = "recovery"
STATION_FEEDBACK_LOCKED = "locked"
STATION_FEEDBACK_STATES = (
    STATION_FEEDBACK_OPEN,
    STATION_FEEDBACK_CAUTION,
    STATION_FEEDBACK_BRAKE,
    STATION_FEEDBACK_RECOVERY,
    STATION_FEEDBACK_LOCKED,
)


@dataclass(frozen=True)
class StationFeedbackConfig:
    """Frozen thresholds for one explicitly enabled development arm."""

    service_ticks: int = 5
    caution_exit_block_streak: int = 5
    brake_exit_block_streak: int = 8
    locked_exit_block_streak: int = 10
    caution_release_drought: int = 12
    brake_release_drought: int = 18
    caution_rejection_streak: int = 5
    brake_rejection_streak: int = 10
    recovery_clear_ticks: int = 5
    recovery_release_count: int = 2
    locked_is_absorbing: bool = True
    early_brake_enabled: bool = False
    early_brake_committed_multiplier: float = 1.4
    early_brake_pressure_streak: int = 3
    early_brake_exit_block_streak: int = 3
    early_brake_release_drought: int = 5
    early_brake_rejection_streak: int = 3
    early_brake_evidence_required: int = 2
    release_signal: str = STATION_FEEDBACK_RELEASE_SIGNAL_LEGACY
    release_handoff_grace_ticks: int = 1
    early_brake_rule: str = STATION_FEEDBACK_EARLY_RULE_MULTI_EVIDENCE

    def __post_init__(self) -> None:
        positive = {
            "service_ticks": self.service_ticks,
            "caution_exit_block_streak": self.caution_exit_block_streak,
            "brake_exit_block_streak": self.brake_exit_block_streak,
            "locked_exit_block_streak": self.locked_exit_block_streak,
            "caution_release_drought": self.caution_release_drought,
            "brake_release_drought": self.brake_release_drought,
            "caution_rejection_streak": self.caution_rejection_streak,
            "brake_rejection_streak": self.brake_rejection_streak,
            "recovery_clear_ticks": self.recovery_clear_ticks,
            "recovery_release_count": self.recovery_release_count,
            "early_brake_pressure_streak": self.early_brake_pressure_streak,
            "early_brake_exit_block_streak": (
                self.early_brake_exit_block_streak
            ),
            "early_brake_release_drought": (
                self.early_brake_release_drought
            ),
            "early_brake_rejection_streak": (
                self.early_brake_rejection_streak
            ),
            "early_brake_evidence_required": (
                self.early_brake_evidence_required
            ),
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.release_handoff_grace_ticks) < 0:
            raise ValueError(
                "release_handoff_grace_ticks must be non-negative"
            )
        if str(self.release_signal) not in STATION_FEEDBACK_RELEASE_SIGNALS:
            raise ValueError(
                "release_signal must be one of: "
                + ", ".join(STATION_FEEDBACK_RELEASE_SIGNALS)
            )
        if str(self.early_brake_rule) not in STATION_FEEDBACK_EARLY_RULES:
            raise ValueError(
                "early_brake_rule must be one of: "
                + ", ".join(STATION_FEEDBACK_EARLY_RULES)
            )
        if (
            self.early_brake_rule == STATION_FEEDBACK_EARLY_RULE_SERVICE_DUE
            and self.release_signal
            != STATION_FEEDBACK_RELEASE_SIGNAL_SERVICE_DUE
        ):
            raise ValueError(
                "service-due early brake requires service-due release signal"
            )
        if not (
            self.caution_exit_block_streak
            < self.brake_exit_block_streak
            < self.locked_exit_block_streak
        ):
            raise ValueError(
                "exit-block thresholds must satisfy caution < brake < locked"
            )
        if self.caution_release_drought >= self.brake_release_drought:
            raise ValueError(
                "release-drought thresholds must satisfy caution < brake"
            )
        if self.caution_rejection_streak >= self.brake_rejection_streak:
            raise ValueError(
                "rejection thresholds must satisfy caution < brake"
            )
        multiplier = float(self.early_brake_committed_multiplier)
        if not math.isfinite(multiplier) or multiplier <= 1.0:
            raise ValueError(
                "early_brake_committed_multiplier must be finite and > 1"
            )
        if int(self.early_brake_evidence_required) > 3:
            raise ValueError(
                "early_brake_evidence_required cannot exceed 3"
            )
        if self.early_brake_enabled:
            if (
                self.early_brake_exit_block_streak
                >= self.brake_exit_block_streak
            ):
                raise ValueError(
                    "early exit-block threshold must precede V1 brake"
                )
            if (
                self.early_brake_release_drought
                >= self.brake_release_drought
            ):
                raise ValueError(
                    "early release-drought threshold must precede V1 brake"
                )
            if (
                self.early_brake_rejection_streak
                >= self.brake_rejection_streak
            ):
                raise ValueError(
                    "early rejection threshold must precede V1 brake"
                )

    @classmethod
    def for_service_ticks(
        cls,
        service_ticks: int,
        **overrides: Any,
    ) -> "StationFeedbackConfig":
        """Derive untuned thresholds from the configured station duration."""

        service = max(1, int(service_ticks))
        caution_block = max(3, service)
        locked_block = max(caution_block + 2, 2 * service)
        brake_block = min(
            locked_block - 1,
            max(caution_block + 1, int(math.ceil(1.5 * service))),
        )
        caution_drought = max(8, 2 * service + 2)
        brake_drought = max(caution_drought + 1, 3 * service + 3)
        caution_rejection = max(3, service)
        brake_rejection = max(caution_rejection + 1, 2 * service)
        values: dict[str, Any] = {
            "service_ticks": service,
            "caution_exit_block_streak": caution_block,
            "brake_exit_block_streak": brake_block,
            "locked_exit_block_streak": locked_block,
            "caution_release_drought": caution_drought,
            "brake_release_drought": brake_drought,
            "caution_rejection_streak": caution_rejection,
            "brake_rejection_streak": brake_rejection,
            "recovery_clear_ticks": max(3, service),
            "recovery_release_count": 2,
            "locked_is_absorbing": True,
            "early_brake_enabled": False,
            "early_brake_committed_multiplier": 1.4,
            "early_brake_pressure_streak": max(
                2, int(math.ceil(0.5 * service))
            ),
            "early_brake_exit_block_streak": max(
                2, int(math.ceil(0.5 * service))
            ),
            "early_brake_release_drought": max(3, service),
            "early_brake_rejection_streak": max(
                2, int(math.ceil(0.5 * service))
            ),
            "early_brake_evidence_required": 2,
            "release_signal": STATION_FEEDBACK_RELEASE_SIGNAL_LEGACY,
            "release_handoff_grace_ticks": 1,
            "early_brake_rule": (
                STATION_FEEDBACK_EARLY_RULE_MULTI_EVIDENCE
            ),
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def for_service_due_release(
        cls,
        service_ticks: int,
        **overrides: Any,
    ) -> "StationFeedbackConfig":
        """Build the isolated V3 profile around an actual release deadline.

        ``release_drought`` thresholds in this profile mean ticks overdue
        after the service robot entered EXITING and the normal one-tick
        handoff grace elapsed.  They no longer count the ordinary service
        countdown itself.
        """

        service = max(1, int(service_ticks))
        caution_overdue = max(2, int(math.ceil(0.4 * service)))
        brake_overdue = max(caution_overdue + 1, service)
        values = cls.for_service_ticks(service).as_dict()
        values.update({
            "release_signal": STATION_FEEDBACK_RELEASE_SIGNAL_SERVICE_DUE,
            "release_handoff_grace_ticks": 1,
            "early_brake_rule": STATION_FEEDBACK_EARLY_RULE_SERVICE_DUE,
            "caution_release_drought": caution_overdue,
            "brake_release_drought": brake_overdue,
            "early_brake_release_drought": caution_overdue,
            "early_brake_exit_block_streak": max(
                3, int(math.ceil(0.5 * service))
            ),
        })
        values.update(overrides)
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        result = {
            "service_ticks": int(self.service_ticks),
            "caution_exit_block_streak": int(
                self.caution_exit_block_streak
            ),
            "brake_exit_block_streak": int(self.brake_exit_block_streak),
            "locked_exit_block_streak": int(self.locked_exit_block_streak),
            "caution_release_drought": int(self.caution_release_drought),
            "brake_release_drought": int(self.brake_release_drought),
            "caution_rejection_streak": int(
                self.caution_rejection_streak
            ),
            "brake_rejection_streak": int(self.brake_rejection_streak),
            "recovery_clear_ticks": int(self.recovery_clear_ticks),
            "recovery_release_count": int(self.recovery_release_count),
            "locked_is_absorbing": bool(self.locked_is_absorbing),
            "early_brake_enabled": bool(self.early_brake_enabled),
            "early_brake_committed_multiplier": float(
                self.early_brake_committed_multiplier
            ),
            "early_brake_pressure_streak": int(
                self.early_brake_pressure_streak
            ),
            "early_brake_exit_block_streak": int(
                self.early_brake_exit_block_streak
            ),
            "early_brake_release_drought": int(
                self.early_brake_release_drought
            ),
            "early_brake_rejection_streak": int(
                self.early_brake_rejection_streak
            ),
            "early_brake_evidence_required": int(
                self.early_brake_evidence_required
            ),
        }
        # Preserve the exact serialized V1/V2 configuration contract.  New
        # fields are emitted only by the explicitly selected V3 profile.
        if (
            self.release_signal != STATION_FEEDBACK_RELEASE_SIGNAL_LEGACY
            or self.early_brake_rule
            != STATION_FEEDBACK_EARLY_RULE_MULTI_EVIDENCE
        ):
            result.update({
                "release_signal": str(self.release_signal),
                "release_handoff_grace_ticks": int(
                    self.release_handoff_grace_ticks
                ),
                "early_brake_rule": str(self.early_brake_rule),
            })
        return result

    @property
    def schema_version(self) -> str:
        if (
            self.release_signal
            == STATION_FEEDBACK_RELEASE_SIGNAL_SERVICE_DUE
        ):
            return STATION_FEEDBACK_RELEASE_AWARE_SCHEMA_VERSION
        return STATION_FEEDBACK_SCHEMA_VERSION


@dataclass(frozen=True)
class StationFeedbackObservation:
    """One read-only station observation at an assignment tick."""

    station_id: int
    tick: int
    capacity: int
    physical_agent_ids: tuple[int, ...]
    committed_load: int
    pre_pick: int
    ready_unadmitted: int
    pipeline_mass: int
    exit_blocked: bool
    admission_attempts: int
    admission_grants: int
    admission_rejections: int
    service_agent_id: Optional[int] = None
    service_status: Optional[str] = None
    service_remaining_ticks: Optional[int] = None
    expected_release_in_ticks: Optional[int] = None
    service_release_due: bool = False

    @property
    def physical(self) -> int:
        return len(self.physical_agent_ids)

    @property
    def upstream_pipeline(self) -> int:
        return int(self.pre_pick + self.ready_unadmitted)


@dataclass(frozen=True)
class StationFeedbackSnapshot:
    """Auditable state-machine output for one station and tick."""

    schema_version: str
    station_id: int
    tick: int
    state: str
    previous_state: str
    transition: Optional[str]
    defer_new_context: bool
    dominant_reason: str
    exit_blocked: bool
    exit_blocked_without_release: bool
    exit_blocked_streak: int
    release_drought: int
    release_signal: str
    service_agent_id: Optional[int]
    service_status: Optional[str]
    service_remaining_ticks: Optional[int]
    expected_release_in_ticks: Optional[int]
    service_release_due: bool
    release_due_without_progress_streak: int
    release_handoff_grace_ticks: int
    release_overdue_ticks: int
    due_exit_blocked: bool
    due_exit_blocked_streak: int
    control_exit_blocked_streak: int
    control_release_drought: int
    rejection_tick_streak: int
    release_delta: int
    admission_attempt_delta: int
    admission_grant_delta: int
    admission_rejection_delta: int
    observation_gap_ticks: int
    physical: int
    committed_load: int
    pre_pick: int
    ready_unadmitted: int
    pipeline_mass: int
    capacity: int
    high_occupancy: bool
    pipeline_pressure: bool
    committed_pressure_threshold: int
    committed_pressure: bool
    committed_pressure_streak: int
    early_brake_evidence_count: int
    early_brake_rule: str
    early_exit_block_evidence: bool
    early_release_drought_evidence: bool
    early_rejection_evidence: bool
    early_brake_condition: bool
    caution_condition: bool
    brake_condition: bool
    locked_condition: bool
    recovery_clear_ticks: int
    recovery_release_count: int

    def as_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": str(self.schema_version),
            "station_id": int(self.station_id),
            "tick": int(self.tick),
            "state": str(self.state),
            "previous_state": str(self.previous_state),
            "transition": self.transition,
            "defer_new_context": bool(self.defer_new_context),
            "dominant_reason": str(self.dominant_reason),
            "exit_blocked": bool(self.exit_blocked),
            "exit_blocked_without_release": bool(
                self.exit_blocked_without_release
            ),
            "exit_blocked_streak": int(self.exit_blocked_streak),
            "release_drought": int(self.release_drought),
            "rejection_tick_streak": int(self.rejection_tick_streak),
            "release_delta": int(self.release_delta),
            "admission_attempt_delta": int(self.admission_attempt_delta),
            "admission_grant_delta": int(self.admission_grant_delta),
            "admission_rejection_delta": int(
                self.admission_rejection_delta
            ),
            "observation_gap_ticks": int(self.observation_gap_ticks),
            "physical": int(self.physical),
            "committed_load": int(self.committed_load),
            "pre_pick": int(self.pre_pick),
            "ready_unadmitted": int(self.ready_unadmitted),
            "pipeline_mass": int(self.pipeline_mass),
            "capacity": int(self.capacity),
            "high_occupancy": bool(self.high_occupancy),
            "pipeline_pressure": bool(self.pipeline_pressure),
            "committed_pressure_threshold": int(
                self.committed_pressure_threshold
            ),
            "committed_pressure": bool(self.committed_pressure),
            "committed_pressure_streak": int(
                self.committed_pressure_streak
            ),
            "early_brake_evidence_count": int(
                self.early_brake_evidence_count
            ),
            "early_exit_block_evidence": bool(
                self.early_exit_block_evidence
            ),
            "early_release_drought_evidence": bool(
                self.early_release_drought_evidence
            ),
            "early_rejection_evidence": bool(
                self.early_rejection_evidence
            ),
            "early_brake_condition": bool(self.early_brake_condition),
            "caution_condition": bool(self.caution_condition),
            "brake_condition": bool(self.brake_condition),
            "locked_condition": bool(self.locked_condition),
            "recovery_clear_ticks": int(self.recovery_clear_ticks),
            "recovery_release_count": int(self.recovery_release_count),
        }
        if self.schema_version == STATION_FEEDBACK_RELEASE_AWARE_SCHEMA_VERSION:
            result.update({
                "release_signal": str(self.release_signal),
                "service_agent_id": (
                    int(self.service_agent_id)
                    if self.service_agent_id is not None
                    else None
                ),
                "service_status": self.service_status,
                "service_remaining_ticks": (
                    int(self.service_remaining_ticks)
                    if self.service_remaining_ticks is not None
                    else None
                ),
                "expected_release_in_ticks": (
                    int(self.expected_release_in_ticks)
                    if self.expected_release_in_ticks is not None
                    else None
                ),
                "service_release_due": bool(self.service_release_due),
                "release_due_without_progress_streak": int(
                    self.release_due_without_progress_streak
                ),
                "release_handoff_grace_ticks": int(
                    self.release_handoff_grace_ticks
                ),
                "release_overdue_ticks": int(self.release_overdue_ticks),
                "due_exit_blocked": bool(self.due_exit_blocked),
                "due_exit_blocked_streak": int(
                    self.due_exit_blocked_streak
                ),
                "control_exit_blocked_streak": int(
                    self.control_exit_blocked_streak
                ),
                "control_release_drought": int(
                    self.control_release_drought
                ),
                "early_brake_rule": str(self.early_brake_rule),
            })
        return result


@dataclass
class _StationFeedbackMemory:
    state: str = STATION_FEEDBACK_OPEN
    last_tick: Optional[int] = None
    physical_agent_ids: tuple[int, ...] = ()
    admission_attempts: int = 0
    admission_grants: int = 0
    admission_rejections: int = 0
    exit_blocked_streak: int = 0
    release_drought: int = 0
    release_due_without_progress_streak: int = 0
    due_exit_blocked_streak: int = 0
    rejection_tick_streak: int = 0
    committed_pressure_streak: int = 0
    recovery_clear_ticks: int = 0
    recovery_release_count: int = 0
    last_snapshot: Optional[StationFeedbackSnapshot] = None


class StationFeedbackController:
    """Maintain independent station flow states without mutating the world."""

    def __init__(self, config: Optional[StationFeedbackConfig] = None) -> None:
        self.config = config or StationFeedbackConfig()
        self._memory: Dict[int, _StationFeedbackMemory] = {}

    def update(
        self,
        observations: Iterable[StationFeedbackObservation],
    ) -> Dict[int, StationFeedbackSnapshot]:
        return {
            int(observation.station_id): self._update_one(observation)
            for observation in observations
        }

    def _update_one(
        self,
        observation: StationFeedbackObservation,
    ) -> StationFeedbackSnapshot:
        station_id = int(observation.station_id)
        tick = int(observation.tick)
        memory = self._memory.setdefault(station_id, _StationFeedbackMemory())
        if memory.last_tick is not None and tick < memory.last_tick:
            raise ValueError("station feedback ticks must be monotone")
        if memory.last_tick == tick and memory.last_snapshot is not None:
            return memory.last_snapshot

        first = memory.last_tick is None
        observation_gap_ticks = (
            0
            if first
            else max(0, tick - int(memory.last_tick) - 1)
        )
        consecutive_observation = first or observation_gap_ticks == 0
        current_physical = tuple(sorted({
            int(agent_id) for agent_id in observation.physical_agent_ids
        }))
        previous_physical = set(memory.physical_agent_ids)
        release_delta = (
            0 if first else len(previous_physical - set(current_physical))
        )
        attempt_delta = (
            0 if first else max(
                0, int(observation.admission_attempts) - memory.admission_attempts
            )
        )
        grant_delta = (
            0 if first else max(
                0, int(observation.admission_grants) - memory.admission_grants
            )
        )
        rejection_delta = (
            0 if first else max(
                0,
                int(observation.admission_rejections)
                - memory.admission_rejections,
            )
        )

        exit_blocked_without_release = bool(
            observation.exit_blocked and release_delta == 0
        )
        if exit_blocked_without_release:
            exit_streak = (
                memory.exit_blocked_streak
                if consecutive_observation else 0
            ) + 1
        else:
            exit_streak = 0

        capacity = max(1, int(observation.capacity))
        high_occupancy = observation.physical >= max(1, capacity - 1)
        drain_demand = (
            observation.physical > 0
            or int(observation.committed_load) > 0
            or int(observation.upstream_pipeline) > 0
        )
        if release_delta > 0 or not (high_occupancy and drain_demand):
            release_drought = 0
        else:
            release_drought = (
                memory.release_drought
                if consecutive_observation else 0
            ) + 1

        # A normal service period is not a release drought.  The release-aware
        # profile starts its clock only after the service robot is actually in
        # EXITING state.  One assignment-tick grace covers the simulator's
        # normal two-phase service->exit handoff; only time beyond that grace
        # is release overdue.
        release_due_without_progress = bool(
            observation.service_release_due and release_delta == 0
        )
        if release_due_without_progress:
            release_due_without_progress_streak = (
                memory.release_due_without_progress_streak
                if consecutive_observation else 0
            ) + 1
        else:
            release_due_without_progress_streak = 0
        release_overdue_ticks = max(
            0,
            int(release_due_without_progress_streak)
            - int(self.config.release_handoff_grace_ticks),
        )
        due_exit_blocked = bool(
            release_due_without_progress and observation.exit_blocked
        )
        if due_exit_blocked:
            due_exit_blocked_streak = (
                memory.due_exit_blocked_streak
                if consecutive_observation else 0
            ) + 1
        else:
            due_exit_blocked_streak = 0

        rejected_without_progress = (
            rejection_delta > 0 and grant_delta == 0 and release_delta == 0
        )
        rejection_streak = 0
        if rejected_without_progress:
            rejection_streak = (
                memory.rejection_tick_streak
                if consecutive_observation else 0
            ) + 1
        pipeline_pressure = (
            int(observation.pipeline_mass) > capacity
            or int(observation.ready_unadmitted) > 0
        )

        cfg = self.config
        release_aware = (
            cfg.release_signal
            == STATION_FEEDBACK_RELEASE_SIGNAL_SERVICE_DUE
        )
        control_exit_streak = (
            due_exit_blocked_streak if release_aware else exit_streak
        )
        control_release_drought = (
            release_overdue_ticks if release_aware else release_drought
        )
        control_exit_blocked_without_release = (
            due_exit_blocked
            if release_aware
            else exit_blocked_without_release
        )
        committed_pressure_threshold = max(
            capacity + 1,
            int(math.ceil(
                capacity * float(cfg.early_brake_committed_multiplier)
            )),
        )
        committed_pressure = bool(
            cfg.early_brake_enabled
            and high_occupancy
            and pipeline_pressure
            and int(observation.committed_load)
            >= committed_pressure_threshold
        )
        if committed_pressure:
            committed_pressure_streak = (
                memory.committed_pressure_streak
                if consecutive_observation else 0
            ) + 1
        else:
            committed_pressure_streak = 0

        early_exit_evidence = bool(
            control_exit_streak >= cfg.early_brake_exit_block_streak
        )
        early_release_evidence = bool(
            control_release_drought >= cfg.early_brake_release_drought
        )
        early_rejection_evidence = bool(
            rejection_streak >= cfg.early_brake_rejection_streak
        )
        early_evidence_count = sum((
            early_exit_evidence,
            early_release_evidence,
            early_rejection_evidence,
        ))
        if cfg.early_brake_rule == STATION_FEEDBACK_EARLY_RULE_SERVICE_DUE:
            early_brake_condition = bool(
                cfg.early_brake_enabled
                and committed_pressure_streak
                >= cfg.early_brake_pressure_streak
                and early_exit_evidence
                and early_release_evidence
            )
        else:
            early_brake_condition = bool(
                cfg.early_brake_enabled
                and committed_pressure_streak
                >= cfg.early_brake_pressure_streak
                and early_evidence_count >= cfg.early_brake_evidence_required
            )
        locked_condition = (
            control_exit_streak >= cfg.locked_exit_block_streak
        )
        brake_exit = (
            control_exit_streak >= cfg.brake_exit_block_streak
            and high_occupancy
            and pipeline_pressure
        )
        brake_drought = (
            control_release_drought >= cfg.brake_release_drought
            and pipeline_pressure
        )
        if release_aware:
            # Capacity rejection is expected while an overbooked station is
            # busy.  It may corroborate an already overdue release, but cannot
            # create BRAKE/CAUTION by itself in the release-aware profile.
            brake_rejection = bool(
                rejection_streak >= cfg.brake_rejection_streak
                and release_overdue_ticks
                >= cfg.caution_release_drought
                and int(observation.ready_unadmitted) > 0
            )
            caution_rejection = bool(
                rejection_streak >= cfg.caution_rejection_streak
                and release_overdue_ticks > 0
            )
        else:
            brake_rejection = bool(
                rejection_streak >= cfg.brake_rejection_streak
                and int(observation.ready_unadmitted) > 0
            )
            caution_rejection = bool(
                rejection_streak >= cfg.caution_rejection_streak
            )
        brake_condition = (
            early_brake_condition
            or brake_exit
            or brake_drought
            or brake_rejection
        )
        caution_exit = bool(
            control_exit_streak >= cfg.caution_exit_block_streak
        )
        early_warning_evidence = bool(
            early_exit_evidence
            or early_release_evidence
            or (early_rejection_evidence and not release_aware)
        )
        caution_condition = (
            pipeline_pressure and (
                caution_exit
                or control_release_drought >= cfg.caution_release_drought
                or caution_rejection
            )
        ) or (
            cfg.early_brake_enabled
            and committed_pressure_streak
            >= cfg.early_brake_pressure_streak
            and early_warning_evidence
        )

        previous_state = str(memory.state)
        state = previous_state
        if (
            previous_state == STATION_FEEDBACK_LOCKED
            and cfg.locked_is_absorbing
        ):
            state = STATION_FEEDBACK_LOCKED
        elif locked_condition:
            state = STATION_FEEDBACK_LOCKED
        elif previous_state == STATION_FEEDBACK_BRAKE:
            if brake_condition:
                state = STATION_FEEDBACK_BRAKE
            elif release_delta > 0:
                state = STATION_FEEDBACK_RECOVERY
                memory.recovery_clear_ticks = 0
                memory.recovery_release_count = int(release_delta)
            else:
                # A preventive brake is lifted only by observed station flow,
                # not merely by a transient disappearance of one warning bit.
                state = STATION_FEEDBACK_BRAKE
        elif previous_state == STATION_FEEDBACK_RECOVERY:
            if brake_condition:
                state = STATION_FEEDBACK_BRAKE
                memory.recovery_clear_ticks = 0
                memory.recovery_release_count = 0
            else:
                memory.recovery_release_count += int(release_delta)
                if (
                    not caution_condition
                    and not control_exit_blocked_without_release
                ):
                    if not consecutive_observation:
                        memory.recovery_clear_ticks = 0
                    memory.recovery_clear_ticks += 1
                else:
                    memory.recovery_clear_ticks = 0
                if (
                    memory.recovery_clear_ticks >= cfg.recovery_clear_ticks
                    and memory.recovery_release_count
                    >= cfg.recovery_release_count
                ):
                    state = STATION_FEEDBACK_OPEN
                    memory.recovery_clear_ticks = 0
                    memory.recovery_release_count = 0
                else:
                    state = STATION_FEEDBACK_RECOVERY
        elif brake_condition:
            state = STATION_FEEDBACK_BRAKE
            memory.recovery_clear_ticks = 0
            memory.recovery_release_count = 0
        elif caution_condition:
            state = STATION_FEEDBACK_CAUTION
        else:
            state = STATION_FEEDBACK_OPEN

        if state == STATION_FEEDBACK_LOCKED:
            reason = "persistent_exit_block"
        elif early_brake_condition:
            reason = (
                "early_service_release_overdue"
                if release_aware
                else "early_committed_pressure"
            )
        elif brake_exit:
            reason = (
                "preventive_due_exit_block"
                if release_aware
                else "preventive_exit_block"
            )
        elif brake_drought:
            reason = (
                "service_release_overdue"
                if release_aware
                else "release_drought"
            )
        elif brake_rejection:
            reason = "rejection_streak"
        elif state == STATION_FEEDBACK_BRAKE:
            reason = "brake_waiting_for_observed_release"
        elif state == STATION_FEEDBACK_CAUTION:
            reason = "early_station_flow_warning"
        elif state == STATION_FEEDBACK_RECOVERY:
            reason = "observed_release_recovery"
        else:
            reason = "normal_station_flow"

        transition = (
            f"{previous_state}->{state}" if state != previous_state else None
        )
        snapshot = StationFeedbackSnapshot(
            schema_version=cfg.schema_version,
            station_id=station_id,
            tick=tick,
            state=state,
            previous_state=previous_state,
            transition=transition,
            defer_new_context=state in (
                STATION_FEEDBACK_BRAKE,
                STATION_FEEDBACK_LOCKED,
            ),
            dominant_reason=reason,
            exit_blocked=bool(observation.exit_blocked),
            exit_blocked_without_release=bool(
                exit_blocked_without_release
            ),
            exit_blocked_streak=int(exit_streak),
            release_drought=int(release_drought),
            release_signal=str(cfg.release_signal),
            service_agent_id=(
                int(observation.service_agent_id)
                if observation.service_agent_id is not None
                else None
            ),
            service_status=observation.service_status,
            service_remaining_ticks=(
                int(observation.service_remaining_ticks)
                if observation.service_remaining_ticks is not None
                else None
            ),
            expected_release_in_ticks=(
                int(observation.expected_release_in_ticks)
                if observation.expected_release_in_ticks is not None
                else None
            ),
            service_release_due=bool(observation.service_release_due),
            release_due_without_progress_streak=int(
                release_due_without_progress_streak
            ),
            release_handoff_grace_ticks=int(
                cfg.release_handoff_grace_ticks
            ),
            release_overdue_ticks=int(release_overdue_ticks),
            due_exit_blocked=bool(due_exit_blocked),
            due_exit_blocked_streak=int(due_exit_blocked_streak),
            control_exit_blocked_streak=int(control_exit_streak),
            control_release_drought=int(control_release_drought),
            rejection_tick_streak=int(rejection_streak),
            release_delta=int(release_delta),
            admission_attempt_delta=int(attempt_delta),
            admission_grant_delta=int(grant_delta),
            admission_rejection_delta=int(rejection_delta),
            observation_gap_ticks=int(observation_gap_ticks),
            physical=int(observation.physical),
            committed_load=int(observation.committed_load),
            pre_pick=int(observation.pre_pick),
            ready_unadmitted=int(observation.ready_unadmitted),
            pipeline_mass=int(observation.pipeline_mass),
            capacity=capacity,
            high_occupancy=bool(high_occupancy),
            pipeline_pressure=bool(pipeline_pressure),
            committed_pressure_threshold=int(committed_pressure_threshold),
            committed_pressure=bool(committed_pressure),
            committed_pressure_streak=int(committed_pressure_streak),
            early_brake_evidence_count=int(early_evidence_count),
            early_brake_rule=str(cfg.early_brake_rule),
            early_exit_block_evidence=bool(early_exit_evidence),
            early_release_drought_evidence=bool(early_release_evidence),
            early_rejection_evidence=bool(early_rejection_evidence),
            early_brake_condition=bool(early_brake_condition),
            caution_condition=bool(caution_condition),
            brake_condition=bool(brake_condition),
            locked_condition=bool(locked_condition),
            recovery_clear_ticks=int(memory.recovery_clear_ticks),
            recovery_release_count=int(memory.recovery_release_count),
        )
        memory.state = state
        memory.last_tick = tick
        memory.physical_agent_ids = current_physical
        memory.admission_attempts = int(observation.admission_attempts)
        memory.admission_grants = int(observation.admission_grants)
        memory.admission_rejections = int(observation.admission_rejections)
        memory.exit_blocked_streak = int(exit_streak)
        memory.release_drought = int(release_drought)
        memory.release_due_without_progress_streak = int(
            release_due_without_progress_streak
        )
        memory.due_exit_blocked_streak = int(due_exit_blocked_streak)
        memory.rejection_tick_streak = int(rejection_streak)
        memory.committed_pressure_streak = int(committed_pressure_streak)
        memory.last_snapshot = snapshot
        return snapshot


def _station_exit_blocked(world: Any, queue: Any) -> bool:
    """Return whether an occupied service slot cannot release to the exit.

    A normal two-phase handoff may satisfy this raw predicate: the previous
    robot is still on the exit cell while the next robot is already in service.
    The controller therefore accumulates a blockage streak only when this raw
    signal is accompanied by no observed physical release progress.
    """

    exit_position = getattr(queue, "exit_position", None)
    service = getattr(queue, "service", None)
    service_agent_id = getattr(service, "agent_id", None)
    if exit_position is None or service_agent_id is None:
        return False
    for agent in world.agents:
        if tuple(agent.position) == tuple(exit_position):
            return True
    return False


def _service_release_projection(
    world: Any,
    queue: Any,
    service_ticks: int,
) -> tuple[Optional[int], Optional[str], Optional[int], Optional[int], bool]:
    """Read the service robot's release phase without mutating the engine.

    Assignment runs before the engine's station-exit phase.  An ``EXITING``
    robot is therefore due to attempt the service->exit handoff in the current
    tick.  ``DELIVERING`` robots still have a normal service countdown and
    must never be treated as a release drought merely because no robot left
    during that countdown.
    """

    service = getattr(queue, "service", None)
    raw_agent_id = getattr(service, "agent_id", None)
    if raw_agent_id is None:
        return None, None, None, None, False
    agent_id = int(raw_agent_id)
    agent = None
    getter = getattr(world, "get_agent", None)
    if callable(getter):
        try:
            agent = getter(agent_id)
        except (KeyError, LookupError):
            agent = None
    if agent is None:
        agent = next(
            (
                candidate
                for candidate in getattr(world, "agents", ())
                if int(getattr(candidate, "agent_id", -1)) == agent_id
            ),
            None,
        )
    if agent is None:
        return agent_id, None, None, None, False

    raw_status = getattr(agent, "status", None)
    status = getattr(raw_status, "name", raw_status)
    status_text = str(status).lower() if status is not None else None
    wait_ticks = max(0, int(getattr(agent, "wait_ticks", 0)))
    if status_text == "exiting":
        return agent_id, status_text, 0, 0, True
    if status_text == "delivering":
        # Delivery completes in the action phase; physical release is attempted
        # in the following tick's station-exit phase.
        return agent_id, status_text, wait_ticks, wait_ticks + 1, False
    return (
        agent_id,
        status_text,
        None,
        max(1, int(service_ticks)) + 1,
        False,
    )


def build_station_feedback_observations(
    world: Any,
    station_snapshots: Mapping[int, Any],
    service_ticks: int = 5,
) -> Sequence[StationFeedbackObservation]:
    """Build lightweight observations without changing station state."""

    rows = []
    tick = int(getattr(world, "tick", 0))
    for raw_station_id, snapshot in sorted(station_snapshots.items()):
        station_id = int(raw_station_id)
        queue = world.station_state.get_queue(station_id)
        if queue is None:
            raise RuntimeError(
                f"missing station queue for feedback station {station_id}"
            )
        metrics = queue.admission_metrics()
        (
            service_agent_id,
            service_status,
            service_remaining_ticks,
            expected_release_in_ticks,
            service_release_due,
        ) = _service_release_projection(world, queue, service_ticks)
        rows.append(StationFeedbackObservation(
            station_id=station_id,
            tick=tick,
            capacity=max(1, int(snapshot.capacity)),
            physical_agent_ids=tuple(sorted(queue.physical_agent_ids())),
            committed_load=int(snapshot.committed_load),
            pre_pick=int(snapshot.pre_pick),
            ready_unadmitted=int(snapshot.ready_unadmitted),
            pipeline_mass=int(snapshot.pipeline_mass),
            exit_blocked=_station_exit_blocked(world, queue),
            admission_attempts=int(metrics.get("attempts", 0)),
            admission_grants=int(metrics.get("granted", 0)),
            admission_rejections=int(metrics.get("rejected_capacity", 0)),
            service_agent_id=service_agent_id,
            service_status=service_status,
            service_remaining_ticks=service_remaining_ticks,
            expected_release_in_ticks=expected_release_in_ticks,
            service_release_due=bool(service_release_due),
        ))
    return tuple(rows)


__all__ = [
    "STATION_FEEDBACK_BRAKE",
    "STATION_FEEDBACK_CAUTION",
    "STATION_FEEDBACK_LOCKED",
    "STATION_FEEDBACK_MODE_ACTIVE",
    "STATION_FEEDBACK_MODE_ACTIVE_V2",
    "STATION_FEEDBACK_MODE_ACTIVE_V3",
    "STATION_FEEDBACK_MODE_OFF",
    "STATION_FEEDBACK_MODE_SHADOW",
    "STATION_FEEDBACK_MODE_SHADOW_V2",
    "STATION_FEEDBACK_MODE_SHADOW_V3",
    "STATION_FEEDBACK_MODES",
    "STATION_FEEDBACK_SUPPORTED_MODES",
    "STATION_FEEDBACK_EARLY_BRAKE_EXPERIMENT_MODES",
    "STATION_FEEDBACK_RELEASE_AWARE_EXPERIMENT_MODES",
    "STATION_FEEDBACK_EARLY_RULE_MULTI_EVIDENCE",
    "STATION_FEEDBACK_EARLY_RULE_SERVICE_DUE",
    "STATION_FEEDBACK_OPEN",
    "STATION_FEEDBACK_RECOVERY",
    "STATION_FEEDBACK_SCHEMA_VERSION",
    "STATION_FEEDBACK_RELEASE_AWARE_SCHEMA_VERSION",
    "STATION_FEEDBACK_RELEASE_SIGNAL_LEGACY",
    "STATION_FEEDBACK_RELEASE_SIGNAL_SERVICE_DUE",
    "STATION_FEEDBACK_STATES",
    "StationFeedbackConfig",
    "StationFeedbackController",
    "StationFeedbackObservation",
    "StationFeedbackSnapshot",
    "build_station_feedback_observations",
    "station_feedback_mode_is_active",
    "station_feedback_mode_is_shadow",
    "station_feedback_mode_uses_early_brake",
    "station_feedback_mode_uses_release_aware_brake",
]
