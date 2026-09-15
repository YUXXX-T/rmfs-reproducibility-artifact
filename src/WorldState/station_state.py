"""
Station State Module
====================
Manages station queue zones: service slots, queue slots, and buffer slots.

Each station has a layered queue zone. Robots enter at the tail (or buffer
if tail is occupied), cascade forward slot-by-slot each tick, and get
serviced at the service slot. Release happens externally via _handle_actions.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger("MAS_RMFS.Engine")
from typing import Dict, List, Optional, Set, Tuple

from Config.config_loader import MapConfig, StationConfig, StationQueueConfig
from WorldState.map_state import CellType


STATION_ADMISSION_PHYSICAL_ONLY = "physical_only_legacy"
STATION_ADMISSION_COMMITTED_V1 = "committed_capacity_v1"
STATION_ADMISSION_COMMITTED_FIFO_V2 = "committed_capacity_fifo_v2"
STATION_ADMISSION_DYNAMIC_ETA_V1 = "dynamic_eta_overbooking_v1"
STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1 = (
    "dynamic_eta_dispatch_queue_v1"
)
_STATION_ADMISSION_MODES = {
    STATION_ADMISSION_PHYSICAL_ONLY,
    STATION_ADMISSION_COMMITTED_V1,
    STATION_ADMISSION_COMMITTED_FIFO_V2,
    STATION_ADMISSION_DYNAMIC_ETA_V1,
    STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1,
}


class SlotType(Enum):
    SERVICE = auto()
    QUEUE = auto()
    BUFFER = auto()


@dataclass
class StationSlot:
    position: Tuple[int, int]
    slot_type: SlotType
    index: int
    agent_id: Optional[int] = None

    @property
    def is_free(self) -> bool:
        return self.agent_id is None

    @property
    def is_occupied(self) -> bool:
        return self.agent_id is not None


@dataclass
class DynamicAdmissionToken:
    """ETA metadata for one admitted robot travelling to a station."""

    grant_tick: int
    initial_eta: int
    predicted_arrival_tick: int
    arrival_tick: Optional[int] = None


class StationQueueState:
    """Per-station queue manager.

    Movement within the zone uses BFS shortest-path: each tick, every
    QUEUING agent advances one cell toward service along the physically
    shortest route through zone cells.  Agents closer to service move first
    so they clear space for those behind.
    """

    def __init__(self, station_id: int, service: StationSlot,
                 queue_slots: List[StationSlot],
                 buffer_slots: List[StationSlot],
                 entry_position: Optional[Tuple[int, int]] = None,
                 exit_position: Optional[Tuple[int, int]] = None):
        self.station_id = station_id
        self.service = service
        self.queue_slots = queue_slots  # Q0=front, Q[-1]=tail
        self.buffer_slots = buffer_slots
        self.entry_position = entry_position
        self.exit_position = exit_position
        self._assigned_agents: Set[int] = set()
        # Preserve the simulator's historical behaviour unless an isolated
        # experiment explicitly enables the corrected admission contract.
        self._admission_mode = STATION_ADMISSION_PHYSICAL_ONLY
        self._admission_attempts = 0
        self._admission_granted = 0
        self._admission_rejected_capacity = 0
        self._admission_idempotent = 0
        # Admission-only experiment instrumentation.  These counters are
        # deliberately mode-agnostic so physical-only, committed, FIFO and
        # ETA-overbooking arms can be compared with the same audit schema.
        self._admission_over_capacity_grants = 0
        self._admission_rollbacks = {
            "entry_occupied": 0,
            "path_failure": 0,
            "other": 0,
        }
        # Ordered reservations that have not acquired a committed token.
        # They are deliberately separate from ``_assigned_agents``: waiting
        # requests do not consume station capacity until promoted.
        self._waiting_reservations: "OrderedDict[int, int]" = OrderedDict()
        self._reservation_sequence = 0
        self._waiting_enqueued = 0
        self._waiting_granted = 0
        self._waiting_cancelled = 0
        self._waiting_requeued = 0
        self._waiting_fairness_rejected = 0
        self._waiting_dispatch_priority_enqueued = 0
        self._waiting_dispatch_priority_fallback = 0
        self._waiting_dispatch_priority_bypass_granted = 0

        # Dynamic ETA-overbooking is an isolated opt-in mode.  The defaults
        # deliberately favour throughput: a healthy station may carry up to
        # 2C physical + in-transit commitments.  ETA weighting and station
        # health only tighten *new* grants; existing tokens are never revoked.
        self._dynamic_tokens: Dict[int, DynamicAdmissionToken] = {}
        self._dynamic_service_ticks = 5
        self._dynamic_max_committed_multiplier = 2.0
        self._dynamic_healthy_extra_ratio = 1.0
        self._dynamic_caution_extra_ratio = 0.5
        self._dynamic_brake_extra_ratio = 0.0
        self._dynamic_eta_near_ticks = 8
        self._dynamic_eta_mid_ticks = 20
        self._dynamic_caution_block_streak = 5
        self._dynamic_brake_block_streak = 10
        self._dynamic_caution_release_drought = 12
        self._dynamic_brake_release_drought = 24
        self._dynamic_first_observed_tick: Optional[int] = None
        self._dynamic_last_observed_tick: Optional[int] = None
        self._dynamic_last_release_tick: Optional[int] = None
        self._dynamic_entry_blocked_streak = 0
        self._dynamic_exit_blocked_streak = 0
        self._dynamic_full_streak = 0
        self._dynamic_entry_blocked_ticks = 0
        self._dynamic_exit_blocked_ticks = 0
        self._dynamic_full_ticks = 0
        self._dynamic_max_entry_blocked_streak = 0
        self._dynamic_max_exit_blocked_streak = 0
        self._dynamic_max_full_streak = 0
        self._dynamic_release_count = 0
        self._dynamic_release_interval_sum = 0
        self._dynamic_release_interval_max = 0
        self._dynamic_physical_capacity_violations = 0
        self._dynamic_granted_by_bucket = {"near": 0, "mid": 0, "far": 0}
        self._dynamic_rejected_by_bucket = {"near": 0, "mid": 0, "far": 0}
        self._dynamic_granted_by_health = {
            "healthy": 0, "caution": 0, "brake": 0,
        }
        self._dynamic_rejected_by_health = {
            "healthy": 0, "caution": 0, "brake": 0,
        }
        self._dynamic_rejected_hard_limit = 0
        self._dynamic_rejected_weighted_limit = 0
        self._dynamic_rejected_near_window = 0
        self._dynamic_rejected_mid_window = 0
        self._dynamic_rollbacks = {
            "entry_occupied": 0,
            "path_failure": 0,
            "other": 0,
        }
        self._dynamic_overbooking_grants = 0
        self._dynamic_max_committed_load = 0
        self._dynamic_max_occupancy = 0
        self._dynamic_max_weighted_mass = 0.0
        self._dynamic_max_weight_limit = 0.0
        self._dynamic_health_ticks = {"healthy": 0, "caution": 0, "brake": 0}
        self._dynamic_eta_sample_count = 0
        self._dynamic_eta_signed_error_sum = 0
        self._dynamic_eta_abs_error_sum = 0
        self._dynamic_eta_abs_error_max = 0
        self._dynamic_eta_late_count = 0
        self._dynamic_eta_early_count = 0

        self._zone_cells: Set[Tuple[int, int]] = self._build_zone_cells()
        self._slot_by_pos: Dict[Tuple[int, int], StationSlot] = {
            s.position: s for s in self._all_slots()
        }
        self._slot_priority: Dict[Tuple[int, int], int] = self._build_slot_priority()

    def _build_zone_cells(self) -> Set[Tuple[int, int]]:
        cells = {self.service.position}
        for s in self.queue_slots:
            cells.add(s.position)
        for s in self.buffer_slots:
            cells.add(s.position)
        if self.entry_position is not None:
            cells.add(self.entry_position)
        return cells

    def _build_slot_priority(self) -> Dict[Tuple[int, int], int]:
        """Slot ordering: service=0, Q0=1, Q1=2, ..., B0=n+1, ... entry=max."""
        pri = {self.service.position: 0}
        for i, s in enumerate(self.queue_slots):
            pri[s.position] = i + 1
        n = len(self.queue_slots)
        for i, s in enumerate(self.buffer_slots):
            pri[s.position] = n + 1 + i
        if self.entry_position is not None:
            pri[self.entry_position] = n + len(self.buffer_slots) + 1
        return pri

    @property
    def tail(self) -> Optional[StationSlot]:
        return self.queue_slots[-1] if self.queue_slots else None

    def get_all_slot_positions(self) -> Set[Tuple[int, int]]:
        positions = {self.service.position}
        for s in self.queue_slots:
            positions.add(s.position)
        for s in self.buffer_slots:
            positions.add(s.position)
        return positions

    @property
    def capacity(self) -> int:
        return 1 + len(self.queue_slots) + len(self.buffer_slots)

    def occupancy(self) -> int:
        """Count agents physically in service + queue + buffer slots (NOT entry)."""
        return sum(1 for s in self._all_slots() if s.is_occupied)

    def physical_agent_ids(self) -> Set[int]:
        """Return agents physically occupying service/queue/buffer slots."""
        return {
            int(slot.agent_id)
            for slot in self._all_slots()
            if slot.agent_id is not None
        }

    def committed_agent_ids(self) -> Set[int]:
        """Return unique physical and admitted in-transit agents.

        ``_assigned_agents`` is the admission-token ledger.  A union with the
        physical slots makes the capacity invariant robust to restored legacy
        snapshots whose token ledger may not contain an already checked-in
        robot.
        """
        return set(self._assigned_agents) | self.physical_agent_ids()

    def committed_load(self) -> int:
        """Number of unique robots consuming current or future queue capacity."""
        return len(self.committed_agent_ids())

    @property
    def admission_mode(self) -> str:
        return self._admission_mode

    def set_admission_mode(self, mode: str) -> None:
        """Select the versioned station-capacity admission contract.

        The corrected mode is opt-in so frozen historical experiments retain
        their exact simulator semantics.  It must be enabled before a run
        starts; switching an already-overcommitted live queue is rejected.
        """
        mode = str(mode)
        if mode not in _STATION_ADMISSION_MODES:
            raise ValueError(
                f"unsupported station admission mode {mode!r}; "
                f"expected one of {sorted(_STATION_ADMISSION_MODES)}"
            )
        if (
            mode in (
                STATION_ADMISSION_COMMITTED_V1,
                STATION_ADMISSION_COMMITTED_FIFO_V2,
            )
            and self.committed_load() > self.capacity
        ):
            raise RuntimeError(
                f"station {self.station_id} is already overcommitted: "
                f"committed_load={self.committed_load()} capacity={self.capacity}"
            )
        if (
            mode in (
                STATION_ADMISSION_DYNAMIC_ETA_V1,
                STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1,
            )
            and self.committed_load() > self.dynamic_hard_limit()
        ):
            raise RuntimeError(
                f"station {self.station_id} exceeds dynamic hard limit: "
                f"committed_load={self.committed_load()} "
                f"hard_limit={self.dynamic_hard_limit()}"
            )
        if self._waiting_reservations and mode != self._admission_mode:
            raise RuntimeError(
                f"station {self.station_id} has pending waiting reservations; "
                f"cannot switch admission mode while waiting requests exist"
            )
        self._admission_mode = mode

    @property
    def uses_fifo_waiting(self) -> bool:
        return self._admission_mode == STATION_ADMISSION_COMMITTED_FIFO_V2

    @property
    def uses_dispatch_priority_waiting(self) -> bool:
        return (
            self._admission_mode
            == STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1
        )

    @property
    def uses_explicit_waiting(self) -> bool:
        return self.uses_fifo_waiting or self.uses_dispatch_priority_waiting

    @property
    def uses_dynamic_eta_overbooking(self) -> bool:
        return self._admission_mode in (
            STATION_ADMISSION_DYNAMIC_ETA_V1,
            STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1,
        )

    def configure_dynamic_admission(
        self,
        *,
        service_ticks: int = 5,
        max_committed_multiplier: float = 2.0,
        healthy_extra_ratio: float = 1.0,
        caution_extra_ratio: float = 0.5,
        brake_extra_ratio: float = 0.0,
        eta_near_ticks: int = 8,
        eta_mid_ticks: int = 20,
        caution_block_streak: Optional[int] = None,
        brake_block_streak: Optional[int] = None,
        caution_release_drought: Optional[int] = None,
        brake_release_drought: Optional[int] = None,
    ) -> None:
        """Configure the opt-in ETA-overbooking controller.

        Ratios are expressed relative to physical station capacity C.  For
        example, ``healthy_extra_ratio=1`` gives a healthy weighted limit of
        2C.  The separate hard count limit remains authoritative even when
        many far-away commitments have low ETA weight.
        """
        service_ticks = max(1, int(service_ticks))
        max_committed_multiplier = float(max_committed_multiplier)
        ratios = (
            float(healthy_extra_ratio),
            float(caution_extra_ratio),
            float(brake_extra_ratio),
        )
        if max_committed_multiplier < 1.0:
            raise ValueError("max_committed_multiplier must be >= 1")
        if any(value < 0.0 for value in ratios):
            raise ValueError("dynamic admission extra ratios must be non-negative")
        if not (0 <= int(eta_near_ticks) < int(eta_mid_ticks)):
            raise ValueError("ETA thresholds must satisfy 0 <= near < mid")

        self._dynamic_service_ticks = service_ticks
        self._dynamic_max_committed_multiplier = max_committed_multiplier
        self._dynamic_healthy_extra_ratio = ratios[0]
        self._dynamic_caution_extra_ratio = ratios[1]
        self._dynamic_brake_extra_ratio = ratios[2]
        self._dynamic_eta_near_ticks = int(eta_near_ticks)
        self._dynamic_eta_mid_ticks = int(eta_mid_ticks)
        self._dynamic_caution_block_streak = int(
            caution_block_streak
            if caution_block_streak is not None
            else max(3, service_ticks)
        )
        self._dynamic_brake_block_streak = int(
            brake_block_streak
            if brake_block_streak is not None
            else max(self._dynamic_caution_block_streak + 1, 2 * service_ticks)
        )
        self._dynamic_caution_release_drought = int(
            caution_release_drought
            if caution_release_drought is not None
            else max(8, 2 * service_ticks + 2)
        )
        self._dynamic_brake_release_drought = int(
            brake_release_drought
            if brake_release_drought is not None
            else max(
                self._dynamic_caution_release_drought + 1,
                4 * service_ticks + 4,
            )
        )
        if self._dynamic_brake_block_streak <= self._dynamic_caution_block_streak:
            raise ValueError("brake_block_streak must exceed caution_block_streak")
        if (
            self._dynamic_brake_release_drought
            <= self._dynamic_caution_release_drought
        ):
            raise ValueError(
                "brake_release_drought must exceed caution_release_drought"
            )

    def dynamic_hard_limit(self) -> int:
        return max(
            int(self.capacity),
            int(round(self.capacity * self._dynamic_max_committed_multiplier)),
        )

    def _dynamic_tick(self, tick: Optional[int] = None) -> int:
        if tick is not None:
            return int(tick)
        if self._dynamic_last_observed_tick is not None:
            return int(self._dynamic_last_observed_tick)
        return 0

    def _dynamic_eta_bucket(self, eta_ticks: int) -> str:
        eta_ticks = max(0, int(eta_ticks))
        if eta_ticks <= self._dynamic_eta_near_ticks:
            return "near"
        if eta_ticks <= self._dynamic_eta_mid_ticks:
            return "mid"
        return "far"

    def _dynamic_eta_weight(self, eta_ticks: int) -> float:
        bucket = self._dynamic_eta_bucket(eta_ticks)
        return {"near": 1.0, "mid": 0.5, "far": 0.25}[bucket]

    def _dynamic_release_drought(self, tick: Optional[int] = None) -> int:
        if self.occupancy() < max(1, self.capacity - 1):
            return 0
        now = self._dynamic_tick(tick)
        reference = self._dynamic_last_release_tick
        if reference is None:
            reference = self._dynamic_first_observed_tick
        if reference is None:
            return 0
        return max(0, now - int(reference))

    def dynamic_health_state(self, tick: Optional[int] = None) -> str:
        block_streak = max(
            self._dynamic_entry_blocked_streak,
            self._dynamic_exit_blocked_streak,
        )
        drought = self._dynamic_release_drought(tick)
        if (
            block_streak >= self._dynamic_brake_block_streak
            or drought >= self._dynamic_brake_release_drought
        ):
            return "brake"
        if (
            block_streak >= self._dynamic_caution_block_streak
            or drought >= self._dynamic_caution_release_drought
        ):
            return "caution"
        return "healthy"

    def dynamic_weight_limit(self, tick: Optional[int] = None) -> float:
        state = self.dynamic_health_state(tick)
        extra_ratio = {
            "healthy": self._dynamic_healthy_extra_ratio,
            "caution": self._dynamic_caution_extra_ratio,
            "brake": self._dynamic_brake_extra_ratio,
        }[state]
        return float(self.capacity) * (1.0 + float(extra_ratio))

    def dynamic_weighted_committed_mass(
        self, tick: Optional[int] = None
    ) -> float:
        now = self._dynamic_tick(tick)
        physical = self.physical_agent_ids()
        mass = float(self.occupancy())
        for agent_id in self._assigned_agents - physical:
            token = self._dynamic_tokens.get(int(agent_id))
            if token is None:
                remaining_eta = 0
            else:
                remaining_eta = max(0, token.predicted_arrival_tick - now)
            mass += self._dynamic_eta_weight(remaining_eta)
        return float(mass)

    def dynamic_release_interval_estimate(self) -> float:
        """Estimate station release cadence without assuming perfect ETA."""
        if self._dynamic_release_count >= 4:
            observed = self._dynamic_release_interval_sum / max(
                1, self._dynamic_release_count - 1
            )
            # A single long historical blockage must not permanently collapse
            # admission.  Keep the estimate within a useful local range.
            return max(
                float(self._dynamic_service_ticks),
                min(float(observed), float(3 * self._dynamic_service_ticks)),
            )
        # Delivery service plus the two-phase station exit is a better cold
        # start than service duration alone for this simulator.
        return float(self._dynamic_service_ticks + 2)

    def dynamic_due_arrival_count(
        self, horizon_ticks: int, tick: Optional[int] = None
    ) -> int:
        now = self._dynamic_tick(tick)
        deadline = now + max(0, int(horizon_ticks))
        physical = self.physical_agent_ids()
        return sum(
            1
            for agent_id, token in self._dynamic_tokens.items()
            if agent_id not in physical
            and token.predicted_arrival_tick <= deadline
        )

    def dynamic_window_limit(
        self, horizon_ticks: int, tick: Optional[int] = None
    ) -> int:
        """Capacity plus predicted releases and an aggressive one-slot slack."""
        health = self.dynamic_health_state(tick)
        if health == "brake":
            slack = 0
        elif health == "caution":
            slack = 1
        else:
            slack = 2
        cadence = self.dynamic_release_interval_estimate()
        release_credits = int(max(0, int(horizon_ticks)) // cadence)
        # The mid horizon is deliberately more permissive: ETA error grows
        # with distance, but there is also time for several station releases.
        # One additional credit avoids treating all mid-bucket commitments as
        # a synchronized burst while the 2C hard limit remains authoritative.
        if int(horizon_ticks) > self._dynamic_eta_near_ticks:
            release_credits += 1
        return min(
            self.dynamic_hard_limit(),
            int(self.capacity + release_credits + slack),
        )

    def dynamic_eta_bucket_counts(
        self, tick: Optional[int] = None
    ) -> Dict[str, int]:
        now = self._dynamic_tick(tick)
        counts = {"near": 0, "mid": 0, "far": 0}
        physical = self.physical_agent_ids()
        for agent_id in self._assigned_agents - physical:
            token = self._dynamic_tokens.get(int(agent_id))
            remaining_eta = (
                max(0, token.predicted_arrival_tick - now)
                if token is not None else 0
            )
            counts[self._dynamic_eta_bucket(remaining_eta)] += 1
        return counts

    def dynamic_token_snapshot(self) -> List[Dict[str, object]]:
        now = self._dynamic_tick()
        physical = self.physical_agent_ids()
        rows = []
        for agent_id, token in sorted(self._dynamic_tokens.items()):
            remaining_eta = max(0, token.predicted_arrival_tick - now)
            rows.append({
                "agent_id": int(agent_id),
                "grant_tick": int(token.grant_tick),
                "initial_eta": int(token.initial_eta),
                "predicted_arrival_tick": int(token.predicted_arrival_tick),
                "remaining_eta": int(remaining_eta),
                "eta_bucket": self._dynamic_eta_bucket(remaining_eta),
                "arrival_tick": (
                    int(token.arrival_tick)
                    if token.arrival_tick is not None else None
                ),
                "physical": bool(agent_id in physical),
            })
        return rows

    def observe_tick(self, world_state) -> None:
        """Observe handoff blockage and release drought for dynamic admission."""
        if not self.uses_dynamic_eta_overbooking:
            return
        tick = int(getattr(world_state, "tick", 0))
        if self._dynamic_first_observed_tick is None:
            self._dynamic_first_observed_tick = tick
        self._dynamic_last_observed_tick = tick
        entry_blocked = False
        exit_blocked = False
        for agent in world_state.agents:
            if self.entry_position is not None and agent.position == self.entry_position:
                active = world_state.task_state.get_active_task_for_agent(
                    agent.agent_id
                )
                is_expected_arrival = (
                    active is not None
                    and str(getattr(active.task_type, "name", active.task_type))
                    == "DELIVER"
                    and getattr(active, "station_id", None) == self.station_id
                )
                if not is_expected_arrival or self.occupancy() >= self.capacity:
                    entry_blocked = True
            if self.exit_position is not None and agent.position == self.exit_position:
                active = world_state.task_state.get_active_task_for_agent(
                    agent.agent_id
                )
                is_expected_exit = (
                    active is not None
                    and str(getattr(active.task_type, "name", active.task_type))
                    == "DELIVER"
                    and getattr(active, "station_id", None) == self.station_id
                )
                if not is_expected_exit:
                    exit_blocked = True
        is_full = self.occupancy() >= self.capacity

        self._dynamic_entry_blocked_streak = (
            self._dynamic_entry_blocked_streak + 1 if entry_blocked else 0
        )
        self._dynamic_exit_blocked_streak = (
            self._dynamic_exit_blocked_streak + 1 if exit_blocked else 0
        )
        self._dynamic_full_streak = self._dynamic_full_streak + 1 if is_full else 0
        self._dynamic_entry_blocked_ticks += int(entry_blocked)
        self._dynamic_exit_blocked_ticks += int(exit_blocked)
        self._dynamic_full_ticks += int(is_full)
        self._dynamic_max_entry_blocked_streak = max(
            self._dynamic_max_entry_blocked_streak,
            self._dynamic_entry_blocked_streak,
        )
        self._dynamic_max_exit_blocked_streak = max(
            self._dynamic_max_exit_blocked_streak,
            self._dynamic_exit_blocked_streak,
        )
        self._dynamic_max_full_streak = max(
            self._dynamic_max_full_streak,
            self._dynamic_full_streak,
        )
        if self.occupancy() > self.capacity:
            self._dynamic_physical_capacity_violations += 1
        health = self.dynamic_health_state(tick)
        self._dynamic_health_ticks[health] += 1
        self._update_dynamic_peaks(tick)

    def _update_dynamic_peaks(self, tick: Optional[int] = None) -> None:
        self._dynamic_max_committed_load = max(
            self._dynamic_max_committed_load, self.committed_load()
        )
        self._dynamic_max_occupancy = max(
            self._dynamic_max_occupancy, self.occupancy()
        )
        self._dynamic_max_weighted_mass = max(
            self._dynamic_max_weighted_mass,
            self.dynamic_weighted_committed_mass(tick),
        )
        self._dynamic_max_weight_limit = max(
            self._dynamic_max_weight_limit,
            self.dynamic_weight_limit(tick),
        )

    def _mark_dynamic_arrival(self, agent_id: int, tick: int) -> None:
        token = self._dynamic_tokens.get(int(agent_id))
        if token is None or token.arrival_tick is not None:
            return
        token.arrival_tick = int(tick)
        error = int(tick) - int(token.predicted_arrival_tick)
        abs_error = abs(error)
        self._dynamic_eta_sample_count += 1
        self._dynamic_eta_signed_error_sum += error
        self._dynamic_eta_abs_error_sum += abs_error
        self._dynamic_eta_abs_error_max = max(
            self._dynamic_eta_abs_error_max, abs_error
        )
        self._dynamic_eta_late_count += int(error > 0)
        self._dynamic_eta_early_count += int(error < 0)

    def _record_dynamic_release(self, tick: int) -> None:
        tick = int(tick)
        if self._dynamic_last_release_tick is not None:
            interval = max(0, tick - self._dynamic_last_release_tick)
            self._dynamic_release_interval_sum += interval
            self._dynamic_release_interval_max = max(
                self._dynamic_release_interval_max, interval
            )
        self._dynamic_last_release_tick = tick
        self._dynamic_release_count += 1

    def waiting_agent_ids(self) -> List[int]:
        """Return pending reservation IDs in the active queue order."""
        if self.uses_dispatch_priority_waiting:
            return [
                int(agent_id)
                for agent_id in sorted(
                    self._waiting_reservations,
                    key=lambda aid: (
                        int(self._waiting_reservations[aid]), int(aid)
                    ),
                )
            ]
        return [int(agent_id) for agent_id in self._waiting_reservations]

    def waiting_reservation_sequence(self, agent_id: int) -> Optional[int]:
        """Return the stable active-mode sequence for a pending reservation."""
        value = self._waiting_reservations.get(int(agent_id))
        return int(value) if value is not None else None

    def is_waiting_reservation(self, agent_id: int) -> bool:
        return int(agent_id) in self._waiting_reservations

    def can_promote_waiting_reservation(self, agent_id: int) -> bool:
        """Return whether a waiting request may attempt admission now.

        FIFO V2 reconsiders only its head after physical capacity is released.
        The dispatch-priority ETA queue instead lets every ready waiter attempt
        the unchanged ETA checks in priority order.  Therefore, an earlier
        near-arrival request may remain queued while a later far-arrival
        request uses otherwise idle overbooking capacity.
        """
        agent_id = int(agent_id)
        if not self.uses_explicit_waiting or not self._waiting_reservations:
            return False
        if self.uses_dispatch_priority_waiting:
            return agent_id in self._waiting_reservations
        head_agent = next(iter(self._waiting_reservations))
        return (
            int(head_agent) == agent_id
            and self.committed_load() < self.capacity
        )

    def enqueue_waiting_reservation(
        self,
        agent_id: int,
        request_tick: Optional[int] = None,
        priority_sequence: Optional[int] = None,
    ) -> int:
        """Register an uncommitted station request exactly once."""
        if not self.uses_explicit_waiting:
            raise RuntimeError(
                "waiting reservations require an explicit waiting mode"
            )
        agent_id = int(agent_id)
        existing = self._waiting_reservations.get(agent_id)
        if existing is not None:
            return int(existing)
        self._reservation_sequence += 1
        if self.uses_dispatch_priority_waiting:
            if priority_sequence is None:
                # Missing task metadata is deterministic and ordered after all
                # normal engine-generated dispatch sequences.
                sequence = 10**12 + self._reservation_sequence
                self._waiting_dispatch_priority_fallback += 1
            else:
                sequence = int(priority_sequence)
                if sequence < 0:
                    raise ValueError("priority_sequence must be non-negative")
                self._waiting_dispatch_priority_enqueued += 1
        else:
            sequence = self._reservation_sequence
        self._waiting_reservations[agent_id] = sequence
        self._waiting_enqueued += 1
        return sequence

    def cancel_waiting_reservation(self, agent_id: int) -> bool:
        """Remove a request whose DELIVER chain was cancelled/reset."""
        agent_id = int(agent_id)
        if agent_id not in self._waiting_reservations:
            return False
        del self._waiting_reservations[agent_id]
        self._waiting_cancelled += 1
        return True

    def requeue_waiting_reservation(
        self,
        agent_id: int,
        request_tick: Optional[int] = None,
        priority_sequence: Optional[int] = None,
    ) -> int:
        """Reinsert a failed transactional activation in its mode's order."""
        agent_id = int(agent_id)
        # FIFO receives a new tail sequence.  The dispatch-priority mode passes
        # the task's original scheduler sequence and therefore keeps its rank.
        self._waiting_reservations.pop(agent_id, None)
        self._waiting_requeued += 1
        return self.enqueue_waiting_reservation(
            agent_id,
            request_tick,
            priority_sequence=priority_sequence,
        )

    def prune_waiting_reservations(self, valid_agent_ids: Set[int]) -> List[int]:
        """Drop stale requests that no longer have an ASSIGNED DELIVER task."""
        valid = {int(agent_id) for agent_id in valid_agent_ids}
        removed = []
        for agent_id in list(self._waiting_reservations):
            if agent_id not in valid:
                del self._waiting_reservations[agent_id]
                self._waiting_cancelled += 1
                removed.append(int(agent_id))
        return removed

    def admission_metrics(self) -> Dict[str, object]:
        """Return a JSON-safe audit snapshot for isolated evaluations."""
        result = {
            "station_id": int(self.station_id),
            "mode": self._admission_mode,
            "capacity": int(self.capacity),
            "occupancy": int(self.occupancy()),
            "committed_load": int(self.committed_load()),
            "assigned_agent_count": int(len(self._assigned_agents)),
            "attempts": int(self._admission_attempts),
            "granted": int(self._admission_granted),
            "rejected_capacity": int(self._admission_rejected_capacity),
            "idempotent": int(self._admission_idempotent),
            "over_capacity_grants": int(
                self._admission_over_capacity_grants
            ),
            "rollbacks": dict(self._admission_rollbacks),
            "waiting_reservation_agent_ids": self.waiting_agent_ids(),
            "waiting_reservation_count": int(len(self._waiting_reservations)),
            "waiting_enqueued": int(self._waiting_enqueued),
            "waiting_granted": int(self._waiting_granted),
            "waiting_cancelled": int(self._waiting_cancelled),
            "waiting_requeued": int(self._waiting_requeued),
            "waiting_fairness_rejected": int(
                self._waiting_fairness_rejected
            ),
            "waiting_order": (
                "dispatch_sequence"
                if self.uses_dispatch_priority_waiting
                else "fifo_enqueue_sequence"
                if self.uses_fifo_waiting
                else None
            ),
            "waiting_reservations": [
                {
                    "agent_id": int(agent_id),
                    "sequence": int(
                        self._waiting_reservations[int(agent_id)]
                    ),
                }
                for agent_id in self.waiting_agent_ids()
            ],
            "waiting_dispatch_priority_enqueued": int(
                self._waiting_dispatch_priority_enqueued
            ),
            "waiting_dispatch_priority_fallback": int(
                self._waiting_dispatch_priority_fallback
            ),
            "waiting_dispatch_priority_bypass_granted": int(
                self._waiting_dispatch_priority_bypass_granted
            ),
            "invariant_holds": bool(
                self.occupancy() <= self.capacity
                and (
                    self._admission_mode
                    not in (
                        STATION_ADMISSION_COMMITTED_V1,
                        STATION_ADMISSION_COMMITTED_FIFO_V2,
                    )
                    or self.committed_load() <= self.capacity
                )
                and (
                    not self.uses_dynamic_eta_overbooking
                    or self.committed_load() <= self.dynamic_hard_limit()
                )
            ),
        }
        if self.uses_dynamic_eta_overbooking:
            eta_samples = max(1, self._dynamic_eta_sample_count)
            release_intervals = max(1, self._dynamic_release_count - 1)
            result["dynamic_eta"] = {
                "hard_limit": int(self.dynamic_hard_limit()),
                "health_state": self.dynamic_health_state(),
                "weighted_committed_mass": round(
                    self.dynamic_weighted_committed_mass(), 6
                ),
                "weighted_limit": round(self.dynamic_weight_limit(), 6),
                "release_interval_estimate": round(
                    self.dynamic_release_interval_estimate(), 6
                ),
                "near_due_arrivals": int(
                    self.dynamic_due_arrival_count(
                        self._dynamic_eta_near_ticks
                    )
                ),
                "mid_due_arrivals": int(
                    self.dynamic_due_arrival_count(
                        self._dynamic_eta_mid_ticks
                    )
                ),
                "near_window_limit": int(
                    self.dynamic_window_limit(self._dynamic_eta_near_ticks)
                ),
                "mid_window_limit": int(
                    self.dynamic_window_limit(self._dynamic_eta_mid_ticks)
                ),
                "eta_bucket_counts": self.dynamic_eta_bucket_counts(),
                "token_count": int(len(self._dynamic_tokens)),
                "tokens": self.dynamic_token_snapshot(),
                "configured": {
                    "service_ticks": int(self._dynamic_service_ticks),
                    "max_committed_multiplier": float(
                        self._dynamic_max_committed_multiplier
                    ),
                    "healthy_extra_ratio": float(
                        self._dynamic_healthy_extra_ratio
                    ),
                    "caution_extra_ratio": float(
                        self._dynamic_caution_extra_ratio
                    ),
                    "brake_extra_ratio": float(
                        self._dynamic_brake_extra_ratio
                    ),
                    "eta_near_ticks": int(self._dynamic_eta_near_ticks),
                    "eta_mid_ticks": int(self._dynamic_eta_mid_ticks),
                    "caution_block_streak": int(
                        self._dynamic_caution_block_streak
                    ),
                    "brake_block_streak": int(
                        self._dynamic_brake_block_streak
                    ),
                    "caution_release_drought": int(
                        self._dynamic_caution_release_drought
                    ),
                    "brake_release_drought": int(
                        self._dynamic_brake_release_drought
                    ),
                },
                "granted_by_bucket": dict(self._dynamic_granted_by_bucket),
                "rejected_by_bucket": dict(self._dynamic_rejected_by_bucket),
                "granted_by_health": dict(self._dynamic_granted_by_health),
                "rejected_by_health": dict(self._dynamic_rejected_by_health),
                "health_ticks": dict(self._dynamic_health_ticks),
                "overbooking_grants": int(self._dynamic_overbooking_grants),
                "rejected_hard_limit": int(
                    self._dynamic_rejected_hard_limit
                ),
                "rejected_weighted_limit": int(
                    self._dynamic_rejected_weighted_limit
                ),
                "rejected_near_window": int(
                    self._dynamic_rejected_near_window
                ),
                "rejected_mid_window": int(
                    self._dynamic_rejected_mid_window
                ),
                "rollbacks": dict(self._dynamic_rollbacks),
                "max_committed_load": int(self._dynamic_max_committed_load),
                "max_occupancy": int(self._dynamic_max_occupancy),
                "max_weighted_committed_mass": round(
                    self._dynamic_max_weighted_mass, 6
                ),
                "max_weighted_limit": round(
                    self._dynamic_max_weight_limit, 6
                ),
                "entry_blocked_ticks": int(
                    self._dynamic_entry_blocked_ticks
                ),
                "exit_blocked_ticks": int(self._dynamic_exit_blocked_ticks),
                "full_ticks": int(self._dynamic_full_ticks),
                "max_entry_blocked_streak": int(
                    self._dynamic_max_entry_blocked_streak
                ),
                "max_exit_blocked_streak": int(
                    self._dynamic_max_exit_blocked_streak
                ),
                "max_full_streak": int(self._dynamic_max_full_streak),
                "release_count": int(self._dynamic_release_count),
                "release_interval_mean": round(
                    self._dynamic_release_interval_sum / release_intervals, 6
                ),
                "release_interval_max": int(
                    self._dynamic_release_interval_max
                ),
                "current_release_drought": int(
                    self._dynamic_release_drought()
                ),
                "physical_capacity_violation_count": int(
                    self._dynamic_physical_capacity_violations
                ),
                "eta_sample_count": int(self._dynamic_eta_sample_count),
                "eta_signed_error_mean": round(
                    self._dynamic_eta_signed_error_sum / eta_samples, 6
                ),
                "eta_abs_error_mean": round(
                    self._dynamic_eta_abs_error_sum / eta_samples, 6
                ),
                "eta_abs_error_max": int(self._dynamic_eta_abs_error_max),
                "eta_late_count": int(self._dynamic_eta_late_count),
                "eta_early_count": int(self._dynamic_eta_early_count),
            }
        return result

    def reserve(
        self,
        agent_id: int,
        request_tick: Optional[int] = None,
        eta_ticks: Optional[int] = None,
    ) -> bool:
        """Acquire one idempotent station admission token for ``agent_id``.

        In committed modes, both robots already in queue slots and admitted
        robots still travelling to the entry consume capacity. FIFO V2 keeps
        rejected requests in enqueue order. The dispatch-priority ETA mode
        keeps rejected requests in scheduler order while reusing the existing
        ETA admission checks below.
        """
        agent_id = int(agent_id)
        self._admission_attempts += 1

        committed = self.committed_agent_ids()
        if agent_id in committed:
            # Repeated activation/replanning of the same DELIVER chain must not
            # consume a second token.  Repair a missing legacy token if the
            # robot is already physically inside the station.
            self._assigned_agents.add(agent_id)
            waiting_sequence = self._waiting_reservations.pop(agent_id, None)
            if waiting_sequence is not None and self.uses_explicit_waiting:
                self._waiting_granted += 1
            if self.uses_dynamic_eta_overbooking and agent_id not in self._dynamic_tokens:
                now = self._dynamic_tick(request_tick)
                is_physical = agent_id in self.physical_agent_ids()
                eta = max(0, int(eta_ticks or 0))
                self._dynamic_tokens[agent_id] = DynamicAdmissionToken(
                    grant_tick=now,
                    initial_eta=eta,
                    predicted_arrival_tick=now + eta,
                    arrival_tick=now if is_physical else None,
                )
            self._admission_idempotent += 1
            return True

        if self.uses_dynamic_eta_overbooking:
            now = self._dynamic_tick(request_tick)
            eta = max(0, int(eta_ticks or 0))
            bucket = self._dynamic_eta_bucket(eta)
            health = self.dynamic_health_state(now)
            load = len(committed)
            projected_mass = (
                self.dynamic_weighted_committed_mass(now)
                + self._dynamic_eta_weight(eta)
            )
            weight_limit = self.dynamic_weight_limit(now)
            projected_near_due = self.dynamic_due_arrival_count(
                self._dynamic_eta_near_ticks, now
            ) + int(eta <= self._dynamic_eta_near_ticks)
            projected_mid_due = self.dynamic_due_arrival_count(
                self._dynamic_eta_mid_ticks, now
            ) + int(eta <= self._dynamic_eta_mid_ticks)
            projected_near_load = self.occupancy() + projected_near_due
            projected_mid_load = self.occupancy() + projected_mid_due
            near_window_limit = self.dynamic_window_limit(
                self._dynamic_eta_near_ticks, now
            )
            mid_window_limit = self.dynamic_window_limit(
                self._dynamic_eta_mid_ticks, now
            )
            rejected = False
            if health == "brake" and load >= self.capacity:
                self._dynamic_rejected_weighted_limit += 1
                rejected = True
            elif load >= self.dynamic_hard_limit():
                self._dynamic_rejected_hard_limit += 1
                rejected = True
            elif projected_near_load > near_window_limit:
                self._dynamic_rejected_near_window += 1
                rejected = True
            elif projected_mid_load > mid_window_limit:
                self._dynamic_rejected_mid_window += 1
                rejected = True
            elif projected_mass > weight_limit + 1e-9:
                self._dynamic_rejected_weighted_limit += 1
                rejected = True
            if rejected:
                self._admission_rejected_capacity += 1
                self._dynamic_rejected_by_bucket[bucket] += 1
                self._dynamic_rejected_by_health[health] += 1
                self._update_dynamic_peaks(now)
                return False

            self._assigned_agents.add(agent_id)
            self._dynamic_tokens[agent_id] = DynamicAdmissionToken(
                grant_tick=now,
                initial_eta=eta,
                predicted_arrival_tick=now + eta,
            )
            self._admission_granted += 1
            self._dynamic_granted_by_bucket[bucket] += 1
            self._dynamic_granted_by_health[health] += 1
            if load >= self.capacity:
                self._dynamic_overbooking_grants += 1
            if self.committed_load() > self.capacity:
                self._admission_over_capacity_grants += 1
            self._update_dynamic_peaks(now)
            if self.committed_load() > self.dynamic_hard_limit():
                violating_load = self.committed_load()
                self._assigned_agents.discard(agent_id)
                self._dynamic_tokens.pop(agent_id, None)
                raise RuntimeError(
                    f"station {self.station_id} dynamic hard limit violated: "
                    f"committed_load={violating_load} "
                    f"hard_limit={self.dynamic_hard_limit()}"
                )
            waiting_sequence = self._waiting_reservations.get(agent_id)
            if waiting_sequence is not None and self.uses_dispatch_priority_waiting:
                if any(
                    int(other_sequence) < int(waiting_sequence)
                    for other_id, other_sequence in self._waiting_reservations.items()
                    if int(other_id) != agent_id
                ):
                    self._waiting_dispatch_priority_bypass_granted += 1
                del self._waiting_reservations[agent_id]
                self._waiting_granted += 1
            return True

        if self._admission_mode in (
            STATION_ADMISSION_COMMITTED_V1,
            STATION_ADMISSION_COMMITTED_FIFO_V2,
        ):
            load = len(committed)
        else:
            load = self.occupancy()

        if load >= self.capacity:
            self._admission_rejected_capacity += 1
            if self.uses_fifo_waiting:
                self.enqueue_waiting_reservation(agent_id, request_tick)
            return False

        if self.uses_fifo_waiting and self._waiting_reservations:
            head_agent = next(iter(self._waiting_reservations))
            if head_agent != agent_id:
                self.enqueue_waiting_reservation(agent_id, request_tick)
                self._waiting_fairness_rejected += 1
                return False
            del self._waiting_reservations[agent_id]
            self._waiting_granted += 1

        self._assigned_agents.add(agent_id)
        self._admission_granted += 1
        if self.committed_load() > self.capacity:
            self._admission_over_capacity_grants += 1

        if (
            self._admission_mode == STATION_ADMISSION_COMMITTED_V1
            and self.committed_load() > self.capacity
        ):
            # Defensive assertion: reserve() is the only token acquisition
            # point, so a violation here is a simulator correctness error.
            violating_load = self.committed_load()
            self._assigned_agents.discard(agent_id)
            raise RuntimeError(
                f"station {self.station_id} admission invariant violated: "
                f"committed_load={violating_load} capacity={self.capacity}"
            )
        return True

    def unreserve(self, agent_id: int, reason: Optional[str] = None):
        """Rollback an admission token before DELIVER activation succeeds.

        A physical queue occupant must retain its token; temporary path
        replanning after activation therefore cannot accidentally create free
        capacity.  Normal completion releases the token via ``release`` or
        ``release_to_exit``.
        """
        agent_id = int(agent_id)
        self._waiting_reservations.pop(agent_id, None)
        if agent_id not in self.physical_agent_ids():
            had_token = (
                agent_id in self._assigned_agents
                or agent_id in self._dynamic_tokens
            )
            if had_token:
                key = (
                    str(reason)
                    if reason in self._admission_rollbacks
                    else "other"
                )
                self._admission_rollbacks[key] += 1
            if self.uses_dynamic_eta_overbooking and agent_id in self._dynamic_tokens:
                key = str(reason) if reason in self._dynamic_rollbacks else "other"
                self._dynamic_rollbacks[key] += 1
            self._assigned_agents.discard(agent_id)
            self._dynamic_tokens.pop(agent_id, None)

    # ---- BFS zone helpers --------------------------------------------------

    def _bfs_dist(self, start: Tuple[int, int], goal: Tuple[int, int],
                  blocked: Set[Tuple[int, int]]) -> int:
        """BFS distance from *start* to *goal* through zone cells, avoiding *blocked*.

        Returns the step count, or a large sentinel (999) if unreachable.
        """
        if start == goal:
            return 0
        from collections import deque
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            pos, dist = queue.popleft()
            for nr, nc in self._neighbors(pos):
                npos = (nr, nc)
                if npos == goal:
                    return dist + 1
                if npos in visited or npos in blocked or npos not in self._zone_cells:
                    continue
                visited.add(npos)
                queue.append((npos, dist + 1))
        return 999

    def _bfs_next_step(self, start: Tuple[int, int], goal: Tuple[int, int],
                       blocked: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Return the next cell on the BFS shortest path from *start* to *goal*.

        Only moves through zone cells, avoiding *blocked*.
        Returns None if no path or already at goal.
        """
        if start == goal:
            return None
        from collections import deque
        visited = {start}
        queue = deque([(start, [start])])
        while queue:
            pos, path = queue.popleft()
            for nr, nc in self._neighbors(pos):
                npos = (nr, nc)
                if npos in visited:
                    continue
                new_path = path + [npos]
                if npos == goal:
                    return new_path[1]
                if npos in blocked or npos not in self._zone_cells:
                    continue
                visited.add(npos)
                queue.append((npos, new_path))
        return None

    @staticmethod
    def _neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = pos
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

    # ---- Main advance (replaces cascade) -----------------------------------

    def cascade(self, world_state) -> List[Tuple[int, Tuple[int, int]]]:
        """BFS-based advance: each QUEUING agent moves 1 step toward service.

        Processing order: agents at lower slot-priority (closer to service)
        move first, freeing their slot for the agent behind.  Each agent
        only targets slots with strictly lower priority than its own — it
        never moves backward in the queue.
        """
        from WorldState.agent_state import AgentStatus

        queuing_agents: List[Tuple[Optional[StationSlot], object]] = []
        for slot in self._all_slots():
            if slot.agent_id is not None:
                agent = world_state.get_agent(slot.agent_id)
                if agent.status == AgentStatus.QUEUING:
                    queuing_agents.append((slot, agent))

        if self.entry_position is not None:
            for agent in world_state.agents:
                if (agent.status == AgentStatus.QUEUING
                        and agent.position == self.entry_position):
                    already = any(a.agent_id == agent.agent_id for _, a in queuing_agents)
                    if not already:
                        queuing_agents.append((None, agent))

        if not queuing_agents:
            return []

        max_pri = max(self._slot_priority.values()) + 1
        queuing_agents.sort(
            key=lambda x: (self._slot_priority.get(x[1].position, max_pri), x[1].agent_id)
        )

        occupied_cells = {a.position for a in world_state.agents}
        service_pos = self.service.position
        moved_agents = []
        claimed_targets: Set[Tuple[int, int]] = set()

        for from_slot, agent in queuing_agents:
            if agent.position == service_pos:
                continue

            my_pri = self._slot_priority.get(agent.position, max_pri)
            target = self._find_target_for_agent(my_pri, claimed_targets)
            if target is None:
                continue

            claimed_targets.add(target)

            blocked = (occupied_cells - {agent.position}) & self._zone_cells
            next_pos = self._bfs_next_step(agent.position, target, blocked)
            if next_pos is None:
                continue

            if next_pos in occupied_cells:
                slot_at_next = self._slot_by_pos.get(next_pos)
                if slot_at_next is None or slot_at_next.agent_id is not None:
                    continue

            old_pos = agent.position

            if from_slot is not None:
                from_slot.agent_id = None
            to_slot = self._slot_by_pos.get(next_pos)
            if to_slot is not None:
                to_slot.agent_id = agent.agent_id

            agent.position = next_pos
            occupied_cells.discard(old_pos)
            occupied_cells.add(next_pos)

            if agent.carried_pod_id is not None:
                pod = world_state.pod_state.get_pod(agent.carried_pod_id)
                if pod:
                    pod.current_position = next_pos

            moved_agents.append((agent.agent_id, next_pos))
            logger.debug(
                "Station %d advance: Agent #%d %s -> %s",
                self.station_id, agent.agent_id, old_pos, next_pos,
            )

            if next_pos == service_pos:
                agent.status = AgentStatus.DELIVERING

        return moved_agents

    def _find_target_for_agent(self, my_priority: int,
                               claimed: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find the closest-to-service free slot with priority < my_priority."""
        if (self.service.is_free
                and self.service.position not in claimed
                and 0 < my_priority):
            return self.service.position

        for slot in self.queue_slots:
            if self._slot_priority[slot.position] >= my_priority:
                break
            if slot.is_free and slot.position not in claimed:
                return slot.position

        for slot in self.buffer_slots:
            if self._slot_priority[slot.position] >= my_priority:
                break
            if slot.is_free and slot.position not in claimed:
                return slot.position

        return None

    def release(self, agent_id: int):
        """Free slot when DELIVER completes."""
        for slot in self._all_slots():
            if slot.agent_id == agent_id:
                slot.agent_id = None
                break
        self._assigned_agents.discard(agent_id)
        self._dynamic_tokens.pop(int(agent_id), None)
        self._waiting_reservations.pop(int(agent_id), None)

    def check_in_from_entry(self, agent_id: int, world_state) -> bool:
        """Mark agent as QUEUING when it arrives at entry_position.

        The agent stays at entry — movement into the zone happens during
        the next cascade (BFS advance) tick.  Returns False if entry is
        wrong or zone is completely full.
        """
        agent = world_state.get_agent(agent_id)
        if agent.position != self.entry_position:
            return False

        if (
            self._admission_mode in (
                STATION_ADMISSION_COMMITTED_V1,
                STATION_ADMISSION_COMMITTED_FIFO_V2,
                STATION_ADMISSION_DYNAMIC_ETA_V1,
                STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1,
            )
            and int(agent_id) not in self.committed_agent_ids()
            and not self.reserve(
                agent_id,
                request_tick=int(getattr(world_state, "tick", 0)),
                eta_ticks=0,
            )
        ):
            return False

        has_space = any(s.is_free for s in self._all_slots())
        if not has_space:
            return False

        from WorldState.agent_state import AgentStatus
        agent.status = AgentStatus.QUEUING
        # The entry itself is outside the counted physical slots.  Once an
        # agent is accepted here its reserved token remains committed while it
        # waits for the next station cascade to move into a free slot.
        self._assigned_agents.add(int(agent_id))
        if self.uses_dynamic_eta_overbooking:
            if int(agent_id) not in self._dynamic_tokens:
                now = int(getattr(world_state, "tick", 0))
                self._dynamic_tokens[int(agent_id)] = DynamicAdmissionToken(
                    grant_tick=now,
                    initial_eta=0,
                    predicted_arrival_tick=now,
                )
            self._mark_dynamic_arrival(
                agent_id, int(getattr(world_state, "tick", 0))
            )
            self._update_dynamic_peaks(int(getattr(world_state, "tick", 0)))
        logger.debug(
            "Station %d: Agent #%d checked in at entry %s (QUEUING, stays put)",
            self.station_id, agent_id, self.entry_position,
        )
        return True

    def release_to_exit(self, agent_id: int, world_state) -> bool:
        """Move agent from service slot to exit_position.

        Returns False if exit is occupied. Slot stays occupied (backpressure).
        """
        if self.exit_position is None:
            return False
        occupied = {a.position for a in world_state.agents if a.agent_id != agent_id}
        if self.exit_position in occupied:
            return False
        for slot in self._all_slots():
            if slot.agent_id == agent_id:
                agent = world_state.get_agent(agent_id)
                old_pos = slot.position
                slot.agent_id = None
                agent.position = self.exit_position
                if agent.carried_pod_id is not None:
                    pod = world_state.pod_state.get_pod(agent.carried_pod_id)
                    if pod:
                        pod.current_position = self.exit_position
                self._assigned_agents.discard(agent_id)
                self._dynamic_tokens.pop(int(agent_id), None)
                self._waiting_reservations.pop(int(agent_id), None)
                if self.uses_dynamic_eta_overbooking:
                    self._record_dynamic_release(
                        int(getattr(world_state, "tick", 0))
                    )
                logger.debug(
                    "Station %d: Agent #%d released from %s%s -> exit %s",
                    self.station_id, agent_id,
                    slot.slot_type.name, old_pos, self.exit_position,
                )
                return True
        return False

    def get_slot_for_agent(self, agent_id: int) -> Optional[StationSlot]:
        """Find slot occupied by agent_id."""
        for slot in self._all_slots():
            if slot.agent_id == agent_id:
                return slot
        return None

    def _all_slots(self):
        yield self.service
        yield from self.queue_slots
        yield from self.buffer_slots


def _auto_layout(station: StationConfig, map_rows: int, map_cols: int,
                 queue_cfg: StationQueueConfig, map_state=None):
    """Auto-generate queue zone positions based on station edge position.

    Truncates queue/buffer if a position overlaps with a pod home or obstacle.
    Returns (service, queue_positions, buffer_positions, entry, exit).
    """
    r, c = station.row, station.col

    if r == 0:
        dr, dc = 1, 0
    elif r == map_rows - 1:
        dr, dc = -1, 0
    elif c == 0:
        dr, dc = 0, 1
    elif c == map_cols - 1:
        dr, dc = 0, -1
    else:
        dr, dc = 1, 0

    def _is_valid(pos):
        pr, pc = pos
        if not (0 <= pr < map_rows and 0 <= pc < map_cols):
            return False
        if map_state is not None:
            cell = map_state.grid[pr][pc]
            if cell in (CellType.OBSTACLE, CellType.POD_HOME):
                return False
        return True

    def _is_free(pos):
        """Entry/exit must be FREE only (not POD_HOME)."""
        pr, pc = pos
        if not (0 <= pr < map_rows and 0 <= pc < map_cols):
            return False
        if map_state is not None:
            return map_state.grid[pr][pc] == CellType.FREE
        return True

    service_pos = (r, c)
    if not _is_valid(service_pos):
        return service_pos, [], [], None, None

    queue_positions = []
    for i in range(queue_cfg.queue_length):
        qr = service_pos[0] + dr * (i + 1)
        qc = service_pos[1] + dc * (i + 1)
        pos = (qr, qc)
        if not _is_valid(pos):
            break
        queue_positions.append(pos)

    tail_pos = queue_positions[-1] if queue_positions else service_pos
    if dr != 0:
        perp_dr, perp_dc = 0, 1
    else:
        perp_dr, perp_dc = 1, 0

    buffer_positions = []
    for i in range(queue_cfg.buffer_length):
        br = tail_pos[0] + perp_dr * (i + 1)
        bc = tail_pos[1] + perp_dc * (i + 1)
        pos = (br, bc)
        if not _is_valid(pos):
            break
        buffer_positions.append(pos)

    zone_cells = {service_pos} | set(queue_positions) | set(buffer_positions)

    entry_pos = None
    anchor = buffer_positions[-1] if buffer_positions else tail_pos
    entry_candidates = [
        (anchor[0] + perp_dr, anchor[1] + perp_dc),
        (anchor[0] + dr, anchor[1] + dc),
        (anchor[0] - dr, anchor[1] - dc),
    ]
    for cand in entry_candidates:
        if _is_free(cand) and cand not in zone_cells:
            entry_pos = cand
            break

    exit_pos = None
    exit_candidates = [
        (service_pos[0] - perp_dr, service_pos[1] - perp_dc),
        (service_pos[0] + perp_dr, service_pos[1] + perp_dc),
    ]
    for cand in exit_candidates:
        if _is_free(cand) and cand not in zone_cells:
            exit_pos = cand
            break

    return service_pos, queue_positions, buffer_positions, entry_pos, exit_pos


class StationState:
    """Container managing all station queue zones."""

    def __init__(self, config, map_state):
        self.stations: Dict[int, StationQueueState] = {}
        self._all_zone_positions: Set[Tuple[int, int]] = set()

        map_cfg = config.map
        zone_marks = {
            CellType.STATION_SERVICE: [],
            CellType.STATION_QUEUE: [],
            CellType.STATION_BUFFER: [],
            CellType.STATION_EXIT: [],
            CellType.STATION_ENTRY: [],
        }

        for station_cfg in map_cfg.stations:
            q_cfg = station_cfg.queue
            if q_cfg is None:
                continue

            # Resolve positions: explicit or auto-layout
            if q_cfg.service is not None and q_cfg.queue is not None:
                service_pos = q_cfg.service
                queue_positions = q_cfg.queue
                buffer_positions = q_cfg.buffer or []
                entry_pos = q_cfg.entry
                exit_pos = q_cfg.exit
            else:
                service_pos, queue_positions, buffer_positions, entry_pos, exit_pos = (
                    _auto_layout(station_cfg, map_cfg.rows, map_cfg.cols, q_cfg, map_state)
                )

            if entry_pos is None:
                logger.warning("Station %d: no valid entry_position found", station_cfg.id)
            if exit_pos is None:
                logger.warning("Station %d: no valid exit_position found", station_cfg.id)

            # Build slots
            service_slot = StationSlot(
                position=service_pos, slot_type=SlotType.SERVICE, index=0
            )
            queue_slots = [
                StationSlot(position=pos, slot_type=SlotType.QUEUE, index=i)
                for i, pos in enumerate(queue_positions)
            ]
            buffer_slots = [
                StationSlot(position=pos, slot_type=SlotType.BUFFER, index=i)
                for i, pos in enumerate(buffer_positions)
            ]

            sq = StationQueueState(
                station_id=station_cfg.id,
                service=service_slot,
                queue_slots=queue_slots,
                buffer_slots=buffer_slots,
                entry_position=entry_pos,
                exit_position=exit_pos,
            )
            self.stations[station_cfg.id] = sq

            # Collect positions for map marking
            station_pos = (station_cfg.row, station_cfg.col)
            if service_pos != station_pos:
                zone_marks[CellType.STATION_SERVICE].append(service_pos)
            for pos in queue_positions:
                zone_marks[CellType.STATION_QUEUE].append(pos)
            for pos in buffer_positions:
                zone_marks[CellType.STATION_BUFFER].append(pos)
            if exit_pos is not None:
                zone_marks[CellType.STATION_EXIT].append(exit_pos)
            if entry_pos is not None:
                zone_marks[CellType.STATION_ENTRY].append(entry_pos)

            self._all_zone_positions |= sq.get_all_slot_positions()

        # Mark zone cells on the map grid
        map_state.mark_station_zone(zone_marks)

    def tick(self, world_state) -> List[Tuple[int, Tuple[int, int]]]:
        """Cascade all stations. No release — that happens in _handle_actions."""
        all_moves = []
        for sq in self.stations.values():
            moves = sq.cascade(world_state)
            all_moves.extend(moves)
            sq.observe_tick(world_state)
        return all_moves

    def get_queue(self, station_id: int) -> Optional[StationQueueState]:
        return self.stations.get(station_id)

    def set_admission_mode(self, mode: str) -> None:
        """Apply one versioned admission mode to every configured station."""
        for queue in self.stations.values():
            queue.set_admission_mode(mode)

    def configure_dynamic_admission(self, **kwargs) -> None:
        """Apply one ETA-overbooking configuration to every station."""
        for queue in self.stations.values():
            queue.configure_dynamic_admission(**kwargs)

    def admission_metrics(self) -> List[Dict[str, object]]:
        """Return per-station admission audit snapshots."""
        return [
            self.stations[station_id].admission_metrics()
            for station_id in sorted(self.stations)
        ]

    def cancel_waiting_reservation(self, agent_id: int) -> None:
        """Cancel a robot's pending station request across all stations."""
        for queue in self.stations.values():
            queue.cancel_waiting_reservation(agent_id)

    def waiting_reservation_station(self, agent_id: int) -> Optional[int]:
        """Return the station holding a pending request, if any."""
        agent_id = int(agent_id)
        for station_id, queue in self.stations.items():
            if queue.is_waiting_reservation(agent_id):
                return int(station_id)
        return None

    def get_service_position(self, station_id: int) -> Optional[Tuple[int, int]]:
        sq = self.stations.get(station_id)
        if sq is not None:
            return sq.service.position
        return None

    def get_entry_position(self, station_id: int) -> Optional[Tuple[int, int]]:
        sq = self.stations.get(station_id)
        if sq is not None:
            return sq.entry_position
        return None

    def get_exit_position(self, station_id: int) -> Optional[Tuple[int, int]]:
        sq = self.stations.get(station_id)
        if sq is not None:
            return sq.exit_position
        return None

    def get_all_zone_positions(self) -> Set[Tuple[int, int]]:
        return self._all_zone_positions
