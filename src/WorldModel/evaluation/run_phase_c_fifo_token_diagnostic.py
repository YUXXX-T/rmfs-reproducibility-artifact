"""Matched diagnostic replay for station admission, token flow, and blockage.

This module is deliberately separate from every Phase-C outcome runner.  It
replays the frozen Dynamic-J policy and recorded order manifest under either
``committed_capacity_v1`` or ``committed_capacity_fifo_v2`` and adds passive
instrumentation only.  The instrumentation records, for all stations and all
ticks:

* committed/physical/in-transit token holders;
* FIFO waiters and the stable queue head;
* station entry/exit occupancy and service-slot release attempts;
* DELIVER path-planning failures and transactional requeue reasons;
* every robot's physical position, status, path progress, and stall signals.

The compressed trace is intended for root-cause analysis.  A compact summary
is written separately and is checked against the already completed V1/V2 run
for the same load/seed.  No policy, queue, planner, or simulator method is
modified on disk, and every runtime wrapper returns the original result
unchanged.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional

from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_dynamic_probe_assigner import (
    DynamicPsiDispatchProbeAssigner,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    LOADS,
    S1_CONFIG,
    TOP_M,
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_psi_dynamic_admission_wait import (
    FifoWaitingAuditProbe,
)
from WorldModel.evaluation.run_phase_c_psi_dynamic_probe import (
    _artifact,
    _atomic_json,
    _load_bundle,
    _manifest_path,
    _read_json,
)
from WorldState.station_state import (
    STATION_ADMISSION_COMMITTED_FIFO_V2,
    STATION_ADMISSION_COMMITTED_V1,
)
from WorldState.task_state import TaskType


SCHEMA_VERSION = "phase_c_fifo_token_diagnostic_v1"
TRACE_SCHEMA_VERSION = "phase_c_fifo_token_trace_v1"
BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_SOURCE_ROOT = BASE_ROOT / "psi_dispatch_ablation_551_560_v1"
DEFAULT_V1_ROOT = BASE_ROOT / "psi_dispatch_dynamic_admission_551_560_v1"
DEFAULT_V2_ROOT = BASE_ROOT / "psi_dispatch_dynamic_admission_wait_551_560_v1"
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "fifo_token_diag_551_560_v1"

MODE_TO_ADMISSION = {
    "committed_v1": STATION_ADMISSION_COMMITTED_V1,
    "fifo_v2": STATION_ADMISSION_COMMITTED_FIFO_V2,
}

REFERENCE_KEYS = (
    "completed_orders",
    "completed_tasks",
    "open_order_count",
    "pending_order_count",
    "completed_order_flow_time_p95",
    "avg_task_duration",
    "avg_excess_delay",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "stall_ratio_mean",
    "stall_ratio_max",
    "assign_calls",
    "model_assign_calls",
    "fallback_greedy_calls",
    "model_inference_calls",
    "order_arrival_count",
    "order_arrival_manifest_sha256",
    "order_arrival_replayed",
)
REFERENCE_WARNING_KEYS_V1 = frozenset({
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
})


def _task_summary(task) -> Optional[dict[str, Any]]:
    if task is None:
        return None
    destination = getattr(task, "destination", None)
    return {
        "task_id": int(task.task_id),
        "type": str(getattr(task.task_type, "name", task.task_type)),
        "status": str(getattr(task.status, "name", task.status)),
        "station_id": (
            int(task.station_id) if task.station_id is not None else None
        ),
        "pod_id": int(task.pod_id) if task.pod_id is not None else None,
        "destination": list(destination) if destination is not None else None,
    }


def _agent_summary(world, agent) -> dict[str, Any]:
    active = world.task_state.get_active_task_for_agent(agent.agent_id)
    nxt = world.task_state.get_next_task_for_agent(agent.agent_id)
    next_step = None
    if agent.has_path and agent.path_index < len(agent.path):
        next_step = list(agent.path[agent.path_index])
    return {
        "agent_id": int(agent.agent_id),
        "position": list(agent.position),
        "status": str(getattr(agent.status, "name", agent.status)),
        "has_path": bool(agent.has_path),
        "path_index": int(agent.path_index),
        "path_length": int(len(agent.path)),
        "path_remaining": max(0, int(len(agent.path) - agent.path_index)),
        "next_step": next_step,
        "moved_this_tick": bool(agent.moved_this_tick),
        "stuck_this_tick": bool(agent.stuck_this_tick),
        "traffic_blocked_this_tick": bool(agent.traffic_blocked_this_tick),
        "plan_failed_streak": int(agent.plan_failed_streak),
        "stationary_ticks": int(agent.stationary_ticks),
        "wait_ticks": int(agent.wait_ticks),
        "assigned_task_id": (
            int(agent.assigned_task_id)
            if agent.assigned_task_id is not None
            else None
        ),
        "carried_pod_id": (
            int(agent.carried_pod_id)
            if agent.carried_pod_id is not None
            else None
        ),
        "station_waiting_station_id": (
            int(agent.station_waiting_station_id)
            if agent.station_waiting_station_id is not None
            else None
        ),
        "station_waiting_since_tick": (
            int(agent.station_waiting_since_tick)
            if agent.station_waiting_since_tick is not None
            else None
        ),
        "station_waiting_sequence": (
            int(agent.station_waiting_sequence)
            if agent.station_waiting_sequence is not None
            else None
        ),
        "active_task": _task_summary(active),
        "next_task": _task_summary(nxt),
    }


def _position_occupant(world, position) -> Optional[int]:
    if position is None:
        return None
    target = tuple(position)
    for agent in world.agents:
        if tuple(agent.position) == target:
            return int(agent.agent_id)
    return None


def _station_for_goal(world, goal) -> Optional[int]:
    target = tuple(goal)
    for station_id, queue in world.station_state.stations.items():
        if queue.entry_position is not None and tuple(queue.entry_position) == target:
            return int(station_id)
        if queue.exit_position is not None and tuple(queue.exit_position) == target:
            return int(station_id)
    return None


class TokenFlowDiagnosticProbe:
    """Passive all-station event and physical-state capture."""

    def __init__(self, engine, trace_tmp_path: Path, trace_stride: int = 1):
        self.engine = engine
        self.trace_tmp_path = Path(trace_tmp_path)
        self.trace_stride = max(1, int(trace_stride))
        self._writer = gzip.open(
            self.trace_tmp_path, "wt", encoding="utf-8", compresslevel=5
        )
        self._pending_events: list[dict[str, Any]] = []
        self._event_counts: Counter[str] = Counter()
        self._event_counts_by_station: dict[int, Counter[str]] = defaultdict(Counter)
        self._requeue_reasons: Counter[str] = Counter()
        self._path_failures_this_tick: dict[tuple[int, int], dict[str, Any]] = {}
        self._token_since: dict[tuple[int, int], int] = {}
        self._previous_position: dict[int, tuple[int, int]] = {
            int(a.agent_id): tuple(a.position) for a in engine.world.agents
        }
        self._in_transit_stationary_streak: dict[tuple[int, int], int] = {}
        self._max_in_transit_stationary_streak: dict[int, int] = defaultdict(int)
        self._max_token_age: dict[int, int] = defaultdict(int)
        self._max_waiting_age: dict[int, int] = defaultdict(int)
        self._max_waiting_depth: dict[int, int] = defaultdict(int)
        self._max_full_wait_streak: dict[int, int] = defaultdict(int)
        self._full_wait_streak: dict[int, int] = defaultdict(int)
        self._max_stable_promotable_head_streak: dict[int, int] = defaultdict(int)
        self._stable_promotable_head: dict[int, tuple[int, Optional[int], int]] = {}
        self._max_exit_blocked_streak: dict[int, int] = defaultdict(int)
        self._exit_blocked_streak: dict[int, int] = defaultdict(int)
        self._zone_moves_this_tick: list[dict[str, Any]] = []
        self._trace_rows = 0
        self._observed_ticks = 0
        self._final_station_snapshot: list[dict[str, Any]] = []
        self._install_runtime_wrappers()

    def _event(
        self,
        kind: str,
        *,
        station_id: Optional[int] = None,
        agent_id: Optional[int] = None,
        **extra,
    ) -> None:
        row = {
            "tick": int(self.engine.world.tick),
            "kind": str(kind),
            "station_id": int(station_id) if station_id is not None else None,
            "agent_id": int(agent_id) if agent_id is not None else None,
            **extra,
        }
        self._pending_events.append(row)
        self._event_counts[str(kind)] += 1
        if station_id is not None:
            self._event_counts_by_station[int(station_id)][str(kind)] += 1

    @staticmethod
    def _queue_core(queue) -> dict[str, Any]:
        waiting_ids = queue.waiting_agent_ids()
        return {
            "capacity": int(queue.capacity),
            "occupancy": int(queue.occupancy()),
            "committed_load": int(queue.committed_load()),
            "assigned_agent_ids": sorted(int(x) for x in queue._assigned_agents),
            "physical_agent_ids": sorted(int(x) for x in queue.physical_agent_ids()),
            "waiting_agent_ids": [int(x) for x in waiting_ids],
            "waiting_sequences": [
                queue.waiting_reservation_sequence(agent_id)
                for agent_id in waiting_ids
            ],
        }

    def _install_runtime_wrappers(self) -> None:
        world = self.engine.world
        for station_id, queue in world.station_state.stations.items():
            sid = int(station_id)
            original_reserve = queue.reserve
            original_unreserve = queue.unreserve
            original_enqueue = queue.enqueue_waiting_reservation
            original_requeue = queue.requeue_waiting_reservation
            original_cancel = queue.cancel_waiting_reservation
            original_check_in = queue.check_in_from_entry
            original_release = queue.release
            original_release_to_exit = queue.release_to_exit

            def reserve(agent_id, request_tick=None, *, _q=queue, _sid=sid, _fn=original_reserve):
                before = self._queue_core(_q)
                result = _fn(agent_id, request_tick=request_tick)
                reason = None
                if not result:
                    if before["committed_load"] >= before["capacity"]:
                        reason = "capacity_full"
                    elif (
                        before["waiting_agent_ids"]
                        and before["waiting_agent_ids"][0] != int(agent_id)
                    ):
                        reason = "fifo_fairness"
                    else:
                        reason = "other"
                self._event(
                    "reserve",
                    station_id=_sid,
                    agent_id=agent_id,
                    result=bool(result),
                    reject_reason=reason,
                    request_tick=request_tick,
                    before=before,
                    after=self._queue_core(_q),
                )
                return result

            def unreserve(agent_id, *, _q=queue, _sid=sid, _fn=original_unreserve):
                before = self._queue_core(_q)
                result = _fn(agent_id)
                self._event(
                    "unreserve",
                    station_id=_sid,
                    agent_id=agent_id,
                    before=before,
                    after=self._queue_core(_q),
                )
                return result

            def enqueue(agent_id, request_tick=None, *, _q=queue, _sid=sid, _fn=original_enqueue):
                existed = _q.is_waiting_reservation(agent_id)
                result = _fn(agent_id, request_tick=request_tick)
                self._event(
                    "enqueue_waiting",
                    station_id=_sid,
                    agent_id=agent_id,
                    already_present=bool(existed),
                    sequence=int(result),
                    request_tick=request_tick,
                )
                return result

            def requeue(agent_id, request_tick=None, *, _q=queue, _sid=sid, _fn=original_requeue):
                tick = int(self.engine.world.tick)
                plan_failure = self._path_failures_this_tick.get((tick, int(agent_id)))
                entry_occupant = _position_occupant(
                    self.engine.world, _q.entry_position
                )
                if plan_failure is not None:
                    reason = "path_failure"
                elif entry_occupant is not None and entry_occupant != int(agent_id):
                    reason = "entry_occupied"
                else:
                    reason = "transactional_other"
                before = self._queue_core(_q)
                result = _fn(agent_id, request_tick=request_tick)
                self._requeue_reasons[reason] += 1
                self._event(
                    "requeue_waiting",
                    station_id=_sid,
                    agent_id=agent_id,
                    reason=reason,
                    request_tick=request_tick,
                    entry_occupant=entry_occupant,
                    plan_failure=plan_failure,
                    new_sequence=int(result),
                    before=before,
                    after=self._queue_core(_q),
                )
                return result

            def cancel(agent_id, *, _q=queue, _sid=sid, _fn=original_cancel):
                before = self._queue_core(_q)
                result = _fn(agent_id)
                self._event(
                    "cancel_waiting",
                    station_id=_sid,
                    agent_id=agent_id,
                    result=bool(result),
                    before=before,
                    after=self._queue_core(_q),
                )
                return result

            def check_in(agent_id, world_state, *, _q=queue, _sid=sid, _fn=original_check_in):
                before = self._queue_core(_q)
                result = _fn(agent_id, world_state)
                self._event(
                    "check_in",
                    station_id=_sid,
                    agent_id=agent_id,
                    result=bool(result),
                    entry_occupant=_position_occupant(world_state, _q.entry_position),
                    before=before,
                    after=self._queue_core(_q),
                )
                return result

            def release(agent_id, *, _q=queue, _sid=sid, _fn=original_release):
                before = self._queue_core(_q)
                result = _fn(agent_id)
                self._event(
                    "release",
                    station_id=_sid,
                    agent_id=agent_id,
                    before=before,
                    after=self._queue_core(_q),
                )
                return result

            def release_to_exit(agent_id, world_state, *, _q=queue, _sid=sid, _fn=original_release_to_exit):
                before = self._queue_core(_q)
                exit_occupant_before = _position_occupant(
                    world_state, _q.exit_position
                )
                result = _fn(agent_id, world_state)
                self._event(
                    "release_to_exit",
                    station_id=_sid,
                    agent_id=agent_id,
                    result=bool(result),
                    exit_occupant_before=exit_occupant_before,
                    before=before,
                    after=self._queue_core(_q),
                )
                return result

            queue.reserve = reserve
            queue.unreserve = unreserve
            queue.enqueue_waiting_reservation = enqueue
            queue.requeue_waiting_reservation = requeue
            queue.cancel_waiting_reservation = cancel
            queue.check_in_from_entry = check_in
            queue.release = release
            queue.release_to_exit = release_to_exit

        planner = self.engine.path_planner
        original_plan = planner.plan

        def plan_wrapper(
            _planner,
            agent,
            goal,
            world_state,
            extra_blocked=None,
        ):
            result = original_plan(
                agent, goal, world_state, extra_blocked=extra_blocked
            )
            if not result and tuple(agent.position) != tuple(goal):
                tick = int(world_state.tick)
                active = world_state.task_state.get_active_task_for_agent(
                    agent.agent_id
                )
                nxt = world_state.task_state.get_next_task_for_agent(agent.agent_id)
                task = active if active is not None else nxt
                station_id = _station_for_goal(world_state, goal)
                if station_id is None and task is not None and task.station_id is not None:
                    station_id = int(task.station_id)
                queue = (
                    world_state.station_state.get_queue(station_id)
                    if station_id is not None
                    else None
                )
                deliver_activation = bool(
                    task is not None
                    and task.task_type == TaskType.DELIVER
                    and queue is not None
                    and queue.entry_position is not None
                    and tuple(goal) == tuple(queue.entry_position)
                )
                row = {
                    "position": list(agent.position),
                    "goal": list(goal),
                    "status": str(getattr(agent.status, "name", agent.status)),
                    "task": _task_summary(task),
                    "deliver_activation": deliver_activation,
                    "extra_blocked": (
                        [list(pos) for pos in sorted(extra_blocked)]
                        if extra_blocked
                        else []
                    ),
                }
                if deliver_activation:
                    self._path_failures_this_tick[(tick, int(agent.agent_id))] = row
                self._event(
                    (
                        "deliver_path_failure"
                        if deliver_activation
                        else "path_failure"
                    ),
                    station_id=station_id,
                    agent_id=agent.agent_id,
                    **row,
                )
            return result

        planner.plan = types.MethodType(plan_wrapper, planner)

        station_state = world.station_state
        original_station_tick = station_state.tick

        def station_tick(world_state):
            result = original_station_tick(world_state)
            self._zone_moves_this_tick = [
                {"agent_id": int(agent_id), "position": list(position)}
                for agent_id, position in result
            ]
            return result

        station_state.tick = station_tick

    def _station_snapshot(self, station_id: int, queue, tick: int) -> dict[str, Any]:
        world = self.engine.world
        committed = set(int(x) for x in queue.committed_agent_ids())
        physical = set(int(x) for x in queue.physical_agent_ids())
        assigned = set(int(x) for x in queue._assigned_agents)
        in_transit = sorted(committed - physical)
        waiting_ids = [int(x) for x in queue.waiting_agent_ids()]

        for agent_id in committed:
            key = (int(station_id), int(agent_id))
            self._token_since.setdefault(key, int(tick))
            self._max_token_age[int(station_id)] = max(
                self._max_token_age[int(station_id)],
                int(tick - self._token_since[key]),
            )
        for key in list(self._token_since):
            if key[0] == int(station_id) and key[1] not in committed:
                del self._token_since[key]

        in_transit_details = []
        for agent_id in in_transit:
            agent = world.get_agent(agent_id)
            if agent is None:
                continue
            key = (int(station_id), int(agent_id))
            previous = self._previous_position.get(agent_id)
            stationary = previous == tuple(agent.position)
            if stationary:
                self._in_transit_stationary_streak[key] = (
                    self._in_transit_stationary_streak.get(key, 0) + 1
                )
            else:
                self._in_transit_stationary_streak[key] = 0
            streak = self._in_transit_stationary_streak[key]
            self._max_in_transit_stationary_streak[int(station_id)] = max(
                self._max_in_transit_stationary_streak[int(station_id)], streak
            )
            in_transit_details.append({
                "agent_id": agent_id,
                "token_age": int(tick - self._token_since.get(key, tick)),
                "stationary_token_streak": int(streak),
                "position": list(agent.position),
                "status": str(getattr(agent.status, "name", agent.status)),
                "path_remaining": max(0, len(agent.path) - agent.path_index),
                "plan_failed_streak": int(agent.plan_failed_streak),
            })

        for key in list(self._in_transit_stationary_streak):
            if key[0] == int(station_id) and key[1] not in in_transit:
                del self._in_transit_stationary_streak[key]

        waiting_details = []
        for agent_id in waiting_ids:
            agent = world.get_agent(agent_id)
            since = (
                int(agent.station_waiting_since_tick)
                if agent is not None and agent.station_waiting_since_tick is not None
                else tick
            )
            age = max(0, int(tick - since))
            self._max_waiting_age[int(station_id)] = max(
                self._max_waiting_age[int(station_id)], age
            )
            waiting_details.append({
                "agent_id": agent_id,
                "sequence": queue.waiting_reservation_sequence(agent_id),
                "since_tick": since,
                "age": age,
                "position": list(agent.position) if agent is not None else None,
            })
        self._max_waiting_depth[int(station_id)] = max(
            self._max_waiting_depth[int(station_id)], len(waiting_ids)
        )

        full = len(committed) >= int(queue.capacity)
        if waiting_ids and full:
            self._full_wait_streak[int(station_id)] += 1
        else:
            self._full_wait_streak[int(station_id)] = 0
        self._max_full_wait_streak[int(station_id)] = max(
            self._max_full_wait_streak[int(station_id)],
            self._full_wait_streak[int(station_id)],
        )

        promotable = bool(waiting_ids and len(committed) < int(queue.capacity))
        if promotable:
            head = waiting_ids[0]
            sequence = queue.waiting_reservation_sequence(head)
            previous = self._stable_promotable_head.get(int(station_id))
            streak = previous[2] + 1 if previous and previous[:2] == (head, sequence) else 1
            self._stable_promotable_head[int(station_id)] = (head, sequence, streak)
            self._max_stable_promotable_head_streak[int(station_id)] = max(
                self._max_stable_promotable_head_streak[int(station_id)], streak
            )
        else:
            self._stable_promotable_head.pop(int(station_id), None)

        service_agent_id = (
            int(queue.service.agent_id)
            if queue.service.agent_id is not None
            else None
        )
        exit_occupant = _position_occupant(world, queue.exit_position)
        if service_agent_id is not None and exit_occupant is not None:
            self._exit_blocked_streak[int(station_id)] += 1
        else:
            self._exit_blocked_streak[int(station_id)] = 0
        self._max_exit_blocked_streak[int(station_id)] = max(
            self._max_exit_blocked_streak[int(station_id)],
            self._exit_blocked_streak[int(station_id)],
        )

        return {
            "station_id": int(station_id),
            "capacity": int(queue.capacity),
            "occupancy": int(queue.occupancy()),
            "committed_load": len(committed),
            "full": bool(full),
            "assigned_agent_ids": sorted(assigned),
            "physical_agent_ids": sorted(physical),
            "in_transit_agent_ids": in_transit,
            "in_transit": in_transit_details,
            "waiting_agent_ids": waiting_ids,
            "waiting": waiting_details,
            "waiting_depth": len(waiting_ids),
            "promotable_waiting_head": bool(promotable),
            "entry_position": (
                list(queue.entry_position)
                if queue.entry_position is not None
                else None
            ),
            "entry_occupant": _position_occupant(world, queue.entry_position),
            "exit_position": (
                list(queue.exit_position)
                if queue.exit_position is not None
                else None
            ),
            "exit_occupant": exit_occupant,
            "service_agent_id": service_agent_id,
            "slots": [
                {
                    "type": str(getattr(slot.slot_type, "name", slot.slot_type)),
                    "index": int(slot.index),
                    "position": list(slot.position),
                    "agent_id": (
                        int(slot.agent_id) if slot.agent_id is not None else None
                    ),
                }
                for slot in [queue.service, *queue.queue_slots, *queue.buffer_slots]
            ],
        }

    def on_tick(self, engine) -> None:
        world = engine.world
        tick = int(world.tick)
        self._observed_ticks += 1
        stations = [
            self._station_snapshot(int(station_id), queue, tick)
            for station_id, queue in sorted(world.station_state.stations.items())
        ]
        self._final_station_snapshot = stations
        if tick % self.trace_stride == 0:
            row = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "tick": tick,
                "traffic_vertex_conflicts": int(
                    getattr(world, "traffic_vertex_conflicts_this_tick", 0)
                ),
                "traffic_swap_conflicts": int(
                    getattr(world, "traffic_swap_conflicts_this_tick", 0)
                ),
                "stations": stations,
                "agents": [
                    _agent_summary(world, agent) for agent in world.agents
                ],
                "zone_moves": self._zone_moves_this_tick,
                "events": self._pending_events,
            }
            self._writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._trace_rows += 1

        for agent in world.agents:
            self._previous_position[int(agent.agent_id)] = tuple(agent.position)
        self._pending_events = []
        self._zone_moves_this_tick = []
        self._path_failures_this_tick = {
            key: value
            for key, value in self._path_failures_this_tick.items()
            if key[0] >= tick
        }

    def close(self) -> None:
        if not self._writer.closed:
            self._writer.close()

    def summary(self) -> dict[str, Any]:
        station_ids = sorted(int(x) for x in self.engine.world.station_state.stations)
        by_station = {}
        for station_id in station_ids:
            by_station[str(station_id)] = {
                "event_counts": dict(
                    sorted(self._event_counts_by_station[station_id].items())
                ),
                "max_waiting_depth": int(self._max_waiting_depth[station_id]),
                "max_waiting_age": int(self._max_waiting_age[station_id]),
                "max_token_age": int(self._max_token_age[station_id]),
                "max_in_transit_stationary_streak": int(
                    self._max_in_transit_stationary_streak[station_id]
                ),
                "max_full_wait_streak": int(
                    self._max_full_wait_streak[station_id]
                ),
                "max_stable_promotable_head_streak": int(
                    self._max_stable_promotable_head_streak[station_id]
                ),
                "max_exit_blocked_streak": int(
                    self._max_exit_blocked_streak[station_id]
                ),
            }
        return {
            "trace_rows": int(self._trace_rows),
            "observed_ticks": int(self._observed_ticks),
            "event_counts": dict(sorted(self._event_counts.items())),
            "requeue_reasons": dict(sorted(self._requeue_reasons.items())),
            "by_station": by_station,
            "final_stations": self._final_station_snapshot,
        }


def _reference_path(root: Path, mode: str, load: str, seed: int) -> Path:
    if mode == "fifo_v2":
        arm = "s1_psi_dynamic_committed_fifo_wait_v2"
    else:
        arm = "s1_psi_dynamic_committed_admission_v1"
    return root / "per_arm" / arm / f"{load}_seed{seed}.json"


def _reference_audit(
    metrics: Mapping[str, Any], reference_path: Path, mode: str
) -> dict[str, Any]:
    if not reference_path.is_file():
        return {
            "passed": False,
            "hard_passed": False,
            "reference_missing": True,
            "reference_path": reference_path.as_posix(),
            "checks": {},
            "warnings": [],
        }
    reference = _read_json(reference_path)
    expected = reference.get("metrics") or {}
    checks = {}
    warnings = []
    warning_keys = (
        REFERENCE_WARNING_KEYS_V1 if mode == "committed_v1" else frozenset()
    )
    for key in REFERENCE_KEYS:
        actual_value = metrics.get(key)
        expected_value = expected.get(key)
        if isinstance(actual_value, float) or isinstance(expected_value, float):
            try:
                passed = abs(float(actual_value) - float(expected_value)) <= 1e-8
            except (TypeError, ValueError):
                passed = actual_value == expected_value
        else:
            passed = actual_value == expected_value
        checks[key] = {
            "passed": bool(passed),
            "severity": "warning" if key in warning_keys else "hard",
            "actual": actual_value,
            "expected": expected_value,
        }
        if not passed and key in warning_keys:
            warnings.append({
                "key": key,
                "actual": actual_value,
                "expected": expected_value,
                "delta": (
                    float(actual_value) - float(expected_value)
                    if actual_value is not None and expected_value is not None
                    else None
                ),
            })
    hard_checks = [
        row for key, row in checks.items() if key not in warning_keys
    ]
    hard_passed = all(row["passed"] for row in hard_checks)
    return {
        # ``passed`` is deliberately the mode-aware gate result.  For V1 the
        # two deadlock fields are baseline-drift warnings, never a fatal gate.
        "passed": bool(hard_passed),
        "hard_passed": bool(hard_passed),
        "reference_missing": False,
        "mode": mode,
        "reference_path": reference_path.as_posix(),
        "reference_sha256": sha256_file(reference_path),
        "checks": checks,
        "warnings": warnings,
    }


def _runtime_hashes() -> dict[str, str]:
    return {
        "diagnostic_runner_sha256": sha256_file(Path(__file__)),
        "engine_sha256": sha256_file(Path("Engine/simulation_engine.py")),
        "station_state_sha256": sha256_file(Path("WorldState/station_state.py")),
        "agent_state_sha256": sha256_file(Path("WorldState/agent_state.py")),
        "planner_sha256": sha256_file(Path(
            "Policies/PathPlanner/PrioritizedPathPlanner/"
            "prioritized_path_planner.py"
        )),
    }


def _run_dir(root: Path, mode: str, load: str, seed: int) -> Path:
    return root / "runs" / mode / f"{load}_seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODE_TO_ADMISSION), required=True)
    parser.add_argument("--load", choices=LOADS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--trace-stride", type=int, default=1)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_V1_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--frozen-bundle", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--allow-reference-mismatch", action="store_true")
    args = parser.parse_args()

    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive")
    if args.trace_stride <= 0:
        raise SystemExit("--trace-stride must be positive")

    bundle_path = args.frozen_bundle or (
        args.source_root / "phase_c_psi_dispatch_frozen_protocol.json"
    )
    bundle, protocol = _load_bundle(bundle_path)
    model_checkpoint = _artifact(bundle, "model_checkpoint")
    psi_head_checkpoint = _artifact(bundle, "psi_head_checkpoint")
    psi_scale_contract = _artifact(bundle, "psi_scale_contract")
    config_path = _artifact(bundle, f"config_{args.load}")
    manifest_path = _manifest_path(args.source_root, args.load, args.seed)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    run_dir = _run_dir(args.output_root, args.mode, args.load, args.seed)
    summary_path = run_dir / "diagnostic_summary.json"
    trace_path = run_dir / "token_flow_trace.jsonl.gz"
    trace_tmp_path = run_dir / "token_flow_trace.jsonl.gz.tmp"
    runtime_hashes = _runtime_hashes()
    if summary_path.is_file():
        existing = _read_json(summary_path)
        meta = existing.get("meta") or {}
        compatible = bool(
            existing.get("schema_version") == SCHEMA_VERSION
            and meta.get("mode") == args.mode
            and meta.get("load") == args.load
            and int(meta.get("seed", -1)) == int(args.seed)
            and int(meta.get("ticks", -1)) == int(args.ticks)
            and int(meta.get("trace_stride", -1)) == int(args.trace_stride)
            and meta.get("frozen_bundle_sha256") == sha256_file(bundle_path)
            and all(meta.get(key) == value for key, value in runtime_hashes.items())
            and trace_path.is_file()
        )
        if not compatible:
            raise FileExistsError(f"incompatible existing diagnostic: {run_dir}")
        if (
            not bool((existing.get("reference_audit") or {}).get("passed"))
            and not args.allow_reference_mismatch
        ):
            raise RuntimeError(
                f"existing diagnostic does not reproduce its reference: "
                f"{summary_path}"
            )
        print(f"[resume] {summary_path}")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "checkpoint_path": str(model_checkpoint),
        "top_m": TOP_M,
        "energy_conv_random_flip_seed": 0,
        **S1_CONFIG,
    }
    assigner = DynamicPsiDispatchProbeAssigner(
        psi_head_checkpoint=str(psi_head_checkpoint),
        psi_scale_contract=str(psi_scale_contract),
        dynamic_trace_enabled=True,
        dynamic_trace_max_records=500,
        **common,
    )

    import WorldModel.evaluation.evaluate_online_v6 as eval_module

    original_builder = eval_module._build_engine
    holder: dict[str, Any] = {}

    def build_with_diagnostic(cfg, task_assigner=None):
        engine = original_builder(cfg, task_assigner=task_assigner)
        engine.world.station_state.set_admission_mode(
            MODE_TO_ADMISSION[args.mode]
        )
        diagnostic = TokenFlowDiagnosticProbe(
            engine, trace_tmp_path=trace_tmp_path, trace_stride=args.trace_stride
        )
        engine.on_tick_callbacks.append(diagnostic.on_tick)
        holder["engine"] = engine
        holder["diagnostic"] = diagnostic
        if args.mode == "fifo_v2":
            fifo_audit = FifoWaitingAuditProbe(engine, trace_max_records=0)
            engine.on_tick_callbacks.append(fifo_audit.on_tick)
            holder["fifo_audit"] = fifo_audit
        return engine

    eval_module._build_engine = build_with_diagnostic
    run_error: Optional[BaseException] = None
    try:
        print(
            f"[run] token diagnostic mode={args.mode} load={args.load} "
            f"seed={args.seed} ticks={args.ticks}"
        )
        metrics = eval_module._run_one_assigner(
            str(config_path),
            assigner,
            int(args.seed),
            int(args.ticks),
            trace_label=f"PhaseCFifoTokenDiagnostic-{args.mode}",
            recorded_orders_path=str(manifest_path),
        )
    except BaseException as exc:  # Preserve the original traceback after close.
        run_error = exc
        metrics = {}
    finally:
        eval_module._build_engine = original_builder
        diagnostic = holder.get("diagnostic")
        if diagnostic is not None:
            diagnostic.close()
    if run_error is not None:
        raise run_error

    diagnostic = holder.get("diagnostic")
    engine = holder.get("engine")
    if diagnostic is None or engine is None:
        raise RuntimeError("diagnostic engine/probe was not attached")
    os.replace(trace_tmp_path, trace_path)

    fifo_audit_payload = None
    if args.mode == "fifo_v2":
        fifo_audit = holder.get("fifo_audit")
        if fifo_audit is None:
            raise RuntimeError("FIFO invariant audit was not attached")
        fifo_audit_payload = fifo_audit.summary()
        if not fifo_audit_payload.get("passed"):
            raise RuntimeError("FIFO invariant audit failed in diagnostic replay")
        metrics.update({
            "waiting_assigned_agent_ticks": fifo_audit_payload[
                "waiting_agent_ticks"
            ],
            "waiting_assigned_ratio": fifo_audit_payload[
                "waiting_assigned_ratio"
            ],
            "waiting_promotions": fifo_audit_payload["waiting_promotions"],
            "waiting_duration_p95_ticks": fifo_audit_payload[
                "waiting_duration_p95_ticks"
            ],
            "waiting_duration_max_ticks": fifo_audit_payload[
                "waiting_duration_max_ticks"
            ],
            "unresolved_waiter_count_final": fifo_audit_payload[
                "unresolved_waiter_count_final"
            ],
        })

    reference_root = args.v2_root if args.mode == "fifo_v2" else args.v1_root
    reference_path = _reference_path(
        reference_root, args.mode, args.load, args.seed
    )
    reference_audit = _reference_audit(metrics, reference_path, args.mode)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "mode": args.mode,
            "station_admission": MODE_TO_ADMISSION[args.mode],
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "trace_stride": int(args.trace_stride),
            "formal_development": False,
            "diagnostic_only": True,
            "policy_logic_changed": False,
            "simulator_logic_changed": False,
            "runtime_wrappers_passive": True,
            "protocol_sha256": str(protocol["protocol_sha256"]),
            "frozen_bundle": bundle_path.as_posix(),
            "frozen_bundle_sha256": sha256_file(bundle_path),
            "manifest": manifest_path.as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            **runtime_hashes,
        },
        "reference_audit": reference_audit,
        "metrics": metrics,
        "fifo_invariant_audit": fifo_audit_payload,
        "token_flow": diagnostic.summary(),
        "station_admission_metrics": engine.world.station_state.admission_metrics(),
        "trace": {
            "path": trace_path.as_posix(),
            "sha256": sha256_file(trace_path),
        },
    }
    _atomic_json(summary_path, payload)
    if not reference_audit["passed"] and not args.allow_reference_mismatch:
        failed = [
            key for key, row in reference_audit.get("checks", {}).items()
            if not row.get("passed")
        ]
        raise RuntimeError(
            f"diagnostic replay changed frozen outcome for {args.mode} "
            f"{args.load} seed={args.seed}: {failed}"
        )
    print(json.dumps({
        "summary": summary_path.as_posix(),
        "trace": trace_path.as_posix(),
        "reference_match": reference_audit["passed"],
        "completed_orders": metrics.get("completed_orders"),
        "requeue_reasons": diagnostic.summary().get("requeue_reasons"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
