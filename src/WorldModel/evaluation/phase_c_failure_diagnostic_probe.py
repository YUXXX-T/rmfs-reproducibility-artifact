"""Diagnostic-only probes for Phase-C closed-loop failure analysis.

This module deliberately extends the frozen probes without modifying them.
Existing Phase-C collection and certification hashes therefore remain valid.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from WorldModel.evaluation.decision_snapshot_probe import (
    DecisionSnapshotProbe as _FrozenDecisionSnapshotProbe,
)
from WorldModel.evaluation.td_stream_probe import TDStreamProbe as _FrozenTDStreamProbe
from WorldState.agent_state import AgentStatus
from WorldState.risk import compute_unified_risk


DIAGNOSTIC_SNAPSHOT_SCHEMA_VERSION = "phasec_failure_decision_snapshot_v1"
DIAGNOSTIC_TRAJECTORY_SCHEMA_VERSION = "phasec_failure_trajectory_v1"


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value))


def _position(value: Any) -> list[int] | None:
    if value is None:
        return None
    return [int(value[0]), int(value[1])]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


class FailureDecisionSnapshotProbe(_FrozenDecisionSnapshotProbe):
    """Capture every online-scored context for selected diagnostic runs."""

    def __init__(self, *args, **kwargs):
        # These settings reproduce the deployed candidate set and retain every
        # rankable decision. Runs using this probe are few and explicitly
        # selected by the diagnostic protocol.
        kwargs.update({
            "sample_interval": 1,
            "candidate_scope": "all_idle_online",
            "attach_assigner_trace": True,
            "require_trace_alignment": True,
            "capture_policy": "interval_all",
            "candidate_robot_mode": "nearest",
            "include_no_assign_candidate": False,
            "max_contexts_per_tick": None,
        })
        super().__init__(*args, **kwargs)

    def on_pre_assignment(self, engine):
        previous = len(self._pending)
        super().on_pre_assignment(engine)
        for payload, row in self._pending[previous:]:
            payload["diagnostic_snapshot_schema_version"] = (
                DIAGNOSTIC_SNAPSHOT_SCHEMA_VERSION
            )
            payload["diagnostic_capture_scope"] = "all_rankable_online_contexts"
            row["diagnostic_snapshot_schema_version"] = (
                DIAGNOSTIC_SNAPSHOT_SCHEMA_VERSION
            )

    def save(self) -> str:
        path = Path(super().save())
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.update({
            "diagnostic_snapshot_schema_version": (
                DIAGNOSTIC_SNAPSHOT_SCHEMA_VERSION
            ),
            "diagnostic_capture_scope": "all_rankable_online_contexts",
            "sample_interval": 1,
            "candidate_scope": "all_idle_online",
        })
        _atomic_write_text(
            path,
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        )
        return str(path)


class FailureTrajectoryProbe(_FrozenTDStreamProbe):
    """Add a human-readable per-tick event stream to the frozen TD stream."""

    _STAT_KEYS = (
        "assign_calls",
        "model_assign_calls",
        "model_inference_calls",
        "decision_contexts_total",
        "energy_conv_contexts",
        "energy_conv_active_contexts",
        "energy_conv_modified_decisions",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory_path = Path(self.out_dir) / (
            f"trajectory_{self.run_id}.jsonl"
        )
        self._trajectory_rows: list[dict[str, Any]] = []
        self._decision_cursor = 0
        self._previous_completed = 0
        self._previous_stats = self._stats_snapshot(None)

    @classmethod
    def _stats_snapshot(cls, engine) -> dict[str, int]:
        stats = getattr(getattr(engine, "task_assigner", None), "stats", {}) or {}
        return {key: int(stats.get(key, 0)) for key in cls._STAT_KEYS}

    @staticmethod
    def _decision_summary(record: dict[str, Any]) -> dict[str, Any]:
        candidates = [
            candidate
            for candidate in record.get("candidates", ())
            if candidate.get("robot_id") is not None
        ]
        selected_robot = record.get("selected_robot")
        baseline_robot = record.get("baseline_robot")
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate.get("robot_id") == selected_robot
            ),
            {},
        )
        base_ranked = sorted(
            candidates,
            key=lambda candidate: (
                float(candidate.get("score", float("inf"))),
                int(candidate.get("robot_id", -1)),
            ),
        )
        base_margin = None
        if len(base_ranked) >= 2:
            base_margin = float(base_ranked[1]["score"]) - float(
                base_ranked[0]["score"]
            )
        conversion_available = any(
            candidate.get("score_conv") is not None for candidate in candidates
        )
        return {
            "context_idx": int(record.get("context_idx", -1)),
            "order_id": int(record.get("order_id", -1)),
            "pod_id": int(record.get("pod_id", -1)),
            "station_id": int(record.get("station_id", -1)),
            "action_status": record.get("action_status"),
            "candidate_count": int(record.get("scored_candidate_count", 0)),
            "selected_robot": selected_robot,
            "baseline_robot": baseline_robot,
            "energy_conversion_modified": bool(
                conversion_available
                and selected_robot is not None
                and baseline_robot is not None
                and int(selected_robot) != int(baseline_robot)
            ),
            "selected_wm_score": selected.get("score"),
            "selected_conversion_score": selected.get("score_conv"),
            "selected_risk_max": selected.get("risk_max"),
            "selected_route_len": selected.get("route_len"),
            "base_rank_margin": base_margin,
        }

    @staticmethod
    def _agent_summary(agent) -> dict[str, Any]:
        return {
            "agent_id": int(agent.agent_id),
            "position": _position(agent.position),
            "status": _enum_name(agent.status),
            "stationary_ticks": int(agent.stationary_ticks),
            "wait_ticks": int(agent.wait_ticks),
            "plan_failed_streak": int(agent.plan_failed_streak),
            "stuck_this_tick": bool(agent.stuck_this_tick),
            "traffic_blocked_this_tick": bool(
                agent.traffic_blocked_this_tick
            ),
            "carried_pod_id": (
                int(agent.carried_pod_id)
                if agent.carried_pod_id is not None
                else None
            ),
            "assigned_task_id": (
                int(agent.assigned_task_id)
                if agent.assigned_task_id is not None
                else None
            ),
        }

    @staticmethod
    def _station_summaries(world) -> list[dict[str, Any]]:
        rows = []
        for station_id, queue in sorted(world.station_state.stations.items()):
            slot_rows = []
            for slot in [queue.service, *queue.queue_slots, *queue.buffer_slots]:
                slot_rows.append({
                    "slot_type": _enum_name(slot.slot_type),
                    "slot_index": int(slot.index),
                    "position": _position(slot.position),
                    "agent_id": (
                        int(slot.agent_id) if slot.agent_id is not None else None
                    ),
                })
            rows.append({
                "station_id": int(station_id),
                "occupancy": int(queue.occupancy()),
                "capacity": int(queue.capacity),
                "assigned_agent_count": len(queue._assigned_agents),
                "assigned_agent_ids": sorted(
                    int(agent_id) for agent_id in queue._assigned_agents
                ),
                "entry_position": _position(queue.entry_position),
                "exit_position": _position(queue.exit_position),
                "slots": slot_rows,
            })
        return rows

    @staticmethod
    def _vertex_conflicts(values: Iterable[Any]) -> list[dict[str, Any]]:
        rows = []
        for position, agent_ids in values:
            rows.append({
                "position": _position(position),
                "agent_ids": [int(agent_id) for agent_id in agent_ids],
            })
        return rows

    @staticmethod
    def _swap_conflicts(values: Iterable[Any]) -> list[dict[str, Any]]:
        rows = []
        for left, right, agent_a, agent_b in values:
            rows.append({
                "left": _position(left),
                "right": _position(right),
                "agent_ids": [int(agent_a), int(agent_b)],
            })
        return rows

    def on_tick(self, engine):
        super().on_tick(engine)
        world = engine.world
        tick = int(world.tick)
        completed = int(world.order_state.total_completed)
        risk = compute_unified_risk(world)

        records = getattr(engine.task_assigner, "decision_trace_records", None) or []
        new_records = records[self._decision_cursor:]
        self._decision_cursor = len(records)
        decisions = [self._decision_summary(record) for record in new_records]

        current_stats = self._stats_snapshot(engine)
        stats_delta = {
            key: current_stats[key] - self._previous_stats[key]
            for key in self._STAT_KEYS
        }
        self._previous_stats = current_stats

        deadlocked_agents = [
            self._agent_summary(agent)
            for agent in world.agents
            if agent.status != AgentStatus.IDLE
            and not agent.is_waiting
            and int(agent.stationary_ticks) >= 10
        ]
        stalled_agents = [
            self._agent_summary(agent)
            for agent in world.agents
            if (
                bool(agent.stuck_this_tick) and not agent.is_waiting
            )
            or int(agent.plan_failed_streak) >= 2
        ]
        status_counts = Counter(_enum_name(agent.status) for agent in world.agents)

        vertex_conflicts = self._vertex_conflicts(
            getattr(engine, "last_vertex_conflicts", ())
        )
        swap_conflicts = self._swap_conflicts(
            getattr(engine, "last_swap_conflicts", ())
        )
        orders = world.order_state.orders.values()
        task_values = world.task_state.tasks.values()
        self._trajectory_rows.append({
            "schema_version": DIAGNOSTIC_TRAJECTORY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "tick": tick,
            "completed_orders_total": completed,
            "completed_orders_delta": completed - self._previous_completed,
            "pending_order_count": len(world.order_state.get_pending_orders()),
            "in_progress_order_count": len(
                world.order_state.get_in_progress_orders()
            ),
            "open_order_count": sum(
                _enum_name(order.status).upper() not in {"COMPLETED", "CANCELLED"}
                for order in orders
            ),
            "idle_robot_count": len(world.get_idle_agents()),
            "active_task_count": sum(
                _enum_name(task.status).upper() in {"ASSIGNED", "IN_PROGRESS"}
                for task in task_values
            ),
            "risk": {
                key: float(value) for key, value in risk.items()
            },
            "status_counts": dict(sorted(status_counts.items())),
            "deadlocked_agents": deadlocked_agents,
            "stalled_agents": stalled_agents,
            "vertex_conflicts": vertex_conflicts,
            "swap_conflicts": swap_conflicts,
            "station_queues": self._station_summaries(world),
            "assigner_stats_delta": stats_delta,
            "decisions": decisions,
        })
        self._previous_completed = completed

    def save(self) -> str:
        td_path = super().save()
        encoded = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in self._trajectory_rows
        )
        _atomic_write_text(self.trajectory_path, encoded)
        return td_path
