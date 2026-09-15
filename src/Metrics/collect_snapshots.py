"""
Snapshot Collector Module
=========================
Captures per-tick world-state snapshots during simulation runs.
Output is JSONL (one JSON line per tick, one file per episode),
intended as training data for Phase 3 world model.

File layout (per episode):
  line 0       — header: {"_header": true, "map": {...}, "config": {...}, ...}
  line 1..N    — per-tick snapshots
"""

import json
import os
import subprocess
from datetime import datetime
from typing import Dict, Optional, Any

from WorldState.world import WorldState
from WorldState.order_state import Order


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


class SnapshotCollector:
    """
    Collects per-tick state snapshots and writes them as JSONL.

    Usage
    -----
    collector = SnapshotCollector("DataGen/snapshots/rmfs")
    engine.on_tick_callbacks.append(collector.on_tick)

    collector.start_episode("warehouse-20x20_PIBT_greedy_r16_ep003")
    collector.write_header(world=engine.world, extra={...})  # optional
    engine.run()
    collector.end_episode()
    """

    def __init__(self, output_dir: str, agent_goals_override: Optional[dict] = None):
        self.output_dir = output_dir
        self.agent_goals_override = agent_goals_override
        self._file = None
        self._tick_count = 0
        self._episode_id: Optional[str] = None
        # path identity per agent — used to detect replans this tick
        self._prev_path_sig: Dict[int, tuple] = {}

    @staticmethod
    def resolve_episode_id(template: str, variables: Dict[str, str]) -> str:
        """Resolve an episode_id template with variable substitution.

        Supported variables (passed via *variables* dict):
          {map}, {planner}, {assigner}, {robots}, {agents}, {timestamp}
        Unknown placeholders are left as-is.
        """
        variables.setdefault("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        for key, val in variables.items():
            template = template.replace("{" + key + "}", str(val))
        return template

    def start_episode(self, episode_id: str):
        """Open a new JSONL file for an episode."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{episode_id}.jsonl")
        self._file = open(path, "w", encoding="utf-8")
        self._tick_count = 0
        self._episode_id = episode_id
        self._prev_path_sig = {}

    def write_header(self, world: WorldState, extra: Optional[Dict[str, Any]] = None):
        """Write one header line with episode-level metadata. Call once, before tick 0."""
        if self._file is None:
            return
        m = world.map_state
        # Derive obstacles from grid (MapState doesn't keep the raw list).
        from WorldState.map_state import CellType
        obstacles = [
            [r, c]
            for r in range(m.rows)
            for c in range(m.cols)
            if m.grid[r][c] == CellType.OBSTACLE
        ]
        header = {
            "_header": True,
            "episode_id": self._episode_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "map": {
                "rows": m.rows,
                "cols": m.cols,
                "obstacles": obstacles,
                "stations": [
                    {"id": sid, "pos": list(pos)}
                    for sid, pos in m.station_positions.items()
                ],
                "pod_homes": [list(p) for p in m.pod_home_positions],
            },
            "agents": {
                "count": len(world.agents),
                "starts": [list(a.position) for a in world.agents],
            },
        }
        if extra:
            header.update(extra)
        self._file.write(json.dumps(header, ensure_ascii=False) + "\n")

    def end_episode(self):
        """Close the current episode file."""
        if self._file is not None:
            self._file.close()
            self._file = None
        self._episode_id = None
        self._prev_path_sig = {}

    def on_tick(self, engine):
        """Engine callback — serialize and write one snapshot line."""
        self.record_tick(engine.world, engine=engine)

    def record_tick(self, world: WorldState, engine=None):
        """Serialize and write one snapshot line. Can be called directly."""
        if self._file is None:
            return
        snapshot = self._serialize_snapshot(world, engine)
        self._file.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        self._tick_count += 1

    def _serialize_snapshot(self, world: WorldState, engine=None) -> dict:
        snap = {
            "tick": world.tick,
            "agent_positions": [
                (a.agent_id, list(a.position))
                for a in world.agents
            ],
            "agent_goals": [
                (a.agent_id, self._resolve_goal(a, world))
                for a in world.agents
            ],
            "agent_statuses": [
                (a.agent_id, a.status.name)
                for a in world.agents
            ],
            "planned_paths": [
                (a.agent_id, [list(p) for p in a.path[a.path_index:]])
                for a in world.agents
            ],
            "task_phases": [
                (a.agent_id, self._resolve_task_phase(a, world))
                for a in world.agents
            ],
            "replans": self._compute_replans(world),
            "pod_positions": [
                (p.pod_id, list(p.current_position), p.is_carried)
                for p in world.pod_state.pods.values()
            ],
            "pending_orders": [
                self._serialize_order(o)
                for o in world.order_state.get_pending_orders()
            ],
            "in_progress_orders": [
                self._serialize_order(o)
                for o in world.order_state.get_in_progress_orders()
            ],
        }
        if engine is not None:
            snap["conflicts"] = {
                "vertex": [
                    {"pos": list(pos), "agents": ids}
                    for pos, ids in engine.last_vertex_conflicts
                ],
                "swap": [
                    {"a_prev": list(ap), "b_prev": list(bp),
                     "agents": [aid_a, aid_b]}
                    for ap, bp, aid_a, aid_b in engine.last_swap_conflicts
                ],
            }
            if engine.metrics.history:
                last = engine.metrics.history[-1]
                snap["metrics"] = {
                    "throughput_cumulative": last.throughput_cumulative,
                    "throughput_delta": last.throughput_delta,
                    "queue_length": last.queue_length,
                    "idle_agents": last.idle_agents,
                    "congestion_count": last.congestion_count,
                    "deadlock_agents": last.deadlock_agents,
                    "planning_ms": last.planning_ms,
                    "assignment_ms": last.assignment_ms,
                }
        return snap

    def _resolve_goal(self, agent, world: WorldState):
        """Return agent goal — from override dict if set, else from active task."""
        if self.agent_goals_override is not None:
            goal = self.agent_goals_override.get(agent.agent_id)
            return list(goal) if goal is not None else None
        active = world.task_state.get_active_task_for_agent(agent.agent_id)
        if active is not None:
            return list(active.destination)
        return None

    @staticmethod
    def _resolve_task_phase(agent, world: WorldState) -> Optional[str]:
        task = world.task_state.get_active_task_for_agent(agent.agent_id)
        if task is None:
            return None
        return task.task_type.name  # PICK / DELIVER / RETURN

    def _compute_replans(self, world: WorldState) -> list:
        """Return list of agent_ids whose planned path changed since last tick."""
        replanned = []
        for a in world.agents:
            sig = (len(a.path), a.path_index,
                   tuple(a.path[a.path_index]) if a.path_index < len(a.path) else None,
                   tuple(a.path[-1]) if a.path else None)
            if self._prev_path_sig.get(a.agent_id) != sig:
                replanned.append(a.agent_id)
            self._prev_path_sig[a.agent_id] = sig
        return replanned

    @staticmethod
    def _serialize_order(order: Order) -> dict:
        return {
            "order_id": order.order_id,
            "sku_demands": order.sku_demands,
            "station_id": order.station_id,
            "status": order.status.name,
            "created_at": order.created_at,
            "pod_ids": order.pod_ids,
            "delivered_pod_ids": order.delivered_pod_ids,
        }
