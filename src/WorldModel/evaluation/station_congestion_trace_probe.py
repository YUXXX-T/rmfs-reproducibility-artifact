"""Compact same-tick station/system congestion instrumentation.

The probe is diagnostic-only.  It subclasses the existing frozen TD stream
probe so the same run also saves stride-aligned graph frames for a later
frozen-encoder ``phi(z)`` audit, while adding a compact JSONL stream of only
physical contemporaneous measurements.  It never mutates simulator state.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from WorldModel.evaluation.phase_c_station_congestion_correlation_protocol import (
    PRIMARY_REGION_HOPS,
    REGION_HOPS,
    TRACE_SCHEMA_VERSION,
    TRAILING_WINDOW,
)
from WorldModel.evaluation.td_stream_probe import TDStreamProbe
from WorldModel.graph_builder import extract_node_features
from WorldState.agent_state import AgentStatus
from WorldState.risk import compute_unified_risk


NODE_CHANNELS = (
    "occupancy",
    "density",
    "wait",
    "blocked",
    "reservation",
    "recent_flow",
    "pod_pressure",
    "station_queue",
    "bottleneck",
    "node_type",
)


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value)).upper()


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / max(len(values), 1))


def _cvar(values: Sequence[float], tail_fraction: float = 0.10) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    count = max(1, int(math.ceil(len(ordered) * float(tail_fraction))))
    return _mean(ordered[-count:])


def _nodes_within_hops(
    seeds: Iterable[int],
    adjacency: Mapping[int, Sequence[int]],
    hops: int,
) -> frozenset[int]:
    distance = {int(seed): 0 for seed in seeds}
    frontier = deque(distance)
    while frontier:
        node = frontier.popleft()
        if distance[node] >= int(hops):
            continue
        next_distance = distance[node] + 1
        for neighbour in adjacency.get(node, ()):
            neighbour = int(neighbour)
            if neighbour in distance:
                continue
            distance[neighbour] = next_distance
            frontier.append(neighbour)
    return frozenset(distance)


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


class StationCongestionTraceProbe(TDStreamProbe):
    """Record compact current-state physical metrics at every simulator tick."""

    def __init__(self, engine, *args, **kwargs):
        super().__init__(engine, *args, **kwargs)
        self.trace_path = Path(self.out_dir) / (
            f"station_congestion_trace_{self.run_id}.jsonl"
        )
        self._rows: list[dict[str, Any]] = []
        self._station_ids = sorted(
            int(value) for value in engine.world.map_state.station_positions
        )
        self._region_nodes: dict[int, dict[int, frozenset[int]]] = {}
        self._station_seed_nodes: dict[int, tuple[int, ...]] = {}
        self._completed_history = {
            station_id: deque(maxlen=TRAILING_WINDOW)
            for station_id in self._station_ids
        }
        self._conflict_history = {
            station_id: deque(maxlen=TRAILING_WINDOW)
            for station_id in self._station_ids
        }
        self._impairment_history = {
            station_id: deque(maxlen=TRAILING_WINDOW)
            for station_id in self._station_ids
        }
        self._global_completed_history: deque[int] = deque(
            maxlen=TRAILING_WINDOW
        )
        self._previous_completed_orders = self._completed_order_ids(
            engine.world
        )
        self._build_station_regions(engine.world)

    @staticmethod
    def _completed_order_ids(world) -> set[int]:
        return {
            int(order.order_id)
            for order in world.order_state.orders.values()
            if _enum_name(order.status) == "COMPLETED"
        }

    def _build_station_regions(self, world) -> None:
        for station_id in self._station_ids:
            positions = []
            service = world.map_state.station_positions.get(station_id)
            if service is not None:
                positions.append(tuple(service))
            queue = world.station_state.stations.get(station_id)
            if queue is not None:
                if queue.entry_position is not None:
                    positions.append(tuple(queue.entry_position))
                if queue.exit_position is not None:
                    positions.append(tuple(queue.exit_position))
            seeds = tuple(sorted({
                int(self._node_map[position])
                for position in positions
                if position in self._node_map
            }))
            if not seeds:
                raise ValueError(
                    f"station {station_id} has no node in the static graph"
                )
            self._station_seed_nodes[station_id] = seeds
            self._region_nodes[station_id] = {
                int(hops): _nodes_within_hops(seeds, self._adj, int(hops))
                for hops in REGION_HOPS
            }

    @staticmethod
    def _conflict_records(engine) -> tuple[list[tuple], list[tuple]]:
        vertex = list(getattr(engine, "last_vertex_conflicts", ()) or ())
        swap = list(getattr(engine, "last_swap_conflicts", ()) or ())
        return vertex, swap

    @staticmethod
    def _node_channel_summary(
        node_features: torch.Tensor,
        node_ids: Sequence[int],
    ) -> dict[str, float]:
        if not node_ids:
            return {
                f"node_{name}_{stat}": 0.0
                for name in NODE_CHANNELS[:9]
                for stat in ("mean", "max", "cvar90")
            }
        ids = torch.as_tensor(node_ids, dtype=torch.long)
        values = node_features.index_select(0, ids)
        result: dict[str, float] = {}
        for channel, name in enumerate(NODE_CHANNELS[:9]):
            data = [float(value) for value in values[:, channel].tolist()]
            result[f"node_{name}_mean"] = _mean(data)
            result[f"node_{name}_max"] = max(data) if data else 0.0
            result[f"node_{name}_cvar90"] = _cvar(data)
        bottleneck = values[:, 8]
        density = values[:, 1]
        denominator = float(bottleneck.sum().item())
        result["node_bottleneck_weighted_density"] = (
            float((bottleneck * density).sum().item()) / denominator
            if denominator > 0.0 else 0.0
        )
        return result

    def _position_node(self, position) -> int | None:
        if position is None:
            return None
        return self._node_map.get(tuple(position))

    def _station_region_metrics(
        self,
        station_id: int,
        hops: int,
        *,
        world,
        node_features: torch.Tensor,
        agent_nodes: Mapping[int, int | None],
        vertex_conflicts: Sequence[tuple],
        swap_conflicts: Sequence[tuple],
    ) -> dict[str, float | int]:
        region = self._region_nodes[station_id][hops]
        region_agents = [
            agent for agent in world.agents
            if agent_nodes.get(int(agent.agent_id)) in region
        ]
        active = [
            agent for agent in region_agents
            if agent.status != AgentStatus.IDLE
        ]
        traffic_blocked = [
            agent for agent in active
            if bool(agent.traffic_blocked_this_tick)
        ]
        stuck = [
            agent for agent in active
            if bool(agent.stuck_this_tick) and not bool(agent.is_waiting)
        ]
        plan_failed = [
            agent for agent in active if int(agent.plan_failed_streak) >= 2
        ]
        deadlocked = [
            agent for agent in active
            if not bool(agent.is_waiting)
            and int(agent.stationary_ticks) >= 10
        ]
        impaired_ids = {
            int(agent.agent_id)
            for agent in [*traffic_blocked, *stuck, *plan_failed, *deadlocked]
        }
        moved = [agent for agent in active if bool(agent.moved_this_tick)]
        waiting = [agent for agent in active if bool(agent.is_waiting)]

        vertex_events = 0
        conflict_agent_ids: set[int] = set()
        for position, agent_ids in vertex_conflicts:
            if self._position_node(position) not in region:
                continue
            vertex_events += 1
            conflict_agent_ids.update(int(value) for value in agent_ids)
        swap_events = 0
        for left, right, agent_a, agent_b in swap_conflicts:
            if (
                self._position_node(left) not in region
                and self._position_node(right) not in region
            ):
                continue
            swap_events += 1
            conflict_agent_ids.update((int(agent_a), int(agent_b)))

        active_count = len(active)
        stationary_values = [int(agent.stationary_ticks) for agent in active]
        metrics: dict[str, float | int] = {
            "region_node_count": len(region),
            "robot_count": len(region_agents),
            "active_robot_count": active_count,
            "waiting_robot_count": len(waiting),
            "traffic_blocked_count": len(traffic_blocked),
            "stuck_count": len(stuck),
            "plan_failed_count": len(plan_failed),
            "deadlocked_count": len(deadlocked),
            "impaired_robot_count": len(impaired_ids),
            "moved_active_count": len(moved),
            "traffic_blocked_active_ratio": _safe_ratio(
                len(traffic_blocked), active_count
            ),
            "stuck_active_ratio": _safe_ratio(len(stuck), active_count),
            "plan_failed_active_ratio": _safe_ratio(
                len(plan_failed), active_count
            ),
            "deadlocked_active_ratio": _safe_ratio(
                len(deadlocked), active_count
            ),
            "mobility_impairment_ratio": _safe_ratio(
                len(impaired_ids), active_count
            ),
            "moved_active_ratio": _safe_ratio(len(moved), active_count),
            "stationary_ticks_mean": _mean(stationary_values),
            "stationary_ticks_max": (
                max(stationary_values) if stationary_values else 0
            ),
            "vertex_conflict_events": vertex_events,
            "swap_conflict_events": swap_events,
            "conflict_events": vertex_events + swap_events,
            "conflict_participant_count": len(conflict_agent_ids),
            "conflict_participant_active_ratio": _safe_ratio(
                len(conflict_agent_ids), active_count
            ),
        }
        metrics.update(self._node_channel_summary(
            node_features, sorted(region)
        ))
        return metrics

    def _station_rows(
        self,
        engine,
        node_features: torch.Tensor,
        newly_completed_by_station: Mapping[int, int],
        vertex_conflicts: Sequence[tuple],
        swap_conflicts: Sequence[tuple],
    ) -> list[dict[str, Any]]:
        world = engine.world
        num_agents = max(len(world.agents), 1)
        station_work_capacity = max(
            float(num_agents) / max(len(self._station_ids), 1), 1.0
        )
        pending_counts = {station_id: 0 for station_id in self._station_ids}
        in_progress_counts = {
            station_id: 0 for station_id in self._station_ids
        }
        open_counts = {station_id: 0 for station_id in self._station_ids}
        order_by_id = {}
        for order in world.order_state.orders.values():
            station_id = int(order.station_id)
            order_by_id[int(order.order_id)] = order
            status = _enum_name(order.status)
            if status == "PENDING":
                pending_counts[station_id] = pending_counts.get(station_id, 0) + 1
            if status == "IN_PROGRESS":
                in_progress_counts[station_id] = (
                    in_progress_counts.get(station_id, 0) + 1
                )
            if status not in {"COMPLETED", "CANCELLED"}:
                open_counts[station_id] = open_counts.get(station_id, 0) + 1

        active_task_counts = {
            station_id: 0 for station_id in self._station_ids
        }
        for task in world.task_state.tasks.values():
            if _enum_name(task.status) not in {"ASSIGNED", "IN_PROGRESS"}:
                continue
            order = order_by_id.get(int(getattr(task, "order_id", -1)))
            if order is None:
                continue
            station_id = int(order.station_id)
            active_task_counts[station_id] = (
                active_task_counts.get(station_id, 0) + 1
            )

        agent_nodes = {
            int(agent.agent_id): self._position_node(agent.position)
            for agent in world.agents
        }
        rows = []
        for station_id in self._station_ids:
            queue = world.station_state.stations.get(station_id)
            if queue is None:
                raise ValueError(f"missing queue state for station {station_id}")
            capacity = max(int(queue.capacity), 1)
            completed_delta = int(newly_completed_by_station.get(station_id, 0))
            regions = {
                f"h{hops}": self._station_region_metrics(
                    station_id,
                    int(hops),
                    world=world,
                    node_features=node_features,
                    agent_nodes=agent_nodes,
                    vertex_conflicts=vertex_conflicts,
                    swap_conflicts=swap_conflicts,
                )
                for hops in REGION_HOPS
            }
            primary = regions[f"h{PRIMARY_REGION_HOPS}"]
            self._completed_history[station_id].append(completed_delta)
            self._conflict_history[station_id].append(
                int(primary["conflict_events"])
            )
            self._impairment_history[station_id].append(
                float(primary["mobility_impairment_ratio"])
            )
            assigned_count = len(queue._assigned_agents)
            row = {
                "station_id": int(station_id),
                "station_seed_nodes": list(
                    self._station_seed_nodes[station_id]
                ),
                "queue_occupancy": int(queue.occupancy()),
                "queue_capacity": int(queue.capacity),
                "queue_occupancy_ratio": _safe_ratio(
                    queue.occupancy(), capacity
                ),
                "assigned_agent_count": int(assigned_count),
                "assigned_agent_capacity_ratio": _safe_ratio(
                    assigned_count, capacity
                ),
                "assigned_agent_work_ratio": _safe_ratio(
                    assigned_count, station_work_capacity
                ),
                "pending_order_count": int(pending_counts.get(station_id, 0)),
                "in_progress_order_count": int(
                    in_progress_counts.get(station_id, 0)
                ),
                "open_order_count": int(open_counts.get(station_id, 0)),
                "active_task_count": int(
                    active_task_counts.get(station_id, 0)
                ),
                "pending_pressure": _safe_ratio(
                    pending_counts.get(station_id, 0), station_work_capacity
                ),
                "in_progress_pressure": _safe_ratio(
                    in_progress_counts.get(station_id, 0),
                    station_work_capacity,
                ),
                "open_order_pressure": _safe_ratio(
                    open_counts.get(station_id, 0), station_work_capacity
                ),
                "active_task_pressure": _safe_ratio(
                    active_task_counts.get(station_id, 0),
                    station_work_capacity,
                ),
                "completed_orders_delta": completed_delta,
                "recent_completed_orders_per_tick": _mean(
                    list(self._completed_history[station_id])
                ),
                "recent_conflict_events_per_tick": _mean(
                    list(self._conflict_history[station_id])
                ),
                "recent_mobility_impairment": _mean(
                    list(self._impairment_history[station_id])
                ),
                "regions": regions,
            }
            rows.append(row)
        return rows

    def _system_row(
        self,
        engine,
        node_features: torch.Tensor,
        station_rows: Sequence[Mapping[str, Any]],
        newly_completed_count: int,
        vertex_conflicts: Sequence[tuple],
        swap_conflicts: Sequence[tuple],
    ) -> dict[str, Any]:
        world = engine.world
        active = [agent for agent in world.agents if agent.status != AgentStatus.IDLE]
        traffic_blocked = [
            agent for agent in active if bool(agent.traffic_blocked_this_tick)
        ]
        stuck = [
            agent for agent in active
            if bool(agent.stuck_this_tick) and not bool(agent.is_waiting)
        ]
        plan_failed = [
            agent for agent in active if int(agent.plan_failed_streak) >= 2
        ]
        deadlocked = [
            agent for agent in active
            if not bool(agent.is_waiting)
            and int(agent.stationary_ticks) >= 10
        ]
        impaired_ids = {
            int(agent.agent_id)
            for agent in [*traffic_blocked, *stuck, *plan_failed, *deadlocked]
        }
        conflict_agent_ids: set[int] = set()
        for _, agent_ids in vertex_conflicts:
            conflict_agent_ids.update(int(value) for value in agent_ids)
        for _, _, agent_a, agent_b in swap_conflicts:
            conflict_agent_ids.update((int(agent_a), int(agent_b)))

        risk = compute_unified_risk(world)
        active_count = len(active)
        self._global_completed_history.append(int(newly_completed_count))
        all_nodes = list(range(node_features.size(0)))
        result: dict[str, Any] = {
            "num_agents": len(world.agents),
            "active_robot_count": active_count,
            "active_robot_ratio": _safe_ratio(active_count, len(world.agents)),
            "idle_robot_count": len(world.get_idle_agents()),
            "traffic_blocked_count": len(traffic_blocked),
            "stuck_count": len(stuck),
            "plan_failed_count": len(plan_failed),
            "deadlocked_count": len(deadlocked),
            "impaired_robot_count": len(impaired_ids),
            "traffic_blocked_active_ratio": _safe_ratio(
                len(traffic_blocked), active_count
            ),
            "stuck_active_ratio": _safe_ratio(len(stuck), active_count),
            "plan_failed_active_ratio": _safe_ratio(
                len(plan_failed), active_count
            ),
            "deadlocked_active_ratio": _safe_ratio(
                len(deadlocked), active_count
            ),
            "mobility_impairment_ratio": _safe_ratio(
                len(impaired_ids), active_count
            ),
            "moved_active_ratio": _safe_ratio(
                sum(bool(agent.moved_this_tick) for agent in active),
                active_count,
            ),
            "vertex_conflict_events": len(vertex_conflicts),
            "swap_conflict_events": len(swap_conflicts),
            "conflict_events": len(vertex_conflicts) + len(swap_conflicts),
            "conflict_participant_count": len(conflict_agent_ids),
            "conflict_participant_active_ratio": _safe_ratio(
                len(conflict_agent_ids), active_count
            ),
            "pending_order_count": len(world.order_state.get_pending_orders()),
            "in_progress_order_count": len(
                world.order_state.get_in_progress_orders()
            ),
            "open_order_count": sum(
                _enum_name(order.status) not in {"COMPLETED", "CANCELLED"}
                for order in world.order_state.orders.values()
            ),
            "active_task_count": sum(
                _enum_name(task.status) in {"ASSIGNED", "IN_PROGRESS"}
                for task in world.task_state.tasks.values()
            ),
            "completed_orders_delta": int(newly_completed_count),
            "recent_completed_orders_per_tick": _mean(
                list(self._global_completed_history)
            ),
            "risk_unified": float(risk["unified_risk"]),
            "risk_stall_ratio": float(risk["stall_ratio"]),
            "risk_deadlock_ratio": float(risk["deadlock_ratio"]),
            "risk_handoff_ratio": float(risk["handoff_ratio"]),
        }
        result.update(self._node_channel_summary(node_features, all_nodes))

        bottleneck_scores = torch.as_tensor(
            self._bottleneck_score, dtype=torch.float32
        )
        bottleneck_count = max(1, int(math.ceil(len(all_nodes) * 0.20)))
        bottleneck_nodes = torch.topk(
            bottleneck_scores, k=bottleneck_count, largest=True
        ).indices.tolist()
        for key, value in self._node_channel_summary(
            node_features, bottleneck_nodes
        ).items():
            result[f"bottleneck_{key}"] = value

        def station_values(key: str) -> list[float]:
            return [float(row[key]) for row in station_rows]

        for key in (
            "queue_occupancy_ratio",
            "assigned_agent_work_ratio",
            "pending_pressure",
            "in_progress_pressure",
            "open_order_pressure",
            "active_task_pressure",
            "recent_completed_orders_per_tick",
            "recent_conflict_events_per_tick",
            "recent_mobility_impairment",
        ):
            values = station_values(key)
            result[f"station_{key}_mean"] = _mean(values)
            result[f"station_{key}_max"] = max(values) if values else 0.0
            result[f"station_{key}_cvar90"] = _cvar(values)
            mean = result[f"station_{key}_mean"]
            result[f"station_{key}_std"] = math.sqrt(_mean([
                (value - mean) ** 2 for value in values
            ]))
        primary_key = f"h{PRIMARY_REGION_HOPS}"
        impairment = [
            float(row["regions"][primary_key]["mobility_impairment_ratio"])
            for row in station_rows
        ]
        result["station_local_impairment_mean"] = _mean(impairment)
        result["station_local_impairment_max"] = (
            max(impairment) if impairment else 0.0
        )
        result["station_local_impairment_cvar90"] = _cvar(impairment)
        return result

    def on_tick(self, engine) -> None:
        super().on_tick(engine)
        world = engine.world
        tick = int(world.tick)
        node_features = extract_node_features(
            world,
            self._node_map,
            self._local_capacity,
            self._bottleneck_score,
            self._node_type_arr,
            self._adj,
            self._flow_counter,
            reservation_window=self.reservation_window,
        )
        completed_now = self._completed_order_ids(world)
        newly_completed = completed_now - self._previous_completed_orders
        self._previous_completed_orders = completed_now
        completed_by_station = {station_id: 0 for station_id in self._station_ids}
        for order_id in newly_completed:
            order = world.order_state.orders.get(int(order_id))
            if order is not None:
                station_id = int(order.station_id)
                completed_by_station[station_id] = (
                    completed_by_station.get(station_id, 0) + 1
                )
        vertex, swap = self._conflict_records(engine)
        station_rows = self._station_rows(
            engine,
            node_features,
            completed_by_station,
            vertex,
            swap,
        )
        system = self._system_row(
            engine,
            node_features,
            station_rows,
            len(newly_completed),
            vertex,
            swap,
        )
        self._rows.append({
            "schema_version": TRACE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "tick": tick,
            "system": system,
            "stations": station_rows,
        })

    def save(self) -> str:
        td_path = super().save()
        encoded = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in self._rows
        )
        _atomic_write_text(self.trace_path, encoded)
        return td_path

