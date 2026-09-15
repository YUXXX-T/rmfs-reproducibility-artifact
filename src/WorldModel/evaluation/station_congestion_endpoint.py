"""Shared helpers for the frozen H=10 station-potential endpoint audit.

The deployed assigner does not import this module.  It reconstructs the
encoder observation at the real isolated-rollout endpoint and evaluates the
already frozen region-aware station congestion head on either a real or a
World-Model-predicted latent.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from WorldModel.core.station_congestion_head import (
    CHANNEL_NAMES,
    StationCongestionHead,
    build_station_targets,
    verify_scale_contract,
)
from WorldModel.data.build_station_congestion_head_dataset import (
    REPRESENTATION_STATION_REGION_MEAN_MAX,
    build_station_region_node_ids,
    build_station_representations,
)
from WorldModel.graph.graph_builder import (
    extract_demand_context,
    extract_edge_features,
    extract_node_features,
)
from WorldState.agent_state import AgentStatus


ENDPOINT_OBSERVATION_SCHEMA_VERSION = (
    "station_congestion_endpoint_observation_v1"
)
ENDPOINT_RECORD_SCHEMA_VERSION = "station_congestion_endpoint_record_v1"
EXPECTED_HEAD_TRAINING_SCHEMA = "station_congestion_head_training_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value)).upper()


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _cvar90(values: torch.Tensor) -> float:
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if values.numel() == 0:
        return 0.0
    count = max(1, int(math.ceil(values.numel() * 0.10)))
    return float(torch.topk(values, k=count, largest=True).values.mean().item())


@dataclass(frozen=True)
class StationLayout:
    """Map-local station indexing used by both real and predicted latents."""

    station_ids: tuple[int, ...]
    station_node_ids: tuple[int, ...]
    station_seed_node_ids: tuple[tuple[int, ...], ...]
    station_region_node_ids: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_ids": list(self.station_ids),
            "station_node_ids": list(self.station_node_ids),
            "station_seed_node_ids": [
                list(values) for values in self.station_seed_node_ids
            ],
            "station_region_node_ids": [
                list(values) for values in self.station_region_node_ids
            ],
        }


def build_station_layout(
    world,
    node_map: Mapping[tuple[int, int], int],
    edge_index: torch.Tensor,
) -> StationLayout:
    station_ids = tuple(sorted(
        int(value) for value in world.map_state.station_positions
    ))
    if not station_ids:
        raise ValueError("world has no stations")
    station_node_ids = []
    seed_rows = []
    seed_ids = []
    for station_id in station_ids:
        service_position = tuple(
            world.map_state.station_positions[station_id]
        )
        if service_position not in node_map:
            raise ValueError(
                f"station {station_id} service node is absent from graph"
            )
        station_node_ids.append(int(node_map[service_position]))
        positions = [service_position]
        queue = world.station_state.get_queue(station_id)
        if queue is not None:
            if queue.entry_position is not None:
                positions.append(tuple(queue.entry_position))
            if queue.exit_position is not None:
                positions.append(tuple(queue.exit_position))
        seeds = tuple(sorted({
            int(node_map[position])
            for position in positions
            if position in node_map
        }))
        if not seeds:
            raise ValueError(f"station {station_id} has no graph seed node")
        seed_ids.append(seeds)
        seed_rows.append({
            "station_id": station_id,
            "station_seed_nodes": list(seeds),
        })
    regions = build_station_region_node_ids(
        seed_rows,
        torch.as_tensor(edge_index, dtype=torch.long),
        num_nodes=len(node_map),
        hops=3,
    )
    return StationLayout(
        station_ids=station_ids,
        station_node_ids=tuple(station_node_ids),
        station_seed_node_ids=tuple(seed_ids),
        station_region_node_ids=tuple(
            tuple(int(value) for value in region) for region in regions
        ),
    )


def build_endpoint_station_rows(
    world,
    node_features: torch.Tensor,
    layout: StationLayout,
    node_map: Mapping[tuple[int, int], int],
) -> list[dict[str, Any]]:
    """Build exactly the five frozen physical target components at endpoint."""

    node_features = torch.as_tensor(node_features, dtype=torch.float32)
    station_count = max(len(layout.station_ids), 1)
    station_work_capacity = max(
        float(len(world.agents)) / float(station_count), 1.0
    )
    in_progress = {station_id: 0 for station_id in layout.station_ids}
    for order in world.order_state.orders.values():
        if _enum_name(order.status) != "IN_PROGRESS":
            continue
        station_id = int(order.station_id)
        in_progress[station_id] = in_progress.get(station_id, 0) + 1
    agent_nodes = {
        int(agent.agent_id): node_map.get(tuple(agent.position))
        for agent in world.agents
    }
    rows = []
    for offset, station_id in enumerate(layout.station_ids):
        region = set(layout.station_region_node_ids[offset])
        active_agents = [
            agent for agent in world.agents
            if agent.status != AgentStatus.IDLE
            and agent_nodes.get(int(agent.agent_id)) in region
        ]
        stationary_ticks_max = max(
            (int(agent.stationary_ticks) for agent in active_agents),
            default=0,
        )
        region_ids = torch.as_tensor(
            layout.station_region_node_ids[offset], dtype=torch.long
        )
        region_features = node_features.index_select(0, region_ids)
        density = region_features[:, 1]
        bottleneck = region_features[:, 8]
        bottleneck_denominator = float(bottleneck.sum().item())
        weighted_density = (
            float((bottleneck * density).sum().item())
            / bottleneck_denominator
            if bottleneck_denominator > 0.0 else 0.0
        )
        queue = world.station_state.get_queue(station_id)
        if queue is None:
            raise ValueError(f"missing queue for station {station_id}")
        assigned_count = len(getattr(queue, "_assigned_agents", ()))
        rows.append({
            "station_id": int(station_id),
            "station_seed_nodes": list(
                layout.station_seed_node_ids[offset]
            ),
            "assigned_agent_capacity_ratio": _safe_ratio(
                assigned_count, int(queue.capacity)
            ),
            "in_progress_pressure": _safe_ratio(
                in_progress.get(station_id, 0), station_work_capacity
            ),
            "regions": {
                "h3": {
                    "stationary_ticks_max": float(stationary_ticks_max),
                    "node_density_mean": float(density.mean().item()),
                    "node_density_cvar90": _cvar90(density),
                    "node_bottleneck_weighted_density": weighted_density,
                }
            },
        })
    return rows


class EndpointObservationObserver:
    """Rebuild the encoder's real endpoint observation during H-step rollout."""

    def __init__(
        self,
        *,
        node_history: torch.Tensor,
        edge_index: torch.Tensor,
        node_map: Mapping[tuple[int, int], int],
        inv_node_map: Mapping[int, tuple[int, int]],
        local_capacity: Sequence[float],
        bottleneck_score: Sequence[float],
        node_type_arr: Sequence[float],
        adj: Mapping[int, Sequence[int]],
        flow_counter: Mapping[int, float],
        edge_flow_counter: Mapping[tuple[int, int], float],
        layout: StationLayout,
        scale_contract: Mapping[str, Any],
        horizon: int,
        reservation_window: int = 1,
    ):
        history = torch.as_tensor(node_history, dtype=torch.float32).clone()
        if history.ndim != 3 or history.size(1) != len(node_map):
            raise ValueError("node_history must have shape (L, N, F)")
        if history.size(0) <= 0:
            raise ValueError("node_history must contain at least one frame")
        verify_scale_contract(scale_contract)
        self._frames = [frame.clone() for frame in history]
        self._history_len = int(history.size(0))
        self._edge_index = torch.as_tensor(
            edge_index, dtype=torch.long
        ).clone()
        self._node_map = dict(node_map)
        self._inv_node_map = dict(inv_node_map)
        self._local_capacity = list(local_capacity)
        self._bottleneck_score = list(bottleneck_score)
        self._node_type_arr = list(node_type_arr)
        self._adj = {int(key): list(value) for key, value in adj.items()}
        self._flow_counter = {
            int(key): float(value) for key, value in flow_counter.items()
        }
        self._edge_flow_counter = {
            (int(key[0]), int(key[1])): float(value)
            for key, value in edge_flow_counter.items()
        }
        self._layout = layout
        self._scale_contract = dict(scale_contract)
        self._horizon = int(horizon)
        self._reservation_window = int(reservation_window)
        self._steps = 0

    def on_post_step(self, world) -> None:
        for node_id in tuple(self._flow_counter):
            self._flow_counter[node_id] *= 0.85
        for agent in world.agents:
            node_id = self._node_map.get(tuple(agent.position))
            if node_id is not None:
                self._flow_counter[node_id] = (
                    self._flow_counter.get(node_id, 0.0) + 1.0
                )
        for edge in tuple(self._edge_flow_counter):
            self._edge_flow_counter[edge] *= 0.85
        for agent in world.agents:
            if not bool(agent.moved_this_tick):
                continue
            previous = tuple(agent.previous_position)
            current = tuple(agent.position)
            if previous == current:
                continue
            left = self._node_map.get(previous)
            right = self._node_map.get(current)
            if left is not None and right is not None:
                edge = (left, right)
                self._edge_flow_counter[edge] = (
                    self._edge_flow_counter.get(edge, 0.0) + 1.0
                )
        frame = extract_node_features(
            world,
            self._node_map,
            self._local_capacity,
            self._bottleneck_score,
            self._node_type_arr,
            self._adj,
            self._flow_counter,
            reservation_window=self._reservation_window,
        )
        self._frames.append(frame.clone())
        if len(self._frames) > self._history_len:
            self._frames.pop(0)
        self._steps += 1

    def finalize(self, world) -> dict[str, Any]:
        if self._steps != self._horizon:
            raise ValueError(
                f"endpoint observer saw {self._steps} steps, "
                f"expected {self._horizon}"
            )
        node_history = torch.stack(self._frames, dim=0)
        edge_features = extract_edge_features(
            self._edge_index,
            self._node_map,
            self._inv_node_map,
            self._local_capacity,
            world,
            self._adj,
            self._edge_flow_counter,
            reservation_window=self._reservation_window,
        )
        demand_context = extract_demand_context(world)
        station_rows = build_endpoint_station_rows(
            world,
            node_history[-1],
            self._layout,
            self._node_map,
        )
        physical_targets = []
        physical_components = []
        for row in station_rows:
            target = build_station_targets(row, self._scale_contract)
            physical_targets.append([
                float(target["channels"][name]) for name in CHANNEL_NAMES
            ])
            physical_components.append(target)
        return {
            "schema_version": ENDPOINT_OBSERVATION_SCHEMA_VERSION,
            "horizon": self._horizon,
            "endpoint_tick": int(world.tick),
            "node_history": node_history,
            "edge_features": edge_features,
            "demand_context": demand_context,
            "station_ids": torch.tensor(
                self._layout.station_ids, dtype=torch.long
            ),
            "station_node_ids": torch.tensor(
                self._layout.station_node_ids, dtype=torch.long
            ),
            "physical_targets": torch.tensor(
                physical_targets, dtype=torch.float32
            ),
            "physical_components": physical_components,
        }


def load_frozen_station_head(
    checkpoint_path: str | Path,
    *,
    model_checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[StationCongestionHead, dict[str, Any]]:
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if payload.get("schema_version") != EXPECTED_HEAD_TRAINING_SCHEMA:
        raise ValueError("wrong station congestion head checkpoint schema")
    if tuple(payload.get("channel_names", ())) != CHANNEL_NAMES:
        raise ValueError("station congestion head channel schema changed")
    verify_scale_contract(payload["scale_contract"])
    if payload.get("scale_contract_sha256") != payload["scale_contract"].get(
        "contract_sha256"
    ):
        raise ValueError("station congestion head scale hash mismatch")
    if payload.get("source_encoder_checkpoint_sha256") != sha256_file(
        model_checkpoint_path
    ):
        raise ValueError(
            "station congestion head was not trained on this encoder checkpoint"
        )
    representation = payload.get("representation") or {}
    if representation.get("name") != REPRESENTATION_STATION_REGION_MEAN_MAX:
        raise ValueError("endpoint audit requires the frozen region-aware head")
    head = StationCongestionHead(int(payload["latent_dim"]))
    head.load_state_dict(payload["state_dict"], strict=True)
    head.to(device)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head, payload


def evaluate_station_psi(
    z: torch.Tensor,
    *,
    head: StationCongestionHead,
    head_payload: Mapping[str, Any],
    layout: StationLayout,
) -> torch.Tensor:
    representation = str(head_payload["representation"]["name"])
    station_features = build_station_representations(
        z,
        layout.station_node_ids,
        representation=representation,
        station_region_node_ids=layout.station_region_node_ids,
    )
    return head.forward_station_latents(station_features)


__all__ = [
    "ENDPOINT_OBSERVATION_SCHEMA_VERSION",
    "ENDPOINT_RECORD_SCHEMA_VERSION",
    "EndpointObservationObserver",
    "StationLayout",
    "build_endpoint_station_rows",
    "build_station_layout",
    "evaluate_station_psi",
    "load_frozen_station_head",
    "sha256_file",
]
