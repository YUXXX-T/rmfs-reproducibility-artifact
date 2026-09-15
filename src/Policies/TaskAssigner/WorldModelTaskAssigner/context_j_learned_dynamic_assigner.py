"""Online learned context-J selector for a fresh-seed closed-loop test.

This module is intentionally additive.  It subclasses the already isolated
dynamic-J probe and changes only the cross-context score:

* the frozen context-J checkpoint predicts ``J(c)``;
* the candidate context is selected again after every successful virtual S1
  robot selection;
* the actual robot choice is delegated to the unchanged S1 scorer;
* no NO_ASSIGN action, hard gate, e_demand/schema change, or FIFO mutation is
  introduced here.

The online feature construction mirrors the context-J training extractor:
station-region latent representation + global latent/demand representation +
the aggregate action-global field of the top five nearest candidate robots,
with the checkpoint's frozen train-only normalization contracts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from Policies.TaskAssigner.base_task_assigner import AssignmentContext
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_dynamic_probe_assigner import (
    DynamicPsiDispatchProbeAssigner,
)
from WorldModel.data.candidate_generator import build_candidate_assignment
from WorldModel.evaluation.station_congestion_endpoint import sha256_file
from WorldModel.training import train_context_dispatch_j_head as context_j_train


LEARNED_CONTEXT_J_SCHEMA_VERSION = "phase_c_context_j_learned_dynamic_v1"
EXPECTED_HEAD_SCHEMA = context_j_train.HEAD_SCHEMA_VERSION
HORIZON = 10
TOP_M_TRAINING_CONTRACT = 5


def _manhattan(left: Sequence[int], right: Sequence[int]) -> int:
    return abs(int(left[0]) - int(right[0])) + abs(
        int(left[1]) - int(right[1])
    )


class LearnedContextJDynamicAssigner(DynamicPsiDispatchProbeAssigner):
    """Dynamic learned-J context selector with the certified S1 robot arm."""

    learned_context_j_schema_version = LEARNED_CONTEXT_J_SCHEMA_VERSION

    def __init__(
        self,
        *,
        context_j_checkpoint: str,
        context_j_trace_enabled: bool = True,
        context_j_trace_max_records: int = 500,
        **kwargs,
    ) -> None:
        self.context_j_checkpoint = str(context_j_checkpoint)
        self.context_j_trace_enabled = bool(context_j_trace_enabled)
        self.context_j_trace_max_records = max(
            0, int(context_j_trace_max_records)
        )
        self._context_j_head = None
        self._context_j_payload: dict[str, Any] | None = None
        self._context_j_checkpoint_sha256: str | None = None
        self._context_j_feature_mean = None
        self._context_j_feature_std = None
        self._context_j_semantic_lower = None
        self._context_j_semantic_span = None
        self._context_j_semantic_constant = None
        self._context_j_state_bundle: dict[str, Any] | None = None
        self._context_j_feature_cache: dict[tuple[Any, ...], tuple[float, dict]] = {}
        super().__init__(**kwargs)
        self.stats.update({
            "context_j_head_loaded": 0,
            "context_j_eval_calls": 0,
            "context_j_eval_time_total_ms": 0.0,
            "context_j_feature_cache_hits": 0,
            "context_j_feature_cache_misses": 0,
            "context_j_no_candidate_rows": 0,
            "context_j_score_sum": 0.0,
            "context_j_semantic_service_sum": 0.0,
            "context_j_semantic_traffic_sum": 0.0,
            "context_j_semantic_marginal_work_sum": 0.0,
            "context_j_semantic_debt_sum": 0.0,
        })

    @staticmethod
    def _as_tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype).flatten()

    def _init(self, world) -> None:
        super()._init(world)
        path = Path(self.context_j_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("context-J checkpoint must contain a mapping")
        if payload.get("head_schema_version") != EXPECTED_HEAD_SCHEMA:
            raise ValueError("unexpected context-J head schema")
        config = payload.get("head_config")
        if not isinstance(config, Mapping):
            raise ValueError("context-J checkpoint lacks head_config")
        if list(config.get("semantic_channels", ())) != list(
            context_j_train.SEMANTIC_CHANNELS
        ):
            raise ValueError("context-J semantic channel contract changed")
        base = payload.get("base_world_model") or {}
        if str(base.get("sha256")) != sha256_file(self.checkpoint_path):
            raise ValueError("context-J head was trained on a different World Model")
        station = payload.get("station_phi") or {}
        if str(station.get("sha256")) != sha256_file(self.psi_head_checkpoint):
            raise ValueError("context-J head was trained on a different station phi")
        target = payload.get("target_contract") or {}
        if int(target.get("horizon", -1)) != HORIZON:
            raise ValueError("online learned-J test requires H=10 head labels")
        if str(target.get("rollout_continuation_mode")) != "behavior":
            raise ValueError(
                "online learned-J test requires behavior-continuation labels"
            )
        if int(config.get("feature_dim", -1)) <= 0:
            raise ValueError("invalid context-J feature dimension")

        device = next(self._model.parameters()).device
        head = context_j_train.ContextDispatchJHead(
            int(config["feature_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]),
            residual_limit=float(config["residual_limit"]),
        ).to(device)
        head.load_state_dict(payload["state_dict"], strict=True)
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)

        feature_contract = payload.get("feature_contract") or {}
        semantic_contract = payload.get("semantic_contract") or {}
        self._context_j_feature_mean = self._as_tensor(
            feature_contract["mean"]
        ).to(device)
        self._context_j_feature_std = self._as_tensor(
            feature_contract["std"]
        ).to(device)
        self._context_j_semantic_lower = self._as_tensor(
            semantic_contract["lower"]
        ).to(device)
        self._context_j_semantic_span = self._as_tensor(
            semantic_contract["span"]
        ).to(device)
        self._context_j_semantic_constant = torch.as_tensor(
            semantic_contract["constant"], dtype=torch.bool, device=device
        ).flatten()
        if self._context_j_semantic_lower.numel() != 4:
            raise ValueError("context-J semantic contract must have four channels")

        self._context_j_head = head
        self._context_j_payload = dict(payload)
        self._context_j_checkpoint_sha256 = sha256_file(path)
        self.stats["context_j_head_loaded"] = 1

    def _encode_context_state(self, world_state) -> dict[str, Any]:
        """Encode the current tick using the same extractor as training."""

        from WorldModel.data.build_station_congestion_head_dataset import (
            build_station_representations,
        )
        from WorldModel.graph_builder import (
            extract_demand_context,
            extract_edge_features,
        )

        node_history = self._feature_history.get_history()
        edge_features = extract_edge_features(
            self._edge_index,
            self._node_map,
            self._inv_node_map,
            self._local_capacity,
            world_state,
            adj=self._adj,
            edge_flow_counter=self._edge_flow_counter,
            reservation_window=self.reservation_window,
        )
        demand = extract_demand_context(world_state)
        with torch.no_grad():
            z, e_demand, _ = self._model.encode_state(
                node_history,
                self._edge_index,
                edge_features,
                demand,
            )
            representation_name = str(
                self._psi_head_payload["representation"]["name"]
            )
            station_representation = build_station_representations(
                z,
                self._psi_layout.station_node_ids,
                representation=representation_name,
                station_region_node_ids=self._psi_layout.station_region_node_ids,
            )
            phi = self._psi_head.forward_station_latents(station_representation)
            global_representation = torch.cat((
                z.mean(dim=0), z.amax(dim=0), e_demand.flatten()
            ))
        self.stats["context_j_state_encode_calls"] = int(
            self.stats.get("context_j_state_encode_calls", 0)
        ) + 1
        return {
            "demand": demand.flatten(),
            "station_representation": station_representation,
            "global_representation": global_representation,
            "phi": phi,
            "feature_dim": int(global_representation.numel()),
        }

    @staticmethod
    def _fixed_context(context: AssignmentContext) -> dict[str, Any]:
        return {
            "order_id": int(context.order_id),
            "pod_id": int(context.pod_id),
            "pod_location": tuple(context.pod_location),
            "station_id": int(context.station_id),
            "station_location": tuple(context.station_location),
            "return_location": tuple(context.return_location),
            "entry_position": (
                tuple(context.entry_position)
                if context.entry_position is not None else None
            ),
            "exit_position": (
                tuple(context.exit_position)
                if context.exit_position is not None else None
            ),
            "order_size": int(context.order_size),
        }

    def _online_context_score(
        self,
        world_state,
        context: AssignmentContext,
        virtual_idle_agents: Sequence,
    ) -> tuple[float, dict[str, Any]]:
        if self._context_j_head is None or self._context_j_state_bundle is None:
            raise RuntimeError("context-J head/state is not initialised")
        device = next(self._context_j_head.parameters()).device
        agent_ids = tuple(sorted(int(agent.agent_id) for agent in virtual_idle_agents))
        key = (int(context.order_id), int(context.pod_id), int(context.station_id), agent_ids)
        cached = self._context_j_feature_cache.get(key)
        if cached is not None:
            self.stats["context_j_feature_cache_hits"] += 1
            return cached
        self.stats["context_j_feature_cache_misses"] += 1
        started = __import__("time").perf_counter()
        candidates = sorted(
            virtual_idle_agents,
            key=lambda agent: (
                _manhattan(agent.position, context.pod_location),
                int(agent.agent_id),
            ),
        )[:TOP_M_TRAINING_CONTRACT]
        if not candidates:
            self.stats["context_j_no_candidate_rows"] += 1
            result = (float("inf"), {"no_candidate": True})
            self._context_j_feature_cache[key] = result
            return result

        from WorldModel.graph_builder import (
            build_action_field,
            compute_preview_legs,
        )

        fixed_context = self._fixed_context(context)
        node_features = self._feature_history.get_latest()
        preview_planner = self.path_planner if self.action_path_mode == 1 else None
        action_rows = []
        for agent in candidates:
            candidate = {
                "robot_id": int(agent.agent_id),
                "robot_start": tuple(agent.position),
            }
            assignment = build_candidate_assignment(candidate, fixed_context)
            legs = compute_preview_legs(
                assignment,
                world_state.map_state,
                self._node_map,
                path_planner=preview_planner,
                world=world_state,
            )
            _, action_global = build_action_field(
                assignment,
                world_state,
                self._node_map,
                self._inv_node_map,
                self._local_capacity,
                node_features=node_features,
                precomputed_legs=legs,
            )
            action_rows.append({"action_global": action_global})
        action = context_j_train._aggregate_action_global(action_rows).to(device)

        bundle = self._context_j_state_bundle
        station_offset = {
            int(station_id): index
            for index, station_id in enumerate(self._psi_layout.station_ids)
        }[int(context.station_id)]
        phi_row = bundle["phi"][station_offset]
        from WorldModel.core.station_congestion_head import CHANNEL_NAMES

        channel_index = {name: int(index) for index, name in enumerate(CHANNEL_NAMES)}
        service = phi_row[channel_index["service"]]
        traffic = phi_row[channel_index["traffic"]]
        marginal_work = 0.5 * action[1] + 0.3 * action[2] + 0.2 * action[5]
        raw_semantic = torch.stack((
            service,
            traffic,
            marginal_work,
            bundle["demand"][5 + station_offset],
        )).to(device)
        station_latent = bundle["station_representation"][station_offset].to(device)
        features_raw = torch.cat((
            station_latent,
            bundle["global_representation"].to(device),
            action,
        ))
        if features_raw.numel() != int(self._context_j_payload["head_config"]["feature_dim"]):
            raise ValueError(
                "online context-J feature dimension differs from checkpoint: "
                f"{features_raw.numel()}"
            )
        features = (
            features_raw - self._context_j_feature_mean
        ) / self._context_j_feature_std
        semantic = (
            (raw_semantic - self._context_j_semantic_lower)
            / self._context_j_semantic_span
        ).clamp(0.0, 1.0)
        semantic = torch.where(
            self._context_j_semantic_constant,
            torch.zeros_like(semantic),
            semantic,
        )
        with torch.no_grad():
            score = float(
                self._context_j_head(
                    features.unsqueeze(0), semantic.unsqueeze(0)
                )[0].item()
            )
        elapsed = (__import__("time").perf_counter() - started) * 1000.0
        self.stats["context_j_eval_calls"] += 1
        self.stats["context_j_eval_time_total_ms"] += elapsed
        self.stats["context_j_score_sum"] += score
        self.stats["context_j_semantic_service_sum"] += float(service.item())
        self.stats["context_j_semantic_traffic_sum"] += float(traffic.item())
        self.stats["context_j_semantic_marginal_work_sum"] += float(
            marginal_work.item()
        )
        self.stats["context_j_semantic_debt_sum"] += float(
            raw_semantic[3].item()
        )
        diagnostics = {
            "no_candidate": False,
            "score": score,
            "service": float(service.item()),
            "traffic": float(traffic.item()),
            "marginal_work": float(marginal_work.item()),
            "service_debt_raw": float(raw_semantic[3].item()),
            "candidate_ids": [int(agent.agent_id) for agent in candidates],
        }
        result = (score, diagnostics)
        self._context_j_feature_cache[key] = result
        return result

    def _dynamic_rows(
        self,
        world_state,
        remaining,
        pending_counts,
        station_channels,
        debt_by_context,
        virtual_idle_agents,
    ) -> list[dict[str, Any]]:
        rows = super()._dynamic_rows(
            world_state,
            remaining,
            pending_counts,
            station_channels,
            debt_by_context,
            virtual_idle_agents,
        )
        for row in rows:
            score, diagnostics = self._online_context_score(
                world_state, row["context"], virtual_idle_agents
            )
            row["analytic_j_score"] = float(row["j_score"])
            row["learned_j_score"] = float(score)
            row["learned_j_diagnostics"] = diagnostics
            row["j_score"] = float(score)
        return rows

    def _append_dynamic_trace(self, record: dict[str, Any]) -> None:
        enriched = dict(record)
        enriched["schema_version"] = LEARNED_CONTEXT_J_SCHEMA_VERSION
        enriched["mode"] = "learned_j_dynamic_interleaved"
        enriched["context_j_checkpoint_sha256"] = self._context_j_checkpoint_sha256
        if self.context_j_trace_enabled:
            super()._append_dynamic_trace(enriched)

    def select_robots(
        self,
        world_state,
        contexts: List[AssignmentContext],
    ) -> Dict[int, int]:
        self._context_j_state_bundle = self._encode_context_state(world_state)
        self._context_j_feature_cache = {}
        try:
            return super().select_robots(world_state, contexts)
        finally:
            self._context_j_state_bundle = None
            self._context_j_feature_cache = {}

    def dynamic_probe_metrics(self) -> dict[str, Any]:
        metrics = super().dynamic_probe_metrics()
        eval_calls = max(int(self.stats.get("context_j_eval_calls", 0)), 1)
        metrics.update({
            "psi_dispatch_schema_version": LEARNED_CONTEXT_J_SCHEMA_VERSION,
            "psi_dispatch_mode": "learned_j_dynamic_interleaved",
            "context_j_head_loaded": bool(self.stats.get("context_j_head_loaded", 0)),
            "context_j_head_checkpoint": self.context_j_checkpoint,
            "context_j_head_checkpoint_sha256": self._context_j_checkpoint_sha256,
            "context_j_eval_calls": int(self.stats.get("context_j_eval_calls", 0)),
            "context_j_eval_time_ms_mean": float(
                self.stats.get("context_j_eval_time_total_ms", 0.0)
            ) / eval_calls,
            "context_j_feature_cache_hits": int(
                self.stats.get("context_j_feature_cache_hits", 0)
            ),
            "context_j_feature_cache_misses": int(
                self.stats.get("context_j_feature_cache_misses", 0)
            ),
            "context_j_no_candidate_rows": int(
                self.stats.get("context_j_no_candidate_rows", 0)
            ),
            "context_j_online_feature_contract": (
                "station_region_latent + global_latent_demand + "
                "top5_nearest_action_global; frozen checkpoint normalization"
            ),
            "context_j_dynamic_update": "virtual_idle_candidate_features_only",
            "context_j_horizon": HORIZON,
            "context_j_rollout_continuation_mode": "behavior",
            "context_j_no_assign_added": False,
            "context_j_hard_gate_added": False,
            "context_j_e_demand_modified": False,
        })
        return metrics


__all__ = [
    "EXPECTED_HEAD_SCHEMA",
    "HORIZON",
    "LEARNED_CONTEXT_J_SCHEMA_VERSION",
    "LearnedContextJDynamicAssigner",
]
