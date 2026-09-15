"""Opt-in station/context dispatch policy built on the frozen S1 scorer.

The existing ``phi_context_rank_assigner`` is intentionally untouched.  This
subclass changes only the *context* order presented to the parent S1 robot
scorer:

* learned ``service`` and ``traffic`` are evaluated once per station;
* exact pending-chain work and context age form ``service_debt``;
* ``J(c) = .5*service + .5*traffic - service_debt`` is used across contexts;
* the selected context is still passed to the unchanged S1 robot scorer.

No ``e_demand`` schema, World Model checkpoint, NO_ASSIGN action, hard gate,
Lyapunov/dispatch-potential term, or station-injection term is introduced.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from Policies.TaskAssigner.base_task_assigner import AssignmentContext
from Policies.TaskAssigner.context_assignment import (
    enumerate_pending_assignment_contexts,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.world_model_task_assigner import (
    WorldModelTaskAssigner,
)
from WorldModel.core.psi_dispatch import (
    PSI_DISPATCH_SCHEMA_VERSION,
    DispatchServiceDurations,
    StaticGraphDistance,
    build_context_debt_features,
    context_key,
    dispatch_cost,
)
from WorldModel.core.station_congestion_head import (
    CHANNEL_NAMES,
    verify_scale_contract,
)
from WorldModel.evaluation.station_congestion_endpoint import (
    StationLayout,
    build_station_layout,
    evaluate_station_psi,
    load_frozen_station_head,
    sha256_file,
)


PSI_CONTEXT_MODES = ("off", "shadow", "j_ascending")


class PsiDispatchContextWorldModelTaskAssigner(WorldModelTaskAssigner):
    """Frozen S1 robot scorer with an explicit cross-context ``J(c)`` layer."""

    psi_dispatch_schema_version = PSI_DISPATCH_SCHEMA_VERSION

    def __init__(
        self,
        *,
        psi_head_checkpoint: str,
        psi_scale_contract: str,
        psi_context_mode: str = "off",
        psi_trace_enabled: bool = False,
        psi_trace_max_records: int = 5000,
        allow_phasec_s0_robot_scorer: bool = False,
        **kwargs,
    ):
        if psi_context_mode not in PSI_CONTEXT_MODES:
            raise ValueError(
                "psi_context_mode must be one of: "
                + ", ".join(PSI_CONTEXT_MODES)
            )
        # The new layer is the only cross-context mechanism in this experiment.
        forbidden_nonzero = {
            "station_injection_weight": float(
                kwargs.get("station_injection_weight", 0.0)
            ),
            "lyapunov_l0_lambda": float(kwargs.get("lyapunov_l0_lambda", 0.0)),
            "work_drift_lambda": float(kwargs.get("work_drift_lambda", 0.0)),
        }
        for name, value in forbidden_nonzero.items():
            if value != 0.0:
                raise ValueError(f"psi dispatch cannot combine with {name}")
        if str(kwargs.get("pool_scoring_mode", "off")) != "off":
            raise ValueError("psi dispatch requires pool_scoring_mode='off'")
        if str(kwargs.get("dispatch_potential_mode", "off")) != "off":
            raise ValueError("psi dispatch requires dispatch_potential_mode='off'")
        if str(kwargs.get("lyapunov_l0_mode", "off")) != "off":
            raise ValueError("psi dispatch requires lyapunov_l0_mode='off'")
        if str(kwargs.get("work_drift_mode", "off")) != "off":
            raise ValueError("psi dispatch requires work_drift_mode='off'")
        if bool(kwargs.get("include_no_assign_candidate", False)):
            raise ValueError("psi dispatch does not add NO_ASSIGN")
        if str(kwargs.get("candidate_context_mode", "prefix")) != "prefix":
            raise ValueError("psi dispatch requires candidate_context_mode='prefix'")
        if float(kwargs.get("energy_conv_random_flip_rate", 0.0)) != 0.0:
            raise ValueError("psi dispatch requires deterministic S1 conversion")
        energy_scoring_mode = str(kwargs.get("energy_scoring_mode", "off"))
        if energy_scoring_mode != "conversion" and not (
            bool(allow_phasec_s0_robot_scorer)
            and energy_scoring_mode == "off"
        ):
            raise ValueError(
                "psi dispatch requires the certified S1 conversion; "
                "the Phase-C S0 scorer is available only through the "
                "explicit factorial-experiment opt-in"
            )

        self.psi_head_checkpoint = str(psi_head_checkpoint)
        self.psi_scale_contract = str(psi_scale_contract)
        self.psi_context_mode = str(psi_context_mode)
        self.psi_trace_enabled = bool(psi_trace_enabled)
        self.psi_trace_max_records = max(0, int(psi_trace_max_records))
        self.psi_robot_scorer_variant = (
            "phasec_s0"
            if energy_scoring_mode == "off"
            else "s1_within_context"
        )

        self._psi_head = None
        self._psi_head_payload: Optional[dict] = None
        self._psi_layout: Optional[StationLayout] = None
        self._psi_head_checkpoint_sha256: Optional[str] = None
        self._psi_scale_contract_sha256: Optional[str] = None
        self._psi_source_encoder_checkpoint_sha256: Optional[str] = None
        self._psi_encoder_contract_verified = False
        self._psi_distance = None
        self._psi_durations: Optional[DispatchServiceDurations] = None
        self._psi_last_final_budget: Optional[int] = None
        self.psi_dispatch_trace_records: list[dict] = []

        super().__init__(**kwargs)
        self.stats.update({
            "psi_dispatch_head_loaded": 0,
            "psi_dispatch_proposal_calls": 0,
            "psi_dispatch_requested_budget_sum": 0,
            "psi_dispatch_superset_size_sum": 0,
            "psi_dispatch_eval_calls": 0,
            "psi_dispatch_eval_time_total_ms": 0.0,
            "psi_dispatch_contexts_seen": 0,
            "psi_dispatch_unique_stations_sum": 0,
            "psi_dispatch_reordered_calls": 0,
            "psi_dispatch_changed_positions": 0,
            "psi_dispatch_replacement_calls": 0,
            "psi_dispatch_replacement_count": 0,
            "psi_dispatch_selected_count": 0,
            "psi_dispatch_service_sum": 0.0,
            "psi_dispatch_traffic_sum": 0.0,
            "psi_dispatch_backlog_sum": 0.0,
            "psi_dispatch_age_sum": 0.0,
            "psi_dispatch_debt_sum": 0.0,
            "psi_dispatch_j_sum": 0.0,
            "psi_dispatch_trace_dropped": 0,
        })

    @property
    def psi_dispatch_applied(self) -> bool:
        return self.psi_context_mode == "j_ascending"

    def _init(self, world) -> None:
        super()._init(world)
        if self.psi_context_mode == "off":
            return
        if not os.path.isfile(self.psi_head_checkpoint):
            raise FileNotFoundError(self.psi_head_checkpoint)
        if not os.path.isfile(self.psi_scale_contract):
            raise FileNotFoundError(self.psi_scale_contract)

        with Path(self.psi_scale_contract).open("r", encoding="utf-8") as handle:
            scale_contract = json.load(handle)
        verify_scale_contract(scale_contract)

        device = next(self._model.parameters()).device
        head, payload = load_frozen_station_head(
            self.psi_head_checkpoint,
            model_checkpoint_path=self.checkpoint_path,
            device=device,
        )
        if payload.get("scale_contract_sha256") != scale_contract.get(
            "contract_sha256"
        ):
            raise ValueError("psi scale contract differs from head checkpoint")

        layout = build_station_layout(world, self._node_map, self._edge_index)
        if tuple(layout.station_ids) != tuple(self._station_ids):
            raise ValueError("psi station layout differs from World Model order")
        if tuple(layout.station_node_ids) != tuple(self._station_node_ids):
            raise ValueError("psi station node layout differs from World Model")

        self._psi_head = head
        self._psi_head_payload = payload
        self._psi_layout = layout
        self._psi_head_checkpoint_sha256 = sha256_file(self.psi_head_checkpoint)
        self._psi_scale_contract_sha256 = sha256_file(self.psi_scale_contract)
        self._psi_source_encoder_checkpoint_sha256 = str(
            payload.get("source_encoder_checkpoint_sha256") or ""
        ) or None
        self._psi_encoder_contract_verified = (
            self._psi_source_encoder_checkpoint_sha256
            == sha256_file(self.checkpoint_path)
        )
        if not self._psi_encoder_contract_verified:
            raise ValueError("psi station head encoder binding was not verified")
        self._psi_distance = StaticGraphDistance(
            self._node_map, self._adj
        )
        self._psi_durations = DispatchServiceDurations.from_world(world)
        self.stats["psi_dispatch_head_loaded"] = 1

    @staticmethod
    def _all_dispatchable_contexts(world_state, assigner) -> List[AssignmentContext]:
        """Return the complete available context set, not an idle prefix."""

        candidates = enumerate_pending_assignment_contexts(
            assigner,
            world_state,
            max_contexts=None,
            include_temporarily_unavailable=True,
            deduplicate_pods=False,
        )
        reserved_pods = {
            int(task.pod_id)
            for task in getattr(world_state.task_state, "tasks", {}).values()
            if getattr(task.status, "name", str(task.status))
            in ("ASSIGNED", "IN_PROGRESS")
        }
        seen_pods: set[int] = set()
        result: List[AssignmentContext] = []
        for context in candidates:
            pod = world_state.pod_state.get_pod(int(context.pod_id))
            if pod is None or bool(getattr(pod, "is_carried", False)):
                continue
            if int(context.pod_id) in reserved_pods:
                continue
            if int(context.pod_id) in seen_pods:
                continue
            seen_pods.add(int(context.pod_id))
            result.append(context)
        return result

    def propose_assignment_contexts(
        self,
        world_state,
        max_contexts: Optional[int] = None,
    ) -> List[AssignmentContext]:
        if self.psi_context_mode == "off":
            return super().propose_assignment_contexts(
                world_state, max_contexts=max_contexts
            )
        contexts = self._all_dispatchable_contexts(world_state, self)
        requested = (
            max(0, int(max_contexts))
            if max_contexts is not None
            else len(world_state.get_idle_agents())
        )
        self._psi_last_final_budget = requested
        self.stats["psi_dispatch_proposal_calls"] += 1
        self.stats["psi_dispatch_requested_budget_sum"] += requested
        self.stats["psi_dispatch_superset_size_sum"] += len(contexts)
        return contexts

    def _evaluate_station_channels(self, world_state) -> Dict[int, Dict[str, float]]:
        if self._psi_head is None or self._psi_head_payload is None:
            raise RuntimeError("psi head was not initialised")
        if self._psi_layout is None:
            raise RuntimeError("psi station layout was not initialised")

        from WorldModel.graph_builder import extract_demand_context, extract_edge_features

        started = time.perf_counter()
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
            z, _, _ = self._model.encode_state(
                node_history,
                self._edge_index,
                edge_features,
                demand,
            )
            prediction = evaluate_station_psi(
                z,
                head=self._psi_head,
                head_payload=self._psi_head_payload,
                layout=self._psi_layout,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.stats["psi_dispatch_eval_calls"] += 1
        self.stats["psi_dispatch_eval_time_total_ms"] += elapsed_ms
        channel_index = {name: int(index) for index, name in enumerate(CHANNEL_NAMES)}
        result: Dict[int, Dict[str, float]] = {}
        for row, station_id in zip(
            prediction.detach().cpu().tolist(), self._psi_layout.station_ids
        ):
            result[int(station_id)] = {
                "service": float(row[channel_index["service"]]),
                "traffic": float(row[channel_index["traffic"]]),
            }
        return result

    def _append_trace(self, payload: dict) -> None:
        if not self.psi_trace_enabled:
            return
        if len(self.psi_dispatch_trace_records) >= self.psi_trace_max_records:
            self.stats["psi_dispatch_trace_dropped"] += 1
            return
        self.psi_dispatch_trace_records.append(payload)

    def select_robots(self, world_state, contexts: List[AssignmentContext]):
        if self.psi_context_mode == "off" or not contexts:
            return super().select_robots(world_state, contexts)
        if self._psi_distance is None or self._psi_durations is None:
            raise RuntimeError("psi dispatch geometry was not initialised")

        station_channels = self._evaluate_station_channels(world_state)
        idle_agents = tuple(world_state.get_idle_agents())
        debt_by_context = build_context_debt_features(
            world_state,
            contexts,
            idle_agents,
            self._psi_distance,
            self._psi_durations,
        )
        rows = []
        for index, context in enumerate(contexts):
            station_id = int(context.station_id)
            channels = station_channels.get(station_id)
            if channels is None:
                raise ValueError(f"missing psi channels for station {station_id}")
            debt = debt_by_context[context_key(context)]
            j_score = dispatch_cost(
                channels["service"],
                channels["traffic"],
                debt.service_debt,
            )
            rows.append({
                "original_index": int(index),
                "order_id": int(context.order_id),
                "pod_id": int(context.pod_id),
                "station_id": station_id,
                "service": float(channels["service"]),
                "traffic": float(channels["traffic"]),
                "service_debt": float(debt.service_debt),
                "backlog_score": float(debt.backlog_score),
                "age_score": float(debt.age_score),
                "unserved_chain_count": int(debt.unserved_chain_count),
                "order_age_ticks": int(debt.order_age_ticks),
                "free_flow_time": float(debt.free_flow_time),
                "j_score": float(j_score),
            })

        order = sorted(
            range(len(rows)),
            key=lambda i: (
                rows[i]["j_score"],
                -rows[i]["service_debt"],
                -rows[i]["age_score"],
                rows[i]["original_index"],
            ),
        )
        reordered = [contexts[index] for index in order]
        changed_positions = sum(
            int(context_key(contexts[index]) != context_key(reordered[index]))
            for index in range(len(contexts))
        )
        final_budget = self._psi_last_final_budget
        if final_budget is None:
            final_budget = len(idle_agents)
        final_budget = max(0, min(int(final_budget), len(contexts)))
        baseline_keys = {
            context_key(context) for context in contexts[:final_budget]
        }
        applied_keys = {
            context_key(context) for context in reordered[:final_budget]
        }
        replacements = len(applied_keys - baseline_keys)

        self.stats["psi_dispatch_contexts_seen"] += len(rows)
        self.stats["psi_dispatch_unique_stations_sum"] += len({
            row["station_id"] for row in rows
        })
        self.stats["psi_dispatch_reordered_calls"] += int(changed_positions > 0)
        self.stats["psi_dispatch_changed_positions"] += changed_positions
        self.stats["psi_dispatch_replacement_calls"] += int(replacements > 0)
        self.stats["psi_dispatch_replacement_count"] += replacements
        for row in rows:
            self.stats["psi_dispatch_service_sum"] += row["service"]
            self.stats["psi_dispatch_traffic_sum"] += row["traffic"]
            self.stats["psi_dispatch_backlog_sum"] += row["backlog_score"]
            self.stats["psi_dispatch_age_sum"] += row["age_score"]
            self.stats["psi_dispatch_debt_sum"] += row["service_debt"]
            self.stats["psi_dispatch_j_sum"] += row["j_score"]

        if self.psi_context_mode == "j_ascending":
            contexts[:] = reordered
        choices = super().select_robots(world_state, contexts)
        selected_indices = set(int(index) for index in choices)
        selected_keys = {
            context_key(contexts[index])
            for index in selected_indices
            if 0 <= index < len(contexts)
        }
        self.stats["psi_dispatch_selected_count"] += len(selected_keys)
        for rank, row_index in enumerate(order):
            row = rows[row_index]
            row["j_rank"] = int(rank)
            row["applied"] = bool(self.psi_context_mode == "j_ascending")
            row["selected"] = context_key(
                reordered[rank] if self.psi_context_mode == "j_ascending" else contexts[row_index]
            ) in selected_keys
        self._append_trace({
            "schema_version": PSI_DISPATCH_SCHEMA_VERSION,
            "tick": int(getattr(world_state, "tick", -1)),
            "mode": self.psi_context_mode,
            "context_count": len(contexts),
            "final_budget": int(final_budget),
            "changed_positions": int(changed_positions),
            "context_replacements": int(replacements),
            "station_channels": station_channels,
            "contexts": rows,
        })
        return choices

    def psi_dispatch_metrics(self) -> dict:
        stats = self.stats
        seen = max(int(stats.get("psi_dispatch_contexts_seen", 0)), 1)
        eval_calls = max(int(stats.get("psi_dispatch_eval_calls", 0)), 1)
        proposal_calls = max(int(stats.get("psi_dispatch_proposal_calls", 0)), 1)
        return {
            "psi_dispatch_schema_version": PSI_DISPATCH_SCHEMA_VERSION,
            "psi_dispatch_mode": self.psi_context_mode,
            "psi_dispatch_head_loaded": bool(
                stats.get("psi_dispatch_head_loaded", 0)
            ),
            "psi_dispatch_head_checkpoint_sha256": self._psi_head_checkpoint_sha256,
            "psi_dispatch_scale_contract_sha256": self._psi_scale_contract_sha256,
            "psi_dispatch_source_encoder_checkpoint_sha256": (
                self._psi_source_encoder_checkpoint_sha256
            ),
            "psi_dispatch_encoder_contract_verified": bool(
                self._psi_encoder_contract_verified
            ),
            "psi_dispatch_proposal_calls": int(
                stats.get("psi_dispatch_proposal_calls", 0)
            ),
            "psi_dispatch_requested_budget_mean": float(
                stats.get("psi_dispatch_requested_budget_sum", 0)
            ) / proposal_calls,
            "psi_dispatch_superset_size_mean": float(
                stats.get("psi_dispatch_superset_size_sum", 0)
            ) / proposal_calls,
            "psi_dispatch_eval_calls": int(
                stats.get("psi_dispatch_eval_calls", 0)
            ),
            "psi_dispatch_eval_time_ms_mean": float(
                stats.get("psi_dispatch_eval_time_total_ms", 0.0)
            ) / eval_calls,
            "psi_dispatch_contexts_seen": int(
                stats.get("psi_dispatch_contexts_seen", 0)
            ),
            "psi_dispatch_unique_stations_mean": float(
                stats.get("psi_dispatch_unique_stations_sum", 0)
            ) / eval_calls,
            "psi_dispatch_reordered_calls": int(
                stats.get("psi_dispatch_reordered_calls", 0)
            ),
            "psi_dispatch_changed_positions": int(
                stats.get("psi_dispatch_changed_positions", 0)
            ),
            "psi_dispatch_replacement_calls": int(
                stats.get("psi_dispatch_replacement_calls", 0)
            ),
            "psi_dispatch_replacement_count": int(
                stats.get("psi_dispatch_replacement_count", 0)
            ),
            "psi_dispatch_service_mean": float(
                stats.get("psi_dispatch_service_sum", 0.0)
            ) / seen,
            "psi_dispatch_traffic_mean": float(
                stats.get("psi_dispatch_traffic_sum", 0.0)
            ) / seen,
            "psi_dispatch_backlog_mean": float(
                stats.get("psi_dispatch_backlog_sum", 0.0)
            ) / seen,
            "psi_dispatch_age_mean": float(
                stats.get("psi_dispatch_age_sum", 0.0)
            ) / seen,
            "psi_dispatch_service_debt_mean": float(
                stats.get("psi_dispatch_debt_sum", 0.0)
            ) / seen,
            "psi_dispatch_j_mean": float(
                stats.get("psi_dispatch_j_sum", 0.0)
            ) / seen,
            "psi_dispatch_trace_records": len(self.psi_dispatch_trace_records),
            "psi_dispatch_trace_dropped": int(
                stats.get("psi_dispatch_trace_dropped", 0)
            ),
            "psi_dispatch_robot_scorer": (
                "WorldModelTaskAssigner.select_robots_unmodified"
            ),
            "psi_dispatch_robot_scorer_variant": (
                self.psi_robot_scorer_variant
            ),
            "psi_dispatch_s1_within_context": (
                self.psi_robot_scorer_variant == "s1_within_context"
            ),
            "psi_dispatch_no_assign_added": False,
            "psi_dispatch_hard_gate_added": False,
            "psi_dispatch_e_demand_modified": False,
            "psi_dispatch_attention_added": False,
        }


__all__ = [
    "PSI_CONTEXT_MODES",
    "PsiDispatchContextWorldModelTaskAssigner",
]
