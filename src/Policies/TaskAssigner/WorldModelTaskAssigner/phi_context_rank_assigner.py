"""Opt-in station-pressure ordering for the frozen Phase-C/S1 assigner.

This module is intentionally isolated from :mod:`world_model_task_assigner`.
Existing frozen experiments continue to import and execute the original
``WorldModelTaskAssigner`` unchanged.  The subclass only changes which fixed
contexts are presented first when the number of dispatchable contexts exceeds
the number of idle robots:

1. build a context superset with the policy-neutral fixed-context helper;
2. encode the current state with the already-frozen World Model encoder;
3. evaluate the frozen station-conditioned ``phi_state`` service head;
4. stably order contexts from lower to higher predicted station pressure;
5. delegate robot scoring and S1 conversion to the parent implementation.

There is no hard gate and no NO_ASSIGN action.  If every available context is
for a high-pressure station, assignments still execute.  The learned pressure
therefore acts only as a cross-context capacity allocator; it never edits the
within-context robot score.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch

from Policies.TaskAssigner.base_task_assigner import AssignmentContext
from Policies.TaskAssigner.context_assignment import (
    propose_fixed_assignment_contexts,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.world_model_task_assigner import (
    WorldModelTaskAssigner,
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


PHI_CONTEXT_RANK_SCHEMA_VERSION = "phase_c_phi_context_rank_v1"
PHI_CONTEXT_MODES = ("off", "shadow", "service_ascending")


def _context_key(context: AssignmentContext) -> tuple[int, int, int]:
    return (
        int(context.order_id),
        int(context.pod_id),
        int(context.station_id),
    )


class PhiContextRankWorldModelTaskAssigner(WorldModelTaskAssigner):
    """Frozen S1 robot scorer with an opt-in ``phi_state`` context order.

    ``service_ascending`` treats predicted service pressure as a cost: lower
    pressure stations consume the finite idle-robot budget first.  ``shadow``
    evaluates and records exactly the same ordering but leaves both the
    context proposal limit and execution order unchanged.
    """

    phi_context_rank_schema_version = PHI_CONTEXT_RANK_SCHEMA_VERSION

    def __init__(
        self,
        *,
        phi_head_checkpoint: str,
        phi_scale_contract: str,
        phi_context_mode: str = "off",
        phi_context_factor: float = 2.0,
        phi_trace_enabled: bool = False,
        phi_trace_max_records: int = 5000,
        **kwargs,
    ):
        if phi_context_mode not in PHI_CONTEXT_MODES:
            raise ValueError(
                "phi_context_mode must be one of: "
                + ", ".join(PHI_CONTEXT_MODES)
            )
        if not math.isfinite(float(phi_context_factor)):
            raise ValueError("phi_context_factor must be finite")
        if float(phi_context_factor) < 1.0:
            raise ValueError("phi_context_factor must be at least 1.0")

        # The first causal test must not mix the learned station head with the
        # old handcrafted station injection or a second cross-context pool.
        if float(kwargs.get("station_injection_weight", 0.0)) != 0.0:
            raise ValueError(
                "phi context ranking cannot be combined with station injection"
            )
        if str(kwargs.get("pool_scoring_mode", "off")) != "off":
            raise ValueError(
                "phi context ranking requires pool_scoring_mode='off'"
            )
        if str(kwargs.get("candidate_context_mode", "prefix")) != "prefix":
            raise ValueError(
                "phi context ranking requires candidate_context_mode='prefix'"
            )

        self.phi_head_checkpoint = str(phi_head_checkpoint)
        self.phi_scale_contract = str(phi_scale_contract)
        self.phi_context_mode = str(phi_context_mode)
        self.phi_context_factor = float(phi_context_factor)
        self.phi_trace_enabled = bool(phi_trace_enabled)
        self.phi_trace_max_records = max(0, int(phi_trace_max_records))

        self._phi_head = None
        self._phi_head_payload: Optional[dict] = None
        self._phi_layout: Optional[StationLayout] = None
        self._phi_service_index = int(CHANNEL_NAMES.index("service"))
        self._phi_head_checkpoint_sha256: Optional[str] = None
        self._phi_scale_contract_sha256: Optional[str] = None
        self._phi_last_final_budget: Optional[int] = None
        self._phi_last_original_contexts: list[AssignmentContext] = []
        self.phi_context_trace_records: list[dict] = []

        super().__init__(**kwargs)
        self.stats.update({
            "phi_context_head_loaded": 0,
            "phi_context_proposal_calls": 0,
            "phi_context_requested_budget_sum": 0,
            "phi_context_superset_size_sum": 0,
            "phi_context_superset_extra_sum": 0,
            "phi_context_eval_calls": 0,
            "phi_context_eval_time_total_ms": 0.0,
            "phi_context_contexts_seen": 0,
            "phi_context_unique_stations_sum": 0,
            "phi_context_multi_station_calls": 0,
            "phi_context_reordered_calls": 0,
            "phi_context_changed_positions": 0,
            "phi_context_replacement_count": 0,
            "phi_context_replacement_calls": 0,
            "phi_context_pressure_count": 0,
            "phi_context_pressure_sum": 0.0,
            "phi_context_pressure_min": float("inf"),
            "phi_context_pressure_max": float("-inf"),
            "phi_context_pressure_spread_sum": 0.0,
            "phi_context_pressure_spread_count": 0,
            "phi_context_baseline_budget_pressure_sum": 0.0,
            "phi_context_applied_budget_pressure_sum": 0.0,
            "phi_context_budget_pressure_count": 0,
            "phi_context_selected_pressure_sum": 0.0,
            "phi_context_selected_pressure_count": 0,
            "phi_context_trace_dropped": 0,
        })

    @property
    def phi_context_applied(self) -> bool:
        return self.phi_context_mode == "service_ascending"

    def _init(self, world) -> None:
        """Initialise the unchanged parent model, then the frozen probe."""

        super()._init(world)
        if self.phi_context_mode == "off":
            return
        if not os.path.isfile(self.phi_head_checkpoint):
            raise FileNotFoundError(self.phi_head_checkpoint)
        if not os.path.isfile(self.phi_scale_contract):
            raise FileNotFoundError(self.phi_scale_contract)

        with Path(self.phi_scale_contract).open("r", encoding="utf-8") as handle:
            scale_contract = json.load(handle)
        verify_scale_contract(scale_contract)

        device = next(self._model.parameters()).device
        head, payload = load_frozen_station_head(
            self.phi_head_checkpoint,
            model_checkpoint_path=self.checkpoint_path,
            device=device,
        )
        if payload.get("scale_contract_sha256") != scale_contract.get(
            "contract_sha256"
        ):
            raise ValueError(
                "external phi scale contract differs from the head checkpoint"
            )

        layout = build_station_layout(
            world,
            self._node_map,
            self._edge_index,
        )
        if tuple(layout.station_ids) != tuple(self._station_ids):
            raise ValueError(
                "phi station layout differs from the World Model station order"
            )
        if tuple(layout.station_node_ids) != tuple(self._station_node_ids):
            raise ValueError(
                "phi station node layout differs from the World Model layout"
            )

        self._phi_head = head
        self._phi_head_payload = payload
        self._phi_layout = layout
        self._phi_head_checkpoint_sha256 = sha256_file(
            self.phi_head_checkpoint
        )
        self._phi_scale_contract_sha256 = sha256_file(
            self.phi_scale_contract
        )
        self.stats["phi_context_head_loaded"] = 1

    def propose_assignment_contexts(
        self,
        world_state,
        max_contexts: Optional[int] = None,
    ) -> List[AssignmentContext]:
        """Return a larger read-only context pool only in applied mode."""

        requested = (
            max(0, int(max_contexts))
            if max_contexts is not None else None
        )
        proposal_limit = requested
        if self.phi_context_applied and requested is not None:
            proposal_limit = max(
                requested,
                int(math.ceil(requested * self.phi_context_factor)),
            )
        contexts = propose_fixed_assignment_contexts(
            self,
            world_state,
            max_contexts=proposal_limit,
        )
        self._phi_last_final_budget = requested
        self._phi_last_original_contexts = list(contexts)
        self.stats["phi_context_proposal_calls"] += 1
        self.stats["phi_context_requested_budget_sum"] += int(requested or 0)
        self.stats["phi_context_superset_size_sum"] += len(contexts)
        self.stats["phi_context_superset_extra_sum"] += max(
            0, len(contexts) - int(requested or len(contexts))
        )
        return contexts

    def _current_station_service_pressure(
        self,
        world_state,
    ) -> dict[int, float]:
        if self._phi_head is None or self._phi_head_payload is None:
            raise RuntimeError("phi head was not initialised")
        if self._phi_layout is None:
            raise RuntimeError("phi station layout was not initialised")

        from WorldModel.graph_builder import (
            extract_demand_context,
            extract_edge_features,
        )

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
                head=self._phi_head,
                head_payload=self._phi_head_payload,
                layout=self._phi_layout,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.stats["phi_context_eval_calls"] += 1
        self.stats["phi_context_eval_time_total_ms"] += elapsed_ms

        values = prediction[:, self._phi_service_index].detach().cpu().tolist()
        if len(values) != len(self._phi_layout.station_ids):
            raise RuntimeError("phi head returned the wrong station count")
        result = {
            int(station_id): float(value)
            for station_id, value in zip(
                self._phi_layout.station_ids,
                values,
            )
        }
        if not all(math.isfinite(value) for value in result.values()):
            raise RuntimeError("phi head produced a non-finite pressure")
        return result

    @staticmethod
    def _stable_pressure_order(
        contexts: Sequence[AssignmentContext],
        pressure_by_station: Mapping[int, float],
    ) -> list[int]:
        missing = sorted({
            int(context.station_id)
            for context in contexts
            if int(context.station_id) not in pressure_by_station
        })
        if missing:
            raise ValueError(
                f"phi pressure is missing station ids: {missing}"
            )
        return sorted(
            range(len(contexts)),
            key=lambda index: (
                float(pressure_by_station[int(contexts[index].station_id)]),
                int(index),
            ),
        )

    def _append_phi_trace(self, record: dict) -> None:
        if not self.phi_trace_enabled:
            return
        if len(self.phi_context_trace_records) >= self.phi_trace_max_records:
            self.stats["phi_context_trace_dropped"] += 1
            return
        self.phi_context_trace_records.append(record)

    def _record_pressure_stats(
        self,
        contexts: Sequence[AssignmentContext],
        pressure_by_station: Mapping[int, float],
    ) -> None:
        values = [
            float(pressure_by_station[int(context.station_id)])
            for context in contexts
        ]
        self.stats["phi_context_contexts_seen"] += len(contexts)
        self.stats["phi_context_unique_stations_sum"] += len({
            int(context.station_id) for context in contexts
        })
        if len({int(context.station_id) for context in contexts}) > 1:
            self.stats["phi_context_multi_station_calls"] += 1
        if not values:
            return
        self.stats["phi_context_pressure_count"] += len(values)
        self.stats["phi_context_pressure_sum"] += sum(values)
        self.stats["phi_context_pressure_min"] = min(
            self.stats["phi_context_pressure_min"], min(values)
        )
        self.stats["phi_context_pressure_max"] = max(
            self.stats["phi_context_pressure_max"], max(values)
        )
        self.stats["phi_context_pressure_spread_sum"] += max(values) - min(values)
        self.stats["phi_context_pressure_spread_count"] += 1

    def select_robots(
        self,
        world_state,
        contexts: List[AssignmentContext],
    ) -> Dict[int, int]:
        if self.phi_context_mode == "off" or not contexts:
            return super().select_robots(world_state, contexts)

        original = list(contexts)
        pressure_by_station = self._current_station_service_pressure(world_state)
        order = self._stable_pressure_order(original, pressure_by_station)
        reordered = [original[index] for index in order]
        self._record_pressure_stats(original, pressure_by_station)

        changed_positions = sum(
            int(_context_key(original[index]) != _context_key(reordered[index]))
            for index in range(len(original))
        )
        if changed_positions:
            self.stats["phi_context_reordered_calls"] += 1
            self.stats["phi_context_changed_positions"] += changed_positions

        final_budget = self._phi_last_final_budget
        if final_budget is None:
            final_budget = len(world_state.get_idle_agents())
        final_budget = max(0, min(int(final_budget), len(original)))
        baseline_budget = original[:final_budget]
        applied_budget = reordered[:final_budget]
        baseline_keys = {_context_key(context) for context in baseline_budget}
        applied_keys = {_context_key(context) for context in applied_budget}
        replacements = len(applied_keys - baseline_keys)
        if replacements:
            self.stats["phi_context_replacement_calls"] += 1
            self.stats["phi_context_replacement_count"] += replacements

        if final_budget > 0:
            baseline_pressure = sum(
                pressure_by_station[int(context.station_id)]
                for context in baseline_budget
            ) / float(final_budget)
            applied_pressure = sum(
                pressure_by_station[int(context.station_id)]
                for context in applied_budget
            ) / float(final_budget)
            self.stats["phi_context_baseline_budget_pressure_sum"] += (
                baseline_pressure
            )
            self.stats["phi_context_applied_budget_pressure_sum"] += (
                applied_pressure
            )
            self.stats["phi_context_budget_pressure_count"] += 1
        else:
            baseline_pressure = 0.0
            applied_pressure = 0.0

        if self.phi_context_applied:
            contexts[:] = reordered

        choices = super().select_robots(world_state, contexts)
        selected_pressure = [
            pressure_by_station[int(contexts[index].station_id)]
            for index in choices
            if 0 <= int(index) < len(contexts)
        ]
        if selected_pressure:
            self.stats["phi_context_selected_pressure_sum"] += sum(
                selected_pressure
            )
            self.stats["phi_context_selected_pressure_count"] += len(
                selected_pressure
            )

        self._append_phi_trace({
            "schema_version": PHI_CONTEXT_RANK_SCHEMA_VERSION,
            "tick": int(getattr(world_state, "tick", -1)),
            "mode": self.phi_context_mode,
            "context_count": len(original),
            "final_budget": int(final_budget),
            "unique_station_count": len({
                int(context.station_id) for context in original
            }),
            "changed_positions": int(changed_positions),
            "context_replacements": int(replacements),
            "baseline_budget_pressure_mean": float(baseline_pressure),
            "applied_budget_pressure_mean": float(applied_pressure),
            "selected_pressure_mean": (
                float(sum(selected_pressure) / len(selected_pressure))
                if selected_pressure else None
            ),
            "pressure_min": min(pressure_by_station.values()),
            "pressure_max": max(pressure_by_station.values()),
            "first_station_before": int(original[0].station_id),
            "first_station_after": int(reordered[0].station_id),
        })
        return choices

    def phi_context_metrics(self) -> dict:
        """Return compact, JSON-safe diagnostics for independent runners."""

        stats = self.stats
        eval_calls = int(stats.get("phi_context_eval_calls", 0))
        proposal_calls = int(stats.get("phi_context_proposal_calls", 0))
        pressure_count = int(stats.get("phi_context_pressure_count", 0))
        spread_count = int(stats.get("phi_context_pressure_spread_count", 0))
        budget_count = int(stats.get("phi_context_budget_pressure_count", 0))
        selected_count = int(stats.get("phi_context_selected_pressure_count", 0))
        pressure_min = float(stats.get("phi_context_pressure_min", float("inf")))
        pressure_max = float(stats.get("phi_context_pressure_max", float("-inf")))
        return {
            "phi_context_schema_version": PHI_CONTEXT_RANK_SCHEMA_VERSION,
            "phi_context_mode": self.phi_context_mode,
            "phi_context_factor": self.phi_context_factor,
            "phi_context_head_loaded": bool(
                stats.get("phi_context_head_loaded", 0)
            ),
            "phi_context_head_checkpoint_sha256": (
                self._phi_head_checkpoint_sha256
            ),
            "phi_context_scale_contract_sha256": (
                self._phi_scale_contract_sha256
            ),
            "phi_context_proposal_calls": proposal_calls,
            "phi_context_requested_budget_mean": (
                float(stats.get("phi_context_requested_budget_sum", 0))
                / max(proposal_calls, 1)
            ),
            "phi_context_superset_size_mean": (
                float(stats.get("phi_context_superset_size_sum", 0))
                / max(proposal_calls, 1)
            ),
            "phi_context_superset_extra_mean": (
                float(stats.get("phi_context_superset_extra_sum", 0))
                / max(proposal_calls, 1)
            ),
            "phi_context_eval_calls": eval_calls,
            "phi_context_eval_time_ms_mean": (
                float(stats.get("phi_context_eval_time_total_ms", 0.0))
                / max(eval_calls, 1)
            ),
            "phi_context_contexts_seen": int(
                stats.get("phi_context_contexts_seen", 0)
            ),
            "phi_context_unique_stations_mean": (
                float(stats.get("phi_context_unique_stations_sum", 0))
                / max(eval_calls, 1)
            ),
            "phi_context_multi_station_calls": int(
                stats.get("phi_context_multi_station_calls", 0)
            ),
            "phi_context_reordered_calls": int(
                stats.get("phi_context_reordered_calls", 0)
            ),
            "phi_context_changed_positions": int(
                stats.get("phi_context_changed_positions", 0)
            ),
            "phi_context_replacement_calls": int(
                stats.get("phi_context_replacement_calls", 0)
            ),
            "phi_context_replacement_count": int(
                stats.get("phi_context_replacement_count", 0)
            ),
            "phi_context_pressure_mean": (
                float(stats.get("phi_context_pressure_sum", 0.0))
                / max(pressure_count, 1)
            ),
            "phi_context_pressure_min": (
                pressure_min if math.isfinite(pressure_min) else None
            ),
            "phi_context_pressure_max": (
                pressure_max if math.isfinite(pressure_max) else None
            ),
            "phi_context_pressure_spread_mean": (
                float(stats.get("phi_context_pressure_spread_sum", 0.0))
                / max(spread_count, 1)
            ),
            "phi_context_baseline_budget_pressure_mean": (
                float(stats.get(
                    "phi_context_baseline_budget_pressure_sum", 0.0
                )) / max(budget_count, 1)
            ),
            "phi_context_applied_budget_pressure_mean": (
                float(stats.get(
                    "phi_context_applied_budget_pressure_sum", 0.0
                )) / max(budget_count, 1)
            ),
            "phi_context_selected_pressure_mean": (
                float(stats.get("phi_context_selected_pressure_sum", 0.0))
                / max(selected_count, 1)
            ),
            "phi_context_trace_records": len(self.phi_context_trace_records),
            "phi_context_trace_dropped": int(
                stats.get("phi_context_trace_dropped", 0)
            ),
            "phi_context_robot_scorer": (
                "WorldModelTaskAssigner.select_robots_unmodified"
            ),
            "phi_context_no_assign_added": False,
            "phi_context_hard_gate_added": False,
        }


__all__ = [
    "PHI_CONTEXT_MODES",
    "PHI_CONTEXT_RANK_SCHEMA_VERSION",
    "PhiContextRankWorldModelTaskAssigner",
]
