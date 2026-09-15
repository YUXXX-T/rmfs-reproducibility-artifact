"""Dynamic J + station-conditioned DEFER_CONTEXT + unchanged S1.

This opt-in assigner is isolated from all frozen Phase-C arms.  It repeatedly
selects the lowest Dynamic-J context, compares that context's service/liveness
credit with the target station's current pipeline risk, and either:

* executes the context through the unchanged S1 robot scorer; or
* defers only that context and continues examining contexts for other stations.

The engine must additionally run committed-capacity admission V1, which is the
separate physical safety invariant.
"""

from __future__ import annotations

from typing import Any, Dict, List

from Policies.TaskAssigner.base_task_assigner import (
    AssignmentContext,
    fulfilled_pod_ids_for_order,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_dynamic_probe_assigner import (
    DynamicPsiDispatchProbeAssigner,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.world_model_task_assigner import (
    WorldModelTaskAssigner,
)
from WorldModel.core.psi_dispatch import (
    build_context_debt_features,
)
from WorldModel.core.station_context_defer import (
    STATION_CONTEXT_DEFER_PIPELINE_PHI_SCHEMA_VERSION,
    STATION_CONTEXT_DEFER_RISK_PIPELINE_PHI_V2,
    STATION_CONTEXT_DEFER_RISK_READY_MAX_V1,
    STATION_CONTEXT_DEFER_SCHEMA_VERSION,
    build_station_dispatch_snapshots,
    eligible_defer_increment,
    station_context_defer_decision,
)
from WorldState.order_state import OrderStatus
from WorldState.task_state import TaskStatus


STATION_CONTEXT_DEFER_ASSIGNER_SCHEMA_VERSION = (
    "phase_c_dynamic_j_station_context_defer_s1_v1"
)


class StationContextDeferDynamicPsiAssigner(
    DynamicPsiDispatchProbeAssigner
):
    """Dynamic J context choice with local defer and unchanged S1 ranking."""

    station_context_defer_assigner_schema_version = (
        STATION_CONTEXT_DEFER_ASSIGNER_SCHEMA_VERSION
    )
    station_context_defer_schema_version = (
        STATION_CONTEXT_DEFER_SCHEMA_VERSION
    )
    station_context_defer_risk_mode = (
        STATION_CONTEXT_DEFER_RISK_READY_MAX_V1
    )
    station_context_defer_mode = (
        "dynamic_j_then_station_context_defer_then_s1"
    )
    station_feedback_batch_filter_hook_enabled = False

    def __init__(self, **kwargs) -> None:
        self._station_context_defer_mass: dict[
            tuple[int, int, int], float
        ] = {}
        self._station_context_defer_ticks: dict[
            tuple[int, int, int], int
        ] = {}
        self._station_context_defer_bound: dict[
            tuple[int, int, int], int
        ] = {}
        self._station_context_defer_clock: dict[
            tuple[int, int, int], float
        ] = {}
        self._station_context_defer_last_update_tick: dict[
            tuple[int, int, int], int
        ] = {}
        super().__init__(**kwargs)
        self.stats.update({
            "station_context_defer_batches": 0,
            "station_context_defer_steps": 0,
            "station_context_defer_evaluations": 0,
            "station_context_defer_decisions": 0,
            "station_context_defer_unique_contexts": 0,
            "station_context_defer_all_remaining_batches": 0,
            "station_context_defer_selected": 0,
            "station_context_defer_debt_override_selected": 0,
            "station_context_defer_debt_updates": 0,
            "station_context_defer_duplicate_tick_updates_suppressed": 0,
            "station_context_defer_risk_sum": 0.0,
            "station_context_defer_context_credit_sum": 0.0,
            "station_context_defer_pipeline_excess_sum": 0.0,
            "station_context_defer_ready_contention_sum": 0.0,
            "station_context_defer_ready_would_dominate_evaluations": 0,
            "station_context_defer_phi_pressure_sum": 0.0,
            "station_context_defer_positive_risk_evaluations": 0,
            "station_context_defer_zero_risk_evaluations": 0,
            "station_context_defer_risk_max": 0.0,
            "station_context_defer_mass_max": 0.0,
            "station_context_defer_eligible_ticks_max": 0,
            "station_context_defer_liveness_bound_violations": 0,
            "station_context_defer_ready_unadmitted_max": 0,
            "station_context_defer_upstream_pipeline_max": 0,
            "station_context_defer_trace_dropped": 0,
        })

    @staticmethod
    def _context_identity(context: AssignmentContext) -> tuple[int, int, int]:
        return (
            int(context.order_id),
            int(context.pod_id),
            int(context.station_id),
        )

    def _prune_defer_mass(self, world_state) -> None:
        active_chain_keys = {
            (int(task.order_id), int(task.pod_id))
            for task in world_state.task_state.tasks.values()
            if task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        }
        for key in list(self._station_context_defer_mass):
            order_id, pod_id, _station_id = key
            order = world_state.order_state.orders.get(order_id)
            if (
                order is None
                or order.status != OrderStatus.PENDING
                or (order_id, pod_id) in active_chain_keys
                or pod_id in fulfilled_pod_ids_for_order(world_state, order)
            ):
                self._station_context_defer_mass.pop(key, None)
                self._station_context_defer_ticks.pop(key, None)
                self._station_context_defer_bound.pop(key, None)
                self._station_context_defer_clock.pop(key, None)
                self._station_context_defer_last_update_tick.pop(key, None)

    def _defer_clock_for_decision(
        self,
        key: tuple[int, int, int],
        observed_free_flow_time: float,
    ) -> float:
        """Return the first-defer chain clock once a ledger exists.

        Dynamic-J recomputes context features every tick.  That is desirable
        for context ordering, but liveness debt must not silently accelerate
        or slow down when the currently available robot set changes.  A
        context therefore uses its current static free-flow estimate until it
        is first deferred, then reuses that frozen value for the lifetime of
        the defer ledger.
        """

        frozen = self._station_context_defer_clock.get(key)
        if frozen is not None:
            return float(frozen)
        observed = float(observed_free_flow_time)
        if observed <= 0.0:
            raise ValueError("observed_free_flow_time must be positive")
        return observed

    def _record_decision_stats(self, decision, snapshot) -> None:
        self.stats["station_context_defer_evaluations"] += 1
        self.stats["station_context_defer_risk_sum"] += float(
            decision.station_risk
        )
        self.stats["station_context_defer_context_credit_sum"] += float(
            decision.context_credit
        )
        self.stats["station_context_defer_pipeline_excess_sum"] += float(
            decision.pipeline_excess_score
        )
        self.stats["station_context_defer_ready_contention_sum"] += float(
            decision.ready_contention_score
        )
        self.stats[
            "station_context_defer_ready_would_dominate_evaluations"
        ] += int(decision.ready_contention_would_dominate)
        self.stats["station_context_defer_phi_pressure_sum"] += float(
            decision.phi_pressure
        )
        self.stats["station_context_defer_risk_max"] = max(
            float(self.stats["station_context_defer_risk_max"]),
            float(decision.station_risk),
        )
        if float(decision.station_risk) > 1e-9:
            self.stats[
                "station_context_defer_positive_risk_evaluations"
            ] += 1
        else:
            self.stats["station_context_defer_zero_risk_evaluations"] += 1
        self.stats["station_context_defer_ready_unadmitted_max"] = max(
            int(self.stats["station_context_defer_ready_unadmitted_max"]),
            int(snapshot.ready_unadmitted),
        )
        self.stats["station_context_defer_upstream_pipeline_max"] = max(
            int(self.stats["station_context_defer_upstream_pipeline_max"]),
            int(snapshot.upstream_pipeline),
        )

    def _prepare_station_context_feedback(
        self,
        world_state,
        station_snapshots,
    ) -> None:
        """Opt-in hook for isolated station-feedback development arms.

        Frozen V1/V2 assigners intentionally do nothing here.  Keeping the
        default hook empty preserves their context order, defer ledgers, task
        materialisation, traces, and admission contract.
        """

    def _station_context_feedback_for_row(
        self,
        world_state,
        chosen,
        snapshot,
    ) -> dict[str, Any] | None:
        """Return an optional auditable station-local defer override."""

        return None

    def _record_station_context_feedback_outcome(
        self,
        feedback: dict[str, Any] | None,
        status: str,
    ) -> None:
        """Opt-in post-decision hook; frozen assigners remain unchanged."""

    def _station_context_feedback_batch_filter(
        self,
        world_state,
        remaining: list[tuple[int, AssignmentContext]],
        station_snapshots,
    ) -> tuple[
        list[tuple[int, AssignmentContext]],
        list[dict[str, Any]],
    ]:
        """Optionally remove station-local contexts before Dynamic-J.

        The frozen assigners return the input unchanged.  Experimental
        subclasses may only remove rows while preserving the relative order
        of every retained context.  Debt features and pending counts have
        already been built from the full context batch before this hook runs,
        so filtering cannot alter the Dynamic-J inputs of healthy stations.
        """

        return list(remaining), []

    def _select_robots_interleaved(
        self,
        world_state,
        contexts: List[AssignmentContext],
        station_channels: Dict[int, Dict[str, float]],
    ) -> Dict[int, int]:
        """Run Dynamic J -> local defer -> S1 repeatedly within one tick."""

        self._prune_defer_mass(world_state)
        idle_agents = tuple(world_state.get_idle_agents())
        debt_by_context = build_context_debt_features(
            world_state,
            contexts,
            idle_agents,
            self._psi_distance,
            self._psi_durations,
        )
        pending_counts = self._initial_pending_counts(debt_by_context)
        station_snapshots = build_station_dispatch_snapshots(world_state)
        self._prepare_station_context_feedback(
            world_state, station_snapshots
        )
        virtual_pipeline_additions = {
            station_id: 0 for station_id in station_snapshots
        }
        original_remaining: list[tuple[int, AssignmentContext]] = list(
            enumerate(contexts)
        )
        if self.station_feedback_batch_filter_hook_enabled:
            remaining, feedback_batch_steps = (
                self._station_context_feedback_batch_filter(
                    world_state,
                    original_remaining,
                    station_snapshots,
                )
            )
            retained_indices = [
                int(index) for index, _context in remaining
            ]
            retained_index_set = set(retained_indices)
            expected_retained_order = [
                int(index)
                for index, _context in original_remaining
                if int(index) in retained_index_set
            ]
            if (
                len(retained_indices) != len(retained_index_set)
                or retained_indices != expected_retained_order
            ):
                raise RuntimeError(
                    "station feedback batch filter may only remove contexts "
                    "without reordering retained contexts"
                )
        else:
            remaining = original_remaining
            feedback_batch_steps = []
            retained_index_set = {
                int(index) for index, _context in original_remaining
            }
        feedback_deferred_keys = {
            self._context_identity(context)
            for index, context in original_remaining
            if int(index) not in retained_index_set
        }
        reserved: set[int] = set()
        choices: Dict[int, int] = {}
        steps: list[dict[str, Any]] = list(feedback_batch_steps)
        rejected: dict[tuple[int, int, int], tuple[float, int]] = {}
        selected_keys: set[tuple[int, int, int]] = set()
        deferred_keys: set[tuple[int, int, int]] = set(
            feedback_deferred_keys
        )
        final_budget = self._psi_last_final_budget
        if final_budget is None:
            final_budget = len(idle_agents)
        final_budget = max(0, min(int(final_budget), len(contexts)))

        while remaining and len(choices) < final_budget:
            virtual_idle = tuple(
                agent for agent in idle_agents
                if int(agent.agent_id) not in reserved
            )
            if not virtual_idle:
                break
            rows = self._dynamic_rows(
                world_state,
                remaining,
                pending_counts,
                station_channels,
                debt_by_context,
                virtual_idle,
            )
            self.stats["psi_dispatch_unique_stations_sum"] += len({
                int(row["station_id"]) for row in rows
            })
            for row in rows:
                self.stats["psi_dispatch_service_sum"] += float(row["service"])
                self.stats["psi_dispatch_traffic_sum"] += float(row["traffic"])
                self.stats["psi_dispatch_backlog_sum"] += float(
                    row["backlog_score"]
                )
                self.stats["psi_dispatch_age_sum"] += float(row["age_score"])
                self.stats["psi_dispatch_debt_sum"] += float(
                    row["service_debt"]
                )
                self.stats["psi_dispatch_j_sum"] += float(row["j_score"])

            ranked = self._sort_rows(rows)
            chosen = ranked[0]
            context = chosen["context"]
            original_index = int(chosen["original_index"])
            key = self._row_key(chosen)
            station_id = int(chosen["station_id"])
            snapshot = station_snapshots.get(station_id)
            if snapshot is None:
                raise RuntimeError(
                    f"missing station snapshot for station {station_id}"
                )
            defer_clock = self._defer_clock_for_decision(
                key, float(chosen["free_flow_time"])
            )
            decision = station_context_defer_decision(
                snapshot,
                service=float(chosen["service"]),
                traffic=float(chosen["traffic"]),
                service_debt=float(chosen["service_debt"]),
                defer_mass=float(
                    self._station_context_defer_mass.get(key, 0.0)
                ),
                free_flow_time=defer_clock,
                virtual_pipeline_additions=int(
                    virtual_pipeline_additions.get(station_id, 0)
                ),
                risk_mode=self.station_context_defer_risk_mode,
            )
            station_feedback = self._station_context_feedback_for_row(
                world_state,
                chosen,
                snapshot,
            )
            feedback_defer = bool(
                station_feedback
                and station_feedback.get("defer_context", False)
            )
            self._record_decision_stats(decision, snapshot)
            eligible_ticks_before = int(
                self._station_context_defer_ticks.get(key, 0)
            )
            liveness_bound = int(
                self._station_context_defer_bound.get(
                    key, decision.worst_case_liveness_bound_ticks
                )
            )
            compact_ranked = [
                {
                    "context": self._row_text(row),
                    "station_id": int(row["station_id"]),
                    "dynamic_j": float(row["j_score"]),
                    "service_debt": float(row["service_debt"]),
                }
                for row in ranked
            ]

            remaining = [
                item for item in remaining
                if self._context_identity(item[1]) != key
            ]
            step_index = len(steps)
            if feedback_defer:
                deferred_keys.add(key)
                self.stats["station_context_defer_decisions"] += 1
                step = {
                    "step": int(step_index),
                    "status": "station_feedback_deferred_context",
                    "selected": self._row_text(chosen),
                    "selected_original_index": original_index,
                    "selected_station": station_id,
                    "dynamic_j": float(chosen["j_score"]),
                    "station_snapshot": snapshot.as_dict(),
                    "defer_decision": decision.as_dict(),
                    "eligible_defer_ticks_before": eligible_ticks_before,
                    "static_defer_free_flow_time": defer_clock,
                    "registered_liveness_bound_ticks": liveness_bound,
                    "ranked_contexts": compact_ranked,
                    "station_feedback": dict(station_feedback),
                }
                steps.append(step)
                # A station in BRAKE/LOCKED is not an eligible normal defer.
                # Its pipeline+phi liveness clock therefore pauses instead of
                # eventually overriding the station-local drain-only state.
                self._record_station_context_feedback_outcome(
                    station_feedback,
                    "station_feedback_deferred_context",
                )
                continue
            if not decision.execute:
                rejected.setdefault(
                    key,
                    (
                        defer_clock,
                        int(decision.worst_case_liveness_bound_ticks),
                    ),
                )
                deferred_keys.add(key)
                self.stats["station_context_defer_decisions"] += 1
                step = {
                    "step": int(step_index),
                    "status": "deferred_context",
                    "selected": self._row_text(chosen),
                    "selected_original_index": original_index,
                    "selected_station": station_id,
                    "dynamic_j": float(chosen["j_score"]),
                    "station_snapshot": snapshot.as_dict(),
                    "defer_decision": decision.as_dict(),
                    "eligible_defer_ticks_before": eligible_ticks_before,
                    "static_defer_free_flow_time": defer_clock,
                    "registered_liveness_bound_ticks": liveness_bound,
                    "ranked_contexts": compact_ranked,
                }
                if station_feedback is not None:
                    step["station_feedback"] = dict(station_feedback)
                steps.append(step)
                self._record_station_context_feedback_outcome(
                    station_feedback,
                    "deferred_context",
                )
                continue

            self._dynamic_reserved_robot_ids = set(reserved)
            try:
                with self._virtual_idle_view(world_state, virtual_idle):
                    one_choice = WorldModelTaskAssigner.select_robots(
                        self, world_state, [context]
                    )
            finally:
                self._dynamic_reserved_robot_ids = set()
            selected_robot = one_choice.get(0)
            status = "selected" if selected_robot is not None else "no_selection"
            if selected_robot is not None:
                selected_robot = int(selected_robot)
                choices[original_index] = selected_robot
                selected_keys.add(key)
                reserved.add(selected_robot)
                pending_counts[station_id] = max(
                    pending_counts.get(station_id, 0) - 1, 0
                )
                virtual_pipeline_additions[station_id] = (
                    virtual_pipeline_additions.get(station_id, 0) + 1
                )
                self._station_context_defer_mass.pop(key, None)
                self._station_context_defer_ticks.pop(key, None)
                self._station_context_defer_bound.pop(key, None)
                self._station_context_defer_clock.pop(key, None)
                self._station_context_defer_last_update_tick.pop(key, None)
                self.stats["station_context_defer_selected"] += 1
                self.stats["dynamic_probe_selected"] += 1
                self.stats["dynamic_probe_virtual_backlog_updates"] += 1
                if decision.debt_override:
                    self.stats[
                        "station_context_defer_debt_override_selected"
                    ] += 1
            else:
                self.stats["dynamic_probe_no_selection"] += 1

            step = {
                "step": int(step_index),
                "status": status,
                "selected": self._row_text(chosen),
                "selected_original_index": original_index,
                "selected_station": station_id,
                "selected_robot": selected_robot,
                "dynamic_j": float(chosen["j_score"]),
                "station_snapshot": snapshot.as_dict(),
                "defer_decision": decision.as_dict(),
                "eligible_defer_ticks_before": eligible_ticks_before,
                "static_defer_free_flow_time": defer_clock,
                "registered_liveness_bound_ticks": liveness_bound,
                "ranked_contexts": compact_ranked,
            }
            if station_feedback is not None:
                step["station_feedback"] = dict(station_feedback)
            steps.append(step)
            self._record_station_context_feedback_outcome(
                station_feedback,
                status,
            )

        current_tick = int(getattr(world_state, "tick", -1))
        for key, (free_flow_time, initial_bound) in rejected.items():
            if key in selected_keys:
                continue
            if (
                self._station_context_defer_last_update_tick.get(key)
                == current_tick
            ):
                self.stats[
                    "station_context_defer_duplicate_tick_updates_suppressed"
                ] += 1
                continue
            if key not in self._station_context_defer_clock:
                self._station_context_defer_clock[key] = float(
                    free_flow_time
                )
            frozen_clock = float(self._station_context_defer_clock[key])
            updated = min(
                float(self._station_context_defer_mass.get(key, 0.0))
                + eligible_defer_increment(frozen_clock),
                1.0,
            )
            self._station_context_defer_mass[key] = updated
            ticks = int(self._station_context_defer_ticks.get(key, 0)) + 1
            self._station_context_defer_ticks[key] = ticks
            self._station_context_defer_last_update_tick[key] = current_tick
            if key not in self._station_context_defer_bound:
                self._station_context_defer_bound[key] = max(
                    int(initial_bound), 0
                )
            bound = int(self._station_context_defer_bound[key])
            self.stats["station_context_defer_eligible_ticks_max"] = max(
                int(self.stats["station_context_defer_eligible_ticks_max"]),
                ticks,
            )
            if ticks > bound:
                self.stats[
                    "station_context_defer_liveness_bound_violations"
                ] += 1
            self.stats["station_context_defer_debt_updates"] += 1
            self.stats["station_context_defer_mass_max"] = max(
                float(self.stats["station_context_defer_mass_max"]), updated
            )

        if deferred_keys and not choices:
            self.stats["station_context_defer_all_remaining_batches"] += 1
        ordered_indices = [
            int(step["selected_original_index"])
            for step in steps if step.get("status") == "selected"
        ]
        original_keys = [self._context_identity(context) for context in contexts]
        ordered_keys = [original_keys[index] for index in ordered_indices]
        changed_positions = sum(
            int(left != right)
            for left, right in zip(original_keys, ordered_keys)
        )
        self.stats["station_context_defer_batches"] += 1
        self.stats["station_context_defer_steps"] += len(steps)
        self.stats["station_context_defer_unique_contexts"] += len(
            deferred_keys
        )
        self.stats["dynamic_probe_batches"] += 1
        self.stats["dynamic_probe_steps"] += len(steps)
        self.stats["psi_dispatch_contexts_seen"] += len(contexts)
        self.stats["dynamic_probe_order_changed_batches"] += int(
            changed_positions > 0
        )
        self.stats["dynamic_probe_changed_positions"] += changed_positions

        self._append_dynamic_trace({
            "schema_version": (
                self.station_context_defer_assigner_schema_version
            ),
            "station_context_defer_schema_version": (
                self.station_context_defer_schema_version
            ),
            "tick": int(getattr(world_state, "tick", -1)),
            "mode": self.station_context_defer_mode,
            "context_count": len(contexts),
            "final_budget": int(final_budget),
            "selected_count": len(choices),
            "deferred_context_count": len(deferred_keys),
            "station_pipeline_at_batch_start": {
                str(station_id): snapshot.as_dict()
                for station_id, snapshot in sorted(station_snapshots.items())
            },
            "virtual_pipeline_additions": {
                str(station_id): int(value)
                for station_id, value in sorted(
                    virtual_pipeline_additions.items()
                )
            },
            "steps": steps,
        })

        # Only selected contexts are materialised; deferred contexts remain in
        # the pending order state and return on a later tick.
        contexts[:] = [contexts[index] for index in ordered_indices]
        original_to_ordered = {
            original_index: ordered_index
            for ordered_index, original_index in enumerate(ordered_indices)
        }
        return {
            original_to_ordered[index]: robot_id
            for index, robot_id in choices.items()
            if index in original_to_ordered
        }

    def dynamic_probe_metrics(self) -> dict[str, Any]:
        metrics = super().dynamic_probe_metrics()
        evaluations = max(
            int(self.stats.get("station_context_defer_evaluations", 0)), 1
        )
        metrics.update({
            "psi_dispatch_schema_version": (
                self.station_context_defer_assigner_schema_version
            ),
            "psi_dispatch_mode": self.station_context_defer_mode,
            "station_context_defer_enabled": True,
            "station_context_defer_schema_version": (
                self.station_context_defer_schema_version
            ),
            "station_context_defer_risk_mode": (
                self.station_context_defer_risk_mode
            ),
            "station_context_defer_ready_contention_in_station_risk": (
                self.station_context_defer_risk_mode
                == STATION_CONTEXT_DEFER_RISK_READY_MAX_V1
            ),
            "station_context_defer_batches": int(
                self.stats.get("station_context_defer_batches", 0)
            ),
            "station_context_defer_steps": int(
                self.stats.get("station_context_defer_steps", 0)
            ),
            "station_context_defer_evaluations": int(
                self.stats.get("station_context_defer_evaluations", 0)
            ),
            "station_context_defer_decisions": int(
                self.stats.get("station_context_defer_decisions", 0)
            ),
            "station_context_defer_unique_contexts": int(
                self.stats.get("station_context_defer_unique_contexts", 0)
            ),
            "station_context_defer_all_remaining_batches": int(
                self.stats.get(
                    "station_context_defer_all_remaining_batches", 0
                )
            ),
            "station_context_defer_selected": int(
                self.stats.get("station_context_defer_selected", 0)
            ),
            "station_context_defer_debt_override_selected": int(
                self.stats.get(
                    "station_context_defer_debt_override_selected", 0
                )
            ),
            "station_context_defer_debt_updates": int(
                self.stats.get("station_context_defer_debt_updates", 0)
            ),
            "station_context_defer_duplicate_tick_updates_suppressed": int(
                self.stats.get(
                    "station_context_defer_duplicate_tick_updates_suppressed",
                    0,
                )
            ),
            "station_context_defer_rate": float(
                self.stats.get("station_context_defer_decisions", 0)
            ) / evaluations,
            "station_context_defer_risk_mean": float(
                self.stats.get("station_context_defer_risk_sum", 0.0)
            ) / evaluations,
            "station_context_defer_context_credit_mean": float(
                self.stats.get(
                    "station_context_defer_context_credit_sum", 0.0
                )
            ) / evaluations,
            "station_context_defer_pipeline_excess_mean": float(
                self.stats.get(
                    "station_context_defer_pipeline_excess_sum", 0.0
                )
            ) / evaluations,
            "station_context_defer_ready_contention_mean": float(
                self.stats.get(
                    "station_context_defer_ready_contention_sum", 0.0
                )
            ) / evaluations,
            "station_context_defer_ready_would_dominate_evaluations": int(
                self.stats.get(
                    "station_context_defer_ready_would_dominate_evaluations",
                    0,
                )
            ),
            "station_context_defer_phi_pressure_mean": float(
                self.stats.get(
                    "station_context_defer_phi_pressure_sum", 0.0
                )
            ) / evaluations,
            "station_context_defer_risk_max": float(
                self.stats.get("station_context_defer_risk_max", 0.0)
            ),
            "station_context_defer_positive_risk_evaluations": int(
                self.stats.get(
                    "station_context_defer_positive_risk_evaluations", 0
                )
            ),
            "station_context_defer_zero_risk_evaluations": int(
                self.stats.get(
                    "station_context_defer_zero_risk_evaluations", 0
                )
            ),
            "station_context_defer_mass_max": float(
                self.stats.get("station_context_defer_mass_max", 0.0)
            ),
            "station_context_defer_eligible_ticks_max": int(
                self.stats.get("station_context_defer_eligible_ticks_max", 0)
            ),
            "station_context_defer_liveness_bound_violations": int(
                self.stats.get(
                    "station_context_defer_liveness_bound_violations", 0
                )
            ),
            "station_context_defer_ledger_size_final": len(
                self._station_context_defer_mass
            ),
            "station_context_defer_clock_ledger_size_final": len(
                self._station_context_defer_clock
            ),
            "station_context_defer_update_tick_ledger_size_final": len(
                self._station_context_defer_last_update_tick
            ),
            "station_context_defer_ready_unadmitted_max": int(
                self.stats.get(
                    "station_context_defer_ready_unadmitted_max", 0
                )
            ),
            "station_context_defer_upstream_pipeline_max": int(
                self.stats.get(
                    "station_context_defer_upstream_pipeline_max", 0
                )
            ),
            "station_context_defer_context_conditioned": True,
            "station_context_defer_continues_other_stations": True,
            "station_context_defer_exact_eta_used": False,
            "station_context_defer_new_trainable_parameters": 0,
            "station_context_defer_liveness_clock": "1/static_free_flow_time",
            "station_context_defer_physical_safety_layer": (
                "committed_capacity_v1"
            ),
            "station_context_defer_fifo_v2_used": False,
            "psi_dispatch_no_assign_added": False,
            "psi_dispatch_hard_gate_added": False,
            "psi_dispatch_e_demand_modified": False,
        })
        return metrics


class StationContextDeferPipelinePhiDynamicPsiAssigner(
    StationContextDeferDynamicPsiAssigner
):
    """V2: ready contention is diagnostic, not an independent max risk."""

    station_context_defer_assigner_schema_version = (
        "phase_c_dynamic_j_station_context_defer_pipeline_phi_s1_v2"
    )
    station_context_defer_schema_version = (
        STATION_CONTEXT_DEFER_PIPELINE_PHI_SCHEMA_VERSION
    )
    station_context_defer_risk_mode = (
        STATION_CONTEXT_DEFER_RISK_PIPELINE_PHI_V2
    )
    station_context_defer_mode = (
        "dynamic_j_then_station_context_defer_pipeline_phi_then_s1"
    )


__all__ = [
    "STATION_CONTEXT_DEFER_ASSIGNER_SCHEMA_VERSION",
    "StationContextDeferDynamicPsiAssigner",
    "StationContextDeferPipelinePhiDynamicPsiAssigner",
]
