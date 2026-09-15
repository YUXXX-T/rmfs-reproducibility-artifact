"""Opt-in station-feedback closure around Dynamic-J + pipeline+phi + S1.

The frozen World Model, station head, Dynamic-J formula, pipeline+phi V2 risk,
and S1 robot scorer remain unchanged.  This isolated subclass adds only a
station-local feedback hook before a context is materialised:

* ``off``: no tracker is created and the parent assigner is unchanged;
* ``shadow_v1``: station states and hypothetical defers are audited only;
* ``active_v1``: contexts targeting a station in ``BRAKE`` or ``LOCKED`` are
  removed once per station from the current batch before Dynamic-J, while
  contexts for healthy stations continue with unchanged debt features.
* ``shadow_v2`` / ``active_v2``: the same audit/filter contract is retained,
  but the controller may enter a reversible BRAKE earlier when sustained
  committed pressure is corroborated by multiple station-flow signals.
* ``shadow_v3`` / ``active_v3``: the filter remains unchanged, while release
  drought is replaced by service-due overdue time after the normal two-phase
  exit handoff grace.

The controller never edits orders, station admission tokens, paths, or tasks
that have already been materialised.  A feedback defer is not eligible for the
pipeline+phi liveness ledger, preventing defer debt from forcing execution
while the station remains drain-only.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Optional

from Policies.TaskAssigner.WorldModelTaskAssigner.station_context_defer_assigner import (
    StationContextDeferPipelinePhiDynamicPsiAssigner,
)
from WorldModel.core.station_feedback_control import (
    STATION_FEEDBACK_MODE_OFF,
    STATION_FEEDBACK_SUPPORTED_MODES,
    STATION_FEEDBACK_SCHEMA_VERSION,
    StationFeedbackConfig,
    StationFeedbackController,
    StationFeedbackSnapshot,
    build_station_feedback_observations,
    station_feedback_mode_is_active,
    station_feedback_mode_is_shadow,
)


STATION_FEEDBACK_ASSIGNER_SCHEMA_VERSION = (
    "phase_c_dynamic_j_pipeline_phi_station_feedback_s1_v3"
)
STATION_FEEDBACK_BATCH_FILTER_SCHEMA_VERSION = (
    "phase_c_station_feedback_batch_filter_v1"
)


class StationFeedbackClosedLoopPipelinePhiDynamicPsiAssigner(
    StationContextDeferPipelinePhiDynamicPsiAssigner
):
    """Dynamic-J context dispatch with an explicit station feedback mode."""

    station_feedback_batch_filter_hook_enabled = True
    station_feedback_assigner_schema_version = (
        STATION_FEEDBACK_ASSIGNER_SCHEMA_VERSION
    )

    def __init__(
        self,
        *,
        station_feedback_mode: str = STATION_FEEDBACK_MODE_OFF,
        station_feedback_config: Optional[
            StationFeedbackConfig | Mapping[str, Any]
        ] = None,
        station_feedback_service_ticks: int = 5,
        station_feedback_trace_max_records: int = 0,
        **kwargs,
    ) -> None:
        mode = str(station_feedback_mode)
        if mode not in STATION_FEEDBACK_SUPPORTED_MODES:
            raise ValueError(
                "station_feedback_mode must be one of: "
                + ", ".join(STATION_FEEDBACK_SUPPORTED_MODES)
            )
        if int(station_feedback_trace_max_records) < 0:
            raise ValueError(
                "station_feedback_trace_max_records must be non-negative"
            )
        if station_feedback_config is None:
            config = StationFeedbackConfig.for_service_ticks(
                station_feedback_service_ticks
            )
        elif isinstance(station_feedback_config, StationFeedbackConfig):
            config = station_feedback_config
        else:
            values = StationFeedbackConfig.for_service_ticks(
                station_feedback_service_ticks
            ).as_dict()
            values.update(dict(station_feedback_config))
            config = StationFeedbackConfig(**values)

        super().__init__(**kwargs)
        self._station_feedback_mode = mode
        self._station_feedback_config = config
        self._station_feedback_controller = (
            None
            if mode == STATION_FEEDBACK_MODE_OFF
            else StationFeedbackController(config)
        )
        self._station_feedback_current: dict[
            int, StationFeedbackSnapshot
        ] = {}
        self._station_feedback_last_prepared_tick: Optional[int] = None
        self._station_feedback_state_ticks: Counter[str] = Counter()
        self._station_feedback_transitions: Counter[str] = Counter()
        self._station_feedback_reason_ticks: Counter[str] = Counter()
        self._station_feedback_batch_filter_tick: Optional[int] = None
        self._station_feedback_batch_filter_stations_this_tick: set[int] = set()
        self._station_feedback_batch_filter_contexts_this_tick: set[
            tuple[int, int, int]
        ] = set()
        self._station_feedback_batch_filter_unique_contexts: set[
            tuple[int, int, int]
        ] = set()
        self.station_feedback_trace_max_records = int(
            station_feedback_trace_max_records
        )
        self.station_feedback_trace_records: list[dict[str, Any]] = []
        self.stats.update({
            "station_feedback_batches": 0,
            "station_feedback_station_observations": 0,
            "station_feedback_context_evaluations": 0,
            "station_feedback_would_defer_evaluations": 0,
            "station_feedback_forced_defers": 0,
            "station_feedback_selected_contexts": 0,
            "station_feedback_contexts_materialized_in_defer_state": 0,
            "station_feedback_off_batches": 0,
            "station_feedback_shadow_batches": 0,
            "station_feedback_active_batches": 0,
            "station_feedback_committed_pressure_station_ticks": 0,
            "station_feedback_early_brake_condition_station_ticks": 0,
            "station_feedback_service_release_due_station_ticks": 0,
            "station_feedback_release_overdue_station_ticks": 0,
            "station_feedback_release_overdue_tick_sum": 0,
            "station_feedback_due_exit_blocked_station_ticks": 0,
            "station_feedback_trace_dropped": 0,
            "station_feedback_batch_filter_batches": 0,
            "station_feedback_batch_filter_contexts_scanned": 0,
            "station_feedback_batch_filter_would_suppress_batches": 0,
            "station_feedback_batch_filter_would_suppress_station_ticks": 0,
            "station_feedback_batch_filter_would_suppress_context_ticks": 0,
            "station_feedback_batch_filter_suppressed_batches": 0,
            "station_feedback_batch_filter_suppressed_station_ticks": 0,
            "station_feedback_batch_filter_suppressed_context_ticks": 0,
            "station_feedback_batch_filter_duplicate_station_tick_records_suppressed": 0,
            "station_feedback_batch_filter_duplicate_context_tick_records_suppressed": 0,
        })

    @property
    def station_feedback_mode(self) -> str:
        return str(self._station_feedback_mode)

    def _prepare_station_context_feedback(
        self,
        world_state,
        station_snapshots,
    ) -> None:
        tick = int(getattr(world_state, "tick", -1))
        if self._station_feedback_last_prepared_tick == tick:
            return
        if self._station_feedback_mode == STATION_FEEDBACK_MODE_OFF:
            self._station_feedback_current = {}
            self._station_feedback_last_prepared_tick = tick
            self.stats["station_feedback_off_batches"] += 1
            return
        controller = self._station_feedback_controller
        if controller is None:
            raise RuntimeError("enabled station feedback lacks a controller")
        observations = build_station_feedback_observations(
            world_state,
            station_snapshots,
            service_ticks=self._station_feedback_config.service_ticks,
        )
        current = controller.update(observations)
        self._station_feedback_current = dict(current)
        self._station_feedback_last_prepared_tick = tick
        self.stats["station_feedback_batches"] += 1
        self.stats["station_feedback_station_observations"] += len(current)
        if station_feedback_mode_is_shadow(self._station_feedback_mode):
            self.stats["station_feedback_shadow_batches"] += 1
        elif station_feedback_mode_is_active(self._station_feedback_mode):
            self.stats["station_feedback_active_batches"] += 1
        for snapshot in current.values():
            self._station_feedback_state_ticks[str(snapshot.state)] += 1
            self._station_feedback_reason_ticks[
                str(snapshot.dominant_reason)
            ] += 1
            self.stats[
                "station_feedback_committed_pressure_station_ticks"
            ] += int(snapshot.committed_pressure)
            self.stats[
                "station_feedback_early_brake_condition_station_ticks"
            ] += int(snapshot.early_brake_condition)
            self.stats[
                "station_feedback_service_release_due_station_ticks"
            ] += int(snapshot.service_release_due)
            self.stats[
                "station_feedback_release_overdue_station_ticks"
            ] += int(snapshot.release_overdue_ticks > 0)
            self.stats[
                "station_feedback_release_overdue_tick_sum"
            ] += int(snapshot.release_overdue_ticks)
            self.stats[
                "station_feedback_due_exit_blocked_station_ticks"
            ] += int(snapshot.due_exit_blocked)
            if snapshot.transition is not None:
                self._station_feedback_transitions[
                    str(snapshot.transition)
                ] += 1
                if (
                    len(self.station_feedback_trace_records)
                    < self.station_feedback_trace_max_records
                ):
                    self.station_feedback_trace_records.append(
                        snapshot.as_dict()
                    )
                else:
                    self.stats["station_feedback_trace_dropped"] += 1

    def _station_context_feedback_for_row(
        self,
        world_state,
        chosen,
        snapshot,
    ) -> dict[str, Any] | None:
        if self._station_feedback_mode == STATION_FEEDBACK_MODE_OFF:
            return None
        station_id = int(chosen["station_id"])
        feedback = self._station_feedback_current.get(station_id)
        if feedback is None:
            raise RuntimeError(
                f"missing station feedback snapshot for station {station_id}"
            )
        would_defer = bool(feedback.defer_new_context)
        defer_context = bool(
            would_defer
            and station_feedback_mode_is_active(self._station_feedback_mode)
        )
        self.stats["station_feedback_context_evaluations"] += 1
        self.stats["station_feedback_would_defer_evaluations"] += int(
            would_defer
        )
        result = feedback.as_dict()
        result.update({
            "mode": str(self._station_feedback_mode),
            "would_defer": would_defer,
            "defer_context": defer_context,
        })
        return result

    def _station_context_feedback_batch_filter(
        self,
        world_state,
        remaining,
        station_snapshots,
    ):
        """Filter all BRAKE/LOCKED contexts once per station and tick.

        Shadow mode performs the same station-local audit without changing
        the candidate list.  Active mode removes blocked-station contexts
        before Dynamic-J, while the parent has already built debt features
        and pending counts from the full batch.  This preserves healthy-
        station scores and avoids repeatedly ranking and deferring each
        blocked context.
        """

        if self._station_feedback_mode == STATION_FEEDBACK_MODE_OFF:
            return list(remaining), []

        self.stats["station_feedback_batch_filter_batches"] += 1
        self.stats["station_feedback_batch_filter_contexts_scanned"] += len(
            remaining
        )
        blocked_by_station: dict[int, list[tuple[int, Any]]] = {}
        for original_index, context in remaining:
            station_id = int(context.station_id)
            feedback = self._station_feedback_current.get(station_id)
            if feedback is None:
                raise RuntimeError(
                    "missing station feedback snapshot for station "
                    f"{station_id}"
                )
            if bool(feedback.defer_new_context):
                blocked_by_station.setdefault(station_id, []).append(
                    (int(original_index), context)
                )

        if not blocked_by_station:
            return list(remaining), []

        self.stats[
            "station_feedback_batch_filter_would_suppress_batches"
        ] += 1
        tick = int(getattr(world_state, "tick", -1))
        if self._station_feedback_batch_filter_tick != tick:
            self._station_feedback_batch_filter_tick = tick
            self._station_feedback_batch_filter_stations_this_tick.clear()
            self._station_feedback_batch_filter_contexts_this_tick.clear()

        new_station_ids: set[int] = set()
        new_context_keys: set[tuple[int, int, int]] = set()
        duplicate_station_records = 0
        duplicate_context_records = 0
        for station_id, items in blocked_by_station.items():
            if (
                station_id
                in self._station_feedback_batch_filter_stations_this_tick
            ):
                duplicate_station_records += 1
            else:
                self._station_feedback_batch_filter_stations_this_tick.add(
                    station_id
                )
                new_station_ids.add(station_id)
            for _original_index, context in items:
                key = self._context_identity(context)
                self._station_feedback_batch_filter_unique_contexts.add(key)
                if (
                    key
                    in self._station_feedback_batch_filter_contexts_this_tick
                ):
                    duplicate_context_records += 1
                else:
                    self._station_feedback_batch_filter_contexts_this_tick.add(
                        key
                    )
                    new_context_keys.add(key)

        self.stats[
            "station_feedback_batch_filter_would_suppress_station_ticks"
        ] += len(new_station_ids)
        self.stats[
            "station_feedback_batch_filter_would_suppress_context_ticks"
        ] += len(new_context_keys)
        self.stats[
            "station_feedback_batch_filter_duplicate_station_tick_records_suppressed"
        ] += duplicate_station_records
        self.stats[
            "station_feedback_batch_filter_duplicate_context_tick_records_suppressed"
        ] += duplicate_context_records

        if station_feedback_mode_is_shadow(self._station_feedback_mode):
            return list(remaining), []
        if not station_feedback_mode_is_active(self._station_feedback_mode):
            raise RuntimeError(
                "unexpected station feedback mode in batch filter: "
                f"{self._station_feedback_mode}"
            )

        self.stats["station_feedback_batch_filter_suppressed_batches"] += 1
        self.stats[
            "station_feedback_batch_filter_suppressed_station_ticks"
        ] += len(new_station_ids)
        self.stats[
            "station_feedback_batch_filter_suppressed_context_ticks"
        ] += len(new_context_keys)
        self.stats["station_feedback_context_evaluations"] += len(
            new_context_keys
        )
        self.stats["station_feedback_would_defer_evaluations"] += len(
            new_context_keys
        )
        self.stats["station_feedback_forced_defers"] += len(new_context_keys)
        self.stats["station_context_defer_decisions"] += len(new_context_keys)

        blocked_station_ids = set(blocked_by_station)
        retained = [
            (int(original_index), context)
            for original_index, context in remaining
            if int(context.station_id) not in blocked_station_ids
        ]
        steps = []
        for step_index, station_id in enumerate(sorted(new_station_ids)):
            feedback = self._station_feedback_current[station_id]
            items = blocked_by_station[station_id]
            first_context = items[0][1]
            feedback_payload = feedback.as_dict()
            feedback_payload.update({
                "mode": str(self._station_feedback_mode),
                "would_defer": True,
                "defer_context": True,
            })
            station_snapshot = station_snapshots.get(station_id)
            if station_snapshot is None:
                raise RuntimeError(
                    "missing station snapshot for station "
                    f"{station_id}"
                )
            steps.append({
                "step": int(step_index),
                "status": "station_feedback_suppressed_station",
                "selected_station": int(station_id),
                "suppressed_context_count": len(items),
                "first_suppressed_context": {
                    "order_id": int(first_context.order_id),
                    "pod_id": int(first_context.pod_id),
                    "station_id": int(first_context.station_id),
                },
                "station_snapshot": station_snapshot.as_dict(),
                "station_feedback": feedback_payload,
            })
        return retained, steps

    def _record_station_context_feedback_outcome(
        self,
        feedback: dict[str, Any] | None,
        status: str,
    ) -> None:
        if feedback is None:
            return
        if status == "station_feedback_deferred_context":
            self.stats["station_feedback_forced_defers"] += 1
            return
        if status == "selected":
            self.stats["station_feedback_selected_contexts"] += 1
            if bool(feedback.get("would_defer", False)):
                self.stats[
                    "station_feedback_contexts_materialized_in_defer_state"
                ] += 1

    def dynamic_probe_metrics(self) -> dict[str, Any]:
        metrics = super().dynamic_probe_metrics()
        evaluations = max(
            int(self.stats.get("station_feedback_context_evaluations", 0)),
            1,
        )
        metrics.update({
            "psi_dispatch_schema_version": (
                self.station_feedback_assigner_schema_version
            ),
            "psi_dispatch_mode": (
                "station_feedback_batch_filter_then_dynamic_j_then_"
                "pipeline_phi_then_s1"
                if station_feedback_mode_is_active(
                    self._station_feedback_mode
                )
                else "dynamic_j_then_pipeline_phi_then_station_feedback_"
                "then_s1"
            ),
            "station_feedback_enabled": (
                self._station_feedback_mode != STATION_FEEDBACK_MODE_OFF
            ),
            "station_feedback_mode": str(self._station_feedback_mode),
            "station_feedback_schema_version": (
                self._station_feedback_config.schema_version
            ),
            "station_feedback_config": self._station_feedback_config.as_dict(),
            "station_feedback_batches": int(
                self.stats.get("station_feedback_batches", 0)
            ),
            "station_feedback_station_observations": int(
                self.stats.get("station_feedback_station_observations", 0)
            ),
            "station_feedback_context_evaluations": int(
                self.stats.get("station_feedback_context_evaluations", 0)
            ),
            "station_feedback_would_defer_evaluations": int(
                self.stats.get(
                    "station_feedback_would_defer_evaluations", 0
                )
            ),
            "station_feedback_would_defer_rate": float(
                self.stats.get(
                    "station_feedback_would_defer_evaluations", 0
                )
            ) / evaluations,
            "station_feedback_forced_defers": int(
                self.stats.get("station_feedback_forced_defers", 0)
            ),
            "station_feedback_selected_contexts": int(
                self.stats.get("station_feedback_selected_contexts", 0)
            ),
            "station_feedback_contexts_materialized_in_defer_state": int(
                self.stats.get(
                    "station_feedback_contexts_materialized_in_defer_state",
                    0,
                )
            ),
            "station_feedback_state_station_ticks": {
                key: int(value)
                for key, value in sorted(
                    self._station_feedback_state_ticks.items()
                )
            },
            "station_feedback_transitions": {
                key: int(value)
                for key, value in sorted(
                    self._station_feedback_transitions.items()
                )
            },
            "station_feedback_dominant_reason_station_ticks": {
                key: int(value)
                for key, value in sorted(
                    self._station_feedback_reason_ticks.items()
                )
            },
            "station_feedback_committed_pressure_station_ticks": int(
                self.stats.get(
                    "station_feedback_committed_pressure_station_ticks", 0
                )
            ),
            "station_feedback_early_brake_condition_station_ticks": int(
                self.stats.get(
                    "station_feedback_early_brake_condition_station_ticks", 0
                )
            ),
            "station_feedback_current_states": {
                str(station_id): snapshot.as_dict()
                for station_id, snapshot in sorted(
                    self._station_feedback_current.items()
                )
            },
            "station_feedback_transition_trace_record_count": len(
                self.station_feedback_trace_records
            ),
            "station_feedback_transition_trace_dropped": int(
                self.stats.get("station_feedback_trace_dropped", 0)
            ),
            "station_feedback_batch_filter_schema_version": (
                STATION_FEEDBACK_BATCH_FILTER_SCHEMA_VERSION
            ),
            "station_feedback_batch_filter_enabled": (
                self._station_feedback_mode != STATION_FEEDBACK_MODE_OFF
            ),
            "station_feedback_batch_filter_active": (
                station_feedback_mode_is_active(self._station_feedback_mode)
            ),
            "station_feedback_batch_filter_batches": int(
                self.stats.get("station_feedback_batch_filter_batches", 0)
            ),
            "station_feedback_batch_filter_contexts_scanned": int(
                self.stats.get(
                    "station_feedback_batch_filter_contexts_scanned", 0
                )
            ),
            "station_feedback_batch_filter_would_suppress_batches": int(
                self.stats.get(
                    "station_feedback_batch_filter_would_suppress_batches",
                    0,
                )
            ),
            "station_feedback_batch_filter_would_suppress_station_ticks": int(
                self.stats.get(
                    "station_feedback_batch_filter_would_suppress_station_ticks",
                    0,
                )
            ),
            "station_feedback_batch_filter_would_suppress_context_ticks": int(
                self.stats.get(
                    "station_feedback_batch_filter_would_suppress_context_ticks",
                    0,
                )
            ),
            "station_feedback_batch_filter_suppressed_batches": int(
                self.stats.get(
                    "station_feedback_batch_filter_suppressed_batches", 0
                )
            ),
            "station_feedback_batch_filter_suppressed_station_ticks": int(
                self.stats.get(
                    "station_feedback_batch_filter_suppressed_station_ticks",
                    0,
                )
            ),
            "station_feedback_batch_filter_suppressed_context_ticks": int(
                self.stats.get(
                    "station_feedback_batch_filter_suppressed_context_ticks",
                    0,
                )
            ),
            "station_feedback_batch_filter_unique_contexts": len(
                self._station_feedback_batch_filter_unique_contexts
            ),
            "station_feedback_batch_filter_duplicate_station_tick_records_suppressed": int(
                self.stats.get(
                    "station_feedback_batch_filter_duplicate_station_tick_records_suppressed",
                    0,
                )
            ),
            "station_feedback_batch_filter_duplicate_context_tick_records_suppressed": int(
                self.stats.get(
                    "station_feedback_batch_filter_duplicate_context_tick_records_suppressed",
                    0,
                )
            ),
            "station_feedback_batch_filter_before_dynamic_j": (
                station_feedback_mode_is_active(self._station_feedback_mode)
            ),
            "station_feedback_batch_filter_preserves_full_batch_debt_features": True,
            "station_feedback_batch_filter_station_tick_deduplicated": True,
            "station_feedback_station_local": True,
            "station_feedback_order_generator_modified": False,
            "station_feedback_task_lifecycle_modified": False,
            "station_feedback_station_admission_modified": False,
            "station_feedback_path_planning_modified": False,
            "station_feedback_s1_modified": False,
            "station_feedback_world_model_modified": False,
            "station_feedback_new_trainable_parameters": 0,
            "station_feedback_admission_contract_owned_by_engine": True,
            "station_context_defer_physical_safety_layer": (
                "external_station_admission_contract"
            ),
            "station_feedback_decision_override_enabled": (
                station_feedback_mode_is_active(self._station_feedback_mode)
            ),
            "station_feedback_liveness_clock_paused_while_deferred": (
                station_feedback_mode_is_active(self._station_feedback_mode)
            ),
            "station_feedback_early_brake_enabled": bool(
                self._station_feedback_config.early_brake_enabled
            ),
            "station_feedback_observation_clock": (
                "assignment ticks with dispatchable contexts"
            ),
            "station_feedback_observation_gap_policy": (
                "never infer an unobserved consecutive streak"
            ),
        })
        if (
            self._station_feedback_config.schema_version
            != STATION_FEEDBACK_SCHEMA_VERSION
        ):
            metrics.update({
                "station_feedback_release_signal": str(
                    self._station_feedback_config.release_signal
                ),
                "station_feedback_early_brake_rule": str(
                    self._station_feedback_config.early_brake_rule
                ),
                "station_feedback_service_release_due_station_ticks": int(
                    self.stats.get(
                        "station_feedback_service_release_due_station_ticks",
                        0,
                    )
                ),
                "station_feedback_release_overdue_station_ticks": int(
                    self.stats.get(
                        "station_feedback_release_overdue_station_ticks", 0
                    )
                ),
                "station_feedback_release_overdue_tick_sum": int(
                    self.stats.get(
                        "station_feedback_release_overdue_tick_sum", 0
                    )
                ),
                "station_feedback_due_exit_blocked_station_ticks": int(
                    self.stats.get(
                        "station_feedback_due_exit_blocked_station_ticks", 0
                    )
                ),
            })
        return metrics


__all__ = [
    "STATION_FEEDBACK_ASSIGNER_SCHEMA_VERSION",
    "STATION_FEEDBACK_BATCH_FILTER_SCHEMA_VERSION",
    "StationFeedbackClosedLoopPipelinePhiDynamicPsiAssigner",
]
