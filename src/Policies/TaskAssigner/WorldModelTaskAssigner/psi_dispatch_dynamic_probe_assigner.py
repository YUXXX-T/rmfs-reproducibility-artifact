"""Opt-in dynamic-J probe built on the frozen S1 robot scorer.

This module is deliberately separate from ``psi_dispatch_context_assigner``.
The existing Phase-C psi arm computes one static J order per assignment batch.
The probe below performs a *virtual backlog refresh* after each hypothetical
context selection and uses that refreshed order for the unchanged S1 scorer.

The probe is not yet a full physical-state dynamic controller: service and
traffic channels remain the station-head values at the beginning of the tick.
The default mode interleaves virtual J refresh with one-context calls to the
parent S1 scorer, while deferring physical task materialisation until the
batch ends.  The purpose is to test the specific stale-J mechanism identified
by the trace replay without changing the certified arm.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

from Policies.TaskAssigner.base_task_assigner import AssignmentContext
from Policies.TaskAssigner.WorldModelTaskAssigner.world_model_task_assigner import (
    WorldModelTaskAssigner,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_context_assigner import (
    PsiDispatchContextWorldModelTaskAssigner,
)
from WorldModel.core.psi_dispatch import (
    bounded_age_score,
    bounded_backlog_score,
    build_context_debt_features,
    context_key,
    dispatch_cost,
    estimate_free_flow_time,
)


DYNAMIC_PROBE_SCHEMA_VERSION = "phase_c_psi_dispatch_dynamic_probe_v1"


class DynamicPsiDispatchProbeAssigner(
    PsiDispatchContextWorldModelTaskAssigner
):
    """Dynamic virtual-debt ordering with the unchanged S1 scorer.

    ``psi_context_mode='shadow'`` is used internally so the inherited
    initialisation loads the frozen station head but never applies the static
    reorder implementation.  In the default interleaved mode, each selected
    context is sent separately to ``WorldModelTaskAssigner.select_robots``;
    its robot is reserved in a virtual idle set before the next J calculation.
    Actual task objects are still committed once at the end of the inherited
    assignment call.
    """

    dynamic_probe_schema_version = DYNAMIC_PROBE_SCHEMA_VERSION

    def __init__(
        self,
        *,
        psi_head_checkpoint: str,
        psi_scale_contract: str,
        dynamic_trace_enabled: bool = True,
        dynamic_trace_max_records: int = 5000,
        dynamic_interleaved: bool = True,
        **kwargs,
    ):
        # Shadow mode loads/validates the frozen head and keeps the inherited
        # proposal path's complete context superset, but does not mutate the
        # caller's context list.
        super().__init__(
            psi_head_checkpoint=psi_head_checkpoint,
            psi_scale_contract=psi_scale_contract,
            psi_context_mode="shadow",
            psi_trace_enabled=False,
            psi_trace_max_records=0,
            **kwargs,
        )
        self.dynamic_trace_enabled = bool(dynamic_trace_enabled)
        self.dynamic_interleaved = bool(dynamic_interleaved)
        self.dynamic_trace_max_records = max(
            0, int(dynamic_trace_max_records)
        )
        self.dynamic_probe_trace_records: list[dict[str, Any]] = []
        self._dynamic_reserved_robot_ids: set[int] = set()
        self.stats.update({
            "dynamic_probe_batches": 0,
            "dynamic_probe_steps": 0,
            "dynamic_probe_selected": 0,
            "dynamic_probe_no_selection": 0,
            "dynamic_probe_order_changed_batches": 0,
            "dynamic_probe_changed_positions": 0,
            "dynamic_probe_virtual_backlog_updates": 0,
            "dynamic_probe_trace_dropped": 0,
            "dynamic_probe_initial_j_mismatch": 0,
        })

    @staticmethod
    def _row_key(row: dict[str, Any]) -> tuple[int, int, int]:
        return (
            int(row["order_id"]),
            int(row["pod_id"]),
            int(row["station_id"]),
        )

    @staticmethod
    def _row_text(row: dict[str, Any]) -> str:
        return (
            f"o{int(row['order_id'])}/p{int(row['pod_id'])}/"
            f"s{int(row['station_id'])}"
        )

    def _append_dynamic_trace(self, record: dict[str, Any]) -> None:
        if not self.dynamic_trace_enabled:
            return
        if len(self.dynamic_probe_trace_records) >= (
            self.dynamic_trace_max_records
        ):
            self.stats["dynamic_probe_trace_dropped"] += 1
            return
        self.dynamic_probe_trace_records.append(record)

    @staticmethod
    def _initial_pending_counts(debt_by_context: Dict) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for debt in debt_by_context.values():
            station = int(debt.station_id)
            counts[station] = max(
                counts.get(station, 0), int(debt.unserved_chain_count)
            )
        return counts

    def _dynamic_rows(
        self,
        world_state,
        remaining: Sequence[Tuple[int, AssignmentContext]],
        pending_counts: Dict[int, int],
        station_channels: Dict[int, Dict[str, float]],
        debt_by_context: Dict,
        virtual_idle_agents: Sequence,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for original_index, context in remaining:
            key = context_key(context)
            base = debt_by_context[key]
            station = int(context.station_id)
            capacity = max(float(base.station_capacity), 1.0)
            backlog_score = bounded_backlog_score(
                pending_counts.get(station, 0), capacity
            )
            free_flow = estimate_free_flow_time(
                context,
                virtual_idle_agents,
                self._psi_distance,
                self._psi_durations,
            )
            age_score = bounded_age_score(
                base.order_age_ticks, free_flow
            )
            service_debt = 0.5 * (backlog_score + age_score)
            channels = station_channels[station]
            j_score = dispatch_cost(
                channels["service"],
                channels["traffic"],
                service_debt,
            )
            rows.append({
                "original_index": int(original_index),
                "order_id": int(context.order_id),
                "pod_id": int(context.pod_id),
                "station_id": station,
                "service": float(channels["service"]),
                "traffic": float(channels["traffic"]),
                "unserved_chain_count": int(
                    pending_counts.get(station, 0)
                ),
                "station_capacity": capacity,
                "backlog_score": float(backlog_score),
                "order_age_ticks": int(base.order_age_ticks),
                "free_flow_time": float(free_flow),
                "age_score": float(age_score),
                "service_debt": float(service_debt),
                "j_score": float(j_score),
                "context": context,
            })
        return rows

    @staticmethod
    def _sort_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: (
                float(row["j_score"]),
                -float(row["service_debt"]),
                -float(row["age_score"]),
                int(row["original_index"]),
            ),
        )

    def _build_dynamic_order(
        self,
        world_state,
        contexts: List[AssignmentContext],
        station_channels: Dict[int, Dict[str, float]],
    ) -> tuple[
        List[AssignmentContext],
        List[int],
        dict[str, Any],
    ]:
        idle_agents = tuple(world_state.get_idle_agents())
        debt_by_context = build_context_debt_features(
            world_state,
            contexts,
            idle_agents,
            self._psi_distance,
            self._psi_durations,
        )
        pending_counts = self._initial_pending_counts(debt_by_context)
        remaining: list[tuple[int, AssignmentContext]] = list(
            enumerate(contexts)
        )
        selected_rows: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        final_budget = self._psi_last_final_budget
        if final_budget is None:
            final_budget = len(idle_agents)
        final_budget = max(0, min(int(final_budget), len(contexts)))

        # The first probe updates the exact pending/debt terms and recomputes
        # free-flow/age against the same virtual idle set.  Robot selection is
        # still done by S1 after this order is built.  This isolates the stale-J
        # mechanism before adding a physical transition update.
        virtual_idle_agents = idle_agents
        step_index = 0
        while remaining and len(choices) < final_budget:
            rows = self._dynamic_rows(
                world_state,
                remaining,
                pending_counts,
                station_channels,
                debt_by_context,
                virtual_idle_agents,
            )
            ranked = self._sort_rows(rows)
            chosen = ranked[0]
            selected_rows.append(chosen)
            selected_context = chosen["context"]
            station = int(chosen["station_id"])
            static_j = None
            # ``j_score`` from the initial debt construction is not stored in
            # the debt dataclass.  Recompute it from the initial pending count
            # for an auditable before/after comparison.
            initial_backlog = bounded_backlog_score(
                debt_by_context[context_key(selected_context)]
                .unserved_chain_count,
                debt_by_context[context_key(selected_context)]
                .station_capacity,
            )
            initial_debt = 0.5 * (
                initial_backlog
                + debt_by_context[context_key(selected_context)].age_score
            )
            static_j = dispatch_cost(
                station_channels[station]["service"],
                station_channels[station]["traffic"],
                initial_debt,
            )
            steps.append({
                "step": int(step_index),
                "selected": self._row_text(chosen),
                "selected_original_index": int(chosen["original_index"]),
                "selected_station": station,
                "dynamic_j": float(chosen["j_score"]),
                "initial_static_j": float(static_j),
                "dynamic_service_debt": float(chosen["service_debt"]),
                "pending_before": {
                    str(key): int(value)
                    for key, value in sorted(pending_counts.items())
                },
                "ranked_contexts": [
                    self._row_text(row) for row in ranked
                ],
                "ranked_j": [float(row["j_score"]) for row in ranked],
            })
            pending_counts[station] = max(
                pending_counts.get(station, 0) - 1, 0
            )
            self.stats["dynamic_probe_virtual_backlog_updates"] += 1
            chosen_key = self._row_key(chosen)
            removed = False
            next_remaining = []
            for item in remaining:
                if not removed and self._row_key({
                    "order_id": item[1].order_id,
                    "pod_id": item[1].pod_id,
                    "station_id": item[1].station_id,
                }) == chosen_key:
                    removed = True
                    continue
                next_remaining.append(item)
            remaining = next_remaining

        # The parent scorer consumes a list.  Put the dynamically selected
        # prefix first; append untouched contexts in their original order so
        # the budget-exhausted suffix remains deterministic.
        selected_keys = {self._row_key(row) for row in selected_rows}
        selected_contexts = [row["context"] for row in selected_rows]
        selected_indices = [int(row["original_index"]) for row in selected_rows]
        suffix_indices = [
            index
            for index, context in enumerate(contexts)
            if self._row_key({
                "order_id": context.order_id,
                "pod_id": context.pod_id,
                "station_id": context.station_id,
            }) not in selected_keys
        ]
        original_indices = selected_indices + suffix_indices
        ordered_contexts = [contexts[index] for index in original_indices]
        original_keys = [
            self._row_key({
                "order_id": context.order_id,
                "pod_id": context.pod_id,
                "station_id": context.station_id,
            })
            for context in contexts
        ]
        ordered_keys = [
            self._row_key({
                "order_id": context.order_id,
                "pod_id": context.pod_id,
                "station_id": context.station_id,
            })
            for context in ordered_contexts
        ]
        changed_positions = sum(
            int(left != right)
            for left, right in zip(original_keys, ordered_keys)
        )
        self.stats["dynamic_probe_batches"] += 1
        self.stats["dynamic_probe_steps"] += len(steps)
        self.stats["psi_dispatch_contexts_seen"] += len(contexts)
        self.stats["dynamic_probe_order_changed_batches"] += int(
            changed_positions > 0
        )
        self.stats["dynamic_probe_changed_positions"] += changed_positions
        if steps:
            initial_rows = self._dynamic_rows(
                world_state,
                list(enumerate(contexts)),
                self._initial_pending_counts(debt_by_context),
                station_channels,
                debt_by_context,
                idle_agents,
            )
            if any(
                abs(
                    float(row["j_score"])
                    - dispatch_cost(
                        row["service"],
                        row["traffic"],
                        debt_by_context[
                            context_key(contexts[int(row["original_index"])])
                        ].service_debt,
                    )
                ) > 1e-6
                for row in initial_rows
            ):
                self.stats["dynamic_probe_initial_j_mismatch"] += 1

        trace = {
            "schema_version": DYNAMIC_PROBE_SCHEMA_VERSION,
            "tick": int(getattr(world_state, "tick", -1)),
            "mode": "j_dynamic_backlog_probe",
            "context_count": len(contexts),
            "final_budget": int(final_budget),
            "changed_positions": int(changed_positions),
            "original_order": [
                self._row_text({
                    "order_id": context.order_id,
                    "pod_id": context.pod_id,
                    "station_id": context.station_id,
                })
                for context in contexts
            ],
            "dynamic_order": [
                self._row_text({
                    "order_id": context.order_id,
                    "pod_id": context.pod_id,
                    "station_id": context.station_id,
                })
                for context in ordered_contexts
            ],
            "station_channels": station_channels,
            "steps": steps,
            "selected_rows": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "context"
                }
                for row in selected_rows
            ],
        }
        self._append_dynamic_trace(trace)
        return ordered_contexts, original_indices, trace

    def _select_robot_candidates(
        self,
        world,
        ctx: AssignmentContext,
        available: List,
        lyapunov_snapshot,
    ):
        """Exclude robots reserved by earlier dynamic steps.

        The parent S1 scorer remains responsible for ranking the filtered
        candidate list.  This hook is active only while the interleaved probe
        calls the parent on one context; normal/static arms never instantiate
        this class.
        """

        filtered = [
            agent
            for agent in available
            if int(agent.agent_id) not in self._dynamic_reserved_robot_ids
        ]
        return super()._select_robot_candidates(
            world, ctx, filtered, lyapunov_snapshot
        )

    @contextmanager
    def _virtual_idle_view(self, world_state, virtual_idle_agents):
        """Expose the virtual idle set to the one-context S1 call.

        Agent status fields are not mutated.  Only the read-only accessor is
        temporarily replaced, so the simulator's actual state remains intact
        until the inherited batch commit happens.
        """

        original = world_state.get_idle_agents
        had_instance_attribute = "get_idle_agents" in vars(world_state)
        setattr(
            world_state,
            "get_idle_agents",
            lambda: list(virtual_idle_agents),
        )
        try:
            yield
        finally:
            if had_instance_attribute:
                setattr(world_state, "get_idle_agents", original)
            else:
                delattr(world_state, "get_idle_agents")

    def _select_robots_interleaved(
        self,
        world_state,
        contexts: List[AssignmentContext],
        station_channels: Dict[int, Dict[str, float]],
    ) -> Dict[int, int]:
        """Select J-min context, then S1 robot, repeatedly.

        The world state is not committed between steps.  Pending debt and the
        idle accessor are maintained as a virtual state; the actual tasks are
        materialised once by ``assign()`` after this method returns.
        """

        idle_agents = tuple(world_state.get_idle_agents())
        debt_by_context = build_context_debt_features(
            world_state,
            contexts,
            idle_agents,
            self._psi_distance,
            self._psi_durations,
        )
        pending_counts = self._initial_pending_counts(debt_by_context)
        remaining: list[tuple[int, AssignmentContext]] = list(
            enumerate(contexts)
        )
        reserved: set[int] = set()
        choices: Dict[int, int] = {}
        steps: list[dict[str, Any]] = []
        final_budget = self._psi_last_final_budget
        if final_budget is None:
            final_budget = len(idle_agents)
        final_budget = max(0, min(int(final_budget), len(contexts)))

        for step_index in range(final_budget):
            if not remaining:
                break
            virtual_idle = tuple(
                agent
                for agent in idle_agents
                if int(agent.agent_id) not in reserved
            )
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
                self.stats["psi_dispatch_service_sum"] += float(
                    row["service"]
                )
                self.stats["psi_dispatch_traffic_sum"] += float(
                    row["traffic"]
                )
                self.stats["psi_dispatch_backlog_sum"] += float(
                    row["backlog_score"]
                )
                self.stats["psi_dispatch_age_sum"] += float(
                    row["age_score"]
                )
                self.stats["psi_dispatch_debt_sum"] += float(
                    row["service_debt"]
                )
                self.stats["psi_dispatch_j_sum"] += float(row["j_score"])
            ranked = self._sort_rows(rows)
            chosen = ranked[0]
            chosen_context = chosen["context"]
            chosen_original_index = int(chosen["original_index"])
            chosen_key = self._row_key(chosen)
            station = int(chosen["station_id"])
            pending_before = {
                str(key): int(value)
                for key, value in sorted(pending_counts.items())
            }
            available_before = [int(agent.agent_id) for agent in virtual_idle]

            self._dynamic_reserved_robot_ids = set(reserved)
            try:
                with self._virtual_idle_view(world_state, virtual_idle):
                    one_choice = WorldModelTaskAssigner.select_robots(
                        self,
                        world_state,
                        [chosen_context],
                    )
            finally:
                self._dynamic_reserved_robot_ids = set()

            selected_robot = one_choice.get(0)
            status = "selected" if selected_robot is not None else "no_selection"
            if selected_robot is not None:
                selected_robot = int(selected_robot)
                choices[chosen_original_index] = selected_robot
                reserved.add(selected_robot)
                pending_counts[station] = max(
                    pending_counts.get(station, 0) - 1, 0
                )
                self.stats["dynamic_probe_selected"] += 1
                self.stats["dynamic_probe_virtual_backlog_updates"] += 1
            else:
                self.stats["dynamic_probe_no_selection"] += 1

            steps.append({
                "step": int(step_index),
                "selected": self._row_text(chosen),
                "selected_original_index": chosen_original_index,
                "selected_station": station,
                "selected_robot": selected_robot,
                "status": status,
                "dynamic_j": float(chosen["j_score"]),
                "dynamic_service_debt": float(chosen["service_debt"]),
                "pending_before": pending_before,
                "available_robot_ids_before": available_before,
                "ranked_contexts": [
                    self._row_text(row) for row in ranked
                ],
                "ranked_j": [float(row["j_score"]) for row in ranked],
            })

            removed = False
            next_remaining: list[tuple[int, AssignmentContext]] = []
            for item in remaining:
                item_key = (
                    int(item[1].order_id),
                    int(item[1].pod_id),
                    int(item[1].station_id),
                )
                if not removed and item_key == chosen_key:
                    removed = True
                    continue
                next_remaining.append(item)
            remaining = next_remaining
            step_index += 1

        selected_keys = {
            (
                int(contexts[index].order_id),
                int(contexts[index].pod_id),
                int(contexts[index].station_id),
            )
            for index in choices
        }
        ordered_indices = [
            int(step["selected_original_index"])
            for step in steps
            if step["status"] == "selected"
        ]
        ordered_indices.extend(
            index
            for index, context in enumerate(contexts)
            if (
                int(context.order_id),
                int(context.pod_id),
                int(context.station_id),
            ) not in selected_keys
        )
        ordered_keys = [
            (
                int(contexts[index].order_id),
                int(contexts[index].pod_id),
                int(contexts[index].station_id),
            )
            for index in ordered_indices
        ]
        original_keys = [
            (
                int(context.order_id),
                int(context.pod_id),
                int(context.station_id),
            )
            for context in contexts
        ]
        changed_positions = sum(
            int(left != right)
            for left, right in zip(original_keys, ordered_keys)
        )
        self.stats["dynamic_probe_batches"] += 1
        self.stats["dynamic_probe_steps"] += len(steps)
        self.stats["psi_dispatch_contexts_seen"] += len(contexts)
        self.stats["dynamic_probe_order_changed_batches"] += int(
            changed_positions > 0
        )
        self.stats["dynamic_probe_changed_positions"] += changed_positions

        self._append_dynamic_trace({
            "schema_version": DYNAMIC_PROBE_SCHEMA_VERSION,
            "tick": int(getattr(world_state, "tick", -1)),
            "mode": "j_dynamic_interleaved_probe",
            "context_count": len(contexts),
            "final_budget": int(final_budget),
            "changed_positions": int(changed_positions),
            "original_order": [
                self._row_text({
                    "order_id": context.order_id,
                    "pod_id": context.pod_id,
                    "station_id": context.station_id,
                })
                for context in contexts
            ],
            "dynamic_order": [
                self._row_text({
                    "order_id": contexts[index].order_id,
                    "pod_id": contexts[index].pod_id,
                    "station_id": contexts[index].station_id,
                })
                for index in ordered_indices
            ],
            "station_channels": station_channels,
            "steps": steps,
            "selected_robot_by_context": {
                self._row_text({
                    "order_id": contexts[index].order_id,
                    "pod_id": contexts[index].pod_id,
                    "station_id": contexts[index].station_id,
                }): int(robot)
                for index, robot in choices.items()
            },
        })
        # The commit helper consumes the list passed by ``assign``.  Reorder
        # that list explicitly and re-index the robot choices to the reordered
        # list, so task materialisation follows the same dynamic sequence as
        # the S1 calls above.  This avoids the exact original-index ambiguity
        # that motivated this probe.
        ordered_contexts = [contexts[index] for index in ordered_indices]
        original_to_ordered = {
            original_index: ordered_index
            for ordered_index, original_index in enumerate(ordered_indices)
        }
        contexts[:] = ordered_contexts
        ordered_choices = {
            original_to_ordered[original_index]: robot
            for original_index, robot in choices.items()
            if original_index in original_to_ordered
        }
        return ordered_choices

    def select_robots(
        self,
        world_state,
        contexts: List[AssignmentContext],
    ) -> Dict[int, int]:
        if not contexts:
            return {}
        if self._psi_distance is None or self._psi_durations is None:
            raise RuntimeError("dynamic psi probe geometry was not initialised")

        station_channels = self._evaluate_station_channels(world_state)
        if self.dynamic_interleaved:
            return self._select_robots_interleaved(
                world_state, contexts, station_channels
            )
        ordered_contexts, original_indices, trace = self._build_dynamic_order(
            world_state, contexts, station_channels
        )

        # Deliberately bypass the static psi wrapper and call the unchanged S1
        # implementation.  The interleaved path reorders the caller's list and
        # re-indexes choices before commit_assignments runs.
        ordered_choices = WorldModelTaskAssigner.select_robots(
            self, world_state, ordered_contexts
        )
        choices: Dict[int, int] = {}
        selected_by_original: Dict[str, int] = {}
        for ordered_index, robot_id in ordered_choices.items():
            if 0 <= int(ordered_index) < len(original_indices):
                original_index = original_indices[int(ordered_index)]
                choices[int(original_index)] = int(robot_id)
                context = contexts[int(original_index)]
                selected_by_original[self._row_text({
                    "order_id": context.order_id,
                    "pod_id": context.pod_id,
                    "station_id": context.station_id,
                })] = int(robot_id)
        self.stats["dynamic_probe_selected"] += len(choices)
        self.stats["dynamic_probe_no_selection"] += max(
            0, len(ordered_choices) - len(choices)
        )
        trace["selected_robot_by_context"] = selected_by_original
        return choices

    def dynamic_probe_metrics(self) -> dict[str, Any]:
        metrics = super().psi_dispatch_metrics()
        mode = (
            "j_dynamic_interleaved_probe"
            if self.dynamic_interleaved
            else "j_dynamic_backlog_probe"
        )
        metrics.update({
            "psi_dispatch_schema_version": DYNAMIC_PROBE_SCHEMA_VERSION,
            "psi_dispatch_mode": mode,
            "psi_dispatch_dynamic_probe": True,
            "dynamic_probe_batches": int(
                self.stats.get("dynamic_probe_batches", 0)
            ),
            "dynamic_probe_steps": int(
                self.stats.get("dynamic_probe_steps", 0)
            ),
            "dynamic_probe_selected": int(
                self.stats.get("dynamic_probe_selected", 0)
            ),
            "dynamic_probe_order_changed_batches": int(
                self.stats.get("dynamic_probe_order_changed_batches", 0)
            ),
            "dynamic_probe_changed_positions": int(
                self.stats.get("dynamic_probe_changed_positions", 0)
            ),
            "dynamic_probe_virtual_backlog_updates": int(
                self.stats.get("dynamic_probe_virtual_backlog_updates", 0)
            ),
            "dynamic_probe_initial_j_mismatch": int(
                self.stats.get("dynamic_probe_initial_j_mismatch", 0)
            ),
            "dynamic_probe_trace_records": len(
                self.dynamic_probe_trace_records
            ),
            "dynamic_probe_trace_dropped": int(
                self.stats.get("dynamic_probe_trace_dropped", 0)
            ),
            "dynamic_probe_choices_reindexed": bool(
                self.dynamic_interleaved
            ),
            "psi_dispatch_robot_scorer": (
                "WorldModelTaskAssigner.select_robots_unmodified"
            ),
            "psi_dispatch_dynamic_update": (
                "virtual_pending_and_idle_candidates"
                if self.dynamic_interleaved
                else "virtual_pending_only"
            ),
            "psi_dispatch_no_assign_added": False,
            "psi_dispatch_hard_gate_added": False,
            "psi_dispatch_e_demand_modified": False,
            "psi_dispatch_attention_added": False,
        })
        return metrics


__all__ = [
    "DYNAMIC_PROBE_SCHEMA_VERSION",
    "DynamicPsiDispatchProbeAssigner",
]
