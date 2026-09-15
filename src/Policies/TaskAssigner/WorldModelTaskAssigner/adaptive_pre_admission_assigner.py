"""Learned dynamic-J + S1 with context-conditioned pre-admission.

This is an additive experimental arm.  It leaves the certified World Model,
station-phi head, learned context-J head, S1 robot scorer, and FIFO-V2 engine
contract unchanged.  The only new decision is made before PICK tasks are
materialised::

    adaptive admissibility filters contexts
        -> learned J selects the next context
        -> unchanged S1 selects that context's robot
        -> virtual pipeline mass is updated and remaining contexts are redone

Deferring one station never prevents an admissible context for another
station from being considered.  When every remaining context is deferred,
the current assignment batch ends without creating a task.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from Policies.TaskAssigner.base_task_assigner import (
    AssignmentContext,
    fulfilled_pod_ids_for_order,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.context_j_learned_dynamic_assigner import (
    LearnedContextJDynamicAssigner,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.world_model_task_assigner import (
    WorldModelTaskAssigner,
)
from WorldModel.core.adaptive_pre_admission import (
    ADAPTIVE_PRE_ADMISSION_SCHEMA_VERSION,
    adaptive_pre_admission_decision,
    build_station_pipeline_snapshots,
    eligible_defer_increment,
)
from WorldModel.core.psi_dispatch import (
    build_context_debt_features,
    context_key,
)
from WorldState.agent_state import AgentStatus
from WorldState.order_state import OrderStatus
from WorldState.task_state import TaskStatus


ADAPTIVE_ASSIGNER_SCHEMA_VERSION = (
    "phase_c_learned_j_adaptive_pre_admission_v1"
)


class AdaptivePreAdmissionLearnedContextJAssigner(
    LearnedContextJDynamicAssigner
):
    """Adaptive station pipeline admissibility around the frozen J+S1 arm."""

    adaptive_assigner_schema_version = ADAPTIVE_ASSIGNER_SCHEMA_VERSION

    def __init__(self, **kwargs) -> None:
        # Context-keyed, dimensionless eligible-defer mass.  It is deliberately
        # separate from the World Model state and does not alter encoder input.
        self._pre_admission_defer_mass: dict[tuple[int, int, int], float] = {}
        super().__init__(**kwargs)
        self.stats.update({
            "pre_admission_batches": 0,
            "pre_admission_steps": 0,
            "pre_admission_context_evaluations": 0,
            "pre_admission_contexts_deferred_unique": 0,
            "pre_admission_all_deferred_batches": 0,
            "pre_admission_selected": 0,
            "pre_admission_debt_override_selected": 0,
            "pre_admission_defer_debt_updates": 0,
            "pre_admission_pipeline_mass_sum": 0.0,
            "pre_admission_pipeline_mass_after_sum": 0.0,
            "pre_admission_effective_limit_sum": 0.0,
            "pre_admission_base_headroom_sum": 0.0,
            "pre_admission_defer_credit_sum": 0.0,
            "pre_admission_pipeline_mass_max": 0.0,
            "pre_admission_waiting_assigned_max": 0,
            "pre_admission_defer_mass_max": 0.0,
            "pre_admission_trace_dropped": 0,
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
        for key in list(self._pre_admission_defer_mass):
            order_id, pod_id, _station_id = key
            order = world_state.order_state.orders.get(order_id)
            if (
                order is None
                or order.status != OrderStatus.PENDING
                or (order_id, pod_id) in active_chain_keys
                or pod_id in fulfilled_pod_ids_for_order(world_state, order)
            ):
                self._pre_admission_defer_mass.pop(key, None)

    def _append_pre_admission_trace(self, record: dict[str, Any]) -> None:
        if not self.dynamic_trace_enabled:
            return
        if len(self.dynamic_probe_trace_records) >= self.dynamic_trace_max_records:
            self.stats["pre_admission_trace_dropped"] += 1
            return
        self.dynamic_probe_trace_records.append(record)

    def _annotate_admission_rows(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        pipeline_snapshots,
        virtual_pipeline_additions: dict[int, int],
        global_waiting_ratio: float,
    ) -> list[dict[str, Any]]:
        annotated: list[dict[str, Any]] = []
        for original in rows:
            row = dict(original)
            station_id = int(row["station_id"])
            snapshot = pipeline_snapshots.get(station_id)
            if snapshot is None:
                raise RuntimeError(
                    f"missing station pipeline snapshot for station {station_id}"
                )
            key = self._row_key(row)
            decision = adaptive_pre_admission_decision(
                snapshot,
                traffic=float(row["traffic"]),
                global_waiting_ratio=float(global_waiting_ratio),
                defer_mass=float(
                    self._pre_admission_defer_mass.get(key, 0.0)
                ),
                virtual_pipeline_additions=int(
                    virtual_pipeline_additions.get(station_id, 0)
                ),
            )
            row["pre_admission"] = decision.as_dict()
            row["pre_admission_execute"] = bool(decision.execute)
            row["pre_admission_debt_override"] = bool(
                decision.debt_override
            )
            annotated.append(row)

            self.stats["pre_admission_context_evaluations"] += 1
            self.stats["pre_admission_pipeline_mass_sum"] += float(
                decision.current_pipeline_mass
            )
            self.stats["pre_admission_pipeline_mass_after_sum"] += float(
                decision.pipeline_mass_after_execute
            )
            self.stats["pre_admission_effective_limit_sum"] += float(
                decision.effective_pipeline_limit
            )
            self.stats["pre_admission_base_headroom_sum"] += float(
                decision.base_headroom
            )
            self.stats["pre_admission_defer_credit_sum"] += float(
                decision.defer_credit
            )
            self.stats["pre_admission_pipeline_mass_max"] = max(
                float(self.stats["pre_admission_pipeline_mass_max"]),
                float(decision.current_pipeline_mass),
            )
            self.stats["pre_admission_waiting_assigned_max"] = max(
                int(self.stats["pre_admission_waiting_assigned_max"]),
                int(snapshot.waiting_assigned),
            )
        return annotated

    def _select_robots_interleaved(
        self,
        world_state,
        contexts: List[AssignmentContext],
        station_channels: Dict[int, Dict[str, float]],
    ) -> Dict[int, int]:
        """Filter EXECUTE/DEFER, then perform dynamic J + unchanged S1."""

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
        pipeline_snapshots = build_station_pipeline_snapshots(world_state)
        virtual_pipeline_additions = {
            int(station_id): 0 for station_id in pipeline_snapshots
        }
        waiting_count = sum(
            int(agent.status == AgentStatus.WAITING_ASSIGNED)
            for agent in world_state.agents
        )
        global_waiting_ratio = waiting_count / max(len(world_state.agents), 1)

        remaining: list[tuple[int, AssignmentContext]] = list(
            enumerate(contexts)
        )
        reserved: set[int] = set()
        choices: Dict[int, int] = {}
        steps: list[dict[str, Any]] = []
        rejected: dict[tuple[int, int, int], float] = {}
        selected_keys: set[tuple[int, int, int]] = set()
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
            base_rows = self._dynamic_rows(
                world_state,
                remaining,
                pending_counts,
                station_channels,
                debt_by_context,
                virtual_idle,
            )
            rows = self._annotate_admission_rows(
                base_rows,
                pipeline_snapshots=pipeline_snapshots,
                virtual_pipeline_additions=virtual_pipeline_additions,
                global_waiting_ratio=global_waiting_ratio,
            )
            ranked = self._sort_rows(rows)
            admissible = [
                row for row in ranked if row["pre_admission_execute"]
            ]
            for row in ranked:
                if row["pre_admission_execute"]:
                    continue
                rejected.setdefault(
                    self._row_key(row), max(float(row["free_flow_time"]), 1.0)
                )

            compact_ranked = [
                {
                    "context": self._row_text(row),
                    "station_id": int(row["station_id"]),
                    "learned_j": float(row["j_score"]),
                    "execute": bool(row["pre_admission_execute"]),
                    "debt_override": bool(
                        row["pre_admission_debt_override"]
                    ),
                    "pipeline_after": float(
                        row["pre_admission"][
                            "pipeline_mass_after_execute"
                        ]
                    ),
                    "pipeline_limit": float(
                        row["pre_admission"][
                            "effective_pipeline_limit"
                        ]
                    ),
                    "traffic": float(row["traffic"]),
                    "defer_credit": float(
                        row["pre_admission"]["defer_credit"]
                    ),
                }
                for row in ranked
            ]
            if not admissible:
                steps.append({
                    "step": int(step_index),
                    "status": "all_remaining_contexts_deferred",
                    "ranked_contexts": compact_ranked,
                })
                self.stats["pre_admission_all_deferred_batches"] += 1
                break

            chosen = admissible[0]
            chosen_context = chosen["context"]
            chosen_original_index = int(chosen["original_index"])
            chosen_key = self._row_key(chosen)
            station_id = int(chosen["station_id"])

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
                selected_keys.add(chosen_key)
                reserved.add(selected_robot)
                pending_counts[station_id] = max(
                    pending_counts.get(station_id, 0) - 1, 0
                )
                virtual_pipeline_additions[station_id] = (
                    virtual_pipeline_additions.get(station_id, 0) + 1
                )
                self._pre_admission_defer_mass.pop(chosen_key, None)
                self.stats["pre_admission_selected"] += 1
                self.stats["dynamic_probe_selected"] += 1
                self.stats["dynamic_probe_virtual_backlog_updates"] += 1
                if chosen["pre_admission_debt_override"]:
                    self.stats["pre_admission_debt_override_selected"] += 1
            else:
                self.stats["dynamic_probe_no_selection"] += 1

            steps.append({
                "step": int(step_index),
                "status": status,
                "selected": self._row_text(chosen),
                "selected_original_index": chosen_original_index,
                "selected_station": station_id,
                "selected_robot": selected_robot,
                "learned_j": float(chosen["j_score"]),
                "pre_admission": dict(chosen["pre_admission"]),
                "ranked_contexts": compact_ranked,
            })

            remaining = [
                item
                for item in remaining
                if self._context_identity(item[1]) != chosen_key
            ]

        # Debt grows once per eligible tick, never once per inner-loop step.
        for key, free_flow_time in rejected.items():
            if key in selected_keys:
                continue
            updated = min(
                self._pre_admission_defer_mass.get(key, 0.0)
                + eligible_defer_increment(free_flow_time),
                1.0,
            )
            self._pre_admission_defer_mass[key] = float(updated)
            self.stats["pre_admission_defer_debt_updates"] += 1
            self.stats["pre_admission_defer_mass_max"] = max(
                float(self.stats["pre_admission_defer_mass_max"]),
                float(updated),
            )

        ordered_indices = [
            int(step["selected_original_index"])
            for step in steps
            if step.get("status") == "selected"
        ]
        original_keys = [self._context_identity(context) for context in contexts]
        ordered_keys = [original_keys[index] for index in ordered_indices]
        changed_positions = sum(
            int(left != right)
            for left, right in zip(original_keys, ordered_keys)
        )
        self.stats["pre_admission_batches"] += 1
        self.stats["pre_admission_steps"] += len(steps)
        self.stats["pre_admission_contexts_deferred_unique"] += len(rejected)
        self.stats["dynamic_probe_batches"] += 1
        self.stats["dynamic_probe_steps"] += len(steps)
        self.stats["psi_dispatch_contexts_seen"] += len(contexts)
        self.stats["dynamic_probe_order_changed_batches"] += int(
            changed_positions > 0
        )
        self.stats["dynamic_probe_changed_positions"] += changed_positions

        self._append_pre_admission_trace({
            "schema_version": ADAPTIVE_ASSIGNER_SCHEMA_VERSION,
            "pre_admission_schema_version": (
                ADAPTIVE_PRE_ADMISSION_SCHEMA_VERSION
            ),
            "tick": int(getattr(world_state, "tick", -1)),
            "mode": "adaptive_pre_admission_then_learned_j_then_s1",
            "context_count": len(contexts),
            "final_budget": int(final_budget),
            "global_waiting_ratio": float(global_waiting_ratio),
            "pipeline_at_batch_start": {
                str(station_id): snapshot.as_dict()
                for station_id, snapshot in sorted(
                    pipeline_snapshots.items()
                )
            },
            "virtual_pipeline_additions": {
                str(key): int(value)
                for key, value in sorted(virtual_pipeline_additions.items())
            },
            "selected_count": len(choices),
            "deferred_context_count": len(rejected),
            "steps": steps,
        })

        # Only EXECUTE contexts reach the inherited materialisation helper.
        contexts[:] = [contexts[index] for index in ordered_indices]
        original_to_ordered = {
            original_index: ordered_index
            for ordered_index, original_index in enumerate(ordered_indices)
        }
        return {
            original_to_ordered[original_index]: robot_id
            for original_index, robot_id in choices.items()
            if original_index in original_to_ordered
        }

    def dynamic_probe_metrics(self) -> dict[str, Any]:
        metrics = super().dynamic_probe_metrics()
        evaluations = max(
            int(self.stats.get("pre_admission_context_evaluations", 0)), 1
        )
        metrics.update({
            "psi_dispatch_schema_version": ADAPTIVE_ASSIGNER_SCHEMA_VERSION,
            "psi_dispatch_mode": (
                "adaptive_pre_admission_then_learned_j_then_s1"
            ),
            "adaptive_pre_admission_enabled": True,
            "adaptive_pre_admission_schema_version": (
                ADAPTIVE_PRE_ADMISSION_SCHEMA_VERSION
            ),
            "pre_admission_batches": int(
                self.stats.get("pre_admission_batches", 0)
            ),
            "pre_admission_steps": int(
                self.stats.get("pre_admission_steps", 0)
            ),
            "pre_admission_context_evaluations": int(
                self.stats.get("pre_admission_context_evaluations", 0)
            ),
            "pre_admission_contexts_deferred_unique": int(
                self.stats.get("pre_admission_contexts_deferred_unique", 0)
            ),
            "pre_admission_all_deferred_batches": int(
                self.stats.get("pre_admission_all_deferred_batches", 0)
            ),
            "pre_admission_selected": int(
                self.stats.get("pre_admission_selected", 0)
            ),
            "pre_admission_debt_override_selected": int(
                self.stats.get("pre_admission_debt_override_selected", 0)
            ),
            "pre_admission_defer_debt_updates": int(
                self.stats.get("pre_admission_defer_debt_updates", 0)
            ),
            "pre_admission_pipeline_mass_mean": float(
                self.stats.get("pre_admission_pipeline_mass_sum", 0.0)
            ) / evaluations,
            "pre_admission_pipeline_mass_after_mean": float(
                self.stats.get("pre_admission_pipeline_mass_after_sum", 0.0)
            ) / evaluations,
            "pre_admission_effective_limit_mean": float(
                self.stats.get("pre_admission_effective_limit_sum", 0.0)
            ) / evaluations,
            "pre_admission_base_headroom_mean": float(
                self.stats.get("pre_admission_base_headroom_sum", 0.0)
            ) / evaluations,
            "pre_admission_defer_credit_mean": float(
                self.stats.get("pre_admission_defer_credit_sum", 0.0)
            ) / evaluations,
            "pre_admission_pipeline_mass_max": float(
                self.stats.get("pre_admission_pipeline_mass_max", 0.0)
            ),
            "pre_admission_waiting_assigned_max": int(
                self.stats.get("pre_admission_waiting_assigned_max", 0)
            ),
            "pre_admission_defer_mass_max": float(
                self.stats.get("pre_admission_defer_mass_max", 0.0)
            ),
            "pre_admission_defer_ledger_size_final": len(
                self._pre_admission_defer_mass
            ),
            "pre_admission_trace_dropped": int(
                self.stats.get("pre_admission_trace_dropped", 0)
            ),
            "pre_admission_pipeline_contract": (
                "one unit per physical, committed-in-transit, "
                "moving-to-pod, or WAITING_ASSIGNED station-bound chain"
            ),
            "pre_admission_exact_eta_used": False,
            "pre_admission_service_channel_role": (
                "learned J context ordering only; exact pipeline occupancy "
                "prevents double-counting service pressure in admission"
            ),
            "pre_admission_fifo_v2_role": "physical_safety_fallback",
            "pre_admission_context_conditioned": True,
            "pre_admission_continue_other_contexts": True,
            "context_j_no_assign_added": False,
            "context_defer_added": True,
            "context_j_hard_gate_added": True,
            "context_j_e_demand_modified": False,
        })
        return metrics


__all__ = [
    "ADAPTIVE_ASSIGNER_SCHEMA_VERSION",
    "AdaptivePreAdmissionLearnedContextJAssigner",
]
