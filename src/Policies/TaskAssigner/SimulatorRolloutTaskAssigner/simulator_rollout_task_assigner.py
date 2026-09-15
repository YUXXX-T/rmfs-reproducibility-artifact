"""Direct H-step simulator oracle for fixed-context robot ranking.

This policy deliberately contains no learned model.  For every fixed
``(order, pod, station)`` context it evaluates each currently legal idle
robot by cloning the live world, force-applying that one candidate, and
running the repository's counterfactual simulator for ``H`` isolated ticks.
The robot with the lowest realized label cost is selected.

The semantics match the Phase-C Round-1 supervision target:

* context order is the original prefix/J0 order;
* all current idle robots are evaluated by default;
* no future orders are generated in a rollout;
* no continuation task assigner is invoked;
* all selected task chains are committed only after candidate scoring.

It is therefore a candidate-ranking oracle for the learned World Model, not
a claim of globally optimal RMFS control or full closed-loop MPC.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Set

from Policies.TaskAssigner.base_task_assigner import (
    AssignmentContext,
    BaseTaskAssigner,
)
from Policies.TaskAssigner.context_assignment import (
    commit_fixed_context_assignments,
    propose_fixed_assignment_contexts,
)
from WorldModel.data.counterfactual_rollout import evaluate_candidate_rollout
from WorldModel.data.candidate_generator import ASSIGN_ROBOT_ACTION_TYPE
from WorldState.task_state import Task


def _manhattan(a, b) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


class SimulatorRolloutTaskAssigner(BaseTaskAssigner):
    """Rank robots with exact isolated counterfactual simulation."""

    requires_external_fixed_context = True

    def __init__(
        self,
        *,
        rollout_horizon: int = 10,
        candidate_limit: int = 0,
        max_contexts_per_tick: int = 0,
        reservation_window: int = 1,
        decision_trace_enabled: bool = False,
        decision_trace_max_records: int = 2000,
    ):
        super().__init__()
        if int(rollout_horizon) <= 0:
            raise ValueError("rollout_horizon must be positive")
        if int(candidate_limit) < 0:
            raise ValueError("candidate_limit must be non-negative")
        if int(max_contexts_per_tick) < 0:
            raise ValueError("max_contexts_per_tick must be non-negative")
        if int(reservation_window) <= 0:
            raise ValueError("reservation_window must be positive")

        self.rollout_horizon = int(rollout_horizon)
        self.candidate_limit = int(candidate_limit)
        self.max_contexts_per_tick = int(max_contexts_per_tick)
        self.reservation_window = int(reservation_window)
        self.decision_trace_enabled = bool(decision_trace_enabled)
        self.decision_trace_max_records = max(
            0, int(decision_trace_max_records)
        )
        self.decision_trace_records: List[dict] = []

        self.path_planner = None
        self._initialized = False
        self._node_map = None
        self._local_capacity = None
        self._bottleneck_score = None
        self._adj = None

        self.stats = {
            "assign_calls": 0,
            "model_assign_calls": 0,
            "fallback_greedy_calls": 0,
            "warmup_defer_calls": 0,
            "model_inference_calls": 0,
            "model_inference_time_total_ms": 0.0,
            "assignment_time_total_ms": 0.0,
            "oracle_contexts_scored": 0,
            "oracle_contexts_selected": 0,
            "oracle_contexts_committed": 0,
            "oracle_contexts_without_candidate": 0,
            "oracle_rollout_calls": 0,
            "oracle_rollout_time_total_ms": 0.0,
            "oracle_complete_rollouts": 0,
            "oracle_incomplete_rollouts": 0,
            "oracle_selected_cost_sum": 0.0,
            "oracle_selected_cost_min": float("inf"),
            "oracle_selected_cost_max": float("-inf"),
            "oracle_selected_nearest_count": 0,
            "oracle_exact_tie_contexts": 0,
            "oracle_selected_vertex_conflicts": 0,
            "oracle_selected_swap_conflicts": 0,
            "oracle_selected_blocked_moves": 0,
            "oracle_trace_dropped": 0,
            "all_idle_candidate_contexts": 0,
            "all_idle_candidates_scored": 0,
        }

    def _init(self, world_state) -> None:
        if self.path_planner is None:
            raise RuntimeError(
                "SimulatorRolloutTaskAssigner requires the engine path planner"
            )
        from WorldModel.graph_builder import build_static_graph

        (
            _edge_index,
            self._node_map,
            _inv_node_map,
            self._local_capacity,
            self._bottleneck_score,
            _node_type,
            self._adj,
        ) = build_static_graph(world_state.map_state)
        self._initialized = True

    def propose_assignment_contexts(
        self,
        world_state,
        max_contexts: Optional[int] = None,
    ) -> List[AssignmentContext]:
        return propose_fixed_assignment_contexts(
            self,
            world_state,
            max_contexts=max_contexts,
        )

    def commit_assignments(
        self,
        world_state,
        contexts: List[AssignmentContext],
        robot_choices: Dict[int, int],
    ) -> List[Task]:
        return commit_fixed_context_assignments(
            world_state,
            contexts,
            robot_choices,
        )

    @staticmethod
    def _fixed_context(context: AssignmentContext) -> dict:
        return {
            "order_id": int(context.order_id),
            "pod_id": int(context.pod_id),
            "pod_location": tuple(context.pod_location),
            "station_id": int(context.station_id),
            "station_location": tuple(context.station_location),
            "entry_position": (
                tuple(context.entry_position)
                if context.entry_position is not None else None
            ),
            "exit_position": (
                tuple(context.exit_position)
                if context.exit_position is not None else None
            ),
            "return_location": tuple(context.return_location),
            "task_type": "PICK",
            "order_size": int(context.order_size),
        }

    def _candidate(self, agent) -> dict:
        scope = (
            "all_idle"
            if self.candidate_limit == 0
            else f"nearest_{self.candidate_limit}"
        )
        return {
            "action_type": ASSIGN_ROBOT_ACTION_TYPE,
            "robot_id": int(agent.agent_id),
            "robot_start": tuple(agent.position),
            "candidate_policy": f"simulator_rollout_{scope}",
            "candidate_selection_mode": scope,
            "chosen": False,
        }

    @staticmethod
    def _complete_rollout(result: dict, horizon: int) -> bool:
        mask = result.get("future_mask")
        if mask is None or int(mask.numel()) != int(horizon):
            return False
        return bool((mask > 0.5).all().item())

    def _append_trace(self, row: dict) -> None:
        if not self.decision_trace_enabled:
            return
        if len(self.decision_trace_records) >= self.decision_trace_max_records:
            self.stats["oracle_trace_dropped"] += 1
            return
        self.decision_trace_records.append(row)

    def select_robots(
        self,
        world_state,
        contexts: List[AssignmentContext],
    ) -> Dict[int, int]:
        idle_agents = list(world_state.get_idle_agents())
        used_agents: Set[int] = set()
        choices: Dict[int, int] = {}

        for context_index, context in enumerate(contexts):
            available = [
                agent
                for agent in idle_agents
                if int(agent.agent_id) not in used_agents
            ]
            if not available:
                break

            nearest = min(
                available,
                key=lambda agent: (
                    _manhattan(agent.position, context.pod_location),
                    int(agent.agent_id),
                ),
            )
            if self.candidate_limit > 0:
                candidates = sorted(
                    available,
                    key=lambda agent: (
                        _manhattan(agent.position, context.pod_location),
                        int(agent.agent_id),
                    ),
                )[: self.candidate_limit]
            else:
                candidates = sorted(
                    available, key=lambda agent: int(agent.agent_id)
                )
                self.stats["all_idle_candidate_contexts"] += 1
                self.stats["all_idle_candidates_scored"] += len(candidates)

            if not candidates:
                self.stats["oracle_contexts_without_candidate"] += 1
                continue

            self.stats["oracle_contexts_scored"] += 1
            fixed_context = self._fixed_context(context)
            scored = []
            for agent in candidates:
                candidate = self._candidate(agent)
                started = time.perf_counter()
                result = evaluate_candidate_rollout(
                    world_state,
                    candidate,
                    fixed_context,
                    world_state.config,
                    self.path_planner,
                    self.rollout_horizon,
                    self._node_map,
                    self._local_capacity,
                    self._bottleneck_score,
                    self._adj,
                    reservation_window=self.reservation_window,
                    rollout_continuation_mode="isolated",
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self.stats["oracle_rollout_calls"] += 1
                self.stats["oracle_rollout_time_total_ms"] += elapsed_ms

                if not self._complete_rollout(result, self.rollout_horizon):
                    self.stats["oracle_incomplete_rollouts"] += 1
                    raise RuntimeError(
                        "incomplete simulator-oracle rollout at "
                        f"tick={world_state.tick}, context={context_index}, "
                        f"robot={agent.agent_id}"
                    )
                self.stats["oracle_complete_rollouts"] += 1

                cost = float(result["realized_cost"])
                if not math.isfinite(cost):
                    raise RuntimeError(
                        "non-finite simulator-oracle cost at "
                        f"tick={world_state.tick}, context={context_index}, "
                        f"robot={agent.agent_id}: {cost}"
                    )
                scored.append({
                    "robot_id": int(agent.agent_id),
                    "cost": cost,
                    "rollout_ms": elapsed_ms,
                    "vertex_conflicts": int(
                        result.get("rollout_vertex_conflicts", 0)
                    ),
                    "swap_conflicts": int(
                        result.get("rollout_swap_conflicts", 0)
                    ),
                    "blocked_moves": int(
                        result.get("rollout_blocked_moves", 0)
                    ),
                })

            best = min(scored, key=lambda row: (row["cost"], row["robot_id"]))
            min_cost = float(best["cost"])
            if sum(
                1 for row in scored
                if abs(float(row["cost"]) - min_cost) <= 1e-12
            ) > 1:
                self.stats["oracle_exact_tie_contexts"] += 1

            robot_id = int(best["robot_id"])
            choices[int(context_index)] = robot_id
            used_agents.add(robot_id)
            self.stats["oracle_contexts_selected"] += 1
            self.stats["oracle_selected_cost_sum"] += min_cost
            self.stats["oracle_selected_cost_min"] = min(
                float(self.stats["oracle_selected_cost_min"]), min_cost
            )
            self.stats["oracle_selected_cost_max"] = max(
                float(self.stats["oracle_selected_cost_max"]), min_cost
            )
            if robot_id == int(nearest.agent_id):
                self.stats["oracle_selected_nearest_count"] += 1
            self.stats["oracle_selected_vertex_conflicts"] += int(
                best["vertex_conflicts"]
            )
            self.stats["oracle_selected_swap_conflicts"] += int(
                best["swap_conflicts"]
            )
            self.stats["oracle_selected_blocked_moves"] += int(
                best["blocked_moves"]
            )

            self._append_trace({
                "tick": int(world_state.tick),
                "context_index": int(context_index),
                "order_id": int(context.order_id),
                "pod_id": int(context.pod_id),
                "station_id": int(context.station_id),
                "candidate_count": len(scored),
                "nearest_robot_id": int(nearest.agent_id),
                "selected_robot_id": robot_id,
                "selected_cost": min_cost,
                "candidates": scored,
            })

        return choices

    def assign(self, world_state) -> List[Task]:
        started = time.perf_counter()
        self.stats["assign_calls"] += 1
        if not self._initialized:
            self._init(world_state)

        if world_state.config.simulation.task_execution_mode == "serial":
            raise RuntimeError(
                "SimulatorRolloutTaskAssigner supports parallel fixed-context "
                "assignment only"
            )

        idle_count = len(world_state.get_idle_agents())
        context_budget = (
            min(idle_count, self.max_contexts_per_tick)
            if self.max_contexts_per_tick > 0
            else idle_count
        )
        contexts = self.propose_assignment_contexts(
            world_state,
            max_contexts=context_budget,
        )
        if not contexts:
            self.stats["assignment_time_total_ms"] += (
                time.perf_counter() - started
            ) * 1000.0
            return []

        robot_choices = self.select_robots(world_state, contexts)
        tasks = self.commit_assignments(
            world_state,
            contexts,
            robot_choices,
        )
        expected_tasks = 3 * len(robot_choices)
        if len(tasks) != expected_tasks:
            raise RuntimeError(
                "simulator-oracle selection/commit mismatch at "
                f"tick={world_state.tick}: selected={len(robot_choices)}, "
                f"tasks={len(tasks)}, expected_tasks={expected_tasks}"
            )
        self.stats["oracle_contexts_committed"] += len(robot_choices)
        self.stats["assignment_time_total_ms"] += (
            time.perf_counter() - started
        ) * 1000.0
        return tasks

    def oracle_metrics(self) -> dict:
        rollouts = int(self.stats["oracle_rollout_calls"])
        selected = int(self.stats["oracle_contexts_selected"])
        minimum = self.stats["oracle_selected_cost_min"]
        maximum = self.stats["oracle_selected_cost_max"]
        return {
            "oracle_policy": self.__class__.__name__,
            "uses_world_model": False,
            "loads_checkpoint": False,
            "context_scheduler": "prefix_j0",
            "candidate_scope": (
                "all_idle"
                if self.candidate_limit == 0
                else f"nearest_{self.candidate_limit}"
            ),
            "rollout_horizon": int(self.rollout_horizon),
            "rollout_continuation_mode": "isolated",
            "future_orders_in_rollout": False,
            "continuation_assigner_in_rollout": False,
            "reservation_window": int(self.reservation_window),
            "max_contexts_per_tick": int(self.max_contexts_per_tick),
            "rollout_calls": rollouts,
            "complete_rollouts": int(
                self.stats["oracle_complete_rollouts"]
            ),
            "incomplete_rollouts": int(
                self.stats["oracle_incomplete_rollouts"]
            ),
            "rollout_time_total_ms": round(
                float(self.stats["oracle_rollout_time_total_ms"]), 6
            ),
            "rollout_time_ms_mean": round(
                float(self.stats["oracle_rollout_time_total_ms"])
                / max(rollouts, 1),
                6,
            ),
            "contexts_scored": int(self.stats["oracle_contexts_scored"]),
            "contexts_selected": selected,
            "contexts_committed": int(
                self.stats["oracle_contexts_committed"]
            ),
            "candidates_per_context_mean": round(
                rollouts / max(int(self.stats["oracle_contexts_scored"]), 1),
                6,
            ),
            "selected_cost_mean": round(
                float(self.stats["oracle_selected_cost_sum"])
                / max(selected, 1),
                6,
            ),
            "selected_cost_min": (
                round(float(minimum), 6) if math.isfinite(float(minimum))
                else None
            ),
            "selected_cost_max": (
                round(float(maximum), 6) if math.isfinite(float(maximum))
                else None
            ),
            "selected_nearest_count": int(
                self.stats["oracle_selected_nearest_count"]
            ),
            "selected_nearest_ratio": round(
                int(self.stats["oracle_selected_nearest_count"])
                / max(selected, 1),
                6,
            ),
            "exact_tie_contexts": int(
                self.stats["oracle_exact_tie_contexts"]
            ),
            "selected_rollout_vertex_conflicts": int(
                self.stats["oracle_selected_vertex_conflicts"]
            ),
            "selected_rollout_swap_conflicts": int(
                self.stats["oracle_selected_swap_conflicts"]
            ),
            "selected_rollout_blocked_moves": int(
                self.stats["oracle_selected_blocked_moves"]
            ),
            "decision_trace_records": len(self.decision_trace_records),
            "decision_trace_dropped": int(
                self.stats["oracle_trace_dropped"]
            ),
        }
