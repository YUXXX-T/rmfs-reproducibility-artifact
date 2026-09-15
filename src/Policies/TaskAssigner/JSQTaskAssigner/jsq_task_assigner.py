"""
JSQ (Join-Shortest-Queue) Task Assigner
========================================
J-level (cross-context sorting) baseline: prioritise assignment contexts
whose target station has the lowest committed load, with greedy nearest-
robot selection within each chosen context.

Pipeline
--------
1. **Propose** all currently dispatchable (order, pod, station) contexts
   without the idle-count truncation used by the greedy baseline.  The full
   superset is needed so that cross-station load comparison can surface
   contexts that a FIFO-ordered prefix would otherwise hide.

2. **Select** iteratively:
   a. Among remaining contexts find the one whose target station has the
      lowest *virtual* committed load (ties broken by original proposal
      order to preserve order-age priority).
   b. For that context assign the nearest idle robot (Manhattan distance
      to the pod).
   c. Increment the virtual committed load for the target station and
      remove the assigned robot from the idle pool before the next step.

3. **Commit** the selected (context, robot) pairs into PICK/DELIVER/RETURN
   task chains using the shared ``commit_fixed_context_assignments`` helper.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from Policies.TaskAssigner.base_task_assigner import (
    BaseTaskAssigner,
    AssignmentContext,
)
from Policies.TaskAssigner.context_assignment import (
    materialize_fixed_order_pod_contexts,
    enumerate_pending_assignment_contexts,
    commit_fixed_context_assignments,
)
from WorldState.task_state import TaskStatus

if TYPE_CHECKING:
    from WorldState.task_state import Task

logger = logging.getLogger("MAS_RMFS.TaskAssigner")


def _manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class JSQTaskAssigner(BaseTaskAssigner):
    """Join-Shortest-Queue context ordering with greedy robot selection.

    Cross-context (J-level) baseline that prioritises contexts whose target
    station has the lowest committed load.  Robot selection within each
    chosen context uses the greedy nearest-idle-agent heuristic.

    The ordering is *dynamic*: after each (context, robot) pair is selected
    the virtual committed load for the assigned station is incremented so
    subsequent iterations see the updated queue picture.  This prevents all
    assignments from flooding the same station when the initial snapshot
    shows identical load values.
    """

    def __init__(self):
        super().__init__()
        self._serial_warned = False

    # ------------------------------------------------------------------
    # assign() -- top-level entry
    # ------------------------------------------------------------------

    def assign(self, world_state) -> List["Task"]:
        mode = getattr(
            getattr(world_state, "config", None), "simulation", None
        )
        if mode and getattr(mode, "task_execution_mode", "parallel") == "serial":
            if not self._serial_warned:
                logger.warning(
                    "JSQTaskAssigner does not implement serial mode; "
                    "falling back to parallel (cross-context) assignment"
                )
                self._serial_warned = True

        materialize_fixed_order_pod_contexts(world_state, self.pod_retriever)

        contexts = self.propose_assignment_contexts(world_state)
        if not contexts:
            return []

        robot_choices = self.select_robots(world_state, contexts)
        return self.commit_assignments(world_state, contexts, robot_choices)

    # ------------------------------------------------------------------
    # propose_assignment_contexts -- full dispatchable superset
    # ------------------------------------------------------------------

    def propose_assignment_contexts(
        self,
        world_state,
        max_contexts: Optional[int] = None,
    ) -> List[AssignmentContext]:
        """Return all currently dispatchable contexts.

        Unlike the greedy baseline which truncates to ``idle_count``, JSQ
        requires the complete superset so that contexts targeting less-
        loaded stations are not hidden by the default FIFO prefix.
        """
        # Collect all pending contexts including temporarily unavailable
        # pods so that the full cross-station picture is visible.  Manual
        # filtering below removes carried/reserved/duplicate pods.
        candidates = enumerate_pending_assignment_contexts(
            self,
            world_state,
            max_contexts=None,
            include_temporarily_unavailable=True,
            deduplicate_pods=False,
        )

        active_statuses = (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        reserved_pods: Set[int] = {
            int(task.pod_id)
            for task in world_state.task_state.tasks.values()
            if task.status in active_statuses
        }

        seen_pods: Set[int] = set()
        result: List[AssignmentContext] = []
        for ctx in candidates:
            pod_id = int(ctx.pod_id)
            pod = world_state.pod_state.get_pod(pod_id)
            if pod is None or bool(getattr(pod, "is_carried", False)):
                continue
            if pod_id in reserved_pods:
                continue
            if pod_id in seen_pods:
                continue
            seen_pods.add(pod_id)
            result.append(ctx)

        if max_contexts is not None:
            result = result[: max(0, int(max_contexts))]
        return result

    # ------------------------------------------------------------------
    # select_robots -- JSQ ordering + greedy nearest
    # ------------------------------------------------------------------

    def select_robots(
        self,
        world_state,
        contexts: List[AssignmentContext],
    ) -> Dict[int, int]:
        if not contexts:
            return {}

        # Snapshot the real committed load for each target station.
        virtual_load: Dict[int, int] = {}
        for ctx in contexts:
            sid = int(ctx.station_id)
            if sid not in virtual_load:
                queue = world_state.station_state.get_queue(sid)
                virtual_load[sid] = (
                    queue.committed_load() if queue is not None else 0
                )

        idle_agents = world_state.get_idle_agents()
        used_agents: Set[int] = set()
        choices: Dict[int, int] = {}

        # Indexed remaining contexts: (original_index, context).
        remaining: List[Tuple[int, AssignmentContext]] = list(
            enumerate(contexts)
        )

        while remaining:
            available = [
                a for a in idle_agents if a.agent_id not in used_agents
            ]
            if not available:
                break

            # JSQ: pick the context whose target station has the lowest
            # virtual committed load.  Tie-break by original proposal
            # index (lower = older order = higher priority), which keeps
            # the baseline deterministic and order-age-aware.
            remaining.sort(
                key=lambda item: (
                    virtual_load.get(int(item[1].station_id), 0),
                    item[0],
                ),
            )
            ctx_index, ctx = remaining[0]

            # Greedy: nearest idle agent to the pod location.
            available.sort(
                key=lambda a: _manhattan_distance(
                    a.position, ctx.pod_location
                ),
            )
            best = available[0]

            choices[ctx_index] = best.agent_id
            used_agents.add(best.agent_id)

            # Virtual committed-load increment: the station will gain one
            # more committed robot once this assignment is materialised.
            virtual_load[int(ctx.station_id)] = (
                virtual_load.get(int(ctx.station_id), 0) + 1
            )

            remaining.pop(0)

        return choices

    # ------------------------------------------------------------------
    # commit_assignments -- delegate to shared helper
    # ------------------------------------------------------------------

    def commit_assignments(
        self,
        world_state,
        contexts: List[AssignmentContext],
        robot_choices: Dict[int, int],
    ) -> List["Task"]:
        return commit_fixed_context_assignments(
            world_state, contexts, robot_choices,
        )
