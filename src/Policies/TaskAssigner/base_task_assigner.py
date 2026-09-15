"""
Base Task Assigner
==================
Abstract base class (interface) for task assignment policies.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from WorldState.task_state import TaskStatus, TaskType

if TYPE_CHECKING:
    from WorldState.world import WorldState
    from WorldState.task_state import Task

logger = logging.getLogger("MAS_RMFS.TaskAssigner")


def fulfilled_pod_ids_for_order(world_state, order) -> Set[int]:
    """Return pod IDs already fulfilled for one order.

    ``Order.delivered_pod_ids`` is the lifecycle source of truth.  The
    completed-DELIVER fallback makes assignment robust to older snapshots in
    which the delivery marker was not persisted even though the task ledger
    already records a successful delivery.  Fulfilment is scoped by order;
    the same returned pod may still be used by a different order.
    """
    fulfilled = {
        int(pod_id)
        for pod_id in getattr(order, "delivered_pod_ids", ())
    }
    task_state = getattr(world_state, "task_state", None)
    tasks = getattr(task_state, "tasks", {})
    order_id = int(getattr(order, "order_id"))
    for task in tasks.values():
        if int(getattr(task, "order_id", -1)) != order_id:
            continue
        if (
            getattr(task, "task_type", None) == TaskType.DELIVER
            and getattr(task, "status", None) == TaskStatus.COMPLETED
        ):
            fulfilled.add(int(getattr(task, "pod_id")))
    return fulfilled


@dataclass
class AssignmentContext:
    """A proposed (order, pod, station) context ready for robot selection.

    Produced by ``propose_assignment_contexts`` — represents one
    dispatchable unit of work without committing any world mutations.
    """
    order_id: int
    pod_id: int
    pod_location: Tuple[int, int]
    station_id: int
    station_location: Tuple[int, int]
    entry_position: Optional[Tuple[int, int]]
    exit_position: Optional[Tuple[int, int]]
    return_location: Tuple[int, int]
    order_size: int


class BaseTaskAssigner(ABC):
    """
    Interface for task assignment policies.

    Implementations decide how to decompose orders into tasks
    and assign them to available agents.

    The propose / select / commit pipeline allows the world model
    data collector and online assigner to hook between phases:

      propose_assignment_contexts  (read-only, except pod_ids materialization)
          ↓
      select_robots               (choose one robot per context)
          ↓
      commit_assignments          (create tasks, mutate world)
    """

    def __init__(self):
        self.pod_return_planner = None
        self.pod_retriever = None

    @abstractmethod
    def assign(self, world_state: "WorldState") -> List["Task"]:
        ...

    def propose_assignment_contexts(
        self,
        world_state: "WorldState",
        max_contexts: Optional[int] = None,
    ) -> List[AssignmentContext]:
        """Return dispatchable (order, pod, station) contexts for this tick.

        Default implementation returns an empty list.
        Subclasses override to provide actual contexts.
        """
        return []

    def select_robots(
        self,
        world_state: "WorldState",
        contexts: List[AssignmentContext],
    ) -> Dict[int, int]:
        """Choose a robot for each context.

        Returns ``{context_index: robot_id}``.
        Default implementation returns an empty dict.
        """
        return {}

    def commit_assignments(
        self,
        world_state: "WorldState",
        contexts: List[AssignmentContext],
        robot_choices: Dict[int, int],
    ) -> List["Task"]:
        """Create task chains and mutate world state.

        Default implementation returns an empty list.
        """
        return []
