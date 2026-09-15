"""
TaskState Module
================
Represents atomic tasks assigned to robot agents.
"""

from enum import Enum, auto
from typing import Optional, Tuple, List


class TaskType(Enum):
    """Types of atomic robot tasks."""
    PICK = auto()     # Go to pod location and pick it up
    DELIVER = auto()  # Carry pod to the target station
    RETURN = auto()   # Return pod to its home location


class TaskStatus(Enum):
    """Lifecycle states of a task."""
    PENDING = auto()      # Created but not yet assigned to an agent
    ASSIGNED = auto()     # Assigned to an agent, waiting to start
    IN_PROGRESS = auto()  # Agent is actively executing this task
    COMPLETED = auto()    # Task finished successfully
    CANCELLED = auto()    # Task cancelled (e.g., pod was unavailable)


class Task:
    """
    An atomic task for a robot agent.

    属性
    ----------
    task_id : int
        Unique task identifier.
    task_type : TaskType
        Type of task (PICK, DELIVER, RETURN).
    order_id : int
        ID of the parent order this task belongs to.
    agent_id : int or None
        ID of the assigned agent.
    pod_id : int
        ID of the pod involved in this task.
    source : tuple[int, int]
        Starting position for this task.
    destination : tuple[int, int]
        Target position for this task.
    status : TaskStatus
        Current lifecycle status.
    """

    _next_id: int = 0

    def __init__(
        self,
        task_type: TaskType,
        order_id: int,
        pod_id: int,
        source: Tuple[int, int],
        destination: Tuple[int, int],
    ):
        self.task_id = Task._next_id
        Task._next_id += 1
        self.task_type = task_type
        self.order_id = order_id
        self.pod_id = pod_id
        self.source = source
        self.destination = destination
        self.agent_id: Optional[int] = None
        self.station_id: Optional[int] = None
        # Opt-in execution-order metadata for station admission experiments.
        # Legacy modes ignore it, so historical task and observation contracts
        # remain unchanged.
        self.station_dispatch_sequence: Optional[int] = None
        self.status: TaskStatus = TaskStatus.PENDING

        self.created_at: Optional[int] = None
        self.assigned_at: Optional[int] = None
        self.started_at: Optional[int] = None
        self.completed_at: Optional[int] = None
        self.free_flow_time: Optional[int] = None
        self.planned_path_len_at_start: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"Task(id={self.task_id}, type={self.task_type.name}, "
            f"pod={self.pod_id}, agent={self.agent_id}, "
            f"status={self.status.name})"
        )


class TaskState:
    """
    Container managing all tasks in the system.

    属性
    ----------
    tasks : dict[int, Task]
        All tasks keyed by task_id.
    """

    def __init__(self):
        self.tasks: dict[int, Task] = {}

    def add_task(self, task: Task):
        """Register a new task."""
        self.tasks[task.task_id] = task

    def get_pending_tasks(self) -> List[Task]:
        """Return all unassigned pending tasks."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]

    def get_tasks_for_agent(self, agent_id: int) -> List[Task]:
        """Return all tasks assigned to a specific agent."""
        return [t for t in self.tasks.values() if t.agent_id == agent_id]

    def get_active_task_for_agent(self, agent_id: int) -> Optional[Task]:
        """Return the currently in-progress task for an agent, if any."""
        for t in self.tasks.values():
            if t.agent_id == agent_id and t.status == TaskStatus.IN_PROGRESS:
                return t
        return None

    def get_next_task_for_agent(self, agent_id: int) -> Optional[Task]:
        """Return the next assigned (but not yet started) task for an agent."""
        for t in self.tasks.values():
            if t.agent_id == agent_id and t.status == TaskStatus.ASSIGNED:
                return t
        return None

    def get_tasks_for_order(self, order_id: int) -> List[Task]:
        """Return all tasks belonging to a specific order."""
        return [t for t in self.tasks.values() if t.order_id == order_id]

    def all_order_tasks_completed(self, order_id: int) -> bool:
        """Check if all tasks for a given order are completed or cancelled."""
        order_tasks = self.get_tasks_for_order(order_id)
        terminal = {TaskStatus.COMPLETED, TaskStatus.CANCELLED}
        return len(order_tasks) > 0 and all(
            t.status in terminal for t in order_tasks
        )

    def get_related_chain_tasks(self, pick_task: "Task") -> List["Task"]:
        """Return DELIVER and RETURN tasks in the same chain as a PICK task."""
        return [
            t for t in self.tasks.values()
            if t.task_id != pick_task.task_id
            and t.order_id == pick_task.order_id
            and t.pod_id == pick_task.pod_id
            and t.agent_id == pick_task.agent_id
            and t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        ]

    def __repr__(self) -> str:
        pending = len(self.get_pending_tasks())
        total = len(self.tasks)
        return f"TaskState(total={total}, pending={pending})"
