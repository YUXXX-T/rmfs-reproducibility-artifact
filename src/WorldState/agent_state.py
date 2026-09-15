"""
AgentState Module
=================
Represents individual robot agent state.
"""

from enum import Enum, auto
from typing import Optional, List, Tuple


class AgentStatus(Enum):
    """Possible states of a robot agent."""
    IDLE = auto()           # Waiting for a task
    MOVING_TO_POD = auto()  # Navigating to pick up a pod
    CARRYING = auto()       # Moving with a pod toward the station
    DELIVERING = auto()     # At station, delivering the pod
    QUEUING = auto()        # Waiting in station queue zone
    RETURNING = auto()      # Returning the pod to its home
    EXITING = auto()        # Processing done, waiting to exit station zone
    MOVING = auto()         # Generic movement (e.g., repositioning)
    # Append new versioned states after every historical member so existing
    # enum numeric values remain byte-compatible with old snapshots/tools.
    WAITING_ASSIGNED = auto()  # DELIVER assigned; awaiting station admission


class AgentState:
    """
    State of a single robot agent.

    属性
    ----------
    agent_id : int
        Unique identifier for this agent.
    position : tuple[int, int]
        Current (row, col) on the grid.
    status : AgentStatus
        Current operational status.
    assigned_task_id : int or None
        ID of the currently assigned task.
    carried_pod_id : int or None
        ID of the pod currently being carried.
    path : list[tuple[int, int]]
        Planned path (sequence of cells to traverse). Empty if idle.
    path_index : int
        Current index into the path list.
    wait_ticks : int
        Number of ticks remaining before the current action completes.
        While > 0 the agent cannot move or accept new tasks.
    """

    def __init__(self, agent_id: int, start_position: Tuple[int, int]):
        self.agent_id = agent_id
        self.position: Tuple[int, int] = start_position
        self.status: AgentStatus = AgentStatus.IDLE
        self.assigned_task_id: Optional[int] = None
        self.carried_pod_id: Optional[int] = None
        self.path: List[Tuple[int, int]] = []
        self.path_index: int = 0
        self.wait_ticks: int = 0
        self.stuck_this_tick: bool = False
        # True only when the movement conflict resolver rejected this
        # robot's intended move in the current tick.  ``stuck_this_tick``
        # also covers an intentional wait action from the space-time path
        # planner, so the two signals must remain distinct for traffic-label
        # collection.
        self.traffic_blocked_this_tick: bool = False
        self.moved_this_tick: bool = False
        self.previous_position: Optional[Tuple[int, int]] = start_position
        self.plan_failed_streak: int = 0
        self.stationary_ticks: int = 0
        # Versioned station-admission wait bookkeeping.  These fields remain
        # empty for legacy/v1 runs and do not change observation dimensions.
        self.station_waiting_station_id: Optional[int] = None
        self.station_waiting_since_tick: Optional[int] = None
        self.station_waiting_sequence: Optional[int] = None

    @property
    def is_idle(self) -> bool:
        """Check if agent is available for new tasks."""
        return self.status == AgentStatus.IDLE

    @property
    def is_waiting(self) -> bool:
        """Check if an action countdown or station admission blocks movement."""
        return (
            self.wait_ticks > 0
            or self.status == AgentStatus.WAITING_ASSIGNED
        )

    def mark_station_waiting(
        self,
        station_id: int,
        since_tick: int,
        sequence: Optional[int] = None,
    ) -> None:
        """Mark a carried-pod robot as waiting for station capacity."""
        self.status = AgentStatus.WAITING_ASSIGNED
        # Admission wait is not a path-planning failure or a physical stall.
        # Clear transient movement diagnostics when entering this phase so
        # the next observation cannot inherit a stale blocked streak.
        self.plan_failed_streak = 0
        self.stuck_this_tick = False
        self.traffic_blocked_this_tick = False
        self.moved_this_tick = False
        self.station_waiting_station_id = int(station_id)
        self.station_waiting_since_tick = int(since_tick)
        self.station_waiting_sequence = (
            int(sequence) if sequence is not None else None
        )

    def clear_station_waiting(self) -> None:
        """Clear versioned station-admission wait metadata."""
        self.station_waiting_station_id = None
        self.station_waiting_since_tick = None
        self.station_waiting_sequence = None

    @property
    def has_path(self) -> bool:
        """Check if agent has remaining steps in its path."""
        return self.path_index < len(self.path)

    def advance(self) -> Optional[Tuple[int, int]]:
        """
        Move the agent one step along its planned path.

        返回值
        -------
        tuple[int, int] or None
            The new position, or None if path is exhausted.
        """
        if self.has_path:
            self.position = self.path[self.path_index]
            self.path_index += 1
            return self.position
        return None

    def clear_path(self):
        """Clear the agent's current path."""
        self.path = []
        self.path_index = 0

    def assign_path(self, path: List[Tuple[int, int]]):
        """Set a new path for the agent to follow."""
        self.path = path
        self.path_index = 0

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.agent_id}, pos={self.position}, "
            f"status={self.status.name}, task={self.assigned_task_id})"
        )
