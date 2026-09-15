"""
Base Path Planner
=================
Abstract base class (interface) for path planning policies.
"""

from abc import ABC, abstractmethod
from typing import List, Set, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from WorldState.world import WorldState
    from WorldState.agent_state import AgentState


class BasePathPlanner(ABC):
    """
    Interface for path planning policies.

    Implementations compute a path from the agent's current position
    to a goal on the grid.
    """

    @abstractmethod
    def plan(
        self,
        agent: "AgentState",
        goal: Tuple[int, int],
        world_state: "WorldState",
        extra_blocked: Optional[Set[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int]]:
        """
        Compute a path from the agent's current position to the goal.

        参数
        ----------
        agent : AgentState
            The agent requesting a path.
        goal : tuple[int, int]
            Target (row, col) position.
        world_state : WorldState
            Current simulation state (for map & obstacle info).

        返回值
        -------
        list[tuple[int, int]]
            Sequence of (row, col) positions forming the path,
            excluding the agent's current position.
            Empty list if no path is found.
        """
        ...
