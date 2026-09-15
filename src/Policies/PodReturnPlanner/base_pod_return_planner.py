"""
Base Pod Return Planner
========================
Abstract base class (interface) for pod return location policies.

After a robot delivers a pod to a workstation, the return planner
decides *where* the pod should be placed back.
"""

from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from WorldState.pod_state import Pod
    from WorldState.world import WorldState


class BasePodReturnPlanner(ABC):
    """
    Interface for pod return location policies.

    Implementations decide where a pod should be dropped off
    after its task at the workstation is complete.
    """

    @abstractmethod
    def plan_return(
        self,
        pod: "Pod",
        station_pos: Tuple[int, int],
        world_state: "WorldState",
    ) -> Tuple[int, int]:
        """
        Decide where to return (drop off) the pod.

        参数
        ----------
        pod : Pod
            The pod being returned.
        station_pos : tuple[int, int]
            The workstation position the pod was just delivered to.
        world_state : WorldState
            Current simulation state (map, agents, pods, etc.).

        返回值
        -------
        tuple[int, int]
            Target (row, col) where the pod should be placed.
        """
        ...
