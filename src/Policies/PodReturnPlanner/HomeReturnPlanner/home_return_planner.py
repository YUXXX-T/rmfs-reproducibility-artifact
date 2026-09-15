"""
Home Return Planner
====================
Default implementation: always returns pods to their original home position.

默认实现：始终将货架返回到其原始主位置。
"""

from typing import Tuple, TYPE_CHECKING

from Policies.PodReturnPlanner.base_pod_return_planner import BasePodReturnPlanner

if TYPE_CHECKING:
    from WorldState.pod_state import Pod
    from WorldState.world import WorldState


class HomeReturnPlanner(BasePodReturnPlanner):
    """
    Always return the pod to its original home position.

    This is the simplest strategy — each pod goes back exactly
    where it started.

    务必将 Pod 返回至其原始归位点。 

    这是最简单的策略——每个 Pod 都准确地返回到其初始位置。
    """

    def plan_return(
        self,
        pod: "Pod",
        station_pos: Tuple[int, int],
        world_state: "WorldState",
    ) -> Tuple[int, int]:
        return pod.home_position
