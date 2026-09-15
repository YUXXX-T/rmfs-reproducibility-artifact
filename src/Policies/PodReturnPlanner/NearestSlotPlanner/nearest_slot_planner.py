"""
Nearest Slot Return Planner
============================
返回值 the pod to the closest available (empty) pod-home slot,
minimizing the robot's travel distance after delivery.

将 Pod 返回至距离最近的可用（空闲）Pod 位，
从而最大程度地缩短机器人完成配送后的移动距离。
"""

from typing import Tuple, TYPE_CHECKING

from Policies.PodReturnPlanner.base_pod_return_planner import BasePodReturnPlanner

if TYPE_CHECKING:
    from WorldState.pod_state import Pod
    from WorldState.world import WorldState


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class NearestSlotPlanner(BasePodReturnPlanner):
    """
    Return the pod to the nearest empty pod-home slot.

    Scans all pod-home positions on the map and picks the one
    closest to the workstation that is not currently occupied by
    another pod.  Falls back to the pod's original home if every
    slot is occupied.

    将 Pod 返回至最近的空闲 Pod 归巢槽位。 

    扫描地图上所有的 Pod 归巢位置，并选取其中
    距离工作站最近且当前未被其他 Pod 占用的槽位。 
    若所有槽位均已被占用，则回退至该 Pod 的原始归巢位置。
    """

    def plan_return(
        self,
        pod: "Pod",
        station_pos: Tuple[int, int],
        world_state: "WorldState",
    ) -> Tuple[int, int]:
        # Collect positions occupied by other grounded pods
        occupied = set()
        for other in world_state.pod_state.pods.values():
            if other.pod_id != pod.pod_id and not other.is_carried:
                occupied.add(other.current_position)

        # Also mark positions where another pod is being returned to
        # (its RETURN task destination) to avoid double-booking
        for task in world_state.task_state.tasks.values():
            from WorldState.task_state import TaskType, TaskStatus
            if (
                task.task_type == TaskType.RETURN
                and task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
                and task.pod_id != pod.pod_id
            ):
                occupied.add(task.destination)

        # Find nearest empty pod-home slot
        best_pos = pod.home_position
        best_dist = _manhattan(station_pos, best_pos)

        for home_pos in world_state.map_state.pod_home_positions:
            if home_pos in occupied:
                continue
            d = _manhattan(station_pos, home_pos)
            if d < best_dist:
                best_dist = d
                best_pos = home_pos

        pod.home_position = best_pos
        return best_pos
