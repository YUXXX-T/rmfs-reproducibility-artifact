"""
Hungarian Task Assigner
=======================
Optimal global assignment baseline using the Hungarian algorithm
to minimize total Manhattan distance between idle agents and pods.

匈牙利任务分配器
=======================
基于匈牙利算法的最优全局分配 baseline，最小化空闲 agent 到 pod 的总曼哈顿距离。
"""

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from Policies.TaskAssigner.base_task_assigner import (
    AssignmentContext,
    BaseTaskAssigner,
    fulfilled_pod_ids_for_order,
)
from Policies.TaskAssigner.context_assignment import (
    commit_fixed_context_assignments,
)
from WorldState.task_state import Task, TaskStatus


def _manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class HungarianTaskAssigner(BaseTaskAssigner):
    """
    Optimal global task assigner using the Hungarian algorithm.

    Collects all (order, pod) pairs and all idle agents, builds a cost
    matrix of Manhattan distances, and solves the linear sum assignment
    to minimize total travel distance.

    基于匈牙利算法的最优全局任务分配器。
    收集所有 (订单, pod) 对与所有空闲 agent，构建曼哈顿距离代价矩阵，
    通过线性和分配求解最小化总行驶距离。
    """

    def assign(self, world_state) -> List[Task]:
        pending_orders = world_state.order_state.get_pending_orders()
        for order in pending_orders:
            if self.pod_retriever is not None:
                if not order.pod_ids:
                    order.pod_ids = self.pod_retriever.retrieve(order, world_state)
                else:
                    fulfilled_pods = fulfilled_pod_ids_for_order(
                        world_state, order,
                    )
                    any_available = any(
                        (p := world_state.pod_state.get_pod(pid)) is not None
                        and not p.is_carried
                        and int(pid) not in fulfilled_pods
                        for pid in order.pod_ids
                    )
                    if not any_available:
                        order.pod_ids = self.pod_retriever.retrieve(order, world_state)

        idle_agents = sorted(
            world_state.get_idle_agents(),
            key=lambda agent: int(agent.agent_id),
        )
        if not idle_agents or not pending_orders:
            return []

        reserved_pods = {
            int(t.pod_id)
            for t in world_state.task_state.tasks.values()
            if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        }

        seen_pods: set = set()
        pods_to_assign: List[Tuple] = []
        for order in pending_orders:
            fulfilled_pods = fulfilled_pod_ids_for_order(world_state, order)
            for pod_id in order.pod_ids:
                pod_id = int(pod_id)
                if (
                    pod_id in fulfilled_pods
                    or pod_id in seen_pods
                    or pod_id in reserved_pods
                ):
                    continue
                pod = world_state.pod_state.get_pod(pod_id)
                if pod is None or pod.is_carried:
                    continue
                station_location = (
                    world_state.station_state.get_service_position(
                        order.station_id
                    )
                )
                if station_location is None:
                    station_location = (
                        world_state.map_state.station_positions.get(
                            order.station_id
                        )
                    )
                if station_location is None:
                    continue
                entry_position = (
                    world_state.station_state.get_entry_position(
                        order.station_id
                    )
                )
                exit_position = (
                    world_state.station_state.get_exit_position(
                        order.station_id
                    )
                )
                pods_to_assign.append((
                    order,
                    pod,
                    tuple(station_location),
                    (
                        tuple(entry_position)
                        if entry_position is not None else None
                    ),
                    (
                        tuple(exit_position)
                        if exit_position is not None else None
                    ),
                ))
                seen_pods.add(pod_id)

        if not pods_to_assign:
            return []

        n_agents = len(idle_agents)
        n_pods = len(pods_to_assign)
        cost = np.zeros((n_agents, n_pods))
        for i, agent in enumerate(idle_agents):
            for j, (_, pod, _, _, _) in enumerate(pods_to_assign):
                cost[i, j] = _manhattan_distance(agent.position, pod.current_position)

        row_idx, col_idx = linear_sum_assignment(cost)

        contexts: List[AssignmentContext] = []
        robot_choices = {}
        for i, j in zip(row_idx, col_idx):
            agent = idle_agents[i]
            order, pod, station_location, entry_position, exit_position = (
                pods_to_assign[j]
            )
            return_source = exit_position or station_location
            if self.pod_return_planner is not None:
                return_location = self.pod_return_planner.plan_return(
                    pod, return_source, world_state
                )
            else:
                return_location = pod.home_position
            if return_location is None:
                return_location = pod.home_position

            context_index = len(contexts)
            contexts.append(AssignmentContext(
                order_id=int(order.order_id),
                pod_id=int(pod.pod_id),
                pod_location=tuple(pod.current_position),
                station_id=int(order.station_id),
                station_location=station_location,
                entry_position=entry_position,
                exit_position=exit_position,
                return_location=tuple(return_location),
                order_size=int(sum(order.sku_demands.values())),
            ))
            robot_choices[context_index] = int(agent.agent_id)

        # The Hungarian solver owns only the global robot/pod matching.  Task
        # materialisation is deliberately shared with WM/Greedy so station
        # queue, exit handoff, order lifecycle and completed-pod guards cannot
        # silently diverge between baselines.
        return commit_fixed_context_assignments(
            world_state,
            contexts,
            robot_choices,
        )
