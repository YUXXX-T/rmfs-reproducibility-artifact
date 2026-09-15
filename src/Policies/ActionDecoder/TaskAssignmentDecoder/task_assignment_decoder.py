"""
任务分配动作解码器
====================
RL 智能体选择要处理的货架，替代贪心任务分配器。

Task Assignment Decoder
=======================
RL agent selects which pod to process, replacing the greedy task assigner.
"""

import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Dict, TYPE_CHECKING

from Policies.ActionDecoder.base_action_decoder import BaseActionDecoder

if TYPE_CHECKING:
    from WorldState.world import WorldState


class TaskAssignmentDecoder(BaseActionDecoder):
    """
    任务分配解码器。

    每个智能体的动作 = Discrete(num_pods + 1):
    - 0: 空闲（不执行任何操作）
    - 1..num_pods: 选择去取对应 ID 的货架

    仅当智能体处于 IDLE 状态时动作才会生效。
    如果所选货架不可用（已被搬运或已被预留），则动作被忽略。
    """

    def action_space(self, world_state: "WorldState") -> spaces.Space:
        num_pods = world_state.pod_state.total_pods
        # 0=空闲, 1..num_pods=选择对应货架
        return spaces.Discrete(num_pods + 1)

    def decode(
        self, actions: Dict[str, int], world_state: "WorldState"
    ) -> None:
        """
        将 RL 动作解码并应用到世界状态。

        对于每个选择了有效货架的空闲智能体：
        1. 查找包含该货架的待处理订单
        2. 创建 PICK → DELIVER → RETURN 任务链
        3. 分配给该智能体
        """
        from WorldState.task_state import Task, TaskType, TaskStatus

        # 收集已被预留的货架
        reserved_pods = set()
        for task in world_state.task_state.tasks.values():
            if task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
                reserved_pods.add(task.pod_id)

        for agent_name, action in actions.items():
            # 解析智能体 ID: "agent_0" → 0
            agent_id = int(agent_name.split("_")[1])
            agent = world_state.agents[agent_id]

            # 仅处理空闲智能体
            if not agent.is_idle:
                continue

            # 动作 0 = 保持空闲
            if action == 0:
                continue

            pod_id = action - 1  # 动作 1..N 映射到货架 0..N-1

            # 验证货架有效性
            pod = world_state.pod_state.get_pod(pod_id)
            if pod is None or pod.is_carried or pod_id in reserved_pods:
                continue

            # 查找包含该货架的待处理订单
            target_order = None
            for order in world_state.order_state.get_pending_orders():
                if pod_id in order.pod_ids:
                    target_order = order
                    break

            if target_order is None:
                # 没有待处理订单需要此货架，跳过
                continue

            # 确定目标工作站 service position
            station_pos = world_state.station_state.get_service_position(
                target_order.station_id
            )
            if station_pos is None:
                station_pos = world_state.map_state.station_positions.get(
                    target_order.station_id
                )
            if station_pos is None:
                continue

            # 确定归还位置
            exit_pos = world_state.station_state.get_exit_position(
                target_order.station_id
            )
            return_source = exit_pos or station_pos
            return_pos = pod.home_position

            # 创建任务链: PICK → DELIVER → RETURN
            pick_task = Task(
                task_type=TaskType.PICK,
                order_id=target_order.order_id,
                pod_id=pod_id,
                source=agent.position,
                destination=pod.current_position,
            )
            pick_task.agent_id = agent_id
            pick_task.status = TaskStatus.ASSIGNED

            deliver_task = Task(
                task_type=TaskType.DELIVER,
                order_id=target_order.order_id,
                pod_id=pod_id,
                source=pod.current_position,
                destination=station_pos,
            )
            deliver_task.agent_id = agent_id
            deliver_task.status = TaskStatus.ASSIGNED
            deliver_task.station_id = target_order.station_id

            return_task = Task(
                task_type=TaskType.RETURN,
                order_id=target_order.order_id,
                pod_id=pod_id,
                source=return_source,
                destination=return_pos,
            )
            return_task.agent_id = agent_id
            return_task.status = TaskStatus.ASSIGNED
            return_task.station_id = target_order.station_id

            # 注册任务
            world_state.task_state.add_task(pick_task)
            world_state.task_state.add_task(deliver_task)
            world_state.task_state.add_task(return_task)

            # 激活智能体
            from WorldState.agent_state import AgentStatus
            agent.status = AgentStatus.MOVING_TO_POD
            agent.assigned_task_id = pick_task.task_id

            # 标记订单为进行中
            from WorldState.order_state import OrderStatus
            target_order.status = OrderStatus.IN_PROGRESS

            # 标记货架为已预留
            reserved_pods.add(pod_id)
