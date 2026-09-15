"""
扁平观测编码器
====================
将 WorldState 编码为一维向量，适用于 MLP 策略网络。

Flat Observation Encoder
========================
Encodes WorldState as a 1D flat vector suitable for MLP policies.
"""

import numpy as np
import gymnasium
from gymnasium import spaces
from typing import TYPE_CHECKING

from Policies.ObservationEncoder.base_observation_encoder import BaseObservationEncoder

if TYPE_CHECKING:
    from WorldState.world import WorldState


class FlatObservationEncoder(BaseObservationEncoder):
    """
    扁平观测编码器。

    将世界状态编码为一维 Box(N,) 向量：
    - 网格展平: rows × cols × 3 (障碍物 + 货架 + 工作站)
    - 各智能体特征: num_agents × 6
    - 订单摘要: 3 维 (待处理数/总数, 进行中数/总数, 已完成数/总数)
    - 归一化 tick: 1 维
    """

    def _obs_size(self, world_state: "WorldState") -> int:
        ms = world_state.map_state
        n_agents = len(world_state.agents)
        grid_flat = ms.rows * ms.cols * 3
        agent_flat = n_agents * 6
        order_summary = 3
        tick_feat = 1
        return grid_flat + agent_flat + order_summary + tick_feat

    def observation_space(self, world_state: "WorldState") -> spaces.Space:
        size = self._obs_size(world_state)
        return spaces.Box(
            low=0.0, high=1.0, shape=(size,), dtype=np.float32
        )

    def encode(self, world_state: "WorldState", agent_id: int) -> np.ndarray:
        ms = world_state.map_state
        rows, cols = ms.rows, ms.cols
        parts = []

        # 网格展平: 3 通道
        grid = np.zeros((rows, cols, 3), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                cell = ms.grid[r][c].name
                if cell == "OBSTACLE":
                    grid[r, c, 0] = 1.0
                elif cell == "STATION":
                    grid[r, c, 1] = 1.0
        for pod in world_state.pod_state.pods.values():
            if not pod.is_carried:
                r, c = pod.current_position
                grid[r, c, 2] = 1.0
        parts.append(grid.flatten())

        # 各智能体特征: 6 维 × num_agents
        for agent in world_state.agents:
            status_map = {
                "IDLE": 0, "MOVING_TO_POD": 1, "WAITING_ASSIGNED": 2,
                "CARRYING": 2,
                "DELIVERING": 3, "RETURNING": 4, "MOVING": 5,
            }
            feat = np.array([
                agent.position[0] / max(rows - 1, 1),
                agent.position[1] / max(cols - 1, 1),
                status_map.get(agent.status.name, 0) / 5.0,
                1.0 if agent.has_path else 0.0,
                1.0 if agent.carried_pod_id is not None else 0.0,
                min(agent.wait_ticks / 10.0, 1.0),
            ], dtype=np.float32)
            parts.append(feat)

        # 订单摘要: 3 维
        os_ = world_state.order_state
        total = max(os_.total_orders, 1)
        parts.append(np.array([
            len(os_.get_pending_orders()) / total,
            len(os_.get_in_progress_orders()) / total,
            os_.total_completed / total,
        ], dtype=np.float32))

        # 归一化 tick
        parts.append(np.array([
            min(world_state.tick / 500.0, 1.0)
        ], dtype=np.float32))

        return np.concatenate(parts)
