"""
网格观测编码器
====================
将 WorldState 编码为多通道网格张量，适用于 CNN 策略网络。

Grid Observation Encoder
========================
Encodes WorldState as a multi-channel grid tensor suitable for CNN policies.
"""

import numpy as np
import gymnasium
from gymnasium import spaces
from typing import TYPE_CHECKING

from Policies.ObservationEncoder.base_observation_encoder import BaseObservationEncoder

if TYPE_CHECKING:
    from WorldState.world import WorldState


class GridObservationEncoder(BaseObservationEncoder):
    """
    多通道网格编码器。

    将世界状态编码为 Dict 空间：
    - "grid": Box(rows, cols, 6) — 6 通道网格
    - "agent_features": Box(num_features,) — 当前智能体的特征向量

    通道定义
    ----------
    0: 障碍物 (1=障碍)
    1: 工作站 (1=工作站)
    2: 货架位置 (1=有静止货架)
    3: 智能体位置 (各智能体 ID+1 的归一化值)
    4: 搬运中的货架 (智能体搬运货架时为 1)
    5: 待处理订单目标工作站热力图
    """

    def observation_space(self, world_state: "WorldState") -> spaces.Space:
        rows = world_state.map_state.rows
        cols = world_state.map_state.cols
        num_agents = len(world_state.agents)

        return spaces.Dict({
            "grid": spaces.Box(
                low=0.0, high=1.0,
                shape=(rows, cols, 6),
                dtype=np.float32,
            ),
            "agent_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(7,),  # 见 _encode_agent_features
                dtype=np.float32,
            ),
        })

    def encode(self, world_state: "WorldState", agent_id: int) -> dict:
        ms = world_state.map_state
        rows, cols = ms.rows, ms.cols
        num_agents = len(world_state.agents)

        grid = np.zeros((rows, cols, 6), dtype=np.float32)

        # 通道 0: 障碍物
        for r in range(rows):
            for c in range(cols):
                if ms.grid[r][c].name == "OBSTACLE":
                    grid[r, c, 0] = 1.0

        # 通道 1: 工作站
        for sid, (r, c) in ms.station_positions.items():
            grid[r, c, 1] = 1.0

        # 通道 2: 静止货架
        for pod in world_state.pod_state.pods.values():
            if not pod.is_carried:
                r, c = pod.current_position
                grid[r, c, 2] = 1.0

        # 通道 3: 智能体位置（归一化）
        for agent in world_state.agents:
            r, c = agent.position
            grid[r, c, 3] = (agent.agent_id + 1) / max(num_agents, 1)

        # 通道 4: 搬运中的货架
        for agent in world_state.agents:
            if agent.carried_pod_id is not None:
                r, c = agent.position
                grid[r, c, 4] = 1.0

        # 通道 5: 待处理订单目标工作站热力图
        pending = world_state.order_state.get_pending_orders()
        in_prog = world_state.order_state.get_in_progress_orders()
        all_active = pending + in_prog
        if all_active:
            for order in all_active:
                station_pos = ms.station_positions.get(order.station_id)
                if station_pos:
                    r, c = station_pos
                    grid[r, c, 5] = min(grid[r, c, 5] + 0.2, 1.0)

        # 智能体特征向量
        agent_feat = self._encode_agent_features(world_state, agent_id)

        return {"grid": grid, "agent_features": agent_feat}

    def _encode_agent_features(
        self, world_state: "WorldState", agent_id: int
    ) -> np.ndarray:
        """
        编码单个智能体的特征向量 (7维):
        [0] row / rows           — 归一化行坐标
        [1] col / cols           — 归一化列坐标
        [2] status_code / 5      — 归一化状态编码
        [3] has_path             — 是否有路径 (0/1)
        [4] is_carrying          — 是否搬运货架 (0/1)
        [5] wait_ticks / 10      — 归一化等待 tick 数
        [6] tick / max_ticks     — 归一化当前 tick
        """
        agent = world_state.agents[agent_id]
        rows = world_state.map_state.rows
        cols = world_state.map_state.cols

        status_map = {
            "IDLE": 0, "MOVING_TO_POD": 1, "WAITING_ASSIGNED": 2,
            "CARRYING": 2,
            "DELIVERING": 3, "RETURNING": 4, "MOVING": 5,
        }
        status_code = status_map.get(agent.status.name, 0) / 5.0

        return np.array([
            agent.position[0] / max(rows - 1, 1),
            agent.position[1] / max(cols - 1, 1),
            status_code,
            1.0 if agent.has_path else 0.0,
            1.0 if agent.carried_pod_id is not None else 0.0,
            min(agent.wait_ticks / 10.0, 1.0),
            min(world_state.tick / 500.0, 1.0),
        ], dtype=np.float32)
