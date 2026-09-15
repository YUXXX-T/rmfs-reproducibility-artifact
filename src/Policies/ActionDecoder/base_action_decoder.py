"""
动作解码器基类
====================
定义将 RL 动作解码并应用到 WorldState 的接口。

Base Action Decoder
===================
Abstract interface for decoding RL actions and applying them to WorldState.
"""

from abc import ABC, abstractmethod
from typing import Dict, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import gymnasium
    from WorldState.world import WorldState


class BaseActionDecoder(ABC):
    """
    将 RL 智能体的动作解码并应用到仿真世界的接口。

    不同实现对应不同的动作语义：
    - TaskAssignmentDecoder: 智能体选择要处理的货架
    - MovementDecoder: 智能体选择移动方向（未来扩展）
    """

    @abstractmethod
    def action_space(
        self, world_state: "WorldState"
    ) -> "gymnasium.spaces.Space":
        """
        返回单个智能体的动作空间定义。

        参数
        ----------
        world_state : WorldState
            用于推断动作维度的世界状态。

        返回值
        -------
        gymnasium.spaces.Space
            动作空间。
        """
        ...

    @abstractmethod
    def decode(
        self, actions: Dict[str, np.ndarray], world_state: "WorldState"
    ) -> None:
        """
        将各智能体的动作解码并应用到世界状态。

        参数
        ----------
        actions : dict[str, np.ndarray]
            智能体名称 → 动作的映射，如 {"agent_0": 3, "agent_1": 0}。
        world_state : WorldState
            当前世界状态（将被就地修改）。
        """
        ...
