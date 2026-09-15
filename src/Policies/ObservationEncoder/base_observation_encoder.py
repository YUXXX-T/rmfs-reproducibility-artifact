"""
观测编码器基类
====================
定义将 WorldState 编码为 RL 观测的接口。

Base Observation Encoder
========================
Abstract interface for encoding WorldState into RL observations.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import gymnasium
    from WorldState.world import WorldState


class BaseObservationEncoder(ABC):
    """
    将 WorldState 编码为智能体观测向量的接口。

    不同实现可以产生不同形状的观测：
    - GridObservationEncoder: 多通道网格 (rows, cols, C) — 适用于 CNN
    - FlatObservationEncoder: 一维向量 (N,) — 适用于 MLP
    """

    @abstractmethod
    def observation_space(
        self, world_state: "WorldState"
    ) -> "gymnasium.spaces.Space":
        """
        返回单个智能体的观测空间定义。

        参数
        ----------
        world_state : WorldState
            用于推断空间维度的世界状态（如网格大小、智能体数量）。

        返回值
        -------
        gymnasium.spaces.Space
            观测空间。
        """
        ...

    @abstractmethod
    def encode(
        self, world_state: "WorldState", agent_id: int
    ) -> np.ndarray:
        """
        将世界状态编码为单个智能体的观测。

        参数
        ----------
        world_state : WorldState
            当前世界状态。
        agent_id : int
            需要编码观测的智能体 ID。

        返回值
        -------
        np.ndarray
            观测向量/矩阵。
        """
        ...
