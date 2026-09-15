"""
奖励函数基类
====================
定义计算 RL 奖励和回合终止的接口。

Base Reward Function
====================
Abstract interface for computing RL reward signals and episode termination.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from WorldState.world import WorldState


class BaseRewardFunction(ABC):
    """
    计算智能体奖励和回合终止条件的接口。

    不同实现可以定义不同的奖励信号：
    - DefaultRewardFunction: 订单完成奖励 + 时间惩罚
    """

    @abstractmethod
    def compute(
        self,
        world_state: "WorldState",
        prev_completed: int,
        agent_id: int,
    ) -> float:
        """
        计算单个智能体的即时奖励。

        参数
        ----------
        world_state : WorldState
            执行动作后的世界状态。
        prev_completed : int
            上一步已完成的订单数（用于计算增量）。
        agent_id : int
            智能体 ID。

        返回值
        -------
        float
            该智能体的奖励值。
        """
        ...

    @abstractmethod
    def is_done(self, world_state: "WorldState") -> bool:
        """
        判断回合是否结束。

        参数
        ----------
        world_state : WorldState
            当前世界状态。

        返回值
        -------
        bool
            True 表示回合结束。
        """
        ...
