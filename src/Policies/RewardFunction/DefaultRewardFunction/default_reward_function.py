"""
默认奖励函数
====================
订单完成奖励 + 时间惩罚 + 冲突惩罚。

Default Reward Function
=======================
Order completion reward + time penalty + conflict penalty.
"""

from typing import TYPE_CHECKING

from Policies.RewardFunction.base_reward_function import BaseRewardFunction

if TYPE_CHECKING:
    from WorldState.world import WorldState


class DefaultRewardFunction(BaseRewardFunction):
    """
    默认奖励函数。

    奖励组成：
    - +10.0 每完成一个订单（全局共享）
    - -0.01 每个 tick（时间压力）
    - -1.0  每个冲突（顶点冲突或边冲突）
    - +0.5  智能体成功拾取货架
    - +0.5  智能体成功归还货架

    终止条件：
    - tick >= max_ticks（默认 500）
    """

    def __init__(self, max_ticks: int = 500):
        self.max_ticks = max_ticks
        self._prev_agent_status = {}  # 追踪状态变化

    def compute(
        self,
        world_state: "WorldState",
        prev_completed: int,
        agent_id: int,
    ) -> float:
        reward = 0.0

        # 时间惩罚
        reward -= 0.01

        # 订单完成奖励（全局共享给所有智能体）
        new_completed = world_state.order_state.total_completed - prev_completed
        reward += new_completed * 10.0

        # 智能体特定奖励：状态变化
        agent = world_state.agents[agent_id]
        prev_status = self._prev_agent_status.get(agent_id)
        curr_status = agent.status.name

        if (
            prev_status == "MOVING_TO_POD"
            and curr_status in ("CARRYING", "WAITING_ASSIGNED")
        ):
            reward += 0.5  # 成功拾取货架
        elif prev_status == "RETURNING" and curr_status == "IDLE":
            reward += 0.5  # 成功归还货架

        # 更新状态追踪
        self._prev_agent_status[agent_id] = curr_status

        return reward

    def is_done(self, world_state: "WorldState") -> bool:
        return world_state.tick >= self.max_ticks

    def reset(self):
        """回合开始时重置内部状态。"""
        self._prev_agent_status.clear()
