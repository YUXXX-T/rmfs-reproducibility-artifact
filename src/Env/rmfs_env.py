"""
MAS-RMFS PettingZoo 多智能体环境
=====================================
基于 PettingZoo ParallelEnv 的多智能体仓库仿真环境。

MAS-RMFS PettingZoo Multi-Agent Environment
============================================
Multi-agent warehouse simulation environment based on PettingZoo ParallelEnv.

所有智能体同时行动（parallel），每个 step 对应一个仿真 tick。
可插拔的 ObservationEncoder、ActionDecoder、RewardFunction 允许
灵活定义观测空间、动作语义和奖励信号。

用法
----
    from Env.rmfs_env import RMFSEnv
    env = RMFSEnv()
    obs, infos = env.reset()
    for _ in range(100):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
"""

import copy
import functools
import os
from typing import Optional, Dict, Any

import numpy as np
from pettingzoo import ParallelEnv
import gymnasium

from Config.config_loader import load_config, SimulationConfig
from Engine.simulation_engine import SimulationEngine
from WorldState.task_state import Task
from WorldState.order_state import Order
import Policies  # noqa: F401 — 触发算法自动注册
from Policies.policy_registry import get_policy

from Policies.ObservationEncoder import BaseObservationEncoder, GridObservationEncoder
from Policies.ActionDecoder import BaseActionDecoder, TaskAssignmentDecoder
from Policies.RewardFunction import BaseRewardFunction, DefaultRewardFunction


class RMFSEnv(ParallelEnv):
    """
    MAS-RMFS 多智能体 PettingZoo 并行环境。

    参数
    ----------
    config_path : str, optional
        JSON 配置文件路径。默认使用 Config/default_config.json。
    obs_encoder : BaseObservationEncoder, optional
        观测编码器。默认使用 GridObservationEncoder。
    action_decoder : BaseActionDecoder, optional
        动作解码器。默认使用 TaskAssignmentDecoder。
    reward_fn : BaseRewardFunction, optional
        奖励函数。默认使用 DefaultRewardFunction。
    max_ticks : int, optional
        回合最大 tick 数（截断条件）。默认 500。
    """

    metadata = {
        "render_modes": ["human"],
        "name": "rmfs_v0",
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        obs_encoder: Optional[BaseObservationEncoder] = None,
        action_decoder: Optional[BaseActionDecoder] = None,
        reward_fn: Optional[BaseRewardFunction] = None,
        max_ticks: int = 500,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # 配置路径
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "Config", "default_config.json",
            )
        self._config_path = config_path
        self._config = load_config(config_path)
        self._max_ticks = max_ticks
        self.render_mode = render_mode

        # 可插拔组件
        self._obs_encoder = obs_encoder or GridObservationEncoder()
        self._action_decoder = action_decoder or TaskAssignmentDecoder()
        self._reward_fn = reward_fn or DefaultRewardFunction(max_ticks=max_ticks)

        # 初始化引擎（用于推断空间维度）
        self._build_engine()

        # PettingZoo 必需属性
        num_agents = len(self._engine.world.agents)
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = list(self.possible_agents)

        # 缓存空间定义
        self._obs_space = self._obs_encoder.observation_space(self._engine.world)
        self._act_space = self._action_decoder.action_space(self._engine.world)

    def _build_engine(self):
        """从配置构建仿真引擎（不含可视化）。"""
        config = load_config(self._config_path)
        self._config = config

        # 重置全局 ID 计数器
        Task._next_id = 0
        Order._next_id = 0

        og_name, og_params = config.policies.order_generator
        ta_name, ta_params = config.policies.task_assigner
        pp_name, pp_params = config.policies.path_planner
        rp_name, rp_params = config.policies.pod_return_planner

        order_generator = get_policy("order_generator", og_name)(
            order_interval=config.simulation.order_interval,
            max_items_per_order=config.simulation.max_items_per_order,
            **og_params,
        )
        task_assigner = get_policy("task_assigner", ta_name)(**ta_params)
        path_planner = get_policy("path_planner", pp_name)(**pp_params)
        pod_return_planner = get_policy("pod_return_planner", rp_name)(**rp_params)
        task_assigner.pod_return_planner = pod_return_planner
        task_assigner.path_planner = path_planner

        self._engine = SimulationEngine(
            config=config,
            order_generator=order_generator,
            task_assigner=task_assigner,
            path_planner=path_planner,
            visualizer=None,
        )

    # ── PettingZoo API ──────────────────────────────────────────

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        """返回指定智能体的观测空间。"""
        return self._obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        """返回指定智能体的动作空间。"""
        return self._act_space

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple:
        """
        重置环境到初始状态。

        返回值
        -------
        observations : dict[str, obs]
            各智能体的初始观测。
        infos : dict[str, dict]
            各智能体的初始信息。
        """
        if seed is not None:
            np.random.seed(seed)

        # 重建引擎
        self._build_engine()
        self.agents = list(self.possible_agents)

        # 重置奖励函数
        if hasattr(self._reward_fn, "reset"):
            self._reward_fn.reset()

        # 清除空间缓存（世界可能不同）
        self.observation_space.cache_clear()
        self.action_space.cache_clear()
        self._obs_space = self._obs_encoder.observation_space(self._engine.world)
        self._act_space = self._action_decoder.action_space(self._engine.world)

        # 编码初始观测
        observations = {}
        infos = {}
        for agent_name in self.agents:
            agent_id = int(agent_name.split("_")[1])
            observations[agent_name] = self._obs_encoder.encode(
                self._engine.world, agent_id
            )
            infos[agent_name] = self._get_info()

        return observations, infos

    def step(
        self, actions: Dict[str, Any]
    ) -> tuple:
        """
        执行一个仿真 tick。

        参数
        ----------
        actions : dict[str, int]
            各智能体的动作，如 {"agent_0": 3, "agent_1": 0}。

        返回值
        -------
        observations : dict[str, obs]
        rewards : dict[str, float]
        terminations : dict[str, bool]
        truncations : dict[str, bool]
        infos : dict[str, dict]
        """
        # 记录 step 前的已完成订单数
        prev_completed = self._engine.world.order_state.total_completed

        # 1) 解码 RL 动作 → 应用到世界状态
        self._action_decoder.decode(actions, self._engine.world)

        # 2) 执行一个仿真 tick
        self._engine._tick()

        # 3) 编码观测、计算奖励
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        done = self._reward_fn.is_done(self._engine.world)
        info = self._get_info()

        for agent_name in self.agents:
            agent_id = int(agent_name.split("_")[1])
            observations[agent_name] = self._obs_encoder.encode(
                self._engine.world, agent_id
            )
            rewards[agent_name] = self._reward_fn.compute(
                self._engine.world, prev_completed, agent_id
            )
            terminations[agent_name] = False  # 无自然终止
            truncations[agent_name] = done    # 超时截断
            infos[agent_name] = info

        # 如果回合结束，清空 agents 列表
        if done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _get_info(self) -> dict:
        """构建信息字典。"""
        ws = self._engine.world
        return {
            "tick": ws.tick,
            "completed_orders": ws.order_state.total_completed,
            "pending_orders": len(ws.order_state.get_pending_orders()),
            "in_progress_orders": len(ws.order_state.get_in_progress_orders()),
            "total_orders": ws.order_state.total_orders,
        }

    def render(self):
        """渲染（当前仅支持文本模式）。"""
        if self.render_mode == "human":
            ws = self._engine.world
            print(
                f"[Tick {ws.tick}] "
                f"Orders: {ws.order_state.total_completed}/{ws.order_state.total_orders} "
                f"Agents: {[a.status.name for a in ws.agents]}"
            )

    def close(self):
        """清理环境资源。"""
        pass
