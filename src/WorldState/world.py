"""
世界状态模块
=================
聚合所有仿真状态组件的门面。
"""

import random
from typing import List, Tuple

from Config.config_loader import SimulationConfig
from WorldState.map_state import MapState, CellType
from WorldState.agent_state import AgentState
from WorldState.order_state import OrderState
from WorldState.task_state import TaskState
from WorldState.pod_state import PodState, Pod
from WorldState.station_state import StationState


class WorldState:
    """
    聚合世界状态 — 仿真的唯一事实来源。

    属性
    ----------
    tick : int
        当前仿真 tick。
    map_state : MapState
        仓库网格地图。
    agents : list[AgentState]
        所有机器人智能体。
    order_state : OrderState
        所有客户订单。
    task_state : TaskState
        所有原子任务。
    pod_state : PodState
        所有货架（Pod）。
    config : SimulationConfig
        已加载的仿真配置。
    """

    def __init__(self, config: SimulationConfig):
        self.tick: int = 0
        self.config = config

        # 初始化地图
        self.map_state = MapState(config.map)

        # 初始化站点队列区域 (before agent placement so zone cells are excluded)
        self.station_state = StationState(config, self.map_state)

        # 在起始位置初始化智能体
        self.agents: List[AgentState] = []
        starts = self._resolve_agent_starts(config)
        for i in range(config.robots.num_robots):
            self.agents.append(AgentState(agent_id=i, start_position=starts[i]))

        # 初始化订单和任务容器
        self.order_state = OrderState()
        self.task_state = TaskState()

        # One-tick traffic telemetry is part of the observable simulator
        # state used by the Lyapunov data collector.  Paths rejected by the
        # movement resolver are cleared immediately, so these counters keep
        # realised conflicts available to the post-step snapshot.
        self.traffic_vertex_conflicts_this_tick: int = 0
        self.traffic_swap_conflicts_this_tick: int = 0

        # 通过 PodInitializer 策略初始化货架
        from Policies.policy_registry import get_policy
        self.pod_state = PodState()
        pi_name = config.policies.pod_initializer[0]
        pi_params = config.policies.pod_initializer[1] if len(config.policies.pod_initializer) > 1 else {}
        PodInitializerCls = get_policy("pod_initializer", pi_name)
        pod_initializer = PodInitializerCls(**pi_params)
        pod_initializer.initialize_pods(self)

    def _resolve_agent_starts(self, config):
        """Return a list of agent start positions of length num_robots.

        Two modes:
        - random_starts=False (default): use config.robots.starts in order;
          pad with free cells for any agent beyond the list.
        - random_starts=True: sample positions from FREE walkable cells
          (excluding stations, pod homes, and zone cells), using the
          GLOBAL random module — so seeding ``random`` upstream gives
          reproducible starts.
        """
        n = config.robots.num_robots

        # Collect positions to avoid when auto-placing agents
        pod_homes = set(self.map_state.pod_home_positions)
        handoff = set()
        for sq in self.station_state.stations.values():
            if sq.entry_position:
                handoff.add(sq.entry_position)
            if sq.exit_position:
                handoff.add(sq.exit_position)

        if not config.robots.random_starts:
            explicit_starts = list(config.robots.starts)
            if len(explicit_starts) >= n:
                return explicit_starts[:n]

            used = set(tuple(s) for s in explicit_starts)
            free_cells = [
                (r, c)
                for r in range(self.map_state.rows)
                for c in range(self.map_state.cols)
                if self.map_state.grid[r][c] == CellType.FREE
                and (r, c) not in used
                and (r, c) not in pod_homes
                and (r, c) not in handoff
            ]
            random.shuffle(free_cells)

            needed = n - len(explicit_starts)
            if len(free_cells) < needed:
                raise ValueError(
                    f"地图上可用的 FREE 格子 ({len(free_cells)}) 不足以放置 "
                    f"{needed} 个缺少起始位置的机器人。请增大地图或减少机器人数量。"
                )
            auto_starts = free_cells[:needed]
            starts = []
            for i in range(n):
                if i < len(explicit_starts):
                    starts.append(explicit_starts[i])
                else:
                    starts.append(auto_starts[i - len(explicit_starts)])
            return starts

        # random_starts=True
        free_cells = [
            (r, c)
            for r in range(self.map_state.rows)
            for c in range(self.map_state.cols)
            if self.map_state.grid[r][c] == CellType.FREE
            and (r, c) not in pod_homes
            and (r, c) not in handoff
        ]
        if len(free_cells) < n:
            raise ValueError(
                f"random_starts=True but only {len(free_cells)} free cells "
                f"available for {n} robots."
            )
        return random.sample(free_cells, n)

    def advance_tick(self):
        """递增仿真 tick 计数器。"""
        self.tick += 1

    def get_agent(self, agent_id: int) -> AgentState:
        """按 ID 获取智能体。"""
        return self.agents[agent_id]

    def get_idle_agents(self) -> List[AgentState]:
        """返回所有当前空闲的智能体。"""
        return [a for a in self.agents if a.is_idle]

    def __repr__(self) -> str:
        return (
            f"WorldState(tick={self.tick}, agents={len(self.agents)}, "
            f"pods={self.pod_state.total_pods}, "
            f"orders={self.order_state.total_orders})"
        )
