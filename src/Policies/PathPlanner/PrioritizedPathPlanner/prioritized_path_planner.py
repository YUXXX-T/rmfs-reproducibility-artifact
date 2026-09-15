"""
Prioritized Path Planner
=========================
Multi-agent pathfinding via Prioritized Planning.

Agents are planned one at a time in priority order. Each agent uses
space-time A* to find a collision-free path, treating higher-priority
agents' planned paths as moving obstacles via a reservation table.

This eliminates most vertex conflicts (two agents on the same cell)
and edge/swap conflicts (two agents crossing the same edge in
opposite directions).

基于优先级的多智能体路径规划。

智能体按优先级顺序依次规划路径。每个智能体使用时空 A* 算法寻找无碰撞路径，并通过预留表将高优先级智能体的已规划路径视为移动障碍物。

这种方法消除了大多数顶点冲突（两个智能体占据同一单元格）和边/交换冲突（两个智能体沿相反方向穿越同一条边）。
"""

import heapq
from typing import List, Set, Tuple, Dict

from Policies.PathPlanner.base_path_planner import BasePathPlanner
from WorldState.agent_state import AgentStatus


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class PrioritizedPathPlanner(BasePathPlanner):
    """
    Prioritized Planning for multi-agent pathfinding.

    Plans paths sequentially — one agent at a time.  Each agent finds
    a collision-free path using **space-time A***, avoiding cells and
    timesteps already reserved by higher-priority agents.

    Priority is determined by the order the engine calls ``plan()``:
    the first agent planned has the highest priority, the second must
    route around the first, etc.

    多智能体路径规划的优先级规划方法

    该方法按顺序（即逐个智能体）进行路径规划。每个智能体利用**时空 A*** 算法来寻找一条无碰撞路径，同时避开已被更高优先级智能体预占的网格单元和时间步。 
    智能体的优先级由规划引擎调用 ``plan()`` 方法的顺序决定：首个被规划的智能体拥有最高优先级；第二个智能体在规划时必须绕开第一个智能体；以此类推。

    Features
    --------
    * **顶点冲突 avoidance** — two agents may not occupy the
      same cell at the same timestep.
    * **Edge (swap) conflict avoidance** — two agents may not swap
      positions in a single timestep.
    * **Wait actions** — an agent can stay in place for one timestep
      to let another agent pass.

    参数
    ----------
    max_horizon : int
        Maximum number of timesteps the space-time search will
        explore.  Limits computation time.  Default 100.
    goal_reserve : int
        Number of extra timesteps to reserve the goal position after
        arrival, preventing later agents from occupying it.  Default 20.
    """

    def __init__(self, max_horizon: int = 100, goal_reserve: int = 20):
        self.max_horizon = max_horizon
        self.goal_reserve = goal_reserve

        # Internal state — reset each tick
        self._vertex_res: Set[Tuple[int, int, int]] = set()
        self._edge_res: Set[Tuple[int, int, int, int, int]] = set()
        self._last_tick: int = -1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plan(
        self,
        agent,
        goal: Tuple[int, int],
        world_state,
        extra_blocked=None,
    ) -> List[Tuple[int, int]]:
        """
        Compute a collision-free path using space-time A*.

        On the first call of each tick the reservation table is rebuilt
        from agents that already have paths.  Each subsequent call adds
        the newly planned path to the table, so lower-priority agents
        automatically avoid higher-priority ones.

        利用时空 A* 算法计算一条无碰撞路径。 

        在每个时间步（tick）的首次调用中，系统会根据已规划好路径的智能体重建预留表。 
        随后的每一次调用都会将新规划的路径添加至该表中，从而确保低优先级智能体能够自动避让高优先级智能体。

        返回值
        -------
        list[tuple[int, int]]
            Path from start to goal (excluding start), possibly with
            "wait" steps (same position repeated).  Empty if no path.
        """
        tick = world_state.tick

        # Reset reservation table once per tick
        if tick != self._last_tick:
            self._last_tick = tick
            self._vertex_res = set()
            self._edge_res = set()
            self._pre_reserve_existing_paths(world_state)

        start = agent.position
        if start == goal:
            return []

        map_state = world_state.map_state

        # Static obstacles: pod positions when carrying a pod + idle agents
        _ZONE_STATUSES = (AgentStatus.QUEUING, AgentStatus.DELIVERING, AgentStatus.EXITING)
        idle_others = {
            ag.position
            for ag in world_state.agents
            if not ag.has_path and not ag.is_waiting
            and ag.status not in _ZONE_STATUSES
        } - {start}

        # WAITING_ASSIGNED has no finite countdown: the robot is parked until
        # station admission promotes it.  Treat its current cell as a static
        # obstacle for other routes, while keeping ordinary action waits on
        # the existing space-time reservation path.
        admission_waiting = {
            ag.position
            for ag in world_state.agents
            if ag.status == AgentStatus.WAITING_ASSIGNED
        } - {start}

        zone_occupied = {
            ag.position
            for ag in world_state.agents
            if ag.status in _ZONE_STATUSES
        } - {start}

        if agent.carried_pod_id is not None:
            static_blocked = {
                pod.current_position
                for pod in world_state.pod_state.pods.values()
                if not pod.is_carried
            } | idle_others | admission_waiting | zone_occupied
        else:
            static_blocked = idle_others | admission_waiting | zone_occupied

        if extra_blocked:
            static_blocked = static_blocked | extra_blocked
        static_blocked = static_blocked - {goal}

        # Plan using space-time A*
        path = self._space_time_astar(start, goal, map_state, static_blocked)

        # Reserve the new path so lower-priority agents avoid it
        if path:
            self._reserve_path(start, path)

        return path

    # ------------------------------------------------------------------
    # Reservation table management
    # ------------------------------------------------------------------

    def _pre_reserve_existing_paths(self, world_state) -> None:
        """
        Populate the reservation table with all agents' current and
        planned future positions, including agents waiting at destinations.
        在预留表中填入所有智能体的当前位置及计划的未来位置，包括正在目的地等待的智能体。
        """
        _ZONE_STATUSES = (AgentStatus.QUEUING, AgentStatus.DELIVERING, AgentStatus.EXITING)
        for agent in world_state.agents:
            pos = agent.position
            # Reserve current position at time-step 0
            self._vertex_res.add((pos[0], pos[1], 0))

            if agent.status in _ZONE_STATUSES:
                for t in range(1, self.max_horizon + 1):
                    self._vertex_res.add((pos[0], pos[1], t))
                continue

            if agent.status == AgentStatus.WAITING_ASSIGNED:
                # Admission wait has no known release tick.  Reserve the
                # parked cell for the complete planning horizon so a newly
                # planned route cannot pass through the waiting robot.
                for t in range(1, self.max_horizon + 1 + self.goal_reserve):
                    self._vertex_res.add((pos[0], pos[1], t))
            elif agent.is_waiting:
                # Agent is parked at this position for wait_ticks more ticks
                # plus a buffer so new paths don't target this cell too early
                # 智能体停驻在此位置，并将继续等待 wait_ticks 个时间步
                # 此外还额外预留了一段缓冲时间，以防止新路径过早地将此单元格选为目标
                for t in range(1, agent.wait_ticks + 1 + self.goal_reserve):
                    self._vertex_res.add((pos[0], pos[1], t))

            elif agent.has_path:
                prev = pos
                remaining = len(agent.path) - agent.path_index
                for i in range(remaining):
                    step = i + 1
                    nxt = agent.path[agent.path_index + i]
                    self._vertex_res.add((nxt[0], nxt[1], step))
                    # Edge reservation (prevents swaps)
                    self._edge_res.add(
                        (nxt[0], nxt[1], prev[0], prev[1], step)
                    )
                    prev = nxt
                # Reserve the final position for extra timesteps
                for extra in range(remaining + 1, remaining + 1 + self.goal_reserve):
                    self._vertex_res.add((prev[0], prev[1], extra))

            else:
                # Agent has no path and is not waiting — sitting still
                # (e.g. just finished PICK/RETURN, stuck at entry, etc.).
                # Reserve for full horizon; table rebuilds each tick so
                # the reservation disappears as soon as the agent moves.
                for t in range(1, self.max_horizon + 1):
                    self._vertex_res.add((pos[0], pos[1], t))

    def _reserve_path(
        self,
        start: Tuple[int, int],
        path: List[Tuple[int, int]],
    ) -> None:
        """Reserve vertex and edge reservations for a newly planned path."""
        prev = start
        for step, pos in enumerate(path, start=1):
            self._vertex_res.add((pos[0], pos[1], step))
            self._edge_res.add((pos[0], pos[1], prev[0], prev[1], step))
            prev = pos
        # Reserve goal after arrival
        for extra in range(len(path) + 1, len(path) + 1 + self.goal_reserve):
            self._vertex_res.add((path[-1][0], path[-1][1], extra))

    # ------------------------------------------------------------------
    # Space-time A*
    # ------------------------------------------------------------------

    def _space_time_astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        map_state,
        static_blocked: Set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        A* search in the (row, col, timestep) state space.

        Actions: move to a cardinal neighbour **or** wait in place.
        Avoids vertex and edge reservations from the reservation table.

        在 (行, 列, 时间步) 状态空间中执行 A* 搜索。 
        动作：移动至相邻的四个正交方向位置 **或** 停留在原地。 
        避开预留表中记录的顶点和边预留。
        """
        counter = 0
        open_set: list = []
        heapq.heappush(
            open_set, (_manhattan(start, goal), counter, start[0], start[1], 0)
        )
        counter += 1

        g_score: Dict[Tuple[int, int, int], int] = {
            (start[0], start[1], 0): 0
        }
        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

        while open_set:
            _, _, r, c, t = heapq.heappop(open_set)

            if (r, c) == goal:
                # Reconstruct path (excluding start)
                path: list = []
                node = (r, c, t)
                while node != (start[0], start[1], 0):
                    path.append((node[0], node[1]))
                    node = came_from[node]
                path.reverse()
                return path

            # Enforce time horizon
            if t >= self.max_horizon:
                continue

            next_t = t + 1

            # Candidate moves: 4 cardinal neighbours + wait-in-place
            candidates = list(map_state.get_neighbors(r, c))
            candidates.append((r, c))  # wait action

            for nr, nc in candidates:
                # Static obstacles
                if (nr, nc) in static_blocked:
                    continue

                # 顶点冲突 check
                if (nr, nc, next_t) in self._vertex_res:
                    continue

                # When arriving at the goal, the agent will stay there
                # (for pickup/delivery/return processing).  Ensure the
                # goal cell is clear for the full occupancy window so we
                # don't conflict with another agent that arrives later.
                # 当智能体抵达目标位置时，它将停留在此处 （以便进行取货/送货/归还等处理）。因此，必须确保
                # 目标单元格在整个占用时间窗口内均保持空闲，以避免与随后抵达的其他智能体发生冲突。
                if (nr, nc) == goal:
                    goal_blocked = False
                    for ft in range(next_t + 1, next_t + 1 + self.goal_reserve):
                        if (nr, nc, ft) in self._vertex_res:
                            goal_blocked = True
                            break
                    if goal_blocked:
                        continue

                # Edge (swap) conflict check — look for reverse direction
                if (r, c, nr, nc, next_t) in self._edge_res:
                    continue

                tentative_g = g_score[(r, c, t)] + 1
                state = (nr, nc, next_t)

                if tentative_g < g_score.get(state, float("inf")):
                    came_from[state] = (r, c, t)
                    g_score[state] = tentative_g
                    f = tentative_g + _manhattan((nr, nc), goal)
                    heapq.heappush(open_set, (f, counter, nr, nc, next_t))
                    counter += 1

        # No path found within the time horizon
        return []
