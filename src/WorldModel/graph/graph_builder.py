"""
Graph Builder Module
====================
Converts WorldState into graph tensors for the ST-GNN world model.

"""

import copy
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch

from WorldState.world import WorldState
from WorldState.map_state import MapState, CellType
from WorldState.agent_state import AgentStatus
from WorldState.task_state import TaskType, TaskStatus


def build_static_graph(map_state: MapState):
    """Build a static graph from the warehouse grid.

    Returns
    -------
    edge_index : torch.LongTensor, shape (2, E)
    node_map : dict[(row, col) -> node_idx]
    inv_node_map : dict[node_idx -> (row, col)]
    local_capacity : list[float], length N
    bottleneck_score : list[float], length N (normalized)
    node_type_arr : list[float], length N
    adj : dict[int, list[int]], adjacency list
    """
    node_map: Dict[Tuple[int, int], int] = {}
    inv_node_map: Dict[int, Tuple[int, int]] = {}
    idx = 0
    for r in range(map_state.rows):
        for c in range(map_state.cols):
            if map_state.grid[r][c] != CellType.OBSTACLE:
                node_map[(r, c)] = idx # node_map 对地图的(r,c)坐标重新进行编号，编号从0开始
                inv_node_map[idx] = (r, c)
                idx += 1

    num_nodes = len(node_map)
    src_list, dst_list = [], []
    # adj 为 (Adjacency List) 邻接表
    adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}

    for (r, c), nid in node_map.items():
        for nr, nc in map_state.get_neighbors(r, c):
            if (nr, nc) in node_map:
                nbr_id = node_map[(nr, nc)]
                src_list.append(nid)
                dst_list.append(nbr_id)
                adj[nid].append(nbr_id)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # 节点 i 的邻居数量（加上 1.0 是加上了自身 / 自环的容量）
    local_capacity = [float(len(adj[i])) + 1.0 for i in range(num_nodes)]

    bottleneck_score = _betweenness_centrality(num_nodes, adj)
    max_b = max(bottleneck_score) if bottleneck_score else 0.0
    if max_b > 0:
        bottleneck_score = [b / max_b for b in bottleneck_score] # 归一化到 [0, 1]

    _type_map = {
        CellType.FREE: 0.0,
        CellType.STATION: 1.0,
        CellType.POD_HOME: 2.0,
    }
    node_type_arr = [
        _type_map.get(map_state.grid[inv_node_map[i][0]][inv_node_map[i][1]], 0.0)
        for i in range(num_nodes)
    ]

    # edge_index 为边列表, node_map 和 inv_node_map 是节点索引和坐标的双向映射, local_capacity 是每个节点的局部容量（邻居数量 + 1）, bottleneck_score 是每个节点的瓶颈得分（基于近似介数中心性计算并归一化）, node_type_arr 是每个节点的类型编码（0=free, 1=station, 2=pod_home）
    return edge_index, node_map, inv_node_map, local_capacity, bottleneck_score, node_type_arr, adj


def _betweenness_centrality(num_nodes: int, adj: Dict[int, List[int]]) -> List[float]:
    """
    Approximate betweenness centrality using BFS from sampled sources. 基于采样源节点的广度优先搜索近似介数中心性。
      介数中心性衡量的是：一个节点在多大程度上扮演了图中其他节点相互连通的"桥梁"。
      在仓库中，这些高得分节点通常是主干道、十字路口、或者连接两个区域的唯一通道。
    """
    # bc 数组就是最后输出的全局累加器，数值越高说明这个节点越可能成为瓶颈。
    bc = [0.0] * num_nodes
    max_sources = min(num_nodes, 50)
    step = max(1, num_nodes // max_sources)
    sources = list(range(0, num_nodes, step))[:max_sources]

    for s in sources:
        dist = [-1] * num_nodes
        dist[s] = 0
        sigma = [0.0] * num_nodes
        sigma[s] = 1.0
        pred: List[List[int]] = [[] for _ in range(num_nodes)]
        queue = deque([s])
        order = []

        # 前向 BFS（找最短路径） 计算从起点 s 到图中所有节点的最短路径，并记录层级关系。
        """
        dist[w]：记录从起点 s 到 w 的最短距离。
        sigma[w]：记录从起点 s 到 w 有几条不同的最短路径（因为格子图往往有多条长度相同的最短路）。
         - 如果 dist[w] == dist[v] + 1，说明从 v 走到 w 也是最短路，那么 w 的路径数就要加上 v 的路径数（sigma[w] += sigma[v]）。
        pred[w]：记录在最短路径树上，w 的前驱节点是谁（可能有多个）。
        order：记录 BFS 访问节点的顺序，必定是按距离从小到大排列的，方便后续逆序回溯。
        """
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1: # 如果 w 还没有被访问过，或者 w 已经被访问过但当前路径也是最短路径，那么都要更新 sigma 和 pred。
                    sigma[w] += sigma[v]
                    pred[w].append(v) # 记录 w 的前驱节点 v，后续回溯时需要知道 w 是通过哪些节点 v 来达到的。v 都只与 w 相差 1 的距离。

        delta = [0.0] * num_nodes
        #  Brandes算法核心: 后向回溯（累加介数得分）
        """
        从距离最远的边缘节点开始往回倒推 (reversed(order))
        v 是 w 的前一站 (for v in pred[w])
        经过 w 的所有车流（记为 1.0 + delta[w]），都要按比例分摊给它的上游路口 v。分摊比例就是 sigma[v] / sigma[w]（即 v 承担了多少条去往 w 的路线）。
        除去起点 s 本身 (if w != s:)，把当前起点的回传得分累加到总得分里 (bc[w] += delta[w])。
        delta_{s}(v) = Σ_{w ∈ pred(v)} {sigma_{sv}/{sigma_{s w}} * (1 + delta_{s}(w))
        """
        for w in reversed(order):
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]

    return bc


def extract_node_features(
    world: WorldState,
    node_map: Dict[Tuple[int, int], int],
    local_capacity: List[float],
    bottleneck_score: List[float],
    node_type_arr: List[float],
    adj: Dict[int, List[int]],
    flow_counter: Optional[Dict[int, float]] = None,
    reservation_window: int = 1,
) -> torch.Tensor:
    """Extract 10-channel node features from current WorldState.

    Channels: self_occupancy, local_density, local_wait_pressure,
              local_blocked_pressure, reservation_pressure,
              recent_flow_pressure, pod_pressure, station_queue,
              bottleneck_score, node_type
    """
    N = len(node_map)
    features = [[0.0] * 10 for _ in range(N)]

    for i in range(N):
        features[i][8] = bottleneck_score[i]
        features[i][9] = node_type_arr[i] / 2.0

    robot_counts = [0.0] * N
    wait_counts = [0.0] * N
    blocked_counts = [0.0] * N

    for agent in world.agents:
        nid = node_map.get(agent.position)
        if nid is not None:
            robot_counts[nid] += 1.0
            if (agent.status != AgentStatus.IDLE
                    and agent.stuck_this_tick
                    and not agent.is_waiting):
                wait_counts[nid] += 1.0
            if agent.status != AgentStatus.IDLE and agent.plan_failed_streak >= 2:
                blocked_counts[nid] += 1.0

    W = max(reservation_window, 1)
    for i in range(N):
        cap = local_capacity[i]
        features[i][0] = min(robot_counts[i], 1.0)
        closed_nbrs = [i] + adj[i]
        local_robot = sum(robot_counts[u] for u in closed_nbrs)
        local_wait = sum(wait_counts[u] for u in closed_nbrs)
        local_blocked = sum(blocked_counts[u] for u in closed_nbrs)
        features[i][1] = local_robot / cap
        features[i][2] = local_wait / cap
        features[i][3] = local_blocked / cap

    reservation_counts = [0.0] * N
    for agent in world.agents:
        if agent.has_path:
            remaining = agent.path[agent.path_index: agent.path_index + W]
            for pos in remaining:
                nid = node_map.get(tuple(pos))
                if nid is not None:
                    reservation_counts[nid] += 1.0
    for i in range(N):
        features[i][4] = min(1.0, reservation_counts[i] / (W * local_capacity[i]))

    if flow_counter:
        for nid, count in flow_counter.items():
            if 0 <= nid < N:
                features[nid][5] = count / local_capacity[nid]

    pod_values = [0.0] * N
    for pod in world.pod_state.pods.values():
        nid = node_map.get(pod.current_position)
        if nid is not None:
            demand_weight = sum(pod.sku_inventory.values()) if pod.sku_inventory else 0
            pod_values[nid] += max(1.0, demand_weight / 50.0)
    max_pod = max(pod_values) if pod_values else 0.0
    if max_pod > 0:
        for i in range(N):
            features[i][6] = pod_values[i] / max_pod

    queue_values = [0.0] * N
    for sid, pos in world.map_state.station_positions.items():
        nid = node_map.get(pos)
        if nid is not None:
            queue_len = 0
            for order in world.order_state.get_in_progress_orders():
                if order.station_id == sid:
                    queue_len += 1
            queue_values[nid] = float(queue_len)
    max_q = max(queue_values) if queue_values else 0.0
    if max_q > 0:
        for i in range(N):
            features[i][7] = queue_values[i] / max_q

    return torch.tensor(features, dtype=torch.float32)


def extract_edge_features(
    edge_index: torch.LongTensor,
    node_map: Dict[Tuple[int, int], int],
    inv_node_map: Dict[int, Tuple[int, int]],
    local_capacity: List[float],
    world: WorldState,
    adj: Dict[int, List[int]],
    edge_flow_counter: Dict[Tuple[int, int], float],
    reservation_window: int = 1,
) -> torch.Tensor:
    """Extract 6-channel edge features.

    Channels:
      0  physical_distance        — always 1.0 (grid)
      1  topological_capacity     — static topology capacity, max-normalized
      2  pod_aware_pressure       — pod blocking context at endpoints
      3  recent_edge_flow         — decayed historical traversal, max-normalized
      4  reservation_load         — directional future-W-step reservation / W
      5  opposite_reservation_load — reservation_load of reverse edge
    """
    W = max(1, reservation_window)
    E = edge_index.shape[1]
    features = [[0.0] * 6 for _ in range(E)]

    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()

    edge_lookup: Dict[Tuple[int, int], int] = {}

    # --- dim 0: physical_distance, dim 1: topological_capacity (raw) ---
    for eidx in range(E):
        s, d = src_list[eidx], dst_list[eidx]
        features[eidx][0] = 1.0
        features[eidx][1] = (local_capacity[s] + local_capacity[d]) / 2.0
        edge_lookup[(s, d)] = eidx

    max_edge_topo_cap = max(f[1] for f in features) if features else 1.0
    if max_edge_topo_cap > 0:
        for eidx in range(E):
            features[eidx][1] /= max_edge_topo_cap

    # --- dim 2: pod_aware_pressure ---
    pod_blocked_positions: set = set()
    for pod in world.pod_state.pods.values():
        if not pod.is_carried:
            pod_blocked_positions.add(pod.current_position)

    num_nodes = len(node_map)
    pod_blocked = [False] * num_nodes
    for nid in range(num_nodes):
        if inv_node_map[nid] in pod_blocked_positions:
            pod_blocked[nid] = True

    loaded_local_pressure = [0.0] * num_nodes
    for nid in range(num_nodes):
        if pod_blocked[nid]:
            loaded_local_pressure[nid] = 0.0
        else:
            free_nbrs = sum(1 for u in adj[nid] if not pod_blocked[u])
            loaded_local_pressure[nid] = 1.0 + free_nbrs

    for eidx in range(E):
        s, d = src_list[eidx], dst_list[eidx]
        if pod_blocked[d]:
            features[eidx][2] = 0.0
        else:
            raw = (loaded_local_pressure[s] + loaded_local_pressure[d]) / 2.0
            features[eidx][2] = raw / max_edge_topo_cap if max_edge_topo_cap > 0 else 0.0

    # --- dim 3: recent_edge_flow ---
    for eidx in range(E):
        s, d = src_list[eidx], dst_list[eidx]
        features[eidx][3] = edge_flow_counter.get((s, d), 0.0)

    max_flow = max(f[3] for f in features) if features else 0.0
    if max_flow > 0:
        for eidx in range(E):
            features[eidx][3] /= max_flow

    # --- dim 4: reservation_load (directional, u->v, future W steps) ---
    for agent in world.agents:
        if agent.has_path:
            future_path = agent.path[agent.path_index: agent.path_index + W]
            prev_pos = agent.position
            for step in future_path:
                next_pos = tuple(step)
                prev_nid = node_map.get(prev_pos)
                next_nid = node_map.get(next_pos)
                if prev_nid is not None and next_nid is not None:
                    eidx = edge_lookup.get((prev_nid, next_nid))
                    if eidx is not None:
                        features[eidx][4] += 1.0
                prev_pos = next_pos

    for eidx in range(E):
        features[eidx][4] /= W

    # --- dim 5: opposite_reservation_load ---
    for eidx in range(E):
        s, d = src_list[eidx], dst_list[eidx]
        reverse_eidx = edge_lookup.get((d, s))
        if reverse_eidx is not None:
            features[eidx][5] = features[reverse_eidx][4]

    return torch.tensor(features, dtype=torch.float32)


def extract_demand_context(
    world: WorldState,
    pending_order_scale: Optional[float] = None,
    item_scale: Optional[float] = None,
    active_order_scale: Optional[float] = None,
    total_backlog_scale: Optional[float] = None,
) -> torch.Tensor:
    """Extract demand context vector (5 + num_stations dimensions).

    Scale parameters default to robot count (pending/active/total_backlog)
    or 100 (item_scale) when not provided.
    """
    pending = world.order_state.get_pending_orders()
    in_progress = world.order_state.get_in_progress_orders()

    num_robots = max(len(world.agents), 1)

    if pending_order_scale is None:
        pending_order_scale = float(num_robots)
    pending_order_scale = max(pending_order_scale, 1.0)
    pending_pressure = len(pending) / pending_order_scale

    total_items = 0.0
    for o in pending:
        total_items += sum(o.sku_demands.values())
    if item_scale is None:
        item_scale = 100.0
    item_scale = max(item_scale, 1.0)
    backlog_pressure = total_items / item_scale

    if active_order_scale is None:
        active_order_scale = float(num_robots)
    active_order_scale = max(active_order_scale, 1.0)
    active_order_pressure = len(in_progress) / active_order_scale

    num_stations = len(world.map_state.station_positions)
    raw_station_backlog = [0.0] * num_stations
    station_ids = sorted(world.map_state.station_positions.keys())
    sid_to_idx = {sid: i for i, sid in enumerate(station_ids)}
    for o in list(pending) + list(in_progress):
        idx = sid_to_idx.get(o.station_id)
        if idx is not None:
            raw_station_backlog[idx] += 1.0

    if total_backlog_scale is None:
        total_backlog_scale = float(num_robots)
    total_backlog_scale = max(total_backlog_scale, 1.0)
    total_station_backlog = sum(raw_station_backlog) / total_backlog_scale

    skew = 0.0
    sb_sum = sum(raw_station_backlog)
    if num_stations > 1 and sb_sum > 0:
        probs = [x / sb_sum for x in raw_station_backlog if x > 0]
        entropy = -sum(p * math.log(p + 1e-8) for p in probs)
        max_entropy = math.log(num_stations)
        skew = 1.0 - entropy / (max_entropy + 1e-8)

    max_sb = max(raw_station_backlog) if raw_station_backlog else 0.0
    if max_sb > 0:
        station_backlog = [x / max_sb for x in raw_station_backlog]
    else:
        station_backlog = [0.0] * num_stations

    context = [
        pending_pressure, backlog_pressure, active_order_pressure,
        total_station_backlog, skew,
    ] + station_backlog
    return torch.tensor(context, dtype=torch.float32)


def _shortest_path_nodes(
    start: Tuple[int, int],
    end: Tuple[int, int],
    map_state: MapState,
    node_map: Dict[Tuple[int, int], int],
) -> List[int]:
    """BFS shortest path returning list of node indices."""
    if start == end:
        nid = node_map.get(start)
        return [nid] if nid is not None else []

    visited = {start}
    parent = {start: None}
    queue = deque([start])

    while queue:
        pos = queue.popleft()
        if pos == end:
            break
        for nbr in map_state.get_neighbors(pos[0], pos[1]):
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = pos
                queue.append(nbr)

    if end not in parent:
        return []

    path_nodes = []
    cur = end
    while cur is not None:
        nid = node_map.get(cur)
        if nid is not None:
            path_nodes.append(nid)
        cur = parent[cur]
    path_nodes.reverse()
    return path_nodes


def _get_path_nodes(start, end, map_state, node_map, path_planner=None,
                    world=None, carrying_pod_id=None, agent_id=None):
    """Get path as node indices. Uses planner if provided, else BFS."""
    if start == end:
        nid = node_map.get(start)
        return [nid] if nid is not None else []

    if path_planner is not None and world is not None:
        if not hasattr(path_planner, "plan"):
            import warnings
            warnings.warn("path_planner has no plan() method, falling back to BFS")
            return _shortest_path_nodes(start, end, map_state, node_map)

        from WorldState.agent_state import AgentState
        temp_agent = AgentState(
            agent_id=agent_id if agent_id is not None else -1,
            start_position=start,
        )
        if carrying_pod_id is not None:
            temp_agent.carried_pod_id = carrying_pod_id
        try:
            raw_path = path_planner.plan(temp_agent, end, world)
        except Exception:
            import warnings
            warnings.warn("path_planner.plan() raised exception, falling back to BFS")
            return _shortest_path_nodes(start, end, map_state, node_map)

        if raw_path:
            nodes = []
            nid = node_map.get(start)
            if nid is not None:
                nodes.append(nid)
            for pos in raw_path:
                nid = node_map.get(tuple(pos))
                if nid is not None:
                    nodes.append(nid)
            return nodes
        return []

    return _shortest_path_nodes(start, end, map_state, node_map)


# ------------------------------------------------------------------
# Preview leg computation with planner isolation
# ------------------------------------------------------------------

def _is_single_step_planner(planner) -> bool:
    return hasattr(planner, '_last_tick') and hasattr(planner, '_decided')


def _clear_collection(obj):
    if isinstance(obj, (dict, list, set)):
        obj.clear()


def _reset_planner_transient_state(planner):
    if hasattr(planner, '_last_tick'):
        planner._last_tick = -1
    for attr in ('_decided', '_next_occupied', '_undecided', '_goals',
                 '_agents', '_cur_pos_index'):
        if hasattr(planner, attr):
            _clear_collection(getattr(planner, attr))
    for attr in ('_cached_paths', '_vertex_res', '_edge_res'):
        if hasattr(planner, attr):
            _clear_collection(getattr(planner, attr))


def _freeze_non_target_agents(world, target_agent_id):
    for agent in world.agents:
        if agent.agent_id == target_agent_id:
            continue
        agent.status = AgentStatus.IDLE
        agent.clear_path()
        agent.wait_ticks = 0
        agent.assigned_task_id = None


def _rollout_single_leg(cloned_world, cloned_planner, agent, goal,
                        carrying_pod_id, node_map, max_steps):
    start = agent.position
    if start == goal:
        nid = node_map.get(start)
        return [nid] if nid is not None else []

    _reset_planner_transient_state(cloned_planner)

    positions = [start]
    for _ in range(max_steps):
        if agent.position == goal:
            break
        try:
            path = cloned_planner.plan(agent, goal, cloned_world)
        except Exception:
            cloned_world.advance_tick()
            continue
        if path:
            next_pos = tuple(path[0])
            if next_pos != agent.position:
                agent.previous_position = agent.position
                agent.position = next_pos
                if carrying_pod_id is not None:
                    pod = cloned_world.pod_state.get_pod(carrying_pod_id)
                    if pod:
                        pod.current_position = next_pos
            positions.append(agent.position)
        cloned_world.advance_tick()

    nodes = []
    for pos in positions:
        nid = node_map.get(pos)
        if nid is not None:
            nodes.append(nid)
    return nodes


def _direct_plan_leg(planner, start, end, world, node_map,
                     carrying_pod_id, agent_id):
    if start == end:
        nid = node_map.get(start)
        return [nid] if nid is not None else []

    from WorldState.agent_state import AgentState
    temp_agent = AgentState(
        agent_id=agent_id if agent_id is not None else -1,
        start_position=start,
    )
    if carrying_pod_id is not None:
        temp_agent.carried_pod_id = carrying_pod_id

    _reset_planner_transient_state(planner)
    try:
        raw_path = planner.plan(temp_agent, end, world)
    except Exception:
        return _shortest_path_nodes(start, end, world.map_state, node_map)

    if not raw_path:
        return _shortest_path_nodes(start, end, world.map_state, node_map)

    nodes = []
    nid = node_map.get(start)
    if nid is not None:
        nodes.append(nid)
    for pos in raw_path:
        nid = node_map.get(tuple(pos))
        if nid is not None:
            nodes.append(nid)
    return nodes


def compute_preview_legs(
    assignment: dict,
    map_state: "MapState",
    node_map: Dict[Tuple[int, int], int],
    path_planner=None,
    world: Optional["WorldState"] = None,
    max_steps_per_leg: int = 200,
) -> Tuple[List[int], List[int], List[int]]:
    """Compute path node lists for all three legs of a candidate assignment.

    Always returns three legs. When *path_planner* and *world* are both
    provided, uses deepcopy isolation; otherwise falls back to BFS.
    For single-step planners (PIBT), performs iterative rollout per leg.
    """
    if str(assignment.get("action_type", "")).lower() == "no_assign":
        return ([], [], [])

    robot_start = assignment["robot_start"]
    pod_loc = assignment["pod_location"]
    station_loc = assignment["station_location"]
    return_loc = assignment["return_location"]
    robot_id = assignment.get("robot_id")
    pod_id = assignment.get("pod_id")

    if path_planner is None or world is None:
        leg1 = _shortest_path_nodes(robot_start, pod_loc, map_state, node_map)
        leg2 = _shortest_path_nodes(pod_loc, station_loc, map_state, node_map)
        leg3 = _shortest_path_nodes(station_loc, return_loc, map_state, node_map)
        return (leg1, leg2, leg3)

    try:
        cloned_world = copy.deepcopy(world)
    except Exception:
        leg1 = _shortest_path_nodes(robot_start, pod_loc, map_state, node_map)
        leg2 = _shortest_path_nodes(pod_loc, station_loc, map_state, node_map)
        leg3 = _shortest_path_nodes(station_loc, return_loc, map_state, node_map)
        return (leg1, leg2, leg3)

    single_step = _is_single_step_planner(path_planner)

    if single_step:
        try:
            cloned_planner = copy.deepcopy(path_planner)
        except Exception:
            leg1 = _shortest_path_nodes(robot_start, pod_loc, map_state, node_map)
            leg2 = _shortest_path_nodes(pod_loc, station_loc, map_state, node_map)
            leg3 = _shortest_path_nodes(station_loc, return_loc, map_state, node_map)
            return (leg1, leg2, leg3)

        _freeze_non_target_agents(cloned_world, robot_id)
        agent = cloned_world.get_agent(robot_id)

        # --- Leg 1: robot -> pod (no carrying) ---
        agent.position = robot_start
        agent.status = AgentStatus.MOVING_TO_POD
        agent.carried_pod_id = None
        agent.clear_path()
        agent.plan_failed_streak = 0
        leg1 = _rollout_single_leg(
            cloned_world, cloned_planner, agent, pod_loc,
            None, node_map, max_steps_per_leg,
        )

        # --- Leg 2: pod -> station (carrying) ---
        agent.position = pod_loc
        agent.status = AgentStatus.CARRYING
        agent.carried_pod_id = pod_id
        agent.clear_path()
        agent.plan_failed_streak = 0
        pod = cloned_world.pod_state.get_pod(pod_id)
        if pod:
            pod.pick_up(robot_id)
            pod.current_position = pod_loc
        leg2 = _rollout_single_leg(
            cloned_world, cloned_planner, agent, station_loc,
            pod_id, node_map, max_steps_per_leg,
        )

        # --- Leg 3: station -> return (carrying) ---
        agent.position = station_loc
        agent.status = AgentStatus.RETURNING
        agent.carried_pod_id = pod_id
        agent.clear_path()
        agent.plan_failed_streak = 0
        if pod:
            pod.current_position = station_loc
        leg3 = _rollout_single_leg(
            cloned_world, cloned_planner, agent, return_loc,
            pod_id, node_map, max_steps_per_leg,
        )

        return (leg1, leg2, leg3)

    # Multi-step planner: fresh clone per leg to avoid reservation leakage
    legs = []
    legs_config = [
        (robot_start, pod_loc, None),
        (pod_loc, station_loc, pod_id),
        (station_loc, return_loc, pod_id),
    ]
    for start, end, carrying in legs_config:
        try:
            leg_planner = copy.deepcopy(path_planner)
        except Exception:
            legs.append(_shortest_path_nodes(start, end, map_state, node_map))
            continue
        legs.append(_direct_plan_leg(
            leg_planner, start, end, cloned_world, node_map,
            carrying_pod_id=carrying, agent_id=robot_id,
        ))
    return tuple(legs)


def build_action_field(
    assignment: dict,
    world: WorldState,
    node_map: Dict[Tuple[int, int], int],
    inv_node_map: Dict[int, Tuple[int, int]],
    local_capacity: List[float],
    path_planner=None,
    node_features: Optional[torch.Tensor] = None,
    station_queue_scale: Optional[float] = None,
    order_size_scale: Optional[float] = None,
    precomputed_legs: Optional[Tuple[List[int], List[int], List[int]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build action field tensors for a candidate assignment.

    Returns (action_node (N, 8), action_global (6,)).
    """
    N = len(node_map)
    action_node = [[0.0] * 8 for _ in range(N)]

    if str(assignment.get("action_type", "")).lower() == "no_assign":
        return (
            torch.zeros((N, 8), dtype=torch.float32),
            torch.zeros(6, dtype=torch.float32),
        )

    robot_start = assignment["robot_start"]
    pod_loc = assignment["pod_location"]
    station_loc = assignment["station_location"]
    return_loc = assignment["return_location"]
    robot_id = assignment.get("robot_id")
    pod_id = assignment.get("pod_id")

    nid = node_map.get(robot_start)
    if nid is not None:
        action_node[nid][0] = 1.0
    nid = node_map.get(pod_loc)
    if nid is not None:
        action_node[nid][1] = 1.0
    nid = node_map.get(station_loc)
    if nid is not None:
        action_node[nid][2] = 1.0
    nid = node_map.get(return_loc)
    if nid is not None:
        action_node[nid][3] = 1.0

    if precomputed_legs is not None:
        path1, path2, path3 = precomputed_legs
    else:
        path1 = _get_path_nodes(
            robot_start, pod_loc, world.map_state, node_map,
            path_planner=path_planner, world=world,
            carrying_pod_id=None, agent_id=robot_id,
        )
        path2 = _get_path_nodes(
            pod_loc, station_loc, world.map_state, node_map,
            path_planner=path_planner, world=world,
            carrying_pod_id=pod_id, agent_id=robot_id,
        )
        path3 = _get_path_nodes(
            station_loc, return_loc, world.map_state, node_map,
            path_planner=path_planner, world=world,
            carrying_pod_id=pod_id, agent_id=robot_id,
        )

    for nid in path1:
        action_node[nid][4] += 1.0
    for nid in path2:
        action_node[nid][5] += 1.0
    for nid in path3:
        action_node[nid][6] += 1.0

    WEIGHT_SUM = 1.0 + 1.5 + 0.8
    for i in range(N):
        action_node[i][7] = (
            1.0 * action_node[i][4]
            + 1.5 * action_node[i][5]
            + 0.8 * action_node[i][6]
        ) / WEIGHT_SUM

    for ch in [4, 5, 6]:
        mx = max(action_node[i][ch] for i in range(N))
        if mx > 0:
            for i in range(N):
                action_node[i][ch] /= mx

    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    dist_r2p = _manhattan(robot_start, pod_loc)
    dist_p2s = _manhattan(pod_loc, station_loc)
    dist_s2r = _manhattan(station_loc, return_loc)

    station_queue = 0
    for sid, pos in world.map_state.station_positions.items():
        if pos == station_loc:
            for o in world.order_state.get_in_progress_orders():
                if o.station_id == sid:
                    station_queue += 1
            break

    if station_queue_scale is None:
        station_queue_scale = max(len(world.agents), 1)
    station_queue_scale = max(float(station_queue_scale), 1.0)

    route_nodes = list(path1)
    route_nodes += path2[1:] if path2 and route_nodes and path2[0] == route_nodes[-1] else path2
    route_nodes += path3[1:] if path3 and route_nodes and path3[0] == route_nodes[-1] else path3

    route_pressure = 0.0
    if route_nodes and node_features is not None:
        total = 0.0
        for nid in route_nodes:
            total += (0.40 * node_features[nid, 1].item()
                      + 0.25 * node_features[nid, 2].item()
                      + 0.25 * node_features[nid, 4].item()
                      + 0.10 * node_features[nid, 5].item())
        route_pressure = total / len(route_nodes)

    if order_size_scale is None:
        sim_cfg = getattr(getattr(world, "config", None), "simulation", None)
        if sim_cfg is not None:
            order_size_scale = (sim_cfg.max_items_per_order
                                * sim_cfg.max_items_per_sku)
        else:
            order_size_scale = 10.0
    order_size_scale = max(float(order_size_scale), 1.0)

    order_size = assignment.get("order_size", 1)
    max_dist = world.map_state.rows + world.map_state.cols

    action_global = torch.tensor([
        dist_r2p / max_dist,
        dist_p2s / max_dist,
        dist_s2r / max_dist,
        station_queue / station_queue_scale,
        route_pressure,
        order_size / order_size_scale,
    ], dtype=torch.float32)

    return torch.tensor(action_node, dtype=torch.float32), action_global


def build_action_edge_field(
    assignment: dict,
    edge_index: torch.LongTensor,
    node_map: Dict[Tuple[int, int], int],
    map_state,
    path_planner=None,
    world=None,
    precomputed_legs: Optional[Tuple[List[int], List[int], List[int]]] = None,
) -> torch.Tensor:
    """Build per-edge action field: (E, 4).

    Channels: action_edge_leg1, action_edge_leg2, action_edge_leg3,
              action_edge_weighted_load.
    Saved in samples but not yet input to model.
    """
    E = edge_index.shape[1]
    action_edge = [[0.0] * 4 for _ in range(E)]

    if str(assignment.get("action_type", "")).lower() == "no_assign":
        return torch.zeros((E, 4), dtype=torch.float32)

    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    edge_lookup = {}
    for eidx in range(E):
        edge_lookup[(src_list[eidx], dst_list[eidx])] = eidx

    robot_id = assignment.get("robot_id")
    pod_id = assignment.get("pod_id")

    segments = [
        (assignment["robot_start"], assignment["pod_location"], None),
        (assignment["pod_location"], assignment["station_location"], pod_id),
        (assignment["station_location"], assignment["return_location"], pod_id),
    ]

    for leg_idx, (start, end, carrying) in enumerate(segments):
        if precomputed_legs is not None:
            path_nodes = precomputed_legs[leg_idx]
        else:
            path_nodes = _get_path_nodes(
                start, end, map_state, node_map,
                path_planner=path_planner, world=world,
                carrying_pod_id=carrying, agent_id=robot_id,
            )
        for i in range(len(path_nodes) - 1):
            eidx = edge_lookup.get((path_nodes[i], path_nodes[i + 1]))
            if eidx is not None:
                action_edge[eidx][leg_idx] = 1.0

    WEIGHT_SUM = 1.0 + 1.5 + 0.8
    for eidx in range(E):
        action_edge[eidx][3] = (
            1.0 * action_edge[eidx][0]
            + 1.5 * action_edge[eidx][1]
            + 0.8 * action_edge[eidx][2]
        ) / WEIGHT_SUM

    return torch.tensor(action_edge, dtype=torch.float32)

# 提取特征，即生成监督信号
def extract_node_labels(
    world: WorldState,
    node_map: Dict[Tuple[int, int], int],
    local_capacity: List[float],
    bottleneck_score: List[float],
    adj: Dict[int, List[int]],
    reservation_window: int = 1,
) -> torch.Tensor:
    """Extract 6-channel node-level labels for supervision.

    Channels: self_occupancy, local_density, local_wait_pressure,
              local_blocked_pressure, reservation_pressure, congestion_score
    """
    N = len(node_map)
    labels = [[0.0] * 6 for _ in range(N)]

    robot_counts = [0.0] * N
    wait_counts = [0.0] * N
    blocked_counts = [0.0] * N
    reservation_counts = [0.0] * N

    W = max(reservation_window, 1)

    for agent in world.agents:
        nid = node_map.get(agent.position)
        if nid is not None:
            robot_counts[nid] += 1.0
            if (agent.status != AgentStatus.IDLE
                    and agent.stuck_this_tick
                    and not agent.is_waiting):
                wait_counts[nid] += 1.0
            if agent.status != AgentStatus.IDLE and agent.plan_failed_streak >= 2:
                blocked_counts[nid] += 1.0

        if agent.has_path:
            remaining = agent.path[agent.path_index: agent.path_index + W]
            for pos in remaining:
                rid = node_map.get(tuple(pos))
                if rid is not None:
                    reservation_counts[rid] += 1.0

    for i in range(N):
        cap = local_capacity[i]
        self_occ = min(robot_counts[i], 1.0)
        closed_nbrs = [i] + adj[i]
        local_density = sum(robot_counts[u] for u in closed_nbrs) / cap
        local_wait = sum(wait_counts[u] for u in closed_nbrs) / cap
        local_blocked = sum(blocked_counts[u] for u in closed_nbrs) / cap
        reservation = min(1.0, reservation_counts[i] / (W * cap))
        congestion = (0.30 * local_density + 0.30 * local_wait
                      + 0.20 * local_blocked + 0.20 * reservation)
        labels[i] = [self_occ, local_density, local_wait, local_blocked,
                     reservation, congestion]

    return torch.tensor(labels, dtype=torch.float32)


def extract_system_labels(
    world: WorldState,
    bottleneck_score: List[float],
    node_map: Dict[Tuple[int, int], int],
    local_capacity: List[float],
    prev_completed: int = 0,
    engine=None,
    baseline_station_queue: Optional[Dict[int, int]] = None,
    window_start_tick: Optional[int] = None,
    risk_override: Optional[float] = None,
    delay_scale: Optional[float] = None,
    station_queue_delta_scale: Optional[float] = None,
    reservation_window: int = 1,
    adj: Optional[Dict[int, List[int]]] = None,
    stalled_ratio_threshold: float = 0.3,
) -> torch.Tensor:
    """ Extract 7-dim system-level labels.
        提取七维系统级标签 通道: 总等待时间、平均超额延迟、站点队列变化量、
                               站点负载失衡度、瓶颈条件风险价值(CVaR)、
                               已完成订单变化量、死锁或严重拥塞风险
    Channels: total_wait_time, average_excess_delay, station_queue_delta,
              station_load_imbalance, bottleneck_CVaR,
              completed_orders_delta, deadlock_or_severe_congestion_risk

    Parameters
    ----------
    baseline_station_queue : dict[station_id, int] or None
        Station queue snapshot at decision time, for proper delta computation.
    window_start_tick : int or None
        Only count tasks completed after this tick for avg_excess_delay.
    """
    # 系统瞬时瘫痪指数, 统计了当前时刻，整个车队中有多少台机器人处于"异常停滞"状态
    num_robots = max(len(world.agents), 1)
    num_stations = len(world.map_state.station_positions)
    if delay_scale is None:
        delay_scale = 10.0
    if station_queue_delta_scale is None:
        station_queue_delta_scale = float(num_stations)

    stall_count = 0
    for agent in world.agents:
        if agent.status == AgentStatus.IDLE:
            continue
        is_stuck = agent.stuck_this_tick and not agent.is_waiting
        is_blocked = agent.plan_failed_streak >= 2
        if is_stuck or is_blocked:
            stall_count += 1

    # 历史窗口内的平均额外延迟，衡量的是已经完成的任务中，实际完成时间与理想完成时间（free_flow_time）之间的差距。这个指标反映了系统在处理订单时的效率和潜在的延迟问题。
    excess_delays = []
    for task in world.task_state.tasks.values():
        # 确保任务已完成，且有理论最短时间
        # free_flow_time (自由流行驶时间)：指的是假设地图上只有这一辆车（没有任何避让、排队、红绿灯），它走完这趟任务的绝对最少时间。
        if (task.completed_at is not None and task.created_at is not None
                and task.free_flow_time is not None):
            # 滑动窗口过滤：只统计最近一段时间完成的任务，如果 window_start_tick 不为 None，则只考虑那些完成时间在 window_start_tick 之后的任务。这有助于评估当前策略在最近一段时间内的表现，避免被过时的历史数据干扰。
            if window_start_tick is not None and task.completed_at <= window_start_tick:
                continue
            actual = task.completed_at - task.created_at
            excess = max(0, actual - task.free_flow_time)
            excess_delays.append(excess)
    avg_excess_delay = sum(excess_delays) / max(len(excess_delays), 1)

    station_ids = sorted(world.map_state.station_positions.keys())
    sid_to_idx = {sid: i for i, sid in enumerate(station_ids)}

    # Station queue occupancy ratio delta (physical occupancy / capacity)
    current_queue_ratios = {}
    for sid in station_ids:
        sq = world.station_state.get_queue(sid)
        if sq is not None:
            current_queue_ratios[sid] = float(sq.occupancy()) / max(sq.capacity, 1)
        else:
            current_queue_ratios[sid] = 0.0

    if baseline_station_queue is not None:
        station_queue_delta = 0.0
        for sid in station_ids:
            base = float(baseline_station_queue.get(sid, 0))
            station_queue_delta += max(0.0, current_queue_ratios[sid] - base)
    else:
        station_queue_delta = sum(max(0.0, v) for v in current_queue_ratios.values())

    # 实时运力倾斜度 统计正在驶向各个工作站的"送货任务"的数量，作为当前各站点的负载指标。然后计算这些负载指标的标准差，作为系统当前的负载失衡度。负载失衡度越高，说明订单越集中在少数几个工作站，可能导致瓶颈和延迟。
    station_load = [0.0] * num_stations
    from WorldState.task_state import TaskStatus as TS, TaskType as TT
    for task in world.task_state.tasks.values():
        if task.task_type == TT.DELIVER and task.status in (TS.ASSIGNED, TS.IN_PROGRESS):
            order = world.order_state.orders.get(task.order_id)
            if order:
                idx = sid_to_idx.get(order.station_id)
                if idx is not None:
                    station_load[idx] += 1.0
    
    # 计算各站运力负载的标准差
    """
    前置特征 skew 看的是 "未分配的静态订单池(Order Backlog)" 偏不偏；这里的 imbalance 看的是 "已经跑在路上的机器人(Deliver Tasks)" 偏不偏。
    一个是静态需求，一个是动态执行。
    """
    mean_load = sum(station_load) / max(num_stations, 1)
    imbalance = 0.0
    if num_stations > 1:
        var = sum((s - mean_load) ** 2 for s in station_load) / num_stations
        imbalance = var ** 0.5

    # 计算咽喉要道的条件风险价值 (CVaR)
    N = len(node_map)
    # 算出每个节点当前的密度 (车数 / 容量)
    node_cong = [0.0] * N
    for agent in world.agents:
        nid = node_map.get(agent.position)
        if nid is not None:
            node_cong[nid] += 1.0 / local_capacity[nid]
    
    # 挑出静态拓扑里，介数中心性最高的前 20% 的节点 (最容易堵的咽喉要道)
    bn_nodes = sorted(range(N), key=lambda i: bottleneck_score[i], reverse=True)
    top_q = max(1, N // 5)
    bn_set = bn_nodes[:top_q]
    # 在这些咽喉要道里，挑出当前"实际上最堵"的前 10%
    bn_cong = sorted([node_cong[i] for i in bn_set], reverse=True)
    top_p = max(1, len(bn_cong) // 10)
    # 计算这最拥堵的 10% 咽喉节点的平均拥堵度
    bottleneck_cvar = sum(bn_cong[:top_p]) / top_p if top_p > 0 else 0.0

    # Override the legacy single-node occupancy CVaR with bottleneck local
    # pressure CVaR.  This keeps the system label semantic as an absolute
    # scalar while making dim 4 match the node-level congestion definition.
    robot_counts = [0.0] * N
    wait_counts = [0.0] * N
    blocked_counts = [0.0] * N
    reservation_counts = [0.0] * N
    W = max(reservation_window, 1)

    for agent in world.agents:
        nid = node_map.get(agent.position)
        if nid is not None:
            robot_counts[nid] += 1.0
            if (agent.status != AgentStatus.IDLE
                    and agent.stuck_this_tick
                    and not agent.is_waiting):
                wait_counts[nid] += 1.0
            if agent.status != AgentStatus.IDLE and agent.plan_failed_streak >= 2:
                blocked_counts[nid] += 1.0

        if agent.has_path:
            remaining = agent.path[agent.path_index: agent.path_index + W]
            for pos in remaining:
                rid = node_map.get(tuple(pos))
                if rid is not None:
                    reservation_counts[rid] += 1.0

    bn_pressure = []
    for i in bn_set:
        cap = local_capacity[i]
        closed_nbrs = [i] + (adj[i] if adj is not None else [])
        local_density = sum(robot_counts[u] for u in closed_nbrs) / cap
        local_wait = sum(wait_counts[u] for u in closed_nbrs) / cap
        local_blocked = sum(blocked_counts[u] for u in closed_nbrs) / cap
        reservation = min(1.0, reservation_counts[i] / (W * cap))
        pressure = (0.30 * local_density + 0.30 * local_wait
                    + 0.20 * local_blocked + 0.20 * reservation)
        bn_pressure.append(pressure)

    bn_pressure.sort(reverse=True)
    top_p = max(1, len(bn_pressure) // 10)
    bottleneck_cvar = sum(bn_pressure[:top_p]) / top_p if top_p > 0 else 0.0

    # 正向业务吞吐量
    completed_delta = float(world.order_state.total_completed - prev_completed)

    # Unified risk signal (replaces old deadlock_risk)
    from WorldState.risk import compute_unified_risk
    risk_info = compute_unified_risk(world, forced_risk_override=risk_override)
    unified_risk = risk_info["unified_risk"]

    return torch.tensor([
        float(stall_count) / num_robots,
        float(avg_excess_delay) / max(delay_scale, 1.0),
        # station_queue_delta: sum of per-station occupancy-ratio increases,
        # divided by num_stations → average per-station delta in [0, 1]
        float(station_queue_delta) / max(station_queue_delta_scale, 1.0),
        float(imbalance) / 5.0,
        float(bottleneck_cvar),
        float(completed_delta),
        float(unified_risk),
    ], dtype=torch.float32)


def extract_station_labels(
    world: WorldState,
    station_queue_scale: Optional[float] = None,
    station_load_scale: Optional[float] = None,
) -> torch.Tensor:
    """Extract per-station labels: (S, 2) — [physical_queue_occupancy_ratio, assigned_load]."""
    station_ids = sorted(world.map_state.station_positions.keys())
    num_stations = len(station_ids)
    sid_to_idx = {sid: i for i, sid in enumerate(station_ids)}

    num_robots = max(len(world.agents), 1)
    if station_load_scale is None:
        station_load_scale = float(num_robots)

    labels = [[0.0, 0.0] for _ in range(num_stations)]

    for sid, sq in world.station_state.stations.items():
        idx = sid_to_idx.get(sid)
        if idx is not None:
            labels[idx][0] = float(sq.occupancy()) / max(sq.capacity, 1)

    from WorldState.task_state import TaskStatus as TS, TaskType as TT
    for task in world.task_state.tasks.values():
        if task.task_type == TT.DELIVER and task.status in (TS.ASSIGNED, TS.IN_PROGRESS):
            order = world.order_state.orders.get(task.order_id)
            if order:
                idx = sid_to_idx.get(order.station_id)
                if idx is not None:
                    labels[idx][1] += 1.0

    for i in range(num_stations):
        labels[i][1] /= max(station_load_scale, 1.0)

    return torch.tensor(labels, dtype=torch.float32)


class FeatureHistory:
    """
    特征历史环形缓冲区
    Ring buffer storing the last L frames of node features.
    """

    def __init__(self, num_nodes: int, feat_dim: int = 10, history_len: int = 4):
        self.history_len = history_len
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        self.buffer: List[torch.Tensor] = []

    def push(self, node_features: torch.Tensor):
        # 滑动窗口更新，保持最近 L 帧的节点特征。每次调用 push 时，会将当前帧的节点特征添加到缓冲区中，如果缓冲区超过了指定的历史长度，就会丢弃最旧的一帧。
        self.buffer.append(node_features.clone())
        if len(self.buffer) > self.history_len:
            self.buffer.pop(0)

    def get_history(self) -> torch.Tensor:
        """Return (L, N, feat_dim). Pads with zeros if < L frames. 获取数据与零填充 (Zero-Padding)"""
        frames = list(self.buffer)
        while len(frames) < self.history_len:
            frames.insert(0, torch.zeros(self.num_nodes, self.feat_dim))
        return torch.stack(frames, dim=0)

    def get_latest(self) -> Optional[torch.Tensor]:
        """Return the most recent frame, or None if buffer is empty."""
        if not self.buffer:
            return None
        return self.buffer[-1]

    @property
    def is_ready(self) -> bool:
        return len(self.buffer) >= self.history_len
