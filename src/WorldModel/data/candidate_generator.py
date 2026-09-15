"""
Candidate Generator
===================
Level 1: same order-pod-station, different robots.
Generates candidate robot assignments for world model evaluation.

New pipeline (preferred):
    generate_robot_candidates(contexts, world, top_m)

Legacy pipeline (deprecated):
    generate_candidates(world, top_m, pod_retriever, pod_return_planner)
"""

from typing import Dict, List, Mapping, Optional, Set, Tuple

from WorldState.world import WorldState
from WorldState.agent_state import AgentStatus
from WorldState.task_state import TaskType

from Policies.TaskAssigner.base_task_assigner import (
    AssignmentContext,
    fulfilled_pod_ids_for_order,
)


ASSIGN_ROBOT_ACTION_TYPE = "assign_robot"
NO_ASSIGN_ACTION_TYPE = "no_assign"
NO_ASSIGN_ACTION_SCHEMA_VERSION = "wm_native_no_assign_action_v1"
NO_ASSIGN_ENCODING = "zero_action_tensors_v1"


def is_no_assign_candidate(candidate: Optional[Mapping]) -> bool:
    """Return whether *candidate* is the native no-assignment action."""
    return bool(
        candidate
        and str(candidate.get("action_type", "")).lower()
        == NO_ASSIGN_ACTION_TYPE
    )


def make_no_assign_candidate() -> dict:
    """Build the stable metadata row for a native ``NO_ASSIGN`` action.

    The action deliberately has no robot or route.  Graph/action builders map
    it to all-zero tensors with unchanged dimensions; the explicit metadata is
    what prevents that encoding from being mistaken for a malformed robot
    assignment.
    """
    return {
        "action_type": NO_ASSIGN_ACTION_TYPE,
        "robot_id": None,
        "robot_start": None,
        "candidate_policy": "native_no_assign",
        "candidate_selection_mode": "native_no_assign",
        "chosen": False,
    }


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _planned_edge_reservations(
    world: WorldState,
    node_map: Mapping[Tuple[int, int], int],
    reservation_window: int,
) -> Dict[Tuple[int, int, int], float]:
    """Return time-indexed reservations already present in ``world``.

    This is deliberately a read-only coverage feature.  It is not used as a
    target or as an online reward; it only helps the collector expose both
    low- and high-conflict robot routes to the counterfactual rollout.
    """
    reservations: Dict[Tuple[int, int, int], float] = {}
    window = max(1, int(reservation_window))
    for agent in world.agents:
        previous = node_map.get(tuple(agent.position))
        if previous is None:
            continue
        path = list(getattr(agent, "path", ()) or ())
        index = max(0, int(getattr(agent, "path_index", 0)))
        for offset, position in enumerate(path[index:index + window], start=1):
            current = node_map.get(tuple(position))
            if current is None:
                continue
            if current != previous:
                key = (previous, current, offset)
                reservations[key] = reservations.get(key, 0.0) + 1.0
            previous = current
    return reservations


def _candidate_preview_features(
    ctx: AssignmentContext,
    agent,
    world: WorldState,
    *,
    lyapunov_config=None,
    lyapunov_snapshot=None,
    node_map: Optional[Mapping[Tuple[int, int], int]] = None,
    edge_flow_counter: Optional[Mapping[Tuple[int, int], float]] = None,
    bottleneck_score=None,
    planned_reservations: Optional[Mapping[Tuple[int, int, int], float]] = None,
    reservation_window: int = 10,
) -> dict:
    """Compute ETA-bin and route-pressure coverage features for one robot."""
    from WorldModel.core.lyapunov import (
        LyapunovL0Config,
        preview_assignment_context,
    )

    if lyapunov_config is None:
        resolved_config = LyapunovL0Config()
    elif isinstance(lyapunov_config, LyapunovL0Config):
        resolved_config = lyapunov_config
    else:
        resolved_config = LyapunovL0Config(**dict(lyapunov_config))

    preview = preview_assignment_context(
        world,
        ctx,
        agent,
        config=resolved_config,
        snapshot=lyapunov_snapshot,
    )

    conflict_score = 0.0
    route_length = 0
    if node_map:
        from WorldModel.graph_builder import compute_preview_legs

        station_position = (
            getattr(ctx, "entry_position", None)
            or getattr(ctx, "station_location")
        )
        assignment = {
            "robot_id": agent.agent_id,
            "robot_start": tuple(agent.position),
            "pod_id": ctx.pod_id,
            "pod_location": tuple(ctx.pod_location),
            "station_location": tuple(station_position),
            "return_location": tuple(ctx.return_location),
            "order_size": ctx.order_size,
        }
        legs = compute_preview_legs(
            assignment,
            world.map_state,
            dict(node_map),
            path_planner=None,
            world=None,
        )
        route: List[int] = []
        for leg in legs:
            leg = list(leg)
            if route and leg and route[-1] == leg[0]:
                route.extend(leg[1:])
            else:
                route.extend(leg)
        route_length = max(0, len(route) - 1)
        reservations = planned_reservations or {}
        recent_flow = edge_flow_counter or {}
        window = max(1, int(reservation_window))
        scored_edges = 0
        for offset, (source, target) in enumerate(
            zip(route, route[1:]), start=1
        ):
            if offset > window:
                break
            same = float(reservations.get((source, target, offset), 0.0))
            opposite = float(reservations.get((target, source, offset), 0.0))
            recent_same = float(recent_flow.get((source, target), 0.0))
            recent_opposite = float(recent_flow.get((target, source), 0.0))
            bottleneck = 0.0
            if bottleneck_score is not None:
                try:
                    bottleneck = float(bottleneck_score[target])
                except (IndexError, KeyError, TypeError, ValueError):
                    bottleneck = 0.0
            conflict_score += (
                same + 2.0 * opposite
                + recent_same + 2.0 * recent_opposite
                + bottleneck
            )
            scored_edges += 1
        conflict_score /= max(scored_edges, 1)

    return {
        "eta": float(preview.eta),
        "eta_bin": int(preview.eta_bin),
        "arrival_delta_preview": float(preview.delta_arrival_potential),
        "route_conflict_preview": float(conflict_score),
        "route_length_preview": int(route_length),
    }


def generate_candidates(
    world: WorldState,
    top_m: int = 5,
    pod_retriever=None,
    pod_return_planner=None,
) -> List[dict]:
    """Generate candidate groups for pending orders. 为待处理订单生成候选组。

    Returns a list of candidate_group dicts, each containing:
      - group_id: str
      - fixed_context: dict (order_id, pod_id, pod_location, station_id,
                              station_location, return_location, order_size)
      - candidates: list of dicts (robot_id, robot_start, candidate_policy, chosen)
    """
    pending = world.order_state.get_pending_orders()
    if not pending:
        return []

    idle_agents = world.get_idle_agents()
    if not idle_agents:
        return []

    groups = []
    for order in pending:
        fulfilled_pods = fulfilled_pod_ids_for_order(world, order)
        pod_ids = order.pod_ids
        if not pod_ids and pod_retriever is not None:
            pod_ids = pod_retriever.retrieve(order, world)
            if pod_ids:
                order.pod_ids = pod_ids

        if not pod_ids:
            continue

        station_loc = world.map_state.station_positions.get(order.station_id)
        if station_loc is None:
            continue

        for pod_id in pod_ids:
            if int(pod_id) in fulfilled_pods:
                continue
            pod = world.pod_state.get_pod(pod_id)
            if pod is None or pod.is_carried:
                continue

            pod_loc = pod.current_position
            return_loc = pod.home_position
            if pod_return_planner is not None:
                try:
                    ret = pod_return_planner.plan_return(pod, station_loc, world)
                    if ret:
                        return_loc = ret
                except Exception:
                    pass

            candidates_by_dist = sorted(
                idle_agents,
                key=lambda a: _manhattan(a.position, pod_loc),
            )[:top_m]

            if not candidates_by_dist:
                continue

            group_id = f"t{world.tick}_o{order.order_id}_p{pod_id}"
            fixed_context = {
                "order_id": order.order_id,
                "pod_id": pod_id,
                "pod_location": pod_loc,
                "station_id": order.station_id,
                "station_location": station_loc,
                "return_location": return_loc,
                "task_type": "PICK",
                "order_size": sum(order.sku_demands.values()),
            }

            cands = []
            for agent in candidates_by_dist:
                cands.append({
                    "robot_id": agent.agent_id,
                    "robot_start": agent.position,
                    "candidate_policy": "nearest",
                    "chosen": False,
                })

            cands[0]["chosen"] = True

            groups.append({
                "group_id": group_id,
                "tick": world.tick,
                "fixed_context": fixed_context,
                "candidates": cands,
            })

    return groups


def generate_robot_candidates(
    contexts: List[AssignmentContext],
    world: WorldState,
    top_m: int = 5,
    *,
    candidate_mode: str = "nearest",
    lyapunov_config=None,
    node_map: Optional[Mapping[Tuple[int, int], int]] = None,
    edge_flow_counter: Optional[Mapping[Tuple[int, int], float]] = None,
    bottleneck_score=None,
    reservation_window: int = 10,
    include_no_assign_candidate: bool = False,
) -> List[dict]:
    """Generate robot candidate groups from pre-built AssignmentContexts.

    For each context, returns top-m nearest idle robots as candidates.
    This is the new pipeline entry point — does NOT scan pending orders.

    Returns the same group structure as the legacy generate_candidates()
    for backward compatibility with data_collector / rollout code.
    """
    if candidate_mode not in ("nearest", "eta_stratified", "stratified"):
        raise ValueError(
            "candidate_mode must be one of: nearest, eta_stratified, stratified"
        )
    if not contexts:
        return []

    idle_agents = world.get_idle_agents()
    if not idle_agents:
        return []

    lyapunov_snapshot = None
    planned_reservations = None
    resolved_config = lyapunov_config
    if candidate_mode != "nearest":
        from WorldModel.core.lyapunov import (
            LyapunovL0Config,
            compute_lyapunov_snapshot,
        )

        if resolved_config is None:
            resolved_config = LyapunovL0Config()
        elif not isinstance(resolved_config, LyapunovL0Config):
            resolved_config = LyapunovL0Config(**dict(resolved_config))
        lyapunov_snapshot = compute_lyapunov_snapshot(world, resolved_config)
        if node_map:
            planned_reservations = _planned_edge_reservations(
                world, node_map, reservation_window
            )

    groups = []
    for ctx in contexts:
        order = getattr(
            getattr(world, "order_state", None), "orders", {}
        ).get(ctx.order_id)
        if (
            order is not None
            and int(ctx.pod_id) in fulfilled_pod_ids_for_order(world, order)
        ):
            continue
        ranked = sorted(
            idle_agents,
            key=lambda a: _manhattan(a.position, ctx.pod_location),
        )

        if not ranked:
            continue

        features_by_id = {}
        selected = []
        reasons: Dict[int, List[str]] = {}

        def add(agent, reason: str):
            agent_reasons = reasons.setdefault(agent.agent_id, [])
            if reason not in agent_reasons:
                agent_reasons.append(reason)
            if agent in selected or len(selected) >= max(1, int(top_m)):
                return
            selected.append(agent)

        add(ranked[0], "nearest")
        if candidate_mode != "nearest":
            for agent in ranked:
                features_by_id[agent.agent_id] = _candidate_preview_features(
                    ctx,
                    agent,
                    world,
                    lyapunov_config=resolved_config,
                    lyapunov_snapshot=lyapunov_snapshot,
                    node_map=node_map,
                    edge_flow_counter=edge_flow_counter,
                    bottleneck_score=bottleneck_score,
                    planned_reservations=planned_reservations,
                    reservation_window=reservation_window,
                )

            if candidate_mode == "stratified":
                low_conflict = min(
                    ranked,
                    key=lambda agent: (
                        features_by_id[agent.agent_id]["route_conflict_preview"],
                        _manhattan(agent.position, ctx.pod_location),
                    ),
                )
                high_conflict = max(
                    ranked,
                    key=lambda agent: (
                        features_by_id[agent.agent_id]["route_conflict_preview"],
                        _manhattan(agent.position, ctx.pod_location),
                    ),
                )
                add(low_conflict, "traffic_low")
                add(high_conflict, "traffic_high")

            by_eta: Dict[int, List] = {}
            for agent in ranked:
                eta_bin = features_by_id[agent.agent_id]["eta_bin"]
                by_eta.setdefault(eta_bin, []).append(agent)
            for eta_bin in sorted(by_eta):
                add(by_eta[eta_bin][0], f"eta_bin_{eta_bin}")

        for agent in ranked:
            if agent not in selected:
                add(agent, "nearest_fill")
        candidates_by_dist = selected[:max(1, int(top_m))]

        group_id = f"t{world.tick}_o{ctx.order_id}_p{ctx.pod_id}"
        fixed_context = {
            "order_id": ctx.order_id,
            "pod_id": ctx.pod_id,
            "pod_location": ctx.pod_location,
            "station_id": ctx.station_id,
            "station_location": ctx.station_location,
            "entry_position": ctx.entry_position,
            "exit_position": ctx.exit_position,
            "return_location": ctx.return_location,
            "task_type": "PICK",
            "order_size": ctx.order_size,
        }

        cands = []
        for agent in candidates_by_dist:
            candidate = {
                "action_type": ASSIGN_ROBOT_ACTION_TYPE,
                "robot_id": agent.agent_id,
                "robot_start": agent.position,
                "candidate_policy": "+".join(
                    reasons.get(agent.agent_id, ["nearest_fill"])
                ),
                "candidate_selection_mode": candidate_mode,
                "chosen": False,
            }
            candidate.update(features_by_id.get(agent.agent_id, {}))
            cands.append(candidate)
        if cands:
            cands[0]["chosen"] = True
        robot_candidate_count = len(cands)
        if include_no_assign_candidate:
            # ``top_m`` limits robot coverage only.  NO_ASSIGN is a distinct
            # native action and must not displace one of the sampled robots.
            cands.append(make_no_assign_candidate())

        groups.append({
            "group_id": group_id,
            "tick": world.tick,
            "fixed_context": fixed_context,
            "candidates": cands,
            "robot_candidate_count": robot_candidate_count,
            "includes_no_assign_candidate": bool(
                include_no_assign_candidate
            ),
        })

    return groups


def compute_heuristic_cost(
    candidate: dict,
    fixed_context: dict,
    world: WorldState,
) -> float:
    """Compute a heuristic pseudo-cost for a candidate assignment. 计算候选分配的启发式伪代价。

    Components: distance_cost, station_load_cost, corridor_congestion_cost. 组成部分：距离代价、站点负载代价、通道拥堵代价。
    """
    if is_no_assign_candidate(candidate):
        raise ValueError(
            "NO_ASSIGN has no valid distance heuristic; use realized rollout "
            "cost and mark heuristic_cost_valid=False"
        )
    robot_start = candidate["robot_start"]
    pod_loc = fixed_context["pod_location"]
    station_loc = fixed_context["station_location"]
    return_loc = fixed_context["return_location"]

    dist_r2p = _manhattan(robot_start, pod_loc)
    dist_p2s = _manhattan(pod_loc, station_loc)
    dist_s2r = _manhattan(station_loc, return_loc)
    distance_cost = float(dist_r2p + dist_p2s + dist_s2r)

    station_load = 0
    for o in world.order_state.get_in_progress_orders():
        if o.station_id == fixed_context["station_id"]:
            station_load += 1
    station_load_cost = float(station_load) * 2.0

    corridor_congestion = 0.0
    for agent in world.agents:
        d = _manhattan(agent.position, pod_loc) + _manhattan(agent.position, station_loc)
        if d <= dist_r2p + dist_p2s + 3:
            corridor_congestion += 1.0
    corridor_cost = corridor_congestion * 0.5

    return distance_cost + station_load_cost + corridor_cost


def build_candidate_assignment(candidate: dict, fixed_context: dict) -> dict:
    """ Convert candidate + fixed_context into an assignment dict for build_action_field.
        将 候选 + 固定上下文 转换为用于构建操作字段的赋值字典。
    """
    if is_no_assign_candidate(candidate):
        return {
            "action_type": NO_ASSIGN_ACTION_TYPE,
            "action_schema_version": NO_ASSIGN_ACTION_SCHEMA_VERSION,
            "action_encoding": NO_ASSIGN_ENCODING,
        }
    return {
        "action_type": ASSIGN_ROBOT_ACTION_TYPE,
        "robot_id": candidate["robot_id"],
        "robot_start": candidate["robot_start"],
        "pod_id": fixed_context["pod_id"],
        "pod_location": fixed_context["pod_location"],
        "station_location": fixed_context["station_location"],
        "return_location": fixed_context["return_location"],
        "order_size": fixed_context.get("order_size", 1),
    }
