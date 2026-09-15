"""
Counterfactual Rollout 
===========================
Clone world -> force apply candidate -> H-tick isolated rollout
-> collect candidate-specific labels + realized cost.
克隆仿真世界 -> 强制应用候选方案 -> H步独立推演
-> 收集候选方案专属标签 + 已实现成本


"""

import copy
import random
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch

from WorldState.world import WorldState
from WorldState.agent_state import AgentStatus
from WorldState.task_state import Task, TaskType, TaskStatus
from WorldState.order_state import Order, OrderStatus
from WorldState.risk import _update_stationary_ticks

from Policies.TaskAssigner.base_task_assigner import (
    fulfilled_pod_ids_for_order,
)

from WorldModel.graph_builder import (
    extract_demand_context,
    extract_node_labels,
    extract_system_labels,
    extract_station_labels,
)
from WorldModel.costs import compute_realized_cost
from WorldModel.candidate_generator import (
    ASSIGN_ROBOT_ACTION_TYPE,
    NO_ASSIGN_ACTION_TYPE,
    is_no_assign_candidate,
)


class RolloutObserverError(RuntimeError):
    """Diagnostic observer failure that must not become a padded rollout."""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _resolve_deliver_goal(world, task):
    """Return the walkable goal for a task.

    For DELIVER tasks the real engine routes to station entry_position;
    replicate that here so the rollout planner doesn't target the
    unwalkable station body cell.
    """
    if task.task_type != TaskType.DELIVER:
        return task.destination
    if not hasattr(world, 'station_state'):
        return task.destination
    sid = task.station_id
    if sid is None:
        for s_id, s_pos in world.map_state.station_positions.items():
            if s_pos == task.destination:
                sid = s_id
                break
    if sid is None:
        return task.destination
    sq = world.station_state.get_queue(sid)
    if sq and sq.entry_position:
        return sq.entry_position
    return task.destination


def _task_free_flow(task_type, source, destination, config):
    dist = _manhattan(source, destination)
    sim = config.simulation
    if task_type == TaskType.PICK:
        service = sim.pickup_duration
    elif task_type == TaskType.DELIVER:
        service = sim.station_process_duration
    elif task_type == TaskType.RETURN:
        service = sim.dropoff_duration
    else:
        service = 0
    return dist + service


# ------------------------------------------------------------------
# Force apply a candidate assignment into a cloned world
# ------------------------------------------------------------------

def force_apply_candidate(world: WorldState, candidate: dict,
                          fixed_context: dict, config) -> bool:
    """Create PICK/DELIVER/RETURN task chain for candidate in world (in-place).

    Returns True if successfully applied, False if candidate is invalid.
    """
    robot_id = candidate["robot_id"]
    agent = world.get_agent(robot_id)
    if agent is None:
        return False
    if agent.status != AgentStatus.IDLE:
        return False
    if agent.position != candidate["robot_start"]:
        return False

    pod_id = fixed_context["pod_id"]
    pod = world.pod_state.get_pod(pod_id)
    if pod is None or pod.is_carried:
        return False
    if pod.current_position != fixed_context["pod_location"]:
        return False

    order_id = fixed_context["order_id"]
    order = world.order_state.orders.get(order_id)
    if order is None or order.status != OrderStatus.PENDING:
        return False
    # A partially dispatched multi-pod order can remain PENDING while an
    # earlier pod chain has already reached the station.  Never inject a
    # second chain for that fulfilled (order, pod) pair into the clone.
    if int(pod_id) in fulfilled_pod_ids_for_order(world, order):
        return False

    pod_loc = fixed_context["pod_location"]
    station_loc = fixed_context["station_location"]
    return_loc = fixed_context["return_location"]
    tick = world.tick

    pick = Task(TaskType.PICK, order_id, pod_id, agent.position, pod_loc)
    pick.agent_id = robot_id
    pick.status = TaskStatus.ASSIGNED
    pick.created_at = tick
    pick.assigned_at = tick
    pick.free_flow_time = _task_free_flow(TaskType.PICK, agent.position, pod_loc, config)

    deliver = Task(TaskType.DELIVER, order_id, pod_id, pod_loc, station_loc)
    deliver.agent_id = robot_id
    deliver.station_id = fixed_context["station_id"]
    deliver.status = TaskStatus.ASSIGNED
    deliver.created_at = tick
    deliver.assigned_at = tick
    deliver.free_flow_time = _task_free_flow(TaskType.DELIVER, pod_loc, station_loc, config)

    return_source = fixed_context.get("exit_position") or station_loc
    ret = Task(TaskType.RETURN, order_id, pod_id, return_source, return_loc)
    ret.agent_id = robot_id
    ret.station_id = fixed_context["station_id"]
    ret.status = TaskStatus.ASSIGNED
    ret.created_at = tick
    ret.assigned_at = tick
    ret.free_flow_time = _task_free_flow(
        TaskType.RETURN, return_source, return_loc, config
    )

    world.task_state.add_task(pick)
    world.task_state.add_task(deliver)
    world.task_state.add_task(ret)

    # A candidate represents one order/pod dispatch context.  Multi-pod
    # orders must remain pending until every pod chain is covered, otherwise
    # the continuation assigner can never materialise the remaining arrival
    # work in the counterfactual world.
    covered_pods = set(int(value) for value in getattr(order, "delivered_pod_ids", ()))
    for task in world.task_state.tasks.values():
        if int(getattr(task, "order_id", -1)) != int(order_id):
            continue
        if task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
            covered_pods.add(int(task.pod_id))
    required_pods = [int(value) for value in getattr(order, "pod_ids", ())]
    if not required_pods or all(value in covered_pods for value in required_pods):
        order.status = OrderStatus.IN_PROGRESS

    agent.status = AgentStatus.MOVING_TO_POD
    agent.assigned_task_id = pick.task_id

    return True


def _no_assign_context_snapshot(world: WorldState, fixed_context: dict) -> dict:
    """Capture the dispatch state that native NO_ASSIGN must not mutate."""
    order_id = fixed_context.get("order_id")
    pod_id = fixed_context.get("pod_id")
    order = getattr(getattr(world, "order_state", None), "orders", {}).get(
        order_id
    )
    pod_state = getattr(world, "pod_state", None)
    pod = pod_state.get_pod(pod_id) if pod_state is not None else None
    tasks = getattr(getattr(world, "task_state", None), "tasks", {})
    context_task_ids = sorted(
        int(task_id)
        for task_id, task in tasks.items()
        if int(getattr(task, "order_id", -1)) == int(order_id)
        and int(getattr(task, "pod_id", -1)) == int(pod_id)
    )
    return {
        "order_id": order_id,
        "pod_id": pod_id,
        "order_status": (
            getattr(getattr(order, "status", None), "name", None)
            if order is not None else None
        ),
        "pod_is_carried": (
            bool(getattr(pod, "is_carried", False))
            if pod is not None else None
        ),
        "pod_position": (
            tuple(getattr(pod, "current_position"))
            if pod is not None else None
        ),
        "context_task_ids": context_task_ids,
        "all_task_ids": sorted(int(task_id) for task_id in tasks),
    }


def _validate_no_assign_candidate(
    world: WorldState,
    fixed_context: dict,
) -> bool:
    """Validate a dispatchable context without creating any task."""
    order_id = fixed_context.get("order_id")
    pod_id = fixed_context.get("pod_id")
    order = getattr(getattr(world, "order_state", None), "orders", {}).get(
        order_id
    )
    if order is None or order.status != OrderStatus.PENDING:
        return False
    if int(pod_id) in fulfilled_pod_ids_for_order(world, order):
        return False
    pod_state = getattr(world, "pod_state", None)
    pod = pod_state.get_pod(pod_id) if pod_state is not None else None
    if pod is None or bool(getattr(pod, "is_carried", False)):
        return False
    if tuple(getattr(pod, "current_position")) != tuple(
        fixed_context.get("pod_location")
    ):
        return False
    tasks = getattr(getattr(world, "task_state", None), "tasks", {})
    for task in tasks.values():
        if (
            int(getattr(task, "order_id", -1)) == int(order_id)
            and int(getattr(task, "pod_id", -1)) == int(pod_id)
            and task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        ):
            return False
    return True


# ------------------------------------------------------------------
# Lightweight simulation stepper (no new orders / no new assignments)
# ------------------------------------------------------------------

def _uses_explicit_station_waiting(world: WorldState) -> bool:
    return any(
        queue.uses_explicit_waiting
        for queue in world.station_state.stations.values()
    )


def _mark_station_waiting_step(agent, queue, task, tick: int) -> None:
    if not queue.uses_explicit_waiting:
        return
    sequence = queue.waiting_reservation_sequence(agent.agent_id)
    if sequence is None:
        sequence = queue.enqueue_waiting_reservation(
            agent.agent_id,
            request_tick=tick,
            priority_sequence=getattr(
                task, "station_dispatch_sequence", None
            ),
        )
    agent.clear_path()
    agent.wait_ticks = 0
    agent.assigned_task_id = task.task_id
    agent.mark_station_waiting(
        station_id=task.station_id,
        since_tick=(
            agent.station_waiting_since_tick
            if agent.station_waiting_since_tick is not None
            else tick
        ),
        sequence=sequence,
    )


def _prune_station_waiting_step(world: WorldState) -> None:
    if not _uses_explicit_station_waiting(world):
        return
    for station_id, queue in world.station_state.stations.items():
        valid = set()
        for agent in world.agents:
            if agent.status != AgentStatus.WAITING_ASSIGNED:
                continue
            nxt = world.task_state.get_next_task_for_agent(agent.agent_id)
            if (
                nxt is not None
                and nxt.task_type == TaskType.DELIVER
                and nxt.station_id == station_id
            ):
                valid.add(agent.agent_id)
        removed = queue.prune_waiting_reservations(valid)
        for agent_id in removed:
            agent = world.get_agent(agent_id)
            if agent is None:
                continue
            agent.clear_station_waiting()
            agent.assigned_task_id = None
            agent.status = (
                AgentStatus.CARRYING
                if agent.carried_pod_id is not None
                else AgentStatus.IDLE
            )

def _plan_and_activate_step(world, path_planner, tick, config):
    """Mirror SimulationEngine._plan_and_activate with DELIVER reservation semantics."""
    station_state = world.station_state
    _prune_station_waiting_step(world)

    # Build handoff blocked set (entry/exit of active stations)
    active_station_ids = set()
    for agent in world.agents:
        if agent.status == AgentStatus.EXITING:
            t = world.task_state.get_active_task_for_agent(agent.agent_id)
            if t:
                active_station_ids.add(t.station_id)
        elif agent.status == AgentStatus.CARRYING and not agent.has_path:
            t = world.task_state.get_active_task_for_agent(agent.agent_id)
            if t and t.task_type == TaskType.DELIVER:
                active_station_ids.add(t.station_id)

    handoff_blocked = set()
    for sq in station_state.stations.values():
        if sq.station_id not in active_station_ids:
            continue
        if sq.entry_position:
            handoff_blocked.add(sq.entry_position)
        if sq.exit_position:
            handoff_blocked.add(sq.exit_position)

    def sort_key(agent):
        waiting = agent.status == AgentStatus.WAITING_ASSIGNED
        sequence = None
        if waiting and agent.station_waiting_station_id is not None:
            queue = station_state.get_queue(agent.station_waiting_station_id)
            if queue is not None:
                sequence = queue.waiting_reservation_sequence(agent.agent_id)
        if waiting:
            return (
                0,
                sequence if sequence is not None else 10**12,
                agent.agent_id,
            )
        return (
            1,
            0 if agent.position in handoff_blocked else 1,
            10**12,
            agent.agent_id,
        )

    agents_sorted = sorted(world.agents, key=sort_key)

    if (
        getattr(path_planner, "supports_batch_planning", False)
        and callable(getattr(path_planner, "plan_batch", None))
    ):
        _plan_and_activate_step_batch(
            world,
            path_planner,
            tick,
            config,
            station_state,
            handoff_blocked,
            agents_sorted,
        )
        return

    for agent in agents_sorted:
        if agent.status in (AgentStatus.QUEUING, AgentStatus.DELIVERING,
                            AgentStatus.EXITING):
            continue
        if agent.is_idle or (
            agent.is_waiting
            and agent.status != AgentStatus.WAITING_ASSIGNED
        ):
            continue

        active = world.task_state.get_active_task_for_agent(agent.agent_id)

        # --- DELIVER branch: transactional with queue.reserve ---
        if active is None:
            nxt = world.task_state.get_next_task_for_agent(agent.agent_id)
            if nxt is not None and nxt.task_type == TaskType.DELIVER:
                queue = station_state.get_queue(nxt.station_id)
                if queue is None or queue.entry_position is None:
                    continue

                if agent.status == AgentStatus.WAITING_ASSIGNED:
                    if not queue.can_promote_waiting_reservation(
                        agent.agent_id
                    ):
                        continue

                er, ec = queue.entry_position
                ar, ac = agent.position
                rough_eta = abs(er - ar) + abs(ec - ac)
                if not queue.reserve(
                    agent.agent_id,
                    request_tick=tick,
                    eta_ticks=rough_eta,
                ):
                    if queue.uses_explicit_waiting:
                        _mark_station_waiting_step(agent, queue, nxt, tick)
                    continue

                goal = queue.entry_position

                # Don't send another agent when entry is occupied
                if any(a.position == goal and a.agent_id != agent.agent_id
                       and not a.has_path
                       for a in world.agents):
                    queue.unreserve(
                        agent.agent_id, reason="entry_occupied"
                    )
                    if queue.uses_explicit_waiting:
                        queue.requeue_waiting_reservation(
                            agent.agent_id,
                            request_tick=tick,
                            priority_sequence=getattr(
                                nxt, "station_dispatch_sequence", None
                            ),
                        )
                        _mark_station_waiting_step(agent, queue, nxt, tick)
                    continue

                extra_blocked = handoff_blocked - {goal, agent.position}
                path = (
                    []
                    if agent.position == goal
                    else path_planner.plan(
                        agent, goal, world,
                        extra_blocked=extra_blocked if extra_blocked else None,
                    )
                )
                if not path and agent.position != goal:
                    queue.unreserve(
                        agent.agent_id, reason="path_failure"
                    )
                    if queue.uses_explicit_waiting:
                        queue.requeue_waiting_reservation(
                            agent.agent_id,
                            request_tick=tick,
                            priority_sequence=getattr(
                                nxt, "station_dispatch_sequence", None
                            ),
                        )
                        _mark_station_waiting_step(agent, queue, nxt, tick)
                    continue

                nxt.status = TaskStatus.IN_PROGRESS
                if nxt.started_at is None:
                    nxt.started_at = tick
                agent.status = AgentStatus.CARRYING
                agent.assigned_task_id = nxt.task_id
                if queue.uses_explicit_waiting:
                    agent.clear_station_waiting()
                agent.assign_path(path)
                agent.plan_failed_streak = 0
                if nxt.free_flow_time is None:
                    nxt.free_flow_time = _task_free_flow(
                        nxt.task_type, nxt.source, nxt.destination, config,
                    )
                continue

        # --- Generic branch: PICK, RETURN, already-active ---
        if active is None:
            nxt = world.task_state.get_next_task_for_agent(agent.agent_id)
            if nxt is None:
                continue
            nxt.status = TaskStatus.IN_PROGRESS
            if nxt.started_at is None:
                nxt.started_at = tick
            active = nxt

            if active.task_type == TaskType.PICK:
                agent.status = AgentStatus.MOVING_TO_POD
            elif active.task_type == TaskType.RETURN:
                agent.status = AgentStatus.RETURNING
            agent.assigned_task_id = active.task_id

        if active.free_flow_time is None:
            active.free_flow_time = _task_free_flow(
                active.task_type, active.source, active.destination, config,
            )

        if not agent.has_path:
            if (active.task_type == TaskType.DELIVER
                    and agent.status == AgentStatus.CARRYING):
                queue = station_state.get_queue(active.station_id)
                if queue and queue.entry_position:
                    goal = queue.entry_position
                else:
                    goal = active.destination
            else:
                goal = active.destination

            if agent.position == goal:
                agent.plan_failed_streak = 0
                continue

            extra_blocked = handoff_blocked - {goal, agent.position}
            path = path_planner.plan(
                agent, goal, world,
                extra_blocked=extra_blocked if extra_blocked else None,
            )
            if path:
                agent.assign_path(path)
                agent.plan_failed_streak = 0
            else:
                agent.plan_failed_streak += 1


def _plan_and_activate_step_batch(
    world,
    path_planner,
    tick,
    config,
    station_state,
    handoff_blocked,
    agents_sorted,
):
    """Counterfactual mirror of Engine._plan_and_activate_batch."""
    requests = []

    def finish_deliver(request, path):
        agent = request["agent"]
        task = request["task"]
        queue = request["queue"]
        task.status = TaskStatus.IN_PROGRESS
        if task.started_at is None:
            task.started_at = tick
        agent.status = AgentStatus.CARRYING
        agent.assigned_task_id = task.task_id
        if queue.uses_explicit_waiting:
            agent.clear_station_waiting()
        agent.assign_path(path)
        agent.plan_failed_streak = 0
        if task.free_flow_time is None:
            task.free_flow_time = _task_free_flow(
                task.task_type, task.source, task.destination, config,
            )

    for agent in agents_sorted:
        if agent.status in (
            AgentStatus.QUEUING,
            AgentStatus.DELIVERING,
            AgentStatus.EXITING,
        ):
            continue
        if agent.is_idle or (
            agent.is_waiting
            and agent.status != AgentStatus.WAITING_ASSIGNED
        ):
            continue

        active = world.task_state.get_active_task_for_agent(agent.agent_id)
        if active is None:
            nxt = world.task_state.get_next_task_for_agent(agent.agent_id)
            if nxt is not None and nxt.task_type == TaskType.DELIVER:
                queue = station_state.get_queue(nxt.station_id)
                if queue is None or queue.entry_position is None:
                    continue
                if agent.status == AgentStatus.WAITING_ASSIGNED:
                    if not queue.can_promote_waiting_reservation(
                        agent.agent_id
                    ):
                        continue

                er, ec = queue.entry_position
                ar, ac = agent.position
                rough_eta = abs(er - ar) + abs(ec - ac)
                if not queue.reserve(
                    agent.agent_id,
                    request_tick=tick,
                    eta_ticks=rough_eta,
                ):
                    if queue.uses_explicit_waiting:
                        _mark_station_waiting_step(agent, queue, nxt, tick)
                    continue

                goal = queue.entry_position
                if any(
                    other.position == goal
                    and other.agent_id != agent.agent_id
                    and not other.has_path
                    for other in world.agents
                ):
                    queue.unreserve(agent.agent_id, reason="entry_occupied")
                    if queue.uses_explicit_waiting:
                        queue.requeue_waiting_reservation(
                            agent.agent_id,
                            request_tick=tick,
                            priority_sequence=getattr(
                                nxt, "station_dispatch_sequence", None
                            ),
                        )
                        _mark_station_waiting_step(agent, queue, nxt, tick)
                    continue

                request = {
                    "kind": "deliver",
                    "agent": agent,
                    "task": nxt,
                    "queue": queue,
                    "goal": goal,
                }
                if agent.position == goal:
                    finish_deliver(request, [])
                else:
                    requests.append(request)
                continue

        if active is None:
            nxt = world.task_state.get_next_task_for_agent(agent.agent_id)
            if nxt is None:
                continue
            nxt.status = TaskStatus.IN_PROGRESS
            if nxt.started_at is None:
                nxt.started_at = tick
            active = nxt
            if active.task_type == TaskType.PICK:
                agent.status = AgentStatus.MOVING_TO_POD
            elif active.task_type == TaskType.RETURN:
                agent.status = AgentStatus.RETURNING
            agent.assigned_task_id = active.task_id

        if active.free_flow_time is None:
            active.free_flow_time = _task_free_flow(
                active.task_type, active.source, active.destination, config,
            )
        if agent.has_path:
            continue
        if (
            active.task_type == TaskType.DELIVER
            and agent.status == AgentStatus.CARRYING
        ):
            queue = station_state.get_queue(active.station_id)
            goal = (
                queue.entry_position
                if queue is not None and queue.entry_position is not None
                else active.destination
            )
        else:
            goal = active.destination
        if agent.position == goal:
            agent.plan_failed_streak = 0
            continue
        requests.append({
            "kind": "generic",
            "agent": agent,
            "task": active,
            "queue": None,
            "goal": goal,
        })

    if not requests:
        return
    paths = path_planner.plan_batch(
        [(request["agent"], request["goal"]) for request in requests],
        world,
        extra_blocked=handoff_blocked if handoff_blocked else None,
    )
    expected = {int(request["agent"].agent_id) for request in requests}
    actual = {int(agent_id) for agent_id in paths}
    if actual != expected:
        raise RuntimeError(
            "counterfactual batch planner robot-set mismatch: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for request in requests:
        agent = request["agent"]
        path = paths[agent.agent_id]
        if not isinstance(path, list) or not path:
            raise RuntimeError(
                "counterfactual single-step batch planner returned an empty "
                f"path for Agent #{agent.agent_id}: {path!r}"
            )

    for request in requests:
        agent = request["agent"]
        path = paths[agent.agent_id]
        if request["kind"] == "deliver":
            finish_deliver(request, path)
        else:
            agent.assign_path(path)
            agent.plan_failed_streak = 0


def _move_agents_step(world):
    """Mirror Engine._move_agents: intended-move conflict resolution."""
    prev_positions = {a.agent_id: a.position for a in world.agents}

    movable = []
    for agent in world.agents:
        agent.previous_position = agent.position
        agent.stuck_this_tick = False
        agent.traffic_blocked_this_tick = False
        agent.moved_this_tick = False
        if agent.status in (
            AgentStatus.QUEUING, AgentStatus.DELIVERING, AgentStatus.EXITING,
        ):
            continue
        if agent.is_waiting:
            continue
        if agent.has_path:
            movable.append(agent)

    intended = {
        agent.agent_id: agent.path[agent.path_index]
        for agent in movable
    }

    occupied = {
        a.position for a in world.agents
        if a.agent_id not in intended
    }

    blocked_aids = set()
    changed = True
    while changed:
        changed = False
        target_counts = {}
        for aid, pos in intended.items():
            if aid in blocked_aids:
                continue
            target_counts.setdefault(pos, []).append(aid)
        for pos, aids in target_counts.items():
            if len(aids) > 1 or pos in occupied:
                for aid in aids:
                    if aid not in blocked_aids:
                        blocked_aids.add(aid)
                        ag = world.get_agent(aid)
                        occupied.add(ag.position)
                        changed = True

    for agent in movable:
        if agent.agent_id in blocked_aids:
            agent.clear_path()
            agent.stuck_this_tick = True
            agent.traffic_blocked_this_tick = True
            continue
        old_pos = agent.position
        new_pos = agent.advance()
        if new_pos == old_pos:
            agent.stuck_this_tick = True
        else:
            agent.moved_this_tick = True
        if new_pos and agent.carried_pod_id is not None:
            pod = world.pod_state.get_pod(agent.carried_pod_id)
            if pod:
                pod.current_position = new_pos

    return prev_positions, len(blocked_aids)


def _handle_actions_step(world, config, tick):
    """Mirror SimulationEngine._handle_actions: arrival check uses
    active_task.destination (service slot), not entry_position.
    DELIVER completion transitions to EXITING (not COMPLETED).
    """
    sim = config.simulation
    for agent in world.agents:
        active = world.task_state.get_active_task_for_agent(agent.agent_id)
        if active is None:
            continue

        if agent.status == AgentStatus.EXITING:
            continue

        # Admission waiting is deliberately separate from service/action
        # waiting.  The DELIVER task remains ASSIGNED until promotion, so a
        # WAITING_ASSIGNED robot must never enter the action countdown path.
        if agent.status == AgentStatus.WAITING_ASSIGNED:
            continue

        if agent.is_waiting:
            agent.wait_ticks -= 1
            if agent.wait_ticks > 0:
                continue
        else:
            if agent.position != active.destination:
                continue
            if agent.has_path:
                continue

            agent.stuck_this_tick = False

            if active.task_type == TaskType.PICK:
                req = sim.pickup_duration
            elif active.task_type == TaskType.DELIVER:
                req = sim.station_process_duration
            elif active.task_type == TaskType.RETURN:
                req = sim.dropoff_duration
            else:
                req = 0

            if req > 0:
                if active.task_type == TaskType.DELIVER:
                    agent.status = AgentStatus.DELIVERING
                agent.wait_ticks = req
                continue

        if active.task_type == TaskType.PICK:
            pod = world.pod_state.get_pod(active.pod_id)
            if pod:
                pod.pick_up(agent.agent_id)
                agent.carried_pod_id = pod.pod_id
            active.status = TaskStatus.COMPLETED
            active.completed_at = tick
            agent.clear_path()
            nxt = world.task_state.get_next_task_for_agent(agent.agent_id)
            if (
                _uses_explicit_station_waiting(world)
                and nxt is not None
                and nxt.task_type == TaskType.DELIVER
            ):
                queue = world.station_state.get_queue(nxt.station_id)
                if queue is not None and queue.entry_position is not None:
                    _mark_station_waiting_step(agent, queue, nxt, tick)

        elif active.task_type == TaskType.DELIVER:
            pod = world.pod_state.get_pod(active.pod_id)
            if pod:
                order = world.order_state.orders.get(active.order_id)
                if order:
                    for sku, demand in order.sku_demands.items():
                        if sku in pod.sku_inventory:
                            pod.sku_inventory[sku] = max(0, pod.sku_inventory[sku] - demand)
                    order.mark_pod_delivered(active.pod_id)
            agent.status = AgentStatus.EXITING

        elif active.task_type == TaskType.RETURN:
            pod = world.pod_state.get_pod(active.pod_id)
            if pod:
                pod.put_down(active.destination)
                agent.carried_pod_id = None
            active.status = TaskStatus.COMPLETED
            active.completed_at = tick
            agent.clear_path()
            agent.assigned_task_id = None
            agent.clear_station_waiting()
            nxt = world.task_state.get_next_task_for_agent(agent.agent_id)
            if nxt is None:
                agent.status = AgentStatus.IDLE
                agent.clear_station_waiting()


def _check_completion_step(world, tick):
    for order in world.order_state.get_in_progress_orders():
        if world.task_state.all_order_tasks_completed(order.order_id):
            order.status = OrderStatus.COMPLETED
            order.completed_at = tick
            order_tasks = world.task_state.get_tasks_for_order(order.order_id)
            for t in order_tasks:
                if t.agent_id is not None:
                    ag = world.get_agent(t.agent_id)
                    if ag and ag.assigned_task_id is None and not ag.is_idle:
                        if world.task_state.get_next_task_for_agent(t.agent_id) is None:
                            ag.status = AgentStatus.IDLE


def _detect_conflicts_step(world, prev_positions):
    """Count vertex and swap conflicts after movement (same logic as Engine)."""
    vertex = 0
    swap = 0
    pos_to_agents = {}
    for agent in world.agents:
        pos_to_agents.setdefault(agent.position, []).append(agent.agent_id)
    for pos, aids in pos_to_agents.items():
        if len(aids) > 1:
            vertex += 1
    agents = world.agents
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a, b = agents[i], agents[j]
            a_prev = prev_positions.get(a.agent_id)
            b_prev = prev_positions.get(b.agent_id)
            if a_prev and b_prev and a_prev != a.position:
                if a.position == b_prev and b.position == a_prev:
                    swap += 1
    world.traffic_vertex_conflicts_this_tick = int(vertex)
    world.traffic_swap_conflicts_this_tick = int(swap)
    return vertex, swap


def _process_station_exits_step(world, tick):
    """Standalone version of SimulationEngine._process_station_exits."""
    for agent in world.agents:
        if agent.status != AgentStatus.EXITING:
            continue
        active_task = world.task_state.get_active_task_for_agent(agent.agent_id)
        if active_task is None:
            continue
        queue = world.station_state.get_queue(active_task.station_id)
        if queue is None:
            continue

        if queue.exit_position and agent.position == queue.exit_position:
            active_task.status = TaskStatus.COMPLETED
            active_task.completed_at = tick
            agent.clear_path()
            agent.status = AgentStatus.CARRYING
            agent.assigned_task_id = None
            agent.clear_station_waiting()
            continue

        queue.release_to_exit(agent.agent_id, world)


def _check_queue_arrivals_step(world, tick, absorbed_entries):
    """Standalone version of SimulationEngine._check_queue_arrivals."""
    for agent in world.agents:
        if agent.status != AgentStatus.CARRYING:
            continue
        active_task = world.task_state.get_active_task_for_agent(agent.agent_id)
        if active_task is None or active_task.task_type != TaskType.DELIVER:
            continue
        queue = world.station_state.get_queue(active_task.station_id)
        if queue is None or queue.entry_position is None:
            continue
        if agent.position != queue.entry_position:
            continue
        if agent.has_path:
            agent.clear_path()
        if queue.entry_position in absorbed_entries:
            continue
        if queue.check_in_from_entry(agent.agent_id, world):
            absorbed_entries.add(queue.entry_position)


def _clear_stale_entry_paths_step(world, tick):
    """Standalone version of SimulationEngine._clear_stale_entry_paths."""
    station_state = world.station_state
    claimed_entries = set()

    for agent in world.agents:
        if agent.status != AgentStatus.CARRYING or not agent.has_path:
            continue
        active_task = world.task_state.get_active_task_for_agent(agent.agent_id)
        if active_task is None or active_task.task_type != TaskType.DELIVER:
            continue
        queue = station_state.get_queue(active_task.station_id)
        if queue is None or queue.entry_position is None:
            continue

        remaining = agent.path[agent.path_index:]
        if not remaining or remaining[-1] != queue.entry_position:
            continue

        er, ec = queue.entry_position
        ar, ac = agent.position
        manhattan = abs(er - ar) + abs(ec - ac)
        if manhattan > 3:
            continue
        if len(remaining) <= manhattan:
            continue

        entry_pos = queue.entry_position
        if entry_pos in claimed_entries:
            continue

        entry_blocked = False
        for other in world.agents:
            if other.agent_id == agent.agent_id:
                continue
            if other.position == entry_pos:
                entry_blocked = True
                break
            if other.has_path and other.path_index < len(other.path):
                if other.path[other.path_index] == entry_pos:
                    entry_blocked = True
                    break
        if entry_blocked:
            continue

        agent.clear_path()


def _refill_backlog_step(world, config, order_generator, tick):
    """Mirror ``SimulationEngine._refill_order_backlog`` in a clone."""
    simulation = getattr(config, "simulation", None)
    floor = int(getattr(simulation, "backlog_floor", 0) or 0)
    if floor <= 0:
        return 0

    pending = len(world.order_state.get_pending_orders())
    generated = 0
    while pending < floor:
        if hasattr(order_generator, "generate_one"):
            order = order_generator.generate_one(world, created_at=tick)
        else:
            orders = order_generator.generate(world)
            order = orders[0] if orders else None
        if order is None:
            break
        world.order_state.add_order(order)
        pending += 1
        generated += 1
    return generated


def _stamp_continuation_tasks(tasks, tick, config):
    """Apply the task metadata normally filled by ``SimulationEngine``."""
    for task in tasks:
        if task.created_at is None:
            task.created_at = tick
        if task.agent_id is not None and task.assigned_at is None:
            task.assigned_at = tick
        if task.free_flow_time is None:
            task.free_flow_time = _task_free_flow(
                task.task_type, task.source, task.destination, config,
            )


def _prepare_behavior_continuation_tick(
    world,
    order_generator,
    task_assigner,
    config,
    *,
    generate_orders: bool,
):
    """Run the pre-movement part of a closed-loop continuation tick.

    At the decision tick the outer engine has already generated/refilled its
    order pool, so ``generate_orders`` is false and only the remaining
    assignments are committed after the forced candidate.  Later rollout
    ticks generate orders, refill the backlog, and assign exactly once.

    Returns ``(new_order_count, new_task_count)`` for coverage diagnostics.
    """
    tick = int(world.tick)
    generated = 0
    if generate_orders:
        orders = order_generator.generate(world)
        for order in orders:
            world.order_state.add_order(order)
            generated += 1

        mode = getattr(
            getattr(config, "simulation", None),
            "backlog_refill_mode",
            "none",
        )
        if mode == "on_pre_assignment":
            generated += _refill_backlog_step(
                world, config, order_generator, tick,
            )

    tasks = task_assigner.assign(world)
    _stamp_continuation_tasks(tasks, tick, config)
    return int(generated), int(len(tasks))


def step_world(world, path_planner, config):
    """One tick of simulation without order generation or task assignment.

    Faithful to SimulationEngine._tick() order, including station queue
    cascade, entry/exit processing, and stationary_ticks update.

    Returns (vertex_conflicts, swap_conflicts, blocked_moves).
    """
    tick = world.tick

    tick_start_positions = {a.agent_id: a.position for a in world.agents}

    world.station_state.tick(world)
    _process_station_exits_step(world, tick)

    absorbed_entries = set()
    _check_queue_arrivals_step(world, tick, absorbed_entries)
    _clear_stale_entry_paths_step(world, tick)

    _plan_and_activate_step(world, path_planner, tick, config)
    prev_positions, blocked = _move_agents_step(world)

    _check_queue_arrivals_step(world, tick, absorbed_entries)

    vertex, swap = _detect_conflicts_step(world, prev_positions)
    _handle_actions_step(world, config, tick)
    _check_completion_step(world, tick)

    _update_stationary_ticks(world, tick_start_positions)

    world.advance_tick()
    return vertex, swap, blocked


# ------------------------------------------------------------------
# Full counterfactual evaluation of one candidate
# ------------------------------------------------------------------

def _make_invalid_result(
    horizon,
    num_nodes,
    num_stations,
    *,
    candidate_action_type=ASSIGN_ROBOT_ACTION_TYPE,
):
    """Return an all-zero result for invalid candidates."""
    return {
        "future_node_labels": torch.zeros(horizon, num_nodes, 6),
        "future_system_labels": torch.zeros(horizon, 7),
        "future_station_labels": torch.zeros(horizon, num_stations, 2),
        "future_mask": torch.zeros(horizon),
        "realized_cost": 0.0,
        "rollout_vertex_conflicts": 0,
        "rollout_swap_conflicts": 0,
        "rollout_blocked_moves": 0,
        "rollout_generated_orders": 0,
        "rollout_assigned_tasks": 0,
        "rollout_step_diagnostics": [],
        "rollout_planned_traffic_positive_steps": 0,
        "rollout_blocked_steps": 0,
        "rollout_continuation_mode": "invalid",
        "future_demand_context": None,
        "lyapunov_l0_valid": False,
        "candidate_action_type": candidate_action_type,
        "no_assign_applied": False,
        "no_assign_audit": None,
    }


def evaluate_candidate_rollout(
    world: WorldState,
    candidate: dict,
    fixed_context: dict,
    config,
    path_planner,
    horizon: int,
    node_map: dict,
    local_capacity: list,
    bottleneck_score: list,
    adj: dict,
    reservation_window: int = 1,
    delay_scale: Optional[float] = None,
    station_queue_delta_scale: Optional[float] = None,
    stalled_ratio_threshold: float = 0.3,
    risk_duration: int = 8,
    station_queue_scale: Optional[float] = None,
    station_load_scale: Optional[float] = None,
    record_lyapunov_l0: bool = False,
    lyapunov_l0_config: Optional[dict] = None,
    rollout_continuation_mode: str = "isolated",
    continuation_order_generator=None,
    continuation_task_assigner=None,
    rollout_observer=None,
) -> dict:
    """Clone world, force candidate, run H ticks, collect labels + cost.

    Protects Task._next_id and Order._next_id from pollution.
    Deep-copies path_planner to avoid state leakage.  ``rollout_observer`` is
    an optional diagnostic-only hook with ``on_post_step(world)`` and
    ``finalize(world)`` methods.  The default path is byte-for-byte
    behaviour-compatible with the historical collector.
    """
    num_nodes = len(node_map)
    num_stations = len(world.map_state.station_positions)
    no_assign = is_no_assign_candidate(candidate)
    candidate_action_type = (
        NO_ASSIGN_ACTION_TYPE if no_assign else ASSIGN_ROBOT_ACTION_TYPE
    )

    if rollout_continuation_mode not in ("isolated", "behavior"):
        raise ValueError(
            "rollout_continuation_mode must be 'isolated' or 'behavior'"
        )
    if rollout_continuation_mode == "behavior" and (
        continuation_order_generator is None
        or continuation_task_assigner is None
    ):
        raise ValueError(
            "behavior continuation requires both an order generator and "
            "a task assigner"
        )
    if no_assign and rollout_continuation_mode != "isolated":
        raise ValueError(
            "native NO_ASSIGN is defined only for isolated rollout; a "
            "behavior continuation would immediately invoke another "
            "scheduler and invalidate the action label"
        )

    saved_task_next_id = Task._next_id
    saved_order_next_id = Order._next_id
    saved_python_rng = random.getstate()
    saved_numpy_rng = np.random.get_state()
    saved_torch_rng = torch.random.get_rng_state()
    try:
        cloned = copy.deepcopy(world)

        l0_config = None
        l0_start = None
        l0_post_action = None
        l0_trajectory = []
        l0_traffic_trajectory = []
        if record_lyapunov_l0:
            from WorldModel.core.lyapunov import (
                LyapunovL0Config,
                TRAFFIC_DIAGNOSTIC_NAMES,
                compute_lyapunov_snapshot,
            )
            l0_config = LyapunovL0Config(**dict(lyapunov_l0_config or {}))
            l0_start = compute_lyapunov_snapshot(cloned, l0_config)

        baseline_station_queues = _snapshot_station_queues(cloned)

        no_assign_before = None
        no_assign_after_apply = None
        if no_assign:
            no_assign_before = _no_assign_context_snapshot(
                cloned, fixed_context
            )
            ok = _validate_no_assign_candidate(cloned, fixed_context)
            no_assign_after_apply = _no_assign_context_snapshot(
                cloned, fixed_context
            )
        else:
            ok = force_apply_candidate(
                cloned, candidate, fixed_context, config
            )
        if not ok:
            invalid = _make_invalid_result(
                horizon,
                num_nodes,
                num_stations,
                candidate_action_type=candidate_action_type,
            )
            if rollout_observer is not None:
                invalid["rollout_observer_output"] = None
            return invalid
        if no_assign and no_assign_before != no_assign_after_apply:
            raise RuntimeError(
                "native NO_ASSIGN mutated the fixed dispatch context"
            )
        if record_lyapunov_l0:
            # Separate the instantaneous assignment/arrival injection from
            # the subsequent H-step physical evolution.  Without this state,
            # an endpoint head cannot tell whether an arrival-bin change came
            # from the candidate itself or from task progress during rollout.
            l0_post_action = compute_lyapunov_snapshot(cloned, l0_config)
            if no_assign and abs(
                float(l0_post_action.total) - float(l0_start.total)
            ) > 1e-9:
                raise RuntimeError(
                    "NO_ASSIGN changed Lyapunov potential before physical "
                    "rollout"
                )

        try:
            rollout_planner = copy.deepcopy(path_planner)
        except Exception:
            rollout_planner = path_planner

        rollout_order_generator = None
        rollout_task_assigner = None
        if rollout_continuation_mode == "behavior":
            try:
                rollout_order_generator = copy.deepcopy(
                    continuation_order_generator
                )
                rollout_task_assigner = copy.deepcopy(
                    continuation_task_assigner
                )
            except Exception as exc:
                raise RuntimeError(
                    "behavior continuation policies must be deepcopy-safe; "
                    "refusing to mutate the live engine policy"
                ) from exc
            if hasattr(rollout_task_assigner, "path_planner"):
                rollout_task_assigner.path_planner = rollout_planner

        window_start_tick = cloned.tick
        prev_completed = cloned.order_state.total_completed

        future_node = []
        future_sys = []
        future_sta = []
        future_mask = []

        total_vertex_conflicts = 0
        total_swap_conflicts = 0
        total_blocked_moves = 0
        total_generated_orders = 0
        total_assigned_tasks = 0

        forced_robot_id = (
            None if no_assign else candidate.get("robot_id")
        )
        forced_no_path_streak = 0
        if delay_scale is None:
            delay_scale = float(horizon)
        risk_scale = max(float(risk_duration), 1.0)

        for k in range(horizon):
            try:
                continuation_tick = int(cloned.tick)
                if rollout_continuation_mode == "behavior":
                    generated, assigned = _prepare_behavior_continuation_tick(
                        cloned,
                        rollout_order_generator,
                        rollout_task_assigner,
                        config,
                        # The decision tick's order generation/refill already
                        # happened before the collector callback.
                        generate_orders=(k > 0),
                    )
                    total_generated_orders += generated
                    total_assigned_tasks += assigned
                vc, sc, bm = step_world(cloned, rollout_planner, config)
                total_vertex_conflicts += vc
                total_swap_conflicts += sc
                total_blocked_moves += bm
                if rollout_observer is not None:
                    try:
                        rollout_observer.on_post_step(cloned)
                    except Exception as exc:
                        raise RolloutObserverError(
                            "counterfactual rollout observer failed"
                        ) from exc
                if (
                    rollout_continuation_mode == "behavior"
                    and getattr(
                        getattr(config, "simulation", None),
                        "backlog_refill_mode",
                        "none",
                    ) == "on_completed"
                ):
                    total_generated_orders += _refill_backlog_step(
                        cloned,
                        config,
                        rollout_order_generator,
                        continuation_tick,
                    )
            except RolloutObserverError:
                raise
            except Exception:
                for _ in range(horizon - k):
                    future_node.append(torch.zeros(num_nodes, 6))
                    future_sys.append(torch.zeros(7))
                    future_sta.append(torch.zeros(num_stations, 2))
                    future_mask.append(0.0)
                break

            if forced_robot_id is None:
                forced_no_path_streak = 0
            else:
                forced_agent = cloned.get_agent(forced_robot_id)
                if (forced_agent is not None
                        and forced_agent.status != AgentStatus.IDLE
                        and forced_agent.plan_failed_streak >= 2):
                    forced_no_path_streak += 1
                else:
                    forced_no_path_streak = 0

            risk_override = max(0, forced_no_path_streak - 1) / risk_scale
            risk_override = max(0.0, min(1.0, risk_override))

            nl = extract_node_labels(cloned, node_map, local_capacity, bottleneck_score,
                                     adj=adj, reservation_window=reservation_window)
            sl = extract_system_labels(
                cloned, bottleneck_score, node_map, local_capacity,
                adj=adj,
                prev_completed=prev_completed, engine=None,
                baseline_station_queue=baseline_station_queues,
                window_start_tick=window_start_tick,
                risk_override=risk_override,
                delay_scale=delay_scale,
                station_queue_delta_scale=station_queue_delta_scale,
                reservation_window=reservation_window,
                stalled_ratio_threshold=stalled_ratio_threshold,
            )
            stl = extract_station_labels(
                cloned,
                station_queue_scale=station_queue_scale,
                station_load_scale=station_load_scale,
            )

            future_node.append(nl)
            future_sys.append(sl)
            future_sta.append(stl)
            future_mask.append(1.0)
            prev_completed = cloned.order_state.total_completed
            if record_lyapunov_l0:
                snapshot = compute_lyapunov_snapshot(cloned, l0_config)
                l0_trajectory.append(snapshot)
                l0_traffic_trajectory.append([
                    float(snapshot.traffic_diagnostics.get(name, 0.0))
                    for name in TRAFFIC_DIAGNOSTIC_NAMES
                ])

        future_node_t = torch.stack(future_node)
        future_sys_t = torch.stack(future_sys)
        future_sta_t = torch.stack(future_sta)
        future_mask_t = torch.tensor(future_mask, dtype=torch.float32)

        realized_cost = compute_realized_cost(future_sys_t)

        complete_rollout = bool(
            future_mask_t.numel() == horizon
            and bool((future_mask_t > 0.5).all())
        )
        result = {
            "future_node_labels": future_node_t,
            "future_system_labels": future_sys_t,
            "future_station_labels": future_sta_t,
            "future_mask": future_mask_t,
            "realized_cost": realized_cost,
            "rollout_vertex_conflicts": total_vertex_conflicts,
            "rollout_swap_conflicts": total_swap_conflicts,
            "rollout_blocked_moves": total_blocked_moves,
            "rollout_generated_orders": total_generated_orders,
            "rollout_assigned_tasks": total_assigned_tasks,
            "rollout_continuation_mode": rollout_continuation_mode,
            "candidate_action_type": candidate_action_type,
            "no_assign_applied": bool(no_assign),
            # The endpoint demand target closes the train/inference signature
            # for future demand prediction; online V-head use remains disabled
            # until such a predictor is trained and supplied.
            "future_demand_context": extract_demand_context(cloned),
            # A rollout exception pads the remaining steps with zeros.  That
            # right-censored endpoint must not supervise the physical head or
            # endpoint-demand predictor.
            "lyapunov_l0_valid": bool(record_lyapunov_l0 and complete_rollout),
        }
        if rollout_observer is not None:
            result["rollout_observer_output"] = (
                rollout_observer.finalize(cloned)
                if complete_rollout else None
            )
        if no_assign:
            no_assign_after_rollout = _no_assign_context_snapshot(
                cloned, fixed_context
            )
            before_task_ids = set(no_assign_before["all_task_ids"])
            after_task_ids = set(no_assign_after_rollout["all_task_ids"])
            result["no_assign_audit"] = {
                "schema_version": "native_no_assign_audit_v1",
                "immediate_context_unchanged": bool(
                    no_assign_before == no_assign_after_apply
                ),
                "tasks_added_immediately": [],
                "tasks_added_during_isolated_rollout": sorted(
                    int(value) for value in after_task_ids - before_task_ids
                ),
                "rollout_generated_orders": int(total_generated_orders),
                "rollout_assigned_tasks": int(total_assigned_tasks),
                "pre_action": no_assign_before,
                "post_action": no_assign_after_apply,
                "endpoint": no_assign_after_rollout,
            }
        else:
            result["no_assign_audit"] = None
        if record_lyapunov_l0 and complete_rollout:
            from WorldModel.core.lyapunov import (
                LYAPUNOV_COLLECTION_SCHEMA_VERSION,
                TRAFFIC_DIAGNOSTIC_NAMES,
                compute_productive_progress,
            )

            l0_end = (
                l0_trajectory[-1] if l0_trajectory
                else compute_lyapunov_snapshot(cloned, l0_config)
            )
            valid_steps = max(int(future_mask_t.sum().item()), 1)
            progress = compute_productive_progress(
                l0_start, l0_end, horizon=valid_steps
            )
            summaries = [snapshot.physical_summary() for snapshot in l0_trajectory]
            station_ids = tuple(sorted(
                int(station_id)
                for station_id in l0_post_action.station_work
            ))
            station_work_trajectory = [
                [
                    float(snapshot.station_work.get(station_id, 0.0))
                    for station_id in station_ids
                ]
                for snapshot in l0_trajectory
            ]
            while len(summaries) < horizon:
                summaries.append((0.0,) * 12)
            while len(l0_traffic_trajectory) < horizon:
                l0_traffic_trajectory.append(
                    [0.0] * len(TRAFFIC_DIAGNOSTIC_NAMES)
                )
            result.update({
                "lyapunov_l0_collection_schema_version": (
                    LYAPUNOV_COLLECTION_SCHEMA_VERSION
                ),
                "lyapunov_l0_start": l0_start.to_dict(),
                "lyapunov_l0_post_action": l0_post_action.to_dict(),
                "lyapunov_l0_end": l0_end.to_dict(),
                "lyapunov_l0_immediate_delta": float(
                    l0_post_action.total - l0_start.total
                ),
                "lyapunov_l0_delta": float(l0_end.total - l0_start.total),
                "lyapunov_l0_trajectory": torch.tensor(
                    summaries, dtype=torch.float32
                ).reshape(horizon, 12),
                # The aggregate physical-summary trajectory is insufficient
                # for station-wise residual targets at intermediate horizons.
                # Persist the already-computed station ledger at every tick so
                # one H=max rollout can provide paired H=5/10/15/20 labels.
                "analytic_work_relief_trajectory_schema_version": (
                    "analytic_work_relief_trajectory_v1"
                ),
                "lyapunov_l0_station_ids": torch.tensor(
                    station_ids, dtype=torch.long
                ),
                "lyapunov_l0_station_work_trajectory": torch.tensor(
                    station_work_trajectory, dtype=torch.float32
                ).reshape(horizon, len(station_ids)),
                "lyapunov_l0_traffic_trajectory_names": tuple(
                    TRAFFIC_DIAGNOSTIC_NAMES
                ),
                "lyapunov_l0_traffic_trajectory": torch.tensor(
                    l0_traffic_trajectory, dtype=torch.float32
                ).reshape(horizon, len(TRAFFIC_DIAGNOSTIC_NAMES)),
                "lyapunov_l0_progress": progress.to_dict(),
                # Persist resolved defaults so the downstream functional is
                # reproducible even when the caller supplied no overrides.
                "lyapunov_l0_config": asdict(l0_config),
            })
        return result
    finally:
        Task._next_id = saved_task_next_id
        Order._next_id = saved_order_next_id
        random.setstate(saved_python_rng)
        np.random.set_state(saved_numpy_rng)
        torch.random.set_rng_state(saved_torch_rng)


def _snapshot_station_queues(world: WorldState) -> Dict[int, float]:
    """Capture current station physical occupancy ratios as baseline for delta."""
    ratios = {}
    for sid in world.map_state.station_positions:
        sq = world.station_state.get_queue(sid)
        if sq is not None:
            ratios[sid] = float(sq.occupancy()) / max(sq.capacity, 1)
        else:
            ratios[sid] = 0.0
    return ratios
