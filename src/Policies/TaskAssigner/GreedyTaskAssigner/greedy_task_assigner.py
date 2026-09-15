"""
Greedy Task Assigner
====================
Default implementation: greedy assignment of orders to nearest idle robots.

Supports two task execution modes:
- ``"parallel"`` (default): each pod in an order is assigned to a
  different idle robot, so multiple robots work in parallel.
  Uses the propose / select / commit pipeline.
- ``"serial"``: all pods in an order are assigned to the same robot,
  which processes them one by one in sequence.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from Policies.TaskAssigner.base_task_assigner import (
    BaseTaskAssigner, AssignmentContext, fulfilled_pod_ids_for_order,
)
from WorldState.task_state import Task, TaskType, TaskStatus
from WorldState.order_state import OrderStatus
from WorldState.agent_state import AgentStatus

logger = logging.getLogger("MAS_RMFS.TaskAssigner")


def _manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class GreedyTaskAssigner(BaseTaskAssigner):

    def __init__(self):
        super().__init__()
        self._serial_propose_warned = False

    # ------------------------------------------------------------------
    # assign() — top-level entry, backward compatible
    # ------------------------------------------------------------------

    def assign(self, world_state) -> List[Task]:
        pending_orders = world_state.order_state.get_pending_orders()
        for order in pending_orders:
            if not order.pod_ids and self.pod_retriever is not None:
                order.pod_ids = self.pod_retriever.retrieve(order, world_state)

        mode = world_state.config.simulation.task_execution_mode
        if mode == "serial":
            return self._assign_serial(world_state)

        contexts = self.propose_assignment_contexts(world_state)
        if not contexts:
            return []
        robot_choices = self.select_robots(world_state, contexts)
        return self.commit_assignments(world_state, contexts, robot_choices)

    # ------------------------------------------------------------------
    # propose_assignment_contexts — read-only (except pod_ids fill)
    # ------------------------------------------------------------------

    def propose_assignment_contexts(
        self,
        world_state,
        max_contexts: Optional[int] = None,
    ) -> List[AssignmentContext]:
        mode = getattr(
            getattr(world_state, 'config', None),
            'simulation', None,
        )
        if mode and getattr(mode, 'task_execution_mode', 'parallel') == "serial":
            if not self._serial_propose_warned:
                logger.info(
                    "serial mode: propose() skipped "
                    "(action_scope=order_pod_robot requires parallel)"
                )
                self._serial_propose_warned = True
            return []

        pending_orders = world_state.order_state.get_pending_orders()
        for order in pending_orders:
            if not order.pod_ids and self.pod_retriever is not None:
                order.pod_ids = self.pod_retriever.retrieve(order, world_state)

        reserved_pods: Set[int] = {
            t.pod_id
            for t in world_state.task_state.tasks.values()
            if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        }
        proposed_pods: Set[int] = set(reserved_pods)

        num_idle = len(world_state.get_idle_agents())
        limit = max(0, int(max_contexts)) if max_contexts is not None else num_idle

        contexts: List[AssignmentContext] = []

        for order in pending_orders:
            if len(contexts) >= limit:
                break
            fulfilled_pods = fulfilled_pod_ids_for_order(world_state, order)
            for pod_id in order.pod_ids:
                if len(contexts) >= limit:
                    break

                if int(pod_id) in fulfilled_pods:
                    continue

                pod = world_state.pod_state.get_pod(pod_id)
                if pod is None or pod.is_carried or pod_id in proposed_pods:
                    continue

                station_loc = world_state.station_state.get_service_position(
                    order.station_id
                )
                if station_loc is None:
                    station_loc = world_state.map_state.station_positions.get(
                        order.station_id
                    )
                if station_loc is None:
                    continue

                entry_pos = world_state.station_state.get_entry_position(
                    order.station_id
                )
                exit_pos = world_state.station_state.get_exit_position(
                    order.station_id
                )

                return_source = exit_pos or station_loc
                if self.pod_return_planner is not None:
                    return_dest = self.pod_return_planner.plan_return(
                        pod, return_source, world_state
                    )
                else:
                    return_dest = pod.home_position

                ctx = AssignmentContext(
                    order_id=order.order_id,
                    pod_id=pod_id,
                    pod_location=pod.current_position,
                    station_id=order.station_id,
                    station_location=station_loc,
                    entry_position=entry_pos,
                    exit_position=exit_pos,
                    return_location=return_dest,
                    order_size=sum(order.sku_demands.values()),
                )
                contexts.append(ctx)
                proposed_pods.add(pod_id)

        return contexts

    # ------------------------------------------------------------------
    # select_robots — greedy nearest
    # ------------------------------------------------------------------

    def select_robots(
        self,
        world_state,
        contexts: List[AssignmentContext],
    ) -> Dict[int, int]:
        idle_agents = world_state.get_idle_agents()
        used_agents: Set[int] = set()
        choices: Dict[int, int] = {}

        for i, ctx in enumerate(contexts):
            available = [
                a for a in idle_agents if a.agent_id not in used_agents
            ]
            if not available:
                break
            available.sort(
                key=lambda a: _manhattan_distance(a.position, ctx.pod_location)
            )
            best = available[0]
            choices[i] = best.agent_id
            used_agents.add(best.agent_id)

        return choices

    # ------------------------------------------------------------------
    # commit_assignments — create tasks, mutate world
    # ------------------------------------------------------------------

    def commit_assignments(
        self,
        world_state,
        contexts: List[AssignmentContext],
        robot_choices: Dict[int, int],
    ) -> List[Task]:
        new_tasks: List[Task] = []
        reserved_pods: Set[int] = {
            t.pod_id
            for t in world_state.task_state.tasks.values()
            if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        }
        committed_order_ids: Set[int] = set()

        for i, ctx in enumerate(contexts):
            robot_id = robot_choices.get(i)
            if robot_id is None:
                continue

            order = world_state.order_state.orders.get(ctx.order_id)
            if order is None or order.status != OrderStatus.PENDING:
                continue
            if int(ctx.pod_id) in fulfilled_pod_ids_for_order(
                world_state, order,
            ):
                continue

            agent = world_state.get_agent(robot_id)
            if agent is None or agent.status != AgentStatus.IDLE:
                continue

            pod = world_state.pod_state.get_pod(ctx.pod_id)
            if pod is None or pod.is_carried or ctx.pod_id in reserved_pods:
                continue

            pick_task = Task(
                task_type=TaskType.PICK,
                order_id=ctx.order_id,
                pod_id=ctx.pod_id,
                source=agent.position,
                destination=ctx.pod_location,
            )
            pick_task.agent_id = robot_id
            pick_task.status = TaskStatus.ASSIGNED

            deliver_task = Task(
                task_type=TaskType.DELIVER,
                order_id=ctx.order_id,
                pod_id=ctx.pod_id,
                source=ctx.pod_location,
                destination=ctx.station_location,
            )
            deliver_task.agent_id = robot_id
            deliver_task.status = TaskStatus.ASSIGNED
            deliver_task.station_id = ctx.station_id

            return_source = ctx.exit_position or ctx.station_location
            return_task = Task(
                task_type=TaskType.RETURN,
                order_id=ctx.order_id,
                pod_id=ctx.pod_id,
                source=return_source,
                destination=ctx.return_location,
            )
            return_task.agent_id = robot_id
            return_task.status = TaskStatus.ASSIGNED
            return_task.station_id = ctx.station_id

            world_state.task_state.add_task(pick_task)
            world_state.task_state.add_task(deliver_task)
            world_state.task_state.add_task(return_task)

            new_tasks.extend([pick_task, deliver_task, return_task])
            reserved_pods.add(ctx.pod_id)
            committed_order_ids.add(int(ctx.order_id))

            agent.status = AgentStatus.MOVING_TO_POD
            agent.assigned_task_id = pick_task.task_id

        self._mark_orders_in_progress(world_state, committed_order_ids)

        return new_tasks

    @staticmethod
    def _mark_orders_in_progress(world_state, committed_order_ids):
        """Mark orders IN_PROGRESS only when ALL their pods are covered."""
        for oid in committed_order_ids:
            order = world_state.order_state.orders.get(oid)
            if order is None or order.status != OrderStatus.PENDING:
                continue
            covered_pods = {
                int(task.pod_id)
                for task in world_state.task_state.tasks.values()
                if int(getattr(task, "order_id", -1)) == int(oid)
                and task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
            }
            covered_pods.update(
                fulfilled_pod_ids_for_order(world_state, order)
            )
            if order.pod_ids and all(
                int(pid) in covered_pods for pid in order.pod_ids
            ):
                order.status = OrderStatus.IN_PROGRESS

    # ------------------------------------------------------------------
    # _assign_serial — unchanged
    # ------------------------------------------------------------------

    def _assign_serial(self, world_state) -> List[Task]:
        new_tasks = []
        pending_orders = world_state.order_state.get_pending_orders()

        reserved_pods = {
            t.pod_id
            for t in world_state.task_state.tasks.values()
            if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        }

        for order in pending_orders:
            fulfilled_pods = fulfilled_pod_ids_for_order(world_state, order)
            order_pods = []
            all_available = True
            for pod_id in order.pod_ids:
                if int(pod_id) in fulfilled_pods:
                    continue
                pod = world_state.pod_state.get_pod(pod_id)
                if pod is None or pod.is_carried or pod_id in reserved_pods:
                    all_available = False
                    break
                order_pods.append(pod)

            if not all_available:
                order.pod_ids = []
                continue
            if not order_pods:
                if order.pod_ids and all(
                    int(pod_id) in fulfilled_pods for pod_id in order.pod_ids
                ):
                    order.status = OrderStatus.IN_PROGRESS
                continue

            idle_agents = world_state.get_idle_agents()
            if not idle_agents:
                continue

            idle_agents.sort(
                key=lambda a: _manhattan_distance(
                    a.position, order_pods[0].current_position
                )
            )
            agent = idle_agents[0]

            station_pos = world_state.station_state.get_service_position(
                order.station_id
            )
            if station_pos is None:
                station_pos = world_state.map_state.station_positions.get(
                    order.station_id
                )
            if station_pos is None:
                continue

            order_tasks = []
            for pod in order_pods:
                pick_task = Task(
                    task_type=TaskType.PICK,
                    order_id=order.order_id,
                    pod_id=pod.pod_id,
                    source=agent.position,
                    destination=pod.current_position,
                )
                pick_task.agent_id = agent.agent_id
                pick_task.status = TaskStatus.ASSIGNED

                deliver_task = Task(
                    task_type=TaskType.DELIVER,
                    order_id=order.order_id,
                    pod_id=pod.pod_id,
                    source=pod.current_position,
                    destination=station_pos,
                )
                deliver_task.agent_id = agent.agent_id
                deliver_task.status = TaskStatus.ASSIGNED
                deliver_task.station_id = order.station_id

                exit_pos = world_state.station_state.get_exit_position(
                    order.station_id
                )
                return_source = exit_pos or station_pos
                if self.pod_return_planner is not None:
                    return_dest = self.pod_return_planner.plan_return(
                        pod, return_source, world_state
                    )
                else:
                    return_dest = pod.home_position

                return_task = Task(
                    task_type=TaskType.RETURN,
                    order_id=order.order_id,
                    pod_id=pod.pod_id,
                    source=return_source,
                    destination=return_dest,
                )
                return_task.agent_id = agent.agent_id
                return_task.status = TaskStatus.ASSIGNED
                return_task.station_id = order.station_id

                world_state.task_state.add_task(pick_task)
                world_state.task_state.add_task(deliver_task)
                world_state.task_state.add_task(return_task)

                order_tasks.extend([pick_task, deliver_task, return_task])
                reserved_pods.add(pod.pod_id)

            new_tasks.extend(order_tasks)

            agent.status = AgentStatus.MOVING_TO_POD
            agent.assigned_task_id = order_tasks[0].task_id

            order.status = OrderStatus.IN_PROGRESS

        return new_tasks
