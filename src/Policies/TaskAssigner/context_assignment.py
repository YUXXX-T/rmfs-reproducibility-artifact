"""Policy-neutral fixed-context proposal and task-chain execution.

These helpers do not choose a robot and do not score competing assignments.
They expose an already-valid order/pod/station context to a robot-ranking
policy and materialise the policy's explicit robot choice into tasks.
"""

from typing import Dict, List, Optional, Set

from Policies.TaskAssigner.base_task_assigner import (
    AssignmentContext,
    fulfilled_pod_ids_for_order,
)
from WorldState.agent_state import AgentStatus
from WorldState.order_state import OrderStatus
from WorldState.task_state import Task, TaskStatus, TaskType


def materialize_fixed_order_pod_contexts(world_state, pod_retriever) -> None:
    """Resolve order inventory context before a robot policy is called.

    This fixes the physical order/pod context for conditional robot ranking;
    it never inspects idle robots or chooses a robot.
    """
    if pod_retriever is None:
        return
    for order in world_state.order_state.get_pending_orders():
        if not order.pod_ids:
            order.pod_ids = pod_retriever.retrieve(order, world_state)


def propose_fixed_assignment_contexts(
    assigner,
    world_state,
    max_contexts: Optional[int] = None,
) -> List[AssignmentContext]:
    """Expose valid fixed contexts without selecting or ranking robots."""
    return enumerate_pending_assignment_contexts(
        assigner,
        world_state,
        max_contexts=max_contexts,
        include_temporarily_unavailable=False,
        deduplicate_pods=True,
    )


def enumerate_pending_assignment_contexts(
    assigner,
    world_state,
    max_contexts: Optional[int] = None,
    *,
    include_temporarily_unavailable: bool = False,
    deduplicate_pods: bool = False,
) -> List[AssignmentContext]:
    """Enumerate materialised pending order/pod chains without robot ranking.

    ``propose_fixed_assignment_contexts`` keeps its legacy, immediately
    dispatchable semantics by calling this helper with unavailable pods
    excluded and pod IDs de-duplicated.  The dispatch-potential ledger uses
    the full form: a pod shared by two pending orders represents two distinct
    service debts, while a chain that already entered its own task pipeline is
    removed.  Reservation by *another* chain only makes a debt temporarily
    ineligible; it does not erase that chain from the ledger.
    """
    pending_orders = world_state.order_state.get_pending_orders()

    active_statuses = (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
    active_chain_keys = {
        (int(task.order_id), int(task.pod_id))
        for task in world_state.task_state.tasks.values()
        if task.status in active_statuses
    }
    reserved_pods: Set[int] = {
        int(task.pod_id)
        for task in world_state.task_state.tasks.values()
        if task.status in active_statuses
    }
    proposed_pods: Set[int] = set(reserved_pods) if deduplicate_pods else set()
    idle_count = len(world_state.get_idle_agents())
    limit = (
        max(0, int(max_contexts))
        if max_contexts is not None
        else (None if include_temporarily_unavailable else idle_count)
    )

    contexts: List[AssignmentContext] = []
    for order in pending_orders:
        if limit is not None and len(contexts) >= limit:
            break
        fulfilled_pods = fulfilled_pod_ids_for_order(world_state, order)
        for pod_id in order.pod_ids:
            if limit is not None and len(contexts) >= limit:
                break
            pod_id = int(pod_id)
            if pod_id in fulfilled_pods:
                continue
            if (int(order.order_id), pod_id) in active_chain_keys:
                continue
            pod = world_state.pod_state.get_pod(pod_id)
            if pod is None:
                continue
            temporarily_unavailable = (
                bool(pod.is_carried) or pod_id in reserved_pods
            )
            if temporarily_unavailable and not include_temporarily_unavailable:
                continue
            if deduplicate_pods and pod_id in proposed_pods:
                continue

            station_location = world_state.station_state.get_service_position(
                order.station_id
            )
            if station_location is None:
                station_location = world_state.map_state.station_positions.get(
                    order.station_id
                )
            if station_location is None:
                continue

            entry_position = world_state.station_state.get_entry_position(
                order.station_id
            )
            exit_position = world_state.station_state.get_exit_position(
                order.station_id
            )
            return_source = exit_position or station_location
            if assigner.pod_return_planner is not None:
                return_location = assigner.pod_return_planner.plan_return(
                    pod, return_source, world_state
                )
            else:
                return_location = pod.home_position

            contexts.append(AssignmentContext(
                order_id=int(order.order_id),
                pod_id=int(pod_id),
                pod_location=tuple(pod.current_position),
                station_id=int(order.station_id),
                station_location=tuple(station_location),
                entry_position=(
                    tuple(entry_position) if entry_position is not None else None
                ),
                exit_position=(
                    tuple(exit_position) if exit_position is not None else None
                ),
                return_location=tuple(return_location),
                order_size=int(sum(order.sku_demands.values())),
            ))
            if deduplicate_pods:
                proposed_pods.add(pod_id)
    return contexts


def _mark_orders_in_progress(world_state, committed_order_ids: Set[int]) -> None:
    for order_id in committed_order_ids:
        order = world_state.order_state.orders.get(order_id)
        if order is None or order.status != OrderStatus.PENDING:
            continue
        covered_pods = {
            int(task.pod_id)
            for task in world_state.task_state.tasks.values()
            if int(getattr(task, "order_id", -1)) == int(order_id)
            and task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        }
        covered_pods.update(fulfilled_pod_ids_for_order(world_state, order))
        if order.pod_ids and all(
            int(pod_id) in covered_pods for pod_id in order.pod_ids
        ):
            order.status = OrderStatus.IN_PROGRESS


def commit_fixed_context_assignments(
    world_state,
    contexts: List[AssignmentContext],
    robot_choices: Dict[int, int],
) -> List[Task]:
    """Materialise explicit context-index -> robot-id choices only."""
    new_tasks: List[Task] = []
    reserved_pods: Set[int] = {
        int(task.pod_id)
        for task in world_state.task_state.tasks.values()
        if task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
    }
    committed_order_ids: Set[int] = set()

    for context_index, context in enumerate(contexts):
        robot_id = robot_choices.get(context_index)
        if robot_id is None:
            continue
        order = world_state.order_state.orders.get(context.order_id)
        if order is None or order.status != OrderStatus.PENDING:
            continue
        if int(context.pod_id) in fulfilled_pod_ids_for_order(world_state, order):
            continue
        agent = world_state.get_agent(robot_id)
        if agent is None or agent.status != AgentStatus.IDLE:
            continue
        pod = world_state.pod_state.get_pod(context.pod_id)
        if (
            pod is None
            or pod.is_carried
            or int(context.pod_id) in reserved_pods
        ):
            continue

        pick_task = Task(
            task_type=TaskType.PICK,
            order_id=context.order_id,
            pod_id=context.pod_id,
            source=agent.position,
            destination=context.pod_location,
        )
        pick_task.agent_id = int(robot_id)
        pick_task.status = TaskStatus.ASSIGNED

        deliver_task = Task(
            task_type=TaskType.DELIVER,
            order_id=context.order_id,
            pod_id=context.pod_id,
            source=context.pod_location,
            destination=context.station_location,
        )
        deliver_task.agent_id = int(robot_id)
        deliver_task.status = TaskStatus.ASSIGNED
        deliver_task.station_id = context.station_id

        return_source = context.exit_position or context.station_location
        return_task = Task(
            task_type=TaskType.RETURN,
            order_id=context.order_id,
            pod_id=context.pod_id,
            source=return_source,
            destination=context.return_location,
        )
        return_task.agent_id = int(robot_id)
        return_task.status = TaskStatus.ASSIGNED
        return_task.station_id = context.station_id

        world_state.task_state.add_task(pick_task)
        world_state.task_state.add_task(deliver_task)
        world_state.task_state.add_task(return_task)
        new_tasks.extend((pick_task, deliver_task, return_task))
        reserved_pods.add(int(context.pod_id))
        committed_order_ids.add(int(context.order_id))
        agent.status = AgentStatus.MOVING_TO_POD
        agent.assigned_task_id = pick_task.task_id

    _mark_orders_in_progress(world_state, committed_order_ids)
    return new_tasks
