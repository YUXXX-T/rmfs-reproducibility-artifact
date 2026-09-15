"""
Unified Risk Signal
===================
Shared risk computation used by both training label generation
(counterfactual_rollout / graph_builder) and online evaluation
(SimulationEngine / OnlineLabelProbe). Ensures label-evaluation consistency.
"""

from typing import Dict, Optional

from WorldState.agent_state import AgentStatus


def _update_stationary_ticks(world, tick_start_positions: Dict[int, tuple]):
    """Update stationary_ticks at end of tick.

    tick_start_positions: dict saved at TICK START (before any station/move logic).
    Must be called by both SimulationEngine._tick() and counterfactual step_world().
    """
    for agent in world.agents:
        start_pos = tick_start_positions.get(agent.agent_id)
        if (
            agent.status != AgentStatus.IDLE
            and not agent.is_waiting
            and start_pos == agent.position
        ):
            agent.stationary_ticks += 1
        else:
            agent.stationary_ticks = 0


def _is_handoff_blocked(agent, world) -> bool:
    """Check if agent is stuck in station handoff (entry/exit/queuing)."""
    if agent.status == AgentStatus.IDLE:
        return False
    if not agent.stuck_this_tick:
        return False
    if agent.is_waiting:
        return False

    if agent.status in (AgentStatus.QUEUING, AgentStatus.EXITING):
        return True

    if agent.status in (AgentStatus.CARRYING, AgentStatus.DELIVERING):
        active = world.task_state.get_active_task_for_agent(agent.agent_id)
        if active is not None:
            from WorldState.task_state import TaskType
            if active.task_type == TaskType.DELIVER:
                order = world.order_state.orders.get(active.order_id)
                if order is not None:
                    entry_pos = world.station_state.get_entry_position(order.station_id)
                    if entry_pos is not None:
                        dist = abs(agent.position[0] - entry_pos[0]) + abs(agent.position[1] - entry_pos[1])
                        if dist <= 2:
                            return True
    return False


def compute_unified_risk(
    world,
    deadlock_threshold: int = 10,
    forced_risk_override: Optional[float] = None,
) -> dict:
    """Unified per-tick risk signal.

    Used in training labels AND online evaluation.

    Parameters
    ----------
    forced_risk_override : float or None
        Counterfactual forced-robot blocked risk.  When a candidate is
        force-applied, the robot may be immediately blocked; this external
        signal is folded into unified_risk via max() so it never lowers
        the risk floor.

    Returns dict with raw components and unified_risk in [0, 1].
    """
    num_robots = max(len(world.agents), 1)

    stall_count = sum(
        1 for a in world.agents
        if (a.stuck_this_tick and not a.is_waiting) or a.plan_failed_streak >= 2
    )
    stall_ratio = stall_count / num_robots

    deadlock_count = sum(
        1 for a in world.agents
        if a.status != AgentStatus.IDLE
        and not a.is_waiting
        and a.stationary_ticks >= deadlock_threshold
    )
    deadlock_ratio = deadlock_count / num_robots

    handoff_blocked = sum(1 for a in world.agents if _is_handoff_blocked(a, world))
    handoff_ratio = handoff_blocked / num_robots

    unified_risk = min(1.0, max(
        stall_ratio / 0.30,
        deadlock_ratio / 0.10,
        handoff_ratio / 0.30,
    ))

    if forced_risk_override is not None:
        forced = max(0.0, min(1.0, forced_risk_override))
        unified_risk = max(unified_risk, forced)

    return {
        "stall_ratio": stall_ratio,
        "deadlock_ratio": deadlock_ratio,
        "handoff_ratio": handoff_ratio,
        "unified_risk": unified_risk,
    }
