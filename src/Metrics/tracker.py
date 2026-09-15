"""
Metrics Tracker Module
======================
Per-tick metric collection for the RMFS simulation.
Records throughput, congestion, deadlocks, and timing data.
"""

import time
from dataclasses import dataclass, field

from WorldState.world import WorldState
from WorldState.agent_state import AgentStatus


@dataclass
class TickMetrics:
    tick: int
    throughput_cumulative: int      # 到当前 tick 完成的总订单数
    throughput_delta: int           # 本 tick 新完成的订单数
    queue_length: int               # 待分配订单队列长度
    idle_agents: int                # 空闲 agent 数
    congestion_count: int           # 本 tick 发生 vertex conflict 的节点数
    deadlock_agents: int            # 连续 N tick 未移动的 agent 数
    planning_ms: float              # 路径规划耗时
    assignment_ms: float            # 任务分配耗时


class MetricsTracker:
    """Per-tick metrics recorder with deadlock detection and timing utilities."""

    def __init__(self, deadlock_threshold: int = 10):
        self.history: list[TickMetrics] = []
        self.deadlock_threshold = deadlock_threshold
        self._agent_stall_counter: dict[int, int] = {}
        self._prev_positions: dict[int, tuple] = {}
        self._timer_stack: dict[str, float] = {}

    def start_timer(self, name: str):
        self._timer_stack[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        elapsed = (time.perf_counter() - self._timer_stack.pop(name)) * 1000
        return elapsed

    def record(self, world: WorldState, vertex_conflicts: int,
               assign_ms: float, plan_ms: float):
        deadlock_count = 0
        for agent in world.agents:
            prev = self._prev_positions.get(agent.agent_id)
            if (
                prev == agent.position
                and agent.status != AgentStatus.IDLE
                and not agent.is_waiting
            ):
                self._agent_stall_counter[agent.agent_id] = \
                    self._agent_stall_counter.get(agent.agent_id, 0) + 1
            else:
                self._agent_stall_counter[agent.agent_id] = 0
            self._prev_positions[agent.agent_id] = agent.position
            if self._agent_stall_counter[agent.agent_id] >= self.deadlock_threshold:
                deadlock_count += 1

        prev_completed = self.history[-1].throughput_cumulative if self.history else 0

        self.history.append(TickMetrics(
            tick=world.tick,
            throughput_cumulative=world.order_state.total_completed,
            throughput_delta=world.order_state.total_completed - prev_completed,
            queue_length=len(world.order_state.get_pending_orders()),
            idle_agents=len(world.get_idle_agents()),
            congestion_count=vertex_conflicts,
            deadlock_agents=deadlock_count,
            planning_ms=plan_ms,
            assignment_ms=assign_ms,
        ))

    def summarize(self) -> dict:
        if not self.history:
            return {}
        total = len(self.history)
        return {
            "total_ticks": total,
            "final_throughput": self.history[-1].throughput_cumulative,
            "avg_throughput_per_100tick": self.history[-1].throughput_cumulative / total * 100,
            "total_deadlock_events": sum(h.deadlock_agents > 0 for h in self.history),
            "total_congestion_events": sum(h.congestion_count for h in self.history),
            "avg_plan_ms": sum(h.planning_ms for h in self.history) / total,
            "avg_assign_ms": sum(h.assignment_ms for h in self.history) / total,
        }
