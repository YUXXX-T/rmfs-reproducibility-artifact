"""Priority Inheritance with Backtracking (PIBT) path planner.

PIBT makes one joint, collision-free next-step decision per simulation tick.
The RMFS engine therefore uses :meth:`plan_batch` after station admission and
task activation have identified the exact robots that may move.  Robots not
present in that batch are pinned for the tick, including station service-zone
robots and robots waiting at a destination.

The sequential :meth:`plan` interface remains available for the standalone
MAPF runner and isolated action-path previews.  RMFS execution must use the
batch interface because DELIVER motion goals are queue-entry cells rather than
the service-cell destination stored on the task object.

Reference
---------
Okumura, K., Machida, M., Defago, X., & Tamura, Y. (2019).
Priority Inheritance with Backtracking for Iterative Multi-agent Path Finding.
IJCAI-19.
"""

from __future__ import annotations

from collections import deque
import random
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, TYPE_CHECKING

from Policies.PathPlanner.base_path_planner import BasePathPlanner

if TYPE_CHECKING:
    from WorldState.agent_state import AgentState
    from WorldState.map_state import MapState
    from WorldState.world import WorldState


Position = Tuple[int, int]


def _bfs_distances(
    goal: Position,
    map_state: "MapState",
    blocked: Optional[Set[Position]] = None,
) -> Dict[Position, int]:
    """Return reverse-BFS distances to ``goal`` under static blockers."""
    dist: Dict[Position, int] = {goal: 0}
    queue: deque[Position] = deque([goal])
    while queue:
        current = queue.popleft()
        next_distance = dist[current] + 1
        for neighbor in map_state.get_neighbors(*current):
            if neighbor in dist:
                continue
            if blocked is not None and neighbor in blocked:
                continue
            dist[neighbor] = next_distance
            queue.append(neighbor)
    return dist


class PIBTPlanner(BasePathPlanner):
    """One-step PIBT with priority inheritance, backtracking and ageing."""

    supports_batch_planning = True
    is_single_step_planner = True

    def __init__(self, seed: int = 0, strict_validation: bool = True):
        self._last_tick = -1
        self._goals: Dict[int, Position] = {}
        self._priorities: Dict[int, int] = {}
        self._arrived: Dict[int, Position] = {}
        self._base_seed = int(seed)
        self._rng = random.Random(seed)
        self.strict_validation = bool(strict_validation)

        self._dist_cache: Dict[Position, Dict[Position, int]] = {}
        self._dist_cache_map_id: Optional[int] = None
        self._carry_dist_cache: Dict[Position, Dict[Position, int]] = {}

        self._decided: Dict[int, Position] = {}
        self._next_occupied: Dict[Position, int] = {}
        self._undecided: Set[int] = set()
        self._agents: Dict[int, "AgentState"] = {}
        self._cur_pos_index: Dict[Position, List[int]] = {}
        self._static_blocked: Set[Position] = set()
        self._routing_blocked: Set[Position] = set()
        self._planning_ids: Optional[FrozenSet[int]] = None
        self._last_batch_audit: Dict[str, object] = {}
        self._runtime: Dict[str, int] = {
            "pibt_batch_calls": 0,
            "pibt_single_calls": 0,
            "pibt_planned_agents": 0,
            "pibt_move_decisions": 0,
            "pibt_wait_decisions": 0,
            "pibt_priority_inheritance_calls": 0,
            "pibt_backtracks": 0,
            "pibt_validation_failures": 0,
            "pibt_max_batch_size": 0,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_goals(self, goals: Dict[int, Position]) -> None:
        """Pre-populate goals for the standalone lifelong-MAPF runner."""
        self._goals.update({int(aid): tuple(goal) for aid, goal in goals.items()})

    def plan_batch(
        self,
        agents_with_goals: List[Tuple["AgentState", Position]],
        world_state: "WorldState",
        extra_blocked: Optional[Set[Position]] = None,
    ) -> Dict[int, List[Position]]:
        """Plan a joint one-step action for exactly the supplied robots.

        Every robot omitted from ``agents_with_goals`` is pinned at its current
        position.  A successful wait is represented by ``[current_position]``;
        an empty path is never used as a PIBT planning-failure sentinel.
        """
        if not agents_with_goals:
            return {}

        planning_agents = {
            int(agent.agent_id): agent for agent, _ in agents_with_goals
        }
        planning_ids = frozenset(planning_agents)
        if len(planning_ids) != len(agents_with_goals):
            raise ValueError("PIBT plan_batch received duplicate agent ids")

        for agent, goal in agents_with_goals:
            self._goals[int(agent.agent_id)] = tuple(goal)

        self._last_tick = int(world_state.tick)
        self._runtime["pibt_batch_calls"] += 1
        self._runtime["pibt_planned_agents"] += len(planning_ids)
        self._runtime["pibt_max_batch_size"] = max(
            self._runtime["pibt_max_batch_size"], len(planning_ids)
        )
        self._run_bulk_pibt(
            world_state,
            planning_ids=planning_ids,
            planning_agents=planning_agents,
            extra_blocked=extra_blocked,
        )
        self._audit_decisions(world_state, planning_ids)

        return {
            agent.agent_id: [
                self._decided.get(agent.agent_id, agent.position)
            ]
            for agent, _ in agents_with_goals
        }

    def plan(
        self,
        agent: "AgentState",
        goal: Position,
        world_state: "WorldState",
        extra_blocked: Optional[Set[Position]] = None,
    ) -> List[Position]:
        """Sequential compatibility interface for MAPF and isolated previews."""
        tick = int(world_state.tick)
        self._goals[int(agent.agent_id)] = tuple(goal)

        world_agent = next(
            (a for a in world_state.agents if a.agent_id == agent.agent_id),
            None,
        )
        if world_agent is not agent:
            self._runtime["pibt_single_calls"] += 1
            return self.plan_batch(
                [(agent, tuple(goal))],
                world_state,
                extra_blocked=extra_blocked,
            )[agent.agent_id]

        if tick != self._last_tick:
            self._last_tick = tick
            self._runtime["pibt_single_calls"] += 1
            self._run_bulk_pibt(
                world_state,
                extra_blocked=extra_blocked,
            )
            self._audit_decisions(
                world_state,
                frozenset(self._decided),
            )

        next_position = self._decided.get(agent.agent_id, agent.position)
        return [next_position]

    def runtime_metrics(self) -> Dict[str, object]:
        """Return read-only planner diagnostics for experiment auditing."""
        return {
            "path_planner_name": self.__class__.__name__,
            "path_planner_single_step": True,
            "path_planner_batch_interface": True,
            "pibt_strict_validation": self.strict_validation,
            **dict(self._runtime),
            "pibt_last_batch_audit": dict(self._last_batch_audit),
        }

    # ------------------------------------------------------------------
    # Distance heuristic
    # ------------------------------------------------------------------

    def _dist(
        self,
        position: Position,
        goal: Position,
        map_state: "MapState",
        carrying: bool = False,
    ) -> int:
        blocked = set(self._routing_blocked)
        if carrying:
            blocked.update(self._static_blocked)
        blocked.discard(goal)

        if carrying:
            if goal not in self._carry_dist_cache:
                self._carry_dist_cache[goal] = _bfs_distances(
                    goal, map_state, blocked
                )
            return self._carry_dist_cache[goal].get(position, 0x7FFF_FFFF)

        map_id = id(map_state)
        if map_id != self._dist_cache_map_id:
            self._dist_cache = {}
            self._dist_cache_map_id = map_id
        if goal not in self._dist_cache:
            self._dist_cache[goal] = _bfs_distances(goal, map_state, blocked)
        return self._dist_cache[goal].get(position, 0x7FFF_FFFF)

    # ------------------------------------------------------------------
    # Bulk PIBT pass
    # ------------------------------------------------------------------

    def _run_bulk_pibt(
        self,
        world_state: "WorldState",
        planning_ids: Optional[FrozenSet[int]] = None,
        planning_agents: Optional[Dict[int, "AgentState"]] = None,
        extra_blocked: Optional[Set[Position]] = None,
    ) -> None:
        self._rng.seed(self._base_seed * 1_000_003 + int(world_state.tick))
        self._init_tick(
            world_state,
            planning_ids=planning_ids,
            planning_agents=planning_agents,
            extra_blocked=extra_blocked,
        )

        active = sorted(
            self._undecided,
            key=lambda aid: (self._priorities.get(aid, 0), aid),
            reverse=True,
        )
        for aid in active:
            if aid in self._undecided:
                self._pibt(aid, None, world_state)

        # Recursive failure intentionally leaves a robot undecided so its
        # parent can roll back.  Any top-level remainder now commits a wait.
        for aid in list(self._undecided):
            position = self._agents[aid].position
            if position not in self._next_occupied:
                self._decide(aid, position)

        self._update_priorities(world_state, planning_ids=planning_ids)

    def _update_priorities(
        self,
        world_state: "WorldState",
        planning_ids: Optional[FrozenSet[int]] = None,
    ) -> None:
        ids = planning_ids if planning_ids is not None else frozenset(self._decided)
        for aid in ids:
            agent = self._agents.get(aid)
            if agent is None:
                continue
            goal = self._get_goal(aid, world_state)
            next_position = self._decided.get(aid, agent.position)
            if next_position == goal:
                self._arrived[aid] = goal
                self._priorities[aid] = -1
                continue
            if self._arrived.get(aid) != goal:
                self._arrived.pop(aid, None)
            # Canonical PIBT ageing: unfinished agents gain priority every
            # tick; priority is reset only on reaching the current goal.
            self._priorities[aid] = max(
                0, self._priorities.get(aid, 0)
            ) + 1

    # ------------------------------------------------------------------
    # Per-tick state
    # ------------------------------------------------------------------

    def _init_tick(
        self,
        world_state: "WorldState",
        planning_ids: Optional[FrozenSet[int]] = None,
        planning_agents: Optional[Dict[int, "AgentState"]] = None,
        extra_blocked: Optional[Set[Position]] = None,
    ) -> None:
        self._decided = {}
        self._next_occupied = {}
        self._undecided = set()
        self._agents = {
            int(agent.agent_id): agent for agent in world_state.agents
        }
        if planning_agents:
            self._agents.update(planning_agents)
        self._cur_pos_index = {}
        self._dist_cache = {}
        self._carry_dist_cache = {}
        self._routing_blocked = set(extra_blocked or ())
        self._planning_ids = planning_ids

        ordered_ids = [int(agent.agent_id) for agent in world_state.agents]
        ordered_ids.extend(aid for aid in self._agents if aid not in ordered_ids)
        pinned_statuses = {"QUEUING", "DELIVERING", "EXITING"}

        for aid in ordered_ids:
            agent = self._agents[aid]
            if planning_ids is not None and aid in planning_ids:
                self._undecided.add(aid)
                self._cur_pos_index.setdefault(agent.position, []).append(aid)
            elif planning_ids is not None:
                self._decide(aid, agent.position)
            elif (
                agent.is_idle
                or agent.is_waiting
                or getattr(agent.status, "name", "") in pinned_statuses
            ):
                self._decide(aid, agent.position)
            else:
                self._undecided.add(aid)
                self._cur_pos_index.setdefault(agent.position, []).append(aid)

        self._static_blocked = {
            pod.current_position
            for pod in world_state.pod_state.pods.values()
            if not pod.is_carried
        }

    # ------------------------------------------------------------------
    # Decision bookkeeping
    # ------------------------------------------------------------------

    def _decide(self, agent_id: int, position: Position) -> None:
        self._decided[agent_id] = position
        self._next_occupied[position] = agent_id
        self._undecided.discard(agent_id)
        agent = self._agents.get(agent_id)
        if agent is None or agent.position not in self._cur_pos_index:
            return
        ids = self._cur_pos_index[agent.position]
        try:
            ids.remove(agent_id)
        except ValueError:
            pass
        if not ids:
            del self._cur_pos_index[agent.position]

    def _undecide(self, agent_id: int) -> None:
        if agent_id not in self._decided:
            return
        position = self._decided.pop(agent_id)
        if self._next_occupied.get(position) == agent_id:
            del self._next_occupied[position]
        self._undecided.add(agent_id)
        agent = self._agents[agent_id]
        ids = self._cur_pos_index.setdefault(agent.position, [])
        if agent_id not in ids:
            ids.append(agent_id)

    def _rollback_to(self, decided_before: Set[int]) -> None:
        for aid in reversed(list(self._decided)):
            if aid not in decided_before:
                self._undecide(aid)

    # ------------------------------------------------------------------
    # Goal lookup and PIBT recursion
    # ------------------------------------------------------------------

    def _get_goal(self, agent_id: int, world_state: "WorldState") -> Position:
        # Explicit batch goals are authoritative.  In RMFS, DELIVER tasks
        # store the service cell while the true path-planning goal is entry.
        if agent_id in self._goals:
            return self._goals[agent_id]
        task = world_state.task_state.get_active_task_for_agent(agent_id)
        if task is not None:
            return task.destination
        task = world_state.task_state.get_next_task_for_agent(agent_id)
        if task is not None:
            return task.destination
        return self._agents[agent_id].position

    def _pibt(
        self,
        agent_id: int,
        parent_id: Optional[int],
        world_state: "WorldState",
    ) -> bool:
        agent = self._agents[agent_id]
        goal = self._get_goal(agent_id, world_state)
        map_state = world_state.map_state

        candidates = list(map_state.get_neighbors(*agent.position))
        candidates.append(agent.position)
        candidates = [
            candidate
            for candidate in candidates
            if (
                candidate not in self._routing_blocked
                or candidate == agent.position
                or candidate == goal
            )
        ]

        if agent.carried_pod_id is not None:
            candidates = [
                candidate
                for candidate in candidates
                if candidate not in self._static_blocked or candidate == goal
            ]

        if parent_id is not None:
            parent_position = self._agents[parent_id].position
            candidates = [
                candidate
                for candidate in candidates
                if candidate != parent_position
            ]

        carrying = agent.carried_pod_id is not None
        self._rng.shuffle(candidates)
        candidates.sort(
            key=lambda candidate: self._dist(
                candidate, goal, map_state, carrying
            )
        )

        for candidate in candidates:
            if candidate in self._next_occupied:
                continue

            occupier_id = self._undecided_at(candidate)
            if occupier_id == agent_id:
                occupier_id = None

            decided_before = set(self._decided)
            self._decide(agent_id, candidate)

            if occupier_id is not None:
                self._runtime["pibt_priority_inheritance_calls"] += 1
                if not self._pibt(occupier_id, agent_id, world_state):
                    self._runtime["pibt_backtracks"] += 1
                    self._rollback_to(decided_before)
                    continue
            return True

        # In a recursive call, the parent may still tentatively claim this
        # robot's current cell.  Leave it undecided until the parent rolls
        # back; only claim a wait when the cell is currently free.
        if agent.position not in self._next_occupied:
            self._decide(agent_id, agent.position)
        return False

    def _undecided_at(self, position: Position) -> Optional[int]:
        ids = self._cur_pos_index.get(position)
        return ids[0] if ids else None

    # ------------------------------------------------------------------
    # Runtime invariant audit
    # ------------------------------------------------------------------

    def _audit_decisions(
        self,
        world_state: "WorldState",
        planning_ids: FrozenSet[int],
    ) -> None:
        missing = sorted(aid for aid in planning_ids if aid not in self._decided)
        invalid_steps = []
        blocked_steps = []
        pod_steps = []
        moves = 0
        waits = 0
        map_state = world_state.map_state

        for aid in sorted(planning_ids):
            agent = self._agents[aid]
            next_position = self._decided.get(aid, agent.position)
            legal = set(map_state.get_neighbors(*agent.position))
            legal.add(agent.position)
            if next_position not in legal:
                invalid_steps.append((aid, agent.position, next_position))
            goal = self._get_goal(aid, world_state)
            if (
                next_position in self._routing_blocked
                and next_position not in {agent.position, goal}
            ):
                blocked_steps.append((aid, next_position))
            if (
                agent.carried_pod_id is not None
                and next_position in self._static_blocked
                and next_position != goal
            ):
                pod_steps.append((aid, next_position))
            if next_position == agent.position:
                waits += 1
            else:
                moves += 1

        targets: Dict[Position, List[int]] = {}
        for aid, position in self._decided.items():
            targets.setdefault(position, []).append(aid)
        vertex_conflicts = [
            (position, sorted(ids))
            for position, ids in targets.items()
            if len(ids) > 1
        ]

        swap_conflicts = []
        planned = sorted(planning_ids)
        for index, aid in enumerate(planned):
            agent_a = self._agents[aid]
            next_a = self._decided.get(aid, agent_a.position)
            for bid in planned[index + 1:]:
                agent_b = self._agents[bid]
                next_b = self._decided.get(bid, agent_b.position)
                if (
                    next_a == agent_b.position
                    and next_b == agent_a.position
                    and agent_a.position != agent_b.position
                ):
                    swap_conflicts.append((aid, bid))

        passed = not any((
            missing,
            invalid_steps,
            blocked_steps,
            pod_steps,
            vertex_conflicts,
            swap_conflicts,
        ))
        self._last_batch_audit = {
            "passed": passed,
            "tick": int(world_state.tick),
            "planned_agents": len(planning_ids),
            "move_decisions": moves,
            "wait_decisions": waits,
            "missing_decisions": missing,
            "invalid_steps": invalid_steps,
            "blocked_steps": blocked_steps,
            "pod_obstacle_steps": pod_steps,
            "vertex_conflicts": vertex_conflicts,
            "swap_conflicts": swap_conflicts,
        }
        self._runtime["pibt_move_decisions"] += moves
        self._runtime["pibt_wait_decisions"] += waits
        if not passed:
            self._runtime["pibt_validation_failures"] += 1
            if self.strict_validation:
                raise RuntimeError(
                    "PIBT one-step invariant audit failed: "
                    f"{self._last_batch_audit}"
                )
