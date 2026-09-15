"""
Base External Solver Planner
=============================
Abstract base class for C++ MAPF solver wrappers.

Subclasses implement ``_build_command`` and ``_parse_output`` for each
specific solver binary (EECBS, LNS2, LaCAM2).

Provides two calling conventions:
- ``plan()``       — per-agent interface compatible with the existing engine.
                     First call each tick triggers a batch solve for all
                     active agents; subsequent calls return cached results.
- ``plan_batch()`` — direct batch interface for Phase 1.3.4 engine integration.
"""

import os
import subprocess
import tempfile
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from Policies.PathPlanner.base_path_planner import BasePathPlanner
from WorldState.map_state import CellType
from Debug.logger import SimLogger

if TYPE_CHECKING:
    from WorldState.world import WorldState
    from WorldState.map_state import MapState
    from WorldState.agent_state import AgentState


class BaseExternalSolverPlanner(BasePathPlanner):

    def __init__(self, binary_path: str, timeout: int = 30, **kwargs):
        self.binary_path = binary_path
        self.timeout = timeout
        self.logger = SimLogger(self.__class__.__name__, level="WARNING")

        self._last_tick: int = -1
        self._goals: Dict[int, Tuple[int, int]] = {}
        self._cached_paths: Dict[int, List[Tuple[int, int]]] = {}
        self._effective_seed: int = 0  # overwritten per-call by _run_solver

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_goals(self, goals: Dict[int, Tuple[int, int]]) -> None:
        """Pre-populate agent goals (used by MAPFRunner before tick 0)."""
        self._goals.update(goals)

    def plan(
        self,
        agent: "AgentState",
        goal: Tuple[int, int],
        world_state: "WorldState",
    ) -> List[Tuple[int, int]]:
        tick = world_state.tick
        self._goals[agent.agent_id] = goal

        if tick != self._last_tick:
            self._last_tick = tick
            self._cached_paths = self._batch_solve(world_state)

        return self._cached_paths.get(agent.agent_id, [])

    def plan_batch(
        self,
        agents_with_goals: List[Tuple["AgentState", Tuple[int, int]]],
        world_state: "WorldState",
    ) -> Dict[int, List[Tuple[int, int]]]:
        if not agents_with_goals:
            return {}
        return self._run_solver(agents_with_goals, world_state)

    # ------------------------------------------------------------------
    # Batch solve (called from plan())
    # ------------------------------------------------------------------

    def _batch_solve(
        self, world_state: "WorldState"
    ) -> Dict[int, List[Tuple[int, int]]]:
        agents_with_goals: List[Tuple["AgentState", Tuple[int, int]]] = []
        for a in world_state.agents:
            if a.is_idle or a.is_waiting or a.has_path:
                continue
            goal = self._get_goal(a.agent_id, world_state)
            if goal is None:
                continue
            agents_with_goals.append((a, goal))

        if not agents_with_goals:
            return {}
        return self._run_solver(agents_with_goals, world_state)

    def _get_goal(
        self, agent_id: int, world_state: "WorldState"
    ) -> Optional[Tuple[int, int]]:
        task = world_state.task_state.get_active_task_for_agent(agent_id)
        if task is not None:
            return task.destination
        task = world_state.task_state.get_next_task_for_agent(agent_id)
        if task is not None:
            return task.destination
        if agent_id in self._goals:
            return self._goals[agent_id]
        return None

    # ------------------------------------------------------------------
    # Solver execution
    # ------------------------------------------------------------------

    def _run_solver(
        self,
        agents_with_goals: List[Tuple["AgentState", Tuple[int, int]]],
        world_state: "WorldState",
    ) -> Dict[int, List[Tuple[int, int]]]:
        if not os.path.isfile(self.binary_path):
            self.logger.warning(
                f"Binary not found: {self.binary_path}"
            )
            return {}

        map_file = scen_file = output_file = None
        try:
            map_file, scen_file = self._write_instance(
                world_state.map_state, agents_with_goals
            )
            output_fd, output_file = tempfile.mkstemp(suffix=".txt")
            os.close(output_fd)

            # Per-tick effective seed: mix base seed with current tick so each
            # solve is reproducible across runs but differs tick-to-tick.
            base = getattr(self, "seed", 0)
            self._effective_seed = (base * 1_000_003 + world_state.tick) & 0x7FFFFFFF

            cmd = self._build_command(
                map_file, scen_file, len(agents_with_goals), output_file
            )
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout
            )

            if result.returncode != 0:
                self.logger.warning(
                    f"Solver exited with code {result.returncode}: "
                    f"{result.stderr[:200]}"
                )

            paths = self._parse_output(output_file, agents_with_goals)
        except subprocess.TimeoutExpired:
            self.logger.warning(
                f"Solver timed out after {self.timeout}s"
            )
            paths = {}
        except Exception as e:
            self.logger.warning(f"Solver error: {e}")
            paths = {}
        finally:
            for f in (map_file, scen_file, output_file):
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except OSError:
                        pass
        return paths

    # ------------------------------------------------------------------
    # Instance file writing
    # ------------------------------------------------------------------

    def _write_instance(
        self,
        map_state: "MapState",
        agents_with_goals: List[Tuple["AgentState", Tuple[int, int]]],
    ) -> Tuple[str, str]:
        map_fd, map_path = tempfile.mkstemp(suffix=".map")
        with os.fdopen(map_fd, "w") as f:
            f.write(f"type octile\nheight {map_state.rows}\n"
                    f"width {map_state.cols}\nmap\n")
            for r in range(map_state.rows):
                row_str = ""
                for c in range(map_state.cols):
                    if map_state.grid[r][c] == CellType.OBSTACLE:
                        row_str += "@"
                    else:
                        row_str += "."
                f.write(row_str + "\n")

        scen_fd, scen_path = tempfile.mkstemp(suffix=".scen")
        with os.fdopen(scen_fd, "w") as f:
            f.write("version 1\n")
            for agent, goal in agents_with_goals:
                sr, sc = agent.position
                gr, gc = goal
                f.write(
                    f"0\t{map_path}\t{map_state.cols}\t{map_state.rows}\t"
                    f"{sc}\t{sr}\t{gc}\t{gr}\t0\n"
                )
        return map_path, scen_path

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_command(
        self,
        map_file: str,
        scen_file: str,
        n_agents: int,
        output_file: str,
    ) -> List[str]:
        ...

    @abstractmethod
    def _parse_output(
        self,
        output_file: str,
        agents_with_goals: List[Tuple["AgentState", Tuple[int, int]]],
    ) -> Dict[int, List[Tuple[int, int]]]:
        ...
