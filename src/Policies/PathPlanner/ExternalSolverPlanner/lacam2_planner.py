"""
LaCAM2 Planner
==============
Wrapper for the LaCAM2 C++ solver.

Reference: Okumura, K. (2023). "Improving LaCAM for Scalable Eventually
Optimal Multi-Agent Pathfinding." IJCAI-23.

LaCAM2 uses a different output format from EECBS/LNS2:
- Coordinates are (col, row) — swapped relative to RMFS (row, col).
- Solution is timestep-indexed: ``t:(c0,r0),(c1,r1),...``
"""

import os
import re
from typing import Dict, List, Tuple, TYPE_CHECKING

from Policies.PathPlanner.base_external_path_planner import (
    BaseExternalSolverPlanner,
)

if TYPE_CHECKING:
    from WorldState.agent_state import AgentState

_SOLVER_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_BINARY = os.path.join(_SOLVER_DIR, "lacam2", "build", "main")

_COORD_RE = re.compile(r"\((\d+),(\d+)\)")


class LaCAM2Planner(BaseExternalSolverPlanner):

    def __init__(
        self,
        binary_path: str = None,
        timeout: int = 30,
        objective: int = 0,
        restart_rate: float = 0.001,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(binary_path or _DEFAULT_BINARY, timeout, **kwargs)
        self.objective = objective
        self.restart_rate = restart_rate
        self.seed = seed

    def _build_command(
        self,
        map_file: str,
        scen_file: str,
        n_agents: int,
        output_file: str,
    ) -> List[str]:
        return [
            self.binary_path,
            "-m", map_file,
            "-i", scen_file,
            "-N", str(n_agents),
            "-o", output_file,
            "-t", str(self.timeout),
            "-O", str(self.objective),
            "-r", str(self.restart_rate),
            "-s", str(self._effective_seed),
            "-v", "0",
        ]

    def _parse_output(
        self,
        output_file: str,
        agents_with_goals: List[Tuple["AgentState", Tuple[int, int]]],
    ) -> Dict[int, List[Tuple[int, int]]]:
        if not os.path.exists(output_file):
            return {}
        try:
            with open(output_file) as f:
                content = f.read()
        except OSError:
            return {}

        if "solved=0" in content or "solution=" not in content:
            return {}

        n_agents = len(agents_with_goals)
        agent_paths: List[List[Tuple[int, int]]] = [[] for _ in range(n_agents)]

        in_solution = False
        for line in content.splitlines():
            if line.startswith("solution="):
                in_solution = True
                continue
            if not in_solution:
                continue
            colon = line.find(":")
            if colon < 0:
                continue
            coords = _COORD_RE.findall(line[colon + 1:])
            for i, (col_s, row_s) in enumerate(coords):
                if i < n_agents:
                    agent_paths[i].append((int(row_s), int(col_s)))

        paths: Dict[int, List[Tuple[int, int]]] = {}
        for i, (agent, _goal) in enumerate(agents_with_goals):
            path = agent_paths[i]
            if len(path) > 1:
                # Trim trailing repeated positions (agent already at goal)
                while len(path) > 2 and path[-1] == path[-2]:
                    path.pop()
                paths[agent.agent_id] = path[1:]
        return paths
