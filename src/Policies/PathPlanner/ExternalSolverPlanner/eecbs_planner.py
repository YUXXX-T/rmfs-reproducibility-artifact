"""
EECBS Planner
=============
Wrapper for the EECBS (Explicit Estimation CBS) C++ solver.

Reference: Li et al. (2021). "EECBS: A Bounded-Suboptimal Search for
Multi-Agent Path Finding." AAAI-21.
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
_DEFAULT_BINARY = os.path.join(_SOLVER_DIR, "EECBS", "build", "eecbs")

_COORD_RE = re.compile(r"\((\d+),(\d+)\)")


class EECBSPlanner(BaseExternalSolverPlanner):

    def __init__(
        self,
        binary_path: str = None,
        timeout: int = 30,
        suboptimality: float = 1.5,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(binary_path or _DEFAULT_BINARY, timeout, **kwargs)
        self.suboptimality = suboptimality
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
            "-a", scen_file,
            "-k", str(n_agents),
            "--outputPaths", output_file,
            "--suboptimality", str(self.suboptimality),
            "-t", str(self.timeout),
            "-s", str(self._effective_seed),
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
                lines = f.readlines()
        except OSError:
            return {}

        paths: Dict[int, List[Tuple[int, int]]] = {}
        for line in lines:
            line = line.strip()
            if not line.startswith("Agent"):
                continue
            colon = line.index(":")
            idx = int(line[6:colon].strip())
            if idx < 0 or idx >= len(agents_with_goals):
                continue
            agent_id = agents_with_goals[idx][0].agent_id
            coords = _COORD_RE.findall(line[colon + 1:])
            path = [(int(r), int(c)) for r, c in coords]
            if len(path) > 1:
                paths[agent_id] = path[1:]
        return paths
