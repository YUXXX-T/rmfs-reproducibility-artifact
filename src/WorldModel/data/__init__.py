"""Data generation, counterfactual rollout, and dataset utilities."""

from importlib import import_module
import sys


def _alias(alias: str, target: str):
    module = import_module(target, __name__)
    sys.modules.setdefault(f"WorldModel.{alias}", module)
    return module


# Keep legacy imports inside moved modules working.
_alias("candidate_generator", ".candidate_generator")
_alias("counterfactual_rollout", ".counterfactual_rollout")
_alias("dataset", ".dataset")
_alias("data_collector", ".data_collector")

from .candidate_generator import (
    build_candidate_assignment,
    compute_heuristic_cost,
    generate_candidates,
)
from .counterfactual_rollout import (
    evaluate_candidate_rollout,
    force_apply_candidate,
    step_world,
)
from .data_collector import WorldModelDataCollector
from .dataset import WorldModelDataset


__all__ = [
    "WorldModelDataCollector",
    "WorldModelDataset",
    "generate_candidates",
    "compute_heuristic_cost",
    "build_candidate_assignment",
    "force_apply_candidate",
    "step_world",
    "evaluate_candidate_rollout",
]
