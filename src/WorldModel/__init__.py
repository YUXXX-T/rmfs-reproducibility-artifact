"""Public API for the WorldModel package."""

from importlib import import_module
from types import ModuleType
import sys


def _alias(alias: str, target: str):
    """Expose a refactored module under its old top-level module name."""
    module = import_module(target, __name__)
    sys.modules.setdefault(f"{__name__}.{alias}", module)
    globals()[alias] = module
    return module


class _LazyAlias(ModuleType):
    """Compatibility proxy for CLI-oriented modules.

    It avoids importing `WorldModel.evaluation.*` modules while Python is
    preparing to execute them with `python -m`.
    """

    def __init__(self, public_name: str, target: str):
        super().__init__(public_name)
        self._target = target

    def _load(self):
        module = import_module(self._target, __name__)
        sys.modules[self.__name__] = module
        globals()[self.__name__.rsplit(".", 1)[-1]] = module
        return module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


def _lazy_alias(alias: str, target: str):
    public_name = f"{__name__}.{alias}"
    module = sys.modules.get(public_name)
    if module is None:
        module = _LazyAlias(public_name, target)
        sys.modules[public_name] = module
    globals()[alias] = module
    return module


# Backward-compatible module paths used by older scripts.
_alias("model", ".core.model")
_alias("costs", ".core.costs")
_alias("graph_builder", ".graph.graph_builder")
_alias("candidate_generator", ".data.candidate_generator")
_alias("counterfactual_rollout", ".data.counterfactual_rollout")
_alias("dataset", ".data.dataset")
_alias("data_collector", ".data.data_collector")
_alias("train", ".training.train")
_lazy_alias("evaluate", ".evaluation.evaluate")
_lazy_alias("check_data_quality", ".evaluation.check_data_quality")
_lazy_alias("evaluate_online_v6", ".evaluation.evaluate_online_v6")
_lazy_alias("validate_dynamics", ".evaluation.validate_dynamics")

from .core import (
    EndpointDemandPredictor,
    FixedLyapunovFunctional,
    LyapunovComponentHead,
    LyapunovLossWeights,
    LyapunovPrediction,
    CandidateEtaPreview,
    ChainWork,
    LyapunovL0Config,
    LyapunovSnapshot,
    PHYSICAL_SUMMARY_NAMES,
    ProductiveProgress,
    RMFSWorldModel,
    StationDecoder,
    TDRiskVHead,
    TDValueHead,
    compute_lyapunov_snapshot,
    compute_productive_progress,
    compute_realized_cost,
    compute_lyapunov_training_loss,
    target_from_analytic_snapshot,
    preview_assignment_context,
    preview_candidate_eta,
)
from .data import (
    WorldModelDataCollector,
    WorldModelDataset,
    build_candidate_assignment,
    compute_heuristic_cost,
    evaluate_candidate_rollout,
    force_apply_candidate,
    generate_candidates,
    step_world,
)
from .graph import (
    FeatureHistory,
    build_action_edge_field,
    build_action_field,
    build_static_graph,
    compute_preview_legs,
    extract_demand_context,
    extract_edge_features,
    extract_node_features,
    extract_node_labels,
    extract_station_labels,
    extract_system_labels,
)
from .training import (
    compute_loss,
    compute_ranking_loss,
    evaluate_ranking,
    evaluate_top1_regret,
    train as train_model,
)


__all__ = [
    "EndpointDemandPredictor",
    "FixedLyapunovFunctional",
    "LyapunovComponentHead",
    "LyapunovLossWeights",
    "LyapunovPrediction",
    "RMFSWorldModel",
    "StationDecoder",
    "TDRiskVHead",
    "TDValueHead",
    "compute_realized_cost",
    "compute_lyapunov_training_loss",
    "target_from_analytic_snapshot",
    "CandidateEtaPreview",
    "ChainWork",
    "LyapunovL0Config",
    "LyapunovSnapshot",
    "ProductiveProgress",
    "PHYSICAL_SUMMARY_NAMES",
    "compute_lyapunov_snapshot",
    "compute_productive_progress",
    "preview_assignment_context",
    "preview_candidate_eta",
    "WorldModelDataCollector",
    "WorldModelDataset",
    "generate_candidates",
    "compute_heuristic_cost",
    "build_candidate_assignment",
    "force_apply_candidate",
    "step_world",
    "evaluate_candidate_rollout",
    "build_static_graph",
    "extract_node_features",
    "extract_edge_features",
    "extract_demand_context",
    "build_action_field",
    "build_action_edge_field",
    "compute_preview_legs",
    "extract_node_labels",
    "extract_system_labels",
    "extract_station_labels",
    "FeatureHistory",
    "train_model",
    "compute_loss",
    "compute_ranking_loss",
    "evaluate_ranking",
    "evaluate_top1_regret",
]
