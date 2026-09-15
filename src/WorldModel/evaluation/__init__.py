"""Evaluation entry points and metric helpers."""

from importlib import import_module


_EXPORTS = {
    "spearman": (".evaluate", "spearman"),
    "roc_auc": (".evaluate", "roc_auc"),
    "evaluate_predictions": (".evaluate", "evaluate_predictions"),
    "evaluate_ranking": (".evaluate", "evaluate_ranking"),
    "compare_throughput": (".evaluate", "compare_throughput"),
    "run_all_seeds": (".evaluate_online_v6", "run_all_seeds"),
    "aggregate_seeds": (".evaluate_online_v6", "aggregate_seeds"),
    "compute_comparison": (".evaluate_online_v6", "compute_comparison"),
    "run_validation": (".validate_dynamics", "run_validation"),
    "build_prediction_targets_schema": (
        ".validate_dynamics",
        "build_prediction_targets_schema",
    ),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
