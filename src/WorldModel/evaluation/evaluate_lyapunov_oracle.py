"""Evaluate analytic Lyapunov action scores before training an online head.

The counterfactual dataset stores several robot candidates for one fixed
order/pod/station context.  This tool intentionally treats ``heuristic_cost``
as a *proxy* base score unless another scalar field is requested.  It reports
how analytic ``Delta L`` trades against that proxy, realised short-horizon
cost, and productive progress.  No result from this script is an online
policy claim.

The current strict schema is L1/v3: unfinished work, station overload, and
cumulative capacity-normalised incoming work form the potential.  Traffic and
stall remain recorded diagnostics for guards, not default potential terms.

The historical ``short_horizon_certification`` and raw-gap slices are retained
for reproducibility of existing reports.  A fixed numeric ``Delta L`` gap is
not scale-invariant and therefore must not be used as the final validity test
for a Lyapunov functional.  New work should use
``evaluate_lyapunov_validity.py`` for the five-layer continuous protocol.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


SCHEMA_VERSION = "lyapunov_oracle_eval_v2"
STRICT_AUDIT_SCHEMA_VERSION = "lyapunov_isolated_audit_v2"
SHORT_HORIZON_GATE_SCHEMA_VERSION = "lyapunov_short_horizon_gate_v2"
CURRENT_L0_COLLECTION_SCHEMA = "lyapunov_l1_collection_v3"
CURRENT_L0_SNAPSHOT_SCHEMA = "lyapunov_l1_snapshot_v3"
ISOLATED_CONTINUATION_MODE = "isolated"
ISOLATED_CONTINUATION_POLICY = "isolated_forced_candidate"
L0_COMPONENT_NAMES = ("work", "station", "traffic", "stall", "arrival")
DEFAULT_LAMBDAS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_PROGRESS_EPS = (0.0, 0.05, 0.10)
DEFAULT_GAP_THRESHOLDS = (1e-9, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
DEFAULT_SELECTION_MARGINS = (0.1, 0.5)
DEFAULT_COMPONENT_DOMINANCE_LIMIT = 0.80
DEFAULT_COMPONENT_SCALE_RATIO_LIMIT = 10.0


def _parse_float_list(text: str) -> Tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("expected non-negative finite comma-separated values")
    return values


def _expand_paths(values: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for value in values:
        matches = sorted(glob.glob(value)) if any(ch in value for ch in "*?") else []
        paths.extend(matches or [value])
    result = []
    for path in paths:
        resolved = str(Path(path).expanduser().resolve())
        if resolved not in result:
            result.append(resolved)
    return result


def _load_samples(paths: Sequence[str]) -> List[dict]:
    samples: List[dict] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected a list of counterfactual samples")
        for sample in payload:
            if not isinstance(sample, dict):
                raise ValueError(f"{path}: sample is not a mapping")
            row = dict(sample)
            row["_source_path"] = path
            samples.append(row)
    return samples


def _sample_identity(sample: Mapping) -> dict:
    return {
        "run_id": str(sample.get("run_id", "unknown")),
        "candidate_group_id": str(sample.get("candidate_group_id", "unknown")),
        "candidate_key": str(sample.get("candidate_key", "unknown")),
        "source_path": str(sample.get("_source_path", "")),
    }


def _close(left: float, right: float, tolerance: float) -> bool:
    return abs(float(left) - float(right)) <= float(tolerance) * max(
        1.0, abs(float(left)), abs(float(right))
    )


def audit_isolated_semantics(
    samples: Sequence[Mapping],
    *,
    required_schema: str = CURRENT_L0_COLLECTION_SCHEMA,
    tolerance: float = 1e-9,
    max_examples: int = 5,
) -> dict:
    """Hard-audit the causal semantics required by an isolated L0 oracle."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")

    check_names = (
        "current_l0_schema",
        "current_snapshot_schema",
        "lyapunov_config_present",
        "frozen_lyapunov_config",
        "isolated_continuation_mode",
        "isolated_continuation_policy",
        "zero_generated_orders",
        "zero_followup_assignments",
        "zero_external_arrival_work",
        "valid_l0_record",
        "post_action_snapshot_present",
        "complete_rollout",
        "rollout_horizon_consistent",
        "candidate_not_completed_at_start",
        "same_start_snapshot_within_group",
    )
    failures = {name: 0 for name in check_names}
    examples = {name: [] for name in check_names}

    def record(name: str, passed: bool, sample: Mapping, observed) -> None:
        if passed:
            return
        failures[name] += 1
        if len(examples[name]) < max_examples:
            examples[name].append({
                **_sample_identity(sample),
                "observed": observed,
            })

    for sample in samples:
        schema = sample.get("lyapunov_l0_collection_schema_version")
        record(
            "current_l0_schema",
            str(schema) == str(required_schema),
            sample,
            schema,
        )
        snapshot_schemas = {}
        for endpoint in ("start", "post_action", "end"):
            snapshot = sample.get(f"lyapunov_l0_{endpoint}")
            snapshot_schemas[endpoint] = (
                snapshot.get("schema_version")
                if isinstance(snapshot, Mapping) else None
            )
        record(
            "current_snapshot_schema",
            all(
                str(value) == CURRENT_L0_SNAPSHOT_SCHEMA
                for value in snapshot_schemas.values()
            ),
            sample,
            snapshot_schemas,
        )
        l0_config = sample.get("lyapunov_l0_config")
        record(
            "lyapunov_config_present",
            isinstance(l0_config, Mapping) and bool(l0_config),
            sample,
            type(l0_config).__name__ if l0_config is not None else None,
        )
        mode = sample.get("rollout_continuation_mode")
        record(
            "isolated_continuation_mode",
            str(mode) == ISOLATED_CONTINUATION_MODE,
            sample,
            mode,
        )
        policy = sample.get("continuation_policy")
        record(
            "isolated_continuation_policy",
            str(policy) == ISOLATED_CONTINUATION_POLICY,
            sample,
            policy,
        )
        generated = float(sample.get("rollout_generated_orders", math.inf))
        record(
            "zero_generated_orders",
            math.isfinite(generated) and abs(generated) <= tolerance,
            sample,
            generated,
        )
        assigned = float(sample.get("rollout_assigned_tasks", math.inf))
        record(
            "zero_followup_assignments",
            math.isfinite(assigned) and abs(assigned) <= tolerance,
            sample,
            assigned,
        )
        arrival = _progress(sample, "arrival_total")
        record(
            "zero_external_arrival_work",
            abs(arrival) <= tolerance,
            sample,
            arrival,
        )
        valid = sample.get("lyapunov_l0_valid")
        record("valid_l0_record", valid is True, sample, valid)
        post_action = sample.get("lyapunov_l0_post_action")
        record(
            "post_action_snapshot_present",
            isinstance(post_action, Mapping),
            sample,
            type(post_action).__name__ if post_action is not None else None,
        )
        mask = sample.get("future_mask")
        complete = False
        observed_mask = None
        if mask is not None:
            tensor = torch.as_tensor(mask).flatten()
            observed_mask = {
                "valid": int((tensor > 0.5).sum().item()),
                "total": int(tensor.numel()),
            }
            complete = bool(tensor.numel() > 0 and (tensor > 0.5).all())
        record("complete_rollout", complete, sample, observed_mask)

        start = sample.get("lyapunov_l0_start")
        post = sample.get("lyapunov_l0_post_action")
        end = sample.get("lyapunov_l0_end")
        progress = sample.get("lyapunov_l0_progress")
        horizon_values = {
            "start_tick": (start or {}).get("tick")
            if isinstance(start, Mapping) else None,
            "post_action_tick": (post or {}).get("tick")
            if isinstance(post, Mapping) else None,
            "end_tick": (end or {}).get("tick")
            if isinstance(end, Mapping) else None,
            "progress_horizon": (progress or {}).get("horizon")
            if isinstance(progress, Mapping) else None,
            "mask_steps": observed_mask["valid"] if observed_mask else None,
        }
        try:
            start_tick = int(horizon_values["start_tick"])
            post_tick = int(horizon_values["post_action_tick"])
            end_tick = int(horizon_values["end_tick"])
            progress_horizon = int(horizon_values["progress_horizon"])
            mask_steps = int(horizon_values["mask_steps"])
            horizon_consistent = bool(
                complete
                and post_tick == start_tick
                and end_tick - start_tick == mask_steps
                and progress_horizon == mask_steps
            )
        except (TypeError, ValueError, OverflowError):
            horizon_consistent = False
        record(
            "rollout_horizon_consistent",
            horizon_consistent,
            sample,
            horizon_values,
        )

        fixed_context = sample.get("fixed_context")
        completed_keys = (
            start.get("completed_chain_keys", ())
            if isinstance(start, Mapping) else ()
        )
        try:
            action_key = (
                int((fixed_context or {})["order_id"]),
                int((fixed_context or {})["pod_id"]),
            )
            completed_set = {
                (int(value[0]), int(value[1]))
                for value in completed_keys
            }
            candidate_not_completed = action_key not in completed_set
            completed_observed = {
                "action_key": list(action_key),
                "already_completed": not candidate_not_completed,
            }
        except (KeyError, TypeError, ValueError, IndexError):
            candidate_not_completed = False
            completed_observed = {
                "fixed_context": fixed_context,
                "completed_chain_keys": completed_keys,
            }
        record(
            "candidate_not_completed_at_start",
            candidate_not_completed,
            sample,
            completed_observed,
        )

    config_signatures = {
        _normalise_tree(sample.get("lyapunov_l0_config"))
        for sample in samples
        if isinstance(sample.get("lyapunov_l0_config"), Mapping)
    }
    if samples:
        record(
            "frozen_lyapunov_config",
            len(config_signatures) == 1 and all(
                isinstance(sample.get("lyapunov_l0_config"), Mapping)
                for sample in samples
            ),
            samples[0],
            {"unique_configs": len(config_signatures)},
        )

    grouped = defaultdict(list)
    for sample in samples:
        grouped[(
            str(sample.get("run_id") or sample.get("_source_path") or "unknown"),
            str(sample.get("candidate_group_id", "unknown")),
        )].append(sample)
    for members in grouped.values():
        signatures = {
            _normalise_tree(member.get("lyapunov_l0_start"))
            for member in members
        }
        record(
            "same_start_snapshot_within_group",
            len(signatures) == 1,
            members[0],
            {"members": len(members), "unique_start_snapshots": len(signatures)},
        )

    checks = {
        name: {
            "passed": failures[name] == 0,
            "failures": int(failures[name]),
            "examples": examples[name],
        }
        for name in check_names
    }
    return {
        "schema_version": STRICT_AUDIT_SCHEMA_VERSION,
        "required_l0_collection_schema": str(required_schema),
        "required_l0_snapshot_schema": CURRENT_L0_SNAPSHOT_SCHEMA,
        "required_continuation_mode": ISOLATED_CONTINUATION_MODE,
        "required_continuation_policy": ISOLATED_CONTINUATION_POLICY,
        "samples": int(len(samples)),
        "checks": checks,
        "passed": bool(samples) and all(row["passed"] for row in checks.values()),
    }


def audit_analytic_invariants(
    samples: Sequence[Mapping],
    *,
    tolerance: float = 1e-6,
    max_examples: int = 5,
) -> dict:
    """Audit non-negativity, conservation, and ledger closure in stored L0 rows."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")

    check_names = (
        "snapshot_present",
        "finite_nonnegative_total",
        "finite_nonnegative_components",
        "total_equals_component_sum",
        "finite_nonnegative_station_work",
        "delta_matches_endpoint_totals",
        "immediate_delta_matches_post_action",
        "assignment_preserves_work_mass",
        "finite_progress_ledger",
        "work_balance_closes",
    )
    failures = {name: 0 for name in check_names}
    examples = {name: [] for name in check_names}
    max_errors = {
        "total_component_sum_abs": 0.0,
        "delta_abs": 0.0,
        "immediate_delta_abs": 0.0,
        "assignment_work_mass_abs": 0.0,
        "work_balance_abs": 0.0,
    }

    def record(name: str, passed: bool, sample: Mapping, observed) -> None:
        if passed:
            return
        failures[name] += 1
        if len(examples[name]) < max_examples:
            examples[name].append({
                **_sample_identity(sample),
                "observed": observed,
            })

    def snapshot_values(sample: Mapping, field: str):
        snapshot = sample.get(field)
        if not isinstance(snapshot, Mapping):
            record("snapshot_present", False, sample, field)
            return None
        total = float(snapshot.get("total", math.nan))
        components = snapshot.get("components")
        station_work = snapshot.get("station_work")
        if not isinstance(components, Mapping) or not isinstance(station_work, Mapping):
            record(
                "snapshot_present",
                False,
                sample,
                {
                    "field": field,
                    "components": type(components).__name__,
                    "station_work": type(station_work).__name__,
                },
            )
            return None
        record("snapshot_present", True, sample, field)

        total_ok = math.isfinite(total) and total >= -tolerance
        record(
            "finite_nonnegative_total",
            total_ok,
            sample,
            {"field": field, "total": total},
        )
        component_values = {
            name: float(components.get(name, math.nan))
            for name in L0_COMPONENT_NAMES
        }
        components_ok = all(
            math.isfinite(value) and value >= -tolerance
            for value in component_values.values()
        )
        record(
            "finite_nonnegative_components",
            components_ok,
            sample,
            {"field": field, "components": component_values},
        )
        component_sum = float(sum(component_values.values()))
        total_error = abs(total - component_sum)
        max_errors["total_component_sum_abs"] = max(
            max_errors["total_component_sum_abs"], total_error
        )
        record(
            "total_equals_component_sum",
            total_ok and components_ok and _close(total, component_sum, tolerance),
            sample,
            {
                "field": field,
                "total": total,
                "component_sum": component_sum,
                "abs_error": total_error,
            },
        )
        work_values = [float(value) for value in station_work.values()]
        work_ok = all(
            math.isfinite(value) and value >= -tolerance
            for value in work_values
        )
        record(
            "finite_nonnegative_station_work",
            work_ok,
            sample,
            {"field": field, "station_work": dict(station_work)},
        )
        return {
            "total": total,
            "work_sum": float(sum(work_values)),
        }

    for sample in samples:
        start = snapshot_values(sample, "lyapunov_l0_start")
        post = snapshot_values(sample, "lyapunov_l0_post_action")
        end = snapshot_values(sample, "lyapunov_l0_end")
        if start is None or post is None or end is None:
            continue

        stored_delta = float(sample.get("lyapunov_l0_delta", math.nan))
        expected_delta = end["total"] - start["total"]
        delta_error = abs(stored_delta - expected_delta)
        max_errors["delta_abs"] = max(max_errors["delta_abs"], delta_error)
        record(
            "delta_matches_endpoint_totals",
            math.isfinite(stored_delta)
            and _close(stored_delta, expected_delta, tolerance),
            sample,
            {
                "stored": stored_delta,
                "expected": expected_delta,
                "abs_error": delta_error,
            },
        )

        immediate = float(sample.get("lyapunov_l0_immediate_delta", math.nan))
        expected_immediate = post["total"] - start["total"]
        immediate_error = abs(immediate - expected_immediate)
        max_errors["immediate_delta_abs"] = max(
            max_errors["immediate_delta_abs"], immediate_error
        )
        record(
            "immediate_delta_matches_post_action",
            math.isfinite(immediate)
            and _close(immediate, expected_immediate, tolerance),
            sample,
            {
                "stored": immediate,
                "expected": expected_immediate,
                "abs_error": immediate_error,
            },
        )

        assignment_error = abs(post["work_sum"] - start["work_sum"])
        max_errors["assignment_work_mass_abs"] = max(
            max_errors["assignment_work_mass_abs"], assignment_error
        )
        record(
            "assignment_preserves_work_mass",
            _close(post["work_sum"], start["work_sum"], tolerance),
            sample,
            {
                "start_work": start["work_sum"],
                "post_action_work": post["work_sum"],
                "abs_error": assignment_error,
            },
        )

        progress = sample.get("lyapunov_l0_progress")
        progress_fields = (
            "productive_total",
            "reverse_total",
            "arrival_total",
            "replan_residual_total",
            "route_plan_churn_total",
        )
        progress_values = {
            name: float((progress or {}).get(name, math.nan))
            for name in progress_fields
        }
        progress_ok = all(math.isfinite(value) for value in progress_values.values())
        progress_ok = progress_ok and all(
            progress_values[name] >= -tolerance
            for name in (
                "productive_total", "reverse_total",
                "arrival_total", "route_plan_churn_total",
            )
        )
        record(
            "finite_progress_ledger",
            progress_ok,
            sample,
            progress_values,
        )
        if progress_ok:
            observed_work_delta = end["work_sum"] - start["work_sum"]
            expected_work_delta = (
                progress_values["arrival_total"]
                - progress_values["productive_total"]
                + progress_values["reverse_total"]
                + progress_values["replan_residual_total"]
            )
            balance_error = abs(observed_work_delta - expected_work_delta)
            max_errors["work_balance_abs"] = max(
                max_errors["work_balance_abs"], balance_error
            )
            record(
                "work_balance_closes",
                _close(observed_work_delta, expected_work_delta, tolerance),
                sample,
                {
                    "observed_work_delta": observed_work_delta,
                    "expected_work_delta": expected_work_delta,
                    "abs_error": balance_error,
                },
            )
        else:
            record(
                "work_balance_closes",
                False,
                sample,
                {"reason": "invalid progress ledger"},
            )

    checks = {
        name: {
            "passed": failures[name] == 0,
            "failures": int(failures[name]),
            "examples": examples[name],
        }
        for name in check_names
    }
    return {
        "schema_version": STRICT_AUDIT_SCHEMA_VERSION,
        "samples": int(len(samples)),
        "tolerance": float(tolerance),
        "checks": checks,
        "max_errors": max_errors,
        "passed": bool(samples) and all(row["passed"] for row in checks.values()),
    }


def build_short_horizon_certification(
    report: Mapping,
    *,
    method: str,
    semantic_audit: Mapping,
    invariant_audit: Mapping,
    required_loads: Sequence[str] = ("low", "mid", "high"),
    min_groups: int = 100,
    min_groups_per_load: int = 20,
    min_seeds_per_load: int = 0,
    max_tie_rate: float = 1.0,
    min_method_flips: int = 20,
    pairwise_gap_threshold: float = 0.1,
    min_pairwise_pairs: int = 100,
    min_pairwise_realized_accuracy: float = 0.55,
    max_realized_cost_mean_delta: float = 0.0,
    min_productive_progress_mean_delta: float = 0.0,
    require_placebo_superiority: bool = True,
) -> dict:
    """Build a pre-registered, held-out short-horizon functional gate."""
    checks = {}

    def add(name: str, passed: bool, observed, requirement: str, kind: str) -> None:
        checks[name] = {
            "passed": bool(passed),
            "observed": observed,
            "requirement": requirement,
            "kind": kind,
        }

    methods = report.get("methods") or {}
    candidate = methods.get(method)
    add(
        "method_exists",
        isinstance(candidate, Mapping),
        method if isinstance(candidate, Mapping) else None,
        f"report.methods contains {method!r}",
        "data",
    )
    add(
        "isolated_semantics",
        bool(semantic_audit.get("passed", False)),
        bool(semantic_audit.get("passed", False)),
        "all strict isolated semantic checks pass",
        "data",
    )
    add(
        "analytic_invariants",
        bool(invariant_audit.get("passed", False)),
        bool(invariant_audit.get("passed", False)),
        "all analytic ledger invariants pass",
        "data",
    )

    groups = int(report.get("groups", 0))
    add(
        "enough_groups",
        groups >= int(min_groups),
        groups,
        f">= {int(min_groups)} aggregate candidate groups",
        "data",
    )
    manifest = report.get("manifest") or {}
    for load in required_loads:
        load_groups = int((manifest.get(load) or {}).get("groups", 0))
        load_seeds = list((manifest.get(load) or {}).get("seeds", ()))
        add(
            f"enough_groups_load:{load}",
            load_groups >= int(min_groups_per_load),
            load_groups,
            f">= {int(min_groups_per_load)} groups for load {load}",
            "data",
        )
        if min_seeds_per_load > 0:
            add(
                f"enough_seeds_load:{load}",
                len(load_seeds) >= int(min_seeds_per_load),
                load_seeds,
                (
                    f">= {int(min_seeds_per_load)} independent seeds for "
                    f"load {load}"
                ),
                "data",
            )

    component_scale = report.get("component_scale") or {}
    component_gates = component_scale.get("gates") or {}
    tie_rate = component_scale.get("tie_rate")
    usable_component_samples = int(component_scale.get("usable_samples", 0))
    add(
        "component_scale_available",
        bool(component_scale) and usable_component_samples > 0,
        usable_component_samples,
        "> 0 samples with complete component endpoints",
        "data",
    )
    add(
        "component_activation_coverage",
        bool(component_gates.get("all_components_activated", False)),
        component_scale.get("inactive_components"),
        "every enabled L0 component is activated by at least one sample",
        "data",
    )
    add(
        "within_group_drift_discrimination",
        tie_rate is not None
        and math.isfinite(float(tie_rate))
        and float(tie_rate) < float(max_tie_rate),
        tie_rate,
        f"tie rate < {float(max_tie_rate):g}",
        "functional",
    )
    add(
        "component_dominance",
        bool(component_gates.get("component_dominance_pass", False)),
        {
            "component": component_scale.get("max_dominant_component"),
            "rate": component_scale.get("max_dominant_rate"),
            "limit": component_gates.get("component_dominance_limit"),
        },
        "no single component dominates more than its registered limit",
        "functional",
    )
    add(
        "component_scale_ratio",
        bool(component_gates.get("active_p90_scale_ratio_pass", False)),
        {
            "ratio": component_scale.get(
                "active_component_p90_scale_ratio"
            ),
            "limit": component_gates.get(
                "active_p90_scale_ratio_limit"
            ),
        },
        "active-component robust p90 scale ratio stays within its limit",
        "functional",
    )

    gap_key = f"{float(pairwise_gap_threshold):g}"
    gap_slice = (
        (report.get("pairwise") or {})
        .get("practical_gap_slices", {})
        .get(gap_key, {})
    )
    pair_count = int(gap_slice.get("pairs", 0))
    pair_accuracy = gap_slice.get("realized_cost_direction_accuracy")
    add(
        "enough_practical_pairs",
        pair_count >= int(min_pairwise_pairs),
        pair_count,
        (
            f">= {int(min_pairwise_pairs)} pairs with "
            f"|Delta L gap| > {float(pairwise_gap_threshold):g}"
        ),
        "data",
    )
    add(
        "pairwise_realized_direction",
        pair_accuracy is not None
        and math.isfinite(float(pair_accuracy))
        and float(pair_accuracy) >= float(min_pairwise_realized_accuracy),
        pair_accuracy,
        f">= {float(min_pairwise_realized_accuracy):g}",
        "functional",
    )

    if isinstance(candidate, Mapping):
        flips = int(candidate.get("flips", 0))
        realized_mean = float(
            (candidate.get("realized_cost_delta") or {}).get("mean", math.inf)
        )
        progress_mean = float(
            (candidate.get("productive_progress_delta") or {}).get(
                "mean", -math.inf
            )
        )
        improvement_mean = float(
            (candidate.get("lyapunov_improvement") or {}).get("mean", -math.inf)
        )
    else:
        flips = 0
        realized_mean = math.inf
        progress_mean = -math.inf
        improvement_mean = -math.inf

    add(
        "action_signal",
        flips >= int(min_method_flips),
        flips,
        f">= {int(min_method_flips)} non-base action selections",
        "functional",
    )
    add(
        "positive_lyapunov_improvement",
        math.isfinite(improvement_mean) and improvement_mean > 0.0,
        improvement_mean,
        "> 0 mean base Delta L minus selected Delta L",
        "functional",
    )
    add(
        "realized_cost_non_degradation",
        math.isfinite(realized_mean)
        and realized_mean <= float(max_realized_cost_mean_delta),
        realized_mean,
        f"<= {float(max_realized_cost_mean_delta):g} mean delta",
        "functional",
    )
    add(
        "productive_progress_non_degradation",
        math.isfinite(progress_mean)
        and progress_mean >= float(min_productive_progress_mean_delta),
        progress_mean,
        f">= {float(min_productive_progress_mean_delta):g} mean delta",
        "functional",
    )

    by_load = report.get("by_load") or {}
    for load in required_loads:
        load_method = ((by_load.get(load) or {}).get("methods") or {}).get(method)
        load_realized = (
            (load_method.get("realized_cost_delta") or {}).get("mean")
            if isinstance(load_method, Mapping) else None
        )
        load_progress = (
            (load_method.get("productive_progress_delta") or {}).get("mean")
            if isinstance(load_method, Mapping) else None
        )
        add(
            f"realized_cost_non_degradation_load:{load}",
            load_realized is not None
            and math.isfinite(float(load_realized))
            and float(load_realized) <= float(max_realized_cost_mean_delta),
            load_realized,
            f"<= {float(max_realized_cost_mean_delta):g} mean delta",
            "functional",
        )
        add(
            f"productive_progress_non_degradation_load:{load}",
            load_progress is not None
            and math.isfinite(float(load_progress))
            and float(load_progress)
            >= float(min_productive_progress_mean_delta),
            load_progress,
            f">= {float(min_productive_progress_mean_delta):g} mean delta",
            "functional",
        )

    placebo = (report.get("matched_random_placebo") or {}).get(method)
    placebo_p05 = (
        (placebo.get("realized_cost_delta") or {}).get("p05")
        if isinstance(placebo, Mapping) else None
    )
    placebo_passed = (
        placebo_p05 is not None
        and math.isfinite(float(placebo_p05))
        and math.isfinite(realized_mean)
        and realized_mean <= float(placebo_p05)
    )
    add(
        "matched_random_placebo_superiority",
        placebo_passed if require_placebo_superiority else True,
        {
            "candidate_realized_cost_mean_delta": realized_mean,
            "matched_random_p05": placebo_p05,
            "required": bool(require_placebo_superiority),
        },
        (
            "candidate mean realized-cost delta <= matched-random p05"
            if require_placebo_superiority else "reported only"
        ),
        "functional",
    )

    failed_data = [
        name for name, row in checks.items()
        if row["kind"] == "data" and not row["passed"]
    ]
    failed_functional = [
        name for name, row in checks.items()
        if row["kind"] == "functional" and not row["passed"]
    ]
    if not checks["isolated_semantics"]["passed"] or not checks[
            "analytic_invariants"]["passed"]:
        verdict = "INVALID_DATA"
    elif failed_data:
        verdict = "INSUFFICIENT_DATA"
    elif failed_functional:
        verdict = "REDESIGN_REQUIRED"
    else:
        verdict = "PASS_SHORT_HORIZON_ORACLE"
    return {
        "schema_version": SHORT_HORIZON_GATE_SCHEMA_VERSION,
        "protocol_status": "legacy_gap_based_diagnostic",
        "superseded_by": "lyapunov_five_layer_validation_v1",
        "scale_invariant_validity_claim": False,
        "scope": (
            "held-out isolated short-horizon analytic action-selection value; "
            "not an online or stochastic-arrival stability proof"
        ),
        "method": method,
        "required_loads": list(required_loads),
        "checks": checks,
        "failed_data_checks": failed_data,
        "failed_functional_checks": failed_functional,
        "verdict": verdict,
        "passed": verdict == "PASS_SHORT_HORIZON_ORACLE",
    }


def _candidate_id(sample: Mapping) -> Tuple:
    info = sample.get("candidate_info") or {}
    return (
        int(info.get("robot_id", -1)),
        str(sample.get("candidate_key", "")),
    )


def _scalar(sample: Mapping, field: str) -> float:
    if field not in sample:
        raise ValueError(f"sample lacks scalar field {field!r}")
    value = float(sample[field])
    if not math.isfinite(value):
        raise ValueError(f"sample field {field!r} is not finite")
    return value


def _progress(sample: Mapping, name: str = "productive_total") -> float:
    progress = sample.get("lyapunov_l0_progress") or {}
    value = float(progress.get(name, 0.0))
    if not math.isfinite(value):
        raise ValueError(f"progress field {name!r} is not finite")
    return value


def _group_samples(samples: Sequence[dict]) -> Dict[Tuple[str, str], List[dict]]:
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    required = (
        "lyapunov_l0_delta", "lyapunov_l0_progress",
        "candidate_group_id", "candidate_info",
    )
    for sample in samples:
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(f"sample lacks required fields: {missing}")
        run_id = str(sample.get("run_id") or sample.get("_source_path") or "unknown")
        group_id = str(sample["candidate_group_id"])
        groups[(run_id, group_id)].append(sample)
    return {
        key: sorted(members, key=_candidate_id)
        for key, members in groups.items() if len(members) >= 2
    }


def _quantiles(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _base_candidate(members: Sequence[dict], base_field: str) -> dict:
    return min(members, key=lambda sample: (
        _scalar(sample, base_field), _candidate_id(sample)
    ))


def _method_summary(
    groups: Mapping[Tuple[str, str], Sequence[dict]],
    selected: Mapping[Tuple[str, str], dict],
    base_field: str,
) -> dict:
    differences = defaultdict(list)
    flips = 0
    for key, members in groups.items():
        base = _base_candidate(members, base_field)
        choice = selected[key]
        flips += int(_candidate_id(choice) != _candidate_id(base))
        differences["base_cost_delta"].append(
            _scalar(choice, base_field) - _scalar(base, base_field)
        )
        if "realized_cost" in base and "realized_cost" in choice:
            differences["realized_cost_delta"].append(
                _scalar(choice, "realized_cost") - _scalar(base, "realized_cost")
            )
        differences["lyapunov_improvement"].append(
            _scalar(base, "lyapunov_l0_delta")
            - _scalar(choice, "lyapunov_l0_delta")
        )
        differences["productive_progress_delta"].append(
            _progress(choice) - _progress(base)
        )
        for name in (
            "reverse_total", "arrival_total", "replan_residual_total",
            "route_plan_churn_total",
        ):
            differences[f"{name}_delta"].append(
                _progress(choice, name) - _progress(base, name)
            )
        if "rollout_blocked_moves" in base and "rollout_blocked_moves" in choice:
            differences["blocked_moves_delta"].append(
                _scalar(choice, "rollout_blocked_moves")
                - _scalar(base, "rollout_blocked_moves")
            )
    n = len(groups)
    return {
        "groups": n,
        "flips": flips,
        "flip_rate": float(flips / n) if n else 0.0,
        **{
            name: _quantiles(values)
            for name, values in differences.items()
        },
    }


def _select_score(
    groups: Mapping[Tuple[str, str], Sequence[dict]],
    base_field: str,
    lambda_l: float,
    progress_epsilon: float | None,
    min_drift_gain: float = 0.0,
) -> Dict[Tuple[str, str], dict]:
    selected = {}
    for key, members in groups.items():
        base = _base_candidate(members, base_field)
        feasible = list(members)
        if progress_epsilon is not None:
            floor = _progress(base) - float(progress_epsilon)
            feasible = [sample for sample in members if _progress(sample) >= floor - 1e-12]
        choice = min(feasible, key=lambda sample: (
            _scalar(sample, base_field)
            + float(lambda_l) * _scalar(sample, "lyapunov_l0_delta"),
            _scalar(sample, base_field),
            _candidate_id(sample),
        ))
        drift_gain = (
            _scalar(base, "lyapunov_l0_delta")
            - _scalar(choice, "lyapunov_l0_delta")
        )
        selected[key] = (
            choice if drift_gain + 1e-12 >= float(min_drift_gain) else base
        )
    return selected


def _select_oracle(
    groups: Mapping[Tuple[str, str], Sequence[dict]],
    base_field: str,
    progress_epsilon: float | None,
    min_drift_gain: float = 0.0,
) -> Dict[Tuple[str, str], dict]:
    selected = {}
    for key, members in groups.items():
        base = _base_candidate(members, base_field)
        feasible = list(members)
        if progress_epsilon is not None:
            floor = _progress(base) - float(progress_epsilon)
            feasible = [sample for sample in members if _progress(sample) >= floor - 1e-12]
        choice = min(feasible, key=lambda sample: (
            _scalar(sample, "lyapunov_l0_delta"),
            _scalar(sample, base_field),
            _candidate_id(sample),
        ))
        drift_gain = (
            _scalar(base, "lyapunov_l0_delta")
            - _scalar(choice, "lyapunov_l0_delta")
        )
        selected[key] = (
            choice if drift_gain + 1e-12 >= float(min_drift_gain) else base
        )
    return selected


def _matched_random_summary(
    groups: Mapping[Tuple[str, str], Sequence[dict]],
    reference: Mapping[Tuple[str, str], dict],
    base_field: str,
    seed: int,
    repeats: int,
) -> dict:
    rows = []
    keys = sorted(groups)
    for repeat in range(repeats):
        rng = random.Random(int(seed) + 104729 * repeat)
        selected = {}
        for key in keys:
            members = groups[key]
            base = _base_candidate(members, base_field)
            if _candidate_id(reference[key]) == _candidate_id(base):
                selected[key] = base
                continue
            alternatives = [
                sample for sample in members
                if _candidate_id(sample) != _candidate_id(base)
            ]
            selected[key] = rng.choice(alternatives)
        rows.append(_method_summary(groups, selected, base_field))

    fields = (
        "flip_rate", "base_cost_delta", "realized_cost_delta",
        "lyapunov_improvement", "productive_progress_delta",
        "blocked_moves_delta",
    )
    result = {"repeats": repeats}
    for field in fields:
        values = []
        for row in rows:
            value = row.get(field)
            if isinstance(value, dict) and "mean" in value:
                values.append(value["mean"])
            elif isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            result[field] = _quantiles(values)
    return result


def _pairwise_report(
    groups: Mapping[Tuple[str, str], Sequence[dict]],
    base_field: str,
    thresholds: Sequence[float],
) -> dict:
    gaps = []
    pairs = []
    group_ranges = []
    for members in groups.values():
        deltas = [_scalar(sample, "lyapunov_l0_delta") for sample in members]
        group_ranges.append(max(deltas) - min(deltas))
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                delta_gap = (
                    _scalar(members[left], "lyapunov_l0_delta")
                    - _scalar(members[right], "lyapunov_l0_delta")
                )
                gaps.append(abs(delta_gap))
                base_gap = (
                    _scalar(members[left], base_field)
                    - _scalar(members[right], base_field)
                )
                realized_gap = None
                if "realized_cost" in members[left] and "realized_cost" in members[right]:
                    realized_gap = (
                        _scalar(members[left], "realized_cost")
                        - _scalar(members[right], "realized_cost")
                    )
                pairs.append((delta_gap, base_gap, realized_gap))

    slices = {}
    for threshold in thresholds:
        active = [pair for pair in pairs if abs(pair[0]) > float(threshold)]
        base_correct = [pair[0] * pair[1] > 0.0 for pair in active]
        realized_correct = [
            pair[0] * pair[2] > 0.0 for pair in active if pair[2] is not None
        ]
        slices[str(threshold)] = {
            "pairs": len(active),
            "base_cost_direction_accuracy": (
                float(np.mean(base_correct)) if base_correct else None
            ),
            "realized_cost_direction_accuracy": (
                float(np.mean(realized_correct)) if realized_correct else None
            ),
        }
    return {
        "all_pair_gaps": _quantiles(gaps),
        "group_delta_ranges": _quantiles(group_ranges),
        "gap_slices_role": (
            "descriptive numerical-scale diagnostics only; not a "
            "scale-invariant Lyapunov validity gate"
        ),
        "practical_gap_slices": slices,
    }


def _infer_load(sample: Mapping) -> str:
    text = " ".join((
        str(sample.get("run_id", "")),
        str(sample.get("_source_path", "")),
    )).lower()
    match = re.search(r"(?:^|[_/\\])(low|mid|high)(?:[_/\\]|$)", text)
    return match.group(1) if match else "unknown"


def _coverage_report(samples: Sequence[Mapping]) -> dict:
    def nonzero(value) -> bool:
        return abs(float(value)) > 1e-12

    counters = defaultdict(int)
    policies = set()
    for sample in samples:
        start = sample.get("lyapunov_l0_start") or {}
        end = sample.get("lyapunov_l0_end") or {}
        start_components = start.get("components") or {}
        end_components = end.get("components") or {}
        progress = sample.get("lyapunov_l0_progress") or {}
        counters["traffic_target_nonzero"] += int(
            nonzero(start_components.get("traffic", 0.0))
            or nonzero(end_components.get("traffic", 0.0))
        )
        counters["arrival_injection_nonzero"] += int(
            nonzero(progress.get("arrival_total", 0.0))
        )
        counters["replan_residual_nonzero"] += int(
            nonzero(progress.get("replan_residual_total", 0.0))
        )
        counters["route_plan_churn_nonzero"] += int(
            nonzero(progress.get("route_plan_churn_total", 0.0))
        )
        counters["vertex_conflict_nonzero"] += int(
            nonzero(sample.get("rollout_vertex_conflicts", 0.0))
        )
        counters["swap_conflict_nonzero"] += int(
            nonzero(sample.get("rollout_swap_conflicts", 0.0))
        )
        counters["blocked_moves_nonzero"] += int(
            nonzero(sample.get("rollout_blocked_moves", 0.0))
        )
        mask = sample.get("future_mask")
        if mask is not None:
            counters["complete_rollout"] += int(
                int(torch.as_tensor(mask).sum().item()) == int(torch.as_tensor(mask).numel())
            )
        policies.add(str(sample.get("continuation_policy", "unknown")))
    n = len(samples)
    return {
        "samples": n,
        **{
            name: {
                "n": int(value),
                "rate": float(value / n) if n else 0.0,
            }
            for name, value in sorted(counters.items())
        },
        "continuation_policies": sorted(policies),
    }


def _normalise_tree(value):
    """Return a deterministic, JSON-like signature for state equality checks."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        return tuple(sorted(
            (str(key), _normalise_tree(item)) for key, item in value.items()
        ))
    if isinstance(value, (list, tuple)):
        return tuple(_normalise_tree(item) for item in value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    return repr(value)


def _eligible_oracle_sample(sample: Mapping) -> bool:
    required = (
        "lyapunov_l0_delta", "lyapunov_l0_progress",
        "candidate_group_id", "candidate_info",
    )
    return all(key in sample for key in required)


def _component_scale_report(
    samples: Sequence[Mapping],
    groups: Mapping[Tuple[str, str], Sequence[dict]],
    dominance_limit: float,
    scale_ratio_limit: float,
) -> dict:
    component_weight_fields = {
        "work": ("work_weight",),
        "station": ("station_weight",),
        "traffic": ("traffic_weight",),
        "stall": ("stall_weight", "plan_fail_weight"),
        "arrival": ("arrival_weight",),
    }
    first_config = next((
        sample.get("lyapunov_l0_config")
        for sample in samples
        if isinstance(sample.get("lyapunov_l0_config"), Mapping)
    ), None)
    enabled_components = []
    for name in L0_COMPONENT_NAMES:
        fields = component_weight_fields[name]
        if first_config is None or any(
            float(first_config.get(field, 1.0)) > 0.0 for field in fields
        ):
            enabled_components.append(name)
    enabled_set = set(enabled_components)

    deltas = {name: [] for name in L0_COMPONENT_NAMES}
    dominant = Counter()
    usable = 0
    for sample in samples:
        start = sample.get("lyapunov_l0_start") or {}
        end = sample.get("lyapunov_l0_end") or {}
        start_components = start.get("components") or {}
        end_components = end.get("components") or {}
        if any(
            name not in start_components or name not in end_components
            for name in L0_COMPONENT_NAMES
        ):
            continue
        values = {
            name: float(end_components[name]) - float(start_components[name])
            for name in L0_COMPONENT_NAMES
        }
        if not all(math.isfinite(value) for value in values.values()):
            continue
        usable += 1
        for name, value in values.items():
            deltas[name].append(value)
        magnitudes = {name: abs(value) for name, value in values.items()}
        largest = max(magnitudes.values(), default=0.0)
        if largest <= 1e-12:
            dominant["all_zero"] += 1
        else:
            winners = [
                name for name, value in magnitudes.items()
                if abs(value - largest) <= 1e-12
            ]
            dominant[winners[0] if len(winners) == 1 else "tie"] += 1

    components = {}
    active_p90 = []
    for name in L0_COMPONENT_NAMES:
        values = np.asarray(deltas[name], dtype=np.float64)
        absolute = np.abs(values)
        active = absolute > 1e-12
        summary = _quantiles(absolute.tolist())
        summary.update({
            "active_samples": int(active.sum()),
            "active_rate": float(active.mean()) if absolute.size else 0.0,
            "signed_mean": float(values.mean()) if values.size else None,
        })
        components[name] = summary
        if name in enabled_set and active.any():
            p90 = float(np.quantile(absolute[active], 0.90))
            components[name]["active_abs_p90"] = p90
            if p90 > 1e-12:
                active_p90.append((name, p90))

    dominant_rates = {
        name: {
            "n": int(count),
            "rate": float(count / usable) if usable else 0.0,
        }
        for name, count in sorted(dominant.items())
    }
    component_dominance = {
        name: row for name, row in dominant_rates.items()
        if name in enabled_set
    }
    max_dominant_component = None
    max_dominant_rate = 0.0
    if component_dominance:
        max_dominant_component, row = max(
            component_dominance.items(), key=lambda item: item[1]["rate"]
        )
        max_dominant_rate = float(row["rate"])

    scale_ratio = None
    if len(active_p90) >= 2:
        scales = [value for _, value in active_p90]
        scale_ratio = float(max(scales) / min(scales))

    tie_groups = 0
    practical_groups = 0
    group_ranges = []
    for members in groups.values():
        values = [float(member["lyapunov_l0_delta"]) for member in members]
        spread = max(values) - min(values)
        group_ranges.append(spread)
        tie_groups += int(spread <= 1e-9)
        practical_groups += int(spread > 0.1)

    inactive_components = [
        name for name, row in components.items()
        if name in enabled_set and row["active_samples"] == 0
    ]
    dominance_pass = bool(
        usable and max_dominant_rate <= float(dominance_limit)
    )
    scale_ratio_pass = bool(
        scale_ratio is not None and scale_ratio <= float(scale_ratio_limit)
    )
    coverage_pass = not inactive_components
    return {
        "usable_samples": usable,
        "components": components,
        "enabled_components": enabled_components,
        "disabled_components": [
            name for name in L0_COMPONENT_NAMES if name not in enabled_set
        ],
        "dominant_component": dominant_rates,
        "max_dominant_component": max_dominant_component,
        "max_dominant_rate": max_dominant_rate,
        "active_component_p90_scale_ratio": scale_ratio,
        "inactive_components": inactive_components,
        "group_delta_ranges": _quantiles(group_ranges),
        "tie_groups": tie_groups,
        "tie_rate": float(tie_groups / len(groups)) if groups else 0.0,
        "practical_gap_gt_0.1_groups": practical_groups,
        "practical_gap_gt_0.1_rate": (
            float(practical_groups / len(groups)) if groups else 0.0
        ),
        "gates": {
            "component_dominance_limit": float(dominance_limit),
            "component_dominance_pass": dominance_pass,
            "active_p90_scale_ratio_limit": float(scale_ratio_limit),
            "active_p90_scale_ratio_pass": scale_ratio_pass,
            "all_components_activated": coverage_pass,
        },
    }


def evaluate_samples(
    samples: Sequence[dict],
    *,
    base_field: str = "heuristic_cost",
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    progress_epsilons: Sequence[float] = DEFAULT_PROGRESS_EPS,
    gap_thresholds: Sequence[float] = DEFAULT_GAP_THRESHOLDS,
    selection_margins: Sequence[float] = DEFAULT_SELECTION_MARGINS,
    random_seed: int = 2026,
    random_repeats: int = 200,
    component_dominance_limit: float = DEFAULT_COMPONENT_DOMINANCE_LIMIT,
    component_scale_ratio_limit: float = DEFAULT_COMPONENT_SCALE_RATIO_LIMIT,
    _include_breakdowns: bool = True,
) -> dict:
    loaded_samples = list(samples)
    eligible_samples = [
        sample for sample in loaded_samples if _eligible_oracle_sample(sample)
    ]
    groups = _group_samples(eligible_samples)
    if not groups:
        raise ValueError("no candidate groups with at least two members")
    for members in groups.values():
        for sample in members:
            _scalar(sample, base_field)
            _scalar(sample, "lyapunov_l0_delta")
            _progress(sample)

    base = {
        key: _base_candidate(members, base_field)
        for key, members in groups.items()
    }
    methods = {
        "base": _method_summary(groups, base, base_field),
    }
    references = {}
    oracle = _select_oracle(groups, base_field, None)
    methods["analytic_oracle"] = _method_summary(groups, oracle, base_field)
    references["analytic_oracle"] = oracle

    for epsilon in progress_epsilons:
        label = f"analytic_oracle_progress_floor_eps={epsilon:g}"
        choice = _select_oracle(groups, base_field, epsilon)
        methods[label] = _method_summary(groups, choice, base_field)
        references[label] = choice
    for margin in selection_margins:
        label = f"analytic_oracle_min_drift_gain={margin:g}"
        choice = _select_oracle(
            groups, base_field, None, min_drift_gain=margin
        )
        methods[label] = _method_summary(groups, choice, base_field)
        references[label] = choice
        for epsilon in progress_epsilons:
            label = (
                f"analytic_oracle_progress_floor_eps={epsilon:g}"
                f"_min_drift_gain={margin:g}"
            )
            choice = _select_oracle(
                groups, base_field, epsilon, min_drift_gain=margin
            )
            methods[label] = _method_summary(groups, choice, base_field)
            references[label] = choice
    for lambda_l in lambdas:
        label = f"base_plus_delta_lambda={lambda_l:g}"
        choice = _select_score(groups, base_field, lambda_l, None)
        methods[label] = _method_summary(groups, choice, base_field)
        references[label] = choice
        for epsilon in progress_epsilons:
            label = f"base_plus_delta_lambda={lambda_l:g}_progress_floor_eps={epsilon:g}"
            choice = _select_score(groups, base_field, lambda_l, epsilon)
            methods[label] = _method_summary(groups, choice, base_field)
            references[label] = choice
        for margin in selection_margins:
            label = (
                f"base_plus_delta_lambda={lambda_l:g}"
                f"_min_drift_gain={margin:g}"
            )
            choice = _select_score(
                groups, base_field, lambda_l, None,
                min_drift_gain=margin,
            )
            methods[label] = _method_summary(groups, choice, base_field)
            references[label] = choice
            for epsilon in progress_epsilons:
                label = (
                    f"base_plus_delta_lambda={lambda_l:g}"
                    f"_progress_floor_eps={epsilon:g}"
                    f"_min_drift_gain={margin:g}"
                )
                choice = _select_score(
                    groups, base_field, lambda_l, epsilon,
                    min_drift_gain=margin,
                )
                methods[label] = _method_summary(groups, choice, base_field)
                references[label] = choice

    placebo = {
        label: _matched_random_summary(
            groups, choice, base_field, random_seed, random_repeats
        )
        for label, choice in references.items()
        if methods[label]["flips"] > 0
    }

    manifest = defaultdict(lambda: {"samples": 0, "groups": set(), "seeds": set()})
    for key, members in groups.items():
        load = _infer_load(members[0])
        manifest[load]["groups"].add(key)
        for sample in members:
            manifest[load]["samples"] += 1
            if sample.get("simulation_seed") is not None:
                manifest[load]["seeds"].add(int(sample["simulation_seed"]))

    component_scale = _component_scale_report(
        [sample for members in groups.values() for sample in members],
        groups,
        dominance_limit=component_dominance_limit,
        scale_ratio_limit=component_scale_ratio_limit,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "semantics": {
            "base_field": base_field,
            "base_field_is_proxy": base_field == "heuristic_cost",
            "lyapunov_improvement": "base_delta_L_minus_selected_delta_L",
            "progress_floor": "selected_P_prod >= base_P_prod - epsilon",
            "minimum_drift_gain": (
                "candidate is used only when base_delta_L - selected_delta_L "
                "meets the registered practical margin"
            ),
            "scope": "fixed order/pod/station, robot candidates only",
            "online_claim": False,
            "raw_gap_thresholds_are_validity_gates": False,
            "five_layer_validator": (
                "WorldModel.evaluation.evaluate_lyapunov_validity"
            ),
        },
        "loaded_samples": len(loaded_samples),
        "samples": sum(len(members) for members in groups.values()),
        "groups": len(groups),
        "manifest": {
            load: {
                "samples": row["samples"],
                "groups": len(row["groups"]),
                "seeds": sorted(row["seeds"]),
            }
            for load, row in sorted(manifest.items())
        },
        "coverage": _coverage_report([
            sample for members in groups.values() for sample in members
        ]),
        "component_scale": component_scale,
        "pairwise": _pairwise_report(groups, base_field, gap_thresholds),
        "methods": methods,
        "matched_random_placebo": placebo,
    }
    if _include_breakdowns:
        samples_by_load = defaultdict(list)
        for members in groups.values():
            for sample in members:
                samples_by_load[_infer_load(sample)].append(sample)
        report["by_load"] = {}
        for load, load_samples in sorted(samples_by_load.items()):
            load_report = evaluate_samples(
                load_samples,
                base_field=base_field,
                lambdas=lambdas,
                progress_epsilons=progress_epsilons,
                gap_thresholds=gap_thresholds,
                selection_margins=selection_margins,
                random_seed=random_seed,
                random_repeats=1,
                component_dominance_limit=component_dominance_limit,
                component_scale_ratio_limit=component_scale_ratio_limit,
                _include_breakdowns=False,
            )
            load_report.pop("matched_random_placebo", None)
            report["by_load"][load] = load_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--base-field", default="heuristic_cost")
    parser.add_argument(
        "--lambdas", type=_parse_float_list,
        default=DEFAULT_LAMBDAS,
    )
    parser.add_argument(
        "--progress-eps", type=_parse_float_list,
        default=DEFAULT_PROGRESS_EPS,
    )
    parser.add_argument(
        "--gap-thresholds", type=_parse_float_list,
        default=DEFAULT_GAP_THRESHOLDS,
        help=(
            "Descriptive Delta-L scale slices retained for historical "
            "reports. They do not certify Lyapunov validity."
        ),
    )
    parser.add_argument(
        "--selection-margins", type=_parse_float_list,
        default=DEFAULT_SELECTION_MARGINS,
        help=("Minimum analytic drift improvement required before a "
              "candidate may replace the base action."),
    )
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument(
        "--component-dominance-limit", type=float,
        default=DEFAULT_COMPONENT_DOMINANCE_LIMIT,
        help=("Maximum fraction of samples whose absolute drift may be "
              "dominated by one L0 component."),
    )
    parser.add_argument(
        "--component-scale-ratio-limit", type=float,
        default=DEFAULT_COMPONENT_SCALE_RATIO_LIMIT,
        help="Maximum robust p90 scale ratio across active L0 components.",
    )
    parser.add_argument(
        "--strict-isolated",
        action="store_true",
        help=(
            "Refuse any sample that is not tagged as a complete current-schema "
            "isolated rollout with zero generated orders, zero follow-up "
            "assignments, and zero external arrival work."
        ),
    )
    parser.add_argument(
        "--required-l0-schema",
        default=CURRENT_L0_COLLECTION_SCHEMA,
        help=(
            "Collection schema required by --strict-isolated. The current "
            "work+station+cumulative-arrival functional uses "
            "lyapunov_l1_collection_v3."
        ),
    )
    parser.add_argument("--semantic-tolerance", type=float, default=1e-9)
    parser.add_argument("--invariant-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--certify-method",
        default=None,
        help=(
            "Method key to evaluate with the held-out short-horizon gate, for "
            "example analytic_oracle_progress_floor_eps=0_min_drift_gain=0.1."
        ),
    )
    parser.add_argument("--certify-required-loads", default="low,mid,high")
    parser.add_argument("--certify-min-groups", type=int, default=100)
    parser.add_argument("--certify-min-groups-per-load", type=int, default=20)
    parser.add_argument("--certify-min-seeds-per-load", type=int, default=3)
    parser.add_argument("--certify-max-tie-rate", type=float, default=0.50)
    parser.add_argument("--certify-min-method-flips", type=int, default=20)
    parser.add_argument("--certify-gap-threshold", type=float, default=0.1)
    parser.add_argument("--certify-min-pairs", type=int, default=100)
    parser.add_argument(
        "--certify-min-pairwise-accuracy", type=float, default=0.55,
    )
    parser.add_argument(
        "--certify-max-realized-cost-mean-delta", type=float, default=0.0,
    )
    parser.add_argument(
        "--certify-min-productive-progress-mean-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--certify-allow-no-placebo-superiority",
        action="store_true",
        help="Do not require the registered method to beat matched random p05.",
    )
    parser.add_argument(
        "--fail-on-certification-failure",
        action="store_true",
        help="Exit with status 2 unless the short-horizon verdict passes.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.random_repeats <= 0:
        raise SystemExit("--random-repeats must be positive")
    if args.semantic_tolerance < 0.0 or args.invariant_tolerance <= 0.0:
        raise SystemExit("audit tolerances must be positive/non-negative")
    if (
        args.certify_min_groups < 0
        or args.certify_min_groups_per_load < 0
        or args.certify_min_seeds_per_load < 0
        or args.certify_min_method_flips < 0
        or args.certify_min_pairs < 0
    ):
        raise SystemExit("certification count thresholds must be non-negative")
    if args.certify_gap_threshold < 0.0:
        raise SystemExit("--certify-gap-threshold must be non-negative")
    if not 0.0 <= args.certify_min_pairwise_accuracy <= 1.0:
        raise SystemExit("--certify-min-pairwise-accuracy must lie in [0, 1]")
    if not 0.0 <= args.certify_max_tie_rate <= 1.0:
        raise SystemExit("--certify-max-tie-rate must lie in [0, 1]")
    if not 0.0 < args.component_dominance_limit <= 1.0:
        raise SystemExit("--component-dominance-limit must lie in (0, 1]")
    if args.component_scale_ratio_limit < 1.0:
        raise SystemExit("--component-scale-ratio-limit must be >= 1")

    paths = _expand_paths(args.data)
    samples = _load_samples(paths)
    semantic_audit = audit_isolated_semantics(
        samples,
        required_schema=args.required_l0_schema,
        tolerance=args.semantic_tolerance,
    )
    invariant_audit = audit_analytic_invariants(
        samples,
        tolerance=args.invariant_tolerance,
    )
    gap_thresholds = tuple(args.gap_thresholds)
    if args.certify_method is not None:
        gap_thresholds = tuple(sorted(set(
            gap_thresholds + (float(args.certify_gap_threshold),)
        )))
    report = evaluate_samples(
        samples,
        base_field=args.base_field,
        lambdas=args.lambdas,
        progress_epsilons=args.progress_eps,
        gap_thresholds=gap_thresholds,
        selection_margins=args.selection_margins,
        random_seed=args.random_seed,
        random_repeats=args.random_repeats,
        component_dominance_limit=args.component_dominance_limit,
        component_scale_ratio_limit=args.component_scale_ratio_limit,
    )
    report["isolated_semantics_audit"] = semantic_audit
    report["analytic_invariant_audit"] = invariant_audit
    if args.certify_method is not None:
        required_loads = tuple(
            value.strip().lower()
            for value in args.certify_required_loads.split(",")
            if value.strip()
        )
        report["short_horizon_certification"] = (
            build_short_horizon_certification(
                report,
                method=args.certify_method,
                semantic_audit=semantic_audit,
                invariant_audit=invariant_audit,
                required_loads=required_loads,
                min_groups=args.certify_min_groups,
                min_groups_per_load=args.certify_min_groups_per_load,
                min_seeds_per_load=args.certify_min_seeds_per_load,
                max_tie_rate=args.certify_max_tie_rate,
                min_method_flips=args.certify_min_method_flips,
                pairwise_gap_threshold=args.certify_gap_threshold,
                min_pairwise_pairs=args.certify_min_pairs,
                min_pairwise_realized_accuracy=(
                    args.certify_min_pairwise_accuracy
                ),
                max_realized_cost_mean_delta=(
                    args.certify_max_realized_cost_mean_delta
                ),
                min_productive_progress_mean_delta=(
                    args.certify_min_productive_progress_mean_delta
                ),
                require_placebo_superiority=not (
                    args.certify_allow_no_placebo_superiority
                ),
            )
        )
    report["source_files"] = paths
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"saved: {output}")
    else:
        print(text)
    certification = report.get("short_horizon_certification")
    if certification is not None:
        print(
            "short-horizon Lyapunov verdict: "
            f"{certification['verdict']}"
        )
    strict_failed = args.strict_isolated and (
        not semantic_audit["passed"] or not invariant_audit["passed"]
    )
    if strict_failed:
        failed = [
            f"semantic:{name}"
            for name, row in semantic_audit["checks"].items()
            if not row["passed"]
        ] + [
            f"invariant:{name}"
            for name, row in invariant_audit["checks"].items()
            if not row["passed"]
        ]
        print("strict isolated Lyapunov audit failed: " + ", ".join(failed))
        raise SystemExit(2)
    if (
        certification is not None
        and args.fail_on_certification_failure
        and not certification["passed"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
