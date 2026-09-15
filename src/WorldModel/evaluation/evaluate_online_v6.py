"""
Online Evaluation — Greedy vs WorldModel (v6, v2 fixes)
========================================================
Multi-seed, multi-checkpoint comparison with deterministic seed control,
assignment timing, fallback stats, and JSON output.

Usage (multi-checkpoint):
    python -m WorldModel.evaluate_online_v6 ^
      --checkpoints DataGen/wm_checkpoints/v6_formal/best_world_model.pt ^
                    DataGen/wm_checkpoints/v6_risk0/best_world_model.pt ^
      --names v6_formal v6_risk0 ^
      --seeds 101 102 103 104 105 ^
      --ticks 1000 ^
      --save-json DataGen/wm_checkpoints/v6_online_compare_v2.json

Usage (single checkpoint, backward-compatible):
    python -m WorldModel.evaluate_online_v6 ^
      --checkpoint DataGen/wm_checkpoints/v6_formal/best_world_model.pt ^
      --seeds 101 102 103 104 105 ^
      --ticks 1000 ^
      --save-json DataGen/wm_checkpoints/v6_formal/online_eval_v2.json
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import torch

from WorldModel.evaluate import _build_engine, _sim_metrics
from WorldModel.graph.graph_builder import build_static_graph, extract_system_labels
from WorldModel.core.costs import DEFAULT_LAMBDAS
from WorldModel.core.long_risk_schema import (
    LONG_RISK_DRIFT_SIGNALS,
    long_risk_runtime_contract,
)
from WorldModel.core.lyapunov import LYAPUNOV_SNAPSHOT_SCHEMA_VERSION


# =====================================================================
# Deterministic seed control
# =====================================================================

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_global_ids():
    from WorldState.task_state import Task
    from WorldState.order_state import Order
    Task._next_id = 0
    Order._next_id = 0


# =====================================================================
# Extended metrics collection
# =====================================================================

def _extended_sim_metrics(engine, ta, elapsed_s: float) -> dict:
    base = _sim_metrics(engine)
    world = engine.world
    num_agents = len(world.agents)
    max_ticks = engine.config.simulation.max_ticks

    base.update({
        "wall_time_s": round(elapsed_s, 2),
        "ticks": max_ticks,
        "num_agents": num_agents,
    })

    def _p95(values: List[float]) -> float:
        if not values:
            return 0.0
        return round(float(np.percentile(np.array(values, dtype=float), 95)), 6)

    pending_orders = world.order_state.get_pending_orders()
    in_progress_orders = world.order_state.get_in_progress_orders()
    completed_orders = world.order_state.get_completed_orders()
    open_orders = list(pending_orders) + list(in_progress_orders)
    open_ages = [
        max(0.0, float(world.tick - getattr(order, "created_at", world.tick)))
        for order in open_orders
    ]
    pending_ages = [
        max(0.0, float(world.tick - getattr(order, "created_at", world.tick)))
        for order in pending_orders
    ]
    completed_flow_times = [
        max(
            0.0,
            float(
                getattr(order, "completed_at", 0)
                - getattr(order, "created_at", 0)
            ),
        )
        for order in completed_orders
        if getattr(order, "completed_at", None) is not None
    ]
    base.update({
        "open_order_count": len(open_orders),
        "pending_order_count": len(pending_orders),
        "open_order_age_p95": _p95(open_ages),
        "pending_order_age_p95": _p95(pending_ages),
        "completed_order_flow_time_p95": _p95(completed_flow_times),
    })

    stats = getattr(ta, "stats", None)
    if stats:
        action_path_mode = getattr(ta, "action_path_mode", None)
        if action_path_mode is not None:
            base.update({
                "action_path_mode": int(action_path_mode),
                "action_route_encoding": (
                    "canonical_bfs_service_cell_v1"
                    if int(action_path_mode) == 0
                    else "runtime_path_planner_preview_v1"
                ),
                "world_model_path_planner_injected": (
                    getattr(ta, "path_planner", None) is engine.path_planner
                ),
            })
        long_risk_schema_version = getattr(
            ta, "long_risk_schema_version", None
        )
        if long_risk_schema_version:
            base.update({
                "long_risk_schema_version": long_risk_schema_version,
                "long_risk_runtime_contract": long_risk_runtime_contract(),
                "energy_drift_signal": getattr(
                    ta, "energy_drift_signal", None
                ),
                "energy_scoring_mode": getattr(
                    ta, "energy_scoring_mode", None
                ),
            })
        checkpoint_action_schema = getattr(
            ta, "_checkpoint_action_schema", None
        )
        if checkpoint_action_schema:
            base["checkpoint_action_schema"] = dict(
                checkpoint_action_schema
            )
        ac = stats.get("assign_calls", 0)
        mac = stats.get("model_assign_calls", 0)
        fgc = stats.get("fallback_greedy_calls", 0)
        mic = stats.get("model_inference_calls", 0)
        att = stats.get("assignment_time_total_ms", 0.0)
        mitt = stats.get("model_inference_time_total_ms", 0.0)
        base.update({
            "assign_calls": ac,
            "model_assign_calls": mac,
            "fallback_greedy_calls": fgc,
            "warmup_defer_calls": stats.get("warmup_defer_calls", 0),
            "model_inference_calls": mic,
            "fallback_greedy_ratio": round(fgc / max(ac, 1), 4),
            "assignment_time_ms_mean": round(att / max(ac, 1), 4),
            "model_inference_time_ms_mean": round(mitt / max(mic, 1), 4),
            "risk_guard_filtered": stats.get("risk_guard_filtered", 0),
            "risk_guard_fallback": stats.get("risk_guard_fallback", 0),
            "online_robot_candidate_scope": "all_idle",
        })
        all_idle_contexts = stats.get("all_idle_candidate_contexts", 0)
        all_idle_candidates = stats.get("all_idle_candidates_scored", 0)
        if all_idle_contexts:
            base.update({
                "all_idle_candidate_contexts": all_idle_contexts,
                "all_idle_candidates_scored": all_idle_candidates,
                "all_idle_candidates_per_context_mean": round(
                    all_idle_candidates / all_idle_contexts, 4
                ),
            })
        no_assign_contexts = stats.get("native_no_assign_contexts", 0)
        if no_assign_contexts:
            no_assign_selected = stats.get(
                "native_no_assign_selected", 0
            )
            no_assign_scored = stats.get("native_no_assign_scored", 0)
            base.update({
                "native_no_assign_enabled": True,
                "native_no_assign_contexts": no_assign_contexts,
                "native_no_assign_scored": no_assign_scored,
                "native_no_assign_selected": no_assign_selected,
                "native_no_assign_assignment_selected": stats.get(
                    "native_no_assign_assignment_selected", 0
                ),
                "native_no_assign_selected_ratio": round(
                    no_assign_selected / max(no_assign_contexts, 1), 6
                ),
                "native_no_assign_score_mean": round(
                    stats.get("native_no_assign_score_sum", 0.0)
                    / max(no_assign_scored, 1),
                    6,
                ),
                "native_no_assign_margin_mean": round(
                    stats.get("native_no_assign_margin_sum", 0.0)
                    / max(no_assign_contexts, 1),
                    6,
                ),
            })

        dispatch_mode = getattr(ta, "dispatch_potential_mode", "off")
        if dispatch_mode != "off":
            snapshot_count = stats.get("dispatch_snapshot_count", 0)
            dispatch_contexts = stats.get("dispatch_contexts", 0)
            crossing_contexts = stats.get("dispatch_crossing_contexts", 0)
            dominance_contexts = stats.get("dispatch_dominance_contexts", 0)
            base.update({
                "dispatch_potential_mode": dispatch_mode,
                "dispatch_snapshot_count": snapshot_count,
                "dispatch_full_pending_contexts_mean": round(
                    stats.get("dispatch_full_pending_contexts_sum", 0)
                    / max(snapshot_count, 1),
                    6,
                ),
                "dispatch_pressure_contexts_mean": round(
                    stats.get("dispatch_pressure_contexts_sum", 0)
                    / max(snapshot_count, 1),
                    6,
                ),
                "dispatch_eligible_contexts_mean": round(
                    stats.get("dispatch_eligible_contexts_sum", 0)
                    / max(snapshot_count, 1),
                    6,
                ),
                "dispatch_unresolved_pending_orders_mean": round(
                    stats.get(
                        "dispatch_unresolved_pending_orders_sum", 0
                    ) / max(snapshot_count, 1),
                    6,
                ),
                "dispatch_unresolved_pending_orders_max": stats.get(
                    "dispatch_unresolved_pending_orders_max", 0
                ),
                "dispatch_contexts": dispatch_contexts,
                "dispatch_candidates": stats.get(
                    "dispatch_candidates", 0
                ),
                "dispatch_exact_wm_ties": stats.get(
                    "dispatch_exact_wm_ties", 0
                ),
                "dispatch_group_span_max": round(
                    stats.get("dispatch_group_span_max", 0.0), 12
                ),
                "dispatch_group_span_violations": stats.get(
                    "dispatch_group_span_violations", 0
                ),
                "dispatch_raw_wm_defer_selected": stats.get(
                    "dispatch_raw_wm_defer_selected", 0
                ),
                "dispatch_bridge_defer_selected": stats.get(
                    "dispatch_bridge_defer_selected", 0
                ),
                "dispatch_actual_defer_selected": stats.get(
                    "dispatch_actual_defer_selected", 0
                ),
                "dispatch_actual_assignment_selected": stats.get(
                    "dispatch_actual_assignment_selected", 0
                ),
                "dispatch_modified_decisions": stats.get(
                    "dispatch_modified_decisions", 0
                ),
                "dispatch_modified_decision_ratio": round(
                    stats.get("dispatch_modified_decisions", 0)
                    / max(dispatch_contexts, 1),
                    6,
                ),
                "dispatch_crossing_contexts": crossing_contexts,
                "dispatch_crossing_assignment_selected": stats.get(
                    "dispatch_crossing_assignment_selected", 0
                ),
                "dispatch_crossing_violations": stats.get(
                    "dispatch_crossing_violations", 0
                ),
                "dispatch_crossing_assignment_rate": round(
                    stats.get(
                        "dispatch_crossing_assignment_selected", 0
                    ) / max(crossing_contexts, 1),
                    6,
                ),
                "dispatch_dominance_contexts": dominance_contexts,
                "dispatch_dominance_raw_wm_defer": stats.get(
                    "dispatch_dominance_raw_wm_defer", 0
                ),
                "dispatch_dominance_final_assignment": stats.get(
                    "dispatch_dominance_final_assignment", 0
                ),
                "dispatch_dominance_final_defer": stats.get(
                    "dispatch_dominance_final_defer", 0
                ),
                "dispatch_relative_gap_mean": round(
                    stats.get("dispatch_relative_gap_sum", 0.0)
                    / max(dispatch_contexts, 1),
                    6,
                ),
                "dispatch_max_debt_before": round(
                    stats.get("dispatch_max_debt_before", 0.0), 12
                ),
                "dispatch_max_continuous_eligible_mass": round(
                    stats.get(
                        "dispatch_max_continuous_eligible_mass", 0.0
                    ),
                    12,
                ),
                "dispatch_max_continuous_eligible_ticks": stats.get(
                    "dispatch_max_continuous_eligible_ticks", 0
                ),
                "dispatch_eligible_debt_increments": stats.get(
                    "dispatch_eligible_debt_increments", 0
                ),
                "dispatch_temporarily_ineligible_contexts": stats.get(
                    "dispatch_temporarily_ineligible_contexts", 0
                ),
                "dispatch_assigned_contexts": stats.get(
                    "dispatch_assigned_contexts", 0
                ),
                "dispatch_streak_bound_violations": stats.get(
                    "dispatch_streak_bound_violations", 0
                ),
                "max_eligible_defer_streak_bound_passed": (
                    stats.get("dispatch_streak_bound_violations", 0) == 0
                ),
            })

        decision_total = stats.get("decision_contexts_total", 0)
        shadow_compared = stats.get("shadow_greedy_compared", 0)
        shadow_match = stats.get("shadow_greedy_match", 0)
        local_compared = stats.get("local_greedy_compared", 0)
        local_match = stats.get("local_greedy_match", 0)
        selected_rank_sum = stats.get("wm_selected_rank_sum", 0.0)
        selected_rank_gt1 = stats.get("wm_selected_rank_gt1", 0)
        extra_distance_sum = stats.get("wm_selected_extra_distance_sum", 0.0)
        extra_distance_positive = stats.get(
            "wm_selected_extra_distance_positive", 0)

        if decision_total:
            base.update({
                "decision_contexts_total": decision_total,
            })
            if shadow_compared:
                base.update({
                    "shadow_greedy_compared": shadow_compared,
                    "shadow_greedy_match": shadow_match,
                    "shadow_greedy_match_ratio": round(
                        shadow_match / shadow_compared, 4
                    ),
                    "shadow_greedy_mismatch_ratio": round(
                        1.0 - shadow_match / shadow_compared, 4
                    ),
                })
            if local_compared:
                base.update({
                    "local_greedy_compared": local_compared,
                    "local_greedy_match": local_match,
                    "local_greedy_match_ratio": round(
                        local_match / local_compared, 4
                    ),
                    "local_greedy_mismatch_ratio": round(
                        1.0 - local_match / local_compared, 4
                    ),
                    "wm_selected_rank_mean": round(
                        selected_rank_sum / local_compared, 4
                    ),
                    "wm_selected_rank_gt1_ratio": round(
                        selected_rank_gt1 / local_compared, 4
                    ),
                    "wm_selected_extra_distance_mean": round(
                        extra_distance_sum / local_compared, 4
                    ),
                    "wm_selected_extra_distance_positive_ratio": round(
                        extra_distance_positive / local_compared, 4
                    ),
                })

        defer_contexts = stats.get("risk_defer_contexts", 0)
        if defer_contexts:
            defer_deferred = stats.get("risk_defer_deferred", 0)
            defer_executed = stats.get("risk_defer_executed", 0)
            random_contexts = stats.get("random_defer_contexts", 0)
            random_deferred = stats.get("random_defer_deferred", 0)
            score_max = stats.get("risk_defer_score_max", float("-inf"))
            score_min = stats.get("risk_defer_score_min", float("inf"))
            base.update({
                "risk_defer_contexts": defer_contexts,
                "risk_defer_deferred": defer_deferred,
                "risk_defer_executed": defer_executed,
                "risk_defer_topq_batches": stats.get("risk_defer_topq_batches", 0),
                "risk_defer_topq_budget_sum": stats.get("risk_defer_topq_budget_sum", 0),
                "risk_defer_rolling_checks": stats.get("risk_defer_rolling_checks", 0),
                "risk_defer_rolling_warmup": stats.get("risk_defer_rolling_warmup", 0),
                "risk_defer_rate": round(
                    defer_deferred / max(defer_contexts, 1), 4),
                "risk_defer_score_mean": round(
                    stats.get("risk_defer_score_sum", 0.0)
                    / max(defer_contexts, 1), 6),
                "risk_defer_score_max": round(score_max, 6),
                "risk_defer_score_min": round(score_min, 6),
                "risk_defer_deferred_score_mean": round(
                    stats.get("risk_defer_deferred_score_sum", 0.0)
                    / max(defer_deferred, 1), 6),
                "risk_defer_executed_score_mean": round(
                    stats.get("risk_defer_executed_score_sum", 0.0)
                    / max(defer_executed, 1), 6),
                "random_defer_contexts": random_contexts,
                "random_defer_deferred": random_deferred,
                "random_defer_rate_actual": round(
                    random_deferred / max(random_contexts, 1), 4)
                if random_contexts else 0.0,
            })
            rolling_checks = stats.get("risk_defer_rolling_checks", 0)
            rolling_warmup = stats.get("risk_defer_rolling_warmup", 0)
            rolling_ready = max(rolling_checks - rolling_warmup, 0)
            if rolling_checks:
                base["risk_defer_rolling_warmup_rate"] = round(
                    rolling_warmup / max(rolling_checks, 1), 4
                )
            if rolling_ready:
                base["risk_defer_rolling_threshold_mean"] = round(
                    stats.get("risk_defer_rolling_threshold_sum", 0.0)
                    / rolling_ready,
                    6,
                )

        rmc = stats.get("risk_max_count", 0)
        risk_max_max = stats.get("risk_max_max", float("-inf"))
        risk_max_min = stats.get("risk_max_min", float("inf"))
        base.update({
            "risk_max_mean": round(stats.get("risk_max_sum", 0) / max(rmc, 1), 6) if rmc > 0 else None,
            "risk_max_max": round(risk_max_max, 6) if rmc > 0 else None,
            "risk_max_min": round(risk_max_min, 6) if rmc > 0 else None,
            "risk_max_count": rmc,
        })

        inj_candidates = stats.get("station_injection_candidates", 0)
        if inj_candidates:
            inj_selected = stats.get("station_injection_selected", 0)
            inj_max = stats.get("station_injection_penalty_max", float("-inf"))
            base.update({
                "station_injection_candidates": inj_candidates,
                "station_injection_penalty_mean": round(
                    stats.get("station_injection_penalty_sum", 0.0)
                    / max(inj_candidates, 1), 6),
                "station_injection_penalty_max": round(inj_max, 6),
                "station_injection_stress_mean": round(
                    stats.get("station_injection_stress_sum", 0.0)
                    / max(inj_candidates, 1), 6),
                "station_injection_fast_mean": round(
                    stats.get("station_injection_fast_sum", 0.0)
                    / max(inj_candidates, 1), 6),
                "station_injection_selected": inj_selected,
                "station_injection_selected_penalty_mean": round(
                    stats.get("station_injection_selected_penalty_sum", 0.0)
                    / max(inj_selected, 1), 6),
                "station_injection_selected_stress_mean": round(
                    stats.get("station_injection_selected_stress_sum", 0.0)
                    / max(inj_selected, 1), 6),
                "station_injection_selected_fast_mean": round(
                    stats.get("station_injection_selected_fast_sum", 0.0)
                    / max(inj_selected, 1), 6),
            })
            diag_contexts = stats.get("station_injection_diag_contexts", 0)
            if diag_contexts:
                diag_flips = stats.get("station_injection_diag_flips", 0)
                margin_count = stats.get(
                    "station_injection_diag_margin_count", 0)
                ratio_count = stats.get(
                    "station_injection_diag_penalty_to_margin_count", 0)
                base.update({
                    "station_injection_diag_contexts": diag_contexts,
                    "station_injection_diag_flips": diag_flips,
                    "station_injection_diag_flip_rate": round(
                        diag_flips / max(diag_contexts, 1), 4),
                    "station_injection_diag_base_margin_mean": round(
                        stats.get(
                            "station_injection_diag_base_margin_sum", 0.0)
                        / max(margin_count, 1),
                        6),
                    "station_injection_diag_with_margin_mean": round(
                        stats.get(
                            "station_injection_diag_with_margin_sum", 0.0)
                        / max(margin_count, 1),
                        6),
                    "station_injection_diag_penalty_spread_mean": round(
                        stats.get(
                            "station_injection_diag_penalty_spread_sum", 0.0)
                        / max(diag_contexts, 1),
                        6),
                    "station_injection_diag_penalty_to_margin_mean": round(
                        stats.get(
                            "station_injection_diag_penalty_to_margin_sum", 0.0)
                        / max(ratio_count, 1),
                        6),
                    "station_injection_diag_base_best_penalty_mean": round(
                        stats.get(
                            "station_injection_diag_base_best_penalty_sum", 0.0)
                        / max(diag_contexts, 1),
                        6),
                    "station_injection_diag_with_best_penalty_mean": round(
                        stats.get(
                            "station_injection_diag_with_best_penalty_sum", 0.0)
                        / max(diag_contexts, 1),
                        6),
                })

        guard_contexts = stats.get("potential_guard_contexts", 0)
        if guard_contexts:
            guard_deferred = stats.get("potential_guard_deferred", 0)
            guard_executed = stats.get("potential_guard_executed", 0)
            guard_random_contexts = stats.get(
                "potential_guard_random_contexts", 0)
            guard_random_deferred = stats.get(
                "potential_guard_random_deferred", 0)
            score_max = stats.get("potential_guard_score_max", float("-inf"))
            score_min = stats.get("potential_guard_score_min", float("inf"))
            base.update({
                "potential_guard_contexts": guard_contexts,
                "potential_guard_deferred": guard_deferred,
                "potential_guard_executed": guard_executed,
                "potential_guard_rate": round(
                    guard_deferred / max(guard_contexts, 1), 4),
                "potential_guard_score_mean": round(
                    stats.get("potential_guard_score_sum", 0.0)
                    / max(guard_contexts, 1),
                    6),
                "potential_guard_score_max": round(score_max, 6),
                "potential_guard_score_min": round(score_min, 6),
                "potential_guard_deferred_score_mean": round(
                    stats.get("potential_guard_deferred_score_sum", 0.0)
                    / max(guard_deferred, 1),
                    6),
                "potential_guard_executed_score_mean": round(
                    stats.get("potential_guard_executed_score_sum", 0.0)
                    / max(guard_executed, 1),
                    6),
                "potential_guard_rolling_checks": stats.get(
                    "potential_guard_rolling_checks", 0),
                "potential_guard_rolling_warmup": stats.get(
                    "potential_guard_rolling_warmup", 0),
                "potential_guard_random_contexts": guard_random_contexts,
                "potential_guard_random_deferred": guard_random_deferred,
                "potential_guard_random_rate_actual": round(
                    guard_random_deferred / max(guard_random_contexts, 1),
                    4)
                if guard_random_contexts else 0.0,
            })
            guard_ready = max(
                stats.get("potential_guard_rolling_checks", 0)
                - stats.get("potential_guard_rolling_warmup", 0),
                0,
            )
            if guard_contexts:
                base["potential_guard_rolling_warmup_rate"] = round(
                    stats.get("potential_guard_rolling_warmup", 0)
                    / max(guard_contexts, 1),
                    4,
                )
            if guard_ready:
                base["potential_guard_rolling_threshold_mean"] = round(
                    stats.get("potential_guard_rolling_threshold_sum", 0.0)
                    / guard_ready,
                    6,
                )

        pool_contexts = stats.get("pool_scoring_contexts", 0)
        if pool_contexts:
            pool_candidates = stats.get("pool_scoring_candidates", 0)
            pool_selected = stats.get("pool_scoring_selected", 0)
            pool_baseline_selected = stats.get(
                "pool_scoring_baseline_selected", 0)
            pool_robot_sub = stats.get("pool_scoring_robot_substitutions", 0)
            pool_order_repl = stats.get("pool_scoring_order_replacements", 0)
            pool_base_missed = stats.get(
                "pool_scoring_baseline_context_missed", 0)
            base.update({
                "pool_scoring_contexts": pool_contexts,
                "pool_scoring_candidates": pool_candidates,
                "pool_scoring_candidates_per_context": round(
                    pool_candidates / max(pool_contexts, 1), 4),
                "pool_scoring_baseline_selected": pool_baseline_selected,
                "pool_scoring_selected": pool_selected,
                "pool_scoring_selected_ratio": round(
                    pool_selected / max(pool_contexts, 1), 4),
                "pool_scoring_robot_substitution_rate": round(
                    pool_robot_sub / max(pool_selected, 1), 4),
                "pool_scoring_order_replacement_rate": round(
                    pool_order_repl / max(pool_selected, 1), 4),
                "pool_scoring_baseline_context_missed_rate": round(
                    pool_base_missed / max(pool_baseline_selected, 1), 4),
                "pool_scoring_selected_base_score_mean": round(
                    stats.get("pool_scoring_selected_base_score_sum", 0.0)
                    / max(pool_selected, 1),
                    6,
                ),
                "pool_scoring_selected_potential_score_mean": round(
                    stats.get("pool_scoring_selected_potential_score_sum", 0.0)
                    / max(pool_selected, 1),
                    6,
                ),
                "pool_scoring_selected_potential_rank_mean": round(
                    stats.get("pool_scoring_selected_potential_rank_sum", 0.0)
                    / max(pool_selected, 1),
                    6,
                ),
                "pool_scoring_selected_final_score_mean": round(
                    stats.get("pool_scoring_selected_final_score_sum", 0.0)
                    / max(pool_selected, 1),
                    6,
                ),
            })

        trace_contexts = stats.get("decision_trace_contexts", 0)
        if trace_contexts:
            base.update({
                "decision_trace_contexts": trace_contexts,
                "decision_trace_dropped": stats.get(
                    "decision_trace_dropped", 0
                ),
            })

        candidate_guard_contexts = stats.get("candidate_set_guard_contexts", 0)
        if candidate_guard_contexts:
            guard_deferred = stats.get("candidate_set_guard_deferred", 0)
            guard_executed = stats.get("candidate_set_guard_executed", 0)
            threshold_count = stats.get(
                "candidate_set_guard_threshold_count", 0
            )
            base.update({
                "candidate_set_guard_contexts": candidate_guard_contexts,
                "candidate_set_guard_deferred": guard_deferred,
                "candidate_set_guard_executed": guard_executed,
                "candidate_set_guard_rate": round(
                    guard_deferred / max(candidate_guard_contexts, 1), 4
                ),
                "candidate_set_guard_warmup": stats.get(
                    "candidate_set_guard_warmup", 0
                ),
                "candidate_set_guard_potential_triggered": stats.get(
                    "candidate_set_guard_potential_triggered", 0
                ),
                "candidate_set_guard_risk_triggered": stats.get(
                    "candidate_set_guard_risk_triggered", 0
                ),
                "candidate_set_guard_random_deferred": stats.get(
                    "candidate_set_guard_random_deferred", 0
                ),
                "candidate_set_guard_min_potential_mean": round(
                    stats.get("candidate_set_guard_min_potential_sum", 0.0)
                    / max(candidate_guard_contexts, 1),
                    6,
                ),
                "candidate_set_guard_min_risk_mean": round(
                    stats.get("candidate_set_guard_min_risk_sum", 0.0)
                    / max(candidate_guard_contexts, 1),
                    6,
                ),
            })
            if threshold_count:
                base.update({
                    "candidate_set_guard_potential_threshold_mean": round(
                        stats.get(
                            "candidate_set_guard_potential_threshold_sum",
                            0.0,
                        )
                        / threshold_count,
                        6,
                    ),
                    "candidate_set_guard_risk_threshold_mean": round(
                        stats.get(
                            "candidate_set_guard_risk_threshold_sum", 0.0
                        )
                        / threshold_count,
                        6,
                    ),
                })

        margin_contexts = stats.get("margin_substitute_contexts", 0)
        if margin_contexts:
            opportunities = stats.get("margin_substitute_opportunities", 0)
            substituted = stats.get("margin_substitute_substituted", 0)
            threshold_count = stats.get(
                "margin_substitute_threshold_count", 0
            )
            base.update({
                "margin_substitute_contexts": margin_contexts,
                "margin_substitute_opportunities": opportunities,
                "margin_substitute_substituted": substituted,
                "margin_substitute_opportunity_rate": round(
                    opportunities / max(margin_contexts, 1), 4
                ),
                "margin_substitute_rate": round(
                    substituted / max(margin_contexts, 1), 4
                ),
                "margin_substitute_warmup": stats.get(
                    "margin_substitute_warmup", 0
                ),
                "margin_substitute_gap_mean": round(
                    stats.get("margin_substitute_gap_sum", 0.0)
                    / max(opportunities, 1),
                    6,
                ),
                "margin_substitute_gain_mean": round(
                    stats.get("margin_substitute_gain_sum", 0.0)
                    / max(opportunities, 1),
                    6,
                ),
            })
            if threshold_count:
                base["margin_substitute_gap_threshold_mean"] = round(
                    stats.get("margin_substitute_gap_threshold_sum", 0.0)
                    / threshold_count,
                    6,
                )

        energy_contexts = stats.get("energy_scoring_contexts", 0)
        if energy_contexts:
            energy_candidates = stats.get("energy_scoring_candidates", 0)
            noop_contexts = stats.get("energy_noop_contexts", 0)
            noop_deferred = stats.get("energy_noop_deferred", 0)
            base.update({
                "energy_scoring_contexts": energy_contexts,
                "energy_scoring_candidates": energy_candidates,
                "energy_candidates_per_context": round(
                    energy_candidates / max(energy_contexts, 1),
                    4,
                ),
                "energy_score_mean": round(
                    stats.get("energy_score_sum", 0.0)
                    / max(energy_candidates, 1),
                    6,
                ),
                "energy_potential_mean": round(
                    stats.get("energy_potential_sum", 0.0)
                    / max(energy_candidates, 1),
                    6,
                ),
                "energy_service_relief_mean": round(
                    stats.get("energy_service_relief_sum", 0.0)
                    / max(energy_candidates, 1),
                    6,
                ),
                "energy_noop_contexts": noop_contexts,
                "energy_noop_deferred": noop_deferred,
                "energy_noop_rate": round(
                    noop_deferred / max(noop_contexts, 1),
                    4,
                ) if noop_contexts else 0.0,
                "energy_noop_score_mean": round(
                    stats.get("energy_noop_score_sum", 0.0)
                    / max(noop_contexts, 1),
                    6,
                ) if noop_contexts else 0.0,
                "energy_noop_cost_mean": round(
                    stats.get("energy_noop_cost_sum", 0.0)
                    / max(noop_contexts, 1),
                    6,
                ) if noop_contexts else 0.0,
                "energy_noop_best_action_score_mean": round(
                    stats.get("energy_noop_best_action_score_sum", 0.0)
                    / max(noop_contexts, 1),
                    6,
                ) if noop_contexts else 0.0,
            })

        conv_contexts = stats.get("energy_conv_contexts", 0)
        if conv_contexts:
            conv_modified = stats.get("energy_conv_modified_decisions", 0)
            conv_active = stats.get("energy_conv_active_contexts", 0)
            base.update({
                "energy_conv_contexts": conv_contexts,
                "energy_conv_active_contexts": conv_active,
                "energy_conv_warmup_contexts": stats.get(
                    "energy_conv_warmup_contexts", 0
                ),
                "energy_conv_modified_decisions": conv_modified,
                "energy_conv_modified_decision_rate": round(
                    conv_modified / max(conv_contexts, 1),
                    4,
                ),
                "energy_conv_active_rate": round(
                    conv_active / max(conv_contexts, 1),
                    4,
                ),
                "energy_conv_gate_g_mean": round(
                    stats.get("energy_conv_gate_g_sum", 0.0)
                    / max(conv_contexts, 1),
                    6,
                ),
                "energy_conv_v_state_mean": round(
                    stats.get("energy_conv_v_state_sum", 0.0)
                    / max(conv_contexts, 1),
                    6,
                ),
                "energy_conv_signal_selected_mean": round(
                    stats.get("energy_conv_signal_selected_sum", 0.0)
                    / max(conv_contexts, 1),
                    6,
                ),
                "energy_conv_z_base_selected_mean": round(
                    stats.get("energy_conv_z_base_selected_sum", 0.0)
                    / max(conv_contexts, 1),
                    6,
                ),
                "energy_conv_z_delta_selected_mean": round(
                    stats.get("energy_conv_z_delta_selected_sum", 0.0)
                    / max(conv_contexts, 1),
                    6,
                ),
            })

        prio_contexts = stats.get("energy_prio_contexts", 0)
        if prio_contexts:
            prio_deferred = stats.get("energy_prio_deferred", 0)
            prio_executed = stats.get("energy_prio_executed", 0)
            base.update({
                "energy_prio_contexts": prio_contexts,
                "energy_prio_deferred": prio_deferred,
                "energy_prio_executed": prio_executed,
                "energy_prio_defer_rate": round(
                    prio_deferred / max(prio_contexts, 1),
                    4,
                ),
                "energy_prio_warmup_contexts": stats.get(
                    "energy_prio_warmup_contexts", 0
                ),
                "energy_prio_warmup_rate": round(
                    stats.get("energy_prio_warmup_contexts", 0)
                    / max(prio_contexts, 1),
                    4,
                ),
                "energy_prio_random_contexts": stats.get(
                    "energy_prio_random_contexts", 0
                ),
                "energy_prio_random_deferred": stats.get(
                    "energy_prio_random_deferred", 0
                ),
                "energy_prio_mu_t_mean": round(
                    stats.get("energy_prio_mu_t_sum", 0.0)
                    / max(prio_contexts, 1),
                    6,
                ),
                "energy_prio_mu_t_max": round(
                    stats.get("energy_prio_mu_t_max", 0.0),
                    6,
                ),
                "energy_prio_v_backlog_mean": round(
                    stats.get("energy_prio_v_backlog_sum", 0.0)
                    / max(prio_contexts, 1),
                    6,
                ),
                "energy_prio_v_backlog_max": round(
                    stats.get("energy_prio_v_backlog_max", 0.0),
                    6,
                ),
                "energy_prio_priority_executed_mean": round(
                    stats.get("energy_prio_priority_executed_sum", 0.0)
                    / max(prio_executed, 1),
                    6,
                ),
                "energy_prio_priority_deferred_mean": round(
                    stats.get("energy_prio_priority_deferred_sum", 0.0)
                    / max(prio_deferred, 1),
                    6,
                ),
                "energy_prio_relief_executed_mean": round(
                    stats.get("energy_prio_relief_executed_sum", 0.0)
                    / max(prio_executed, 1),
                    6,
                ),
                "energy_prio_relief_deferred_mean": round(
                    stats.get("energy_prio_relief_deferred_sum", 0.0)
                    / max(prio_deferred, 1),
                    6,
                ),
                "energy_prio_drift_executed_mean": round(
                    stats.get("energy_prio_drift_executed_sum", 0.0)
                    / max(prio_executed, 1),
                    6,
                ),
                "energy_prio_drift_deferred_mean": round(
                    stats.get("energy_prio_drift_deferred_sum", 0.0)
                    / max(prio_deferred, 1),
                    6,
                ),
                "energy_prio_noop_executed_mean": round(
                    stats.get("energy_prio_noop_executed_sum", 0.0)
                    / max(prio_executed, 1),
                    6,
                ),
                "energy_prio_noop_deferred_mean": round(
                    stats.get("energy_prio_noop_deferred_sum", 0.0)
                    / max(prio_deferred, 1),
                    6,
                ),
            })

        l0_states = stats.get("lyapunov_l0_state_count", 0)
        l0_candidates = stats.get("lyapunov_l0_arrival_candidates", 0)
        l0_selected = stats.get("lyapunov_l0_arrival_selected", 0)
        if l0_states or l0_candidates:
            base.update({
                "lyapunov_l0_state_count": l0_states,
                "lyapunov_l0_state_mean": round(
                    stats.get("lyapunov_l0_state_sum", 0.0)
                    / max(l0_states, 1),
                    6,
                ),
                "lyapunov_l0_arrival_candidates": l0_candidates,
                "lyapunov_l0_arrival_delta_mean": round(
                    stats.get("lyapunov_l0_arrival_delta_sum", 0.0)
                    / max(l0_candidates, 1),
                    6,
                ),
                "lyapunov_l0_arrival_delta_min": (
                    round(stats["lyapunov_l0_arrival_delta_min"], 6)
                    if l0_candidates else None
                ),
                "lyapunov_l0_arrival_delta_max": (
                    round(stats["lyapunov_l0_arrival_delta_max"], 6)
                    if l0_candidates else None
                ),
                "lyapunov_l0_candidate_eta_mean": round(
                    stats.get("lyapunov_l0_arrival_eta_sum", 0.0)
                    / max(l0_candidates, 1),
                    6,
                ),
                "lyapunov_l0_arrival_selected": l0_selected,
                "lyapunov_l0_selected_delta_mean": round(
                    stats.get(
                        "lyapunov_l0_arrival_selected_delta_sum", 0.0
                    ) / max(l0_selected, 1),
                    6,
                ),
                "lyapunov_l0_selected_eta_mean": round(
                    stats.get(
                        "lyapunov_l0_arrival_selected_eta_sum", 0.0
                    ) / max(l0_selected, 1),
                    6,
                ),
            })

        work_contexts = stats.get("work_drift_contexts", 0)
        work_candidates = stats.get("work_drift_candidates", 0)
        if work_contexts or work_candidates:
            base.update({
                "work_drift_mode": getattr(ta, "work_drift_mode", "off"),
                "work_drift_lambda": float(
                    getattr(ta, "work_drift_lambda", 0.0)
                ),
                "work_drift_contexts": work_contexts,
                "work_drift_candidates": work_candidates,
                "work_drift_candidates_per_context": round(
                    work_candidates / max(work_contexts, 1), 6
                ),
                "work_drift_candidate_superset_contexts": stats.get(
                    "work_drift_candidate_superset_contexts", 0
                ),
                "work_drift_max_candidate_count": stats.get(
                    "work_drift_max_candidate_count", 0
                ),
                "work_drift_raw_mean": round(
                    stats.get("work_drift_raw_sum", 0.0)
                    / max(work_candidates, 1),
                    6,
                ),
                "work_drift_raw_min": (
                    round(stats.get("work_drift_raw_min", 0.0), 6)
                    if work_candidates else None
                ),
                "work_drift_raw_max": (
                    round(stats.get("work_drift_raw_max", 0.0), 6)
                    if work_candidates else None
                ),
                "work_drift_exact_tie_contexts": stats.get(
                    "work_drift_exact_tie_contexts", 0
                ),
                "work_drift_wm_exact_tie_contexts": stats.get(
                    "work_drift_wm_exact_tie_contexts", 0
                ),
                "work_drift_modified_decisions": stats.get(
                    "work_drift_modified_decisions", 0
                ),
                "work_drift_modified_decision_rate": round(
                    stats.get("work_drift_modified_decisions", 0)
                    / max(work_contexts, 1),
                    6,
                ),
                "work_drift_selected_group_range_mean": round(
                    stats.get("work_drift_selected_group_range_sum", 0.0)
                    / max(work_contexts, 1),
                    6,
                ),
                "work_drift_selected_raw_mean": round(
                    stats.get("work_drift_selected_raw_sum", 0.0)
                    / max(work_contexts, 1),
                    6,
                ),
            })

        eta_contexts = stats.get("eta_stratified_contexts", 0)
        context_calls = stats.get("context_stratified_calls", 0)
        if eta_contexts or context_calls:
            base.update({
                "eta_stratified_contexts": eta_contexts,
                "eta_stratified_non_nearest_added": stats.get(
                    "eta_stratified_non_nearest_added", 0
                ),
                "eta_stratified_non_nearest_per_context": round(
                    stats.get("eta_stratified_non_nearest_added", 0)
                    / max(eta_contexts, 1),
                    6,
                ),
                "route_conflict_representatives": stats.get(
                    "route_conflict_representatives", 0
                ),
                "route_conflict_non_nearest_added": stats.get(
                    "route_conflict_non_nearest_added", 0
                ),
                "context_stratified_calls": context_calls,
                "context_stratified_available": stats.get(
                    "context_stratified_available", 0
                ),
                "context_stratified_selected": stats.get(
                    "context_stratified_selected", 0
                ),
            })

    risk_max_values = getattr(ta, "risk_max_values", None)
    if risk_max_values:
        import numpy as _np
        vals = _np.array(risk_max_values)
        base.update({
            "risk_max_p50": round(float(_np.percentile(vals, 50)), 6),
            "risk_max_p90": round(float(_np.percentile(vals, 90)), 6),
            "risk_max_p95": round(float(_np.percentile(vals, 95)), 6),
            "risk_max_p99": round(float(_np.percentile(vals, 99)), 6),
        })

    # Planner-specific diagnostics are opt-in.  PP/A*/external planners do
    # not expose ``runtime_metrics``, so their historical output schema and
    # execution path remain untouched.  PIBT uses these fields to prove that
    # the joint one-step interface was active and invariant-clean.
    planner_metrics = getattr(engine.path_planner, "runtime_metrics", None)
    if callable(planner_metrics):
        base.update(planner_metrics())
        engine_summary = engine.metrics.summarize()
        base.update({
            "path_planner_engine_deadlock_event_ticks": int(
                engine_summary.get("total_deadlock_events", 0)
            ),
            "path_planner_engine_vertex_conflicts": int(
                getattr(engine, "total_vertex_conflicts_detected", 0)
            ),
            "path_planner_engine_swap_conflicts": int(
                getattr(engine, "total_swap_conflicts_detected", 0)
            ),
            "path_planner_engine_avg_plan_ms": round(
                float(engine_summary.get("avg_plan_ms", 0.0)), 6
            ),
        })

    return base


# =====================================================================
# Online label probe — reuses training-time label extraction per tick
# =====================================================================

_WM_COST_LAMBDAS = list(DEFAULT_LAMBDAS)

class OnlineLabelProbe:
    """Per-tick system label extraction using the same function as training.

    wm_label_cost reuses the congestion/pressure channels and their weights
    from DEFAULT_LAMBDAS, excluding the throughput reward (ch5,
    completed_orders_delta) because throughput is evaluated separately
    via completed_orders.
    """

    def __init__(self, engine):
        (_, self._node_map, _, self._local_capacity,
         self._bottleneck_score, _, self._adj) = build_static_graph(
            engine.world.map_state
        )
        self._prev_completed = engine.world.order_state.total_completed
        self.records = []
        self._risk_components = []  # per-tick raw risk components

    def on_tick(self, engine):
        labels = extract_system_labels(
            engine.world,
            self._bottleneck_score,
            self._node_map,
            self._local_capacity,
            adj=self._adj,
            prev_completed=self._prev_completed,
        )
        self._prev_completed = engine.world.order_state.total_completed
        self.records.append(labels)

        from WorldState.risk import compute_unified_risk
        self._risk_components.append(compute_unified_risk(engine.world))

    def summary(self) -> dict:
        _zero = [0.0] * 7
        if not self.records:
            return {
                "wm_label_cost": 0.0,
                "wait_or_stall": 0.0,
                "avg_excess_delay_label": 0.0,
                "station_pressure": 0.0,
                "bottleneck_CVaR": 0.0,
                "completed_orders_delta_sum": 0.0,
                "unified_risk": 0.0,
                "system_label_means": _zero,
                "system_label_max": _zero,
            }

        stacked = torch.stack(self.records)  # (T, 7)
        means = stacked.mean(dim=0).tolist()
        maxes = stacked.max(dim=0).values.tolist()

        wait_or_stall = means[0]
        excess_label = means[1]
        sta_queue_delta = means[2]
        sta_load_imb = means[3]
        station_pressure = sta_queue_delta + sta_load_imb
        bottleneck = means[4]
        completed_sum = stacked[:, 5].sum().item()
        deadlock = maxes[6]

        lam = _WM_COST_LAMBDAS
        wm_cost = (
            lam[0] * means[0]
            + lam[1] * means[1]
            + lam[2] * means[2]
            + lam[3] * means[3]
            + lam[4] * means[4]
            + lam[6] * maxes[6]
        )

        result = {
            "wm_label_cost": round(wm_cost, 4),
            "wait_or_stall": round(wait_or_stall, 4),
            "avg_excess_delay_label": round(excess_label, 4),
            "station_pressure": round(station_pressure, 4),
            "bottleneck_CVaR": round(bottleneck, 4),
            "completed_orders_delta_sum": round(completed_sum, 4),
            "unified_risk": round(deadlock, 4),
            "system_label_means": [round(v, 6) for v in means],
            "system_label_max": [round(v, 6) for v in maxes],
            "congestion_events": int((stacked[:, 6] >= 0.7).sum().item()),
            "severe_events": int((stacked[:, 6] >= 1.0).sum().item()),
            "risk_rate_per_100": round(
                int((stacked[:, 6] >= 0.7).sum().item())
                / max(len(self.records), 1) * 100, 4
            ),
        }

        if self._risk_components:
            stall_vals = [r["stall_ratio"] for r in self._risk_components]
            dl_vals = [r["deadlock_ratio"] for r in self._risk_components]
            ho_vals = [r["handoff_ratio"] for r in self._risk_components]
            result.update({
                "stall_ratio_mean": round(sum(stall_vals) / len(stall_vals), 6),
                "stall_ratio_max": round(max(stall_vals), 6),
                "deadlock_ratio_mean": round(sum(dl_vals) / len(dl_vals), 6),
                "deadlock_ratio_max": round(max(dl_vals), 6),
                "handoff_ratio_mean": round(sum(ho_vals) / len(ho_vals), 6),
                "handoff_ratio_max": round(max(ho_vals), 6),
            })

        return result


def _round_float(value, ndigits: int = 6):
    if value is None:
        return None
    return round(float(value), ndigits)


def _future_window_metrics(probe: OnlineLabelProbe, tick: int, horizons: List[int]):
    """Attach post-decision outcome windows to trace records.

    The trace is emitted during assignment. We use tick+1 as the first future
    label so the target window does not include the state before the selected
    action can have an effect.
    """
    metrics = {}
    total = len(probe.records)
    start = max(0, min(total, int(tick) + 1))

    for horizon in horizons:
        h = int(horizon)
        end = max(start, min(total, start + h))
        prefix = f"future_{h}"
        metrics[f"{prefix}_count"] = int(max(0, end - start))
        if end <= start:
            metrics.update({
                f"{prefix}_station_pressure_mean": None,
                f"{prefix}_bottleneck_CVaR_mean": None,
                f"{prefix}_completed_orders_delta_sum": None,
                f"{prefix}_congestion_events": 0,
                f"{prefix}_severe_events": 0,
                f"{prefix}_unified_risk_max": None,
                f"{prefix}_deadlock_ratio_mean": None,
                f"{prefix}_wm_label_cost": None,
            })
            continue

        window = torch.stack(probe.records[start:end])
        means = window.mean(dim=0)
        maxes = window.max(dim=0).values
        station_pressure = means[2] + means[3]
        wm_cost = (
            _WM_COST_LAMBDAS[0] * means[0]
            + _WM_COST_LAMBDAS[1] * means[1]
            + _WM_COST_LAMBDAS[2] * means[2]
            + _WM_COST_LAMBDAS[3] * means[3]
            + _WM_COST_LAMBDAS[4] * means[4]
            + _WM_COST_LAMBDAS[6] * maxes[6]
        )
        metrics.update({
            f"{prefix}_station_pressure_mean": _round_float(
                station_pressure.item()
            ),
            f"{prefix}_bottleneck_CVaR_mean": _round_float(means[4].item()),
            f"{prefix}_completed_orders_delta_sum": _round_float(
                window[:, 5].sum().item()
            ),
            f"{prefix}_congestion_events": int(
                (window[:, 6] >= 0.7).sum().item()
            ),
            f"{prefix}_severe_events": int(
                (window[:, 6] >= 1.0).sum().item()
            ),
            f"{prefix}_unified_risk_max": _round_float(maxes[6].item()),
            f"{prefix}_wm_label_cost": _round_float(wm_cost.item()),
        })

        risk_window = probe._risk_components[start:end]
        if risk_window:
            dl_vals = [float(r.get("deadlock_ratio", 0.0)) for r in risk_window]
            stall_vals = [float(r.get("stall_ratio", 0.0)) for r in risk_window]
            metrics[f"{prefix}_deadlock_ratio_mean"] = _round_float(
                sum(dl_vals) / len(dl_vals)
            )
            metrics[f"{prefix}_stall_ratio_mean"] = _round_float(
                sum(stall_vals) / len(stall_vals)
            )
        else:
            metrics[f"{prefix}_deadlock_ratio_mean"] = None
            metrics[f"{prefix}_stall_ratio_mean"] = None

    return metrics


def _write_decision_trace_jsonl(
    path: str,
    records: List[dict],
    probe: OnlineLabelProbe,
    seed: int,
    label: str,
    horizons: List[int],
):
    if not path or not records:
        return 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    written = 0
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            out = dict(record)
            out["seed"] = int(seed)
            out["assigner"] = label
            out.update(_future_window_metrics(
                probe, int(record.get("tick", -1)), horizons
            ))
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
    return written


ORDER_ARRIVAL_MANIFEST_SCHEMA_VERSION = "layer5_order_arrival_manifest_v1"


def _order_arrival_manifest(engine) -> dict:
    """Return the realised exogenous order stream, independent of outcomes."""

    orders = sorted(
        engine.world.order_state.orders.values(),
        key=lambda order: (
            int(getattr(order, "created_at", 0)),
            int(getattr(order, "order_id", -1)),
        ),
    )
    rows = []
    for order in orders:
        rows.append({
            "tick": int(getattr(order, "created_at", 0)),
            "order_id": int(getattr(order, "order_id", -1)),
            "station_id": int(getattr(order, "station_id")),
            "sku_demands": {
                str(key): int(value)
                for key, value in sorted(
                    dict(getattr(order, "sku_demands", {})).items(),
                    key=lambda item: str(item[0]),
                )
            },
        })
    canonical = {
        "schema_version": ORDER_ARRIVAL_MANIFEST_SCHEMA_VERSION,
        "orders": rows,
    }
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        **canonical,
        "total_orders": len(rows),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


# =====================================================================
# Single assigner runner (one seed, one assigner)
# =====================================================================

def _run_one_assigner(
    config_path: str,
    ta,
    seed: int,
    max_ticks: int,
    trace_label: Optional[str] = None,
    decision_trace_jsonl: Optional[str] = None,
    decision_trace_horizons: Optional[List[int]] = None,
    td_stream_dir: Optional[str] = None,
    td_stream_frame_stride: int = 5,
    td_stream_run_id: Optional[str] = None,
    td_stream_meta: Optional[dict] = None,
    td_stream_record_lyapunov_l0: bool = False,
    td_stream_lyapunov_l0_config: Optional[dict] = None,
    decision_snapshot_dir: Optional[str] = None,
    decision_snapshot_interval: int = 5,
    decision_snapshot_top_m: Optional[int] = None,
    decision_snapshot_run_id: Optional[str] = None,
    decision_snapshot_meta: Optional[dict] = None,
    decision_snapshot_candidate_scope: str = "top_m_snapshot",
    decision_snapshot_attach_trace: bool = False,
    decision_snapshot_require_trace_alignment: bool = False,
    decision_snapshot_capture_policy: str = "interval_all",
    decision_snapshot_candidate_robot_mode: str = "nearest",
    decision_snapshot_include_no_assign: bool = False,
    decision_snapshot_max_contexts_per_tick: Optional[int] = None,
    decision_snapshot_phase_c_round: Optional[str] = None,
    decision_snapshot_exclude_external_baselines: bool = False,
    recorded_orders_path: Optional[str] = None,
    save_order_manifest: Optional[str] = None,
    path_planner_override: Optional[str] = None,
    path_planner_params_override: Optional[dict] = None,
) -> dict:
    from Config.config_loader import load_config

    set_global_seed(seed)
    reset_global_ids()

    cfg = load_config(config_path)
    cfg.simulation.max_ticks = max_ticks
    cfg.simulation.seed = seed
    if path_planner_override:
        cfg.policies.path_planner = (
            str(path_planner_override),
            dict(path_planner_params_override or {}),
        )
    if decision_snapshot_attach_trace:
        if not hasattr(ta, "decision_trace_enabled"):
            raise TypeError(
                "decision snapshot trace attachment requires a trace-capable "
                "task assigner"
            )
        ta.decision_trace_enabled = True
    if recorded_orders_path:
        recorded_path = os.path.abspath(recorded_orders_path)
        if not os.path.isfile(recorded_path):
            raise FileNotFoundError(recorded_path)
        cfg.policies.order_generator = (
            "RecordedOrderGenerator",
            {
                "recorded_orders_path": recorded_path,
                "immediate_dispatch": False,
            },
        )
        # The manifest already contains initial-pool, scheduled-arrival and
        # backlog-refill orders at their realised baseline ticks.  Re-seeding
        # or refilling here would duplicate them and break the paired test.
        cfg.simulation.initial_order_pool_size = 0
        cfg.simulation.backlog_floor = 0
        cfg.simulation.backlog_refill_mode = "none"

    t0 = time.time()
    engine = _build_engine(cfg, task_assigner=ta)
    probe = OnlineLabelProbe(engine)
    engine.on_tick_callbacks.append(probe.on_tick)
    td_probe = None
    if td_stream_dir:
        from WorldModel.evaluation.td_stream_probe import TDStreamProbe
        td_probe = TDStreamProbe(
            engine,
            out_dir=td_stream_dir,
            frame_stride=td_stream_frame_stride,
            run_id=td_stream_run_id or f"run_seed{seed}",
            meta=td_stream_meta,
            record_lyapunov_l0=td_stream_record_lyapunov_l0,
            lyapunov_l0_config=(
                asdict(ta.lyapunov_l0_config)
                if hasattr(ta, "lyapunov_l0_config")
                else dict(td_stream_lyapunov_l0_config or {})
            ),
        )
        engine.on_tick_callbacks.append(td_probe.on_tick)
    snap_probe = None
    if decision_snapshot_dir:
        from WorldModel.evaluation.decision_snapshot_probe import (
            DecisionSnapshotProbe,
        )
        snap_probe = DecisionSnapshotProbe(
            engine,
            out_dir=decision_snapshot_dir,
            run_id=decision_snapshot_run_id or f"run_seed{seed}",
            sample_interval=decision_snapshot_interval,
            top_m=(decision_snapshot_top_m
                   if decision_snapshot_top_m is not None else 5),
            meta=decision_snapshot_meta,
            reservation_window=int(
                getattr(ta, "reservation_window", 1)
            ),
            candidate_scope=decision_snapshot_candidate_scope,
            attach_assigner_trace=decision_snapshot_attach_trace,
            require_trace_alignment=(
                decision_snapshot_require_trace_alignment
            ),
            lyapunov_l0_config=(
                asdict(ta.lyapunov_l0_config)
                if hasattr(ta, "lyapunov_l0_config")
                else dict(td_stream_lyapunov_l0_config or {})
            ),
            capture_policy=decision_snapshot_capture_policy,
            candidate_robot_mode=decision_snapshot_candidate_robot_mode,
            include_no_assign_candidate=(
                decision_snapshot_include_no_assign
            ),
            max_contexts_per_tick=(
                decision_snapshot_max_contexts_per_tick
            ),
            phase_c_round=decision_snapshot_phase_c_round,
            exclude_external_baselines=(
                decision_snapshot_exclude_external_baselines
            ),
        )
        engine.pre_assignment_callbacks.append(snap_probe.on_pre_assignment)
        engine.on_tick_callbacks.append(snap_probe.on_tick)
    engine.run()
    elapsed = time.time() - t0

    m = _extended_sim_metrics(engine, ta, elapsed)
    m.update(probe.summary())
    order_manifest = _order_arrival_manifest(engine)
    m["order_arrival_manifest_schema_version"] = order_manifest[
        "schema_version"
    ]
    m["order_arrival_manifest_sha256"] = order_manifest["manifest_sha256"]
    m["order_arrival_count"] = order_manifest["total_orders"]
    m["order_arrival_replayed"] = bool(recorded_orders_path)
    if save_order_manifest:
        manifest_path = os.path.abspath(save_order_manifest)
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                existing_manifest = json.load(handle)
            if existing_manifest != order_manifest:
                raise FileExistsError(
                    "existing order-arrival manifest differs from the "
                    f"deterministic rerun: {manifest_path}"
                )
        else:
            os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(order_manifest, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        m["order_arrival_manifest_path"] = manifest_path
    if td_probe is not None:
        td_probe.save()
        m["td_stream_saved_frames"] = td_probe.saved_frames
    if snap_probe is not None:
        snap_probe.save()
        m["decision_snapshots_saved"] = snap_probe.saved_snapshots
    trace_records = getattr(ta, "decision_trace_records", None)
    if decision_trace_jsonl and trace_records is not None:
        saved = _write_decision_trace_jsonl(
            decision_trace_jsonl,
            trace_records,
            probe,
            seed,
            trace_label or ta.__class__.__name__,
            decision_trace_horizons or [50, 100, 200],
        )
        m["decision_trace_saved"] = saved
    return m


# =====================================================================
# Multi-seed, multi-checkpoint runner
# =====================================================================

def run_all_seeds(
    config_path: str,
    checkpoint_map: Dict[str, str],
    seeds: List[int],
    max_ticks: int,
    top_m: int = 5,
    risk_threshold: Optional[float] = None,
    risk_weight: Optional[float] = None,
    long_risk_beta: Optional[Dict[str, float]] = None,
    risk_defer_mode: str = "off",
    risk_defer_threshold: Optional[float] = None,
    risk_defer_score: str = "combined",
    risk_defer_weight_risk_max: float = 1.0,
    risk_defer_weight_terminal: float = 0.5,
    risk_defer_weight_cvar: float = 0.2,
    risk_defer_weight_peak: float = 0.0,
    risk_defer_weight_delta: float = 0.0,
    risk_defer_topq: float = 0.0,
    risk_defer_rolling_window: int = 200,
    risk_defer_rolling_quantile: float = 0.90,
    risk_defer_min_history: int = 50,
    random_defer_rate: float = 0.0,
    random_defer_seed: int = 0,
    station_injection_weight: float = 0.0,
    station_injection_eta_norm: float = 0.35,
    station_injection_near_radius: int = 4,
    station_injection_diagnostic_only: bool = False,
    potential_guard_mode: str = "off",
    potential_guard_rolling_window: int = 200,
    potential_guard_rolling_quantile: float = 0.90,
    potential_guard_min_history: int = 50,
    potential_guard_weight_state: float = 0.5,
    potential_guard_weight_station: float = 1.0,
    potential_guard_weight_injection: float = 1.0,
    potential_guard_weight_risk_max: float = 1.0,
    potential_guard_weight_terminal: float = 0.5,
    potential_guard_weight_cvar: float = 0.2,
    potential_guard_weight_peak: float = 0.0,
    potential_guard_weight_delta: float = 0.0,
    pool_scoring_mode: str = "off",
    pool_scoring_lambda: float = 0.0,
    pool_scoring_context_factor: float = 1.0,
    decision_trace_jsonl: Optional[str] = None,
    decision_trace_max_records: int = 200000,
    decision_trace_horizons: Optional[List[int]] = None,
    td_stream_dir: Optional[str] = None,
    td_stream_frame_stride: int = 5,
    td_stream_record_lyapunov_l0: bool = False,
    td_stream_lyapunov_l0_config: Optional[dict] = None,
    decision_snapshot_dir: Optional[str] = None,
    decision_snapshot_interval: int = 5,
    decision_snapshot_top_m: Optional[int] = None,
    decision_snapshot_arms: str = "wm",
    candidate_set_guard_mode: str = "off",
    candidate_set_guard_features: str = "either",
    candidate_set_guard_rolling_window: int = 200,
    candidate_set_guard_rolling_quantile: float = 0.90,
    candidate_set_guard_min_history: int = 50,
    margin_substitute_mode: str = "off",
    margin_substitute_rolling_window: int = 200,
    margin_substitute_gap_quantile: float = 0.25,
    margin_substitute_min_history: int = 50,
    energy_scoring_mode: str = "off",
    energy_potential_form: str = "endpoint",
    energy_score_lambda: float = 0.0,
    energy_discount: float = 0.95,
    energy_drift_signal: str = "combo",
    energy_gate_mode: str = "sigmoid",
    energy_gate_window: int = 200,
    energy_gate_warmup: int = 50,
    energy_gate_quantile: float = 0.80,
    energy_conv_lambda: float = 0.25,
    energy_conv_random_flip_rate: float = 0.0,
    energy_conv_random_flip_seed: int = 2026,
    energy_prio_drift_lambda: float = 1.0,
    energy_prio_theta: float = 1.0,
    energy_prio_window: int = 200,
    energy_prio_warmup: int = 50,
    energy_mu_ref: float = 1.0,
    energy_mu_min: float = 0.5,
    energy_mu_max: float = 2.0,
    energy_weight_wait: float = 0.0,
    energy_weight_excess: float = 0.0,
    energy_weight_station: float = 1.0,
    energy_weight_bottleneck: float = 1.0,
    energy_weight_severe: float = 2.0,
    energy_weight_completed: float = 0.0,
    energy_weight_long_terminal: float = 0.0,
    energy_weight_long_cvar: float = 0.0,
    energy_weight_long_peak: float = 0.0,
    energy_weight_long_delta: float = 0.0,
    energy_service_relief_scale: float = 1.0,
    energy_service_weight_order_age: float = 1.0,
    energy_service_weight_station_pending: float = 1.0,
    energy_service_weight_order_size: float = 0.0,
    energy_noop_base_margin: float = 0.0,
    energy_noop_cost_scale: float = 10.0,
    energy_noop_weight_order_age: float = 1.0,
    energy_noop_weight_station_pending: float = 1.0,
    energy_noop_weight_global_pending: float = 0.5,
    energy_noop_weight_idle: float = 0.5,
    energy_age_norm: float = 200.0,
    lyapunov_l0_mode: str = "off",
    lyapunov_l0_lambda: float = 0.0,
    lyapunov_l0_config: Optional[dict] = None,
    candidate_robot_mode: str = "nearest",
    candidate_context_mode: str = "prefix",
    candidate_context_factor: float = 2.0,
    candidate_explore_seed: int = 0,
    include_no_assign_candidate: bool = False,
    world_model_only: bool = False,
) -> Dict[int, Dict[str, dict]]:
    from Policies.TaskAssigner import GreedyTaskAssigner, WorldModelTaskAssigner

    per_seed = {}

    for seed in seeds:
        print(f"  --- Seed {seed} ---")
        seed_results = {}

        if not world_model_only:
            greedy_ta = GreedyTaskAssigner()
            greedy_meta = {
                "arm_label": "Greedy",
                "seed": seed,
                "config": config_path,
                "checkpoint_path": None,
                "lyapunov_l0": {
                    "mode": "off",
                    "config": lyapunov_l0_config,
                    "candidate_delta_semantics": (
                        "eta_arrival_barrier_preview_v1"
                    ),
                },
            }
            m = _run_one_assigner(
                config_path, greedy_ta, seed, max_ticks,
                td_stream_dir=td_stream_dir,
                td_stream_frame_stride=td_stream_frame_stride,
                td_stream_run_id=f"Greedy_seed{seed}",
                td_stream_meta=greedy_meta,
                td_stream_record_lyapunov_l0=td_stream_record_lyapunov_l0,
                td_stream_lyapunov_l0_config=td_stream_lyapunov_l0_config,
                decision_snapshot_dir=(
                    decision_snapshot_dir
                    if decision_snapshot_arms == "all" else None
                ),
                decision_snapshot_interval=decision_snapshot_interval,
                decision_snapshot_top_m=(
                    decision_snapshot_top_m
                    if decision_snapshot_top_m is not None else top_m
                ),
                decision_snapshot_run_id=f"Greedy_seed{seed}",
                decision_snapshot_meta=greedy_meta,
            )
            seed_results["Greedy"] = m
            print(f"    seed={seed}  {'Greedy':12s}  "
                  f"orders={m['completed_orders']:3d}  "
                  f"excess={m['avg_excess_delay']:.1f}  "
                  f"wm_cost={m.get('wm_label_cost', 0):.2f}  "
                  f"wait={m.get('wait_or_stall', 0):.2f}  "
                  f"sta_p={m.get('station_pressure', 0):.2f}  "
                  f"bneck={m.get('bottleneck_CVaR', 0):.2f}  "
                  f"risk={m.get('unified_risk', 0):.2f}")

        for label, ckpt in checkpoint_map.items():
            wm_ta = WorldModelTaskAssigner(
                checkpoint_path=ckpt, top_m=top_m,
                risk_threshold=risk_threshold,
                risk_weight=risk_weight,
                long_risk_beta=long_risk_beta,
                risk_defer_mode=risk_defer_mode,
                risk_defer_threshold=risk_defer_threshold,
                risk_defer_score=risk_defer_score,
                risk_defer_weight_risk_max=risk_defer_weight_risk_max,
                risk_defer_weight_terminal=risk_defer_weight_terminal,
                risk_defer_weight_cvar=risk_defer_weight_cvar,
                risk_defer_weight_peak=risk_defer_weight_peak,
                risk_defer_weight_delta=risk_defer_weight_delta,
                risk_defer_topq=risk_defer_topq,
                risk_defer_rolling_window=risk_defer_rolling_window,
                risk_defer_rolling_quantile=risk_defer_rolling_quantile,
                risk_defer_min_history=risk_defer_min_history,
                random_defer_rate=random_defer_rate,
                random_defer_seed=random_defer_seed + seed,
                station_injection_weight=station_injection_weight,
                station_injection_eta_norm=station_injection_eta_norm,
                station_injection_near_radius=station_injection_near_radius,
                station_injection_diagnostic_only=(
                    station_injection_diagnostic_only
                ),
                potential_guard_mode=potential_guard_mode,
                potential_guard_rolling_window=potential_guard_rolling_window,
                potential_guard_rolling_quantile=potential_guard_rolling_quantile,
                potential_guard_min_history=potential_guard_min_history,
                potential_guard_weight_state=potential_guard_weight_state,
                potential_guard_weight_station=potential_guard_weight_station,
                potential_guard_weight_injection=potential_guard_weight_injection,
                potential_guard_weight_risk_max=potential_guard_weight_risk_max,
                potential_guard_weight_terminal=potential_guard_weight_terminal,
                potential_guard_weight_cvar=potential_guard_weight_cvar,
                potential_guard_weight_peak=potential_guard_weight_peak,
                potential_guard_weight_delta=potential_guard_weight_delta,
                pool_scoring_mode=pool_scoring_mode,
                pool_scoring_lambda=pool_scoring_lambda,
                pool_scoring_context_factor=pool_scoring_context_factor,
                decision_trace_enabled=bool(decision_trace_jsonl),
                decision_trace_max_records=decision_trace_max_records,
                candidate_set_guard_mode=candidate_set_guard_mode,
                candidate_set_guard_features=candidate_set_guard_features,
                candidate_set_guard_rolling_window=(
                    candidate_set_guard_rolling_window
                ),
                candidate_set_guard_rolling_quantile=(
                    candidate_set_guard_rolling_quantile
                ),
                candidate_set_guard_min_history=(
                    candidate_set_guard_min_history
                ),
                margin_substitute_mode=margin_substitute_mode,
                margin_substitute_rolling_window=(
                    margin_substitute_rolling_window
                ),
                margin_substitute_gap_quantile=margin_substitute_gap_quantile,
                margin_substitute_min_history=margin_substitute_min_history,
                energy_scoring_mode=energy_scoring_mode,
                energy_potential_form=energy_potential_form,
                energy_score_lambda=energy_score_lambda,
                energy_discount=energy_discount,
                energy_drift_signal=energy_drift_signal,
                energy_gate_mode=energy_gate_mode,
                energy_gate_window=energy_gate_window,
                energy_gate_warmup=energy_gate_warmup,
                energy_gate_quantile=energy_gate_quantile,
                energy_conv_lambda=energy_conv_lambda,
                energy_conv_random_flip_rate=energy_conv_random_flip_rate,
                energy_conv_random_flip_seed=(
                    energy_conv_random_flip_seed + seed
                ),
                energy_prio_drift_lambda=energy_prio_drift_lambda,
                energy_prio_theta=energy_prio_theta,
                energy_prio_window=energy_prio_window,
                energy_prio_warmup=energy_prio_warmup,
                energy_mu_ref=energy_mu_ref,
                energy_mu_min=energy_mu_min,
                energy_mu_max=energy_mu_max,
                energy_weight_wait=energy_weight_wait,
                energy_weight_excess=energy_weight_excess,
                energy_weight_station=energy_weight_station,
                energy_weight_bottleneck=energy_weight_bottleneck,
                energy_weight_severe=energy_weight_severe,
                energy_weight_completed=energy_weight_completed,
                energy_weight_long_terminal=energy_weight_long_terminal,
                energy_weight_long_cvar=energy_weight_long_cvar,
                energy_weight_long_peak=energy_weight_long_peak,
                energy_weight_long_delta=energy_weight_long_delta,
                energy_service_relief_scale=energy_service_relief_scale,
                energy_service_weight_order_age=(
                    energy_service_weight_order_age
                ),
                energy_service_weight_station_pending=(
                    energy_service_weight_station_pending
                ),
                energy_service_weight_order_size=energy_service_weight_order_size,
                energy_noop_base_margin=energy_noop_base_margin,
                energy_noop_cost_scale=energy_noop_cost_scale,
                energy_noop_weight_order_age=energy_noop_weight_order_age,
                energy_noop_weight_station_pending=(
                    energy_noop_weight_station_pending
                ),
                energy_noop_weight_global_pending=(
                    energy_noop_weight_global_pending
                ),
                energy_noop_weight_idle=energy_noop_weight_idle,
                energy_age_norm=energy_age_norm,
                lyapunov_l0_mode=lyapunov_l0_mode,
                lyapunov_l0_lambda=lyapunov_l0_lambda,
                lyapunov_l0_config=lyapunov_l0_config,
                candidate_robot_mode=candidate_robot_mode,
                candidate_context_mode=candidate_context_mode,
                candidate_context_factor=candidate_context_factor,
                candidate_explore_seed=candidate_explore_seed + seed,
                include_no_assign_candidate=include_no_assign_candidate,
            )
            m = _run_one_assigner(
                config_path,
                wm_ta,
                seed,
                max_ticks,
                trace_label=label,
                decision_trace_jsonl=decision_trace_jsonl,
                decision_trace_horizons=decision_trace_horizons,
                td_stream_dir=td_stream_dir,
                td_stream_frame_stride=td_stream_frame_stride,
                td_stream_run_id=f"{label}_seed{seed}",
                td_stream_meta={
                    "arm_label": label,
                    "seed": seed,
                    "config": config_path,
                    "checkpoint_path": ckpt,
                    "native_no_assign_candidate": bool(
                        include_no_assign_candidate
                    ),
                    "energy_conv": {
                        "energy_scoring_mode": energy_scoring_mode,
                        "energy_conv_lambda": energy_conv_lambda,
                        "energy_gate_mode": energy_gate_mode,
                        "energy_gate_window": energy_gate_window,
                        "energy_gate_warmup": energy_gate_warmup,
                        "energy_gate_quantile": energy_gate_quantile,
                    },
                    "lyapunov_l0": {
                        "version": LYAPUNOV_SNAPSHOT_SCHEMA_VERSION,
                        "mode": lyapunov_l0_mode,
                        "lambda": lyapunov_l0_lambda,
                        "config": lyapunov_l0_config,
                        "candidate_robot_mode": candidate_robot_mode,
                        "candidate_context_mode": candidate_context_mode,
                        "candidate_context_factor": candidate_context_factor,
                        "candidate_delta_semantics": (
                            "eta_arrival_barrier_preview_v1"
                        ),
                    },
                },
                td_stream_record_lyapunov_l0=td_stream_record_lyapunov_l0,
                td_stream_lyapunov_l0_config=td_stream_lyapunov_l0_config,
                decision_snapshot_dir=decision_snapshot_dir,
                decision_snapshot_interval=decision_snapshot_interval,
                # None inherits --top-m: the snapshot candidate set must
                # equal the deployed assigner's action scope.
                decision_snapshot_top_m=(
                    decision_snapshot_top_m
                    if decision_snapshot_top_m is not None else top_m
                ),
                decision_snapshot_run_id=f"{label}_seed{seed}",
                decision_snapshot_meta={
                    "arm_label": label,
                    "seed": seed,
                    "config": config_path,
                    "checkpoint_path": ckpt,
                    "lyapunov_l0": {
                        "version": LYAPUNOV_SNAPSHOT_SCHEMA_VERSION,
                        "mode": lyapunov_l0_mode,
                        "lambda": lyapunov_l0_lambda,
                        "config": lyapunov_l0_config,
                        "candidate_robot_mode": candidate_robot_mode,
                        "candidate_context_mode": candidate_context_mode,
                        "candidate_context_factor": candidate_context_factor,
                        "candidate_delta_semantics": (
                            "eta_arrival_barrier_preview_v1"
                        ),
                    },
                },
            )
            seed_results[label] = m
            fbr = m.get("fallback_greedy_ratio", 0)
            defer = m.get("risk_defer_rate", 0)
            print(f"    seed={seed}  {label:12s}  "
                  f"orders={m['completed_orders']:3d}  "
                  f"excess={m['avg_excess_delay']:.1f}  "
                  f"wm_cost={m.get('wm_label_cost', 0):.2f}  "
                  f"wait={m.get('wait_or_stall', 0):.2f}  "
                  f"sta_p={m.get('station_pressure', 0):.2f}  "
                  f"bneck={m.get('bottleneck_CVaR', 0):.2f}  "
                  f"risk={m.get('unified_risk', 0):.2f}  "
                  f"fb={fbr:.4f}  "
                  f"defer={defer:.4f}")

        per_seed[seed] = seed_results

    return per_seed


# =====================================================================
# Multi-seed aggregation
# =====================================================================

def aggregate_seeds(per_seed: Dict[int, Dict[str, dict]]) -> Dict[str, dict]:
    assigners = set()
    for seed_data in per_seed.values():
        assigners.update(seed_data.keys())

    agg = {}
    for assigner in sorted(assigners):
        all_metrics = {}
        for seed_data in per_seed.values():
            m = seed_data.get(assigner, {})
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    all_metrics.setdefault(k, []).append(v)

        summary = {}
        for k, vals in all_metrics.items():
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            summary[f"{k}_mean"] = round(mean, 4)
            summary[f"{k}_std"] = round(std, 4)
        agg[assigner] = summary

    return agg


def compute_comparison(agg: dict, wm_label: str = "WorldModel") -> dict:
    g = agg.get("Greedy", {})
    w = agg.get(wm_label, {})

    g_orders = g.get("completed_orders_mean", 0)
    w_orders = w.get("completed_orders_mean", 0)
    tp_imp = (w_orders - g_orders) / max(g_orders, 1)

    g_wm_cost = g.get("wm_label_cost_mean", 0)
    w_wm_cost = w.get("wm_label_cost_mean", 0)
    wm_cost_diff = w_wm_cost - g_wm_cost

    fbr = w.get("fallback_greedy_ratio_mean", None)

    pass_basic = (
        w_orders >= g_orders * 0.95
        and w_wm_cost <= g_wm_cost
        and (fbr is None or fbr <= 0.2)
    )

    pass_strong = (w_orders > g_orders and w_wm_cost < g_wm_cost)

    result = {
        "throughput_improvement_mean": round(tp_imp, 4),
        "wm_cost_diff": round(wm_cost_diff, 4),
        "pass_basic": pass_basic,
        "pass_strong": pass_strong,
    }
    if fbr is not None:
        result["fallback_greedy_ratio_mean"] = fbr
    return result


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Online Evaluation: WorldModel, with optional Greedy baseline",
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint (backward-compatible)")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                        help="Multiple checkpoints to compare")
    parser.add_argument("--names", type=str, nargs="+", default=None,
                        help="Labels for each checkpoint (must match --checkpoints length)")
    parser.add_argument(
        "--world-model-only",
        action="store_true",
        help=(
            "Run only the World Model arm. No Greedy baseline process is "
            "constructed or executed."
        ),
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[101, 102, 103, 104, 105])
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument(
        "--top-m", type=int, default=5,
        help=(
            "Training/snapshot candidate count only. Online World Model "
            "inference always scores all currently idle robots."
        ),
    )
    parser.add_argument("--risk-threshold", type=float, default=None,
                        help="A1 risk guard: filter candidates with predicted risk_max > threshold")
    parser.add_argument("--risk-weight", type=float, default=None,
                        help="A2 risk weight override: replaces checkpoint lambdas[6]")
    parser.add_argument("--long-risk-beta-peak", type=float, default=0.0,
                        help="B2 long-risk: beta for peak_q95 (default: 0 = disabled)")
    parser.add_argument("--long-risk-beta-cvar", type=float, default=0.0,
                        help="B2 long-risk: beta for cvar_q90")
    parser.add_argument("--long-risk-beta-terminal", type=float, default=0.0,
                        help="B2 long-risk: beta for terminal_q90")
    parser.add_argument("--long-risk-beta-delta", type=float, default=0.0,
                        help="B2 long-risk: beta for delta_group_q90")
    parser.add_argument("--risk-defer-mode", type=str, default="off",
                        choices=[
                            "off", "risk", "random", "risk_topq", "random_topq",
                            "risk_rolling", "random_rolling",
                        ],
                        help=("Pre-6.1 diagnostic trick, not the final method. "
                              "'risk' skips by threshold, 'random' skips by "
                              "probability, and '*_topq' uses a fixed per-call "
                              "defer budget; '*_rolling' uses a rolling score "
                              "quantile."))
    parser.add_argument("--risk-defer-threshold", type=float, default=None,
                        help="Risk-defer threshold; required when --risk-defer-mode risk")
    parser.add_argument("--risk-defer-score", type=str, default="combined",
                        choices=["risk_max", "long_risk", "combined"],
                        help="Risk score source for risk-defer")
    parser.add_argument("--risk-defer-weight-risk-max", type=float, default=1.0)
    parser.add_argument("--risk-defer-weight-terminal", type=float, default=0.5)
    parser.add_argument("--risk-defer-weight-cvar", type=float, default=0.2)
    parser.add_argument("--risk-defer-weight-peak", type=float, default=0.0)
    parser.add_argument("--risk-defer-weight-delta", type=float, default=0.0)
    parser.add_argument("--risk-defer-topq", "--risk-defer-budget-rate",
                        dest="risk_defer_topq", type=float, default=0.0,
                        help="Per-assign-call defer budget for risk_topq/random_topq")
    parser.add_argument("--risk-defer-rolling-window", type=int, default=200,
                        help="Number of recent context scores for rolling quantile")
    parser.add_argument("--risk-defer-rolling-quantile", type=float, default=0.90,
                        help="Rolling quantile threshold for risk_rolling/random_rolling")
    parser.add_argument("--risk-defer-min-history", type=int, default=50,
                        help="Warmup context count before rolling guard activates")
    parser.add_argument("--random-defer-rate", type=float, default=0.0,
                        help="Per-context random defer probability for random control")
    parser.add_argument("--random-defer-seed", type=int, default=2026)
    parser.add_argument("--station-injection-weight", type=float, default=0.0,
                        help=("6.1 diagnostic: add penalty for quickly injecting "
                              "robots into high-pressure station regions "
                              "(0=disabled)."))
    parser.add_argument("--station-injection-eta-norm", type=float, default=0.35,
                        help="Normalized route-length threshold for fast station injection")
    parser.add_argument("--station-injection-near-radius", type=int, default=4,
                        help="Radius for counting robots near each station")
    parser.add_argument("--station-injection-diagnostic-only",
                        action="store_true",
                        help=("Compute station-injection counterfactual "
                              "flip diagnostics without adding the penalty "
                              "to the executed score."))
    parser.add_argument("--potential-guard-mode", type=str, default="off",
                        choices=["off", "rolling", "random_rolling"],
                        help=("Pre-6.1 candidate admission guard. 'rolling' "
                              "defers selected candidates whose potential "
                              "score is above the recent rolling quantile; "
                              "'random_rolling' is the matched random control."))
    parser.add_argument("--potential-guard-rolling-window", type=int,
                        default=200)
    parser.add_argument("--potential-guard-rolling-quantile", type=float,
                        default=0.90)
    parser.add_argument("--potential-guard-min-history", type=int, default=50)
    parser.add_argument("--potential-guard-weight-state", type=float,
                        default=0.5)
    parser.add_argument("--potential-guard-weight-station", type=float,
                        default=1.0)
    parser.add_argument("--potential-guard-weight-injection", type=float,
                        default=1.0)
    parser.add_argument("--potential-guard-weight-risk-max", type=float,
                        default=1.0)
    parser.add_argument("--potential-guard-weight-terminal", type=float,
                        default=0.5)
    parser.add_argument("--potential-guard-weight-cvar", type=float,
                        default=0.2)
    parser.add_argument("--potential-guard-weight-peak", type=float,
                        default=0.0)
    parser.add_argument("--potential-guard-weight-delta", type=float,
                        default=0.0)
    parser.add_argument("--pool-scoring-mode", type=str, default="off",
                        choices=[
                            "off", "potential_rank", "context_rank",
                            "energy_score", "energy_conversion",
                            "random_defer_matched",
                        ],
                        help=("6.1 candidate-pool scoring. 'potential_rank' "
                              "scores all order+robot candidates together "
                              "with a batch-relative potential rank term; "
                              "'context_rank' only re-ranks robots within "
                              "each context and preserves context order; "
                              "'energy_score' globally compares order+robot "
                              "actions using the Phase 6.1 score(a); "
                              "'energy_conversion' is the S2 priority/no-op "
                              "branch on top of S1 conversion; "
                              "'random_defer_matched' is its matched random "
                              "pacing control."))
    parser.add_argument("--pool-scoring-lambda", type=float, default=0.0,
                        help="Weight for batch-relative potential rank")
    parser.add_argument("--pool-scoring-context-factor", type=float, default=1.0,
                        help=("Candidate contexts requested per idle robot in "
                              "pool scoring; >1 opens order/station choice"))
    parser.add_argument("--decision-trace-jsonl", "--save-decision-trace",
                        dest="decision_trace_jsonl", type=str, default=None,
                        help=("Save trace-only per-context candidate diagnostics "
                              "as JSONL. This does not change decisions."))
    parser.add_argument("--decision-trace-max-records", type=int,
                        default=200000,
                        help="Maximum trace records kept per assigner run")
    parser.add_argument("--decision-trace-horizons", type=int, nargs="+",
                        default=[50, 100, 200],
                        help="Future tick windows attached to each trace record")
    parser.add_argument("--td-stream-dir", type=str, default=None,
                        help=("Dump per-run TD adjacent-segment streams "
                              "(td_stream_v1: per-tick unified risk + strided "
                              "observation frames) into this directory. "
                              "Dumps ALL online arms including Greedy; "
                              "observation-only, does not change decisions."))
    parser.add_argument("--td-stream-frame-stride", type=int, default=5,
                        help="Ticks between captured frames in the TD stream")
    parser.add_argument(
        "--td-stream-record-lyapunov-l0",
        "--td-stream-lyapunov-l0",
        action="store_true",
        help=("Add analytic L0 components and productive/reverse/arrival "
              "work flow to "
              "each TD stream. Observation-only; does not change decisions."),
    )
    parser.add_argument(
        "--td-stream-lyapunov-config",
        type=str,
        default=None,
        help=("Optional JSON file containing LyapunovL0Config fields. Supplying "
              "it also enables --td-stream-record-lyapunov-l0."),
    )
    parser.add_argument("--decision-snapshot-dir", type=str, default=None,
                        help=("Dump Phase C decision-point snapshots "
                              "(phaseC_decision_snapshot_v1, contract superset "
                              "of phaseB_b0_decision_snapshot_v1) into this "
                              "directory. Capture side does full interval "
                              "sampling; priority selection is offline. "
                              "Observation-only, does not change decisions."))
    parser.add_argument("--decision-snapshot-interval", type=int, default=5,
                        help=("Ticks between decision-point captures "
                              "(default 5 = phaseB sample_interval = TD "
                              "frame stride, keeping the tick grids aligned)"))
    parser.add_argument("--decision-snapshot-top-m", type=int, default=None,
                        help=("Candidates per group in snapshots. Default "
                              "None inherits --top-m so the snapshot "
                              "candidate set equals the deployed assigner's "
                              "action scope; set lower only to save storage."))
    parser.add_argument("--decision-snapshot-arms", type=str, default="wm",
                        choices=["wm", "all"],
                        help=("Which arms dump decision snapshots. Default "
                              "'wm' skips Greedy (its states are already in "
                              "D_0; Phase C wants the WM-induced "
                              "distribution)."))
    parser.add_argument("--candidate-set-guard-mode", type=str, default="off",
                        choices=["off", "rolling", "random_rolling"],
                        help=("Adaptive all-bad candidate-set guard. "
                              "'rolling' defers a context when candidate-set "
                              "minimum potential/risk is above its recent "
                              "rolling quantile; no fixed absolute threshold "
                              "is used."))
    parser.add_argument("--candidate-set-guard-features", type=str,
                        default="either",
                        choices=["either", "potential", "risk"],
                        help=("Which candidate-set signal triggers the guard. "
                              "'either' uses candidate_min_potential OR "
                              "candidate_min_risk_score."))
    parser.add_argument("--candidate-set-guard-rolling-window", type=int,
                        default=200)
    parser.add_argument("--candidate-set-guard-rolling-quantile", type=float,
                        default=0.90)
    parser.add_argument("--candidate-set-guard-min-history", type=int,
                        default=50)
    parser.add_argument("--margin-substitute-mode", type=str, default="off",
                        choices=["off", "rolling"],
                        help=("Adaptive margin-gated substitute. It only "
                              "switches to the best-potential robot when the "
                              "baseline is not already best-potential and the "
                              "base-score gap is in a recent low quantile."))
    parser.add_argument("--margin-substitute-rolling-window", type=int,
                        default=200)
    parser.add_argument("--margin-substitute-gap-quantile", type=float,
                        default=0.25)
    parser.add_argument("--margin-substitute-min-history", type=int,
                        default=50)
    parser.add_argument("--energy-scoring-mode", type=str, default="off",
                        choices=["off", "additive", "with_noop", "conversion"],
                        help=("Phase 6.1 energy-shaped scoring. 'additive' "
                              "adds V_pred(s,a) to candidate scores; "
                              "'with_noop' also compares against a dynamic "
                              "no-op opportunity-cost candidate; 'conversion' "
                              "uses gated within-context drift scoring."))
    parser.add_argument("--energy-potential-form", type=str,
                        default="endpoint",
                        choices=["discounted", "endpoint"],
                        help=("Potential form used by additive/with_noop "
                              "energy scoring."))
    parser.add_argument("--energy-score-lambda", type=float, default=0.0)
    parser.add_argument("--energy-discount", type=float, default=0.95)
    parser.add_argument("--energy-drift-signal", type=str, default="combo",
                        choices=list(LONG_RISK_DRIFT_SIGNALS),
                        help=("Long-risk channel used by conversion scoring: "
                              "combo=0.2*peak_q95+0.3*cvar_q90+"
                              "0.5*terminal_q90, terminal=C1 ablation, "
                              "event_logit=reproduction of the legacy S1 "
                              "runtime semantics."))
    parser.add_argument("--energy-gate-mode", type=str, default="sigmoid",
                        choices=["sigmoid", "hard", "off"],
                        help=("Conversion gate: rolling sigmoid, hard gate, "
                              "or always-on ablation."))
    parser.add_argument("--energy-gate-window", type=int, default=200)
    parser.add_argument("--energy-gate-warmup", type=int, default=50)
    parser.add_argument("--energy-gate-quantile", type=float, default=0.80)
    parser.add_argument("--energy-conv-lambda", type=float, default=0.25,
                        help=("Lambda_E for conversion scoring. This is "
                              "separate from --energy-score-lambda."))
    parser.add_argument("--energy-conv-random-flip-rate", type=float,
                        default=0.0,
                        help=("S4 matched-random control: per decision "
                              "context, flip the selected candidate to a "
                              "uniformly random non-best one with this "
                              "probability INSTEAD of drift scoring. Match "
                              "the rate to A1's measured "
                              "energy_conv_modified_decision_rate. Requires "
                              "--energy-scoring-mode conversion."))
    parser.add_argument("--energy-conv-random-flip-seed", type=int,
                        default=2026)
    parser.add_argument("--energy-prio-drift-lambda", type=float, default=1.0,
                        help="S2 priority drift penalty weight")
    parser.add_argument("--energy-prio-theta", type=float, default=1.0,
                        help=("S2 defer threshold coefficient; >=1e6 disables "
                              "defer and leaves priority ordering only"))
    parser.add_argument("--energy-prio-window", type=int, default=200,
                        help="Rolling window for S2 relief/drift/no-op std")
    parser.add_argument("--energy-prio-warmup", type=int, default=50,
                        help="Warmup contexts before S2 priority can defer")
    parser.add_argument("--energy-mu-ref", type=float, default=1.0,
                        help="Reference backlog potential for mu_t")
    parser.add_argument("--energy-mu-min", type=float, default=0.5,
                        help="Lower clip for mu_t")
    parser.add_argument("--energy-mu-max", type=float, default=2.0,
                        help="Upper clip for mu_t")
    parser.add_argument("--energy-weight-wait", type=float, default=0.0)
    parser.add_argument("--energy-weight-excess", type=float, default=0.0)
    parser.add_argument("--energy-weight-station", type=float, default=1.0)
    parser.add_argument("--energy-weight-bottleneck", type=float, default=1.0)
    parser.add_argument("--energy-weight-severe", type=float, default=2.0)
    parser.add_argument("--energy-weight-completed", type=float, default=0.0)
    parser.add_argument("--energy-weight-long-terminal", type=float,
                        default=0.0)
    parser.add_argument("--energy-weight-long-cvar", type=float, default=0.0)
    parser.add_argument("--energy-weight-long-peak", type=float, default=0.0)
    parser.add_argument("--energy-weight-long-delta", type=float, default=0.0)
    parser.add_argument("--energy-service-relief-scale", type=float,
                        default=1.0)
    parser.add_argument("--energy-service-weight-order-age", type=float,
                        default=1.0)
    parser.add_argument("--energy-service-weight-station-pending", type=float,
                        default=1.0)
    parser.add_argument("--energy-service-weight-order-size", type=float,
                        default=0.0)
    parser.add_argument("--energy-noop-base-margin", type=float, default=0.0)
    parser.add_argument("--energy-noop-cost-scale", type=float, default=10.0)
    parser.add_argument("--energy-noop-weight-order-age", type=float,
                        default=1.0)
    parser.add_argument("--energy-noop-weight-station-pending", type=float,
                        default=1.0)
    parser.add_argument("--energy-noop-weight-global-pending", type=float,
                        default=0.5)
    parser.add_argument("--energy-noop-weight-idle", type=float, default=0.5)
    parser.add_argument("--energy-age-norm", type=float, default=200.0)
    parser.add_argument(
        "--lyapunov-l0-mode",
        choices=["off", "diagnostic", "additive"],
        default="off",
        help=(
            "Analytic L0 ETA-arrival preview: off, trace-only diagnostic, "
            "or additive candidate penalty. This is not yet learned H-step "
            "full-L0 drift."
        ),
    )
    parser.add_argument("--lyapunov-l0-lambda", type=float, default=0.0)
    parser.add_argument(
        "--lyapunov-l0-work-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--lyapunov-l0-station-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--lyapunov-l0-traffic-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--lyapunov-l0-stall-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--lyapunov-l0-plan-fail-weight", type=float, default=0.5
    )
    parser.add_argument(
        "--lyapunov-l0-arrival-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--lyapunov-l0-eta-bins",
        type=int,
        nargs="+",
        default=[10, 25, 50],
    )
    parser.add_argument(
        "--lyapunov-l0-reservation-window", type=int, default=10
    )
    parser.add_argument(
        "--candidate-robot-mode",
        choices=["nearest", "eta_stratified", "stratified"],
        default="nearest",
        help=(
            "Compatibility/diagnostic setting for collection metadata. It "
            "does not truncate online World Model robot candidates; online "
            "scope is always all idle robots."
        ),
    )
    parser.add_argument(
        "--candidate-context-mode",
        choices=["prefix", "stratified"],
        default="prefix",
        help=(
            "Context coverage. prefix preserves legacy pending-order order; "
            "stratified samples a larger pool then keeps pressure/ETA/route/"
            "conflict representatives."
        ),
    )
    parser.add_argument(
        "--candidate-context-factor", type=float, default=2.0
    )
    parser.add_argument(
        "--candidate-explore-seed", type=int, default=2026
    )
    parser.add_argument(
        "--include-no-assign",
        action="store_true",
        help=(
            "Enable the native model-scored NO_ASSIGN action. The checkpoint "
            "must carry the complete wm_native_no_assign_action_v1 training "
            "schema; old checkpoints are rejected."
        ),
    )
    parser.add_argument("--save-json", type=str, default=None)

    args = parser.parse_args()

    if args.risk_defer_mode == "risk" and args.risk_defer_threshold is None:
        parser.error("--risk-defer-threshold is required when --risk-defer-mode risk")
    if args.risk_defer_mode == "random" and not (0.0 <= args.random_defer_rate <= 1.0):
        parser.error("--random-defer-rate must be in [0, 1]")
    if args.risk_defer_mode in ("risk_topq", "random_topq"):
        if not (0.0 < args.risk_defer_topq <= 1.0):
            parser.error("--risk-defer-topq must be in (0, 1] for *_topq modes")
    if args.risk_defer_mode in ("risk_rolling", "random_rolling"):
        if args.risk_defer_rolling_window <= 0:
            parser.error("--risk-defer-rolling-window must be positive")
        if args.risk_defer_min_history <= 0:
            parser.error("--risk-defer-min-history must be positive")
        if not (0.0 < args.risk_defer_rolling_quantile < 1.0):
            parser.error("--risk-defer-rolling-quantile must be in (0, 1)")
    if args.station_injection_weight < 0.0:
        parser.error("--station-injection-weight must be >= 0")
    if args.station_injection_eta_norm <= 0.0:
        parser.error("--station-injection-eta-norm must be > 0")
    if args.station_injection_near_radius < 0:
        parser.error("--station-injection-near-radius must be >= 0")
    if args.potential_guard_mode != "off":
        if args.potential_guard_rolling_window <= 0:
            parser.error("--potential-guard-rolling-window must be positive")
        if args.potential_guard_min_history <= 0:
            parser.error("--potential-guard-min-history must be positive")
        if not (0.0 < args.potential_guard_rolling_quantile < 1.0):
            parser.error(
                "--potential-guard-rolling-quantile must be in (0, 1)"
            )
    if args.pool_scoring_lambda < 0.0:
        parser.error("--pool-scoring-lambda must be >= 0")
    if args.pool_scoring_context_factor < 1.0:
        parser.error("--pool-scoring-context-factor must be >= 1")
    if args.pool_scoring_mode in (
        "energy_conversion", "random_defer_matched",
    ):
        if args.energy_scoring_mode != "conversion":
            parser.error(
                "--pool-scoring-mode energy_conversion/random_defer_matched "
                "requires --energy-scoring-mode conversion"
            )
        if args.energy_prio_drift_lambda < 0.0:
            parser.error("--energy-prio-drift-lambda must be >= 0")
        if args.energy_prio_theta < 0.0:
            parser.error("--energy-prio-theta must be >= 0")
        if args.energy_prio_window <= 0:
            parser.error("--energy-prio-window must be positive")
        if args.energy_prio_warmup < 0:
            parser.error("--energy-prio-warmup must be >= 0")
        if args.energy_mu_ref <= 0.0:
            parser.error("--energy-mu-ref must be > 0")
        if args.energy_mu_min < 0.0 or args.energy_mu_max < 0.0:
            parser.error("--energy-mu-min/max must be >= 0")
        if args.energy_mu_min > args.energy_mu_max:
            parser.error("--energy-mu-min must be <= --energy-mu-max")
        if (
            args.pool_scoring_mode == "random_defer_matched"
            and not (0.0 <= args.random_defer_rate <= 1.0)
        ):
            parser.error(
                "--random-defer-rate must be in [0, 1] for "
                "random_defer_matched"
            )
    if args.decision_trace_max_records < 0:
        parser.error("--decision-trace-max-records must be >= 0")
    if any(h <= 0 for h in args.decision_trace_horizons):
        parser.error("--decision-trace-horizons must be positive integers")
    if args.td_stream_frame_stride <= 0:
        parser.error("--td-stream-frame-stride must be positive")
    if ((args.td_stream_record_lyapunov_l0
         or args.td_stream_lyapunov_config)
            and not args.td_stream_dir):
        parser.error(
            "TD-stream Lyapunov recording requires --td-stream-dir"
        )
    if args.decision_snapshot_interval <= 0:
        parser.error("--decision-snapshot-interval must be positive")
    if (args.decision_snapshot_top_m is not None
            and args.decision_snapshot_top_m <= 0):
        parser.error("--decision-snapshot-top-m must be positive")
    if args.candidate_set_guard_mode != "off":
        if args.candidate_set_guard_rolling_window <= 0:
            parser.error("--candidate-set-guard-rolling-window must be positive")
        if args.candidate_set_guard_min_history <= 0:
            parser.error("--candidate-set-guard-min-history must be positive")
        if not (0.0 < args.candidate_set_guard_rolling_quantile < 1.0):
            parser.error(
                "--candidate-set-guard-rolling-quantile must be in (0, 1)"
            )
    if args.margin_substitute_mode != "off":
        if args.margin_substitute_rolling_window <= 0:
            parser.error("--margin-substitute-rolling-window must be positive")
        if args.margin_substitute_min_history <= 0:
            parser.error("--margin-substitute-min-history must be positive")
        if not (0.0 < args.margin_substitute_gap_quantile < 1.0):
            parser.error("--margin-substitute-gap-quantile must be in (0, 1)")
    if args.energy_scoring_mode != "off":
        if args.energy_score_lambda < 0.0:
            parser.error("--energy-score-lambda must be >= 0")
        if not (0.0 <= args.energy_discount <= 1.0):
            parser.error("--energy-discount must be in [0, 1]")
        if args.energy_age_norm <= 0.0:
            parser.error("--energy-age-norm must be > 0")
    if (
        args.energy_conv_random_flip_rate > 0.0
        and args.energy_scoring_mode != "conversion"
    ):
        parser.error(
            "--energy-conv-random-flip-rate requires "
            "--energy-scoring-mode conversion"
        )
    if not (0.0 <= args.energy_conv_random_flip_rate <= 1.0):
        parser.error("--energy-conv-random-flip-rate must be in [0, 1]")
    if args.energy_scoring_mode == "conversion":
        if args.energy_conv_lambda < 0.0:
            parser.error("--energy-conv-lambda must be >= 0")
        if args.energy_gate_window <= 0:
            parser.error("--energy-gate-window must be positive")
        if args.energy_gate_warmup < 0:
            parser.error("--energy-gate-warmup must be >= 0")
        if not (0.0 < args.energy_gate_quantile < 1.0):
            parser.error("--energy-gate-quantile must be in (0, 1)")
    if args.lyapunov_l0_lambda < 0.0:
        parser.error("--lyapunov-l0-lambda must be >= 0")
    l0_weights = (
        args.lyapunov_l0_work_weight,
        args.lyapunov_l0_station_weight,
        args.lyapunov_l0_traffic_weight,
        args.lyapunov_l0_stall_weight,
        args.lyapunov_l0_plan_fail_weight,
        args.lyapunov_l0_arrival_weight,
    )
    if any(value < 0.0 for value in l0_weights):
        parser.error("--lyapunov-l0 component weights must be non-negative")
    if args.lyapunov_l0_reservation_window <= 0:
        parser.error("--lyapunov-l0-reservation-window must be positive")
    if (
        any(edge <= 0 for edge in args.lyapunov_l0_eta_bins)
        or sorted(set(args.lyapunov_l0_eta_bins))
        != args.lyapunov_l0_eta_bins
    ):
        parser.error(
            "--lyapunov-l0-eta-bins must be strictly increasing positives"
        )
    if args.candidate_context_factor < 1.0:
        parser.error("--candidate-context-factor must be >= 1")

    if args.checkpoints:
        ckpts = args.checkpoints
        names = args.names or [f"WorldModel_{i}" for i in range(len(ckpts))]
        if len(names) != len(ckpts):
            parser.error("--names must have same length as --checkpoints")
    elif args.checkpoint:
        ckpts = [args.checkpoint]
        names = ["WorldModel"]
    else:
        parser.error("Must provide --checkpoint or --checkpoints")

    checkpoint_map = dict(zip(names, ckpts))

    config_path = args.config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "Config", "world_model_config.json",
        )

    td_stream_lyapunov_l0_config = {
        "work_weight": args.lyapunov_l0_work_weight,
        "station_weight": args.lyapunov_l0_station_weight,
        "traffic_weight": args.lyapunov_l0_traffic_weight,
        "stall_weight": args.lyapunov_l0_stall_weight,
        "plan_fail_weight": args.lyapunov_l0_plan_fail_weight,
        "arrival_weight": args.lyapunov_l0_arrival_weight,
        "eta_bin_edges": tuple(args.lyapunov_l0_eta_bins),
        "reservation_window": args.lyapunov_l0_reservation_window,
    }
    if args.td_stream_lyapunov_config:
        try:
            with open(args.td_stream_lyapunov_config, "r", encoding="utf-8") as f:
                file_config = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(
                f"cannot load --td-stream-lyapunov-config: {exc}"
            )
        if not isinstance(file_config, dict):
            parser.error("--td-stream-lyapunov-config must contain a JSON object")
        td_stream_lyapunov_l0_config.update(file_config)
    if "eta_bin_edges" in td_stream_lyapunov_l0_config:
        td_stream_lyapunov_l0_config["eta_bin_edges"] = tuple(
            int(value)
            for value in td_stream_lyapunov_l0_config["eta_bin_edges"]
        )
    try:
        from WorldModel.core.lyapunov import LyapunovL0Config
        td_stream_lyapunov_l0_config = asdict(
            LyapunovL0Config(**td_stream_lyapunov_l0_config)
        )
    except (TypeError, ValueError) as exc:
        parser.error(f"invalid Lyapunov L0 configuration: {exc}")
    td_stream_record_lyapunov_l0 = bool(
        args.td_stream_record_lyapunov_l0
        or args.td_stream_lyapunov_config
        or (
            args.td_stream_dir
            and (
                args.lyapunov_l0_mode != "off"
                or args.candidate_robot_mode != "nearest"
                or args.candidate_context_mode != "prefix"
            )
        )
    )

    print("=" * 62)
    print(
        "  Online Evaluation - "
        + ("WorldModel only" if args.world_model_only else "Greedy vs WorldModel")
        + " (v6, fixed-context)"
    )
    print("=" * 62)
    for label, ckpt in checkpoint_map.items():
        print(f"  {label}: {ckpt}")
    long_risk_beta = {}
    if args.long_risk_beta_peak:
        long_risk_beta["peak"] = args.long_risk_beta_peak
    if args.long_risk_beta_cvar:
        long_risk_beta["cvar"] = args.long_risk_beta_cvar
    if args.long_risk_beta_terminal:
        long_risk_beta["terminal"] = args.long_risk_beta_terminal
    if args.long_risk_beta_delta:
        long_risk_beta["delta"] = args.long_risk_beta_delta

    print(f"  Config    : {config_path}")
    print(f"  Seeds     : {args.seeds}")
    print(f"  Ticks     : {args.ticks}")
    print(f"  top_m     : {args.top_m} (training/snapshot only)")
    print("  online robot candidates: ALL currently idle robots")
    print(f"  native NO_ASSIGN candidate: {args.include_no_assign}")
    print(f"  risk_thr  : {args.risk_threshold}")
    print(f"  risk_wt   : {args.risk_weight}")
    if long_risk_beta:
        print(f"  lr_beta   : {long_risk_beta}")
    print(f"  defer    : mode={args.risk_defer_mode}, "
          f"score={args.risk_defer_score}, "
          f"thr={args.risk_defer_threshold}, "
          f"topq={args.risk_defer_topq}, "
          f"rolling_q={args.risk_defer_rolling_quantile}, "
          f"rolling_window={args.risk_defer_rolling_window}, "
          f"rand_rate={args.random_defer_rate}")
    print(f"  injection: weight={args.station_injection_weight}, "
          f"eta_norm={args.station_injection_eta_norm}, "
          f"near_radius={args.station_injection_near_radius}, "
          f"diagnostic_only={args.station_injection_diagnostic_only}")
    print(f"  potential_guard: mode={args.potential_guard_mode}, "
          f"q={args.potential_guard_rolling_quantile}, "
          f"window={args.potential_guard_rolling_window}, "
          f"min_history={args.potential_guard_min_history}")
    print(f"  pool_scoring: mode={args.pool_scoring_mode}, "
          f"lambda={args.pool_scoring_lambda}, "
          f"context_factor={args.pool_scoring_context_factor}")
    if args.decision_trace_jsonl:
        os.makedirs(os.path.dirname(args.decision_trace_jsonl) or ".",
                    exist_ok=True)
        with open(args.decision_trace_jsonl, "w", encoding="utf-8"):
            pass
        print(f"  decision_trace: {args.decision_trace_jsonl}, "
              f"horizons={args.decision_trace_horizons}, "
              f"max_records/run={args.decision_trace_max_records}")
    if args.td_stream_dir:
        os.makedirs(args.td_stream_dir, exist_ok=True)
        print(f"  td_stream: dir={args.td_stream_dir}, "
              f"frame_stride={args.td_stream_frame_stride} "
              f"(all online arms incl. Greedy)")
        if td_stream_record_lyapunov_l0:
            print("  td_stream L0: enabled (analytic labels + productive flow)")
    if args.decision_snapshot_dir:
        os.makedirs(args.decision_snapshot_dir, exist_ok=True)
        print(f"  decision_snapshot: dir={args.decision_snapshot_dir}, "
              f"interval={args.decision_snapshot_interval}, "
              f"top_m={args.decision_snapshot_top_m or args.top_m} "
              f"(inherit --top-m), arms={args.decision_snapshot_arms}")
    print(f"  candidate_set_guard: mode={args.candidate_set_guard_mode}, "
          f"features={args.candidate_set_guard_features}, "
          f"q={args.candidate_set_guard_rolling_quantile}, "
          f"window={args.candidate_set_guard_rolling_window}, "
          f"min_history={args.candidate_set_guard_min_history}")
    print(f"  margin_substitute: mode={args.margin_substitute_mode}, "
          f"gap_q={args.margin_substitute_gap_quantile}, "
          f"window={args.margin_substitute_rolling_window}, "
          f"min_history={args.margin_substitute_min_history}")
    print(f"  energy_scoring: mode={args.energy_scoring_mode}, "
          f"lambda={args.energy_score_lambda}, "
          f"discount={args.energy_discount}, "
          f"potential_form={args.energy_potential_form}, "
          f"drift={args.energy_drift_signal}, "
          f"gate={args.energy_gate_mode}, "
          f"gate_q={args.energy_gate_quantile}, "
          f"conv_lambda={args.energy_conv_lambda}, "
          f"conv_flip_rate={args.energy_conv_random_flip_rate}, "
          f"noop_scale={args.energy_noop_cost_scale}, "
          f"age_norm={args.energy_age_norm}")
    print(f"  energy_prio  : drift_lambda={args.energy_prio_drift_lambda}, "
          f"theta={args.energy_prio_theta}, "
          f"window={args.energy_prio_window}, "
          f"warmup={args.energy_prio_warmup}, "
          f"mu_ref={args.energy_mu_ref}, "
          f"mu=[{args.energy_mu_min}, {args.energy_mu_max}]")
    print(
        "  lyapunov_l0 : "
        f"mode={args.lyapunov_l0_mode}, "
        f"lambda={args.lyapunov_l0_lambda}, "
        f"eta_bins={td_stream_lyapunov_l0_config['eta_bin_edges']}, "
        "robot_pool=all_idle_online, "
        f"collection_mode={args.candidate_robot_mode}, "
        f"context_pool={args.candidate_context_mode}, "
        f"context_factor={args.candidate_context_factor}"
    )
    if args.lyapunov_l0_mode == "additive":
        print(
            "  L0 online term: ETA arrival-barrier preview only; "
            "not learned full H-step Lyapunov drift"
        )
    print()

    per_seed = run_all_seeds(
        config_path=config_path,
        checkpoint_map=checkpoint_map,
        seeds=args.seeds,
        max_ticks=args.ticks,
        top_m=args.top_m,
        risk_threshold=args.risk_threshold,
        risk_weight=args.risk_weight,
        long_risk_beta=long_risk_beta or None,
        risk_defer_mode=args.risk_defer_mode,
        risk_defer_threshold=args.risk_defer_threshold,
        risk_defer_score=args.risk_defer_score,
        risk_defer_weight_risk_max=args.risk_defer_weight_risk_max,
        risk_defer_weight_terminal=args.risk_defer_weight_terminal,
        risk_defer_weight_cvar=args.risk_defer_weight_cvar,
        risk_defer_weight_peak=args.risk_defer_weight_peak,
        risk_defer_weight_delta=args.risk_defer_weight_delta,
        risk_defer_topq=args.risk_defer_topq,
        risk_defer_rolling_window=args.risk_defer_rolling_window,
        risk_defer_rolling_quantile=args.risk_defer_rolling_quantile,
        risk_defer_min_history=args.risk_defer_min_history,
        random_defer_rate=args.random_defer_rate,
        random_defer_seed=args.random_defer_seed,
        station_injection_weight=args.station_injection_weight,
        station_injection_eta_norm=args.station_injection_eta_norm,
        station_injection_near_radius=args.station_injection_near_radius,
        station_injection_diagnostic_only=(
            args.station_injection_diagnostic_only
        ),
        potential_guard_mode=args.potential_guard_mode,
        potential_guard_rolling_window=args.potential_guard_rolling_window,
        potential_guard_rolling_quantile=args.potential_guard_rolling_quantile,
        potential_guard_min_history=args.potential_guard_min_history,
        potential_guard_weight_state=args.potential_guard_weight_state,
        potential_guard_weight_station=args.potential_guard_weight_station,
        potential_guard_weight_injection=args.potential_guard_weight_injection,
        potential_guard_weight_risk_max=args.potential_guard_weight_risk_max,
        potential_guard_weight_terminal=args.potential_guard_weight_terminal,
        potential_guard_weight_cvar=args.potential_guard_weight_cvar,
        potential_guard_weight_peak=args.potential_guard_weight_peak,
        potential_guard_weight_delta=args.potential_guard_weight_delta,
        pool_scoring_mode=args.pool_scoring_mode,
        pool_scoring_lambda=args.pool_scoring_lambda,
        pool_scoring_context_factor=args.pool_scoring_context_factor,
        decision_trace_jsonl=args.decision_trace_jsonl,
        decision_trace_max_records=args.decision_trace_max_records,
        decision_trace_horizons=args.decision_trace_horizons,
        td_stream_dir=args.td_stream_dir,
        td_stream_frame_stride=args.td_stream_frame_stride,
        td_stream_record_lyapunov_l0=td_stream_record_lyapunov_l0,
        td_stream_lyapunov_l0_config=td_stream_lyapunov_l0_config,
        decision_snapshot_dir=args.decision_snapshot_dir,
        decision_snapshot_interval=args.decision_snapshot_interval,
        decision_snapshot_top_m=args.decision_snapshot_top_m,
        decision_snapshot_arms=args.decision_snapshot_arms,
        candidate_set_guard_mode=args.candidate_set_guard_mode,
        candidate_set_guard_features=args.candidate_set_guard_features,
        candidate_set_guard_rolling_window=(
            args.candidate_set_guard_rolling_window
        ),
        candidate_set_guard_rolling_quantile=(
            args.candidate_set_guard_rolling_quantile
        ),
        candidate_set_guard_min_history=args.candidate_set_guard_min_history,
        margin_substitute_mode=args.margin_substitute_mode,
        margin_substitute_rolling_window=args.margin_substitute_rolling_window,
        margin_substitute_gap_quantile=args.margin_substitute_gap_quantile,
        margin_substitute_min_history=args.margin_substitute_min_history,
        energy_scoring_mode=args.energy_scoring_mode,
        energy_potential_form=args.energy_potential_form,
        energy_score_lambda=args.energy_score_lambda,
        energy_discount=args.energy_discount,
        energy_drift_signal=args.energy_drift_signal,
        energy_gate_mode=args.energy_gate_mode,
        energy_gate_window=args.energy_gate_window,
        energy_gate_warmup=args.energy_gate_warmup,
        energy_gate_quantile=args.energy_gate_quantile,
        energy_conv_lambda=args.energy_conv_lambda,
        energy_conv_random_flip_rate=args.energy_conv_random_flip_rate,
        energy_conv_random_flip_seed=args.energy_conv_random_flip_seed,
        energy_prio_drift_lambda=args.energy_prio_drift_lambda,
        energy_prio_theta=args.energy_prio_theta,
        energy_prio_window=args.energy_prio_window,
        energy_prio_warmup=args.energy_prio_warmup,
        energy_mu_ref=args.energy_mu_ref,
        energy_mu_min=args.energy_mu_min,
        energy_mu_max=args.energy_mu_max,
        energy_weight_wait=args.energy_weight_wait,
        energy_weight_excess=args.energy_weight_excess,
        energy_weight_station=args.energy_weight_station,
        energy_weight_bottleneck=args.energy_weight_bottleneck,
        energy_weight_severe=args.energy_weight_severe,
        energy_weight_completed=args.energy_weight_completed,
        energy_weight_long_terminal=args.energy_weight_long_terminal,
        energy_weight_long_cvar=args.energy_weight_long_cvar,
        energy_weight_long_peak=args.energy_weight_long_peak,
        energy_weight_long_delta=args.energy_weight_long_delta,
        energy_service_relief_scale=args.energy_service_relief_scale,
        energy_service_weight_order_age=(
            args.energy_service_weight_order_age
        ),
        energy_service_weight_station_pending=(
            args.energy_service_weight_station_pending
        ),
        energy_service_weight_order_size=args.energy_service_weight_order_size,
        energy_noop_base_margin=args.energy_noop_base_margin,
        energy_noop_cost_scale=args.energy_noop_cost_scale,
        energy_noop_weight_order_age=args.energy_noop_weight_order_age,
        energy_noop_weight_station_pending=(
            args.energy_noop_weight_station_pending
        ),
        energy_noop_weight_global_pending=(
            args.energy_noop_weight_global_pending
        ),
        energy_noop_weight_idle=args.energy_noop_weight_idle,
        energy_age_norm=args.energy_age_norm,
        lyapunov_l0_mode=args.lyapunov_l0_mode,
        lyapunov_l0_lambda=args.lyapunov_l0_lambda,
        lyapunov_l0_config=td_stream_lyapunov_l0_config,
        candidate_robot_mode=args.candidate_robot_mode,
        candidate_context_mode=args.candidate_context_mode,
        candidate_context_factor=args.candidate_context_factor,
        candidate_explore_seed=args.candidate_explore_seed,
        include_no_assign_candidate=args.include_no_assign,
        world_model_only=args.world_model_only,
    )

    agg = aggregate_seeds(per_seed)

    comparisons = {}
    if not args.world_model_only:
        for label in names:
            comparisons[label] = compute_comparison(agg, label)

    # ---- Summary ----
    print("\n" + "=" * 62)
    print("  Aggregate Results")
    print("=" * 62)
    for assigner, metrics in agg.items():
        print(f"\n  {assigner}:")
        print(f"    completed_orders : {metrics.get('completed_orders_mean', 0):.1f} "
              f"+/- {metrics.get('completed_orders_std', 0):.1f}")
        print(f"    avg_excess_delay : {metrics.get('avg_excess_delay_mean', 0):.1f} "
              f"+/- {metrics.get('avg_excess_delay_std', 0):.1f}")
        print(f"    wm_label_cost    : {metrics.get('wm_label_cost_mean', 0):.4f} "
              f"+/- {metrics.get('wm_label_cost_std', 0):.4f}")
        print(f"    wait_or_stall    : {metrics.get('wait_or_stall_mean', 0):.4f} "
              f"+/- {metrics.get('wait_or_stall_std', 0):.4f}")
        print(f"    station_pressure : {metrics.get('station_pressure_mean', 0):.4f} "
              f"+/- {metrics.get('station_pressure_std', 0):.4f}")
        print(f"    bottleneck_CVaR  : {metrics.get('bottleneck_CVaR_mean', 0):.4f} "
              f"+/- {metrics.get('bottleneck_CVaR_std', 0):.4f}")
        print(f"    unified_risk     : {metrics.get('unified_risk_mean', 0):.4f} "
              f"+/- {metrics.get('unified_risk_std', 0):.4f}")
        if "lyapunov_l0_state_mean_mean" in metrics:
            print(
                "    lyapunov_l0     : "
                f"state={metrics['lyapunov_l0_state_mean_mean']:.4f} "
                "arrival_delta="
                f"{metrics.get('lyapunov_l0_arrival_delta_mean_mean', 0):.4f} "
                "selected_delta="
                f"{metrics.get('lyapunov_l0_selected_delta_mean_mean', 0):.4f}"
            )
        if "eta_stratified_contexts_mean" in metrics:
            print(
                "    candidate_pool  : "
                "eta_non_nearest/context="
                f"{metrics.get('eta_stratified_non_nearest_per_context_mean', 0):.3f} "
                "context_stratified="
                f"{metrics.get('context_stratified_calls_mean', 0):.1f}"
            )
        if "fallback_greedy_ratio_mean" in metrics:
            print(f"    fallback_ratio   : {metrics['fallback_greedy_ratio_mean']:.4f}")
        if "native_no_assign_selected_ratio_mean" in metrics:
            print(
                "    native_no_assign : "
                f"selected={metrics['native_no_assign_selected_ratio_mean']:.4f} "
                "contexts="
                f"{metrics.get('native_no_assign_contexts_mean', 0):.1f}"
            )
        if "risk_defer_rate_mean" in metrics:
            print(f"    defer_rate      : {metrics['risk_defer_rate_mean']:.4f} "
                  f"+/- {metrics.get('risk_defer_rate_std', 0):.4f}")
        if "station_injection_penalty_mean_mean" in metrics:
            print(f"    inj_penalty     : "
                  f"{metrics['station_injection_penalty_mean_mean']:.4f} "
                  f"+/- {metrics.get('station_injection_penalty_mean_std', 0):.4f}")
            print(f"    inj_selected    : "
                  f"{metrics.get('station_injection_selected_penalty_mean_mean', 0):.4f}")
            if "station_injection_diag_flip_rate_mean" in metrics:
                print(f"    inj_flip        : "
                      f"{metrics['station_injection_diag_flip_rate_mean']:.4f}")
                print(f"    inj_margin/spread: "
                      f"{metrics.get('station_injection_diag_base_margin_mean_mean', 0):.4f} / "
                      f"{metrics.get('station_injection_diag_penalty_spread_mean_mean', 0):.4f}")
        if "potential_guard_rate_mean" in metrics:
            print(f"    guard_rate      : "
                  f"{metrics['potential_guard_rate_mean']:.4f} "
                  f"+/- {metrics.get('potential_guard_rate_std', 0):.4f}")
            print(f"    guard_score     : exec="
                  f"{metrics.get('potential_guard_executed_score_mean_mean', 0):.4f} "
                  f"defer="
                  f"{metrics.get('potential_guard_deferred_score_mean_mean', 0):.4f}")
        if "pool_scoring_selected_ratio_mean" in metrics:
            print(f"    pool_selected   : "
                  f"{metrics['pool_scoring_selected_ratio_mean']:.4f} "
                  f"contexts, cand/context="
                  f"{metrics.get('pool_scoring_candidates_per_context_mean', 0):.2f}")
            print(f"    pool_subst      : robot="
                  f"{metrics.get('pool_scoring_robot_substitution_rate_mean', 0):.4f} "
                  f"order="
                  f"{metrics.get('pool_scoring_order_replacement_rate_mean', 0):.4f} "
                  f"pot_rank="
                  f"{metrics.get('pool_scoring_selected_potential_rank_mean_mean', 0):.4f}")
        if "candidate_set_guard_rate_mean" in metrics:
            print(f"    cand_guard      : "
                  f"{metrics['candidate_set_guard_rate_mean']:.4f} "
                  f"+/- {metrics.get('candidate_set_guard_rate_std', 0):.4f}")
            print(f"    cand_guard_thr  : pot="
                  f"{metrics.get('candidate_set_guard_potential_threshold_mean_mean', 0):.4f} "
                  f"risk="
                  f"{metrics.get('candidate_set_guard_risk_threshold_mean_mean', 0):.4f}")
        if "margin_substitute_rate_mean" in metrics:
            print(f"    margin_subst    : "
                  f"{metrics['margin_substitute_rate_mean']:.4f} "
                  f"opp="
                  f"{metrics.get('margin_substitute_opportunity_rate_mean', 0):.4f} "
                  f"gap_thr="
                  f"{metrics.get('margin_substitute_gap_threshold_mean_mean', 0):.4f}")
        if "energy_scoring_contexts_mean" in metrics:
            print(f"    energy          : pot="
                  f"{metrics.get('energy_potential_mean_mean', 0):.4f} "
                  f"relief="
                  f"{metrics.get('energy_service_relief_mean_mean', 0):.4f} "
                  f"score="
                  f"{metrics.get('energy_score_mean_mean', 0):.4f}")
            if "energy_noop_rate_mean" in metrics:
                print(f"    energy_noop     : "
                      f"{metrics.get('energy_noop_rate_mean', 0):.4f} "
                      f"cost="
                      f"{metrics.get('energy_noop_cost_mean_mean', 0):.4f} "
                      f"noop_score="
                      f"{metrics.get('energy_noop_score_mean_mean', 0):.4f}")
        if "energy_prio_defer_rate_mean" in metrics:
            print(f"    energy_prio    : defer="
                  f"{metrics.get('energy_prio_defer_rate_mean', 0):.4f} "
                  f"mu={metrics.get('energy_prio_mu_t_mean_mean', 0):.4f} "
                  f"warmup="
                  f"{metrics.get('energy_prio_warmup_rate_mean', 0):.4f}")
            print(f"    prio split     : priority exec/defer="
                  f"{metrics.get('energy_prio_priority_executed_mean_mean', 0):.4f}/"
                  f"{metrics.get('energy_prio_priority_deferred_mean_mean', 0):.4f} "
                  f"relief exec/defer="
                  f"{metrics.get('energy_prio_relief_executed_mean_mean', 0):.4f}/"
                  f"{metrics.get('energy_prio_relief_deferred_mean_mean', 0):.4f}")

    for label, comp in comparisons.items():
        print(f"\n  --- {label} vs Greedy ---")
        print(f"  Throughput improvement: {comp['throughput_improvement_mean']:+.2%}")
        print(f"  wm_label_cost diff   : {comp['wm_cost_diff']:+.4f}  (lower=better)")
        if "fallback_greedy_ratio_mean" in comp:
            print(f"  Fallback ratio       : {comp['fallback_greedy_ratio_mean']:.4f}")
        print(f"  Pass basic           : {'YES' if comp['pass_basic'] else 'NO'}")
        print(f"  Pass strong          : {'YES' if comp['pass_strong'] else 'NO'}")
    print("=" * 62)

    # ---- Save JSON ----
    if args.save_json:
        per_seed_serializable = {
            str(k): v for k, v in per_seed.items()
        }
        output = {
            "meta": {
                "checkpoints": checkpoint_map,
                "config": config_path,
                "seeds": args.seeds,
                "ticks": args.ticks,
                "top_m": args.top_m,
                "world_model_only": args.world_model_only,
                "online_robot_candidate_scope": "all_idle",
                "native_no_assign_candidate": bool(
                    args.include_no_assign
                ),
                "risk_threshold": args.risk_threshold,
                "risk_weight": args.risk_weight,
                "long_risk_beta": long_risk_beta or None,
                "risk_defer_note": (
                    "diagnostic trick / risk-aware guard baseline; "
                    "not the final 6.1 method design"
                ),
                "risk_defer_mode": args.risk_defer_mode,
                "risk_defer_threshold": args.risk_defer_threshold,
                "risk_defer_topq": args.risk_defer_topq,
                "risk_defer_rolling_window": args.risk_defer_rolling_window,
                "risk_defer_rolling_quantile": args.risk_defer_rolling_quantile,
                "risk_defer_min_history": args.risk_defer_min_history,
                "risk_defer_score": args.risk_defer_score,
                "risk_defer_weights": {
                    "risk_max": args.risk_defer_weight_risk_max,
                    "terminal": args.risk_defer_weight_terminal,
                    "cvar": args.risk_defer_weight_cvar,
                    "peak": args.risk_defer_weight_peak,
                    "delta": args.risk_defer_weight_delta,
                },
                "random_defer_rate": args.random_defer_rate,
                "random_defer_seed": args.random_defer_seed,
                "station_injection_note": (
                    "6.1 diagnostic additive scoring term; penalizes fast "
                    "injection into high-pressure station regions. It does "
                    "not replace learned cost, bottleneck_CVaR, or long-risk "
                    "beta terms."
                ),
                "station_injection_weight": args.station_injection_weight,
                "station_injection_eta_norm": args.station_injection_eta_norm,
                "station_injection_near_radius": args.station_injection_near_radius,
                "station_injection_diagnostic_only": (
                    args.station_injection_diagnostic_only
                ),
                "potential_guard_note": (
                    "pre-6.1 candidate admission diagnostic; uses rolling "
                    "relative potential ranking, not an absolute value "
                    "threshold"
                ),
                "potential_guard_mode": args.potential_guard_mode,
                "potential_guard_rolling_window": (
                    args.potential_guard_rolling_window
                ),
                "potential_guard_rolling_quantile": (
                    args.potential_guard_rolling_quantile
                ),
                "potential_guard_min_history": args.potential_guard_min_history,
                "potential_guard_weights": {
                    "state": args.potential_guard_weight_state,
                    "station": args.potential_guard_weight_station,
                    "injection": args.potential_guard_weight_injection,
                    "risk_max": args.potential_guard_weight_risk_max,
                    "terminal": args.potential_guard_weight_terminal,
                    "cvar": args.potential_guard_weight_cvar,
                    "peak": args.potential_guard_weight_peak,
                    "delta": args.potential_guard_weight_delta,
                },
                "pool_scoring_note": (
                    "Candidate-pool scoring modes. potential_rank/context_rank/"
                    "energy_score are pre-6.1 diagnostics. energy_conversion "
                    "is S2 cross-context priority plus defer/no-op on top of "
                    "the frozen S1 conversion score. random_defer_matched "
                    "keeps S1 context order and randomly defers contexts at "
                    "the configured matched rate."
                ),
                "pool_scoring_mode": args.pool_scoring_mode,
                "pool_scoring_lambda": args.pool_scoring_lambda,
                "pool_scoring_context_factor": (
                    args.pool_scoring_context_factor
                ),
                "decision_trace_jsonl": args.decision_trace_jsonl,
                "decision_trace_horizons": args.decision_trace_horizons,
                "td_stream_dir": args.td_stream_dir,
                "td_stream_frame_stride": args.td_stream_frame_stride,
                "td_stream_lyapunov_l0": td_stream_record_lyapunov_l0,
                "lyapunov_l0": {
                    "version": LYAPUNOV_SNAPSHOT_SCHEMA_VERSION,
                    "mode": args.lyapunov_l0_mode,
                    "lambda": args.lyapunov_l0_lambda,
                    "config": td_stream_lyapunov_l0_config,
                    "online_robot_candidate_scope": "all_idle",
                    "candidate_robot_mode": args.candidate_robot_mode,
                    "candidate_context_mode": args.candidate_context_mode,
                    "candidate_context_factor": (
                        args.candidate_context_factor
                    ),
                    "candidate_explore_seed": args.candidate_explore_seed,
                    "candidate_delta_semantics": (
                        "eta_arrival_barrier_preview_v1"
                    ),
                    "full_h_step_drift_online": False,
                },
                "decision_snapshot_dir": args.decision_snapshot_dir,
                "decision_snapshot_interval": args.decision_snapshot_interval,
                "decision_snapshot_top_m": (
                    args.decision_snapshot_top_m
                    if args.decision_snapshot_top_m is not None
                    else args.top_m
                ),
                "decision_snapshot_arms": args.decision_snapshot_arms,
                "candidate_set_guard_note": (
                    "adaptive all-bad candidate-set guard; uses rolling "
                    "quantiles of candidate_min_potential and/or "
                    "candidate_min_risk_score, not fixed absolute thresholds"
                ),
                "candidate_set_guard_mode": args.candidate_set_guard_mode,
                "candidate_set_guard_features": (
                    args.candidate_set_guard_features
                ),
                "candidate_set_guard_rolling_window": (
                    args.candidate_set_guard_rolling_window
                ),
                "candidate_set_guard_rolling_quantile": (
                    args.candidate_set_guard_rolling_quantile
                ),
                "candidate_set_guard_min_history": (
                    args.candidate_set_guard_min_history
                ),
                "margin_substitute_note": (
                    "adaptive substitute; only replaces the baseline robot "
                    "when it is not the best-potential candidate and the "
                    "best-potential base-score gap is small relative to the "
                    "recent rolling distribution"
                ),
                "margin_substitute_mode": args.margin_substitute_mode,
                "margin_substitute_rolling_window": (
                    args.margin_substitute_rolling_window
                ),
                "margin_substitute_gap_quantile": (
                    args.margin_substitute_gap_quantile
                ),
                "margin_substitute_min_history": (
                    args.margin_substitute_min_history
                ),
                "energy_scoring_note": (
                    "Phase 6.1 energy-shaped scoring. Candidate actions use "
                    "WorldModel rollout system_preds to compute V_pred(s,a). "
                    "with_noop adds a dynamic no-op candidate whose cost is "
                    "state/context opportunity cost, not a fixed defer "
                    "threshold or budget."
                ),
                "energy_scoring_mode": args.energy_scoring_mode,
                "energy_potential_form": args.energy_potential_form,
                "energy_score_lambda": args.energy_score_lambda,
                "energy_discount": args.energy_discount,
                "energy_conversion": {
                    "drift_signal": args.energy_drift_signal,
                    "long_risk_contract": long_risk_runtime_contract(),
                    "gate_mode": args.energy_gate_mode,
                    "gate_window": args.energy_gate_window,
                    "gate_warmup": args.energy_gate_warmup,
                    "gate_quantile": args.energy_gate_quantile,
                    "lambda": args.energy_conv_lambda,
                    "random_flip_rate": args.energy_conv_random_flip_rate,
                    "random_flip_seed": args.energy_conv_random_flip_seed,
                },
                "energy_priority": {
                    "drift_lambda": args.energy_prio_drift_lambda,
                    "theta": args.energy_prio_theta,
                    "window": args.energy_prio_window,
                    "warmup": args.energy_prio_warmup,
                    "mu_ref": args.energy_mu_ref,
                    "mu_min": args.energy_mu_min,
                    "mu_max": args.energy_mu_max,
                    "matched_random_rate": args.random_defer_rate,
                },
                "energy_weights": {
                    "wait": args.energy_weight_wait,
                    "excess": args.energy_weight_excess,
                    "station": args.energy_weight_station,
                    "bottleneck": args.energy_weight_bottleneck,
                    "severe": args.energy_weight_severe,
                    "completed": args.energy_weight_completed,
                    "long_terminal": args.energy_weight_long_terminal,
                    "long_cvar": args.energy_weight_long_cvar,
                    "long_peak": args.energy_weight_long_peak,
                    "long_delta": args.energy_weight_long_delta,
                },
                "energy_service_relief": {
                    "scale": args.energy_service_relief_scale,
                    "order_age": args.energy_service_weight_order_age,
                    "station_pending": (
                        args.energy_service_weight_station_pending
                    ),
                    "order_size": args.energy_service_weight_order_size,
                },
                "energy_noop": {
                    "base_margin": args.energy_noop_base_margin,
                    "cost_scale": args.energy_noop_cost_scale,
                    "order_age": args.energy_noop_weight_order_age,
                    "station_pending": (
                        args.energy_noop_weight_station_pending
                    ),
                    "global_pending": (
                        args.energy_noop_weight_global_pending
                    ),
                    "idle": args.energy_noop_weight_idle,
                    "age_norm": args.energy_age_norm,
                },
                "order_stream_control": "python_random_np_random_torch_seeded_per_assigner",
            },
            "per_seed": per_seed_serializable,
            "aggregate": agg,
            "comparisons": comparisons,
        }
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  JSON saved: {args.save_json}")


if __name__ == "__main__":
    main()
