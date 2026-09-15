"""
Validate state potential V(s_t) against future online risk.

This is a pre-6.1 diagnostic.  It tests whether observable state
potentials have predictive power for future risk under an online policy
trajectory.  It does not test action-conditioned drift scoring yet.
"""

import argparse
import csv
import json
import math
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from WorldModel.core.costs import CONGESTION_LAMBDAS
from WorldModel.evaluate import _build_engine
from WorldModel.evaluation.evaluate_online_v6 import (
    _extended_sim_metrics,
    reset_global_ids,
    set_global_seed,
)
from WorldModel.graph.graph_builder import build_static_graph, extract_system_labels
from WorldState.risk import compute_unified_risk


SYSTEM_CHANNELS = [
    "wait_or_stall",
    "avg_excess_delay",
    "station_queue_delta",
    "station_load_imbalance",
    "bottleneck_CVaR",
    "completed_orders_delta",
    "unified_risk_label",
]


EXTRA_STATE_CHANNELS = [
    "station_queue_occ_mean",
    "station_queue_occ_max",
    "station_queue_occ_cvar10",
    "station_near_robot_max_pressure",
    "station_near_robot_imbalance",
    "active_delivery_pressure",
    "active_task_pressure",
    "active_delivery_station_max_pressure",
    "active_delivery_station_imbalance",
    "active_delivery_target_dist_mean",
    "active_delivery_target_dist_cvar10",
    "active_delivery_cross_station_ratio",
    "pending_order_pressure",
    "pending_task_pressure",
    "pending_order_station_max_pressure",
    "pending_order_station_imbalance",
    "pending_order_station_max_ratio",
    "pending_order_station_concentration",
    "in_progress_order_pressure",
]


def _safe_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _std(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def _manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _rank_average(values: List[float]) -> List[float]:
    """Average ranks for ties, 0-based."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[indexed[j]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j
    return ranks


def _pearson(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 3 or len(y) != n:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


def _spearman(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3 or len(y) != len(x):
        return None
    return _pearson(_rank_average(x), _rank_average(y))


def _roc_auc(labels: List[int], scores: List[float]) -> Optional[float]:
    if len(labels) != len(scores) or not labels:
        return None
    pos = sum(1 for v in labels if v)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = _rank_average(scores)
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y)
    # Mann-Whitney U with 0-based ranks.
    u = rank_sum_pos - pos * (pos - 1) / 2.0
    return u / (pos * neg)


def _top_tail_mean(values: List[float], tail_frac: float = 0.10) -> float:
    if not values:
        return 0.0
    k = max(1, int(math.ceil(len(values) * tail_frac)))
    return sum(sorted(values, reverse=True)[:k]) / k


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _hinge_square(value: float, threshold: float) -> float:
    excess = max(0.0, float(value) - threshold)
    return excess * excess


def _extract_extra_state_channels(world) -> Dict[str, float]:
    from WorldState.order_state import OrderStatus
    from WorldState.task_state import TaskStatus, TaskType

    num_agents = max(len(world.agents), 1)
    station_positions = dict(world.map_state.station_positions)
    station_ids = sorted(station_positions.keys())
    num_stations = max(len(station_ids), 1)
    station_index = {sid: i for i, sid in enumerate(station_ids)}
    map_rows = len(world.map_state.grid)
    map_cols = len(world.map_state.grid[0]) if map_rows > 0 else 1
    map_diameter = max(map_rows + map_cols - 2, 1)

    station_occ = []
    for sid in station_ids:
        sq = world.station_state.stations.get(sid)
        if sq is not None:
            station_occ.append(float(sq.occupancy()) / max(sq.capacity, 1))
        else:
            station_occ.append(0.0)
    if station_occ:
        station_queue_occ_mean = _mean(station_occ)
        station_queue_occ_max = max(station_occ)
        station_queue_occ_cvar10 = _top_tail_mean(station_occ, 0.10)
    else:
        station_queue_occ_mean = 0.0
        station_queue_occ_max = 0.0
        station_queue_occ_cvar10 = 0.0

    near_robot_counts = [0.0] * num_stations
    near_radius = 4
    for agent in world.agents:
        if not station_positions:
            continue
        nearest_sid, nearest_dist = min(
            ((_sid, _manhattan(agent.position, pos))
             for _sid, pos in station_positions.items()),
            key=lambda x: x[1],
        )
        if nearest_dist <= near_radius:
            near_robot_counts[station_index[nearest_sid]] += 1.0

    station_near_robot_max_pressure = (
        max(near_robot_counts) / num_agents if near_robot_counts else 0.0
    )
    station_near_robot_imbalance = _std(near_robot_counts) / num_agents

    active_delivery = 0
    active_tasks = 0
    pending_tasks = 0
    active_delivery_by_station = [0.0] * num_stations
    active_delivery_target_dist = []
    active_delivery_cross_station = 0
    for task in world.task_state.tasks.values():
        if task.status == TaskStatus.PENDING:
            pending_tasks += 1
        if task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
            active_tasks += 1
            if task.task_type == TaskType.DELIVER:
                active_delivery += 1
                order = world.order_state.orders.get(task.order_id)
                if order is not None and order.station_id in station_index:
                    target_idx = station_index[order.station_id]
                    active_delivery_by_station[target_idx] += 1.0
                    if task.agent_id is not None:
                        agent = world.get_agent(task.agent_id)
                        target_pos = station_positions.get(order.station_id)
                        if target_pos is not None:
                            d = _manhattan(agent.position, target_pos)
                            active_delivery_target_dist.append(
                                min(1.0, float(d) / map_diameter)
                            )
                            if station_positions:
                                nearest_sid, _ = min(
                                    ((_sid, _manhattan(agent.position, pos))
                                     for _sid, pos in station_positions.items()),
                                    key=lambda x: x[1],
                                )
                                if nearest_sid != order.station_id:
                                    active_delivery_cross_station += 1

    pending_orders = 0
    in_progress_orders = 0
    pending_order_by_station = [0.0] * num_stations
    for order in world.order_state.orders.values():
        if order.status == OrderStatus.PENDING:
            pending_orders += 1
            if order.station_id in station_index:
                pending_order_by_station[station_index[order.station_id]] += 1.0
        elif order.status == OrderStatus.IN_PROGRESS:
            in_progress_orders += 1

    active_delivery_station_max_pressure = (
        max(active_delivery_by_station) / num_agents
        if active_delivery_by_station else 0.0
    )
    active_delivery_station_imbalance = (
        _std(active_delivery_by_station) / num_agents
    )
    if active_delivery_target_dist:
        active_delivery_target_dist_mean = _mean(active_delivery_target_dist)
        active_delivery_target_dist_cvar10 = _top_tail_mean(
            active_delivery_target_dist, 0.10
        )
    else:
        active_delivery_target_dist_mean = 0.0
        active_delivery_target_dist_cvar10 = 0.0
    active_delivery_cross_station_ratio = (
        active_delivery_cross_station / max(active_delivery, 1)
    )

    pending_order_station_max = (
        max(pending_order_by_station) if pending_order_by_station else 0.0
    )
    pending_order_station_max_pressure = (
        pending_order_station_max / num_agents
    )
    pending_order_station_imbalance = (
        _std(pending_order_by_station) / num_agents
    )
    pending_order_station_max_ratio = (
        pending_order_station_max / max(sum(pending_order_by_station), 1.0)
    )
    pending_order_station_concentration = (
        pending_order_station_max_pressure * pending_order_station_max_ratio
    )

    return {
        "station_queue_occ_mean": station_queue_occ_mean,
        "station_queue_occ_max": station_queue_occ_max,
        "station_queue_occ_cvar10": station_queue_occ_cvar10,
        "station_near_robot_max_pressure": _clamp01(
            station_near_robot_max_pressure
        ),
        "station_near_robot_imbalance": _clamp01(
            station_near_robot_imbalance
        ),
        "active_delivery_pressure": _clamp01(active_delivery / num_agents),
        "active_task_pressure": _clamp01(active_tasks / num_agents),
        "active_delivery_station_max_pressure": _clamp01(
            active_delivery_station_max_pressure
        ),
        "active_delivery_station_imbalance": _clamp01(
            active_delivery_station_imbalance
        ),
        "active_delivery_target_dist_mean": _clamp01(
            active_delivery_target_dist_mean
        ),
        "active_delivery_target_dist_cvar10": _clamp01(
            active_delivery_target_dist_cvar10
        ),
        "active_delivery_cross_station_ratio": _clamp01(
            active_delivery_cross_station_ratio
        ),
        "pending_order_pressure": _clamp01(pending_orders / num_agents),
        "pending_task_pressure": _clamp01(pending_tasks / num_agents),
        "pending_order_station_max_pressure": _clamp01(
            pending_order_station_max_pressure
        ),
        "pending_order_station_imbalance": _clamp01(
            pending_order_station_imbalance
        ),
        "pending_order_station_max_ratio": _clamp01(
            pending_order_station_max_ratio
        ),
        "pending_order_station_concentration": _clamp01(
            pending_order_station_concentration
        ),
        "in_progress_order_pressure": _clamp01(in_progress_orders / num_agents),
    }


class PotentialTraceProbe:
    """Collect per-tick state labels and risk components."""

    def __init__(self, engine):
        (_, self._node_map, _, self._local_capacity,
         self._bottleneck_score, _, self._adj) = build_static_graph(
            engine.world.map_state
        )
        self._prev_completed = engine.world.order_state.total_completed
        self.records: List[dict] = []

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
        risk = compute_unified_risk(engine.world)
        extra = _extract_extra_state_channels(engine.world)

        label_vals = [_safe_float(v) for v in labels.tolist()]
        rec = {
            "tick": int(engine.world.tick),
            "wait_or_stall": label_vals[0],
            "avg_excess_delay": label_vals[1],
            "station_queue_delta": label_vals[2],
            "station_load_imbalance": label_vals[3],
            "station_pressure": label_vals[2] + label_vals[3],
            "bottleneck_CVaR": label_vals[4],
            "completed_orders_delta": label_vals[5],
            "unified_risk_label": label_vals[6],
            "stall_ratio": float(risk["stall_ratio"]),
            "deadlock_ratio": float(risk["deadlock_ratio"]),
            "handoff_ratio": float(risk["handoff_ratio"]),
            "unified_risk": float(risk["unified_risk"]),
        }
        rec.update(extra)
        rec.update(_compute_potentials(rec))
        self.records.append(rec)

    def summary(self) -> dict:
        if not self.records:
            return {
                "trace_ticks": 0,
                "station_pressure": 0.0,
                "bottleneck_CVaR": 0.0,
                "unified_risk": 0.0,
                "severe_events": 0,
                "congestion_events": 0,
            }
        result = {"trace_ticks": len(self.records)}
        for key in (
            "wait_or_stall",
            "avg_excess_delay",
            "station_pressure",
            "bottleneck_CVaR",
            "unified_risk",
            "stall_ratio",
            "deadlock_ratio",
            "handoff_ratio",
            *EXTRA_STATE_CHANNELS,
        ):
            vals = [r[key] for r in self.records]
            result[f"{key}_mean"] = round(_mean(vals), 6)
            result[f"{key}_max"] = round(max(vals), 6)
        result["completed_orders_delta_sum"] = round(
            sum(r["completed_orders_delta"] for r in self.records), 6
        )
        result["congestion_events"] = sum(
            1 for r in self.records if r["unified_risk"] >= 0.7
        )
        result["severe_events"] = sum(
            1 for r in self.records if r["unified_risk"] >= 1.0
        )
        return result


def _compute_potentials(rec: dict) -> Dict[str, float]:
    """Candidate V(s) definitions from the Phase B / 6.1 notes."""
    lam = CONGESTION_LAMBDAS
    stall_norm = _clamp01(rec["stall_ratio"] / 0.30)
    deadlock_norm = _clamp01(rec["deadlock_ratio"] / 0.10)
    handoff_norm = _clamp01(rec["handoff_ratio"] / 0.30)
    queue_mean = _clamp01(rec["station_queue_occ_mean"])
    queue_cvar = _clamp01(rec["station_queue_occ_cvar10"])
    queue_max = _clamp01(rec["station_queue_occ_max"])
    active_delivery = _clamp01(rec["active_delivery_pressure"])
    active_task = _clamp01(rec["active_task_pressure"])
    backlog = _clamp01(max(
        rec["pending_order_pressure"],
        rec["pending_task_pressure"],
    ))
    in_progress = _clamp01(rec["in_progress_order_pressure"])
    bneck = max(0.0, rec["bottleneck_CVaR"])
    station_pressure_norm = _clamp01(rec["station_pressure"] / 4.0)
    pending_concentration = _clamp01(
        rec["pending_order_station_concentration"]
    )
    active_station_imb = _clamp01(
        rec["active_delivery_station_imbalance"]
    )
    cross_station = _clamp01(rec["active_delivery_cross_station_ratio"])
    target_dist_tail = _clamp01(rec["active_delivery_target_dist_cvar10"])
    near_robot_imb = _clamp01(rec["station_near_robot_imbalance"])

    system_cost_state = (
        lam[0] * rec["wait_or_stall"]
        + lam[1] * rec["avg_excess_delay"]
        + lam[2] * rec["station_queue_delta"]
        + lam[3] * rec["station_load_imbalance"]
        + lam[4] * rec["bottleneck_CVaR"]
        + lam[6] * rec["unified_risk_label"]
    )
    v2_linear_no_bneck = (
        0.18 * queue_mean
        + 0.17 * queue_cvar
        + 0.15 * rec["station_load_imbalance"]
        + 0.15 * handoff_norm
        + 0.15 * stall_norm
        + 0.10 * deadlock_norm
        + 0.05 * active_delivery
        + 0.03 * backlog
        + 0.02 * in_progress
    )
    v2_barrier_no_bneck = (
        0.24 * _hinge_square(queue_mean, 0.55)
        + 0.24 * _hinge_square(queue_cvar, 0.70)
        + 0.10 * _hinge_square(queue_max, 0.85)
        + 0.16 * _hinge_square(handoff_norm, 0.35)
        + 0.16 * _hinge_square(stall_norm, 0.35)
        + 0.06 * _hinge_square(deadlock_norm, 0.20)
        + 0.03 * _hinge_square(active_task, 0.75)
        + 0.01 * _hinge_square(backlog, 0.75)
    )
    v_mid_linear_no_bneck = (
        0.25 * station_pressure_norm
        + 0.18 * handoff_norm
        + 0.18 * stall_norm
        + 0.08 * deadlock_norm
        + 0.09 * rec["pending_order_pressure"]
        + 0.08 * pending_concentration
        + 0.06 * active_station_imb
        + 0.04 * cross_station
        + 0.03 * target_dist_tail
        + 0.01 * near_robot_imb
    )
    v_mid_spatial_no_bneck = (
        0.22 * station_pressure_norm
        + 0.18 * pending_concentration
        + 0.15 * active_station_imb
        + 0.15 * cross_station
        + 0.10 * target_dist_tail
        + 0.10 * near_robot_imb
        + 0.05 * handoff_norm
        + 0.05 * stall_norm
    )
    return {
        # Direct implementation of section 6.1 with current system channels.
        "V_system_cost": system_cost_state,
        "V_system_equal": (
            rec["wait_or_stall"]
            + rec["avg_excess_delay"]
            + rec["station_queue_delta"]
            + rec["station_load_imbalance"]
            + rec["bottleneck_CVaR"]
            + rec["unified_risk_label"]
        ),
        # Phase-B energy interpretation: station pressure + bottleneck +
        # risk/stall/handoff pressure, with throughput excluded.
        "V_phaseB_energy": (
            rec["station_pressure"]
            + rec["bottleneck_CVaR"]
            + rec["unified_risk"]
            + rec["stall_ratio"]
            + rec["handoff_ratio"]
        ),
        "V_phaseB_compact": (
            rec["station_pressure"]
            + rec["bottleneck_CVaR"]
            + rec["unified_risk"]
        ),
        # Ablations used to tell whether the signal is dominated by one part.
        "V_station_only": rec["station_pressure"],
        "V_bneck_only": rec["bottleneck_CVaR"],
        "V_risk_only": rec["unified_risk"],
        "V_stall_handoff": (
            rec["stall_ratio"] + rec["deadlock_ratio"] + rec["handoff_ratio"]
        ),
        # V2 tests whether explicit station queue saturation and operational
        # pressures explain mid-load risk better than bottleneck_CVaR alone.
        "V2_linear": v2_linear_no_bneck + 0.02 * bneck,
        "V2_linear_no_bneck": v2_linear_no_bneck,
        # Barrier potential: safe region is lightly penalized; pressure near
        # capacity grows superlinearly.  This keeps high productive station
        # activity from being treated as automatically bad.
        "V2_barrier": v2_barrier_no_bneck + 0.02 * _hinge_square(bneck, 0.15),
        "V2_barrier_no_bneck": v2_barrier_no_bneck,
        # Mid-load hypothesis potentials.  These test whether the failure mode
        # is better explained by station-specific order concentration and
        # spatial mismatch than by static aisle bottlenecks.
        "V_mid_linear": v_mid_linear_no_bneck + 0.01 * bneck,
        "V_mid_linear_no_bneck": v_mid_linear_no_bneck,
        "V_mid_spatial": v_mid_spatial_no_bneck + 0.01 * bneck,
        "V_mid_spatial_no_bneck": v_mid_spatial_no_bneck,
    }


POTENTIAL_NAMES = [
    "V_system_cost",
    "V_system_equal",
    "V_phaseB_energy",
    "V_phaseB_compact",
    "V_station_only",
    "V_bneck_only",
    "V_risk_only",
    "V_stall_handoff",
    "V2_linear",
    "V2_linear_no_bneck",
    "V2_barrier",
    "V2_barrier_no_bneck",
    "V_mid_linear",
    "V_mid_linear_no_bneck",
    "V_mid_spatial",
    "V_mid_spatial_no_bneck",
]


def _future_targets(
    records: List[dict],
    idx: int,
    window: int,
    event_thresholds: Tuple[float, float],
) -> dict:
    future = records[idx + 1: idx + 1 + window]
    risk_vals = [r["unified_risk"] for r in future]
    label_risk_vals = [r["unified_risk_label"] for r in future]
    station_vals = [r["station_pressure"] for r in future]
    bneck_vals = [r["bottleneck_CVaR"] for r in future]
    completed_vals = [r["completed_orders_delta"] for r in future]
    t0, t1 = event_thresholds
    return {
        "future_unified_max": max(risk_vals),
        "future_unified_mean": _mean(risk_vals),
        "future_unified_cvar10": _top_tail_mean(risk_vals, 0.10),
        "future_label_risk_max": max(label_risk_vals),
        "future_station_pressure_max": max(station_vals),
        "future_station_pressure_mean": _mean(station_vals),
        "future_bneck_max": max(bneck_vals),
        "future_bneck_mean": _mean(bneck_vals),
        "future_completed_sum": sum(completed_vals),
        f"future_event_ge_{t0:g}": int(max(risk_vals) >= t0),
        f"future_event_ge_{t1:g}": int(max(risk_vals) >= t1),
    }


def _top_quantile_stats(
    scores: List[float],
    target: List[float],
    event: List[int],
    frac: float,
) -> dict:
    if not scores:
        return {}
    n = len(scores)
    k = max(1, int(math.ceil(n * frac)))
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    top_idx = order[:k]
    base_mean = _mean(target)
    top_mean = _mean(target[i] for i in top_idx)
    base_event = _mean(event)
    top_event = _mean(event[i] for i in top_idx)
    return {
        f"top{int(frac * 100)}_n": k,
        f"top{int(frac * 100)}_future_unified_max_mean": top_mean,
        f"top{int(frac * 100)}_future_unified_max_lift": (
            top_mean / base_mean if base_mean > 1e-12 else None
        ),
        f"top{int(frac * 100)}_event_rate": top_event,
        f"top{int(frac * 100)}_event_lift": (
            top_event / base_event if base_event > 1e-12 else None
        ),
    }


def _bin_stats(
    scores: List[float],
    target: List[float],
    event: List[int],
    bins: int = 5,
) -> List[dict]:
    if not scores:
        return []
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    out = []
    for b in range(bins):
        lo = int(n * b / bins)
        hi = int(n * (b + 1) / bins)
        idx = order[lo:hi]
        if not idx:
            continue
        out.append({
            "bin": b + 1,
            "n": len(idx),
            "score_min": min(scores[i] for i in idx),
            "score_max": max(scores[i] for i in idx),
            "future_unified_max_mean": _mean(target[i] for i in idx),
            "event_rate": _mean(event[i] for i in idx),
        })
    return out


def analyze_records(
    records: List[dict],
    windows: List[int],
    warmup_ticks: int = 0,
    event_thresholds: Tuple[float, float] = (0.7, 1.0),
    top_fracs: Tuple[float, ...] = (0.10, 0.20),
) -> Dict[str, dict]:
    return analyze_traces(
        [records],
        windows=windows,
        warmup_ticks=warmup_ticks,
        event_thresholds=event_thresholds,
        top_fracs=top_fracs,
    )


def analyze_traces(
    traces: List[List[dict]],
    windows: List[int],
    warmup_ticks: int = 0,
    event_thresholds: Tuple[float, float] = (0.7, 1.0),
    top_fracs: Tuple[float, ...] = (0.10, 0.20),
) -> Dict[str, dict]:
    analysis = {}
    if not traces:
        return analysis
    t0, t1 = event_thresholds
    event_key_0 = f"future_event_ge_{t0:g}"
    event_key_1 = f"future_event_ge_{t1:g}"

    for window in windows:
        rows = []
        for records in traces:
            for i, rec in enumerate(records):
                if rec["tick"] < warmup_ticks:
                    continue
                if i + window >= len(records):
                    continue
                target = _future_targets(records, i, window, event_thresholds)
                rows.append((rec, target))

        window_result = {
            "n": len(rows),
            "warmup_ticks": warmup_ticks,
            "event_thresholds": list(event_thresholds),
            "potentials": {},
        }
        if not rows:
            analysis[str(window)] = window_result
            continue

        target_unified_max = [t["future_unified_max"] for _, t in rows]
        target_event0 = [t[event_key_0] for _, t in rows]
        target_event1 = [t[event_key_1] for _, t in rows]

        for name in POTENTIAL_NAMES:
            scores = [r[name] for r, _ in rows]
            metrics = {
                "spearman_future_unified_max": _spearman(
                    scores, target_unified_max
                ),
                "spearman_future_unified_mean": _spearman(
                    scores, [t["future_unified_mean"] for _, t in rows]
                ),
                "spearman_future_unified_cvar10": _spearman(
                    scores, [t["future_unified_cvar10"] for _, t in rows]
                ),
                "spearman_future_station_pressure_max": _spearman(
                    scores, [t["future_station_pressure_max"] for _, t in rows]
                ),
                "spearman_future_bneck_max": _spearman(
                    scores, [t["future_bneck_max"] for _, t in rows]
                ),
                "auc_future_event_0": _roc_auc(target_event0, scores),
                "auc_future_event_1": _roc_auc(target_event1, scores),
                "event_rate_0": _mean(target_event0),
                "event_rate_1": _mean(target_event1),
                "score_mean": _mean(scores),
                "score_std": float(np.std(scores)) if scores else 0.0,
                "future_unified_max_mean": _mean(target_unified_max),
            }
            for frac in top_fracs:
                metrics.update(
                    _top_quantile_stats(
                        scores, target_unified_max, target_event0, frac
                    )
                )
            metrics["bins_by_score"] = _bin_stats(
                scores, target_unified_max, target_event0, bins=5
            )
            window_result["potentials"][name] = metrics
        analysis[str(window)] = window_result
    return analysis


def _run_one_trace(config_path: str, ta, seed: int, max_ticks: int) -> dict:
    from Config.config_loader import load_config

    set_global_seed(seed)
    reset_global_ids()

    cfg = load_config(config_path)
    cfg.simulation.max_ticks = max_ticks
    cfg.simulation.seed = seed

    t0 = time.time()
    engine = _build_engine(cfg, task_assigner=ta)
    probe = PotentialTraceProbe(engine)
    engine.on_tick_callbacks.append(probe.on_tick)
    engine.run()
    elapsed = time.time() - t0

    metrics = _extended_sim_metrics(engine, ta, elapsed)
    metrics.update(probe.summary())
    return {
        "metrics": metrics,
        "records": probe.records,
    }


def _build_assigners(
    checkpoints: Dict[str, str],
    include_greedy: bool,
    top_m: int,
    risk_threshold: Optional[float],
    risk_weight: Optional[float],
    long_risk_beta: Optional[Dict[str, float]],
) -> Dict[str, object]:
    from Policies.TaskAssigner import GreedyTaskAssigner, WorldModelTaskAssigner

    assigners: Dict[str, object] = {}
    if include_greedy:
        assigners["Greedy"] = GreedyTaskAssigner()
    for name, ckpt in checkpoints.items():
        assigners[name] = WorldModelTaskAssigner(
            checkpoint_path=ckpt,
            top_m=top_m,
            risk_threshold=risk_threshold,
            risk_weight=risk_weight,
            long_risk_beta=long_risk_beta,
        )
    return assigners


def run_validation(
    config_path: str,
    checkpoints: Dict[str, str],
    seeds: List[int],
    ticks: int,
    windows: List[int],
    include_greedy: bool = True,
    top_m: int = 5,
    risk_threshold: Optional[float] = None,
    risk_weight: Optional[float] = None,
    long_risk_beta: Optional[Dict[str, float]] = None,
    warmup_ticks: int = 0,
    event_thresholds: Tuple[float, float] = (0.7, 1.0),
    save_traces: bool = False,
) -> dict:
    per_seed = {}
    aggregate_traces: Dict[str, List[List[dict]]] = {}
    aggregate_metrics: Dict[str, Dict[str, List[float]]] = {}

    for seed in seeds:
        print(f"  --- Seed {seed} ---")
        per_seed[str(seed)] = {}
        assigners = _build_assigners(
            checkpoints=checkpoints,
            include_greedy=include_greedy,
            top_m=top_m,
            risk_threshold=risk_threshold,
            risk_weight=risk_weight,
            long_risk_beta=long_risk_beta,
        )
        for name, ta in assigners.items():
            result = _run_one_trace(config_path, ta, seed, ticks)
            metrics = result["metrics"]
            records = result["records"]
            analysis = analyze_records(
                records,
                windows=windows,
                warmup_ticks=warmup_ticks,
                event_thresholds=event_thresholds,
            )
            per_seed[str(seed)][name] = {
                "metrics": metrics,
                "analysis": analysis,
            }
            if save_traces:
                per_seed[str(seed)][name]["records"] = records

            aggregate_traces.setdefault(name, []).append(records)
            metric_bucket = aggregate_metrics.setdefault(name, {})
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    metric_bucket.setdefault(k, []).append(float(v))

            print(
                f"    {name:16s} orders={metrics.get('completed_orders', 0):4} "
                f"station={metrics.get('station_pressure_mean', 0):.4f} "
                f"bneck={metrics.get('bottleneck_CVaR_mean', 0):.4f} "
                f"risk={metrics.get('unified_risk_mean', 0):.4f} "
                f"severe={metrics.get('severe_events', 0)}"
            )

    aggregate = {}
    for name, traces in aggregate_traces.items():
        metric_summary = {}
        for k, vals in aggregate_metrics.get(name, {}).items():
            if vals:
                metric_summary[f"{k}_mean"] = round(_mean(vals), 6)
                metric_summary[f"{k}_std"] = round(float(np.std(vals)), 6)
        aggregate[name] = {
            "metrics": metric_summary,
            "analysis": analyze_traces(
                traces,
                windows=windows,
                warmup_ticks=warmup_ticks,
                event_thresholds=event_thresholds,
            ),
        }

    return {
        "meta": {
            "config": config_path,
            "checkpoints": checkpoints,
            "include_greedy": include_greedy,
            "seeds": seeds,
            "ticks": ticks,
            "windows": windows,
            "warmup_ticks": warmup_ticks,
            "event_thresholds": list(event_thresholds),
            "top_m": top_m,
            "risk_threshold": risk_threshold,
            "risk_weight": risk_weight,
            "long_risk_beta": long_risk_beta,
            "potential_note": (
                "V(s_t) uses observable current-state labels/risk components. "
                "It deliberately excludes action-conditioned long_risk_head "
                "predictions; those belong to later Delta-V scoring."
            ),
            "potentials": POTENTIAL_NAMES,
            "system_channels": SYSTEM_CHANNELS,
            "extra_state_channels": EXTRA_STATE_CHANNELS,
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def save_summary_csv(result: dict, path: str):
    rows = []
    for assigner, payload in result["aggregate"].items():
        analysis = payload.get("analysis", {})
        for window, window_result in analysis.items():
            for potential, metrics in window_result.get("potentials", {}).items():
                row = {
                    "assigner": assigner,
                    "window": window,
                    "potential": potential,
                    "n": window_result.get("n", 0),
                }
                for key, value in metrics.items():
                    if key == "bins_by_score":
                        continue
                    row[key] = value
                rows.append(row)

    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    leading = ["assigner", "window", "potential", "n"]
    fieldnames = leading + [k for k in fieldnames if k not in leading]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_json_sanitize(rows))


def _parse_long_risk_beta(args) -> Optional[Dict[str, float]]:
    beta = {}
    if args.long_risk_beta_peak:
        beta["peak"] = args.long_risk_beta_peak
    if args.long_risk_beta_cvar:
        beta["cvar"] = args.long_risk_beta_cvar
    if args.long_risk_beta_terminal:
        beta["terminal"] = args.long_risk_beta_terminal
    if args.long_risk_beta_delta:
        beta["delta"] = args.long_risk_beta_delta
    return beta or None


def main():
    parser = argparse.ArgumentParser(
        description="Validate V(s_t) against future risk windows."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None)
    parser.add_argument("--names", type=str, nargs="+", default=None)
    parser.add_argument("--no-greedy", action="store_true",
                        help="Do not include the Greedy baseline trace")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[101, 102])
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--windows", type=int, nargs="+",
                        default=[10, 50, 100, 200])
    parser.add_argument("--warmup-ticks", type=int, default=50)
    parser.add_argument("--event-thresholds", type=float, nargs=2,
                        default=[0.7, 1.0])
    parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument("--risk-threshold", type=float, default=None)
    parser.add_argument("--risk-weight", type=float, default=None)
    parser.add_argument("--long-risk-beta-peak", type=float, default=0.0)
    parser.add_argument("--long-risk-beta-cvar", type=float, default=0.0)
    parser.add_argument("--long-risk-beta-terminal", type=float, default=0.0)
    parser.add_argument("--long-risk-beta-delta", type=float, default=0.0)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--save-csv", type=str, default=None)
    parser.add_argument("--save-traces", action="store_true",
                        help="Include per-tick records in the JSON")
    args = parser.parse_args()

    if args.checkpoints:
        ckpts = args.checkpoints
        names = args.names or [f"WorldModel_{i}" for i in range(len(ckpts))]
        if len(names) != len(ckpts):
            parser.error("--names must have same length as --checkpoints")
    elif args.checkpoint:
        ckpts = [args.checkpoint]
        names = ["WorldModel"]
    else:
        ckpts = []
        names = []
        if args.no_greedy:
            parser.error("Provide a checkpoint or enable Greedy")

    checkpoint_map = dict(zip(names, ckpts))
    config_path = args.config
    if config_path is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        config_path = os.path.join(root, "Config", "world_model_config.json")

    long_risk_beta = _parse_long_risk_beta(args)

    print("=" * 70)
    print("  Potential V(s_t) vs Future Risk Validation")
    print("=" * 70)
    print(f"  Config    : {config_path}")
    print(f"  Seeds     : {args.seeds}")
    print(f"  Ticks     : {args.ticks}")
    print(f"  Windows   : {args.windows}")
    print(f"  Warmup    : {args.warmup_ticks}")
    print(f"  Events    : >= {args.event_thresholds}")
    print(f"  Greedy    : {not args.no_greedy}")
    for name, ckpt in checkpoint_map.items():
        print(f"  {name:10s}: {ckpt}")
    if long_risk_beta:
        print(f"  LR beta   : {long_risk_beta}")
    print()

    result = run_validation(
        config_path=config_path,
        checkpoints=checkpoint_map,
        seeds=args.seeds,
        ticks=args.ticks,
        windows=args.windows,
        include_greedy=not args.no_greedy,
        top_m=args.top_m,
        risk_threshold=args.risk_threshold,
        risk_weight=args.risk_weight,
        long_risk_beta=long_risk_beta,
        warmup_ticks=args.warmup_ticks,
        event_thresholds=(args.event_thresholds[0], args.event_thresholds[1]),
        save_traces=args.save_traces,
    )

    print("\n" + "=" * 70)
    print("  Aggregate Signal Check")
    print("=" * 70)
    for assigner, payload in result["aggregate"].items():
        print(f"\n  {assigner}:")
        for window in args.windows:
            win = payload["analysis"].get(str(window), {})
            potentials = win.get("potentials", {})
            ranked = []
            for name, metrics in potentials.items():
                sp = metrics.get("spearman_future_unified_max")
                auc = metrics.get("auc_future_event_0")
                lift = metrics.get("top10_future_unified_max_lift")
                ranked.append((
                    -999.0 if sp is None else sp,
                    name,
                    auc,
                    lift,
                ))
            ranked.sort(reverse=True)
            if ranked:
                sp, name, auc, lift = ranked[0]
                sp_txt = "NA" if sp == -999.0 else f"{sp:.3f}"
                auc_txt = "NA" if auc is None else f"{auc:.3f}"
                lift_txt = "NA" if lift is None else f"{lift:.3f}"
                print(
                    f"    W={window:3d} best={name:18s} "
                    f"spearman={sp_txt} auc@event0={auc_txt} "
                    f"top10_lift={lift_txt}"
                )
    print("=" * 70)

    result = _json_sanitize(result)
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  JSON saved: {args.save_json}")
    if args.save_csv:
        save_summary_csv(result, args.save_csv)
        print(f"  CSV saved : {args.save_csv}")


if __name__ == "__main__":
    main()
