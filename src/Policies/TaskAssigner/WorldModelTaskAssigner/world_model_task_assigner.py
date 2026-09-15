"""World-Model robot-ranking task assigner.

For each valid fixed order/pod/station context, the online policy evaluates
*all currently idle robots* with the World Model and selects by predicted
score.  Training-time ``top_m`` sampling is not an online action-space limit.
No Greedy/Hungarian robot choice or fallback is allowed in this class.
"""

import hashlib
import json
import math
import os
import random
import time
from collections import deque
from typing import Dict, List, Optional, Set

import torch

from Policies.TaskAssigner.base_task_assigner import (
    AssignmentContext,
    BaseTaskAssigner,
)
from Policies.TaskAssigner.context_assignment import (
    commit_fixed_context_assignments,
    enumerate_pending_assignment_contexts,
    propose_fixed_assignment_contexts,
)
from WorldModel.core.long_risk_schema import (
    LONG_RISK_DRIFT_SIGNALS,
    LONG_RISK_SCHEMA_VERSION,
    decode_long_risk_predictions,
    long_risk_drift_signal_key,
)
from WorldState.task_state import Task


def _manhattan_distance(a, b) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


class WorldModelTaskAssigner(BaseTaskAssigner):
    """Rank every idle robot for each supplied fixed assignment context."""

    requires_external_fixed_context = True

    def __init__(
        self,
        checkpoint_path: str = "DataGen/wm_checkpoints/world_model.pt",
        top_m: int = 5,
        hidden_dim: int = 64,
        reservation_window: int = 1,
        action_path_mode: int = 0,
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
        decision_trace_enabled: bool = False,
        decision_trace_max_records: int = 200000,
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
        energy_conv_random_flip_seed: int = 0,
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
        lyapunov_l0_work_weight: float = 1.0,
        lyapunov_l0_station_weight: float = 1.0,
        lyapunov_l0_traffic_weight: float = 1.0,
        lyapunov_l0_stall_weight: float = 1.0,
        lyapunov_l0_plan_fail_weight: float = 0.5,
        lyapunov_l0_arrival_weight: float = 1.0,
        lyapunov_l0_eta_bins: tuple = (10, 25, 50),
        lyapunov_l0_config: Optional[Dict[str, object]] = None,
        candidate_robot_mode: str = "nearest",
        candidate_context_mode: str = "prefix",
        candidate_context_factor: float = 2.0,
        candidate_explore_seed: int = 0,
        include_no_assign_candidate: bool = False,
        dispatch_potential_mode: str = "off",
        work_drift_mode: str = "off",
        work_drift_head_path: Optional[str] = None,
        work_drift_layer4_report_path: Optional[str] = None,
        work_drift_lambda: float = 0.25,
        work_drift_layer5_evaluation: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.top_m = top_m
        self.hidden_dim = hidden_dim
        self.reservation_window = reservation_window
        self.action_path_mode = action_path_mode
        self.risk_threshold = risk_threshold
        self.risk_weight = risk_weight
        self.long_risk_beta = long_risk_beta or {}
        valid_defer_modes = (
            "off", "risk", "random", "risk_topq", "random_topq",
            "risk_rolling", "random_rolling",
        )
        if risk_defer_mode not in valid_defer_modes:
            raise ValueError(
                "risk_defer_mode must be one of: "
                + ", ".join(valid_defer_modes)
            )
        if risk_defer_score not in ("risk_max", "long_risk", "combined"):
            raise ValueError(
                "risk_defer_score must be one of: risk_max, long_risk, combined"
            )
        self.risk_defer_mode = risk_defer_mode
        self.risk_defer_threshold = risk_defer_threshold
        self.risk_defer_score = risk_defer_score
        self.risk_defer_weight_risk_max = risk_defer_weight_risk_max
        self.risk_defer_weight_terminal = risk_defer_weight_terminal
        self.risk_defer_weight_cvar = risk_defer_weight_cvar
        self.risk_defer_weight_peak = risk_defer_weight_peak
        self.risk_defer_weight_delta = risk_defer_weight_delta
        self.risk_defer_topq = max(0.0, min(1.0, risk_defer_topq))
        self.risk_defer_rolling_window = max(1, int(risk_defer_rolling_window))
        self.risk_defer_rolling_quantile = max(
            0.0, min(1.0, risk_defer_rolling_quantile)
        )
        self.risk_defer_min_history = max(1, int(risk_defer_min_history))
        self.random_defer_rate = max(0.0, min(1.0, random_defer_rate))
        self.random_defer_seed = random_defer_seed
        self._random_defer_rng = random.Random(random_defer_seed)
        self._risk_defer_history = deque(maxlen=self.risk_defer_rolling_window)
        self.station_injection_weight = max(0.0, float(station_injection_weight))
        self.station_injection_eta_norm = max(
            1e-6, float(station_injection_eta_norm)
        )
        self.station_injection_near_radius = max(
            0, int(station_injection_near_radius)
        )
        self.station_injection_diagnostic_only = bool(
            station_injection_diagnostic_only
        )
        valid_potential_guard_modes = ("off", "rolling", "random_rolling")
        if potential_guard_mode not in valid_potential_guard_modes:
            raise ValueError(
                "potential_guard_mode must be one of: "
                + ", ".join(valid_potential_guard_modes)
            )
        self.potential_guard_mode = potential_guard_mode
        self.potential_guard_rolling_window = max(
            1, int(potential_guard_rolling_window)
        )
        self.potential_guard_rolling_quantile = max(
            0.0, min(1.0, potential_guard_rolling_quantile)
        )
        self.potential_guard_min_history = max(
            1, int(potential_guard_min_history)
        )
        self.potential_guard_weight_state = float(potential_guard_weight_state)
        self.potential_guard_weight_station = float(potential_guard_weight_station)
        self.potential_guard_weight_injection = float(
            potential_guard_weight_injection
        )
        self.potential_guard_weight_risk_max = float(
            potential_guard_weight_risk_max
        )
        self.potential_guard_weight_terminal = float(
            potential_guard_weight_terminal
        )
        self.potential_guard_weight_cvar = float(potential_guard_weight_cvar)
        self.potential_guard_weight_peak = float(potential_guard_weight_peak)
        self.potential_guard_weight_delta = float(potential_guard_weight_delta)
        self._potential_guard_history = deque(
            maxlen=self.potential_guard_rolling_window
        )
        valid_pool_scoring_modes = (
            "off", "potential_rank", "context_rank", "energy_score",
            "energy_conversion", "random_defer_matched",
        )
        if pool_scoring_mode not in valid_pool_scoring_modes:
            raise ValueError(
                "pool_scoring_mode must be one of: "
                + ", ".join(valid_pool_scoring_modes)
            )
        self.pool_scoring_mode = pool_scoring_mode
        self.pool_scoring_lambda = max(0.0, float(pool_scoring_lambda))
        self.pool_scoring_context_factor = max(
            1.0, float(pool_scoring_context_factor)
        )
        self.decision_trace_enabled = bool(decision_trace_enabled)
        self.decision_trace_max_records = max(
            0, int(decision_trace_max_records)
        )
        self.decision_trace_records: List[dict] = []
        valid_candidate_guard_modes = ("off", "rolling", "random_rolling")
        if candidate_set_guard_mode not in valid_candidate_guard_modes:
            raise ValueError(
                "candidate_set_guard_mode must be one of: "
                + ", ".join(valid_candidate_guard_modes)
            )
        if candidate_set_guard_features not in ("either", "potential", "risk"):
            raise ValueError(
                "candidate_set_guard_features must be one of: "
                "either, potential, risk"
            )
        self.candidate_set_guard_mode = candidate_set_guard_mode
        self.candidate_set_guard_features = candidate_set_guard_features
        self.candidate_set_guard_rolling_window = max(
            1, int(candidate_set_guard_rolling_window)
        )
        self.candidate_set_guard_rolling_quantile = max(
            0.0, min(1.0, candidate_set_guard_rolling_quantile)
        )
        self.candidate_set_guard_min_history = max(
            1, int(candidate_set_guard_min_history)
        )
        self._candidate_set_guard_potential_history = deque(
            maxlen=self.candidate_set_guard_rolling_window
        )
        self._candidate_set_guard_risk_history = deque(
            maxlen=self.candidate_set_guard_rolling_window
        )
        valid_margin_substitute_modes = ("off", "rolling")
        if margin_substitute_mode not in valid_margin_substitute_modes:
            raise ValueError(
                "margin_substitute_mode must be one of: "
                + ", ".join(valid_margin_substitute_modes)
            )
        self.margin_substitute_mode = margin_substitute_mode
        self.margin_substitute_rolling_window = max(
            1, int(margin_substitute_rolling_window)
        )
        self.margin_substitute_gap_quantile = max(
            0.0, min(1.0, margin_substitute_gap_quantile)
        )
        self.margin_substitute_min_history = max(
            1, int(margin_substitute_min_history)
        )
        self._margin_substitute_gap_history = deque(
            maxlen=self.margin_substitute_rolling_window
        )
        valid_energy_modes = ("off", "additive", "with_noop", "conversion")
        if energy_scoring_mode not in valid_energy_modes:
            raise ValueError(
                "energy_scoring_mode must be one of: "
                + ", ".join(valid_energy_modes)
            )
        if energy_potential_form not in ("discounted", "endpoint"):
            raise ValueError(
                "energy_potential_form must be one of: discounted, endpoint"
            )
        if energy_drift_signal not in LONG_RISK_DRIFT_SIGNALS:
            raise ValueError(
                "energy_drift_signal must be one of: "
                + ", ".join(LONG_RISK_DRIFT_SIGNALS)
            )
        if energy_gate_mode not in ("sigmoid", "hard", "off"):
            raise ValueError("energy_gate_mode must be one of: sigmoid, hard, off")
        self.energy_scoring_mode = energy_scoring_mode
        self.energy_potential_form = energy_potential_form
        self.energy_score_lambda = float(energy_score_lambda)
        self.energy_discount = max(0.0, min(1.0, float(energy_discount)))
        self.energy_drift_signal = energy_drift_signal
        self.long_risk_schema_version = LONG_RISK_SCHEMA_VERSION
        self.energy_gate_mode = energy_gate_mode
        self.energy_gate_window = max(1, int(energy_gate_window))
        self.energy_gate_warmup = max(0, int(energy_gate_warmup))
        self.energy_gate_quantile = max(0.0, min(1.0, float(energy_gate_quantile)))
        self.energy_conv_lambda = max(0.0, float(energy_conv_lambda))
        self.energy_conv_random_flip_rate = max(
            0.0, min(1.0, float(energy_conv_random_flip_rate))
        )
        self.energy_conv_random_flip_seed = int(energy_conv_random_flip_seed)
        self._energy_conv_flip_rng = random.Random(energy_conv_random_flip_seed)
        self._energy_gate_vstate_history = deque(maxlen=self.energy_gate_window)
        if (
            self.pool_scoring_mode in ("energy_conversion", "random_defer_matched")
            and self.energy_scoring_mode != "conversion"
        ):
            raise ValueError(
                "pool_scoring_mode energy_conversion/random_defer_matched "
                "requires energy_scoring_mode='conversion'"
            )
        self.energy_prio_drift_lambda = max(
            0.0, float(energy_prio_drift_lambda)
        )
        self.energy_prio_theta = max(0.0, float(energy_prio_theta))
        self.energy_prio_window = max(1, int(energy_prio_window))
        self.energy_prio_warmup = max(0, int(energy_prio_warmup))
        self.energy_mu_ref = max(1e-6, float(energy_mu_ref))
        self.energy_mu_min = float(energy_mu_min)
        self.energy_mu_max = float(energy_mu_max)
        if self.energy_mu_min > self.energy_mu_max:
            self.energy_mu_min, self.energy_mu_max = (
                self.energy_mu_max, self.energy_mu_min
            )
        self._prio_relief_history = deque(maxlen=self.energy_prio_window)
        self._prio_drift_history = deque(maxlen=self.energy_prio_window)
        self._prio_noop_history = deque(maxlen=self.energy_prio_window)
        self.energy_weight_wait = float(energy_weight_wait)
        self.energy_weight_excess = float(energy_weight_excess)
        self.energy_weight_station = float(energy_weight_station)
        self.energy_weight_bottleneck = float(energy_weight_bottleneck)
        self.energy_weight_severe = float(energy_weight_severe)
        self.energy_weight_completed = float(energy_weight_completed)
        self.energy_weight_long_terminal = float(energy_weight_long_terminal)
        self.energy_weight_long_cvar = float(energy_weight_long_cvar)
        self.energy_weight_long_peak = float(energy_weight_long_peak)
        self.energy_weight_long_delta = float(energy_weight_long_delta)
        self.energy_service_relief_scale = float(energy_service_relief_scale)
        self.energy_service_weight_order_age = float(
            energy_service_weight_order_age
        )
        self.energy_service_weight_station_pending = float(
            energy_service_weight_station_pending
        )
        self.energy_service_weight_order_size = float(
            energy_service_weight_order_size
        )
        self.energy_noop_base_margin = float(energy_noop_base_margin)
        self.energy_noop_cost_scale = float(energy_noop_cost_scale)
        self.energy_noop_weight_order_age = float(energy_noop_weight_order_age)
        self.energy_noop_weight_station_pending = float(
            energy_noop_weight_station_pending
        )
        self.energy_noop_weight_global_pending = float(
            energy_noop_weight_global_pending
        )
        self.energy_noop_weight_idle = float(energy_noop_weight_idle)
        self.energy_age_norm = max(1.0, float(energy_age_norm))
        if lyapunov_l0_mode not in ("off", "diagnostic", "additive"):
            raise ValueError(
                "lyapunov_l0_mode must be one of: off, diagnostic, additive"
            )
        if candidate_robot_mode not in (
            "nearest", "eta_stratified", "stratified",
        ):
            raise ValueError(
                "candidate_robot_mode must be one of: nearest, "
                "eta_stratified, stratified"
            )
        if candidate_context_mode not in ("prefix", "stratified"):
            raise ValueError(
                "candidate_context_mode must be one of: prefix, stratified"
            )
        if float(candidate_context_factor) < 1.0:
            raise ValueError("candidate_context_factor must be >= 1")
        self.lyapunov_l0_mode = lyapunov_l0_mode
        self.lyapunov_l0_lambda = max(0.0, float(lyapunov_l0_lambda))
        valid_work_drift_modes = (
            "off",
            "group_range_additive",
            "analytic_h5_group_range",
        )
        if work_drift_mode not in valid_work_drift_modes:
            raise ValueError(
                "work_drift_mode must be one of: "
                + ", ".join(valid_work_drift_modes)
            )
        if float(work_drift_lambda) < 0.0:
            raise ValueError("work_drift_lambda must be non-negative")
        if work_drift_mode != "off" and float(work_drift_lambda) <= 0.0:
            raise ValueError("Layer-5 work_drift_lambda must be positive")
        if work_drift_mode == "group_range_additive" and not work_drift_head_path:
            raise ValueError(
                "work_drift_head_path is required for work-drift scoring"
            )
        if (
            work_drift_mode == "group_range_additive"
            and not work_drift_layer4_report_path
        ):
            raise ValueError(
                "the held-out Layer-4 report is required for work-drift scoring"
            )
        if work_drift_mode == "analytic_h5_group_range" and (
            work_drift_head_path or work_drift_layer4_report_path
        ):
            raise ValueError(
                "analytic_h5_group_range is parameter-free and forbids head/report "
                "artifacts"
            )
        self.work_drift_mode = work_drift_mode
        self.work_drift_head_path = work_drift_head_path
        self.work_drift_layer4_report_path = work_drift_layer4_report_path
        self.work_drift_lambda = float(work_drift_lambda)
        self.work_drift_layer5_evaluation = bool(
            work_drift_layer5_evaluation
        )
        self.include_no_assign_candidate = bool(
            include_no_assign_candidate
        )
        valid_dispatch_potential_modes = ("off", "diagnostic", "bridge")
        if dispatch_potential_mode not in valid_dispatch_potential_modes:
            raise ValueError(
                "dispatch_potential_mode must be one of: "
                + ", ".join(valid_dispatch_potential_modes)
            )
        self.dispatch_potential_mode = str(dispatch_potential_mode)
        if (
            self.dispatch_potential_mode != "off"
            and not self.include_no_assign_candidate
        ):
            raise ValueError(
                "dispatch potential requires the checkpoint-audited native "
                "NO_ASSIGN candidate"
            )
        if self.work_drift_mode != "off":
            mixed_features = []
            if self.risk_threshold is not None or self.risk_weight is not None:
                mixed_features.append("risk threshold/override")
            if self.long_risk_beta:
                mixed_features.append("long-risk beta")
            if self.risk_defer_mode != "off":
                mixed_features.append("risk defer")
            if self.random_defer_rate > 0.0:
                mixed_features.append("random defer")
            if self.station_injection_weight > 0.0:
                mixed_features.append("station injection")
            if self.potential_guard_mode != "off":
                mixed_features.append("potential guard")
            if self.pool_scoring_mode != "off":
                mixed_features.append("pool scoring")
            if self.candidate_set_guard_mode != "off":
                mixed_features.append("candidate-set guard")
            if self.margin_substitute_mode != "off":
                mixed_features.append("margin substitute")
            if self.energy_scoring_mode != "off":
                mixed_features.append("legacy energy scoring")
            if self.lyapunov_l0_mode != "off":
                mixed_features.append("legacy analytic L0 action scoring")
            if candidate_robot_mode != "nearest":
                mixed_features.append("candidate robot sampling/stratification")
            if candidate_context_mode != "prefix":
                mixed_features.append("candidate context sampling/stratification")
            if mixed_features:
                raise ValueError(
                    "work-drift Layer-5 scoring must be isolated from: "
                    + ", ".join(mixed_features)
                )
        self.candidate_robot_mode = candidate_robot_mode
        self.candidate_context_mode = candidate_context_mode
        self.candidate_context_factor = float(candidate_context_factor)
        self._candidate_explore_rng = random.Random(candidate_explore_seed)
        from WorldModel.core.lyapunov import LyapunovL0Config
        lyapunov_config_values = {
            "work_weight": float(lyapunov_l0_work_weight),
            "station_weight": float(lyapunov_l0_station_weight),
            "traffic_weight": float(lyapunov_l0_traffic_weight),
            "stall_weight": float(lyapunov_l0_stall_weight),
            "plan_fail_weight": float(lyapunov_l0_plan_fail_weight),
            "arrival_weight": float(lyapunov_l0_arrival_weight),
            "eta_bin_edges": tuple(int(v) for v in lyapunov_l0_eta_bins),
            "reservation_window": max(1, int(reservation_window)),
        }
        lyapunov_config_values.update(dict(lyapunov_l0_config or {}))
        if "eta_bin_edges" in lyapunov_config_values:
            lyapunov_config_values["eta_bin_edges"] = tuple(
                int(value)
                for value in lyapunov_config_values["eta_bin_edges"]
            )
        self.lyapunov_l0_config = LyapunovL0Config(
            **lyapunov_config_values
        )
        self.path_planner = None

        self._initialized = False
        self._model = None
        self._edge_index = None
        self._node_map = None
        self._inv_node_map = None
        self._local_capacity = None
        self._bottleneck_score = None
        self._node_type_arr = None
        self._adj = None
        self._num_nodes = 0
        self._feature_history = None
        self._flow_counter = {}
        self._edge_flow_counter: dict = {}
        self._last_edge_flow_update_tick: int = -1
        self._station_node_ids = []
        self._station_ids = []
        self._work_drift_head = None
        self._work_drift_payload = None
        self._work_drift_layer4_report = None
        self._work_drift_horizon = (
            5 if self.work_drift_mode == "analytic_h5_group_range" else 0
        )
        self._analytic_work_service_durations = None
        self._checkpoint_action_schema = {}
        self._dispatch_distance = None
        self._dispatch_service_durations = None
        self._dispatch_debt_ledger = None
        self._dispatch_snapshot = None

        if self.include_no_assign_candidate:
            mixed_features = []
            if self.risk_threshold is not None or self.risk_weight is not None:
                mixed_features.append("risk threshold/override")
            if self.long_risk_beta:
                mixed_features.append("long-risk beta")
            if self.risk_defer_mode != "off" or self.random_defer_rate > 0.0:
                mixed_features.append("risk/random defer")
            if self.station_injection_weight > 0.0:
                mixed_features.append("station injection")
            if self.potential_guard_mode != "off":
                mixed_features.append("potential guard")
            if self.pool_scoring_mode != "off":
                mixed_features.append("pool scoring")
            if self.candidate_set_guard_mode != "off":
                mixed_features.append("candidate-set guard")
            if self.margin_substitute_mode != "off":
                mixed_features.append("margin substitute")
            if self.energy_scoring_mode != "off":
                mixed_features.append("legacy energy/no-op scoring")
            if self.lyapunov_l0_mode != "off":
                mixed_features.append("legacy analytic L0 scoring")
            if self.work_drift_mode != "off":
                mixed_features.append("work-drift auxiliary")
            if mixed_features:
                raise ValueError(
                    "native NO_ASSIGN must be evaluated by the same pure "
                    "World Model score, isolated from: "
                    + ", ".join(mixed_features)
                )
        if self.dispatch_potential_mode != "off":
            dispatch_conflicts = []
            if self.candidate_context_mode != "prefix":
                dispatch_conflicts.append("candidate context stratification")
            if self.candidate_robot_mode != "nearest":
                dispatch_conflicts.append("candidate robot stratification")
            if dispatch_conflicts:
                raise ValueError(
                    "Gate-1 dispatch potential must use its frozen debt/age "
                    "proposal order and all-idle robot scope, isolated from: "
                    + ", ".join(dispatch_conflicts)
                )

        self.stats = {
            "assign_calls": 0,
            "model_assign_calls": 0,
            "fallback_greedy_calls": 0,
            "warmup_defer_calls": 0,
            "native_no_assign_contexts": 0,
            "native_no_assign_scored": 0,
            "native_no_assign_selected": 0,
            "native_no_assign_assignment_selected": 0,
            "native_no_assign_score_sum": 0.0,
            "native_no_assign_margin_sum": 0.0,
            "all_idle_candidate_contexts": 0,
            "all_idle_candidates_scored": 0,
            "model_inference_calls": 0,
            "assignment_time_total_ms": 0.0,
            "model_inference_time_total_ms": 0.0,
            "risk_guard_filtered": 0,
            "risk_guard_fallback": 0,
            "risk_max_count": 0,
            "risk_max_sum": 0.0,
            "risk_max_max": float("-inf"),
            "risk_max_min": float("inf"),
            "decision_contexts_total": 0,
            "shadow_greedy_compared": 0,
            "shadow_greedy_match": 0,
            "local_greedy_compared": 0,
            "local_greedy_match": 0,
            "wm_selected_rank_sum": 0.0,
            "wm_selected_rank_gt1": 0,
            "wm_selected_extra_distance_sum": 0.0,
            "wm_selected_extra_distance_positive": 0,
            "risk_defer_contexts": 0,
            "risk_defer_deferred": 0,
            "risk_defer_executed": 0,
            "risk_defer_score_sum": 0.0,
            "risk_defer_score_max": float("-inf"),
            "risk_defer_score_min": float("inf"),
            "risk_defer_deferred_score_sum": 0.0,
            "risk_defer_executed_score_sum": 0.0,
            "risk_defer_topq_batches": 0,
            "risk_defer_topq_budget_sum": 0,
            "risk_defer_rolling_checks": 0,
            "risk_defer_rolling_warmup": 0,
            "risk_defer_rolling_threshold_sum": 0.0,
            "random_defer_contexts": 0,
            "random_defer_deferred": 0,
            "station_injection_candidates": 0,
            "station_injection_penalty_sum": 0.0,
            "station_injection_penalty_max": float("-inf"),
            "station_injection_stress_sum": 0.0,
            "station_injection_fast_sum": 0.0,
            "station_injection_selected": 0,
            "station_injection_selected_penalty_sum": 0.0,
            "station_injection_selected_stress_sum": 0.0,
            "station_injection_selected_fast_sum": 0.0,
            "station_injection_diag_contexts": 0,
            "station_injection_diag_flips": 0,
            "station_injection_diag_margin_count": 0,
            "station_injection_diag_base_margin_sum": 0.0,
            "station_injection_diag_with_margin_sum": 0.0,
            "station_injection_diag_penalty_spread_sum": 0.0,
            "station_injection_diag_penalty_to_margin_sum": 0.0,
            "station_injection_diag_penalty_to_margin_count": 0,
            "station_injection_diag_base_best_penalty_sum": 0.0,
            "station_injection_diag_with_best_penalty_sum": 0.0,
            "potential_guard_contexts": 0,
            "potential_guard_deferred": 0,
            "potential_guard_executed": 0,
            "potential_guard_score_sum": 0.0,
            "potential_guard_score_max": float("-inf"),
            "potential_guard_score_min": float("inf"),
            "potential_guard_deferred_score_sum": 0.0,
            "potential_guard_executed_score_sum": 0.0,
            "potential_guard_rolling_checks": 0,
            "potential_guard_rolling_warmup": 0,
            "potential_guard_rolling_threshold_sum": 0.0,
            "potential_guard_random_contexts": 0,
            "potential_guard_random_deferred": 0,
            "pool_scoring_contexts": 0,
            "pool_scoring_candidates": 0,
            "pool_scoring_baseline_selected": 0,
            "pool_scoring_selected": 0,
            "pool_scoring_robot_substitutions": 0,
            "pool_scoring_order_replacements": 0,
            "pool_scoring_baseline_context_missed": 0,
            "pool_scoring_selected_base_score_sum": 0.0,
            "pool_scoring_selected_potential_score_sum": 0.0,
            "pool_scoring_selected_potential_rank_sum": 0.0,
            "pool_scoring_selected_final_score_sum": 0.0,
            "decision_trace_contexts": 0,
            "decision_trace_dropped": 0,
            "candidate_set_guard_contexts": 0,
            "candidate_set_guard_deferred": 0,
            "candidate_set_guard_executed": 0,
            "candidate_set_guard_warmup": 0,
            "candidate_set_guard_potential_triggered": 0,
            "candidate_set_guard_risk_triggered": 0,
            "candidate_set_guard_random_deferred": 0,
            "candidate_set_guard_min_potential_sum": 0.0,
            "candidate_set_guard_min_risk_sum": 0.0,
            "candidate_set_guard_potential_threshold_sum": 0.0,
            "candidate_set_guard_risk_threshold_sum": 0.0,
            "candidate_set_guard_threshold_count": 0,
            "margin_substitute_contexts": 0,
            "margin_substitute_opportunities": 0,
            "margin_substitute_substituted": 0,
            "margin_substitute_warmup": 0,
            "margin_substitute_gap_sum": 0.0,
            "margin_substitute_gain_sum": 0.0,
            "margin_substitute_gap_threshold_sum": 0.0,
            "margin_substitute_threshold_count": 0,
            "energy_scoring_contexts": 0,
            "energy_scoring_candidates": 0,
            "energy_score_sum": 0.0,
            "energy_potential_sum": 0.0,
            "energy_service_relief_sum": 0.0,
            "energy_noop_contexts": 0,
            "energy_noop_deferred": 0,
            "energy_noop_score_sum": 0.0,
            "energy_noop_cost_sum": 0.0,
            "energy_noop_best_action_score_sum": 0.0,
            "energy_conv_contexts": 0,
            "energy_conv_active_contexts": 0,
            "energy_conv_warmup_contexts": 0,
            "energy_conv_modified_decisions": 0,
            "energy_conv_gate_g_sum": 0.0,
            "energy_conv_v_state_sum": 0.0,
            "energy_conv_signal_selected_sum": 0.0,
            "energy_conv_z_base_selected_sum": 0.0,
            "energy_conv_z_delta_selected_sum": 0.0,
            "energy_prio_contexts": 0,
            "energy_prio_deferred": 0,
            "energy_prio_executed": 0,
            "energy_prio_warmup_contexts": 0,
            "energy_prio_random_contexts": 0,
            "energy_prio_random_deferred": 0,
            "energy_prio_mu_t_sum": 0.0,
            "energy_prio_mu_t_max": 0.0,
            "energy_prio_v_backlog_sum": 0.0,
            "energy_prio_v_backlog_max": 0.0,
            "energy_prio_priority_executed_sum": 0.0,
            "energy_prio_priority_deferred_sum": 0.0,
            "energy_prio_relief_executed_sum": 0.0,
            "energy_prio_relief_deferred_sum": 0.0,
            "energy_prio_drift_executed_sum": 0.0,
            "energy_prio_drift_deferred_sum": 0.0,
            "energy_prio_noop_executed_sum": 0.0,
            "energy_prio_noop_deferred_sum": 0.0,
            "lyapunov_l0_state_count": 0,
            "lyapunov_l0_state_sum": 0.0,
            "lyapunov_l0_arrival_candidates": 0,
            "lyapunov_l0_arrival_delta_sum": 0.0,
            "lyapunov_l0_arrival_delta_min": float("inf"),
            "lyapunov_l0_arrival_delta_max": float("-inf"),
            "lyapunov_l0_arrival_eta_sum": 0.0,
            "lyapunov_l0_arrival_selected": 0,
            "lyapunov_l0_arrival_selected_delta_sum": 0.0,
            "lyapunov_l0_arrival_selected_eta_sum": 0.0,
            "eta_stratified_contexts": 0,
            "eta_stratified_non_nearest_added": 0,
            "route_conflict_representatives": 0,
            "route_conflict_non_nearest_added": 0,
            "context_stratified_calls": 0,
            "context_stratified_available": 0,
            "context_stratified_selected": 0,
            "context_pressure_representatives": 0,
            "context_eta_representatives": 0,
            "context_route_representatives": 0,
            "context_conflict_representatives": 0,
            "context_explore_representatives": 0,
            "work_drift_contexts": 0,
            "work_drift_candidates": 0,
            "work_drift_candidate_superset_contexts": 0,
            "work_drift_max_candidate_count": 0,
            "work_drift_raw_sum": 0.0,
            "work_drift_raw_min": float("inf"),
            "work_drift_raw_max": float("-inf"),
            "work_drift_exact_tie_contexts": 0,
            "work_drift_wm_exact_tie_contexts": 0,
            "work_drift_modified_decisions": 0,
            "work_drift_selected_group_range_sum": 0.0,
            "work_drift_selected_raw_sum": 0.0,
        }
        self.stats.update({
            "dispatch_snapshot_count": 0,
            "dispatch_full_pending_contexts_sum": 0,
            "dispatch_pressure_contexts_sum": 0,
            "dispatch_eligible_contexts_sum": 0,
            "dispatch_unresolved_pending_orders_sum": 0,
            "dispatch_unresolved_pending_orders_max": 0,
            "dispatch_contexts": 0,
            "dispatch_candidates": 0,
            "dispatch_exact_wm_ties": 0,
            "dispatch_group_span_violations": 0,
            "dispatch_group_span_max": 0.0,
            "dispatch_raw_wm_defer_selected": 0,
            "dispatch_bridge_defer_selected": 0,
            "dispatch_actual_defer_selected": 0,
            "dispatch_actual_assignment_selected": 0,
            "dispatch_modified_decisions": 0,
            "dispatch_crossing_contexts": 0,
            "dispatch_crossing_assignment_selected": 0,
            "dispatch_crossing_violations": 0,
            "dispatch_dominance_contexts": 0,
            "dispatch_dominance_raw_wm_defer": 0,
            "dispatch_dominance_final_assignment": 0,
            "dispatch_dominance_final_defer": 0,
            "dispatch_relative_gap_sum": 0.0,
            "dispatch_max_debt_before": 0.0,
            "dispatch_max_continuous_eligible_mass": 0.0,
            "dispatch_max_continuous_eligible_ticks": 0,
            "dispatch_eligible_debt_increments": 0,
            "dispatch_temporarily_ineligible_contexts": 0,
            "dispatch_assigned_contexts": 0,
            "dispatch_streak_bound_violations": 0,
        })
        self.risk_max_values: List[float] = []

    def propose_assignment_contexts(
        self,
        world_state,
        max_contexts: Optional[int] = None,
    ) -> List[AssignmentContext]:
        """Expose a valid fixed context without selecting a robot."""
        return propose_fixed_assignment_contexts(
            self, world_state, max_contexts=max_contexts
        )

    def commit_assignments(
        self,
        world_state,
        contexts: List[AssignmentContext],
        robot_choices: Dict[int, int],
    ) -> List[Task]:
        """Materialise only robot IDs explicitly selected by the model."""
        return commit_fixed_context_assignments(
            world_state, contexts, robot_choices
        )

    @staticmethod
    def _scalar(value, default: float = 0.0) -> float:
        if value is None:
            return default
        if hasattr(value, "item"):
            value = value.item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _append_decision_trace(self, record: dict):
        if not self.decision_trace_enabled:
            return
        if len(self.decision_trace_records) >= self.decision_trace_max_records:
            self.stats["decision_trace_dropped"] += 1
            return
        self.decision_trace_records.append(record)
        self.stats["decision_trace_contexts"] += 1

    @staticmethod
    def _quantile_from_history(history, quantile: float) -> Optional[float]:
        hist = list(history)
        if not hist:
            return None
        sorted_hist = sorted(hist)
        idx = int((len(sorted_hist) - 1) * quantile)
        return sorted_hist[max(0, min(len(sorted_hist) - 1, idx))]

    def _energy_from_system_preds(self, system_preds, lr_components: dict) -> float:
        """Phase 6.1 state potential evaluated on predicted rollout labels.

        This is an action-level potential: each candidate action is rolled out
        by the world model, then mapped to the same scalar system-energy space.
        """
        if system_preds is None:
            return 0.0
        K = int(system_preds.shape[0])
        if K <= 0:
            return 0.0

        if self.energy_potential_form == "discounted":
            discounts = torch.tensor(
                [self.energy_discount ** k for k in range(K)],
                device=system_preds.device,
                dtype=system_preds.dtype,
            )
            norm = discounts.sum().clamp_min(1e-6)

            def reduce_dim(dim: int) -> torch.Tensor:
                return (discounts * system_preds[:, dim]).sum() / norm

            station_pressure = reduce_dim(2) + reduce_dim(3)
            completed = reduce_dim(5)
            bottleneck = reduce_dim(4)
            severe = system_preds[:, 6].max()
        else:
            tail2 = system_preds[-min(2, K):]
            tail3 = system_preds[-min(3, K):]

            def reduce_dim(dim: int) -> torch.Tensor:
                return tail2[:, dim].mean()

            station_pressure = reduce_dim(2) + reduce_dim(3)
            completed = reduce_dim(5)
            bottleneck = reduce_dim(4)
            severe = tail3[:, 6].max()

        energy = (
            self.energy_weight_wait * reduce_dim(0)
            + self.energy_weight_excess * reduce_dim(1)
            + self.energy_weight_station * station_pressure
            + self.energy_weight_bottleneck * bottleneck
            + self.energy_weight_severe * severe
            - self.energy_weight_completed * completed
        )
        energy = energy + (
            self.energy_weight_long_peak
            * float(lr_components.get("long_risk_peak_q95", 0.0))
            + self.energy_weight_long_cvar
            * float(lr_components.get("long_risk_cvar_q90", 0.0))
            + self.energy_weight_long_terminal
            * float(lr_components.get("long_risk_terminal_q90", 0.0))
            + self.energy_weight_long_delta
            * max(0.0, float(lr_components.get("long_risk_delta_group_q90", 0.0)))
        )
        return self._scalar(energy)

    @staticmethod
    def _z_scores_or_rank(values: List[float]) -> List[float]:
        m = len(values)
        if m <= 0:
            return []
        if m == 1:
            return [0.0]
        mu = sum(values) / float(m)
        sd = (sum((x - mu) ** 2 for x in values) / float(m)) ** 0.5
        if sd < 1e-6:
            order = sorted(range(m), key=lambda i: values[i])
            ranks = [0.0] * m
            denom = float(max(m - 1, 1))
            for pos, idx in enumerate(order):
                ranks[idx] = float(pos) / denom
            return ranks
        return [(x - mu) / sd for x in values]

    @staticmethod
    def _positive_delta_scale(values: List[float]) -> List[float]:
        m = len(values)
        if m <= 0:
            return []
        if m == 1:
            return [0.0]
        v_min = min(values)
        deltas = [x - v_min for x in values]
        mu = sum(deltas) / float(m)
        sd = (sum((x - mu) ** 2 for x in deltas) / float(m)) ** 0.5
        if sd < 1e-6:
            spread = max(deltas) - min(deltas)
            if spread < 1e-9:
                return [0.0] * m
            order = sorted(range(m), key=lambda i: deltas[i])
            ranks = [0.0] * m
            denom = float(max(m - 1, 1))
            for pos, idx in enumerate(order):
                ranks[idx] = float(pos) / denom
            return ranks
        return [x / sd for x in deltas]

    def _energy_gate_value(self, v_state: float) -> float:
        if self.energy_gate_mode == "off":
            return 1.0
        hist = self._energy_gate_vstate_history
        if len(hist) < self.energy_gate_warmup:
            return 0.0
        threshold = self._quantile_from_history(hist, self.energy_gate_quantile)
        if threshold is None:
            return 0.0
        if self.energy_gate_mode == "hard":
            return 1.0 if v_state > threshold else 0.0

        p90 = self._quantile_from_history(hist, 0.90)
        p70 = self._quantile_from_history(hist, 0.70)
        if p90 is None or p70 is None:
            return 0.0
        tau = max((p90 - p70) / 4.0, 1e-6)
        z = max(-60.0, min(60.0, (v_state - threshold) / tau))
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def _rolling_std(history: deque) -> float:
        values = [float(v) for v in history]
        if not values:
            return 1e-6
        mu = sum(values) / float(len(values))
        var = sum((x - mu) ** 2 for x in values) / float(len(values))
        return max(var ** 0.5, 1e-6)

    def _energy_backlog_potential(self, world) -> float:
        pending = world.order_state.get_pending_orders()
        num_agents = max(len(world.agents), 1)
        if not pending:
            return 0.0
        ages = [
            max(0.0, float(world.tick - getattr(order, "created_at", world.tick)))
            for order in pending
        ]
        mean_age = sum(ages) / float(len(ages)) if ages else 0.0
        return (
            float(len(pending)) / float(num_agents)
            + mean_age / self.energy_age_norm
        )

    def _energy_mu_t(self, world) -> tuple[float, float]:
        v_backlog = self._energy_backlog_potential(world)
        mu_t = v_backlog / self.energy_mu_ref
        mu_t = max(self.energy_mu_min, min(self.energy_mu_max, mu_t))
        return mu_t, v_backlog

    def _planned_edge_pressure(self, world) -> Dict[tuple, float]:
        """Time-agnostic near-term edge reservation proxy for pool coverage."""
        pressure: Dict[tuple, float] = {}
        if not self._node_map:
            return pressure
        for other in getattr(world, "agents", ()):
            previous = self._node_map.get(tuple(other.position))
            if previous is None:
                continue
            path = list(getattr(other, "path", ()) or ())
            index = max(0, int(getattr(other, "path_index", 0)))
            for position in path[index:index + max(1, self.reservation_window)]:
                current = self._node_map.get(tuple(position))
                if current is None or current == previous:
                    previous = current if current is not None else previous
                    continue
                pressure[(previous, current)] = (
                    pressure.get((previous, current), 0.0) + 1.0
                )
                previous = current
        return pressure

    def _route_conflict_score(
        self,
        world,
        ctx: AssignmentContext,
        agent,
        planned_pressure: Optional[Dict[tuple, float]] = None,
    ) -> float:
        """Fast, read-only route conflict proxy used only for representation.

        It combines recent directed edge flow, currently planned reservations,
        opposite-direction pressure, and static bottleneck centrality.  The
        value is a cost; it never rewards distance.
        """
        if not self._node_map:
            return 0.0
        from WorldModel.graph_builder import compute_preview_legs

        entry_position = getattr(ctx, "entry_position", None)
        station_position = entry_position or getattr(
            ctx, "station_location", None
        )
        if station_position is None:
            return float("inf")

        assignment = {
            "robot_id": agent.agent_id,
            "robot_start": tuple(agent.position),
            "pod_id": ctx.pod_id,
            "pod_location": tuple(ctx.pod_location),
            "station_location": tuple(station_position),
            "return_location": tuple(ctx.return_location),
            "order_size": ctx.order_size,
        }
        legs = compute_preview_legs(
            assignment,
            world.map_state,
            self._node_map,
            path_planner=None,
            world=None,
        )
        planned_pressure = planned_pressure or {}
        total = 0.0
        edges = 0
        for leg in legs:
            for source, target in zip(leg, leg[1:]):
                recent = float(self._edge_flow_counter.get((source, target), 0.0))
                recent_opp = float(self._edge_flow_counter.get((target, source), 0.0))
                reserved = float(planned_pressure.get((source, target), 0.0))
                reserved_opp = float(planned_pressure.get((target, source), 0.0))
                bottleneck = 0.0
                if self._bottleneck_score is not None:
                    try:
                        bottleneck = float(self._bottleneck_score[target])
                    except (IndexError, TypeError, ValueError):
                        bottleneck = 0.0
                total += (
                    recent + reserved
                    + 2.0 * (recent_opp + reserved_opp)
                    + bottleneck
                )
                edges += 1
        return total / max(edges, 1)

    @staticmethod
    def _normalise_costs(values: List[float]) -> List[float]:
        if not values:
            return []
        low, high = min(values), max(values)
        if high - low <= 1e-9:
            return [0.0] * len(values)
        return [(value - low) / (high - low) for value in values]

    def _select_context_representatives(
        self,
        world,
        contexts: List[AssignmentContext],
        budget: int,
    ) -> List[AssignmentContext]:
        """Pressure/route/ETA/conflict/exploration-stratified context pool."""
        if self.candidate_context_mode == "prefix" or len(contexts) <= budget:
            return contexts[:budget]
        from WorldModel.core.lyapunov import (
            compute_lyapunov_snapshot,
            preview_assignment_context,
        )

        self.stats["context_stratified_calls"] += 1
        self.stats["context_stratified_available"] += len(contexts)
        snapshot = compute_lyapunov_snapshot(world, self.lyapunov_l0_config)
        idle = list(world.get_idle_agents())
        planned_pressure = self._planned_edge_pressure(world)
        rows = []
        for index, ctx in enumerate(contexts):
            if idle:
                nearest = min(
                    idle,
                    key=lambda item: _manhattan_distance(
                        item.position, ctx.pod_location
                    ),
                )
                route = (
                    _manhattan_distance(nearest.position, ctx.pod_location)
                    + _manhattan_distance(
                        ctx.pod_location,
                        getattr(ctx, "entry_position", None)
                        or ctx.station_location,
                    )
                )
                preview = preview_assignment_context(
                    world, ctx, nearest,
                    config=self.lyapunov_l0_config,
                    snapshot=snapshot,
                )
                conflict = self._route_conflict_score(
                    world, ctx, nearest, planned_pressure
                )
                eta_bin = preview.eta_bin
                arrival_delta = float(preview.delta_arrival_potential)
            else:
                route = float("inf")
                conflict = float("inf")
                eta_bin = -1
                arrival_delta = 0.0
            rows.append({
                "index": index,
                "ctx": ctx,
                "pressure": float(snapshot.station_work.get(ctx.station_id, 0.0)),
                "route": float(route),
                "conflict": float(conflict),
                "eta_bin": int(eta_bin),
                "arrival_delta": arrival_delta,
            })

        selected = []
        selected_indices = set()

        def add(row, stat_key=None):
            if row is None or row["index"] in selected_indices:
                return
            if len(selected) >= budget:
                return
            selected.append(row)
            selected_indices.add(row["index"])
            if stat_key:
                self.stats[stat_key] += 1

        add(max(rows, key=lambda row: (row["pressure"], -row["route"])),
            "context_pressure_representatives")
        add(min(rows, key=lambda row: (row["route"], row["conflict"])),
            "context_route_representatives")
        add(min(rows, key=lambda row: (row["conflict"], row["route"])),
            "context_conflict_representatives")

        by_eta: Dict[int, List[dict]] = {}
        for row in rows:
            by_eta.setdefault(row["eta_bin"], []).append(row)
        for eta_bin in sorted(by_eta):
            add(
                min(by_eta[eta_bin], key=lambda row: (
                    row["arrival_delta"], row["route"]
                )),
                "context_eta_representatives",
            )

        remaining = [row for row in rows if row["index"] not in selected_indices]
        if remaining and len(selected) < budget:
            add(
                self._candidate_explore_rng.choice(remaining),
                "context_explore_representatives",
            )

        pressure_cost = self._normalise_costs([
            -row["pressure"] for row in rows
        ])
        route_cost = self._normalise_costs([row["route"] for row in rows])
        conflict_cost = self._normalise_costs([
            row["conflict"] for row in rows
        ])
        arrival_cost = self._normalise_costs([
            row["arrival_delta"] for row in rows
        ])
        for index, row in enumerate(rows):
            row["fill_cost"] = (
                pressure_cost[index] + route_cost[index]
                + conflict_cost[index] + arrival_cost[index]
            )
        for row in sorted(rows, key=lambda item: (
            item["fill_cost"], item["route"], item["index"]
        )):
            add(row)

        self.stats["context_stratified_selected"] += len(selected)
        return [row["ctx"] for row in selected]

    def _select_robot_candidates(
        self,
        world,
        ctx: AssignmentContext,
        available: List,
        lyapunov_snapshot=None,
    ) -> List:
        """Return every currently available idle robot.

        ``top_m`` and stratification belong to collection/training only.  At
        inference time no legal idle robot may be hidden from the World Model.
        Agent-id ordering is deterministic and has no distance preference.
        """
        candidates = sorted(available, key=lambda agent: int(agent.agent_id))
        self.stats["all_idle_candidate_contexts"] += 1
        self.stats["all_idle_candidates_scored"] += len(candidates)
        return candidates

    def _apply_energy_conversion(self, ctx_records: List[dict]) -> None:
        """Apply within-context gated drift scoring after all candidates exist.

        `score_conv` is deliberately context-local because z-normalization
        destroys cross-context comparability.
        """
        if not ctx_records:
            return

        if self.energy_conv_random_flip_rate > 0.0:
            self._apply_energy_conversion_random_flip(ctx_records)
            return

        signal_key = long_risk_drift_signal_key(self.energy_drift_signal)
        base_scores = [float(r.get("score", r.get("base_score", 0.0)))
                       for r in ctx_records]
        drift_values = [float(r.get(signal_key, 0.0)) for r in ctx_records]

        v_state = min(base_scores)
        history_len = len(self._energy_gate_vstate_history)
        gate_g = self._energy_gate_value(v_state)
        self._energy_gate_vstate_history.append(v_state)

        z_base = self._z_scores_or_rank(base_scores)
        z_delta = self._positive_delta_scale(drift_values)

        for i, record in enumerate(ctx_records):
            score_conv = (
                z_base[i]
                + self.energy_conv_lambda * gate_g * z_delta[i]
            )
            record["score_conv"] = score_conv
            record["energy_conv_signal"] = drift_values[i]
            record["energy_conv_delta_v"] = drift_values[i] - min(drift_values)
            record["energy_conv_z_base"] = z_base[i]
            record["energy_conv_z_delta"] = z_delta[i]
            record["energy_conv_gate_g"] = gate_g
            record["energy_conv_v_state"] = v_state

        base_best = min(
            ctx_records,
            key=lambda r: (r["score"], r["best_robot"]),
        )
        conv_best = min(
            ctx_records,
            key=lambda r: (r["score_conv"], r["best_robot"]),
        )

        self.stats["energy_conv_contexts"] += 1
        self.stats["energy_conv_gate_g_sum"] += gate_g
        self.stats["energy_conv_v_state_sum"] += v_state
        if self.energy_gate_mode != "off" and history_len < self.energy_gate_warmup:
            self.stats["energy_conv_warmup_contexts"] += 1
        if gate_g > 0.5:
            self.stats["energy_conv_active_contexts"] += 1
        if conv_best["best_robot"] != base_best["best_robot"]:
            self.stats["energy_conv_modified_decisions"] += 1
        self.stats["energy_conv_signal_selected_sum"] += float(
            conv_best.get("energy_conv_signal", 0.0)
        )
        self.stats["energy_conv_z_base_selected_sum"] += float(
            conv_best.get("energy_conv_z_base", 0.0)
        )
        self.stats["energy_conv_z_delta_selected_sum"] += float(
            conv_best.get("energy_conv_z_delta", 0.0)
        )

    def _apply_energy_conversion_random_flip(
        self, ctx_records: List[dict]
    ) -> None:
        """S4 matched-random control: keep the base ranking, but with
        probability `energy_conv_random_flip_rate` flip the argmin to a
        uniformly random other candidate.

        Rate is matched offline to A1's measured
        energy_conv_modified_decision_rate; the drift signal and gate are
        never consulted, so any effect is attributable to generic
        perturbation rather than drift-informed scheduling.
        """
        base_scores = [
            float(r.get("score", r.get("base_score", 0.0)))
            for r in ctx_records
        ]
        for record, base in zip(ctx_records, base_scores):
            record["score_conv"] = base
            record["energy_conv_gate_g"] = 0.0

        self.stats["energy_conv_contexts"] += 1

        if len(ctx_records) < 2:
            return
        if (
            self._energy_conv_flip_rng.random()
            >= self.energy_conv_random_flip_rate
        ):
            return

        best_idx = min(
            range(len(ctx_records)),
            key=lambda i: (base_scores[i], ctx_records[i]["best_robot"]),
        )
        other_indices = [
            i for i in range(len(ctx_records)) if i != best_idx
        ]
        target_idx = self._energy_conv_flip_rng.choice(other_indices)
        ctx_records[target_idx]["score_conv"] = min(base_scores) - 1.0
        self.stats["energy_conv_modified_decisions"] += 1

    def _context_opportunity_features(
        self,
        world,
        ctx: AssignmentContext,
        snapshot: dict,
    ) -> dict:
        num_agents = max(len(world.agents), 1)
        pending_orders = world.order_state.get_pending_orders()
        order = world.order_state.orders.get(ctx.order_id)
        order_age = 0.0
        if order is not None:
            order_age = max(0.0, float(world.tick - order.created_at))
        order_age_norm = order_age / self.energy_age_norm

        station_data = snapshot.get("stations", {}).get(ctx.station_id, {})
        station_pending = float(station_data.get("pending_pressure", 0.0))
        station_queue = float(station_data.get("queue_occ", 0.0))
        global_pending = float(len(pending_orders)) / float(num_agents)
        idle_fraction = float(len(world.get_idle_agents())) / float(num_agents)
        order_size_norm = float(ctx.order_size) / float(num_agents)
        return {
            "order_age_norm": order_age_norm,
            "station_pending_pressure": station_pending,
            "station_queue_occ": station_queue,
            "global_pending_pressure": global_pending,
            "idle_fraction": idle_fraction,
            "order_size_norm": order_size_norm,
        }

    def _energy_service_relief(self, features: dict) -> float:
        relief = (
            self.energy_service_weight_order_age
            * features["order_age_norm"]
            + self.energy_service_weight_station_pending
            * features["station_pending_pressure"]
            + self.energy_service_weight_order_size
            * features["order_size_norm"]
        )
        return self.energy_service_relief_scale * relief

    def _energy_noop_cost(self, features: dict) -> float:
        return (
            self.energy_noop_weight_order_age * features["order_age_norm"]
            + self.energy_noop_weight_station_pending
            * features["station_pending_pressure"]
            + self.energy_noop_weight_global_pending
            * features["global_pending_pressure"]
            + self.energy_noop_weight_idle * features["idle_fraction"]
        )

    def _compute_risk_defer_score(self, details: dict) -> float:
        """Diagnostic score for the pre-6.1 risk-defer trick.

        This hard guard only tests whether learned risk can identify dangerous
        assignments. It is not the final scheduling method; 6.1 should replace
        it with soft energy/drift scoring.
        """
        score = 0.0
        risk_max = float(details.get("risk_max", 0.0))
        if self.risk_defer_score in ("risk_max", "combined"):
            score += self.risk_defer_weight_risk_max * risk_max

        if self.risk_defer_score in ("long_risk", "combined"):
            lr = self._long_risk_components(details)
            score += (
                self.risk_defer_weight_peak
                * lr["long_risk_peak_q95"]
            )
            score += (
                self.risk_defer_weight_cvar
                * lr["long_risk_cvar_q90"]
            )
            score += (
                self.risk_defer_weight_terminal
                * lr["long_risk_terminal_q90"]
            )
            score += (
                self.risk_defer_weight_delta
                * max(0.0, lr["long_risk_delta_group_q90"])
            )
        return score

    def _long_risk_components(self, details: dict) -> dict:
        """Decode the optional LongRiskHead output through one strict schema."""

        return decode_long_risk_predictions(
            details.get("long_risk_preds"),
            scalar=self._scalar,
        )

    def _station_injection_snapshot(self, world) -> dict:
        """Per-station pressure used by the 6.1 injection diagnostic.

        This does not replace the learned cost / bottleneck terms. It only
        adds an optional online penalty for actions that quickly inject robots
        into already pressured station regions.
        """
        from WorldState.order_state import OrderStatus
        from WorldState.task_state import TaskStatus, TaskType

        station_positions = dict(world.map_state.station_positions)
        station_ids = sorted(station_positions.keys())
        num_agents = max(len(world.agents), 1)
        data = {
            sid: {
                "queue_occ": 0.0,
                "pending_pressure": 0.0,
                "active_delivery_pressure": 0.0,
                "near_robot_pressure": 0.0,
            }
            for sid in station_ids
        }

        for sid in station_ids:
            sq = world.station_state.stations.get(sid)
            if sq is not None:
                data[sid]["queue_occ"] = float(sq.occupancy()) / max(sq.capacity, 1)

        for order in world.order_state.orders.values():
            if order.status == OrderStatus.PENDING and order.station_id in data:
                data[order.station_id]["pending_pressure"] += 1.0 / num_agents

        for task in world.task_state.tasks.values():
            if (task.task_type == TaskType.DELIVER
                    and task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)):
                order = world.order_state.orders.get(task.order_id)
                if order is not None and order.station_id in data:
                    data[order.station_id]["active_delivery_pressure"] += (
                        1.0 / num_agents
                    )

        radius = self.station_injection_near_radius
        if radius > 0 and station_positions:
            for agent in world.agents:
                nearest_sid, nearest_dist = min(
                    ((sid, _manhattan_distance(agent.position, pos))
                     for sid, pos in station_positions.items()),
                    key=lambda x: x[1],
                )
                if nearest_dist <= radius and nearest_sid in data:
                    data[nearest_sid]["near_robot_pressure"] += 1.0 / num_agents

        return {
            "stations": data,
            "station_positions": station_positions,
        }

    def _station_stress_from_data(self, station_data: dict) -> float:
        return (
            0.20 * station_data["queue_occ"]
            + 0.35 * min(1.0, station_data["pending_pressure"])
            + 0.30 * min(1.0, station_data["active_delivery_pressure"])
            + 0.15 * min(1.0, station_data["near_robot_pressure"])
        )

    def _station_global_stress(self, snapshot: dict) -> float:
        stations = snapshot.get("stations", {})
        if not stations:
            return 0.0
        return max(self._station_stress_from_data(v) for v in stations.values())

    def _station_injection_penalty(
        self,
        agent,
        ctx: AssignmentContext,
        snapshot: dict,
        map_diameter: float,
    ) -> tuple[float, float, float, float]:
        station_data = snapshot.get("stations", {}).get(ctx.station_id)
        if not station_data:
            return 0.0, 0.0, 0.0, 0.0

        stress = self._station_stress_from_data(station_data)
        route_len = (
            _manhattan_distance(agent.position, ctx.pod_location)
            + _manhattan_distance(ctx.pod_location, ctx.station_location)
        )
        route_norm = float(route_len) / max(map_diameter, 1.0)
        fast_injection = max(
            0.0, 1.0 - route_norm / self.station_injection_eta_norm
        )
        raw_penalty = stress * fast_injection
        weighted = self.station_injection_weight * raw_penalty
        return weighted, raw_penalty, stress, fast_injection

    def _compute_potential_guard_score(
        self,
        details: dict,
        state_station_stress: float,
        station_stress: float,
        injection_raw: float,
    ) -> float:
        """Relative guard score for candidate admission.

        The score is only compared against recent candidates, so its absolute
        value is intentionally not interpreted as a universal threshold.
        """
        score = (
            self.potential_guard_weight_state * state_station_stress
            + self.potential_guard_weight_station * station_stress
            + self.potential_guard_weight_injection * injection_raw
        )
        score += self.potential_guard_weight_risk_max * float(
            details.get("risk_max", 0.0)
        )

        lr = self._long_risk_components(details)
        score += self.potential_guard_weight_peak * lr["long_risk_peak_q95"]
        score += self.potential_guard_weight_cvar * lr["long_risk_cvar_q90"]
        score += (
            self.potential_guard_weight_terminal
            * lr["long_risk_terminal_q90"]
        )
        score += (
            self.potential_guard_weight_delta
            * max(0.0, lr["long_risk_delta_group_q90"])
        )
        return score

    def _record_station_injection_diag(self, score_records: List[dict]):
        if not score_records:
            return

        base_ranked = sorted(score_records, key=lambda r: r["base_score"])
        with_ranked = sorted(
            score_records, key=lambda r: r["with_injection_score"]
        )
        base_best = base_ranked[0]
        with_best = with_ranked[0]

        self.stats["station_injection_diag_contexts"] += 1
        if base_best["robot_id"] != with_best["robot_id"]:
            self.stats["station_injection_diag_flips"] += 1

        self.stats["station_injection_diag_base_best_penalty_sum"] += (
            base_best["station_injection_penalty"]
        )
        self.stats["station_injection_diag_with_best_penalty_sum"] += (
            with_best["station_injection_penalty"]
        )

        penalties = [r["station_injection_penalty"] for r in score_records]
        spread = max(penalties) - min(penalties)
        self.stats["station_injection_diag_penalty_spread_sum"] += spread

        if len(base_ranked) >= 2:
            base_margin = (
                base_ranked[1]["base_score"] - base_ranked[0]["base_score"]
            )
            with_margin = (
                with_ranked[1]["with_injection_score"]
                - with_ranked[0]["with_injection_score"]
            )
            self.stats["station_injection_diag_margin_count"] += 1
            self.stats["station_injection_diag_base_margin_sum"] += base_margin
            self.stats["station_injection_diag_with_margin_sum"] += with_margin
            if base_margin > 1e-9:
                self.stats["station_injection_diag_penalty_to_margin_sum"] += (
                    spread / base_margin
                )
                self.stats[
                    "station_injection_diag_penalty_to_margin_count"
                ] += 1

    def _topq_budget_count(self, n: int) -> int:
        if n <= 0 or self.risk_defer_topq <= 0.0:
            return 0
        target = n * self.risk_defer_topq
        count = int(target)
        if self._random_defer_rng.random() < (target - count):
            count += 1
        return max(0, min(n, count))

    def _rolling_threshold(self) -> Optional[float]:
        hist = list(self._risk_defer_history)
        if len(hist) < self.risk_defer_min_history:
            return None
        sorted_hist = sorted(hist)
        idx = int((len(sorted_hist) - 1) * self.risk_defer_rolling_quantile)
        return sorted_hist[max(0, min(len(sorted_hist) - 1, idx))]

    def _potential_guard_rolling_threshold(self) -> Optional[float]:
        hist = list(self._potential_guard_history)
        if len(hist) < self.potential_guard_min_history:
            return None
        sorted_hist = sorted(hist)
        idx = int((len(sorted_hist) - 1) * self.potential_guard_rolling_quantile)
        return sorted_hist[max(0, min(len(sorted_hist) - 1, idx))]

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _exact_group_range(values: List[float]) -> List[float]:
        """Frozen no-gap group transform used by Layers 3--5."""
        if not values:
            return []
        mean = sum(float(value) for value in values) / float(len(values))
        value_range = max(values) - min(values)
        if value_range == 0.0:
            return [0.0 for _ in values]
        return [(float(value) - mean) / value_range for value in values]

    def _prepare_dispatch_snapshot(self, world_state):
        """Build the action-independent full pending-chain ledger."""
        if getattr(self, "dispatch_potential_mode", "off") == "off":
            return None
        if (
            self._dispatch_debt_ledger is None
            or self._dispatch_distance is None
            or self._dispatch_service_durations is None
        ):
            raise RuntimeError("dispatch potential was not initialised")
        contexts = enumerate_pending_assignment_contexts(
            self,
            world_state,
            max_contexts=None,
            include_temporarily_unavailable=True,
            deduplicate_pods=False,
        )
        snapshot = self._dispatch_debt_ledger.snapshot(
            world_state,
            contexts,
            self._dispatch_distance,
            self._dispatch_service_durations,
        )
        self._dispatch_snapshot = snapshot
        self.stats["dispatch_snapshot_count"] += 1
        self.stats["dispatch_full_pending_contexts_sum"] += len(
            snapshot.contexts
        )
        self.stats["dispatch_pressure_contexts_sum"] += len(
            snapshot.pressure_keys
        )
        self.stats["dispatch_eligible_contexts_sum"] += len(
            snapshot.eligible_free_flow
        )
        self.stats["dispatch_unresolved_pending_orders_sum"] += int(
            snapshot.unresolved_pending_orders
        )
        self.stats["dispatch_unresolved_pending_orders_max"] = max(
            self.stats["dispatch_unresolved_pending_orders_max"],
            int(snapshot.unresolved_pending_orders),
        )
        self.stats["dispatch_max_debt_before"] = max(
            self.stats["dispatch_max_debt_before"],
            max(snapshot.debt_before.values(), default=0.0),
        )
        return snapshot

    def _finalize_dispatch_snapshot(self, snapshot, tasks: List[Task]) -> None:
        """Update service debt exactly once from committed PICK tasks."""
        if snapshot is None:
            return
        from WorldModel.core.dispatch_potential import context_key
        from WorldState.task_state import TaskType

        committed = {
            (int(task.order_id), int(task.pod_id))
            for task in tasks
            if task.task_type == TaskType.PICK
        }
        assigned_keys = {
            context_key(context)
            for context in snapshot.contexts
            if (int(context.order_id), int(context.pod_id)) in committed
        }
        summary = self._dispatch_debt_ledger.finalize(
            snapshot,
            assigned_keys,
        )
        self.stats["dispatch_eligible_debt_increments"] += int(
            summary["eligible_debt_increments"]
        )
        self.stats["dispatch_temporarily_ineligible_contexts"] += int(
            summary["temporarily_ineligible_contexts"]
        )
        self.stats["dispatch_assigned_contexts"] += int(
            summary["assigned_contexts"]
        )
        self.stats["dispatch_max_continuous_eligible_mass"] = max(
            self.stats["dispatch_max_continuous_eligible_mass"],
            float(summary["max_continuous_eligible_mass"]),
        )
        self.stats["dispatch_max_continuous_eligible_ticks"] = max(
            self.stats["dispatch_max_continuous_eligible_ticks"],
            int(summary["max_continuous_eligible_ticks"]),
        )
        if not bool(summary["streak_bound_passed"]):
            self.stats["dispatch_streak_bound_violations"] = 1

    def _apply_dispatch_potential(
        self,
        ctx: AssignmentContext,
        scored_records: List[dict],
        no_assign_record: dict,
    ) -> dict:
        """Attach frozen group-range + analytic drift and choose one action."""
        if self.dispatch_potential_mode == "off":
            raise RuntimeError("dispatch scoring requested while disabled")
        if self._dispatch_snapshot is None:
            raise RuntimeError("dispatch snapshot missing for current tick")
        if not scored_records:
            raise ValueError("dispatch scoring requires robot candidates")

        raw_records = list(scored_records) + [no_assign_record]
        raw_values = [float(record["score"]) for record in raw_records]
        group_values = self._exact_group_range(raw_values)
        raw_range = max(raw_values) - min(raw_values)
        group_span = max(group_values) - min(group_values)
        if raw_range == 0.0:
            self.stats["dispatch_exact_wm_ties"] += 1
        if group_span > 1.0 + 1e-12:
            self.stats["dispatch_group_span_violations"] += 1
        self.stats["dispatch_group_span_max"] = max(
            self.stats["dispatch_group_span_max"], group_span
        )

        terms = self._dispatch_snapshot.terms(ctx)
        if not terms.eligible:
            raise RuntimeError(
                "dispatch proposal exposed a context without a reachable "
                "idle robot"
            )
        for record, wm_group in zip(scored_records, group_values[:-1]):
            record["dispatch_wm_group_range"] = float(wm_group)
            record["dispatch_delta_l"] = float(terms.delta_l_assign)
            record["dispatch_composite_score"] = float(
                wm_group + terms.delta_l_assign
            )
            record["dispatch_action_type"] = "assign_robot"
        no_assign_record["dispatch_wm_group_range"] = float(group_values[-1])
        no_assign_record["dispatch_delta_l"] = float(terms.delta_l_defer)
        no_assign_record["dispatch_composite_score"] = float(
            group_values[-1] + terms.delta_l_defer
        )
        no_assign_record["dispatch_action_type"] = "no_assign"

        raw_best_assignment = min(
            scored_records,
            key=lambda row: (float(row["score"]), int(row["best_robot"])),
        )
        bridge_best_assignment = min(
            scored_records,
            key=lambda row: (
                float(row["dispatch_composite_score"]),
                int(row["best_robot"]),
            ),
        )
        raw_defer = bool(
            float(no_assign_record["score"])
            < float(raw_best_assignment["score"])
        )
        bridge_defer = bool(
            float(no_assign_record["dispatch_composite_score"])
            < float(bridge_best_assignment["dispatch_composite_score"])
        )
        actual_defer = (
            bridge_defer
            if self.dispatch_potential_mode == "bridge"
            else raw_defer
        )
        selected_assignment = (
            bridge_best_assignment
            if self.dispatch_potential_mode == "bridge"
            else raw_best_assignment
        )
        selected_record = (
            no_assign_record if actual_defer else selected_assignment
        )
        dominance_margin = float(
            no_assign_record["dispatch_composite_score"]
            - bridge_best_assignment["dispatch_composite_score"]
        )
        audit = {
            **terms.to_dict(),
            "mode": self.dispatch_potential_mode,
            "wm_raw_range": float(raw_range),
            "wm_group_span": float(group_span),
            "raw_wm_defer_selected": raw_defer,
            "bridge_defer_selected": bridge_defer,
            "actual_defer_selected": actual_defer,
            "dominance_margin": dominance_margin,
        }
        for record in raw_records:
            record["dispatch_potential"] = audit

        self.stats["dispatch_contexts"] += 1
        self.stats["dispatch_candidates"] += len(raw_records)
        self.stats["dispatch_relative_gap_sum"] += float(
            terms.relative_defer_minus_assign
        )
        if raw_defer:
            self.stats["dispatch_raw_wm_defer_selected"] += 1
        if bridge_defer:
            self.stats["dispatch_bridge_defer_selected"] += 1
        if raw_defer != bridge_defer:
            self.stats["dispatch_modified_decisions"] += 1
        if actual_defer:
            self.stats["dispatch_actual_defer_selected"] += 1
        else:
            self.stats["dispatch_actual_assignment_selected"] += 1
        if terms.crossing_reached:
            self.stats["dispatch_crossing_contexts"] += 1
            if bridge_defer:
                self.stats["dispatch_crossing_violations"] += 1
            else:
                self.stats["dispatch_crossing_assignment_selected"] += 1
        if terms.potential_dominates_wm:
            self.stats["dispatch_dominance_contexts"] += 1
            if raw_defer:
                self.stats["dispatch_dominance_raw_wm_defer"] += 1
            if bridge_defer:
                self.stats["dispatch_dominance_final_defer"] += 1
            else:
                self.stats["dispatch_dominance_final_assignment"] += 1
        return {
            "selected_record": selected_record,
            "selected_assignment": selected_assignment,
            "actual_defer": actual_defer,
            "raw_defer": raw_defer,
            "bridge_defer": bridge_defer,
            "audit": audit,
        }

    def _load_work_drift_head(self):
        if self.work_drift_mode in ("off", "analytic_h5_group_range"):
            return
        head_path = str(self.work_drift_head_path)
        report_path = str(self.work_drift_layer4_report_path)
        for path in (head_path, report_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"required work-drift certification artifact not found: {path}"
                )
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("schema_version") != "work_drift_group_range_evaluation_v1":
            raise ValueError("unexpected Layer-4 work-drift report schema")
        if (
            report.get("status")
            != "WORLD_MODEL_WORK_GROUP_RANGE_ESTIMATION_SUPPORTED"
            or not bool(report.get("passed"))
            or not bool(report.get("supported"))
        ):
            raise ValueError("work-drift scoring requires a passed Layer-4 report")

        from WorldModel.core.work_drift_head import WorkDriftHead
        head, payload = WorkDriftHead.from_checkpoint(head_path)
        actual_head_hash = self._sha256_file(head_path)
        certified_head_hash = str(
            report.get("head_checkpoint", {}).get("sha256", "")
        )
        if actual_head_hash != certified_head_hash:
            raise ValueError(
                "work-drift head SHA256 does not match the held-out Layer-4 report"
            )
        actual_wm_hash = self._sha256_file(self.checkpoint_path)
        certified_wm_hash = str(
            report.get("base_world_model", {}).get("sha256", "")
        )
        payload_wm_hash = str(
            payload.get("base_world_model", {}).get("sha256", "")
        )
        if actual_wm_hash != certified_wm_hash or actual_wm_hash != payload_wm_hash:
            raise ValueError(
                "base World Model SHA256 differs from the head/Layer-4 certificate"
            )
        protocol = payload.get("formal_protocol", {})
        report_protocol = report.get("formal_protocol", {})
        if (
            protocol.get("protocol_sha256")
            != report_protocol.get("protocol_sha256")
        ):
            raise ValueError(
                "work-drift checkpoint and Layer-4 report use different protocols"
            )
        if int(protocol.get("horizon", -1)) != 10:
            raise ValueError("Layer-5 work-drift scoring is frozen to H=10")
        forbidden = protocol.get("forbidden", {})
        if not all(bool(forbidden.get(name)) for name in (
            "unknown_future_orders",
            "future_demand_predictor",
            "continuation_policy",
            "td_risk_v_head",
            "greedy_or_external_assignment_policy",
            "hard_gap_or_load_gate",
        )):
            raise ValueError("work-drift checkpoint lacks the frozen isolation contract")
        if not bool(payload.get("online_ready", False)):
            if not self.work_drift_layer5_evaluation:
                raise ValueError(
                    "the head is not deployment-certified; enable it only inside "
                    "the explicit Layer-5 evaluation protocol"
                )
        if head.num_stations != len(self._station_node_ids):
            raise ValueError("work-drift head station count does not match the map")
        schema_ids = sorted(
            int(value) for value in payload.get("work_schema", {}).get(
                "station_ids", ()
            )
        )
        if schema_ids != self._station_ids:
            raise ValueError(
                "work-drift head station ledger IDs do not match the online map"
            )
        model_device = next(self._model.parameters()).device
        head.to(model_device).eval()
        self._work_drift_head = head
        self._work_drift_payload = payload
        self._work_drift_layer4_report = report
        self._work_drift_horizon = int(protocol["horizon"])

    def _apply_work_drift_group_range(
        self,
        scored_records: List[dict],
    ) -> None:
        """Softly fuse WM and predicted work drift inside one fixed context.

        Both signals use the exact no-gap group-range transform.  Normalising
        the WM score preserves its candidate ordering, while the preregistered
        conservative lambda keeps work drift auxiliary rather than allowing it
        to replace the WM score.  No load, raw-gap, arrival, TD or continuation
        gate enters this operation.
        """
        if self.work_drift_mode == "off" or not scored_records:
            return
        wm_values = [float(row["score"]) for row in scored_records]
        work_values = [float(row["work_drift_raw"]) for row in scored_records]
        wm_group = self._exact_group_range(wm_values)
        work_group = self._exact_group_range(work_values)
        for row, wm_value, work_value in zip(
            scored_records, wm_group, work_group
        ):
            row["work_drift_wm_group_range"] = float(wm_value)
            row["work_drift_group_range"] = float(work_value)
            row["work_drift_combined_score"] = float(
                wm_value + self.work_drift_lambda * work_value
            )

        self.stats["work_drift_contexts"] += 1
        self.stats["work_drift_candidates"] += len(scored_records)
        self.stats["work_drift_max_candidate_count"] = max(
            self.stats["work_drift_max_candidate_count"], len(scored_records)
        )
        if len(scored_records) > 10:
            self.stats["work_drift_candidate_superset_contexts"] += 1
        if max(work_values) - min(work_values) == 0.0:
            self.stats["work_drift_exact_tie_contexts"] += 1
        if max(wm_values) - min(wm_values) == 0.0:
            self.stats["work_drift_wm_exact_tie_contexts"] += 1
        base_best = min(
            scored_records, key=lambda row: (row["score"], row["best_robot"])
        )
        auxiliary_best = min(
            scored_records,
            key=lambda row: (
                row["work_drift_combined_score"], row["score"],
                row["best_robot"],
            ),
        )
        if base_best["best_robot"] != auxiliary_best["best_robot"]:
            self.stats["work_drift_modified_decisions"] += 1

    def _init(self, world):
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                "WorldModelTaskAssigner checkpoint not found; refusing to "
                "construct a random model or fall back to Greedy: "
                f"{self.checkpoint_path}"
            )
        from WorldModel.graph_builder import build_static_graph, FeatureHistory
        from WorldModel.model import RMFSWorldModel

        (self._edge_index, self._node_map, self._inv_node_map,
         self._local_capacity, self._bottleneck_score,
         self._node_type_arr, self._adj) = build_static_graph(world.map_state)
        self._num_nodes = len(self._node_map)
        self._feature_history = FeatureHistory(self._num_nodes, 10, 4)
        if self.dispatch_potential_mode != "off":
            from WorldModel.core.dispatch_potential import (
                DispatchDebtLedger,
                DispatchServiceDurations,
                StaticShortestPathDistance,
            )
            self._dispatch_distance = StaticShortestPathDistance(
                self._node_map,
                self._adj,
            )
            self._dispatch_service_durations = (
                DispatchServiceDurations.from_world(world)
            )
            self._dispatch_debt_ledger = DispatchDebtLedger()

        self._station_ids = sorted(
            int(value) for value in world.map_state.station_positions.keys()
        )
        for station_id in self._station_ids:
            spos = world.map_state.station_positions[station_id]
            nid = self._node_map.get(spos)
            if nid is not None:
                self._station_node_ids.append(nid)

        num_stations = len(world.map_state.station_positions)
        demand_dim = 5 + num_stations

        model_config = {
            "node_feat_dim": 10,
            "edge_feat_dim": 6,
            "demand_dim": demand_dim,
            "action_node_dim": 8,
            "action_global_dim": 6,
            "hidden_dim": self.hidden_dim,
            "num_spatial_layers": 3,
            "rollout_horizon": 10,
            "num_stations": num_stations,
        }

        ckpt = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        self._checkpoint_action_schema = (
            dict(ckpt.get("action_schema") or {})
            if isinstance(ckpt, dict) else {}
        )
        if self.include_no_assign_candidate:
            from WorldModel.candidate_generator import (
                NO_ASSIGN_ACTION_SCHEMA_VERSION,
                NO_ASSIGN_ENCODING,
            )
            schema = self._checkpoint_action_schema
            if (
                schema.get("schema_version")
                != NO_ASSIGN_ACTION_SCHEMA_VERSION
                or not bool(schema.get("supports_no_assign_candidate"))
                or schema.get("no_assign_encoding") != NO_ASSIGN_ENCODING
                or not bool(schema.get("complete_group_coverage"))
                or not bool(schema.get("zero_encoding_verified"))
            ):
                raise ValueError(
                    "checkpoint was not trained on the complete native "
                    "NO_ASSIGN action set; refusing to score unseen all-zero "
                    "actions online"
                )
        if isinstance(ckpt, dict) and "model_config" in ckpt:
            saved_cfg = ckpt["model_config"]
            for k in ("node_feat_dim", "edge_feat_dim", "demand_dim",
                      "action_node_dim", "action_global_dim", "hidden_dim",
                      "num_stations", "rollout_horizon"):
                if k in saved_cfg:
                    model_config[k] = saved_cfg[k]
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        self._model = RMFSWorldModel(**model_config)

        self._model.load_state_dict(state_dict)

        self._model.eval()
        if self.work_drift_mode == "analytic_h5_group_range":
            from WorldModel.core.analytic_work_relief import (
                WorkServiceDurations,
            )
            self._analytic_work_service_durations = (
                WorkServiceDurations.from_world(world)
            )
        self._load_work_drift_head()
        self._initialized = True

    def _update_features(self, world):
        from WorldModel.graph_builder import extract_node_features

        self._flow_counter = {}
        for agent in world.agents:
            nid = self._node_map.get(agent.position)
            if nid is not None:
                self._flow_counter[nid] = self._flow_counter.get(nid, 0.0) + 1.0

        nf = extract_node_features(
            world, self._node_map, self._local_capacity,
            self._bottleneck_score, self._node_type_arr,
            self._adj, self._flow_counter,
            reservation_window=self.reservation_window,
        )
        self._feature_history.push(nf)

        if world.tick != self._last_edge_flow_update_tick:
            self._last_edge_flow_update_tick = world.tick
            for key in self._edge_flow_counter:
                self._edge_flow_counter[key] *= 0.85
            for agent in world.agents:
                if agent.moved_this_tick and agent.previous_position != agent.position:
                    prev_nid = self._node_map.get(agent.previous_position)
                    cur_nid = self._node_map.get(agent.position)
                    if prev_nid is not None and cur_nid is not None:
                        edge_key = (prev_nid, cur_nid)
                        self._edge_flow_counter[edge_key] = (
                            self._edge_flow_counter.get(edge_key, 0.0) + 1.0
                        )

    # ------------------------------------------------------------------
    # assign() — route to greedy or model depending on readiness
    # ------------------------------------------------------------------

    def assign(self, world_state) -> List[Task]:
        t_start = time.perf_counter()
        self.stats["assign_calls"] += 1

        if not self._initialized:
            self._init(world_state)

        self._update_features(world_state)

        mode = world_state.config.simulation.task_execution_mode
        if mode == "serial":
            raise RuntimeError(
                "WorldModelTaskAssigner supports parallel fixed-context robot "
                "ranking only; serial mode must not fall back to Greedy"
            )

        dispatch_snapshot = self._prepare_dispatch_snapshot(world_state)
        if not self._feature_history.is_ready:
            self.stats["warmup_defer_calls"] += 1
            self._finalize_dispatch_snapshot(dispatch_snapshot, [])
            self.stats["assignment_time_total_ms"] += (time.perf_counter() - t_start) * 1000
            return []

        self.stats["model_assign_calls"] += 1
        idle_count = len(world_state.get_idle_agents())
        max_contexts = None
        if self.pool_scoring_mode in (
            "potential_rank", "energy_score", "energy_conversion",
            "random_defer_matched",
        ):
            max_contexts = max(
                idle_count,
                int(round(idle_count * self.pool_scoring_context_factor)),
            )
        final_context_budget = (
            max_contexts if max_contexts is not None else idle_count
        )
        proposal_limit = final_context_budget
        if self.candidate_context_mode == "stratified":
            proposal_limit = max(
                final_context_budget,
                int(round(
                    final_context_budget * self.candidate_context_factor
                )),
            )
        if self.dispatch_potential_mode == "bridge":
            from WorldModel.core.dispatch_potential import prioritise_contexts
            contexts = prioritise_contexts(
                dispatch_snapshot,
                proposal_limit,
            )
        else:
            contexts = self.propose_assignment_contexts(
                world_state, max_contexts=proposal_limit
            )
        if (
            self.dispatch_potential_mode != "bridge"
            and self.candidate_context_mode == "stratified"
        ):
            contexts = self._select_context_representatives(
                world_state, contexts, final_context_budget
            )
        if not contexts:
            self._finalize_dispatch_snapshot(dispatch_snapshot, [])
            self.stats["assignment_time_total_ms"] += (time.perf_counter() - t_start) * 1000
            return []
        robot_choices = self.select_robots(world_state, contexts)
        result = self.commit_assignments(world_state, contexts, robot_choices)
        self._finalize_dispatch_snapshot(dispatch_snapshot, result)
        self.stats["assignment_time_total_ms"] += (time.perf_counter() - t_start) * 1000
        return result

    # ------------------------------------------------------------------
    # select_robots() — model-based scoring
    # ------------------------------------------------------------------

    def select_robots(
        self,
        world_state,
        contexts: List[AssignmentContext],
    ) -> Dict[int, int]:
        from WorldModel.graph_builder import (
            extract_edge_features, extract_demand_context,
            build_action_field, compute_preview_legs,
        )
        from WorldModel.candidate_generator import build_candidate_assignment

        node_hist = self._feature_history.get_history()
        nf_latest = self._feature_history.get_latest()
        edge_feat = extract_edge_features(
            self._edge_index, self._node_map, self._inv_node_map,
            self._local_capacity, world_state,
            adj=self._adj,
            edge_flow_counter=self._edge_flow_counter,
            reservation_window=self.reservation_window,
        )
        demand = extract_demand_context(world_state)

        with torch.no_grad():
            z, e_demand, edge_attr = self._model.encode_state(
                node_hist, self._edge_index, edge_feat, demand,
            )

        preview_planner = (
            self.path_planner if self.action_path_mode == 1 else None
        )

        idle_agents = world_state.get_idle_agents()
        used_agents: Set[int] = set()
        choices: Dict[int, int] = {}
        # Greedy comparison belongs in an independent observer/baseline arm,
        # never inside the World Model decision call stack.
        shadow_greedy_choices: Dict[int, int] = {}
        topq_records = []
        is_topq_mode = self.risk_defer_mode in ("risk_topq", "random_topq")
        is_rolling_mode = self.risk_defer_mode in ("risk_rolling", "random_rolling")
        use_station_terms = (
            self.station_injection_weight > 0.0
            or self.potential_guard_mode != "off"
            or self.pool_scoring_mode != "off"
            or self.decision_trace_enabled
            or self.candidate_set_guard_mode != "off"
            or self.margin_substitute_mode != "off"
            or self.energy_scoring_mode != "off"
        )
        if use_station_terms:
            injection_snapshot = self._station_injection_snapshot(world_state)
            map_rows = len(world_state.map_state.grid)
            map_cols = len(world_state.map_state.grid[0]) if map_rows > 0 else 1
            map_diameter = float(max(map_rows + map_cols - 2, 1))
            potential_state_stress = self._station_global_stress(
                injection_snapshot
            )
        else:
            injection_snapshot = {}
            map_diameter = 1.0
            potential_state_stress = 0.0

        lyapunov_snapshot = None
        if (
            self.lyapunov_l0_mode != "off"
            or self.candidate_robot_mode != "nearest"
            or self.work_drift_mode != "off"
        ):
            from WorldModel.core.lyapunov import compute_lyapunov_snapshot
            lyapunov_snapshot = compute_lyapunov_snapshot(
                world_state, self.lyapunov_l0_config
            )
            self.stats["lyapunov_l0_state_count"] += 1
            self.stats["lyapunov_l0_state_sum"] += lyapunov_snapshot.total

        current_station_work = None
        if self.work_drift_mode != "off":
            if lyapunov_snapshot is None:
                raise RuntimeError("work-drift scoring was not initialised")
            if self.work_drift_mode == "group_range_additive":
                if self._work_drift_head is None:
                    raise RuntimeError("learned work-drift head was not initialised")
                current_station_work = torch.tensor(
                    [
                        float(
                            lyapunov_snapshot.station_work.get(station_id, 0.0)
                        )
                        for station_id in self._station_ids
                    ],
                    dtype=z.dtype,
                    device=z.device,
                )
                certified_capacity = self._work_drift_head.work_capacity.to(
                    dtype=z.dtype, device=z.device
                )
                if not bool(torch.allclose(
                    certified_capacity,
                    torch.full_like(
                        certified_capacity,
                        float(lyapunov_snapshot.work_capacity),
                    ),
                    rtol=0.0,
                    atol=1e-6,
                )):
                    raise RuntimeError(
                        "online analytic work capacity differs from the "
                        "certified head"
                    )
            elif self._analytic_work_service_durations is None:
                raise RuntimeError(
                    "analytic H=5 work durations were not initialised"
                )

        def note_defer_candidate(score_for_defer: float):
            self.stats["risk_defer_contexts"] += 1
            self.stats["risk_defer_score_sum"] += score_for_defer
            self.stats["risk_defer_score_max"] = max(
                self.stats["risk_defer_score_max"], score_for_defer
            )
            self.stats["risk_defer_score_min"] = min(
                self.stats["risk_defer_score_min"], score_for_defer
            )

        def note_potential_guard_candidate(score_for_guard: float):
            self.stats["potential_guard_contexts"] += 1
            self.stats["potential_guard_score_sum"] += score_for_guard
            self.stats["potential_guard_score_max"] = max(
                self.stats["potential_guard_score_max"], score_for_guard
            )
            self.stats["potential_guard_score_min"] = min(
                self.stats["potential_guard_score_min"], score_for_guard
            )

        def serializable_pos(pos):
            if pos is None:
                return None
            return [int(pos[0]), int(pos[1])]

        def record_decision_trace(
            context_idx: int,
            ctx: AssignmentContext,
            candidates: List,
            candidate_ids: List[int],
            local_greedy_robot: Optional[int],
            local_greedy_dist: int,
            scored_records: List[dict],
            selected_record: Optional[dict],
            mode: str,
            action_status: str = "would_execute",
            native_no_assign_record: Optional[dict] = None,
        ):
            if not self.decision_trace_enabled:
                return
            if not scored_records:
                self._append_decision_trace({
                    "tick": int(getattr(world_state, "tick", -1)),
                    "context_idx": int(context_idx),
                    "order_id": int(ctx.order_id),
                    "pod_id": int(ctx.pod_id),
                    "station_id": int(ctx.station_id),
                    "candidate_count": int(len(candidates)),
                    "scored_candidate_count": 0,
                    "mode": mode,
                    "action_status": "no_scored_candidate",
                })
                return

            base_ranked = sorted(
                scored_records, key=lambda r: (r["score"], r["best_robot"])
            )
            potential_ranked = sorted(
                scored_records,
                key=lambda r: (r["potential_guard_score"], r["best_robot"]),
            )
            baseline = base_ranked[0]
            best_potential = potential_ranked[0]
            denom = max(len(scored_records) - 1, 1)
            potential_rank_by_robot = {
                r["best_robot"]: float(rank) / float(denom)
                for rank, r in enumerate(potential_ranked)
            }
            base_rank_by_robot = {
                r["best_robot"]: float(rank) / float(denom)
                for rank, r in enumerate(base_ranked)
            }
            potential_values = [
                r["potential_guard_score"] for r in scored_records
            ]
            score_values = [r["score"] for r in scored_records]
            risk_values = [r["risk_defer_score"] for r in scored_records]
            selected = selected_record or baseline
            baseline_potential = baseline["potential_guard_score"]
            min_potential = best_potential["potential_guard_score"]
            selected_robot = selected.get("best_robot")

            trace_candidates = []
            for r in scored_records:
                trace_candidates.append({
                    "robot_id": int(r["best_robot"]),
                    "robot_start": serializable_pos(r.get("robot_start")),
                    "score": self._scalar(r.get("score")),
                    "base_score": self._scalar(r.get("base_score")),
                    "score_conv": self._scalar(r.get("score_conv")),
                    "energy_conv_signal": self._scalar(
                        r.get("energy_conv_signal")
                    ),
                    "energy_conv_delta_v": self._scalar(
                        r.get("energy_conv_delta_v")
                    ),
                    "energy_conv_z_base": self._scalar(
                        r.get("energy_conv_z_base")
                    ),
                    "energy_conv_z_delta": self._scalar(
                        r.get("energy_conv_z_delta")
                    ),
                    "energy_conv_gate_g": self._scalar(
                        r.get("energy_conv_gate_g")
                    ),
                    "energy_conv_v_state": self._scalar(
                        r.get("energy_conv_v_state")
                    ),
                    "energy_prio_priority": self._scalar(
                        r.get("energy_prio_priority")
                    ),
                    "energy_prio_relief_raw": self._scalar(
                        r.get("energy_prio_relief_raw")
                    ),
                    "energy_prio_relief_n": self._scalar(
                        r.get("energy_prio_relief_n")
                    ),
                    "energy_prio_drift_raw": self._scalar(
                        r.get("energy_prio_drift_raw")
                    ),
                    "energy_prio_drift_n": self._scalar(
                        r.get("energy_prio_drift_n")
                    ),
                    "energy_prio_noop_raw": self._scalar(
                        r.get("energy_prio_noop_raw")
                    ),
                    "energy_prio_noop_n": self._scalar(
                        r.get("energy_prio_noop_n")
                    ),
                    "energy_prio_mu_t": self._scalar(
                        r.get("energy_prio_mu_t")
                    ),
                    "energy_prio_v_backlog": self._scalar(
                        r.get("energy_prio_v_backlog")
                    ),
                    "wm_cost": self._scalar(r.get("wm_cost")),
                    "dispatch_wm_group_range": self._scalar(
                        r.get("dispatch_wm_group_range")
                    ),
                    "dispatch_delta_l": self._scalar(
                        r.get("dispatch_delta_l")
                    ),
                    "dispatch_composite_score": self._scalar(
                        r.get("dispatch_composite_score")
                    ),
                    "dispatch_potential": r.get("dispatch_potential"),
                    "work_drift_raw": self._scalar(
                        r.get("work_drift_raw")
                    ),
                    "work_drift_wm_group_range": self._scalar(
                        r.get("work_drift_wm_group_range")
                    ),
                    "work_drift_group_range": self._scalar(
                        r.get("work_drift_group_range")
                    ),
                    "work_drift_combined_score": self._scalar(
                        r.get("work_drift_combined_score")
                    ),
                    "work_drift_semantics": r.get(
                        "work_drift_semantics", "off"
                    ),
                    "work_drift_horizon": r.get("work_drift_horizon"),
                    "work_drift_chain_free_flow_work": r.get(
                        "work_drift_chain_free_flow_work"
                    ),
                    "work_drift_candidate_nominal_relief_mass": r.get(
                        "work_drift_candidate_nominal_relief_mass"
                    ),
                    "work_drift_active_pipeline_chains": r.get(
                        "work_drift_active_pipeline_chains"
                    ),
                    "work_drift_current_potential": r.get(
                        "work_drift_current_potential"
                    ),
                    "work_drift_endpoint_potential": r.get(
                        "work_drift_endpoint_potential"
                    ),
                    "risk_defer_score": self._scalar(
                        r.get("risk_defer_score")
                    ),
                    "potential_score": self._scalar(
                        r.get("potential_guard_score")
                    ),
                    "base_rank": self._scalar(
                        base_rank_by_robot.get(r["best_robot"])
                    ),
                    "potential_rank": self._scalar(
                        potential_rank_by_robot.get(r["best_robot"])
                    ),
                    "risk_max": self._scalar(r.get("risk_max")),
                    "long_risk_peak_q90": self._scalar(
                        r.get("long_risk_peak_q90")
                    ),
                    "long_risk_event_logit": self._scalar(
                        r.get("long_risk_event_logit")
                    ),
                    "long_risk_peak_q95": self._scalar(
                        r.get("long_risk_peak_q95")
                    ),
                    "long_risk_cvar_q90": self._scalar(
                        r.get("long_risk_cvar_q90")
                    ),
                    "long_risk_terminal_q90": self._scalar(
                        r.get("long_risk_terminal_q90")
                    ),
                    "long_risk_delta_group_q90": self._scalar(
                        r.get("long_risk_delta_group_q90")
                    ),
                    "long_risk_combo_q90": self._scalar(
                        r.get("long_risk_combo_q90")
                    ),
                    "long_risk_quantile_combo": self._scalar(
                        r.get("long_risk_quantile_combo")
                    ),
                    "station_injection_penalty": self._scalar(
                        r.get("station_injection_penalty")
                    ),
                    "station_injection_raw": self._scalar(
                        r.get("station_injection_raw")
                    ),
                    "station_injection_stress": self._scalar(
                        r.get("station_injection_stress")
                    ),
                    "station_injection_fast": self._scalar(
                        r.get("station_injection_fast")
                    ),
                    "energy_potential": self._scalar(
                        r.get("energy_potential")
                    ),
                    "energy_service_relief": self._scalar(
                        r.get("energy_service_relief")
                    ),
                    "lyapunov_l0_delta": self._scalar(
                        r.get("lyapunov_l0_delta")
                    ),
                    "lyapunov_l0_eta": self._scalar(
                        r.get("lyapunov_l0_eta")
                    ),
                    "lyapunov_l0_eta_bin": int(
                        r.get("lyapunov_l0_eta_bin", -1)
                    ),
                    "lyapunov_l0_delta_semantics": (
                        "eta_arrival_barrier_preview_v1"
                    ),
                    "energy_order_age_norm": self._scalar(
                        r.get("energy_order_age_norm")
                    ),
                    "energy_station_pending_pressure": self._scalar(
                        r.get("energy_station_pending_pressure")
                    ),
                    "energy_global_pending_pressure": self._scalar(
                        r.get("energy_global_pending_pressure")
                    ),
                    "energy_idle_fraction": self._scalar(
                        r.get("energy_idle_fraction")
                    ),
                    "robot_to_pod_dist": int(r.get("robot_to_pod_dist", 0)),
                    "pod_to_station_dist": int(
                        r.get("pod_to_station_dist", 0)
                    ),
                    "route_len": int(r.get("route_len", 0)),
                    "is_selected": r["best_robot"] == selected_robot,
                    "is_base_best": r["best_robot"] == baseline["best_robot"],
                    "is_best_potential": (
                        r["best_robot"] == best_potential["best_robot"]
                    ),
                    "is_local_greedy": r["best_robot"] == local_greedy_robot,
                })
            if native_no_assign_record is not None:
                trace_candidates.append({
                    "action_type": "no_assign",
                    "robot_id": None,
                    "robot_start": None,
                    "score": self._scalar(
                        native_no_assign_record.get("score")
                    ),
                    "base_score": self._scalar(
                        native_no_assign_record.get("base_score")
                    ),
                    "wm_cost": self._scalar(
                        native_no_assign_record.get("wm_cost")
                    ),
                    "dispatch_wm_group_range": self._scalar(
                        native_no_assign_record.get(
                            "dispatch_wm_group_range"
                        )
                    ),
                    "dispatch_delta_l": self._scalar(
                        native_no_assign_record.get("dispatch_delta_l")
                    ),
                    "dispatch_composite_score": self._scalar(
                        native_no_assign_record.get(
                            "dispatch_composite_score"
                        )
                    ),
                    "dispatch_potential": native_no_assign_record.get(
                        "dispatch_potential"
                    ),
                    "risk_max": self._scalar(
                        native_no_assign_record.get("risk_max")
                    ),
                    "is_selected": selected_robot is None,
                    "is_base_best": False,
                    "is_best_potential": False,
                    "is_local_greedy": False,
                })

            self._append_decision_trace({
                "tick": int(getattr(world_state, "tick", -1)),
                "context_idx": int(context_idx),
                "order_id": int(ctx.order_id),
                "pod_id": int(ctx.pod_id),
                "station_id": int(ctx.station_id),
                "pod_location": serializable_pos(ctx.pod_location),
                "station_location": serializable_pos(ctx.station_location),
                "return_location": serializable_pos(ctx.return_location),
                "order_size": int(ctx.order_size),
                "mode": mode,
                "action_status": action_status,
                "candidate_count": int(
                    len(candidates)
                    + (1 if native_no_assign_record is not None else 0)
                ),
                "scored_candidate_count": int(
                    len(scored_records)
                    + (1 if native_no_assign_record is not None else 0)
                ),
                "candidate_ids": [int(v) for v in candidate_ids],
                "local_greedy_robot": (
                    int(local_greedy_robot)
                    if local_greedy_robot is not None else None
                ),
                "selected_robot": (
                    int(selected_robot) if selected_robot is not None else None
                ),
                "selected_action_type": (
                    "no_assign" if selected_robot is None else "assign_robot"
                ),
                "native_no_assign_score": (
                    self._scalar(native_no_assign_record.get("score"))
                    if native_no_assign_record is not None else None
                ),
                "dispatch_potential_mode": self.dispatch_potential_mode,
                "dispatch_potential": (
                    selected.get("dispatch_potential")
                    if selected is not None else None
                ),
                "selected_dispatch_wm_group_range": self._scalar(
                    selected.get("dispatch_wm_group_range")
                    if selected is not None else None
                ),
                "selected_dispatch_delta_l": self._scalar(
                    selected.get("dispatch_delta_l")
                    if selected is not None else None
                ),
                "selected_dispatch_composite_score": self._scalar(
                    selected.get("dispatch_composite_score")
                    if selected is not None else None
                ),
                "work_drift_mode": self.work_drift_mode,
                "work_drift_lambda": (
                    float(self.work_drift_lambda)
                    if self.work_drift_mode != "off" else 0.0
                ),
                "work_drift_horizon": (
                    int(self._work_drift_horizon)
                    if self.work_drift_mode != "off" else None
                ),
                "work_drift_modified_decision": (
                    self.work_drift_mode != "off"
                    and selected_robot is not None
                    and int(selected_robot) != int(baseline["best_robot"])
                ),
                "baseline_robot": int(baseline["best_robot"]),
                "best_potential_robot": int(best_potential["best_robot"]),
                "baseline_base_score": self._scalar(baseline.get("score")),
                "selected_base_score": self._scalar(selected.get("score")),
                "baseline_work_drift_raw": self._scalar(
                    baseline.get("work_drift_raw")
                ),
                "selected_work_drift_raw": self._scalar(
                    selected.get("work_drift_raw")
                ),
                "baseline_work_drift_combined_score": self._scalar(
                    baseline.get("work_drift_combined_score")
                ),
                "selected_work_drift_combined_score": self._scalar(
                    selected.get("work_drift_combined_score")
                ),
                "baseline_score_conv": self._scalar(
                    baseline.get("score_conv")
                ),
                "selected_score_conv": self._scalar(
                    selected.get("score_conv")
                ),
                "energy_conv_gate_g": self._scalar(
                    selected.get("energy_conv_gate_g")
                ),
                "energy_conv_v_state": self._scalar(
                    selected.get("energy_conv_v_state")
                ),
                "energy_prio_priority": self._scalar(
                    selected.get("energy_prio_priority")
                ),
                "energy_prio_relief_raw": self._scalar(
                    selected.get("energy_prio_relief_raw")
                ),
                "energy_prio_relief_n": self._scalar(
                    selected.get("energy_prio_relief_n")
                ),
                "energy_prio_drift_raw": self._scalar(
                    selected.get("energy_prio_drift_raw")
                ),
                "energy_prio_drift_n": self._scalar(
                    selected.get("energy_prio_drift_n")
                ),
                "energy_prio_noop_raw": self._scalar(
                    selected.get("energy_prio_noop_raw")
                ),
                "energy_prio_noop_n": self._scalar(
                    selected.get("energy_prio_noop_n")
                ),
                "energy_prio_mu_t": self._scalar(
                    selected.get("energy_prio_mu_t")
                ),
                "lyapunov_l0_delta": self._scalar(
                    selected.get("lyapunov_l0_delta")
                ),
                "lyapunov_l0_eta": self._scalar(
                    selected.get("lyapunov_l0_eta")
                ),
                "lyapunov_l0_eta_bin": int(
                    selected.get("lyapunov_l0_eta_bin", -1)
                ),
                "lyapunov_l0_delta_semantics": (
                    "eta_arrival_barrier_preview_v1"
                ),
                "energy_prio_v_backlog": self._scalar(
                    selected.get("energy_prio_v_backlog")
                ),
                "candidate_min_base": self._scalar(min(score_values)),
                "candidate_max_base": self._scalar(max(score_values)),
                "base_margin": self._scalar(
                    base_ranked[1]["score"] - base_ranked[0]["score"]
                    if len(base_ranked) >= 2 else 0.0
                ),
                "baseline_potential": self._scalar(baseline_potential),
                "selected_potential": self._scalar(
                    selected.get("potential_guard_score")
                ),
                "candidate_min_potential": self._scalar(min_potential),
                "candidate_max_potential": self._scalar(max(potential_values)),
                "candidate_potential_spread": self._scalar(
                    max(potential_values) - min(potential_values)
                ),
                "potential_gain_vs_baseline": self._scalar(
                    baseline_potential - min_potential
                ),
                "best_potential_base_gap": self._scalar(
                    best_potential["score"] - baseline["score"]
                ),
                "baseline_potential_rank": self._scalar(
                    potential_rank_by_robot.get(baseline["best_robot"])
                ),
                "selected_potential_rank": self._scalar(
                    potential_rank_by_robot.get(selected_robot)
                ),
                "baseline_is_best_potential": (
                    baseline["best_robot"] == best_potential["best_robot"]
                ),
                "candidate_min_risk_score": self._scalar(min(risk_values)),
                "candidate_max_risk_score": self._scalar(max(risk_values)),
                "candidate_risk_spread": self._scalar(
                    max(risk_values) - min(risk_values)
                ),
                "candidates": trace_candidates,
            })

        def summarize_context_records(scored_records: List[dict]) -> Optional[dict]:
            if not scored_records:
                return None
            base_ranked = sorted(
                scored_records, key=lambda r: (r["score"], r["best_robot"])
            )
            potential_ranked = sorted(
                scored_records,
                key=lambda r: (r["potential_guard_score"], r["best_robot"]),
            )
            baseline = base_ranked[0]
            best_potential = potential_ranked[0]
            potential_values = [
                r["potential_guard_score"] for r in scored_records
            ]
            risk_values = [r["risk_defer_score"] for r in scored_records]
            return {
                "baseline": baseline,
                "best_potential": best_potential,
                "candidate_min_potential": min(potential_values),
                "candidate_min_risk_score": min(risk_values),
                "potential_gain_vs_baseline": (
                    baseline["potential_guard_score"]
                    - best_potential["potential_guard_score"]
                ),
                "best_potential_base_gap": (
                    best_potential["score"] - baseline["score"]
                ),
                "baseline_is_best_potential": (
                    baseline["best_robot"] == best_potential["best_robot"]
                ),
            }

        def apply_energy_noop(
            ctx: AssignmentContext,
            scored_records: List[dict],
        ) -> tuple[bool, str, float]:
            if self.energy_scoring_mode != "with_noop":
                return False, "off", 0.0
            if not scored_records:
                return False, "no_records", 0.0

            features = self._context_opportunity_features(
                world_state, ctx, injection_snapshot
            )
            noop_cost = self._energy_noop_cost(features)
            min_base_score = min(r["base_score"] for r in scored_records)
            best_action_score = min(r["score"] for r in scored_records)
            noop_score = (
                min_base_score
                + self.energy_noop_base_margin
                + self.energy_noop_cost_scale * noop_cost
            )
            self.stats["energy_noop_contexts"] += 1
            self.stats["energy_noop_score_sum"] += noop_score
            self.stats["energy_noop_cost_sum"] += noop_cost
            self.stats[
                "energy_noop_best_action_score_sum"
            ] += best_action_score
            if noop_score <= best_action_score:
                self.stats["energy_noop_deferred"] += 1
                return True, "energy_noop", noop_score
            return False, "execute", noop_score

        def apply_candidate_set_guard(summary: dict) -> tuple[bool, str]:
            if self.candidate_set_guard_mode == "off":
                return False, "off"

            min_potential = float(summary["candidate_min_potential"])
            min_risk = float(summary["candidate_min_risk_score"])
            self.stats["candidate_set_guard_contexts"] += 1
            self.stats["candidate_set_guard_min_potential_sum"] += min_potential
            self.stats["candidate_set_guard_min_risk_sum"] += min_risk

            use_potential = self.candidate_set_guard_features in (
                "either", "potential"
            )
            use_risk = self.candidate_set_guard_features in ("either", "risk")
            potential_ready = (
                not use_potential
                or len(self._candidate_set_guard_potential_history)
                >= self.candidate_set_guard_min_history
            )
            risk_ready = (
                not use_risk
                or len(self._candidate_set_guard_risk_history)
                >= self.candidate_set_guard_min_history
            )

            should_defer = False
            reason = "executed"
            if not (potential_ready and risk_ready):
                self.stats["candidate_set_guard_warmup"] += 1
                reason = "warmup"
            else:
                potential_threshold = None
                risk_threshold = None
                potential_triggered = False
                risk_triggered = False
                if use_potential:
                    potential_threshold = self._quantile_from_history(
                        self._candidate_set_guard_potential_history,
                        self.candidate_set_guard_rolling_quantile,
                    )
                    self.stats[
                        "candidate_set_guard_potential_threshold_sum"
                    ] += float(potential_threshold)
                    if min_potential > potential_threshold:
                        potential_triggered = True
                        self.stats[
                            "candidate_set_guard_potential_triggered"
                        ] += 1
                if use_risk:
                    risk_threshold = self._quantile_from_history(
                        self._candidate_set_guard_risk_history,
                        self.candidate_set_guard_rolling_quantile,
                    )
                    self.stats[
                        "candidate_set_guard_risk_threshold_sum"
                    ] += float(risk_threshold)
                    if min_risk > risk_threshold:
                        risk_triggered = True
                        self.stats["candidate_set_guard_risk_triggered"] += 1
                self.stats["candidate_set_guard_threshold_count"] += 1

                if self.candidate_set_guard_mode == "rolling":
                    should_defer = potential_triggered or risk_triggered
                    if should_defer:
                        reason = (
                            "potential_and_risk"
                            if potential_triggered and risk_triggered
                            else "potential"
                            if potential_triggered
                            else "risk"
                        )
                else:
                    if (self._random_defer_rng.random()
                            > self.candidate_set_guard_rolling_quantile):
                        should_defer = True
                        reason = "random"
                        self.stats["candidate_set_guard_random_deferred"] += 1

            self._candidate_set_guard_potential_history.append(min_potential)
            self._candidate_set_guard_risk_history.append(min_risk)
            if should_defer:
                self.stats["candidate_set_guard_deferred"] += 1
            else:
                self.stats["candidate_set_guard_executed"] += 1
            return should_defer, reason

        def apply_margin_substitute(summary: dict) -> tuple[dict, str]:
            baseline = summary["baseline"]
            if self.margin_substitute_mode == "off":
                return baseline, "off"

            self.stats["margin_substitute_contexts"] += 1
            if summary["baseline_is_best_potential"]:
                return baseline, "baseline_is_best_potential"

            best_potential = summary["best_potential"]
            gap = float(summary["best_potential_base_gap"])
            gain = float(summary["potential_gain_vs_baseline"])
            self.stats["margin_substitute_opportunities"] += 1
            self.stats["margin_substitute_gap_sum"] += gap
            self.stats["margin_substitute_gain_sum"] += gain

            if (len(self._margin_substitute_gap_history)
                    < self.margin_substitute_min_history):
                self.stats["margin_substitute_warmup"] += 1
                self._margin_substitute_gap_history.append(gap)
                return baseline, "warmup"

            gap_threshold = self._quantile_from_history(
                self._margin_substitute_gap_history,
                self.margin_substitute_gap_quantile,
            )
            self.stats["margin_substitute_gap_threshold_sum"] += float(
                gap_threshold
            )
            self.stats["margin_substitute_threshold_count"] += 1
            self._margin_substitute_gap_history.append(gap)

            if gap <= gap_threshold:
                self.stats["margin_substitute_substituted"] += 1
                return best_potential, "substituted"
            return baseline, "gap_too_large"

        def commit_record(record: dict):
            best_robot = record["best_robot"]
            choices[record["context_idx"]] = best_robot
            used_agents.add(best_robot)

            shadow_robot = record["shadow_robot"]
            if shadow_robot is not None:
                self.stats["shadow_greedy_compared"] += 1
                if best_robot == shadow_robot:
                    self.stats["shadow_greedy_match"] += 1

            local_greedy_robot = record["local_greedy_robot"]
            if local_greedy_robot is not None:
                self.stats["local_greedy_compared"] += 1
                if best_robot == local_greedy_robot:
                    self.stats["local_greedy_match"] += 1

            candidate_ids = record["candidate_ids"]
            if best_robot in candidate_ids:
                selected_rank = candidate_ids.index(best_robot) + 1
                self.stats["wm_selected_rank_sum"] += float(selected_rank)
                if selected_rank > 1:
                    self.stats["wm_selected_rank_gt1"] += 1
                selected_agent = record["candidates"][selected_rank - 1]
                extra_dist = (
                    _manhattan_distance(selected_agent.position, record["pod_location"])
                    - record["local_greedy_dist"]
                )
                self.stats["wm_selected_extra_distance_sum"] += float(extra_dist)
                if extra_dist > 0:
                    self.stats["wm_selected_extra_distance_positive"] += 1

            if (self.station_injection_weight > 0.0
                    and "station_injection_penalty" in record):
                self.stats["station_injection_selected"] += 1
                self.stats["station_injection_selected_penalty_sum"] += (
                    record["station_injection_penalty"]
                )
                self.stats["station_injection_selected_stress_sum"] += (
                    record.get("station_injection_stress", 0.0)
                )
                self.stats["station_injection_selected_fast_sum"] += (
                    record.get("station_injection_fast", 0.0)
                )
            if self.lyapunov_l0_mode != "off":
                self.stats["lyapunov_l0_arrival_selected"] += 1
                self.stats[
                    "lyapunov_l0_arrival_selected_delta_sum"
                ] += float(
                    record.get("lyapunov_l0_delta", 0.0)
                )
                self.stats[
                    "lyapunov_l0_arrival_selected_eta_sum"
                ] += float(record.get("lyapunov_l0_eta", 0.0))
            if self.work_drift_mode != "off":
                self.stats["work_drift_selected_group_range_sum"] += float(
                    record.get("work_drift_group_range", 0.0)
                )
                self.stats["work_drift_selected_raw_sum"] += float(
                    record.get("work_drift_raw", 0.0)
                )

        def score_candidate(
            context_idx: int,
            ctx: AssignmentContext,
            agent,
            candidates: List,
            candidate_ids: List[int],
            local_greedy_robot: Optional[int],
            local_greedy_dist: int,
            injection_score_records: List[dict],
        ) -> Optional[dict]:
            cand = {
                "robot_id": agent.agent_id,
                "robot_start": agent.position,
            }
            fc = {
                "order_id": ctx.order_id,
                "pod_id": ctx.pod_id,
                "pod_location": ctx.pod_location,
                "station_id": ctx.station_id,
                "station_location": ctx.station_location,
                "entry_position": ctx.entry_position,
                "exit_position": ctx.exit_position,
                "return_location": ctx.return_location,
                "order_size": ctx.order_size,
            }
            assignment = build_candidate_assignment(cand, fc)
            legs = compute_preview_legs(
                assignment, world_state.map_state, self._node_map,
                path_planner=preview_planner, world=world_state,
            )
            action_node, action_global = build_action_field(
                assignment, world_state, self._node_map,
                self._inv_node_map, self._local_capacity,
                node_features=nf_latest,
                precomputed_legs=legs,
            )

            t_inf = time.perf_counter()
            with torch.no_grad():
                cost, details = self._model.predict_cost(
                    z, e_demand, edge_attr,
                    action_node, action_global, self._edge_index,
                    self._station_node_ids,
                    return_details=True,
                    risk_weight=self.risk_weight,
                )
            self.stats["model_inference_calls"] += 1
            self.stats["model_inference_time_total_ms"] += (
                time.perf_counter() - t_inf
            ) * 1000

            work_drift_raw = 0.0
            work_drift_audit = {
                "work_drift_semantics": "off",
                "work_drift_horizon": None,
                "work_drift_chain_free_flow_work": None,
                "work_drift_candidate_nominal_relief_mass": None,
                "work_drift_active_pipeline_chains": None,
                "work_drift_current_potential": None,
                "work_drift_endpoint_potential": None,
            }
            if self.work_drift_mode == "group_range_additive":
                z_endpoint = details.get("z_endpoint")
                if z_endpoint is None or current_station_work is None:
                    raise RuntimeError(
                        "World Model did not expose its action-conditioned endpoint"
                    )
                score_horizon = int(details.get("rollout_horizon", -1))
                if score_horizon <= 0:
                    raise RuntimeError("World Model omitted its score horizon")
                if score_horizon > self._work_drift_horizon:
                    raise RuntimeError(
                        "base WM score horizon exceeds the certified work horizon"
                    )
                with torch.no_grad():
                    if score_horizon < self._work_drift_horizon:
                        # Preserve the checkpoint's native cost horizon while
                        # extending only the same frozen action-conditioned
                        # latent trajectory to the certified H=10 endpoint.
                        z_endpoint = self._model.continue_latent_rollout(
                            z_endpoint,
                            e_demand,
                            edge_attr,
                            action_node,
                            action_global,
                            self._edge_index,
                            start_step=score_horizon,
                            end_step=self._work_drift_horizon,
                        )
                    work_prediction = self._work_drift_head(
                        z,
                        z_endpoint,
                        e_demand,
                        self._station_node_ids,
                        current_station_work,
                    )
                work_drift_raw = self._scalar(
                    work_prediction.raw_work_drift
                )
                work_drift_audit.update({
                    "work_drift_semantics": "learned_endpoint_head_h10_v1",
                    "work_drift_horizon": int(self._work_drift_horizon),
                })
            elif self.work_drift_mode == "analytic_h5_group_range":
                from WorldModel.core.analytic_work_relief import (
                    compute_virtual_candidate_work_drift,
                )
                analytic = compute_virtual_candidate_work_drift(
                    lyapunov_snapshot,
                    candidate_info=cand,
                    fixed_context=fc,
                    service_durations=self._analytic_work_service_durations,
                    horizon=self._work_drift_horizon,
                    work_weight=self.lyapunov_l0_config.work_weight,
                )
                work_drift_raw = float(analytic.raw_work_drift)
                work_drift_audit.update({
                    "work_drift_semantics": (
                        "parameter_free_virtual_post_action_analytic_h5_v1"
                    ),
                    "work_drift_horizon": int(analytic.horizon),
                    "work_drift_chain_free_flow_work": float(
                        analytic.chain_free_flow_work
                    ),
                    "work_drift_candidate_nominal_relief_mass": float(
                        analytic.candidate_nominal_relief_mass
                    ),
                    "work_drift_active_pipeline_chains": int(
                        analytic.active_pipeline_chains
                    ),
                    "work_drift_current_potential": float(
                        analytic.current_work_potential
                    ),
                    "work_drift_endpoint_potential": float(
                        analytic.endpoint_work_potential
                    ),
                })

            if self.work_drift_mode != "off":
                self.stats["work_drift_raw_sum"] += work_drift_raw
                self.stats["work_drift_raw_min"] = min(
                    self.stats["work_drift_raw_min"], work_drift_raw
                )
                self.stats["work_drift_raw_max"] = max(
                    self.stats["work_drift_raw_max"], work_drift_raw
                )

            risk_max = self._scalar(details.get("risk_max"))
            self.stats["risk_max_count"] += 1
            self.stats["risk_max_sum"] += risk_max
            self.stats["risk_max_max"] = max(
                self.stats["risk_max_max"], risk_max
            )
            self.stats["risk_max_min"] = min(
                self.stats["risk_max_min"], risk_max
            )
            self.risk_max_values.append(risk_max)

            if self.risk_threshold is not None:
                if risk_max > self.risk_threshold:
                    self.stats["risk_guard_filtered"] += 1
                    return None

            wm_cost = self._scalar(cost)
            base_score = wm_cost
            risk_defer_score = self._compute_risk_defer_score(details)
            lr_components = self._long_risk_components(details)

            # B2: Add long-risk penalties if betas are configured.
            if self.long_risk_beta:
                base_score += (
                    self.long_risk_beta.get("peak", 0.0)
                    * lr_components["long_risk_peak_q95"]
                )
                base_score += (
                    self.long_risk_beta.get("cvar", 0.0)
                    * lr_components["long_risk_cvar_q90"]
                )
                base_score += (
                    self.long_risk_beta.get("terminal", 0.0)
                    * lr_components["long_risk_terminal_q90"]
                )
                base_score += (
                    self.long_risk_beta.get("delta", 0.0)
                    * max(0.0, lr_components["long_risk_delta_group_q90"])
                )

            injection_penalty, injection_raw, injection_stress, injection_fast = (
                self._station_injection_penalty(
                    agent, ctx, injection_snapshot, map_diameter
                )
            )
            potential_guard_score = self._compute_potential_guard_score(
                details,
                potential_state_stress,
                injection_stress,
                injection_raw,
            )
            score = base_score
            if self.station_injection_weight > 0.0:
                with_injection_score = base_score + injection_penalty
                if not self.station_injection_diagnostic_only:
                    score = with_injection_score
                self.stats["station_injection_candidates"] += 1
                self.stats["station_injection_penalty_sum"] += injection_penalty
                self.stats["station_injection_penalty_max"] = max(
                    self.stats["station_injection_penalty_max"],
                    injection_penalty,
                )
                self.stats["station_injection_stress_sum"] += injection_stress
                self.stats["station_injection_fast_sum"] += injection_fast
                injection_score_records.append({
                    "robot_id": agent.agent_id,
                    "base_score": base_score,
                    "with_injection_score": with_injection_score,
                    "station_injection_penalty": injection_penalty,
                })

            lyapunov_l0_delta = 0.0
            lyapunov_l0_eta = 0.0
            lyapunov_l0_eta_bin = -1
            if self.lyapunov_l0_mode != "off":
                from WorldModel.core.lyapunov import preview_assignment_context
                l0_preview = preview_assignment_context(
                    world_state,
                    ctx,
                    agent,
                    config=self.lyapunov_l0_config,
                    snapshot=lyapunov_snapshot,
                )
                lyapunov_l0_delta = float(l0_preview.delta_total)
                lyapunov_l0_eta = float(l0_preview.eta)
                lyapunov_l0_eta_bin = int(l0_preview.eta_bin)
                if self.lyapunov_l0_mode == "additive":
                    score += self.lyapunov_l0_lambda * lyapunov_l0_delta
                self.stats["lyapunov_l0_arrival_candidates"] += 1
                self.stats[
                    "lyapunov_l0_arrival_delta_sum"
                ] += lyapunov_l0_delta
                self.stats[
                    "lyapunov_l0_arrival_eta_sum"
                ] += lyapunov_l0_eta
                self.stats["lyapunov_l0_arrival_delta_min"] = min(
                    self.stats["lyapunov_l0_arrival_delta_min"],
                    lyapunov_l0_delta,
                )
                self.stats["lyapunov_l0_arrival_delta_max"] = max(
                    self.stats["lyapunov_l0_arrival_delta_max"],
                    lyapunov_l0_delta,
                )

            energy_potential = 0.0
            energy_service_relief = 0.0
            opportunity_features = {}
            if self.energy_scoring_mode in ("additive", "with_noop"):
                opportunity_features = self._context_opportunity_features(
                    world_state, ctx, injection_snapshot
                )
                energy_potential = self._energy_from_system_preds(
                    details.get("system_preds"), lr_components
                )
                energy_service_relief = self._energy_service_relief(
                    opportunity_features
                )
                score = (
                    score
                    + self.energy_score_lambda * energy_potential
                    - energy_service_relief
                )
                self.stats["energy_scoring_candidates"] += 1
                self.stats["energy_score_sum"] += score
                self.stats["energy_potential_sum"] += energy_potential
                self.stats[
                    "energy_service_relief_sum"
                ] += energy_service_relief

            robot_to_pod_dist = _manhattan_distance(
                agent.position, ctx.pod_location
            )
            pod_to_station_dist = _manhattan_distance(
                ctx.pod_location, ctx.station_location
            )
            route_len = robot_to_pod_dist + pod_to_station_dist

            result = {
                "context_idx": context_idx,
                "best_robot": agent.agent_id,
                "robot_start": agent.position,
                "score": score,
                "base_score": base_score,
                "wm_cost": wm_cost,
                "risk_max": risk_max,
                "risk_defer_score": risk_defer_score,
                "potential_guard_score": potential_guard_score,
                "shadow_robot": shadow_greedy_choices.get(context_idx),
                "local_greedy_robot": local_greedy_robot,
                "candidate_ids": candidate_ids,
                "candidates": candidates,
                "pod_location": ctx.pod_location,
                "local_greedy_dist": local_greedy_dist,
                "station_injection_penalty": injection_penalty,
                "station_injection_raw": injection_raw,
                "station_injection_stress": injection_stress,
                "station_injection_fast": injection_fast,
                "energy_potential": energy_potential,
                "energy_service_relief": energy_service_relief,
                "lyapunov_l0_delta": lyapunov_l0_delta,
                "lyapunov_l0_eta": lyapunov_l0_eta,
                "lyapunov_l0_eta_bin": lyapunov_l0_eta_bin,
                "lyapunov_l0_delta_semantics": (
                    "eta_arrival_barrier_preview_v1"
                ),
                "energy_order_age_norm": opportunity_features.get(
                    "order_age_norm", 0.0
                ),
                "energy_station_pending_pressure": opportunity_features.get(
                    "station_pending_pressure", 0.0
                ),
                "energy_global_pending_pressure": opportunity_features.get(
                    "global_pending_pressure", 0.0
                ),
                "energy_idle_fraction": opportunity_features.get(
                    "idle_fraction", 0.0
                ),
                "robot_to_pod_dist": robot_to_pod_dist,
                "pod_to_station_dist": pod_to_station_dist,
                "route_len": route_len,
                "work_drift_raw": work_drift_raw,
            }
            result.update(work_drift_audit)
            result.update(lr_components)
            return result

        def score_native_no_assign(context_idx: int) -> Optional[dict]:
            """Score native NO_ASSIGN with the exact same frozen WM call."""
            if not self.include_no_assign_candidate:
                return None
            assignment = {
                "action_type": "no_assign",
                "action_schema_version": "wm_native_no_assign_action_v1",
                "action_encoding": "zero_action_tensors_v1",
            }
            action_node, action_global = build_action_field(
                assignment,
                world_state,
                self._node_map,
                self._inv_node_map,
                self._local_capacity,
                node_features=nf_latest,
            )
            t_inf = time.perf_counter()
            with torch.no_grad():
                cost, details = self._model.predict_cost(
                    z,
                    e_demand,
                    edge_attr,
                    action_node,
                    action_global,
                    self._edge_index,
                    self._station_node_ids,
                    return_details=True,
                    risk_weight=None,
                )
            self.stats["model_inference_calls"] += 1
            self.stats["model_inference_time_total_ms"] += (
                time.perf_counter() - t_inf
            ) * 1000
            score = self._scalar(cost)
            risk_max = self._scalar(details.get("risk_max"))
            self.stats["risk_max_count"] += 1
            self.stats["risk_max_sum"] += risk_max
            self.stats["risk_max_max"] = max(
                self.stats["risk_max_max"], risk_max
            )
            self.stats["risk_max_min"] = min(
                self.stats["risk_max_min"], risk_max
            )
            self.risk_max_values.append(risk_max)
            self.stats["native_no_assign_scored"] += 1
            self.stats["native_no_assign_score_sum"] += score
            return {
                "action_type": "no_assign",
                "context_idx": context_idx,
                "best_robot": None,
                "score": score,
                "base_score": score,
                "wm_cost": score,
                "risk_max": risk_max,
            }

        if self.pool_scoring_mode != "off":
            pool_records = []
            per_context_records: Dict[int, List[dict]] = {}

            for i, ctx in enumerate(contexts):
                available = list(idle_agents)
                if not available:
                    break
                candidates = self._select_robot_candidates(
                    world_state, ctx, available, lyapunov_snapshot
                )
                candidate_ids = [a.agent_id for a in candidates]
                local_greedy_robot = None
                local_greedy_dist = (
                    min(
                        _manhattan_distance(agent.position, ctx.pod_location)
                        for agent in candidates
                    )
                    if candidates else 0
                )
                self.stats["decision_contexts_total"] += 1
                self.stats["pool_scoring_contexts"] += 1
                injection_score_records = []
                ctx_scored_records = []

                for agent in candidates:
                    record = score_candidate(
                        i, ctx, agent, candidates, candidate_ids,
                        local_greedy_robot, local_greedy_dist,
                        injection_score_records,
                    )
                    if record is None:
                        continue
                    ctx_scored_records.append(record)

                if self.station_injection_weight > 0.0:
                    self._record_station_injection_diag(injection_score_records)
                if self.energy_scoring_mode == "conversion":
                    self._apply_energy_conversion(ctx_scored_records)
                if self.energy_scoring_mode != "off" and ctx_scored_records:
                    self.stats["energy_scoring_contexts"] += 1
                should_energy_noop, noop_reason, _ = apply_energy_noop(
                    ctx, ctx_scored_records
                )
                if should_energy_noop:
                    record_decision_trace(
                        i, ctx, candidates, candidate_ids,
                        local_greedy_robot, local_greedy_dist,
                        ctx_scored_records, None,
                        mode=self.pool_scoring_mode,
                        action_status="energy_noop_deferred:" + noop_reason,
                    )
                    continue
                for record in ctx_scored_records:
                    pool_records.append(record)
                    per_context_records.setdefault(i, []).append(record)

            if not pool_records:
                return choices

            self.stats["pool_scoring_candidates"] += len(pool_records)

            baseline_records = []
            baseline_used_agents: Set[int] = set()
            for i in range(len(contexts)):
                ctx_records = sorted(
                    per_context_records.get(i, []),
                    key=lambda r: r["score"],
                )
                for record in ctx_records:
                    if record["best_robot"] in baseline_used_agents:
                        continue
                    baseline_records.append(record)
                    baseline_used_agents.add(record["best_robot"])
                    break
            baseline_contexts = {
                record["context_idx"] for record in baseline_records
            }
            baseline_robot_by_context = {
                record["context_idx"]: record["best_robot"]
                for record in baseline_records
            }
            self.stats["pool_scoring_baseline_selected"] += len(
                baseline_records
            )

            if self.pool_scoring_mode == "energy_score":
                selected_records = []
                selected_contexts: Set[int] = set()
                selected_agents: Set[int] = set()
                for record in sorted(
                    pool_records,
                    key=lambda r: (
                        r["score"],
                        r["context_idx"],
                        r["best_robot"],
                    ),
                ):
                    if record["context_idx"] in selected_contexts:
                        continue
                    if record["best_robot"] in selected_agents:
                        continue
                    selected_records.append(record)
                    selected_contexts.add(record["context_idx"])
                    selected_agents.add(record["best_robot"])
                    if len(selected_agents) >= len(idle_agents):
                        break

                self.stats["pool_scoring_selected"] += len(selected_records)
                self.stats["pool_scoring_order_replacements"] += len(
                    selected_contexts - baseline_contexts
                )
                self.stats["pool_scoring_baseline_context_missed"] += len(
                    baseline_contexts - selected_contexts
                )

                selected_by_context = {
                    record["context_idx"]: record
                    for record in selected_records
                }
                for i, ctx in enumerate(contexts):
                    ctx_records = per_context_records.get(i, [])
                    if not ctx_records:
                        continue
                    first = ctx_records[0]
                    record_decision_trace(
                        i, ctx, first["candidates"], first["candidate_ids"],
                        first["local_greedy_robot"],
                        first["local_greedy_dist"],
                        ctx_records,
                        selected_by_context.get(i),
                        mode="energy_score",
                        action_status=(
                            "selected_by_pool"
                            if i in selected_contexts else "missed_by_pool"
                        ),
                    )

                for record in selected_records:
                    self.stats["pool_scoring_selected_base_score_sum"] += (
                        record["score"]
                    )
                    self.stats[
                        "pool_scoring_selected_potential_score_sum"
                    ] += record["potential_guard_score"]
                    self.stats["pool_scoring_selected_final_score_sum"] += (
                        record["score"]
                    )
                    base_robot = baseline_robot_by_context.get(
                        record["context_idx"]
                    )
                    if base_robot is not None and base_robot != record[
                        "best_robot"
                    ]:
                        self.stats[
                            "pool_scoring_robot_substitutions"
                        ] += 1
                    commit_record(record)

                return choices

            if self.pool_scoring_mode == "context_rank":
                selected_records = []
                selected_contexts: Set[int] = set()
                selected_agents: Set[int] = set()

                for i in range(len(contexts)):
                    ctx_records_all = per_context_records.get(i, [])
                    if not ctx_records_all:
                        continue
                    base_ranked = sorted(
                        ctx_records_all,
                        key=lambda r: (
                            r.get("score_conv", r["score"])
                            if self.energy_scoring_mode == "conversion"
                            else r["score"],
                            r["best_robot"],
                        ),
                    )
                    pot_ranked = sorted(
                        ctx_records_all,
                        key=lambda r: (
                            r["potential_guard_score"], r["best_robot"]
                        ),
                    )
                    denom = max(len(ctx_records_all) - 1, 1)
                    for rank, record in enumerate(base_ranked):
                        record["base_rank"] = float(rank) / float(denom)
                    for rank, record in enumerate(pot_ranked):
                        record["potential_rank"] = float(rank) / float(denom)
                    for record in ctx_records_all:
                        record["pool_score"] = (
                            record["base_rank"]
                            + self.pool_scoring_lambda
                            * record["potential_rank"]
                        )

                    for record in sorted(
                        ctx_records_all,
                        key=lambda r: (
                            r["pool_score"],
                            r["base_rank"],
                            r.get("score_conv", r["score"])
                            if self.energy_scoring_mode == "conversion"
                            else r["score"],
                            r["best_robot"],
                        ),
                    ):
                        if record["best_robot"] in selected_agents:
                            continue
                        selected_records.append(record)
                        selected_contexts.add(record["context_idx"])
                        selected_agents.add(record["best_robot"])
                        break

                    if len(selected_agents) >= len(idle_agents):
                        break

                self.stats["pool_scoring_selected"] += len(selected_records)
                self.stats["pool_scoring_order_replacements"] += 0
                self.stats["pool_scoring_baseline_context_missed"] += len(
                    baseline_contexts - selected_contexts
                )

                selected_by_context = {
                    record["context_idx"]: record
                    for record in selected_records
                }
                for i, ctx in enumerate(contexts):
                    ctx_records = per_context_records.get(i, [])
                    if not ctx_records:
                        continue
                    first = ctx_records[0]
                    record_decision_trace(
                        i, ctx, first["candidates"], first["candidate_ids"],
                        first["local_greedy_robot"],
                        first["local_greedy_dist"],
                        ctx_records,
                        selected_by_context.get(i),
                        mode="context_rank",
                        action_status=(
                            "selected_by_pool"
                            if i in selected_contexts else "missed_by_pool"
                        ),
                    )

                for record in selected_records:
                    self.stats["pool_scoring_selected_base_score_sum"] += (
                        record["score"]
                    )
                    self.stats[
                        "pool_scoring_selected_potential_score_sum"
                    ] += record["potential_guard_score"]
                    self.stats[
                        "pool_scoring_selected_potential_rank_sum"
                    ] += record.get("potential_rank", 0.0)
                    self.stats["pool_scoring_selected_final_score_sum"] += (
                        record.get("pool_score", record["score"])
                    )
                    base_robot = baseline_robot_by_context.get(
                        record["context_idx"]
                    )
                    if base_robot is not None and base_robot != record[
                        "best_robot"
                    ]:
                        self.stats[
                            "pool_scoring_robot_substitutions"
                        ] += 1
                    commit_record(record)

                return choices

            if self.pool_scoring_mode in (
                "energy_conversion", "random_defer_matched",
            ):
                mu_t, v_backlog = self._energy_mu_t(world_state)
                warmup_ready = (
                    len(self._prio_relief_history) >= self.energy_prio_warmup
                    and len(self._prio_drift_history) >= self.energy_prio_warmup
                    and len(self._prio_noop_history) >= self.energy_prio_warmup
                )
                relief_scale = self._rolling_std(self._prio_relief_history)
                drift_scale = self._rolling_std(self._prio_drift_history)
                noop_scale = self._rolling_std(self._prio_noop_history)

                context_rows = []
                raw_relief_values = []
                raw_drift_values = []
                raw_noop_values = []

                for i, ctx in enumerate(contexts):
                    ctx_records_all = per_context_records.get(i, [])
                    if not ctx_records_all:
                        continue
                    ranked = sorted(
                        ctx_records_all,
                        key=lambda r: (
                            r.get("score_conv", r["score"]),
                            r["best_robot"],
                        ),
                    )
                    s1_best = ranked[0]
                    features = self._context_opportunity_features(
                        world_state, ctx, injection_snapshot
                    )
                    relief_raw = self._energy_service_relief(features)
                    drift_raw = (
                        float(s1_best.get("energy_conv_gate_g", 0.0))
                        * float(s1_best.get("energy_conv_delta_v", 0.0))
                    )
                    noop_raw = self._energy_noop_cost(features)

                    raw_relief_values.append(relief_raw)
                    raw_drift_values.append(drift_raw)
                    raw_noop_values.append(noop_raw)

                    if warmup_ready:
                        relief_n = relief_raw / relief_scale
                        drift_n = drift_raw / drift_scale
                        noop_n = noop_raw / noop_scale
                        priority = (
                            mu_t * relief_n
                            - self.energy_prio_drift_lambda * drift_n
                        )
                    else:
                        relief_n = 0.0
                        drift_n = 0.0
                        noop_n = 0.0
                        priority = 0.0

                    for record in ctx_records_all:
                        record["energy_prio_priority"] = priority
                        record["energy_prio_relief_raw"] = relief_raw
                        record["energy_prio_relief_n"] = relief_n
                        record["energy_prio_drift_raw"] = drift_raw
                        record["energy_prio_drift_n"] = drift_n
                        record["energy_prio_noop_raw"] = noop_raw
                        record["energy_prio_noop_n"] = noop_n
                        record["energy_prio_mu_t"] = mu_t
                        record["energy_prio_v_backlog"] = v_backlog
                        record["pool_score"] = -priority

                    context_rows.append({
                        "context_idx": i,
                        "ctx": ctx,
                        "records": ctx_records_all,
                        "ranked": ranked,
                        "priority": priority,
                        "relief_raw": relief_raw,
                        "relief_n": relief_n,
                        "drift_raw": drift_raw,
                        "drift_n": drift_n,
                        "noop_raw": noop_raw,
                        "noop_n": noop_n,
                    })

                if not context_rows:
                    return choices

                self._prio_relief_history.extend(raw_relief_values)
                self._prio_drift_history.extend(raw_drift_values)
                self._prio_noop_history.extend(raw_noop_values)

                num_rows = len(context_rows)
                self.stats["energy_prio_contexts"] += num_rows
                self.stats["energy_prio_mu_t_sum"] += mu_t * num_rows
                self.stats["energy_prio_mu_t_max"] = max(
                    self.stats["energy_prio_mu_t_max"], mu_t
                )
                self.stats["energy_prio_v_backlog_sum"] += (
                    v_backlog * num_rows
                )
                self.stats["energy_prio_v_backlog_max"] = max(
                    self.stats["energy_prio_v_backlog_max"], v_backlog
                )
                if not warmup_ready:
                    self.stats["energy_prio_warmup_contexts"] += num_rows

                s1_baseline_records = []
                s1_used_agents: Set[int] = set()
                for i in range(len(contexts)):
                    ctx_records = sorted(
                        per_context_records.get(i, []),
                        key=lambda r: (
                            r.get("score_conv", r["score"]),
                            r["best_robot"],
                        ),
                    )
                    for record in ctx_records:
                        if record["best_robot"] in s1_used_agents:
                            continue
                        s1_baseline_records.append(record)
                        s1_used_agents.add(record["best_robot"])
                        break
                s1_baseline_contexts = {
                    record["context_idx"] for record in s1_baseline_records
                }
                s1_baseline_robot_by_context = {
                    record["context_idx"]: record["best_robot"]
                    for record in s1_baseline_records
                }

                if self.pool_scoring_mode == "random_defer_matched":
                    ordered_rows = sorted(
                        context_rows,
                        key=lambda row: row["context_idx"],
                    )
                else:
                    ordered_rows = sorted(
                        context_rows,
                        key=lambda row: (
                            -row["priority"] if warmup_ready else 0.0,
                            row["context_idx"],
                        ),
                    )
                selected_records = []
                selected_contexts: Set[int] = set()
                selected_agents: Set[int] = set()
                selected_by_context = {}
                status_by_context = {
                    row["context_idx"]: "missed_by_capacity"
                    for row in context_rows
                }

                for row in ordered_rows:
                    context_idx = row["context_idx"]
                    should_defer = False
                    if self.pool_scoring_mode == "random_defer_matched":
                        self.stats["energy_prio_random_contexts"] += 1
                        self.stats["random_defer_contexts"] += 1
                        if self._random_defer_rng.random() < self.random_defer_rate:
                            should_defer = True
                            self.stats["energy_prio_random_deferred"] += 1
                            self.stats["random_defer_deferred"] += 1
                    elif (
                        warmup_ready
                        and self.energy_prio_theta < 1e6
                        and row["priority"]
                        < -self.energy_prio_theta * row["noop_n"]
                    ):
                        should_defer = True

                    if should_defer:
                        self.stats["energy_prio_deferred"] += 1
                        self.stats["energy_prio_priority_deferred_sum"] += (
                            row["priority"]
                        )
                        self.stats["energy_prio_relief_deferred_sum"] += (
                            row["relief_raw"]
                        )
                        self.stats["energy_prio_drift_deferred_sum"] += (
                            row["drift_raw"]
                        )
                        self.stats["energy_prio_noop_deferred_sum"] += (
                            row["noop_raw"]
                        )
                        status_by_context[context_idx] = (
                            "random_matched_deferred"
                            if self.pool_scoring_mode == "random_defer_matched"
                            else "priority_deferred"
                        )
                        continue

                    selected_record = None
                    for record in row["ranked"]:
                        if record["best_robot"] in selected_agents:
                            continue
                        selected_record = record
                        break

                    if selected_record is None:
                        status_by_context[context_idx] = "missed_robot_conflict"
                        continue

                    selected_records.append(selected_record)
                    selected_contexts.add(context_idx)
                    selected_agents.add(selected_record["best_robot"])
                    selected_by_context[context_idx] = selected_record
                    self.stats["energy_prio_executed"] += 1
                    self.stats["energy_prio_priority_executed_sum"] += (
                        row["priority"]
                    )
                    self.stats["energy_prio_relief_executed_sum"] += (
                        row["relief_raw"]
                    )
                    self.stats["energy_prio_drift_executed_sum"] += (
                        row["drift_raw"]
                    )
                    self.stats["energy_prio_noop_executed_sum"] += (
                        row["noop_raw"]
                    )
                    status_by_context[context_idx] = (
                        "selected_by_random_matched"
                        if self.pool_scoring_mode == "random_defer_matched"
                        else "selected_by_priority"
                    )

                    if len(selected_agents) >= len(idle_agents):
                        break

                self.stats["pool_scoring_selected"] += len(selected_records)
                self.stats["pool_scoring_order_replacements"] += len(
                    selected_contexts - s1_baseline_contexts
                )
                self.stats["pool_scoring_baseline_context_missed"] += len(
                    s1_baseline_contexts - selected_contexts
                )

                for row in context_rows:
                    ctx_records = row["records"]
                    first = ctx_records[0]
                    record_decision_trace(
                        row["context_idx"],
                        row["ctx"],
                        first["candidates"],
                        first["candidate_ids"],
                        first["local_greedy_robot"],
                        first["local_greedy_dist"],
                        ctx_records,
                        selected_by_context.get(row["context_idx"]),
                        mode=self.pool_scoring_mode,
                        action_status=status_by_context[row["context_idx"]],
                    )

                for record in selected_records:
                    self.stats["pool_scoring_selected_base_score_sum"] += (
                        record["score"]
                    )
                    self.stats[
                        "pool_scoring_selected_potential_score_sum"
                    ] += record["potential_guard_score"]
                    self.stats["pool_scoring_selected_final_score_sum"] += (
                        record.get("pool_score", record["score"])
                    )
                    base_robot = s1_baseline_robot_by_context.get(
                        record["context_idx"]
                    )
                    if base_robot is not None and base_robot != record[
                        "best_robot"
                    ]:
                        self.stats[
                            "pool_scoring_robot_substitutions"
                        ] += 1
                    commit_record(record)

                return choices

            ranked_by_potential = sorted(
                pool_records,
                key=lambda r: r["potential_guard_score"],
            )
            denom = max(len(ranked_by_potential) - 1, 1)
            for rank, record in enumerate(ranked_by_potential):
                record["potential_rank"] = float(rank) / float(denom)
                record["pool_score"] = (
                    record["score"]
                    + self.pool_scoring_lambda * record["potential_rank"]
                )

            selected_records = []
            selected_contexts: Set[int] = set()
            selected_agents: Set[int] = set()
            for record in sorted(
                pool_records,
                key=lambda r: (
                    r["pool_score"],
                    r["score"],
                    r["context_idx"],
                    r["best_robot"],
                ),
            ):
                if record["context_idx"] in selected_contexts:
                    continue
                if record["best_robot"] in selected_agents:
                    continue
                selected_records.append(record)
                selected_contexts.add(record["context_idx"])
                selected_agents.add(record["best_robot"])
                if len(selected_agents) >= len(idle_agents):
                    break

            self.stats["pool_scoring_selected"] += len(selected_records)
            self.stats["pool_scoring_order_replacements"] += len(
                selected_contexts - baseline_contexts
            )
            self.stats["pool_scoring_baseline_context_missed"] += len(
                baseline_contexts - selected_contexts
            )

            selected_by_context = {
                record["context_idx"]: record
                for record in selected_records
            }
            for i, ctx in enumerate(contexts):
                ctx_records = per_context_records.get(i, [])
                if not ctx_records:
                    continue
                first = ctx_records[0]
                record_decision_trace(
                    i, ctx, first["candidates"], first["candidate_ids"],
                    first["local_greedy_robot"], first["local_greedy_dist"],
                    ctx_records,
                    selected_by_context.get(i),
                    mode="potential_rank",
                    action_status=(
                        "selected_by_pool"
                        if i in selected_contexts else "missed_by_pool"
                    ),
                )

            for record in selected_records:
                self.stats["pool_scoring_selected_base_score_sum"] += (
                    record["score"]
                )
                self.stats["pool_scoring_selected_potential_score_sum"] += (
                    record["potential_guard_score"]
                )
                self.stats["pool_scoring_selected_potential_rank_sum"] += (
                    record.get("potential_rank", 0.0)
                )
                self.stats["pool_scoring_selected_final_score_sum"] += (
                    record.get("pool_score", record["score"])
                )
                base_robot = baseline_robot_by_context.get(record["context_idx"])
                if base_robot is not None and base_robot != record["best_robot"]:
                    self.stats["pool_scoring_robot_substitutions"] += 1
                commit_record(record)

            return choices

        for i, ctx in enumerate(contexts):
            available = [
                a for a in idle_agents if a.agent_id not in used_agents
            ]
            if not available:
                break

            candidates = self._select_robot_candidates(
                world_state, ctx, available, lyapunov_snapshot
            )
            candidate_ids = [a.agent_id for a in candidates]
            local_greedy_robot = None
            local_greedy_dist = (
                min(
                    _manhattan_distance(agent.position, ctx.pod_location)
                    for agent in candidates
                )
                if candidates else 0
            )
            self.stats["decision_contexts_total"] += 1

            best_cost = float("inf")
            best_robot = None
            best_risk_defer_score = None
            best_potential_guard_score = None
            best_injection_penalty = 0.0
            best_injection_raw = 0.0
            best_injection_stress = 0.0
            best_injection_fast = 0.0
            best_record = None
            scored_records = []
            injection_score_records = []

            for agent in candidates:
                record = score_candidate(
                    i, ctx, agent, candidates, candidate_ids,
                    local_greedy_robot, local_greedy_dist,
                    injection_score_records,
                )
                if record is None:
                    continue
                scored_records.append(record)

                if record["score"] < best_cost:
                    best_cost = record["score"]
                    best_robot = record["best_robot"]
                    best_record = record
                    best_risk_defer_score = record["risk_defer_score"]
                    best_potential_guard_score = record[
                        "potential_guard_score"
                    ]
                    best_injection_penalty = record["station_injection_penalty"]
                    best_injection_raw = record["station_injection_raw"]
                    best_injection_stress = record["station_injection_stress"]
                    best_injection_fast = record["station_injection_fast"]

            if self.station_injection_weight > 0.0:
                self._record_station_injection_diag(injection_score_records)

            self._apply_work_drift_group_range(scored_records)

            if self.energy_scoring_mode == "conversion":
                self._apply_energy_conversion(scored_records)

            if scored_records:
                best_record = min(
                    scored_records,
                    key=lambda r: (
                        r["work_drift_combined_score"]
                        if self.work_drift_mode != "off"
                        else (
                            r.get("score_conv", r["score"])
                            if self.energy_scoring_mode == "conversion"
                            else r["score"]
                        ),
                        r["score"] if self.work_drift_mode != "off" else 0.0,
                        r["best_robot"],
                    ),
                )
                best_cost = (
                    best_record["work_drift_combined_score"]
                    if self.work_drift_mode != "off"
                    else (
                        best_record.get("score_conv", best_record["score"])
                        if self.energy_scoring_mode == "conversion"
                        else best_record["score"]
                    )
                )
                best_robot = best_record["best_robot"]
                best_risk_defer_score = best_record["risk_defer_score"]
                best_potential_guard_score = best_record[
                    "potential_guard_score"
                ]
                best_injection_penalty = best_record["station_injection_penalty"]
                best_injection_raw = best_record["station_injection_raw"]
                best_injection_stress = best_record["station_injection_stress"]
                best_injection_fast = best_record["station_injection_fast"]

            native_no_assign_record = None
            if self.include_no_assign_candidate:
                self.stats["native_no_assign_contexts"] += 1
                native_no_assign_record = score_native_no_assign(i)
                if (
                    native_no_assign_record is not None
                    and best_robot is not None
                ):
                    dispatch_decision = None
                    if self.dispatch_potential_mode != "off":
                        dispatch_decision = self._apply_dispatch_potential(
                            ctx,
                            scored_records,
                            native_no_assign_record,
                        )
                        best_record = dispatch_decision[
                            "selected_assignment"
                        ]
                        best_robot = best_record["best_robot"]
                        if self.dispatch_potential_mode == "bridge":
                            best_cost = float(
                                best_record["dispatch_composite_score"]
                            )
                            no_assign_cost = float(
                                native_no_assign_record[
                                    "dispatch_composite_score"
                                ]
                            )
                        else:
                            # Diagnostic mode must leave legacy metrics as
                            # well as actions untouched.  Composite margins
                            # remain available under dispatch_* trace fields.
                            best_cost = float(best_record["score"])
                            no_assign_cost = float(
                                native_no_assign_record["score"]
                            )
                        should_defer_native = bool(
                            dispatch_decision["actual_defer"]
                        )
                        trace_mode = (
                            "sequential_dispatch_bridge"
                            if self.dispatch_potential_mode == "bridge"
                            else "sequential_dispatch_diagnostic"
                        )
                    else:
                        no_assign_cost = float(
                            native_no_assign_record["score"]
                        )
                        should_defer_native = bool(
                            no_assign_cost < float(best_cost)
                        )
                        trace_mode = "sequential_native_action_set"

                    # Exact ties execute assignment.  Gate-1 has no learned
                    # or hand-tuned scale: the only transformation is the
                    # frozen exact group range plus analytic drift.
                    margin = float(best_cost - no_assign_cost)
                    self.stats["native_no_assign_margin_sum"] += margin
                    if should_defer_native:
                        self.stats["native_no_assign_selected"] += 1
                        record_decision_trace(
                            i,
                            ctx,
                            candidates,
                            candidate_ids,
                            local_greedy_robot,
                            local_greedy_dist,
                            scored_records,
                            native_no_assign_record,
                            mode=trace_mode,
                            action_status=(
                                "dispatch_defer_selected"
                                if dispatch_decision is not None
                                else "native_no_assign_selected"
                            ),
                            native_no_assign_record=native_no_assign_record,
                        )
                        continue
                    self.stats[
                        "native_no_assign_assignment_selected"
                    ] += 1

            # If every idle robot is rejected by the risk gate, defer this
            # fixed context.  A nearest-robot substitute would silently
            # reintroduce a non-World-Model allocation decision.

            if best_robot is not None:
                context_summary = summarize_context_records(scored_records)
                action_status = "selected_before_guard"
                if context_summary is not None:
                    if self.energy_scoring_mode != "off":
                        self.stats["energy_scoring_contexts"] += 1
                    should_energy_noop, noop_reason, noop_score = (
                        apply_energy_noop(ctx, scored_records)
                    )
                    if should_energy_noop:
                        record_decision_trace(
                            i, ctx, candidates, candidate_ids,
                            local_greedy_robot, local_greedy_dist,
                            scored_records, best_record,
                            mode="sequential",
                            action_status=(
                                "energy_noop_deferred:" + noop_reason
                            ),
                        )
                        continue

                    should_candidate_defer, guard_reason = (
                        apply_candidate_set_guard(context_summary)
                    )
                    if should_candidate_defer:
                        record_decision_trace(
                            i, ctx, candidates, candidate_ids,
                            local_greedy_robot, local_greedy_dist,
                            scored_records, best_record,
                            mode="sequential",
                            action_status=(
                                "candidate_set_deferred:" + guard_reason
                            ),
                        )
                        continue

                    # In "off" mode apply_margin_substitute returns the raw
                    # `score` argmin, which would silently revert conversion
                    # flips (score_conv argmin), so only run it when enabled.
                    if self.margin_substitute_mode != "off":
                        selected_record, substitute_reason = (
                            apply_margin_substitute(context_summary)
                        )
                        if selected_record is not best_record:
                            best_record = selected_record
                            best_robot = selected_record["best_robot"]
                            best_cost = (
                                selected_record.get(
                                    "score_conv", selected_record["score"]
                                )
                                if self.energy_scoring_mode == "conversion"
                                else selected_record["score"]
                            )
                            best_risk_defer_score = selected_record[
                                "risk_defer_score"
                            ]
                            best_potential_guard_score = selected_record[
                                "potential_guard_score"
                            ]
                            best_injection_penalty = selected_record[
                                "station_injection_penalty"
                            ]
                            best_injection_raw = selected_record[
                                "station_injection_raw"
                            ]
                            best_injection_stress = selected_record[
                                "station_injection_stress"
                            ]
                            best_injection_fast = selected_record[
                                "station_injection_fast"
                            ]
                        if substitute_reason == "substituted":
                            action_status = "margin_substituted"
                        elif substitute_reason not in ("off",):
                            action_status = "selected:" + substitute_reason

                record_decision_trace(
                    i, ctx, candidates, candidate_ids,
                    local_greedy_robot, local_greedy_dist,
                    scored_records, best_record,
                    mode=(
                        "sequential_dispatch_bridge"
                        if self.dispatch_potential_mode == "bridge"
                        else (
                            "sequential_dispatch_diagnostic"
                            if self.dispatch_potential_mode == "diagnostic"
                            else "sequential"
                        )
                    ),
                    action_status=action_status,
                    native_no_assign_record=native_no_assign_record,
                )

                if self.potential_guard_mode != "off":
                    score_for_guard = (
                        float(best_potential_guard_score)
                        if best_potential_guard_score is not None else 0.0
                    )
                    note_potential_guard_candidate(score_for_guard)

                    should_guard_defer = False
                    threshold = self._potential_guard_rolling_threshold()
                    self.stats["potential_guard_rolling_checks"] += 1
                    if threshold is None:
                        self.stats["potential_guard_rolling_warmup"] += 1
                    else:
                        self.stats["potential_guard_rolling_threshold_sum"] += (
                            threshold
                        )
                        if self.potential_guard_mode == "rolling":
                            if score_for_guard > threshold:
                                should_guard_defer = True
                        else:
                            self.stats["potential_guard_random_contexts"] += 1
                            if (self._random_defer_rng.random()
                                    > self.potential_guard_rolling_quantile):
                                should_guard_defer = True
                                self.stats[
                                    "potential_guard_random_deferred"
                                ] += 1
                    self._potential_guard_history.append(score_for_guard)

                    if should_guard_defer:
                        self.stats["potential_guard_deferred"] += 1
                        self.stats["potential_guard_deferred_score_sum"] += (
                            score_for_guard
                        )
                        continue

                    self.stats["potential_guard_executed"] += 1
                    self.stats["potential_guard_executed_score_sum"] += (
                        score_for_guard
                    )

                if self.risk_defer_mode != "off":
                    score_for_defer = (
                        float(best_risk_defer_score)
                        if best_risk_defer_score is not None else 0.0
                    )
                    note_defer_candidate(score_for_defer)

                    if is_topq_mode:
                        topq_records.append({
                            "context_idx": i,
                            "best_robot": best_robot,
                            "score_for_defer": score_for_defer,
                            "shadow_robot": shadow_greedy_choices.get(i),
                            "local_greedy_robot": local_greedy_robot,
                            "candidate_ids": candidate_ids,
                            "candidates": candidates,
                            "pod_location": ctx.pod_location,
                            "local_greedy_dist": local_greedy_dist,
                            "station_injection_penalty": best_injection_penalty,
                            "station_injection_raw": best_injection_raw,
                            "station_injection_stress": best_injection_stress,
                            "station_injection_fast": best_injection_fast,
                            "lyapunov_l0_delta": (
                                best_record.get("lyapunov_l0_delta", 0.0)
                                if best_record is not None else 0.0
                            ),
                            "lyapunov_l0_eta": (
                                best_record.get("lyapunov_l0_eta", 0.0)
                                if best_record is not None else 0.0
                            ),
                            "lyapunov_l0_eta_bin": (
                                best_record.get("lyapunov_l0_eta_bin", -1)
                                if best_record is not None else -1
                            ),
                        })
                        # Budgeted top-q is a diagnostic ablation over the
                        # would-be choices in this assign call. We reserve the
                        # robot here so later contexts see the same sequential
                        # availability as the baseline scorer.
                        used_agents.add(best_robot)
                        continue

                    # Diagnostic trick: skip the whole context when the selected
                    # candidate looks risky. This is a hard-guard ablation, not
                    # the intended 6.1 scheduler design.
                    should_defer = False
                    if self.risk_defer_mode == "risk":
                        if (self.risk_defer_threshold is not None
                                and score_for_defer > self.risk_defer_threshold):
                            should_defer = True
                    elif self.risk_defer_mode == "random":
                        self.stats["random_defer_contexts"] += 1
                        if self._random_defer_rng.random() < self.random_defer_rate:
                            should_defer = True
                            self.stats["random_defer_deferred"] += 1
                    elif is_rolling_mode:
                        threshold = self._rolling_threshold()
                        self.stats["risk_defer_rolling_checks"] += 1
                        if threshold is None:
                            self.stats["risk_defer_rolling_warmup"] += 1
                        else:
                            self.stats["risk_defer_rolling_threshold_sum"] += threshold
                            if self.risk_defer_mode == "risk_rolling":
                                if score_for_defer > threshold:
                                    should_defer = True
                            else:
                                self.stats["random_defer_contexts"] += 1
                                if self._random_defer_rng.random() > self.risk_defer_rolling_quantile:
                                    should_defer = True
                                    self.stats["random_defer_deferred"] += 1
                        self._risk_defer_history.append(score_for_defer)

                    if should_defer:
                        self.stats["risk_defer_deferred"] += 1
                        self.stats["risk_defer_deferred_score_sum"] += score_for_defer
                        continue

                    self.stats["risk_defer_executed"] += 1
                    self.stats["risk_defer_executed_score_sum"] += score_for_defer

                commit_record({
                    "context_idx": i,
                    "best_robot": best_robot,
                    "shadow_robot": shadow_greedy_choices.get(i),
                    "local_greedy_robot": local_greedy_robot,
                    "candidate_ids": candidate_ids,
                    "candidates": candidates,
                    "pod_location": ctx.pod_location,
                    "local_greedy_dist": local_greedy_dist,
                    "station_injection_penalty": best_injection_penalty,
                    "station_injection_raw": best_injection_raw,
                    "station_injection_stress": best_injection_stress,
                    "station_injection_fast": best_injection_fast,
                    "lyapunov_l0_delta": (
                        best_record.get("lyapunov_l0_delta", 0.0)
                        if best_record is not None else 0.0
                    ),
                    "lyapunov_l0_eta": (
                        best_record.get("lyapunov_l0_eta", 0.0)
                        if best_record is not None else 0.0
                    ),
                    "lyapunov_l0_eta_bin": (
                        best_record.get("lyapunov_l0_eta_bin", -1)
                        if best_record is not None else -1
                    ),
                })
            else:
                record_decision_trace(
                    i, ctx, candidates, candidate_ids,
                    local_greedy_robot, local_greedy_dist,
                    scored_records, best_record,
                    mode="sequential",
                    action_status="no_selection",
                    native_no_assign_record=native_no_assign_record,
                )

        if is_topq_mode and topq_records:
            budget = self._topq_budget_count(len(topq_records))
            self.stats["risk_defer_topq_batches"] += 1
            self.stats["risk_defer_topq_budget_sum"] += budget
            if self.risk_defer_mode == "risk_topq":
                ranked = sorted(
                    range(len(topq_records)),
                    key=lambda j: topq_records[j]["score_for_defer"],
                    reverse=True,
                )
                deferred = set(ranked[:budget])
            else:
                self.stats["random_defer_contexts"] += len(topq_records)
                deferred = set(self._random_defer_rng.sample(
                    range(len(topq_records)), budget
                )) if budget > 0 else set()
                self.stats["random_defer_deferred"] += len(deferred)

            for j, record in enumerate(topq_records):
                score_for_defer = record["score_for_defer"]
                if j in deferred:
                    self.stats["risk_defer_deferred"] += 1
                    self.stats["risk_defer_deferred_score_sum"] += score_for_defer
                    continue
                self.stats["risk_defer_executed"] += 1
                self.stats["risk_defer_executed_score_sum"] += score_for_defer
                commit_record(record)

        return choices
