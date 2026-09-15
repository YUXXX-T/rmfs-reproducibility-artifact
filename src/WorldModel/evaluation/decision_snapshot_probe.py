"""Phase C decision-point snapshot probe for online evaluation runs.

Registered on BOTH engine callback lists:

  - ``on_pre_assignment`` (after order generation / backlog refill, before
    ``task_assigner.assign()``) captures full restart snapshots per candidate
    group — the same phase and payload contract as ``data_collector``'s
    ``phaseB_b0_decision_snapshot_v1``, so
    ``generate_long_risk_labels.process_snapshot`` consumes these files
    unchanged (it reads fields and does not check schema_version).
  - ``on_tick`` (post-step) attaches the same-tick ``decision_meta``
    (per-tick assigner stats diff + post-step unified risk) to the pending
    snapshots, then maintains the feature history / flow counters used by
    the next capture.

Registered boundaries (td_bootstrap_6p2_plan.md §8):

  - The legacy default remains interval capture.  The analytic Layer-5 mode
    may pre-buffer every ready decision tick and retain all decisions that
    changed the pure-WM choice plus interval controls.  This uses only the
    same-tick assigner trace; realised-future priority remains an offline
    trace/snapshot join and never changes the online policy.
  - The TD stream frame at tick ``tau`` is the post-step state of ``tau``
    and is NOT this snapshot's decision state ``s_t``; Phase C training
    samples must take ``s_t`` from the snapshot's own ``world_snapshot`` /
    ``node_history``.
  - ``propose_assignment_contexts`` may prefill ``order.pod_ids``
    (idempotent with assign's own prefill); non-perturbation is certified
    by the bitwise with/without-probe smoke check, not by declaration.
"""

import copy
import datetime
import json
import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from WorldModel.candidate_generator import generate_robot_candidates
from WorldModel.graph_builder import (
    FeatureHistory,
    build_static_graph,
    extract_demand_context,
    extract_edge_features,
    extract_node_features,
    extract_system_labels,
)
from WorldState.order_state import Order
from WorldState.risk import compute_unified_risk
from WorldState.task_state import Task

STATS_KEYS = (
    ("energy_conv_gate_g_sum", "gate_g_sum_d"),
    ("energy_conv_contexts", "conv_contexts_d"),
    ("energy_conv_active_contexts", "conv_active_d"),
    ("energy_conv_modified_decisions", "conv_modified_d"),
    ("shadow_greedy_compared", "shadow_greedy_compared_d"),
    ("shadow_greedy_match", "shadow_greedy_match_d"),
    ("fallback_greedy_calls", "fallback_greedy_d"),
    ("model_assign_calls", "model_assign_d"),
)

PHASE_C_SNAPSHOT_SCHEMA_VERSION = "phaseC_decision_snapshot_v3"
PHASE_C_SNAPINDEX_SCHEMA_VERSION = "phaseC_decision_snapindex_v3"
PHASE_C_DATA_CONTRACT = "wm_on_policy_isolated_counterfactual_v1"


class DecisionSnapshotProbe:
    """Full decision-point snapshots (contract ⊇ phaseB v1) for one run."""

    def __init__(
        self,
        engine,
        out_dir: str,
        run_id: str = "run",
        sample_interval: int = 5,
        top_m: int = 5,
        min_group_size: int = 2,
        meta: Optional[dict] = None,
        reservation_window: int = 1,
        candidate_scope: str = "top_m_snapshot",
        attach_assigner_trace: bool = False,
        require_trace_alignment: bool = False,
        lyapunov_l0_config: Optional[dict] = None,
        capture_policy: str = "interval_all",
        candidate_robot_mode: str = "nearest",
        include_no_assign_candidate: bool = False,
        max_contexts_per_tick: Optional[int] = None,
        phase_c_round: Optional[str] = None,
        exclude_external_baselines: bool = False,
    ):
        self.out_dir = out_dir
        self.run_id = run_id
        self.sample_interval = max(1, int(sample_interval))
        self.top_m = int(top_m)
        self.min_group_size = int(min_group_size)
        self.meta = dict(meta or {})
        self.reservation_window = reservation_window
        if candidate_scope not in ("top_m_snapshot", "all_idle_online"):
            raise ValueError(
                "candidate_scope must be top_m_snapshot or all_idle_online"
            )
        self.candidate_scope = candidate_scope
        self.attach_assigner_trace = bool(attach_assigner_trace)
        self.require_trace_alignment = bool(require_trace_alignment)
        if self.require_trace_alignment and not self.attach_assigner_trace:
            raise ValueError(
                "require_trace_alignment requires attach_assigner_trace"
            )
        self.lyapunov_l0_config = dict(lyapunov_l0_config or {})
        if capture_policy not in (
            "interval_all",
            "all_modified_plus_interval_controls",
        ):
            raise ValueError("unexpected decision snapshot capture_policy")
        if (
            capture_policy == "all_modified_plus_interval_controls"
            and not self.require_trace_alignment
        ):
            raise ValueError(
                "modified-decision capture requires strict trace alignment"
            )
        self.capture_policy = capture_policy
        if candidate_robot_mode not in (
            "nearest", "eta_stratified", "stratified",
        ):
            raise ValueError(
                "candidate_robot_mode must be nearest, eta_stratified, or "
                "stratified"
            )
        self.candidate_robot_mode = candidate_robot_mode
        self.include_no_assign_candidate = bool(
            include_no_assign_candidate
        )
        if max_contexts_per_tick is not None and int(max_contexts_per_tick) <= 0:
            raise ValueError("max_contexts_per_tick must be positive")
        self.max_contexts_per_tick = (
            None
            if max_contexts_per_tick is None
            else int(max_contexts_per_tick)
        )
        self.phase_c_round = phase_c_round
        self.exclude_external_baselines = bool(exclude_external_baselines)
        if self.exclude_external_baselines:
            arm_label = str(self.meta.get("arm_label") or "")
            forbidden = ("greedy", "hungarian")
            if any(token in arm_label.lower() for token in forbidden):
                raise ValueError(
                    "Phase C training snapshots must come from the World "
                    "Model on-policy arm, not an external baseline"
                )
        if self.include_no_assign_candidate and self.attach_assigner_trace:
            raise ValueError(
                "Phase C counterfactual NO_ASSIGN capture must remain "
                "observation-only; do not replace it with the deployed "
                "assigner trace candidate list"
            )
        os.makedirs(out_dir, exist_ok=True)

        (self._edge_index, self._node_map, self._inv_node_map,
         self._local_capacity, self._bottleneck_score,
         self._node_type_arr, self._adj) = build_static_graph(
            engine.world.map_state)

        self._fh = FeatureHistory(len(self._node_map), feat_dim=10,
                                  history_len=4)
        self._flow_counter: Dict[int, float] = {}
        self._edge_flow_counter: Dict[Tuple[int, int], float] = {}
        self._last_edge_tick = -1

        # station_node_ids by the station_positions convention
        # (sorted station ids -> node_map), same as data_collector._init_graph.
        self._station_node_ids: List[int] = []
        sp = engine.world.map_state.station_positions
        for sid in sorted(sp.keys()):
            nid = self._node_map.get(sp[sid])
            if nid is not None:
                self._station_node_ids.append(nid)

        # Greedy arm has no .stats — diffs stay zero there.
        self._prev_stats = self._read_stats(engine)

        self._pending: List[Tuple[dict, dict]] = []
        self._index_rows: List[dict] = []
        self._trace_cursor = 0
        self._prev_completed = int(
            engine.world.order_state.total_completed
        )

    @staticmethod
    def _read_stats(engine) -> Dict[str, float]:
        stats = getattr(engine.task_assigner, "stats", None) or {}
        return {k: float(stats.get(k, 0.0)) for k, _ in STATS_KEYS}

    @property
    def saved_snapshots(self) -> int:
        return len(self._index_rows)

    # ------------------------------------------------------------------
    # pre-assignment phase: capture restart snapshots
    # ------------------------------------------------------------------

    def on_pre_assignment(self, engine):
        world = engine.world
        tick = int(world.tick)
        if (
            self.capture_policy == "interval_all"
            and tick % self.sample_interval != 0
        ):
            return
        if not self._fh.is_ready:
            return

        context_limit = self.max_contexts_per_tick
        if self.require_trace_alignment:
            # The pure World-Model assigner opens at most one fixed context
            # per currently idle robot.  Mirroring that budget prevents the
            # probe from manufacturing extra high-load contexts that the
            # online policy never scored and therefore cannot align.
            trace_limit = len(world.get_idle_agents())
            context_limit = (
                trace_limit
                if context_limit is None
                else min(context_limit, trace_limit)
            )
        if context_limit is None:
            contexts = engine.task_assigner.propose_assignment_contexts(world)
        else:
            contexts = engine.task_assigner.propose_assignment_contexts(
                world, max_contexts=context_limit
            )
        if not contexts:
            return
        candidate_limit = self.top_m
        if self.candidate_scope == "all_idle_online":
            candidate_limit = max(len(world.get_idle_agents()), 1)
        groups = generate_robot_candidates(
            contexts,
            world,
            candidate_limit,
            candidate_mode=self.candidate_robot_mode,
            lyapunov_config=self.lyapunov_l0_config,
            node_map=self._node_map,
            edge_flow_counter=self._edge_flow_counter,
            bottleneck_score=self._bottleneck_score,
            reservation_window=self.reservation_window,
            include_no_assign_candidate=self.include_no_assign_candidate,
        )
        groups = [
            group for group in groups
            if int(group.get(
                "robot_candidate_count", len(group["candidates"])
            )) >= self.min_group_size
        ]
        if not groups:
            return

        risk_at_decision = compute_unified_risk(world)
        r_at_decision = float(risk_at_decision["unified_risk"])
        system_labels = extract_system_labels(
            world,
            self._bottleneck_score,
            self._node_map,
            self._local_capacity,
            adj=self._adj,
            prev_completed=self._prev_completed,
        )
        system_values = [float(value) for value in system_labels.tolist()]
        state_diagnostics = {
            "idle_robot_count": len(world.get_idle_agents()),
            "pending_order_count": len(world.order_state.get_pending_orders()),
            "open_order_count": sum(
                str(getattr(order.status, "name", order.status)).upper()
                not in {"COMPLETED", "CANCELLED"}
                for order in world.order_state.orders.values()
            ),
            "active_task_count": sum(
                str(getattr(task.status, "name", task.status)).upper()
                in {"ASSIGNED", "IN_PROGRESS"}
                for task in world.task_state.tasks.values()
            ),
            "system_labels": system_values,
            "station_pressure": (
                system_values[2] + system_values[3]
                if len(system_values) >= 4 else None
            ),
            "risk_components": {
                key: float(value)
                for key, value in risk_at_decision.items()
                if isinstance(value, (int, float))
            },
        }
        history = self._fh.get_history()
        edge_feat = extract_edge_features(
            self._edge_index, self._node_map, self._inv_node_map,
            self._local_capacity, world, adj=self._adj,
            edge_flow_counter=self._edge_flow_counter,
            reservation_window=self.reservation_window,
        )
        demand = extract_demand_context(world)
        station_ids = torch.tensor(self._station_node_ids, dtype=torch.long)
        # One immutable restart image is sufficient for every fixed context at
        # this decision tick.  Re-copying the full world/path-planner per group
        # makes long all-modified capture needlessly CPU-bound.
        restart_world = copy.deepcopy(world)
        restart_config = copy.deepcopy(engine.config)
        restart_path_planner = copy.deepcopy(engine.path_planner)
        restart_order_generator = copy.deepcopy(engine.order_generator)
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()

        for group in groups:
            payload = {
                "schema_version": PHASE_C_SNAPSHOT_SCHEMA_VERSION,
                "phase_c_data_contract": PHASE_C_DATA_CONTRACT,
                "phase_c_round": self.phase_c_round,
                "training_source_policy": "world_model_on_policy",
                "external_baseline_training_samples": False,
                "td_target_enabled": False,
                "td_value_head_enabled": False,
                "rollout_label_contract": {
                    "continuation_mode": "isolated",
                    "future_order_generation": False,
                    "continuation_scheduler": False,
                    "candidate_actions": (
                        "stratified_idle_robots_plus_native_no_assign"
                        if self.include_no_assign_candidate
                        else "sampled_idle_robots"
                    ),
                },
                # provenance (new vs phaseB v1)
                "run_id": self.run_id,
                "arm_label": self.meta.get("arm_label"),
                "load": self.meta.get("load"),
                "seed": self.meta.get("seed"),
                "config_path": self.meta.get("config"),
                "checkpoint_path": self.meta.get("checkpoint_path"),
                "path_planner_override": self.meta.get(
                    "path_planner_override"
                ),
                "path_planner_params_override": self.meta.get(
                    "path_planner_params_override"
                ),
                "order_manifest_sha256": (
                    self.meta.get("order_manifest_sha256")
                    or self.meta.get("paired_order_manifest_sha256")
                ),
                "online_candidate_scope": self.candidate_scope,
                "top_m_is_online_limit": False,
                "snapshot_candidate_robot_mode": self.candidate_robot_mode,
                "snapshot_includes_no_assign": (
                    self.include_no_assign_candidate
                ),
                "snapshot_max_contexts_per_tick": (
                    self.max_contexts_per_tick
                ),
                # phaseB_b0_decision_snapshot_v1 field contract
                # (generate_long_risk_labels.process_snapshot input)
                "decision_tick": tick,
                "candidate_group_id": group["group_id"],
                "world_snapshot": restart_world,
                "task_next_id": Task._next_id,
                "order_next_id": Order._next_id,
                "python_rng_state": python_rng_state,
                "numpy_rng_state": numpy_rng_state,
                "torch_rng_state": torch_rng_state,
                "config": restart_config,
                "path_planner_state": restart_path_planner,
                "order_generator_state": restart_order_generator,
                "candidates": copy.deepcopy(group["candidates"]),
                "fixed_context": copy.deepcopy(group["fixed_context"]),
                # decision-state observation (offline Phase-B sample rebuild)
                "r_at_decision": r_at_decision,
                "node_history": history.clone(),
                "edge_index": self._edge_index,
                "edge_features": edge_feat,
                "demand_context": demand,
                "station_node_ids": station_ids,
                "flow_counter": dict(self._flow_counter),
                "edge_flow_counter": dict(self._edge_flow_counter),
                "lyapunov_l0_config": copy.deepcopy(
                    self.lyapunov_l0_config
                ),
                "phase_c_state_diagnostics": copy.deepcopy(
                    state_diagnostics
                ),
            }
            row = {
                "tick": tick,
                "group_id": group["group_id"],
                "order_id": int(group["fixed_context"]["order_id"]),
                "pod_id": int(group["fixed_context"]["pod_id"]),
                "station_id": int(group["fixed_context"]["station_id"]),
                "n_candidates": len(group["candidates"]),
                "captured_candidate_scope": self.candidate_scope,
                "r_at_decision": r_at_decision,
                "phase_c_state_diagnostics": copy.deepcopy(
                    state_diagnostics
                ),
                "robot_candidate_count": int(group.get(
                    "robot_candidate_count", len(group["candidates"])
                )),
                "includes_no_assign_candidate": bool(group.get(
                    "includes_no_assign_candidate", False
                )),
            }
            self._pending.append((payload, row))

    # ------------------------------------------------------------------
    # post-step phase: decision_meta + feature/counter maintenance
    # ------------------------------------------------------------------

    def on_tick(self, engine):
        world = engine.world
        tick = int(world.tick)
        self._prev_completed = int(world.order_state.total_completed)

        # Stats diff every tick (sampling-tick-only differencing would
        # attribute the whole previous interval to one decision tick).
        cur = self._read_stats(engine)
        diff = {out: cur[k] - self._prev_stats[k] for k, out in STATS_KEYS}
        self._prev_stats = cur

        trace_rows = []
        if self.attach_assigner_trace:
            records = getattr(
                engine.task_assigner, "decision_trace_records", None
            )
            if records is None:
                raise RuntimeError(
                    "assigner trace attachment requested but no trace exists"
                )
            trace_rows = records[self._trace_cursor:]
            self._trace_cursor = len(records)

        if self._pending:
            meta = dict(diff)
            meta["r_post_step"] = float(
                compute_unified_risk(world)["unified_risk"])
            for payload, row in self._pending:
                trace = None
                if self.attach_assigner_trace:
                    for candidate_trace in trace_rows:
                        if (
                            int(candidate_trace.get("tick", -1))
                            == int(payload["decision_tick"])
                            and int(candidate_trace.get("order_id", -1))
                            == int(payload["fixed_context"]["order_id"])
                            and int(candidate_trace.get("pod_id", -1))
                            == int(payload["fixed_context"]["pod_id"])
                        ):
                            trace = candidate_trace
                            break
                    if trace is None and self.require_trace_alignment:
                        raise RuntimeError(
                            "decision snapshot could not be aligned with the "
                            "online assigner trace"
                        )
                if trace is not None:
                    trace = copy.deepcopy(trace)
                    payload["online_decision"] = trace
                    actual_ids = [
                        int(value) for value in trace.get("candidate_ids", ())
                    ]
                    by_id = {
                        int(candidate["robot_id"]): candidate
                        for candidate in payload["candidates"]
                    }
                    actual_candidates = []
                    trace_candidates = {
                        int(candidate["robot_id"]): candidate
                        for candidate in trace.get("candidates", ())
                    }
                    for robot_id in actual_ids:
                        candidate = by_id.get(robot_id)
                        traced = trace_candidates.get(robot_id) or {}
                        if candidate is None:
                            candidate = {
                                "robot_id": robot_id,
                                "robot_start": traced.get("robot_start"),
                                "candidate_policy": "online_trace_recovered",
                                "candidate_selection_mode": "all_idle_online",
                                "chosen": False,
                            }
                        else:
                            candidate = dict(candidate)
                        candidate.update({
                            "chosen": robot_id == trace.get("selected_robot"),
                            "online_wm_score": traced.get("score"),
                            "online_analytic_work_drift": traced.get(
                                "work_drift_raw"
                            ),
                            "online_combined_score": traced.get(
                                "work_drift_combined_score"
                            ),
                        })
                        actual_candidates.append(candidate)
                    payload["candidates"] = actual_candidates
                    payload["online_candidate_ids"] = actual_ids
                    payload["online_candidate_count"] = len(actual_ids)
                    payload["online_decision_trace_aligned"] = True
                    row.update({
                        "trace_aligned": True,
                        "n_candidates": len(actual_ids),
                        "selected_robot": trace.get("selected_robot"),
                        "baseline_robot": trace.get("baseline_robot"),
                        "work_drift_modified_decision": trace.get(
                            "work_drift_modified_decision"
                        ),
                    })
                else:
                    payload["online_decision_trace_aligned"] = False
                    row["trace_aligned"] = False
                if self.capture_policy == "all_modified_plus_interval_controls":
                    if (
                        trace is None
                        or trace.get("selected_robot") is None
                        or int(trace.get("scored_candidate_count", 0)) < 2
                        or int(trace.get("work_drift_horizon", -1)) <= 0
                    ):
                        # Control snapshots must be immediately replayable;
                        # no-scored/deferred contexts do not define competing
                        # online robot rankings.
                        continue
                    is_control_tick = (
                        int(payload["decision_tick"]) % self.sample_interval == 0
                    )
                    is_modified = bool(
                        (trace or {}).get("work_drift_modified_decision")
                    )
                    if not is_control_tick and not is_modified:
                        continue
                    payload["snapshot_selection_reason"] = (
                        "analytic_modified_selection"
                        if is_modified else "interval_control"
                    )
                    row["snapshot_selection_reason"] = payload[
                        "snapshot_selection_reason"
                    ]
                payload["decision_meta"] = dict(meta)
                fname = (f"{self.run_id}_tick{payload['decision_tick']}"
                         f"_group{payload['candidate_group_id']}.pkl")
                with open(os.path.join(self.out_dir, fname), "wb") as f:
                    pickle.dump(payload, f,
                                protocol=pickle.HIGHEST_PROTOCOL)
                row["decision_meta"] = dict(meta)
                row["file"] = fname
                self._index_rows.append(row)
            self._pending = []

        # Counter maintenance mirrors data_collector.on_post_tick
        # (decay 0.85; edge counter counts actual moves).
        for nid in self._flow_counter:
            self._flow_counter[nid] *= 0.85
        for agent in world.agents:
            nid = self._node_map.get(agent.position)
            if nid is not None:
                self._flow_counter[nid] = self._flow_counter.get(nid, 0.0) + 1.0
        if tick != self._last_edge_tick:
            self._last_edge_tick = tick
            for key in self._edge_flow_counter:
                self._edge_flow_counter[key] *= 0.85
            for agent in world.agents:
                if agent.moved_this_tick and agent.previous_position != agent.position:
                    a = self._node_map.get(agent.previous_position)
                    b = self._node_map.get(agent.position)
                    if a is not None and b is not None:
                        self._edge_flow_counter[(a, b)] = (
                            self._edge_flow_counter.get((a, b), 0.0) + 1.0
                        )

        nf = extract_node_features(
            world, self._node_map, self._local_capacity,
            self._bottleneck_score, self._node_type_arr, self._adj,
            self._flow_counter, reservation_window=self.reservation_window,
        )
        self._fh.push(nf)

    # ------------------------------------------------------------------

    def save(self) -> str:
        path = os.path.join(self.out_dir, f"snapindex_{self.run_id}.json")
        header = {
            "schema_version": PHASE_C_SNAPINDEX_SCHEMA_VERSION,
            "phase_c_data_contract": PHASE_C_DATA_CONTRACT,
            "phase_c_round": self.phase_c_round,
            "training_source_policy": "world_model_on_policy",
            "external_baseline_training_samples": False,
            "td_target_enabled": False,
            "td_value_head_enabled": False,
            "run_id": self.run_id,
            "arm_label": self.meta.get("arm_label"),
            "load": self.meta.get("load"),
            "seed": self.meta.get("seed"),
            "config_path": self.meta.get("config"),
            "checkpoint_path": self.meta.get("checkpoint_path"),
            "path_planner_override": self.meta.get(
                "path_planner_override"
            ),
            "path_planner_params_override": self.meta.get(
                "path_planner_params_override"
            ),
            "sample_interval": self.sample_interval,
            "top_m": self.top_m,
            "candidate_scope": self.candidate_scope,
            "attach_assigner_trace": self.attach_assigner_trace,
            "require_trace_alignment": self.require_trace_alignment,
            "capture_policy": self.capture_policy,
            "min_group_size": self.min_group_size,
            "candidate_robot_mode": self.candidate_robot_mode,
            "include_no_assign_candidate": (
                self.include_no_assign_candidate
            ),
            "max_contexts_per_tick": self.max_contexts_per_tick,
            "date": datetime.date.today().isoformat(),
            "n_snapshots": len(self._index_rows),
            "decisions": self._index_rows,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(header, f, indent=2, ensure_ascii=False)
        return path
