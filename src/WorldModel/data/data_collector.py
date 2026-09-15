"""
World Model Data Collector (v3 — counterfactual)
=================================================
V2 adjustments from adjust_code_v2.md:
  - Sampling happens BEFORE task_assigner.assign() via pre_assignment_callbacks
  - Each candidate gets its own counterfactual rollout (clone world → force apply → H ticks)
  - realized_cost comes from short rollout, not heuristic
  - No pending buffer — samples are finalized immediately
  - No dummy samples — skip if no pending orders or idle robots
  - station_node_ids saved per sample
  - action_edge (E, 4) saved per sample
  - Data quality stats printed at save time
"""

import os
import sys
import copy
import inspect
import time
import random
import pickle
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import torch

from WorldState.world import WorldState
from WorldState.agent_state import AgentStatus

from WorldModel.graph_builder import (
    build_static_graph,
    extract_node_features,
    extract_edge_features,
    extract_demand_context,
    build_action_field,
    build_action_edge_field,
    compute_preview_legs,
    FeatureHistory,
)
from WorldModel.candidate_generator import (
    ASSIGN_ROBOT_ACTION_TYPE,
    NO_ASSIGN_ACTION_SCHEMA_VERSION,
    NO_ASSIGN_ACTION_TYPE,
    NO_ASSIGN_ENCODING,
    generate_robot_candidates,
    compute_heuristic_cost,
    build_candidate_assignment,
    is_no_assign_candidate,
)
from WorldModel.counterfactual_rollout import evaluate_candidate_rollout


def summarize_lyapunov_collection(samples) -> dict:
    """Summarise v2 arrival/traffic coverage without altering legacy rows."""
    l0_samples = [
        sample for sample in samples
        if sample.get("lyapunov_l0_start") is not None
    ]
    schemas = Counter(str(sample.get(
        "lyapunov_l0_collection_schema_version", "legacy_or_missing"
    )) for sample in l0_samples)
    traffic_nonzero = 0
    blocked_samples = 0
    blocked_with_signal = 0
    post_action_present = 0
    arrival_records = {"start": 0, "post_action": 0, "end": 0}
    already_arrived_records = 0

    for sample in l0_samples:
        if sample.get("lyapunov_l0_post_action") is not None:
            post_action_present += 1
        for endpoint in arrival_records:
            snapshot = sample.get(f"lyapunov_l0_{endpoint}") or {}
            manifest = snapshot.get("arrival_manifest") or []
            arrival_records[endpoint] += len(manifest)
            already_arrived_records += sum(
                str(record.get("agent_status", "")).upper()
                in {"QUEUING", "DELIVERING", "EXITING"}
                for record in manifest
            )

        names = list(sample.get("lyapunov_l0_traffic_trajectory_names") or ())
        trajectory = sample.get("lyapunov_l0_traffic_trajectory")
        max_pressure = 0.0
        if trajectory is not None and "total_pressure" in names:
            column = names.index("total_pressure")
            tensor = torch.as_tensor(trajectory)
            valid = int(torch.as_tensor(
                sample.get("future_mask", [])
            ).sum().item())
            if tensor.ndim == 2 and tensor.size(0) > 0:
                valid = max(0, min(valid, tensor.size(0)))
                if valid > 0:
                    max_pressure = float(tensor[:valid, column].max().item())
        if max_pressure > 0.0:
            traffic_nonzero += 1
        if int(sample.get("rollout_blocked_moves", 0)) > 0:
            blocked_samples += 1
            if max_pressure > 0.0:
                blocked_with_signal += 1

    return {
        "valid_samples": len(l0_samples),
        "collection_schema_histogram": dict(sorted(schemas.items())),
        "post_action_snapshot_samples": post_action_present,
        "arrival_manifest_records": arrival_records,
        "already_arrived_manifest_records": already_arrived_records,
        "traffic_nonzero_samples": traffic_nonzero,
        "blocked_samples": blocked_samples,
        "blocked_samples_with_nonzero_traffic": blocked_with_signal,
        "blocked_to_traffic_coverage": (
            float(blocked_with_signal) / float(blocked_samples)
            if blocked_samples else None
        ),
    }


class WorldModelDataCollector:
    """Collects counterfactual training samples with pre-assignment sampling."""

    def __init__(
        self,
        output_dir: str = "DataGen/wm_data",
        history_len: int = 4,
        rollout_horizon: int = 10,
        sample_interval: int = 5,
        top_m_candidates: int = 5,
        min_group_size: int = 2,
        reservation_window: int = 1,
        action_path_mode: int = 0,
        delay_scale=None,
        station_queue_delta_scale=None,
        stalled_ratio_threshold: float = 0.3,
        risk_duration: int = 8,
        station_queue_scale=None,
        station_load_scale=None,
        max_groups_per_tick: Optional[int] = None,
        save_snapshots: bool = False,
        snapshot_dir: Optional[str] = None,
        run_id: str = "",
        record_lyapunov_l0: bool = False,
        lyapunov_l0_config: Optional[dict] = None,
        rollout_continuation_mode: str = "isolated",
        candidate_robot_mode: str = "nearest",
        include_no_assign_candidate: bool = False,
    ):
        self.output_dir = output_dir
        self.history_len = history_len
        self.rollout_horizon = rollout_horizon
        self.sample_interval = sample_interval
        self.top_m_candidates = top_m_candidates
        self.min_group_size = min_group_size
        self.reservation_window = reservation_window
        self.action_path_mode = action_path_mode
        self._delay_scale = delay_scale
        self._station_queue_delta_scale = station_queue_delta_scale
        self._stalled_ratio_threshold = stalled_ratio_threshold
        self._risk_duration = risk_duration
        self._station_queue_scale = station_queue_scale
        self._station_load_scale = station_load_scale
        self._max_groups_per_tick = max_groups_per_tick
        self._save_snapshots = save_snapshots
        self._snapshot_dir = snapshot_dir
        self._run_id = run_id
        self._snapshot_count = 0
        self._record_lyapunov_l0 = bool(record_lyapunov_l0)
        self._lyapunov_l0_config = dict(lyapunov_l0_config or {})
        if rollout_continuation_mode not in ("isolated", "behavior"):
            raise ValueError(
                "rollout_continuation_mode must be 'isolated' or 'behavior'"
            )
        self._rollout_continuation_mode = rollout_continuation_mode
        if candidate_robot_mode not in (
            "nearest", "eta_stratified", "stratified",
        ):
            raise ValueError(
                "candidate_robot_mode must be one of: nearest, "
                "eta_stratified, stratified"
            )
        self._candidate_robot_mode = candidate_robot_mode
        self._include_no_assign_candidate = bool(
            include_no_assign_candidate
        )
        if (
            self._include_no_assign_candidate
            and rollout_continuation_mode != "isolated"
        ):
            raise ValueError(
                "native NO_ASSIGN collection requires isolated rollout"
            )
        if "candidate_mode" not in inspect.signature(
                generate_robot_candidates).parameters:
            raise RuntimeError(
                "WorldModel/data/candidate_generator.py is out of sync: "
                "generate_robot_candidates() must accept candidate_mode. "
                "Deploy the repaired candidate_generator.py before collecting."
            )

        self._initialized = False
        self._edge_index = None
        self._node_map = None
        self._inv_node_map = None
        self._local_capacity = None
        self._bottleneck_score = None
        self._node_type_arr = None
        self._adj = None
        self._num_nodes = 0
        self._num_stations = 0
        self._station_node_ids: List[int] = []

        self._feature_history: Optional[FeatureHistory] = None
        self._flow_counter: Dict[int, float] = {}
        self._edge_flow_counter: Dict[Tuple[int, int], float] = {}
        self._last_edge_flow_update_tick: int = -1

        self._finalized_samples: List[dict] = []
        self._group_sizes: List[int] = []
        self._sample_count = 0
        self._group_count = 0

    def _init_graph(self, world: WorldState):
        (
            self._edge_index,
            self._node_map,
            self._inv_node_map,
            self._local_capacity,
            self._bottleneck_score,
            self._node_type_arr,
            self._adj,
        ) = build_static_graph(world.map_state)
        self._num_nodes = len(self._node_map)
        self._num_stations = len(world.map_state.station_positions)
        self._feature_history = FeatureHistory(
            self._num_nodes, feat_dim=10, history_len=self.history_len
        )
        station_ids = sorted(world.map_state.station_positions.keys())
        for sid in station_ids:
            spos = world.map_state.station_positions[sid]
            nid = self._node_map.get(spos)
            if nid is not None:
                self._station_node_ids.append(nid)

        self._initialized = True

    # ------------------------------------------------------------------
    # on_pre_assignment — called BEFORE task_assigner.assign()
    # ------------------------------------------------------------------

    def on_pre_assignment(self, engine):
        """Generate candidate groups and run counterfactual rollouts.

        This is the main data collection entry point, called via
        engine.pre_assignment_callbacks before task assignment.
        """
        world = engine.world
        if not self._initialized:
            self._init_graph(world)

        tick = world.tick
        if tick < self.history_len:
            return
        if tick % self.sample_interval != 0:
            return
        if not self._feature_history or not self._feature_history.is_ready:
            return

        contexts = engine.task_assigner.propose_assignment_contexts(
            world, max_contexts=self._max_groups_per_tick,
        )
        if not contexts:
            return

        groups = generate_robot_candidates(
            contexts,
            world,
            self.top_m_candidates,
            candidate_mode=self._candidate_robot_mode,
            lyapunov_config=self._lyapunov_l0_config,
            node_map=self._node_map,
            edge_flow_counter=self._edge_flow_counter,
            bottleneck_score=self._bottleneck_score,
            reservation_window=max(
                self.reservation_window,
                int(self._lyapunov_l0_config.get(
                    "reservation_window", self.reservation_window
                )),
            ),
            include_no_assign_candidate=(
                self._include_no_assign_candidate
            ),
        )
        if not groups:
            return

        eligible = sum(
            1 for g in groups
            if int(g.get("robot_candidate_count", len(g["candidates"])))
            >= self.min_group_size
        )
        total_cands = sum(
            len(g["candidates"]) for g in groups
            if int(g.get("robot_candidate_count", len(g["candidates"])))
            >= self.min_group_size
        )
        sys.stdout.write(
            f"\n    [tick {tick}] pre_assign: {len(contexts)} ctx -> "
            f"{len(groups)} grp ({eligible} eligible), "
            f"~{total_cands} cands, H={self.rollout_horizon}\n"
        )
        sys.stdout.flush()
        _pa_t0 = time.time()

        history = self._feature_history.get_history()
        nf_latest = self._feature_history.get_latest()
        edge_feat = extract_edge_features(
            self._edge_index, self._node_map, self._inv_node_map,
            self._local_capacity, world,
            adj=self._adj,
            edge_flow_counter=self._edge_flow_counter,
            reservation_window=self.reservation_window,
        )
        demand = extract_demand_context(world)

        preview_planner = engine.path_planner if self.action_path_mode == 1 else None

        _gi = 0
        for group in groups:
            fc = group["fixed_context"]
            candidates = group["candidates"]
            robot_candidate_count = int(
                group.get("robot_candidate_count", len(candidates))
            )

            # ``min_group_size`` remains a robot-coverage requirement.  A
            # single robot plus NO_ASSIGN is not silently promoted into an
            # otherwise ineligible ranking group.
            if robot_candidate_count < self.min_group_size:
                continue

            _gi += 1
            _grp_t0 = time.time()
            self._group_count += 1
            self._group_sizes.append(len(candidates))

            # --- B0.7: Save decision-point snapshot for long-risk label generation ---
            if self._save_snapshots and self._snapshot_dir:
                from WorldState.task_state import Task
                from WorldState.order_state import Order
                snapshot_payload = {
                    "schema_version": "phaseB_b0_decision_snapshot_v1",
                    "run_id": self._run_id,
                    "decision_tick": tick,
                    "candidate_group_id": group["group_id"],
                    "world_snapshot": copy.deepcopy(world),
                    "task_next_id": Task._next_id,
                    "order_next_id": Order._next_id,
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.random.get_rng_state(),
                    "config": copy.deepcopy(engine.config),
                    "path_planner_state": copy.deepcopy(engine.path_planner),
                    "order_generator_state": copy.deepcopy(engine.order_generator),
                    "candidates": copy.deepcopy(candidates),
                    "fixed_context": copy.deepcopy(fc),
                }
                os.makedirs(self._snapshot_dir, exist_ok=True)
                snap_path = os.path.join(
                    self._snapshot_dir,
                    f"{self._run_id}_tick{tick}_group{group['group_id']}.pkl",
                )
                with open(snap_path, "wb") as f:
                    pickle.dump(snapshot_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                self._snapshot_count += 1

            for cand in candidates:
                no_assign = is_no_assign_candidate(cand)
                action_type = (
                    NO_ASSIGN_ACTION_TYPE
                    if no_assign else ASSIGN_ROBOT_ACTION_TYPE
                )
                assignment = build_candidate_assignment(cand, fc)
                legs = compute_preview_legs(
                    assignment, world.map_state, self._node_map,
                    path_planner=preview_planner, world=world,
                )
                an, ag = build_action_field(
                    assignment, world, self._node_map,
                    self._inv_node_map, self._local_capacity,
                    node_features=nf_latest,
                    precomputed_legs=legs,
                )
                ae = build_action_edge_field(
                    assignment, self._edge_index, self._node_map,
                    world.map_state,
                    precomputed_legs=legs,
                )

                rollout_result = evaluate_candidate_rollout(
                    world=world,
                    candidate=cand,
                    fixed_context=fc,
                    config=engine.config,
                    path_planner=engine.path_planner,
                    horizon=self.rollout_horizon,
                    node_map=self._node_map,
                    local_capacity=self._local_capacity,
                    bottleneck_score=self._bottleneck_score,
                    adj=self._adj,
                    reservation_window=self.reservation_window,
                    delay_scale=self._delay_scale,
                    station_queue_delta_scale=self._station_queue_delta_scale,
                    stalled_ratio_threshold=self._stalled_ratio_threshold,
                    risk_duration=self._risk_duration,
                    station_queue_scale=self._station_queue_scale,
                    station_load_scale=self._station_load_scale,
                    record_lyapunov_l0=self._record_lyapunov_l0,
                    lyapunov_l0_config=self._lyapunov_l0_config,
                    rollout_continuation_mode=self._rollout_continuation_mode,
                    continuation_order_generator=(
                        engine.order_generator
                        if self._rollout_continuation_mode == "behavior"
                        else None
                    ),
                    continuation_task_assigner=(
                        engine.task_assigner
                        if self._rollout_continuation_mode == "behavior"
                        else None
                    ),
                )

                heuristic_cost_valid = not no_assign
                heuristic_cost = (
                    compute_heuristic_cost(cand, fc, world)
                    if heuristic_cost_valid else None
                )
                candidate_robot_token = (
                    "NO_ASSIGN" if no_assign else str(cand["robot_id"])
                )

                sample = {
                    # Stable provenance is required for run/seed-level splits
                    # and for within-context ranking.  ``candidate_group_id``
                    # alone can collide because tick/order counters restart in
                    # every simulation process.
                    "run_id": self._run_id,
                    "simulation_seed": getattr(
                        getattr(getattr(world, "config", None), "simulation", None),
                        "seed",
                        None,
                    ),
                    "continuation_policy": (
                        (
                            f"{type(engine.task_assigner).__module__}."
                            f"{type(engine.task_assigner).__name__}"
                        )
                        if self._rollout_continuation_mode == "behavior"
                        else (
                            "isolated_native_no_assign"
                            if no_assign
                            else "isolated_forced_candidate"
                        )
                    ),
                    "rollout_continuation_mode": rollout_result.get(
                        "rollout_continuation_mode",
                        self._rollout_continuation_mode,
                    ),
                    "node_history": history.clone(),
                    "edge_index": self._edge_index,
                    "edge_features": edge_feat,
                    "demand_context": demand,
                    "action_node": an,
                    "action_global": ag,
                    "action_edge": ae,
                    "candidate_group_id": group["group_id"],
                    "decision_tick": tick,
                    "action_type": action_type,
                    "action_schema_version": (
                        NO_ASSIGN_ACTION_SCHEMA_VERSION
                        if self._include_no_assign_candidate else None
                    ),
                    "action_encoding": (
                        NO_ASSIGN_ENCODING if no_assign else "route_fields_v1"
                    ),
                    "candidate_key": (
                        f"{group['group_id']}"
                        f"_r{candidate_robot_token}"
                        f"_o{fc['order_id']}"
                        f"_p{fc['pod_id']}"
                        f"_s{fc['station_id']}"
                        f"_t{tick}"
                    ),
                    "candidate_info": {
                        "action_type": action_type,
                        "robot_id": cand["robot_id"],
                        "robot_start": cand["robot_start"],
                        "candidate_policy": cand["candidate_policy"],
                        "candidate_selection_mode": cand.get(
                            "candidate_selection_mode", "nearest"
                        ),
                        "eta": cand.get("eta"),
                        "eta_bin": cand.get("eta_bin"),
                        "arrival_delta_preview": cand.get(
                            "arrival_delta_preview"
                        ),
                        "route_conflict_preview": cand.get(
                            "route_conflict_preview"
                        ),
                        "route_length_preview": cand.get(
                            "route_length_preview"
                        ),
                    },
                    "fixed_context": {
                        "order_id": fc["order_id"],
                        "pod_id": fc["pod_id"],
                        "pod_location": fc["pod_location"],
                        "station_id": fc["station_id"],
                        "station_location": fc["station_location"],
                        "entry_position": fc.get("entry_position"),
                        "exit_position": fc.get("exit_position"),
                        "return_location": fc["return_location"],
                    },
                    "station_node_ids": torch.tensor(
                        self._station_node_ids, dtype=torch.long,
                    ),
                    "future_node_labels": rollout_result["future_node_labels"],
                    "future_system_labels": rollout_result["future_system_labels"],
                    "future_station_labels": rollout_result["future_station_labels"],
                    "future_mask": rollout_result["future_mask"],
                    "realized_cost": rollout_result["realized_cost"],
                    "heuristic_cost": heuristic_cost,
                    "heuristic_cost_valid": heuristic_cost_valid,
                    "rollout_vertex_conflicts": rollout_result["rollout_vertex_conflicts"],
                    "rollout_swap_conflicts": rollout_result["rollout_swap_conflicts"],
                    "rollout_blocked_moves": rollout_result["rollout_blocked_moves"],
                    "rollout_generated_orders": rollout_result.get(
                        "rollout_generated_orders", 0
                    ),
                    "rollout_assigned_tasks": rollout_result.get(
                        "rollout_assigned_tasks", 0
                    ),
                    "future_demand_context": rollout_result.get(
                        "future_demand_context"
                    ),
                    "no_assign_applied": bool(
                        rollout_result.get("no_assign_applied", False)
                    ),
                    "no_assign_audit": rollout_result.get(
                        "no_assign_audit"
                    ),
                }

                if self._record_lyapunov_l0:
                    from WorldModel.core.lyapunov import (
                        LYAPUNOV_COLLECTION_SCHEMA_VERSION,
                    )
                    # Tag even invalid/right-censored samples so a file-level
                    # validator can distinguish v2 data from an untagged v1
                    # sample instead of silently mixing the two schemas.
                    sample.update({
                        "lyapunov_l0_collection_schema_version": (
                            LYAPUNOV_COLLECTION_SCHEMA_VERSION
                        ),
                        "lyapunov_l0_valid": bool(
                            rollout_result.get("lyapunov_l0_valid", False)
                        ),
                    })

                if rollout_result.get("lyapunov_l0_valid"):
                    sample.update({
                        "lyapunov_l0_collection_schema_version": rollout_result[
                            "lyapunov_l0_collection_schema_version"
                        ],
                        "lyapunov_l0_start": rollout_result["lyapunov_l0_start"],
                        "lyapunov_l0_post_action": rollout_result[
                            "lyapunov_l0_post_action"
                        ],
                        "lyapunov_l0_end": rollout_result["lyapunov_l0_end"],
                        "lyapunov_l0_immediate_delta": rollout_result[
                            "lyapunov_l0_immediate_delta"
                        ],
                        "lyapunov_l0_delta": rollout_result["lyapunov_l0_delta"],
                        "lyapunov_l0_trajectory": rollout_result[
                            "lyapunov_l0_trajectory"
                        ],
                        "analytic_work_relief_trajectory_schema_version": (
                            rollout_result[
                                "analytic_work_relief_trajectory_schema_version"
                            ]
                        ),
                        "lyapunov_l0_station_ids": rollout_result[
                            "lyapunov_l0_station_ids"
                        ],
                        "lyapunov_l0_station_work_trajectory": rollout_result[
                            "lyapunov_l0_station_work_trajectory"
                        ],
                        "lyapunov_l0_progress": rollout_result[
                            "lyapunov_l0_progress"
                        ],
                        "lyapunov_l0_traffic_trajectory_names": rollout_result[
                            "lyapunov_l0_traffic_trajectory_names"
                        ],
                        "lyapunov_l0_traffic_trajectory": rollout_result[
                            "lyapunov_l0_traffic_trajectory"
                        ],
                        "lyapunov_l0_config": rollout_result[
                            "lyapunov_l0_config"
                        ],
                    })

                self._finalized_samples.append(sample)
                self._sample_count += 1

            _grp_dt = time.time() - _grp_t0
            sys.stdout.write(
                f"      grp {_gi}/{eligible}: {len(candidates)} cands "
                f"({_grp_dt:.1f}s) samples={self._sample_count}\n"
            )
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # on_post_tick — called after full tick (movement, actions, etc.)
    # ------------------------------------------------------------------

    def on_post_tick(self, engine):
        """Maintain feature history and flow counter after each tick."""
        world = engine.world
        if not self._initialized:
            self._init_graph(world)

        self._update_flow_counter(world)
        self._update_edge_flow_counter(world)

        node_feat = extract_node_features(
            world, self._node_map, self._local_capacity,
            self._bottleneck_score, self._node_type_arr,
            self._adj,
            self._flow_counter,
            reservation_window=self.reservation_window,
        )
        self._feature_history.push(node_feat)

    # ------------------------------------------------------------------
    # Pairwise ranking data
    # ------------------------------------------------------------------

    def build_pairwise_data(self, epsilon: float = 0.01) -> List[dict]:
        """Build pairwise ranking data from candidate groups using realized_cost."""
        groups: Dict[str, List[dict]] = {}
        for sample in self._finalized_samples:
            gid = sample.get("candidate_group_id", "")
            if gid not in groups:
                groups[gid] = []
            groups[gid].append(sample)

        pairs = []
        for gid, members in groups.items():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    ci = members[i]["realized_cost"]
                    cj = members[j]["realized_cost"]
                    if abs(ci - cj) > epsilon:
                        if ci < cj:
                            pairs.append({
                                "better": i, "worse": j, "group": gid,
                                "sample_i": members[i], "sample_j": members[j],
                            })
                        else:
                            pairs.append({
                                "better": j, "worse": i, "group": gid,
                                "sample_i": members[j], "sample_j": members[i],
                            })
        return pairs

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _update_flow_counter(self, world: WorldState, decay: float = 0.85):
        for nid in self._flow_counter:
            self._flow_counter[nid] *= decay
        for agent in world.agents:
            nid = self._node_map.get(agent.position)
            if nid is not None:
                self._flow_counter[nid] = self._flow_counter.get(nid, 0.0) + 1.0

    def _update_edge_flow_counter(self, world: WorldState, decay: float = 0.85):
        if world.tick == self._last_edge_flow_update_tick:
            return
        self._last_edge_flow_update_tick = world.tick
        for key in self._edge_flow_counter:
            self._edge_flow_counter[key] *= decay
        for agent in world.agents:
            if agent.moved_this_tick and agent.previous_position != agent.position:
                prev_nid = self._node_map.get(agent.previous_position)
                cur_nid = self._node_map.get(agent.position)
                if prev_nid is not None and cur_nid is not None:
                    edge_key = (prev_nid, cur_nid)
                    self._edge_flow_counter[edge_key] = (
                        self._edge_flow_counter.get(edge_key, 0.0) + 1.0
                    )

    def save(self, filename: str = "wm_train_data.pt"):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        torch.save(self._finalized_samples, path)
        self._print_data_quality()
        return path

    def _print_data_quality(self):
        """Print data quality stats per adjust_v2_labels_schema.md §4."""
        groups: Dict[str, List[dict]] = {}
        for s in self._finalized_samples:
            gid = s.get("candidate_group_id", "")
            if gid not in groups:
                groups[gid] = []
            groups[gid].append(s)

        total = len(self._finalized_samples)
        group_count = len(groups)
        sizes = [len(m) for m in groups.values()]
        size_counter = Counter(sizes)

        pair_count = 0
        cost_stds = []
        for members in groups.values():
            costs = [m["realized_cost"] for m in members]
            for i in range(len(costs)):
                for j in range(i + 1, len(costs)):
                    if abs(costs[i] - costs[j]) > 0.01:
                        pair_count += 1
            if len(costs) >= 2:
                mean_c = sum(costs) / len(costs)
                var_c = sum((c - mean_c) ** 2 for c in costs) / len(costs)
                cost_stds.append(var_c ** 0.5)

        mean_group_size = sum(sizes) / max(len(sizes), 1)
        mean_cost_std = sum(cost_stds) / max(len(cost_stds), 1)

        total_vertex = sum(s.get("rollout_vertex_conflicts", 0)
                           for s in self._finalized_samples)
        total_swap = sum(s.get("rollout_swap_conflicts", 0)
                         for s in self._finalized_samples)
        total_blocked = sum(s.get("rollout_blocked_moves", 0)
                            for s in self._finalized_samples)
        total_generated_orders = sum(
            s.get("rollout_generated_orders", 0)
            for s in self._finalized_samples
        )
        total_assigned_tasks = sum(
            s.get("rollout_assigned_tasks", 0)
            for s in self._finalized_samples
        )
        samples_with_conflicts = sum(
            1 for s in self._finalized_samples
            if s.get("rollout_vertex_conflicts", 0) > 0
            or s.get("rollout_swap_conflicts", 0) > 0
        )
        no_assign_samples = [
            s for s in self._finalized_samples
            if s.get("action_type") == NO_ASSIGN_ACTION_TYPE
        ]
        no_assign_audit_failures = sum(
            1 for sample in no_assign_samples
            if not bool(
                (sample.get("no_assign_audit") or {}).get(
                    "immediate_context_unchanged", False
                )
            )
            or bool(
                (sample.get("no_assign_audit") or {}).get(
                    "tasks_added_during_isolated_rollout", []
                )
            )
        )

        print(f"\n  === Data Quality Report ===")
        print(f"  total_samples            : {total}")
        print(f"  candidate_group_count    : {group_count}")
        print(f"  mean_group_size          : {mean_group_size:.2f}")
        print(f"  group_size_histogram     : {dict(sorted(size_counter.items()))}")
        print(f"  pairwise_pair_count      : {pair_count}")
        print(f"  realized_cost_std_mean   : {mean_cost_std:.4f}")
        print(f"  rollout_vertex_conflicts : {total_vertex}")
        print(f"  rollout_swap_conflicts   : {total_swap}")
        print(f"  rollout_blocked_moves    : {total_blocked}")
        print(f"  rollout_generated_orders : {total_generated_orders}")
        print(f"  rollout_assigned_tasks   : {total_assigned_tasks}")
        print(f"  rollout_continuation     : {self._rollout_continuation_mode}")
        print(f"  native_no_assign_samples : {len(no_assign_samples)}")
        print(f"  no_assign_audit_failures : {no_assign_audit_failures}")
        print(f"  samples_with_conflicts   : {samples_with_conflicts}/{total}")
        lyapunov_quality = summarize_lyapunov_collection(
            self._finalized_samples
        )
        if lyapunov_quality["valid_samples"]:
            print(
                "  l0_collection_schemas    : "
                f"{lyapunov_quality['collection_schema_histogram']}"
            )
            print(
                "  l0_arrival_records       : "
                f"{lyapunov_quality['arrival_manifest_records']}"
            )
            print(
                "  l0_traffic_nonzero       : "
                f"{lyapunov_quality['traffic_nonzero_samples']}/"
                f"{lyapunov_quality['valid_samples']}"
            )
            print(
                "  blocked->traffic coverage: "
                f"{lyapunov_quality['blocked_samples_with_nonzero_traffic']}/"
                f"{lyapunov_quality['blocked_samples']}"
            )
            print(
                "  arrived-in-manifest      : "
                f"{lyapunov_quality['already_arrived_manifest_records']}"
            )
        if self._save_snapshots:
            print(f"  snapshots_saved          : {self._snapshot_count}")
            print(f"  snapshot_dir             : {self._snapshot_dir}")

    @property
    def num_samples(self) -> int:
        return len(self._finalized_samples)

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def demand_dim(self) -> int:
        if self._finalized_samples:
            return self._finalized_samples[0]["demand_context"].shape[0]
        if self._num_stations > 0:
            return 5 + self._num_stations
        return 9
