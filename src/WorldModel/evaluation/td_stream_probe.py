"""TD adjacent-segment stream probe for online evaluation runs (6.2 V2).

Registered as an ``on_tick`` callback (post-step phase: fires after the world
step completes and before ``advance_tick()``, i.e. the same frame convention
as ``dump_td_tuples._rollout_candidate_capture``). Records ``unified_risk``
every tick and observation frames at stride ticks, so a deployment run
natively yields ``(s_t, risk_seq[t..t+K], s_{t+K})`` segments — assembled
offline by ``WorldModel/data/build_td_stream_tuples.py``.

Scope declaration: the product is the 6.2 V-mode on-policy TD data source
only, NOT the complete Phase C round-1 data closure. Candidate ranking /
counterfactual relabeling still requires the decision-point snapshot dump
(roadmap §4.a) — a separate task.
"""

import datetime
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch

from WorldModel.graph_builder import (
    FeatureHistory,
    build_static_graph,
    extract_demand_context,
    extract_edge_features,
    extract_node_features,
)
from WorldModel.core.lyapunov import (
    LYAPUNOV_SNAPSHOT_SCHEMA_VERSION,
    LyapunovL0Config,
    PHYSICAL_SUMMARY_NAMES,
    ProductiveProgress,
    compute_lyapunov_snapshot,
    compute_productive_progress,
)
from WorldState.risk import compute_unified_risk

GATE_META_KEYS = (
    "energy_conv_contexts",
    "energy_conv_active_contexts",
    "energy_conv_modified_decisions",
    "energy_conv_gate_g_sum",
)


class TDStreamProbe:
    """Per-tick risk stream + strided observation frames for one online run."""

    def __init__(
        self,
        engine,
        out_dir: str,
        frame_stride: int = 5,
        run_id: str = "run",
        meta: Optional[dict] = None,
        reservation_window: int = 1,
        record_lyapunov_l0: bool = False,
        lyapunov_l0_config: Optional[dict] = None,
    ):
        self.out_dir = out_dir
        self.frame_stride = max(1, int(frame_stride))
        self.run_id = run_id
        self.meta = dict(meta or {})
        self.reservation_window = reservation_window
        self.record_lyapunov_l0 = bool(record_lyapunov_l0)
        config_values = dict(lyapunov_l0_config or {})
        if "eta_bin_edges" in config_values:
            config_values["eta_bin_edges"] = tuple(
                int(value) for value in config_values["eta_bin_edges"]
            )
        self.lyapunov_l0_config = LyapunovL0Config(**config_values)

        (self._edge_index, self._node_map, self._inv_node_map,
         self._local_capacity, self._bottleneck_score,
         self._node_type_arr, self._adj) = build_static_graph(
            engine.world.map_state)

        self._fh = FeatureHistory(len(self._node_map), feat_dim=10,
                                  history_len=4)
        self._flow_counter: Dict[int, float] = {}
        self._edge_flow_counter: Dict[Tuple[int, int], float] = {}
        self._last_edge_tick = -1

        self._station_node_ids: List[int] = []
        for spos in engine.world.map_state.station_positions.values():
            nid = self._node_map.get(spos)
            if nid is not None:
                self._station_node_ids.append(nid)

        # Greedy arm has no .stats — gate_meta rows stay zero there.
        self._prev_stats = self._read_stats(engine)

        self._tick_seq: List[int] = []
        self._risk_seq: List[float] = []
        self._risk_components: List[Tuple[float, float, float]] = []
        self._gate_meta: List[Tuple[float, ...]] = []
        self._frames_by_tick: Dict[int, dict] = {}
        self._previous_lyapunov_snapshot = None
        self._lyapunov_summary: List[Tuple[float, ...]] = []
        self._productive_progress: List[
            Tuple[float, float, float, float, float]
        ] = []
        self._productive_progress_by_station: List[dict] = []

    @staticmethod
    def _read_stats(engine) -> Dict[str, float]:
        stats = getattr(engine.task_assigner, "stats", None) or {}
        return {k: float(stats.get(k, 0.0)) for k in GATE_META_KEYS}

    @property
    def saved_frames(self) -> int:
        return len(self._frames_by_tick)

    def on_tick(self, engine):
        world = engine.world
        tick = int(world.tick)

        risk = compute_unified_risk(world)
        self._tick_seq.append(tick)
        self._risk_seq.append(float(risk["unified_risk"]))
        self._risk_components.append((
            float(risk["stall_ratio"]),
            float(risk["deadlock_ratio"]),
            float(risk["handoff_ratio"]),
        ))

        lyapunov_snapshot = None
        if self.record_lyapunov_l0:
            lyapunov_snapshot = compute_lyapunov_snapshot(
                world, self.lyapunov_l0_config
            )
            if self._previous_lyapunov_snapshot is None:
                progress = ProductiveProgress(
                    horizon=1,
                    productive_by_station={},
                    reverse_by_station={},
                    arrivals_by_station={},
                )
            else:
                progress = compute_productive_progress(
                    self._previous_lyapunov_snapshot,
                    lyapunov_snapshot,
                    horizon=max(
                        lyapunov_snapshot.tick
                        - self._previous_lyapunov_snapshot.tick,
                        1,
                    ),
                )
            self._previous_lyapunov_snapshot = lyapunov_snapshot
            self._lyapunov_summary.append(lyapunov_snapshot.physical_summary())
            self._productive_progress.append((
                progress.productive_total,
                progress.reverse_total,
                progress.arrival_total,
                progress.replan_residual_total,
                progress.route_plan_churn_total,
            ))
            self._productive_progress_by_station.append(progress.to_dict())

        # Counter maintenance mirrors dump_td_tuples._post_step (decay 0.85).
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

        if tick % self.frame_stride == 0 and self._fh.is_ready:
            frame = {
                "node_history": self._fh.get_history().clone(),
                "edge_features": extract_edge_features(
                    self._edge_index, self._node_map, self._inv_node_map,
                    self._local_capacity, world, adj=self._adj,
                    edge_flow_counter=self._edge_flow_counter,
                    reservation_window=self.reservation_window,
                ),
                "demand_context": extract_demand_context(world),
            }
            if lyapunov_snapshot is not None:
                frame["lyapunov_l0"] = lyapunov_snapshot.to_dict()
            self._frames_by_tick[tick] = frame

        cur = self._read_stats(engine)
        self._gate_meta.append(tuple(
            cur[k] - self._prev_stats[k] for k in GATE_META_KEYS))
        self._prev_stats = cur

    def save(self) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        out_path = os.path.join(self.out_dir, f"tdstream_{self.run_id}.pt")
        payload = dict(self.meta)
        payload.update({
            "schema_version": "td_stream_v1",
            "run_id": self.run_id,
            "frame_stride": self.frame_stride,
            "ticks": len(self._tick_seq),
            "edge_index": self._edge_index,
            "station_node_ids": list(self._station_node_ids),
            "gate_meta_keys": list(GATE_META_KEYS),
            "date": datetime.date.today().isoformat(),
            "tick_seq": torch.tensor(self._tick_seq, dtype=torch.int64),
            "risk_seq_full": torch.tensor(self._risk_seq, dtype=torch.float32),
            "risk_components": torch.tensor(
                self._risk_components, dtype=torch.float32).reshape(-1, 3),
            "gate_meta": torch.tensor(
                self._gate_meta, dtype=torch.float32).reshape(-1, len(GATE_META_KEYS)),
            "frames_by_tick": self._frames_by_tick,
            "lyapunov_l0_enabled": self.record_lyapunov_l0,
        })
        if self.record_lyapunov_l0:
            payload.update({
                "lyapunov_l0_version": LYAPUNOV_SNAPSHOT_SCHEMA_VERSION,
                "lyapunov_l0_config": asdict(self.lyapunov_l0_config),
                "lyapunov_l0_summary": torch.tensor(
                    self._lyapunov_summary, dtype=torch.float32
                ).reshape(-1, len(PHYSICAL_SUMMARY_NAMES)),
                "lyapunov_l0_summary_names": list(PHYSICAL_SUMMARY_NAMES),
                "productive_progress": torch.tensor(
                    self._productive_progress, dtype=torch.float32
                ).reshape(-1, 5),
                "productive_progress_names": [
                    "productive", "reverse", "arrival", "replan_residual",
                    "route_plan_churn",
                ],
                "productive_progress_by_station": (
                    self._productive_progress_by_station
                ),
            })
        torch.save(payload, out_path)
        return out_path
