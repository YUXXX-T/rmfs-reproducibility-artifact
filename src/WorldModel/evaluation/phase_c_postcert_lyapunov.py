"""Read-only analytic Lyapunov shadowing for Phase-C post-cert diagnosis.

The helpers in this module deliberately do not alter the selected action.  A
normal :class:`WorldModelTaskAssigner` first makes the exact Phase-C decision;
only then do we attach parameter-free analytic diagnostics to its in-memory
decision trace.  This keeps the failed 471--480 certification immutable while
allowing us to ask whether ``L_work`` would have exposed repeated NO_ASSIGN.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, Mapping, MutableMapping, Sequence

from Policies.TaskAssigner.WorldModelTaskAssigner.world_model_task_assigner import (
    WorldModelTaskAssigner,
)
from WorldModel.core.analytic_work_relief import (
    WorkServiceDurations,
    compute_snapshot_nominal_work_relief,
    compute_virtual_candidate_work_drift,
    work_potential,
)
from WorldModel.core.lyapunov import (
    compute_lyapunov_snapshot,
    preview_assignment_context,
)


SHADOW_SCHEMA_VERSION = "phase_c_postcert_lyapunov_shadow_v1"
NO_ASSIGN_DRIFT_SCHEMA_VERSION = "analytic_no_assign_work_drift_v1"


def _snapshot_payload(snapshot) -> dict:
    if hasattr(snapshot, "to_dict"):
        return snapshot.to_dict()
    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    raise TypeError("snapshot must be a LyapunovSnapshot or mapping")


def compute_no_assign_work_drift(
    snapshot,
    *,
    horizon: int,
    work_weight: float,
) -> dict:
    """Return free-flow ``L_work`` drift when the pending context is untouched.

    Existing pipeline chains are allowed to progress analytically.  Pending
    chains, including the fixed context under consideration, remain pending and
    therefore receive no nominal service.  This exactly mirrors the isolated
    NO_ASSIGN semantics used by the Phase-C training labels, without invoking a
    simulator continuation policy.
    """

    payload = _snapshot_payload(snapshot)
    forecast = compute_snapshot_nominal_work_relief(
        payload,
        horizon=int(horizon),
    )
    capacity = float(payload.get("work_capacity", 0.0))
    if capacity <= 0.0:
        raise ValueError("snapshot.work_capacity must be positive")
    current = tuple(float(value) for value in forecast.post_action_station_work)
    endpoint = tuple(
        float(value) for value in forecast.nominal_endpoint_station_work
    )
    current_potential = work_potential(
        current,
        work_capacity=capacity,
        work_weight=float(work_weight),
    )
    endpoint_potential = work_potential(
        endpoint,
        work_capacity=capacity,
        work_weight=float(work_weight),
    )
    return {
        "schema_version": NO_ASSIGN_DRIFT_SCHEMA_VERSION,
        "horizon": int(horizon),
        "active_pipeline_chains": int(forecast.active_pipeline_chains),
        "current_station_work": list(current),
        "nominal_station_relief": [
            float(value) for value in forecast.nominal_station_relief
        ],
        "endpoint_station_work": list(endpoint),
        "current_work_potential": float(current_potential),
        "endpoint_work_potential": float(endpoint_potential),
        "raw_work_drift": float(endpoint_potential - current_potential),
    }


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return float(sum(rows) / max(len(rows), 1))


def _fixed_context_from_trace(record: Mapping, context) -> dict:
    return {
        "order_id": int(record["order_id"]),
        "pod_id": int(record["pod_id"]),
        "pod_location": getattr(context, "pod_location", record["pod_location"]),
        "station_id": int(record["station_id"]),
        "station_location": getattr(
            context, "station_location", record["station_location"]
        ),
        "entry_position": getattr(context, "entry_position", None),
        "exit_position": getattr(context, "exit_position", None),
        "return_location": getattr(
            context, "return_location", record["return_location"]
        ),
        "order_size": int(
            getattr(context, "order_size", record.get("order_size", 0))
        ),
    }


def build_shadow_record(
    *,
    snapshot,
    trace_record: Mapping,
    context,
    world,
    service_durations: WorkServiceDurations,
    horizons: Sequence[int],
    work_weight: float,
    lyapunov_config,
) -> dict:
    """Attach WM-vs-NO_ASSIGN analytic evidence to one unchanged decision."""

    payload = _snapshot_payload(snapshot)
    candidates = list(trace_record.get("candidates") or ())
    robot_rows = [
        row for row in candidates if row.get("robot_id") is not None
    ]
    no_assign_rows = [
        row
        for row in candidates
        if row.get("action_type") == "no_assign"
        or row.get("robot_id") is None
    ]
    if not robot_rows:
        raise ValueError("shadow trace has no scored robot candidate")
    if len(no_assign_rows) != 1:
        raise ValueError("shadow trace must contain exactly one NO_ASSIGN row")

    wm_best = min(
        robot_rows,
        key=lambda row: (float(row["score"]), int(row["robot_id"])),
    )
    no_assign = no_assign_rows[0]
    fixed_context = _fixed_context_from_trace(trace_record, context)
    candidate_info = {
        "robot_id": int(wm_best["robot_id"]),
        "robot_start": wm_best["robot_start"],
    }
    agent_by_id = {
        int(agent.agent_id): agent for agent in getattr(world, "agents", ())
    }
    best_agent = agent_by_id.get(int(wm_best["robot_id"]))
    if best_agent is None:
        raise ValueError("WM-best robot is absent from the live world")

    immediate = preview_assignment_context(
        world,
        context,
        best_agent,
        config=lyapunov_config,
        snapshot=snapshot,
    )

    horizon_rows = {}
    for horizon in sorted({int(value) for value in horizons}):
        no_assign_drift = compute_no_assign_work_drift(
            snapshot,
            horizon=horizon,
            work_weight=float(work_weight),
        )
        robot_drift = compute_virtual_candidate_work_drift(
            snapshot,
            candidate_info=candidate_info,
            fixed_context=fixed_context,
            service_durations=service_durations,
            horizon=horizon,
            work_weight=float(work_weight),
        ).to_dict()
        advantage = float(
            no_assign_drift["raw_work_drift"]
            - robot_drift["raw_work_drift"]
        )
        current_work = float(no_assign_drift["current_work_potential"])
        horizon_rows[str(horizon)] = {
            "no_assign": no_assign_drift,
            "wm_best_robot": robot_drift,
            # Positive means assigning the WM-best robot dissipates more work.
            "analytic_assign_advantage": advantage,
            "analytic_assign_advantage_relative": float(
                advantage / max(abs(current_work), 1e-12)
            ),
            "analytic_prefers_assignment": bool(advantage > 1e-12),
        }

    chain_states = [
        str(chain.get("state", ""))
        for chain in (payload.get("chains") or {}).values()
    ]
    station_work = [
        float(value) for value in (payload.get("station_work") or {}).values()
    ]
    queue_ratios = [
        float(value)
        for value in (payload.get("station_queue_ratio") or {}).values()
    ]
    traffic = {
        str(key): float(value)
        for key, value in (payload.get("traffic_diagnostics") or {}).items()
        if isinstance(value, (int, float))
    }
    order = getattr(getattr(world, "order_state", None), "orders", {}).get(
        int(trace_record["order_id"])
    )
    order_age = None
    if order is not None:
        order_age = max(
            0,
            int(getattr(world, "tick", 0))
            - int(getattr(order, "created_at", 0)),
        )

    return {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "tick": int(trace_record.get("tick", -1)),
        "context_idx": int(trace_record.get("context_idx", -1)),
        "order_id": int(trace_record["order_id"]),
        "pod_id": int(trace_record["pod_id"]),
        "station_id": int(trace_record["station_id"]),
        "selected_action_type": trace_record.get("selected_action_type"),
        "action_status": trace_record.get("action_status"),
        "candidate_count": int(trace_record.get("candidate_count", 0)),
        "wm_best_robot_id": int(wm_best["robot_id"]),
        "wm_best_robot_score": float(wm_best["score"]),
        "no_assign_wm_score": float(no_assign["score"]),
        # Positive means the frozen WM prefers NO_ASSIGN.
        "wm_no_assign_advantage": float(
            float(wm_best["score"]) - float(no_assign["score"])
        ),
        "immediate_assignment_arrival_delta": float(immediate.delta_total),
        "immediate_assignment_eta": float(immediate.eta),
        "immediate_assignment_eta_bin": int(immediate.eta_bin),
        "naive_immediate_l0_prefers_no_assign": bool(
            float(immediate.delta_total) > 0.0
        ),
        "current_lyapunov": {
            "total": float(payload["total"]),
            "components": {
                str(key): float(value)
                for key, value in (payload.get("components") or {}).items()
            },
            "station_work_sum": float(sum(station_work)),
            "station_work_max": float(max(station_work, default=0.0)),
            "station_queue_ratio_mean": _mean(queue_ratios),
            "station_queue_ratio_max": float(max(queue_ratios, default=0.0)),
            "pending_chains": int(chain_states.count("pending")),
            "pipeline_chains": int(chain_states.count("pipeline")),
            "unresolved_pending_orders": int(
                payload.get("unresolved_pending_orders", 0)
            ),
            "traffic_diagnostics": traffic,
        },
        "context_order_age": order_age,
        "horizons": horizon_rows,
        "candidate_scope": "frozen_wm_best_robot_vs_global_no_assign",
    }


class LyapunovShadowWorldModelTaskAssigner(WorldModelTaskAssigner):
    """Exact Phase-C assigner with post-decision, non-perturbing diagnostics."""

    def __init__(
        self,
        *args,
        shadow_horizons: Sequence[int] = (5, 10, 20, 50),
        **kwargs,
    ):
        if kwargs.get("work_drift_mode", "off") != "off":
            raise ValueError("post-cert shadow forbids active work-drift scoring")
        if kwargs.get("lyapunov_l0_mode", "off") != "off":
            raise ValueError("post-cert shadow forbids active Lyapunov scoring")
        kwargs["include_no_assign_candidate"] = True
        kwargs["decision_trace_enabled"] = True
        super().__init__(*args, **kwargs)
        horizons = tuple(sorted({int(value) for value in shadow_horizons}))
        if not horizons or min(horizons) <= 0:
            raise ValueError("shadow_horizons must be positive")
        self.shadow_horizons = horizons
        self.lyapunov_shadow_records: list[dict] = []
        self._shadow_service_durations = None
        self._shadow_chain_no_assign_streak: MutableMapping[str, int] = {}
        self._shadow_global_no_assign_streak = 0

    def select_robots(self, world_state, contexts):
        snapshot = compute_lyapunov_snapshot(
            world_state,
            self.lyapunov_l0_config,
        )
        if self._shadow_service_durations is None:
            self._shadow_service_durations = WorkServiceDurations.from_world(
                world_state
            )
        before = len(self.decision_trace_records)
        choices = super().select_robots(world_state, contexts)
        new_records = self.decision_trace_records[before:]
        for trace in new_records:
            context_idx = int(trace.get("context_idx", -1))
            if context_idx < 0 or context_idx >= len(contexts):
                continue
            shadow = build_shadow_record(
                snapshot=snapshot,
                trace_record=trace,
                context=contexts[context_idx],
                world=world_state,
                service_durations=self._shadow_service_durations,
                horizons=self.shadow_horizons,
                work_weight=float(self.lyapunov_l0_config.work_weight),
                lyapunov_config=self.lyapunov_l0_config,
            )
            chain_key = f"{shadow['order_id']}:{shadow['pod_id']}"
            selected_no_assign = shadow["selected_action_type"] == "no_assign"
            chain_before = int(
                self._shadow_chain_no_assign_streak.get(chain_key, 0)
            )
            global_before = int(self._shadow_global_no_assign_streak)
            if selected_no_assign:
                chain_after = chain_before + 1
                global_after = global_before + 1
                self._shadow_chain_no_assign_streak[chain_key] = chain_after
            else:
                chain_after = 0
                global_after = 0
                self._shadow_chain_no_assign_streak.pop(chain_key, None)
            self._shadow_global_no_assign_streak = global_after
            shadow.update({
                "chain_no_assign_streak_before": chain_before,
                "chain_no_assign_streak_after": chain_after,
                "global_no_assign_streak_before": global_before,
                "global_no_assign_streak_after": global_after,
            })
            self.lyapunov_shadow_records.append(shadow)
        return choices


def shadow_config_dict(assigner: LyapunovShadowWorldModelTaskAssigner) -> dict:
    return {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "horizons": list(assigner.shadow_horizons),
        "lyapunov_config": asdict(assigner.lyapunov_l0_config),
        "read_only": True,
        "active_policy_modification": False,
        "candidate_scope": "frozen_wm_best_robot_vs_global_no_assign",
    }


__all__ = [
    "NO_ASSIGN_DRIFT_SCHEMA_VERSION",
    "SHADOW_SCHEMA_VERSION",
    "LyapunovShadowWorldModelTaskAssigner",
    "build_shadow_record",
    "compute_no_assign_work_drift",
    "shadow_config_dict",
]
