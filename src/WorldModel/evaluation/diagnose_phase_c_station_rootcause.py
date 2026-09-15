"""Diagnostic-only exact replay for station handoff/queue lockups.

This module is intentionally separate from all frozen Phase-C runners.  It
reuses the deployed Dynamic-J arm and a recorded order manifest and adds the
:class:`FailureTrajectoryProbe` state capture.  By default simulator semantics
remain unchanged; ``--station-admission committed_capacity_v1`` enables only
the versioned station admission correction for a diagnostic replay.
"""

from __future__ import annotations

import argparse
import json
import logging
import types
from pathlib import Path
from typing import Any, Mapping

from Policies.TaskAssigner import WorldModelTaskAssigner
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_context_assigner import (
    PsiDispatchContextWorldModelTaskAssigner,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_dynamic_probe_assigner import (
    DynamicPsiDispatchProbeAssigner,
)
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
from WorldModel.evaluation.phase_c_failure_diagnostic_probe import (
    FailureTrajectoryProbe,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    S1_CONFIG,
    TOP_M,
    sha256_file,
)
from WorldState.station_state import (
    STATION_ADMISSION_COMMITTED_V1,
    STATION_ADMISSION_COMMITTED_FIFO_V2,
    STATION_ADMISSION_PHYSICAL_ONLY,
)


DEFAULT_BUNDLE = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "psi_dispatch_ablation_551_560_v1/phase_c_psi_dispatch_frozen_protocol.json"
)
DEFAULT_MANIFEST = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "diagnostic_station_rootcause_mid560_manifest220_v1/"
    "order_manifests/orders_mid_seed560.json"
)
DEFAULT_OUTPUT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "diagnostic_station_rootcause_mid560_trace220_v1"
)


class FullFailureTrajectoryProbe(FailureTrajectoryProbe):
    """Failure trajectory plus every robot's physical state per tick."""

    capture_through_tick: int | None = None

    @staticmethod
    def _task_summary(world, agent_id: int) -> dict[str, Any]:
        active = world.task_state.get_active_task_for_agent(agent_id)
        nxt = world.task_state.get_next_task_for_agent(agent_id)

        def one(task):
            if task is None:
                return None
            return {
                "task_id": int(task.task_id),
                "type": str(getattr(task.task_type, "name", task.task_type)),
                "station_id": (
                    int(task.station_id)
                    if task.station_id is not None else None
                ),
                "pod_id": int(task.pod_id) if task.pod_id is not None else None,
                "status": str(getattr(task.status, "name", task.status)),
                "destination": list(task.destination),
            }

        return {"active": one(active), "next": one(nxt)}

    def on_tick(self, engine):
        tick = int(engine.world.tick)
        if (
            self.capture_through_tick is not None
            and tick > int(self.capture_through_tick)
        ):
            return
        super().on_tick(engine)
        world = engine.world
        row = self._trajectory_rows[-1]
        all_agents = []
        for agent in world.agents:
            next_path = None
            remaining_path = []
            if agent.has_path and agent.path_index < len(agent.path):
                next_path = list(agent.path[agent.path_index])
                remaining_path = [
                    list(position)
                    for position in agent.path[agent.path_index:]
                ]
            all_agents.append({
                "agent_id": int(agent.agent_id),
                "position": list(agent.position),
                "previous_position": list(agent.previous_position),
                "status": str(getattr(agent.status, "name", agent.status)),
                "has_path": bool(agent.has_path),
                "path_index": int(agent.path_index),
                "path_length": int(len(agent.path)),
                "next_path": next_path,
                "remaining_path": remaining_path,
                "plan_failed_streak": int(agent.plan_failed_streak),
                "stationary_ticks": int(agent.stationary_ticks),
                "wait_ticks": int(agent.wait_ticks),
                "moved_this_tick": bool(agent.moved_this_tick),
                "stuck_this_tick": bool(agent.stuck_this_tick),
                "traffic_blocked_this_tick": bool(
                    agent.traffic_blocked_this_tick
                ),
                "carried_pod_id": (
                    int(agent.carried_pod_id)
                    if agent.carried_pod_id is not None else None
                ),
                "assigned_task_id": (
                    int(agent.assigned_task_id)
                    if agent.assigned_task_id is not None else None
                ),
                "tasks": self._task_summary(world, int(agent.agent_id)),
            })
        row["all_agents"] = all_agents


def _queue_snapshot(queue) -> dict[str, Any]:
    return {
        "occupancy": int(queue.occupancy()),
        "capacity": int(queue.capacity),
        "admission_mode": str(queue.admission_mode),
        "assigned_agent_ids": sorted(int(x) for x in queue._assigned_agents),
        "waiting_reservation_agent_ids": queue.waiting_agent_ids(),
        "waiting_reservation_count": len(queue.waiting_agent_ids()),
        "committed_agent_ids": sorted(
            int(x) for x in queue.committed_agent_ids()
        ),
        "committed_load": int(queue.committed_load()),
        "committed_capacity_invariant": bool(
            queue.committed_load() <= queue.capacity
        ),
        "slots": [
            {
                "slot_type": str(getattr(slot.slot_type, "name", slot.slot_type)),
                "index": int(slot.index),
                "position": list(slot.position),
                "agent_id": int(slot.agent_id) if slot.agent_id is not None else None,
            }
            for slot in [queue.service, *queue.queue_slots, *queue.buffer_slots]
        ],
    }


def _agent_snapshot(world, agent_id: int) -> dict[str, Any] | None:
    agent = world.get_agent(agent_id)
    if agent is None:
        return None
    active = world.task_state.get_active_task_for_agent(agent_id)
    return {
        "agent_id": int(agent_id),
        "position": list(agent.position),
        "status": str(getattr(agent.status, "name", agent.status)),
        "has_path": bool(agent.has_path),
        "path_index": int(agent.path_index),
        "path_length": int(len(agent.path)),
        "plan_failed_streak": int(agent.plan_failed_streak),
        "assigned_task_id": (
            int(agent.assigned_task_id)
            if agent.assigned_task_id is not None else None
        ),
        "active_task": (
            {
                "task_id": int(active.task_id),
                "type": str(getattr(active.task_type, "name", active.task_type)),
                "station_id": (
                    int(active.station_id)
                    if active.station_id is not None else None
                ),
                "destination": list(active.destination),
            }
            if active is not None else None
        ),
    }


def _capture_enabled(engine, capture_through_tick: int | None) -> bool:
    return (
        capture_through_tick is None
        or int(engine.world.tick) <= int(capture_through_tick)
    )


def _initial_world_snapshot(engine) -> dict[str, Any]:
    world = engine.world
    return {
        "tick": int(world.tick),
        "agents": [
            {
                "agent_id": int(agent.agent_id),
                "position": list(agent.position),
                "status": str(getattr(agent.status, "name", agent.status)),
                "carried_pod_id": (
                    int(agent.carried_pod_id)
                    if agent.carried_pod_id is not None else None
                ),
            }
            for agent in world.agents
        ],
        "station_queues": {
            str(station_id): _queue_snapshot(queue)
            for station_id, queue in sorted(
                world.station_state.stations.items()
            )
        },
    }


def _install_runtime_probe(
    engine, capture_through_tick: int | None
) -> None:
    """Attach non-mutating event capture to one freshly built engine.

    The original diagnostic watched station 2 only.  Branch diagnosis cannot
    assume that every seed locks at the same station, so this version records
    admission/check-in/release events and path failures for every station.
    """
    queues = dict(engine.world.station_state.stations)
    engine._station_rootcause_events = []
    engine._station_rootcause_plan_failures = []
    if not queues:
        return

    def event(
        station_id: int,
        queue,
        kind: str,
        agent_id: int | None = None,
        **extra,
    ):
        if not _capture_enabled(engine, capture_through_tick):
            return
        engine._station_rootcause_events.append({
            "tick": int(engine.world.tick),
            "station_id": int(station_id),
            "kind": kind,
            "agent": int(agent_id) if agent_id is not None else None,
            **extra,
        })

    for station_id, queue in sorted(queues.items()):
        original_reserve = queue.reserve
        original_unreserve = queue.unreserve
        original_check_in = queue.check_in_from_entry
        original_release = queue.release_to_exit

        def reserve(
            agent_id,
            request_tick=None,
            *,
            _station_id=station_id,
            _queue=queue,
            _original=original_reserve,
        ):
            if not _capture_enabled(engine, capture_through_tick):
                return _original(agent_id, request_tick=request_tick)
            before = _queue_snapshot(_queue)
            result = _original(agent_id, request_tick=request_tick)
            event(
                _station_id,
                _queue,
                "reserve",
                agent_id,
                result=bool(result),
                before=before,
                request_tick=request_tick,
                after=_queue_snapshot(_queue),
            )
            return result

        def unreserve(
            agent_id,
            *,
            _station_id=station_id,
            _queue=queue,
            _original=original_unreserve,
        ):
            if not _capture_enabled(engine, capture_through_tick):
                return _original(agent_id)
            before = _queue_snapshot(_queue)
            result = _original(agent_id)
            event(
                _station_id,
                _queue,
                "unreserve",
                agent_id,
                before=before,
                after=_queue_snapshot(_queue),
            )
            return result

        def check_in(
            agent_id,
            world_state,
            *,
            _station_id=station_id,
            _queue=queue,
            _original=original_check_in,
        ):
            if not _capture_enabled(engine, capture_through_tick):
                return _original(agent_id, world_state)
            before = _queue_snapshot(_queue)
            result = _original(agent_id, world_state)
            event(
                _station_id,
                _queue,
                "check_in",
                agent_id,
                result=bool(result),
                agent_state=_agent_snapshot(world_state, agent_id),
                before=before,
                after=_queue_snapshot(_queue),
            )
            return result

        def release(
            agent_id,
            world_state,
            *,
            _station_id=station_id,
            _queue=queue,
            _original=original_release,
        ):
            if not _capture_enabled(engine, capture_through_tick):
                return _original(agent_id, world_state)
            before = _queue_snapshot(_queue)
            result = _original(agent_id, world_state)
            event(
                _station_id,
                _queue,
                "release_to_exit",
                agent_id,
                result=bool(result),
                agent_state=_agent_snapshot(world_state, agent_id),
                before=before,
                after=_queue_snapshot(_queue),
            )
            return result

        # Functions attached to an instance are not descriptors, so these
        # wrappers intentionally take only the explicit call arguments.
        queue.reserve = reserve
        queue.unreserve = unreserve
        queue.check_in_from_entry = check_in
        queue.release_to_exit = release

    planner = engine.path_planner
    original_plan = planner.plan

    def plan(agent, goal, world_state, extra_blocked=None):
        result = original_plan(
            agent, goal, world_state, extra_blocked=extra_blocked
        )
        if (
            not result
            and _capture_enabled(engine, capture_through_tick)
        ):
            active = world_state.task_state.get_active_task_for_agent(agent.agent_id)
            station_id = (
                int(active.station_id)
                if active is not None and active.station_id is not None
                else None
            )
            active_queue = queues.get(station_id)
            engine._station_rootcause_plan_failures.append({
                "tick": int(world_state.tick),
                "agent": int(agent.agent_id),
                "position": list(agent.position),
                "goal": list(goal),
                "status": str(getattr(agent.status, "name", agent.status)),
                "has_path": bool(agent.has_path),
                "plan_failed_streak": int(agent.plan_failed_streak),
                "active_task": (
                    {
                        "task_id": int(active.task_id),
                        "type": str(
                            getattr(active.task_type, "name", active.task_type)
                        ),
                        "station_id": station_id,
                        "destination": list(active.destination),
                    }
                    if active is not None else None
                ),
                "station_queue": (
                    _queue_snapshot(active_queue)
                    if active_queue is not None else None
                ),
                "nearby_agents": [
                    _agent_snapshot(world_state, other.agent_id)
                    for other in world_state.agents
                    if abs(other.position[0] - agent.position[0])
                    + abs(other.position[1] - agent.position[1]) <= 2
                ],
                "vertex_res_count": len(getattr(planner, "_vertex_res", ())),
                "edge_res_count": len(getattr(planner, "_edge_res", ())),
                "extra_blocked": (
                    [list(pos) for pos in sorted(extra_blocked)]
                    if extra_blocked else []
                ),
            })
        return result

    planner.plan = types.MethodType(
        lambda _self, agent, goal, world_state, extra_blocked=None:
        plan(agent, goal, world_state, extra_blocked),
        planner,
    )

    original_build = engine.world.station_state.tick

    def station_tick(world_state):
        capture = _capture_enabled(engine, capture_through_tick)
        before = (
            {
                int(station_id): _queue_snapshot(queue)
                for station_id, queue in sorted(queues.items())
            }
            if capture else {}
        )
        result = original_build(world_state)
        if capture:
            for station_id, queue in sorted(queues.items()):
                moved = [
                    {"agent": int(agent_id), "position": list(pos)}
                    for agent_id, pos in result
                    if queue.get_slot_for_agent(agent_id) is not None
                    or agent_id in queue._assigned_agents
                ]
                after = _queue_snapshot(queue)
                if moved or before[int(station_id)] != after:
                    event(
                        int(station_id),
                        queue,
                        "cascade",
                        before=before[int(station_id)],
                        moved=moved,
                        after=after,
                    )
        return result

    engine.world.station_state.tick = station_tick


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _trace_through_tick(
    rows: list[dict[str, Any]], capture_through_tick: int
) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if int(row.get("tick", -1)) <= int(capture_through_tick)
    ]


def _reference_audit(
    metrics: Mapping[str, Any], reference_path: Path | None
) -> dict[str, Any]:
    if reference_path is None:
        return {"available": False, "passed": None, "checks": {}}
    reference = _read_json(reference_path)
    expected = reference.get("metrics") or {}
    checks: dict[str, bool] = {}
    compared: dict[str, dict[str, Any]] = {}
    exact_keys = (
        "ticks",
        "completed_orders",
        "completed_tasks",
        "open_order_count",
        "pending_order_count",
        "order_arrival_count",
        "order_arrival_manifest_sha256",
        "assign_calls",
        "model_assign_calls",
        "model_inference_calls",
        "decision_contexts_total",
        "energy_conv_contexts",
        "energy_conv_active_contexts",
        "energy_conv_modified_decisions",
        "psi_dispatch_eval_calls",
        "psi_dispatch_contexts_seen",
        "dynamic_probe_batches",
        "dynamic_probe_steps",
        "dynamic_probe_order_changed_batches",
    )
    float_keys = (
        "stall_ratio_mean",
        "stall_ratio_max",
    )
    for key in exact_keys:
        if key not in expected:
            continue
        observed = metrics.get(key)
        wanted = expected.get(key)
        checks[key] = observed == wanted
        compared[key] = {"observed": observed, "expected": wanted}
    for key in float_keys:
        if key not in expected:
            continue
        observed = metrics.get(key)
        wanted = expected.get(key)
        passed = (
            observed is not None
            and wanted is not None
            and abs(float(observed) - float(wanted)) <= 1e-6
        )
        checks[key] = passed
        compared[key] = {"observed": observed, "expected": wanted}
    # The diagnostic adds callbacks after the simulator's frozen online probe.
    # Physical deadlock summaries can therefore drift by a few post-step
    # frames while the complete/order/task outcome remains exactly identical.
    # Record those fields, but do not make them replay-fatal.
    warnings: dict[str, dict[str, Any]] = {}
    for key in ("deadlock_ratio_mean", "deadlock_ratio_max"):
        if key not in expected:
            continue
        observed = metrics.get(key)
        wanted = expected.get(key)
        warnings[key] = {
            "observed": observed,
            "expected": wanted,
            "delta": (
                float(observed) - float(wanted)
                if observed is not None and wanted is not None else None
            ),
        }
    return {
        "available": True,
        "reference_result": reference_path.as_posix(),
        "passed": all(checks.values()),
        "checks": checks,
        "compared": compared,
        "warnings": warnings,
    }


def _artifact(bundle: dict[str, Any], key: str) -> Path:
    value = (bundle.get("artifacts") or {}).get(key)
    if not isinstance(value, dict):
        raise ValueError(f"frozen bundle lacks artifact {key!r}")
    path = Path(str(value.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha256 = value.get("sha256")
    if expected_sha256 and sha256_file(path) != str(expected_sha256):
        raise ValueError(f"frozen artifact changed: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=("dynamic", "static_psi", "shadow", "s1"),
        default="dynamic",
    )
    parser.add_argument("--load", choices=("low", "mid", "high"), default="mid")
    parser.add_argument("--seed", type=int, default=560)
    parser.add_argument("--ticks", type=int, default=220)
    parser.add_argument(
        "--capture-through-tick",
        type=int,
        default=500,
        help=(
            "save dense physical and decision traces only through this tick; "
            "the simulation still runs for --ticks to reproduce final metrics"
        ),
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--reference-result",
        type=Path,
        default=None,
        help="frozen per-arm JSON used to audit exact replay equivalence",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--station-admission",
        choices=(
            STATION_ADMISSION_PHYSICAL_ONLY,
            STATION_ADMISSION_COMMITTED_V1,
            STATION_ADMISSION_COMMITTED_FIFO_V2,
        ),
        default=STATION_ADMISSION_PHYSICAL_ONLY,
        help=(
            "versioned station admission contract; the default exactly "
            "preserves historical frozen semantics"
        ),
    )
    args = parser.parse_args()

    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive")
    if args.capture_through_tick < 0:
        raise SystemExit("--capture-through-tick must be non-negative")
    capture_through_tick = min(
        int(args.capture_through_tick), int(args.ticks)
    )
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite diagnostic output: {args.output_root}"
        )
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.reference_result is not None and not args.reference_result.is_file():
        raise FileNotFoundError(args.reference_result)

    bundle = _read_json(args.bundle)
    model_checkpoint = _artifact(bundle, "model_checkpoint")
    psi_head_checkpoint = _artifact(bundle, "psi_head_checkpoint")
    psi_scale_contract = _artifact(bundle, "psi_scale_contract")
    config_path = _artifact(bundle, f"config_{args.load}")

    common = {
        "checkpoint_path": str(model_checkpoint),
        "top_m": TOP_M,
        "energy_conv_random_flip_seed": 0,
        **S1_CONFIG,
        "decision_trace_enabled": True,
        "decision_trace_max_records": max(2000, int(args.ticks) * 20),
    }
    if args.arm == "dynamic":
        assigner = DynamicPsiDispatchProbeAssigner(
            psi_head_checkpoint=str(psi_head_checkpoint),
            psi_scale_contract=str(psi_scale_contract),
            dynamic_trace_enabled=True,
            dynamic_trace_max_records=max(300, int(args.ticks)),
            **common,
        )
    elif args.arm in ("static_psi", "shadow"):
        assigner = PsiDispatchContextWorldModelTaskAssigner(
            psi_head_checkpoint=str(psi_head_checkpoint),
            psi_scale_contract=str(psi_scale_contract),
            psi_context_mode=(
                "j_ascending" if args.arm == "static_psi" else "shadow"
            ),
            psi_trace_enabled=True,
            psi_trace_max_records=max(300, int(args.ticks)),
            **common,
        )
    else:
        assigner = WorldModelTaskAssigner(**common)

    # The frozen TD runner imports the class from this module at call time.
    # Swap only for this process and restore it immediately afterwards.  The
    # engine builder is patched in the same isolated process so the runtime
    # event hooks never affect a normal runner.
    import WorldModel.evaluation.td_stream_probe as td_module
    import WorldModel.evaluation.evaluate_online_v6 as eval_module

    original_probe = td_module.TDStreamProbe
    original_builder = eval_module._build_engine
    FullFailureTrajectoryProbe.capture_through_tick = capture_through_tick
    td_module.TDStreamProbe = FullFailureTrajectoryProbe
    engine_holder: dict[str, Any] = {}

    def build_and_probe(cfg, task_assigner=None):
        engine = original_builder(cfg, task_assigner=task_assigner)
        engine.world.station_state.set_admission_mode(args.station_admission)
        engine_holder["engine"] = engine
        engine_holder["initial_world"] = _initial_world_snapshot(engine)
        _install_runtime_probe(engine, capture_through_tick)
        return engine

    eval_module._build_engine = build_and_probe
    args.output_root.mkdir(parents=True, exist_ok=False)
    run_id = (
        f"station_rootcause_{args.arm}_{args.load}_"
        f"seed{args.seed}_t{args.ticks}_{args.station_admission}"
    )
    diagnostic_schema_version = "phasec_multistation_rootcause_v4"
    try:
        # The diagnostic is deliberately quiet; state is captured in JSONL,
        # not inferred from a warning log.
        logging.getLogger("MAS_RMFS.Engine").setLevel(logging.ERROR)
        metrics = _run_one_assigner(
            str(config_path),
            assigner,
            int(args.seed),
            int(args.ticks),
            trace_label="PhaseCStationRootCauseDiagnostic",
            td_stream_dir=str(args.output_root),
            # The human-readable trajectory is the primary artifact.  Keep
            # the auxiliary TD stream sparse so 15 exact replays remain small.
            td_stream_frame_stride=max(capture_through_tick + 1, 1),
            td_stream_run_id=run_id,
            td_stream_meta={
                "diagnostic_schema_version": diagnostic_schema_version,
                "arm": args.arm,
                "load": args.load,
                "seed": int(args.seed),
                "ticks": int(args.ticks),
                "capture_through_tick": capture_through_tick,
                "manifest": args.manifest.as_posix(),
                "station_admission": args.station_admission,
            },
            recorded_orders_path=str(args.manifest),
        )
    finally:
        td_module.TDStreamProbe = original_probe
        eval_module._build_engine = original_builder

    if hasattr(assigner, "dynamic_probe_metrics"):
        metrics.update(assigner.dynamic_probe_metrics())
    elif hasattr(assigner, "psi_dispatch_metrics"):
        metrics.update(assigner.psi_dispatch_metrics())

    engine = engine_holder.get("engine")
    if engine is None:
        raise RuntimeError("diagnostic engine was not captured")

    station_events = list(
        getattr(engine, "_station_rootcause_events", [])
    )
    plan_failures = list(
        getattr(engine, "_station_rootcause_plan_failures", [])
    )
    robot_decisions = _trace_through_tick(
        list(getattr(assigner, "decision_trace_records", []) or []),
        capture_through_tick,
    )
    context_trace: list[dict[str, Any]] = []
    context_trace_kind = None
    if hasattr(assigner, "dynamic_probe_trace_records"):
        context_trace_kind = "dynamic_j"
        context_trace = _trace_through_tick(
            list(assigner.dynamic_probe_trace_records),
            capture_through_tick,
        )
    elif hasattr(assigner, "psi_dispatch_trace_records"):
        context_trace_kind = (
            "static_j" if args.arm == "static_psi" else "shadow_j"
        )
        context_trace = _trace_through_tick(
            list(assigner.psi_dispatch_trace_records),
            capture_through_tick,
        )

    station_events_path = args.output_root / f"station_events_{run_id}.jsonl"
    plan_failures_path = args.output_root / f"path_failures_{run_id}.jsonl"
    robot_trace_path = args.output_root / f"robot_decisions_{run_id}.jsonl"
    context_trace_path = (
        args.output_root / f"context_order_{run_id}.jsonl"
        if context_trace_kind is not None else None
    )
    _write_jsonl(station_events_path, station_events)
    _write_jsonl(plan_failures_path, plan_failures)
    _write_jsonl(robot_trace_path, robot_decisions)
    if context_trace_path is not None:
        _write_jsonl(context_trace_path, context_trace)

    reference_audit = _reference_audit(metrics, args.reference_result)

    payload = {
        "schema_version": diagnostic_schema_version,
        "run_id": run_id,
        "arm": args.arm,
        "load": args.load,
        "seed": int(args.seed),
        "ticks": int(args.ticks),
        "capture_through_tick": capture_through_tick,
        "station_admission": args.station_admission,
        "bundle": args.bundle.as_posix(),
        "manifest": args.manifest.as_posix(),
        "reference_audit": reference_audit,
        "initial_world": engine_holder.get("initial_world"),
        "metrics": metrics,
        "station_admission_metrics": (
            engine.world.station_state.admission_metrics()
        ),
        "station_events": station_events_path.as_posix(),
        "station_event_count": len(station_events),
        "path_failures": plan_failures_path.as_posix(),
        "path_failure_count": len(plan_failures),
        "robot_decisions": robot_trace_path.as_posix(),
        "robot_decision_count": len(robot_decisions),
        "context_order_trace_kind": context_trace_kind,
        "context_order_trace": (
            context_trace_path.as_posix()
            if context_trace_path is not None else None
        ),
        "context_order_trace_count": len(context_trace),
        "trajectory": (
            args.output_root / f"trajectory_{run_id}.jsonl"
        ).as_posix(),
        "td_stream": (
            args.output_root / f"tdstream_{run_id}.pt"
        ).as_posix(),
    }
    (args.output_root / "diagnostic_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_root": args.output_root.as_posix(),
        "trajectory": payload["trajectory"],
        "completed_orders": metrics.get("completed_orders"),
        "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
        "reference_audit_passed": reference_audit.get("passed"),
        "station_event_count": len(station_events),
        "path_failure_count": len(plan_failures),
    }, ensure_ascii=False, indent=2))
    if reference_audit.get("available") and not reference_audit.get("passed"):
        failed = [
            key for key, passed in reference_audit["checks"].items()
            if not passed
        ]
        raise RuntimeError(
            "diagnostic replay differs from frozen result: "
            + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
