"""Run the isolated dispatch-preserving ETA station-queue experiment.

The control arm is the existing ETA V3 admission policy.  The treatment arm
uses the same ETA limits but gives post-PICK DELIVER tasks an explicit waiting
lifecycle ordered by the task chains returned by each scheduler.  No model,
S1/J1 scorer, station service, release, or path-planning rule is changed.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    LOADS,
    canonical_sha256,
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_psi_dynamic_admission import (
    _atomic_json,
    _read_json,
)
from WorldModel.evaluation.run_phase_c_sj_admission_factorial import (
    ArmSpec,
    _factor_metrics,
    _load_run_inputs,
    _make_assigner,
    _manifest_contract,
    _max_mapping,
    _policy_audit,
    _policy_contract,
    _validate_manifest_only,
)
from WorldModel.evaluation.run_phase_c_station_admission_restoration import (
    DEFAULT_BUNDLE,
    DEFAULT_V1_REFERENCE_ROOT,
    DEFAULT_V3_REFERENCE_ROOT,
    FORMAL_SEEDS,
    FORMAL_TICKS,
    StationAdmissionRestorationAuditProbe,
    _dynamic_config,
)
from WorldState.agent_state import AgentStatus
from WorldState.station_state import (
    STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1,
    STATION_ADMISSION_DYNAMIC_ETA_V1,
)
from WorldState.task_state import TaskStatus, TaskType


SCHEMA_VERSION = "phase_c_dispatch_preserving_eta_queue_arm_v1"
CONTRACT_VERSION = "phase_c_dispatch_preserving_eta_queue_contract_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_SOURCE_ROOT = (
    BASE_ROOT / "station_admission_capacity_frontier_source_high_551_560_v1"
)
DEFAULT_OUTPUT_ROOT = (
    BASE_ROOT / "station_admission_dispatch_preserving_frontier_high_551_560_v1"
)

ETA_CONTROL_KEY = "eta_v3_control"
DISPATCH_QUEUE_KEY = "dispatch_eta_queue_v1"
ADMISSION_MODES = {
    ETA_CONTROL_KEY: STATION_ADMISSION_DYNAMIC_ETA_V1,
    DISPATCH_QUEUE_KEY: STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1,
}


def _build_arm_specs() -> dict[str, ArmSpec]:
    policies = (
        ("s1_j1", "world_model", "s1", "j1", "S1 + static J1"),
        ("s0_j0", "world_model", "s0", "j0", "Phase C (S0 + J0)"),
        ("greedy", "greedy", None, None, "Greedy"),
    )
    labels = {
        ETA_CONTROL_KEY: "ETA V3 control",
        DISPATCH_QUEUE_KEY: "dispatch-preserving ETA queue V1",
    }
    result = {}
    for policy_key, family, selector, scheduler, policy_label in policies:
        for admission_key, mode in ADMISSION_MODES.items():
            spec = ArmSpec(
                key=f"{policy_key}_{admission_key}",
                label=f"{policy_label} + {labels[admission_key]}",
                policy_key=policy_key,
                policy_family=family,
                robot_selector=selector,
                context_scheduler=scheduler,
                admission_key=admission_key,
                admission_mode=mode,
            )
            result[spec.key] = spec
    return result


ARM_SPECS = _build_arm_specs()
ARM_KEYS = tuple(ARM_SPECS)

EXPERIMENT_CONTRACT = {
    "schema_version": CONTRACT_VERSION,
    "policies": ["s1_j1", "s0_j0", "greedy"],
    "admissions": {
        ETA_CONTROL_KEY: {
            "mode": STATION_ADMISSION_DYNAMIC_ETA_V1,
            "waiting_lifecycle": False,
        },
        DISPATCH_QUEUE_KEY: {
            "mode": STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1,
            "waiting_lifecycle": True,
            "priority": "engine-stamped task-chain dispatch sequence",
            "fallback": "stable sequence after normal engine priorities",
            "bypass": (
                "later ready waiter may proceed only when earlier waiters "
                "remain infeasible under unchanged ETA admission checks"
            ),
        },
    },
    "isolation": {
        "world_model_checkpoint_changed": False,
        "s1_or_j1_changed": False,
        "path_planner_changed": False,
        "station_service_or_release_changed": False,
        "eta_capacity_equations_changed": False,
    },
    "arm_keys": list(ARM_KEYS),
}
EXPERIMENT_CONTRACT = {
    **EXPERIMENT_CONTRACT,
    "contract_sha256": canonical_sha256(EXPERIMENT_CONTRACT),
}


def _output_path(root: Path, arm: str, load: str, seed: int) -> Path:
    return root / "per_arm" / arm / f"{load}_seed{seed}.json"


def _runtime_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "task_state": Path("WorldState/task_state.py"),
        "station_state": Path("WorldState/station_state.py"),
        "simulation_engine": Path("Engine/simulation_engine.py"),
        "counterfactual_rollout": Path(
            "WorldModel/data/counterfactual_rollout.py"
        ),
        "world_model_assigner": Path(
            "Policies/TaskAssigner/WorldModelTaskAssigner/"
            "world_model_task_assigner.py"
        ),
        "static_j_assigner": Path(
            "Policies/TaskAssigner/WorldModelTaskAssigner/"
            "psi_dispatch_context_assigner.py"
        ),
        "dynamic_j_assigner": Path(
            "Policies/TaskAssigner/WorldModelTaskAssigner/"
            "psi_dispatch_dynamic_probe_assigner.py"
        ),
        "greedy_assigner": Path(
            "Policies/TaskAssigner/GreedyTaskAssigner/"
            "greedy_task_assigner.py"
        ),
        "hungarian_assigner": Path(
            "Policies/TaskAssigner/HungarianTaskAssigner/"
            "hungarian_task_assigner.py"
        ),
        "evaluate_online": Path(
            "WorldModel/evaluation/evaluate_online_v6.py"
        ),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    rows = sorted(float(value) for value in values)
    if len(rows) == 1:
        return rows[0]
    position = max(0.0, min(1.0, float(quantile))) * (len(rows) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return rows[lower]
    weight = position - lower
    return rows[lower] * (1.0 - weight) + rows[upper] * weight


class DispatchWaitingAuditProbe:
    """Audit only the new waiting lifecycle; ETA invariants use the main probe."""

    def __init__(self, engine, trace_max_records: int = 0):
        self.engine = engine
        self.trace_max_records = max(0, int(trace_max_records))
        self.trace: list[dict[str, Any]] = []
        self.trace_dropped = 0
        self.observed_ticks = 0
        self.waiting_agent_ticks = 0
        self.waiting_age_samples: list[float] = []
        self.waiting_duration_samples: list[float] = []
        self.waiting_promotions = 0
        self.waiting_cancellations = 0
        self.observed_bypass_promotions = 0
        self.max_waiting_depth: dict[int, int] = {}
        self.violations: list[dict[str, Any]] = []
        self.violation_count = 0
        self._previous_waiting: dict[int, dict[str, int]] = {}

    def _violation(self, payload: Mapping[str, Any]) -> None:
        self.violation_count += 1
        if len(self.violations) < 100:
            self.violations.append(dict(payload))

    def on_tick(self, engine) -> None:
        world = engine.world
        tick = int(world.tick)
        self.observed_ticks += 1
        current: dict[int, dict[str, int]] = {}
        trace_stations = []

        for station_id, queue in sorted(world.station_state.stations.items()):
            station_id = int(station_id)
            waiting_ids = queue.waiting_agent_ids()
            sequences = [
                queue.waiting_reservation_sequence(agent_id)
                for agent_id in waiting_ids
            ]
            self.max_waiting_depth[station_id] = max(
                len(waiting_ids), self.max_waiting_depth.get(station_id, 0)
            )
            if queue.admission_mode != STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1:
                self._violation({
                    "tick": tick,
                    "station_id": station_id,
                    "reason": "wrong_admission_mode",
                    "actual": queue.admission_mode,
                })
            if any(value is None for value in sequences):
                self._violation({
                    "tick": tick,
                    "station_id": station_id,
                    "reason": "missing_queue_sequence",
                })
            elif [int(value) for value in sequences] != sorted(
                int(value) for value in sequences
            ):
                self._violation({
                    "tick": tick,
                    "station_id": station_id,
                    "reason": "queue_not_dispatch_ordered",
                    "sequences": sequences,
                })
            if queue.committed_load() > queue.dynamic_hard_limit():
                self._violation({
                    "tick": tick,
                    "station_id": station_id,
                    "reason": "dynamic_hard_limit_exceeded",
                    "committed_load": queue.committed_load(),
                    "hard_limit": queue.dynamic_hard_limit(),
                })

            for agent_id, sequence in zip(waiting_ids, sequences):
                agent = world.get_agent(agent_id)
                task = world.task_state.get_next_task_for_agent(agent_id)
                reasons = []
                if agent is None:
                    reasons.append("missing_agent")
                else:
                    if agent.status != AgentStatus.WAITING_ASSIGNED:
                        reasons.append("agent_not_waiting_assigned")
                    if agent.has_path:
                        reasons.append("waiting_agent_has_path")
                    if agent.carried_pod_id is None:
                        reasons.append("waiting_agent_has_no_pod")
                    if agent.station_waiting_station_id != station_id:
                        reasons.append("agent_station_mismatch")
                    if agent.station_waiting_sequence != sequence:
                        reasons.append("agent_sequence_mismatch")
                if task is None or task.task_type != TaskType.DELIVER:
                    reasons.append("next_task_not_deliver")
                else:
                    if task.status != TaskStatus.ASSIGNED:
                        reasons.append("deliver_not_assigned")
                    if task.station_id != station_id:
                        reasons.append("deliver_station_mismatch")
                    dispatch_sequence = getattr(
                        task, "station_dispatch_sequence", None
                    )
                    if dispatch_sequence is None:
                        reasons.append("missing_task_dispatch_sequence")
                    elif int(dispatch_sequence) != int(sequence):
                        reasons.append("task_dispatch_sequence_mismatch")
                if world.task_state.get_active_task_for_agent(agent_id) is not None:
                    reasons.append("active_task_present")
                if int(agent_id) in queue.committed_agent_ids():
                    reasons.append("waiting_agent_has_committed_token")
                if reasons:
                    self._violation({
                        "tick": tick,
                        "station_id": station_id,
                        "agent_id": int(agent_id),
                        "reasons": reasons,
                    })
                since = (
                    int(agent.station_waiting_since_tick)
                    if agent is not None
                    and agent.station_waiting_since_tick is not None
                    else tick
                )
                self.waiting_agent_ticks += 1
                self.waiting_age_samples.append(max(0.0, tick - since))
                current[int(agent_id)] = {
                    "station_id": station_id,
                    "sequence": int(sequence) if sequence is not None else 10**15,
                    "since_tick": since,
                }

            if self.trace_max_records > 0:
                trace_stations.append({
                    "station_id": station_id,
                    "waiting_agent_ids": waiting_ids,
                    "waiting_sequences": sequences,
                    "committed_load": int(queue.committed_load()),
                    "hard_limit": int(queue.dynamic_hard_limit()),
                })

        for agent_id, previous in self._previous_waiting.items():
            if agent_id in current:
                continue
            agent = world.get_agent(agent_id)
            active = world.task_state.get_active_task_for_agent(agent_id)
            queue = world.station_state.get_queue(previous["station_id"])
            promoted = bool(
                agent is not None
                and agent.status == AgentStatus.CARRYING
                and active is not None
                and active.task_type == TaskType.DELIVER
                and active.status == TaskStatus.IN_PROGRESS
                and queue is not None
                and agent_id in queue.committed_agent_ids()
            )
            if promoted:
                self.waiting_promotions += 1
                self.waiting_duration_samples.append(
                    max(0.0, tick - previous["since_tick"])
                )
                older_remains = any(
                    row["station_id"] == previous["station_id"]
                    and row["sequence"] < previous["sequence"]
                    for row in current.values()
                )
                self.observed_bypass_promotions += int(older_remains)
            else:
                self.waiting_cancellations += 1

        if trace_stations:
            row = {"tick": tick, "stations": trace_stations}
            if len(self.trace) < self.trace_max_records:
                self.trace.append(row)
            else:
                self.trace_dropped += 1
        self._previous_waiting = current

    def summary(self) -> dict[str, Any]:
        station_metrics = self.engine.world.station_state.admission_metrics()
        final_tick = int(self.engine.world.tick)
        fallback_count = sum(
            int(row.get("waiting_dispatch_priority_fallback", 0))
            for row in station_metrics
        )
        queue_bypass_count = sum(
            int(row.get("waiting_dispatch_priority_bypass_granted", 0))
            for row in station_metrics
        )
        return {
            "mode": STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1,
            "passed": bool(
                self.violation_count == 0
                and all(bool(row.get("invariant_holds")) for row in station_metrics)
                and all(
                    row.get("mode")
                    == STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1
                    for row in station_metrics
                )
            ),
            "violation_count": int(self.violation_count),
            "violations": self.violations,
            "observed_ticks": int(self.observed_ticks),
            "waiting_agent_ticks": int(self.waiting_agent_ticks),
            "waiting_assigned_ratio": float(
                self.waiting_agent_ticks
                / max(self.observed_ticks * len(self.engine.world.agents), 1)
            ),
            "waiting_age_p50_ticks": _percentile(
                self.waiting_age_samples, 0.50
            ),
            "waiting_age_p95_ticks": _percentile(
                self.waiting_age_samples, 0.95
            ),
            "waiting_duration_p50_ticks": _percentile(
                self.waiting_duration_samples, 0.50
            ),
            "waiting_duration_p95_ticks": _percentile(
                self.waiting_duration_samples, 0.95
            ),
            "waiting_promotions": int(self.waiting_promotions),
            "waiting_cancellations_observed": int(self.waiting_cancellations),
            "observed_bypass_promotions": int(
                self.observed_bypass_promotions
            ),
            "queue_bypass_grants": int(queue_bypass_count),
            "dispatch_priority_fallback_count": int(fallback_count),
            "unresolved_waiter_count_final": int(len(self._previous_waiting)),
            "oldest_waiter_age_final_ticks": max(
                (
                    max(0, final_tick - row["since_tick"])
                    for row in self._previous_waiting.values()
                ),
                default=None,
            ),
            "max_waiting_depth": {
                str(key): int(value)
                for key, value in sorted(self.max_waiting_depth.items())
            },
            "trace_record_count": len(self.trace),
            "trace_dropped": int(self.trace_dropped),
            "trace": self.trace,
            "station_metrics_final": station_metrics,
        }


@contextmanager
def _engine_contract(
    admission_mode: str,
    *,
    dynamic_config: Mapping[str, Any],
    station_trace_stride: int,
    station_trace_max_records: int,
    queue_trace_max_records: int,
):
    import WorldModel.evaluation.evaluate_online_v6 as eval_module

    if admission_mode not in ADMISSION_MODES.values():
        raise ValueError(f"unsupported experiment admission mode: {admission_mode}")
    original_builder = eval_module._build_engine
    holder: dict[str, Any] = {}

    def build(cfg, task_assigner=None):
        cfg.simulation.log_level = "ERROR"
        engine = original_builder(cfg, task_assigner=task_assigner)
        engine.world.station_state.configure_dynamic_admission(
            service_ticks=int(cfg.simulation.station_process_duration),
            **dict(dynamic_config),
        )
        engine.world.station_state.set_admission_mode(admission_mode)
        station_probe = StationAdmissionRestorationAuditProbe(
            engine,
            admission_mode,
            trace_stride=station_trace_stride,
            trace_max_records=station_trace_max_records,
        )
        engine.on_tick_callbacks.append(station_probe.on_tick)
        holder["engine"] = engine
        holder["station_probe"] = station_probe
        if admission_mode == STATION_ADMISSION_DYNAMIC_ETA_DISPATCH_QUEUE_V1:
            queue_probe = DispatchWaitingAuditProbe(
                engine, trace_max_records=queue_trace_max_records
            )
            engine.on_tick_callbacks.append(queue_probe.on_tick)
            holder["queue_probe"] = queue_probe
        return engine

    eval_module._build_engine = build
    try:
        yield holder
    finally:
        eval_module._build_engine = original_builder


def _run_audit(
    spec: ArmSpec,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    station_audit: Mapping[str, Any],
    queue_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    checks = {
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "manifest_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_count_matches": int(
            metrics.get("order_arrival_count", -1)
        ) == int(manifest.get("total_orders", -2)),
        "station_admission_audit": bool(station_audit.get("passed")),
        "station_admission_mode": (
            station_audit.get("mode") == spec.admission_mode
        ),
        "physical_capacity_never_exceeded": int(
            station_audit.get("physical_capacity_violation_count", -1)
        ) == 0,
        "dynamic_hard_limit_never_exceeded": int(
            station_audit.get("dynamic_hard_limit_violation_count", -1)
        ) == 0,
        **_policy_audit(spec, metrics),
    }
    if spec.admission_key == DISPATCH_QUEUE_KEY:
        checks.update({
            "dispatch_waiting_probe_attached": queue_audit is not None,
            "dispatch_waiting_lifecycle": bool(
                queue_audit and queue_audit.get("passed")
            ),
            "dispatch_priority_metadata_complete": int(
                (queue_audit or {}).get(
                    "dispatch_priority_fallback_count", -1
                )
            ) == 0,
        })
    else:
        checks["dispatch_waiting_probe_not_attached"] = queue_audit is None
        checks["control_has_no_waiting_reservations"] = all(
            int(row.get("waiting_reservation_count", -1)) == 0
            for row in station_audit.get("station_metrics_final", [])
        )
    return {"passed": all(checks.values()), "checks": checks}


def _resume_compatible(
    path: Path,
    *,
    spec: ArmSpec,
    args: argparse.Namespace,
    bundle_sha: str,
    runtime_hashes: Mapping[str, str],
    policy_fingerprint: str,
    manifest_sha: str,
    dynamic_config: Mapping[str, Any],
) -> bool:
    if not path.is_file():
        return False
    payload = _read_json(path)
    meta = payload.get("meta") or {}
    checks = {
        "schema": payload.get("schema_version") == SCHEMA_VERSION,
        "contract": (
            meta.get("experiment_contract_sha256")
            == EXPERIMENT_CONTRACT["contract_sha256"]
        ),
        "arm": meta.get("arm_key") == spec.key,
        "load": meta.get("load") == args.load,
        "seed": int(meta.get("seed", -1)) == int(args.seed),
        "ticks": int(meta.get("ticks", -1)) == int(args.ticks),
        "admission": meta.get("station_admission") == spec.admission_mode,
        "bundle": meta.get("frozen_bundle_sha256") == bundle_sha,
        "runtime": meta.get("runtime_code_sha256") == dict(runtime_hashes),
        "policy": (
            (meta.get("policy_contract") or {}).get("fingerprint_sha256")
            == policy_fingerprint
        ),
        "manifest": (
            (payload.get("manifest") or {}).get("content_sha256")
            == manifest_sha
        ),
        "dynamic_config": (
            meta.get("dynamic_admission_config") == dict(dynamic_config)
        ),
        "audit": bool((payload.get("audit") or {}).get("passed")),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"cannot resume incompatible output {path}: {failed}")
    print(f"[resume] {spec.key} {args.load} seed={args.seed}: {path}")
    return True


def _run_arm(args: argparse.Namespace) -> None:
    if args.arm is None:
        raise SystemExit("--arm is required for --mode arm")
    spec = ARM_SPECS[args.arm]
    (
        _bundle,
        source_protocol,
        model_checkpoint,
        psi_head_checkpoint,
        psi_scale_contract,
        config_path,
    ) = _load_run_inputs(args)
    manifest_contract = _manifest_contract(args)
    manifest_path = manifest_contract["path"]
    manifest = manifest_contract["payload"]
    runtime_hashes = _runtime_hashes()
    policy_contract = _policy_contract(
        spec,
        model_checkpoint=model_checkpoint,
        psi_head_checkpoint=psi_head_checkpoint,
        psi_scale_contract=psi_scale_contract,
        runtime_hashes=runtime_hashes,
    )
    dynamic_config = _dynamic_config(args)
    bundle_sha = sha256_file(args.frozen_bundle)
    output = _output_path(
        args.output_root, spec.key, args.load, int(args.seed)
    )
    if _resume_compatible(
        output,
        spec=spec,
        args=args,
        bundle_sha=bundle_sha,
        runtime_hashes=runtime_hashes,
        policy_fingerprint=policy_contract["fingerprint_sha256"],
        manifest_sha=str(manifest["manifest_sha256"]),
        dynamic_config=dynamic_config,
    ):
        return

    assigner = _make_assigner(
        spec,
        model_checkpoint=model_checkpoint,
        psi_head_checkpoint=psi_head_checkpoint,
        psi_scale_contract=psi_scale_contract,
        trace_max_records=args.policy_trace_max_records,
    )
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    print(
        f"[run] arm={spec.key} load={args.load} seed={args.seed} "
        f"ticks={args.ticks} admission={spec.admission_mode}"
    )
    with _engine_contract(
        spec.admission_mode,
        dynamic_config=dynamic_config,
        station_trace_stride=args.station_trace_stride,
        station_trace_max_records=args.station_trace_max_records,
        queue_trace_max_records=args.queue_trace_max_records,
    ) as holder:
        metrics = _run_one_assigner(
            str(config_path),
            assigner,
            int(args.seed),
            int(args.ticks),
            trace_label=spec.label,
            recorded_orders_path=str(manifest_path),
        )
    station_probe = holder.get("station_probe")
    if not isinstance(station_probe, StationAdmissionRestorationAuditProbe):
        raise RuntimeError("station admission audit probe was not attached")
    station_audit = station_probe.summary()
    queue_probe = holder.get("queue_probe")
    queue_audit = queue_probe.summary() if queue_probe is not None else None

    metrics.update(_factor_metrics(spec, assigner))
    metrics.update({
        "station_admission_mode": spec.admission_mode,
        "station_capacity_rejections": station_audit.get(
            "capacity_rejections"
        ),
        "station_over_capacity_grants": station_audit.get(
            "over_capacity_grants"
        ),
        "station_committed_over_capacity_tick_count": station_audit.get(
            "committed_over_capacity_station_tick_count"
        ),
        "station_ticks_with_any_committed_over_capacity": station_audit.get(
            "ticks_with_any_committed_over_capacity"
        ),
    })
    if queue_audit is not None:
        metrics.update({
            "waiting_assigned_agent_ticks": queue_audit.get(
                "waiting_agent_ticks"
            ),
            "waiting_assigned_ratio": queue_audit.get(
                "waiting_assigned_ratio"
            ),
            "waiting_promotions": queue_audit.get("waiting_promotions"),
            "waiting_duration_p95_ticks": queue_audit.get(
                "waiting_duration_p95_ticks"
            ),
            "dispatch_queue_bypass_grants": queue_audit.get(
                "queue_bypass_grants"
            ),
            "dispatch_queue_fallback_count": queue_audit.get(
                "dispatch_priority_fallback_count"
            ),
            "unresolved_waiter_count_final": queue_audit.get(
                "unresolved_waiter_count_final"
            ),
        })

    audit = _run_audit(
        spec, metrics, manifest, station_audit, queue_audit
    )
    if not audit["passed"]:
        failed = [
            name for name, passed in audit["checks"].items() if not passed
        ]
        raise RuntimeError(
            f"dispatch queue arm audit failed {spec.key} {args.load} "
            f"seed={args.seed}: {failed}"
        )

    if spec.context_scheduler == "j1":
        policy_trace = assigner.psi_dispatch_trace_records
    else:
        policy_trace = []
    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "experiment_contract_sha256": EXPERIMENT_CONTRACT[
                "contract_sha256"
            ],
            "source_protocol_sha256": source_protocol.get(
                "protocol_sha256"
            ),
            "frozen_bundle": args.frozen_bundle.as_posix(),
            "frozen_bundle_sha256": bundle_sha,
            "arm_key": spec.key,
            "arm_label": spec.label,
            "policy_key": spec.policy_key,
            "policy_family": spec.policy_family,
            "robot_selector": spec.robot_selector,
            "context_scheduler": spec.context_scheduler,
            "policy_contract": policy_contract,
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "formal": not bool(args.development),
            "admission_key": spec.admission_key,
            "station_admission": spec.admission_mode,
            "dynamic_admission_config": dynamic_config,
            "source_manifest_root": args.source_root.as_posix(),
            "runtime_code_sha256": runtime_hashes,
            "engine_log_level": "ERROR",
        },
        "experiment_contract": EXPERIMENT_CONTRACT,
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "historical_manifest_contract": manifest_contract["historical"],
        "audit": audit,
        "station_admission_audit": station_audit,
        "dispatch_waiting_audit": queue_audit,
        "metrics": metrics,
        "policy_trace": policy_trace,
    }
    _atomic_json(output, payload)
    print(f"[done] {output}")
    print(json.dumps({
        "completed_orders": metrics.get("completed_orders"),
        "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
        "stall_ratio_mean": metrics.get("stall_ratio_mean"),
        "max_committed_load": _max_mapping(
            station_audit.get("max_committed_load")
        ),
        "max_occupancy": _max_mapping(station_audit.get("max_occupancy")),
        "waiting_assigned_ratio": metrics.get("waiting_assigned_ratio"),
        "audit_passed": audit["passed"],
    }, indent=2, ensure_ascii=False))


def _validate_args(args: argparse.Namespace) -> None:
    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive")
    if args.station_trace_stride <= 0:
        raise SystemExit("--station-trace-stride must be positive")
    if any(value < 0 for value in (
        args.policy_trace_max_records,
        args.station_trace_max_records,
        args.queue_trace_max_records,
    )):
        raise SystemExit("trace record limits must be non-negative")
    if args.mode in ("manifest", "arm") and args.seed is None:
        raise SystemExit("--seed is required")
    if not args.development:
        if args.ticks != FORMAL_TICKS:
            raise SystemExit(f"formal runs freeze --ticks={FORMAL_TICKS}")
        if args.seed not in FORMAL_SEEDS:
            raise SystemExit(
                f"formal seed must be in {list(FORMAL_SEEDS)}: {args.seed}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("manifest", "arm"), required=True)
    parser.add_argument("--arm", choices=ARM_KEYS, default=None)
    parser.add_argument("--load", choices=LOADS, default="high")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ticks", type=int, default=FORMAL_TICKS)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frozen-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--manifest-reference-root",
        type=Path,
        default=DEFAULT_V1_REFERENCE_ROOT,
    )
    parser.add_argument(
        "--v3-reference-root", type=Path, default=DEFAULT_V3_REFERENCE_ROOT
    )
    parser.add_argument("--require-v3-reference", action="store_true")
    parser.add_argument("--skip-historical-contract", action="store_true")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--policy-trace-max-records", type=int, default=0)
    parser.add_argument("--station-trace-stride", type=int, default=10)
    parser.add_argument("--station-trace-max-records", type=int, default=0)
    parser.add_argument("--queue-trace-max-records", type=int, default=0)
    parser.add_argument("--max-committed-multiplier", type=float, default=2.0)
    parser.add_argument("--healthy-extra-ratio", type=float, default=1.0)
    parser.add_argument("--caution-extra-ratio", type=float, default=0.5)
    parser.add_argument("--brake-extra-ratio", type=float, default=0.0)
    parser.add_argument("--eta-near-ticks", type=int, default=8)
    parser.add_argument("--eta-mid-ticks", type=int, default=20)
    args = parser.parse_args()
    _validate_args(args)
    if args.mode == "manifest":
        _validate_manifest_only(args)
    else:
        _run_arm(args)


if __name__ == "__main__":
    main()


__all__ = [
    "ARM_KEYS",
    "ARM_SPECS",
    "DISPATCH_QUEUE_KEY",
    "ETA_CONTROL_KEY",
    "EXPERIMENT_CONTRACT",
]
