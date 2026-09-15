"""Run the versioned Dynamic-J arm with FIFO station admission waiting.

This runner is intentionally separate from the committed-capacity V1 runner.
The policy, model checkpoint, order manifests, path planner, and station exit
logic are unchanged.  The opt-in simulator contract adds an explicit
``WAITING_ASSIGNED`` state for a carried pod whose DELIVER task is waiting for
station capacity, and promotes requests in FIFO order when a token becomes
available.

The output root and arm key are new, so this experiment cannot overwrite any
frozen Phase-C or earlier V1 artifact.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_dynamic_probe_assigner import (
    DynamicPsiDispatchProbeAssigner,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    LOADS,
    S1_CONFIG,
    TOP_M,
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_psi_dynamic_probe import (
    _artifact,
    _atomic_json,
    _load_bundle,
    _manifest_path,
    _read_json,
)
from WorldState.agent_state import AgentStatus
from WorldState.station_state import STATION_ADMISSION_COMMITTED_FIFO_V2
from WorldState.task_state import TaskStatus, TaskType


SCHEMA_VERSION = "phase_c_psi_dynamic_fifo_wait_arm_v2"
ARM_KEY = "s1_psi_dynamic_committed_fifo_wait_v2"
ARM_LABEL = "PhaseCS1PsiDynamicCommittedFifoWaitV2"
BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_SOURCE_ROOT = BASE_ROOT / "psi_dispatch_ablation_551_560_v1"
DEFAULT_LEGACY_DYNAMIC_ROOT = BASE_ROOT / "psi_dispatch_dynamic_probe_551_560_v1"
DEFAULT_COMMITTED_ROOT = BASE_ROOT / "psi_dispatch_dynamic_admission_551_560_v1"
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "psi_dispatch_dynamic_admission_wait_551_560_v1"


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(q)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


class FifoWaitingAuditProbe:
    """Audit state/task/token invariants and station-waiting exposure."""

    def __init__(self, engine, trace_max_records: int = 5000):
        self.engine = engine
        self.trace_max_records = max(0, int(trace_max_records))
        self.trace: list[dict[str, Any]] = []
        self.trace_dropped = 0
        self.max_committed_load: dict[int, int] = {}
        self.max_occupancy: dict[int, int] = {}
        self.max_waiting_depth: dict[int, int] = {}
        self.observed_ticks = 0
        self.waiting_agent_ticks = 0
        self.waiting_age_samples: list[float] = []
        self.waiting_duration_samples: list[float] = []
        self.waiting_promotions = 0
        self.waiting_cancellations = 0
        self.capacity_violations: list[dict[str, Any]] = []
        self.token_mismatches: list[dict[str, Any]] = []
        self.waiting_semantic_violations: list[dict[str, Any]] = []
        self.fifo_order_violations: list[dict[str, Any]] = []
        self._violation_totals = {
            "capacity": 0,
            "token": 0,
            "waiting_semantic": 0,
            "fifo": 0,
        }
        self._previous_waiting: dict[int, dict[str, int]] = {}

    @staticmethod
    def _next_deliver(world, agent_id: int):
        return world.task_state.get_next_task_for_agent(int(agent_id))

    def _record_violation(self, target: list, row: dict[str, Any]) -> None:
        if target is self.capacity_violations:
            key = "capacity"
        elif target is self.token_mismatches:
            key = "token"
        elif target is self.waiting_semantic_violations:
            key = "waiting_semantic"
        else:
            key = "fifo"
        self._violation_totals[key] += 1
        # Keep example payloads bounded while preserving exact totals.
        if len(target) < 100:
            target.append(row)

    def on_tick(self, engine) -> None:
        world = engine.world
        tick = int(world.tick)
        self.observed_ticks += 1
        current_waiting: dict[int, dict[str, int]] = {}
        tick_waiting: list[dict[str, Any]] = []

        for agent in world.agents:
            if agent.status != AgentStatus.WAITING_ASSIGNED:
                continue
            aid = int(agent.agent_id)
            station_id = agent.station_waiting_station_id
            queue = (
                world.station_state.get_queue(station_id)
                if station_id is not None else None
            )
            sequence = (
                queue.waiting_reservation_sequence(aid)
                if queue is not None else None
            )
            since = agent.station_waiting_since_tick
            if station_id is None or queue is None or sequence is None:
                self._record_violation(
                    self.waiting_semantic_violations,
                    {
                        "tick": tick,
                        "agent_id": aid,
                        "reason": "missing_station_or_fifo_metadata",
                        "station_id": station_id,
                        "sequence": sequence,
                    },
                )
                continue
            if since is None:
                self._record_violation(
                    self.waiting_semantic_violations,
                    {
                        "tick": tick,
                        "agent_id": aid,
                        "reason": "missing_waiting_since_tick",
                    },
                )
                since = tick
            next_deliver = self._next_deliver(world, aid)
            active = world.task_state.get_active_task_for_agent(aid)
            reasons: list[str] = []
            if next_deliver is None or next_deliver.task_type != TaskType.DELIVER:
                reasons.append("next_task_not_deliver")
            elif next_deliver.status != TaskStatus.ASSIGNED:
                reasons.append("deliver_not_assigned")
            if (
                next_deliver is not None
                and next_deliver.station_id != int(station_id)
            ):
                reasons.append("deliver_station_mismatch")
            if (
                next_deliver is not None
                and agent.assigned_task_id != next_deliver.task_id
            ):
                reasons.append("assigned_task_id_mismatch")
            if agent.station_waiting_sequence != int(sequence):
                reasons.append("agent_fifo_sequence_mismatch")
            if active is not None:
                reasons.append("active_task_present")
            if agent.carried_pod_id is None:
                reasons.append("carried_pod_missing")
            elif (
                next_deliver is not None
                and int(next_deliver.pod_id) != int(agent.carried_pod_id)
            ):
                reasons.append("carried_pod_task_mismatch")
            if agent.has_path:
                reasons.append("waiting_agent_has_path")
            if aid in queue.committed_agent_ids():
                reasons.append("waiting_agent_has_committed_token")
            if reasons:
                self._record_violation(
                    self.waiting_semantic_violations,
                    {
                        "tick": tick,
                        "agent_id": aid,
                        "station_id": int(station_id),
                        "reasons": reasons,
                    },
                )

            age = max(0.0, float(tick - int(since)))
            self.waiting_agent_ticks += 1
            self.waiting_age_samples.append(age)
            current_waiting[aid] = {
                "station_id": int(station_id),
                "sequence": int(sequence),
                "since_tick": int(since),
            }
            tick_waiting.append({
                "agent_id": aid,
                "station_id": int(station_id),
                "sequence": int(sequence),
                "since_tick": int(since),
                "age_ticks": age,
            })

        # A waiter that disappears from the FIFO list is either promoted or
        # cancelled.  Promotion is identifiable by CARRYING + active DELIVER
        # + a committed token; the previous request tick gives its exact wait.
        for aid, previous in self._previous_waiting.items():
            if aid in current_waiting:
                continue
            agent = world.get_agent(aid)
            active = world.task_state.get_active_task_for_agent(aid)
            queue = world.station_state.get_queue(previous["station_id"])
            promoted = bool(
                agent is not None
                and agent.status == AgentStatus.CARRYING
                and active is not None
                and active.task_type == TaskType.DELIVER
                and active.status == TaskStatus.IN_PROGRESS
                and queue is not None
                and aid in queue.committed_agent_ids()
            )
            if promoted:
                self.waiting_promotions += 1
                self.waiting_duration_samples.append(
                    max(0.0, float(tick - previous["since_tick"]))
                )
                # If an older waiter remained with its original sequence,
                # this promotion bypassed it.  A requeued failed head gets a
                # new sequence and is intentionally not flagged here.
                for other_id, other in self._previous_waiting.items():
                    if other_id == aid or other["station_id"] != previous["station_id"]:
                        continue
                    current_other = current_waiting.get(other_id)
                    if (
                        other["sequence"] < previous["sequence"]
                        and current_other is not None
                        and current_other["sequence"] == other["sequence"]
                    ):
                        self._record_violation(
                            self.fifo_order_violations,
                            {
                                "tick": tick,
                                "station_id": previous["station_id"],
                                "promoted_agent_id": aid,
                                "promoted_sequence": previous["sequence"],
                                "older_waiting_agent_id": other_id,
                                "older_sequence": other["sequence"],
                            },
                        )
            else:
                self.waiting_cancellations += 1

        for station_id, queue in world.station_state.stations.items():
            station_id = int(station_id)
            committed = int(queue.committed_load())
            occupancy = int(queue.occupancy())
            waiting_ids = queue.waiting_agent_ids()
            self.max_committed_load[station_id] = max(
                committed, self.max_committed_load.get(station_id, 0)
            )
            self.max_occupancy[station_id] = max(
                occupancy, self.max_occupancy.get(station_id, 0)
            )
            self.max_waiting_depth[station_id] = max(
                len(waiting_ids), self.max_waiting_depth.get(station_id, 0)
            )
            if committed > int(queue.capacity):
                self._record_violation(
                    self.capacity_violations,
                    {
                        "tick": tick,
                        "station_id": station_id,
                        "committed_load": committed,
                        "capacity": int(queue.capacity),
                        "committed_agent_ids": sorted(
                            int(x) for x in queue.committed_agent_ids()
                        ),
                    },
                )
            physical_ids = queue.physical_agent_ids()
            for aid in sorted(queue._assigned_agents):
                if aid in physical_ids:
                    continue
                active = world.task_state.get_active_task_for_agent(aid)
                active_type = (
                    str(getattr(active.task_type, "name", active.task_type))
                    if active is not None else None
                )
                active_station = (
                    int(active.station_id)
                    if active is not None and active.station_id is not None
                    else None
                )
                if active_type == "DELIVER" and active_station == station_id:
                    continue
                self._record_violation(
                    self.token_mismatches,
                    {
                        "tick": tick,
                        "station_id": station_id,
                        "agent_id": int(aid),
                        "active_task_type": active_type,
                        "active_station_id": active_station,
                    },
                )
            sequences = [
                queue.waiting_reservation_sequence(aid)
                for aid in waiting_ids
            ]
            if any(value is None for value in sequences) or sequences != sorted(sequences):
                self._record_violation(
                    self.fifo_order_violations,
                    {
                        "tick": tick,
                        "station_id": station_id,
                        "waiting_agent_ids": waiting_ids,
                        "sequences": sequences,
                        "reason": "waiting_sequence_not_fifo",
                    },
                )
            for aid in waiting_ids:
                agent = world.get_agent(aid)
                if agent is None or agent.status != AgentStatus.WAITING_ASSIGNED:
                    self._record_violation(
                        self.waiting_semantic_violations,
                        {
                            "tick": tick,
                            "station_id": station_id,
                            "agent_id": int(aid),
                            "reason": "stale_waiting_reservation",
                        },
                    )

        if len(self.trace) < self.trace_max_records:
            self.trace.append({
                "tick": tick,
                "waiting_agents": tick_waiting,
                "stations": [
                    {
                        "station_id": int(station_id),
                        "capacity": int(queue.capacity),
                        "occupancy": int(queue.occupancy()),
                        "committed_load": int(queue.committed_load()),
                        "waiting_agent_ids": queue.waiting_agent_ids(),
                        "waiting_depth": len(queue.waiting_agent_ids()),
                    }
                    for station_id, queue in sorted(
                        world.station_state.stations.items()
                    )
                ],
            })
        else:
            self.trace_dropped += 1
        self._previous_waiting = current_waiting

    def summary(self) -> dict[str, Any]:
        station_metrics = self.engine.world.station_state.admission_metrics()
        final_tick = int(getattr(self.engine.world, "tick", 0))
        unresolved_by_station: dict[str, int] = {}
        oldest_by_station: dict[str, float] = {}
        for row in self._previous_waiting.values():
            station = str(int(row["station_id"]))
            unresolved_by_station[station] = unresolved_by_station.get(station, 0) + 1
            age = max(0.0, float(final_tick - int(row["since_tick"])))
            oldest_by_station[station] = max(
                age, oldest_by_station.get(station, 0.0)
            )
        return {
            "mode": STATION_ADMISSION_COMMITTED_FIFO_V2,
            "passed": bool(
                not self.capacity_violations
                and not self.token_mismatches
                and not self.waiting_semantic_violations
                and not self.fifo_order_violations
                and all(bool(row["invariant_holds"]) for row in station_metrics)
                and all(
                    row["mode"] == STATION_ADMISSION_COMMITTED_FIFO_V2
                    for row in station_metrics
                )
            ),
            "max_committed_load": {
                str(key): int(value)
                for key, value in sorted(self.max_committed_load.items())
            },
            "max_occupancy": {
                str(key): int(value)
                for key, value in sorted(self.max_occupancy.items())
            },
            "max_waiting_depth": {
                str(key): int(value)
                for key, value in sorted(self.max_waiting_depth.items())
            },
            "waiting_agent_ticks": int(self.waiting_agent_ticks),
            "waiting_assigned_ratio": float(
                self.waiting_agent_ticks
                / max(self.observed_ticks * len(self.engine.world.agents), 1)
            ),
            "waiting_age_p50_ticks": _percentile(self.waiting_age_samples, 0.50),
            "waiting_age_p95_ticks": _percentile(self.waiting_age_samples, 0.95),
            "waiting_age_max_ticks": max(self.waiting_age_samples, default=None),
            "waiting_promotions": int(self.waiting_promotions),
            "waiting_cancellations_observed": int(self.waiting_cancellations),
            "waiting_duration_p50_ticks": _percentile(
                self.waiting_duration_samples, 0.50
            ),
            "waiting_duration_p95_ticks": _percentile(
                self.waiting_duration_samples, 0.95
            ),
            "waiting_duration_max_ticks": max(
                self.waiting_duration_samples, default=None
            ),
            "unresolved_waiter_count_final": int(len(self._previous_waiting)),
            "unresolved_waiters_by_station_final": unresolved_by_station,
            "oldest_waiter_age_by_station_final": oldest_by_station,
            "oldest_waiter_age_final_ticks": max(
                oldest_by_station.values(), default=None
            ),
            "capacity_violation_count": int(self._violation_totals["capacity"]),
            "capacity_violations": self.capacity_violations[:100],
            "token_mismatch_count": int(self._violation_totals["token"]),
            "token_mismatches": self.token_mismatches[:100],
            "waiting_semantic_violation_count": int(
                self._violation_totals["waiting_semantic"]
            ),
            "waiting_semantic_violations": self.waiting_semantic_violations[:100],
            "fifo_order_violation_count": int(self._violation_totals["fifo"]),
            "fifo_order_violations": self.fifo_order_violations[:100],
            "trace": self.trace,
            "trace_dropped": int(self.trace_dropped),
            "station_metrics_final": station_metrics,
        }


def _output_path(root: Path, load: str, seed: int) -> Path:
    return root / "per_arm" / ARM_KEY / f"{load}_seed{seed}.json"


def _runtime_code_hashes() -> dict[str, str]:
    return {
        "runner_code_sha256": sha256_file(Path(__file__)),
        "station_state_code_sha256": sha256_file(
            Path("WorldState/station_state.py")
        ),
        "engine_code_sha256": sha256_file(Path("Engine/simulation_engine.py")),
        "prioritized_planner_code_sha256": sha256_file(
            Path(
                "Policies/PathPlanner/PrioritizedPathPlanner/"
                "prioritized_path_planner.py"
            )
        ),
        "counterfactual_code_sha256": sha256_file(
            Path("WorldModel/data/counterfactual_rollout.py")
        ),
    }


def _reference_metrics(
    source_root: Path,
    legacy_dynamic_root: Path,
    committed_root: Path,
    load: str,
    seed: int,
) -> dict[str, Any]:
    references = {
        "s1_baseline": (
            source_root / "per_arm" / "s1" / f"{load}_seed{seed}.json"
        ),
        "greedy_manifest": (
            source_root / "per_arm" / "greedy_manifest" / f"{load}_seed{seed}.json"
        ),
        "s1_psi_shadow": (
            source_root / "per_arm" / "s1_psi_shadow" / f"{load}_seed{seed}.json"
        ),
        "s1_psi_dispatch_static": (
            source_root / "per_arm" / "s1_psi_dispatch" / f"{load}_seed{seed}.json"
        ),
        "s1_psi_dynamic_legacy": (
            legacy_dynamic_root
            / "per_arm"
            / "s1_psi_dynamic_probe"
            / f"{load}_seed{seed}.json"
        ),
        "s1_psi_dynamic_committed_v1": (
            committed_root
            / "per_arm"
            / "s1_psi_dynamic_committed_admission_v1"
            / f"{load}_seed{seed}.json"
        ),
    }
    keys = (
        "completed_orders",
        "completed_tasks",
        "deadlock_ratio_mean",
        "deadlock_ratio_max",
        "stall_ratio_mean",
        "stall_ratio_max",
        "pending_order_count",
        "open_order_count",
    )
    result: dict[str, Any] = {}
    for name, path in references.items():
        if not path.is_file():
            continue
        payload = _read_json(path)
        metrics = payload.get("metrics") or {}
        result[name] = {
            "path": path.as_posix(),
            **{key: metrics.get(key) for key in keys},
        }
    return result


def _audit(
    metrics: Mapping[str, Any],
    manifest_payload: Mapping[str, Any],
    admission_audit: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "manifest_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest_payload.get("manifest_sha256")
        ),
        "manifest_count_matches": int(metrics.get("order_arrival_count", -1))
        == int(manifest_payload.get("total_orders", -2)),
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
        "s1_used": int(metrics.get("energy_conv_contexts", 0)) > 0,
        "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
        "dynamic_batches_seen": int(metrics.get("dynamic_probe_batches", 0)) > 0,
        "dynamic_choices_reindexed": bool(
            metrics.get("dynamic_probe_choices_reindexed", False)
        ),
        "dynamic_head_loaded": bool(metrics.get("psi_dispatch_head_loaded")),
        "parent_robot_scorer_preserved": (
            metrics.get("psi_dispatch_robot_scorer")
            == "WorldModelTaskAssigner.select_robots_unmodified"
        ),
        "no_assign_added": not bool(
            metrics.get("psi_dispatch_no_assign_added", True)
        ),
        "hard_gate_added": not bool(
            metrics.get("psi_dispatch_hard_gate_added", True)
        ),
        "e_demand_modified": not bool(
            metrics.get("psi_dispatch_e_demand_modified", True)
        ),
        "fifo_waiting_lifecycle_invariant": bool(admission_audit.get("passed")),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", choices=LOADS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--legacy-dynamic-root", type=Path, default=DEFAULT_LEGACY_DYNAMIC_ROOT
    )
    parser.add_argument(
        "--committed-root", type=Path, default=DEFAULT_COMMITTED_ROOT
    )
    parser.add_argument("--frozen-bundle", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--trace-max-records", type=int, default=500)
    args = parser.parse_args()

    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive")
    bundle_path = args.frozen_bundle or (
        args.source_root / "phase_c_psi_dispatch_frozen_protocol.json"
    )
    bundle, protocol = _load_bundle(bundle_path)
    model_checkpoint = _artifact(bundle, "model_checkpoint")
    psi_head_checkpoint = _artifact(bundle, "psi_head_checkpoint")
    psi_scale_contract = _artifact(bundle, "psi_scale_contract")
    config_path = _artifact(bundle, f"config_{args.load}")
    manifest_path = _manifest_path(args.source_root, args.load, args.seed)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    runtime_code_hashes = _runtime_code_hashes()

    output_path = _output_path(args.output_root, args.load, args.seed)
    if output_path.is_file():
        existing = _read_json(output_path)
        meta = existing.get("meta") or {}
        if (
            existing.get("schema_version") != SCHEMA_VERSION
            or meta.get("station_admission") != STATION_ADMISSION_COMMITTED_FIFO_V2
            or meta.get("load") != args.load
            or int(meta.get("seed", -1)) != int(args.seed)
            or int(meta.get("ticks", -1)) != int(args.ticks)
            or meta.get("frozen_bundle_sha256") != sha256_file(bundle_path)
            or any(
                meta.get(key) != value
                for key, value in runtime_code_hashes.items()
            )
        ):
            raise FileExistsError(f"incompatible existing output: {output_path}")
        print(f"[resume] {output_path}")
        return

    common = {
        "checkpoint_path": str(model_checkpoint),
        "top_m": TOP_M,
        "energy_conv_random_flip_seed": 0,
        **S1_CONFIG,
    }
    assigner = DynamicPsiDispatchProbeAssigner(
        psi_head_checkpoint=str(psi_head_checkpoint),
        psi_scale_contract=str(psi_scale_contract),
        dynamic_trace_enabled=True,
        dynamic_trace_max_records=int(args.trace_max_records),
        **common,
    )

    import WorldModel.evaluation.evaluate_online_v6 as eval_module

    original_builder = eval_module._build_engine
    holder: dict[str, Any] = {}

    def build_with_fifo_admission(cfg, task_assigner=None):
        engine = original_builder(cfg, task_assigner=task_assigner)
        engine.world.station_state.set_admission_mode(
            STATION_ADMISSION_COMMITTED_FIFO_V2
        )
        audit_probe = FifoWaitingAuditProbe(
            engine, trace_max_records=int(args.trace_max_records)
        )
        engine.on_tick_callbacks.append(audit_probe.on_tick)
        holder["engine"] = engine
        holder["audit_probe"] = audit_probe
        return engine

    eval_module._build_engine = build_with_fifo_admission
    try:
        print(
            f"[run] dynamic FIFO waiting load={args.load} "
            f"seed={args.seed} ticks={args.ticks}"
        )
        metrics = eval_module._run_one_assigner(
            str(config_path),
            assigner,
            int(args.seed),
            int(args.ticks),
            trace_label=ARM_LABEL,
            recorded_orders_path=str(manifest_path),
        )
    finally:
        eval_module._build_engine = original_builder

    audit_probe = holder.get("audit_probe")
    if audit_probe is None:
        raise RuntimeError("FIFO waiting audit probe was not attached")
    admission_audit = audit_probe.summary()
    metrics.update(assigner.dynamic_probe_metrics())
    # Keep the admission channel next to the ordinary simulator metrics so a
    # consumer does not mistake legal station waiting for physical deadlock.
    metrics.update({
        "waiting_assigned_agent_ticks": admission_audit["waiting_agent_ticks"],
        "waiting_assigned_ratio": admission_audit["waiting_assigned_ratio"],
        "waiting_promotions": admission_audit["waiting_promotions"],
        "waiting_duration_p95_ticks": admission_audit[
            "waiting_duration_p95_ticks"
        ],
        "waiting_duration_max_ticks": admission_audit[
            "waiting_duration_max_ticks"
        ],
        "unresolved_waiter_count_final": admission_audit[
            "unresolved_waiter_count_final"
        ],
    })
    manifest_payload = _read_json(manifest_path)
    audit = _audit(metrics, manifest_payload, admission_audit)
    if not audit["passed"]:
        failed = [key for key, passed in audit["checks"].items() if not passed]
        raise RuntimeError(
            f"dynamic FIFO waiting audit failed "
            f"{args.load} seed={args.seed}: {failed}"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": str(protocol["protocol_sha256"]),
            "frozen_bundle": bundle_path.as_posix(),
            "frozen_bundle_sha256": sha256_file(bundle_path),
            "arm_key": ARM_KEY,
            "arm_label": ARM_LABEL,
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "formal_development": False,
            "station_admission": STATION_ADMISSION_COMMITTED_FIFO_V2,
            "waiting_assigned_state": True,
            "deliver_task_stays_assigned_until_promotion": True,
            "fifo_promotion": True,
            "policy_logic_changed": False,
            "model_checkpoint_changed": False,
            "station_exit_logic_changed": False,
            "path_planner_logic_changed": False,
            "model_checkpoint": model_checkpoint.as_posix(),
            "psi_head_checkpoint": psi_head_checkpoint.as_posix(),
            "psi_scale_contract": psi_scale_contract.as_posix(),
            "source_manifest_root": args.source_root.as_posix(),
            "runner_code": Path(__file__).as_posix(),
            **runtime_code_hashes,
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest_payload.get("manifest_sha256"),
            "total_orders": manifest_payload.get("total_orders"),
        },
        "reference_metrics": _reference_metrics(
            args.source_root,
            args.legacy_dynamic_root,
            args.committed_root,
            args.load,
            args.seed,
        ),
        "audit": audit,
        "station_admission_audit": admission_audit,
        "metrics": metrics,
        "dynamic_probe_trace": assigner.dynamic_probe_trace_records,
    }
    _atomic_json(output_path, payload)
    print(f"[done] {output_path}")
    print(json.dumps({
        "completed_orders": metrics.get("completed_orders"),
        "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
        "waiting_agent_ticks": admission_audit.get("waiting_agent_ticks"),
        "waiting_duration_p95_ticks": admission_audit.get(
            "waiting_duration_p95_ticks"
        ),
        "max_waiting_depth": admission_audit.get("max_waiting_depth"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
