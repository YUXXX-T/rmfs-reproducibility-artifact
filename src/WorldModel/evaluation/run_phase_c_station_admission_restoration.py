"""Run the strict Phase-C station-admission restoration matrix.

The formal comparison replays one frozen order-arrival manifest per seed and
changes only two factors: task-assignment policy and station admission mode.
In particular, ``physical_only_legacy`` is the true restoration arm: current
physical station occupancy remains bounded by configured slots, while admitted
robots still travelling to the station do not consume a committed-capacity
budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from Policies.TaskAssigner import GreedyTaskAssigner, HungarianTaskAssigner
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_dynamic_probe_assigner import (
    DynamicPsiDispatchProbeAssigner,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    LOADS,
    S1_CONFIG,
    TOP_M,
    canonical_sha256,
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_psi_dynamic_admission import (
    _artifact,
    _atomic_json,
    _load_bundle,
    _read_json,
)
from WorldModel.evaluation.run_phase_c_psi_dynamic_admission_wait import (
    FifoWaitingAuditProbe,
)
from WorldState.station_state import (
    STATION_ADMISSION_COMMITTED_FIFO_V2,
    STATION_ADMISSION_COMMITTED_V1,
    STATION_ADMISSION_DYNAMIC_ETA_V1,
    STATION_ADMISSION_PHYSICAL_ONLY,
)


SCHEMA_VERSION = "phase_c_station_admission_restoration_arm_v1"
SUMMARY_SCHEMA_VERSION = "phase_c_station_admission_restoration_summary_v1"
MANIFEST_SOURCE_SCHEMA_VERSION = (
    "phase_c_station_admission_restoration_manifest_source_v1"
)

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_SOURCE_ROOT = (
    BASE_ROOT / "psi_dispatch_dynamic_eta_overbooking_stress_source_551_560_v1"
)
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "station_admission_restoration_551_560_v1"
DEFAULT_BUNDLE = (
    BASE_ROOT
    / "psi_dispatch_ablation_551_560_v1"
    / "phase_c_psi_dispatch_frozen_protocol.json"
)
DEFAULT_V1_REFERENCE_ROOT = BASE_ROOT / "psi_dispatch_dynamic_admission_551_560_v1"
DEFAULT_V3_REFERENCE_ROOT = (
    BASE_ROOT / "psi_dispatch_dynamic_eta_overbooking_551_560_v3_cpu"
)
FORMAL_SEEDS = tuple(range(551, 561))
FORMAL_TICKS = 1500
MANIFEST_SOURCE_KEY = "manifest_source_greedy_physical_only"


@dataclass(frozen=True)
class ArmSpec:
    key: str
    label: str
    policy: str
    admission_mode: str


ARM_SPECS = {
    spec.key: spec
    for spec in (
        ArmSpec(
            "dynamic_j_physical_only_restored_v1",
            "Dynamic J + physical-only (restored)",
            "dynamic_j",
            STATION_ADMISSION_PHYSICAL_ONLY,
        ),
        ArmSpec(
            "dynamic_j_committed_v1",
            "Dynamic J + committed V1",
            "dynamic_j",
            STATION_ADMISSION_COMMITTED_V1,
        ),
        ArmSpec(
            "dynamic_j_eta_overbooking_v3",
            "Dynamic J + ETA V3",
            "dynamic_j",
            STATION_ADMISSION_DYNAMIC_ETA_V1,
        ),
        ArmSpec(
            "greedy_physical_only",
            "Greedy + physical-only",
            "greedy",
            STATION_ADMISSION_PHYSICAL_ONLY,
        ),
        ArmSpec(
            "greedy_fifo_v2",
            "Greedy + FIFO V2",
            "greedy",
            STATION_ADMISSION_COMMITTED_FIFO_V2,
        ),
        ArmSpec(
            "hungarian_physical_only",
            "Hungarian + physical-only",
            "hungarian",
            STATION_ADMISSION_PHYSICAL_ONLY,
        ),
        ArmSpec(
            "hungarian_fifo_v2",
            "Hungarian + FIFO V2",
            "hungarian",
            STATION_ADMISSION_COMMITTED_FIFO_V2,
        ),
    )
}
ARM_KEYS = tuple(ARM_SPECS)

COMPARISONS = (
    (
        "dynamic_physical_minus_committed_v1",
        "dynamic_j_physical_only_restored_v1",
        "dynamic_j_committed_v1",
    ),
    (
        "dynamic_physical_minus_eta_v3",
        "dynamic_j_physical_only_restored_v1",
        "dynamic_j_eta_overbooking_v3",
    ),
    (
        "dynamic_physical_minus_greedy_physical",
        "dynamic_j_physical_only_restored_v1",
        "greedy_physical_only",
    ),
    (
        "dynamic_physical_minus_hungarian_physical",
        "dynamic_j_physical_only_restored_v1",
        "hungarian_physical_only",
    ),
    (
        "dynamic_physical_minus_greedy_fifo",
        "dynamic_j_physical_only_restored_v1",
        "greedy_fifo_v2",
    ),
    (
        "dynamic_physical_minus_hungarian_fifo",
        "dynamic_j_physical_only_restored_v1",
        "hungarian_fifo_v2",
    ),
    (
        "greedy_fifo_minus_physical",
        "greedy_fifo_v2",
        "greedy_physical_only",
    ),
    (
        "hungarian_fifo_minus_physical",
        "hungarian_fifo_v2",
        "hungarian_physical_only",
    ),
)


def _manifest_path(root: Path, load: str, seed: int) -> Path:
    return root / "order_manifests" / f"orders_{load}_seed{seed}.json"


def _output_path(root: Path, arm: str, load: str, seed: int) -> Path:
    return root / "per_arm" / arm / f"{load}_seed{seed}.json"


def _canonical_manifest_sha(payload: Mapping[str, Any]) -> str:
    canonical = {
        "schema_version": payload.get("schema_version"),
        "orders": payload.get("orders"),
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    rows = payload.get("orders")
    checks = {
        "schema": payload.get("schema_version")
        == "layer5_order_arrival_manifest_v1",
        "orders": isinstance(rows, list),
        "count": isinstance(rows, list)
        and int(payload.get("total_orders", -1)) == len(rows),
        "hash": _canonical_manifest_sha(payload)
        == payload.get("manifest_sha256"),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"invalid order manifest {path}: {failed}")
    return payload


def _reference_manifest(
    root: Path,
    arm: str,
    load: str,
    seed: int,
) -> dict[str, Any] | None:
    path = root / "per_arm" / arm / f"{load}_seed{seed}.json"
    if not path.is_file():
        return None
    payload = _read_json(path)
    manifest = payload.get("manifest") or {}
    content_sha = manifest.get("content_sha256")
    total_orders = manifest.get("total_orders")
    if not content_sha or total_orders is None:
        raise ValueError(f"reference result lacks manifest contract: {path}")
    return {
        "path": path.as_posix(),
        "content_sha256": str(content_sha),
        "total_orders": int(total_orders),
    }


def _validate_historical_contract(
    manifest: Mapping[str, Any],
    *,
    v1_root: Path,
    v3_root: Path,
    load: str,
    seed: int,
    require_v3: bool,
    skip: bool,
) -> dict[str, Any]:
    if skip:
        return {"skipped": True, "references": {}}

    references: dict[str, Any] = {}
    v1 = _reference_manifest(
        v1_root,
        "s1_psi_dynamic_committed_admission_v1",
        load,
        seed,
    )
    if v1 is None:
        raise FileNotFoundError(
            "missing mandatory Dynamic-J committed V1 manifest contract for "
            f"{load} seed={seed}: {v1_root}"
        )
    references["dynamic_j_committed_v1"] = v1

    v3 = _reference_manifest(
        v3_root,
        "s1_psi_dynamic_eta_overbooking_v3",
        load,
        seed,
    )
    if v3 is not None:
        references["historical_eta_v3"] = v3
    elif require_v3:
        raise FileNotFoundError(
            "missing required historical ETA V3 manifest contract for "
            f"{load} seed={seed}: {v3_root}"
        )

    expected_sha = str(manifest.get("manifest_sha256"))
    expected_count = int(manifest.get("total_orders", -1))
    mismatches = []
    for name, reference in references.items():
        if (
            reference["content_sha256"] != expected_sha
            or int(reference["total_orders"]) != expected_count
        ):
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(
            f"manifest differs from historical paired contract {load} "
            f"seed={seed}: {mismatches}"
        )
    return {
        "skipped": False,
        "passed": True,
        "content_sha256": expected_sha,
        "total_orders": expected_count,
        "references": references,
    }


def _runtime_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "station_state": Path("WorldState/station_state.py"),
        "simulation_engine": Path("Engine/simulation_engine.py"),
        "counterfactual_rollout": Path(
            "WorldModel/data/counterfactual_rollout.py"
        ),
        "evaluate_online": Path("WorldModel/evaluation/evaluate_online_v6.py"),
        "dynamic_j_assigner": Path(
            "Policies/TaskAssigner/WorldModelTaskAssigner/"
            "psi_dispatch_dynamic_probe_assigner.py"
        ),
        "greedy_assigner": Path(
            "Policies/TaskAssigner/GreedyTaskAssigner/greedy_task_assigner.py"
        ),
        "hungarian_assigner": Path(
            "Policies/TaskAssigner/HungarianTaskAssigner/"
            "hungarian_task_assigner.py"
        ),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _dynamic_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_committed_multiplier": float(args.max_committed_multiplier),
        "healthy_extra_ratio": float(args.healthy_extra_ratio),
        "caution_extra_ratio": float(args.caution_extra_ratio),
        "brake_extra_ratio": float(args.brake_extra_ratio),
        "eta_near_ticks": int(args.eta_near_ticks),
        "eta_mid_ticks": int(args.eta_mid_ticks),
    }


def _policy_contract(
    spec: ArmSpec,
    *,
    model_checkpoint: Path,
    psi_head_checkpoint: Path,
    psi_scale_contract: Path,
    runtime_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if spec.policy == "dynamic_j":
        contract = {
            "policy": "DynamicPsiDispatchProbeAssigner",
            "checkpoint": model_checkpoint.as_posix(),
            "checkpoint_sha256": sha256_file(model_checkpoint),
            "psi_head_checkpoint": psi_head_checkpoint.as_posix(),
            "psi_head_checkpoint_sha256": sha256_file(psi_head_checkpoint),
            "psi_scale_contract": psi_scale_contract.as_posix(),
            "psi_scale_contract_sha256": sha256_file(psi_scale_contract),
            "top_m": int(TOP_M),
            "energy_conv_random_flip_seed": 0,
            "s1_config": dict(S1_CONFIG),
            "dynamic_interleaved": True,
            "policy_code_sha256": runtime_hashes["dynamic_j_assigner"],
        }
    elif spec.policy == "greedy":
        contract = {
            "policy": "GreedyTaskAssigner",
            "policy_code_sha256": runtime_hashes["greedy_assigner"],
        }
    elif spec.policy == "hungarian":
        contract = {
            "policy": "HungarianTaskAssigner",
            "objective": "global Manhattan-distance assignment",
            "policy_code_sha256": runtime_hashes["hungarian_assigner"],
        }
    else:
        raise ValueError(spec.policy)
    return {
        **contract,
        "fingerprint_sha256": canonical_sha256(contract),
    }


def _make_assigner(
    spec: ArmSpec,
    *,
    model_checkpoint: Path,
    psi_head_checkpoint: Path,
    psi_scale_contract: Path,
    trace_max_records: int,
):
    if spec.policy == "greedy":
        return GreedyTaskAssigner()
    if spec.policy == "hungarian":
        return HungarianTaskAssigner()
    if spec.policy == "dynamic_j":
        return DynamicPsiDispatchProbeAssigner(
            psi_head_checkpoint=str(psi_head_checkpoint),
            psi_scale_contract=str(psi_scale_contract),
            dynamic_trace_enabled=int(trace_max_records) > 0,
            dynamic_trace_max_records=int(trace_max_records),
            dynamic_interleaved=True,
            checkpoint_path=str(model_checkpoint),
            top_m=TOP_M,
            energy_conv_random_flip_seed=0,
            **S1_CONFIG,
        )
    raise ValueError(spec.policy)


class StationAdmissionRestorationAuditProbe:
    """Collect one common station audit schema for every matrix arm."""

    def __init__(
        self,
        engine,
        expected_mode: str,
        *,
        trace_stride: int = 5,
        trace_max_records: int = 300,
    ):
        self.engine = engine
        self.expected_mode = str(expected_mode)
        self.trace_stride = max(1, int(trace_stride))
        self.trace_max_records = max(0, int(trace_max_records))
        self.trace: list[dict[str, Any]] = []
        self.trace_dropped = 0
        self.max_occupancy: dict[int, int] = {}
        self.max_committed_load: dict[int, int] = {}
        self.max_in_transit: dict[int, int] = {}
        self.over_capacity_station_ticks: dict[int, int] = {}
        self.over_capacity_excess_sum: dict[int, int] = {}
        self.ticks_with_any_over_capacity: set[int] = set()
        self._totals = {
            "physical_capacity": 0,
            "committed_capacity": 0,
            "dynamic_hard_limit": 0,
            "token_mismatch": 0,
            "mode_mismatch": 0,
        }
        self.physical_capacity_violations: list[dict[str, Any]] = []
        self.committed_capacity_violations: list[dict[str, Any]] = []
        self.dynamic_hard_limit_violations: list[dict[str, Any]] = []
        self.token_mismatches: list[dict[str, Any]] = []
        self.mode_mismatches: list[dict[str, Any]] = []

    def _record(self, key: str, target: list, row: dict[str, Any]) -> None:
        self._totals[key] += 1
        if len(target) < 100:
            target.append(row)

    def on_tick(self, engine) -> None:
        world = engine.world
        tick = int(world.tick)
        trace_stations = []
        any_over_capacity = False
        for station_id, queue in sorted(
            world.station_state.stations.items()
        ):
            station_id = int(station_id)
            capacity = int(queue.capacity)
            occupancy = int(queue.occupancy())
            physical_ids = {
                int(agent_id) for agent_id in queue.physical_agent_ids()
            }
            committed_ids = {
                int(agent_id) for agent_id in queue.committed_agent_ids()
            }
            committed = len(committed_ids)
            in_transit = len(committed_ids - physical_ids)
            self.max_occupancy[station_id] = max(
                occupancy, self.max_occupancy.get(station_id, 0)
            )
            self.max_committed_load[station_id] = max(
                committed, self.max_committed_load.get(station_id, 0)
            )
            self.max_in_transit[station_id] = max(
                in_transit, self.max_in_transit.get(station_id, 0)
            )

            if queue.admission_mode != self.expected_mode:
                self._record(
                    "mode_mismatch",
                    self.mode_mismatches,
                    {
                        "tick": tick,
                        "station_id": station_id,
                        "expected": self.expected_mode,
                        "actual": queue.admission_mode,
                    },
                )
            if occupancy > capacity:
                self._record(
                    "physical_capacity",
                    self.physical_capacity_violations,
                    {
                        "tick": tick,
                        "station_id": station_id,
                        "capacity": capacity,
                        "occupancy": occupancy,
                        "physical_agent_ids": sorted(physical_ids),
                    },
                )
            if committed > capacity:
                any_over_capacity = True
                self.over_capacity_station_ticks[station_id] = (
                    self.over_capacity_station_ticks.get(station_id, 0) + 1
                )
                self.over_capacity_excess_sum[station_id] = (
                    self.over_capacity_excess_sum.get(station_id, 0)
                    + committed
                    - capacity
                )
                if self.expected_mode in (
                    STATION_ADMISSION_COMMITTED_V1,
                    STATION_ADMISSION_COMMITTED_FIFO_V2,
                ):
                    self._record(
                        "committed_capacity",
                        self.committed_capacity_violations,
                        {
                            "tick": tick,
                            "station_id": station_id,
                            "capacity": capacity,
                            "committed_load": committed,
                            "committed_agent_ids": sorted(committed_ids),
                        },
                    )
            if (
                queue.uses_dynamic_eta_overbooking
                and committed > int(queue.dynamic_hard_limit())
            ):
                self._record(
                    "dynamic_hard_limit",
                    self.dynamic_hard_limit_violations,
                    {
                        "tick": tick,
                        "station_id": station_id,
                        "committed_load": committed,
                        "hard_limit": int(queue.dynamic_hard_limit()),
                        "committed_agent_ids": sorted(committed_ids),
                    },
                )

            for agent_id in sorted(queue._assigned_agents):
                if int(agent_id) in physical_ids:
                    continue
                active = world.task_state.get_active_task_for_agent(agent_id)
                active_type = (
                    str(getattr(active.task_type, "name", active.task_type))
                    if active is not None
                    else None
                )
                active_station = (
                    int(active.station_id)
                    if active is not None and active.station_id is not None
                    else None
                )
                if active_type == "DELIVER" and active_station == station_id:
                    continue
                self._record(
                    "token_mismatch",
                    self.token_mismatches,
                    {
                        "tick": tick,
                        "station_id": station_id,
                        "agent_id": int(agent_id),
                        "active_task_type": active_type,
                        "active_station_id": active_station,
                    },
                )

            if (
                self.trace_max_records > 0
                and tick % self.trace_stride == 0
            ):
                row = {
                    "station_id": station_id,
                    "mode": queue.admission_mode,
                    "capacity": capacity,
                    "occupancy": occupancy,
                    "committed_load": committed,
                    "in_transit_commitments": in_transit,
                    "committed_over_capacity": committed > capacity,
                    "committed_excess": max(0, committed - capacity),
                    "admission_attempts": int(queue._admission_attempts),
                    "admission_granted": int(queue._admission_granted),
                    "capacity_rejections": int(
                        queue._admission_rejected_capacity
                    ),
                    "over_capacity_grants": int(
                        queue._admission_over_capacity_grants
                    ),
                    "rollbacks": dict(queue._admission_rollbacks),
                }
                if queue.uses_dynamic_eta_overbooking:
                    row["dynamic_eta"] = {
                        "hard_limit": int(queue.dynamic_hard_limit()),
                        "health_state": queue.dynamic_health_state(tick),
                        "weighted_committed_mass": round(
                            float(queue.dynamic_weighted_committed_mass(tick)),
                            6,
                        ),
                        "weighted_limit": round(
                            float(queue.dynamic_weight_limit(tick)), 6
                        ),
                        "near_window_limit": int(
                            queue.dynamic_window_limit(
                                queue._dynamic_eta_near_ticks, tick
                            )
                        ),
                        "mid_window_limit": int(
                            queue.dynamic_window_limit(
                                queue._dynamic_eta_mid_ticks, tick
                            )
                        ),
                        "eta_bucket_counts": queue.dynamic_eta_bucket_counts(
                            tick
                        ),
                    }
                trace_stations.append(row)

        if any_over_capacity:
            self.ticks_with_any_over_capacity.add(tick)
        if trace_stations:
            row = {"tick": tick, "stations": trace_stations}
            if len(self.trace) < self.trace_max_records:
                self.trace.append(row)
            else:
                self.trace_dropped += 1

    def summary(self) -> dict[str, Any]:
        station_metrics = self.engine.world.station_state.admission_metrics()
        capacity_rejections = sum(
            int(row.get("rejected_capacity", 0)) for row in station_metrics
        )
        over_capacity_grants = sum(
            int(row.get("over_capacity_grants", 0)) for row in station_metrics
        )
        rollbacks = {"entry_occupied": 0, "path_failure": 0, "other": 0}
        for row in station_metrics:
            for key in rollbacks:
                rollbacks[key] += int((row.get("rollbacks") or {}).get(key, 0))
        passed = bool(
            self._totals["physical_capacity"] == 0
            and self._totals["token_mismatch"] == 0
            and self._totals["mode_mismatch"] == 0
            and self._totals["committed_capacity"] == 0
            and self._totals["dynamic_hard_limit"] == 0
            and all(bool(row.get("invariant_holds")) for row in station_metrics)
            and all(
                row.get("mode") == self.expected_mode
                for row in station_metrics
            )
        )
        return {
            "mode": self.expected_mode,
            "passed": passed,
            "physical_capacity_contract": "occupancy <= configured capacity",
            "committed_capacity_contract": (
                "not enforced for in-transit commitments"
                if self.expected_mode == STATION_ADMISSION_PHYSICAL_ONLY
                else (
                    "committed_load <= physical capacity"
                    if self.expected_mode
                    in (
                        STATION_ADMISSION_COMMITTED_V1,
                        STATION_ADMISSION_COMMITTED_FIFO_V2,
                    )
                    else "ETA/window controller with finite hard limit"
                )
            ),
            "max_occupancy": {
                str(key): int(value)
                for key, value in sorted(self.max_occupancy.items())
            },
            "max_committed_load": {
                str(key): int(value)
                for key, value in sorted(self.max_committed_load.items())
            },
            "max_in_transit_commitments": {
                str(key): int(value)
                for key, value in sorted(self.max_in_transit.items())
            },
            "committed_over_capacity_station_ticks": {
                str(key): int(value)
                for key, value in sorted(
                    self.over_capacity_station_ticks.items()
                )
            },
            "committed_over_capacity_station_tick_count": int(
                sum(self.over_capacity_station_ticks.values())
            ),
            "ticks_with_any_committed_over_capacity": int(
                len(self.ticks_with_any_over_capacity)
            ),
            "committed_over_capacity_excess_sum": {
                str(key): int(value)
                for key, value in sorted(self.over_capacity_excess_sum.items())
            },
            "over_capacity_grants": int(over_capacity_grants),
            "capacity_rejections": int(capacity_rejections),
            "rollbacks": rollbacks,
            "physical_capacity_violation_count": int(
                self._totals["physical_capacity"]
            ),
            "physical_capacity_violations": self.physical_capacity_violations,
            "committed_capacity_violation_count": int(
                self._totals["committed_capacity"]
            ),
            "committed_capacity_violations": (
                self.committed_capacity_violations
            ),
            "dynamic_hard_limit_violation_count": int(
                self._totals["dynamic_hard_limit"]
            ),
            "dynamic_hard_limit_violations": (
                self.dynamic_hard_limit_violations
            ),
            "token_mismatch_count": int(self._totals["token_mismatch"]),
            "token_mismatches": self.token_mismatches,
            "mode_mismatch_count": int(self._totals["mode_mismatch"]),
            "mode_mismatches": self.mode_mismatches,
            "trace_stride": int(self.trace_stride),
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
    fifo_trace_max_records: int,
) -> Iterator[dict[str, Any]]:
    import WorldModel.evaluation.evaluate_online_v6 as eval_module

    original_builder = eval_module._build_engine
    holder: dict[str, Any] = {}

    def build(cfg, task_assigner=None):
        # Physical-only deliberately creates strong station pressure.  The
        # engine otherwise emits the same failed-path warning on every retry,
        # which can dominate wall time and produce multi-gigabyte Slurm logs.
        # Structured rollback counters below preserve the information without
        # changing any simulation decision.
        cfg.simulation.log_level = "ERROR"
        engine = original_builder(cfg, task_assigner=task_assigner)
        if admission_mode == STATION_ADMISSION_DYNAMIC_ETA_V1:
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
        if admission_mode == STATION_ADMISSION_COMMITTED_FIFO_V2:
            fifo_probe = FifoWaitingAuditProbe(
                engine,
                trace_max_records=int(fifo_trace_max_records),
            )
            engine.on_tick_callbacks.append(fifo_probe.on_tick)
            holder["fifo_probe"] = fifo_probe
        return engine

    eval_module._build_engine = build
    try:
        yield holder
    finally:
        eval_module._build_engine = original_builder


def _policy_audit(
    spec: ArmSpec,
    metrics: Mapping[str, Any],
) -> dict[str, bool]:
    model_calls = int(metrics.get("model_assign_calls") or 0)
    s1_contexts = int(metrics.get("energy_conv_contexts") or 0)
    fallback_calls = int(metrics.get("fallback_greedy_calls") or 0)
    if spec.policy == "dynamic_j":
        return {
            "model_used": model_calls > 0,
            "s1_used": s1_contexts > 0,
            "no_greedy_fallback": fallback_calls == 0,
            "dynamic_batches_seen": int(
                metrics.get("dynamic_probe_batches") or 0
            )
            > 0,
            "dynamic_choices_reindexed": bool(
                metrics.get("dynamic_probe_choices_reindexed", False)
            ),
            "dynamic_head_loaded": bool(
                metrics.get("psi_dispatch_head_loaded", False)
            ),
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
        }
    return {
        "external_baseline_has_no_model_calls": model_calls == 0,
        "external_baseline_has_no_s1_contexts": s1_contexts == 0,
        "external_baseline_has_no_greedy_fallback": fallback_calls == 0,
    }


def _run_audit(
    spec: ArmSpec,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    station_audit: Mapping[str, Any],
    fifo_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    checks = {
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "manifest_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_count_matches": int(
            metrics.get("order_arrival_count", -1)
        )
        == int(manifest.get("total_orders", -2)),
        "station_admission_audit": bool(station_audit.get("passed")),
        "station_admission_mode": (
            station_audit.get("mode") == spec.admission_mode
        ),
        **_policy_audit(spec, metrics),
    }
    if spec.admission_mode == STATION_ADMISSION_COMMITTED_FIFO_V2:
        checks["fifo_waiting_lifecycle"] = bool(
            fifo_audit and fifo_audit.get("passed")
        )
        checks["fifo_mode"] = bool(
            fifo_audit
            and fifo_audit.get("mode")
            == STATION_ADMISSION_COMMITTED_FIFO_V2
        )
    else:
        checks["fifo_waiting_not_attached"] = fifo_audit is None
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
        "arm": meta.get("arm_key") == spec.key,
        "load": meta.get("load") == args.load,
        "seed": int(meta.get("seed", -1)) == int(args.seed),
        "ticks": int(meta.get("ticks", -1)) == int(args.ticks),
        "admission": meta.get("station_admission") == spec.admission_mode,
        "bundle": meta.get("frozen_bundle_sha256") == bundle_sha,
            "runtime": meta.get("runtime_code_sha256") == dict(runtime_hashes),
            "engine_log_level": meta.get("engine_log_level") == "ERROR",
        "policy": (
            (meta.get("policy_contract") or {}).get("fingerprint_sha256")
            == policy_fingerprint
        ),
        "manifest": (
            (payload.get("manifest") or {}).get("content_sha256")
            == manifest_sha
        ),
        "dynamic_config": (
            meta.get("dynamic_admission_config")
            == (
                dict(dynamic_config)
                if spec.admission_mode == STATION_ADMISSION_DYNAMIC_ETA_V1
                else None
            )
        ),
        "audit": bool((payload.get("audit") or {}).get("passed")),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"cannot resume incompatible output {path}: {failed}")
    print(f"[resume] {spec.key} {args.load} seed={args.seed}: {path}")
    return True


def _load_run_inputs(args: argparse.Namespace):
    bundle, protocol = _load_bundle(args.frozen_bundle)
    model_checkpoint = _artifact(bundle, "model_checkpoint")
    psi_head_checkpoint = _artifact(bundle, "psi_head_checkpoint")
    psi_scale_contract = _artifact(bundle, "psi_scale_contract")
    config_path = _artifact(bundle, f"config_{args.load}")
    return (
        bundle,
        protocol,
        model_checkpoint,
        psi_head_checkpoint,
        psi_scale_contract,
        config_path,
    )


def _prepare_manifest(args: argparse.Namespace) -> None:
    (
        _bundle,
        protocol,
        _model_checkpoint,
        _psi_head_checkpoint,
        _psi_scale_contract,
        config_path,
    ) = _load_run_inputs(args)
    manifest_path = _manifest_path(args.source_root, args.load, args.seed)
    generated = False
    if not manifest_path.is_file():
        from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

        source_output = _output_path(
            args.source_root,
            MANIFEST_SOURCE_KEY,
            args.load,
            args.seed,
        )
        if source_output.is_file():
            raise FileExistsError(
                f"manifest source output exists but manifest is missing: "
                f"{source_output}"
            )
        print(
            f"[run] manifest source Greedy physical-only {args.load} "
            f"seed={args.seed} ticks={args.ticks}"
        )
        with _engine_contract(
            STATION_ADMISSION_PHYSICAL_ONLY,
            dynamic_config=_dynamic_config(args),
            station_trace_stride=args.station_trace_stride,
            station_trace_max_records=0,
            fifo_trace_max_records=0,
        ) as holder:
            metrics = _run_one_assigner(
                str(config_path),
                GreedyTaskAssigner(),
                int(args.seed),
                int(args.ticks),
                trace_label="RestorationManifestSourceGreedyPhysicalOnly",
                save_order_manifest=str(manifest_path),
            )
        station_probe = holder.get("station_probe")
        if station_probe is None:
            raise RuntimeError("manifest source station probe was not attached")
        station_audit = station_probe.summary()
        if not station_audit.get("passed"):
            raise RuntimeError(
                f"manifest source station audit failed {args.load} "
                f"seed={args.seed}"
            )
        generated = True
    manifest = _validate_manifest(manifest_path)
    contract = _validate_historical_contract(
        manifest,
        v1_root=args.manifest_reference_root,
        v3_root=args.v3_reference_root,
        load=args.load,
        seed=args.seed,
        require_v3=args.require_v3_reference,
        skip=args.skip_historical_contract,
    )
    if generated:
        source_output = _output_path(
            args.source_root,
            MANIFEST_SOURCE_KEY,
            args.load,
            args.seed,
        )
        payload = {
            "schema_version": MANIFEST_SOURCE_SCHEMA_VERSION,
            "meta": {
                "protocol_sha256": protocol.get("protocol_sha256"),
                "load": args.load,
                "seed": int(args.seed),
                "ticks": int(args.ticks),
                "policy": "GreedyTaskAssigner",
                "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
                "formal_matrix_arm": False,
                "purpose": "manifest generation only; formal Greedy is replayed",
                "runtime_code_sha256": _runtime_hashes(),
                "engine_log_level": "ERROR",
            },
            "manifest": {
                "path": manifest_path.as_posix(),
                "file_sha256": sha256_file(manifest_path),
                "content_sha256": manifest.get("manifest_sha256"),
                "total_orders": manifest.get("total_orders"),
            },
            "historical_contract": contract,
            "station_admission_audit": station_audit,
            "metrics": metrics,
        }
        _atomic_json(source_output, payload)
        print(f"[done] manifest source {source_output}")
    print(json.dumps({
        "manifest": manifest_path.as_posix(),
        "content_sha256": manifest.get("manifest_sha256"),
        "total_orders": manifest.get("total_orders"),
        "historical_contract": contract,
    }, indent=2, ensure_ascii=False))


def _run_arm(args: argparse.Namespace) -> None:
    if args.arm is None:
        raise SystemExit("--arm is required for --mode arm")
    spec = ARM_SPECS[args.arm]
    (
        _bundle,
        protocol,
        model_checkpoint,
        psi_head_checkpoint,
        psi_scale_contract,
        config_path,
    ) = _load_run_inputs(args)
    manifest_path = _manifest_path(args.source_root, args.load, args.seed)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"prepare manifest before running arms: {manifest_path}"
        )
    manifest = _validate_manifest(manifest_path)
    historical_contract = _validate_historical_contract(
        manifest,
        v1_root=args.manifest_reference_root,
        v3_root=args.v3_reference_root,
        load=args.load,
        seed=args.seed,
        require_v3=args.require_v3_reference,
        skip=args.skip_historical_contract,
    )
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
        fifo_trace_max_records=args.fifo_trace_max_records,
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
    if station_probe is None:
        raise RuntimeError("station admission audit probe was not attached")
    station_audit = station_probe.summary()
    fifo_probe = holder.get("fifo_probe")
    fifo_audit = fifo_probe.summary() if fifo_probe is not None else None

    if spec.policy == "dynamic_j":
        metrics.update(assigner.dynamic_probe_metrics())
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
    if fifo_audit is not None:
        metrics.update({
            "waiting_assigned_agent_ticks": fifo_audit.get(
                "waiting_agent_ticks"
            ),
            "waiting_assigned_ratio": fifo_audit.get(
                "waiting_assigned_ratio"
            ),
            "waiting_promotions": fifo_audit.get("waiting_promotions"),
            "waiting_duration_p95_ticks": fifo_audit.get(
                "waiting_duration_p95_ticks"
            ),
            "unresolved_waiter_count_final": fifo_audit.get(
                "unresolved_waiter_count_final"
            ),
        })

    audit = _run_audit(
        spec, metrics, manifest, station_audit, fifo_audit
    )
    if not audit["passed"]:
        failed = [
            key for key, passed in audit["checks"].items() if not passed
        ]
        raise RuntimeError(
            f"restoration arm audit failed {spec.key} {args.load} "
            f"seed={args.seed}: {failed}"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": protocol.get("protocol_sha256"),
            "frozen_bundle": args.frozen_bundle.as_posix(),
            "frozen_bundle_sha256": bundle_sha,
            "arm_key": spec.key,
            "arm_label": spec.label,
            "policy_family": spec.policy,
            "policy_contract": policy_contract,
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "formal": not bool(args.development),
            "station_admission": spec.admission_mode,
            "dynamic_admission_config": (
                dynamic_config
                if spec.admission_mode == STATION_ADMISSION_DYNAMIC_ETA_V1
                else None
            ),
            "physical_only_semantics": (
                "reserve checks current physical occupancy; in-transit "
                "commitments have no count cap; check-in still requires a "
                "free physical slot"
                if spec.admission_mode == STATION_ADMISSION_PHYSICAL_ONLY
                else None
            ),
            "source_manifest_root": args.source_root.as_posix(),
            "runtime_code_sha256": runtime_hashes,
            "engine_log_level": "ERROR",
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "historical_manifest_contract": historical_contract,
        "audit": audit,
        "station_admission_audit": station_audit,
        "fifo_waiting_audit": fifo_audit,
        "metrics": metrics,
        "dynamic_j_trace": (
            assigner.dynamic_probe_trace_records
            if spec.policy == "dynamic_j"
            else []
        ),
    }
    _atomic_json(output, payload)
    print(f"[done] {output}")
    print(json.dumps({
        "completed_orders": metrics.get("completed_orders"),
        "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
        "stall_ratio_mean": metrics.get("stall_ratio_mean"),
        "max_committed_load": station_audit.get("max_committed_load"),
        "max_occupancy": station_audit.get("max_occupancy"),
        "over_capacity_grants": station_audit.get("over_capacity_grants"),
        "ticks_with_any_committed_over_capacity": station_audit.get(
            "ticks_with_any_committed_over_capacity"
        ),
    }, indent=2, ensure_ascii=False))


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [_number(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    return round(statistics.fmean(clean), 6) if clean else None


def _std(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [_number(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    if len(clean) == 1:
        return 0.0
    return round(statistics.stdev(clean), 6)


def _max_mapping(mapping: Mapping[str, Any] | None) -> int:
    if not mapping:
        return 0
    return max((int(value) for value in mapping.values()), default=0)


def _aggregate_arm(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    rollback_totals = {"entry_occupied": 0, "path_failure": 0, "other": 0}
    for payload in payloads:
        meta = payload.get("meta") or {}
        metrics = payload.get("metrics") or {}
        station = payload.get("station_admission_audit") or {}
        for key in rollback_totals:
            rollback_totals[key] += int(
                (station.get("rollbacks") or {}).get(key, 0)
            )
        rows.append({
            "seed": int(meta.get("seed")),
            "completed_orders": metrics.get("completed_orders"),
            "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
            "stall_ratio_mean": metrics.get("stall_ratio_mean"),
            "pending_order_count": metrics.get("pending_order_count"),
            "open_order_count": metrics.get("open_order_count"),
            "wall_time_s": metrics.get("wall_time_s"),
            "max_occupancy": _max_mapping(station.get("max_occupancy")),
            "max_committed_load": _max_mapping(
                station.get("max_committed_load")
            ),
            "max_in_transit_commitments": _max_mapping(
                station.get("max_in_transit_commitments")
            ),
            "over_capacity_grants": int(
                station.get("over_capacity_grants", 0)
            ),
            "committed_over_capacity_station_tick_count": int(
                station.get(
                    "committed_over_capacity_station_tick_count", 0
                )
            ),
            "ticks_with_any_committed_over_capacity": int(
                station.get("ticks_with_any_committed_over_capacity", 0)
            ),
            "capacity_rejections": int(
                station.get("capacity_rejections", 0)
            ),
            "physical_capacity_violations": int(
                station.get("physical_capacity_violation_count", 0)
            ),
            "token_mismatches": int(
                station.get("token_mismatch_count", 0)
            ),
            "manifest_sha256": (payload.get("manifest") or {}).get(
                "content_sha256"
            ),
            "path": (payload.get("_path") or ""),
        })
    rows.sort(key=lambda row: row["seed"])
    completed_mean = _mean(rows, "completed_orders")
    ticks = int((payloads[0].get("meta") or {}).get("ticks", 0)) if payloads else 0
    return {
        "run_count": len(rows),
        "completed_orders_mean": completed_mean,
        "completed_orders_std": _std(rows, "completed_orders"),
        "orders_per_1000_ticks_mean": (
            round(float(completed_mean) * 1000.0 / ticks, 6)
            if completed_mean is not None and ticks > 0
            else None
        ),
        "deadlock_ratio_mean": _mean(rows, "deadlock_ratio_mean"),
        "stall_ratio_mean": _mean(rows, "stall_ratio_mean"),
        "pending_order_count_mean": _mean(rows, "pending_order_count"),
        "open_order_count_mean": _mean(rows, "open_order_count"),
        "wall_time_s_sum": round(
            sum(_number(row.get("wall_time_s")) or 0.0 for row in rows), 6
        ),
        "max_occupancy": max(
            (row["max_occupancy"] for row in rows), default=0
        ),
        "max_committed_load": max(
            (row["max_committed_load"] for row in rows), default=0
        ),
        "max_in_transit_commitments": max(
            (row["max_in_transit_commitments"] for row in rows), default=0
        ),
        "over_capacity_grants_sum": sum(
            row["over_capacity_grants"] for row in rows
        ),
        "committed_over_capacity_station_ticks_sum": sum(
            row["committed_over_capacity_station_tick_count"] for row in rows
        ),
        "seeds_with_committed_over_capacity": sum(
            int(row["ticks_with_any_committed_over_capacity"] > 0)
            for row in rows
        ),
        "capacity_rejections_sum": sum(
            row["capacity_rejections"] for row in rows
        ),
        "rollbacks_sum": rollback_totals,
        "physical_capacity_violations_sum": sum(
            row["physical_capacity_violations"] for row in rows
        ),
        "token_mismatches_sum": sum(
            row["token_mismatches"] for row in rows
        ),
        "runs": rows,
    }


def _paired_comparison(
    name: str,
    left_key: str,
    right_key: str,
    payloads_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    left = {
        int((payload.get("meta") or {}).get("seed")): payload
        for payload in payloads_by_arm[left_key]
    }
    right = {
        int((payload.get("meta") or {}).get("seed")): payload
        for payload in payloads_by_arm[right_key]
    }
    rows = []
    wins = ties = losses = 0
    deadlock_wins = deadlock_ties = deadlock_losses = 0
    for seed in sorted(set(left) & set(right)):
        left_metrics = left[seed].get("metrics") or {}
        right_metrics = right[seed].get("metrics") or {}
        left_orders = int(left_metrics.get("completed_orders", 0))
        right_orders = int(right_metrics.get("completed_orders", 0))
        order_delta = left_orders - right_orders
        wins += int(order_delta > 0)
        ties += int(order_delta == 0)
        losses += int(order_delta < 0)
        left_deadlock = float(left_metrics.get("deadlock_ratio_mean", 0.0))
        right_deadlock = float(right_metrics.get("deadlock_ratio_mean", 0.0))
        deadlock_delta = left_deadlock - right_deadlock
        deadlock_wins += int(deadlock_delta < -1e-12)
        deadlock_ties += int(abs(deadlock_delta) <= 1e-12)
        deadlock_losses += int(deadlock_delta > 1e-12)
        rows.append({
            "seed": seed,
            "left_completed_orders": left_orders,
            "right_completed_orders": right_orders,
            "completed_orders_delta": order_delta,
            "deadlock_ratio_delta": round(deadlock_delta, 6),
            "stall_ratio_delta": round(
                float(left_metrics.get("stall_ratio_mean", 0.0))
                - float(right_metrics.get("stall_ratio_mean", 0.0)),
                6,
            ),
        })
    mean_delta = _mean(rows, "completed_orders_delta")
    right_mean = statistics.fmean(
        [float(row["right_completed_orders"]) for row in rows]
    ) if rows else 0.0
    return {
        "name": name,
        "direction": "left minus right",
        "left_arm": left_key,
        "right_arm": right_key,
        "paired_seed_count": len(rows),
        "completed_orders_delta_mean": mean_delta,
        "completed_orders_relative_delta_percent": (
            round(float(mean_delta) * 100.0 / right_mean, 6)
            if mean_delta is not None and right_mean
            else None
        ),
        "throughput_wins_ties_losses": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
        },
        "deadlock_ratio_delta_mean": _mean(rows, "deadlock_ratio_delta"),
        "deadlock_lower_wins_ties_losses": {
            "wins": deadlock_wins,
            "ties": deadlock_ties,
            "losses": deadlock_losses,
        },
        "stall_ratio_delta_mean": _mean(rows, "stall_ratio_delta"),
        "pairs": rows,
    }


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Station admission restoration summary",
        "",
        (
            "`physical_only_legacy` is the true restoration condition: "
            "physical occupancy remains bounded, but in-transit commitments "
            "do not consume a committed-capacity budget."
        ),
        "",
        "| Arm | Throughput | Deadlock | Stall | Max committed | Over-cap grants |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_KEYS:
        row = summary["arms"][arm]
        lines.append(
            f"| {ARM_SPECS[arm].label} | "
            f"{row.get('completed_orders_mean')} | "
            f"{row.get('deadlock_ratio_mean')} | "
            f"{row.get('stall_ratio_mean')} | "
            f"{row.get('max_committed_load')} | "
            f"{row.get('over_capacity_grants_sum')} |"
        )
    lines.extend([
        "",
        "## Paired comparisons",
        "",
        "| Comparison (left - right) | Order delta | Relative | W/T/L | Deadlock delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, row in summary["comparisons"].items():
        wtl = row["throughput_wins_ties_losses"]
        lines.append(
            f"| {name} | {row.get('completed_orders_delta_mean')} | "
            f"{row.get('completed_orders_relative_delta_percent')}% | "
            f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
            f"{row.get('deadlock_ratio_delta_mean')} |"
        )
    evidence = summary["restoration_evidence"]
    lines.extend([
        "",
        "## Restoration evidence",
        "",
        f"- Dynamic-J physical-only exceeded committed capacity in "
        f"{evidence['dynamic_physical_seeds_with_over_capacity']} seeds.",
        f"- Maximum Dynamic-J physical-only committed load: "
        f"{evidence['dynamic_physical_max_committed_load']}.",
        f"- Maximum Dynamic-J physical-only physical occupancy: "
        f"{evidence['dynamic_physical_max_occupancy']}.",
        f"- Physical capacity violations across all arms: "
        f"{evidence['physical_capacity_violations_all_arms']}.",
        "",
        "## Integrity",
        "",
        f"- Manifest pairing passed: {summary['integrity']['manifest_pairing_passed']}",
        f"- Policy fingerprints paired correctly: "
        f"{summary['integrity']['policy_pairing_passed']}",
        f"- All arm audits passed: {summary['integrity']['all_arm_audits_passed']}",
        "",
    ])
    return "\n".join(lines)


def _summarize(args: argparse.Namespace) -> None:
    payloads_by_arm: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in ARM_KEYS
    }
    missing = []
    for arm in ARM_KEYS:
        for seed in args.seeds:
            path = _output_path(args.output_root, arm, args.load, seed)
            if not path.is_file():
                missing.append(path.as_posix())
                continue
            payload = _read_json(path)
            payload["_path"] = path.as_posix()
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"wrong restoration result schema: {path}")
            if not bool((payload.get("audit") or {}).get("passed")):
                raise RuntimeError(f"failed arm audit in result: {path}")
            payloads_by_arm[arm].append(payload)
    if missing and not args.allow_incomplete:
        raise FileNotFoundError(
            "missing restoration results:\n" + "\n".join(missing)
        )

    manifest_checks = {}
    manifest_pairing_passed = True
    for seed in args.seeds:
        manifest_path = _manifest_path(args.source_root, args.load, seed)
        if not manifest_path.is_file():
            manifest_checks[str(seed)] = {"missing_source_manifest": True}
            manifest_pairing_passed = False
            continue
        manifest = _validate_manifest(manifest_path)
        contract = _validate_historical_contract(
            manifest,
            v1_root=args.manifest_reference_root,
            v3_root=args.v3_reference_root,
            load=args.load,
            seed=seed,
            require_v3=args.require_v3_reference,
            skip=args.skip_historical_contract,
        )
        arm_shas = {
            str((payload.get("manifest") or {}).get("content_sha256"))
            for arm in ARM_KEYS
            for payload in payloads_by_arm[arm]
            if int((payload.get("meta") or {}).get("seed", -1)) == seed
        }
        expected = str(manifest.get("manifest_sha256"))
        passed = arm_shas == {expected} if not args.allow_incomplete else (
            not arm_shas or arm_shas == {expected}
        )
        manifest_pairing_passed &= passed
        manifest_checks[str(seed)] = {
            "passed": passed,
            "source_content_sha256": expected,
            "source_total_orders": int(manifest.get("total_orders", 0)),
            "arm_content_sha256_values": sorted(arm_shas),
            "historical_contract": contract,
        }

    policy_checks = {}
    policy_pairing_passed = True
    policy_groups = {
        "dynamic_j": (
            "dynamic_j_physical_only_restored_v1",
            "dynamic_j_committed_v1",
            "dynamic_j_eta_overbooking_v3",
        ),
        "greedy": ("greedy_physical_only", "greedy_fifo_v2"),
        "hungarian": (
            "hungarian_physical_only",
            "hungarian_fifo_v2",
        ),
    }
    for policy, arms in policy_groups.items():
        fingerprints = {
            str(
                ((payload.get("meta") or {}).get("policy_contract") or {}).get(
                    "fingerprint_sha256"
                )
            )
            for arm in arms
            for payload in payloads_by_arm[arm]
        }
        passed = (
            len(fingerprints) == 1
            and "None" not in fingerprints
            and "" not in fingerprints
        )
        policy_pairing_passed &= passed
        policy_checks[policy] = {
            "passed": passed,
            "arms": list(arms),
            "fingerprints": sorted(fingerprints),
        }

    arms = {
        arm: _aggregate_arm(payloads_by_arm[arm]) for arm in ARM_KEYS
    }
    comparisons = {
        name: _paired_comparison(
            name, left, right, payloads_by_arm
        )
        for name, left, right in COMPARISONS
    }
    all_audits_passed = all(
        bool((payload.get("audit") or {}).get("passed"))
        for payloads in payloads_by_arm.values()
        for payload in payloads
    )
    physical_violations = sum(
        int(row.get("physical_capacity_violations_sum", 0))
        for row in arms.values()
    )
    dynamic_physical = arms["dynamic_j_physical_only_restored_v1"]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "meta": {
            "load": args.load,
            "seeds": [int(seed) for seed in args.seeds],
            "ticks": int(args.ticks),
            "source_root": args.source_root.as_posix(),
            "output_root": args.output_root.as_posix(),
            "runner": Path(__file__).as_posix(),
            "runner_sha256": sha256_file(Path(__file__)),
            "missing_results": missing,
        },
        "causal_contract": {
            "one_manifest_per_seed": True,
            "all_formal_arms_replay_manifest": True,
            "dynamic_j_three_arms_share_one_policy_fingerprint": True,
            "physical_only_is_true_restoration": True,
            "physical_only_definition": (
                "no in-transit committed-count cap; physical check-in remains "
                "bounded by station slots"
            ),
            "eta_v3_definition": (
                "cap=2C (14 for C=7) plus ETA weighted and release-window "
                "admission; it is overbooking, not unrestricted restoration"
            ),
        },
        "integrity": {
            "manifest_pairing_passed": bool(manifest_pairing_passed),
            "manifest_checks": manifest_checks,
            "policy_pairing_passed": bool(policy_pairing_passed),
            "policy_checks": policy_checks,
            "all_arm_audits_passed": bool(all_audits_passed),
        },
        "arms": arms,
        "comparisons": comparisons,
        "restoration_evidence": {
            "dynamic_physical_committed_cap_exercised": bool(
                dynamic_physical["seeds_with_committed_over_capacity"] > 0
            ),
            "dynamic_physical_seeds_with_over_capacity": int(
                dynamic_physical["seeds_with_committed_over_capacity"]
            ),
            "dynamic_physical_over_capacity_grants": int(
                dynamic_physical["over_capacity_grants_sum"]
            ),
            "dynamic_physical_over_capacity_station_ticks": int(
                dynamic_physical[
                    "committed_over_capacity_station_ticks_sum"
                ]
            ),
            "dynamic_physical_max_committed_load": int(
                dynamic_physical["max_committed_load"]
            ),
            "dynamic_physical_max_occupancy": int(
                dynamic_physical["max_occupancy"]
            ),
            "physical_capacity_violations_all_arms": int(
                physical_violations
            ),
        },
    }
    if not args.allow_incomplete:
        integrity = summary["integrity"]
        if not all((
            integrity["manifest_pairing_passed"],
            integrity["policy_pairing_passed"],
            integrity["all_arm_audits_passed"],
            physical_violations == 0,
        )):
            raise RuntimeError("restoration summary integrity checks failed")

    validation = args.output_root / "validation"
    json_path = validation / f"station_admission_restoration_{args.load}.json"
    markdown_path = validation / (
        f"station_admission_restoration_{args.load}.md"
    )
    _atomic_json(json_path, summary)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = _markdown_summary(summary)
    if markdown_path.is_file():
        if markdown_path.read_text(encoding="utf-8") != markdown:
            raise FileExistsError(
                f"refusing to overwrite changed summary: {markdown_path}"
            )
    else:
        markdown_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "summary_json": json_path.as_posix(),
        "summary_markdown": markdown_path.as_posix(),
        "integrity": summary["integrity"],
        "restoration_evidence": summary["restoration_evidence"],
    }, indent=2, ensure_ascii=False))


def _validate_args(args: argparse.Namespace) -> None:
    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive")
    if args.station_trace_stride <= 0:
        raise SystemExit("--station-trace-stride must be positive")
    if any(
        value < 0
        for value in (
            args.policy_trace_max_records,
            args.station_trace_max_records,
            args.fifo_trace_max_records,
        )
    ):
        raise SystemExit("trace record limits must be non-negative")
    if not args.development:
        if args.ticks != FORMAL_TICKS:
            raise SystemExit(
                f"formal restoration freezes --ticks={FORMAL_TICKS}"
            )
        seeds = args.seeds if args.mode == "summary" else [args.seed]
        invalid = [seed for seed in seeds if seed not in FORMAL_SEEDS]
        if invalid:
            raise SystemExit(
                f"formal seeds must be in {list(FORMAL_SEEDS)}: {invalid}"
            )
        if args.skip_historical_contract:
            raise SystemExit(
                "formal restoration cannot skip the historical manifest contract"
            )
    if args.mode in ("manifest", "arm") and args.seed is None:
        raise SystemExit("--seed is required for manifest and arm modes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("manifest", "arm", "summary"), required=True
    )
    parser.add_argument("--arm", choices=ARM_KEYS, default=None)
    parser.add_argument("--load", choices=LOADS, default="high")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS)
    )
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
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--policy-trace-max-records", type=int, default=0)
    parser.add_argument("--station-trace-stride", type=int, default=5)
    parser.add_argument("--station-trace-max-records", type=int, default=300)
    parser.add_argument("--fifo-trace-max-records", type=int, default=0)
    parser.add_argument("--max-committed-multiplier", type=float, default=2.0)
    parser.add_argument("--healthy-extra-ratio", type=float, default=1.0)
    parser.add_argument("--caution-extra-ratio", type=float, default=0.5)
    parser.add_argument("--brake-extra-ratio", type=float, default=0.0)
    parser.add_argument("--eta-near-ticks", type=int, default=8)
    parser.add_argument("--eta-mid-ticks", type=int, default=20)
    args = parser.parse_args()
    _validate_args(args)
    if args.mode == "manifest":
        _prepare_manifest(args)
    elif args.mode == "arm":
        _run_arm(args)
    else:
        _summarize(args)


if __name__ == "__main__":
    main()
