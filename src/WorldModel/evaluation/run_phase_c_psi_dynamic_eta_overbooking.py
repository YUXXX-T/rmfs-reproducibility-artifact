"""Run Dynamic-J with aggressive, station-aware ETA overbooking.

This isolated development arm removes the V1 constraint that all physical and
in-transit commitments must fit inside the seven physical station slots.  The
physical slot check remains unchanged.  New DELIVER commitments are admitted
with a rough Manhattan ETA, a weighted time-bucket budget, a 2C hard ceiling,
and a station-health brake for sustained entry/exit blockage or release drought.
"""

from __future__ import annotations

import argparse
import json
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
from WorldModel.evaluation.run_phase_c_psi_dynamic_admission import (
    _artifact,
    _atomic_json,
    _load_bundle,
    _manifest_path,
    _read_json,
)
from WorldState.station_state import STATION_ADMISSION_DYNAMIC_ETA_V1


SCHEMA_VERSION = "phase_c_psi_dynamic_eta_overbooking_v3"
ARM_KEY = "s1_psi_dynamic_eta_overbooking_v3"
ARM_LABEL = "PhaseCS1PsiDynamicEtaOverbookingV3"
BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_SOURCE_ROOT = BASE_ROOT / "psi_dispatch_ablation_551_560_v1"
DEFAULT_LEGACY_DYNAMIC_ROOT = BASE_ROOT / "psi_dispatch_dynamic_probe_551_560_h1500_v1"
DEFAULT_V1_ROOT = BASE_ROOT / "psi_dispatch_dynamic_admission_551_560_v1"
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "psi_dispatch_dynamic_eta_overbooking_551_560_v3"


class DynamicEtaAdmissionAuditProbe:
    """Capture physical safety, token lifecycle, and overbooking data flow."""

    def __init__(self, engine, trace_stride: int = 5, trace_max_records: int = 2000):
        self.engine = engine
        self.trace_stride = max(1, int(trace_stride))
        self.trace_max_records = max(0, int(trace_max_records))
        self.flow_trace: list[dict[str, Any]] = []
        self.flow_trace_dropped = 0
        self.max_committed_load: dict[int, int] = {}
        self.max_occupancy: dict[int, int] = {}
        self.max_weighted_mass: dict[int, float] = {}
        self.max_weight_limit: dict[int, float] = {}
        self.physical_capacity_violations: list[dict[str, Any]] = []
        self.hard_limit_violations: list[dict[str, Any]] = []
        self.token_mismatches: list[dict[str, Any]] = []

    def on_tick(self, engine) -> None:
        world = engine.world
        trace_stations = []
        for station_id, queue in world.station_state.stations.items():
            station_id = int(station_id)
            committed = int(queue.committed_load())
            occupancy = int(queue.occupancy())
            weighted = float(queue.dynamic_weighted_committed_mass(world.tick))
            weight_limit = float(queue.dynamic_weight_limit(world.tick))
            hard_limit = int(queue.dynamic_hard_limit())
            self.max_committed_load[station_id] = max(
                committed, self.max_committed_load.get(station_id, 0)
            )
            self.max_occupancy[station_id] = max(
                occupancy, self.max_occupancy.get(station_id, 0)
            )
            self.max_weighted_mass[station_id] = max(
                weighted, self.max_weighted_mass.get(station_id, 0.0)
            )
            self.max_weight_limit[station_id] = max(
                weight_limit, self.max_weight_limit.get(station_id, 0.0)
            )
            if occupancy > int(queue.capacity):
                self.physical_capacity_violations.append({
                    "tick": int(world.tick),
                    "station_id": station_id,
                    "occupancy": occupancy,
                    "capacity": int(queue.capacity),
                    "physical_agent_ids": sorted(queue.physical_agent_ids()),
                })
            if committed > hard_limit:
                self.hard_limit_violations.append({
                    "tick": int(world.tick),
                    "station_id": station_id,
                    "committed_load": committed,
                    "hard_limit": hard_limit,
                    "committed_agent_ids": sorted(queue.committed_agent_ids()),
                })

            physical = queue.physical_agent_ids()
            for agent_id in sorted(queue._assigned_agents):
                if agent_id in physical:
                    continue
                active = world.task_state.get_active_task_for_agent(agent_id)
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
                self.token_mismatches.append({
                    "tick": int(world.tick),
                    "station_id": station_id,
                    "agent_id": int(agent_id),
                    "active_task_type": active_type,
                    "active_station_id": active_station,
                })
            if int(world.tick) % self.trace_stride == 0:
                trace_stations.append({
                    "station_id": station_id,
                    "occupancy": occupancy,
                    "committed_load": committed,
                    "weighted_committed_mass": round(weighted, 6),
                    "weighted_limit": round(weight_limit, 6),
                    "hard_limit": hard_limit,
                    "health_state": queue.dynamic_health_state(world.tick),
                    "release_interval_estimate": round(
                        queue.dynamic_release_interval_estimate(), 6
                    ),
                    "near_due_arrivals": int(
                        queue.dynamic_due_arrival_count(
                            queue._dynamic_eta_near_ticks, world.tick
                        )
                    ),
                    "mid_due_arrivals": int(
                        queue.dynamic_due_arrival_count(
                            queue._dynamic_eta_mid_ticks, world.tick
                        )
                    ),
                    "near_window_limit": int(
                        queue.dynamic_window_limit(
                            queue._dynamic_eta_near_ticks, world.tick
                        )
                    ),
                    "mid_window_limit": int(
                        queue.dynamic_window_limit(
                            queue._dynamic_eta_mid_ticks, world.tick
                        )
                    ),
                    "eta_bucket_counts": queue.dynamic_eta_bucket_counts(
                        world.tick
                    ),
                    "entry_blocked_streak": int(
                        queue._dynamic_entry_blocked_streak
                    ),
                    "exit_blocked_streak": int(
                        queue._dynamic_exit_blocked_streak
                    ),
                    "full_streak": int(queue._dynamic_full_streak),
                    "release_drought": int(
                        queue._dynamic_release_drought(world.tick)
                    ),
                    "release_count": int(queue._dynamic_release_count),
                    "token_count": int(len(queue._dynamic_tokens)),
                })
        if trace_stations:
            row = {"tick": int(world.tick), "stations": trace_stations}
            if len(self.flow_trace) < self.trace_max_records:
                self.flow_trace.append(row)
            else:
                self.flow_trace_dropped += 1

    def summary(self) -> dict[str, Any]:
        station_metrics = self.engine.world.station_state.admission_metrics()
        dynamic_rows = [row.get("dynamic_eta") or {} for row in station_metrics]
        return {
            "mode": STATION_ADMISSION_DYNAMIC_ETA_V1,
            "passed": (
                not self.physical_capacity_violations
                and not self.hard_limit_violations
                and not self.token_mismatches
                and all(bool(row["invariant_holds"]) for row in station_metrics)
                and all(
                    row["mode"] == STATION_ADMISSION_DYNAMIC_ETA_V1
                    for row in station_metrics
                )
                and all(
                    int(row.get("physical_capacity_violation_count", 0)) == 0
                    for row in dynamic_rows
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
            "max_weighted_mass": {
                str(key): round(float(value), 6)
                for key, value in sorted(self.max_weighted_mass.items())
            },
            "max_weight_limit": {
                str(key): round(float(value), 6)
                for key, value in sorted(self.max_weight_limit.items())
            },
            "physical_capacity_violation_count": len(
                self.physical_capacity_violations
            ),
            "physical_capacity_violations": self.physical_capacity_violations[:100],
            "hard_limit_violation_count": len(self.hard_limit_violations),
            "hard_limit_violations": self.hard_limit_violations[:100],
            "token_mismatch_count": len(self.token_mismatches),
            "token_mismatches": self.token_mismatches[:100],
            "flow_trace_stride": int(self.trace_stride),
            "flow_trace_record_count": len(self.flow_trace),
            "flow_trace_dropped": int(self.flow_trace_dropped),
            "flow_trace": self.flow_trace,
            "station_metrics_final": station_metrics,
        }


def _output_path(root: Path, load: str, seed: int) -> Path:
    return root / "per_arm" / ARM_KEY / f"{load}_seed{seed}.json"


def _metrics_from_path(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _read_json(path)
    metrics = payload.get("metrics") or {}
    keys = (
        "completed_orders",
        "completed_tasks",
        "deadlock_ratio_mean",
        "deadlock_ratio_max",
        "stall_ratio_mean",
        "stall_ratio_max",
        "handoff_ratio_mean",
        "pending_order_count",
        "open_order_count",
        "wall_time_s",
    )
    return {"path": path.as_posix(), **{key: metrics.get(key) for key in keys}}


def _reference_metrics(
    source_root: Path,
    legacy_dynamic_root: Path,
    v1_root: Path,
    load: str,
    seed: int,
) -> dict[str, Any]:
    candidates = {
        "phase_c_s1": (
            source_root / "per_arm" / "s1_psi_shadow" / f"{load}_seed{seed}.json"
        ),
        "dynamic_j_physical_only": (
            legacy_dynamic_root
            / "per_arm"
            / "s1_psi_dynamic_probe"
            / f"{load}_seed{seed}.json"
        ),
        "dynamic_j_committed_v1": (
            v1_root
            / "per_arm"
            / "s1_psi_dynamic_committed_admission_v1"
            / f"{load}_seed{seed}.json"
        ),
    }
    result = {}
    for name, path in candidates.items():
        row = _metrics_from_path(path)
        if row is not None:
            result[name] = row
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
        "dynamic_eta_admission_safe": bool(admission_audit.get("passed")),
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
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_V1_ROOT)
    parser.add_argument("--frozen-bundle", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--trace-max-records", type=int, default=500)
    parser.add_argument("--station-trace-stride", type=int, default=5)
    parser.add_argument("--station-trace-max-records", type=int, default=2000)
    parser.add_argument("--max-committed-multiplier", type=float, default=2.0)
    parser.add_argument("--healthy-extra-ratio", type=float, default=1.0)
    parser.add_argument("--caution-extra-ratio", type=float, default=0.5)
    parser.add_argument("--brake-extra-ratio", type=float, default=0.0)
    parser.add_argument("--eta-near-ticks", type=int, default=8)
    parser.add_argument("--eta-mid-ticks", type=int, default=20)
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

    dynamic_config = {
        "max_committed_multiplier": float(args.max_committed_multiplier),
        "healthy_extra_ratio": float(args.healthy_extra_ratio),
        "caution_extra_ratio": float(args.caution_extra_ratio),
        "brake_extra_ratio": float(args.brake_extra_ratio),
        "eta_near_ticks": int(args.eta_near_ticks),
        "eta_mid_ticks": int(args.eta_mid_ticks),
    }
    output_path = _output_path(args.output_root, args.load, args.seed)
    if output_path.is_file():
        existing = _read_json(output_path)
        meta = existing.get("meta") or {}
        if (
            existing.get("schema_version") != SCHEMA_VERSION
            or meta.get("station_admission") != STATION_ADMISSION_DYNAMIC_ETA_V1
            or meta.get("dynamic_admission_config") != dynamic_config
            or meta.get("load") != args.load
            or int(meta.get("seed", -1)) != int(args.seed)
            or int(meta.get("ticks", -1)) != int(args.ticks)
        ):
            raise FileExistsError(f"incompatible existing output: {output_path}")
        print(f"[resume] {output_path}")
        return

    assigner = DynamicPsiDispatchProbeAssigner(
        psi_head_checkpoint=str(psi_head_checkpoint),
        psi_scale_contract=str(psi_scale_contract),
        dynamic_trace_enabled=True,
        dynamic_trace_max_records=int(args.trace_max_records),
        checkpoint_path=str(model_checkpoint),
        top_m=TOP_M,
        energy_conv_random_flip_seed=0,
        **S1_CONFIG,
    )

    import WorldModel.evaluation.evaluate_online_v6 as eval_module

    original_builder = eval_module._build_engine
    holder: dict[str, Any] = {}

    def build_with_dynamic_eta_admission(cfg, task_assigner=None):
        engine = original_builder(cfg, task_assigner=task_assigner)
        engine.world.station_state.configure_dynamic_admission(
            service_ticks=int(cfg.simulation.station_process_duration),
            **dynamic_config,
        )
        engine.world.station_state.set_admission_mode(
            STATION_ADMISSION_DYNAMIC_ETA_V1
        )
        audit_probe = DynamicEtaAdmissionAuditProbe(
            engine,
            trace_stride=int(args.station_trace_stride),
            trace_max_records=int(args.station_trace_max_records),
        )
        engine.on_tick_callbacks.append(audit_probe.on_tick)
        holder["engine"] = engine
        holder["audit_probe"] = audit_probe
        return engine

    eval_module._build_engine = build_with_dynamic_eta_admission
    try:
        print(
            f"[run] dynamic ETA overbooking load={args.load} "
            f"seed={args.seed} ticks={args.ticks} config={dynamic_config}"
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
        raise RuntimeError("dynamic ETA admission audit probe was not attached")
    admission_audit = audit_probe.summary()
    metrics.update(assigner.dynamic_probe_metrics())
    manifest_payload = _read_json(manifest_path)
    audit = _audit(metrics, manifest_payload, admission_audit)
    if not audit["passed"]:
        failed = [key for key, passed in audit["checks"].items() if not passed]
        raise RuntimeError(
            f"dynamic ETA admission audit failed "
            f"{args.load} seed={args.seed}: {failed}"
        )

    runner_path = Path(__file__)
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
            "station_admission": STATION_ADMISSION_DYNAMIC_ETA_V1,
            "dynamic_admission_config": dynamic_config,
            "rough_eta": "Manhattan distance from robot position to station entry",
            "physical_capacity_contract": (
                "check_in_from_entry still requires a free physical station slot"
            ),
            "existing_tokens_revoked_on_health_drop": False,
            "policy_logic_changed_from_dynamic_probe": False,
            "model_checkpoint": model_checkpoint.as_posix(),
            "psi_head_checkpoint": psi_head_checkpoint.as_posix(),
            "psi_scale_contract": psi_scale_contract.as_posix(),
            "source_manifest_root": args.source_root.as_posix(),
            "runner_code": runner_path.as_posix(),
            "runner_code_sha256": sha256_file(runner_path),
            "station_state_code_sha256": sha256_file(
                Path("WorldState/station_state.py")
            ),
            "engine_code_sha256": sha256_file(Path("Engine/simulation_engine.py")),
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
            args.v1_root,
            args.load,
            args.seed,
        ),
        "audit": audit,
        "station_admission_audit": admission_audit,
        "metrics": metrics,
        "dynamic_probe_trace": assigner.dynamic_probe_trace_records,
    }
    _atomic_json(output_path, payload)
    station_rows = admission_audit["station_metrics_final"]
    print(f"[done] {output_path}")
    print(json.dumps({
        "completed_orders": metrics.get("completed_orders"),
        "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
        "stall_ratio_mean": metrics.get("stall_ratio_mean"),
        "capacity_rejections": sum(
            int(row["rejected_capacity"]) for row in station_rows
        ),
        "overbooking_grants": sum(
            int((row.get("dynamic_eta") or {}).get("overbooking_grants", 0))
            for row in station_rows
        ),
        "max_committed_load": admission_audit["max_committed_load"],
        "max_occupancy": admission_audit["max_occupancy"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
