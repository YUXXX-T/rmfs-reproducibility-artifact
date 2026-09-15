"""Run Dynamic-J with the isolated committed-capacity station admission fix.

This development arm changes exactly one simulator contract: a station's
capacity includes both physical queue occupants and admitted robots travelling
to its entry.  Frozen Phase-C runners keep the historical physical-only mode.
The order manifest, Dynamic-J policy, S1 robot scorer, model checkpoint and
station exit/path-planning logic are otherwise reused unchanged.
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
from WorldModel.evaluation.run_phase_c_psi_dynamic_probe import (
    _artifact,
    _atomic_json,
    _load_bundle,
    _manifest_path,
    _read_json,
)
from WorldState.station_state import STATION_ADMISSION_COMMITTED_V1


SCHEMA_VERSION = "phase_c_psi_dynamic_committed_admission_v1"
ARM_KEY = "s1_psi_dynamic_committed_admission_v1"
ARM_LABEL = "PhaseCS1PsiDynamicCommittedAdmissionV1"
BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_SOURCE_ROOT = BASE_ROOT / "psi_dispatch_ablation_551_560_v1"
DEFAULT_LEGACY_DYNAMIC_ROOT = BASE_ROOT / "psi_dispatch_dynamic_probe_551_560_v1"
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "psi_dispatch_dynamic_admission_551_560_v1"


class StationAdmissionAuditProbe:
    """Audit the committed-capacity invariant and token lifecycle each tick."""

    def __init__(self, engine):
        self.engine = engine
        self.max_committed_load: dict[int, int] = {}
        self.max_occupancy: dict[int, int] = {}
        self.capacity_violations: list[dict[str, Any]] = []
        self.token_mismatches: list[dict[str, Any]] = []

    def on_tick(self, engine) -> None:
        world = engine.world
        for station_id, queue in world.station_state.stations.items():
            station_id = int(station_id)
            committed = int(queue.committed_load())
            occupancy = int(queue.occupancy())
            self.max_committed_load[station_id] = max(
                committed, self.max_committed_load.get(station_id, 0)
            )
            self.max_occupancy[station_id] = max(
                occupancy, self.max_occupancy.get(station_id, 0)
            )
            if committed > int(queue.capacity):
                self.capacity_violations.append({
                    "tick": int(world.tick),
                    "station_id": station_id,
                    "committed_load": committed,
                    "capacity": int(queue.capacity),
                    "committed_agent_ids": sorted(
                        int(x) for x in queue.committed_agent_ids()
                    ),
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

    def summary(self) -> dict[str, Any]:
        station_metrics = self.engine.world.station_state.admission_metrics()
        return {
            "mode": STATION_ADMISSION_COMMITTED_V1,
            "passed": (
                not self.capacity_violations
                and not self.token_mismatches
                and all(bool(row["invariant_holds"]) for row in station_metrics)
                and all(
                    row["mode"] == STATION_ADMISSION_COMMITTED_V1
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
            "capacity_violation_count": len(self.capacity_violations),
            "capacity_violations": self.capacity_violations[:100],
            "token_mismatch_count": len(self.token_mismatches),
            "token_mismatches": self.token_mismatches[:100],
            "station_metrics_final": station_metrics,
        }


def _output_path(root: Path, load: str, seed: int) -> Path:
    return root / "per_arm" / ARM_KEY / f"{load}_seed{seed}.json"


def _reference_metrics(
    source_root: Path,
    legacy_dynamic_root: Path,
    load: str,
    seed: int,
) -> dict[str, Any]:
    references = {
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
    }
    result: dict[str, Any] = {}
    keys = (
        "completed_orders",
        "completed_tasks",
        "deadlock_ratio_mean",
        "deadlock_ratio_max",
        "stall_ratio_mean",
        "pending_order_count",
        "open_order_count",
    )
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
        "committed_admission_invariant": bool(admission_audit.get("passed")),
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

    output_path = _output_path(args.output_root, args.load, args.seed)
    if output_path.is_file():
        existing = _read_json(output_path)
        meta = existing.get("meta") or {}
        if (
            existing.get("schema_version") != SCHEMA_VERSION
            or meta.get("station_admission") != STATION_ADMISSION_COMMITTED_V1
            or meta.get("load") != args.load
            or int(meta.get("seed", -1)) != int(args.seed)
            or int(meta.get("ticks", -1)) != int(args.ticks)
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

    def build_with_committed_admission(cfg, task_assigner=None):
        engine = original_builder(cfg, task_assigner=task_assigner)
        engine.world.station_state.set_admission_mode(
            STATION_ADMISSION_COMMITTED_V1
        )
        audit_probe = StationAdmissionAuditProbe(engine)
        engine.on_tick_callbacks.append(audit_probe.on_tick)
        holder["engine"] = engine
        holder["audit_probe"] = audit_probe
        return engine

    eval_module._build_engine = build_with_committed_admission
    try:
        print(
            f"[run] dynamic committed admission load={args.load} "
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
        raise RuntimeError("station admission audit probe was not attached")
    admission_audit = audit_probe.summary()
    metrics.update(assigner.dynamic_probe_metrics())
    manifest_payload = _read_json(manifest_path)
    audit = _audit(metrics, manifest_payload, admission_audit)
    if not audit["passed"]:
        failed = [key for key, passed in audit["checks"].items() if not passed]
        raise RuntimeError(
            f"dynamic committed-admission audit failed "
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
            "station_admission": STATION_ADMISSION_COMMITTED_V1,
            "only_simulator_change": (
                "station committed capacity includes admitted in-transit "
                "and physical-slot robots"
            ),
            "station_exit_logic_changed": False,
            "path_planner_logic_changed": False,
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
        "capacity_rejections": sum(
            int(row["rejected_capacity"])
            for row in admission_audit["station_metrics_final"]
        ),
        "max_committed_load": admission_audit["max_committed_load"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
