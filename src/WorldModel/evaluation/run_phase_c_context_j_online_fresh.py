"""Fresh-seed online comparison for the learned context-J head.

The protocol uses one Greedy-generated order manifest per (load, seed), then
replays that exact manifest for every arm under the same FIFO-V2 admission:

    greedy_manifest, hungarian, phasec, s1,
    learned_j_old (531--540 head), learned_j_new (571--590 head)

The learned arms differ only in the context-J checkpoint.  They use dynamic
``J -> S1 robot`` selection; the World Model, S1 configuration, admission
engine, order stream, and tick budget are shared.  This file is additive and
does not modify any existing result directory or deployed assigner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping

from Policies.TaskAssigner import (
    GreedyTaskAssigner,
    HungarianTaskAssigner,
    WorldModelTaskAssigner,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.context_j_learned_dynamic_assigner import (
    LearnedContextJDynamicAssigner,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    LOAD_CONFIGS,
    S1_CONFIG,
    sha256_file,
)
from WorldModel.evaluation.phase_c_s1_hungarian_protocol import PHASEC_CONFIG
from WorldModel.evaluation.run_phase_c_s1_fifo_pair import (
    _fifo_engine,
    _manifest_path,
    _validate_manifest,
)
from WorldState.station_state import STATION_ADMISSION_COMMITTED_FIFO_V2


SCHEMA_VERSION = "phase_c_context_j_online_fresh_v1"
PROTOCOL_SCHEMA_VERSION = "phase_c_context_j_online_fresh_protocol_v1"
BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "ctxj_online_591_600_v1"
SEEDS = tuple(range(591, 601))
LOADS = ("low", "mid", "high")
TICKS = 1500
ARMS = (
    "greedy_manifest",
    "hungarian",
    "phasec",
    "s1",
    "learned_j_old",
    "learned_j_new",
)
ARM_LABELS = {
    "greedy_manifest": "GreedyManifestSource",
    "hungarian": "HungarianFifoV2",
    "phasec": "PhaseCPureFifoV2",
    "s1": "PhaseCS1FifoV2",
    "learned_j_old": "PhaseCLearnedJOldFifoV2",
    "learned_j_new": "PhaseCLearnedJNewFifoV2",
}
MODEL_CHECKPOINT = BASE_ROOT / "model_round1_v1/best_regret_world_model.pt"
PHI_HEAD = BASE_ROOT / (
    "station_congestion_head_region_dev_511_520_v1/"
    "linear_head_v1/best_station_congestion_head.pt"
)
PHI_SCALE_CONTRACT = BASE_ROOT / (
    "station_congestion_head_region_dev_511_520_v1/"
    "station_congestion_scale_contract.json"
)
OLD_CONTEXT_HEAD = BASE_ROOT / (
    "context_dispatch_j_head_train_531_540_v1/best_context_j_head.pt"
)
NEW_CONTEXT_HEAD = BASE_ROOT / (
    "context_dispatch_j_head_train_571_590_v1/best_context_j_head.pt"
)
RUNNER_PATH = Path(__file__)
LEARNED_ASSIGNER_PATH = Path(
    "Policies/TaskAssigner/WorldModelTaskAssigner/"
    "context_j_learned_dynamic_assigner.py"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite changed output: {path}")
        return
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _output_path(root: Path, arm: str, load: str, seed: int) -> Path:
    return root / "per_arm" / arm / f"{load}_seed{seed}.json"


def _protocol_path(root: Path) -> Path:
    return root / "context_j_online_fresh_protocol.json"


def _make_protocol(root: Path) -> dict[str, Any]:
    required = {
        "model_checkpoint": MODEL_CHECKPOINT,
        "phi_head": PHI_HEAD,
        "phi_scale_contract": PHI_SCALE_CONTRACT,
        "old_context_head": OLD_CONTEXT_HEAD,
        "new_context_head": NEW_CONTEXT_HEAD,
        "runner": RUNNER_PATH,
        "learned_assigner": LEARNED_ASSIGNER_PATH,
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    configs = {}
    for load, path in LOAD_CONFIGS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        configs[load] = {"path": path.as_posix(), "sha256": sha256_file(path)}
    payload: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "purpose": (
            "fresh-seed online closed-loop comparison of old/new learned "
            "context-J heads with unchanged S1 robot scoring"
        ),
        "seeds": list(SEEDS),
        "loads": list(LOADS),
        "ticks": TICKS,
        "arms": list(ARMS),
        "manifest_source": "GreedyTaskAssigner",
        "exact_manifest_replay": True,
        "station_admission": STATION_ADMISSION_COMMITTED_FIFO_V2,
        "model_checkpoint": {
            "path": MODEL_CHECKPOINT.as_posix(),
            "sha256": sha256_file(MODEL_CHECKPOINT),
        },
        "station_phi": {
            "path": PHI_HEAD.as_posix(),
            "sha256": sha256_file(PHI_HEAD),
            "scale_contract": PHI_SCALE_CONTRACT.as_posix(),
            "scale_contract_sha256": sha256_file(PHI_SCALE_CONTRACT),
        },
        "context_heads": {
            "old": {
                "path": OLD_CONTEXT_HEAD.as_posix(),
                "sha256": sha256_file(OLD_CONTEXT_HEAD),
                "training_block": "531-540",
            },
            "new": {
                "path": NEW_CONTEXT_HEAD.as_posix(),
                "sha256": sha256_file(NEW_CONTEXT_HEAD),
                "training_block": "571-590",
            },
        },
        "load_configs": configs,
        "s1_config": dict(S1_CONFIG),
        "phasec_config": dict(PHASEC_CONFIG),
        "learned_j_contract": {
            "selection": "dynamic_next_context_then_unchanged_S1_robot",
            "feature_contract": (
                "station_region_latent + global_latent_demand + "
                "top5_nearest_action_global"
            ),
            "normalisation": "checkpoint_frozen_train_only_contract",
            "horizon": 10,
            "rollout_continuation_mode": "behavior",
            "no_assign_added": False,
            "hard_gate_added": False,
            "e_demand_modified": False,
            "virtual_update": "idle_candidate_features_only",
        },
        "code_hashes": {
            "runner": sha256_file(RUNNER_PATH),
            "learned_assigner": sha256_file(LEARNED_ASSIGNER_PATH),
        },
    }
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol": payload,
        "protocol_sha256": _canonical_sha(payload),
    }


def _load_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _protocol_path(root)
    protocol_bundle = _read_json(path)
    if protocol_bundle.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(f"unexpected fresh online protocol: {path}")
    protocol = protocol_bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("fresh online protocol lacks protocol payload")
    claimed = str(protocol_bundle.get("protocol_sha256", ""))
    if _canonical_sha(protocol) != claimed:
        raise ValueError("fresh online protocol hash mismatch")
    return dict(protocol_bundle), dict(protocol)


def _freeze(root: Path) -> None:
    expected = _make_protocol(root)
    path = _protocol_path(root)
    if path.is_file():
        existing = _read_json(path)
        if existing != expected:
            raise FileExistsError(f"existing fresh protocol differs: {path}")
        print(f"[resume] protocol {path}")
        return
    _atomic_json(path, expected)
    print(f"[freeze] protocol sha256={expected['protocol_sha256']}")
    print(f"[freeze] wrote {path}")


def _manifest_path_for(root: Path, load: str, seed: int) -> Path:
    return _manifest_path(root, load, seed)


def _manifest_audit(metrics: Mapping[str, Any], probe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifest_source_fifo_passed": bool(probe.get("passed")),
        "manifest_written": bool(metrics.get("order_arrival_manifest_path")),
        "fifo_mode": probe.get("mode") == STATION_ADMISSION_COMMITTED_FIFO_V2,
    }


def _prepare_manifest(root: Path, load: str, seed: int, ticks: int) -> None:
    path = _manifest_path_for(root, load, seed)
    if path.is_file():
        _validate_manifest(path)
        existing_output = _output_path(root, "greedy_manifest", load, seed)
        if existing_output.is_file():
            print(f"[resume] manifest/output {existing_output}")
            return
        print(f"[resume] manifest exists; rebuilding missing source output {path}")
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    path.parent.mkdir(parents=True, exist_ok=True)
    with _fifo_engine(trace_max_records=0) as holder:
        metrics = _run_one_assigner(
            str(LOAD_CONFIGS[load]),
            GreedyTaskAssigner(),
            int(seed),
            int(ticks),
            trace_label=ARM_LABELS["greedy_manifest"],
            save_order_manifest=str(path),
        )
    probe = holder.get("probe")
    if probe is None:
        raise RuntimeError("manifest source did not attach FIFO probe")
    probe_summary = probe.summary()
    _validate_manifest(path)
    audit = _manifest_audit(metrics, probe_summary)
    if not all(audit.values()):
        raise RuntimeError(f"manifest audit failed {load} seed={seed}: {audit}")
    bundle, protocol = _load_protocol(root)
    output = _output_path(root, "greedy_manifest", load, seed)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "arm": "greedy_manifest",
            "arm_label": ARM_LABELS["greedy_manifest"],
            "load": load,
            "seed": int(seed),
            "ticks": int(ticks),
            "protocol_sha256": bundle["protocol_sha256"],
        },
        "manifest": {
            "path": path.as_posix(),
            "file_sha256": sha256_file(path),
            "content_sha256": _read_json(path).get("manifest_sha256"),
        },
        "audit": {"passed": all(audit.values()), "checks": audit},
        "admission_audit": probe_summary,
        "metrics": metrics,
    }
    _atomic_json(output, payload)
    print(f"[done] manifest and greedy output {output}")


def _make_assigner(arm: str, protocol: Mapping[str, Any], trace_max: int):
    model = str(protocol["model_checkpoint"]["path"])
    common = {
        "checkpoint_path": model,
        "top_m": 10,
        "energy_conv_random_flip_seed": 0,
    }
    if arm == "hungarian":
        return HungarianTaskAssigner()
    if arm == "phasec":
        return WorldModelTaskAssigner(**common, **PHASEC_CONFIG)
    if arm == "s1":
        return WorldModelTaskAssigner(**common, **S1_CONFIG)
    if arm in ("learned_j_old", "learned_j_new"):
        head_key = "old" if arm.endswith("old") else "new"
        return LearnedContextJDynamicAssigner(
            context_j_checkpoint=str(protocol["context_heads"][head_key]["path"]),
            psi_head_checkpoint=str(protocol["station_phi"]["path"]),
            psi_scale_contract=str(protocol["station_phi"]["scale_contract"]),
            dynamic_trace_enabled=True,
            dynamic_trace_max_records=trace_max,
            **common,
            **S1_CONFIG,
        )
    raise ValueError(f"unsupported executable arm: {arm}")


def _base_audit(
    arm: str,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> dict[str, bool]:
    checks = {
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "manifest_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_count_matches": int(metrics.get("order_arrival_count", -1))
        == int(manifest.get("total_orders", -2)),
        "fifo_admission_invariant": bool(admission.get("passed")),
        "fifo_mode": admission.get("mode") == STATION_ADMISSION_COMMITTED_FIFO_V2,
    }
    if arm in {"hungarian", "phasec", "s1", "learned_j_old", "learned_j_new"}:
        checks.update({
            "model_or_reference_completed": (
                arm in {"hungarian"}
                or int(metrics.get("model_assign_calls", 0)) > 0
            ),
            "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
        })
    if arm in {"phasec", "s1", "learned_j_old", "learned_j_new"}:
        checks["native_no_assign_disabled"] = not bool(
            metrics.get("native_no_assign_enabled", False)
        )
    if arm == "s1":
        checks["s1_used"] = int(metrics.get("energy_conv_contexts", 0)) > 0
    if arm in {"learned_j_old", "learned_j_new"}:
        checks.update({
            "s1_used": int(metrics.get("energy_conv_contexts", 0)) > 0,
            "context_j_head_loaded": bool(metrics.get("context_j_head_loaded")),
            "context_j_evaluated": int(metrics.get("context_j_eval_calls", 0)) > 0,
            "dynamic_steps_seen": int(metrics.get("dynamic_probe_steps", 0)) > 0,
            "parent_s1_preserved": (
                metrics.get("psi_dispatch_robot_scorer")
                == "WorldModelTaskAssigner.select_robots_unmodified"
            ),
            "no_assign_added": not bool(metrics.get("context_j_no_assign_added", True)),
            "hard_gate_added": not bool(metrics.get("context_j_hard_gate_added", True)),
            "e_demand_modified": not bool(
                metrics.get("context_j_e_demand_modified", True)
            ),
        })
    return checks


def _run_arm(root: Path, arm: str, load: str, seed: int, ticks: int, trace_max: int) -> None:
    bundle, protocol = _load_protocol(root)
    if arm == "greedy_manifest":
        _prepare_manifest(root, load, seed, ticks)
        return
    manifest_path = _manifest_path_for(root, load, seed)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = _validate_manifest(manifest_path)
    output = _output_path(root, arm, load, seed)
    if output.is_file():
        existing = _read_json(output)
        if (
            existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("meta", {}).get("protocol_sha256")
            == bundle["protocol_sha256"]
            and existing.get("meta", {}).get("arm") == arm
            and existing.get("meta", {}).get("load") == load
            and int(existing.get("meta", {}).get("seed", -1)) == int(seed)
            and bool(existing.get("audit", {}).get("passed"))
        ):
            print(f"[resume] {output}")
            return
        raise FileExistsError(f"incompatible existing output: {output}")

    assigner = _make_assigner(arm, protocol, trace_max)
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    with _fifo_engine(trace_max) as holder:
        metrics = _run_one_assigner(
            str(LOAD_CONFIGS[load]),
            assigner,
            int(seed),
            int(ticks),
            trace_label=ARM_LABELS[arm],
            recorded_orders_path=str(manifest_path),
        )
    probe = holder.get("probe")
    if probe is None:
        raise RuntimeError("FIFO audit probe was not attached")
    admission = probe.summary()
    if hasattr(assigner, "dynamic_probe_metrics"):
        metrics.update(assigner.dynamic_probe_metrics())
    metrics.update({
        "station_admission_mode": STATION_ADMISSION_COMMITTED_FIFO_V2,
        "waiting_assigned_agent_ticks": admission.get("waiting_agent_ticks"),
        "waiting_assigned_ratio": admission.get("waiting_assigned_ratio"),
        "waiting_promotions": admission.get("waiting_promotions"),
        "waiting_duration_p95_ticks": admission.get("waiting_duration_p95_ticks"),
        "unresolved_waiter_count_final": admission.get("unresolved_waiter_count_final"),
    })
    checks = _base_audit(arm, metrics, manifest, admission)
    audit = {"passed": all(checks.values()), "checks": checks}
    if not audit["passed"]:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"{arm} audit failed {load} seed={seed}: {failed}")

    traces = []
    if hasattr(assigner, "dynamic_probe_trace_records"):
        traces = list(assigner.dynamic_probe_trace_records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "arm": arm,
            "arm_label": ARM_LABELS[arm],
            "load": load,
            "seed": int(seed),
            "ticks": int(ticks),
            "protocol_sha256": bundle["protocol_sha256"],
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "audit": audit,
        "admission_audit": admission,
        "metrics": metrics,
        "dynamic_trace": traces,
    }
    _atomic_json(output, payload)
    print(f"[done] {output}")


def _summarise(root: Path) -> None:
    bundle, protocol = _load_protocol(root)
    rows = []
    missing = []
    failed = []
    for arm in ARMS:
        for load in LOADS:
            for seed in SEEDS:
                path = _output_path(root, arm, load, seed)
                if not path.is_file():
                    missing.append(path.as_posix())
                    continue
                payload = _read_json(path)
                if not bool(payload.get("audit", {}).get("passed")):
                    failed.append(path.as_posix())
                    continue
                metrics = payload.get("metrics") or {}
                rows.append({
                    "arm": arm,
                    "load": load,
                    "seed": int(seed),
                    "completed_orders": metrics.get("completed_orders"),
                    "completed_tasks": metrics.get("completed_tasks"),
                    "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
                    "deadlock_ratio_max": metrics.get("deadlock_ratio_max"),
                    "stall_ratio_mean": metrics.get("stall_ratio_mean"),
                    "wall_time_s": metrics.get("wall_time_s"),
                })
    if missing or failed:
        raise RuntimeError(
            f"summary incomplete: missing={len(missing)} failed={len(failed)}"
        )

    def numeric(values):
        return [float(value) for value in values if value is not None]

    aggregate = {}
    for arm in ARMS:
        aggregate[arm] = {}
        for load in LOADS:
            selected = [row for row in rows if row["arm"] == arm and row["load"] == load]
            aggregate[arm][load] = {
                "runs": len(selected),
                "completed_orders_mean": statistics.mean(
                    numeric(row["completed_orders"] for row in selected)
                ),
                "completed_orders_std": statistics.pstdev(
                    numeric(row["completed_orders"] for row in selected)
                ),
                "deadlock_ratio_mean_mean": statistics.mean(
                    numeric(row["deadlock_ratio_mean"] for row in selected)
                ),
                "deadlock_ratio_max_mean": statistics.mean(
                    numeric(row["deadlock_ratio_max"] for row in selected)
                ),
                "stall_ratio_mean_mean": statistics.mean(
                    numeric(row["stall_ratio_mean"] for row in selected)
                ),
                "wall_time_s_mean": statistics.mean(
                    numeric(row["wall_time_s"] for row in selected)
                ),
            }
    report = {
        "schema_version": "phase_c_context_j_online_fresh_summary_v1",
        "protocol_sha256": bundle["protocol_sha256"],
        "expected_runs": len(ARMS) * len(LOADS) * len(SEEDS),
        "completed_runs": len(rows),
        "rows": rows,
        "aggregate": aggregate,
    }
    _atomic_json(root / "online_summary.json", report)
    manifest_lines = []
    for path in sorted(root.glob("per_arm/**/*.json")):
        manifest_lines.append(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        )
    manifest_lines.append(
        f"{sha256_file(root / 'online_summary.json')}  online_summary.json"
    )
    manifest_lines.append(
        f"{sha256_file(_protocol_path(root))}  {_protocol_path(root).name}"
    )
    (root / "results.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "completed_runs": len(rows),
        "expected_runs": report["expected_runs"],
        "output_root": root.as_posix(),
    }, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "manifest", "arm", "summarize"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--load", choices=LOADS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--trace-max-records", type=int, default=200)
    args = parser.parse_args()
    root = args.output_root
    if args.mode == "freeze":
        _freeze(root)
    elif args.mode == "manifest":
        if args.load is None or args.seed is None:
            raise SystemExit("manifest mode requires --load and --seed")
        _load_protocol(root)
        _prepare_manifest(root, args.load, args.seed, args.ticks)
    elif args.mode == "arm":
        if args.arm is None or args.arm == "greedy_manifest":
            raise SystemExit("arm mode requires an executable non-manifest --arm")
        if args.load is None or args.seed is None:
            raise SystemExit("arm mode requires --load and --seed")
        _run_arm(root, args.arm, args.load, args.seed, args.ticks, args.trace_max_records)
    else:
        _summarise(root)


if __name__ == "__main__":
    main()
