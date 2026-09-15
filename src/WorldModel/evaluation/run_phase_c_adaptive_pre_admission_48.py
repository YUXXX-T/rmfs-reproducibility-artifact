"""48-robot Adaptive Pre-Admission supplement on exact seeds 591--600.

The completed ``ctxj_online_591_600_v1`` block is an immutable reference.
This runner replays its exact order manifests and runs only one new arm::

    context-conditioned adaptive pre-admission
        -> learned dynamic J (571--590 head)
        -> unchanged certified S1 robot scorer
        -> unchanged FIFO-V2 physical fallback

No baseline arm, order stream, World Model, station head, context-J head, S1
parameter, or simulator admission rule is regenerated or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from Policies.TaskAssigner.WorldModelTaskAssigner.adaptive_pre_admission_assigner import (
    AdaptivePreAdmissionLearnedContextJAssigner,
)
from WorldModel.evaluation import run_phase_c_context_j_online_fresh as reference
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    S1_CONFIG,
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_s1_fifo_pair import (
    _fifo_engine,
    _validate_manifest,
)
from WorldState.station_state import STATION_ADMISSION_COMMITTED_FIFO_V2


SCHEMA_VERSION = "phase_c_adaptive_pre_admission_48_run_v1"
PROTOCOL_SCHEMA_VERSION = "phase_c_adaptive_pre_admission_48_protocol_v1"
SUMMARY_SCHEMA_VERSION = "phase_c_adaptive_pre_admission_48_summary_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_REFERENCE_ROOT = BASE_ROOT / "ctxj_online_591_600_v1"
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "ctxj_adaptive_pre_admission_591_600_v1"

SEEDS = reference.SEEDS
LOADS = reference.LOADS
TICKS = reference.TICKS
REFERENCE_ARMS = reference.ARMS
ARM = "learned_j_adaptive_pre_admission"
ALL_ARMS = REFERENCE_ARMS + (ARM,)
ARM_LABEL = "PhaseCLearnedJAdaptivePreAdmission48FifoV2"

MODEL_CHECKPOINT = reference.MODEL_CHECKPOINT
PHI_HEAD = reference.PHI_HEAD
PHI_SCALE_CONTRACT = reference.PHI_SCALE_CONTRACT
CONTEXT_HEAD = reference.NEW_CONTEXT_HEAD
RUNNER_PATH = Path(
    "WorldModel/evaluation/run_phase_c_adaptive_pre_admission_48.py"
)
REFERENCE_RUNNER_PATH = Path(
    "WorldModel/evaluation/run_phase_c_context_j_online_fresh.py"
)
ASSIGNER_PATH = Path(
    "Policies/TaskAssigner/WorldModelTaskAssigner/"
    "adaptive_pre_admission_assigner.py"
)
LEARNED_J_ASSIGNER_PATH = Path(
    "Policies/TaskAssigner/WorldModelTaskAssigner/"
    "context_j_learned_dynamic_assigner.py"
)
CORE_PATH = Path("WorldModel/core/adaptive_pre_admission.py")

SUMMARY_METRICS = (
    "completion_fraction",
    "completed_orders",
    "completed_tasks",
    "avg_task_duration",
    "avg_excess_delay",
    "open_order_count",
    "pending_order_count",
    "station_pressure",
    "bottleneck_CVaR",
    "stall_ratio_mean",
    "stall_ratio_max",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "congestion_events",
    "severe_events",
    "waiting_assigned_agent_ticks",
    "waiting_assigned_ratio",
    "waiting_promotions",
    "waiting_duration_p95_ticks",
    "unresolved_waiter_count_final",
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


def _protocol_path(root: Path) -> Path:
    return root / "adaptive_pre_admission_protocol.json"


def _frozen_inputs_path(root: Path) -> Path:
    return root / "frozen_inputs.sha256"


def _output_path(root: Path, load: str, seed: int) -> Path:
    return root / "per_arm" / ARM / f"{load}_seed{seed}.json"


def _reference_protocol_path(root: Path) -> Path:
    return root / "context_j_online_fresh_protocol.json"


def _reference_summary_path(root: Path) -> Path:
    return root / "online_summary.json"


def _reference_results_path(root: Path) -> Path:
    return root / "results.sha256"


def _validate_reference(reference_root: Path) -> tuple[dict, dict, dict]:
    bundle, protocol = reference._load_protocol(reference_root)
    summary_path = _reference_summary_path(reference_root)
    results_path = _reference_results_path(reference_root)
    if not summary_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(
            "reference requires online_summary.json and results.sha256"
        )
    summary = _read_json(summary_path)
    rows = summary.get("rows")
    expected = len(REFERENCE_ARMS) * len(LOADS) * len(SEEDS)
    if (
        summary.get("schema_version")
        != "phase_c_context_j_online_fresh_summary_v1"
        or not isinstance(rows, list)
        or len(rows) != expected
        or int(summary.get("completed_runs", -1)) != expected
    ):
        raise ValueError("591--600 reference summary is incomplete")
    observed = {str(row.get("arm")) for row in rows}
    if observed != set(REFERENCE_ARMS):
        raise ValueError(f"unexpected reference arms: {sorted(observed)}")
    return bundle, protocol, summary


def _manifest_entry(reference_root: Path, load: str, seed: int) -> dict[str, Any]:
    path = reference._manifest_path_for(reference_root, load, seed)
    manifest = _validate_manifest(path)
    return {
        "path": path.as_posix(),
        "file_sha256": sha256_file(path),
        "content_sha256": manifest.get("manifest_sha256"),
        "total_orders": int(manifest.get("total_orders", -1)),
    }


def _make_protocol(reference_root: Path) -> dict[str, Any]:
    reference_bundle, reference_protocol, reference_summary = (
        _validate_reference(reference_root)
    )
    required = {
        "model_checkpoint": MODEL_CHECKPOINT,
        "station_phi": PHI_HEAD,
        "station_phi_scale_contract": PHI_SCALE_CONTRACT,
        "context_j_head": CONTEXT_HEAD,
        "runner": RUNNER_PATH,
        "reference_runner": REFERENCE_RUNNER_PATH,
        "adaptive_assigner": ASSIGNER_PATH,
        "learned_j_assigner": LEARNED_J_ASSIGNER_PATH,
        "adaptive_core": CORE_PATH,
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    manifests = {
        load: {
            str(seed): _manifest_entry(reference_root, load, seed)
            for seed in SEEDS
        }
        for load in LOADS
    }
    payload: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "purpose": (
            "48-robot controlled test of context-conditioned adaptive "
            "pre-admission on the exact 591-600 learned-J online block"
        ),
        "arm": ARM,
        "reference_arms": list(REFERENCE_ARMS),
        "seeds": list(SEEDS),
        "loads": list(LOADS),
        "ticks": TICKS,
        "num_robots": 48,
        "exact_manifest_replay": True,
        "station_admission": STATION_ADMISSION_COMMITTED_FIFO_V2,
        "reference": {
            "root": reference_root.as_posix(),
            "protocol_path": _reference_protocol_path(reference_root).as_posix(),
            "protocol_file_sha256": sha256_file(
                _reference_protocol_path(reference_root)
            ),
            "protocol_sha256": reference_bundle["protocol_sha256"],
            "summary_path": _reference_summary_path(reference_root).as_posix(),
            "summary_sha256": sha256_file(
                _reference_summary_path(reference_root)
            ),
            "results_path": _reference_results_path(reference_root).as_posix(),
            "results_sha256": sha256_file(
                _reference_results_path(reference_root)
            ),
            "completed_runs": reference_summary["completed_runs"],
        },
        "model_checkpoint": dict(reference_protocol["model_checkpoint"]),
        "station_phi": dict(reference_protocol["station_phi"]),
        "context_j_head": dict(reference_protocol["context_heads"]["new"]),
        "load_configs": dict(reference_protocol["load_configs"]),
        "s1_config": dict(S1_CONFIG),
        "adaptive_pre_admission_contract": {
            "decision_point": "before_PICK_task_materialisation",
            "selection": (
                "admissible contexts -> learned dynamic J -> unchanged S1"
            ),
            "context_conditioned": True,
            "defer_one_station_then_continue_other_contexts": True,
            "all_contexts_deferred_ends_current_assignment_batch": True,
            "pipeline_mass": {
                "physical": 1.0,
                "committed_in_transit": 1.0,
                "moving_to_pod": 1.0,
                "waiting_assigned": 1.0,
            },
            "base_headroom": (
                "(1-traffic)*(1-local_waiting_ratio)*"
                "(1-global_waiting_ratio)"
            ),
            "defer_credit": (
                "clip(sum eligible rejected ticks/static_free_flow_time,0,1)"
            ),
            "effective_pipeline_limit": (
                "capacity*(1+max(base_headroom,defer_credit))"
            ),
            "execute_predicate": (
                "pipeline_mass_after_execute <= effective_pipeline_limit"
            ),
            "static_free_flow_role": "defer_debt_clock_only_not_arrival_ETA",
            "service_channel_role": "learned_J_context_ordering_only",
            "traffic_channel_role": "adaptive_prefetch_headroom",
            "virtual_pipeline_update_after_each_execute": True,
            "exact_eta_used": False,
            "new_trainable_parameters": 0,
            "fifo_v2_preserved_as_physical_safety_fallback": True,
        },
        "manifests": manifests,
        "code_hashes": {
            name: {"path": path.as_posix(), "sha256": sha256_file(path)}
            for name, path in required.items()
            if name not in {
                "model_checkpoint",
                "station_phi",
                "station_phi_scale_contract",
                "context_j_head",
            }
        },
    }
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol": payload,
        "protocol_sha256": _canonical_sha(payload),
    }


def _write_frozen_inputs(root: Path, bundle: Mapping[str, Any]) -> None:
    protocol = bundle["protocol"]
    lines = [
        f"{sha256_file(_protocol_path(root))}  {_protocol_path(root).name}",
        (
            f"{protocol['reference']['protocol_file_sha256']}  "
            f"{protocol['reference']['protocol_path']}"
        ),
        (
            f"{protocol['reference']['summary_sha256']}  "
            f"{protocol['reference']['summary_path']}"
        ),
        (
            f"{protocol['reference']['results_sha256']}  "
            f"{protocol['reference']['results_path']}"
        ),
        (
            f"{protocol['model_checkpoint']['sha256']}  "
            f"{protocol['model_checkpoint']['path']}"
        ),
        (
            f"{protocol['station_phi']['sha256']}  "
            f"{protocol['station_phi']['path']}"
        ),
        (
            f"{protocol['station_phi']['scale_contract_sha256']}  "
            f"{protocol['station_phi']['scale_contract']}"
        ),
        (
            f"{protocol['context_j_head']['sha256']}  "
            f"{protocol['context_j_head']['path']}"
        ),
    ]
    for entry in protocol["code_hashes"].values():
        lines.append(f"{entry['sha256']}  {entry['path']}")
    for load in LOADS:
        config = protocol["load_configs"][load]
        lines.append(f"{config['sha256']}  {config['path']}")
        for seed in SEEDS:
            manifest = protocol["manifests"][load][str(seed)]
            lines.append(f"{manifest['file_sha256']}  {manifest['path']}")
    _frozen_inputs_path(root).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _freeze(root: Path, reference_root: Path) -> None:
    expected = _make_protocol(reference_root)
    path = _protocol_path(root)
    if path.is_file():
        existing = _read_json(path)
        if existing != expected:
            raise FileExistsError(f"existing adaptive protocol differs: {path}")
        print(f"[resume] protocol {path}")
    else:
        _atomic_json(path, expected)
        print(f"[freeze] protocol sha256={expected['protocol_sha256']}")
        print(f"[freeze] wrote {path}")
    _write_frozen_inputs(root, expected)
    print(f"[freeze] wrote {_frozen_inputs_path(root)}")


def _load_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = _read_json(_protocol_path(root))
    if bundle.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unexpected adaptive pre-admission protocol schema")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("adaptive protocol lacks payload")
    if _canonical_sha(protocol) != bundle.get("protocol_sha256"):
        raise ValueError("adaptive protocol hash mismatch")

    checks = [
        (
            protocol["reference"]["protocol_path"],
            protocol["reference"]["protocol_file_sha256"],
            "reference protocol",
        ),
        (
            protocol["reference"]["summary_path"],
            protocol["reference"]["summary_sha256"],
            "reference summary",
        ),
        (
            protocol["reference"]["results_path"],
            protocol["reference"]["results_sha256"],
            "reference results manifest",
        ),
        (
            protocol["model_checkpoint"]["path"],
            protocol["model_checkpoint"]["sha256"],
            "model checkpoint",
        ),
        (
            protocol["station_phi"]["path"],
            protocol["station_phi"]["sha256"],
            "station phi",
        ),
        (
            protocol["station_phi"]["scale_contract"],
            protocol["station_phi"]["scale_contract_sha256"],
            "station phi scale contract",
        ),
        (
            protocol["context_j_head"]["path"],
            protocol["context_j_head"]["sha256"],
            "context J head",
        ),
    ]
    for file_path, expected, label in checks:
        if sha256_file(file_path) != expected:
            raise ValueError(f"{label} changed after freeze")
    for name, entry in protocol["code_hashes"].items():
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise ValueError(f"code input changed after freeze: {name}")
    for load in LOADS:
        config = protocol["load_configs"][load]
        if sha256_file(config["path"]) != config["sha256"]:
            raise ValueError(f"load config changed: {load}")
        for seed in SEEDS:
            entry = protocol["manifests"][load][str(seed)]
            if sha256_file(entry["path"]) != entry["file_sha256"]:
                raise ValueError(f"manifest changed: {load} seed={seed}")
    return dict(bundle), dict(protocol)


def _make_assigner(protocol: Mapping[str, Any], trace_max: int):
    return AdaptivePreAdmissionLearnedContextJAssigner(
        context_j_checkpoint=str(protocol["context_j_head"]["path"]),
        psi_head_checkpoint=str(protocol["station_phi"]["path"]),
        psi_scale_contract=str(protocol["station_phi"]["scale_contract"]),
        dynamic_trace_enabled=True,
        dynamic_trace_max_records=int(trace_max),
        checkpoint_path=str(protocol["model_checkpoint"]["path"]),
        top_m=10,
        energy_conv_random_flip_seed=0,
        **S1_CONFIG,
    )


def _arm_audit(
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    admission: Mapping[str, Any],
    ticks: int,
) -> dict[str, bool]:
    return {
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "manifest_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_count_matches": int(metrics.get("order_arrival_count", -1))
        == int(manifest.get("total_orders", -2)),
        "fifo_admission_invariant": bool(admission.get("passed")),
        "fifo_mode": admission.get("mode")
        == STATION_ADMISSION_COMMITTED_FIFO_V2,
        "num_agents_is_48": int(metrics.get("num_agents", -1)) == 48,
        "ticks_match": int(metrics.get("ticks", -1)) == int(ticks),
        "world_model_used": int(metrics.get("model_assign_calls", 0)) > 0,
        "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
        "native_no_assign_disabled": not bool(
            metrics.get("native_no_assign_enabled", False)
        ),
        "s1_used": int(metrics.get("energy_conv_contexts", 0)) > 0,
        "context_j_head_loaded": bool(metrics.get("context_j_head_loaded")),
        "context_j_evaluated": int(metrics.get("context_j_eval_calls", 0)) > 0,
        "dynamic_steps_seen": int(metrics.get("dynamic_probe_steps", 0)) > 0,
        "parent_s1_preserved": (
            metrics.get("psi_dispatch_robot_scorer")
            == "WorldModelTaskAssigner.select_robots_unmodified"
        ),
        "adaptive_pre_admission_enabled": bool(
            metrics.get("adaptive_pre_admission_enabled", False)
        ),
        "context_defer_added": bool(metrics.get("context_defer_added", False)),
        "context_conditioned": bool(
            metrics.get("pre_admission_context_conditioned", False)
        ),
        "continues_other_contexts": bool(
            metrics.get("pre_admission_continue_other_contexts", False)
        ),
        "adaptive_gate_declared": bool(
            metrics.get("context_j_hard_gate_added", False)
        ),
        "pre_admission_evaluated": int(
            metrics.get("pre_admission_context_evaluations", 0)
        ) > 0,
        "no_exact_eta": not bool(
            metrics.get("pre_admission_exact_eta_used", True)
        ),
        "e_demand_unchanged": not bool(
            metrics.get("context_j_e_demand_modified", True)
        ),
    }


def _run_arm(
    root: Path,
    load: str,
    seed: int,
    ticks: int,
    trace_max: int,
) -> None:
    bundle, protocol = _load_protocol(root)
    if int(ticks) != int(protocol["ticks"]):
        raise ValueError("tick budget differs from frozen protocol")
    output = _output_path(root, load, seed)
    if output.is_file():
        existing = _read_json(output)
        if (
            existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("meta", {}).get("protocol_sha256")
            == bundle["protocol_sha256"]
            and existing.get("meta", {}).get("load") == load
            and int(existing.get("meta", {}).get("seed", -1)) == int(seed)
            and bool(existing.get("audit", {}).get("passed"))
        ):
            print(f"[resume] {output}")
            return
        raise FileExistsError(f"incompatible existing output: {output}")

    manifest_entry = protocol["manifests"][load][str(seed)]
    manifest_path = Path(manifest_entry["path"])
    manifest = _validate_manifest(manifest_path)
    assigner = _make_assigner(protocol, trace_max)
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    config_path = str(protocol["load_configs"][load]["path"])
    with _fifo_engine(trace_max_records=0) as holder:
        metrics = _run_one_assigner(
            config_path,
            assigner,
            int(seed),
            int(ticks),
            trace_label=ARM_LABEL,
            recorded_orders_path=str(manifest_path),
        )
    probe = holder.get("probe")
    if probe is None:
        raise RuntimeError("FIFO audit probe was not attached")
    admission = probe.summary()
    metrics.update(assigner.dynamic_probe_metrics())
    metrics.update({
        "station_admission_mode": STATION_ADMISSION_COMMITTED_FIFO_V2,
        "waiting_assigned_agent_ticks": admission.get("waiting_agent_ticks"),
        "waiting_assigned_ratio": admission.get("waiting_assigned_ratio"),
        "waiting_promotions": admission.get("waiting_promotions"),
        "waiting_duration_p95_ticks": admission.get(
            "waiting_duration_p95_ticks"
        ),
        "unresolved_waiter_count_final": admission.get(
            "unresolved_waiter_count_final"
        ),
    })
    checks = _arm_audit(metrics, manifest, admission, ticks)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            f"adaptive arm audit failed {load} seed={seed}: {failed}"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "arm": ARM,
            "arm_label": ARM_LABEL,
            "load": load,
            "seed": int(seed),
            "ticks": int(ticks),
            "num_robots": 48,
            "protocol_sha256": bundle["protocol_sha256"],
            "reference_protocol_sha256": protocol["reference"][
                "protocol_sha256"
            ],
        },
        "controller": {
            "selection_order": (
                "adaptive admissibility -> learned dynamic J -> unchanged S1"
            ),
            "exact_eta_used": False,
            "fifo_v2_preserved": True,
            "new_trainable_parameters": 0,
        },
        "manifest": dict(manifest_entry),
        "audit": {"passed": True, "checks": checks},
        "admission_audit": admission,
        "metrics": metrics,
        "dynamic_trace": list(assigner.dynamic_probe_trace_records),
    }
    _atomic_json(output, payload)
    print(f"[done] {output}")


def _metric_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    meta = payload.get("meta") or {}
    metrics = payload.get("metrics") or {}
    admission = payload.get("admission_audit") or {}
    arrivals = float(metrics.get("order_arrival_count", 0) or 0)
    completed = float(metrics.get("completed_orders", 0) or 0)
    row: dict[str, Any] = {
        "arm": str(meta["arm"]),
        "load": str(meta["load"]),
        "seed": int(meta["seed"]),
        "order_arrival_count": int(arrivals),
        "completion_fraction": completed / arrivals if arrivals > 0 else None,
    }
    for metric in SUMMARY_METRICS:
        if metric == "completion_fraction":
            continue
        value = metrics.get(metric)
        if value is None:
            fallback = {
                "waiting_assigned_agent_ticks": "waiting_agent_ticks",
                "waiting_assigned_ratio": "waiting_assigned_ratio",
                "waiting_promotions": "waiting_promotions",
                "waiting_duration_p95_ticks": "waiting_duration_p95_ticks",
                "unresolved_waiter_count_final": "unresolved_waiter_count_final",
            }.get(metric)
            if fallback is not None:
                value = admission.get(fallback)
        row[metric] = value
    return row


def _load_reference_rows(reference_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in REFERENCE_ARMS:
        for load in LOADS:
            for seed in SEEDS:
                path = reference._output_path(reference_root, arm, load, seed)
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = _read_json(path)
                if not bool((payload.get("audit") or {}).get("passed")):
                    raise ValueError(f"reference run audit failed: {path}")
                rows.append(_metric_row(payload))
    return rows


def _numeric(rows: Iterable[Mapping[str, Any]], metric: str) -> list[float]:
    result = []
    for row in rows:
        value = row.get(metric)
        if value is not None:
            result.append(float(value))
    return result


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for arm in ALL_ARMS:
        report[arm] = {}
        for load in LOADS:
            selected = [
                row for row in rows if row["arm"] == arm and row["load"] == load
            ]
            summary: dict[str, Any] = {"runs": len(selected)}
            for metric in SUMMARY_METRICS:
                values = _numeric(selected, metric)
                summary[f"{metric}_mean"] = (
                    statistics.mean(values) if values else None
                )
                summary[f"{metric}_std"] = (
                    statistics.pstdev(values) if values else None
                )
            report[arm][load] = summary
    return report


def _bootstrap_seed_cluster(
    deltas_by_seed: Mapping[int, list[float]],
    *,
    replicates: int = 20000,
    random_seed: int,
) -> list[float]:
    seeds = sorted(deltas_by_seed)
    rng = random.Random(random_seed)
    values: list[float] = []
    for _ in range(replicates):
        sample: list[float] = []
        for _slot in seeds:
            sampled = rng.choice(seeds)
            sample.extend(deltas_by_seed[sampled])
        values.append(statistics.mean(sample))
    values.sort()
    return [
        values[int(0.025 * (len(values) - 1))],
        values[int(0.975 * (len(values) - 1))],
    ]


def _paired_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {
        (str(row["arm"]), str(row["load"]), int(row["seed"])): row
        for row in rows
    }
    report: dict[str, Any] = {}
    for reference_arm in REFERENCE_ARMS:
        arm_report: dict[str, Any] = {}
        for metric_index, metric in enumerate(SUMMARY_METRICS):
            deltas_by_seed = {seed: [] for seed in SEEDS}
            by_load = {load: [] for load in LOADS}
            wins = ties = losses = 0
            for load in LOADS:
                for seed in SEEDS:
                    current = indexed[(ARM, load, seed)].get(metric)
                    other = indexed[(reference_arm, load, seed)].get(metric)
                    if current is None or other is None:
                        continue
                    delta = float(current) - float(other)
                    deltas_by_seed[seed].append(delta)
                    by_load[load].append(delta)
                    if metric == "completed_orders":
                        if delta > 0:
                            wins += 1
                        elif delta < 0:
                            losses += 1
                        else:
                            ties += 1
            flat = [value for values in deltas_by_seed.values() for value in values]
            if not flat:
                continue
            metric_report: dict[str, Any] = {
                "mean_delta": statistics.mean(flat),
                "seed_cluster_bootstrap_ci95": _bootstrap_seed_cluster(
                    deltas_by_seed,
                    random_seed=20260812 + metric_index,
                ),
                "mean_delta_by_load": {
                    load: statistics.mean(values) if values else None
                    for load, values in by_load.items()
                },
            }
            if metric == "completed_orders":
                metric_report["wins_ties_losses"] = {
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                }
            arm_report[metric] = metric_report
        report[f"{ARM}_minus_{reference_arm}"] = arm_report
    return report


def _summarise(root: Path) -> None:
    bundle, protocol = _load_protocol(root)
    reference_root = Path(str(protocol["reference"]["root"]))
    rows = _load_reference_rows(reference_root)
    for load in LOADS:
        for seed in SEEDS:
            path = _output_path(root, load, seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = _read_json(path)
            if not bool((payload.get("audit") or {}).get("passed")):
                raise ValueError(f"adaptive run audit failed: {path}")
            if payload.get("meta", {}).get("protocol_sha256") != bundle[
                "protocol_sha256"
            ]:
                raise ValueError(f"adaptive run protocol mismatch: {path}")
            rows.append(_metric_row(payload))
    expected = len(ALL_ARMS) * len(LOADS) * len(SEEDS)
    if len(rows) != expected:
        raise RuntimeError(f"summary rows={len(rows)} expected={expected}")

    adaptive_payloads = [
        _read_json(_output_path(root, load, seed))
        for load in LOADS
        for seed in SEEDS
    ]
    total_evaluations = sum(
        int(payload["metrics"].get("pre_admission_context_evaluations", 0))
        for payload in adaptive_payloads
    )
    total_deferred = sum(
        int(payload["metrics"].get("pre_admission_contexts_deferred_unique", 0))
        for payload in adaptive_payloads
    )
    if total_evaluations <= 0:
        raise RuntimeError("adaptive pre-admission was never evaluated")
    if total_deferred <= 0:
        raise RuntimeError(
            "adaptive pre-admission never deferred a context; the control "
            "point was inactive and the ablation is not informative"
        )
    report = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "protocol_sha256": bundle["protocol_sha256"],
        "reference_protocol_sha256": protocol["reference"]["protocol_sha256"],
        "num_robots": 48,
        "reference_runs": len(REFERENCE_ARMS) * len(LOADS) * len(SEEDS),
        "new_adaptive_runs": len(LOADS) * len(SEEDS),
        "combined_runs": len(rows),
        "audit": {
            "passed": True,
            "all_adaptive_run_audits_passed": True,
            "exact_591_600_manifests_replayed": True,
            "world_model_phi_context_j_s1_fifo_unchanged": True,
            "adaptive_pre_admission_only_new_controller_layer": True,
            "pre_admission_was_active": total_evaluations > 0,
            "at_least_one_context_was_deferred": total_deferred > 0,
        },
        "adaptive_activity": {
            "context_evaluations": total_evaluations,
            "contexts_deferred_unique": total_deferred,
        },
        "rows": rows,
        "aggregate": _aggregate(rows),
        "paired_comparisons": _paired_comparisons(rows),
    }
    _atomic_json(root / "online_summary.json", report)
    lines = []
    for path in sorted(root.glob("per_arm/**/*.json")):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    for path in (
        root / "online_summary.json",
        _protocol_path(root),
        _frozen_inputs_path(root),
    ):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "results.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "reference_runs": report["reference_runs"],
        "new_adaptive_runs": report["new_adaptive_runs"],
        "combined_runs": report["combined_runs"],
        "adaptive_context_evaluations": total_evaluations,
        "adaptive_contexts_deferred": total_deferred,
        "output_root": root.as_posix(),
    }, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("freeze", "arm", "summarize"), required=True
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT
    )
    parser.add_argument("--load", choices=LOADS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--trace-max-records", type=int, default=200)
    args = parser.parse_args()

    if args.mode == "freeze":
        _freeze(args.output_root, args.reference_root)
        return
    if args.seed is not None and args.seed not in SEEDS:
        raise SystemExit(f"seed must be in {SEEDS[0]}--{SEEDS[-1]}")
    if args.mode == "arm":
        if args.load is None or args.seed is None:
            raise SystemExit("arm mode requires --load and --seed")
        _run_arm(
            args.output_root,
            args.load,
            args.seed,
            args.ticks,
            args.trace_max_records,
        )
        return
    _summarise(args.output_root)


if __name__ == "__main__":
    main()
