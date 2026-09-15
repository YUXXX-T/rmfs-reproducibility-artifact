"""Complete the missing learned context-J arm for the frozen 60-robot test.

This runner is deliberately additive.  It does not edit or regenerate the
completed four-arm result block at::

    phasec_s1_robot60_601_610_v1

Instead, it reuses that block's audited 60-robot configurations and exact
Greedy order manifests, runs only the final ``learned_j_new`` policy, and
builds a combined five-arm summary by reference.  The deployed contract is::

    dynamic learned J selects the next context
        -> the unchanged certified S1 scorer selects its robot

No World Model, station-phi, context-J head, FIFO rule, or policy parameter is
retrained or retuned for 60 robots.
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

from Policies.TaskAssigner.WorldModelTaskAssigner.context_j_learned_dynamic_assigner import (
    LearnedContextJDynamicAssigner,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    S1_CONFIG,
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_robot60_generalization import (
    LOADS,
    MODEL_CHECKPOINT,
    SEEDS,
    SUMMARY_METRICS,
    TARGET_ROBOT_COUNT,
    TICKS,
    _config_path,
    _load_protocol as _load_reference_protocol,
)
from WorldModel.evaluation.run_phase_c_s1_fifo_pair import (
    _fifo_engine,
    _manifest_path,
    _validate_manifest,
)
from WorldState.station_state import STATION_ADMISSION_COMMITTED_FIFO_V2


SCHEMA_VERSION = "phase_c_robot60_context_j_missing_run_v1"
PROTOCOL_SCHEMA_VERSION = "phase_c_robot60_context_j_missing_protocol_v1"
SUMMARY_SCHEMA_VERSION = "phase_c_robot60_context_j_combined_summary_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_REFERENCE_ROOT = BASE_ROOT / "phasec_s1_robot60_601_610_v1"
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "ctxj_robot60_601_610_v1"

ARM = "learned_j_new"
REFERENCE_ARMS = ("greedy_manifest", "hungarian", "phasec", "s1")
COMBINED_ARMS = REFERENCE_ARMS + (ARM,)
ARM_LABEL = "PhaseCLearnedJNewRobot60FifoV2"

PHI_HEAD = BASE_ROOT / (
    "station_congestion_head_region_dev_511_520_v1/"
    "linear_head_v1/best_station_congestion_head.pt"
)
PHI_SCALE_CONTRACT = BASE_ROOT / (
    "station_congestion_head_region_dev_511_520_v1/"
    "station_congestion_scale_contract.json"
)
NEW_CONTEXT_HEAD = BASE_ROOT / (
    "context_dispatch_j_head_train_571_590_v1/best_context_j_head.pt"
)

RUNNER_PATH = Path(__file__)
LEARNED_ASSIGNER_PATH = Path(
    "Policies/TaskAssigner/WorldModelTaskAssigner/"
    "context_j_learned_dynamic_assigner.py"
)
REFERENCE_RUNNER_PATH = Path(
    "WorldModel/evaluation/run_phase_c_robot60_generalization.py"
)
FIFO_HELPER_PATH = Path("WorldModel/evaluation/run_phase_c_s1_fifo_pair.py")

PAIRED_METRICS = (
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
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _protocol_path(root: Path) -> Path:
    return root / "robot60_context_j_protocol.json"


def _frozen_inputs_path(root: Path) -> Path:
    return root / "frozen_inputs.sha256"


def _output_path(root: Path, load: str, seed: int) -> Path:
    return root / "per_arm" / ARM / f"{load}_seed{seed}.json"


def _reference_summary_path(reference_root: Path) -> Path:
    return reference_root / "online_summary.json"


def _reference_protocol_path(reference_root: Path) -> Path:
    return reference_root / "robot60_generalization_protocol.json"


def _manifest_entry(reference_root: Path, load: str, seed: int) -> dict[str, Any]:
    path = _manifest_path(reference_root, load, seed)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing frozen reference manifest: {path}; restore the original "
            "order_manifests/ directory before submitting the J supplement"
        )
    manifest = _validate_manifest(path)
    return {
        "path": path.as_posix(),
        "file_sha256": sha256_file(path),
        "content_sha256": manifest.get("manifest_sha256"),
        "total_orders": manifest.get("total_orders"),
    }


def _validate_reference_summary(reference_root: Path) -> dict[str, Any]:
    path = _reference_summary_path(reference_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = _read_json(path)
    audit = summary.get("audit") or {}
    if not bool(audit.get("passed")):
        raise ValueError("reference 60-robot summary audit did not pass")
    if int(summary.get("target_robot_count", -1)) != TARGET_ROBOT_COUNT:
        raise ValueError("reference summary is not the frozen 60-robot block")
    if int(summary.get("completed_runs", -1)) != 120:
        raise ValueError("reference summary is not the complete 120-run block")
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != 120:
        raise ValueError("reference summary rows are incomplete")
    observed_arms = {str(row.get("arm")) for row in rows if isinstance(row, Mapping)}
    if observed_arms != set(REFERENCE_ARMS):
        raise ValueError(f"unexpected reference arms: {sorted(observed_arms)}")
    return summary


def _make_protocol(reference_root: Path) -> dict[str, Any]:
    reference_bundle, reference_protocol = _load_reference_protocol(reference_root)
    reference_summary = _validate_reference_summary(reference_root)
    required = {
        "model_checkpoint": MODEL_CHECKPOINT,
        "station_phi": PHI_HEAD,
        "station_phi_scale_contract": PHI_SCALE_CONTRACT,
        "context_j_new_head": NEW_CONTEXT_HEAD,
        "runner": RUNNER_PATH,
        "learned_assigner": LEARNED_ASSIGNER_PATH,
        "reference_runner": REFERENCE_RUNNER_PATH,
        "fifo_helper": FIFO_HELPER_PATH,
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    manifests: dict[str, dict[str, Any]] = {}
    for load in LOADS:
        manifests[load] = {
            str(seed): _manifest_entry(reference_root, load, seed) for seed in SEEDS
        }

    payload: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "purpose": (
            "add the omitted final learned context-J arm to the frozen "
            "60-robot 601-610 paired generalisation block"
        ),
        "arm": ARM,
        "reference_arms": list(REFERENCE_ARMS),
        "seeds": list(SEEDS),
        "loads": list(LOADS),
        "ticks": TICKS,
        "source_robot_count": reference_protocol["source_robot_count"],
        "target_robot_count": TARGET_ROBOT_COUNT,
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
            "summary_sha256": sha256_file(_reference_summary_path(reference_root)),
            "completed_runs": reference_summary["completed_runs"],
        },
        "model_checkpoint": dict(reference_protocol["model_checkpoint"]),
        "load_configs": dict(reference_protocol["load_configs"]),
        "station_phi": {
            "path": PHI_HEAD.as_posix(),
            "sha256": sha256_file(PHI_HEAD),
            "scale_contract": PHI_SCALE_CONTRACT.as_posix(),
            "scale_contract_sha256": sha256_file(PHI_SCALE_CONTRACT),
        },
        "context_j_head": {
            "path": NEW_CONTEXT_HEAD.as_posix(),
            "sha256": sha256_file(NEW_CONTEXT_HEAD),
            "training_block": "571-590",
        },
        "s1_config": dict(S1_CONFIG),
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
        "zero_shot": {
            "world_model_retrained": False,
            "station_phi_retrained": False,
            "context_j_retrained": False,
            "s1_retuned": False,
            "fifo_modified": False,
            "allowed_config_changes": ["robots.num_robots"],
        },
        "manifests": manifests,
        "code_hashes": {
            name: {"path": path.as_posix(), "sha256": sha256_file(path)}
            for name, path in required.items()
            if name not in {"model_checkpoint", "station_phi", "station_phi_scale_contract", "context_j_new_head"}
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
    for load in LOADS:
        config = protocol["load_configs"][load]
        lines.append(f"{config['derived_sha256']}  {config['derived_path']}")
        for seed in SEEDS:
            manifest = protocol["manifests"][load][str(seed)]
            lines.append(f"{manifest['file_sha256']}  {manifest['path']}")
    for entry in protocol["code_hashes"].values():
        lines.append(f"{entry['sha256']}  {entry['path']}")
    _frozen_inputs_path(root).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _freeze(root: Path, reference_root: Path) -> None:
    expected = _make_protocol(reference_root)
    path = _protocol_path(root)
    if path.is_file():
        existing = _read_json(path)
        if existing != expected:
            raise FileExistsError(f"existing robot60 J protocol differs: {path}")
        print(f"[resume] protocol {path}")
    else:
        _atomic_json(path, expected)
        print(f"[freeze] protocol sha256={expected['protocol_sha256']}")
        print(f"[freeze] wrote {path}")
    _write_frozen_inputs(root, expected)
    print(f"[freeze] wrote {_frozen_inputs_path(root)}")


def _load_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _protocol_path(root)
    bundle = _read_json(path)
    if bundle.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(f"unexpected robot60 J protocol schema: {path}")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("robot60 J protocol lacks protocol payload")
    if _canonical_sha(protocol) != str(bundle.get("protocol_sha256", "")):
        raise ValueError("robot60 J protocol hash mismatch")

    reference_root = Path(str(protocol["reference"]["root"]))
    reference_bundle, reference_protocol = _load_reference_protocol(reference_root)
    if reference_bundle["protocol_sha256"] != protocol["reference"]["protocol_sha256"]:
        raise ValueError("reference protocol identity changed")
    if sha256_file(protocol["reference"]["protocol_path"]) != protocol["reference"]["protocol_file_sha256"]:
        raise ValueError("reference protocol file changed")
    if sha256_file(protocol["reference"]["summary_path"]) != protocol["reference"]["summary_sha256"]:
        raise ValueError("reference summary changed")
    _validate_reference_summary(reference_root)

    if dict(protocol["s1_config"]) != dict(S1_CONFIG):
        raise ValueError("current S1 configuration differs from frozen J protocol")
    checks = (
        (protocol["model_checkpoint"]["path"], protocol["model_checkpoint"]["sha256"], "model checkpoint"),
        (protocol["station_phi"]["path"], protocol["station_phi"]["sha256"], "station phi"),
        (
            protocol["station_phi"]["scale_contract"],
            protocol["station_phi"]["scale_contract_sha256"],
            "station phi scale contract",
        ),
        (protocol["context_j_head"]["path"], protocol["context_j_head"]["sha256"], "context J head"),
    )
    for file_path, claimed, label in checks:
        if sha256_file(file_path) != claimed:
            raise ValueError(f"{label} differs from frozen protocol")
    for name, entry in protocol["code_hashes"].items():
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise ValueError(f"code input differs from frozen protocol: {name}")
    for load in LOADS:
        config = protocol["load_configs"][load]
        if sha256_file(config["derived_path"]) != config["derived_sha256"]:
            raise ValueError(f"derived 60-robot config changed: {load}")
        if config != reference_protocol["load_configs"][load]:
            raise ValueError(f"J protocol no longer uses reference config: {load}")
        for seed in SEEDS:
            entry = protocol["manifests"][load][str(seed)]
            if sha256_file(entry["path"]) != entry["file_sha256"]:
                raise ValueError(f"reference manifest changed: {load} seed={seed}")
            manifest = _validate_manifest(Path(entry["path"]))
            if manifest.get("manifest_sha256") != entry["content_sha256"]:
                raise ValueError(f"reference manifest content changed: {load} seed={seed}")
    return dict(bundle), dict(protocol)


def _make_assigner(protocol: Mapping[str, Any], trace_max: int):
    return LearnedContextJDynamicAssigner(
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
        "fifo_mode": admission.get("mode") == STATION_ADMISSION_COMMITTED_FIFO_V2,
        "num_agents_is_60": int(metrics.get("num_agents", -1))
        == TARGET_ROBOT_COUNT,
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
        "no_assign_added": not bool(metrics.get("context_j_no_assign_added", True)),
        "hard_gate_added": not bool(metrics.get("context_j_hard_gate_added", True)),
        "e_demand_modified": not bool(
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
        raise ValueError(
            f"ticks={ticks} differs from frozen protocol ticks={protocol['ticks']}"
        )
    output = _output_path(root, load, seed)
    if output.is_file():
        existing = _read_json(output)
        if (
            existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("meta", {}).get("protocol_sha256")
            == bundle["protocol_sha256"]
            and existing.get("meta", {}).get("arm") == ARM
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

    with _fifo_engine(trace_max_records=0) as holder:
        metrics = _run_one_assigner(
            _config_path(protocol, load),
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
    if hasattr(assigner, "dynamic_probe_metrics"):
        metrics.update(assigner.dynamic_probe_metrics())
    metrics.update(
        {
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
        }
    )
    checks = _arm_audit(metrics, manifest, admission, ticks)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            f"{ARM} audit failed {load} seed={seed}: {failed}"
        )

    traces: list[Any] = []
    if hasattr(assigner, "dynamic_probe_trace_records"):
        traces = list(assigner.dynamic_probe_trace_records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "arm": ARM,
            "arm_label": ARM_LABEL,
            "load": load,
            "seed": int(seed),
            "ticks": int(ticks),
            "num_robots": TARGET_ROBOT_COUNT,
            "protocol_sha256": bundle["protocol_sha256"],
            "reference_protocol_sha256": protocol["reference"][
                "protocol_sha256"
            ],
        },
        "generalization": {
            "source_robot_count": protocol["source_robot_count"],
            "target_robot_count": TARGET_ROBOT_COUNT,
            "zero_shot": True,
            "changed_config_fields": ["robots.num_robots"],
        },
        "manifest": dict(manifest_entry),
        "audit": {"passed": True, "checks": checks},
        "admission_audit": admission,
        "metrics": metrics,
        "dynamic_trace": traces,
    }
    _atomic_json(output, payload)
    print(f"[done] {output}")


def _numeric(rows: Iterable[Mapping[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if value is not None:
            values.append(float(value))
    return values


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("completion_fraction",) + SUMMARY_METRICS
    aggregate: dict[str, Any] = {}
    for arm in COMBINED_ARMS:
        aggregate[arm] = {}
        for load in LOADS:
            selected = [
                row for row in rows if row["arm"] == arm and row["load"] == load
            ]
            summary: dict[str, Any] = {"runs": len(selected)}
            for metric in metrics:
                values = _numeric(selected, metric)
                summary[f"{metric}_mean"] = (
                    statistics.mean(values) if values else None
                )
                summary[f"{metric}_std"] = (
                    statistics.pstdev(values) if values else None
                )
            aggregate[arm][load] = summary
    return aggregate


def _bootstrap_seed_cluster(
    deltas_by_seed: Mapping[int, list[float]],
    *,
    replicates: int = 20000,
    random_seed: int = 20260811,
) -> list[float]:
    seeds = sorted(deltas_by_seed)
    rng = random.Random(random_seed)
    samples: list[float] = []
    for _ in range(replicates):
        values: list[float] = []
        for _slot in seeds:
            sampled_seed = rng.choice(seeds)
            values.extend(deltas_by_seed[sampled_seed])
        samples.append(statistics.mean(values))
    samples.sort()
    lower = samples[int(0.025 * (len(samples) - 1))]
    upper = samples[int(0.975 * (len(samples) - 1))]
    return [lower, upper]


def _paired_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {
        (str(row["arm"]), str(row["load"]), int(row["seed"])): row
        for row in rows
    }
    comparisons: dict[str, Any] = {}
    for reference_arm in REFERENCE_ARMS:
        arm_report: dict[str, Any] = {}
        for metric_index, metric in enumerate(PAIRED_METRICS):
            deltas_by_seed: dict[int, list[float]] = {seed: [] for seed in SEEDS}
            load_means: dict[str, float] = {}
            completed_wins = completed_ties = completed_losses = 0
            for load in LOADS:
                load_deltas: list[float] = []
                for seed in SEEDS:
                    current = indexed[(ARM, load, seed)].get(metric)
                    reference = indexed[(reference_arm, load, seed)].get(metric)
                    if current is None or reference is None:
                        continue
                    delta = float(current) - float(reference)
                    deltas_by_seed[seed].append(delta)
                    load_deltas.append(delta)
                    if metric == "completed_orders":
                        if delta > 0:
                            completed_wins += 1
                        elif delta < 0:
                            completed_losses += 1
                        else:
                            completed_ties += 1
                load_means[load] = statistics.mean(load_deltas)
            flat = [value for values in deltas_by_seed.values() for value in values]
            metric_report: dict[str, Any] = {
                "mean_delta": statistics.mean(flat),
                "seed_cluster_bootstrap_ci95": _bootstrap_seed_cluster(
                    deltas_by_seed,
                    random_seed=20260811 + metric_index,
                ),
                "mean_delta_by_load": load_means,
            }
            if metric == "completed_orders":
                metric_report["wins_ties_losses"] = {
                    "wins": completed_wins,
                    "ties": completed_ties,
                    "losses": completed_losses,
                }
            arm_report[metric] = metric_report
        comparisons[f"{ARM}_minus_{reference_arm}"] = arm_report
    return comparisons


def _summarise(root: Path) -> None:
    bundle, protocol = _load_protocol(root)
    reference_root = Path(str(protocol["reference"]["root"]))
    reference_summary = _validate_reference_summary(reference_root)
    reference_rows = [dict(row) for row in reference_summary["rows"]]
    rows = list(reference_rows)
    missing: list[str] = []
    failed: list[str] = []

    for load in LOADS:
        for seed in SEEDS:
            path = _output_path(root, load, seed)
            if not path.is_file():
                missing.append(path.as_posix())
                continue
            payload = _read_json(path)
            if not bool(payload.get("audit", {}).get("passed")):
                failed.append(path.as_posix())
                continue
            meta = payload.get("meta") or {}
            if meta.get("protocol_sha256") != bundle["protocol_sha256"]:
                failed.append(f"protocol:{path.as_posix()}")
                continue
            metrics = payload.get("metrics") or {}
            if int(metrics.get("num_agents", -1)) != TARGET_ROBOT_COUNT:
                failed.append(f"robot_count:{path.as_posix()}")
                continue
            manifest = payload.get("manifest") or {}
            frozen_manifest = protocol["manifests"][load][str(seed)]
            if manifest.get("content_sha256") != frozen_manifest["content_sha256"]:
                failed.append(f"manifest:{path.as_posix()}")
                continue
            row: dict[str, Any] = {
                "arm": ARM,
                "load": load,
                "seed": int(seed),
                "order_arrival_count": metrics.get("order_arrival_count"),
            }
            for metric in SUMMARY_METRICS:
                row[metric] = metrics.get(metric)
            arrivals = float(metrics.get("order_arrival_count", 0) or 0)
            completed = float(metrics.get("completed_orders", 0) or 0)
            row["completion_fraction"] = completed / arrivals if arrivals > 0 else None
            rows.append(row)

    if missing or failed or len(rows) != 150:
        raise RuntimeError(
            "combined summary incomplete: "
            f"rows={len(rows)}/150 missing={len(missing)} failed={len(failed)}"
        )

    report = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "protocol_sha256": bundle["protocol_sha256"],
        "reference_protocol_sha256": protocol["reference"]["protocol_sha256"],
        "source_robot_count": protocol["source_robot_count"],
        "target_robot_count": TARGET_ROBOT_COUNT,
        "reference_runs": len(reference_rows),
        "new_j_runs": 30,
        "combined_runs": len(rows),
        "audit": {
            "passed": True,
            "reference_summary_passed": True,
            "all_j_run_audits_passed": True,
            "all_j_runs_use_60_robots": True,
            "exact_reference_manifests_replayed": True,
            "zero_shot_checkpoints_unchanged": True,
        },
        "rows": rows,
        "aggregate": _aggregate(rows),
        "paired_comparisons": _paired_comparisons(rows),
    }
    _atomic_json(root / "online_summary.json", report)

    manifest_lines = []
    for path in sorted(root.glob("per_arm/**/*.json")):
        manifest_lines.append(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        )
    for path in (
        root / "online_summary.json",
        _protocol_path(root),
        _frozen_inputs_path(root),
    ):
        manifest_lines.append(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        )
    (root / "results.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "reference_runs": len(reference_rows),
                "new_j_runs": 30,
                "combined_runs": len(rows),
                "output_root": root.as_posix(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


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
