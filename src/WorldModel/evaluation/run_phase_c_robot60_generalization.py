"""Zero-shot 60-robot generalisation of the frozen Phase-C/S1 policy.

Only the fleet size changes from the trained/evaluated 48-robot setup:

* the 20x20 map, four stations, queue semantics, order process, World Model,
  S1 configuration, and FIFO-V2 admission remain frozen;
* one Greedy-generated order manifest is replayed exactly by every arm;
* Greedy, Hungarian, pure Phase C, and Phase C + S1 are compared on fresh
  seeds 601--610 under low/mid/high demand.

The 60-robot JSON configurations are derived into the result directory and
audited to differ from the 48-robot source configurations only at
``robots.num_robots``.  No deployed policy source is modified by this file.
"""

from __future__ import annotations

import argparse
import copy
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
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    LOAD_CONFIGS as SOURCE_LOAD_CONFIGS,
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


SCHEMA_VERSION = "phase_c_robot60_generalization_run_v1"
PROTOCOL_SCHEMA_VERSION = "phase_c_robot60_generalization_protocol_v1"
SUMMARY_SCHEMA_VERSION = "phase_c_robot60_generalization_summary_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "phasec_s1_robot60_601_610_v1"
MODEL_CHECKPOINT = BASE_ROOT / "model_round1_v1/best_regret_world_model.pt"

SOURCE_ROBOT_COUNT = 48
TARGET_ROBOT_COUNT = 60
SEEDS = tuple(range(601, 611))
LOADS = ("low", "mid", "high")
TICKS = 1500
ARMS = ("greedy_manifest", "hungarian", "phasec", "s1")
ARM_LABELS = {
    "greedy_manifest": "GreedyRobot60ManifestSource",
    "hungarian": "HungarianRobot60FifoV2",
    "phasec": "PhaseCPureRobot60FifoV2",
    "s1": "PhaseCS1Robot60FifoV2",
}

RUNNER_PATH = Path(__file__)
SOURCE_FILES = {
    "world_model_assigner": Path(
        "Policies/TaskAssigner/WorldModelTaskAssigner/world_model_task_assigner.py"
    ),
    "phasec_s1_protocol": Path(
        "WorldModel/evaluation/phase_c_s1_hungarian_protocol.py"
    ),
    "fifo_pair_helpers": Path(
        "WorldModel/evaluation/run_phase_c_s1_fifo_pair.py"
    ),
    "station_state": Path("WorldState/station_state.py"),
    "world_model": Path("WorldModel/core/model.py"),
    "graph_builder": Path("WorldModel/graph/graph_builder.py"),
}

SUMMARY_METRICS = (
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
    "wall_time_s",
    "assignment_time_ms_mean",
    "waiting_assigned_ratio",
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


def _changed_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        keys = sorted(set(left) | set(right))
        for key in keys:
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_changed_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        paths = []
        if len(left) != len(right):
            paths.append(f"{prefix}.length")
        for index, (lvalue, rvalue) in enumerate(zip(left, right)):
            paths.extend(_changed_paths(lvalue, rvalue, f"{prefix}[{index}]"))
        return paths
    return [] if left == right else [prefix]


def _derived_config_path(root: Path, load: str) -> Path:
    return root / "frozen_configs" / f"world_model_config_PP_60_{load}.json"


def _derive_configs(root: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for load in LOADS:
        source_path = Path(SOURCE_LOAD_CONFIGS[load])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source = _read_json(source_path)
        robots = source.get("robots")
        if not isinstance(robots, Mapping):
            raise ValueError(f"source config lacks robots object: {source_path}")
        if int(robots.get("num_robots", -1)) != SOURCE_ROBOT_COUNT:
            raise ValueError(
                f"source config is not {SOURCE_ROBOT_COUNT}-robot: {source_path}"
            )
        derived = copy.deepcopy(source)
        derived["robots"]["num_robots"] = TARGET_ROBOT_COUNT
        changed = _changed_paths(source, derived)
        if changed != ["robots.num_robots"]:
            raise AssertionError(
                f"derived {load} config changed unexpected fields: {changed}"
            )
        derived_path = _derived_config_path(root, load)
        _atomic_json(derived_path, derived)
        entries[load] = {
            "source_path": source_path.as_posix(),
            "source_sha256": sha256_file(source_path),
            "derived_path": derived_path.as_posix(),
            "derived_sha256": sha256_file(derived_path),
            "changed_paths": changed,
            "source_robot_count": SOURCE_ROBOT_COUNT,
            "target_robot_count": TARGET_ROBOT_COUNT,
        }
    return entries


def _output_path(root: Path, arm: str, load: str, seed: int) -> Path:
    return root / "per_arm" / arm / f"{load}_seed{seed}.json"


def _protocol_path(root: Path) -> Path:
    return root / "robot60_generalization_protocol.json"


def _frozen_inputs_path(root: Path) -> Path:
    return root / "frozen_inputs.sha256"


def _make_protocol(root: Path) -> dict[str, Any]:
    if not MODEL_CHECKPOINT.is_file():
        raise FileNotFoundError(MODEL_CHECKPOINT)
    for name, path in SOURCE_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    config_entries = _derive_configs(root)
    payload: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "purpose": (
            "zero-shot fleet-size generalisation from 48 to 60 robots with "
            "the frozen Phase-C World Model, certified S1 configuration, and "
            "committed-capacity FIFO-V2 admission"
        ),
        "seeds": list(SEEDS),
        "loads": list(LOADS),
        "ticks": TICKS,
        "arms": list(ARMS),
        "source_robot_count": SOURCE_ROBOT_COUNT,
        "target_robot_count": TARGET_ROBOT_COUNT,
        "allowed_config_changes": ["robots.num_robots"],
        "order_process_unchanged_from_48_robot_configs": True,
        "manifest_source": "GreedyTaskAssigner",
        "exact_manifest_replay": True,
        "station_admission": STATION_ADMISSION_COMMITTED_FIFO_V2,
        "zero_shot": {
            "world_model_retrained": False,
            "s1_retuned": False,
            "fifo_modified": False,
            "policy_code_modified": False,
        },
        "model_checkpoint": {
            "path": MODEL_CHECKPOINT.as_posix(),
            "sha256": sha256_file(MODEL_CHECKPOINT),
        },
        "load_configs": config_entries,
        "phasec_config": dict(PHASEC_CONFIG),
        "s1_config": dict(S1_CONFIG),
        "code_hashes": {
            "runner": {
                "path": RUNNER_PATH.as_posix(),
                "sha256": sha256_file(RUNNER_PATH),
            },
            **{
                name: {"path": path.as_posix(), "sha256": sha256_file(path)}
                for name, path in SOURCE_FILES.items()
            },
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
        f"{protocol['model_checkpoint']['sha256']}  {protocol['model_checkpoint']['path']}",
    ]
    for load in LOADS:
        entry = protocol["load_configs"][load]
        lines.append(f"{entry['source_sha256']}  {entry['source_path']}")
        lines.append(f"{entry['derived_sha256']}  {entry['derived_path']}")
    for entry in protocol["code_hashes"].values():
        lines.append(f"{entry['sha256']}  {entry['path']}")
    _frozen_inputs_path(root).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _freeze(root: Path) -> None:
    expected = _make_protocol(root)
    path = _protocol_path(root)
    if path.is_file():
        existing = _read_json(path)
        if existing != expected:
            raise FileExistsError(f"existing robot60 protocol differs: {path}")
        print(f"[resume] protocol {path}")
    else:
        _atomic_json(path, expected)
        print(f"[freeze] protocol sha256={expected['protocol_sha256']}")
        print(f"[freeze] wrote {path}")
    _write_frozen_inputs(root, expected)
    print(f"[freeze] wrote {_frozen_inputs_path(root)}")


def _verify_protocol_inputs(protocol: Mapping[str, Any]) -> None:
    model = protocol["model_checkpoint"]
    if sha256_file(model["path"]) != model["sha256"]:
        raise ValueError("World Model checkpoint differs from frozen protocol")
    for load in LOADS:
        entry = protocol["load_configs"][load]
        if sha256_file(entry["source_path"]) != entry["source_sha256"]:
            raise ValueError(f"source {load} config differs from frozen protocol")
        if sha256_file(entry["derived_path"]) != entry["derived_sha256"]:
            raise ValueError(f"derived {load} config differs from frozen protocol")
        source = _read_json(Path(entry["source_path"]))
        derived = _read_json(Path(entry["derived_path"]))
        if _changed_paths(source, derived) != ["robots.num_robots"]:
            raise ValueError(f"derived {load} config changed more than robot count")
        if int(derived["robots"]["num_robots"]) != TARGET_ROBOT_COUNT:
            raise ValueError(f"derived {load} config is not 60-robot")
    for name, entry in protocol["code_hashes"].items():
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise ValueError(f"code input differs from frozen protocol: {name}")


def _load_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _protocol_path(root)
    bundle = _read_json(path)
    if bundle.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(f"unexpected robot60 protocol schema: {path}")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("robot60 protocol lacks protocol payload")
    if _canonical_sha(protocol) != str(bundle.get("protocol_sha256", "")):
        raise ValueError("robot60 protocol hash mismatch")
    _verify_protocol_inputs(protocol)
    return dict(bundle), dict(protocol)


def _config_path(protocol: Mapping[str, Any], load: str) -> str:
    return str(protocol["load_configs"][load]["derived_path"])


def _manifest_path_for(root: Path, load: str, seed: int) -> Path:
    return _manifest_path(root, load, seed)


def _prepare_manifest(root: Path, load: str, seed: int, ticks: int) -> None:
    bundle, protocol = _load_protocol(root)
    path = _manifest_path_for(root, load, seed)
    if path.is_file():
        _validate_manifest(path)
        existing_output = _output_path(root, "greedy_manifest", load, seed)
        if existing_output.is_file():
            print(f"[resume] manifest/output {existing_output}")
            return
        print(f"[resume] rebuilding missing Greedy output for {path}")

    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    path.parent.mkdir(parents=True, exist_ok=True)
    with _fifo_engine(trace_max_records=0) as holder:
        metrics = _run_one_assigner(
            _config_path(protocol, load),
            GreedyTaskAssigner(),
            int(seed),
            int(ticks),
            trace_label=ARM_LABELS["greedy_manifest"],
            save_order_manifest=str(path),
        )
    probe = holder.get("probe")
    if probe is None:
        raise RuntimeError("Greedy manifest source did not attach FIFO probe")
    admission = probe.summary()
    manifest = _validate_manifest(path)
    checks = {
        "manifest_written": bool(metrics.get("order_arrival_manifest_path")),
        "fifo_admission_invariant": bool(admission.get("passed")),
        "fifo_mode": admission.get("mode") == STATION_ADMISSION_COMMITTED_FIFO_V2,
        "num_agents_is_60": int(metrics.get("num_agents", -1)) == TARGET_ROBOT_COUNT,
        "ticks_match": int(metrics.get("ticks", -1)) == int(ticks),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Greedy manifest audit failed {load} seed={seed}: {failed}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "arm": "greedy_manifest",
            "arm_label": ARM_LABELS["greedy_manifest"],
            "load": load,
            "seed": int(seed),
            "ticks": int(ticks),
            "num_robots": TARGET_ROBOT_COUNT,
            "protocol_sha256": bundle["protocol_sha256"],
        },
        "generalization": {
            "source_robot_count": SOURCE_ROBOT_COUNT,
            "target_robot_count": TARGET_ROBOT_COUNT,
            "zero_shot": True,
            "changed_config_fields": ["robots.num_robots"],
        },
        "manifest": {
            "path": path.as_posix(),
            "file_sha256": sha256_file(path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "audit": {"passed": True, "checks": checks},
        "admission_audit": admission,
        "metrics": metrics,
    }
    _atomic_json(_output_path(root, "greedy_manifest", load, seed), payload)
    print(f"[done] manifest and Greedy output {load} seed={seed}")


def _make_assigner(arm: str, protocol: Mapping[str, Any]):
    common = {
        "checkpoint_path": str(protocol["model_checkpoint"]["path"]),
        "top_m": 10,
        "energy_conv_random_flip_seed": 0,
    }
    if arm == "hungarian":
        return HungarianTaskAssigner()
    if arm == "phasec":
        return WorldModelTaskAssigner(**common, **PHASEC_CONFIG)
    if arm == "s1":
        return WorldModelTaskAssigner(**common, **S1_CONFIG)
    raise ValueError(f"unsupported executable arm: {arm}")


def _arm_audit(
    arm: str,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    admission: Mapping[str, Any],
    ticks: int,
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
        "num_agents_is_60": int(metrics.get("num_agents", -1)) == TARGET_ROBOT_COUNT,
        "ticks_match": int(metrics.get("ticks", -1)) == int(ticks),
        "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
    }
    if arm in {"phasec", "s1"}:
        checks.update({
            "world_model_used": int(metrics.get("model_assign_calls", 0)) > 0,
            "native_no_assign_disabled": not bool(
                metrics.get("native_no_assign_enabled", False)
            ),
        })
    if arm == "s1":
        checks["s1_conversion_used"] = int(
            metrics.get("energy_conv_contexts", 0)
        ) > 0
    return checks


def _run_arm(root: Path, arm: str, load: str, seed: int, ticks: int) -> None:
    bundle, protocol = _load_protocol(root)
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

    assigner = _make_assigner(arm, protocol)
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    with _fifo_engine(trace_max_records=0) as holder:
        metrics = _run_one_assigner(
            _config_path(protocol, load),
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
    metrics.update({
        "station_admission_mode": STATION_ADMISSION_COMMITTED_FIFO_V2,
        "waiting_assigned_agent_ticks": admission.get("waiting_agent_ticks"),
        "waiting_assigned_ratio": admission.get("waiting_assigned_ratio"),
        "waiting_promotions": admission.get("waiting_promotions"),
        "waiting_duration_p95_ticks": admission.get("waiting_duration_p95_ticks"),
        "unresolved_waiter_count_final": admission.get("unresolved_waiter_count_final"),
    })
    checks = _arm_audit(arm, metrics, manifest, admission, ticks)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"{arm} audit failed {load} seed={seed}: {failed}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "arm": arm,
            "arm_label": ARM_LABELS[arm],
            "load": load,
            "seed": int(seed),
            "ticks": int(ticks),
            "num_robots": TARGET_ROBOT_COUNT,
            "protocol_sha256": bundle["protocol_sha256"],
        },
        "generalization": {
            "source_robot_count": SOURCE_ROBOT_COUNT,
            "target_robot_count": TARGET_ROBOT_COUNT,
            "zero_shot": True,
            "changed_config_fields": ["robots.num_robots"],
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "audit": {"passed": True, "checks": checks},
        "admission_audit": admission,
        "metrics": metrics,
    }
    _atomic_json(output, payload)
    print(f"[done] {output}")


def _numeric(rows: list[dict[str, Any]], metric: str) -> list[float]:
    return [float(row[metric]) for row in rows if row.get(metric) is not None]


def _summarise(root: Path) -> None:
    bundle, protocol = _load_protocol(root)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[str] = []
    manifest_hashes: dict[tuple[str, int], set[str]] = {}
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
                meta = payload.get("meta") or {}
                if meta.get("protocol_sha256") != bundle["protocol_sha256"]:
                    failed.append(f"protocol:{path.as_posix()}")
                    continue
                metrics = payload.get("metrics") or {}
                if int(metrics.get("num_agents", -1)) != TARGET_ROBOT_COUNT:
                    failed.append(f"robot_count:{path.as_posix()}")
                    continue
                key = (load, int(seed))
                manifest_hashes.setdefault(key, set()).add(
                    str(payload.get("manifest", {}).get("content_sha256"))
                )
                row = {
                    "arm": arm,
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

    manifest_mismatches = [
        {"load": load, "seed": seed, "hashes": sorted(hashes)}
        for (load, seed), hashes in sorted(manifest_hashes.items())
        if len(hashes) != 1
    ]
    expected = len(ARMS) * len(LOADS) * len(SEEDS)
    if missing or failed or manifest_mismatches or len(rows) != expected:
        raise RuntimeError(
            "summary incomplete: "
            f"rows={len(rows)}/{expected} missing={len(missing)} "
            f"failed={len(failed)} manifest_mismatches={len(manifest_mismatches)}"
        )

    aggregate: dict[str, dict[str, dict[str, Any]]] = {}
    aggregate_metrics = ("completion_fraction",) + SUMMARY_METRICS
    for arm in ARMS:
        aggregate[arm] = {}
        for load in LOADS:
            selected = [
                row for row in rows if row["arm"] == arm and row["load"] == load
            ]
            summary: dict[str, Any] = {"runs": len(selected)}
            for metric in aggregate_metrics:
                metric_values = _numeric(selected, metric)
                if not metric_values:
                    summary[f"{metric}_mean"] = None
                    summary[f"{metric}_std"] = None
                else:
                    summary[f"{metric}_mean"] = statistics.mean(metric_values)
                    summary[f"{metric}_std"] = statistics.pstdev(metric_values)
            aggregate[arm][load] = summary

    report = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "protocol_sha256": bundle["protocol_sha256"],
        "source_robot_count": SOURCE_ROBOT_COUNT,
        "target_robot_count": TARGET_ROBOT_COUNT,
        "expected_runs": expected,
        "completed_runs": len(rows),
        "audit": {
            "passed": True,
            "all_run_audits_passed": True,
            "all_runs_use_60_robots": True,
            "paired_manifest_hashes_match": True,
            "zero_shot_checkpoint_unchanged": (
                sha256_file(MODEL_CHECKPOINT)
                == protocol["model_checkpoint"]["sha256"]
            ),
        },
        "rows": rows,
        "aggregate": aggregate,
    }
    _atomic_json(root / "online_summary.json", report)

    manifest_lines = []
    for path in sorted(root.glob("per_arm/**/*.json")):
        manifest_lines.append(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        )
    for path in sorted((root / "frozen_configs").glob("*.json")):
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
    print(json.dumps({
        "completed_runs": len(rows),
        "expected_runs": expected,
        "source_robot_count": SOURCE_ROBOT_COUNT,
        "target_robot_count": TARGET_ROBOT_COUNT,
        "output_root": root.as_posix(),
    }, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("freeze", "manifest", "arm", "summarize"), required=True
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--load", choices=LOADS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int, default=TICKS)
    args = parser.parse_args()
    root = args.output_root

    if args.mode == "freeze":
        _freeze(root)
        return
    if args.seed is not None and args.seed not in SEEDS:
        raise SystemExit(f"seed must be in {SEEDS[0]}--{SEEDS[-1]}")
    if args.mode == "manifest":
        if args.load is None or args.seed is None:
            raise SystemExit("manifest mode requires --load and --seed")
        _prepare_manifest(root, args.load, args.seed, args.ticks)
        return
    if args.mode == "arm":
        if args.arm is None or args.arm == "greedy_manifest":
            raise SystemExit("arm mode requires hungarian, phasec, or s1")
        if args.load is None or args.seed is None:
            raise SystemExit("arm mode requires --load and --seed")
        _run_arm(root, args.arm, args.load, args.seed, args.ticks)
        return
    _summarise(root)


if __name__ == "__main__":
    main()
