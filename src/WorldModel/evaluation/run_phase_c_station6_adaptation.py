"""Six-station World-Model adaptation campaign.

The scale-generalisation campaign deliberately treats the 30x30/72-robot/
6-station cell as a baseline-only schema probe: the historical checkpoint has
``demand_dim=9`` (five global channels plus four station channels).  This
runner is the *separate adaptation campaign* needed before making any
six-station World-Model claim.

The campaign is intentionally implemented as an experiment orchestrator.  It
does not modify the simulator, graph builder, task assigners, station queues,
or training code.  Its stages are resumable and leave a complete lineage:

1. derive and hash a 30x30/72-robot/6-station configuration;
2. adapt the 4-station checkpoint by expanding only the demand input layer;
3. collect Greedy counterfactual data and train a six-station behaviour WM;
4. collect pure six-station WM-on-policy snapshots, build isolated H=10
   counterfactual labels, and train the six-station Phase-C core;
5. collect a second, seed-disjoint snapshot block from the trained core,
   generate W=200 labels, and train a LongRiskHead-only repair;
6. fit a six-station station-conditioned head (state-level J1 probe) on the
   final encoder;
7. run paired Greedy/Hungarian/Phase-C/Combo-S1+J1 evaluations.

The final Combo arm is therefore an adapted six-station model, never a frozen
four-station zero-shot result.  Core-adaptation seeds 711--720, held-out
online-evaluation seeds 721--730, and final LongRiskHead/J1 seeds 731--740
are mutually disjoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "phase_c_station6_adaptation_protocol_v3"
RUN_SCHEMA_VERSION = "phase_c_station6_adaptation_run_v3"
SUMMARY_SCHEMA_VERSION = "phase_c_station6_adaptation_summary_v3"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "phasec_station6_adaptation_711_740_v1"

# This is the corrected 4-station lineage used by the scale campaign.  It is
# only an adapter source; no result from it is reported as six-station
# zero-shot.
SOURCE_CHECKPOINT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_longrisk_head_repair_681_700_v1/"
    "training/long_risk_head_only_v1/best_long_risk_world_model.pt"
)

LOADS = ("low", "mid", "high")
CORE_DATA_SEEDS = tuple(range(711, 721))
CORE_TRAIN_SEEDS = tuple(range(711, 718))
CORE_VAL_SEEDS = (718, 719)
CORE_TEST_SEEDS = (720,)
ONLINE_TEST_SEEDS = tuple(range(721, 731))
REPAIR_DATA_SEEDS = tuple(range(731, 741))
REPAIR_TRAIN_SEEDS = tuple(range(731, 738))
REPAIR_VAL_SEEDS = (738, 739)
REPAIR_TEST_SEEDS = (740,)
TICKS = 1500
STATION_COUNT = 6
MAP_ROWS = 30
MAP_COLS = 30
ROBOT_COUNT = 72
TOP_M = 10
SNAPSHOT_INTERVAL = 20
MAX_CONTEXTS_PER_TICK = 2
ROLLOUT_HORIZON = 10
LONG_RISK_HORIZON = 200
LONG_RISK_TERMINAL_WINDOW = 50

ARM_KEYS = ("greedy", "hungarian", "phasec", "combo_s1_j1")
REPORT_METRICS = (
    "completed_orders",
    "completed_tasks",
    "avg_task_duration",
    "avg_excess_delay",
    "open_order_count",
    "pending_order_count",
    "stall_ratio_mean",
    "stall_ratio_max",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "congestion_events",
    "severe_events",
    "risk_rate_per_100",
    "assignment_time_ms_mean",
    "wall_time_s",
)

FROZEN_CODE_INPUTS = (
    Path("WorldModel/evaluation/run_phase_c_station6_adaptation.py"),
    Path("WorldModel/evaluation/run_phase_c_scale_generalization.py"),
    Path("WorldModel/evaluation/evaluate_online_v6.py"),
    Path("WorldModel/evaluation/decision_snapshot_probe.py"),
    Path("WorldModel/data/data_collector.py"),
    Path("WorldModel/data/build_phase_c_round1_dataset.py"),
    Path("WorldModel/data/generate_long_risk_labels.py"),
    Path("WorldModel/data/fuse_and_split.py"),
    Path("WorldModel/training/run_train_v6.py"),
    Path("WorldModel/training/train_long_risk_head_only.py"),
    Path("WorldModel/training/train_station_congestion_head.py"),
    Path(
        "Policies/TaskAssigner/WorldModelTaskAssigner/"
        "world_model_task_assigner.py"
    ),
    Path(
        "Policies/TaskAssigner/WorldModelTaskAssigner/"
        "psi_dispatch_context_assigner.py"
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        import torch

        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    log: Path | None = None,
    env_overrides: Mapping[str, str] | None = None,
) -> None:
    """Run a child command with an optional durable log."""

    child_env = None
    if env_overrides:
        child_env = os.environ.copy()
        child_env.update({str(key): str(value) for key, value in env_overrides.items()})
    log and log.parent.mkdir(parents=True, exist_ok=True)
    if log is None:
        subprocess.run(
            list(command), cwd=str(cwd), check=True, env=child_env
        )
        return
    with log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("$ " + " ".join(str(x) for x in command) + "\n")
        handle.flush()
        subprocess.run(
            list(command),
            cwd=str(cwd),
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=child_env,
        )


def _cpu_training_env(max_threads: int = 60) -> dict[str, str]:
    allocated = int(
        os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 1)
    )
    requested = int(
        os.environ.get("PHASEC_STATION6_TRAIN_THREADS") or allocated
    )
    threads = max(1, min(int(max_threads), allocated, requested))
    value = str(threads)
    return {
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
        "VECLIB_MAXIMUM_THREADS": value,
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _station_specs(rows: int, cols: int) -> list[dict[str, Any]]:
    offset = max(5, cols // 4)
    positions = [
        (0, offset),
        (0, cols - 1 - offset),
        (rows - 1, offset),
        (rows - 1, cols - 1 - offset),
        (rows // 2, 0),
        (rows // 2, cols - 1),
    ]
    if len(set(positions)) != STATION_COUNT:
        raise ValueError(f"six-station positions are not unique: {positions}")
    return [
        {
            "id": index + 1,
            "row": int(row),
            "col": int(col),
            "queue": {
                "queue_length": 3,
                "buffer_length": 1,
                "direction": "auto",
            },
        }
        for index, (row, col) in enumerate(positions)
    ]


def _pod_zones(rows: int, cols: int) -> list[dict[str, int]]:
    zone_rows, zone_cols = 6, 2
    row_max = rows - 4 - zone_rows
    col_max = cols - 4 - zone_cols
    if row_max < 4 or col_max < 4:
        raise ValueError(f"map too small for pod zones: {rows}x{cols}")
    starts_r = [4, max(4, row_max - 6)]
    starts_c = [int(round(4 + (col_max - 4) * i / 3.0)) for i in range(4)]
    starts = [(r, c) for r in starts_r for c in starts_c]
    if len(set(starts)) != 8:
        raise ValueError(f"pod zone anchors are not unique: {starts}")
    return [
        {"origin_row": int(r), "origin_col": int(c), "num_rows": zone_rows, "num_cols": zone_cols}
        for r, c in starts
    ]


def _derive_config(source: Mapping[str, Any]) -> dict[str, Any]:
    derived = copy.deepcopy(source)
    derived.setdefault("map", {})["rows"] = MAP_ROWS
    derived["map"]["cols"] = MAP_COLS
    derived["map"]["stations"] = _station_specs(MAP_ROWS, MAP_COLS)
    derived["map"].pop("pod_layout", None)
    derived["map"]["pod_zones"] = _pod_zones(MAP_ROWS, MAP_COLS)
    robots = derived.setdefault("robots", {})
    robots["num_robots"] = ROBOT_COUNT
    robots["starts"] = []
    robots["random_starts"] = True
    return derived


def _changed_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                result.append(child)
            else:
                result.extend(_changed_paths(left[key], right[key], child))
        return result
    if isinstance(left, list) and isinstance(right, list):
        result = []
        if len(left) != len(right):
            result.append(f"{prefix}.length")
        for index, (lvalue, rvalue) in enumerate(zip(left, right)):
            result.extend(_changed_paths(lvalue, rvalue, f"{prefix}[{index}]"))
        return result
    return [] if left == right else [prefix]


def _config_path(root: Path, load: str) -> Path:
    return root / "configs" / f"world_model_config_{load}.json"


def _protocol_path(root: Path) -> Path:
    return root / "station6_adaptation_protocol.json"


def _physical_only_mode_ok(station_audit: Mapping[str, Any]) -> bool:
    """Check the exact simulator admission-mode token.

    The historical token is ``physical_only_legacy`` (despite the shorter
    human-facing phrase used in reports).  Keeping the comparison in one
    helper prevents a false audit failure when the mode is otherwise correct.
    """

    from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY

    return station_audit.get("mode") == STATION_ADMISSION_PHYSICAL_ONLY


def _adapter_path(root: Path) -> Path:
    return root / "adapter" / "stage1_station6_adapted.pt"


def _adaptation_schema(checkpoint: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "state_dict" not in payload:
        raise ValueError(f"unsupported source checkpoint: {checkpoint}")
    cfg = dict(payload.get("model_config") or {})
    return {
        "source_demand_dim": int(cfg.get("demand_dim", -1)),
        "source_num_stations": int(cfg.get("num_stations", -1)),
        "target_demand_dim": 5 + STATION_COUNT,
        "target_num_stations": STATION_COUNT,
    }


def _prepare(root: Path, ticks: int = TICKS) -> dict[str, Any]:
    if int(ticks) <= 0:
        raise ValueError("ticks must be positive")
    root.mkdir(parents=True, exist_ok=True)
    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(SOURCE_CHECKPOINT)
    load_configs: dict[str, dict[str, Any]] = {}
    from WorldModel.evaluation.run_phase_c_scale_generalization import (
        VARIANTS as SCALE_VARIANTS,
        _derive_config as derive_scale_config,
    )

    scale_variant = next(
        variant for variant in SCALE_VARIANTS if variant.key == "map30_r72_s6"
    )
    for load in LOADS:
        source_path = Path(f"Config/world_model_config_PP_48_{load}.json")
        source = _read_json(source_path)
        derived = _derive_config(source)
        scale_reference = derive_scale_config(source, scale_variant)
        if derived != scale_reference:
            raise AssertionError(
                f"station6 adaptation config differs from scale baseline: {load}"
            )
        changed = _changed_paths(source, derived)
        allowed = {
            "map.rows", "map.cols", "map.stations", "map.pod_zones",
            "robots.num_robots", "robots.starts", "robots.random_starts",
        }
        if any(
            not any(path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[") for prefix in allowed)
            for path in changed
        ):
            raise AssertionError(f"config changed outside scale contract: {changed}")
        target = _config_path(root, load)
        _atomic_json(target, derived)
        load_configs[load] = {
            "source": source_path.as_posix(),
            "source_sha256": _sha256_file(source_path),
            "derived": target.as_posix(),
            "derived_sha256": _sha256_file(target),
            "changed_paths": changed,
            "matches_scale_variant": scale_variant.key,
        }

    schema = _adaptation_schema(SOURCE_CHECKPOINT)
    if schema["source_num_stations"] != 4 or schema["source_demand_dim"] != 9:
        raise ValueError(f"expected 4-station/9-channel source, got {schema}")
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "explicit six-station adaptation, not frozen zero-shot generalisation",
        "target": {
            "map_rows": MAP_ROWS,
            "map_cols": MAP_COLS,
            "robots": ROBOT_COUNT,
            "stations": STATION_COUNT,
            "demand_dim": 5 + STATION_COUNT,
        },
        "loads": list(LOADS),
        "core_data_seeds": list(CORE_DATA_SEEDS),
        "core_train_seeds": list(CORE_TRAIN_SEEDS),
        "core_val_seeds": list(CORE_VAL_SEEDS),
        "core_offline_test_seeds": list(CORE_TEST_SEEDS),
        "online_test_seeds": list(ONLINE_TEST_SEEDS),
        "repair_data_seeds": list(REPAIR_DATA_SEEDS),
        "repair_train_seeds": list(REPAIR_TRAIN_SEEDS),
        "repair_val_seeds": list(REPAIR_VAL_SEEDS),
        "repair_offline_test_seeds": list(REPAIR_TEST_SEEDS),
        "online_test_disjoint_from_all_data": not bool(
            (set(CORE_DATA_SEEDS) | set(REPAIR_DATA_SEEDS))
            & set(ONLINE_TEST_SEEDS)
        ),
        "core_and_repair_data_disjoint": not bool(
            set(CORE_DATA_SEEDS) & set(REPAIR_DATA_SEEDS)
        ),
        "ticks": int(ticks),
        "source_checkpoint": {
            "path": SOURCE_CHECKPOINT.as_posix(),
            "sha256": _sha256_file(SOURCE_CHECKPOINT),
            "schema": schema,
        },
        "frozen_code_inputs": {
            path.as_posix(): _sha256_file(path)
            for path in FROZEN_CODE_INPUTS
        },
        "adapter": {
            "method": "demand_encoder_station_channel_mean_expand_v1",
            "copied_columns": "all original five global plus four station channels",
            "new_columns": "two extra station channels initialized to mean of original four station columns",
            "all_other_tensors": "must remain bitwise equal in adapter",
        },
        "domain_adaptation": {
            "behavior_policy": "GreedyTaskAssigner",
            "counterfactual_candidate_mode": "stratified",
            "include_native_no_assign": True,
            "sample_interval": SNAPSHOT_INTERVAL,
            "top_m": TOP_M,
            "max_contexts_per_tick": MAX_CONTEXTS_PER_TICK,
            "short_rollout_horizon": ROLLOUT_HORIZON,
            "station_admission": "physical_only",
        },
        "phase_c_core": {
            "behavior_policy": "pure_world_model",
            "training_source_policy": "world_model_on_policy",
            "external_baseline_training_samples": False,
            "snapshot_seeds": list(CORE_DATA_SEEDS),
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "snapshot_top_m": TOP_M,
            "max_contexts_per_tick": MAX_CONTEXTS_PER_TICK,
            "candidate_robot_mode": "stratified",
            "include_native_no_assign": True,
            "isolated_counterfactual_horizon": ROLLOUT_HORIZON,
            "initial_checkpoint": "six_station_greedy_domain_adapted_world_model",
            "station_admission": "physical_only",
        },
        "long_risk": {
            "snapshot_behavior_policy": "trained_six_station_phase_c_core",
            "snapshot_seeds": list(REPAIR_DATA_SEEDS),
            "snapshot_data_disjoint_from_phase_c_core": True,
            "horizon": LONG_RISK_HORIZON,
            "terminal_window": LONG_RISK_TERMINAL_WINDOW,
            "continuation": "greedy",
            "head_training": "LongRiskHead-only",
        },
        "station_head": {
            "snapshot_seeds": list(REPAIR_DATA_SEEDS),
            "source_encoder": "trained_six_station_phase_c_core",
            "label_scope": "state_level_current_station_congestion",
        },
        "online_evaluation": {
            "seeds": list(ONLINE_TEST_SEEDS),
            "manifest_source": "GreedyTaskAssigner",
            "greedy_generates_one_manifest_per_load_seed": True,
            "hungarian_phasec_combo_exact_replay": True,
            "disjoint_from_adaptation_data": True,
        },
        "claims": {
            "frozen_zero_shot_six_station": False,
            "adapted_world_model": True,
            "phase_c_core_retrained_on_world_model_on_policy_data": True,
            "station_head_state_level_adaptation": True,
        },
        "load_configs": load_configs,
    }
    bundle = {"protocol": protocol, "protocol_sha256": _canonical_sha(protocol)}
    _atomic_json(_protocol_path(root), bundle)
    _write_adapter(root)
    return bundle


def _load_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = _read_json(_protocol_path(root))
    if bundle.get("protocol", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected station6 protocol schema")
    protocol = bundle["protocol"]
    if _canonical_sha(protocol) != bundle.get("protocol_sha256"):
        raise ValueError("station6 protocol hash mismatch")
    if tuple(protocol.get("core_data_seeds", ())) != CORE_DATA_SEEDS:
        raise ValueError("station6 core-data seed block changed")
    if (
        tuple(protocol.get("core_train_seeds", ())) != CORE_TRAIN_SEEDS
        or tuple(protocol.get("core_val_seeds", ())) != CORE_VAL_SEEDS
        or tuple(protocol.get("core_offline_test_seeds", ()))
        != CORE_TEST_SEEDS
    ):
        raise ValueError("station6 core train/val/test split changed")
    if tuple(protocol.get("online_test_seeds", ())) != ONLINE_TEST_SEEDS:
        raise ValueError("station6 online-test seed block changed")
    if tuple(protocol.get("repair_data_seeds", ())) != REPAIR_DATA_SEEDS:
        raise ValueError("station6 repair-data seed block changed")
    if (
        tuple(protocol.get("repair_train_seeds", ())) != REPAIR_TRAIN_SEEDS
        or tuple(protocol.get("repair_val_seeds", ())) != REPAIR_VAL_SEEDS
        or tuple(protocol.get("repair_offline_test_seeds", ()))
        != REPAIR_TEST_SEEDS
    ):
        raise ValueError("station6 repair train/val/test split changed")
    if (
        set(CORE_TRAIN_SEEDS)
        | set(CORE_VAL_SEEDS)
        | set(CORE_TEST_SEEDS)
    ) != set(CORE_DATA_SEEDS):
        raise ValueError("station6 core split does not cover its data seeds")
    if (
        set(REPAIR_TRAIN_SEEDS)
        | set(REPAIR_VAL_SEEDS)
        | set(REPAIR_TEST_SEEDS)
    ) != set(REPAIR_DATA_SEEDS):
        raise ValueError("station6 repair split does not cover its data seeds")
    all_data_seeds = set(CORE_DATA_SEEDS) | set(REPAIR_DATA_SEEDS)
    if all_data_seeds & set(ONLINE_TEST_SEEDS):
        raise ValueError("station6 online-test seeds overlap adaptation data")
    if set(CORE_DATA_SEEDS) & set(REPAIR_DATA_SEEDS):
        raise ValueError("station6 core and repair seed blocks overlap")
    if not bool(protocol.get("online_test_disjoint_from_all_data")):
        raise ValueError("station6 protocol does not certify held-out online seeds")
    if not bool(protocol.get("core_and_repair_data_disjoint")):
        raise ValueError("station6 protocol does not separate core/repair data")
    for load in LOADS:
        row = protocol["load_configs"][load]
        if _sha256_file(Path(row["source"])) != row["source_sha256"]:
            raise ValueError(f"source config changed: {load}")
        if _sha256_file(Path(row["derived"])) != row["derived_sha256"]:
            raise ValueError(f"derived config changed: {load}")
    if _sha256_file(SOURCE_CHECKPOINT) != protocol["source_checkpoint"]["sha256"]:
        raise ValueError("source checkpoint changed after protocol freeze")
    frozen_code = protocol.get("frozen_code_inputs") or {}
    expected_code_paths = {path.as_posix() for path in FROZEN_CODE_INPUTS}
    if set(frozen_code) != expected_code_paths:
        raise ValueError("station6 frozen-code input set changed")
    for path in FROZEN_CODE_INPUTS:
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256_file(path) != frozen_code[path.as_posix()]:
            raise ValueError(f"station6 frozen code changed: {path}")
    return bundle, protocol


def _require_protocol_ticks(protocol: Mapping[str, Any], ticks: int) -> None:
    expected = int(protocol.get("ticks", -1))
    if int(ticks) != expected:
        raise ValueError(
            f"requested ticks={ticks} differs from frozen protocol ticks={expected}"
        )


def _write_adapter(root: Path) -> None:
    """Create and audit the 4→6 demand-schema adapter."""

    output = _adapter_path(root)
    audit_path = output.with_suffix(".audit.json")
    if output.is_file() and audit_path.is_file():
        audit = _read_json(audit_path)
        if audit.get("passed") and audit.get("output_sha256") == _sha256_file(output):
            return
        raise ValueError(f"existing station6 adapter failed audit: {output}")

    import torch
    from WorldModel.core.model import RMFSWorldModel

    source_payload = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    if not isinstance(source_payload, dict):
        raise ValueError("source checkpoint payload must be a dictionary")
    source_state = source_payload.get("state_dict")
    if not isinstance(source_state, Mapping):
        raise ValueError("source checkpoint lacks state_dict")
    key = "demand_encoder.net.0.weight"
    old_weight = source_state.get(key)
    if old_weight is None or tuple(old_weight.shape)[1] != 9:
        raise ValueError(f"unexpected demand encoder tensor: {key} {getattr(old_weight, 'shape', None)}")
    new_weight = torch.empty((old_weight.shape[0], 11), dtype=old_weight.dtype)
    new_weight[:, :9] = old_weight
    # The input order is [five global channels, station_1..station_S].  A
    # neutral mean initialization avoids privileging either new station.
    new_weight[:, 9:] = old_weight[:, 5:9].mean(dim=1, keepdim=True)
    target_state = {name: value.detach().clone() for name, value in source_state.items()}
    target_state[key] = new_weight
    target_payload = copy.deepcopy(source_payload)
    target_payload["state_dict"] = target_state
    cfg = dict(target_payload.get("model_config") or {})
    cfg["demand_dim"] = 11
    cfg["num_stations"] = STATION_COUNT
    target_payload["model_config"] = cfg
    target_payload["station6_adapter"] = {
        "schema_version": "station6_demand_schema_adapter_v1",
        "source_checkpoint": SOURCE_CHECKPOINT.as_posix(),
        "source_checkpoint_sha256": _sha256_file(SOURCE_CHECKPOINT),
        "source_shape": [int(old_weight.shape[0]), int(old_weight.shape[1])],
        "target_shape": [int(new_weight.shape[0]), int(new_weight.shape[1])],
        "new_channel_initialization": "mean(original_station_columns_1_to_4)",
    }
    _atomic_torch_save(output, target_payload)
    # Strict-load validates all dimensions, including station_node_ids-related
    # decoder construction, without changing any runtime module.
    model = RMFSWorldModel(**cfg)
    model.load_state_dict(target_state, strict=True)
    checks = {
        "source_checkpoint_schema": tuple(old_weight.shape) == (64, 9),
        "target_demand_dim": tuple(new_weight.shape) == (64, 11),
        "copied_original_columns_bitwise": bool(torch.equal(new_weight[:, :9], old_weight)),
        "new_columns_finite": bool(torch.isfinite(new_weight[:, 9:]).all()),
        "strict_model_load": True,
        "non_demand_tensors_bitwise_equal": all(
            torch.equal(value, source_state[name])
            for name, value in target_state.items() if name != key
        ),
        "model_config_target": cfg.get("demand_dim") == 11 and cfg.get("num_stations") == 6,
    }
    audit = {
        "schema_version": "station6_demand_schema_adapter_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "output": output.as_posix(),
        "output_sha256": _sha256_file(output),
        "source_sha256": _sha256_file(SOURCE_CHECKPOINT),
        "changed_tensor": key,
    }
    _atomic_json(audit_path, audit)
    if not audit["passed"]:
        raise RuntimeError(f"station6 adapter audit failed: {audit}")


def _config(protocol: Mapping[str, Any], load: str) -> Path:
    if load not in LOADS:
        raise ValueError(load)
    return Path(protocol["load_configs"][load]["derived"])


def _base_manifest_path(root: Path, load: str, seed: int) -> Path:
    return root / "base_manifests" / load / f"orders_{load}_seed{seed}.json"


def _snapshot_manifest_path(
    root: Path, stage: str, load: str, seed: int
) -> Path:
    if stage not in ("core", "repair"):
        raise ValueError(stage)
    return (
        root
        / f"{stage}_manifests"
        / load
        / f"orders_{load}_seed{seed}.json"
    )


def _eval_manifest_path(root: Path, load: str, seed: int) -> Path:
    return (
        root
        / "evaluation"
        / "manifests"
        / load
        / f"orders_{load}_seed{seed}.json"
    )


def _base_dir(root: Path, load: str, seed: int) -> Path:
    return root / "base_data" / load / f"seed{seed}"


def _snapshot_dir(root: Path, stage: str, load: str, seed: int) -> Path:
    if stage not in ("core", "repair"):
        raise ValueError(stage)
    return root / f"{stage}_wm_snapshots" / load / f"seed{seed}"


def _snapshot_data_dir(root: Path, stage: str, load: str, seed: int) -> Path:
    if stage not in ("core", "repair"):
        raise ValueError(stage)
    return root / f"{stage}_snapshot_data" / load / f"seed{seed}"


def _label_path(root: Path, load: str, seed: int) -> Path:
    return root / "long_risk_labels" / load / f"seed{seed}" / "long_risk_labels.pt"


def _check_manifest(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema_version") != "order_arrival_manifest_v1":
        # The helper's schema has changed historically; require the fields
        # that make a paired replay safe instead of guessing a version.
        for key in ("manifest_sha256", "total_orders", "orders"):
            if key not in value:
                raise ValueError(f"invalid order manifest: {path}")
    return value


def _run_base_cell(root: Path, load: str, seed: int, ticks: int = TICKS) -> None:
    if int(seed) not in CORE_DATA_SEEDS:
        raise ValueError(f"base/core seed must be in {list(CORE_DATA_SEEDS)}")
    bundle, protocol = _load_protocol(root)
    _require_protocol_ticks(protocol, ticks)
    out_dir = _base_dir(root, load, seed)
    data_path = out_dir / "data.pt"
    gen_config_path = out_dir / "gen_config.json"
    summary_path = out_dir / "collection_summary.json"
    manifest_path = _base_manifest_path(root, load, seed)
    if data_path.is_file() and gen_config_path.is_file() and summary_path.is_file() and manifest_path.is_file():
        summary = _read_json(summary_path)
        if summary.get("protocol_sha256") == bundle["protocol_sha256"] and summary.get("audit", {}).get("passed"):
            print(f"[resume] base collection {load} seed={seed}")
            return
        raise ValueError(f"incompatible existing base collection: {out_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"partial base collection exists: {out_dir}")

    import torch
    from Config.config_loader import load_config
    from Policies.TaskAssigner import GreedyTaskAssigner
    from WorldModel.data.data_collector import WorldModelDataCollector
    # Import the online evaluation module itself rather than copying the
    # builder symbol.  The physical-only audit contract temporarily wraps
    # ``evaluate_online_v6._build_engine``; importing the function directly
    # here would bypass that wrapper and make the audit appear empty even
    # though the simulator was otherwise configured correctly.
    import WorldModel.evaluation.evaluate_online_v6 as evaluate_online
    from WorldModel.evaluation.evaluate_online_v6 import (
        _order_arrival_manifest,
        reset_global_ids,
        set_global_seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(seed))
    reset_global_ids()
    cfg = load_config(str(_config(protocol, load)))
    cfg.simulation.max_ticks = int(ticks)
    cfg.simulation.seed = int(seed)
    replayed = manifest_path.is_file()
    if replayed:
        cfg.policies.order_generator = (
            "RecordedOrderGenerator",
            {"recorded_orders_path": str(manifest_path), "immediate_dispatch": False},
        )
        cfg.simulation.initial_order_pool_size = 0
        cfg.simulation.backlog_floor = 0
        cfg.simulation.backlog_refill_mode = "none"
    assigner = GreedyTaskAssigner()
    collector = WorldModelDataCollector(
        output_dir=str(out_dir),
        history_len=4,
        rollout_horizon=ROLLOUT_HORIZON,
        sample_interval=SNAPSHOT_INTERVAL,
        top_m_candidates=TOP_M,
        min_group_size=2,
        reservation_window=1,
        max_groups_per_tick=MAX_CONTEXTS_PER_TICK,
        candidate_robot_mode="stratified",
        include_no_assign_candidate=True,
    )
    from WorldModel.evaluation.run_phase_c_s1_j1_long_risk_correction import _physical_only_audit_contract
    with _physical_only_audit_contract() as holder:
        # Build the engine while the audit contract is active.  The contract
        # wraps the online builder and attaches the station probe at
        # construction time; building it before entering the context would
        # leave ``holder['station_probe']`` unset.
        engine = evaluate_online._build_engine(cfg, task_assigner=assigner)
        engine.pre_assignment_callbacks.append(collector.on_pre_assignment)
        engine.on_tick_callbacks.append(collector.on_post_tick)
        engine.run()
    manifest = _order_arrival_manifest(engine)
    if manifest_path.is_file():
        if _check_manifest(manifest_path) != manifest:
            raise ValueError(f"recorded manifest changed during base collection: {manifest_path}")
    else:
        _atomic_json(manifest_path, manifest)
    samples = collector._finalized_samples
    if not samples:
        raise RuntimeError(f"base collection produced no samples: {load} seed={seed}")
    run_id = f"station6_base_{load}_seed{seed}"
    for sample in samples:
        sample["run_id"] = run_id
        sample["source_seed"] = int(seed)
        sample["source_load_level"] = load
    _atomic_torch_save(data_path, samples)
    _atomic_json(gen_config_path, {
        "schema_version": "station6_base_collection_v1",
        "run_id": run_id,
        "seed": int(seed),
        "load_level": load,
        "stations": STATION_COUNT,
        "demand_dim": 5 + STATION_COUNT,
        "behavior_policy": "GreedyTaskAssigner",
        "station_admission": "physical_only",
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _sha256_file(manifest_path),
    })
    station_audit = holder.get("station_probe").summary() if holder.get("station_probe") else {}
    checks = {
        "nonempty": len(samples) > 0,
        "six_stations": int(torch.as_tensor(samples[0]["station_node_ids"]).numel()) == STATION_COUNT,
        "demand_dim_11": int(torch.as_tensor(samples[0]["demand_context"]).numel()) == 11,
        "manifest_hash": manifest.get("manifest_sha256") == _check_manifest(manifest_path).get("manifest_sha256"),
        "station_audit_passed": bool(station_audit.get("passed")),
        "physical_only": _physical_only_mode_ok(station_audit),
        "physical_capacity_no_violation": int(
            station_audit.get("physical_capacity_violation_count", -1)
        )
        == 0,
        "no_committed_cap_contract": station_audit.get(
            "committed_capacity_contract"
        )
        == "not enforced for in-transit commitments",
    }
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "protocol_sha256": bundle["protocol_sha256"],
        "load": load,
        "seed": int(seed),
        "ticks": int(ticks),
        "data_path": data_path.as_posix(),
        "data_sha256": _sha256_file(data_path),
        "gen_config_path": gen_config_path.as_posix(),
        "gen_config_sha256": _sha256_file(gen_config_path),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _sha256_file(manifest_path),
        "samples": len(samples),
        "metrics": {"completed_orders": len(engine.world.order_state.get_completed_orders())},
        "station_audit": station_audit,
        "audit": {"passed": all(checks.values()), "checks": checks},
    }
    _atomic_json(summary_path, summary)
    if not summary["audit"]["passed"]:
        raise RuntimeError(f"base collection audit failed: {summary}")
    print(f"[complete] base collection {load} seed={seed} samples={len(samples)}")


def _all_base_data(root: Path) -> list[Path]:
    return [
        _base_dir(root, load, seed) / "data.pt"
        for load in LOADS for seed in CORE_DATA_SEEDS
    ]


def _all_snapshot_data(root: Path, stage: str) -> list[Path]:
    seeds = CORE_DATA_SEEDS if stage == "core" else REPAIR_DATA_SEEDS
    return [
        _snapshot_data_dir(root, stage, load, seed) / "data.pt"
        for load in LOADS for seed in seeds
    ]


def _fuse_base(root: Path) -> Path:
    _, protocol = _load_protocol(root)
    output = root / "base_fused_seed_split_v1"
    data = _all_base_data(root)
    if not all(path.is_file() for path in data):
        raise FileNotFoundError("base data collection is incomplete")
    required = [output / "wm_train_data_fused.pt", output / "splits.json", output / "quality_report.json"]
    if all(path.is_file() for path in required):
        return output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"partial base fusion exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "WorldModel.data.fuse_and_split",
        "--inputs", *[str(path) for path in data],
        "--output-dir", str(output), "--output-name", "wm_train_data_fused.pt",
        "--split-unit", "source_seed",
        "--train-seeds", ",".join(map(str, CORE_TRAIN_SEEDS)),
        "--val-seeds", ",".join(map(str, CORE_VAL_SEEDS)),
        "--test-seeds", ",".join(map(str, CORE_TEST_SEEDS)),
    ]
    _run(command, cwd=_repo_root(), log=root / "logs" / "fuse_base.log")
    _atomic_json(root / "base_fusion_lineage.json", {
        "protocol_sha256": _canonical_sha(protocol),
        "inputs": [{"path": path.as_posix(), "sha256": _sha256_file(path)} for path in data],
        "output": [path.as_posix() for path in required],
    })
    return output


def _train_base(root: Path) -> Path:
    _load_protocol(root)
    _write_adapter(root)
    fused = _fuse_base(root)
    data = fused / "wm_train_data_fused.pt"
    splits = fused / "splits.json"
    model_root = root / "training" / "world_model_v1"
    lineage_path = root / "base_training_lineage.json"
    checkpoint = model_root / "best_regret_world_model.pt"
    if not checkpoint.is_file():
        checkpoint = model_root / "world_model.pt"
    if checkpoint.is_file() and (model_root / "train_summary.json").is_file():
        if not lineage_path.is_file():
            raise FileNotFoundError(
                f"completed base checkpoint lacks lineage: {lineage_path}"
            )
        lineage = _read_json(lineage_path)
        if (
            lineage.get("checkpoint_sha256") == _sha256_file(checkpoint)
            and lineage.get("source_adapter_sha256")
            == _sha256_file(_adapter_path(root))
        ):
            return checkpoint
        raise ValueError(f"existing base checkpoint failed lineage audit: {checkpoint}")
    if model_root.exists() and any(model_root.iterdir()):
        raise FileExistsError(f"partial base model training exists: {model_root}")
    command = [
        sys.executable, "-m", "WorldModel.training.run_train_v6",
        "--data", str(data), "--splits", str(splits), "--save-dir", str(model_root),
        "--skip-stage1", "--stage1-checkpoint", str(_adapter_path(root)),
        "--stage2-horizon", str(ROLLOUT_HORIZON), "--stage2-epochs", "30",
        "--stage2-lr", "1e-4", "--stage2-alpha-rank", "0.1",
        "--stage2-freeze-epochs", "0", "--stage2-unfreeze-mode", "all",
        "--stage2-val-dynamics-interval", "1", "--early-stopping-patience", "8",
        "--early-stopping-monitor", "val_top1_regret_mean",
        "--max-train-pairs", "20000", "--max-val-pairs", "5000",
        "--balanced-sampler", "--pair-balance-by", "source_load_level",
        "--alpha-long-risk", "0.0", "--device", "cpu",
    ]
    _run(
        command,
        cwd=_repo_root(),
        log=root / "logs" / "train_base.log",
        env_overrides=_cpu_training_env(),
    )
    checkpoint = model_root / "best_regret_world_model.pt"
    if not checkpoint.is_file():
        checkpoint = model_root / "world_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("base training emitted no checkpoint")
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("model_config") or {}
    if int(cfg.get("demand_dim", -1)) != 11 or int(cfg.get("num_stations", -1)) != STATION_COUNT:
        raise ValueError(f"base model schema is not six-station: {cfg}")
    _atomic_json(lineage_path, {
        "source_adapter": _adapter_path(root).as_posix(),
        "source_adapter_sha256": _sha256_file(_adapter_path(root)),
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "model_config": cfg,
        "cpu_training_threads": int(_cpu_training_env()["OMP_NUM_THREADS"]),
        "long_risk_head_status": "not supervised in this stage; repaired in later stage",
    })
    return checkpoint


def _snapshot_stage_contract(
    root: Path, stage: str
) -> tuple[tuple[int, ...], Path, str, str]:
    if stage == "core":
        return (
            CORE_DATA_SEEDS,
            _train_base(root),
            "Station6PhaseCCoreBehavior",
            "station6_phase_c_core",
        )
    if stage == "repair":
        return (
            REPAIR_DATA_SEEDS,
            _train_phasec_core(root),
            "Station6PhaseCRepairBehavior",
            "station6_phase_c_repair",
        )
    raise ValueError(stage)


def _run_snapshot_cell(
    root: Path, stage: str, load: str, seed: int, ticks: int = TICKS
) -> None:
    bundle, protocol = _load_protocol(root)
    _require_protocol_ticks(protocol, ticks)
    allowed_seeds, checkpoint, trace_label, phase_round = (
        _snapshot_stage_contract(root, stage)
    )
    if int(seed) not in allowed_seeds:
        raise ValueError(
            f"{stage} snapshot seed must be in {list(allowed_seeds)}"
        )
    out = _snapshot_dir(root, stage, load, seed)
    summary_path = out / "collection_summary.json"
    manifest = _snapshot_manifest_path(root, stage, load, seed)
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if (
            summary.get("protocol_sha256") == bundle["protocol_sha256"]
            and summary.get("checkpoint_sha256") == _sha256_file(checkpoint)
            and summary.get("audit", {}).get("passed")
            and manifest.is_file()
        ):
            print(f"[resume] {stage} WM snapshots {load} seed={seed}")
            return
        raise ValueError(f"incompatible snapshot collection: {out}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"partial snapshot collection exists: {out}")
    if manifest.exists():
        raise FileExistsError(
            f"orphan snapshot manifest exists without a completed run: {manifest}"
        )
    out.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    from Policies.TaskAssigner import WorldModelTaskAssigner
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
    from WorldModel.evaluation.run_phase_c_s1_j1_long_risk_correction import (
        _physical_only_audit_contract,
    )

    # Match the formal Phase-C lineage: the online behaviour policy chooses
    # with the pure S0 scorer (top_m=1), while the observation-only snapshot
    # probe records a wider stratified candidate set plus native NO_ASSIGN.
    assigner = WorldModelTaskAssigner(
        checkpoint_path=str(checkpoint),
        top_m=1,
        include_no_assign_candidate=False,
    )
    snapshot_path = out / "snapshots"
    run_id = f"station6_{stage}_{load}_seed{seed}"
    with _physical_only_audit_contract() as holder:
        metrics = _run_one_assigner(
            str(_config(protocol, load)),
            assigner,
            int(seed),
            int(ticks),
            trace_label=trace_label,
            decision_snapshot_dir=str(snapshot_path),
            decision_snapshot_interval=SNAPSHOT_INTERVAL,
            decision_snapshot_top_m=TOP_M,
            decision_snapshot_run_id=run_id,
            decision_snapshot_meta={
                "arm_label": trace_label,
                "load": load,
                "seed": int(seed),
                "config": _config(protocol, load).as_posix(),
                "checkpoint_path": checkpoint.as_posix(),
                "checkpoint_sha256": _sha256_file(checkpoint),
                "phase_c_round": phase_round,
                "protocol_sha256": bundle["protocol_sha256"],
                "external_baseline_training_samples": False,
            },
            decision_snapshot_candidate_scope="top_m_snapshot",
            decision_snapshot_capture_policy="interval_all",
            decision_snapshot_candidate_robot_mode="stratified",
            decision_snapshot_include_no_assign=True,
            decision_snapshot_max_contexts_per_tick=MAX_CONTEXTS_PER_TICK,
            decision_snapshot_phase_c_round=phase_round,
            decision_snapshot_exclude_external_baselines=True,
            save_order_manifest=str(manifest),
        )
    station_audit = (
        holder.get("station_probe").summary()
        if holder.get("station_probe")
        else {}
    )
    indexes = list(snapshot_path.glob("snapindex_*.json"))
    snapshot_files = sorted(snapshot_path.glob("*.pkl"))
    manifest_payload = _check_manifest(manifest)
    snapshot_schema_checks = {
        "six_stations": False,
        "demand_dim_11": False,
        "world_model_on_policy": False,
        "external_baselines_excluded": False,
    }
    if snapshot_files:
        with snapshot_files[0].open("rb") as handle:
            first_snapshot = pickle.load(handle)
        import torch

        snapshot_schema_checks = {
            "six_stations": int(
                torch.as_tensor(first_snapshot["station_node_ids"]).numel()
            )
            == STATION_COUNT,
            "demand_dim_11": int(
                torch.as_tensor(first_snapshot["demand_context"]).numel()
            )
            == 11,
            "world_model_on_policy": first_snapshot.get(
                "training_source_policy"
            )
            == "world_model_on_policy",
            "external_baselines_excluded": not bool(
                first_snapshot.get("external_baseline_training_samples")
            ),
        }
    checks = {
        "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
        "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
        "snapshots": (
            int(metrics.get("decision_snapshots_saved", 0)) > 0
            and len(indexes) == 1
            and bool(snapshot_files)
        ),
        "manifest_generated_by_world_model_run": not bool(
            metrics.get("order_arrival_replayed")
        ),
        "manifest": metrics.get("order_arrival_manifest_sha256")
        == manifest_payload.get("manifest_sha256"),
        "station_audit_passed": bool(station_audit.get("passed")),
        "physical_only": _physical_only_mode_ok(station_audit),
        "physical_capacity_no_violation": int(
            station_audit.get("physical_capacity_violation_count", -1)
        )
        == 0,
        "no_committed_cap_contract": station_audit.get(
            "committed_capacity_contract"
        )
        == "not enforced for in-transit commitments",
        **snapshot_schema_checks,
    }
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "protocol_sha256": bundle["protocol_sha256"],
        "stage": stage,
        "load": load,
        "seed": int(seed),
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "manifest": manifest.as_posix(),
        "manifest_sha256": _sha256_file(manifest),
        "snapshot_dir": snapshot_path.as_posix(),
        "metrics": metrics,
        "station_audit": station_audit,
        "audit": {"passed": all(checks.values()), "checks": checks},
    }
    _atomic_json(summary_path, summary)
    if not summary["audit"]["passed"]:
        raise RuntimeError(f"snapshot collection audit failed: {summary}")
    print(
        f"[complete] {stage} WM snapshots {load} seed={seed} "
        f"snapshots={metrics.get('decision_snapshots_saved')}"
    )


def _run_label_cell(root: Path, load: str, seed: int) -> None:
    if int(seed) not in REPAIR_DATA_SEEDS:
        raise ValueError(
            f"long-risk seed must be in {list(REPAIR_DATA_SEEDS)}"
        )
    snapshot_path = _snapshot_dir(root, "repair", load, seed) / "snapshots"
    output = _label_path(root, load, seed)
    marker = output.with_suffix(".json")
    if output.is_file() and marker.is_file():
        return
    if not snapshot_path.is_dir():
        raise FileNotFoundError(snapshot_path)
    command = [
        sys.executable, "-m", "WorldModel.data.generate_long_risk_labels",
        "--snapshot-dir", str(snapshot_path), "--output", str(output),
        "--W", str(LONG_RISK_HORIZON), "--terminal-window", str(LONG_RISK_TERMINAL_WINDOW),
        "--snapshot-sampling", "uniform", "--snapshot-seed", "20260824",
        "--continuation-policy", "greedy",
    ]
    _run(command, cwd=_repo_root(), log=root / "logs" / f"longrisk_{load}_seed{seed}.log")
    import torch
    labels = torch.load(output, map_location="cpu", weights_only=False)
    if not isinstance(labels, list) or not labels:
        raise RuntimeError(f"long-risk label generation produced no labels: {output}")
    _atomic_json(marker, {
        "schema_version": "station6_longrisk_label_run_v1",
        "output": output.as_posix(), "sha256": _sha256_file(output),
        "labels": len(labels), "horizon": LONG_RISK_HORIZON,
    })


def _run_snapshot_data_cell(
    root: Path, stage: str, load: str, seed: int
) -> None:
    allowed_seeds = (
        CORE_DATA_SEEDS if stage == "core" else REPAIR_DATA_SEEDS
        if stage == "repair"
        else ()
    )
    if int(seed) not in allowed_seeds:
        raise ValueError(
            f"{stage} dataset seed must be in {list(allowed_seeds)}"
        )
    snapshot_path = _snapshot_dir(root, stage, load, seed) / "snapshots"
    output = _snapshot_data_dir(root, stage, load, seed)
    data = output / "data.pt"
    if data.is_file() and (output / "data_meta.json").is_file():
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"partial snapshot data exists: {output}")
    command = [
        sys.executable, "-m", "WorldModel.data.build_phase_c_round1_dataset",
        "--snapshot-dir", str(snapshot_path),
        "--lyapunov-config", "Config/lyapunov_l1_config.json",
        "--output-dir", str(output), "--output-name", "data.pt",
        "--horizon", str(ROLLOUT_HORIZON), "--formal",
    ]
    _run(
        command,
        cwd=_repo_root(),
        log=root / "logs" / f"{stage}_snapshot_data_{load}_seed{seed}.log",
    )


def _fuse_core(root: Path) -> Path:
    output = root / "phasec_core_fused_seed_split_v1"
    data = _all_snapshot_data(root, "core")
    if not all(path.is_file() for path in data):
        raise FileNotFoundError("Phase-C core snapshot datasets are incomplete")
    required = [
        output / "phase_c_round1_fused.pt",
        output / "splits.json",
        output / "quality_report.json",
    ]
    if all(path.is_file() for path in required):
        return output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"partial Phase-C core fusion exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "WorldModel.data.fuse_and_split",
        "--inputs",
        *[str(path) for path in data],
        "--output-dir",
        str(output),
        "--output-name",
        "phase_c_round1_fused.pt",
        "--require-lyapunov-l0-schema",
        "lyapunov_l1_collection_v3",
        "--split-unit",
        "source_seed",
        "--train-seeds",
        ",".join(map(str, CORE_TRAIN_SEEDS)),
        "--val-seeds",
        ",".join(map(str, CORE_VAL_SEEDS)),
        "--test-seeds",
        ",".join(map(str, CORE_TEST_SEEDS)),
    ]
    _run(command, cwd=_repo_root(), log=root / "logs" / "fuse_phasec_core.log")
    _atomic_json(
        root / "phasec_core_fusion_lineage.json",
        {
            "protocol_sha256": _load_protocol(root)[0]["protocol_sha256"],
            "training_source_policy": "world_model_on_policy",
            "external_baseline_training_samples": False,
            "inputs": [
                {"path": path.as_posix(), "sha256": _sha256_file(path)}
                for path in data
            ],
            "outputs": [path.as_posix() for path in required],
        },
    )
    return output


def _train_phasec_core(root: Path) -> Path:
    base = _train_base(root)
    fused = _fuse_core(root)
    output = root / "training" / "phasec_core_v1"
    checkpoint = output / "best_regret_world_model.pt"
    if not checkpoint.is_file():
        checkpoint = output / "world_model.pt"
    lineage_path = root / "phasec_core_training_lineage.json"
    if checkpoint.is_file() and lineage_path.is_file():
        lineage = _read_json(lineage_path)
        if (
            lineage.get("audit", {}).get("passed")
            and lineage.get("checkpoint_sha256") == _sha256_file(checkpoint)
        ):
            return checkpoint
        raise ValueError(f"existing Phase-C core failed lineage audit: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"partial Phase-C core training exists: {output}")
    command = [
        sys.executable,
        "-m",
        "WorldModel.training.run_train_v6",
        "--data",
        str(fused / "phase_c_round1_fused.pt"),
        "--splits",
        str(fused / "splits.json"),
        "--save-dir",
        str(output),
        "--skip-stage1",
        "--stage1-checkpoint",
        str(base),
        "--stage2-horizon",
        str(ROLLOUT_HORIZON),
        "--stage2-epochs",
        "30",
        "--stage2-lr",
        "1e-4",
        "--stage2-alpha-rank",
        "0.1",
        "--stage2-freeze-epochs",
        "0",
        "--stage2-unfreeze-mode",
        "all",
        "--stage2-val-dynamics-interval",
        "1",
        "--early-stopping-patience",
        "8",
        "--early-stopping-monitor",
        "val_top1_regret_mean",
        "--max-train-pairs",
        "20000",
        "--max-val-pairs",
        "5000",
        "--balanced-sampler",
        "--pair-balance-by",
        "source_load_level",
        "--alpha-long-risk",
        "0.0",
        "--device",
        "cpu",
    ]
    _run(
        command,
        cwd=_repo_root(),
        log=root / "logs" / "train_phasec_core.log",
        env_overrides=_cpu_training_env(),
    )
    checkpoint = output / "best_regret_world_model.pt"
    if not checkpoint.is_file():
        checkpoint = output / "world_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("Phase-C core training emitted no checkpoint")

    import torch

    base_payload = torch.load(base, map_location="cpu", weights_only=False)
    core_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base_state = base_payload.get("state_dict") or {}
    core_state = core_payload.get("state_dict") or {}
    if set(base_state) != set(core_state):
        raise ValueError("base/core checkpoint tensor keys differ")
    long_risk_keys = sorted(
        key for key in core_state if key.startswith("long_risk_head.")
    )
    non_long_risk_keys = sorted(
        key for key in core_state if not key.startswith("long_risk_head.")
    )
    cfg = core_payload.get("model_config") or {}
    action_schema = core_payload.get("action_schema") or {}
    checks = {
        "six_station_schema": (
            int(cfg.get("demand_dim", -1)) == 11
            and int(cfg.get("num_stations", -1)) == STATION_COUNT
        ),
        "native_no_assign_schema": bool(
            action_schema.get("supports_no_assign_candidate")
        )
        and bool(action_schema.get("complete_group_coverage")),
        "long_risk_unsupervised_and_bitwise_preserved": bool(long_risk_keys)
        and all(
            torch.equal(base_state[key], core_state[key])
            for key in long_risk_keys
        ),
        "phase_c_non_head_training_occurred": any(
            not torch.equal(base_state[key], core_state[key])
            for key in non_long_risk_keys
        ),
    }
    lineage = {
        "schema_version": "station6_phase_c_core_training_lineage_v1",
        "protocol_sha256": _load_protocol(root)[0]["protocol_sha256"],
        "initial_checkpoint": base.as_posix(),
        "initial_checkpoint_sha256": _sha256_file(base),
        "training_data": (fused / "phase_c_round1_fused.pt").as_posix(),
        "training_source_policy": "world_model_on_policy",
        "external_baseline_training_samples": False,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "model_config": cfg,
        "action_schema": action_schema,
        "cpu_training_threads": int(_cpu_training_env()["OMP_NUM_THREADS"]),
        "audit": {"passed": all(checks.values()), "checks": checks},
    }
    _atomic_json(lineage_path, lineage)
    if not lineage["audit"]["passed"]:
        raise RuntimeError(f"Phase-C core lineage audit failed: {checks}")
    return checkpoint


def _fuse_longrisk(root: Path) -> Path:
    _, protocol = _load_protocol(root)
    output = root / "longrisk_fused_seed_split_v1"
    data = _all_snapshot_data(root, "repair")
    labels = [
        _label_path(root, load, seed)
        for load in LOADS for seed in REPAIR_DATA_SEEDS
    ]
    if not all(path.is_file() for path in data + labels):
        raise FileNotFoundError("snapshot data/long-risk labels are incomplete")
    required = [output / "phase_c_round1_fused.pt", output / "splits.json", output / "quality_report.json"]
    if all(path.is_file() for path in required):
        return output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"partial long-risk fusion exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "WorldModel.data.fuse_and_split",
        "--inputs", *[str(path) for path in data],
        "--long-risk-inputs", *[str(path) for path in labels],
        "--output-dir", str(output), "--output-name", "phase_c_round1_fused.pt",
        "--require-lyapunov-l0-schema", "lyapunov_l1_collection_v3",
        "--split-unit", "source_seed",
        "--train-seeds", ",".join(map(str, REPAIR_TRAIN_SEEDS)),
        "--val-seeds", ",".join(map(str, REPAIR_VAL_SEEDS)),
        "--test-seeds", ",".join(map(str, REPAIR_TEST_SEEDS)),
    ]
    _run(command, cwd=_repo_root(), log=root / "logs" / "fuse_longrisk.log")
    _atomic_json(root / "longrisk_fusion_lineage.json", {
        "protocol_sha256": _load_protocol(root)[0]["protocol_sha256"],
        "inputs": [{"data": d.as_posix(), "labels": l.as_posix(), "data_sha256": _sha256_file(d), "labels_sha256": _sha256_file(l)} for d, l in zip(data, labels)],
        "outputs": [p.as_posix() for p in required],
    })
    return output


def _train_longrisk(root: Path) -> Path:
    phasec_core = _train_phasec_core(root)
    stage1_reference = _train_base(root)
    fused = _fuse_longrisk(root)
    output = root / "training" / "long_risk_head_only_v1"
    checkpoint = output / "best_long_risk_world_model.pt"
    audit_path = output / "tensor_audit.json"
    summary_path = output / "train_summary.json"
    if checkpoint.is_file() and audit_path.is_file() and summary_path.is_file():
        audit = _read_json(audit_path)
        summary = _read_json(summary_path)
        source = summary.get("source_checkpoint") or {}
        stage1 = summary.get("stage1_reference") or {}
        if (
            audit.get("passed")
            and source.get("sha256") == _sha256_file(phasec_core)
            and stage1.get("sha256") == _sha256_file(stage1_reference)
            and summary.get("output_checkpoint_sha256")
            == _sha256_file(checkpoint)
        ):
            return checkpoint
        raise ValueError(f"existing LongRiskHead repair failed audit: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"partial long-risk training exists: {output}")
    command = [
        sys.executable, "-m", "WorldModel.training.train_long_risk_head_only",
        "--checkpoint", str(phasec_core),
        "--stage1-reference", str(stage1_reference),
        "--data", str(fused / "phase_c_round1_fused.pt"),
        "--splits", str(fused / "splits.json"), "--output-root", str(output),
        "--epochs", "80", "--batch-size", "1024", "--eval-batch-size", "4096",
        "--learning-rate", "1e-3", "--weight-decay", "1e-4", "--patience", "15",
        "--device", "cpu", "--torch-threads", "4",
    ]
    _run(
        command,
        cwd=_repo_root(),
        log=root / "logs" / "train_longrisk.log",
        env_overrides=_cpu_training_env(max_threads=4),
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _station_dataset(root: Path) -> Path:
    """Build a station-region latent dataset directly from WM snapshots.

    This is intentionally state-level: it trains the same shared
    StationCongestionHead used by J1, but does not claim the separate H=10
    endpoint transport certificate from the original four-station campaign.
    """

    _, protocol = _load_protocol(root)
    protocol_ticks = int(protocol["ticks"])
    final_model = _train_longrisk(root)
    output = root / "station_head_dataset" / "station_congestion_latents.pt"
    scale_path = output.parent / "station_congestion_scale_contract.json"
    if output.is_file() and scale_path.is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    import torch
    from WorldModel.core.station_congestion_head import (
        COMPONENT_NAMES, CHANNEL_NAMES, build_station_targets, extract_station_components, fit_scale_contract,
    )
    from WorldModel.data.build_station_congestion_head_dataset import (
        REPRESENTATION_STATION_REGION_MEAN_MAX, build_station_representations,
    )
    from WorldModel.evaluation.station_congestion_endpoint import build_endpoint_station_rows, build_station_layout
    from WorldModel.evaluation.evaluate import _load_model
    model, _ = _load_model(str(final_model))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    records: list[dict[str, Any]] = []
    run_index = 0
    frame_group_counter = 0
    for load_index, load in enumerate(LOADS):
        for seed in REPAIR_DATA_SEEDS:
            snap_dir = _snapshot_dir(root, "repair", load, seed) / "snapshots"
            files = sorted(snap_dir.glob("*.pkl"))
            for path in files:
                with path.open("rb") as handle:
                    snapshot = pickle.load(handle)
                world = snapshot["world_snapshot"]
                edge_index = torch.as_tensor(snapshot["edge_index"], dtype=torch.long)
                node_history = torch.as_tensor(snapshot["node_history"], dtype=torch.float32)
                edge_features = torch.as_tensor(snapshot["edge_features"], dtype=torch.float32)
                demand = torch.as_tensor(snapshot["demand_context"], dtype=torch.float32)
                from WorldModel.graph.graph_builder import build_static_graph
                (_, node_map, _, _, _, _, _) = build_static_graph(world.map_state)
                layout = build_station_layout(world, node_map, edge_index)
                with torch.no_grad():
                    z, _, _ = model.encode_state(node_history, edge_index, edge_features, demand)
                reps = build_station_representations(
                    z, layout.station_node_ids,
                    representation=REPRESENTATION_STATION_REGION_MEAN_MAX,
                    station_region_node_ids=layout.station_region_node_ids,
                ).detach().cpu()
                rows = build_endpoint_station_rows(world, node_history[-1], layout, node_map)
                system = snapshot.get("phase_c_state_diagnostics") or {}
                frame_group = frame_group_counter
                frame_group_counter += 1
                idle_count = system.get("idle_robot_count")
                if idle_count is None:
                    idle_count = sum(
                        1 for agent in world.agents
                        if str(getattr(getattr(agent, "status", None), "name", getattr(agent, "status", ""))).upper() == "IDLE"
                    )
                active_ratio = 1.0 - float(idle_count) / max(float(len(world.agents)), 1.0)
                recorded_active_ratio = system.get("active_robot_ratio")
                if recorded_active_ratio is None:
                    recorded_active_ratio = active_ratio
                for offset, row in enumerate(rows):
                    records.append({
                        "latent": reps[offset],
                        "raw": extract_station_components(row),
                        # Preserve the complete canonical endpoint row.  The
                        # target builder must see node_density_mean and the
                        # weighted-density baseline, not an approximation made
                        # from already-derived components.
                        "station_row": copy.deepcopy(row),
                        "load": load,
                        "load_code": load_index,
                        "seed": int(seed),
                        "tick": int(snapshot.get("decision_tick", 0)),
                        "run_index": run_index,
                        "station_id": int(row["station_id"]),
                        "frame_group": frame_group,
                        "global_open_order_count": float(system.get("open_order_count", 0.0)),
                        "global_active_robot_ratio": float(recorded_active_ratio),
                        "region_node_count": len(layout.station_region_node_ids[offset]),
                    })
            run_index += 1
    if not records:
        raise RuntimeError("station head dataset has no records")
    train_rows = [
        row for row in records if row["seed"] in REPAIR_TRAIN_SEEDS
    ]
    scale_contract = fit_scale_contract(
        [row["raw"] for row in train_rows],
        fitted_seeds=REPAIR_TRAIN_SEEDS,
        source_protocol_sha256=_load_protocol(root)[0]["protocol_sha256"],
    )
    _atomic_json(scale_path, scale_contract)
    split_buffers: dict[str, dict[str, list[Any]]] = {
        name: {key: [] for key in ("latents", "targets", "raw_components", "normalised_components", "frame_group", "run_index", "station_id", "seed", "tick", "tick_fraction", "arm_code", "load_code", "global_open_order_count", "global_active_robot_ratio", "region_node_count")}
        for name in ("train", "val", "test")
    }
    def split(seed: int) -> str:
        return (
            "train"
            if seed in REPAIR_TRAIN_SEEDS
            else "val"
            if seed in REPAIR_VAL_SEEDS
            else "test"
        )
    for row in records:
        # Use the exact endpoint row emitted by the canonical station
        # observer.  Reconstructing it from ``raw`` would silently replace
        # node_density_mean and the weighted baseline and change J1 labels.
        target = build_station_targets(row["station_row"], scale_contract)
        name = split(row["seed"])
        b = split_buffers[name]
        b["latents"].append(row["latent"])
        b["targets"].append(torch.tensor([target["channels"][key] for key in CHANNEL_NAMES], dtype=torch.float32))
        b["raw_components"].append(torch.tensor([row["raw"][key] for key in COMPONENT_NAMES], dtype=torch.float32))
        b["normalised_components"].append(torch.tensor([target["normalised_components"][key] for key in COMPONENT_NAMES], dtype=torch.float32))
        b["frame_group"].append(row["frame_group"])
        b["run_index"].append(row["run_index"])
        b["station_id"].append(row["station_id"])
        b["seed"].append(row["seed"])
        b["tick"].append(row["tick"])
        b["tick_fraction"].append(row["tick"] / max(protocol_ticks, 1))
        b["arm_code"].append(0)
        b["load_code"].append(row["load_code"])
        b["global_open_order_count"].append(row["global_open_order_count"])
        b["global_active_robot_ratio"].append(row["global_active_robot_ratio"])
        b["region_node_count"].append(row["region_node_count"])
    def finalise(buffer: Mapping[str, list[Any]]) -> dict[str, Any]:
        count = len(buffer["latents"])
        if not count:
            raise RuntimeError("empty station dataset split")
        return {
            "latents": torch.stack(buffer["latents"]).reshape(count, -1),
            "targets": torch.stack(buffer["targets"]).reshape(count, 2),
            "raw_components": torch.stack(buffer["raw_components"]).reshape(count, 5),
            "normalised_components": torch.stack(buffer["normalised_components"]).reshape(count, 5),
            "frame_group": torch.tensor(buffer["frame_group"], dtype=torch.int64),
            "run_index": torch.tensor(buffer["run_index"], dtype=torch.int64),
            "station_id": torch.tensor(buffer["station_id"], dtype=torch.int64),
            "seed": torch.tensor(buffer["seed"], dtype=torch.int64),
            "tick": torch.tensor(buffer["tick"], dtype=torch.int64),
            "tick_fraction": torch.tensor(buffer["tick_fraction"], dtype=torch.float32),
            "arm_code": torch.tensor(buffer["arm_code"], dtype=torch.int64),
            "load_code": torch.tensor(buffer["load_code"], dtype=torch.int64),
            "global_open_order_count": torch.tensor(buffer["global_open_order_count"], dtype=torch.float32),
            "global_active_robot_ratio": torch.tensor(buffer["global_active_robot_ratio"], dtype=torch.float32),
            "region_node_count": torch.tensor(buffer["region_node_count"], dtype=torch.int64),
        }
    splits = {name: finalise(buffer) for name, buffer in split_buffers.items()}
    from WorldModel.data.build_station_congestion_head_dataset import DATASET_SCHEMA_VERSION
    dataset = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "development_only": False,
        "source_protocol_sha256": _load_protocol(root)[0]["protocol_sha256"],
        "source_checkpoint": final_model.as_posix(),
        "source_checkpoint_sha256": _sha256_file(final_model),
        "checkpoint_label_schema": "station6_state_level_current_v1",
        "latent_dim": int(splits["train"]["latents"].shape[1]),
        "encoder_latent_dim": 64,
        "representation": {"name": "station_region_mean_max", "primary_region_hops": 3, "station_ids_required": True, "features": ["station_node", "region_mean", "region_max"]},
        "channel_names": ["traffic", "service"],
        "component_names": list(COMPONENT_NAMES),
        "scale_contract": scale_contract,
        "split_seeds": {
            "train": list(REPAIR_TRAIN_SEEDS),
            "val": list(REPAIR_VAL_SEEDS),
            "test": list(REPAIR_TEST_SEEDS),
        },
        "arm_names": ["station6_adapted_wm"],
        "load_names": list(LOADS),
        "run_ids": [
            f"station6_repair_{load}_seed{seed}"
            for load in LOADS for seed in REPAIR_DATA_SEEDS
        ],
        "splits": splits,
    }
    _atomic_torch_save(output, dataset)
    _atomic_json(output.parent / "dataset_summary.json", {
        "schema_version": "station6_state_head_dataset_summary_v1",
        "records": len(records), "split_sizes": {name: len(value["targets"]) for name, value in splits.items()},
        "source_checkpoint": final_model.as_posix(), "source_checkpoint_sha256": _sha256_file(final_model),
        "representation": dataset["representation"], "scale_contract_sha256": scale_contract["contract_sha256"],
        "endpoint_transport_certificate": False,
    })
    return output


def _train_station_head(root: Path) -> Path:
    dataset = _station_dataset(root)
    output = root / "training" / "station_head_state_level_v1"
    checkpoint = output / "best_station_congestion_head.pt"
    if checkpoint.is_file() and (output / "train_summary.json").is_file():
        return checkpoint
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"partial station-head training exists: {output}")
    command = [
        sys.executable, "-m", "WorldModel.training.train_station_congestion_head",
        "--dataset", str(dataset), "--output-root", str(output),
        "--epochs", "100", "--batch-size", "4096", "--learning-rate", "1e-3",
        "--weight-decay", "1e-4", "--patience", "15", "--seed", "20260824",
        "--device", "cpu", "--torch-threads", "4", "--formal",
    ]
    _run(
        command,
        cwd=_repo_root(),
        log=root / "logs" / "train_station_head.log",
        env_overrides=_cpu_training_env(max_threads=4),
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _make_assigner(arm: str, root: Path):
    from Policies.TaskAssigner import GreedyTaskAssigner, HungarianTaskAssigner, WorldModelTaskAssigner
    from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_context_assigner import PsiDispatchContextWorldModelTaskAssigner
    from WorldModel.evaluation.phase_c_combo_s1_j1_paper_protocol import selector_config
    from WorldModel.evaluation.phase_c_s1_hungarian_protocol import PHASEC_CONFIG
    if arm == "greedy":
        return GreedyTaskAssigner()
    if arm == "hungarian":
        return HungarianTaskAssigner()
    checkpoint = _train_longrisk(root)
    if arm == "phasec":
        return WorldModelTaskAssigner(checkpoint_path=str(checkpoint), top_m=TOP_M, **dict(PHASEC_CONFIG))
    if arm == "combo_s1_j1":
        head = _train_station_head(root)
        scale = root / "station_head_dataset" / "station_congestion_scale_contract.json"
        return PsiDispatchContextWorldModelTaskAssigner(
            psi_head_checkpoint=str(head), psi_scale_contract=str(scale),
            psi_context_mode="j_ascending", psi_trace_enabled=False,
            psi_trace_max_records=0, allow_phasec_s0_robot_scorer=False,
            checkpoint_path=str(checkpoint), top_m=TOP_M,
            energy_conv_random_flip_seed=0, **dict(selector_config()),
        )
    raise ValueError(arm)


def _eval_cell(root: Path, arm: str, load: str, seed: int, ticks: int = TICKS) -> None:
    if int(seed) not in ONLINE_TEST_SEEDS:
        raise ValueError(
            f"online evaluation seed {seed} is outside the held-out block "
            f"{ONLINE_TEST_SEEDS}"
        )
    bundle, protocol = _load_protocol(root)
    _require_protocol_ticks(protocol, ticks)
    output = root / "evaluation" / "per_arm" / arm / f"{load}_seed{seed}.json"
    manifest = _eval_manifest_path(root, load, seed)
    expected_checkpoint = (
        _train_longrisk(root) if arm in ("phasec", "combo_s1_j1") else None
    )
    expected_head = (
        _train_station_head(root) if arm == "combo_s1_j1" else None
    )
    if output.is_file():
        payload = _read_json(output)
        meta = payload.get("meta", {})
        resume_checks = {
            "audit": bool(payload.get("audit", {}).get("passed")),
            "protocol": meta.get("protocol_sha256") == bundle["protocol_sha256"],
            "manifest_file": manifest.is_file(),
            "manifest_hash": manifest.is_file()
            and meta.get("manifest_sha256") == _sha256_file(manifest),
            "checkpoint": expected_checkpoint is None
            or meta.get("checkpoint_sha256")
            == _sha256_file(expected_checkpoint),
            "station_head": expected_head is None
            or meta.get("station_head_sha256") == _sha256_file(expected_head),
        }
        if all(resume_checks.values()):
            return
        raise ValueError(
            f"incompatible evaluation output: {output}: "
            f"{[name for name, passed in resume_checks.items() if not passed]}"
        )
    manifest_existed = manifest.is_file()
    if arm != "greedy" and not manifest_existed:
        raise FileNotFoundError(
            f"held-out Greedy manifest must be generated before {arm}: {manifest}"
        )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
    from WorldModel.evaluation.run_phase_c_s1_j1_long_risk_correction import _physical_only_audit_contract
    assigner = _make_assigner(arm, root)
    replay_kwargs = (
        {"recorded_orders_path": str(manifest)}
        if manifest_existed
        else {"save_order_manifest": str(manifest)}
    )
    with _physical_only_audit_contract() as holder:
        metrics = _run_one_assigner(
            str(_config(protocol, load)), assigner, int(seed), int(ticks),
            trace_label=f"Station6_{arm}", **replay_kwargs,
        )
    if hasattr(assigner, "psi_dispatch_metrics"):
        metrics.update(assigner.psi_dispatch_metrics())
    station_audit = holder.get("station_probe").summary() if holder.get("station_probe") else {}
    manifest_payload = _check_manifest(manifest)
    checks = {
        "manifest": metrics.get("order_arrival_manifest_sha256") == manifest_payload.get("manifest_sha256"),
        "manifest_count": int(metrics.get("order_arrival_count", -1))
        == int(manifest_payload.get("total_orders", -2)),
        "ticks": int(metrics.get("ticks", -1)) == int(ticks),
        "robot_count": int(metrics.get("num_agents", -1)) == ROBOT_COUNT,
        "paired_replay": (
            not manifest_existed and arm == "greedy"
        ) or bool(metrics.get("order_arrival_replayed")),
        "physical_only": _physical_only_mode_ok(station_audit),
        "station_audit_passed": bool(station_audit.get("passed")),
        "physical_capacity_no_violation": int(
            station_audit.get("physical_capacity_violation_count", -1)
        )
        == 0,
        "no_committed_cap_contract": station_audit.get(
            "committed_capacity_contract"
        )
        == "not enforced for in-transit commitments",
    }
    if arm in ("greedy", "hungarian"):
        checks["no_model_calls"] = int(metrics.get("model_assign_calls", 0)) == 0
    if arm in ("phasec", "combo_s1_j1"):
        from WorldModel.core.long_risk_schema import (
            LONG_RISK_SCHEMA_VERSION,
            long_risk_runtime_contract,
        )

        checks.update({
            "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
            "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
            "long_risk_schema": metrics.get("long_risk_schema_version")
            == LONG_RISK_SCHEMA_VERSION,
            "long_risk_contract": metrics.get("long_risk_runtime_contract")
            == long_risk_runtime_contract(),
        })
    if arm == "phasec":
        checks["phasec_energy_off"] = metrics.get("energy_scoring_mode") == "off"
    if arm == "combo_s1_j1":
        checks.update({
            "combo_energy_mode": metrics.get("energy_scoring_mode")
            == "conversion",
            "combo_signal": metrics.get("energy_drift_signal") == "combo",
            "combo_contexts_seen": int(
                metrics.get("energy_conv_contexts", 0)
            )
            > 0,
            "j1_head_loaded": bool(metrics.get("psi_dispatch_head_loaded")),
            "j1_mode": metrics.get("psi_dispatch_mode") == "j_ascending",
            "j1_head_hash": metrics.get(
                "psi_dispatch_head_checkpoint_sha256"
            )
            == _sha256_file(expected_head),
            "j1_scale_hash": metrics.get(
                "psi_dispatch_scale_contract_sha256"
            )
            == _sha256_file(
                root
                / "station_head_dataset"
                / "station_congestion_scale_contract.json"
            ),
            "j1_evaluated": int(metrics.get("psi_dispatch_eval_calls", 0)) > 0,
            "j1_contexts_seen": int(
                metrics.get("psi_dispatch_contexts_seen", 0)
            )
            > 0,
            "j1_parent_robot_scorer_preserved": metrics.get(
                "psi_dispatch_robot_scorer"
            )
            == "WorldModelTaskAssigner.select_robots_unmodified",
            "j1_encoder_binding": bool(
                metrics.get("psi_dispatch_encoder_contract_verified")
            ),
            "j1_s1_variant": metrics.get(
                "psi_dispatch_robot_scorer_variant"
            )
            == "s1_within_context",
        })
    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": bundle["protocol_sha256"],
            "arm": arm, "load": load, "seed": int(seed),
            "ticks": int(ticks), "stations": STATION_COUNT,
            "checkpoint": (
                expected_checkpoint.as_posix()
                if expected_checkpoint is not None else None
            ),
            "checkpoint_sha256": (
                _sha256_file(expected_checkpoint)
                if expected_checkpoint is not None else None
            ),
            "station_head": (
                expected_head.as_posix() if expected_head is not None else None
            ),
            "station_head_sha256": (
                _sha256_file(expected_head) if expected_head is not None else None
            ),
            "manifest_path": manifest.as_posix(),
            "manifest_sha256": _sha256_file(manifest),
            "manifest_generated_by_this_run": not manifest_existed,
            "held_out_online_seed": True,
        },
        "metrics": metrics, "station_audit": station_audit,
        "audit": {"passed": all(checks.values()), "checks": checks},
    }
    _atomic_json(output, payload)
    if not payload["audit"]["passed"]:
        raise RuntimeError(f"evaluation audit failed: {output}: {checks}")


def _summarize(root: Path) -> None:
    bundle, _ = _load_protocol(root)
    rows = []
    for arm in ARM_KEYS:
        for load in LOADS:
            for seed in ONLINE_TEST_SEEDS:
                path = root / "evaluation" / "per_arm" / arm / f"{load}_seed{seed}.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = _read_json(path)
                if not payload.get("audit", {}).get("passed"):
                    raise ValueError(f"failed evaluation audit: {path}")
                row = {"arm": arm, "load": load, "seed": seed}
                metrics = payload.get("metrics") or {}
                for metric in REPORT_METRICS:
                    row[metric] = metrics.get(metric)
                rows.append(row)
    aggregate: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARM_KEYS:
        aggregate[arm] = {}
        for load in LOADS:
            selected = [row for row in rows if row["arm"] == arm and row["load"] == load]
            cell = {"runs": len(selected)}
            for metric in REPORT_METRICS:
                values = [float(row[metric]) for row in selected if row.get(metric) is not None]
                cell[f"{metric}_mean"] = statistics.fmean(values) if values else None
                cell[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None
            aggregate[arm][load] = cell
    comparisons = {}
    for baseline in ("greedy", "hungarian", "phasec"):
        name = f"combo_s1_j1_minus_{baseline}"
        comparisons[name] = {}
        for load in LOADS:
            cell = {}
            for metric in REPORT_METRICS:
                left = [row[metric] for row in rows if row["arm"] == "combo_s1_j1" and row["load"] == load and row.get(metric) is not None]
                right = [row[metric] for row in rows if row["arm"] == baseline and row["load"] == load and row.get(metric) is not None]
                cell[metric] = (statistics.fmean([float(a) - float(b) for a, b in zip(left, right)]) if left and len(left) == len(right) else None)
            comparisons[name][load] = cell
    expected_rows = len(ARM_KEYS) * len(LOADS) * len(ONLINE_TEST_SEEDS)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "protocol_sha256": bundle["protocol_sha256"],
        "target": {"map": [MAP_ROWS, MAP_COLS], "robots": ROBOT_COUNT, "stations": STATION_COUNT, "demand_dim": 11},
        "zero_shot_claim": False,
        "phase_c_core_data_seeds": list(CORE_DATA_SEEDS),
        "long_risk_and_j1_data_seeds": list(REPAIR_DATA_SEEDS),
        "held_out_online_test_seeds": list(ONLINE_TEST_SEEDS),
        "online_test_disjoint_from_all_adaptation_data": not bool(
            (set(CORE_DATA_SEEDS) | set(REPAIR_DATA_SEEDS))
            & set(ONLINE_TEST_SEEDS)
        ),
        "phase_c_core_checkpoint": _train_phasec_core(root).as_posix(),
        "adapted_checkpoint": _train_longrisk(root).as_posix(),
        "station_head": _train_station_head(root).as_posix(),
        "rows": rows, "aggregate": aggregate, "comparisons": comparisons,
        "audit": {
            "passed": (
                not bool(
                    (set(CORE_DATA_SEEDS) | set(REPAIR_DATA_SEEDS))
                    & set(ONLINE_TEST_SEEDS)
                )
                and not bool(set(CORE_DATA_SEEDS) & set(REPAIR_DATA_SEEDS))
                and len(rows) == expected_rows
            ),
            "expected_rows": expected_rows,
            "rows": len(rows),
        },
    }
    _atomic_json(root / "evaluation" / "summary.json", summary)
    print(json.dumps({"output": (root / "evaluation" / "summary.json").as_posix(), "rows": len(rows)}, indent=2))


def _cells(mode: str) -> list[tuple[str, int]]:
    if mode in ("collect_base", "collect_core_snapshots", "build_core_data"):
        seeds = CORE_DATA_SEEDS
    elif mode in (
        "collect_repair_snapshots",
        "label_longrisk",
        "build_repair_data",
    ):
        seeds = REPAIR_DATA_SEEDS
    else:
        raise ValueError(mode)
    return [(load, seed) for load in LOADS for seed in seeds]


def _parallel_cells(root: Path, mode: str, workers: int, ticks: int = TICKS) -> None:
    failures = []
    # Simulation cells mutate module-level RNG/ID state.  Threads would make
    # otherwise paired cells race on those globals; isolated worker processes
    # preserve deterministic per-seed replay while still using the CPU node.
    with ProcessPoolExecutor(
        max_workers=max(1, int(workers)),
        mp_context=mp.get_context("spawn"),
    ) as pool:
        futures = {}
        for load, seed in _cells(mode):
            if mode == "collect_base":
                future = pool.submit(_run_base_cell, root, load, seed, ticks)
            elif mode == "collect_core_snapshots":
                future = pool.submit(
                    _run_snapshot_cell, root, "core", load, seed, ticks
                )
            elif mode == "collect_repair_snapshots":
                future = pool.submit(
                    _run_snapshot_cell, root, "repair", load, seed, ticks
                )
            elif mode == "label_longrisk":
                future = pool.submit(_run_label_cell, root, load, seed)
            elif mode == "build_core_data":
                future = pool.submit(
                    _run_snapshot_data_cell, root, "core", load, seed
                )
            elif mode == "build_repair_data":
                future = pool.submit(
                    _run_snapshot_data_cell, root, "repair", load, seed
                )
            else:
                raise ValueError(mode)
            futures[future] = (load, seed)
        for future in as_completed(futures):
            cell = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((cell, repr(exc)))
    if failures:
        raise RuntimeError(f"{mode} failures: {failures[:10]}")


def _parallel_eval(root: Path, workers: int, ticks: int = TICKS) -> None:
    def run_batch(cells: Sequence[tuple[str, str, int]], label: str) -> None:
        failures = []
        with ProcessPoolExecutor(
            max_workers=max(1, int(workers)),
            mp_context=mp.get_context("spawn"),
        ) as pool:
            futures = {
                pool.submit(_eval_cell, root, arm, load, seed, ticks): (
                    arm, load, seed
                )
                for arm, load, seed in cells
            }
            for future in as_completed(futures):
                cell = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append((cell, repr(exc)))
        if failures:
            raise RuntimeError(f"{label} failures: {failures[:10]}")

    # The held-out Greedy arm is the sole manifest producer.  Finish it first
    # so every other arm can only replay an immutable paired arrival stream.
    greedy_cells = [
        ("greedy", load, seed)
        for load in LOADS for seed in ONLINE_TEST_SEEDS
    ]
    run_batch(greedy_cells, "held-out Greedy manifest/evaluation")
    replay_cells = [
        (arm, load, seed)
        for arm in ARM_KEYS if arm != "greedy"
        for load in LOADS for seed in ONLINE_TEST_SEEDS
    ]
    run_batch(replay_cells, "held-out paired evaluation")


def _all(root: Path, workers: int, ticks: int = TICKS) -> None:
    _prepare(root, ticks)
    _parallel_cells(root, "collect_base", workers, ticks)
    _fuse_base(root)
    _train_base(root)
    _parallel_cells(root, "collect_core_snapshots", workers, ticks)
    _parallel_cells(root, "build_core_data", workers)
    _fuse_core(root)
    _train_phasec_core(root)
    _parallel_cells(root, "collect_repair_snapshots", workers, ticks)
    _parallel_cells(root, "label_longrisk", workers)
    _parallel_cells(root, "build_repair_data", workers)
    _fuse_longrisk(root)
    _train_longrisk(root)
    _station_dataset(root)
    _train_station_head(root)
    _parallel_eval(root, workers, ticks)
    _summarize(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "prepare", "collect_base", "fuse_base", "train_base",
            "collect_core_snapshots", "build_core_data", "fuse_core",
            "train_core", "collect_repair_snapshots", "label_longrisk",
            "build_repair_data", "fuse_longrisk", "train_longrisk",
            "build_station_dataset", "train_station_head", "evaluate",
            "summarize", "all",
        ),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--load", choices=LOADS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=ARM_KEYS)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--ticks", type=int, default=TICKS)
    args = parser.parse_args()
    root = args.output_root
    if args.mode == "prepare":
        _prepare(root, args.ticks)
    elif args.mode == "collect_base":
        if args.load is None or args.seed is None: parser.error("collect_base requires --load/--seed")
        _run_base_cell(root, args.load, args.seed, args.ticks)
    elif args.mode == "fuse_base":
        _fuse_base(root)
    elif args.mode == "train_base":
        _train_base(root)
    elif args.mode == "collect_core_snapshots":
        if args.load is None or args.seed is None: parser.error("collect_core_snapshots requires --load/--seed")
        _run_snapshot_cell(root, "core", args.load, args.seed, args.ticks)
    elif args.mode == "build_core_data":
        if args.load is None or args.seed is None: parser.error("build_core_data requires --load/--seed")
        _run_snapshot_data_cell(root, "core", args.load, args.seed)
    elif args.mode == "fuse_core":
        _fuse_core(root)
    elif args.mode == "train_core":
        _train_phasec_core(root)
    elif args.mode == "collect_repair_snapshots":
        if args.load is None or args.seed is None: parser.error("collect_repair_snapshots requires --load/--seed")
        _run_snapshot_cell(root, "repair", args.load, args.seed, args.ticks)
    elif args.mode == "label_longrisk":
        if args.load is None or args.seed is None: parser.error("label_longrisk requires --load/--seed")
        _run_label_cell(root, args.load, args.seed)
    elif args.mode == "build_repair_data":
        if args.load is None or args.seed is None: parser.error("build_repair_data requires --load/--seed")
        _run_snapshot_data_cell(root, "repair", args.load, args.seed)
    elif args.mode == "fuse_longrisk":
        _fuse_longrisk(root)
    elif args.mode == "train_longrisk":
        _train_longrisk(root)
    elif args.mode == "build_station_dataset":
        _station_dataset(root)
    elif args.mode == "train_station_head":
        _train_station_head(root)
    elif args.mode == "evaluate":
        if args.arm is None or args.load is None or args.seed is None: parser.error("evaluate requires --arm/--load/--seed")
        _eval_cell(root, args.arm, args.load, args.seed, args.ticks)
    elif args.mode == "summarize":
        _summarize(root)
    else:
        _all(root, args.workers, args.ticks)


if __name__ == "__main__":
    main()
