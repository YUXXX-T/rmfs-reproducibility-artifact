"""Frozen map-size, fleet-size, and station-count generalisation campaign.

This runner is deliberately isolated from the simulator and policy modules.  It
derives JSON configurations into the campaign output directory, records the
exact input hashes, creates one Greedy order manifest per
``(variant, load, seed)``, and replays that manifest for every executable arm.

The frozen Phase-C checkpoint was trained with four stations and a demand
vector of ``5 + 4 = 9`` channels.  Consequently, four-station map/fleet
variants are valid zero-shot World-Model tests.  A six-station variant is
included as a simulator-compatible baseline-only probe and is explicitly
marked ``unsupported_checkpoint_schema`` for World-Model arms; it must not be
reported as World-Model station-count generalisation without an adapted
checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from Policies.TaskAssigner import (
    GreedyTaskAssigner,
    HungarianTaskAssigner,
    WorldModelTaskAssigner,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_context_assigner import (
    PsiDispatchContextWorldModelTaskAssigner,
)
from WorldModel.evaluation.phase_c_long_risk_head_repair_protocol import (
    REBOUND_PSI_HEAD,
    REPAIRED_CHECKPOINT,
    PSI_SCALE_CONTRACT,
    combo_selector_config,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    LOAD_CONFIGS as SOURCE_LOAD_CONFIGS,
    sha256_file,
)
from WorldModel.evaluation.phase_c_s1_hungarian_protocol import PHASEC_CONFIG
from WorldModel.evaluation.run_phase_c_s1_fifo_pair import _validate_manifest
from WorldModel.evaluation.run_phase_c_s1_j1_long_risk_correction import (
    _physical_only_audit_contract,
)
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


SCHEMA_VERSION = "phase_c_scale_generalization_run_v1"
PROTOCOL_SCHEMA_VERSION = "phase_c_scale_generalization_protocol_v1"
SUMMARY_SCHEMA_VERSION = "phase_c_scale_generalization_summary_v1"
PREFLIGHT_SCHEMA_VERSION = "phase_c_scale_generalization_preflight_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "phasec_scale_generalization_701_710_v1"
MODEL_CHECKPOINT = Path(REPAIRED_CHECKPOINT)
PSI_HEAD_CHECKPOINT = Path(REBOUND_PSI_HEAD)
PSI_SCALE_CONTRACT_PATH = Path(PSI_SCALE_CONTRACT)
MODEL_SUMMARY = BASE_ROOT / "model_round1_v1" / "train_summary.json"

LOADS = ("low", "mid", "high")
SEEDS = tuple(range(701, 711))
TICKS = 1500
TOP_M = 10


@dataclass(frozen=True)
class Variant:
    key: str
    rows: int
    cols: int
    robots: int
    stations: int
    description: str
    preserve_20x20_layout: bool = False

    @property
    def wm_compatible(self) -> bool:
        return self.stations == 4

    @property
    def arms(self) -> tuple[str, ...]:
        if self.wm_compatible:
            return ("greedy", "hungarian", "phasec", "combo_s1_j1")
        # A six-station config is useful to verify that the simulator and
        # analytic baselines accept it, but the frozen WM cannot consume its
        # 11-dimensional demand vector.
        return ("greedy", "hungarian")


VARIANTS: tuple[Variant, ...] = (
    Variant(
        "map20_r36_s4", 20, 20, 36, 4,
        "20x20 baseline topology with a sparse 36-robot fleet",
        preserve_20x20_layout=True,
    ),
    Variant(
        "map20_r48_s4", 20, 20, 48, 4,
        "20x20 baseline topology and nominal 48-robot fleet",
        preserve_20x20_layout=True,
    ),
    Variant(
        "map20_r60_s4", 20, 20, 60, 4,
        "20x20 baseline topology with a dense 60-robot fleet",
        preserve_20x20_layout=True,
    ),
    Variant(
        "map30_r48_s4", 30, 30, 48, 4,
        "30x30 map with the nominal fleet",
    ),
    Variant(
        "map30_r72_s4", 30, 30, 72, 4,
        "30x30 map with a dense 72-robot fleet",
    ),
    Variant(
        "map40_r96_s4", 40, 40, 96, 4,
        "40x40 map with a dense 96-robot fleet",
    ),
    Variant(
        "map30_r72_s6", 30, 30, 72, 6,
        "30x30 map with six stations (baseline-only schema probe)",
    ),
)
VARIANT_BY_KEY = {variant.key: variant for variant in VARIANTS}

SUMMARY_METRICS = (
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
    "wall_time_s",
    "assignment_time_ms_mean",
)

RUNNER_PATH = Path(__file__)
SOURCE_FILES = {
    "runner": RUNNER_PATH,
    "phasec_protocol": Path(
        "WorldModel/evaluation/phase_c_s1_hungarian_protocol.py"
    ),
    "combo_protocol": Path(
        "WorldModel/evaluation/phase_c_combo_s1_j1_paper_protocol.py"
    ),
    "psi_protocol": Path(
        "WorldModel/evaluation/phase_c_psi_dispatch_ablation_protocol.py"
    ),
    "longrisk_protocol": Path(
        "WorldModel/evaluation/phase_c_long_risk_head_repair_protocol.py"
    ),
    "physical_audit_helper": Path(
        "WorldModel/evaluation/run_phase_c_s1_j1_long_risk_correction.py"
    ),
    "station_restoration_audit": Path(
        "WorldModel/evaluation/run_phase_c_station_admission_restoration.py"
    ),
    "manifest_helper": Path(
        "WorldModel/evaluation/run_phase_c_s1_fifo_pair.py"
    ),
    "evaluate_online": Path("WorldModel/evaluation/evaluate_online_v6.py"),
    "evaluate_engine": Path("WorldModel/evaluation/evaluate.py"),
    "task_assigner_init": Path("Policies/TaskAssigner/__init__.py"),
    "task_assigner_base": Path("Policies/TaskAssigner/base_task_assigner.py"),
    "greedy_assigner": Path(
        "Policies/TaskAssigner/GreedyTaskAssigner/greedy_task_assigner.py"
    ),
    "hungarian_assigner": Path(
        "Policies/TaskAssigner/HungarianTaskAssigner/hungarian_task_assigner.py"
    ),
    "world_model_assigner": Path(
        "Policies/TaskAssigner/WorldModelTaskAssigner/"
        "world_model_task_assigner.py"
    ),
    "psi_assigner": Path(
        "Policies/TaskAssigner/WorldModelTaskAssigner/"
        "psi_dispatch_context_assigner.py"
    ),
    "world_model": Path("WorldModel/core/model.py"),
    "long_risk_schema": Path("WorldModel/core/long_risk_schema.py"),
    "graph_builder": Path("WorldModel/graph/graph_builder.py"),
    "station_endpoint": Path(
        "WorldModel/evaluation/station_congestion_endpoint.py"
    ),
    "station_head": Path("WorldModel/core/station_congestion_head.py"),
    "station_state": Path("WorldState/station_state.py"),
    "config_loader": Path("Config/config_loader.py"),
    "world_state": Path("WorldState/world.py"),
    "map_state": Path("WorldState/map_state.py"),
    "agent_state": Path("WorldState/agent_state.py"),
    "order_state": Path("WorldState/order_state.py"),
    "task_state": Path("WorldState/task_state.py"),
    "pod_state": Path("WorldState/pod_state.py"),
    "simulation_engine": Path("Engine/simulation_engine.py"),
}


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


def _station_specs(rows: int, cols: int, count: int) -> list[dict[str, Any]]:
    if count == 4:
        offset = max(3, cols // 4)
        positions = [
            (0, offset),
            (0, cols - 1 - offset),
            (rows - 1, offset),
            (rows - 1, cols - 1 - offset),
        ]
    elif count == 6:
        offset = max(5, cols // 4)
        positions = [
            (0, offset),
            (0, cols - 1 - offset),
            (rows - 1, offset),
            (rows - 1, cols - 1 - offset),
            (rows // 2, 0),
            (rows // 2, cols - 1),
        ]
    else:
        raise ValueError(f"unsupported station count for this protocol: {count}")
    if len(set(positions)) != count:
        raise ValueError(f"station positions are not unique: {positions}")
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


def _scaled_pod_zones(rows: int, cols: int) -> list[dict[str, int]]:
    """Place the same eight 6x2 pod blocks at scale-aware coordinates.

    Keeping the number of pod homes fixed is important: a map-size test should
    not silently increase inventory merely because the grid became larger.
    """

    zone_rows = 6
    zone_cols = 2
    row_max = rows - 4 - zone_rows
    col_max = cols - 4 - zone_cols
    if row_max < 4 or col_max < 4:
        raise ValueError(f"map too small for scaled pod zones: {rows}x{cols}")
    starts_r = [4, max(4, row_max - 6)]
    starts_c = [
        int(round(4 + (col_max - 4) * index / 3.0))
        for index in range(4)
    ]
    starts = [(row, col) for row in starts_r for col in starts_c]
    if len(starts) != 8 or len(set(starts)) != len(starts):
        raise ValueError(f"scaled pod-zone anchors are not eight unique cells: {starts}")
    return [
        {
            "origin_row": int(row),
            "origin_col": int(col),
            "num_rows": zone_rows,
            "num_cols": zone_cols,
        }
        for row, col in starts
    ]


def _config_path(root: Path, variant: str, load: str) -> Path:
    return root / "frozen_configs" / variant / f"world_model_config_{load}.json"


def _derive_config(source: Mapping[str, Any], variant: Variant) -> dict[str, Any]:
    derived = copy.deepcopy(source)
    robots = derived.setdefault("robots", {})
    robots["num_robots"] = variant.robots
    robots["starts"] = []
    robots["random_starts"] = True

    if variant.preserve_20x20_layout:
        # The 20x20 control cells differ only in fleet size.  This makes the
        # robot-count contrast directly paired with the historical topology.
        return derived

    map_cfg = derived.setdefault("map", {})
    map_cfg["rows"] = variant.rows
    map_cfg["cols"] = variant.cols
    map_cfg["stations"] = _station_specs(
        variant.rows, variant.cols, variant.stations
    )
    map_cfg.pop("pod_layout", None)
    map_cfg["pod_zones"] = _scaled_pod_zones(variant.rows, variant.cols)
    return derived


def _model_schema() -> dict[str, Any]:
    if MODEL_SUMMARY.is_file():
        payload = _read_json(MODEL_SUMMARY)
        value = payload.get("model_config")
        if isinstance(value, Mapping):
            return {
                "demand_dim": int(value.get("demand_dim", -1)),
                "num_stations": int(value.get("num_stations", -1)),
            }
    return {"demand_dim": 9, "num_stations": 4}


def _derive_configs(root: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        entries[variant.key] = {}
        for load in LOADS:
            source_path = Path(SOURCE_LOAD_CONFIGS[load])
            source = _read_json(source_path)
            source_stations = source.get("map", {}).get("stations", [])
            source_robots = source.get("robots", {}).get("num_robots")
            if len(source_stations) != 4 or int(source_robots) != 48:
                raise ValueError(
                    f"scale source must be the frozen 48-robot/4-station config: {source_path}"
                )
            derived = _derive_config(source, variant)
            changed = _changed_paths(source, derived)
            allowed = (
                {"robots.num_robots"}
                if variant.preserve_20x20_layout
                else {"map.rows", "map.cols", "map.stations", "map.pod_zones", "robots.num_robots"}
            )
            if any(
                not any(path == prefix or path.startswith(prefix + "[")
                        or path.startswith(prefix + ".") for prefix in allowed)
                for path in changed
            ):
                raise AssertionError(
                    f"derived {variant.key}/{load} changed fields outside scale contract: {changed}"
                )
            path = _config_path(root, variant.key, load)
            _atomic_json(path, derived)
            entries[variant.key][load] = {
                "source_path": source_path.as_posix(),
                "source_sha256": sha256_file(source_path),
                "derived_path": path.as_posix(),
                "derived_sha256": sha256_file(path),
                "changed_paths": changed,
                "rows": variant.rows,
                "cols": variant.cols,
                "robots": variant.robots,
                "stations": variant.stations,
            }
    return entries


def _protocol_path(root: Path) -> Path:
    return root / "scale_generalization_protocol.json"


def _preflight_path(root: Path) -> Path:
    return root / "scale_generalization_preflight.json"


def _frozen_inputs_path(root: Path) -> Path:
    return root / "frozen_inputs.sha256"


def _make_protocol(root: Path) -> dict[str, Any]:
    for path in (MODEL_CHECKPOINT, PSI_HEAD_CHECKPOINT, PSI_SCALE_CONTRACT_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, path in SOURCE_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    schema = _model_schema()
    configs = _derive_configs(root)
    variants_payload = {}
    for variant in VARIANTS:
        variants_payload[variant.key] = {
            "rows": variant.rows,
            "cols": variant.cols,
            "robots": variant.robots,
            "stations": variant.stations,
            "description": variant.description,
            "arms": list(variant.arms),
            "wm_compatible": variant.wm_compatible,
            "compatibility_reason": (
                "frozen checkpoint demand schema matches four stations"
                if variant.wm_compatible
                else (
                    "unsupported_checkpoint_schema: frozen checkpoint has "
                    f"num_stations={schema['num_stations']}, demand_dim="
                    f"{schema['demand_dim']}; six stations require demand_dim=11"
                )
            ),
        }
    payload: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "purpose": (
            "zero-shot map-size and fleet-size generalisation of the repaired "
            "Phase-C/Combo-S1+J1 policy, with an explicit six-station "
            "simulator baseline-only schema probe"
        ),
        "seeds": list(SEEDS),
        "loads": list(LOADS),
        "ticks": TICKS,
        "top_m": TOP_M,
        "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
        "manifest_source": "GreedyTaskAssigner",
        "exact_manifest_replay": True,
        "variants": variants_payload,
        "load_configs": configs,
        "model_checkpoint": {
            "path": MODEL_CHECKPOINT.as_posix(),
            "sha256": sha256_file(MODEL_CHECKPOINT),
            "model_schema": schema,
        },
        "static_j1_artifacts": {
            "head": PSI_HEAD_CHECKPOINT.as_posix(),
            "head_sha256": sha256_file(PSI_HEAD_CHECKPOINT),
            "scale_contract": PSI_SCALE_CONTRACT_PATH.as_posix(),
            "scale_contract_sha256": sha256_file(PSI_SCALE_CONTRACT_PATH),
        },
        "zero_shot": {
            "world_model_retrained_for_new_map_or_fleet": False,
            "checkpoint_uses_pretrained_longrisk_repair": True,
            "s1_retuned": False,
            "j1_retuned": False,
            "station_admission_modified": False,
            "core_policy_code_modified": False,
        },
        "configs": {
            "four_station_control": (
                "20x20 controls preserve the original explicit station/pod "
                "layout and change only robots.num_robots"
            ),
            "larger_maps": (
                "stations use edge-aware auto queue layout; 6x2 pod blocks "
                "are generated in the interior with a four-cell margin"
            ),
        },
        "phasec_config": dict(PHASEC_CONFIG),
        "combo_selector_config": dict(combo_selector_config("combo")),
        "source_hashes": {
            name: {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in SOURCE_FILES.items()
        },
    }
    return {"schema_version": PROTOCOL_SCHEMA_VERSION, "protocol": payload,
            "protocol_sha256": _canonical_sha(payload)}


def _write_frozen_inputs(root: Path, bundle: Mapping[str, Any]) -> None:
    protocol = bundle["protocol"]
    lines = [
        f"{sha256_file(_protocol_path(root))}  {_protocol_path(root).name}",
        f"{protocol['model_checkpoint']['sha256']}  {protocol['model_checkpoint']['path']}",
        f"{protocol['static_j1_artifacts']['head_sha256']}  {protocol['static_j1_artifacts']['head']}",
        f"{protocol['static_j1_artifacts']['scale_contract_sha256']}  {protocol['static_j1_artifacts']['scale_contract']}",
    ]
    for variant in VARIANTS:
        for load in LOADS:
            entry = protocol["load_configs"][variant.key][load]
            lines.append(f"{entry['source_sha256']}  {entry['source_path']}")
            lines.append(f"{entry['derived_sha256']}  {entry['derived_path']}")
    for entry in protocol["source_hashes"].values():
        lines.append(f"{entry['sha256']}  {entry['path']}")
    _frozen_inputs_path(root).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _freeze(root: Path) -> None:
    expected = _make_protocol(root)
    path = _protocol_path(root)
    if path.is_file():
        existing = _read_json(path)
        if existing != expected:
            raise FileExistsError(f"existing protocol differs: {path}")
        print(f"[resume] protocol {path}")
    else:
        _atomic_json(path, expected)
        print(f"[freeze] protocol sha256={expected['protocol_sha256']}")
    _write_frozen_inputs(root, expected)
    print(f"[freeze] wrote {_frozen_inputs_path(root)}")


def _verify_protocol_inputs(protocol: Mapping[str, Any]) -> None:
    model = protocol["model_checkpoint"]
    if sha256_file(model["path"]) != model["sha256"]:
        raise ValueError("World Model checkpoint differs from frozen protocol")
    static = protocol["static_j1_artifacts"]
    if sha256_file(static["head"]) != static["head_sha256"]:
        raise ValueError("static-J1 head differs from frozen protocol")
    if sha256_file(static["scale_contract"]) != static["scale_contract_sha256"]:
        raise ValueError("static-J1 scale contract differs from frozen protocol")
    for variant in VARIANTS:
        for load in LOADS:
            entry = protocol["load_configs"][variant.key][load]
            if sha256_file(entry["source_path"]) != entry["source_sha256"]:
                raise ValueError(f"source config differs: {variant.key}/{load}")
            if sha256_file(entry["derived_path"]) != entry["derived_sha256"]:
                raise ValueError(f"derived config differs: {variant.key}/{load}")
    for name, entry in protocol["source_hashes"].items():
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise ValueError(f"source code differs from protocol: {name}")


def _load_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = _read_json(_protocol_path(root))
    if bundle.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unexpected scale protocol schema")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol payload is missing")
    if _canonical_sha(protocol) != str(bundle.get("protocol_sha256", "")):
        raise ValueError("protocol hash mismatch")
    _verify_protocol_inputs(protocol)
    return bundle, dict(protocol)


def _run_config_path(protocol: Mapping[str, Any], variant: str, load: str) -> str:
    return str(protocol["load_configs"][variant][load]["derived_path"])


def _manifest_path(root: Path, variant: str, load: str, seed: int) -> Path:
    return root / "order_manifests" / variant / f"orders_{load}_seed{seed}.json"


def _output_path(root: Path, variant: str, arm: str, load: str, seed: int) -> Path:
    return root / "per_arm" / variant / arm / f"{load}_seed{seed}.json"


def _variant(variant: str) -> Variant:
    try:
        return VARIANT_BY_KEY[variant]
    except KeyError as exc:
        raise ValueError(f"unknown variant: {variant}") from exc


def _run_simulation(
    config_path: str,
    assigner: Any,
    seed: int,
    ticks: int,
    trace_label: str,
    *,
    recorded_orders_path: str | None = None,
    save_order_manifest: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    with _physical_only_audit_contract() as holder:
        metrics = _run_one_assigner(
            config_path,
            assigner,
            int(seed),
            int(ticks),
            trace_label=trace_label,
            recorded_orders_path=recorded_orders_path,
            save_order_manifest=save_order_manifest,
        )
    if hasattr(assigner, "psi_dispatch_metrics"):
        metrics.update(assigner.psi_dispatch_metrics())
    station_probe = holder.get("station_probe")
    if station_probe is None:
        raise RuntimeError("physical-only station probe was not attached")
    station_audit = station_probe.summary()
    metrics.update({
        "station_admission_mode": STATION_ADMISSION_PHYSICAL_ONLY,
        "station_capacity_rejections": int(
            station_audit.get("capacity_rejections", 0)
        ),
        "station_max_occupancy": max(
            (int(value) for value in (station_audit.get("max_occupancy") or {}).values()),
            default=0,
        ),
        "station_max_committed_load": max(
            (int(value) for value in (station_audit.get("max_committed_load") or {}).values()),
            default=0,
        ),
    })
    return dict(metrics), dict(station_audit)


def _make_assigner(arm: str, protocol: Mapping[str, Any]):
    checkpoint = str(protocol["model_checkpoint"]["path"])
    if arm == "greedy":
        return GreedyTaskAssigner()
    if arm == "hungarian":
        return HungarianTaskAssigner()
    if arm == "phasec":
        return WorldModelTaskAssigner(
            checkpoint_path=checkpoint,
            top_m=TOP_M,
            **dict(PHASEC_CONFIG),
        )
    if arm == "combo_s1_j1":
        static = protocol["static_j1_artifacts"]
        return PsiDispatchContextWorldModelTaskAssigner(
            psi_head_checkpoint=str(static["head"]),
            psi_scale_contract=str(static["scale_contract"]),
            psi_context_mode="j_ascending",
            psi_trace_enabled=False,
            psi_trace_max_records=0,
            allow_phasec_s0_robot_scorer=False,
            checkpoint_path=checkpoint,
            top_m=TOP_M,
            energy_conv_random_flip_seed=0,
            **dict(combo_selector_config("combo")),
        )
    raise ValueError(f"unsupported arm: {arm}")


def _arm_contract(arm: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "arm_key": arm,
        "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
        "checkpoint": None,
    }
    if arm == "phasec":
        contract.update({
            "policy_family": "world_model",
            "robot_selector": "phasec_s0",
            "checkpoint": protocol["model_checkpoint"]["path"],
            "checkpoint_sha256": protocol["model_checkpoint"]["sha256"],
        })
    elif arm == "combo_s1_j1":
        static = protocol["static_j1_artifacts"]
        contract.update({
            "policy_family": "world_model_static_j1",
            "robot_selector": "corrected_quantile_combo_s1",
            "context_scheduler": "static_j1",
            "energy_drift_signal": "combo",
            "checkpoint": protocol["model_checkpoint"]["path"],
            "checkpoint_sha256": protocol["model_checkpoint"]["sha256"],
            "psi_head": static["head"],
            "psi_head_sha256": static["head_sha256"],
            "psi_scale_contract": static["scale_contract"],
            "psi_scale_contract_sha256": static["scale_contract_sha256"],
        })
    else:
        contract["policy_family"] = "analytic_baseline"
    return {**contract, "fingerprint_sha256": _canonical_sha(contract)}


def _common_checks(
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    station_audit: Mapping[str, Any],
    variant: Variant,
    ticks: int,
) -> dict[str, bool]:
    return {
        "manifest_count_matches": int(metrics.get("order_arrival_count", -1))
        == int(manifest.get("total_orders", -2)),
        "manifest_hash_matches": metrics.get("order_arrival_manifest_sha256")
        == manifest.get("manifest_sha256"),
        "ticks_match": int(metrics.get("ticks", -1)) == int(ticks),
        "robot_count_matches": int(metrics.get("num_agents", -1)) == variant.robots,
        "physical_only_mode": station_audit.get("mode")
        == STATION_ADMISSION_PHYSICAL_ONLY,
        "station_audit_passed": bool(station_audit.get("passed")),
        "physical_capacity_never_exceeded": int(
            station_audit.get("physical_capacity_violation_count", -1)
        ) == 0,
        "no_committed_cap_contract": station_audit.get(
            "committed_capacity_contract"
        ) == "not enforced for in-transit commitments",
        "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
    }


def _arm_checks(
    arm: str,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    station_audit: Mapping[str, Any],
    variant: Variant,
    ticks: int,
    *,
    generated_manifest: bool,
) -> dict[str, bool]:
    checks = _common_checks(metrics, manifest, station_audit, variant, ticks)
    checks["manifest_mode_valid"] = (
        generated_manifest or bool(metrics.get("order_arrival_replayed"))
    )
    if arm in ("phasec", "combo_s1_j1"):
        checks["world_model_used"] = int(metrics.get("model_assign_calls", 0)) > 0
    if arm == "phasec":
        checks["phasec_energy_off"] = metrics.get("energy_scoring_mode") == "off"
    if arm == "combo_s1_j1":
        checks.update({
            "combo_energy_mode": metrics.get("energy_scoring_mode") == "conversion",
            "combo_signal": metrics.get("energy_drift_signal") == "combo",
            "combo_contexts_seen": int(metrics.get("energy_conv_contexts", 0)) > 0,
            "j1_mode": metrics.get("psi_dispatch_mode") == "j_ascending",
            "j1_head_loaded": bool(metrics.get("psi_dispatch_head_loaded")),
            "j1_evaluated": int(metrics.get("psi_dispatch_eval_calls", 0)) > 0,
            "j1_encoder_binding": bool(
                metrics.get("psi_dispatch_encoder_contract_verified")
            ),
            "j1_s1_variant": metrics.get("psi_dispatch_robot_scorer_variant")
            == "s1_within_context",
        })
    return checks


def _prepare_manifest(root: Path, variant_key: str, load: str, seed: int, ticks: int) -> None:
    variant = _variant(variant_key)
    bundle, protocol = _load_protocol(root)
    if load not in LOADS or seed not in SEEDS:
        raise ValueError(f"invalid load/seed: {load}/{seed}")
    manifest_path = _manifest_path(root, variant_key, load, seed)
    output = _output_path(root, variant_key, "greedy", load, seed)
    manifest_exists = manifest_path.is_file()
    if manifest_exists:
        _validate_manifest(manifest_path)
    if manifest_exists and output.is_file():
        existing = _read_json(output)
        meta = existing.get("meta") or {}
        if (
            meta.get("protocol_sha256") == bundle["protocol_sha256"]
            and meta.get("variant") == variant_key
            and meta.get("arm") == "greedy"
            and meta.get("load") == load
            and int(meta.get("seed", -1)) == seed
            and bool(existing.get("audit", {}).get("passed"))
        ):
            print(f"[resume] Greedy manifest/output {variant_key}/{load}/seed{seed}")
            return
        raise FileExistsError(f"incompatible existing Greedy output: {output}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_kwargs: dict[str, str] = (
        {"recorded_orders_path": str(manifest_path)}
        if manifest_exists
        else {"save_order_manifest": str(manifest_path)}
    )
    metrics, station_audit = _run_simulation(
        _run_config_path(protocol, variant_key, load),
        GreedyTaskAssigner(),
        seed,
        ticks,
        f"ScaleGeneralization/{variant_key}/Greedy/{load}/seed{seed}",
        **run_kwargs,
    )
    manifest = _validate_manifest(manifest_path)
    checks = _arm_checks(
        "greedy", metrics, manifest, station_audit, variant, ticks,
        generated_manifest=not manifest_exists,
    )
    if not all(checks.values()):
        raise RuntimeError(
            f"Greedy manifest audit failed {variant_key}/{load}/seed{seed}: "
            + ", ".join(key for key, value in checks.items() if not value)
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": bundle["protocol_sha256"],
            "variant": variant_key,
            "load": load,
            "seed": int(seed),
            "ticks": int(ticks),
            "arm": "greedy",
            "config": _run_config_path(protocol, variant_key, load),
            "map_rows": variant.rows,
            "map_cols": variant.cols,
            "num_robots": variant.robots,
            "num_stations": variant.stations,
            "manifest_mode": "generated" if not manifest_exists else "replayed",
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "station_audit": station_audit,
        "audit": {"passed": all(checks.values()), "checks": checks},
        "metrics": metrics,
    }
    _atomic_json(output, payload)
    print(f"[done] Greedy manifest {variant_key}/{load}/seed{seed}")


def _run_arm(root: Path, variant_key: str, arm: str, load: str, seed: int, ticks: int) -> None:
    variant = _variant(variant_key)
    if arm not in variant.arms:
        raise ValueError(f"arm {arm} is not executable for variant {variant_key}")
    bundle, protocol = _load_protocol(root)
    manifest_path = _manifest_path(root, variant_key, load, seed)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _validate_manifest(manifest_path)
    output = _output_path(root, variant_key, arm, load, seed)
    contract = _arm_contract(arm, protocol)
    if output.is_file():
        existing = _read_json(output)
        meta = existing.get("meta") or {}
        if (
            meta.get("protocol_sha256") == bundle["protocol_sha256"]
            and meta.get("variant") == variant_key
            and meta.get("arm") == arm
            and meta.get("load") == load
            and int(meta.get("seed", -1)) == seed
            and (meta.get("policy_contract") or {}).get("fingerprint_sha256")
            == contract["fingerprint_sha256"]
            and bool((existing.get("audit") or {}).get("passed"))
        ):
            print(f"[resume] {output}")
            return
        raise FileExistsError(f"incompatible existing output: {output}")
    assigner = _make_assigner(arm, protocol)
    metrics, station_audit = _run_simulation(
        _run_config_path(protocol, variant_key, load),
        assigner,
        seed,
        ticks,
        f"ScaleGeneralization/{variant_key}/{arm}/{load}/seed{seed}",
        recorded_orders_path=str(manifest_path),
    )
    checks = _arm_checks(
        arm, metrics, manifest, station_audit, variant, ticks,
        generated_manifest=False,
    )
    if not all(checks.values()):
        raise RuntimeError(
            f"arm audit failed {variant_key}/{arm}/{load}/seed{seed}: "
            + ", ".join(key for key, value in checks.items() if not value)
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": bundle["protocol_sha256"],
            "variant": variant_key,
            "load": load,
            "seed": int(seed),
            "ticks": int(ticks),
            "arm": arm,
            "config": _run_config_path(protocol, variant_key, load),
            "map_rows": variant.rows,
            "map_cols": variant.cols,
            "num_robots": variant.robots,
            "num_stations": variant.stations,
            "policy_contract": contract,
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "station_audit": station_audit,
        "audit": {"passed": all(checks.values()), "checks": checks},
        "metrics": metrics,
    }
    _atomic_json(output, payload)
    print(f"[done] {output}")


def _preflight(root: Path) -> None:
    bundle, protocol = _load_protocol(root)
    model_schema = protocol["model_checkpoint"]["model_schema"]
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        entry = {
            "variant": variant.key,
            "rows": variant.rows,
            "cols": variant.cols,
            "robots": variant.robots,
            "stations": variant.stations,
            "wm_compatible": variant.wm_compatible,
            "expected_arms": list(variant.arms),
            "status": "ok" if variant.wm_compatible else "baseline_only",
            "reason": (
                "frozen checkpoint matches station schema"
                if variant.wm_compatible
                else (
                    "unsupported_checkpoint_schema: checkpoint has "
                    f"{model_schema['num_stations']} stations/{model_schema['demand_dim']} "
                    f"demand channels; runtime config has {variant.stations} stations"
                )
            ),
            "loads": {},
        }
        for load in LOADS:
            path = Path(_run_config_path(protocol, variant.key, load))
            load_result: dict[str, Any] = {
                "path": path.as_posix(),
                "config_exists": path.is_file(),
                "config_loadable": False,
                "world_initializable": False,
                "error": None,
            }
            try:
                from Config.config_loader import load_config
                from WorldState.world import WorldState

                cfg = load_config(str(path))
                load_result["config_loadable"] = True
                world = WorldState(cfg)
                load_result["world_initializable"] = True
                load_result["actual_station_count"] = len(world.map_state.station_positions)
                load_result["actual_robot_count"] = len(world.agents)
                load_result["station_count_matches"] = (
                    load_result["actual_station_count"] == variant.stations
                )
                load_result["robot_count_matches"] = (
                    load_result["actual_robot_count"] == variant.robots
                )
                load_result["free_cells"] = sum(
                    1
                    for row in world.map_state.grid
                    for cell in row
                    if getattr(cell, "name", "") == "FREE"
                )
            except Exception as exc:  # report, do not hide the variant
                load_result["error"] = f"{type(exc).__name__}: {exc}"
            entry["loads"][load] = load_result
        if variant.wm_compatible:
            wm_probe: dict[str, Any] = {
                "phasec_init": False,
                "combo_s1_j1_init": False,
                "error": None,
            }
            try:
                from Config.config_loader import load_config
                from WorldState.world import WorldState

                probe_cfg = load_config(
                    _run_config_path(protocol, variant.key, "low")
                )
                probe_world = WorldState(probe_cfg)
                phasec_probe = _make_assigner("phasec", protocol)
                phasec_probe._init(probe_world)
                wm_probe["phasec_init"] = True
                combo_probe = _make_assigner("combo_s1_j1", protocol)
                combo_probe._init(probe_world)
                wm_probe["combo_s1_j1_init"] = True
            except Exception as exc:
                wm_probe["error"] = f"{type(exc).__name__}: {exc}"
            entry["world_model_preflight"] = wm_probe
        else:
            entry["world_model_preflight"] = {
                "phasec_init": False,
                "combo_s1_j1_init": False,
                "status": "skipped_unsupported_checkpoint_schema",
            }
        rows.append(entry)
    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "protocol_sha256": bundle["protocol_sha256"],
        "model_schema": model_schema,
        "variants": rows,
        "audit": {
            "all_configs_loadable": all(
                value["config_loadable"]
                for row in rows for value in row["loads"].values()
            ),
            "all_worlds_initializable": all(
                value["world_initializable"]
                for row in rows for value in row["loads"].values()
            ),
            "all_station_counts_match": all(
                value.get("station_count_matches", False)
                for row in rows for value in row["loads"].values()
            ),
            "all_robot_counts_match": all(
                value.get("robot_count_matches", False)
                for row in rows for value in row["loads"].values()
            ),
            "all_compatible_wm_initializations_pass": all(
                row["world_model_preflight"].get("phasec_init")
                and row["world_model_preflight"].get("combo_s1_j1_init")
                for row in rows if row["wm_compatible"]
            ),
            "six_station_world_model_blocked_explicitly": any(
                row["stations"] > model_schema["num_stations"]
                and row["status"] == "baseline_only"
                for row in rows
            ),
        },
    }
    _atomic_json(_preflight_path(root), report)
    for row in rows:
        print(
            f"[preflight] {row['variant']}: {row['status']} "
            f"{row['rows']}x{row['cols']} robots={row['robots']} "
            f"stations={row['stations']} arms={','.join(row['expected_arms'])}"
        )
    if not all(
        report["audit"][key]
        for key in (
            "all_configs_loadable",
            "all_worlds_initializable",
            "all_station_counts_match",
            "all_robot_counts_match",
            "all_compatible_wm_initializations_pass",
        )
    ):
        raise RuntimeError(
            "scale preflight failed; inspect "
            f"{_preflight_path(root)} before starting online runs"
        )


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarise(root: Path) -> None:
    bundle, protocol = _load_protocol(root)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[str] = []
    manifest_hashes: dict[tuple[str, str, int], set[str]] = {}
    for variant in VARIANTS:
        for arm in variant.arms:
            for load in LOADS:
                for seed in SEEDS:
                    path = _output_path(root, variant.key, arm, load, seed)
                    if not path.is_file():
                        missing.append(path.as_posix())
                        continue
                    payload = _read_json(path)
                    if not bool((payload.get("audit") or {}).get("passed")):
                        failed.append(path.as_posix())
                        continue
                    meta = payload.get("meta") or {}
                    if meta.get("protocol_sha256") != bundle["protocol_sha256"]:
                        failed.append(f"protocol:{path.as_posix()}")
                        continue
                    manifest = payload.get("manifest") or {}
                    key = (variant.key, load, seed)
                    manifest_hashes.setdefault(key, set()).add(
                        str(manifest.get("content_sha256"))
                    )
                    metrics = payload.get("metrics") or {}
                    arrivals = float(metrics.get("order_arrival_count", 0) or 0)
                    completed = float(metrics.get("completed_orders", 0) or 0)
                    row = {
                        "variant": variant.key,
                        "arm": arm,
                        "load": load,
                        "seed": seed,
                        "map_rows": variant.rows,
                        "map_cols": variant.cols,
                        "num_robots": variant.robots,
                        "num_stations": variant.stations,
                        "order_arrival_count": metrics.get("order_arrival_count"),
                        "completion_fraction": completed / arrivals if arrivals else None,
                    }
                    for metric in SUMMARY_METRICS:
                        row[metric] = metrics.get(metric)
                    rows.append(row)
    mismatches = [
        {"variant": key[0], "load": key[1], "seed": key[2], "hashes": sorted(value)}
        for key, value in sorted(manifest_hashes.items()) if len(value) != 1
    ]
    expected = sum(len(variant.arms) for variant in VARIANTS) * len(LOADS) * len(SEEDS)
    if missing or failed or mismatches or len(rows) != expected:
        raise RuntimeError(
            "scale summary incomplete: "
            f"rows={len(rows)}/{expected} missing={len(missing)} "
            f"failed={len(failed)} manifest_mismatches={len(mismatches)}"
        )
    aggregate: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    aggregate_metrics = ("completion_fraction",) + SUMMARY_METRICS
    for variant in VARIANTS:
        aggregate[variant.key] = {}
        for arm in variant.arms:
            aggregate[variant.key][arm] = {}
            for load in LOADS:
                selected = [
                    row for row in rows
                    if row["variant"] == variant.key
                    and row["arm"] == arm
                    and row["load"] == load
                ]
                summary: dict[str, Any] = {"runs": len(selected)}
                for metric in aggregate_metrics:
                    values = [
                        value for row in selected
                        if (value := _number(row.get(metric))) is not None
                    ]
                    summary[f"{metric}_mean"] = (
                        statistics.mean(values) if values else None
                    )
                    summary[f"{metric}_std"] = (
                        statistics.pstdev(values) if values else None
                    )
                aggregate[variant.key][arm][load] = summary
    report = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "protocol_sha256": bundle["protocol_sha256"],
        "expected_runs": expected,
        "completed_runs": len(rows),
        "variants": {
            variant.key: {
                "rows": variant.rows,
                "cols": variant.cols,
                "robots": variant.robots,
                "stations": variant.stations,
                "wm_compatible": variant.wm_compatible,
                "arms": list(variant.arms),
            }
            for variant in VARIANTS
        },
        "audit": {
            "passed": True,
            "all_run_audits_passed": True,
            "paired_manifest_hashes_match": True,
            "repaired_checkpoint_unchanged": (
                sha256_file(MODEL_CHECKPOINT)
                == protocol["model_checkpoint"]["sha256"]
            ),
        },
        "rows": rows,
        "aggregate": aggregate,
    }
    _atomic_json(root / "online_summary.json", report)
    hashes: list[str] = []
    for path in sorted(root.glob("per_arm/**/*.json")):
        hashes.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    for path in sorted((root / "frozen_configs").glob("**/*.json")):
        hashes.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    for path in (_protocol_path(root), _preflight_path(root), root / "online_summary.json", _frozen_inputs_path(root)):
        if path.is_file():
            hashes.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "results.sha256").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print(json.dumps({"completed_runs": len(rows), "expected_runs": expected, "output_root": root.as_posix()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "preflight", "manifest", "arm", "summarize"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--variant", choices=tuple(VARIANT_BY_KEY))
    parser.add_argument("--arm", choices=("greedy", "hungarian", "phasec", "combo_s1_j1"))
    parser.add_argument("--load", choices=LOADS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int, default=TICKS)
    args = parser.parse_args()
    if args.mode == "freeze":
        _freeze(args.output_root)
        return
    if args.mode == "preflight":
        _preflight(args.output_root)
        return
    if args.mode in ("manifest", "arm"):
        if args.variant is None or args.load is None or args.seed is None:
            raise SystemExit(f"{args.mode} requires --variant --load --seed")
        if args.seed not in SEEDS:
            raise SystemExit(f"seed must be in {SEEDS[0]}--{SEEDS[-1]}")
        if args.mode == "manifest":
            _prepare_manifest(args.output_root, args.variant, args.load, args.seed, args.ticks)
        else:
            if args.arm is None:
                raise SystemExit("arm mode requires --arm")
            _run_arm(args.output_root, args.variant, args.arm, args.load, args.seed, args.ticks)
        return
    _summarise(args.output_root)


if __name__ == "__main__":
    main()
