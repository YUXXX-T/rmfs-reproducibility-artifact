"""Paired zero-shot map/fleet-density generalisation campaign.

The protocol keeps the four-station service interface and its explicit
seven-slot U-shaped queues fixed while scaling the warehouse geometry.  Pod
blocks are extended only along rows; their width remains two cells so every
stored pod can leave the block laterally.  The nine evaluated configurations
form a 3 x 3 map-size/fleet-density factorial:

    20x20:  48,  60,  72 robots,  8 blocks of  6x2 pods
    30x30: 108, 135, 162 robots, 12 blocks of  9x2 pods
    40x40: 192, 240, 288 robots, 16 blocks of 12x2 pods

All maps therefore have pod density 0.24, while the three fleet levels have
robot densities 0.12, 0.15, and 0.18.  One Greedy-generated order manifest is
replayed across Greedy, Hungarian, JSQ, Phase C, and repaired Combo S1+J1.

This module intentionally reuses the audited execution and summary machinery
from ``run_phase_c_scale_generalization``.  Only campaign metadata, map
derivation, JSQ registration, and the stronger layout preflight are replaced;
no simulator or deployed policy module is modified.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from WorldModel.evaluation import run_phase_c_scale_generalization as base


SCHEMA_VERSION = "phase_c_density_scale_run_v1"
PROTOCOL_SCHEMA_VERSION = "phase_c_density_scale_protocol_v1"
SUMMARY_SCHEMA_VERSION = "phase_c_density_scale_summary_v1"
PREFLIGHT_SCHEMA_VERSION = "phase_c_density_scale_preflight_v1"

DEFAULT_OUTPUT_ROOT = (
    base.BASE_ROOT / "phasec_density_scale_generalization_701_710_v1"
)
LOADS = ("low", "mid", "high")
SEEDS = tuple(range(701, 711))
TICKS = 1500
TOP_M = 10
ARMS = ("greedy", "hungarian", "jsq", "phasec", "combo_s1_j1")
MANIFEST_SOURCE_VARIANT = "map20_r48_s4"


@dataclass(frozen=True)
class Variant:
    key: str
    rows: int
    cols: int
    robots: int
    stations: int
    description: str
    pod_zone_rows: int
    pod_zone_count: int
    robot_density: float

    @property
    def preserve_20x20_layout(self) -> bool:
        # All maps are regenerated from the same explicit scale contract,
        # including 20x20.  Its generated geometry is byte-for-byte
        # equivalent to the source map, apart from requested fleet size.
        return False

    @property
    def wm_compatible(self) -> bool:
        return self.stations == 4

    @property
    def arms(self) -> tuple[str, ...]:
        return ARMS


def _variant_set(size: int, zone_rows: int, zone_count: int) -> tuple[Variant, ...]:
    robot_levels = {
        20: (48, 60, 72),
        30: (108, 135, 162),
        40: (192, 240, 288),
    }[size]
    density_labels = ("sparse", "medium", "dense")
    return tuple(
        Variant(
            key=f"map{size}_r{robots}_s4",
            rows=size,
            cols=size,
            robots=robots,
            stations=4,
            description=(
                f"{size}x{size}, four explicit seven-slot stations, "
                f"{density_labels[index]} robot density"
            ),
            pod_zone_rows=zone_rows,
            pod_zone_count=zone_count,
            robot_density=robots / float(size * size),
        )
        for index, robots in enumerate(robot_levels)
    )


VARIANTS = (
    *_variant_set(20, 6, 8),
    *_variant_set(30, 9, 12),
    *_variant_set(40, 12, 16),
)
VARIANT_BY_KEY = {variant.key: variant for variant in VARIANTS}


def _explicit_config_path(variant: Variant, load: str) -> Path:
    return Path(
        "Config"
    ) / f"world_model_config_PP_{variant.rows}x{variant.cols}_r{variant.robots}_{load}.json"


def _station_columns(cols: int) -> tuple[int, int]:
    # Preserve the source station coordinate (4 on indices 0..19) under a
    # normalized map resize.  The right station remains the exact mirror.
    left = int(round(4 * (cols - 1) / 19.0))
    return left, cols - 1 - left


def _station_spec(
    station_id: int,
    row: int,
    col: int,
    *,
    inward_row: int,
    queue_side: int,
) -> dict[str, Any]:
    """Return the original 20x20 seven-slot U queue at a new edge anchor."""

    inner = row + inward_row
    return {
        "id": station_id,
        "row": row,
        "col": col,
        "queue": {
            "service": [row, col],
            "queue": [
                [row, col + queue_side],
                [row, col + 2 * queue_side],
                [row, col + 3 * queue_side],
                [inner, col + 3 * queue_side],
                [inner, col + 2 * queue_side],
            ],
            "buffer": [[inner, col + queue_side]],
            "entry": [inner, col],
            "exit": [row, col - queue_side],
        },
    }


def _explicit_station_specs(rows: int, cols: int) -> list[dict[str, Any]]:
    left, right = _station_columns(cols)
    return [
        _station_spec(1, 0, left, inward_row=1, queue_side=-1),
        _station_spec(2, 0, right, inward_row=1, queue_side=1),
        _station_spec(3, rows - 1, left, inward_row=-1, queue_side=-1),
        _station_spec(4, rows - 1, right, inward_row=-1, queue_side=1),
    ]


def _pod_layout(size: int) -> tuple[list[int], list[int], int]:
    if size == 20:
        return [3, 11], [2, 6, 10, 14], 6
    if size == 30:
        return [4, 17], [3, 7, 11, 15, 19, 23], 9
    if size == 40:
        return [6, 22], [4, 8, 12, 16, 20, 24, 28, 32], 12
    raise ValueError(f"unsupported density-scale map size: {size}")


def _pod_zones(rows: int, cols: int) -> list[dict[str, int]]:
    if rows != cols:
        raise ValueError(f"this protocol requires square maps: {rows}x{cols}")
    row_starts, col_starts, zone_rows = _pod_layout(rows)
    # Preserve the source JSON's column-major, top/bottom-paired zone order.
    return [
        {
            "origin_row": row,
            "origin_col": col,
            "num_rows": zone_rows,
            "num_cols": 2,
        }
        for col in col_starts
        for row in row_starts
    ]


def _derive_config(source: Mapping[str, Any], variant: Variant) -> dict[str, Any]:
    import copy

    derived = copy.deepcopy(source)
    map_cfg = derived.setdefault("map", {})
    map_cfg["rows"] = variant.rows
    map_cfg["cols"] = variant.cols
    map_cfg["stations"] = _explicit_station_specs(variant.rows, variant.cols)
    map_cfg.pop("pod_layout", None)
    map_cfg["pod_zones"] = _pod_zones(variant.rows, variant.cols)

    robots = derived.setdefault("robots", {})
    robots["num_robots"] = variant.robots
    robots["starts"] = []
    robots["random_starts"] = True
    return derived


def _derive_configs(root: Path) -> dict[str, dict[str, Any]]:
    """Freeze reviewed configs and reject drift from the scale contract."""

    entries: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        entries[variant.key] = {}
        for load in LOADS:
            source_path = Path(base.SOURCE_LOAD_CONFIGS[load])
            source = base._read_json(source_path)
            if (
                len(source.get("map", {}).get("stations", [])) != 4
                or int(source.get("robots", {}).get("num_robots", -1)) != 48
            ):
                raise ValueError(
                    f"scale source must be 48-robot/4-station: {source_path}"
                )
            expected = _derive_config(source, variant)
            explicit_path: Path | None = None
            if variant.rows in (30, 40):
                explicit_path = _explicit_config_path(variant, load)
                if not explicit_path.is_file():
                    raise FileNotFoundError(explicit_path)
                reviewed = base._read_json(explicit_path)
                if reviewed != expected:
                    changed = base._changed_paths(expected, reviewed)
                    raise ValueError(
                        f"reviewed config violates scale contract: {explicit_path}; "
                        f"changed paths={changed[:20]}"
                    )
                derived = reviewed
            else:
                derived = expected

            changed = base._changed_paths(source, derived)
            allowed = {
                "map.rows",
                "map.cols",
                "map.stations",
                "map.pod_zones",
                "robots.num_robots",
                "robots.starts",
                "robots.random_starts",
            }
            unexpected = [
                path
                for path in changed
                if not any(
                    path == prefix
                    or path.startswith(prefix + ".")
                    or path.startswith(prefix + "[")
                    for prefix in allowed
                )
            ]
            if unexpected:
                raise AssertionError(
                    f"{variant.key}/{load} changes fields outside the "
                    f"scale contract: {unexpected}"
                )

            frozen_path = base._config_path(root, variant.key, load)
            base._atomic_json(frozen_path, derived)
            entry: dict[str, Any] = {
                "source_path": source_path.as_posix(),
                "source_sha256": base.sha256_file(source_path),
                "derived_path": frozen_path.as_posix(),
                "derived_sha256": base.sha256_file(frozen_path),
                "changed_paths": changed,
                "rows": variant.rows,
                "cols": variant.cols,
                "robots": variant.robots,
                "stations": variant.stations,
                "config_origin": (
                    "reviewed_explicit_config"
                    if explicit_path is not None
                    else "verified_20x20_source_derivation"
                ),
            }
            if explicit_path is not None:
                entry["explicit_config_path"] = explicit_path.as_posix()
                entry["explicit_config_sha256"] = base.sha256_file(explicit_path)
            entries[variant.key][load] = entry
    return entries


_ORIGINAL_MAKE_ASSIGNER = base._make_assigner
_ORIGINAL_ARM_CONTRACT = base._arm_contract


def _make_assigner(arm: str, protocol: Mapping[str, Any]):
    if arm == "jsq":
        from Policies.TaskAssigner.JSQTaskAssigner import JSQTaskAssigner

        return JSQTaskAssigner()
    return _ORIGINAL_MAKE_ASSIGNER(arm, protocol)


def _arm_contract(arm: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    if arm != "jsq":
        return _ORIGINAL_ARM_CONTRACT(arm, protocol)
    contract = {
        "arm_key": "jsq",
        "station_admission": base.STATION_ADMISSION_PHYSICAL_ONLY,
        "checkpoint": None,
        "policy_family": "analytic_baseline",
        "context_scheduler": "dynamic_join_shortest_committed_queue",
        "robot_selector": "greedy_nearest_within_context",
    }
    return {
        **contract,
        "fingerprint_sha256": base._canonical_sha(contract),
    }


def _protocol_path(root: Path) -> Path:
    return root / "density_scale_protocol.json"


def _preflight_path(root: Path) -> Path:
    return root / "density_scale_preflight.json"


def _manifest_path(root: Path, variant: str, load: str, seed: int) -> Path:
    del variant
    return root / "order_manifests" / f"orders_{load}_seed{seed}.json"


def _make_protocol(root: Path) -> dict[str, Any]:
    for path in (
        base.MODEL_CHECKPOINT,
        base.PSI_HEAD_CHECKPOINT,
        base.PSI_SCALE_CONTRACT_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, path in base.SOURCE_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    schema = base._model_schema()
    if schema != {"demand_dim": 9, "num_stations": 4}:
        raise ValueError(f"unexpected frozen World Model station schema: {schema}")
    configs = base._derive_configs(root)
    variants_payload = {
        variant.key: {
            "rows": variant.rows,
            "cols": variant.cols,
            "robots": variant.robots,
            "stations": variant.stations,
            "description": variant.description,
            "arms": list(variant.arms),
            "wm_compatible": True,
            "pod_zone_shape": [variant.pod_zone_rows, 2],
            "pod_zone_count": variant.pod_zone_count,
            "pod_homes": int(0.24 * variant.rows * variant.cols),
            "pod_density": 0.24,
            "robot_density": variant.robot_density,
            "robot_per_pod": variant.robots
            / float(0.24 * variant.rows * variant.cols),
        }
        for variant in VARIANTS
    }
    payload: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "purpose": (
            "zero-shot graph-size and fleet-density transfer under a fixed "
            "four-station service interface"
        ),
        "seeds": list(SEEDS),
        "loads": list(LOADS),
        "ticks": TICKS,
        "top_m": TOP_M,
        "station_admission": base.STATION_ADMISSION_PHYSICAL_ONLY,
        "path_planner": "PrioritizedPathPlanner",
        "manifest_source": "GreedyTaskAssigner",
        "manifest_source_variant": MANIFEST_SOURCE_VARIANT,
        "exact_manifest_replay": True,
        "exact_manifest_replay_across_all_variants": True,
        "variants": variants_payload,
        "load_configs": configs,
        "model_checkpoint": {
            "path": base.MODEL_CHECKPOINT.as_posix(),
            "sha256": base.sha256_file(base.MODEL_CHECKPOINT),
            "model_schema": schema,
        },
        "static_j1_artifacts": {
            "head": base.PSI_HEAD_CHECKPOINT.as_posix(),
            "head_sha256": base.sha256_file(base.PSI_HEAD_CHECKPOINT),
            "scale_contract": base.PSI_SCALE_CONTRACT_PATH.as_posix(),
            "scale_contract_sha256": base.sha256_file(
                base.PSI_SCALE_CONTRACT_PATH
            ),
        },
        "zero_shot": {
            "world_model_retrained_for_new_map_or_fleet": False,
            "checkpoint_uses_pretrained_longrisk_repair": True,
            "s1_retuned": False,
            "j1_retuned": False,
            "station_admission_modified": False,
            "core_policy_code_modified": False,
        },
        "scale_contract": {
            "station_count": 4,
            "station_physical_slots": 7,
            "station_queue_slots": 5,
            "station_buffer_slots": 1,
            "station_layout": "explicit_scaled_U_shape",
            "station_service_parameters_unchanged": True,
            "pod_zone_width": 2,
            "pod_zone_rows_by_map": {"20": 6, "30": 9, "40": 12},
            "pod_density": 0.24,
            "robot_densities": [0.12, 0.15, 0.18],
            "arrival_parameters_unchanged_across_maps": True,
            "canonical_arrival_stream_shared_across_maps": True,
            "large_map_configs_are_reviewed_explicit_json": True,
        },
        "phasec_config": dict(base.PHASEC_CONFIG),
        "combo_selector_config": dict(base.combo_selector_config("combo")),
        "source_hashes": {
            name: {"path": path.as_posix(), "sha256": base.sha256_file(path)}
            for name, path in base.SOURCE_FILES.items()
        },
    }
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol": payload,
        "protocol_sha256": base._canonical_sha(payload),
    }


def _raw_station_cells(station: Mapping[str, Any]) -> set[tuple[int, int]]:
    queue = station["queue"]
    values = [queue["service"], *queue["queue"], *queue["buffer"]]
    if queue.get("entry") is not None:
        values.append(queue["entry"])
    if queue.get("exit") is not None:
        values.append(queue["exit"])
    return {tuple(value) for value in values}


def _reachable_nonpod_cells(world: Any) -> tuple[int, int, set[tuple[int, int]]]:
    from WorldState.map_state import CellType

    blocked_types = {
        CellType.OBSTACLE,
        CellType.STATION,
        CellType.POD_HOME,
        CellType.STATION_SERVICE,
        CellType.STATION_QUEUE,
        CellType.STATION_BUFFER,
    }
    traversable = {
        (row, col)
        for row in range(world.map_state.rows)
        for col in range(world.map_state.cols)
        if world.map_state.grid[row][col] not in blocked_types
    }
    if not traversable:
        return 0, 0, set()
    reached = {next(iter(traversable))}
    pending = deque(reached)
    while pending:
        row, col = pending.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (row + dr, col + dc)
            if nxt in traversable and nxt not in reached:
                reached.add(nxt)
                pending.append(nxt)
    return len(traversable), len(reached), reached


def _layout_audit(config_path: Path, variant: Variant) -> dict[str, Any]:
    from Config.config_loader import load_config
    from WorldState.map_state import CellType
    from WorldState.world import WorldState

    raw = base._read_json(config_path)
    cfg = load_config(str(config_path))
    world = WorldState(cfg)
    zones = raw["map"]["pod_zones"]
    stations = raw["map"]["stations"]
    zone_cells: set[tuple[int, int]] = set()
    laterally_exposed = True
    for zone in zones:
        r0, c0 = int(zone["origin_row"]), int(zone["origin_col"])
        height, width = int(zone["num_rows"]), int(zone["num_cols"])
        for row in range(r0, r0 + height):
            zone_cells.update((row, col) for col in range(c0, c0 + width))
            for col, outside_col in ((c0, c0 - 1), (c0 + 1, c0 + 2)):
                if not (0 <= outside_col < variant.cols):
                    laterally_exposed = False
                elif world.map_state.grid[row][outside_col] == CellType.POD_HOME:
                    laterally_exposed = False

    station_cells: set[tuple[int, int]] = set()
    station_cells_unique = True
    station_internal_cells_unique = True
    for station in stations:
        queue = station["queue"]
        raw_cells = [
            queue["service"],
            *queue["queue"],
            *queue["buffer"],
            queue["entry"],
            queue["exit"],
        ]
        cells = _raw_station_cells(station)
        if len(raw_cells) != 9 or len(cells) != 9:
            station_internal_cells_unique = False
        if station_cells & cells:
            station_cells_unique = False
        station_cells |= cells

    traversable, reached_count, reached = _reachable_nonpod_cells(world)
    entry_exit = {
        tuple(station["queue"][kind])
        for station in stations
        for kind in ("entry", "exit")
    }
    station_capacities = {
        int(station_id): queue.capacity
        for station_id, queue in world.station_state.stations.items()
    }
    pod_type_counts = Counter(
        str(pod.pod_type) for pod in world.pod_state.pods.values()
    )
    expected_pods = int(round(variant.rows * variant.cols * 0.24))
    free_cells = sum(
        cell == CellType.FREE
        for row in world.map_state.grid
        for cell in row
    )
    checks = {
        "dimensions_match": (
            world.map_state.rows == variant.rows
            and world.map_state.cols == variant.cols
        ),
        "robot_count_matches": len(world.agents) == variant.robots,
        "four_stations": len(world.station_state.stations) == 4,
        "explicit_queue_coordinates": all(
            all(key in station["queue"] for key in (
                "service", "queue", "buffer", "entry", "exit"
            ))
            for station in stations
        ),
        "five_queue_slots_each": all(
            len(station["queue"]["queue"]) == 5 for station in stations
        ),
        "one_buffer_slot_each": all(
            len(station["queue"]["buffer"]) == 1 for station in stations
        ),
        "seven_physical_slots_each": all(
            capacity == 7 for capacity in station_capacities.values()
        ),
        "station_internal_cells_unique": station_internal_cells_unique,
        "station_cells_do_not_overlap": station_cells_unique,
        "station_and_pod_cells_disjoint": not bool(station_cells & zone_cells),
        "pod_zone_count_matches": len(zones) == variant.pod_zone_count,
        "pod_zone_height_matches": all(
            int(zone["num_rows"]) == variant.pod_zone_rows for zone in zones
        ),
        "pod_zone_width_is_two": all(
            int(zone["num_cols"]) == 2 for zone in zones
        ),
        "pod_zone_cells_unique": len(zone_cells) == expected_pods,
        "pod_count_matches_density": world.pod_state.total_pods == expected_pods,
        "pod_density_is_0_24": (
            world.pod_state.total_pods / float(variant.rows * variant.cols)
            == 0.24
        ),
        "all_pod_rows_laterally_exposed": laterally_exposed,
        "carrying_traversable_space_connected": traversable == reached_count,
        "all_station_entries_exits_reachable": entry_exit <= reached,
        "enough_free_cells_for_robot_starts": free_cells >= variant.robots,
        "all_random_robot_starts_unique": (
            len({agent.position for agent in world.agents}) == variant.robots
        ),
    }
    return {
        "path": config_path.as_posix(),
        "rows": variant.rows,
        "cols": variant.cols,
        "robots": variant.robots,
        "robot_density": variant.robot_density,
        "pod_zone_shape": [variant.pod_zone_rows, 2],
        "pod_zone_count": len(zones),
        "pod_count": world.pod_state.total_pods,
        "pod_density": world.pod_state.total_pods
        / float(variant.rows * variant.cols),
        "robot_per_pod": variant.robots / float(world.pod_state.total_pods),
        "pod_type_counts": dict(sorted(pod_type_counts.items())),
        "station_capacities": station_capacities,
        "station_cells": len(station_cells),
        "free_cells": free_cells,
        "carrying_traversable_cells": traversable,
        "carrying_reachable_cells": reached_count,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _preflight(root: Path) -> None:
    bundle, protocol = base._load_protocol(root)
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        entry: dict[str, Any] = {
            "variant": variant.key,
            "description": variant.description,
            "loads": {},
            "world_model_preflight": {
                "phasec_init": False,
                "combo_s1_j1_init": False,
                "error": None,
            },
        }
        for load in LOADS:
            path = Path(base._run_config_path(protocol, variant.key, load))
            try:
                audit = _layout_audit(path, variant)
            except Exception as exc:
                audit = {
                    "path": path.as_posix(),
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "checks": {},
                }
            entry["loads"][load] = audit

        try:
            from Config.config_loader import load_config
            from WorldState.world import WorldState

            probe_cfg = load_config(
                base._run_config_path(protocol, variant.key, "low")
            )
            probe_world = WorldState(probe_cfg)
            phasec = _make_assigner("phasec", protocol)
            phasec._init(probe_world)
            entry["world_model_preflight"]["phasec_init"] = True
            combo = _make_assigner("combo_s1_j1", protocol)
            combo._init(probe_world)
            entry["world_model_preflight"]["combo_s1_j1_init"] = True
        except Exception as exc:
            entry["world_model_preflight"]["error"] = (
                f"{type(exc).__name__}: {exc}"
            )
        rows.append(entry)

    audit = {
        "all_layout_checks_pass": all(
            load_audit.get("passed", False)
            for row in rows
            for load_audit in row["loads"].values()
        ),
        "all_world_model_initializations_pass": all(
            row["world_model_preflight"]["phasec_init"]
            and row["world_model_preflight"]["combo_s1_j1_init"]
            for row in rows
        ),
        "model_schema_is_four_station": (
            protocol["model_checkpoint"]["model_schema"]
            == {"demand_dim": 9, "num_stations": 4}
        ),
    }
    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "protocol_sha256": bundle["protocol_sha256"],
        "variants": rows,
        "audit": {**audit, "passed": all(audit.values())},
    }
    base._atomic_json(_preflight_path(root), report)
    for row in rows:
        load_status = ",".join(
            f"{load}={'ok' if value.get('passed') else 'FAIL'}"
            for load, value in row["loads"].items()
        )
        print(f"[preflight] {row['variant']}: {load_status}")
    if not report["audit"]["passed"]:
        raise RuntimeError(
            f"density-scale preflight failed; inspect {_preflight_path(root)}"
        )


def _pairing_audit_path(root: Path) -> Path:
    return root / "cross_variant_manifest_audit.json"


def _cross_variant_pairing_audit(root: Path) -> dict[str, Any]:
    bundle, _ = base._load_protocol(root)
    groups: list[dict[str, Any]] = []
    all_pass = True
    expected_references = len(VARIANTS) * len(ARMS)
    for load in LOADS:
        for seed in SEEDS:
            manifest_path = _manifest_path(root, "", load, seed)
            manifest = base._validate_manifest(manifest_path)
            expected_hash = str(manifest.get("manifest_sha256"))
            observed_hashes: set[str] = set()
            observed_paths: set[str] = set()
            references = 0
            missing: list[str] = []
            for variant in VARIANTS:
                for arm in ARMS:
                    output = base._output_path(
                        root, variant.key, arm, load, seed
                    )
                    if not output.is_file():
                        missing.append(output.as_posix())
                        continue
                    payload = base._read_json(output)
                    observed_hashes.add(
                        str((payload.get("manifest") or {}).get("content_sha256"))
                    )
                    observed_paths.add(
                        str((payload.get("manifest") or {}).get("path"))
                    )
                    references += 1
            passed = (
                not missing
                and references == expected_references
                and observed_hashes == {expected_hash}
                and observed_paths == {manifest_path.as_posix()}
            )
            all_pass = all_pass and passed
            groups.append({
                "load": load,
                "seed": seed,
                "manifest_path": manifest_path.as_posix(),
                "manifest_sha256": expected_hash,
                "run_references": references,
                "expected_run_references": expected_references,
                "observed_hashes": sorted(observed_hashes),
                "observed_paths": sorted(observed_paths),
                "missing": missing,
                "passed": passed,
            })
    return {
        "schema_version": "phase_c_density_scale_cross_variant_pairing_v1",
        "protocol_sha256": bundle["protocol_sha256"],
        "manifest_source_variant": MANIFEST_SOURCE_VARIANT,
        "groups": groups,
        "audit": {
            "passed": all_pass,
            "shared_manifest_count": len(groups),
            "expected_shared_manifest_count": len(LOADS) * len(SEEDS),
            "all_arms_and_variants_replay_identical_stream_per_group": all_pass,
        },
    }


def _summarise(root: Path) -> None:
    pairing = _cross_variant_pairing_audit(root)
    if not pairing["audit"]["passed"]:
        raise RuntimeError(
            "cross-variant manifest pairing audit failed; campaign is incomplete"
        )
    base._atomic_json(_pairing_audit_path(root), pairing)
    base._summarise(root)
    results_path = root / "results.sha256"
    pairing_line = (
        f"{base.sha256_file(_pairing_audit_path(root))}  "
        f"{_pairing_audit_path(root).relative_to(root).as_posix()}"
    )
    lines = results_path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if not line.endswith(
        "  cross_variant_manifest_audit.json"
    )]
    results_path.write_text(
        "\n".join([*lines, pairing_line]) + "\n", encoding="utf-8"
    )


def _configure_worker_threads() -> None:
    intra = max(1, int(os.environ.get("RMFS_DENSITY_WORKER_THREADS", "1")))
    inter = max(
        1, int(os.environ.get("RMFS_DENSITY_WORKER_INTEROP_THREADS", "1"))
    )
    try:
        import torch

        torch.set_num_threads(intra)
        torch.set_num_interop_threads(inter)
    except RuntimeError:
        # A caller embedding this runner may already have initialized the
        # inter-op pool; Slurm worker processes execute this path only once.
        pass


def _configure_base() -> None:
    """Bind the shared audited runner to this isolated campaign contract."""

    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.PROTOCOL_SCHEMA_VERSION = PROTOCOL_SCHEMA_VERSION
    base.SUMMARY_SCHEMA_VERSION = SUMMARY_SCHEMA_VERSION
    base.PREFLIGHT_SCHEMA_VERSION = PREFLIGHT_SCHEMA_VERSION
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    base.LOADS = LOADS
    base.SEEDS = SEEDS
    base.TICKS = TICKS
    base.TOP_M = TOP_M
    base.VARIANTS = VARIANTS
    base.VARIANT_BY_KEY = VARIANT_BY_KEY
    base.RUNNER_PATH = Path(__file__)
    source_files = dict(base.SOURCE_FILES)
    source_files["runner"] = Path(__file__)
    source_files["shared_scale_runner"] = Path(
        "WorldModel/evaluation/run_phase_c_scale_generalization.py"
    )
    source_files["jsq_assigner"] = Path(
        "Policies/TaskAssigner/JSQTaskAssigner/jsq_task_assigner.py"
    )
    source_files["pod_initializer"] = Path(
        "Policies/PodInitializer/DefaultPodInitializer/__init__.py"
    )
    source_files["order_generator"] = Path(
        "Policies/OrderGenerator/ZipfOrderGenerator/zipf_order_generator.py"
    )
    for variant in VARIANTS:
        if variant.rows == 20:
            continue
        for load in LOADS:
            source_files[f"config_{variant.key}_{load}"] = (
                _explicit_config_path(variant, load)
            )
    base.SOURCE_FILES = source_files
    base._derive_config = _derive_config
    base._derive_configs = _derive_configs
    base._make_assigner = _make_assigner
    base._arm_contract = _arm_contract
    base._make_protocol = _make_protocol
    base._protocol_path = _protocol_path
    base._preflight_path = _preflight_path
    base._manifest_path = _manifest_path
    base._preflight = _preflight


def main() -> None:
    _configure_base()
    _configure_worker_threads()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("freeze", "preflight", "manifest", "arm", "summarize"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--variant", choices=tuple(VARIANT_BY_KEY))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--load", choices=LOADS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int, default=TICKS)
    args = parser.parse_args()

    if args.mode == "freeze":
        base._freeze(args.output_root)
        return
    if args.mode == "preflight":
        _preflight(args.output_root)
        return
    if args.mode in ("manifest", "arm"):
        if args.variant is None or args.load is None or args.seed is None:
            raise SystemExit(
                f"{args.mode} requires --variant --load --seed"
            )
        if args.seed not in SEEDS:
            raise SystemExit(f"seed must be in {SEEDS[0]}--{SEEDS[-1]}")
        if args.mode == "manifest":
            if args.variant != MANIFEST_SOURCE_VARIANT:
                raise SystemExit(
                    "manifest mode is restricted to the canonical source "
                    f"variant {MANIFEST_SOURCE_VARIANT}; run other Greedy "
                    "variants with --mode arm --arm greedy"
                )
            base._prepare_manifest(
                args.output_root,
                args.variant,
                args.load,
                args.seed,
                args.ticks,
            )
        else:
            if args.arm is None:
                raise SystemExit("arm mode requires --arm")
            base._run_arm(
                args.output_root,
                args.variant,
                args.arm,
                args.load,
                args.seed,
                args.ticks,
            )
        return
    _summarise(args.output_root)


if __name__ == "__main__":
    main()
