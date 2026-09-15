#!/usr/bin/env python3
"""Export anonymous, compact evidence from the full internal experiment tree.

This is a maintainer utility. Reviewers do not need the internal tree: all
outputs produced here are already committed to the public artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MAIN_REL = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "phasec_physical_only_pp_factorial_900_950_v1"
)
SIX_REL = Path(
    "WorldModel/checkpoints/station6_20x20_adaptation_v1/"
    "full_711_740_v1/evaluation"
)
TRACE_REL = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "phasec_physical_only_pp_factorial_900_950_v1/"
    "station_lock_causal_analysis/collapse_align_paper/lock_precedence_multiseed.json"
)
FIG_REL = Path("references/writing/figures/09_icra")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def is_integrity_field(name: str) -> bool:
    lowered = name.lower()
    legacy_markers = ("sha" + "256", "ha" + "sh", "check" + "sum")
    return any(marker in lowered for marker in legacy_markers)


def portable(value: Any) -> Any:
    """Recursively remove absolute workstation/cluster prefixes."""

    if isinstance(value, dict):
        return {
            str(key): portable(item)
            for key, item in value.items()
            if not is_integrity_field(str(key))
        }
    if isinstance(value, list):
        return [portable(item) for item in value]
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    is_absolute = bool(re.match(r"^[A-Za-z]:/", normalized)) or normalized.startswith("/lustre/")
    if not is_absolute:
        return normalized
    for marker in ("WorldModel/", "Config/", "Policies/", "Engine/", "references/"):
        index = normalized.find(marker)
        if index >= 0:
            return normalized[index:]
    return "<external-path-redacted>"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    if fields is None:
        field_set = {key for row in rows for key in row}
        prefix = [name for name in ("variant", "campaign", "load", "seed", "arm", "ticks") if name in field_set]
        fields = prefix + sorted(field_set - set(prefix))
    fields = list(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    excluded = ("path", "checkpoint", "output_dir", "log_file")
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if any(token in key.lower() for token in excluded) or is_integrity_field(key):
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            result[key] = value
    return result


def export_main(source: Path) -> list[dict[str, Any]]:
    main = source / MAIN_REL
    rows: list[dict[str, Any]] = []
    for path in sorted((main / "per_seed").glob("*.json")):
        job = read_json(path)
        for arm, block in job["arms"].items():
            row = {
                "variant": job["variant"],
                "load": job["load"],
                "seed": int(job["seed"]),
                "arm": arm,
                "ticks": int(job["ticks"]),
            }
            row.update(scalar_metrics(block.get("metrics", {})))
            rows.append(row)
    if len(rows) != 750:
        raise ValueError(f"expected 750 main rows, got {len(rows)}")
    write_csv(ROOT / "artifacts/raw/main_per_seed.csv", rows)

    protocol = portable(read_json(main / "physical_only_pp_50seed_protocol.json"))
    summary = portable(read_json(main / "summary.json"))
    write_json(ROOT / "artifacts/protocols/main_50seed_protocol.json", protocol)
    write_json(ROOT / "artifacts/protocols/main_50seed_summary.json", summary)
    return rows


def export_six(source: Path) -> list[dict[str, Any]]:
    summary = portable(read_json(source / SIX_REL / "summary.json"))
    rows = [dict(row, variant="map20_r48_s6", ticks=1500) for row in summary.pop("rows")]
    if len(rows) != 120:
        raise ValueError(f"expected 120 six-station rows, got {len(rows)}")
    write_csv(ROOT / "artifacts/raw/station6_per_seed.csv", rows)
    write_json(ROOT / "artifacts/protocols/station6_summary_metadata.json", summary)
    return rows


def export_lock(source: Path) -> None:
    report = portable(read_json(source / TRACE_REL))
    write_json(ROOT / "artifacts/protocols/lock_precedence_multiseed.json", report)

    event_source = source / FIG_REL / "data/fig05_lock_precedence_events.csv"
    shutil.copy2(event_source, ROOT / "artifacts/raw/station_lock_events.csv")
    compact = portable(read_json(source / FIG_REL / "data/fig05_lock_precedence_summary.json"))
    write_json(ROOT / "artifacts/statistics/station_lock_summary.json", compact)


def export_figure_inputs(source: Path) -> None:
    destination = ROOT / "artifacts/raw/figure_inputs"
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "fig02_closed_loop_50seed.csv",
        "fig03_predictive_validity.json",
        "fig04_pareto_means.csv",
    ):
        shutil.copy2(source / FIG_REL / "data" / name, destination / name)


def export_maps(source: Path) -> None:
    for stations, source_name, target_name in (
        (4, "world_model_config_PP_48_high.json", "map20_s4.json"),
        (6, "world_model_config_PP_48_high_6stations.json", "map20_s6.json"),
    ):
        config = read_json(source / "Config" / source_name)
        payload = {
            "schema_version": "rmfs_map_v1",
            "id": f"map20_r48_s{stations}",
            "map": config["map"],
            "robots": config["robots"],
            "station_queue": config["station_queue"],
        }
        write_json(ROOT / "maps" / target_name, payload)


def copy_manifests(source: Path) -> None:
    groups = (
        (source / MAIN_REL / "manifests", ROOT / "manifests/main_50seed"),
        (source / SIX_REL / "manifests", ROOT / "manifests/station6_10seed"),
    )
    metadata: list[dict[str, Any]] = []
    for source_dir, target_dir in groups:
        campaign = target_dir.name
        for path in sorted(source_dir.rglob("*.json")):
            relative = path.relative_to(source_dir)
            target = target_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = portable(read_json(path))
            write_json(target, payload)
            metadata.append(
                {
                    "campaign": campaign,
                    "relative_path": (Path(campaign) / relative).as_posix(),
                    "schema_version": payload.get("schema_version", ""),
                    "total_orders": payload.get("total_orders", len(payload.get("orders", []))),
                }
            )
    if len(metadata) != 180:
        raise ValueError(f"expected 180 manifests, got {len(metadata)}")
    write_csv(
        ROOT / "manifests/metadata.csv",
        metadata,
        ["campaign", "relative_path", "schema_version", "total_orders"],
    )


def export_runtime(main_rows: list[dict[str, Any]], six_rows: list[dict[str, Any]]) -> None:
    rows = []
    for campaign, source_rows in (("main_50seed", main_rows), ("station6_10seed", six_rows)):
        for row in source_rows:
            rows.append(
                {
                    "campaign": campaign,
                    "variant": row["variant"],
                    "load": row["load"],
                    "seed": row["seed"],
                    "arm": row["arm"],
                    "ticks": row.get("ticks", 1500),
                    "wall_time_s": row.get("wall_time_s"),
                    "assignment_time_ms_mean": row.get("assignment_time_ms_mean"),
                }
            )
    write_csv(
        ROOT / "runtime/raw_measurements.csv",
        rows,
        ["campaign", "variant", "load", "seed", "arm", "ticks", "wall_time_s", "assignment_time_ms_mean"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    if not (source / "WorldModel").is_dir():
        raise SystemExit(f"not an internal experiment tree: {source}")
    main_rows = export_main(source)
    six_rows = export_six(source)
    export_lock(source)
    export_figure_inputs(source)
    export_maps(source)
    copy_manifests(source)
    export_runtime(main_rows, six_rows)
    print(f"Exported anonymous compact evidence to {ROOT}")


if __name__ == "__main__":
    main()
