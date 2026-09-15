"""Wrap the available-run endpoint analysis with explicit recovery caveats."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from WorldModel.evaluation.analyze_phase_c_station_congestion_endpoint import (
    analyze,
)
from WorldModel.evaluation.pc_endpoint_recovery import (
    DEFAULT_RECOVERY_ROOT,
    RECOVERY_SCHEMA_VERSION,
)
from WorldModel.evaluation.phase_c_station_congestion_endpoint_protocol import (
    sha256_file,
)


EXPLORATORY_ANALYSIS_SCHEMA_VERSION = (
    "station_congestion_endpoint_exploratory_analysis_v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(
                f"refusing to overwrite changed exploratory analysis: {path}"
            )
        return
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _headline(base: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for layer_name, layer in (base.get("layers") or {}).items():
        result[layer_name] = {}
        for subset in ("all_stations", "context_station"):
            result[layer_name][subset] = {}
            for channel in ("traffic", "service"):
                metrics = layer[subset][channel]
                result[layer_name][subset][channel] = {
                    "primary_per_run_spearman_mean": metrics.get(
                        "per_run_spearman_mean"
                    ),
                    "primary_per_run_cluster_ci95": metrics.get(
                        "per_run_cluster_ci95"
                    ),
                    "primary_run_sign_consistency": metrics.get(
                        "run_sign_consistency"
                    ),
                    "secondary_pooled_spearman": metrics.get(
                        "pooled_spearman"
                    ),
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "bias": metrics.get("bias"),
                    "candidate_ranking_at_context_station": metrics.get(
                        "candidate_ranking_at_context_station"
                    ),
                    "sign": metrics.get("sign"),
                }
    return result


def run(recovery_root: Path) -> dict[str, Any]:
    recovery_manifest_path = (
        recovery_root / "exploratory_recovery_manifest.json"
    )
    recovery = _read_json(recovery_manifest_path)
    if recovery.get("schema_version") != RECOVERY_SCHEMA_VERSION:
        raise ValueError("wrong exploratory recovery manifest schema")
    if bool(recovery.get("formal_protocol_passed")):
        raise ValueError("recovery manifest must not claim formal passage")
    base = analyze(recovery_root, require_complete=False)
    base_path = recovery_root / "station_congestion_endpoint_h10_analysis.json"
    report = {
        "schema_version": EXPLORATORY_ANALYSIS_SCHEMA_VERSION,
        "development_only": True,
        "exploratory_underfilled": True,
        "formal_protocol_passed": False,
        "source_protocol_sha256": recovery["source_protocol_sha256"],
        "recovery_counts": recovery["counts"],
        "included_runs": recovery["included_runs"],
        "excluded_runs": recovery["excluded_runs"],
        "primary_analysis_contract": {
            "primary_unit": "source run",
            "primary_metrics": (
                "per-run Spearman mean and run-cluster bootstrap CI95"
            ),
            "pooled_metrics": "secondary because snapshot counts differ",
            "high_load_coverage": "underfilled",
            "permitted_use": (
                "decide whether a denser v2 collection is worth running"
            ),
            "forbidden_use": (
                "formal endpoint transport certification or publication claim"
            ),
        },
        "headline": _headline(base),
        "base_analysis": base,
        "source_files": {
            "recovery_manifest": recovery_manifest_path.name,
            "recovery_manifest_sha256": sha256_file(
                recovery_manifest_path
            ),
            "base_analysis": base_path.name,
            "base_analysis_sha256": sha256_file(base_path),
        },
    }
    report_path = recovery_root / (
        "pc_endpoint_h10_recovery_analysis.json"
    )
    _atomic_json(report_path, report)
    manifest = recovery_root / "exploratory_analyzed_outputs.sha256"
    manifest_text = "".join((
        f"{sha256_file(report_path)}  {report_path.name}\n",
        f"{sha256_file(base_path)}  {base_path.name}\n",
        f"{sha256_file(recovery_manifest_path)}  "
        f"{recovery_manifest_path.name}\n",
    ))
    if manifest.is_file():
        if manifest.read_text(encoding="utf-8") != manifest_text:
            raise FileExistsError(
                f"refusing to overwrite changed manifest: {manifest}"
            )
    else:
        manifest.write_text(manifest_text, encoding="utf-8", newline="\n")
    print(f"[complete] exploratory endpoint analysis: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT
    )
    args = parser.parse_args()
    run(args.recovery_root)


if __name__ == "__main__":
    main()
