"""Freeze the independent analytic-H5 Layer-5 localisation bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from WorldModel.evaluation.analytic_work_layer5_protocol import (
    EFFICIENCY_DECOMPOSITION_REPORT,
    EXPECTED_BASE_WORLD_MODEL_SHA256,
    EXPECTED_EFFICIENCY_DECOMPOSITION_REPORT_SHA256,
    EXPECTED_HORIZON_DEVELOPMENT_REPORT_SHA256,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    HORIZON_DEVELOPMENT_REPORT,
    protocol,
    sha256_file,
)


SOURCE_FILES = (
    "WorldModel/core/analytic_work_relief.py",
    "Policies/TaskAssigner/WorldModelTaskAssigner/world_model_task_assigner.py",
    "WorldModel/evaluation/evaluate_online_v6.py",
    "WorldModel/evaluation/decision_snapshot_probe.py",
    "WorldModel/evaluation/analytic_work_layer5_protocol.py",
    "WorldModel/evaluation/freeze_analytic_work_layer5.py",
    "WorldModel/evaluation/run_analytic_work_layer5.py",
    "WorldModel/evaluation/replay_analytic_work_snapshots.py",
    "WorldModel/evaluation/run_analytic_work_layer5_postfailure_focus.sh",
)


def _artifact(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": path.as_posix(), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--lyapunov-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    world_model = Path(args.world_model)
    lyapunov_config = Path(args.lyapunov_config)
    world_model_artifact = _artifact(world_model)
    if world_model_artifact["sha256"] != EXPECTED_BASE_WORLD_MODEL_SHA256:
        raise SystemExit("base World Model hash differs from the frozen model")
    horizon_report = _artifact(Path(HORIZON_DEVELOPMENT_REPORT))
    if (
        horizon_report["sha256"]
        != EXPECTED_HORIZON_DEVELOPMENT_REPORT_SHA256
    ):
        raise SystemExit("analytic horizon development report hash changed")
    efficiency_report = _artifact(Path(EFFICIENCY_DECOMPOSITION_REPORT))
    if (
        efficiency_report["sha256"]
        != EXPECTED_EFFICIENCY_DECOMPOSITION_REPORT_SHA256
    ):
        raise SystemExit("analytic efficiency decomposition report hash changed")

    source = {}
    for value in SOURCE_FILES:
        path = Path(value)
        source[value] = _artifact(path)
    bundle = {
        "schema_version": FROZEN_BUNDLE_SCHEMA_VERSION,
        "protocol": protocol(),
        "artifacts": {
            "base_world_model": world_model_artifact,
            "lyapunov_config": _artifact(lyapunov_config),
            "horizon_development_report": horizon_report,
            "efficiency_decomposition_report": efficiency_report,
        },
        "source_files": source,
        "no_learned_auxiliary_head": True,
        "preserves_failed_legacy_layer5_bundle": True,
    }

    output = Path(args.output)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != bundle:
            raise SystemExit(
                f"frozen analytic Layer-5 bundle changed: {output}"
            )
        print(f"[audit] frozen analytic Layer-5 bundle unchanged: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[freeze] wrote {output}")
    print(
        "analytic Layer-5 protocol sha256 = "
        + str(bundle["protocol"]["protocol_sha256"])
    )
    print("learned work/TD head = False")


if __name__ == "__main__":
    main()
