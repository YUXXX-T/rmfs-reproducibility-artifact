"""Attach a frozen Layer-5 certificate to the immutable Layer-1--4 report."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from WorldModel.evaluation.evaluate_lyapunov_validity import (
    _role_decision,
    evaluate_closed_loop_attachment,
)
from WorldModel.evaluation.validate_work_drift_layer5 import (
    SCHEMA_VERSION as LAYER5_SCHEMA_VERSION,
)
from WorldModel.evaluation.work_drift_layer5_protocol import sha256_file


SCHEMA_VERSION = "five_layer_lyapunov_with_work_drift_layer5_attachment_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-layer4-report", required=True)
    parser.add_argument("--layer5-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_layer4_report)
    layer5_path = Path(args.layer5_report)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite five-layer attachment: {output}")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    layer5_source = json.loads(layer5_path.read_text(encoding="utf-8"))
    if base.get("verdict") != "READY_FOR_FROZEN_CLOSED_LOOP_VALIDATION":
        raise SystemExit("base report is not ready for frozen Layer 5")
    for name in (
        "layer1_state_potential",
        "layer2_action_controllable_drift",
        "layer3_incremental_information",
        "layer4_world_model_drift_estimation",
    ):
        if not bool((base.get(name) or {}).get("supported")):
            raise SystemExit(f"base report does not support {name}")
    if layer5_source.get("schema_version") != LAYER5_SCHEMA_VERSION:
        raise SystemExit("incompatible Layer-5 report schema")

    attached = evaluate_closed_loop_attachment(layer5_source)
    result = copy.deepcopy(base)
    result["schema_version"] = SCHEMA_VERSION
    result["base_report_schema_version"] = base.get("schema_version")
    result["layer5_closed_loop_stability"] = attached
    layers = {
        name: result[name]
        for name in (
            "layer1_state_potential",
            "layer2_action_controllable_drift",
            "layer3_incremental_information",
            "layer4_world_model_drift_estimation",
            "layer5_closed_loop_stability",
        )
    }
    decision = _role_decision(layers)
    result["decision"] = decision
    result["verdict"] = decision["recommended_role"]
    result["passed"] = (
        decision["recommended_role"] == "ONLINE_ACTION_AUXILIARY_SUPPORTED"
    )
    result["attachment_provenance"] = {
        "base_layer4_report": str(base_path),
        "base_layer4_report_sha256": sha256_file(base_path),
        "layer5_report": str(layer5_path),
        "layer5_report_sha256": sha256_file(layer5_path),
        "base_report_modified": False,
        "layers_1_to_4_recomputed": False,
        "layer5_source_verdict": layer5_source.get("verdict"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {output}")
    print("Layer 5 status =", attached.get("status"))
    print("Layer 5 supported =", attached.get("supported"))
    print("five-layer role =", result["verdict"])
    print("note: original Layer-1--4 report was not modified or overwritten")


if __name__ == "__main__":
    main()
