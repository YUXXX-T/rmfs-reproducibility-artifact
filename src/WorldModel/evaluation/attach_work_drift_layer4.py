"""Attach a certified work-drift Layer-4 report to an immutable Layer-3 report."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from WorldModel.evaluation.evaluate_lyapunov_validity import (
    _role_decision,
    evaluate_drift_prediction,
)


SCHEMA_VERSION = "five_layer_lyapunov_with_work_drift_layer4_attachment_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-layer3-report", required=True)
    parser.add_argument("--layer4-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base_path = Path(args.base_layer3_report)
    layer4_path = Path(args.layer4_report)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing attachment: {output}")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    learned = json.loads(layer4_path.read_text(encoding="utf-8"))
    layer3 = base.get("layer3_incremental_information") or {}
    formal = layer3.get("formal_candidate_contract") or {}
    if layer3.get("candidate_component") != "work_group_range":
        raise SystemExit("base report is not the frozen work_group_range Layer-3 report")
    if not bool(layer3.get("supported")) or not bool(formal.get("passed")):
        raise SystemExit("base Layer 3 is not formally supported")
    if learned.get("schema_version") != "work_drift_group_range_evaluation_v1":
        raise SystemExit("incompatible Layer-4 report schema")

    attached_layer4 = evaluate_drift_prediction(
        [], predicted_drift_field=None, learned_report=learned
    )
    result = copy.deepcopy(base)
    result["schema_version"] = SCHEMA_VERSION
    result["base_report_schema_version"] = base.get("schema_version")
    result["layer4_world_model_drift_estimation"] = attached_layer4
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
        "base_layer3_report": str(base_path),
        "base_layer3_report_sha256": _sha256(base_path),
        "layer4_report": str(layer4_path),
        "layer4_report_sha256": _sha256(layer4_path),
        "base_report_modified": False,
        "layers_1_to_3_recomputed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {output}")
    print("Layer 3 status =", layer3.get("status"))
    print("Layer 4 status =", attached_layer4.get("status"))
    print("Layer 4 supported =", attached_layer4.get("supported"))
    print("five-layer role =", result["verdict"])
    print("note: original Layer-3 report was not modified or overwritten")


if __name__ == "__main__":
    main()
