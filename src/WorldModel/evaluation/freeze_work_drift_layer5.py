"""Audit prerequisites and write the immutable Layer-5 protocol bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from WorldModel.evaluation.work_drift_layer5_protocol import (
    EXPECTED_BASE_WORLD_MODEL_SHA256,
    EXPECTED_HEAD_SHA256,
    EXPECTED_LAYER4_PROTOCOL_SHA256,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    derive_frozen_lambda,
    formal_protocol,
    sha256_file,
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def build_frozen_bundle(
    *,
    world_model_path: Path,
    head_path: Path,
    layer3_report_path: Path,
    layer4_report_path: Path,
    five_layer_l4_report_path: Path,
) -> dict:
    for path in (
        world_model_path,
        head_path,
        layer3_report_path,
        layer4_report_path,
        five_layer_l4_report_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    wm_hash = sha256_file(world_model_path)
    head_hash = sha256_file(head_path)
    if wm_hash != EXPECTED_BASE_WORLD_MODEL_SHA256:
        raise ValueError("base World Model SHA256 changed")
    if head_hash != EXPECTED_HEAD_SHA256:
        raise ValueError("WorkDriftHead SHA256 changed")

    layer3 = _read_json(layer3_report_path)
    layer4 = _read_json(layer4_report_path)
    combined = _read_json(five_layer_l4_report_path)
    lambda_audit = derive_frozen_lambda(layer3)

    if (
        layer4.get("schema_version")
        != "work_drift_group_range_evaluation_v1"
        or layer4.get("status")
        != "WORLD_MODEL_WORK_GROUP_RANGE_ESTIMATION_SUPPORTED"
        or not bool(layer4.get("passed"))
        or not bool(layer4.get("supported"))
    ):
        raise ValueError("Layer-4 held-out report is not certified")
    layer4_protocol = layer4.get("formal_protocol") or {}
    if layer4_protocol.get("protocol_sha256") != EXPECTED_LAYER4_PROTOCOL_SHA256:
        raise ValueError("unexpected Layer-4 protocol hash")
    if (layer4.get("head_checkpoint") or {}).get("sha256") != head_hash:
        raise ValueError("Layer-4 report refers to a different WorkDriftHead")
    if (layer4.get("base_world_model") or {}).get("sha256") != wm_hash:
        raise ValueError("Layer-4 report refers to a different World Model")

    required_layers = (
        "layer1_state_potential",
        "layer2_action_controllable_drift",
        "layer3_incremental_information",
        "layer4_world_model_drift_estimation",
    )
    if combined.get("verdict") != "READY_FOR_FROZEN_CLOSED_LOOP_VALIDATION":
        raise ValueError("five-layer report is not ready for frozen Layer 5")
    if not all(bool((combined.get(name) or {}).get("supported")) for name in required_layers):
        raise ValueError("five-layer report does not support every Layer 1--4")
    layer5 = combined.get("layer5_closed_loop_stability") or {}
    if layer5.get("status") != "NOT_EVALUATED_MISSING_CLOSED_LOOP_REPORT":
        raise ValueError("Layer 5 was already evaluated in the prerequisite report")

    payload = torch.load(head_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "work_drift_group_range_head_v1":
        raise ValueError("unexpected WorkDriftHead checkpoint schema")
    if (payload.get("base_world_model") or {}).get("sha256") != wm_hash:
        raise ValueError("WorkDriftHead was trained against a different World Model")
    if (
        (payload.get("formal_protocol") or {}).get("protocol_sha256")
        != EXPECTED_LAYER4_PROTOCOL_SHA256
    ):
        raise ValueError("WorkDriftHead protocol differs from Layer 4")

    protocol = formal_protocol()
    return {
        "schema_version": FROZEN_BUNDLE_SCHEMA_VERSION,
        "formal_protocol": protocol,
        "integration": {
            "lambda_audit": lambda_audit,
            "formula": protocol["integration"]["formula"],
            "candidate_transform": protocol["integration"]["group_range"],
        },
        "artifacts": {
            "base_world_model": {
                "path": str(world_model_path),
                "sha256": wm_hash,
            },
            "work_drift_head": {
                "path": str(head_path),
                "sha256": head_hash,
            },
            "layer3_report": {
                "path": str(layer3_report_path),
                "sha256": sha256_file(layer3_report_path),
            },
            "layer4_report": {
                "path": str(layer4_report_path),
                "sha256": sha256_file(layer4_report_path),
            },
            "five_layer_l4_report": {
                "path": str(five_layer_l4_report_path),
                "sha256": sha256_file(five_layer_l4_report_path),
            },
        },
        "audit": {
            "layers_1_to_4_supported": True,
            "layer5_previously_unevaluated": True,
            "head_online_ready_before_layer5": bool(payload.get("online_ready", False)),
            "layer5_seed_results_read": False,
            "post_freeze_lambda_tuning_allowed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--layer3-report", required=True)
    parser.add_argument("--layer4-report", required=True)
    parser.add_argument("--five-layer-l4-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    bundle = build_frozen_bundle(
        world_model_path=Path(args.world_model),
        head_path=Path(args.head),
        layer3_report_path=Path(args.layer3_report),
        layer4_report_path=Path(args.layer4_report),
        five_layer_l4_report_path=Path(args.five_layer_l4_report),
    )
    text = json.dumps(bundle, indent=2, ensure_ascii=False) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != text:
            raise SystemExit(f"refusing to change frozen Layer-5 bundle: {output}")
        print(f"[audit] frozen Layer-5 bundle unchanged: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"[freeze] wrote {output}")
    print("Layer-5 protocol sha256 =", bundle["formal_protocol"]["protocol_sha256"])
    print("frozen lambda =", bundle["integration"]["lambda_audit"]["value"])
    print("certification seeds =", bundle["formal_protocol"]["formal_test"]["seeds"])


if __name__ == "__main__":
    main()
