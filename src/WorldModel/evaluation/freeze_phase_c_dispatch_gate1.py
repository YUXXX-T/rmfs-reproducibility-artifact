"""Audit and write the immutable Phase-C dispatch Gate-1 bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from WorldModel.evaluation.phase_c_dispatch_gate1_protocol import (
    EXPECTED_BASELINE_SHA256,
    EXPECTED_CANDIDATE_SHA256,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOAD_CONFIGS,
    formal_protocol,
    sha256_file,
)


NO_ASSIGN_SCHEMA = "wm_native_no_assign_action_v1"
NO_ASSIGN_ENCODING = "zero_action_tensors_v1"


def _checkpoint(path: Path, *, candidate: bool) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = EXPECTED_CANDIDATE_SHA256 if candidate else EXPECTED_BASELINE_SHA256
    digest = sha256_file(path)
    if digest != expected:
        raise ValueError(f"unexpected checkpoint SHA256: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"unsupported World Model checkpoint: {path}")
    schema = dict(payload.get("action_schema") or {})
    if candidate:
        checks = {
            "schema": schema.get("schema_version") == NO_ASSIGN_SCHEMA,
            "support": bool(schema.get("supports_no_assign_candidate")),
            "encoding": schema.get("no_assign_encoding") == NO_ASSIGN_ENCODING,
            "coverage": bool(schema.get("complete_group_coverage")),
            "zero_verified": bool(schema.get("zero_encoding_verified")),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError("candidate NO_ASSIGN audit failed: " + ", ".join(failed))
    elif bool(schema.get("supports_no_assign_candidate")):
        raise ValueError("Stage-1 unexpectedly supports native NO_ASSIGN")
    return {
        "path": path.as_posix(),
        "sha256": digest,
        "model_config": dict(payload.get("model_config") or {}),
        "action_schema": schema,
        "label_schema_version": payload.get("label_schema_version"),
    }


def _file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": path.as_posix(), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol = formal_protocol()
    bundle = {
        "schema_version": FROZEN_BUNDLE_SCHEMA_VERSION,
        "formal_protocol": protocol,
        "artifacts": {
            "baseline_checkpoint": _checkpoint(
                Path(args.baseline), candidate=False
            ),
            "candidate_checkpoint": _checkpoint(
                Path(args.candidate), candidate=True
            ),
            "design_plan": _file(Path(args.plan)),
            "load_configs": {
                load: _file(Path(path)) for load, path in LOAD_CONFIGS.items()
            },
        },
        "audit": {
            "seeds_481_486_results_read_before_freeze": False,
            "round1_seeds_471_480_used_for_threshold_or_scale": False,
            "tunable_dispatch_parameters": 0,
            "checkpoint_retraining_for_gate1": False,
            "new_learned_head": False,
        },
    }
    output = Path(args.output)
    text = json.dumps(bundle, indent=2, ensure_ascii=False) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != text:
            raise SystemExit(f"refusing to change frozen Gate-1 bundle: {output}")
        print(f"[audit] frozen dispatch Gate-1 bundle unchanged: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".partial")
        partial.unlink(missing_ok=True)
        partial.write_text(text, encoding="utf-8")
        partial.replace(output)
        print(f"[freeze] wrote {output}")
    print("dispatch Gate-1 protocol sha256 =", protocol["protocol_sha256"])
    print("development seeds =", protocol["test"]["seeds"])
    print("ticks =", protocol["test"]["ticks"])


if __name__ == "__main__":
    main()
