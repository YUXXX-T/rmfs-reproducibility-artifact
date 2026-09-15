"""Audit prerequisites and write the immutable Phase-C online cert bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from WorldModel.evaluation.phase_c_round1_cert_protocol import (
    BASELINE_HORIZON,
    CANDIDATE_HORIZON,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_TRAINING_PROTOCOL_SHA256,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOAD_CONFIGS,
    ONLINE_REPORT_SCHEMA_VERSION,
    REQUIRED_LOADS,
    canonical_sha256,
    formal_protocol,
    sha256_file,
)


NO_ASSIGN_SCHEMA = "wm_native_no_assign_action_v1"
NO_ASSIGN_ENCODING = "zero_action_tensors_v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _checkpoint_audit(path: Path, *, candidate: bool) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    expected = EXPECTED_CANDIDATE_SHA256 if candidate else EXPECTED_BASELINE_SHA256
    if digest != expected:
        raise ValueError(f"unexpected checkpoint SHA256: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"unsupported World Model checkpoint: {path}")
    config = dict(payload.get("model_config") or {})
    horizon = int(config.get("rollout_horizon", -1))
    expected_horizon = CANDIDATE_HORIZON if candidate else BASELINE_HORIZON
    if horizon != expected_horizon:
        raise ValueError(
            f"{path}: expected native H={expected_horizon}, got H={horizon}"
        )
    schema = dict(payload.get("action_schema") or {})
    if candidate:
        checks = {
            "schema_version": schema.get("schema_version") == NO_ASSIGN_SCHEMA,
            "supports_no_assign_candidate": bool(
                schema.get("supports_no_assign_candidate")
            ),
            "no_assign_encoding": (
                schema.get("no_assign_encoding") == NO_ASSIGN_ENCODING
            ),
            "complete_group_coverage": bool(schema.get("complete_group_coverage")),
            "zero_encoding_verified": bool(schema.get("zero_encoding_verified")),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError("candidate NO_ASSIGN audit failed: " + ", ".join(failed))
    elif schema.get("supports_no_assign_candidate"):
        raise ValueError("Stage-1 baseline unexpectedly enables native NO_ASSIGN")
    return {
        "path": path.as_posix(),
        "sha256": digest,
        "model_config": config,
        "rollout_horizon": horizon,
        "label_schema_version": payload.get("label_schema_version"),
        "action_schema": schema,
    }


def _training_audit(protocol_path: Path, summary_path: Path) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    if protocol.get("protocol_sha256") != EXPECTED_TRAINING_PROTOCOL_SHA256:
        raise ValueError("unexpected Phase-C training protocol hash")
    canonical = dict(protocol)
    canonical.pop("protocol_sha256", None)
    if canonical_sha256(canonical) != EXPECTED_TRAINING_PROTOCOL_SHA256:
        raise ValueError("Phase-C training protocol content/hash mismatch")
    expected_seeds = {
        "train_seeds": list(range(461, 468)),
        "validation_seeds": [468, 469],
        "offline_test_seeds": [470],
        "reserved_final_online_certification_seeds": list(range(471, 481)),
    }
    for key, expected in expected_seeds.items():
        if [int(value) for value in protocol.get(key, ())] != expected:
            raise ValueError(f"training protocol has unexpected {key}")
    if protocol.get("training", {}).get("early_stopping_monitor") != (
        "val_top1_regret_mean"
    ):
        raise ValueError("candidate selection was not frozen to validation regret")
    if (
        protocol.get("behavior_checkpoint_sha256") != EXPECTED_BASELINE_SHA256
        or bool(protocol.get("external_baseline_training_samples"))
        or bool(protocol.get("td_target_enabled"))
        or bool(protocol.get("td_value_head_enabled"))
        or bool(protocol.get("td_risk_v_head_enabled"))
    ):
        raise ValueError("unexpected Phase-C training-policy semantics")

    summary = _read_json(summary_path)
    stage2 = summary.get("stage2") or {}
    if int(stage2.get("best_epoch_by_regret", -1)) != 2:
        raise ValueError("expected Phase-C best-regret checkpoint at epoch 2")
    if int(stage2.get("best_epoch_by_rank", -1)) != 10:
        raise ValueError("unexpected diagnostic best-rank epoch")
    schema = summary.get("action_schema") or {}
    if schema.get("schema_version") != NO_ASSIGN_SCHEMA:
        raise ValueError("training summary lacks native NO_ASSIGN schema")
    return {
        "protocol": {
            "path": protocol_path.as_posix(),
            "file_sha256": sha256_file(protocol_path),
            "protocol_sha256": protocol["protocol_sha256"],
        },
        "summary": {
            "path": summary_path.as_posix(),
            "sha256": sha256_file(summary_path),
            "best_epoch_by_regret": stage2["best_epoch_by_regret"],
            "best_val_regret": stage2.get("best_val_regret"),
            "best_epoch_by_rank": stage2["best_epoch_by_rank"],
        },
    }


def _offline_audit(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline = _read_json(baseline_path)
    candidate = _read_json(candidate_path)
    for report in (baseline, candidate):
        meta = report.get("meta") or {}
        if meta.get("split") != "test" or int(meta.get("num_groups", -1)) != 130:
            raise ValueError("offline report is not the frozen seed-470 test split")
    b_rank = baseline.get("ranking") or {}
    c_rank = candidate.get("ranking") or {}
    if float(c_rank.get("top1_regret_mean", 1e9)) >= float(
        b_rank.get("top1_regret_mean", -1e9)
    ):
        raise ValueError("held-out seed470 did not improve top-1 regret")
    return {
        "baseline": {
            "path": baseline_path.as_posix(),
            "sha256": sha256_file(baseline_path),
            "ranking": b_rank,
        },
        "candidate": {
            "path": candidate_path.as_posix(),
            "sha256": sha256_file(candidate_path),
            "ranking": c_rank,
        },
        "checkpoint_selection_changed_after_test": False,
    }


def _pilot_audit(paths: list[Path]) -> dict[str, Any]:
    reports = {}
    observed_loads = []
    for path in paths:
        report = _read_json(path)
        if report.get("schema_version") != ONLINE_REPORT_SCHEMA_VERSION:
            raise ValueError(f"unexpected pilot schema: {path}")
        meta = report.get("meta") or {}
        load = str(meta.get("load"))
        observed_loads.append(load)
        if bool(meta.get("formal")) or [int(v) for v in meta.get("seeds", ())] != [9972]:
            raise ValueError(f"pilot is not the frozen development seed9972: {path}")
        if int(meta.get("ticks", -1)) != 500:
            raise ValueError(f"pilot is not the frozen 500-tick run: {path}")
        if not all(bool(row.get("passed")) for row in (report.get("pair_audits") or {}).values()):
            raise ValueError(f"pilot mechanism audit failed: {path}")
        checkpoint = meta.get("candidate_checkpoint") or {}
        if checkpoint.get("sha256") != EXPECTED_CANDIDATE_SHA256:
            raise ValueError(f"pilot used another candidate checkpoint: {path}")
        reports[load] = {
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "seeds": meta.get("seeds"),
            "ticks": meta.get("ticks"),
        }
    if tuple(sorted(observed_loads)) != tuple(sorted(REQUIRED_LOADS)):
        raise ValueError("development pilots must cover low/mid/high exactly")
    return reports


def _config_audit() -> dict[str, Any]:
    reports = {}
    for load, raw_path in LOAD_CONFIGS.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        reports[load] = {
            "path": path.as_posix(),
            "sha256": sha256_file(path),
        }
    return reports


def build_frozen_bundle(
    *,
    baseline_path: Path,
    candidate_path: Path,
    training_protocol_path: Path,
    training_summary_path: Path,
    offline_baseline_path: Path,
    offline_candidate_path: Path,
    pilot_paths: list[Path],
) -> dict[str, Any]:
    baseline = _checkpoint_audit(baseline_path, candidate=False)
    candidate = _checkpoint_audit(candidate_path, candidate=True)
    training = _training_audit(training_protocol_path, training_summary_path)
    offline = _offline_audit(offline_baseline_path, offline_candidate_path)
    pilots = _pilot_audit(pilot_paths)
    load_configs = _config_audit()
    protocol = formal_protocol()
    return {
        "schema_version": FROZEN_BUNDLE_SCHEMA_VERSION,
        "formal_protocol": protocol,
        "artifacts": {
            "baseline_checkpoint": baseline,
            "candidate_checkpoint": candidate,
            "training": training,
            "offline_seed470": offline,
            "development_pilots": pilots,
            "load_configs": load_configs,
        },
        "audit": {
            "candidate_selected_before_offline_test": True,
            "offline_test_seed_used_for_checkpoint_selection": False,
            "certification_seed_results_read": False,
            "post_freeze_checkpoint_or_threshold_tuning_allowed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--training-protocol", required=True)
    parser.add_argument("--training-summary", required=True)
    parser.add_argument("--offline-baseline", required=True)
    parser.add_argument("--offline-candidate", required=True)
    parser.add_argument("--pilot-reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    bundle = build_frozen_bundle(
        baseline_path=Path(args.baseline),
        candidate_path=Path(args.candidate),
        training_protocol_path=Path(args.training_protocol),
        training_summary_path=Path(args.training_summary),
        offline_baseline_path=Path(args.offline_baseline),
        offline_candidate_path=Path(args.offline_candidate),
        pilot_paths=[Path(path) for path in args.pilot_reports],
    )
    text = json.dumps(bundle, indent=2, ensure_ascii=False) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != text:
            raise SystemExit(f"refusing to change frozen Phase-C bundle: {output}")
        print(f"[audit] frozen Phase-C online bundle unchanged: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".partial")
        if partial.exists():
            partial.unlink()
        partial.write_text(text, encoding="utf-8")
        partial.replace(output)
        print(f"[freeze] wrote {output}")
    print(
        "Phase-C online protocol sha256 =",
        bundle["formal_protocol"]["protocol_sha256"],
    )
    print("certification seeds =", bundle["formal_protocol"]["formal_test"]["seeds"])
    print("ticks =", bundle["formal_protocol"]["formal_test"]["ticks"])


if __name__ == "__main__":
    main()
