"""Run the paired online audit for the LongRiskHead-only repair campaign."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from Policies.TaskAssigner import (
    GreedyTaskAssigner,
    HungarianTaskAssigner,
    WorldModelTaskAssigner,
)
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_context_assigner import (
    PsiDispatchContextWorldModelTaskAssigner,
)
from WorldModel.core.long_risk_schema import (
    LONG_RISK_SCHEMA_VERSION,
    long_risk_runtime_contract,
)
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
from WorldModel.evaluation.phase_c_long_risk_head_repair_protocol import (
    ARM_KEYS,
    ARM_LABELS,
    COMPARISONS,
    EVALUATION_BUNDLE_SCHEMA_VERSION,
    EVALUATION_ROOT,
    LOAD_CONFIGS,
    LOADS,
    ONLINE_TEST_SEEDS,
    PHASEC_CONFIG,
    PSI_SCALE_CONTRACT,
    REBOUND_PSI_HEAD,
    REPAIRED_CHECKPOINT,
    REPORT_METRICS,
    SOURCE_CHECKPOINT,
    SOURCE_PSI_HEAD,
    TICKS,
    TOP_M,
    canonical_sha256,
    combo_selector_config,
    formal_protocol,
    sha256_file,
)
from WorldModel.evaluation.run_phase_c_s1_j1_long_risk_correction import (
    _physical_only_audit_contract,
)
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


BUNDLE_FILENAME = "phase_c_long_risk_head_repair_evaluation_bundle.json"
OUTPUT_SCHEMA_VERSION = "phase_c_long_risk_head_repair_online_output_v1"
TRAINING_SUMMARY = REPAIRED_CHECKPOINT.parent / "train_summary.json"
TENSOR_AUDIT = REPAIRED_CHECKPOINT.parent / "tensor_audit.json"
REBOUND_SUMMARY = REBOUND_PSI_HEAD.parent / "rebind_summary.json"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_exact(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite changed file: {path}")
        return
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _bundle_path(root: Path) -> Path:
    return root / BUNDLE_FILENAME


def _policy_contract(arm: str) -> dict:
    if arm not in ARM_KEYS:
        raise ValueError(arm)
    contract: dict[str, Any] = {
        "arm": arm,
        "label": ARM_LABELS[arm],
        "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
        "in_transit_committed_cap": None,
    }
    if arm in ("greedy", "hungarian"):
        contract.update({"policy_family": "analytic_baseline", "checkpoint": None})
    elif arm in ("source_phasec", "repaired_phasec", "repaired_combo_j0"):
        checkpoint = (
            SOURCE_CHECKPOINT if arm == "source_phasec" else REPAIRED_CHECKPOINT
        )
        signal = "none" if arm != "repaired_combo_j0" else "combo"
        contract.update({
            "policy_family": "world_model",
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint),
            "robot_selector": "phasec_s0" if signal == "none" else "corrected_quantile_combo_s1",
            "energy_drift_signal": signal,
            "context_scheduler": "j0_prefix",
        })
    else:
        checkpoint = SOURCE_CHECKPOINT if arm == "source_combo_j1" else REPAIRED_CHECKPOINT
        head = SOURCE_PSI_HEAD if arm == "source_combo_j1" else REBOUND_PSI_HEAD
        signal = "event_logit" if arm == "repaired_event_j1" else "combo"
        contract.update({
            "policy_family": "world_model_static_j1",
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint),
            "psi_head_checkpoint": head.as_posix(),
            "psi_head_checkpoint_sha256": sha256_file(head),
            "psi_scale_contract": PSI_SCALE_CONTRACT.as_posix(),
            "psi_scale_contract_sha256": sha256_file(PSI_SCALE_CONTRACT),
            "robot_selector": "corrected_quantile_combo_s1",
            "energy_drift_signal": signal,
            "context_scheduler": "static_j1",
            "psi_context_mode": "j_ascending",
            "j_refresh": "once_per_proposal_batch",
        })
    return {**contract, "fingerprint_sha256": canonical_sha256(contract)}


def _make_assigner(arm: str):
    if arm == "greedy":
        return GreedyTaskAssigner()
    if arm == "hungarian":
        return HungarianTaskAssigner()
    if arm == "source_phasec":
        return WorldModelTaskAssigner(
            checkpoint_path=str(SOURCE_CHECKPOINT),
            top_m=TOP_M,
            **dict(PHASEC_CONFIG),
        )
    if arm == "repaired_phasec":
        return WorldModelTaskAssigner(
            checkpoint_path=str(REPAIRED_CHECKPOINT),
            top_m=TOP_M,
            **dict(PHASEC_CONFIG),
        )
    if arm == "repaired_combo_j0":
        return WorldModelTaskAssigner(
            checkpoint_path=str(REPAIRED_CHECKPOINT),
            top_m=TOP_M,
            energy_conv_random_flip_seed=0,
            **combo_selector_config("combo"),
        )
    if arm in ("source_combo_j1", "repaired_event_j1", "repaired_combo_j1"):
        if arm == "source_combo_j1":
            checkpoint = SOURCE_CHECKPOINT
            head = SOURCE_PSI_HEAD
            signal = "combo"
        else:
            checkpoint = REPAIRED_CHECKPOINT
            head = REBOUND_PSI_HEAD
            signal = "event_logit" if arm == "repaired_event_j1" else "combo"
        return PsiDispatchContextWorldModelTaskAssigner(
            psi_head_checkpoint=str(head),
            psi_scale_contract=str(PSI_SCALE_CONTRACT),
            psi_context_mode="j_ascending",
            psi_trace_enabled=False,
            psi_trace_max_records=0,
            allow_phasec_s0_robot_scorer=False,
            checkpoint_path=str(checkpoint),
            top_m=TOP_M,
            energy_conv_random_flip_seed=0,
            **combo_selector_config(signal),
        )
    raise ValueError(arm)


def _manifest_path(root: Path, load: str, seed: int) -> Path:
    return root / "order_manifests" / f"orders_{load}_seed{seed}.json"


def _output_path(root: Path, arm: str, load: str, seed: int) -> Path:
    return root / "per_arm" / arm / f"{load}_seed{seed}.json"


def _audit(
    arm: str,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    station_audit: Mapping[str, Any] | None,
) -> dict:
    checks = {
        "manifest_hash": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_count": int(metrics.get("order_arrival_count", -1))
        == int(manifest.get("total_orders", -2)),
        "physical_only": (
            station_audit is not None
            and station_audit.get("mode") == STATION_ADMISSION_PHYSICAL_ONLY
        ),
        "physical_capacity_no_violation": (
            station_audit is not None
            and int(station_audit.get("physical_capacity_violation_count", -1)) == 0
        ),
    }
    if arm == "greedy":
        checks["manifest_generated"] = not bool(metrics.get("order_arrival_replayed"))
        checks["no_model_calls"] = int(metrics.get("model_assign_calls", 0)) == 0
    elif arm == "hungarian":
        checks["manifest_replayed"] = bool(metrics.get("order_arrival_replayed"))
        checks["no_model_calls"] = int(metrics.get("model_assign_calls", 0)) == 0
    else:
        checks.update({
            "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
            "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
            "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
            "long_risk_schema": metrics.get("long_risk_schema_version")
            == LONG_RISK_SCHEMA_VERSION,
            "long_risk_contract": metrics.get("long_risk_runtime_contract")
            == long_risk_runtime_contract(),
        })
        if arm in ("source_combo_j1", "repaired_event_j1", "repaired_combo_j1"):
            expected_head = SOURCE_PSI_HEAD if arm == "source_combo_j1" else REBOUND_PSI_HEAD
            checks.update({
                "j1_mode": metrics.get("psi_dispatch_mode") == "j_ascending",
                "j1_head_loaded": bool(metrics.get("psi_dispatch_head_loaded")),
                "j1_head_hash": metrics.get("psi_dispatch_head_checkpoint_sha256")
                == sha256_file(expected_head),
                "j1_evaluated": int(metrics.get("psi_dispatch_eval_calls", 0)) > 0,
            })
        else:
            checks["j0_mode"] = metrics.get("psi_dispatch_mode") in (None, "off")
    if station_audit is not None:
        checks["admission_modes_exact"] = station_audit.get("admission_modes") == [
            STATION_ADMISSION_PHYSICAL_ONLY
        ] or station_audit.get("mode") == STATION_ADMISSION_PHYSICAL_ONLY
    return {"passed": all(checks.values()), "checks": checks}


def _freeze(root: Path) -> None:
    for path in (SOURCE_CHECKPOINT, SOURCE_PSI_HEAD, PSI_SCALE_CONTRACT):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not REPAIRED_CHECKPOINT.is_file() or not REBOUND_PSI_HEAD.is_file():
        raise FileNotFoundError(
            "repaired checkpoint and rebound station head must be created first"
        )
    for path in (TRAINING_SUMMARY, TENSOR_AUDIT, REBOUND_SUMMARY):
        if not path.is_file():
            raise FileNotFoundError(path)
    tensor_audit = _read_json(TENSOR_AUDIT)
    if not tensor_audit.get("passed"):
        raise ValueError("LongRiskHead tensor audit did not pass")
    training_summary = _read_json(TRAINING_SUMMARY)
    if not (training_summary.get("trained_head") or {}).get("test"):
        raise ValueError("LongRiskHead offline test metrics are missing")
    source_head_payload = torch.load(
        SOURCE_PSI_HEAD, map_location="cpu", weights_only=False
    )
    rebound_head_payload = torch.load(
        REBOUND_PSI_HEAD, map_location="cpu", weights_only=False
    )
    if source_head_payload.get("source_encoder_checkpoint_sha256") != sha256_file(
        SOURCE_CHECKPOINT
    ):
        raise ValueError("source static-J1 head is not bound to source Phase-C")
    if rebound_head_payload.get("source_encoder_checkpoint_sha256") != sha256_file(
        REPAIRED_CHECKPOINT
    ):
        raise ValueError("rebound static-J1 head is not bound to repaired Phase-C")
    protocol = formal_protocol()
    artifacts = {}
    for name, path in {
        "source_checkpoint": SOURCE_CHECKPOINT,
        "source_psi_head": SOURCE_PSI_HEAD,
        "psi_scale_contract": PSI_SCALE_CONTRACT,
        "repaired_checkpoint": REPAIRED_CHECKPOINT,
        "rebound_psi_head": REBOUND_PSI_HEAD,
        "training_summary": TRAINING_SUMMARY,
        "tensor_audit": TENSOR_AUDIT,
        "rebound_summary": REBOUND_SUMMARY,
    }.items():
        artifacts[name] = {"path": path.as_posix(), "sha256": sha256_file(path)}
    payload = {
        "schema_version": EVALUATION_BUNDLE_SCHEMA_VERSION,
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "artifacts": artifacts,
        "arms": {arm: _policy_contract(arm) for arm in ARM_KEYS},
    }
    root.mkdir(parents=True, exist_ok=True)
    _write_json_exact(_bundle_path(root), payload)
    print(f"[complete] evaluation bundle: {_bundle_path(root)}")


def _verify_bundle(path: Path) -> dict:
    bundle = _read_json(path)
    if bundle.get("schema_version") != EVALUATION_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"wrong bundle schema: {path}")
    protocol = bundle.get("protocol")
    if canonical_sha256(protocol) != bundle.get("protocol_sha256"):
        raise ValueError("bundle protocol hash mismatch")
    if protocol != formal_protocol():
        raise ValueError("bundle protocol differs from the current frozen protocol")
    expected_artifacts = {
        "source_checkpoint": SOURCE_CHECKPOINT,
        "source_psi_head": SOURCE_PSI_HEAD,
        "psi_scale_contract": PSI_SCALE_CONTRACT,
        "repaired_checkpoint": REPAIRED_CHECKPOINT,
        "rebound_psi_head": REBOUND_PSI_HEAD,
        "training_summary": TRAINING_SUMMARY,
        "tensor_audit": TENSOR_AUDIT,
        "rebound_summary": REBOUND_SUMMARY,
    }
    artifacts = bundle.get("artifacts") or {}
    if set(artifacts) != set(expected_artifacts):
        raise ValueError("bundle artifact set differs from the frozen contract")
    for name, expected_path in expected_artifacts.items():
        artifact = artifacts[name]
        artifact_path = Path(str(artifact.get("path", "")))
        if artifact_path.as_posix() != expected_path.as_posix():
            raise ValueError(f"artifact path changed: {name}")
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if sha256_file(artifact_path) != artifact.get("sha256"):
            raise ValueError(f"artifact hash mismatch: {name}")
    expected_arms = {arm: _policy_contract(arm) for arm in ARM_KEYS}
    if bundle.get("arms") != expected_arms:
        raise ValueError("bundle arm contracts differ from the frozen policy")
    return bundle


def _run_arm(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    bundle = _verify_bundle(Path(args.frozen_bundle or _bundle_path(root)))
    if args.arm not in ARM_KEYS or args.load not in LOADS:
        raise ValueError("unknown arm/load")
    seed = int(args.seed)
    if seed not in ONLINE_TEST_SEEDS:
        raise ValueError(f"online test seed must be in {list(ONLINE_TEST_SEEDS)}")
    if int(args.ticks) != TICKS:
        raise ValueError(f"ticks must equal {TICKS}")
    manifest_path = _manifest_path(root, args.load, seed)
    output_path = _output_path(root, args.arm, args.load, seed)
    if output_path.is_file():
        payload = _read_json(output_path)
        meta = payload.get("meta") or {}
        policy = meta.get("policy_contract") or {}
        checks = {
            "schema": payload.get("schema_version") == OUTPUT_SCHEMA_VERSION,
            "arm": meta.get("arm") == args.arm,
            "label": meta.get("arm_label") == ARM_LABELS[args.arm],
            "load": meta.get("load") == args.load,
            "seed": int(meta.get("seed", -1)) == seed,
            "ticks": int(meta.get("ticks", -1)) == int(args.ticks),
            "bundle": meta.get("bundle_sha256")
            == sha256_file(Path(args.frozen_bundle or _bundle_path(root))),
            "policy": policy.get("fingerprint_sha256")
            == _policy_contract(args.arm)["fingerprint_sha256"],
            "audit": bool((payload.get("audit") or {}).get("passed")),
        }
        if all(checks.values()):
            print(f"[resume] {output_path}")
            return
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(
            f"incompatible existing output {output_path}: {failed}"
        )
    if args.arm != "greedy" and not manifest_path.is_file():
        raise FileNotFoundError(f"missing greedy manifest: {manifest_path}")
    assigner = _make_assigner(args.arm)
    run_kwargs = {}
    if args.arm == "greedy":
        run_kwargs["save_order_manifest"] = str(manifest_path)
    else:
        run_kwargs["recorded_orders_path"] = str(manifest_path)
    with _physical_only_audit_contract() as holder:
        metrics = _run_one_assigner(
            LOAD_CONFIGS[args.load],
            assigner,
            seed,
            int(args.ticks),
            trace_label=ARM_LABELS[args.arm],
            **run_kwargs,
        )
    if hasattr(assigner, "psi_dispatch_metrics"):
        metrics.update(assigner.psi_dispatch_metrics())
    station_audit = None
    probe = holder.get("station_probe")
    if probe is not None:
        station_audit = probe.summary()
        metrics["station_admission_mode"] = STATION_ADMISSION_PHYSICAL_ONLY
        metrics["station_audit"] = station_audit
    manifest = _read_json(manifest_path)
    audit = _audit(args.arm, metrics, manifest, station_audit)
    if not audit["passed"]:
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise RuntimeError(f"audit failed: {', '.join(failed)}")
    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "meta": {
            "arm": args.arm,
            "arm_label": ARM_LABELS[args.arm],
            "load": args.load,
            "seed": seed,
            "ticks": int(args.ticks),
            "manifest_mode": "generated" if args.arm == "greedy" else "replayed",
            "manifest_path": manifest_path.as_posix(),
            "policy_contract": _policy_contract(args.arm),
            "bundle_sha256": sha256_file(Path(args.frozen_bundle or _bundle_path(root))),
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "sha256": manifest.get("manifest_sha256"),
            "file_sha256": sha256_file(manifest_path),
            "total_orders": manifest.get("total_orders"),
        },
        "metrics": metrics,
        "station_audit": station_audit,
        "audit": audit,
    }
    _write_json_exact(output_path, payload)
    print(f"[complete] {output_path}")


def _summary(root: Path) -> None:
    bundle = _verify_bundle(_bundle_path(root))
    rows = []
    for arm in ARM_KEYS:
        for load in LOADS:
            for seed in ONLINE_TEST_SEEDS:
                path = _output_path(root, arm, load, seed)
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = _read_json(path)
                if not (payload.get("audit") or {}).get("passed"):
                    raise ValueError(f"failed output audit: {path}")
                metrics = payload.get("metrics") or {}
                row = {"arm": arm, "load": load, "seed": seed}
                row.update({metric: metrics.get(metric) for metric in REPORT_METRICS})
                rows.append(row)

    nonperturbation_metrics = tuple(
        metric
        for metric in REPORT_METRICS
        if metric not in ("wall_time_s", "assignment_time_ms_mean")
    )
    nonperturbation_checks = {}
    for load in LOADS:
        for seed in ONLINE_TEST_SEEDS:
            source = next(
                row
                for row in rows
                if row["arm"] == "source_phasec"
                and row["load"] == load
                and row["seed"] == seed
            )
            repaired = next(
                row
                for row in rows
                if row["arm"] == "repaired_phasec"
                and row["load"] == load
                and row["seed"] == seed
            )
            cell = f"{load}_seed{seed}"
            nonperturbation_checks[cell] = {
                metric: source[metric] == repaired[metric]
                for metric in nonperturbation_metrics
            }
    failed_nonperturbation = [
        f"{cell}:{metric}"
        for cell, checks in nonperturbation_checks.items()
        for metric, passed in checks.items()
        if not passed
    ]
    if failed_nonperturbation:
        raise RuntimeError(
            "head-only Phase-C nonperturbation control failed: "
            + ", ".join(failed_nonperturbation[:20])
        )
    csv_path = root / "validation" / "paired_rows.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def mean_for(arm: str, load: str, metric: str) -> float | None:
        values = [
            float(row[metric])
            for row in rows
            if row["arm"] == arm
            and row["load"] == load
            and row[metric] is not None
        ]
        return statistics.fmean(values) if values else None

    aggregate = {}
    for arm in ARM_KEYS:
        aggregate[arm] = {
            load: {
                metric: mean_for(arm, load, metric)
                for metric in REPORT_METRICS
            }
            for load in LOADS
        }
    comparisons = {}
    for name, baseline, candidate in COMPARISONS:
        comparisons[name] = {}
        for load in LOADS:
            cell = {}
            for metric in REPORT_METRICS:
                left = mean_for(candidate, load, metric)
                right = mean_for(baseline, load, metric)
                cell[metric] = (
                    None if left is None or right is None else left - right
                )
            comparisons[name][load] = cell
    summary = {
        "schema_version": "phase_c_long_risk_head_repair_summary_v1",
        "bundle_sha256": sha256_file(_bundle_path(root)),
        "loads": list(LOADS),
        "seeds": list(ONLINE_TEST_SEEDS),
        "arms": dict(ARM_LABELS),
        "aggregate_mean": aggregate,
        "paired_differences_candidate_minus_baseline": comparisons,
        "source_vs_repaired_phasec_nonperturbation": {
            "passed": True,
            "compared_metrics": list(nonperturbation_metrics),
            "checks": nonperturbation_checks,
        },
        "artifact_audit": {
            "passed": True,
            "rows": len(rows),
            "expected_rows": len(ARM_KEYS) * len(LOADS) * len(ONLINE_TEST_SEEDS),
        },
    }
    _write_json_exact(root / "validation" / "summary.json", summary)
    print(f"[complete] summary: {root / 'validation' / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "arm", "summary"), required=True)
    parser.add_argument("--output-root", default=str(EVALUATION_ROOT))
    parser.add_argument("--frozen-bundle", default=None)
    parser.add_argument("--arm", choices=ARM_KEYS)
    parser.add_argument("--load", choices=LOADS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int, default=TICKS)
    args = parser.parse_args()
    root = Path(args.output_root)
    if args.mode == "freeze":
        _freeze(root)
    elif args.mode == "arm":
        if args.arm is None or args.load is None or args.seed is None:
            parser.error("arm mode requires --arm, --load, and --seed")
        _run_arm(args)
    else:
        _summary(root)


if __name__ == "__main__":
    main()
