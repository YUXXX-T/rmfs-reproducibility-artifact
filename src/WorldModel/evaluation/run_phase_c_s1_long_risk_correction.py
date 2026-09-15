"""Run and summarize the corrected LongRiskHead S1 comparison campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping

from Policies.TaskAssigner import (
    GreedyTaskAssigner,
    HungarianTaskAssigner,
    WorldModelTaskAssigner,
)
from WorldModel.core.long_risk_schema import (
    LONG_RISK_SCHEMA_VERSION,
    long_risk_runtime_contract,
)
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
from WorldModel.evaluation.phase_c_s1_long_risk_correction_protocol import (
    ARM_KEYS,
    ARM_LABELS,
    BUNDLE_SCHEMA_VERSION,
    CANDIDATE_CHECKPOINT,
    COMPARISONS,
    LOAD_CONFIGS,
    LOADS,
    OUTPUT_ROOT,
    PER_SEED_SCHEMA_VERSION,
    PHASEC_CONFIG,
    REPORT_METRICS,
    SEEDS,
    S1_SIGNAL_CONFIGS,
    SUMMARY_SCHEMA_VERSION,
    TICKS,
    TOP_M,
    canonical_sha256,
    formal_protocol,
    sha256_file,
)


BUNDLE_FILENAME = "phase_c_s1_long_risk_correction_frozen_protocol.json"
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260818

REQUIRED_ARTIFACTS = {
    "candidate_checkpoint": CANDIDATE_CHECKPOINT,
    "config_low": LOAD_CONFIGS["low"],
    "config_mid": LOAD_CONFIGS["mid"],
    "config_high": LOAD_CONFIGS["high"],
    "long_risk_schema": "WorldModel/core/long_risk_schema.py",
    "world_model_core": "WorldModel/core/model.py",
    "world_model_assigner": (
        "Policies/TaskAssigner/WorldModelTaskAssigner/"
        "world_model_task_assigner.py"
    ),
    "online_evaluator": "WorldModel/evaluation/evaluate_online_v6.py",
    "base_s1_protocol": (
        "WorldModel/evaluation/phase_c_s1_hungarian_protocol.py"
    ),
    "correction_protocol": (
        "WorldModel/evaluation/"
        "phase_c_s1_long_risk_correction_protocol.py"
    ),
    "correction_runner": (
        "WorldModel/evaluation/run_phase_c_s1_long_risk_correction.py"
    ),
    "submission_script": (
        "WorldModel/evaluation/"
        "run_phase_c_s1_long_risk_correction_60cpu.slurm"
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_text_exact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite changed file: {path}")
        print(f"[resume] unchanged {path}")
        return
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json_exact(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_exact(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def _bundle_path(root: Path) -> Path:
    return root / BUNDLE_FILENAME


def _freeze_bundle(root: Path) -> dict[str, Any]:
    protocol = formal_protocol()
    artifacts: dict[str, dict[str, str]] = {}
    for name, raw_path in REQUIRED_ARTIFACTS.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts[name] = {
            "path": path.as_posix(),
            "sha256": sha256_file(path),
        }
    payload = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "artifacts": artifacts,
    }
    _write_json_exact(_bundle_path(root), payload)
    return payload


def _verify_bundle(path: Path) -> dict[str, Any]:
    bundle = _read_json(path)
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"wrong bundle schema: {path}")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("bundle lacks protocol")
    expected_protocol = str(bundle.get("protocol_sha256", ""))
    if canonical_sha256(protocol) != expected_protocol:
        raise ValueError("bundle protocol hash mismatch")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("bundle lacks artifacts")
    for name, expected_path in REQUIRED_ARTIFACTS.items():
        artifact = artifacts.get(name)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"bundle lacks artifact {name}")
        path = Path(str(artifact.get("path", "")))
        if path.as_posix() != Path(expected_path).as_posix():
            raise ValueError(f"artifact path changed for {name}: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != str(artifact.get("sha256", "")):
            raise ValueError(f"frozen artifact changed: {path}")
    return bundle


def _paths(root: Path, load: str, seed: int, arm: str) -> tuple[Path, Path]:
    manifest = root / "order_manifests" / f"orders_{load}_seed{seed}.json"
    output = root / "per_arm" / arm / f"{load}_seed{seed}.json"
    return manifest, output


def _make_assigner(arm: str):
    if arm == "greedy":
        return GreedyTaskAssigner()
    if arm == "hungarian":
        return HungarianTaskAssigner()
    if arm == "phasec":
        return WorldModelTaskAssigner(
            checkpoint_path=CANDIDATE_CHECKPOINT,
            top_m=TOP_M,
            **PHASEC_CONFIG,
        )
    signal_by_arm = {
        "s1_event": "event_logit",
        "s1_terminal": "terminal",
        "s1_combo": "combo",
    }
    signal = signal_by_arm.get(arm)
    if signal is not None:
        return WorldModelTaskAssigner(
            checkpoint_path=CANDIDATE_CHECKPOINT,
            top_m=TOP_M,
            energy_conv_random_flip_seed=0,
            **S1_SIGNAL_CONFIGS[signal],
        )
    raise ValueError(arm)


def _audit_metrics(
    arm: str,
    metrics: Mapping[str, Any],
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "manifest_file_exists": manifest_path.is_file(),
        "manifest_content_hash_matches": (
            metrics.get("order_arrival_manifest_sha256")
            == manifest.get("manifest_sha256")
        ),
        "manifest_count_matches": (
            int(metrics.get("order_arrival_count", -1))
            == int(manifest.get("total_orders", -2))
        ),
    }
    if arm == "greedy":
        checks["manifest_is_source"] = not bool(
            metrics.get("order_arrival_replayed")
        )
    else:
        checks["manifest_was_replayed"] = bool(
            metrics.get("order_arrival_replayed")
        )

    model_arm = arm in ("phasec", "s1_event", "s1_terminal", "s1_combo")
    if model_arm:
        checks.update({
            "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
            "no_greedy_fallback": int(
                metrics.get("fallback_greedy_calls", 0)
            ) == 0,
            "native_no_assign_disabled": not bool(
                metrics.get("native_no_assign_enabled", False)
            ),
            "long_risk_schema_matches": (
                metrics.get("long_risk_schema_version")
                == LONG_RISK_SCHEMA_VERSION
            ),
            "long_risk_contract_matches": (
                metrics.get("long_risk_runtime_contract")
                == long_risk_runtime_contract()
            ),
        })
    else:
        checks["external_baseline_has_no_model_calls"] = int(
            metrics.get("model_assign_calls", 0)
        ) == 0

    if arm == "phasec":
        checks.update({
            "phasec_conversion_disabled": int(
                metrics.get("energy_conv_contexts", 0)
            ) == 0,
            "phasec_energy_mode_off": (
                metrics.get("energy_scoring_mode") == "off"
            ),
        })
    signal_by_arm = {
        "s1_event": "event_logit",
        "s1_terminal": "terminal",
        "s1_combo": "combo",
    }
    signal = signal_by_arm.get(arm)
    if signal is not None:
        checks.update({
            "s1_contexts_observed": int(
                metrics.get("energy_conv_contexts", 0)
            ) > 0,
            "s1_conversion_mode": (
                metrics.get("energy_scoring_mode") == "conversion"
            ),
            "s1_signal_matches_arm": (
                metrics.get("energy_drift_signal") == signal
            ),
        })
    return {"passed": all(checks.values()), "checks": checks}


def _resume_ok(
    path: Path,
    *,
    arm: str,
    load: str,
    seed: int,
    ticks: int,
    bundle_sha256: str,
    protocol_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    payload = _read_json(path)
    meta = payload.get("meta") or {}
    checks = {
        "schema": payload.get("schema_version") == PER_SEED_SCHEMA_VERSION,
        "arm": meta.get("arm_key") == arm,
        "label": meta.get("arm") == ARM_LABELS[arm],
        "load": meta.get("load") == load,
        "seed": int(meta.get("seed", -1)) == seed,
        "ticks": int(meta.get("ticks", -1)) == ticks,
        "bundle": meta.get("frozen_bundle_sha256") == bundle_sha256,
        "protocol": meta.get("protocol_sha256") == protocol_sha256,
        "audit": bool((payload.get("audit") or {}).get("passed")),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"cannot resume incompatible {path}: {failed}")
    print(f"[resume] arm={arm} load={load} seed={seed}: {path}")
    return True


def _run_arm(args: argparse.Namespace) -> None:
    if args.arm not in ARM_KEYS:
        raise ValueError(args.arm)
    if args.load not in LOADS:
        raise ValueError(args.load)
    if int(args.ticks) <= 0:
        raise ValueError("ticks must be positive")
    if not args.development:
        if int(args.seed) not in SEEDS:
            raise ValueError(f"formal seed must be one of {list(SEEDS)}")
        if int(args.ticks) != TICKS:
            raise ValueError(f"formal ticks must equal {TICKS}")

    root = Path(args.output_root)
    bundle_path = Path(args.frozen_bundle or _bundle_path(root))
    bundle = _verify_bundle(bundle_path)
    protocol_sha256 = str(bundle["protocol_sha256"])
    bundle_sha256 = sha256_file(bundle_path)
    manifest_path, output_path = _paths(
        root, args.load, int(args.seed), args.arm
    )
    if _resume_ok(
        output_path,
        arm=args.arm,
        load=args.load,
        seed=int(args.seed),
        ticks=int(args.ticks),
        bundle_sha256=bundle_sha256,
        protocol_sha256=protocol_sha256,
    ):
        return
    if args.arm != "greedy" and not manifest_path.is_file():
        raise FileNotFoundError(
            f"Greedy manifest must finish first: {manifest_path}"
        )

    assigner = _make_assigner(args.arm)
    run_kwargs: dict[str, Any] = {}
    if args.arm == "greedy":
        run_kwargs["save_order_manifest"] = str(manifest_path)
    else:
        run_kwargs["recorded_orders_path"] = str(manifest_path)
    print(
        f"[run] arm={args.arm} load={args.load} seed={args.seed} "
        f"ticks={args.ticks}"
    )
    metrics = _run_one_assigner(
        LOAD_CONFIGS[args.load],
        assigner,
        int(args.seed),
        int(args.ticks),
        trace_label=ARM_LABELS[args.arm],
        **run_kwargs,
    )
    manifest = _read_json(manifest_path)
    audit = _audit_metrics(args.arm, metrics, manifest_path, manifest)
    if not audit["passed"]:
        failed = [
            name for name, passed in audit["checks"].items() if not passed
        ]
        raise RuntimeError(
            f"audit failed for {args.arm} {args.load} seed={args.seed}: "
            + ", ".join(failed)
        )
    signal_by_arm = {
        "s1_event": "event_logit",
        "s1_terminal": "terminal",
        "s1_combo": "combo",
    }
    payload = {
        "schema_version": PER_SEED_SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": protocol_sha256,
            "frozen_bundle": bundle_path.as_posix(),
            "frozen_bundle_sha256": bundle_sha256,
            "formal": not args.development,
            "arm_key": args.arm,
            "arm": ARM_LABELS[args.arm],
            "load": args.load,
            "seed": int(args.seed),
            "ticks": int(args.ticks),
            "top_m": TOP_M,
            "config": LOAD_CONFIGS[args.load],
            "checkpoint": CANDIDATE_CHECKPOINT if args.arm not in (
                "greedy", "hungarian"
            ) else None,
            "energy_drift_signal": signal_by_arm.get(args.arm),
            "long_risk_runtime_contract": (
                long_risk_runtime_contract()
                if args.arm not in ("greedy", "hungarian") else None
            ),
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "audit": audit,
        "metrics": metrics,
    }
    _write_json_exact(output_path, payload)
    print(f"[done] {output_path}")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else float("nan"),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_ci(values: list[float], seed_offset: int) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    count = len(values)
    means = [
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_REPEATS)
    ]
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return buffer.getvalue()


def _summarize(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    bundle_path = Path(args.frozen_bundle or _bundle_path(root))
    bundle = _verify_bundle(bundle_path)
    protocol_sha256 = str(bundle["protocol_sha256"])
    bundle_sha256 = sha256_file(bundle_path)

    rows: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    input_files: list[Path] = [bundle_path]
    audit_checks: dict[str, bool] = {}
    for load in LOADS:
        for seed in SEEDS:
            manifest_path, _ = _paths(root, load, seed, "greedy")
            manifest = _read_json(manifest_path)
            input_files.append(manifest_path)
            arm_metrics: dict[str, Mapping[str, Any]] = {}
            for arm in ARM_KEYS:
                _, output_path = _paths(root, load, seed, arm)
                payload = _read_json(output_path)
                input_files.append(output_path)
                meta = payload.get("meta") or {}
                audit = payload.get("audit") or {}
                metrics = payload.get("metrics") or {}
                key = f"{arm}_{load}_{seed}"
                audit_checks[f"{key}_identity"] = (
                    payload.get("schema_version") == PER_SEED_SCHEMA_VERSION
                    and meta.get("protocol_sha256") == protocol_sha256
                    and meta.get("frozen_bundle_sha256") == bundle_sha256
                    and meta.get("arm_key") == arm
                    and meta.get("load") == load
                    and int(meta.get("seed", -1)) == seed
                    and int(meta.get("ticks", -1)) == TICKS
                )
                audit_checks[f"{key}_audit"] = bool(audit.get("passed"))
                audit_checks[f"{key}_manifest"] = (
                    metrics.get("order_arrival_manifest_sha256")
                    == manifest.get("manifest_sha256")
                    and int(metrics.get("order_arrival_count", -1))
                    == int(manifest.get("total_orders", -2))
                )
                arm_metrics[arm] = metrics
            rows[(load, seed)] = arm_metrics
    if not all(audit_checks.values()):
        failed = [name for name, passed in audit_checks.items() if not passed]
        raise RuntimeError("summary audit failed: " + ", ".join(failed[:20]))

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_json: dict[str, Any] = {}
    for arm in ARM_KEYS:
        aggregate_json[arm] = {}
        for load in LOADS:
            aggregate_json[arm][load] = {}
            for metric in REPORT_METRICS:
                values = [
                    value
                    for seed in SEEDS
                    if (
                        value := _number(rows[(load, seed)][arm].get(metric))
                    ) is not None
                ]
                summary = _mean_std(values)
                aggregate_json[arm][load][metric] = summary
                aggregate_rows.append({
                    "arm_key": arm,
                    "arm": ARM_LABELS[arm],
                    "load": load,
                    "metric": metric,
                    **summary,
                })

    contrast_rows: list[dict[str, Any]] = []
    contrast_json: dict[str, Any] = {}
    for comparison_index, (name, baseline, candidate) in enumerate(COMPARISONS):
        comparison_report: dict[str, Any] = {
            "baseline": baseline,
            "candidate": candidate,
            "definition": "candidate_minus_baseline",
            "metrics": {},
        }
        for metric_index, metric in enumerate(REPORT_METRICS):
            metric_report: dict[str, Any] = {}
            per_seed_cluster: list[float] = []
            for seed in SEEDS:
                seed_deltas = []
                for load in LOADS:
                    left = _number(rows[(load, seed)][baseline].get(metric))
                    right = _number(rows[(load, seed)][candidate].get(metric))
                    if left is not None and right is not None:
                        seed_deltas.append(right - left)
                if len(seed_deltas) == len(LOADS):
                    per_seed_cluster.append(
                        sum(seed_deltas) / len(seed_deltas)
                    )
            for load_index, load in enumerate(LOADS):
                values = []
                for seed in SEEDS:
                    left = _number(rows[(load, seed)][baseline].get(metric))
                    right = _number(rows[(load, seed)][candidate].get(metric))
                    if left is not None and right is not None:
                        values.append(right - left)
                summary = _mean_std(values)
                summary["ci95_low"], summary["ci95_high"] = _bootstrap_ci(
                    values,
                    comparison_index * 100000
                    + metric_index * 100
                    + load_index,
                )
                metric_report[load] = summary
                contrast_rows.append({
                    "comparison": name,
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    "scope": load,
                    **summary,
                })
            overall = _mean_std(per_seed_cluster)
            overall["ci95_low"], overall["ci95_high"] = _bootstrap_ci(
                per_seed_cluster,
                comparison_index * 100000 + metric_index * 100 + 90,
            )
            metric_report["overall_seed_cluster"] = overall
            contrast_rows.append({
                "comparison": name,
                "baseline": baseline,
                "candidate": candidate,
                "metric": metric,
                "scope": "overall_seed_cluster",
                **overall,
            })
            comparison_report["metrics"][metric] = metric_report
        contrast_json[name] = comparison_report

    mechanism = {}
    for arm in ("s1_event", "s1_terminal", "s1_combo"):
        mechanism[arm] = {
            "contexts": sum(
                int(rows[(load, seed)][arm].get("energy_conv_contexts", 0))
                for load in LOADS for seed in SEEDS
            ),
            "active_contexts": sum(
                int(
                    rows[(load, seed)][arm].get(
                        "energy_conv_active_contexts", 0
                    )
                )
                for load in LOADS for seed in SEEDS
            ),
            "modified_decisions": sum(
                int(
                    rows[(load, seed)][arm].get(
                        "energy_conv_modified_decisions", 0
                    )
                )
                for load in LOADS for seed in SEEDS
            ),
        }

    validation_dir = root / "validation"
    aggregate_csv = validation_dir / "aggregate_summary.csv"
    contrast_csv = validation_dir / "paired_contrasts.csv"
    summary_json = validation_dir / "correction_summary.json"
    _write_text_exact(
        aggregate_csv,
        _csv_text(
            aggregate_rows,
            ["arm_key", "arm", "load", "metric", "n", "mean", "std"],
        ),
    )
    _write_text_exact(
        contrast_csv,
        _csv_text(
            contrast_rows,
            [
                "comparison", "baseline", "candidate", "metric", "scope",
                "n", "mean", "std", "ci95_low", "ci95_high",
            ],
        ),
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "protocol_sha256": protocol_sha256,
        "frozen_bundle_sha256": bundle_sha256,
        "artifact_audit": {
            "passed": True,
            "checks": len(audit_checks),
            "manifests": len(LOADS) * len(SEEDS),
            "per_arm_outputs": len(LOADS) * len(SEEDS) * len(ARM_KEYS),
        },
        "long_risk_runtime_contract": long_risk_runtime_contract(),
        "mechanism": mechanism,
        "aggregate": aggregate_json,
        "comparisons": contrast_json,
    }
    _write_json_exact(summary_json, summary)

    validated_files = sorted(
        input_files + [aggregate_csv, contrast_csv, summary_json],
        key=lambda path: path.as_posix(),
    )
    _write_text_exact(
        validation_dir / "validated_outputs.sha256",
        "".join(
            f"{sha256_file(path)}  {path.as_posix()}\n"
            for path in validated_files
        ),
    )
    print(f"[complete] summary={summary_json}")
    print(f"[complete] aggregate={aggregate_csv}")
    print(f"[complete] contrasts={contrast_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "arm", "summary"), required=True)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--frozen-bundle", default=None)
    parser.add_argument("--arm", choices=ARM_KEYS)
    parser.add_argument("--load", choices=LOADS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    if args.mode == "freeze":
        bundle = _freeze_bundle(root)
        print(f"[complete] bundle={_bundle_path(root)}")
        print(f"[complete] protocol_sha256={bundle['protocol_sha256']}")
        return
    if args.mode == "arm":
        if args.arm is None or args.load is None or args.seed is None:
            parser.error("--mode arm requires --arm, --load, and --seed")
        _run_arm(args)
        return
    _summarize(args)


if __name__ == "__main__":
    main()
