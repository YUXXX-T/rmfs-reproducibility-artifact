"""Strict audit for the expanded 571--590 context-J behavior labels.

The default mode is a data/label audit.  It checks every run's frozen
generation contract and computes the actual multi-context context targets and
non-tie pair signal used by train_context_dispatch_j_head.py.  An optional
--model-validation flag additionally runs the frozen H=10 decoder validator;
that audit is opt-in because it is substantially more expensive than checking
the labels themselves.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from WorldModel.core.costs import DEFAULT_LAMBDAS, compute_realized_cost
from WorldModel.data.dataset import WorldModelDataset
from WorldModel.evaluation.context_j_behavior_h10_protocol import (
    FROZEN_BUNDLE_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    canonical_sha256,
    sha256_file,
)
from WorldModel.evaluation import validate_phase_c_behavior_h10 as frozen_h10_validator


TARGET_GAMMA = 0.95
TARGET_LOCAL_WEIGHTS = (1.0, 1.0)
DEFAULT_PAIR_EPSILON = 0.01
EXPECTED_LOADS = ("low", "mid", "high")
EXPECTED_SEEDS = tuple(range(571, 591))


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _digest(path: Path) -> str:
    return sha256_file(path)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _station_ids(config_path: Path) -> tuple[int, ...]:
    payload = _json(config_path)
    stations = payload.get("map", {}).get("stations", [])
    ids = tuple(int(row["id"]) for row in stations)
    if not ids:
        raise ValueError(f"no stations in config: {config_path}")
    return ids


def _fixed_context(sample: Mapping[str, Any]) -> tuple[int, int, int]:
    context = sample.get("fixed_context")
    if not isinstance(context, Mapping):
        raise ValueError("fixed_context is not a mapping")
    return (
        int(context["order_id"]),
        int(context["pod_id"]),
        int(context["station_id"]),
    )


def _frame_key(sample: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(sample.get("run_id", "")),
        int(sample["simulation_seed"]),
        int(sample["decision_tick"]),
    )


def _context_group_key(sample: Mapping[str, Any]) -> tuple[Any, ...]:
    return _frame_key(sample) + (
        str(sample["candidate_group_id"]),
    ) + _fixed_context(sample)


def _target_cost(
    sample: Mapping[str, Any],
    *,
    station_ids: Sequence[int],
    horizon: int,
    pair_epsilon: float,
) -> float:
    labels = torch.as_tensor(sample["future_system_labels"], dtype=torch.float32)
    mask = torch.as_tensor(sample["future_mask"], dtype=torch.float32).flatten()
    if labels.ndim != 2 or labels.size(1) != len(DEFAULT_LAMBDAS):
        raise ValueError("future_system_labels must have shape (H, 7)")
    if labels.size(0) < horizon or mask.numel() < horizon:
        raise ValueError("stored label horizon is shorter than protocol H=10")
    if not bool((mask[:horizon] > 0.5).all()):
        raise ValueError("right-censored H=10 label")
    labels = labels[:horizon]
    if not bool(torch.isfinite(labels).all()):
        raise ValueError("future system labels contain NaN/Inf")
    global_cost = float(
        compute_realized_cost(labels, lambdas=list(DEFAULT_LAMBDAS))
    )
    station_labels = torch.as_tensor(
        sample["future_station_labels"], dtype=torch.float32
    )
    if (
        station_labels.ndim != 3
        or station_labels.size(0) < horizon
        or station_labels.size(1) != len(station_ids)
        or station_labels.size(2) != len(TARGET_LOCAL_WEIGHTS)
    ):
        raise ValueError(
            "future_station_labels has an incompatible station/channel shape"
        )
    station_labels = station_labels[:horizon]
    if not bool(torch.isfinite(station_labels).all()):
        raise ValueError("future station labels contain NaN/Inf")
    station_id = _fixed_context(sample)[2]
    if station_id not in station_ids:
        raise ValueError(f"unknown station id {station_id}")
    station_offset = tuple(station_ids).index(station_id)
    local_cost = 0.0
    for step in range(horizon):
        station_value = station_labels[step, station_offset, :]
        local_cost += (TARGET_GAMMA ** step) * float(
            station_value[0].item() * TARGET_LOCAL_WEIGHTS[0]
            + station_value[1].item() * TARGET_LOCAL_WEIGHTS[1]
        )
    value = global_cost + local_cost
    if not math.isfinite(value):
        raise ValueError("non-finite target cost")
    return value


def _strict_generation_checks(
    run_dir: Path,
    *,
    load: str,
    seed: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    expected = protocol["collection"]
    marker_path = run_dir / "run_complete.json"
    data_path = run_dir / "behavior_h10_data.pt"
    gen_path = run_dir / "gen_config.json"
    meta_path = run_dir / "behavior_h10_data_meta.json"
    checks: dict[str, bool] = {
        "run_directory_present": run_dir.is_dir(),
        "marker_present": marker_path.is_file(),
        "data_present": data_path.is_file(),
        "generation_config_present": gen_path.is_file(),
        "metadata_present": meta_path.is_file(),
    }
    marker: dict[str, Any] = {}
    generation: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    if marker_path.is_file():
        marker = _json(marker_path)
        checks.update({
            "marker_protocol_matches": marker.get("protocol_sha256")
            == protocol.get("_protocol_sha256"),
            "marker_load_matches": str(marker.get("load")) == load,
            "marker_seed_matches": _as_int(marker.get("seed")) == seed,
            "marker_data_hash_matches": (
                data_path.is_file()
                and marker.get("data_sha256") == _digest(data_path)
            ),
            "marker_generation_hash_matches": (
                gen_path.is_file()
                and marker.get("generation_config_sha256") == _digest(gen_path)
            ),
            "marker_metadata_hash_matches": (
                meta_path.is_file()
                and marker.get("metadata_sha256") == _digest(meta_path)
            ),
        })
    if gen_path.is_file():
        generation = _json(gen_path)
        checks.update({
            "seed_matches": _as_int(generation.get("seed")) == seed,
            "load_matches": str(generation.get("load_level")) == load,
            "ticks_match": _as_int(generation.get("ticks")) == int(expected["ticks"]),
            "horizon_match": _as_int(generation.get("horizon"))
            == int(expected["horizon"]),
            "sample_interval_match": _as_int(generation.get("sample_interval"))
            == int(expected["sample_interval"]),
            "reservation_window_match": _as_int(
                generation.get("reservation_window")
            ) == int(expected["reservation_window"]),
            "top_m_match": _as_int(generation.get("top_m"))
            == int(expected["top_m"]),
            "min_group_match": _as_int(generation.get("min_group"))
            == int(expected["min_group"]),
            "delay_scale_match": _as_float(generation.get("delay_scale"))
            == float(expected["delay_scale"]),
            "assignment_mode_match": str(generation.get("assignment_mode"))
            == str(expected["assignment_mode"]),
            "behavior_continuation": generation.get("rollout_continuation_mode")
            == "behavior",
            "behavior_policy_match": str(generation.get("task_assigner"))
            == str(expected["behavior_policy"]),
            "candidate_mode_match": str(generation.get("candidate_robot_mode"))
            == str(expected["candidate_robot_mode"]),
            "no_assign_absent": not bool(
                generation.get("include_no_assign_candidate")
            ),
            "max_groups_unlimited": generation.get("max_groups_per_tick") is None,
        })
    if meta_path.is_file():
        metadata = _json(meta_path)
        params = metadata.get("generation_params") or {}
        checks.update({
            "metadata_seed_matches": _as_int(params.get("seed")) == seed,
            "metadata_load_matches": str(params.get("load_level")) == load,
            "metadata_behavior_continuation": params.get(
                "rollout_continuation_mode"
            ) == "behavior",
            "metadata_no_assign_absent": not bool(
                params.get("include_no_assign_candidate")
            ),
            "metadata_max_groups_unlimited": params.get(
                "max_groups_per_tick"
            ) is None,
        })
    checks["passed"] = all(checks.values())
    return {
        "load": load,
        "seed": seed,
        "run": run_dir.name,
        "paths": {
            "data": data_path.as_posix(),
            "generation_config": gen_path.as_posix(),
            "metadata": meta_path.as_posix(),
            "marker": marker_path.as_posix(),
        },
        "generation_config": generation,
        "metadata": metadata,
        "data_quality": metadata.get("data_quality", {}),
        "checks": checks,
        "passed": checks["passed"],
    }


def _signal_for_run(
    data_path: Path,
    *,
    load: str,
    seed: int,
    station_ids: Sequence[int],
    horizon: int,
    pair_epsilon: float,
) -> dict[str, Any]:
    dataset = WorldModelDataset.from_file(str(data_path))
    contexts: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    frames: dict[tuple[str, int, int], set[tuple[int, int, int]]] = defaultdict(set)
    action_rows = 0
    no_assign_rows = 0
    invalid_rows: list[str] = []
    for index, sample in enumerate(dataset.samples):
        action_rows += 1
        if str(sample.get("rollout_continuation_mode")) != "behavior":
            invalid_rows.append(f"sample[{index}].continuation")
        if bool(sample.get("no_assign_applied")) or str(
            sample.get("action_type", "")
        ) == "no_assign":
            no_assign_rows += 1
            invalid_rows.append(f"sample[{index}].no_assign")
        try:
            frame = _frame_key(sample)
            context = _fixed_context(sample)
            group = _context_group_key(sample)
            target = _target_cost(
                sample,
                station_ids=station_ids,
                horizon=horizon,
                pair_epsilon=pair_epsilon,
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            invalid_rows.append(f"sample[{index}]: {exc}")
            continue
        contexts[group].append(target)
        frames[frame].add(context)
    for group, values in contexts.items():
        if len(values) < 2:
            invalid_rows.append(
                f"candidate_group_below_min_group:{group}:size={len(values)}"
            )
    if action_rows == 0:
        invalid_rows.append("empty_dataset")
    if not contexts:
        invalid_rows.append("no_candidate_context_groups")
    multi_context_frames = {
        frame: context_set for frame, context_set in frames.items() if len(context_set) >= 2
    }
    context_costs_by_frame: dict[
        tuple[str, int, int], dict[tuple[int, int, int], float]
    ] = defaultdict(dict)
    for group, values in contexts.items():
        frame = group[:3]
        context = tuple(int(value) for value in group[4:7])
        if context in context_costs_by_frame[frame]:
            invalid_rows.append(f"duplicate_context:{frame}:{context}")
        context_costs_by_frame[frame][context] = min(values)
    raw_pairs = 0
    non_tie_pairs = 0
    for frame, costs in context_costs_by_frame.items():
        if len(costs) < 2:
            continue
        for left, right in itertools.combinations(costs.values(), 2):
            raw_pairs += 1
            if abs(left - right) > pair_epsilon:
                non_tie_pairs += 1
    split = (
        "train" if seed <= 582 else
        "val" if seed <= 586 else
        "test"
    )
    # Metadata is read by the caller; this field is kept compact here.
    return {
        "load": load,
        "seed": seed,
        "split": split,
        "samples": action_rows,
        "candidate_context_groups": len(contexts),
        "multi_context_frames": len(multi_context_frames),
        "context_rows": sum(len(values) for values in context_costs_by_frame.values()
                            if len(values) >= 2),
        "raw_context_pairs": raw_pairs,
        "non_tie_context_pairs": non_tie_pairs,
        "no_assign_rows": no_assign_rows,
        "invalid_rows": invalid_rows[:20],
        "invalid_count": len(invalid_rows),
        "dataset_summary": dataset.summary(),
    }


def _aggregate_signal(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "runs": len(rows),
        "samples": sum(int(row["samples"]) for row in rows),
        "candidate_context_groups": sum(
            int(row["candidate_context_groups"]) for row in rows
        ),
        "multi_context_frames": sum(int(row["multi_context_frames"]) for row in rows),
        "context_rows": sum(int(row["context_rows"]) for row in rows),
        "raw_context_pairs": sum(int(row["raw_context_pairs"]) for row in rows),
        "non_tie_context_pairs": sum(int(row["non_tie_context_pairs"]) for row in rows),
        "no_assign_rows": sum(int(row["no_assign_rows"]) for row in rows),
        "invalid_count": sum(int(row["invalid_count"]) for row in rows),
    }
    return result


def run_audit(
    *,
    data_root: Path,
    bundle_path: Path,
    save_json: Path,
    pair_epsilon: float,
    allow_partial: bool,
    run_model_validation: bool,
    model_report_path: Path | None,
    torch_threads: int,
) -> dict[str, Any]:
    bundle = _json(bundle_path)
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"wrong frozen bundle schema: {bundle_path}")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("frozen bundle has no protocol")
    protocol_sha = canonical_sha256(protocol)
    if protocol_sha != bundle.get("protocol_sha256"):
        raise ValueError("frozen protocol hash mismatch")
    protocol = dict(protocol)
    protocol["_protocol_sha256"] = protocol_sha
    loads = tuple(protocol["inputs"]["loads"].keys())
    seeds = tuple(int(value) for value in protocol["inputs"]["seeds"])
    expected_runs = len(loads) * len(seeds)
    if tuple(loads) != EXPECTED_LOADS or tuple(seeds) != EXPECTED_SEEDS:
        raise ValueError("bundle does not contain the frozen 571--590 protocol")
    collection = protocol["collection"]
    if int(collection["horizon"]) != 10 or int(collection["sample_interval"]) != 10:
        raise ValueError("unexpected H/sample interval in context-J protocol")
    if collection.get("max_groups_per_tick") is not None:
        raise ValueError("context-J protocol must collect all dispatchable contexts")
    frozen_h10_validator._verify_inputs(bundle, Path.cwd())

    summary_path = data_root / "collection_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    collection_summary = _json(summary_path)
    if collection_summary.get("protocol_sha256") != protocol_sha:
        raise ValueError("collection summary protocol hash mismatch")

    run_reports: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    started = time.time()
    for load in loads:
        station_ids = _station_ids(Path(str(protocol["inputs"]["loads"][load])))
        for seed in seeds:
            run_dir = data_root / "runs" / f"{load}_seed{seed}"
            data_path = run_dir / "behavior_h10_data.pt"
            report = _strict_generation_checks(
                run_dir, load=load, seed=seed, protocol=protocol
            )
            if not data_path.is_file():
                missing.append({"load": load, "seed": seed, "path": data_path.as_posix()})
                run_reports.append(report)
                continue
            if not report["passed"]:
                failed.append({
                    "load": load,
                    "seed": seed,
                    "failed_checks": [
                        name for name, passed in report["checks"].items() if not passed
                    ],
                })
                run_reports.append(report)
                continue
            signal = _signal_for_run(
                data_path,
                load=load,
                seed=seed,
                station_ids=station_ids,
                horizon=int(collection["horizon"]),
                pair_epsilon=float(pair_epsilon),
            )
            report["signal"] = signal
            if signal["invalid_count"]:
                failed.append({
                    "load": load,
                    "seed": seed,
                    "failed_checks": ["label_signal_integrity"],
                    "invalid_count": signal["invalid_count"],
                })
            signal_rows.append(signal)
            run_reports.append(report)
            print(
                f"[audit] {load}/seed{seed}: samples={signal['samples']} "
                f"frames={signal['multi_context_frames']} "
                f"contexts={signal['context_rows']} "
                f"non_tie_pairs={signal['non_tie_context_pairs']}",
                flush=True,
            )

    by_split = {}
    for split in ("train", "val", "test"):
        rows = [row for row in signal_rows if row["split"] == split]
        by_split[split] = _aggregate_signal(rows)
    gates = {
        "all_expected_runs_present": not missing and len(signal_rows) == expected_runs,
        "all_run_checks_passed": not failed,
        "no_native_no_assign_rows": all(
            int(row["no_assign_rows"]) == 0 for row in signal_rows
        ),
        "train_signal_sufficient": by_split["train"]["non_tie_context_pairs"] >= 150,
        "val_signal_sufficient": by_split["val"]["non_tie_context_pairs"] >= 40,
        "test_signal_sufficient": by_split["test"]["non_tie_context_pairs"] >= 40,
    }
    gates["all_gates_passed"] = all(gates.values())
    signal_audit = {
        split: {
            "non_tie_context_pairs": int(by_split[split]["non_tie_context_pairs"]),
            "minimum_recommended_pairs": int(limit),
            "sufficient_for_training": int(
                by_split[split]["non_tie_context_pairs"]
            ) >= int(limit),
        }
        for split, limit in {"train": 150, "val": 40, "test": 40}.items()
    }
    signal_audit["all_splits_sufficient"] = all(
        row["sufficient_for_training"]
        for name, row in signal_audit.items()
        if name != "all_splits_sufficient"
    )
    result: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_sha256": protocol_sha,
        "bundle": bundle_path.as_posix(),
        "data_root": data_root.as_posix(),
        "expected_runs": expected_runs,
        "completed_runs": len(signal_rows),
        "missing_runs": missing,
        "failed_runs": failed,
        "partial": bool(missing or failed),
        "pair_epsilon": float(pair_epsilon),
        "splits": by_split,
        "signal_audit": signal_audit,
        "runs": run_reports,
        "gates": gates,
        "runtime_seconds": time.time() - started,
        "interpretation": {
            "multi_context_frame": (
                "one run/seed/decision tick containing at least two distinct "
                "fixed order-pod-station contexts"
            ),
            "context_row": (
                "one context after collapsing its robot candidates to the "
                "minimum true H=10 behavior-continuation cost"
            ),
            "non_tie_pair": (
                "a pair of distinct contexts in one frame whose target J cost "
                "differs by more than pair_epsilon"
            ),
            "online_connection_allowed": False,
            "next_step": (
                "fit the context-J head with train 571--582, select only on "
                "val 583--586, and hold out test 587--590"
            ),
        },
    }
    if run_model_validation:
        if model_report_path is None:
            raise ValueError("model_report_path is required with --model-validation")
        model_result = frozen_h10_validator.run_evaluation(
            data_root=data_root,
            bundle_path=bundle_path,
            save_json=model_report_path,
            device_name="cpu",
            torch_threads=int(torch_threads),
            max_samples=0,
            allow_partial=allow_partial,
        )
        result["frozen_model_validation"] = {
            "report": model_report_path.as_posix(),
            "completed_runs": model_result.get("completed_runs"),
            "expected_runs": model_result.get("expected_runs"),
            "aggregate": model_result.get("aggregate"),
        }
    save_json.parent.mkdir(parents=True, exist_ok=True)
    save_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not gates["all_gates_passed"] and not allow_partial:
        raise RuntimeError(
            "context-J label audit failed; see gates/failed_runs in "
            f"{save_json}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--pair-epsilon", type=float, default=DEFAULT_PAIR_EPSILON)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--model-validation", action="store_true")
    parser.add_argument("--model-report", type=Path, default=None)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    if args.pair_epsilon < 0:
        raise SystemExit("--pair-epsilon must be non-negative")
    result = run_audit(
        data_root=args.data_root,
        bundle_path=args.bundle,
        save_json=args.save_json,
        pair_epsilon=args.pair_epsilon,
        allow_partial=args.allow_partial,
        run_model_validation=args.model_validation,
        model_report_path=args.model_report,
        torch_threads=args.torch_threads,
    )
    print("=" * 88)
    print("Context-J behavior H=10 label audit")
    print("=" * 88)
    print("runs =", result["completed_runs"], "/", result["expected_runs"])
    for split, row in result["splits"].items():
        print(
            split,
            "frames=", row["multi_context_frames"],
            "contexts=", row["context_rows"],
            "non_tie_pairs=", row["non_tie_context_pairs"],
        )
    print("all_gates_passed =", result["gates"]["all_gates_passed"])
    print("JSON saved:", args.save_json)


if __name__ == "__main__":
    main()
