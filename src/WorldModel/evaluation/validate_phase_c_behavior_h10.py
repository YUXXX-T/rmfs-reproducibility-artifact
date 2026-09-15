"""Validate frozen H=10 predictions under behavior continuation.

The data files contain candidate-conditioned labels collected with
``rollout_continuation_mode=behavior``.  For every stored candidate sample,
this evaluator calls ``encode_state`` afresh and then performs the frozen
H=10 model rollout.  Thus the re-encoding boundary is the real decision
sample/tick, while the ten-step model transition remains unchanged.

This module is intentionally separate from the historical isolated-horizon
validator; no old report or protocol is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from WorldModel.data.dataset import WorldModelDataset
from WorldModel.evaluation.evaluate import _load_model
from WorldModel.evaluation.phase_c_behavior_h10_protocol import (
    FROZEN_BUNDLE_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    canonical_sha256,
    sha256_file,
)


SYSTEM_CHANNELS = (
    "total_wait_time",
    "average_excess_delay",
    "station_queue_delta",
    "station_load_imbalance",
    "bottleneck_CVaR",
    "completed_orders_delta",
    "deadlock_or_severe_congestion_risk",
)
NODE_CHANNELS = (
    "self_occupancy",
    "local_density",
    "local_wait_pressure",
    "local_blocked_pressure",
    "reservation_pressure",
    "congestion_score",
)
STATION_CHANNELS = ("station_queue", "assigned_load")


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mu = _mean(values)
    return math.sqrt(_mean([(value - mu) ** 2 for value in values]))


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rankdata(values: Sequence[float]) -> list[float]:
    """Average-rank ties (tie-aware Spearman without scipy)."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        value = values[order[start]]
        while end < len(order) and values[order[end]] == value:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 3 or len(left) != len(right):
        return 0.0
    ml, mr = _mean(left), _mean(right)
    numerator = math.fsum((a - ml) * (b - mr) for a, b in zip(left, right))
    denominator = math.sqrt(
        math.fsum((a - ml) ** 2 for a in left)
        * math.fsum((b - mr) ** 2 for b in right)
    )
    return numerator / denominator if denominator > 1e-12 else 0.0


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 3 or len(left) != len(right):
        return 0.0
    return _pearson(_rankdata(left), _rankdata(right))


def _bundle(predictions: Sequence[float], targets: Sequence[float]) -> dict:
    pairs = [
        (float(prediction), float(target))
        for prediction, target in zip(predictions, targets)
        if math.isfinite(float(prediction)) and math.isfinite(float(target))
    ]
    if not pairs:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "pearson": None,
            "spearman": None,
            "pred_mean": None,
            "pred_std": None,
            "target_mean": None,
            "target_std": None,
        }
    pred = [pair[0] for pair in pairs]
    target = [pair[1] for pair in pairs]
    errors = [a - b for a, b in pairs]
    return {
        "n": len(pairs),
        "mae": _mean([abs(value) for value in errors]),
        "rmse": math.sqrt(_mean([value * value for value in errors])),
        "bias": _mean(errors),
        "pearson": _pearson(pred, target),
        "spearman": _spearman(pred, target),
        "pred_mean": _mean(pred),
        "pred_std": _std(pred),
        "target_mean": _mean(target),
        "target_std": _std(target),
    }


def _metric_store(horizon: int) -> dict:
    return {
        "system": {
            str(step): {
                channel: {"pred": [], "target": []}
                for channel in SYSTEM_CHANNELS
            }
            for step in range(1, horizon + 1)
        },
        "completed_cumulative": {
            str(step): {"pred": [], "target": []}
            for step in range(1, horizon + 1)
        },
        "node_density": {
            str(step): {"pred": [], "target": []}
            for step in range(1, horizon + 1)
        },
        "node_congestion": {
            str(step): {"pred": [], "target": []}
            for step in range(1, horizon + 1)
        },
        "station_queue": {
            str(step): {"pred": [], "target": []}
            for step in range(1, horizon + 1)
        },
        "station_load": {
            str(step): {"pred": [], "target": []}
            for step in range(1, horizon + 1)
        },
    }


def _summarize_store(store: Mapping[str, Any]) -> dict:
    system_by_step = {}
    for step, channels in store["system"].items():
        system_by_step[step] = {
            "channels": {
                name: _bundle(values["pred"], values["target"])
                for name, values in channels.items()
            }
        }
    return {
        "system_by_step": system_by_step,
        "completed_orders_cumulative": {
            step: _bundle(values["pred"], values["target"])
            for step, values in store["completed_cumulative"].items()
        },
        "node_density_by_step": {
            step: _bundle(values["pred"], values["target"])
            for step, values in store["node_density"].items()
        },
        "node_congestion_by_step": {
            step: _bundle(values["pred"], values["target"])
            for step, values in store["node_congestion"].items()
        },
        "station_queue_by_step": {
            step: _bundle(values["pred"], values["target"])
            for step, values in store["station_queue"].items()
        },
        "station_load_by_step": {
            step: _bundle(values["pred"], values["target"])
            for step, values in store["station_load"].items()
        },
    }


def _group_key(sample: Mapping[str, Any], load: str, seed: int) -> str:
    return (
        f"{load}|{seed}|{sample.get('run_id', '')}|"
        f"{sample.get('candidate_group_id', '')}"
    )


def _add_sample(
    store: dict,
    model,
    sample: Mapping[str, Any],
    device: torch.device,
    horizon: int,
    delay_scale: float,
) -> tuple[bool, dict[str, Any]]:
    station_ids = sample.get("station_node_ids")
    station_ids = torch.as_tensor(station_ids).tolist() if station_ids is not None else None
    # Re-encode at the sample's current decision state.  Do not move this
    # outside the loop: samples at different decision ticks have different
    # order/backlog/traffic states.
    z, demand_embedding, edge_embedding = model.encode_state(
        sample["node_history"].to(device),
        sample["edge_index"].to(device),
        sample["edge_features"].to(device),
        sample["demand_context"].to(device),
    )
    node_pred, system_pred, station_pred = model.rollout(
        z,
        demand_embedding,
        edge_embedding,
        sample["action_node"].to(device),
        sample["action_global"].to(device),
        sample["edge_index"].to(device),
        station_ids,
        K=horizon,
    )[:3]
    target_system = torch.as_tensor(sample["future_system_labels"])
    target_node = torch.as_tensor(sample["future_node_labels"])
    target_station = torch.as_tensor(sample["future_station_labels"])
    mask = sample.get("future_mask")
    mask = torch.as_tensor(mask) if mask is not None else None
    if target_system.shape[0] < horizon or system_pred.shape[0] < horizon:
        raise ValueError("incomplete H=10 system prediction/target")
    if mask is not None and mask.shape[0] < horizon:
        raise ValueError("incomplete H=10 future mask")

    complete = True
    cumulative_prediction = 0.0
    cumulative_target = 0.0
    endpoint_prediction: list[float] | None = None
    endpoint_target: list[float] | None = None
    endpoint_station_prediction: list[list[float]] | None = None
    endpoint_station_target: list[list[float]] | None = None
    for offset in range(horizon):
        if mask is not None and float(mask[offset].item()) < 0.5:
            complete = False
            continue
        step = str(offset + 1)
        prediction = system_pred[offset].detach().cpu().reshape(-1)
        target = target_system[offset].detach().cpu().reshape(-1)
        for channel, name in enumerate(SYSTEM_CHANNELS):
            prediction_value = float(prediction[channel].item())
            target_value = float(target[channel].item())
            if channel == 1:
                prediction_value *= delay_scale
                target_value *= delay_scale
            store["system"][step][name]["pred"].append(prediction_value)
            store["system"][step][name]["target"].append(target_value)
            if offset == horizon - 1:
                if endpoint_prediction is None:
                    endpoint_prediction = []
                    endpoint_target = []
                endpoint_prediction.append(prediction_value)
                endpoint_target.append(target_value)

        # completed_orders_delta is a per-step event; also report its prefix
        # sum so H=10 completion is not mistaken for a one-tick event.
        cumulative_prediction += float(prediction[5].item())
        cumulative_target += float(target[5].item())
        store["completed_cumulative"][step]["pred"].append(
            cumulative_prediction
        )
        store["completed_cumulative"][step]["target"].append(
            cumulative_target
        )

        pred_node = torch.as_tensor(node_pred[offset]).detach().cpu()
        true_node = target_node[offset].detach().cpu()
        if pred_node.shape == true_node.shape and pred_node.numel():
            for key, channel in (("node_density", 1), ("node_congestion", 5)):
                store[key][step]["pred"].extend(pred_node[:, channel].tolist())
                store[key][step]["target"].extend(true_node[:, channel].tolist())

        pred_station = torch.as_tensor(station_pred[offset]).detach().cpu()
        true_station = target_station[offset].detach().cpu()
        if pred_station.shape == true_station.shape and pred_station.numel():
            store["station_queue"][step]["pred"].extend(
                pred_station[:, 0].tolist()
            )
            store["station_queue"][step]["target"].extend(
                true_station[:, 0].tolist()
            )
            store["station_load"][step]["pred"].extend(
                pred_station[:, 1].tolist()
            )
            store["station_load"][step]["target"].extend(
                true_station[:, 1].tolist()
            )
            if offset == horizon - 1:
                endpoint_station_prediction = pred_station.tolist()
                endpoint_station_target = true_station.tolist()

    fixed = sample.get("fixed_context") or {}
    candidate = sample.get("candidate_info") or {}
    endpoint_record = {
        "candidate_key": str(sample.get("candidate_key") or ""),
        "candidate_group_id": str(sample.get("candidate_group_id") or ""),
        "decision_tick": _as_int(sample.get("decision_tick")),
        "action_type": str(sample.get("action_type") or ""),
        "order_id": fixed.get("order_id"),
        "pod_id": fixed.get("pod_id"),
        "station_id": fixed.get("station_id"),
        "robot_id": candidate.get("robot_id"),
        "rollout_generated_orders": _as_int(
            sample.get("rollout_generated_orders"), 0
        ),
        "rollout_assigned_tasks": _as_int(
            sample.get("rollout_assigned_tasks"), 0
        ),
        "system_prediction": dict(zip(SYSTEM_CHANNELS, endpoint_prediction or [])),
        "system_target": dict(zip(SYSTEM_CHANNELS, endpoint_target or [])),
        "system_error": dict(zip(
            SYSTEM_CHANNELS,
            [
                prediction - target
                for prediction, target in zip(
                    endpoint_prediction or [], endpoint_target or []
                )
            ],
        )),
        "completed_orders_cumulative_prediction": cumulative_prediction,
        "completed_orders_cumulative_target": cumulative_target,
        "station_prediction": endpoint_station_prediction,
        "station_target": endpoint_station_target,
    }
    return complete, endpoint_record


def _load_bundle(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"wrong behavior bundle schema: {path}")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("behavior bundle lacks protocol")
    if canonical_sha256(protocol) != payload.get("protocol_sha256"):
        raise ValueError("behavior protocol hash mismatch")
    return payload


def _verify_inputs(bundle: Mapping[str, Any], repo_root: Path) -> None:
    artifacts = bundle.get("artifacts") or {}
    rows = [artifacts.get("model_checkpoint")]
    rows.extend((artifacts.get("load_configs") or {}).values())
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("malformed frozen input artifact")
        raw = Path(str(row["path"]))
        path = raw if raw.is_absolute() else repo_root / raw
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != row.get("sha256"):
            raise ValueError(f"frozen input changed: {path}")


def _run_files(data_root: Path, loads: Iterable[str], seeds: Iterable[int]):
    for load in loads:
        for seed in seeds:
            run_dir = data_root / "runs" / f"{load}_seed{seed}"
            yield load, int(seed), run_dir, run_dir / "behavior_h10_data.pt"


def _validate_run_metadata(
    run_dir: Path,
    data_path: Path,
    load: str,
    seed: int,
    protocol: Mapping[str, Any],
) -> dict:
    gen_path = run_dir / "gen_config.json"
    checks = {"generation_config_present": gen_path.is_file()}
    generation = {}
    if gen_path.is_file():
        generation = json.loads(gen_path.read_text(encoding="utf-8"))
        expected = protocol["collection"]
        checks.update({
            "seed_matches": _as_int(generation.get("seed"), -1) == seed,
            "load_matches": str(generation.get("load_level")) == load,
            "ticks_match": _as_int(generation.get("ticks"), -1)
            == int(expected["ticks"]),
            "horizon_matches": _as_int(generation.get("horizon"), -1)
            == int(expected["horizon"]),
            "sample_interval_matches": _as_int(
                generation.get("sample_interval"), -1
            ) == int(expected["sample_interval"]),
            "top_m_matches": _as_int(generation.get("top_m"), -1)
            == int(expected["top_m"]),
            "min_group_matches": _as_int(generation.get("min_group"), -1)
            == int(expected["min_group"]),
            "delay_scale_matches": float(generation.get("delay_scale") or 0.0)
            == float(expected["delay_scale"]),
            "behavior_continuation": generation.get(
                "rollout_continuation_mode"
            ) == "behavior",
            "behavior_policy_matches": str(generation.get("task_assigner"))
            == str(expected["behavior_policy"]),
            "candidate_mode_matches": str(
                generation.get("candidate_robot_mode")
            ) == str(expected["candidate_robot_mode"]),
            "no_native_no_assign": not bool(
                generation.get("include_no_assign_candidate")
            ),
        })
    return {
        "data": data_path.as_posix(),
        "data_sha256": sha256_file(data_path),
        "generation_config": generation,
        "checks": checks,
        "passed": all(checks.values()),
    }


def evaluate_run(
    *,
    model,
    data_path: Path,
    load: str,
    seed: int,
    device: torch.device,
    horizon: int,
    delay_scale: float,
    max_samples: int,
) -> dict:
    dataset = WorldModelDataset.from_file(str(data_path))
    samples = list(dataset.samples)
    if max_samples > 0:
        samples = samples[:max_samples]
    store = _metric_store(horizon)
    groups: dict[str, dict[str, int]] = {}
    mode_histogram: Counter[str] = Counter()
    decision_ticks: set[tuple[str, int]] = set()
    complete_count = 0
    endpoint_records: list[dict[str, Any]] = []
    started = time.time()

    model.eval()
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            mode = str(sample.get("rollout_continuation_mode") or "")
            mode_histogram[mode] += 1
            if mode != "behavior":
                raise ValueError(
                    f"{data_path}: sample[{index}] continuation mode={mode!r}"
                )
            if bool(sample.get("no_assign_applied")):
                raise ValueError(f"{data_path}: native NO_ASSIGN sample")
            group = _group_key(sample, load, seed)
            tick = _as_int(sample.get("decision_tick"))
            decision_ticks.add((str(sample.get("run_id") or ""), tick))
            row = groups.setdefault(
                group, {"generated_orders": 0, "assigned_tasks": 0, "decision_tick": tick}
            )
            row["generated_orders"] = max(
                row["generated_orders"],
                _as_int(sample.get("rollout_generated_orders"), 0),
            )
            row["assigned_tasks"] = max(
                row["assigned_tasks"],
                _as_int(sample.get("rollout_assigned_tasks"), 0),
            )
            complete, endpoint_record = _add_sample(
                store, model, sample, device, horizon, delay_scale
            )
            complete_count += int(complete)
            endpoint_records.append(endpoint_record)
            if (index + 1) % 100 == 0:
                print(f"  {load}/seed{seed}: {index + 1}/{len(samples)}")

    summary = _summarize_store(store)
    groups_with_generated = sum(
        row["generated_orders"] > 0 for row in groups.values()
    )
    groups_with_assigned = sum(
        row["assigned_tasks"] > 0 for row in groups.values()
    )
    coverage_checks = {
        "samples_present": len(samples) > 0,
        "candidate_groups_present": len(groups) > 0,
        "all_samples_complete_h10": complete_count == len(samples),
        "all_samples_behavior_mode": set(mode_histogram) == {"behavior"},
        "future_orders_observed": groups_with_generated > 0,
        "continuation_assignments_observed": groups_with_assigned > 0,
        "decision_ticks_reencoded": len(decision_ticks) > 0,
    }
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "load": load,
        "seed": seed,
        "data": data_path.as_posix(),
        "samples": len(samples),
        "candidate_groups": len(groups),
        "decision_ticks_reencoded": len(decision_ticks),
        "complete_horizon_samples": complete_count,
        "mode_histogram": dict(mode_histogram),
        "groups_with_generated_orders": groups_with_generated,
        "groups_with_assigned_tasks": groups_with_assigned,
        "generated_orders_group_total": sum(
            row["generated_orders"] for row in groups.values()
        ),
        "assigned_tasks_group_total": sum(
            row["assigned_tasks"] for row in groups.values()
        ),
        "fresh_encode_per_sample": True,
        "coverage_audit": {
            "checks": coverage_checks,
            "passed": all(coverage_checks.values()),
        },
        "h10_endpoint_records": endpoint_records,
        "metrics": summary,
        "runtime_seconds": time.time() - started,
    }


def _aggregate(run_reports: list[dict], loads: Sequence[str], horizon: int) -> dict:
    aggregate = {
        "coverage": {
            "samples": sum(row["samples"] for row in run_reports),
            "candidate_groups": sum(row["candidate_groups"] for row in run_reports),
            "decision_ticks_reencoded": sum(
                row["decision_ticks_reencoded"] for row in run_reports
            ),
            "complete_horizon_samples": sum(
                row["complete_horizon_samples"] for row in run_reports
            ),
            "groups_with_generated_orders": sum(
                row["groups_with_generated_orders"] for row in run_reports
            ),
            "groups_with_assigned_tasks": sum(
                row["groups_with_assigned_tasks"] for row in run_reports
            ),
            "generated_orders_group_total": sum(
                row["generated_orders_group_total"] for row in run_reports
            ),
            "assigned_tasks_group_total": sum(
                row["assigned_tasks_group_total"] for row in run_reports
            ),
        },
        "runs_by_load": {},
        "system_by_step_run_mean": {},
        "system_h10_by_load": {},
    }
    for load in loads:
        rows = [row for row in run_reports if row["load"] == load]
        aggregate["runs_by_load"][load] = [row["seed"] for row in rows]
        for step in range(1, horizon + 1):
            key = str(step)
            channels = {}
            for name in SYSTEM_CHANNELS:
                metrics = [
                    row["metrics"]["system_by_step"][key]["channels"][name]
                    for row in rows
                ]
                metrics = [metric for metric in metrics if metric["mae"] is not None]
                channels[name] = {
                    "run_count": len(metrics),
                    "mae_mean": _mean([metric["mae"] for metric in metrics]),
                    "rmse_mean": _mean([metric["rmse"] for metric in metrics]),
                    "spearman_mean": _mean(
                        [metric["spearman"] for metric in metrics]
                    ),
                }
            aggregate["system_by_step_run_mean"].setdefault(key, {})[load] = channels
        aggregate["system_h10_by_load"][load] = aggregate[
            "system_by_step_run_mean"
        ][str(horizon)][load]
    return aggregate


def run_evaluation(
    *,
    data_root: Path,
    bundle_path: Path,
    save_json: Path,
    device_name: str,
    torch_threads: int,
    max_samples: int,
    allow_partial: bool,
) -> dict:
    bundle = _load_bundle(bundle_path)
    repo_root = Path.cwd()
    _verify_inputs(bundle, repo_root)
    protocol = bundle["protocol"]
    model_path = Path(str(bundle["artifacts"]["model_checkpoint"]["path"]))
    if not model_path.is_absolute():
        model_path = repo_root / model_path
    torch.set_num_threads(max(int(torch_threads), 1))
    device = _device(device_name)
    model, label_schema = _load_model(str(model_path))
    model = model.to(device)
    model.eval()
    loads = tuple(protocol["inputs"]["loads"].keys())
    seeds = tuple(int(value) for value in protocol["inputs"]["seeds"])
    horizon = int(protocol["collection"]["horizon"])
    delay_scale = float(protocol["collection"]["delay_scale"])
    training_horizon = int(getattr(model, "rollout_horizon", -1))
    step_embedding = getattr(getattr(model, "transition", None), "step_embed", None)
    step_capacity = (
        int(step_embedding.num_embeddings) if step_embedding is not None else None
    )
    if training_horizon != horizon:
        raise RuntimeError(
            f"checkpoint training horizon={training_horizon}, protocol H={horizon}"
        )
    if step_capacity is not None and step_capacity < horizon:
        raise RuntimeError(
            f"step embedding capacity={step_capacity} cannot cover H={horizon}"
        )

    run_reports = []
    missing = []
    started = time.time()
    for load, seed in ((load, seed) for load in loads for seed in seeds):
        run_dir = data_root / "runs" / f"{load}_seed{seed}"
        data_path = run_dir / "behavior_h10_data.pt"
        if not data_path.is_file():
            missing.append({"load": load, "seed": seed, "path": data_path.as_posix()})
            continue
        metadata = _validate_run_metadata(
            run_dir, data_path, load, seed, protocol
        )
        if not metadata["passed"]:
            failed = [
                name for name, passed in metadata["checks"].items() if not passed
            ]
            raise RuntimeError(
                f"run metadata audit failed {load}/seed{seed}: {failed}"
            )
        report = evaluate_run(
            model=model,
            data_path=data_path,
            load=load,
            seed=seed,
            device=device,
            horizon=horizon,
            delay_scale=delay_scale,
            max_samples=max_samples,
        )
        if not report["coverage_audit"]["passed"]:
            failed = [
                name
                for name, passed in report["coverage_audit"]["checks"].items()
                if not passed
            ]
            raise RuntimeError(
                f"behavior coverage audit failed {load}/seed{seed}: {failed}"
            )
        report["metadata_audit"] = metadata
        run_reports.append(report)
        run_json = run_dir / "behavior_h10_eval.json"
        run_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    if missing and not allow_partial:
        raise RuntimeError(
            "missing behavior H=10 runs: "
            + ", ".join(f"{row['load']}/seed{row['seed']}" for row in missing)
        )
    if not run_reports:
        raise RuntimeError("no behavior H=10 runs available")

    result = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_sha256": bundle["protocol_sha256"],
        "bundle": bundle_path.as_posix(),
        "data_root": data_root.as_posix(),
        "checkpoint": model_path.as_posix(),
        "checkpoint_sha256": sha256_file(model_path),
        "checkpoint_label_schema": label_schema,
        "checkpoint_horizon_contract": {
            "training_horizon": training_horizon,
            "step_embedding_capacity": step_capacity,
            "requested_horizon": horizon,
            "passed": True,
        },
        "device": str(device),
        "torch_threads": int(torch_threads),
        "expected_runs": len(loads) * len(seeds),
        "completed_runs": len(run_reports),
        "missing_runs": missing,
        "partial": bool(missing),
        "fresh_encode_per_sample": True,
        "behavior_continuation_required": True,
        "aggregate": _aggregate(run_reports, loads, horizon),
        "runs": run_reports,
        "runtime_seconds": time.time() - started,
        "psi_pre_connection_allowed": False,
        "interpretation": {
            "reencode_boundary": (
                "encode_state is called independently for every decision-point "
                "sample; the frozen H-step candidate-conditioned transition is "
                "not retrained or altered"
            ),
            "completed_orders_delta": (
                "raw channel is a per-step event; cumulative prefix is reported "
                "separately at H=10"
            ),
        },
    }
    save_json.parent.mkdir(parents=True, exist_ok=True)
    save_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def print_summary(result: Mapping[str, Any]) -> None:
    print("=" * 92)
    print("Phase-C behavior/closed-loop H=10 validation")
    print("=" * 92)
    print(
        "runs =", result["completed_runs"], "/", result["expected_runs"],
        "samples =", result["aggregate"]["coverage"]["samples"],
        "groups =", result["aggregate"]["coverage"]["candidate_groups"],
    )
    coverage = result["aggregate"]["coverage"]
    print(
        "re-encoded decision ticks =", coverage["decision_ticks_reencoded"],
        "groups with generated orders =", coverage["groups_with_generated_orders"],
        "groups with assigned tasks =", coverage["groups_with_assigned_tasks"],
    )
    for load, channels in result["aggregate"]["system_h10_by_load"].items():
        print(f"\nload={load}, H=10 (mean of complete run reports)")
        for name in SYSTEM_CHANNELS:
            metric = channels[name]
            print(
                f"  {name:<38} "
                f"MAE={metric['mae_mean']:.4f} "
                f"RMSE={metric['rmse_mean']:.4f} "
                f"Spearman={metric['spearman_mean']:+.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = run_evaluation(
        data_root=args.data_root,
        bundle_path=args.bundle,
        save_json=args.save_json,
        device_name=args.device,
        torch_threads=args.torch_threads,
        max_samples=args.max_samples,
        allow_partial=args.allow_partial,
    )
    print_summary(result)
    print("\nJSON saved:", args.save_json)


if __name__ == "__main__":
    main()
