"""Validate the frozen station-region head on fresh 531--540 traces.

The validator consumes only same-tick physical labels from the fresh runs.
It deliberately does not build endpoint labels, rollout deltas, or an online
score.  A compressed row-level artifact is retained so the result can be
plotted without rerunning the simulator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from WorldModel.core.station_congestion_head import (
    CHANNEL_NAMES,
    HEAD_SCHEMA_VERSION,
    StationCongestionHead,
    build_station_targets,
    verify_scale_contract,
)
from WorldModel.data.build_station_congestion_head_dataset import (
    _load_td,
    _read_trace,
    _station_rows,
    align_station_node_ids,
    build_station_region_node_ids,
    build_station_representations,
)
from WorldModel.evaluation.evaluate import _load_model
from WorldModel.evaluation.phase_c_phi_state_531_540_protocol import (
    ARMS,
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    HEAD_CHECKPOINT,
    LOADS,
    OUTPUT_ROOT,
    PRIMARY_REGION_HOPS,
    REPORT_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    SCALE_CONTRACT,
    SCHEMA_VERSION,
    SEEDS,
    TICKS,
    canonical_sha256,
    sha256_file,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3 or right.size != left.size:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator <= 1e-15:
        return float("nan")
    return float((left @ right) / denominator)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size != right.size or left.size < 3:
        return float("nan")
    if np.all(left == left[0]) or np.all(right == right[0]):
        return float("nan")
    return _pearson(_rankdata(left), _rankdata(right))


def _ci(values: Sequence[float], *, seed: int) -> list[float]:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return [float("nan"), float("nan")]
    if finite.size == 1:
        return [float(finite[0]), float(finite[0])]
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(
        finite, size=(int(BOOTSTRAP_REPEATS), finite.size), replace=True
    )
    return [float(value) for value in np.quantile(draws.mean(axis=1), [0.025, 0.975])]


def _metric_block(
    prediction: np.ndarray,
    target: np.ndarray,
    run_ids: Sequence[str],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    residual = prediction - target
    per_run: list[float] = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, run_id in enumerate(run_ids):
        grouped[str(run_id)].append(index)
    for indices in grouped.values():
        value = _spearman(prediction[indices], target[indices])
        if math.isfinite(value):
            per_run.append(value)
    return {
        "samples": int(prediction.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "bias": float(np.mean(residual)),
        "prediction_std": float(np.std(prediction)),
        "target_std": float(np.std(target)),
        "pooled_spearman": float(_spearman(prediction, target)),
        "per_run_spearman_mean": (
            float(np.mean(per_run)) if per_run else float("nan")
        ),
        "per_run_spearman_median": (
            float(np.median(per_run)) if per_run else float("nan")
        ),
        "per_run_cluster_ci95": _ci(per_run, seed=BOOTSTRAP_SEED + seed_offset),
        "run_sign_consistency": (
            float(np.mean(np.asarray(per_run) > 0.0)) if per_run else float("nan")
        ),
    }


def _within_tick_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    run_ids: Sequence[str],
    ticks: Sequence[int],
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (run_id, tick) in enumerate(zip(run_ids, ticks)):
        groups[(str(run_id), int(tick))].append(index)
    correlations: list[float] = []
    top1: list[bool] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        rho = _spearman(prediction[indices], target[indices])
        if math.isfinite(rho):
            correlations.append(rho)
        pred_best = indices[int(np.argmax(prediction[indices]))]
        target_best = float(np.max(target[indices]))
        top1.append(bool(target[pred_best] >= target_best - 1e-12))
    return {
        "groups": len(groups),
        "valid_groups": len(correlations),
        "spearman_mean": float(np.mean(correlations)) if correlations else float("nan"),
        "spearman_median": float(np.median(correlations)) if correlations else float("nan"),
        "positive_fraction": (
            float(np.mean(np.asarray(correlations) > 0.0))
            if correlations else float("nan")
        ),
        "top1_hit_rate": float(np.mean(top1)) if top1 else float("nan"),
    }


def _subset_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    run_ids: Sequence[str],
    ticks: Sequence[int],
    groups: Sequence[str],
    values: Sequence[str],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    indices = np.asarray(
        [index for index, value in enumerate(groups) if str(value) in set(values)],
        dtype=np.int64,
    )
    if indices.size == 0:
        return {"samples": 0}
    result = _metric_block(
        prediction[indices], target[indices],
        [run_ids[index] for index in indices], seed_offset=seed_offset,
    )
    result["within_tick_station"] = _within_tick_metrics(
        prediction[indices], target[indices],
        [run_ids[index] for index in indices],
        [ticks[index] for index in indices],
    )
    return result


def _verify_bundle(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = _read_json(path)
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"wrong phi_state frozen bundle: {path}")
    protocol = dict(bundle.get("protocol") or {})
    claimed = str(protocol.pop("protocol_sha256", ""))
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong phi_state protocol schema")
    if canonical_sha256(protocol) != claimed:
        raise ValueError("phi_state protocol hash mismatch")
    protocol["protocol_sha256"] = claimed
    return bundle, protocol


def _verify_artifact(bundle: Mapping[str, Any], key: str) -> Path:
    value = (bundle.get("artifacts") or {}).get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing frozen artifact {key}")
    path = Path(str(value.get("path", "")))
    expected = str(value.get("sha256", ""))
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"frozen artifact mismatch: {path}")
    return path


def _discover_runs(input_root: Path) -> list[dict[str, Any]]:
    result = []
    for summary_path in sorted((input_root / "runs").glob("*/phi_state_run_summary.json")):
        summary = _read_json(summary_path)
        if summary.get("schema_version") != RUN_SCHEMA_VERSION:
            raise ValueError(f"wrong run schema: {summary_path}")
        if not bool((summary.get("audit") or {}).get("passed")):
            raise ValueError(f"run audit failed: {summary_path}")
        run_dir = summary_path.parent
        manifest = run_dir / str(summary.get("run_outputs_manifest", "run_outputs.sha256"))
        if sha256_file(manifest) != summary.get("run_outputs_sha256"):
            raise ValueError(f"run output manifest hash mismatch: {run_dir}")
        for raw in manifest.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            expected, relative = raw.split(None, 1)
            path = run_dir / relative.strip()
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"run artifact hash mismatch: {path}")
        outputs = summary.get("outputs") or {}
        result.append({
            "run_id": run_dir.name,
            "arm": str(summary["arm"]),
            "load": str(summary["load"]),
            "seed": int(summary["seed"]),
            "ticks": int(summary["ticks"]),
            "run_dir": run_dir,
            "trace_path": run_dir / str(outputs["trace"]),
            "td_path": run_dir / str(outputs["td_stream"]),
        })
    if not result:
        raise ValueError(f"no complete runs under {input_root / 'runs'}")
    return result


def _load_head(path: Path, device: torch.device) -> tuple[StationCongestionHead, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("head_schema_version") != HEAD_SCHEMA_VERSION:
        raise ValueError("wrong station head schema")
    latent_dim = int(payload.get("latent_dim", -1))
    head = StationCongestionHead(latent_dim).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head, payload


def validate(
    *,
    input_root: Path,
    frozen_bundle: Path,
    checkpoint: Path,
    scale_contract_path: Path,
    output_root: Path,
    device: torch.device,
    torch_threads: int,
    reference_validation: Path | None,
    expected_arms: Sequence[str],
    allow_subset: bool,
) -> dict[str, Any]:
    bundle, protocol = _verify_bundle(frozen_bundle)
    protocol_sha = str(protocol["protocol_sha256"])
    frozen_head = _verify_artifact(bundle, "station_head_checkpoint")
    frozen_scale = _verify_artifact(bundle, "station_scale_contract")
    if checkpoint != frozen_head or scale_contract_path != frozen_scale:
        raise ValueError("runtime checkpoint/scale path differs from frozen protocol")
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    scale_contract = _read_json(scale_contract_path)
    verify_scale_contract(scale_contract)
    if sha256_file(scale_contract_path) != protocol["frozen_inputs"]["scale_contract_sha256"]:
        raise ValueError("scale contract hash differs from protocol")

    runs = _discover_runs(input_root)
    selected_arms = tuple(str(value) for value in expected_arms)
    if not selected_arms or any(arm not in ARMS for arm in selected_arms):
        raise ValueError(f"invalid expected arms: {selected_arms}")
    if len(set(selected_arms)) != len(selected_arms):
        raise ValueError("expected arms contain duplicates")
    if not allow_subset and set(selected_arms) != set(ARMS):
        raise ValueError("partial validation requires --allow-subset")
    expected = {
        (arm, load, seed)
        for arm in selected_arms for load in LOADS for seed in SEEDS
    }
    full_expected = {
        (arm, load, seed) for arm in ARMS for load in LOADS for seed in SEEDS
    }
    observed = {(r["arm"], r["load"], r["seed"]) for r in runs}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"run coverage mismatch: missing={missing[:5]} extra={extra[:5]}")

    torch.set_num_threads(max(int(torch_threads), 1))
    model, label_schema = _load_model(str(_verify_artifact(bundle, "source_candidate_checkpoint")))
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head, head_payload = _load_head(checkpoint, device)
    if int(head_payload.get("latent_dim", -1)) != 192:
        raise ValueError("frozen region head must have latent_dim=192")

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    row_run: list[str] = []
    row_load: list[str] = []
    row_arm: list[str] = []
    row_seed: list[int] = []
    row_tick: list[int] = []
    run_summaries = []

    with torch.no_grad():
        for run_index, source in enumerate(sorted(runs, key=lambda r: r["run_id"])):
            payload = _load_td(source["td_path"])
            frames = payload["frames_by_tick"]
            trace = _read_trace(source["trace_path"])
            ticks = sorted(int(value) for value in frames)
            if not ticks:
                raise ValueError(f"empty frames: {source['run_id']}")
            first_tick = ticks[0]
            template_rows = _station_rows(trace[first_tick])
            template_station_ids = [
                int(row["station_id"]) for row in template_rows
            ]
            aligned_ids = align_station_node_ids(
                template_rows, payload.get("station_node_ids") or []
            )
            first_frame = frames[first_tick]
            node_count = int(
                (trace[first_tick].get("system") or {}).get(
                    "node_count", first_frame["node_history"].size(1)
                )
            )
            regions = build_station_region_node_ids(
                template_rows,
                payload["edge_index"],
                num_nodes=node_count,
                hops=PRIMARY_REGION_HOPS,
            )
            run_pred = []
            run_true = []
            for tick in ticks:
                trace_row = trace.get(tick)
                if trace_row is None:
                    raise ValueError(f"missing trace tick {tick}: {source['run_id']}")
                station_rows = _station_rows(trace_row)
                if [int(row["station_id"]) for row in station_rows] != template_station_ids:
                    raise ValueError(
                        f"station order/count changed within {source['run_id']}"
                    )
                frame = frames[tick]
                z, _, _ = model.encode_state(
                    frame["node_history"].to(device),
                    payload["edge_index"].to(device),
                    frame["edge_features"].to(device),
                    frame["demand_context"].to(device),
                )
                station_latents = build_station_representations(
                    z,
                    aligned_ids,
                    representation="station_region_mean_max",
                    station_region_node_ids=regions,
                )
                pred = head.forward_station_latents(station_latents)
                true = torch.tensor([
                    [
                        build_station_targets(row, scale_contract)["channels"][name]
                        for name in CHANNEL_NAMES
                    ]
                    for row in station_rows
                ], dtype=torch.float32, device=device)
                pred_np = pred.detach().cpu().numpy()
                true_np = true.detach().cpu().numpy()
                predictions.append(pred_np)
                targets.append(true_np)
                run_pred.extend(pred_np.tolist())
                run_true.extend(true_np.tolist())
                for station_offset in range(len(station_rows)):
                    row_run.append(source["run_id"])
                    row_load.append(source["load"])
                    row_arm.append(source["arm"])
                    row_seed.append(source["seed"])
                    row_tick.append(tick)
            run_summaries.append({
                "run_id": source["run_id"],
                "arm": source["arm"],
                "load": source["load"],
                "seed": source["seed"],
                "frames": len(ticks),
                "station_rows": len(run_pred),
                "traffic_spearman": _spearman(
                    np.asarray(run_pred)[:, 0], np.asarray(run_true)[:, 0]
                ),
                "service_spearman": _spearman(
                    np.asarray(run_pred)[:, 1], np.asarray(run_true)[:, 1]
                ),
            })
            print(
                f"[validate] {run_index + 1}/{len(runs)} {source['run_id']} "
                f"frames={len(ticks)}",
                flush=True,
            )

    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    run_ids = np.asarray(row_run)
    loads = np.asarray(row_load)
    arms = np.asarray(row_arm)
    ticks = np.asarray(row_tick, dtype=np.int64)
    seeds = np.asarray(row_seed, dtype=np.int64)
    channels: dict[str, Any] = {}
    for index, name in enumerate(CHANNEL_NAMES):
        channels[name] = _metric_block(
            prediction[:, index], target[:, index], row_run,
            seed_offset=index,
        )
        channels[name]["within_tick_station"] = _within_tick_metrics(
            prediction[:, index], target[:, index], row_run, row_tick
        )
        channels[name]["by_load"] = {
            load: _subset_metrics(
                prediction[:, index], target[:, index], row_run, row_tick,
                row_load, [load], seed_offset=100 + index,
            ) for load in LOADS
        }
        channels[name]["by_arm"] = {
            arm: _subset_metrics(
                prediction[:, index], target[:, index], row_run, row_tick,
                row_arm, [arm], seed_offset=200 + index,
            ) for arm in ARMS
        }

    service_metrics = channels["service"]
    service_within = service_metrics["within_tick_station"]
    service_bars = (protocol.get("analysis") or {}).get(
        "primary_service_bars", {}
    )
    service_gate_checks = {
        "pooled_spearman": float(service_metrics["pooled_spearman"])
        >= float(service_bars.get("pooled_spearman_ge", 0.80)),
        "per_run_ci95_lower": float(
            service_metrics["per_run_cluster_ci95"][0]
        ) >= float(service_bars.get("per_run_cluster_ci95_lower_ge", 0.70)),
        "within_tick_mean": float(service_within["spearman_mean"])
        >= float(service_bars.get("within_tick_station_spearman_mean_ge", 0.60)),
        "within_tick_positive_fraction": float(
            service_within["positive_fraction"]
        ) >= float(service_bars.get("within_tick_positive_fraction_ge", 0.80)),
    }

    reference = None
    if reference_validation is not None and reference_validation.is_file():
        ref = _read_json(reference_validation)
        reference = {
            "path": reference_validation.as_posix(),
            "sha256": sha256_file(reference_validation),
            "test": {
                name: ((ref.get("splits") or {}).get("test") or {})
                .get("channels", {}).get(name, {})
                for name in CHANNEL_NAMES
            },
        }
        for name in CHANNEL_NAMES:
            reference["test"][name] = dict(reference["test"][name])
            reference["test"][name]["source"] = "511_520_test"

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "development_only": True,
        "protocol_sha256": protocol_sha,
        "input_root": input_root.as_posix(),
        "head_checkpoint": checkpoint.as_posix(),
        "scale_contract": scale_contract_path.as_posix(),
        "head_representation": head_payload.get("representation"),
        "model_checkpoint_label_schema": label_schema,
        "coverage": {
            "expected_runs": len(expected),
            "observed_runs": len(runs),
            "expected_seeds": list(SEEDS),
            "arms": list(selected_arms),
            "full_protocol_arms": list(ARMS),
            "full_protocol_coverage": observed == full_expected,
            "subset_allowed": bool(allow_subset),
            "loads": list(LOADS),
            "samples": int(prediction.shape[0]),
        },
        "channels": channels,
        "primary_service_bars": service_bars,
        "primary_service_gate": {
            "passed": all(service_gate_checks.values()),
            "checks": service_gate_checks,
        },
        "run_summaries": run_summaries,
        "reference_511_520": reference,
        "interpretation": {
            "phi_state_is_primary": True,
            "psi_pre_is_not_evaluated_here": True,
            "delta_psi_is_not_computed": True,
            "online_policy_changed": False,
            "same_tick_labels_only": True,
        },
        "audit": {
            "passed": True,
            "fresh_seed_block": True,
            "locked_501_510_used": False,
            "head_frozen": True,
            "scale_contract_frozen": True,
            "scale_refit_on_531_540": False,
            "world_model_retrained": False,
            "same_tick_target": True,
            "endpoint_target": False,
            "delta_psi": False,
            "online_policy_changed": False,
            "service_primary_gate_evaluated": True,
            "traffic_auxiliary_only": True,
            "full_protocol_coverage": observed == full_expected,
            "exploratory_subset": observed != full_expected,
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    np.savez_compressed(
        staging / "phi_state_predictions.npz",
        prediction=prediction.astype(np.float32),
        target=target.astype(np.float32),
        run_id=run_ids,
        load=loads,
        arm=arms,
        seed=seeds,
        tick=ticks,
    )
    report["outputs"] = {
        "prediction_rows": int(prediction.shape[0]),
        "prediction_file": "phi_state_predictions.npz",
        "validation_file": "phi_state_validation.json",
        "manifest_file": "validation_outputs.sha256",
    }
    report_path = staging / "phi_state_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = staging / "validation_outputs.sha256"
    manifest.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != manifest.name
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    staging.rename(output_root)
    print(f"[complete] phi_state validation: {output_root}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--frozen-bundle",
        type=Path,
        default=OUTPUT_ROOT / "phase_c_phi_state_frozen_protocol.json",
    )
    parser.add_argument("--checkpoint", type=Path, default=HEAD_CHECKPOINT)
    parser.add_argument("--scale-contract", type=Path, default=SCALE_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT / "validation")
    parser.add_argument("--reference-validation", type=Path, default=Path(
        "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
        "station_congestion_head_region_dev_511_520_v1/linear_head_v1/"
        "state_level_validation.json"
    ))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument(
        "--expected-arms",
        default=",".join(ARMS),
        help="Comma-separated collected arms; defaults to the full protocol.",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Permit an explicitly labelled exploratory arm subset.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")
    validate(
        input_root=args.input_root,
        frozen_bundle=args.frozen_bundle,
        checkpoint=args.checkpoint,
        scale_contract_path=args.scale_contract,
        output_root=args.output_root,
        device=device,
        torch_threads=args.torch_threads,
        reference_validation=args.reference_validation,
        expected_arms=tuple(
            value.strip() for value in args.expected_arms.split(",")
            if value.strip()
        ),
        allow_subset=bool(args.allow_subset),
    )


if __name__ == "__main__":
    main()
