"""Run paired pure-WM and parameter-free analytic-H5 Layer-5 arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from Policies.TaskAssigner import WorldModelTaskAssigner
from WorldModel.evaluation.analytic_work_layer5_protocol import (
    ANALYTIC_HORIZON,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOAD_CONFIGS,
    REQUIRED_LOADS,
    RESERVATION_WINDOW,
    SNAPSHOT_INTERVAL,
    TICKS,
    TOP_M_METADATA_ONLY,
    protocol,
    sha256_file,
    validate_exact_seed_set,
)
from WorldModel.evaluation.evaluate_online_v6 import (
    _run_one_assigner,
    aggregate_seeds,
)


BASELINE_LABEL = "WorldModel"
ANALYTIC_LABEL = "WorldModel+AnalyticWorkH5"
SCHEMA_VERSION = "analytic_work_layer5_paired_online_v1"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _verify_bundle(
    bundle: dict,
    *,
    world_model: Path,
    lyapunov_config: Path,
) -> tuple[dict, float]:
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unexpected analytic Layer-5 frozen bundle schema")
    frozen_protocol = bundle.get("protocol") or {}
    if frozen_protocol != protocol():
        raise ValueError("frozen analytic Layer-5 protocol differs from source")
    artifacts = bundle.get("artifacts") or {}
    for name, path in (
        ("base_world_model", world_model),
        ("lyapunov_config", lyapunov_config),
    ):
        expected = (artifacts.get(name) or {}).get("sha256")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen artifact changed: {name}")
    for name in (
        "horizon_development_report",
        "efficiency_decomposition_report",
    ):
        artifact = artifacts.get(name) or {}
        path = Path(str(artifact.get("path") or ""))
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise ValueError(f"frozen evidence artifact changed: {name}")
    for name, artifact in (bundle.get("source_files") or {}).items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise ValueError(f"frozen analytic Layer-5 source changed: {name}")
    return frozen_protocol, float(
        frozen_protocol["integration"]["lambda"]
    )


def _arm_meta(
    *,
    label: str,
    load: str,
    seed: int,
    config: str,
    checkpoint: str,
    frozen_protocol: dict,
    analytic_enabled: bool,
    order_manifest_sha256: str | None,
) -> dict:
    return {
        "arm_label": label,
        "load": load,
        "seed": int(seed),
        "config": config,
        "checkpoint_path": checkpoint,
        "protocol_sha256": frozen_protocol["protocol_sha256"],
        "role": frozen_protocol["role"],
        "fixed_order_pod_station_context": True,
        "online_robot_candidate_scope": "all_idle",
        "top_m_is_online_limit": False,
        "normal_arrivals_in_simulator": True,
        "paired_order_manifest_sha256": order_manifest_sha256,
        "future_unknown_orders_in_wm_rollout": False,
        "continuation_policy_in_wm_rollout": False,
        "td_risk_v_head": False,
        "greedy_or_external_assignment_policy": False,
        "analytic_work": {
            "enabled": bool(analytic_enabled),
            "mode": (
                "analytic_h5_group_range" if analytic_enabled else "off"
            ),
            "horizon": ANALYTIC_HORIZON if analytic_enabled else None,
            "learned_head": False,
            "candidate_residual": False,
            "shared_efficiency_eta": False,
        },
    }


def _require_new_path(path: Path, description: str) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {description}: {path}")


def _retry_path(path: Path, *, resume: bool) -> Path:
    """Keep an interrupted arm intact and allocate a fresh retry path."""

    if not path.exists() or not resume:
        return path
    for attempt in range(1, 1000):
        if path.suffix:
            candidate = path.with_name(
                f"{path.stem}.retry{attempt}{path.suffix}"
            )
        else:
            candidate = path.with_name(f"{path.name}_retry{attempt}")
        if not candidate.exists():
            print(f"[resume] preserving partial arm at {path}; retry={candidate}")
            return candidate
    raise RuntimeError(f"too many interrupted retries for {path}")


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_reused_baseline(
    report_path: Path,
    *,
    load: str,
    config: str,
    seed: int,
    ticks: int,
    manifests_dir: Path,
) -> tuple[dict, Path, dict]:
    """Reuse the expensive pure-WM arm from the failed Layer-5 run."""

    report = _read_json(report_path)
    if report.get("schema_version") != "work_drift_layer5_paired_online_v1":
        raise ValueError("unexpected reused Layer-5 baseline report schema")
    meta = report.get("meta") or {}
    if str(meta.get("load")) != str(load):
        raise ValueError("reused baseline report load does not match --load")
    if Path(str(meta.get("config"))).as_posix() != Path(config).as_posix():
        raise ValueError("reused baseline report config does not match --config")
    if int(meta.get("ticks", -1)) != int(ticks):
        raise ValueError("reused baseline report ticks do not match --ticks")
    seed_payload = (report.get("per_seed") or {}).get(str(int(seed))) or {}
    baseline_metrics = seed_payload.get(BASELINE_LABEL)
    if not isinstance(baseline_metrics, dict):
        raise ValueError(f"reused baseline report is missing seed {seed}")

    manifest_path = manifests_dir / f"layer5_{load}_seed{seed}_orders.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "layer5_order_arrival_manifest_v1":
        raise ValueError(f"unexpected order manifest schema: {manifest_path}")
    if (
        str(manifest.get("manifest_sha256"))
        != str(baseline_metrics.get("order_arrival_manifest_sha256"))
        or int(manifest.get("total_orders", -1))
        != int(baseline_metrics.get("order_arrival_count", -2))
    ):
        raise ValueError(
            "reused baseline metrics and realised order manifest differ: "
            f"load={load} seed={seed}"
        )
    return dict(baseline_metrics), manifest_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-bundle", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--lyapunov-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--load", choices=REQUIRED_LOADS, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--top-m", type=int, default=TOP_M_METADATA_ONLY)
    parser.add_argument(
        "--snapshot-interval", type=int, default=SNAPSHOT_INTERVAL
    )
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--baseline-online",
        default=None,
        help=(
            "optional failed Layer-5 online_<load>_431_440.json; reuse its "
            "pure-WM metrics instead of rerunning the expensive baseline arm"
        ),
    )
    parser.add_argument(
        "--order-manifests-dir",
        default=None,
        help=(
            "directory containing layer5_<load>_seed<seed>_orders.json for "
            "--baseline-online reuse"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume at completed pure-WM arm or paired per-seed boundaries",
    )
    parser.add_argument(
        "--formal-development",
        action="store_true",
        help=(
            "enforce the frozen 461--463 development seed/config/tick set; "
            "this remains localisation, not a Layer-5 certificate"
        ),
    )
    args = parser.parse_args()
    if bool(args.baseline_online) != bool(args.order_manifests_dir):
        raise SystemExit(
            "--baseline-online and --order-manifests-dir must be supplied together"
        )

    bundle_path = Path(args.frozen_bundle)
    frozen_protocol, analytic_lambda = _verify_bundle(
        _read_json(bundle_path),
        world_model=Path(args.world_model),
        lyapunov_config=Path(args.lyapunov_config),
    )
    seeds = [int(value) for value in args.seeds]
    if args.formal_development:
        validate_exact_seed_set(seeds)
        if int(args.ticks) != TICKS:
            raise SystemExit(
                f"frozen analytic Layer-5 development uses --ticks={TICKS}"
            )
        if int(args.top_m) != TOP_M_METADATA_ONLY:
            raise SystemExit(
                "frozen analytic Layer-5 development uses metadata "
                f"--top-m={TOP_M_METADATA_ONLY}"
            )
        if int(args.snapshot_interval) != SNAPSHOT_INTERVAL:
            raise SystemExit(
                "frozen analytic Layer-5 development uses snapshot interval "
                f"{SNAPSHOT_INTERVAL}"
            )
        expected_config = Path(LOAD_CONFIGS[args.load]).as_posix()
        if Path(args.config).as_posix() != expected_config:
            raise SystemExit(
                f"frozen {args.load} load requires {expected_config}"
            )

    output = Path(args.output)
    if output.exists():
        if args.resume:
            print(f"[complete] paired analytic Layer-5 output already exists: {output}")
            return
        _require_new_path(output, "paired analytic Layer-5 output")
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = artifacts_dir / "order_manifests"
    traces_dir = artifacts_dir / "decision_traces"
    snapshots_root = artifacts_dir / "decision_snapshots"
    for path in (manifests_dir, traces_dir, snapshots_root):
        path.mkdir(parents=True, exist_ok=True)
    lyapunov_config = _read_json(Path(args.lyapunov_config))
    reused_baseline_report = (
        Path(args.baseline_online) if args.baseline_online else None
    )
    reused_manifests_dir = (
        Path(args.order_manifests_dir) if args.order_manifests_dir else None
    )
    if reused_baseline_report is not None and not reused_baseline_report.is_file():
        raise FileNotFoundError(reused_baseline_report)
    if reused_manifests_dir is not None and not reused_manifests_dir.is_dir():
        raise FileNotFoundError(reused_manifests_dir)

    per_seed: dict[int, dict[str, dict]] = {}
    for seed in seeds:
        seed_result = artifacts_dir / (
            f"paired_{args.load}_seed{seed}.json"
        )
        if args.resume and seed_result.is_file():
            payload = _read_json(seed_result)
            meta = payload.get("meta") or {}
            if (
                int(meta.get("seed", -1)) != int(seed)
                or meta.get("load") != args.load
                or meta.get("protocol_sha256")
                != frozen_protocol["protocol_sha256"]
            ):
                raise RuntimeError(f"invalid per-seed resume artifact: {seed_result}")
            arms = payload.get("arms") or {}
            if set(arms) != {BASELINE_LABEL, ANALYTIC_LABEL}:
                raise RuntimeError(f"incomplete per-seed resume artifact: {seed_result}")
            for name in (
                "baseline_result",
                "order_manifest",
                "analytic_trace",
                "snapshot_index",
            ):
                path = Path(str(meta.get(f"{name}_path") or ""))
                expected = meta.get(f"{name}_sha256")
                if not path.is_file() or sha256_file(path) != expected:
                    raise RuntimeError(
                        f"per-seed resume artifact changed: {name}={path}"
                    )
            per_seed[seed] = arms
            print(f"[resume] completed {args.load} seed={seed} from {seed_result}")
            continue
        _require_new_path(seed_result, "per-seed analytic Layer-5 result")
        baseline_result = artifacts_dir / (
            f"baseline_{args.load}_seed{seed}.json"
        )
        resumed_baseline = None
        resumed_manifest = None
        resumed_manifest_path = None
        if args.resume and baseline_result.is_file():
            payload = _read_json(baseline_result)
            meta = payload.get("meta") or {}
            if (
                int(meta.get("seed", -1)) != int(seed)
                or meta.get("load") != args.load
                or meta.get("protocol_sha256")
                != frozen_protocol["protocol_sha256"]
            ):
                raise RuntimeError(
                    f"invalid baseline resume artifact: {baseline_result}"
                )
            resumed_baseline = payload.get("metrics")
            resumed_manifest_path = Path(str(meta.get("order_manifest_path")))
            if not isinstance(resumed_baseline, dict):
                raise RuntimeError(
                    f"baseline resume metrics are missing: {baseline_result}"
                )
            resumed_manifest = _read_json(resumed_manifest_path)
            if (
                str(resumed_manifest.get("manifest_sha256"))
                != str(meta.get("order_manifest_sha256"))
            ):
                raise RuntimeError(
                    f"baseline resume manifest changed: {resumed_manifest_path}"
                )
            resumed_trace_path = meta.get("decision_trace_path")
            if resumed_trace_path:
                path = Path(str(resumed_trace_path))
                if (
                    not path.is_file()
                    or sha256_file(path) != meta.get("decision_trace_sha256")
                ):
                    raise RuntimeError(
                        f"baseline resume trace changed: {path}"
                    )
            print(
                f"[resume] completed baseline {args.load} seed={seed} "
                f"from {baseline_result}"
            )
        else:
            _require_new_path(
                baseline_result, "per-seed pure-WM baseline result"
            )
        order_manifest = manifests_dir / (
            f"analytic_layer5_{args.load}_seed{seed}_orders.json"
        )
        baseline_trace_path = traces_dir / (
            f"analytic_layer5_{args.load}_wm_seed{seed}.jsonl"
        )
        analytic_trace_path = traces_dir / (
            f"analytic_layer5_{args.load}_analytic_seed{seed}.jsonl"
        )
        snapshot_dir = snapshots_root / f"{args.load}_seed{seed}"
        analytic_trace_path = _retry_path(
            analytic_trace_path, resume=bool(args.resume)
        )
        snapshot_dir = _retry_path(
            snapshot_dir, resume=bool(args.resume)
        )
        if reused_baseline_report is None and resumed_baseline is None:
            _require_new_path(baseline_trace_path, "pure-WM decision trace")
        _require_new_path(analytic_trace_path, "analytic decision trace")
        _require_new_path(snapshot_dir, "analytic decision snapshot directory")

        if resumed_baseline is not None:
            baseline_metrics = resumed_baseline
            order_manifest = resumed_manifest_path
            manifest = resumed_manifest
        elif reused_baseline_report is not None:
            baseline_metrics, order_manifest, manifest = _load_reused_baseline(
                reused_baseline_report,
                load=args.load,
                config=args.config,
                seed=seed,
                ticks=args.ticks,
                manifests_dir=reused_manifests_dir,
            )
            print(
                f"\n--- Analytic Layer 5 {args.load} seed={seed}: "
                "reuse frozen pure-WorldModel baseline ---"
            )
        else:
            print(
                f"\n--- Analytic Layer 5 {args.load} seed={seed}: "
                "pure WorldModel ---"
            )
            baseline = WorldModelTaskAssigner(
                checkpoint_path=args.world_model,
                top_m=args.top_m,
                reservation_window=RESERVATION_WINDOW,
                lyapunov_l0_config=lyapunov_config,
                candidate_robot_mode="nearest",
                candidate_context_mode="prefix",
                candidate_context_factor=1.0,
                decision_trace_enabled=True,
            )
            baseline_metrics = _run_one_assigner(
                args.config,
                baseline,
                seed,
                args.ticks,
                trace_label=BASELINE_LABEL,
                decision_trace_jsonl=str(baseline_trace_path),
                decision_trace_horizons=[5, 50, 100, 200],
                save_order_manifest=str(order_manifest),
            )
            manifest = _read_json(order_manifest)
        manifest_sha256 = str(manifest["manifest_sha256"])
        if not baseline_result.exists():
            baseline_trace_hash = (
                sha256_file(baseline_trace_path)
                if baseline_trace_path.is_file() else None
            )
            _write_json_atomic(baseline_result, {
                "schema_version": "analytic_work_layer5_baseline_seed_v1",
                "meta": {
                    "load": args.load,
                    "seed": int(seed),
                    "protocol_sha256": frozen_protocol["protocol_sha256"],
                    "order_manifest_path": str(order_manifest),
                    "order_manifest_sha256": manifest_sha256,
                    "decision_trace_path": (
                        str(baseline_trace_path)
                        if baseline_trace_hash is not None else None
                    ),
                    "decision_trace_sha256": baseline_trace_hash,
                    "source": (
                        "reused_failed_layer5_baseline"
                        if reused_baseline_report is not None
                        else "new_pure_world_model_run"
                    ),
                },
                "metrics": baseline_metrics,
            })

        print(
            f"--- Analytic Layer 5 {args.load} seed={seed}: "
            "WM+parameter-free H5 analytic work ---"
        )
        analytic = WorldModelTaskAssigner(
            checkpoint_path=args.world_model,
            top_m=args.top_m,
            reservation_window=RESERVATION_WINDOW,
            lyapunov_l0_config=lyapunov_config,
            candidate_robot_mode="nearest",
            candidate_context_mode="prefix",
            candidate_context_factor=1.0,
            work_drift_mode="analytic_h5_group_range",
            work_drift_lambda=analytic_lambda,
        )
        analytic_meta = _arm_meta(
            label=ANALYTIC_LABEL,
            load=args.load,
            seed=seed,
            config=args.config,
            checkpoint=args.world_model,
            frozen_protocol=frozen_protocol,
            analytic_enabled=True,
            order_manifest_sha256=manifest_sha256,
        )
        analytic_metrics = _run_one_assigner(
            args.config,
            analytic,
            seed,
            args.ticks,
            trace_label=ANALYTIC_LABEL,
            decision_trace_jsonl=str(analytic_trace_path),
            decision_trace_horizons=[5, 50, 100, 200],
            decision_snapshot_dir=str(snapshot_dir),
            decision_snapshot_interval=int(args.snapshot_interval),
            decision_snapshot_top_m=int(args.top_m),
            decision_snapshot_run_id=(
                f"analytic_layer5_{args.load}_seed{seed}"
            ),
            decision_snapshot_meta=analytic_meta,
            decision_snapshot_candidate_scope="all_idle_online",
            decision_snapshot_attach_trace=True,
            decision_snapshot_require_trace_alignment=True,
            decision_snapshot_capture_policy=(
                "all_modified_plus_interval_controls"
            ),
            recorded_orders_path=str(order_manifest),
        )
        if (
            baseline_metrics.get("order_arrival_manifest_sha256")
            != analytic_metrics.get("order_arrival_manifest_sha256")
            or baseline_metrics.get("order_arrival_count")
            != analytic_metrics.get("order_arrival_count")
        ):
            raise RuntimeError(
                "paired arms did not receive the exact same realised order stream"
            )
        if int(analytic_metrics.get("fallback_greedy_calls", 0)) != 0:
            raise RuntimeError("analytic arm used a forbidden fallback")
        if int(analytic_metrics.get("work_drift_contexts", 0)) <= 0:
            raise RuntimeError("analytic H5 scoring was never exercised")
        if int(analytic_metrics.get("decision_snapshots_saved", 0)) <= 0:
            raise RuntimeError("no aligned Phase-C candidate snapshots were saved")

        per_seed[seed] = {
            BASELINE_LABEL: baseline_metrics,
            ANALYTIC_LABEL: analytic_metrics,
        }
        snapshot_index = snapshot_dir / (
            f"snapindex_analytic_layer5_{args.load}_seed{seed}.json"
        )
        for path, description in (
            (analytic_trace_path, "analytic decision trace"),
            (snapshot_index, "analytic snapshot index"),
            (baseline_result, "baseline result"),
            (Path(order_manifest), "order manifest"),
        ):
            if not path.is_file():
                raise RuntimeError(f"missing {description}: {path}")
        seed_payload = {
            "schema_version": "analytic_work_layer5_per_seed_v1",
            "meta": {
                "load": args.load,
                "seed": int(seed),
                "protocol_sha256": frozen_protocol["protocol_sha256"],
                "order_stream_manifest_sha256": manifest_sha256,
                "baseline_reused": reused_baseline_report is not None,
                "baseline_online_source": (
                    str(reused_baseline_report)
                    if reused_baseline_report is not None else None
                ),
                "baseline_result_path": str(baseline_result),
                "baseline_result_sha256": sha256_file(baseline_result),
                "order_manifest_path": str(order_manifest),
                "order_manifest_sha256": sha256_file(order_manifest),
                "analytic_trace_path": str(analytic_trace_path),
                "analytic_trace_sha256": sha256_file(analytic_trace_path),
                "snapshot_index_path": str(snapshot_index),
                "snapshot_index_sha256": sha256_file(snapshot_index),
            },
            "arms": per_seed[seed],
        }
        _write_json_atomic(seed_result, seed_payload)
        print(
            f"[done] {args.load} seed={seed} cost "
            f"{baseline_metrics.get('wm_label_cost')} -> "
            f"{analytic_metrics.get('wm_label_cost')}; orders "
            f"{baseline_metrics.get('completed_orders')} -> "
            f"{analytic_metrics.get('completed_orders')}; modified="
            f"{analytic_metrics.get('work_drift_modified_decision_rate')}"
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "formal_development": bool(args.formal_development),
            "certification_status": frozen_protocol["certification_status"],
            "load": args.load,
            "config": args.config,
            "seeds": seeds,
            "ticks": int(args.ticks),
            "top_m": int(args.top_m),
            "top_m_is_online_limit": False,
            "online_robot_candidate_scope": "all_idle",
            "paired_arms": [BASELINE_LABEL, ANALYTIC_LABEL],
            "normal_arrivals": True,
            "analytic_horizon": ANALYTIC_HORIZON,
            "analytic_lambda": analytic_lambda,
            "learned_work_or_td_head": False,
            "protocol_sha256": frozen_protocol["protocol_sha256"],
            "frozen_bundle": str(bundle_path),
            "frozen_bundle_sha256": sha256_file(bundle_path),
            "artifacts_dir": str(artifacts_dir),
            "snapshot_interval": int(args.snapshot_interval),
            "baseline_reused": reused_baseline_report is not None,
            "baseline_online_source": (
                str(reused_baseline_report)
                if reused_baseline_report is not None else None
            ),
            "baseline_online_source_sha256": (
                sha256_file(reused_baseline_report)
                if reused_baseline_report is not None else None
            ),
            "order_manifests_dir": str(
                reused_manifests_dir or manifests_dir
            ),
        },
        "per_seed": {str(seed): values for seed, values in per_seed.items()},
        "aggregate": aggregate_seeds(per_seed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nsaved paired analytic Layer-5 result: {output}")


if __name__ == "__main__":
    main()
