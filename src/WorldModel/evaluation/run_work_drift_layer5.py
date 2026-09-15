"""Run paired pure-WM and WM+WorkDrift Layer-5 online arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Policies.TaskAssigner import WorldModelTaskAssigner
from WorldModel.evaluation.evaluate_online_v6 import (
    _run_one_assigner,
    aggregate_seeds,
)
from WorldModel.evaluation.work_drift_layer5_protocol import (
    CERTIFICATION_SEEDS,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    LOAD_CONFIGS,
    RESERVATION_WINDOW,
    REQUIRED_LOADS,
    TICKS,
    TOP_M_METADATA_ONLY,
    formal_protocol,
    sha256_file,
    validate_exact_seed_set,
)


BASELINE_LABEL = "WorldModel"
WORK_LABEL = "WorldModel+WorkDrift"
SCHEMA_VERSION = "work_drift_layer5_paired_online_v1"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _verify_bundle(
    bundle: dict,
    *,
    world_model: Path,
    head: Path,
    layer4_report: Path,
) -> tuple[dict, float]:
    if bundle.get("schema_version") != FROZEN_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unexpected frozen Layer-5 bundle schema")
    protocol = bundle.get("formal_protocol") or {}
    if protocol != formal_protocol():
        raise ValueError("frozen Layer-5 protocol differs from source protocol")
    artifacts = bundle.get("artifacts") or {}
    checks = (
        ("base_world_model", world_model),
        ("work_drift_head", head),
        ("layer4_report", layer4_report),
    )
    for name, path in checks:
        expected = (artifacts.get(name) or {}).get("sha256")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen artifact changed: {name}")
    value = float((bundle.get("integration") or {})["lambda_audit"]["value"])
    if value != float(protocol["integration"]["lambda"]):
        raise ValueError("bundle lambda differs from formal protocol")
    return protocol, value


def _arm_meta(
    *,
    label: str,
    load: str,
    seed: int,
    config: str,
    checkpoint: str,
    protocol: dict,
    work_enabled: bool,
    work_lambda: float,
    order_stream_mode: str,
) -> dict:
    return {
        "arm_label": label,
        "load": load,
        "seed": int(seed),
        "config": config,
        "checkpoint_path": checkpoint,
        "five_layer_role": "layer5_normal_arrival_closed_loop_stability",
        "layer5_protocol_sha256": protocol["protocol_sha256"],
        "fixed_order_pod_station_context": True,
        "online_robot_candidate_scope": "all_idle",
        "top_m_is_online_limit": False,
        "normal_arrivals_in_simulator": True,
        "future_unknown_orders_in_wm_rollout": False,
        "continuation_policy_in_wm_rollout": False,
        "td_risk_v_head": False,
        "greedy_or_external_assignment_policy": False,
        "paired_order_stream_mode": order_stream_mode,
        "work_drift": {
            "enabled": bool(work_enabled),
            "mode": "group_range_additive" if work_enabled else "off",
            "lambda": float(work_lambda) if work_enabled else 0.0,
            "hard_gap_gate": False,
            "load_gate": False,
            "endpoint_horizon": 10 if work_enabled else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-bundle", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--layer4-report", required=True)
    parser.add_argument("--lyapunov-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--load", choices=REQUIRED_LOADS, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--top-m", type=int, default=TOP_M_METADATA_ONLY)
    parser.add_argument("--streams-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--formal",
        action="store_true",
        help="enforce the exact frozen seeds/config/ticks and refuse overwrite",
    )
    args = parser.parse_args()

    bundle_path = Path(args.frozen_bundle)
    bundle = _read_json(bundle_path)
    protocol, work_lambda = _verify_bundle(
        bundle,
        world_model=Path(args.world_model),
        head=Path(args.head),
        layer4_report=Path(args.layer4_report),
    )
    seeds = [int(value) for value in args.seeds]
    if args.formal:
        validate_exact_seed_set(seeds)
        if int(args.ticks) != TICKS:
            raise SystemExit(f"formal Layer 5 freezes --ticks={TICKS}")
        expected_config = Path(LOAD_CONFIGS[args.load]).as_posix()
        actual_config = Path(args.config).as_posix()
        if actual_config != expected_config:
            raise SystemExit(
                f"formal {args.load} load requires {expected_config}, got {actual_config}"
            )
        if int(args.top_m) != TOP_M_METADATA_ONLY:
            raise SystemExit(
                f"formal Layer 5 freezes metadata --top-m={TOP_M_METADATA_ONLY}"
            )

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite paired Layer-5 result: {output}")
    streams_dir = Path(args.streams_dir)
    streams_dir.mkdir(parents=True, exist_ok=True)
    order_manifest_dir = streams_dir.parent / "order_manifests"
    order_manifest_dir.mkdir(parents=True, exist_ok=True)
    lyapunov_config = _read_json(Path(args.lyapunov_config))

    per_seed: dict[int, dict[str, dict]] = {}
    for seed in seeds:
        baseline_stream = streams_dir / (
            f"tdstream_layer5_{args.load}_wm_seed{seed}.pt"
        )
        work_stream = streams_dir / (
            f"tdstream_layer5_{args.load}_work_seed{seed}.pt"
        )
        order_manifest = order_manifest_dir / (
            f"layer5_{args.load}_seed{seed}_orders.json"
        )
        if args.formal and (
            baseline_stream.exists()
            or work_stream.exists()
            or order_manifest.exists()
        ):
            print(
                "[resume] rerunning the frozen seed deterministically; the "
                "order manifest must match exactly before outputs are accepted"
            )
        print(f"\n--- Layer 5 {args.load} seed={seed}: pure WorldModel ---")
        baseline = WorldModelTaskAssigner(
            checkpoint_path=args.world_model,
            top_m=args.top_m,
            reservation_window=RESERVATION_WINDOW,
            lyapunov_l0_config=lyapunov_config,
            candidate_robot_mode="nearest",
            candidate_context_mode="prefix",
            candidate_context_factor=1.0,
        )
        baseline_meta = _arm_meta(
            label=BASELINE_LABEL,
            load=args.load,
            seed=seed,
            config=args.config,
            checkpoint=args.world_model,
            protocol=protocol,
            work_enabled=False,
            work_lambda=0.0,
            order_stream_mode="generate_then_freeze_manifest",
        )
        baseline_metrics = _run_one_assigner(
            args.config,
            baseline,
            seed,
            args.ticks,
            trace_label=BASELINE_LABEL,
            td_stream_dir=str(streams_dir),
            td_stream_frame_stride=int(args.ticks) + 1,
            td_stream_run_id=f"layer5_{args.load}_wm_seed{seed}",
            td_stream_meta=baseline_meta,
            td_stream_record_lyapunov_l0=True,
            td_stream_lyapunov_l0_config=lyapunov_config,
            save_order_manifest=str(order_manifest),
        )

        print(f"--- Layer 5 {args.load} seed={seed}: WM+WorkDrift ---")
        work = WorldModelTaskAssigner(
            checkpoint_path=args.world_model,
            top_m=args.top_m,
            reservation_window=RESERVATION_WINDOW,
            lyapunov_l0_config=lyapunov_config,
            candidate_robot_mode="nearest",
            candidate_context_mode="prefix",
            candidate_context_factor=1.0,
            work_drift_mode="group_range_additive",
            work_drift_head_path=args.head,
            work_drift_layer4_report_path=args.layer4_report,
            work_drift_lambda=work_lambda,
            work_drift_layer5_evaluation=True,
        )
        work_meta = _arm_meta(
            label=WORK_LABEL,
            load=args.load,
            seed=seed,
            config=args.config,
            checkpoint=args.world_model,
            protocol=protocol,
            work_enabled=True,
            work_lambda=work_lambda,
            order_stream_mode="replay_frozen_baseline_manifest",
        )
        work_metrics = _run_one_assigner(
            args.config,
            work,
            seed,
            args.ticks,
            trace_label=WORK_LABEL,
            td_stream_dir=str(streams_dir),
            td_stream_frame_stride=int(args.ticks) + 1,
            td_stream_run_id=f"layer5_{args.load}_work_seed{seed}",
            td_stream_meta=work_meta,
            td_stream_record_lyapunov_l0=True,
            td_stream_lyapunov_l0_config=lyapunov_config,
            recorded_orders_path=str(order_manifest),
        )
        if (
            baseline_metrics.get("order_arrival_manifest_sha256")
            != work_metrics.get("order_arrival_manifest_sha256")
            or baseline_metrics.get("order_arrival_count")
            != work_metrics.get("order_arrival_count")
        ):
            raise RuntimeError(
                "paired arms did not receive the exact same realised order stream"
            )
        per_seed[seed] = {
            BASELINE_LABEL: baseline_metrics,
            WORK_LABEL: work_metrics,
        }
        print(
            f"[done] {args.load} seed={seed} "
            f"cost {baseline_metrics.get('wm_label_cost')} -> "
            f"{work_metrics.get('wm_label_cost')}; "
            f"orders {baseline_metrics.get('completed_orders')} -> "
            f"{work_metrics.get('completed_orders')}"
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "formal": bool(args.formal),
            "load": args.load,
            "config": args.config,
            "seeds": seeds,
            "ticks": int(args.ticks),
            "top_m": int(args.top_m),
            "top_m_is_online_limit": False,
            "online_robot_candidate_scope": "all_idle",
            "paired_arms": [BASELINE_LABEL, WORK_LABEL],
            "normal_arrivals": True,
            "paired_order_arrivals": (
                "baseline-generated manifest replayed exactly in WorkDrift arm"
            ),
            "layer5_protocol_sha256": protocol["protocol_sha256"],
            "work_drift_lambda": work_lambda,
            "frozen_bundle": str(bundle_path),
            "frozen_bundle_sha256": sha256_file(bundle_path),
            "streams_dir": str(streams_dir),
            "order_manifest_dir": str(order_manifest_dir),
        },
        "per_seed": {str(seed): values for seed, values in per_seed.items()},
        "aggregate": aggregate_seeds(per_seed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nsaved paired Layer-5 result: {output}")


if __name__ == "__main__":
    main()
