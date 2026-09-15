"""Collect source-Phase-C snapshots for LongRiskHead-only repair.

This is an isolated variant of the historical Phase-C collector.  It freezes
the source checkpoint and records only pure World-Model on-policy snapshots;
the seed range and output root are specific to the repair campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from Policies.TaskAssigner import WorldModelTaskAssigner
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
from WorldModel.evaluation.phase_c_long_risk_head_repair_protocol import (
    COLLECTION_SCHEMA_VERSION,
    LOAD_CONFIGS,
    LYAPUNOV_CONFIG,
    MAX_CONTEXTS_PER_TICK,
    ROLLOUT_HORIZON,
    SNAPSHOT_INTERVAL,
    SNAPSHOT_TOP_M,
    SOURCE_CHECKPOINT,
    TICKS,
    TRAIN_SEEDS,
    VAL_SEEDS,
    OFFLINE_TEST_SEEDS,
)
from WorldModel.evaluation.run_phase_c_s1_j1_long_risk_correction import (
    _physical_only_audit_contract,
)
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


PROTOCOL_SCHEMA_VERSION = "phase_c_long_risk_head_repair_collection_protocol_v1"
ALL_DATA_SEEDS = TRAIN_SEEDS + VAL_SEEDS + OFFLINE_TEST_SEEDS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _seed_role(seed: int) -> str:
    if seed in TRAIN_SEEDS:
        return "train"
    if seed in VAL_SEEDS:
        return "validation"
    if seed in OFFLINE_TEST_SEEDS:
        return "offline_test"
    raise ValueError(f"seed {seed} is outside the frozen repair data set")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(SOURCE_CHECKPOINT))
    parser.add_argument("--config", required=True)
    parser.add_argument("--load", choices=tuple(LOAD_CONFIGS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--snapshot-interval", type=int, default=SNAPSHOT_INTERVAL)
    parser.add_argument("--snapshot-top-m", type=int, default=SNAPSHOT_TOP_M)
    parser.add_argument(
        "--max-contexts-per-tick", type=int, default=MAX_CONTEXTS_PER_TICK
    )
    parser.add_argument("--candidate-robot-mode", default="stratified")
    parser.add_argument("--lyapunov-config", default=str(LYAPUNOV_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--path-planner-override", default=None)
    parser.add_argument("--path-planner-params-json", default="{}")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    config = Path(args.config)
    lyapunov_path = Path(args.lyapunov_config)
    for path in (checkpoint, config, lyapunov_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if config.as_posix() != Path(LOAD_CONFIGS[args.load]).as_posix():
        raise ValueError(
            f"repair collection for {args.load} requires {LOAD_CONFIGS[args.load]}"
        )
    seeds = [int(seed) for seed in args.seeds]
    if tuple(seeds) != ALL_DATA_SEEDS:
        raise ValueError(
            f"repair collection freezes seeds {list(ALL_DATA_SEEDS)} in order"
        )
    if int(args.ticks) != TICKS:
        raise ValueError(f"repair collection freezes ticks={TICKS}")
    if int(args.snapshot_interval) != SNAPSHOT_INTERVAL:
        raise ValueError("repair collection freezes snapshot interval=20")
    if int(args.snapshot_top_m) != SNAPSHOT_TOP_M:
        raise ValueError("repair collection freezes snapshot top_m=10")
    if int(args.max_contexts_per_tick) != MAX_CONTEXTS_PER_TICK:
        raise ValueError("repair collection freezes max contexts per tick=2")
    try:
        planner_params = json.loads(args.path_planner_params_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid planner params JSON: {exc}") from exc
    if not isinstance(planner_params, dict):
        raise ValueError("planner params must be a JSON object")
    if planner_params and not args.path_planner_override:
        raise ValueError("planner params require a planner override")

    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "collection_schema_version": COLLECTION_SCHEMA_VERSION,
        "purpose": "supervised W=200 LongRiskHead repair data",
        "load": args.load,
        "config": config.as_posix(),
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "seeds": seeds,
        "seed_roles": {str(seed): _seed_role(seed) for seed in seeds},
        "ticks": int(args.ticks),
        "snapshot_interval": int(args.snapshot_interval),
        "snapshot_top_m": int(args.snapshot_top_m),
        "max_contexts_per_tick": int(args.max_contexts_per_tick),
        "candidate_robot_mode": args.candidate_robot_mode,
        "behavior_policy": "source_phasec_s0j0",
        "training_source_policy": "world_model_on_policy",
        "external_baseline_training_samples": False,
        "td_target_enabled": False,
        "td_value_head_enabled": False,
        "td_risk_v_head_enabled": False,
        "candidate_label_continuation_mode": "isolated",
        "future_label_horizon": ROLLOUT_HORIZON,
        "path_planner_override": args.path_planner_override,
        "path_planner_params_override": planner_params,
        "lyapunov_config": lyapunov_path.as_posix(),
        "lyapunov_config_sha256": _sha256_file(lyapunov_path),
    }
    protocol["protocol_sha256"] = _canonical_sha256(protocol)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    load_dir = output_dir / args.load
    load_dir.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for seed in seeds:
        run_id = f"phaseC_longrisk_repair_{args.load}_seed{seed}"
        run_dir = load_dir / f"seed{seed}"
        snapshot_dir = run_dir / "snapshots"
        manifest_path = run_dir / "order_manifest.json"
        metrics_path = run_dir / "metrics.json"
        if metrics_path.is_file():
            existing = _read_json(metrics_path)
            if (
                existing.get("protocol_sha256") != protocol["protocol_sha256"]
                or int(existing.get("seed", -1)) != seed
            ):
                raise RuntimeError(f"incompatible completed run: {run_dir}")
            per_seed[str(seed)] = existing["metrics"]
            print(f"[resume] {args.load} seed={seed}")
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"ambiguous partial run: {run_dir}")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        assigner = WorldModelTaskAssigner(
            checkpoint_path=str(checkpoint),
            top_m=1,
            include_no_assign_candidate=False,
        )
        with _physical_only_audit_contract() as holder:
            metrics = _run_one_assigner(
                str(config),
                assigner,
                seed,
                int(args.ticks),
                trace_label="PhaseC_LongRiskRepair_SourceS0J0",
                decision_snapshot_dir=str(snapshot_dir),
                decision_snapshot_interval=int(args.snapshot_interval),
                decision_snapshot_top_m=int(args.snapshot_top_m),
                decision_snapshot_run_id=run_id,
                decision_snapshot_meta={
                    "arm_label": "PhaseC_LongRiskRepair_SourceS0J0",
                    "load": args.load,
                    "seed": seed,
                    "seed_role": _seed_role(seed),
                    "config": config.as_posix(),
                    "checkpoint_path": checkpoint.as_posix(),
                    "phase_c_round": "long_risk_head_repair",
                    "protocol_sha256": protocol["protocol_sha256"],
                    "training_source_policy": "world_model_on_policy",
                    "external_baseline_training_samples": False,
                    "td_target_enabled": False,
                    "td_value_head_enabled": False,
                    "td_risk_v_head_enabled": False,
                    "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
                    "path_planner_override": args.path_planner_override,
                    "path_planner_params_override": planner_params,
                },
                decision_snapshot_candidate_scope="top_m_snapshot",
                decision_snapshot_capture_policy="interval_all",
                decision_snapshot_candidate_robot_mode=args.candidate_robot_mode,
                decision_snapshot_include_no_assign=True,
                decision_snapshot_max_contexts_per_tick=int(
                    args.max_contexts_per_tick
                ),
                decision_snapshot_phase_c_round="long_risk_head_repair",
                decision_snapshot_exclude_external_baselines=True,
                td_stream_lyapunov_l0_config=_read_json(lyapunov_path),
                save_order_manifest=str(manifest_path),
                path_planner_override=args.path_planner_override,
                path_planner_params_override=planner_params,
            )
        station_probe = holder.get("station_probe")
        station_audit = station_probe.summary() if station_probe is not None else {}
        if not station_audit.get("passed"):
            raise RuntimeError("physical-only station audit failed during collection")
        if int(metrics.get("fallback_greedy_calls", 0)) != 0:
            raise RuntimeError("source Phase-C behavior used Greedy fallback")
        if int(metrics.get("decision_snapshots_saved", 0)) <= 0:
            raise RuntimeError("no decision snapshots were produced")
        run_record = {
            "schema_version": "phase_c_long_risk_repair_run_v1",
            "protocol_sha256": protocol["protocol_sha256"],
            "load": args.load,
            "seed": seed,
            "seed_role": _seed_role(seed),
            "run_id": run_id,
            "snapshot_index": str(
                snapshot_dir / f"snapindex_{run_id}.json"
            ),
            "order_manifest": str(manifest_path),
            "metrics": metrics,
            "station_audit": station_audit,
        }
        metrics_path.write_text(
            json.dumps(run_record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        per_seed[str(seed)] = metrics
        print(
            f"[done] {args.load} seed={seed} snapshots="
            f"{metrics['decision_snapshots_saved']}"
        )

    result = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "protocol": protocol,
        "load": args.load,
        "per_seed": per_seed,
        "snapshot_count": sum(
            int(metrics.get("decision_snapshots_saved", 0))
            for metrics in per_seed.values()
        ),
    }
    report_path = output_dir / f"collection_{args.load}.json"
    if report_path.is_file():
        if _read_json(report_path) != result:
            raise RuntimeError(f"refusing to overwrite changed report: {report_path}")
    else:
        report_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"[complete] repair collection report: {report_path}")


if __name__ == "__main__":
    main()
