"""Collect pure-World-Model on-policy decision snapshots for Phase C round 1.

This command does not label candidates and does not train a model.  It runs
the frozen pure World Model under normal arrivals and records observation-only
decision snapshots.  Each snapshot contains a stratified robot subset plus a
native NO_ASSIGN counterfactual candidate for later isolated replay.

External assignment baselines (Greedy/Hungarian) are deliberately absent from
the training-data path.  They remain sealed evaluation-only baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from Policies.TaskAssigner import WorldModelTaskAssigner
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner


SCHEMA_VERSION = "phase_c_round1_snapshot_collection_v1"
PROTOCOL_SCHEMA_VERSION = "phase_c_round1_protocol_v1"
FORMAL_SEEDS = tuple(range(461, 471))
FINAL_CERTIFICATION_SEEDS = tuple(range(471, 481))
FORMAL_TICKS = 1500
FORMAL_SNAPSHOT_INTERVAL = 20
FORMAL_SNAPSHOT_TOP_M = 10
FORMAL_MAX_CONTEXTS_PER_TICK = 2
FORMAL_ROLLOUT_HORIZON = 10
LOAD_CONFIGS = {
    "low": "Config/world_model_config_PP_48_low.json",
    "mid": "Config/world_model_config_PP_48_mid.json",
    "high": "Config/world_model_config_PP_48_high.json",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: dict) -> str:
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
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _validate_formal_args(args, seeds: list[int]) -> None:
    if tuple(seeds) != FORMAL_SEEDS:
        raise SystemExit("formal Phase C round 1 freezes seeds 461--470")
    if int(args.ticks) != FORMAL_TICKS:
        raise SystemExit("formal Phase C round 1 freezes --ticks=1500")
    if int(args.snapshot_interval) != FORMAL_SNAPSHOT_INTERVAL:
        raise SystemExit(
            "formal Phase C round 1 freezes --snapshot-interval=20"
        )
    if int(args.snapshot_top_m) != FORMAL_SNAPSHOT_TOP_M:
        raise SystemExit(
            "formal Phase C round 1 freezes --snapshot-top-m=10"
        )
    if int(args.max_contexts_per_tick) != FORMAL_MAX_CONTEXTS_PER_TICK:
        raise SystemExit(
            "formal Phase C round 1 freezes --max-contexts-per-tick=2"
        )
    if Path(args.config).as_posix() != Path(
        LOAD_CONFIGS[args.load]
    ).as_posix():
        raise SystemExit(
            f"formal {args.load} collection requires "
            f"{LOAD_CONFIGS[args.load]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--load", choices=tuple(LOAD_CONFIGS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--ticks", type=int, default=FORMAL_TICKS)
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=FORMAL_SNAPSHOT_INTERVAL,
    )
    parser.add_argument(
        "--snapshot-top-m", type=int, default=FORMAL_SNAPSHOT_TOP_M
    )
    parser.add_argument(
        "--max-contexts-per-tick",
        type=int,
        default=FORMAL_MAX_CONTEXTS_PER_TICK,
    )
    parser.add_argument(
        "--candidate-robot-mode",
        choices=("stratified", "eta_stratified", "nearest"),
        default="stratified",
    )
    parser.add_argument("--lyapunov-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--path-planner-override", default=None)
    parser.add_argument("--path-planner-params-json", default="{}")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    try:
        path_planner_params = json.loads(args.path_planner_params_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"invalid --path-planner-params-json: {exc}"
        ) from exc
    if not isinstance(path_planner_params, dict):
        raise SystemExit("--path-planner-params-json must decode to an object")
    if path_planner_params and not args.path_planner_override:
        raise SystemExit(
            "--path-planner-params-json requires --path-planner-override"
        )

    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) != len(set(seeds)):
        raise SystemExit("duplicate seeds are not allowed")
    if any(value <= 0 for value in (
        args.ticks,
        args.snapshot_interval,
        args.snapshot_top_m,
        args.max_contexts_per_tick,
    )):
        raise SystemExit("ticks and snapshot limits must be positive")
    if args.formal:
        _validate_formal_args(args, seeds)

    checkpoint = Path(args.checkpoint)
    config = Path(args.config)
    lyapunov_config_path = Path(args.lyapunov_config)
    for path in (checkpoint, config, lyapunov_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    lyapunov_config = _read_json(lyapunov_config_path)

    repo_root = Path(__file__).resolve().parents[2]
    probe_source = repo_root / "WorldModel/evaluation/decision_snapshot_probe.py"
    assigner_source = repo_root / (
        "Policies/TaskAssigner/WorldModelTaskAssigner/"
        "world_model_task_assigner.py"
    )
    runner_source = Path(__file__).resolve()

    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "formal": bool(args.formal),
        "round": "phase_c_round1",
        "objective": "repair_world_model_closed_loop_distribution_shift",
        "load": args.load,
        "config": config.as_posix(),
        "path_planner_override": args.path_planner_override,
        "path_planner_params_override": path_planner_params,
        "seeds": seeds,
        "ticks": int(args.ticks),
        "behavior_policy": "pure_world_model",
        "behavior_checkpoint": checkpoint.as_posix(),
        "behavior_checkpoint_sha256": _sha256_file(checkpoint),
        "normal_arrivals": True,
        "training_source_policy": "world_model_on_policy",
        "external_baseline_training_samples": False,
        "hungarian_training_samples": False,
        "greedy_training_samples": False,
        "td_target_enabled": False,
        "td_value_head_enabled": False,
        "td_risk_v_head_enabled": False,
        "future_unknown_orders_in_candidate_rollout": False,
        "continuation_scheduler_in_candidate_rollout": False,
        "candidate_label_continuation_mode": "isolated",
        "candidate_actions": "stratified_idle_robots_plus_native_no_assign",
        "snapshot_interval": int(args.snapshot_interval),
        "snapshot_top_m_robot_candidates": int(args.snapshot_top_m),
        "snapshot_top_m_is_online_limit": False,
        "max_contexts_per_tick": int(args.max_contexts_per_tick),
        "candidate_robot_mode": args.candidate_robot_mode,
        "future_label_horizon": FORMAL_ROLLOUT_HORIZON,
        "phase_c_data_seeds": list(FORMAL_SEEDS),
        "reserved_final_certification_seeds": list(
            FINAL_CERTIFICATION_SEEDS
        ),
        "lyapunov_config": lyapunov_config_path.as_posix(),
        "lyapunov_config_sha256": _sha256_file(lyapunov_config_path),
        "runner_source_sha256": _sha256_file(runner_source),
        "probe_source_sha256": _sha256_file(probe_source),
        "assigner_source_sha256": _sha256_file(assigner_source),
    }
    protocol["protocol_sha256"] = _canonical_sha256(protocol)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    load_dir = output_dir / args.load
    load_dir.mkdir(parents=True, exist_ok=True)

    per_seed = {}
    for seed in seeds:
        run_id = f"phaseC_r1_{args.load}_seed{seed}"
        run_dir = load_dir / f"seed{seed}"
        snapshot_dir = run_dir / "snapshots"
        manifest_path = run_dir / "order_manifest.json"
        metrics_path = run_dir / "metrics.json"

        if metrics_path.is_file():
            existing = _read_json(metrics_path)
            if (
                existing.get("protocol_sha256")
                != protocol["protocol_sha256"]
                or int(existing.get("seed", -1)) != seed
            ):
                raise RuntimeError(
                    f"refusing incompatible completed Phase C run: {run_dir}"
                )
            index_path = snapshot_dir / f"snapindex_{run_id}.json"
            if not index_path.is_file() or not manifest_path.is_file():
                raise RuntimeError(
                    f"completed metrics lack snapshot/manifest artifacts: {run_dir}"
                )
            print(f"[skip] completed {args.load} seed={seed}")
            per_seed[str(seed)] = existing["metrics"]
            continue

        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(
                "refusing ambiguous partial Phase C run; move it aside or "
                f"complete it explicitly: {run_dir}"
            )
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[start] Phase C round1 load={args.load} seed={seed}")
        assigner = WorldModelTaskAssigner(
            checkpoint_path=str(checkpoint),
            top_m=1,
            include_no_assign_candidate=False,
        )
        metrics = _run_one_assigner(
            str(config),
            assigner,
            seed,
            int(args.ticks),
            trace_label="PhaseC_WorldModel",
            decision_snapshot_dir=str(snapshot_dir),
            decision_snapshot_interval=int(args.snapshot_interval),
            decision_snapshot_top_m=int(args.snapshot_top_m),
            decision_snapshot_run_id=run_id,
            decision_snapshot_meta={
                "arm_label": "PhaseC_WorldModel",
                "load": args.load,
                "seed": seed,
                "config": config.as_posix(),
                "path_planner_override": args.path_planner_override,
                "path_planner_params_override": path_planner_params,
                "checkpoint_path": checkpoint.as_posix(),
                "phase_c_round": "phase_c_round1",
                "protocol_sha256": protocol["protocol_sha256"],
                "external_baseline_training_samples": False,
            },
            decision_snapshot_candidate_scope="top_m_snapshot",
            decision_snapshot_capture_policy="interval_all",
            decision_snapshot_candidate_robot_mode=(
                args.candidate_robot_mode
            ),
            decision_snapshot_include_no_assign=True,
            decision_snapshot_max_contexts_per_tick=int(
                args.max_contexts_per_tick
            ),
            decision_snapshot_phase_c_round="phase_c_round1",
            decision_snapshot_exclude_external_baselines=True,
            td_stream_lyapunov_l0_config=lyapunov_config,
            save_order_manifest=str(manifest_path),
            path_planner_override=args.path_planner_override,
            path_planner_params_override=path_planner_params,
        )
        if int(metrics.get("fallback_greedy_calls", 0)) != 0:
            raise RuntimeError("pure Phase C World Model used Greedy fallback")
        if int(metrics.get("decision_snapshots_saved", 0)) <= 0:
            raise RuntimeError("Phase C run produced no decision snapshots")

        run_record = {
            "schema_version": "phase_c_round1_run_v1",
            "protocol_sha256": protocol["protocol_sha256"],
            "load": args.load,
            "seed": seed,
            "run_id": run_id,
            "snapshot_index": str(
                snapshot_dir / f"snapindex_{run_id}.json"
            ),
            "order_manifest": str(manifest_path),
            "metrics": metrics,
        }
        metrics_path.write_text(
            json.dumps(run_record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        per_seed[str(seed)] = metrics
        print(
            f"[done] {args.load} seed={seed} snapshots="
            f"{metrics['decision_snapshots_saved']} orders="
            f"{metrics.get('completed_orders')}"
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "load": args.load,
        "per_seed": per_seed,
        "snapshot_count": sum(
            int(metrics.get("decision_snapshots_saved", 0))
            for metrics in per_seed.values()
        ),
    }
    report_path = output_dir / f"collection_{args.load}.json"
    if report_path.exists():
        existing = _read_json(report_path)
        if existing != result:
            raise RuntimeError(f"refusing to overwrite changed report: {report_path}")
    else:
        report_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"\nsaved Phase C collection report: {report_path}")
    print(f"protocol sha256: {protocol['protocol_sha256']}")
    print(f"snapshots: {result['snapshot_count']}")


if __name__ == "__main__":
    main()
