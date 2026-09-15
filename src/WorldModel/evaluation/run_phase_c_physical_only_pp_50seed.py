"""Fixed-protocol 50-seed PP dispatch factorial on Windows.

This runner is intentionally isolated from the existing scale-generalisation
campaign.  It evaluates one nominal 20x20/48-robot/4-station configuration
under ``physical_only_legacy`` admission and replays one Greedy-generated
arrival manifest across five assignment arms:

    Greedy, Hungarian, JSQ, Phase C, Combo S1+J1

The default seed set is exactly 900--949 (50 seeds; the requested 900--950
range interpreted as a 50-seed interval), with low/mid/high loads
and 1500 ticks per run.  A worker executes all five arms for one
``(load, seed)`` pair, so the manifest is generated once and reused exactly.
Workers write independent log files; no stdout PIPE is used, avoiding the
Windows pipe deadlock observed in the earlier JSQ campaign.

Typical Windows usage (rmfs conda environment)::

    python -m WorldModel.evaluation.run_phase_c_physical_only_pp_50seed \
        --workers 8 --worker-threads 1 --worker-interop-threads 1

The runner never selects seeds based on outcomes.  Existing successful job
files are resumed, and the final summary contains every completed seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import runpy
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping


BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "phasec_physical_only_pp_factorial_900_950_v1"
CONFIGS = {
    "low": Path("Config/world_model_config_PP_48_low.json"),
    "mid": Path("Config/world_model_config_PP_48_mid.json"),
    "high": Path("Config/world_model_config_PP_48_high.json"),
}
LOADS = tuple(CONFIGS)
# 900--950 is a 51-number interval if interpreted literally.  The protocol
# uses 50 seeds, therefore the inclusive seed set is 900--949.
SEEDS = tuple(range(900, 950))
TICKS = 1500
VARIANT = "map20_r48_s4"
ARMS = ("Greedy", "Hungarian", "JSQ", "PhaseC", "ComboS1J1")
STATION_ADMISSION = "physical_only_legacy"
PROTOCOL_SCHEMA = "phase_c_physical_only_pp_50seed_protocol_v1"
RESULT_SCHEMA = "phase_c_physical_only_pp_50seed_job_v1"
SUMMARY_SCHEMA = "phase_c_physical_only_pp_50seed_summary_v1"

_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
_WORKER_THREADS_ENV = "RMFS_PP50_WORKER_THREADS"
_WORKER_INTEROP_ENV = "RMFS_PP50_WORKER_INTEROP_THREADS"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite changed output: {path}")
        return
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _configure_worker_threads(intra_op: int, inter_op: int) -> None:
    intra_op = max(1, int(intra_op))
    inter_op = max(1, int(inter_op))
    for name in _THREAD_ENV_NAMES:
        os.environ[name] = str(intra_op)
    os.environ[_WORKER_THREADS_ENV] = str(intra_op)
    os.environ[_WORKER_INTEROP_ENV] = str(inter_op)
    try:
        import torch

        torch.set_num_threads(intra_op)
        torch.set_num_interop_threads(inter_op)
    except (ImportError, RuntimeError):
        # The bootstrap normally sets these before importing WorldModel.
        pass


def _add_multiobjective_diagnostics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add transparent, non-primary diagnostics for trade-off analysis.

    These quantities are derived from the simulator's already reported
    metrics.  No hand-tuned weighted score is introduced: throughput,
    latency, and stability remain separate dimensions in the final Pareto
    analysis.
    """
    arrivals = max(int(metrics.get("order_arrival_count", 0)), 1)
    completed = max(float(metrics.get("completed_orders", 0.0)), 0.0)
    open_orders = max(float(metrics.get("open_order_count", 0.0)), 0.0)
    pending_orders = max(float(metrics.get("pending_order_count", 0.0)), 0.0)
    backlog = open_orders + pending_orders
    congestion = max(float(metrics.get("congestion_events", 0.0)), 1.0)
    severe = max(float(metrics.get("severe_events", 0.0)), 1.0)

    # Completion/backlog ratios are scale-free and useful when arrival counts
    # vary slightly across seeds.  Event-normalised throughput is diagnostic
    # only; it must not replace completed_orders as the primary endpoint.
    metrics["completion_fraction"] = round(completed / arrivals, 6)
    metrics["final_backlog_total"] = round(backlog, 6)
    metrics["final_backlog_fraction"] = round(backlog / arrivals, 6)
    metrics["completed_orders_per_congestion_event"] = round(
        completed / congestion, 6
    )
    metrics["completed_orders_per_severe_event"] = round(
        completed / severe, 6
    )
    return metrics


def _protocol_payload(root: Path) -> dict[str, Any]:
    from WorldModel.evaluation.phase_c_long_risk_head_repair_protocol import (
        REBOUND_PSI_HEAD,
        REPAIRED_CHECKPOINT,
        PSI_SCALE_CONTRACT,
    )

    source_files = {
        "runner": Path(__file__),
        "phasec_protocol": Path(
            "WorldModel/evaluation/phase_c_s1_hungarian_protocol.py"
        ),
        "combo_protocol": Path(
            "WorldModel/evaluation/phase_c_long_risk_head_repair_protocol.py"
        ),
        "scale_runner_helpers": Path(
            "WorldModel/evaluation/run_phase_c_scale_generalization.py"
        ),
        "evaluate_online": Path("WorldModel/evaluation/evaluate_online_v6.py"),
        "greedy_assigner": Path(
            "Policies/TaskAssigner/GreedyTaskAssigner/greedy_task_assigner.py"
        ),
        "hungarian_assigner": Path(
            "Policies/TaskAssigner/HungarianTaskAssigner/hungarian_task_assigner.py"
        ),
        "jsq_assigner": Path(
            "Policies/TaskAssigner/JSQTaskAssigner/jsq_task_assigner.py"
        ),
        "world_model_assigner": Path(
            "Policies/TaskAssigner/WorldModelTaskAssigner/world_model_task_assigner.py"
        ),
        "psi_assigner": Path(
            "Policies/TaskAssigner/WorldModelTaskAssigner/psi_dispatch_context_assigner.py"
        ),
        "physical_audit_helper": Path(
            "WorldModel/evaluation/run_phase_c_s1_j1_long_risk_correction.py"
        ),
    }
    missing = [str(path) for path in source_files.values() if not path.is_file()]
    required = [Path(REPAIRED_CHECKPOINT), Path(REBOUND_PSI_HEAD), Path(PSI_SCALE_CONTRACT)]
    missing.extend(str(path) for path in required if not path.is_file())
    missing.extend(str(path) for path in CONFIGS.values() if not path.is_file())
    if missing:
        raise FileNotFoundError("missing fixed-protocol input(s): " + ", ".join(missing))

    payload = {
        "schema_version": PROTOCOL_SCHEMA,
        "protocol": {
            "variant": VARIANT,
            "map_rows": 20,
            "map_cols": 20,
            "num_robots": 48,
            "num_stations": 4,
            "planner": "PP",
            "station_admission": STATION_ADMISSION,
            "loads": list(LOADS),
            "seeds": list(SEEDS),
            "ticks": TICKS,
            "arms": list(ARMS),
            "manifest_source": "GreedyTaskAssigner",
            "exact_manifest_replay": True,
            "checkpoint": {
                "path": Path(REPAIRED_CHECKPOINT).as_posix(),
                "sha256": _sha256_file(Path(REPAIRED_CHECKPOINT)),
            },
            "psi_head": {
                "path": Path(REBOUND_PSI_HEAD).as_posix(),
                "sha256": _sha256_file(Path(REBOUND_PSI_HEAD)),
            },
            "psi_scale_contract": {
                "path": Path(PSI_SCALE_CONTRACT).as_posix(),
                "sha256": _sha256_file(Path(PSI_SCALE_CONTRACT)),
            },
            "configs": {
                load: {
                    "path": path.as_posix(),
                    "sha256": _sha256_file(path),
                }
                for load, path in CONFIGS.items()
            },
            "source_hashes": {
                name: {"path": path.as_posix(), "sha256": _sha256_file(path)}
                for name, path in source_files.items()
            },
        },
    }
    payload["protocol_sha256"] = _canonical_sha(payload["protocol"])
    return payload


def _freeze_protocol(root: Path) -> dict[str, Any]:
    path = root / "physical_only_pp_50seed_protocol.json"
    payload = _protocol_payload(root)
    if path.is_file():
        existing = _read_json(path)
        if existing != payload:
            raise FileExistsError(f"existing protocol differs: {path}")
    else:
        _atomic_json(path, payload)
    return payload


def _manifest_path(root: Path, load: str, seed: int) -> Path:
    return root / "manifests" / f"greedy_{load}_seed{seed}_orders.json"


def _job_path(root: Path, load: str, seed: int) -> Path:
    return root / "per_seed" / f"{load}_seed{seed}.json"


def _make_assigner(arm: str, protocol: Mapping[str, Any]):
    from Policies.TaskAssigner import (
        GreedyTaskAssigner,
        HungarianTaskAssigner,
        WorldModelTaskAssigner,
    )
    from Policies.TaskAssigner.JSQTaskAssigner import JSQTaskAssigner
    from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_context_assigner import (
        PsiDispatchContextWorldModelTaskAssigner,
    )
    from WorldModel.evaluation.phase_c_long_risk_head_repair_protocol import (
        combo_selector_config,
    )
    from WorldModel.evaluation.phase_c_s1_hungarian_protocol import PHASEC_CONFIG

    checkpoint = protocol["checkpoint"]["path"]
    if arm == "Greedy":
        return GreedyTaskAssigner()
    if arm == "Hungarian":
        return HungarianTaskAssigner()
    if arm == "JSQ":
        return JSQTaskAssigner()
    if arm == "PhaseC":
        return WorldModelTaskAssigner(
            checkpoint_path=checkpoint,
            top_m=10,
            **dict(PHASEC_CONFIG),
        )
    if arm == "ComboS1J1":
        static = protocol
        return PsiDispatchContextWorldModelTaskAssigner(
            psi_head_checkpoint=static["psi_head"]["path"],
            psi_scale_contract=static["psi_scale_contract"]["path"],
            psi_context_mode="j_ascending",
            psi_trace_enabled=False,
            psi_trace_max_records=0,
            allow_phasec_s0_robot_scorer=False,
            checkpoint_path=checkpoint,
            top_m=10,
            energy_conv_random_flip_seed=0,
            **dict(combo_selector_config("combo")),
        )
    raise ValueError(f"unknown arm: {arm}")


def _validate_manifest(path: Path) -> dict[str, Any]:
    from WorldModel.evaluation.run_phase_c_s1_fifo_pair import _validate_manifest as validate

    return validate(path)


def _run_arm(
    config_path: Path,
    arm: str,
    seed: int,
    ticks: int,
    manifest_path: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    from WorldModel.evaluation.run_phase_c_scale_generalization import _run_simulation

    assigner = _make_assigner(arm, protocol)
    metrics, station_audit = _run_simulation(
        str(config_path),
        assigner,
        seed,
        ticks,
        f"PhysicalOnlyPP50/{VARIANT}/{arm}/seed{seed}",
        recorded_orders_path=str(manifest_path),
    )
    metrics = _add_multiobjective_diagnostics(metrics)
    manifest = _validate_manifest(manifest_path)
    checks = {
        "manifest_count_matches": int(metrics.get("order_arrival_count", -1))
        == int(manifest.get("total_orders", -2)),
        "manifest_hash_matches": metrics.get("order_arrival_manifest_sha256")
        == manifest.get("manifest_sha256"),
        "ticks_match": int(metrics.get("ticks", -1)) == ticks,
        "robot_count_matches": int(metrics.get("num_agents", -1)) == 48,
        "physical_only_mode": station_audit.get("mode") == STATION_ADMISSION,
        "station_audit_passed": bool(station_audit.get("passed")),
        "physical_capacity_never_exceeded": int(
            station_audit.get("physical_capacity_violation_count", -1)
        )
        == 0,
        "no_committed_cap_contract": station_audit.get(
            "committed_capacity_contract"
        )
        == "not enforced for in-transit commitments",
        "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
    }
    if arm in ("PhaseC", "ComboS1J1"):
        checks["world_model_used"] = int(metrics.get("model_assign_calls", 0)) > 0
    if arm == "ComboS1J1":
        checks.update(
            {
                "combo_energy_mode": metrics.get("energy_scoring_mode") == "conversion",
                "combo_signal": metrics.get("energy_drift_signal") == "combo",
                "j1_mode": metrics.get("psi_dispatch_mode") == "j_ascending",
                "j1_head_loaded": bool(metrics.get("psi_dispatch_head_loaded")),
                "j1_evaluated": int(metrics.get("psi_dispatch_eval_calls", 0)) > 0,
                "j1_encoder_binding": bool(
                    metrics.get("psi_dispatch_encoder_contract_verified")
                ),
                "j1_s1_variant": metrics.get("psi_dispatch_robot_scorer_variant")
                == "s1_within_context",
            }
        )
    if not all(checks.values()):
        failed = ", ".join(key for key, value in checks.items() if not value)
        raise RuntimeError(f"{arm} audit failed seed={seed}: {failed}")
    return {
        "arm": arm,
        "metrics": metrics,
        "station_audit": station_audit,
        "audit": {"passed": True, "checks": checks},
    }


def _worker_main(args: argparse.Namespace) -> None:
    _configure_worker_threads(args.worker_threads, args.worker_interop_threads)
    root = Path(args.root)
    load = args.load
    seed = int(args.seed)
    ticks = int(args.ticks)
    protocol_bundle = _read_json(root / "physical_only_pp_50seed_protocol.json")
    protocol = protocol_bundle["protocol"]
    config_path = Path(protocol["configs"][load]["path"])
    manifest_path = _manifest_path(root, load, seed)
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.is_file():
        existing = _read_json(result_path)
        if existing.get("protocol_sha256") == protocol_bundle.get("protocol_sha256"):
            print(f"[resume] {load} seed={seed}")
            return
        raise FileExistsError(f"incompatible existing job result: {result_path}")

    manifest_exists = manifest_path.is_file()
    if manifest_exists:
        _validate_manifest(manifest_path)

    # Generate the shared order stream exactly once with Greedy.  If an
    # interrupted run left a valid manifest, replay it to recover Greedy's
    # metrics without overwriting the manifest.
    from WorldModel.evaluation.run_phase_c_scale_generalization import _run_simulation
    from Policies.TaskAssigner import GreedyTaskAssigner

    greedy_kwargs = (
        {"recorded_orders_path": str(manifest_path)}
        if manifest_exists
        else {"save_order_manifest": str(manifest_path)}
    )
    greedy_metrics, greedy_audit = _run_simulation(
        str(config_path),
        GreedyTaskAssigner(),
        seed,
        ticks,
        f"PhysicalOnlyPP50/{VARIANT}/Greedy/seed{seed}",
        **greedy_kwargs,
    )
    greedy_metrics = _add_multiobjective_diagnostics(greedy_metrics)
    manifest = _validate_manifest(manifest_path)
    greedy_checks = {
        "manifest_count_matches": int(greedy_metrics.get("order_arrival_count", -1))
        == int(manifest.get("total_orders", -2)),
        "manifest_hash_matches": greedy_metrics.get("order_arrival_manifest_sha256")
        == manifest.get("manifest_sha256"),
        "physical_only_mode": greedy_audit.get("mode") == STATION_ADMISSION,
        "station_audit_passed": bool(greedy_audit.get("passed")),
        "physical_capacity_never_exceeded": int(
            greedy_audit.get("physical_capacity_violation_count", -1)
        )
        == 0,
        "no_committed_cap_contract": greedy_audit.get(
            "committed_capacity_contract"
        )
        == "not enforced for in-transit commitments",
    }
    if not all(greedy_checks.values()):
        failed = ", ".join(key for key, value in greedy_checks.items() if not value)
        raise RuntimeError(f"Greedy manifest audit failed seed={seed}: {failed}")

    arms: dict[str, dict[str, Any]] = {
        "Greedy": {
            "arm": "Greedy",
            "metrics": greedy_metrics,
            "station_audit": greedy_audit,
            "audit": {"passed": True, "checks": greedy_checks},
        }
    }
    for arm in ("Hungarian", "JSQ", "PhaseC", "ComboS1J1"):
        arms[arm] = _run_arm(
            config_path, arm, seed, ticks, manifest_path, protocol
        )

    payload = {
        "schema_version": RESULT_SCHEMA,
        "protocol_sha256": protocol_bundle["protocol_sha256"],
        "variant": VARIANT,
        "load": load,
        "seed": seed,
        "ticks": ticks,
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": _sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "arms": arms,
    }
    _atomic_json(result_path, payload)
    summary = " ".join(
        f"{arm}={arms[arm]['metrics'].get('completed_orders', '?')}"
        for arm in ARMS
    )
    print(f"[done] {load} seed={seed} {summary}")


def _bootstrap_ci(values: list[float], seed: int = 20260905, trials: int = 10000):
    if not values:
        return None
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(trials)
    )
    return [
        round(means[int(math.floor(0.025 * (trials - 1)))], 6),
        round(means[int(math.ceil(0.975 * (trials - 1)))], 6),
    ]


_PARETO_DIRECTIONS = {
    "completed_orders": "higher",
    "avg_excess_delay": "lower",
    "deadlock_ratio_mean": "lower",
    "congestion_events": "lower",
}


def _pareto_analysis(aggregate: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return the non-dominated arms on the mean trade-off surface.

    This is deliberately reported as a Pareto set rather than collapsing the
    objectives into an arbitrary scalar.  An arm is dominated only when
    another arm is no worse on every listed dimension and strictly better on
    at least one.
    """
    arms = [arm for arm in ARMS if arm in aggregate]

    def value(arm: str, metric: str) -> float | None:
        raw = aggregate.get(arm, {}).get(f"{metric}_mean")
        return float(raw) if isinstance(raw, (int, float)) else None

    def dominates(left: str, right: str) -> bool:
        strict = False
        for metric, direction in _PARETO_DIRECTIONS.items():
            lv = value(left, metric)
            rv = value(right, metric)
            if lv is None or rv is None:
                return False
            if direction == "higher":
                if lv < rv:
                    return False
                strict = strict or lv > rv
            else:
                if lv > rv:
                    return False
                strict = strict or lv < rv
        return strict

    dominated_by: dict[str, list[str]] = {arm: [] for arm in arms}
    for arm in arms:
        dominated_by[arm] = [
            other for other in arms if other != arm and dominates(other, arm)
        ]
    front = [arm for arm in arms if not dominated_by[arm]]
    return {
        "dimensions": dict(_PARETO_DIRECTIONS),
        "pareto_front_arms": front,
        "dominated_by": dominated_by,
    }


def _aggregate_summary(root: Path, protocol_bundle: Mapping[str, Any]) -> dict[str, Any]:
    from WorldModel.evaluation.evaluate_online_v6 import aggregate_seeds

    rows: dict[str, dict[int, dict[str, dict[str, Any]]]] = {
        load: {} for load in LOADS
    }
    missing: list[str] = []
    for load in LOADS:
        for seed in SEEDS:
            path = _job_path(root, load, seed)
            if not path.is_file():
                missing.append(f"{load}/seed{seed}")
                continue
            payload = _read_json(path)
            if payload.get("protocol_sha256") != protocol_bundle["protocol_sha256"]:
                raise ValueError(f"protocol mismatch in {path}")
            rows[load][seed] = {
                arm: data["metrics"]
                for arm, data in payload["arms"].items()
            }
    aggregate: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    focus = {
        "completed_orders": "higher",
        "completed_tasks": "higher",
        "avg_task_duration": "lower",
        "avg_excess_delay": "lower",
        "open_order_count": "lower",
        "pending_order_count": "lower",
        "congestion_events": "lower",
        "severe_events": "lower",
        "stall_ratio_mean": "lower",
        "deadlock_ratio_mean": "lower",
        "risk_rate_per_100": "lower",
        "completion_fraction": "higher",
        "final_backlog_total": "lower",
        "final_backlog_fraction": "lower",
        "completed_orders_per_congestion_event": "higher",
        "completed_orders_per_severe_event": "higher",
    }
    for load in LOADS:
        aggregate[load] = aggregate_seeds(rows[load])
        comparisons[load] = {}
        for baseline in ("Greedy", "JSQ"):
            comparisons[load][baseline] = {}
            for arm in ("PhaseC", "ComboS1J1", "Hungarian", "JSQ"):
                if arm == baseline:
                    continue
                per_metric: dict[str, Any] = {}
                for metric, direction in focus.items():
                    deltas = []
                    wins = losses = ties = 0
                    for seed in sorted(rows[load]):
                        b = rows[load][seed].get(baseline, {}).get(metric)
                        a = rows[load][seed].get(arm, {}).get(metric)
                        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                            continue
                        raw = float(a) - float(b)
                        favourable = raw if direction == "higher" else -raw
                        deltas.append(raw)
                        if favourable > 0:
                            wins += 1
                        elif favourable < 0:
                            losses += 1
                        else:
                            ties += 1
                    if deltas:
                        per_metric[metric] = {
                            "direction": direction,
                            "paired_seeds": len(deltas),
                            "arm_minus_baseline_mean": round(sum(deltas) / len(deltas), 6),
                            "arm_minus_baseline_ci95": _bootstrap_ci(deltas),
                            "wins": wins,
                            "losses": losses,
                            "ties": ties,
                        }
                comparisons[load][baseline][arm] = per_metric
    report = {
        "schema_version": SUMMARY_SCHEMA,
        "protocol_sha256": protocol_bundle["protocol_sha256"],
        "expected_jobs": len(LOADS) * len(SEEDS),
        "completed_jobs": sum(len(value) for value in rows.values()),
        "missing_jobs": missing,
        "variant": VARIANT,
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "ticks": TICKS,
        "arms": list(ARMS),
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "multiobjective_analysis": {
            load: _pareto_analysis(aggregate[load]) for load in LOADS
        },
    }
    _atomic_json(root / "summary.json", report)
    return report


def _tail(path: Path, chars: int = 5000) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return content if len(content) <= chars else "...[truncated]...\n" + content[-chars:]


def _terminate(active: dict[tuple[str, int], tuple[subprocess.Popen, Any, Path]]) -> None:
    for proc, _handle, _log in active.values():
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
    for proc, _handle, _log in active.values():
        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass


def _run_coordinator(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    protocol_bundle = _freeze_protocol(root)
    logs = root / "worker_logs"
    logs.mkdir(parents=True, exist_ok=True)
    root.joinpath("manifests").mkdir(parents=True, exist_ok=True)
    jobs = [(load, seed) for load in LOADS for seed in SEEDS]
    pending = [job for job in jobs if not _job_path(root, *job).is_file()]
    print(
        f"Fixed physical-only PP factorial: {len(LOADS)} loads x {len(SEEDS)} seeds "
        f"x {len(ARMS)} arms = {len(LOADS) * len(SEEDS) * len(ARMS)} simulations"
    )
    print(f"output: {root}")
    print(f"workers: {args.workers}; worker threads: {args.worker_threads}/{args.worker_interop_threads}")
    if not pending:
        print("all job files already exist; rebuilding summary")
        report = _aggregate_summary(root, protocol_bundle)
        print(json.dumps({"completed_jobs": report["completed_jobs"], "expected_jobs": report["expected_jobs"]}, indent=2))
        return

    active: dict[tuple[str, int], tuple[subprocess.Popen, Any, Path]] = {}
    completed = len(jobs) - len(pending)
    python = sys.executable
    bootstrap = (
        "import os,runpy,torch;"
        "torch.set_num_threads(int(os.environ.get('RMFS_PP50_WORKER_THREADS','1')));"
        "torch.set_num_interop_threads(int(os.environ.get('RMFS_PP50_WORKER_INTEROP_THREADS','1')));"
        "runpy.run_module('WorldModel.evaluation.run_phase_c_physical_only_pp_50seed', run_name='__main__')"
    )
    try:
        while pending or active:
            while pending and len(active) < max(1, int(args.workers)):
                load, seed = pending.pop(0)
                result = _job_path(root, load, seed)
                log_path = logs / f"{load}_seed{seed}.log"
                log_handle = log_path.open("w", encoding="utf-8", buffering=1)
                env = os.environ.copy()
                for name in _THREAD_ENV_NAMES:
                    env[name] = str(args.worker_threads)
                env[_WORKER_THREADS_ENV] = str(args.worker_threads)
                env[_WORKER_INTEROP_ENV] = str(args.worker_interop_threads)
                cmd = [
                    python,
                    "-c",
                    bootstrap,
                    "--_worker",
                    "--root",
                    str(root),
                    "--load",
                    load,
                    "--seed",
                    str(seed),
                    "--ticks",
                    str(TICKS),
                    "--result",
                    str(result),
                    "--worker-threads",
                    str(args.worker_threads),
                    "--worker-interop-threads",
                    str(args.worker_interop_threads),
                ]
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                active[(load, seed)] = (proc, log_handle, log_path)

            finished: list[tuple[str, int]] = []
            for key, (proc, _handle, log_path) in active.items():
                ret = proc.poll()
                if ret is None:
                    continue
                load, seed = key
                if ret != 0:
                    raise RuntimeError(
                        f"worker failed {load} seed={seed} code={ret}; "
                        f"log={log_path}\n{_tail(log_path)}"
                    )
                result = _job_path(root, load, seed)
                if not result.is_file():
                    raise RuntimeError(
                        f"worker exited without result {load} seed={seed}; "
                        f"log={log_path}\n{_tail(log_path)}"
                    )
                completed += 1
                print(f"[{completed}/{len(jobs)}] completed {load} seed={seed}")
                finished.append(key)
            for key in finished:
                proc, handle, _log = active.pop(key)
                handle.close()
            if active and not finished:
                time.sleep(0.5)
    finally:
        _terminate(active)
        for _proc, handle, _log in active.values():
            handle.close()

    report = _aggregate_summary(root, protocol_bundle)
    print(json.dumps({"completed_jobs": report["completed_jobs"], "expected_jobs": report["expected_jobs"], "summary": str(root / 'summary.json')}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--worker-interop-threads", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--load", choices=LOADS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--ticks", type=int, default=TICKS, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        if args.root is None or args.load is None or args.seed is None or args.result is None:
            raise SystemExit("worker arguments are incomplete")
        _worker_main(args)
        return
    if args.summarize:
        root = Path(args.output_root)
        bundle = _read_json(root / "physical_only_pp_50seed_protocol.json")
        report = _aggregate_summary(root, bundle)
        print(json.dumps({"completed_jobs": report["completed_jobs"], "expected_jobs": report["expected_jobs"], "missing_jobs": report["missing_jobs"]}, indent=2))
        return
    _run_coordinator(args)


if __name__ == "__main__":
    main()
