"""Greedy vs JSQ paired comparison across three load levels and 10 seeds.

Usage
-----
    python -m WorldModel.evaluation.run_greedy_jsq_comparison ^
        --seeds 701 702 703 704 705 706 707 708 709 710 ^
        --ticks 1500 ^
        --loads low mid high ^
        --workers 8 ^
        --worker-threads 1 --worker-interop-threads 1 ^
        --output results/greedy_jsq_comparison.json

The script runs two stages per (load, seed) pair:

  1. **Manifest stage** -- run the Greedy assigner and save the realised order
     arrival manifest.
  2. **Arm stage** -- replay the *exact same* order stream into the JSQ
     assigner so throughput is compared under identical demand.

With ``--workers N`` (N > 1) pairs are dispatched as independent
``subprocess.Popen`` invocations (via ``--_worker`` mode) so up to N pairs
run concurrently.  Each worker is a fresh Python process with its own
memory space, avoiding the Windows ``spawn`` MemoryError that
``ProcessPoolExecutor`` would trigger when torch is imported.

Each worker's numerical-library and PyTorch thread counts are constrained by
``--worker-threads`` and ``--worker-interop-threads`` (both default to 1),
which is important on Windows hosts where PyTorch otherwise inherits large
machine-wide defaults.

Results are aggregated across seeds (mean +/- std) and a paired bootstrap
confidence interval (95 %) is computed for each focus metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, TextIO, Tuple

# ── constants ─────────────────────────────────────────────────────

SCHEMA_VERSION = "greedy_jsq_paired_comparison_v1"
GREEDY_LABEL = "Greedy"
JSQ_LABEL = "JSQ"
DEFAULT_SEEDS = tuple(range(701, 711))
DEFAULT_TICKS = 1500
DEFAULT_WORKER_THREADS = 1
DEFAULT_WORKER_INTEROP_THREADS = 1

# A worker is a complete simulation process.  Leaving the BLAS/OpenMP and
# PyTorch defaults in place makes every worker inherit the host-wide defaults
# (on the user's Windows machine these were 24 intra-op and 64 inter-op
# threads), so a 20-worker run creates hundreds of runnable threads.  These
# names are intentionally runner-specific; they are not used by the simulator
# itself and therefore cannot alter the experiment semantics.
_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
_WORKER_THREADS_ENV = "RMFS_JSQ_WORKER_THREADS"
_WORKER_INTEROP_ENV = "RMFS_JSQ_WORKER_INTEROP_THREADS"

# ``python -m WorldModel...`` imports ``WorldModel/__init__.py`` before the
# module's ``main`` function.  That package imports torch, so setting thread
# limits only inside ``_worker_main`` is too late to guarantee the inter-op
# limit.  The bootstrap imports torch and sets both limits before runpy loads
# the package.  The normal worker function repeats the idempotent check as a
# defensive fallback for direct/unit-test invocation.
_WORKER_BOOTSTRAP = (
    "import os,runpy,torch;"
    "torch.set_num_threads(int(os.environ.get('RMFS_JSQ_WORKER_THREADS','1')));"
    "torch.set_num_interop_threads(int(os.environ.get('RMFS_JSQ_WORKER_INTEROP_THREADS','1')));"
    "runpy.run_module('WorldModel.evaluation.run_greedy_jsq_comparison', run_name='__main__')"
)

LOAD_CONFIGS = {
    "low":  "Config/world_model_config_PP_48_low.json",
    "mid":  "Config/world_model_config_PP_48_mid.json",
    "high": "Config/world_model_config_PP_48_high.json",
}

# Direction: from the system objective's perspective.
# "higher" means larger is better; "lower" means smaller is better.
FOCUS_METRICS = {
    "completed_orders":              "higher",
    "completed_tasks":               "higher",
    "avg_task_duration":             "lower",
    "avg_excess_delay":              "lower",
    "open_order_count":              "lower",
    "pending_order_count":           "lower",
    "open_order_age_p95":            "lower",
    "completed_order_flow_time_p95": "lower",
}


def _configure_worker_threads(intra_op: int, inter_op: int) -> None:
    """Constrain numerical-library and PyTorch threads for one worker.

    This is deliberately local to this comparison runner.  It is applied in
    the child process before importing simulator/policy modules, and does not
    change the simulator's configuration or the assignment algorithms.
    """
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
        # RuntimeError is raised if a caller has already started a torch
        # parallel region.  The subprocess bootstrap normally sets this early;
        # retaining the fallback keeps direct unit-test invocation safe.
        pass


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    """Return a bounded worker-log tail for actionable failure messages."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    return "...[tail truncated]...\n" + text[-max_chars:]


def _terminate_active_workers(
    active: Dict[Tuple[str, int], Tuple[subprocess.Popen, str, TextIO]],
) -> None:
    """Best-effort cleanup used when a worker fails or the coordinator stops."""
    if not active:
        return
    for proc, _result_file, _log_handle in active.values():
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
    for proc, _result_file, _log_handle in active.values():
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    for _proc, _result_file, log_handle in active.values():
        try:
            log_handle.close()
        except OSError:
            pass


# ── helpers ───────────────────────────────────────────────────────

def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def _std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _bootstrap_ci95(
    values: list[float],
    *,
    seed: int = 20260903,
    trials: int = 10_000,
) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = random.Random(seed)
    n = len(values)
    boot = sorted(
        _mean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(trials)
    )
    lo = boot[int(math.floor(0.025 * (trials - 1)))]
    hi = boot[int(math.ceil(0.975 * (trials - 1)))]
    return [round(float(lo), 6), round(float(hi), 6)]


def _paired_summary(
    per_seed: dict[int, dict[str, dict]],
    baseline_label: str,
    comparison_label: str,
) -> dict:
    summary = {}
    for metric, direction in FOCUS_METRICS.items():
        rows = []
        for seed in sorted(per_seed):
            base_val = per_seed[seed].get(baseline_label, {}).get(metric)
            comp_val = per_seed[seed].get(comparison_label, {}).get(metric)
            if not isinstance(base_val, (int, float)):
                continue
            if not isinstance(comp_val, (int, float)):
                continue
            raw_delta = float(comp_val) - float(base_val)
            favourable = raw_delta if direction == "higher" else -raw_delta
            rows.append({
                "seed": int(seed),
                "greedy": float(base_val),
                "jsq": float(comp_val),
                "jsq_minus_greedy": round(raw_delta, 6),
                "favourable_improvement": round(favourable, 6),
            })
        if not rows:
            continue
        raw = [r["jsq_minus_greedy"] for r in rows]
        fav = [r["favourable_improvement"] for r in rows]
        summary[metric] = {
            "direction": direction,
            "paired_seeds": len(rows),
            "jsq_minus_greedy_mean": round(_mean(raw), 6),
            "jsq_minus_greedy_ci95": _bootstrap_ci95(raw),
            "favourable_improvement_mean": round(_mean(fav), 6),
            "favourable_improvement_ci95": _bootstrap_ci95(fav),
            "jsq_wins": sum(1 for r in rows if r["favourable_improvement"] > 0),
            "greedy_wins": sum(1 for r in rows if r["favourable_improvement"] < 0),
            "ties": sum(1 for r in rows if r["favourable_improvement"] == 0),
            "per_seed": rows,
        }
    return summary


def _print_load_table(
    load: str,
    per_seed: dict[int, dict[str, dict]],
    paired: dict,
) -> None:
    """Print a human-readable paired comparison table for one load level."""
    header_metrics = [
        "completed_orders",
        "avg_task_duration",
        "avg_excess_delay",
        "open_order_count",
        "open_order_age_p95",
        "completed_order_flow_time_p95",
    ]
    short_names = {
        "completed_orders": "Orders",
        "avg_task_duration": "AvgDur",
        "avg_excess_delay": "AvgExcess",
        "open_order_count": "OpenOrd",
        "open_order_age_p95": "Age_p95",
        "completed_order_flow_time_p95": "Flow_p95",
    }

    print(f"\n{'=' * 80}")
    print(f"  LOAD: {load.upper()}")
    print(f"{'=' * 80}")

    seeds = sorted(per_seed.keys())
    for label in (GREEDY_LABEL, JSQ_LABEL):
        vals = {}
        for m in header_metrics:
            v = [per_seed[s][label].get(m, 0) for s in seeds
                 if isinstance(per_seed[s].get(label, {}).get(m), (int, float))]
            vals[m] = (_mean(v), _std(v)) if v else (0, 0)

        cols = "  ".join(
            f"{short_names.get(m, m):>10s}" for m in header_metrics
        )
        row = "  ".join(
            f"{vals[m][0]:>10.2f}" for m in header_metrics
        )
        std_row = "  ".join(
            f"{'+-' + f'{vals[m][1]:.2f}':>8s}" for m in header_metrics
        )
        if label == GREEDY_LABEL:
            print(f"\n  {'':12s}{cols}")
            print(f"  {'-' * (12 + len(cols))}")
        print(f"  {label:12s}{row}")
        print(f"  {'':12s}{std_row}")

    # Delta row
    print(f"  {'-' * (12 + len(cols))}")
    delta_parts = []
    for m in header_metrics:
        p = paired.get(m)
        if p:
            d = p["jsq_minus_greedy_mean"]
            delta_parts.append(f"{d:>+10.2f}")
        else:
            delta_parts.append(f"{'N/A':>10s}")
    print(f"  {'JSQ-Greedy':12s}{'  '.join(delta_parts)}")

    print()
    for m in header_metrics:
        p = paired.get(m)
        if not p:
            continue
        ci = p.get("favourable_improvement_ci95")
        ci_str = (
            f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "N/A"
        )
        sig = ""
        if ci and ci[0] > 0:
            sig = " *JSQ*"
        elif ci and ci[1] < 0:
            sig = " *Greedy*"
        print(
            f"  {short_names.get(m, m):>12s}: "
            f"JSQ wins {p['jsq_wins']}/{p['paired_seeds']}, "
            f"CI95 {ci_str}{sig}"
        )


# ── subprocess worker entry point ────────────────────────────────

def _worker_main(args: argparse.Namespace) -> None:
    """Run one (Greedy, JSQ) pair inside a dedicated subprocess.

    Writes a JSON result file that the coordinator reads back.
    """
    _configure_worker_threads(
        args._worker_threads
        if args._worker_threads is not None
        else os.environ.get(_WORKER_THREADS_ENV, DEFAULT_WORKER_THREADS),
        args._worker_interop_threads
        if args._worker_interop_threads is not None
        else os.environ.get(
            _WORKER_INTEROP_ENV, DEFAULT_WORKER_INTEROP_THREADS
        ),
    )

    from Policies.TaskAssigner.GreedyTaskAssigner import GreedyTaskAssigner
    from Policies.TaskAssigner.JSQTaskAssigner import JSQTaskAssigner
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    load = args._worker_load
    seed = int(args._worker_seed)
    config_path = args._worker_config
    ticks = int(args._worker_ticks)
    manifest_path = args._worker_manifest
    result_path = args._worker_result

    # Stage 1: Greedy (generate manifest)
    greedy_ta = GreedyTaskAssigner()
    greedy_metrics = _run_one_assigner(
        config_path, greedy_ta, seed, ticks,
        trace_label=GREEDY_LABEL,
        save_order_manifest=manifest_path,
    )

    # Stage 2: JSQ (replay manifest)
    jsq_ta = JSQTaskAssigner()
    jsq_metrics = _run_one_assigner(
        config_path, jsq_ta, seed, ticks,
        trace_label=JSQ_LABEL,
        recorded_orders_path=manifest_path,
    )

    # Validate manifest replay integrity
    g_sha = greedy_metrics.get("order_arrival_manifest_sha256")
    j_sha = jsq_metrics.get("order_arrival_manifest_sha256")
    g_cnt = greedy_metrics.get("order_arrival_count", -1)
    j_cnt = jsq_metrics.get("order_arrival_count", -1)
    if g_sha != j_sha or g_cnt != j_cnt:
        raise RuntimeError(
            f"{load} seed={seed}: manifest integrity violation"
        )

    payload = {
        "load": load, "seed": seed,
        GREEDY_LABEL: greedy_metrics,
        JSQ_LABEL: jsq_metrics,
    }
    Path(result_path).write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    g_orders = greedy_metrics.get("completed_orders", "?")
    j_orders = jsq_metrics.get("completed_orders", "?")
    print(f"[done] {load} seed={seed}  Greedy={g_orders}  JSQ={j_orders}")


def _run_pair_sequential(
    load: str, seed: int, config_path: str,
    ticks: int, manifest_path: str,
) -> Tuple[dict, dict]:
    """Run one pair in-process (--workers 1)."""
    from Policies.TaskAssigner.GreedyTaskAssigner import GreedyTaskAssigner
    from Policies.TaskAssigner.JSQTaskAssigner import JSQTaskAssigner
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    greedy_ta = GreedyTaskAssigner()
    g = _run_one_assigner(
        config_path, greedy_ta, seed, ticks,
        trace_label=GREEDY_LABEL,
        save_order_manifest=manifest_path,
    )
    jsq_ta = JSQTaskAssigner()
    j = _run_one_assigner(
        config_path, jsq_ta, seed, ticks,
        trace_label=JSQ_LABEL,
        recorded_orders_path=manifest_path,
    )
    g_sha = g.get("order_arrival_manifest_sha256")
    j_sha = j.get("order_arrival_manifest_sha256")
    if g_sha != j_sha or g.get("order_arrival_count") != j.get("order_arrival_count"):
        raise RuntimeError(
            f"{load} seed={seed}: manifest integrity violation"
        )
    return g, j


# ── main ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired Greedy vs JSQ comparison across load levels.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=list(DEFAULT_SEEDS),
        help=f"Random seeds (default: {list(DEFAULT_SEEDS)})",
    )
    parser.add_argument(
        "--ticks", type=int, default=DEFAULT_TICKS,
        help=f"Simulation ticks per run (default: {DEFAULT_TICKS})",
    )
    parser.add_argument(
        "--loads", nargs="+",
        choices=list(LOAD_CONFIGS.keys()),
        default=list(LOAD_CONFIGS.keys()),
        help="Load levels to test (default: all three)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1 = sequential)",
    )
    parser.add_argument(
        "--worker-threads", type=int, default=DEFAULT_WORKER_THREADS,
        help=(
            "Intra-op/BLAS threads per subprocess worker "
            f"(default: {DEFAULT_WORKER_THREADS})"
        ),
    )
    parser.add_argument(
        "--worker-interop-threads", type=int,
        default=DEFAULT_WORKER_INTEROP_THREADS,
        help=(
            "PyTorch inter-op threads per subprocess worker "
            f"(default: {DEFAULT_WORKER_INTEROP_THREADS})"
        ),
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path for JSON result file",
    )
    parser.add_argument(
        "--manifests-dir", type=str, default=None,
        help="Directory for order manifests (default: temp dir)",
    )

    # Hidden sub-command for subprocess workers
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-load", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-config", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-ticks", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-manifest", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-result", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-threads", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--_worker-interop-threads", type=int, help=argparse.SUPPRESS
    )

    args = parser.parse_args()

    # ── Worker mode: run a single pair and exit ───────────────────
    if args._worker:
        _worker_main(args)
        return

    # ── Coordinator mode ──────────────────────────────────────────
    seeds = [int(s) for s in args.seeds]
    if len(seeds) != len(set(seeds)):
        raise SystemExit("duplicate seeds")
    ticks = int(args.ticks)
    loads = list(args.loads)
    workers = max(1, int(args.workers))
    worker_threads = max(1, int(args.worker_threads))
    worker_interop_threads = max(1, int(args.worker_interop_threads))
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite: {output}")

    if args.manifests_dir:
        manifests_dir = Path(args.manifests_dir)
        manifests_dir.mkdir(parents=True, exist_ok=True)
        tmp_cleanup = False
    else:
        manifests_dir = Path(tempfile.mkdtemp(prefix="jsq_manifests_"))
        tmp_cleanup = True

    total_pairs = len(loads) * len(seeds)
    print(
        f"Greedy vs JSQ comparison: "
        f"{len(loads)} loads x {len(seeds)} seeds x 2 arms = "
        f"{total_pairs * 2} runs"
    )
    print(f"workers: {workers}  |  ticks: {ticks}")
    print(
        f"worker torch threads: intra={worker_threads}, "
        f"interop={worker_interop_threads}"
    )
    print(f"order manifests: {manifests_dir}")

    # Build job list
    jobs: List[Tuple[str, int, str, int, str]] = []
    for load in loads:
        for seed in seeds:
            manifest_path = str(
                manifests_dir / f"greedy_{load}_seed{seed}_orders.json"
            )
            jobs.append((load, seed, LOAD_CONFIGS[load], ticks, manifest_path))

    wall_t0 = time.time()
    pair_results: Dict[Tuple[str, int], Tuple[dict, dict]] = {}
    worker_logs_dir: str | None = None

    if workers <= 1:
        # ── Sequential ────────────────────────────────────────────
        for idx, (load, seed, cfg, tk, mf) in enumerate(jobs):
            print(f"\n[{idx + 1}/{total_pairs}] {load} seed={seed}")
            g, j = _run_pair_sequential(load, seed, cfg, tk, mf)
            pair_results[(load, seed)] = (g, j)
            print(
                f"  Greedy: orders={g.get('completed_orders')}, "
                f"excess={g.get('avg_excess_delay', 0):.1f}  |  "
                f"JSQ: orders={j.get('completed_orders')}, "
                f"excess={j.get('avg_excess_delay', 0):.1f}"
            )
    else:
        # ── Parallel via subprocess.Popen ─────────────────────────
        # Parallel workers write logs to files.  A PIPE read only after
        # ``poll()`` can deadlock on Windows when warning output fills it.
        import shutil

        results_dir = Path(tempfile.mkdtemp(prefix="jsq_results_"))
        output.parent.mkdir(parents=True, exist_ok=True)
        logs_dir = output.parent / f"{output.stem}.worker_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        worker_logs_dir = str(logs_dir)
        python = sys.executable

        active: Dict[
            Tuple[str, int], Tuple[subprocess.Popen, str, TextIO]
        ] = {}
        pending = list(jobs)
        completed = 0

        try:
            while pending or active:
                # Launch new workers up to the concurrency limit.
                while pending and len(active) < workers:
                    load, seed, cfg, tk, mf = pending.pop(0)
                    result_file = str(results_dir / f"{load}_seed{seed}.json")
                    log_path = logs_dir / f"{load}_seed{seed}.log"
                    log_handle = log_path.open(
                        "w", encoding="utf-8", errors="replace", buffering=1
                    )
                    worker_env = os.environ.copy()
                    for name in _THREAD_ENV_NAMES:
                        worker_env[name] = str(worker_threads)
                    worker_env[_WORKER_THREADS_ENV] = str(worker_threads)
                    worker_env[_WORKER_INTEROP_ENV] = str(
                        worker_interop_threads
                    )
                    # Bootstrap torch before runpy imports WorldModel.  The
                    # package imports torch eagerly, so a normal ``python -m``
                    # invocation would set inter-op threads too late.
                    cmd = [
                        python, "-c", _WORKER_BOOTSTRAP,
                        "--_worker",
                        "--_worker-load", load,
                        "--_worker-seed", str(seed),
                        "--_worker-config", cfg,
                        "--_worker-ticks", str(tk),
                        "--_worker-manifest", mf,
                        "--_worker-result", result_file,
                        "--_worker-threads", str(worker_threads),
                        "--_worker-interop-threads",
                        str(worker_interop_threads),
                        "--output", str(output),  # required by argparse
                    ]
                    try:
                        proc = subprocess.Popen(
                            cmd,
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                            env=worker_env,
                        )
                    except BaseException:
                        log_handle.close()
                        raise
                    active[(load, seed)] = (proc, result_file, log_handle)

                # Poll running workers.
                done_keys = []
                for key, (proc, result_file, _log_handle) in active.items():
                    ret = proc.poll()
                    if ret is None:
                        continue
                    completed += 1
                    load, seed = key
                    log_path = logs_dir / f"{load}_seed{seed}.log"
                    if ret != 0:
                        tail = _tail_text(log_path)
                        raise RuntimeError(
                            f"{load} seed={seed}: worker exited with code "
                            f"{ret}; log={log_path}\n{tail}"
                        )
                    result_path = Path(result_file)
                    if not result_path.is_file():
                        tail = _tail_text(log_path)
                        raise RuntimeError(
                            f"{load} seed={seed}: worker exited successfully "
                            f"but did not write {result_path}; log={log_path}\n"
                            f"{tail}"
                        )
                    payload = json.loads(
                        result_path.read_text(encoding="utf-8")
                    )
                    pair_results[(load, seed)] = (
                        payload[GREEDY_LABEL], payload[JSQ_LABEL]
                    )
                    print(
                        f"[{completed}/{total_pairs}] {load} seed={seed}  "
                        f"Greedy={payload[GREEDY_LABEL].get('completed_orders')} "
                        f"JSQ={payload[JSQ_LABEL].get('completed_orders')}"
                    )
                    done_keys.append(key)

                for key in done_keys:
                    _proc, _result_file, log_handle = active.pop(key)
                    log_handle.close()

                if active and not done_keys:
                    time.sleep(0.5)
        finally:
            _terminate_active_workers(active)
            shutil.rmtree(results_dir, ignore_errors=True)

    wall_elapsed = time.time() - wall_t0

    # ── Assemble per-load results ─────────────────────────────────

    from WorldModel.evaluation.evaluate_online_v6 import aggregate_seeds

    all_results: Dict[str, dict] = {}
    for load in loads:
        per_seed: Dict[int, Dict[str, dict]] = {}
        for seed in seeds:
            g, j = pair_results[(load, seed)]
            per_seed[seed] = {GREEDY_LABEL: g, JSQ_LABEL: j}

        paired = _paired_summary(per_seed, GREEDY_LABEL, JSQ_LABEL)
        _print_load_table(load, per_seed, paired)

        all_results[load] = {
            "config": LOAD_CONFIGS[load],
            "seeds": seeds,
            "ticks": ticks,
            "per_seed": {str(s): v for s, v in per_seed.items()},
            "aggregate": aggregate_seeds(per_seed),
            "paired_comparison": paired,
        }

    # ── Save JSON ─────────────────────────────────────────────────
    result = {
        "schema_version": SCHEMA_VERSION,
        "arms": [GREEDY_LABEL, JSQ_LABEL],
        "loads": loads,
        "seeds": seeds,
        "ticks": ticks,
        "workers": workers,
        "worker_threads": worker_threads,
        "worker_interop_threads": worker_interop_threads,
        "worker_logs_dir": worker_logs_dir,
        "wall_time_s": round(wall_elapsed, 2),
        "results_by_load": all_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nresults saved: {output}")

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY  ({len(seeds)} seeds x {ticks} ticks, {workers} workers)")
    print(f"{'=' * 80}")
    header = (
        f"  {'Load':>6s}  {'Metric':>14s}  "
        f"{'Greedy':>10s}  {'JSQ':>10s}  "
        f"{'Delta':>10s}  {'JSQ wins':>10s}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for load in loads:
        paired = all_results[load]["paired_comparison"]
        agg = all_results[load]["aggregate"]
        first = True
        for metric in (
            "completed_orders",
            "avg_excess_delay",
            "open_order_age_p95",
        ):
            g_mean = agg.get(GREEDY_LABEL, {}).get(f"{metric}_mean", 0)
            j_mean = agg.get(JSQ_LABEL, {}).get(f"{metric}_mean", 0)
            p = paired.get(metric, {})
            delta = p.get("jsq_minus_greedy_mean", 0)
            wins = f"{p.get('jsq_wins', 0)}/{p.get('paired_seeds', 0)}"
            load_col = load.upper() if first else ""
            print(
                f"  {load_col:>6s}  {metric:>14s}  "
                f"{g_mean:>10.2f}  {j_mean:>10.2f}  "
                f"{delta:>+10.2f}  {wins:>10s}"
            )
            first = False
        print()

    print(f"total wall time: {wall_elapsed:.1f}s")

    if tmp_cleanup:
        import shutil
        shutil.rmtree(manifests_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
