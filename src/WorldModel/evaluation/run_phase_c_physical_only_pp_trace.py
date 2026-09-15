"""Per-tick TRACE re-collection for the physical-only PP factorial.

Purpose
-------
The 50-seed PP factorial (``run_phase_c_physical_only_pp_50seed.py``) recorded
only aggregate, end-of-run station metrics (``trace_record_count == 0``).  The
offline causal analysis therefore established *mediation* (avoiding station-lock
carries the high-load method gain) but could NOT test *temporal precedence*
(does lock onset precede throughput collapse) because there was no per-tick,
INDEPENDENT throughput signal on disk.

This runner re-collects the SAME runs at per-tick resolution so a multi-seed
precedence test (Granger / cross-correlation / change-point ordering) becomes
possible.  It records, per sampled tick:

  * the station-lock signal already produced by the frozen audit probe
    (``admission_attempts`` per station -> the ``share_attempts`` lock metric),
  * an INDEPENDENT throughput signal: ``order_state.total_completed`` plus
    pending / in-progress backlog.

Collection-only guarantee
--------------------------
NOTHING in the model / assignment / engine-step path is modified.  This runner:

  * replays the frozen factorial's Greedy manifests (identical arrival stream),
  * uses the identical checkpoints/configs (read from the factorial protocol),
  * attaches TWO read-only ``on_tick`` observers via ``engine.on_tick_callbacks``
    -- the existing ``StationAdmissionRestorationAuditProbe`` (with tracing
    enabled instead of ``trace_max_records=0``) and a new ``ThroughputTraceProbe``
    that only reads ``world.order_state``.

Determinism (seed + manifest replay) means each re-collected run reproduces the
corresponding factorial run; the runner cross-checks ``completed_orders`` against
the frozen factorial per-seed file and records the comparison.

Typical usage (multi_robot / torch conda environment)::

    python -m WorldModel.evaluation.run_phase_c_physical_only_pp_trace \
        --workers 6 --worker-threads 1 --worker-interop-threads 1

    # local wiring smoke (no manifest replay, short run, Greedy only):
    python -m WorldModel.evaluation.run_phase_c_physical_only_pp_trace --smoke

Defaults target the claim (high load; locking Greedy vs non-locking ComboS1J1);
override with --loads / --arms / --seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

# Reuse the frozen factorial's protocol/assigner/manifest machinery verbatim.
# Importing this module pulls only stdlib at import time (no torch), so the
# coordinator can run under the base interpreter; workers run under torch.
from WorldModel.evaluation.run_phase_c_physical_only_pp_50seed import (
    CONFIGS,
    LOADS,
    SEEDS,
    TICKS,
    VARIANT,
    ARMS as ALL_ARMS,
    STATION_ADMISSION,
    DEFAULT_OUTPUT_ROOT as FACTORIAL_ROOT,
    _make_assigner,
    _add_multiobjective_diagnostics,
    _validate_manifest,
    _read_json,
    _atomic_json,
    _sha256_file,
    _configure_worker_threads,
    _manifest_path,
    _job_path,
    _THREAD_ENV_NAMES,
    _WORKER_THREADS_ENV,
    _WORKER_INTEROP_ENV,
    _tail,
    _terminate,
)

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "phasec_physical_only_pp_trace_v1"
DEFAULT_TRACE_STRIDE = 10
DEFAULT_LOADS = ("high",)
DEFAULT_ARMS = ("Greedy", "ComboS1J1")
RESULT_SCHEMA = "phase_c_physical_only_pp_trace_job_v1"


def _trace_max_records(ticks: int, stride: int) -> int:
    """Cap sized so no sampled tick is dropped (+ small safety buffer)."""
    stride = max(1, int(stride))
    return math.ceil(int(ticks) / stride) + 8


class ThroughputTraceProbe:
    """Read-only per-tick INDEPENDENT throughput / backlog observer.

    Attached via ``engine.on_tick_callbacks``; it only reads
    ``world.order_state`` and never mutates simulation state.  Samples on the
    same stride as the station probe so the two series align tick-for-tick.
    """

    def __init__(self, engine, *, stride: int):
        self.engine = engine
        self.stride = max(1, int(stride))
        self.trace: list[dict[str, int]] = []

    def on_tick(self, engine) -> None:
        world = engine.world
        tick = int(world.tick)
        if tick % self.stride != 0:
            return
        order_state = world.order_state
        self.trace.append(
            {
                "tick": tick,
                "completed": int(order_state.total_completed),
                "pending": len(order_state.get_pending_orders()),
                "in_progress": len(order_state.get_in_progress_orders()),
            }
        )

    def summary(self) -> dict[str, Any]:
        return {
            "signal": "order_state.total_completed (independent of station probe)",
            "stride": int(self.stride),
            "record_count": len(self.trace),
            "trace": self.trace,
        }


@contextmanager
def _traced_audit_contract(
    trace_stride: int, trace_max_records: int
) -> Iterator[dict[str, Any]]:
    """Isolated copy of the physical-only audit contract with tracing ON.

    Mirrors ``run_phase_c_s1_j1_long_risk_correction._physical_only_audit_contract``
    exactly, except (a) ``trace_max_records > 0`` so the station trace is
    recorded, and (b) a second read-only throughput probe is attached.  The
    frozen contract/probe files are NOT modified.
    """

    import WorldModel.evaluation.evaluate_online_v6 as eval_module
    from WorldModel.evaluation.run_phase_c_station_admission_restoration import (
        StationAdmissionRestorationAuditProbe,
        STATION_ADMISSION_PHYSICAL_ONLY,
    )

    original_builder = eval_module._build_engine
    holder: dict[str, Any] = {}

    def build(cfg, task_assigner=None):
        # Logging-only change (identical to the frozen contract): physical-only
        # failed-path warnings can otherwise dominate output.  No simulation
        # decision is affected.
        cfg.simulation.log_level = "ERROR"
        engine = original_builder(cfg, task_assigner=task_assigner)
        modes = {
            queue.admission_mode
            for queue in engine.world.station_state.stations.values()
        }
        if modes != {STATION_ADMISSION_PHYSICAL_ONLY}:
            raise RuntimeError(
                "trace runner expected constructor-default physical-only "
                f"station admission, got {sorted(modes)}"
            )
        station_probe = StationAdmissionRestorationAuditProbe(
            engine,
            STATION_ADMISSION_PHYSICAL_ONLY,
            trace_stride=trace_stride,
            trace_max_records=trace_max_records,
        )
        throughput_probe = ThroughputTraceProbe(engine, stride=trace_stride)
        engine.on_tick_callbacks.append(station_probe.on_tick)
        engine.on_tick_callbacks.append(throughput_probe.on_tick)
        holder["engine"] = engine
        holder["station_probe"] = station_probe
        holder["throughput_probe"] = throughput_probe
        return engine

    eval_module._build_engine = build
    try:
        yield holder
    finally:
        eval_module._build_engine = original_builder


def _run_simulation_traced(
    config_path: str,
    assigner: Any,
    seed: int,
    ticks: int,
    trace_label: str,
    *,
    recorded_orders_path: str | None = None,
    save_order_manifest: str | None = None,
    trace_stride: int = DEFAULT_TRACE_STRIDE,
    trace_max_records: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Mirror of scale_generalization._run_simulation using the traced contract.

    Returns ``(metrics, station_audit, throughput_trace)``.
    """
    from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner

    if trace_max_records is None:
        trace_max_records = _trace_max_records(ticks, trace_stride)

    with _traced_audit_contract(trace_stride, trace_max_records) as holder:
        metrics = _run_one_assigner(
            config_path,
            assigner,
            int(seed),
            int(ticks),
            trace_label=trace_label,
            recorded_orders_path=recorded_orders_path,
            save_order_manifest=save_order_manifest,
        )
    if hasattr(assigner, "psi_dispatch_metrics"):
        metrics.update(assigner.psi_dispatch_metrics())

    station_probe = holder.get("station_probe")
    throughput_probe = holder.get("throughput_probe")
    if station_probe is None or throughput_probe is None:
        raise RuntimeError("trace probes were not attached")
    station_audit = station_probe.summary()
    metrics.update(
        {
            "station_admission_mode": station_audit.get("mode"),
            "station_capacity_rejections": int(
                station_audit.get("capacity_rejections", 0)
            ),
        }
    )
    return dict(metrics), dict(station_audit), throughput_probe.summary()


def _factorial_completed(source_root: Path, load: str, seed: int) -> dict[str, Any]:
    """Return {arm: completed_orders} from the frozen factorial per-seed file."""
    path = _job_path(source_root, load, seed)
    if not path.is_file():
        return {}
    payload = _read_json(path)
    return {
        arm: data.get("metrics", {}).get("completed_orders")
        for arm, data in payload.get("arms", {}).items()
    }


def _run_arm_traced(
    config_path: Path,
    arm: str,
    seed: int,
    ticks: int,
    manifest_path: Path,
    protocol: Mapping[str, Any],
    trace_stride: int,
    factorial_completed: Mapping[str, Any],
) -> dict[str, Any]:
    assigner = _make_assigner(arm, protocol)
    metrics, station_audit, throughput_trace = _run_simulation_traced(
        str(config_path),
        assigner,
        seed,
        ticks,
        f"PhysicalOnlyPPTrace/{VARIANT}/{arm}/seed{seed}",
        recorded_orders_path=str(manifest_path),
        trace_stride=trace_stride,
    )
    metrics = _add_multiobjective_diagnostics(metrics)
    manifest = _validate_manifest(manifest_path)

    completed = metrics.get("completed_orders")
    factorial_value = factorial_completed.get(arm)
    reproduces_factorial = (
        factorial_value is not None and int(completed) == int(factorial_value)
    )

    checks = {
        "manifest_count_matches": int(metrics.get("order_arrival_count", -1))
        == int(manifest.get("total_orders", -2)),
        "manifest_hash_matches": metrics.get("order_arrival_manifest_sha256")
        == manifest.get("manifest_sha256"),
        "ticks_match": int(metrics.get("ticks", -1)) == ticks,
        "physical_only_mode": station_audit.get("mode") == STATION_ADMISSION,
        "station_audit_passed": bool(station_audit.get("passed")),
        "physical_capacity_never_exceeded": int(
            station_audit.get("physical_capacity_violation_count", -1)
        )
        == 0,
        "station_trace_recorded": int(station_audit.get("trace_record_count", 0)) > 0,
        "station_trace_not_dropped": int(station_audit.get("trace_dropped", 0)) == 0,
        "throughput_trace_recorded": int(throughput_trace.get("record_count", 0)) > 0,
    }
    if not all(checks.values()):
        failed = ", ".join(key for key, value in checks.items() if not value)
        raise RuntimeError(f"{arm} trace audit failed seed={seed}: {failed}")

    return {
        "arm": arm,
        "completed_orders": completed,
        "factorial_completed_orders": factorial_value,
        "reproduces_factorial_completed_orders": reproduces_factorial,
        "metrics": metrics,
        "station_trace": {
            "stride": station_audit.get("trace_stride"),
            "record_count": station_audit.get("trace_record_count"),
            "dropped": station_audit.get("trace_dropped"),
            "trace": station_audit.get("trace"),
        },
        "throughput_trace": throughput_trace,
        "audit": {"passed": True, "checks": checks},
    }


def _worker_main(args: argparse.Namespace) -> None:
    _configure_worker_threads(args.worker_threads, args.worker_interop_threads)
    source_root = Path(args.source_root)
    load = args.load
    seed = int(args.seed)
    ticks = int(args.ticks)
    trace_stride = int(args.trace_stride)
    arms = tuple(args.arms.split(",")) if args.arms else DEFAULT_ARMS
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.is_file():
        print(f"[resume] {load} seed={seed}")
        return

    protocol_bundle = _read_json(
        source_root / "physical_only_pp_50seed_protocol.json"
    )
    protocol = protocol_bundle["protocol"]
    config_path = Path(protocol["configs"][load]["path"])

    manifest_path = _manifest_path(source_root, load, seed)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "frozen factorial manifest required for identical replay: "
            f"{manifest_path}"
        )
    manifest = _validate_manifest(manifest_path)
    factorial_completed = _factorial_completed(source_root, load, seed)

    arm_results: dict[str, dict[str, Any]] = {}
    for arm in arms:
        arm_results[arm] = _run_arm_traced(
            config_path,
            arm,
            seed,
            ticks,
            manifest_path,
            protocol,
            trace_stride,
            factorial_completed,
        )

    payload = {
        "schema_version": RESULT_SCHEMA,
        "source_factorial_root": source_root.as_posix(),
        "source_protocol_sha256": protocol_bundle.get("protocol_sha256"),
        "variant": VARIANT,
        "load": load,
        "seed": seed,
        "ticks": ticks,
        "trace_stride": trace_stride,
        "manifest": {
            "path": manifest_path.as_posix(),
            "file_sha256": _sha256_file(manifest_path),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "arms": arm_results,
    }
    _atomic_json(result_path, payload)
    summary = " ".join(
        f"{arm}={arm_results[arm]['completed_orders']}"
        f"({'ok' if arm_results[arm]['reproduces_factorial_completed_orders'] else 'DIFF'})"
        for arm in arms
    )
    print(f"[done] {load} seed={seed} {summary}")


def _trace_job_path(root: Path, load: str, seed: int) -> Path:
    return root / "per_seed" / f"{load}_seed{seed}.json"


def _run_smoke(args: argparse.Namespace) -> None:
    """Local wiring check: short Greedy run, live orders, verify both probes."""
    _configure_worker_threads(args.worker_threads, args.worker_interop_threads)
    load = args.loads.split(",")[0] if args.loads else DEFAULT_LOADS[0]
    ticks = int(args.smoke_ticks)
    stride = int(args.trace_stride)
    config_path = CONFIGS[load]
    print(f"[smoke] load={load} ticks={ticks} stride={stride} arm=Greedy (live orders)")

    from Policies.TaskAssigner import GreedyTaskAssigner

    metrics, station_audit, throughput_trace = _run_simulation_traced(
        str(config_path),
        GreedyTaskAssigner(),
        int(args.smoke_seed),
        ticks,
        f"PhysicalOnlyPPTrace/SMOKE/{VARIANT}/Greedy/seed{args.smoke_seed}",
        trace_stride=stride,
    )
    st = station_audit
    tp = throughput_trace
    print(f"[smoke] completed_orders(metric)={metrics.get('completed_orders')}")
    print(
        f"[smoke] station trace: stride={st.get('trace_stride')} "
        f"records={st.get('trace_record_count')} dropped={st.get('trace_dropped')}"
    )
    print(
        f"[smoke] throughput trace: stride={tp.get('stride')} "
        f"records={tp.get('record_count')}"
    )
    if st.get("trace"):
        first = st["trace"][0]
        stations = first.get("stations", [])
        print(
            f"[smoke] station row @tick={first.get('tick')}: "
            + ", ".join(
                f"s{s['station_id']}(att={s['admission_attempts']},"
                f"gr={s['admission_granted']},rej={s['capacity_rejections']})"
                for s in stations
            )
        )
    if tp.get("trace"):
        head = tp["trace"][: min(4, len(tp["trace"]))]
        tail = tp["trace"][-1]
        print(f"[smoke] throughput head: {head}")
        print(f"[smoke] throughput last: {tail}")
    ok = (
        int(st.get("trace_record_count", 0)) > 0
        and int(tp.get("record_count", 0)) > 0
        and bool(st.get("passed"))
        and int(st.get("physical_capacity_violation_count", -1)) == 0
    )
    print(f"[smoke] RESULT: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def _run_coordinator(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    source_root = Path(args.source_root)
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "worker_logs"
    logs.mkdir(parents=True, exist_ok=True)

    loads = tuple(args.loads.split(",")) if args.loads else DEFAULT_LOADS
    arms = args.arms if args.arms else ",".join(DEFAULT_ARMS)
    seeds = (
        tuple(int(s) for s in args.seeds.split(",")) if args.seeds else SEEDS
    )

    proto_path = source_root / "physical_only_pp_50seed_protocol.json"
    if not proto_path.is_file():
        raise FileNotFoundError(f"source factorial protocol not found: {proto_path}")

    jobs = [(load, seed) for load in loads for seed in seeds]
    pending = [job for job in jobs if not _trace_job_path(root, *job).is_file()]
    print(
        f"Per-tick trace re-collection: loads={loads} arms=[{arms}] "
        f"seeds={len(seeds)} -> {len(jobs)} (load,seed) jobs"
    )
    print(f"source factorial: {source_root}")
    print(f"output: {root}")
    print(f"workers: {args.workers}; trace_stride={args.trace_stride}")
    if not pending:
        print("all trace job files already exist; nothing to do")
        return

    active: dict[tuple[str, int], tuple[subprocess.Popen, Any, Path]] = {}
    completed = len(jobs) - len(pending)
    python = sys.executable
    bootstrap = (
        "import os,runpy,torch;"
        "torch.set_num_threads(int(os.environ.get('RMFS_PP50_WORKER_THREADS','1')));"
        "torch.set_num_interop_threads(int(os.environ.get('RMFS_PP50_WORKER_INTEROP_THREADS','1')));"
        "runpy.run_module('WorldModel.evaluation.run_phase_c_physical_only_pp_trace', run_name='__main__')"
    )
    try:
        while pending or active:
            while pending and len(active) < max(1, int(args.workers)):
                load, seed = pending.pop(0)
                result = _trace_job_path(root, load, seed)
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
                    "--source-root",
                    str(source_root),
                    "--load",
                    load,
                    "--seed",
                    str(seed),
                    "--ticks",
                    str(args.ticks),
                    "--trace-stride",
                    str(args.trace_stride),
                    "--arms",
                    arms,
                    "--result",
                    str(result),
                    "--worker-threads",
                    str(args.worker_threads),
                    "--worker-interop-threads",
                    str(args.worker_interop_threads),
                ]
                proc = subprocess.Popen(
                    cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env
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
                if not _trace_job_path(root, load, seed).is_file():
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

    print(json.dumps({"completed_jobs": completed, "expected_jobs": len(jobs)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=FACTORIAL_ROOT,
                        help="frozen factorial root providing protocol + manifests")
    parser.add_argument("--loads", default=",".join(DEFAULT_LOADS),
                        help="comma-separated subset of low,mid,high")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                        help="comma-separated arms to trace")
    parser.add_argument("--seeds", default="",
                        help="comma-separated seeds; default = full 900-949")
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--trace-stride", type=int, default=DEFAULT_TRACE_STRIDE)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--worker-interop-threads", type=int, default=1)
    # smoke
    parser.add_argument("--smoke", action="store_true",
                        help="short local wiring check (Greedy, live orders)")
    parser.add_argument("--smoke-ticks", type=int, default=60)
    parser.add_argument("--smoke-seed", type=int, default=900)
    # worker (hidden)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--load", choices=LOADS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        if args.load is None or args.seed is None or args.result is None:
            raise SystemExit("worker arguments are incomplete")
        # worker reuses --source-root, --arms, --ticks, --trace-stride
        _worker_main(args)
        return
    if args.smoke:
        _run_smoke(args)
        return
    _run_coordinator(args)


if __name__ == "__main__":
    main()
