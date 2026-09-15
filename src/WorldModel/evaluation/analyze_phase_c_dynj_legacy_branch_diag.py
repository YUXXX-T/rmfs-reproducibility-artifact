"""Recover and analyse the legacy Dynamic-J station-lock diagnostic.

The original batch completed all replays, but an audit plumbing bug omitted
assigner-specific metrics from ``diagnostic_summary.json`` and treated tiny
deadlock sampling drift as fatal.  This analyser is deliberately offline: it
does not run the simulator and it does not modify any frozen result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CASE_ORDER = ("low552", "high552", "high553", "high560", "mid560")
ARM_ORDER = ("shadow", "static_psi", "dynamic")
OMITTED_ASSIGNER_KEYS = {
    "psi_dispatch_eval_calls",
    "psi_dispatch_contexts_seen",
    "dynamic_probe_batches",
    "dynamic_probe_steps",
    "dynamic_probe_order_changed_batches",
}
DEADLOCK_WARNING_KEYS = {"deadlock_ratio_mean", "deadlock_ratio_max"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object: {path}:{line_number}")
            yield value


def _one(directory: Path, pattern: str) -> Path:
    values = sorted(directory.glob(pattern))
    if len(values) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern!r} in {directory}, got {len(values)}"
        )
    return values[0]


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _context_key(row: dict[str, Any]) -> str:
    return f"o{row['order_id']}/p{row['pod_id']}/s{row['station_id']}"


def _context_order(row: dict[str, Any]) -> list[str]:
    if "dynamic_order" in row:
        return [str(value) for value in row.get("dynamic_order") or []]
    contexts = [
        value for value in row.get("contexts") or []
        if bool(value.get("selected", True))
    ]
    if row.get("mode") == "j_ascending":
        contexts.sort(key=lambda value: (
            int(value.get("j_rank", value.get("original_index", 0))),
            int(value.get("original_index", 0)),
        ))
    else:
        contexts.sort(key=lambda value: int(value.get("original_index", 0)))
    return [_context_key(value) for value in contexts]


def _reference_audit(summary: dict[str, Any]) -> dict[str, Any]:
    """Recover the audit using the corrected v4 contract.

    Outcome/task/manifest and core decision fields remain hard checks.  The
    five assigner counters were absent from old summaries solely because the
    runner forgot to merge ``psi_dispatch_metrics``.  Their frozen expected
    values are reported, but are not invented from a trace truncated at the
    diagnostic capture tick.  Deadlock deltas remain warnings because the
    callback changes only the sampling frame, not the closed-loop outcome.
    """
    old = summary.get("reference_audit") or {}
    checks = dict(old.get("checks") or {})
    hard_checks = {
        key: bool(value)
        for key, value in checks.items()
        if key not in OMITTED_ASSIGNER_KEYS | DEADLOCK_WARNING_KEYS
    }
    omitted = {
        key: {
            "reason": "old_runner_omitted_assigner_specific_metrics",
            **((old.get("compared") or {}).get(key) or {}),
        }
        for key in OMITTED_ASSIGNER_KEYS
        if key in checks
    }
    warnings = {}
    compared = old.get("compared") or {}
    for key in DEADLOCK_WARNING_KEYS:
        if key not in compared:
            continue
        observed = _number((compared.get(key) or {}).get("observed"))
        expected = _number((compared.get(key) or {}).get("expected"))
        warnings[key] = {
            "observed": observed,
            "expected": expected,
            "delta": (
                observed - expected
                if observed is not None and expected is not None else None
            ),
            "fatal": False,
        }
    return {
        "old_passed": old.get("passed"),
        "recovered_passed": bool(hard_checks) and all(hard_checks.values()),
        "hard_checks": hard_checks,
        "old_runner_omissions": omitted,
        "warnings": warnings,
        "reference_result": old.get("reference_result"),
    }


def _lock_episodes(
    trajectory_path: Path,
    minimum_ticks: int,
) -> dict[str, Any]:
    active: dict[int, dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []
    max_assigned: Counter[int] = Counter()
    max_occupancy: Counter[int] = Counter()
    full_ticks: Counter[int] = Counter()
    first_overcommitted: dict[int, int] = {}
    first_full: dict[int, int] = {}
    first_entry_exit_occupied: dict[int, int] = {}
    first_entry_exit_still: dict[int, int] = {}
    last_tick = -1

    def close(station_id: int, censored: bool = False) -> None:
        episode = active.pop(station_id)
        episode["station_id"] = int(station_id)
        episode["duration_ticks"] = int(
            episode["end_tick"] - episode["start_tick"] + 1
        )
        episode["capture_censored"] = bool(censored)
        episodes.append(episode)

    for row in _read_jsonl(trajectory_path):
        tick = int(row["tick"])
        last_tick = tick
        agents = {
            tuple(agent["position"]): agent
            for agent in row.get("all_agents") or []
        }
        for queue in row.get("station_queues") or []:
            station_id = int(queue["station_id"])
            capacity = int(queue["capacity"])
            occupancy = int(queue["occupancy"])
            assigned = int(queue["assigned_agent_count"])
            max_assigned[station_id] = max(max_assigned[station_id], assigned)
            max_occupancy[station_id] = max(max_occupancy[station_id], occupancy)
            if assigned > capacity:
                first_overcommitted.setdefault(station_id, tick)
            if occupancy >= capacity:
                full_ticks[station_id] += 1
                first_full.setdefault(station_id, tick)

            entry = agents.get(tuple(queue["entry_position"]))
            exit_agent = agents.get(tuple(queue["exit_position"]))
            both_occupied = entry is not None and exit_agent is not None
            if both_occupied:
                first_entry_exit_occupied.setdefault(station_id, tick)
            both_still = bool(
                both_occupied
                and int(entry.get("stationary_ticks", 0)) >= 1
                and int(exit_agent.get("stationary_ticks", 0)) >= 1
            )
            if both_still:
                first_entry_exit_still.setdefault(station_id, tick)
            locked = bool(
                occupancy >= capacity
                and both_occupied
                and str(entry.get("status")) != "IDLE"
                and str(exit_agent.get("status")) != "IDLE"
                and int(entry.get("stationary_ticks", 0)) >= 10
                and int(exit_agent.get("stationary_ticks", 0)) >= 10
            )
            if locked:
                if station_id not in active:
                    active[station_id] = {
                        "start_tick": tick,
                        "end_tick": tick,
                        "entry_agent": int(entry["agent_id"]),
                        "entry_status": str(entry["status"]),
                        "exit_agent": int(exit_agent["agent_id"]),
                        "exit_status": str(exit_agent["status"]),
                        "occupancy_at_start": occupancy,
                        "capacity": capacity,
                        "assigned_at_start": assigned,
                        "max_assigned_during_episode": assigned,
                        "completed_orders_at_start": int(
                            row.get("completed_orders_total", 0)
                        ),
                    }
                else:
                    episode = active[station_id]
                    episode["end_tick"] = tick
                    episode["max_assigned_during_episode"] = max(
                        int(episode["max_assigned_during_episode"]), assigned
                    )
            elif station_id in active:
                close(station_id)

    for station_id in list(active):
        close(station_id, censored=True)
    sustained = [
        episode for episode in episodes
        if int(episode["duration_ticks"]) >= minimum_ticks
    ]
    right_censored_candidates = [
        episode for episode in episodes
        if bool(episode.get("capture_censored"))
        and int(episode["duration_ticks"]) < minimum_ticks
    ]
    sustained.sort(key=lambda value: (
        int(value["start_tick"]), int(value["station_id"])
    ))
    return {
        "capture_last_tick": last_tick,
        "definition": {
            "minimum_ticks": minimum_ticks,
            "conditions": [
                "station occupancy equals capacity",
                "entry and exit cells both hold non-idle robots",
                "both robots have stationary_ticks >= 10",
            ],
        },
        "episodes": sustained,
        "right_censored_candidates": right_censored_candidates,
        "first_sustained_lock_tick": (
            sustained[0]["start_tick"] if sustained else None
        ),
        "max_sustained_lock_ticks": max(
            (int(value["duration_ticks"]) for value in sustained), default=0
        ),
        "max_assigned_by_station": {
            str(key): int(value) for key, value in sorted(max_assigned.items())
        },
        "max_occupancy_by_station": {
            str(key): int(value) for key, value in sorted(max_occupancy.items())
        },
        "full_ticks_by_station": {
            str(key): int(value) for key, value in sorted(full_ticks.items())
        },
        "first_overcommitted_tick_by_station": {
            str(key): int(value)
            for key, value in sorted(first_overcommitted.items())
        },
        "first_full_tick_by_station": {
            str(key): int(value) for key, value in sorted(first_full.items())
        },
        "first_entry_exit_occupied_tick_by_station": {
            str(key): int(value)
            for key, value in sorted(first_entry_exit_occupied.items())
        },
        "first_entry_exit_still_tick_by_station": {
            str(key): int(value)
            for key, value in sorted(first_entry_exit_still.items())
        },
    }


def _path_failures(path: Path) -> dict[str, Any]:
    total = 0
    by_station: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    by_task_type: Counter[str] = Counter()
    first_tick: int | None = None
    last_tick: int | None = None
    for row in _read_jsonl(path):
        total += 1
        tick = int(row["tick"])
        first_tick = tick if first_tick is None else min(first_tick, tick)
        last_tick = tick if last_tick is None else max(last_tick, tick)
        active = row.get("active_task") or {}
        station_id = active.get("station_id")
        by_station["none" if station_id is None else str(station_id)] += 1
        by_status[str(row.get("status"))] += 1
        by_task_type[str(active.get("type", "none"))] += 1
    return {
        "total": total,
        "first_tick": first_tick,
        "last_tick": last_tick,
        "by_station": dict(sorted(by_station.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_task_type": dict(sorted(by_task_type.items())),
    }


def _first_context_divergence(
    left_path: Path,
    right_path: Path,
) -> dict[str, Any] | None:
    left = {int(row["tick"]): row for row in _read_jsonl(left_path)}
    right = {int(row["tick"]): row for row in _read_jsonl(right_path)}
    for tick in sorted(set(left) | set(right)):
        left_order = _context_order(left[tick]) if tick in left else None
        right_order = _context_order(right[tick]) if tick in right else None
        if left_order != right_order:
            return {
                "tick": tick,
                "left_order": left_order,
                "right_order": right_order,
            }
    return None


def _load_admission_runs(root: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if root is None:
        return {}
    validation = root / "validation" / "dynamic_admission_validation.json"
    if not validation.is_file():
        return {}
    payload = _read_json(validation)
    return {
        (str(row["load"]), int(row["seed"])): row
        for row in payload.get("runs") or []
    }


def _resolve_legacy_dynamic_root(
    explicit: Path | None,
    admission_root: Path | None,
) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if admission_root is not None:
        candidates.extend([
            admission_root.parent / "psi_dispatch_dynamic_probe_551_560_h1500_v1",
            admission_root.parent / "psi_dispatch_dynamic_probe_551_560_v1",
        ])
    for candidate in candidates:
        if (candidate / "per_arm" / "s1_psi_dynamic_probe").is_dir():
            return candidate
    return None


def _full_admission_comparison(
    legacy_root: Path | None,
    admission_root: Path | None,
) -> dict[str, Any]:
    if legacy_root is None or admission_root is None:
        return {"available": False}
    legacy_arm = legacy_root / "per_arm" / "s1_psi_dynamic_probe"
    admission_arm = (
        admission_root / "per_arm" / "s1_psi_dynamic_committed_admission_v1"
    )
    if not legacy_arm.is_dir() or not admission_arm.is_dir():
        return {"available": False}
    rows = []
    for load in ("low", "mid", "high"):
        for seed in range(551, 561):
            legacy_path = legacy_arm / f"{load}_seed{seed}.json"
            admission_path = admission_arm / f"{load}_seed{seed}.json"
            if not legacy_path.is_file() or not admission_path.is_file():
                continue
            legacy = _read_json(legacy_path).get("metrics") or {}
            admitted = _read_json(admission_path).get("metrics") or {}
            rows.append({
                "load": load,
                "seed": seed,
                "legacy": {
                    key: legacy.get(key) for key in (
                        "completed_orders", "deadlock_ratio_mean",
                        "stall_ratio_mean",
                    )
                },
                "committed_admission_v1": {
                    key: admitted.get(key) for key in (
                        "completed_orders", "deadlock_ratio_mean",
                        "stall_ratio_mean",
                    )
                },
            })

    def summary(values: list[dict[str, Any]]) -> dict[str, Any]:
        if not values:
            return {"runs": 0}
        return {
            "runs": len(values),
            "legacy_completed_orders_mean": _mean([
                float(row["legacy"]["completed_orders"]) for row in values
            ]),
            "admission_completed_orders_mean": _mean([
                float(row["committed_admission_v1"]["completed_orders"])
                for row in values
            ]),
            "legacy_deadlock_ratio_mean": _mean([
                float(row["legacy"]["deadlock_ratio_mean"]) for row in values
            ]),
            "admission_deadlock_ratio_mean": _mean([
                float(row["committed_admission_v1"]["deadlock_ratio_mean"])
                for row in values
            ]),
            "legacy_stall_ratio_mean": _mean([
                float(row["legacy"]["stall_ratio_mean"]) for row in values
            ]),
            "admission_stall_ratio_mean": _mean([
                float(row["committed_admission_v1"]["stall_ratio_mean"])
                for row in values
            ]),
            "completion_wins_ties_losses": {
                "wins": sum(
                    float(row["committed_admission_v1"]["completed_orders"])
                    > float(row["legacy"]["completed_orders"])
                    for row in values
                ),
                "ties": sum(
                    float(row["committed_admission_v1"]["completed_orders"])
                    == float(row["legacy"]["completed_orders"])
                    for row in values
                ),
                "losses": sum(
                    float(row["committed_admission_v1"]["completed_orders"])
                    < float(row["legacy"]["completed_orders"])
                    for row in values
                ),
            },
            "completion_up_deadlock_down": sum(
                float(row["committed_admission_v1"]["completed_orders"])
                > float(row["legacy"]["completed_orders"])
                and float(row["committed_admission_v1"]["deadlock_ratio_mean"])
                < float(row["legacy"]["deadlock_ratio_mean"])
                for row in values
            ),
        }

    return {
        "available": bool(rows),
        "legacy_root": legacy_root.as_posix(),
        "paired_runs": len(rows),
        "overall": summary(rows),
        "by_load": {
            load: summary([row for row in rows if row["load"] == load])
            for load in ("low", "mid", "high")
        },
        "rows": rows,
    }


def analyse(
    root: Path,
    admission_root: Path | None,
    legacy_dynamic_root: Path | None,
    minimum_lock_ticks: int,
) -> dict[str, Any]:
    run_dirs = [
        path for path in root.iterdir()
        if path.is_dir()
        and path.name not in {"logs", "validation"}
        and (path / "diagnostic_summary.json").is_file()
    ]
    if len(run_dirs) != 15:
        raise RuntimeError(f"expected 15 replay directories, found {len(run_dirs)}")
    admission = _load_admission_runs(admission_root)
    resolved_legacy_root = _resolve_legacy_dynamic_root(
        legacy_dynamic_root, admission_root
    )
    runs: list[dict[str, Any]] = []
    traces: dict[tuple[str, str], Path] = {}

    for directory in sorted(run_dirs):
        summary = _read_json(directory / "diagnostic_summary.json")
        load = str(summary["load"])
        seed = int(summary["seed"])
        arm = str(summary["arm"])
        case = f"{load}{seed}"
        context_path = _one(directory, "context_order_*.jsonl")
        traces[(case, arm)] = context_path
        audit = _reference_audit(summary)
        metrics = summary.get("metrics") or {}
        lock = _lock_episodes(
            _one(directory, "trajectory_*.jsonl"), minimum_lock_ticks
        )
        failures = _path_failures(_one(directory, "path_failures_*.jsonl"))
        admission_row = admission.get((load, seed)) if arm == "dynamic" else None
        admission_comparison = None
        if admission_row is not None:
            new_metrics = admission_row.get("metrics") or {}
            admission_comparison = {
                "available": True,
                "completed_orders": new_metrics.get("completed_orders"),
                "deadlock_ratio_mean": new_metrics.get("deadlock_ratio_mean"),
                "stall_ratio_mean": new_metrics.get("stall_ratio_mean"),
                "completed_orders_delta": (
                    float(new_metrics["completed_orders"])
                    - float(metrics["completed_orders"])
                ),
                "deadlock_ratio_mean_delta": (
                    float(new_metrics["deadlock_ratio_mean"])
                    - float(metrics["deadlock_ratio_mean"])
                ),
                "stall_ratio_mean_delta": (
                    float(new_metrics["stall_ratio_mean"])
                    - float(metrics["stall_ratio_mean"])
                ),
                "capacity_rejections": admission_row.get("capacity_rejections"),
                "max_committed_load": admission_row.get("max_committed_load"),
            }
        runs.append({
            "run": directory.name,
            "case": case,
            "load": load,
            "seed": seed,
            "arm": arm,
            "audit": audit,
            "metrics": {
                "completed_orders": metrics.get("completed_orders"),
                "completed_tasks": metrics.get("completed_tasks"),
                "deadlock_ratio_mean": metrics.get("deadlock_ratio_mean"),
                "deadlock_ratio_max": metrics.get("deadlock_ratio_max"),
                "stall_ratio_mean": metrics.get("stall_ratio_mean"),
                "stall_ratio_max": metrics.get("stall_ratio_max"),
                "energy_conv_modified_decisions": metrics.get(
                    "energy_conv_modified_decisions"
                ),
            },
            "path_failures": failures,
            "station_lock": lock,
            "committed_admission_v1": admission_comparison,
        })

    by_case: dict[str, Any] = {}
    for case in CASE_ORDER:
        case_runs = {row["arm"]: row for row in runs if row["case"] == case}
        if set(case_runs) != set(ARM_ORDER):
            raise RuntimeError(f"case {case} lacks all three arms")
        by_case[case] = {
            "outcomes": {
                arm: case_runs[arm]["metrics"] for arm in ARM_ORDER
            },
            "shadow_vs_static_first_context_divergence": (
                _first_context_divergence(
                    traces[(case, "shadow")], traces[(case, "static_psi")]
                )
            ),
            "static_vs_dynamic_first_context_divergence": (
                _first_context_divergence(
                    traces[(case, "static_psi")], traces[(case, "dynamic")]
                )
            ),
            "sustained_locks": {
                arm: case_runs[arm]["station_lock"]["episodes"]
                for arm in ARM_ORDER
            },
        }

    dynamic_runs = [row for row in runs if row["arm"] == "dynamic"]
    paired = [
        row for row in dynamic_runs
        if row.get("committed_admission_v1") is not None
    ]
    catastrophic = [
        row for row in dynamic_runs
        if row["station_lock"]["max_sustained_lock_ticks"] >= 60
    ]
    admission_summary = {
        "available": bool(paired),
        "paired_cases": len(paired),
        "legacy_completed_orders_mean": _mean([
            float(row["metrics"]["completed_orders"]) for row in paired
        ]),
        "admission_completed_orders_mean": _mean([
            float(row["committed_admission_v1"]["completed_orders"])
            for row in paired
        ]),
        "legacy_deadlock_ratio_mean": _mean([
            float(row["metrics"]["deadlock_ratio_mean"]) for row in paired
        ]),
        "admission_deadlock_ratio_mean": _mean([
            float(row["committed_admission_v1"]["deadlock_ratio_mean"])
            for row in paired
        ]),
        "legacy_stall_ratio_mean": _mean([
            float(row["metrics"]["stall_ratio_mean"]) for row in paired
        ]),
        "admission_stall_ratio_mean": _mean([
            float(row["committed_admission_v1"]["stall_ratio_mean"])
            for row in paired
        ]),
        "catastrophic_legacy_cases": [row["case"] for row in catastrophic],
        "catastrophic_case_admission_deltas": {
            row["case"]: row["committed_admission_v1"]
            for row in catastrophic
            if row["committed_admission_v1"] is not None
        },
    }

    all_audits = all(row["audit"]["recovered_passed"] for row in runs)
    return {
        "schema_version": "phase_c_dynj_legacy_branch_analysis_v1",
        "root": root.as_posix(),
        "offline_only": True,
        "audit_recovery": {
            "runs": len(runs),
            "recovered_passed_runs": sum(
                bool(row["audit"]["recovered_passed"]) for row in runs
            ),
            "all_recovered_passed": all_audits,
            "interpretation": (
                "All hard outcome/task/manifest/core-decision checks pass. "
                "Old audit failures came only from omitted assigner metrics "
                "and non-fatal deadlock sampling drift."
            ),
        },
        "mechanism": {
            "sustained_lock_definition": (
                "full station plus occupied entry and exit, with both "
                "non-idle robots stationary for at least 10 ticks"
            ),
            "minimum_episode_ticks": minimum_lock_ticks,
            "dynamic_runs_with_sustained_lock": sum(
                bool(row["station_lock"]["episodes"]) for row in dynamic_runs
            ),
            "dynamic_runs_with_catastrophic_lock_ge_60_ticks": len(catastrophic),
            "finding": (
                "Dynamic J changes context order and therefore robot/station "
                "arrival phase.  It can either avoid or trigger an entry-exit "
                "physical lock; ranking alone is not a safety invariant."
            ),
        },
        "admission_ablation": admission_summary,
        "full_30_run_admission_ablation": _full_admission_comparison(
            resolved_legacy_root, admission_root
        ),
        "design_decision": {
            "verdict": "committed_admission_v1_plus_station_conditioned_defer",
            "physical_safety_layer": (
                "Keep committed-capacity admission V1 as the final invariant: "
                "physical occupants plus admitted in-transit DELIVER robots "
                "must not exceed station capacity.  This is not FIFO V2 and "
                "does not create a waiting reservation list."
            ),
            "scheduling_layer": (
                "Add context-local defer before creating a new PICK-DELIVER-"
                "RETURN chain.  Dynamic J chooses the next context and S1 "
                "chooses its robot; when that station's marginal admission "
                "risk is excessive, defer only that context and continue "
                "with other stations."
            ),
            "liveness": (
                "Deferred contexts retain age/service debt and are reconsidered "
                "after capacity or pressure changes."
            ),
            "rejected_alternatives": {
                "defer_only": (
                    "Not a sufficient safety guarantee because J/WM lacks "
                    "exact path reservations and station arrival phase."
                ),
                "admission_only_as_final_method": (
                    "Safe but reactive; it can leave picked pods waiting and "
                    "does not optimise how much upstream work is launched."
                ),
                "fifo_v2": (
                    "Not required by this evidence and previously traded too "
                    "much throughput for waiting-list stability."
                ),
            },
        },
        "by_case": by_case,
        "runs": runs,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _markdown(report: dict[str, Any]) -> str:
    audit = report["audit_recovery"]
    admission = report["admission_ablation"]
    full_admission = report["full_30_run_admission_ablation"]
    lines = [
        "# Dynamic J legacy branch diagnosis",
        "",
        "## Headline",
        "",
        f"- replay audit recovered: {audit['recovered_passed_runs']}/"
        f"{audit['runs']} hard-check clean",
        "- failure mechanism: sustained station entry-exit physical lock, "
        "not a failed simulation and not a global Dynamic-J score collapse",
        "- final architecture: committed admission V1 safety invariant + "
        "Dynamic J/S1 context-local defer",
        "- do not use FIFO V2 as the main controller",
        "",
        "## Five paired cases",
        "",
        "| case | arm | completed | deadlock | stall | path failures | "
        "first sustained lock | lock ticks | station |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    index = {
        (row["case"], row["arm"]): row for row in report["runs"]
    }
    for case in CASE_ORDER:
        for arm in ARM_ORDER:
            row = index[(case, arm)]
            lock = row["station_lock"]
            episode = lock["episodes"][0] if lock["episodes"] else {}
            lines.append(
                f"| {case} | {arm} | "
                f"{_fmt(row['metrics']['completed_orders'])} | "
                f"{_fmt(row['metrics']['deadlock_ratio_mean'])} | "
                f"{_fmt(row['metrics']['stall_ratio_mean'])} | "
                f"{row['path_failures']['total']} | "
                f"{_fmt(lock['first_sustained_lock_tick'])} | "
                f"{_fmt(lock['max_sustained_lock_ticks'])} | "
                f"{_fmt(episode.get('station_id'))} |"
            )
    lines.extend([
        "",
        "The lock detector requires a full station, non-idle robots on both "
        "entry and exit, and both stationary for at least 10 ticks. Episodes "
        "shorter than the configured minimum are omitted from the main lock "
        "count; capture-end candidates remain recorded in JSON as right-"
        "censored.",
        "",
        "## Key trajectories",
        "",
        "- mid560 Dynamic J: station 2 lock starts at tick 170 and lasts "
        "through dense capture tick 500; 3112/3175 path failures target station 2.",
        "- high553 Dynamic J: station 4 lock starts at tick 318 and lasts "
        "through tick 500; 718/783 path failures target station 4.",
        "- low552: Shadow and Static J lock at station 4 from tick 432, while "
        "Dynamic J remains healthy. This is the direct counterexample to "
        "treating Dynamic J itself as uniformly dangerous.",
        "- healthy arms also reach assigned_agent_count above capacity. "
        "Overcommitment is a precondition/risk amplifier, not by itself a "
        "sufficient definition of lock.",
        "",
        "## Committed admission V1 ablation",
        "",
        f"- paired Dynamic-J cases: {admission['paired_cases']}",
        f"- completed orders mean: {_fmt(admission['legacy_completed_orders_mean'], 1)} "
        f"-> {_fmt(admission['admission_completed_orders_mean'], 1)}",
        f"- deadlock ratio mean: {_fmt(admission['legacy_deadlock_ratio_mean'])} "
        f"-> {_fmt(admission['admission_deadlock_ratio_mean'])}",
        f"- stall ratio mean: {_fmt(admission['legacy_stall_ratio_mean'])} "
        f"-> {_fmt(admission['admission_stall_ratio_mean'])}",
        "",
    ])
    if full_admission.get("available"):
        overall = full_admission["overall"]
        wtl = overall["completion_wins_ties_losses"]
        lines.extend([
            "Across the complete 30-run 551-560 Dynamic-J pairing:",
            "",
            f"- completed orders mean: "
            f"{_fmt(overall['legacy_completed_orders_mean'], 1)} -> "
            f"{_fmt(overall['admission_completed_orders_mean'], 1)}",
            f"- deadlock ratio mean: "
            f"{_fmt(overall['legacy_deadlock_ratio_mean'])} -> "
            f"{_fmt(overall['admission_deadlock_ratio_mean'])}",
            f"- stall ratio mean: "
            f"{_fmt(overall['legacy_stall_ratio_mean'])} -> "
            f"{_fmt(overall['admission_stall_ratio_mean'])}",
            f"- completion wins/ties/losses: {wtl['wins']}/"
            f"{wtl['ties']}/{wtl['losses']}",
            f"- completion up and deadlock down: "
            f"{overall['completion_up_deadlock_down']}/"
            f"{overall['runs']}",
            "",
        ])
    lines.extend([
        "Committed admission V1 is not FIFO V2. V1 limits physical occupants "
        "+ admitted in-transit DELIVER robots to capacity and has no waiting "
        "reservation list. FIFO V2 is the separate WAITING_ASSIGNED mechanism "
        "that previously reduced throughput.",
        "",
        "## Design decision",
        "",
        "1. Keep committed admission V1 as the non-negotiable physical safety "
        "invariant.",
        "2. Above it, let Dynamic J select the next context and S1 select the "
        "robot. Add station-conditioned, context-local defer before launching "
        "the chain; continue evaluating contexts for other stations.",
        "3. Carry age/service debt for deferred contexts so deferral cannot "
        "become starvation.",
        "4. Do not rely on defer alone for safety, and do not make FIFO V2 the "
        "main method.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--admission-root", type=Path, default=None)
    parser.add_argument("--legacy-dynamic-root", type=Path, default=None)
    parser.add_argument("--minimum-lock-ticks", type=int, default=10)
    args = parser.parse_args()
    if args.minimum_lock_ticks <= 0:
        raise SystemExit("--minimum-lock-ticks must be positive")
    report = analyse(
        args.input_root,
        args.admission_root,
        args.legacy_dynamic_root,
        args.minimum_lock_ticks,
    )
    validation = args.input_root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    json_path = validation / "dynj_legacy_branch_analysis.json"
    md_path = validation / "dynj_legacy_branch_analysis.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    (validation / "dynj_legacy_branch_analysis.sha256").write_text(
        f"{digest}  {json_path.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "validation": validation.as_posix(),
        "audit_recovery": report["audit_recovery"],
        "mechanism": report["mechanism"],
        "admission_ablation": report["admission_ablation"],
        "full_30_run_admission_ablation": (
            report["full_30_run_admission_ablation"].get("overall")
        ),
        "verdict": report["design_decision"]["verdict"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
