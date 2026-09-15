"""Validate and summarize the paired service-due station-feedback sweep."""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from WorldModel.core.station_feedback_control import (
    STATION_FEEDBACK_MODE_ACTIVE_V2,
    STATION_FEEDBACK_MODE_ACTIVE_V3,
    STATION_FEEDBACK_MODE_OFF,
    STATION_FEEDBACK_MODE_SHADOW_V3,
    STATION_FEEDBACK_RELEASE_AWARE_EXPERIMENT_MODES,
    STATION_FEEDBACK_RELEASE_SIGNAL_SERVICE_DUE,
)
from WorldModel.evaluation.analyze_phase_c_station_feedback_closed_loop import (
    BEHAVIOR_EQUIVALENCE_KEYS,
    _mark_collapses,
    _mean,
    _parse_points,
    _write_csv,
)
from WorldModel.evaluation.run_phase_c_psi_dynamic_admission import (
    _atomic_json,
    _read_json,
)
from WorldModel.evaluation.run_phase_c_station_feedback_release_aware import (
    ARM_KEYS,
    EXPERIMENT_CONTRACT_VERSION,
    SCHEMA_VERSION,
)


SUMMARY_SCHEMA_VERSION = "phase_c_station_feedback_release_aware_summary_v1"


def _output_path(
    root: Path,
    tag: str,
    mode: str,
    load: str,
    seed: int,
) -> Path:
    return (
        root / tag / "per_arm" / ARM_KEYS[mode]
        / f"{load}_seed{seed}.json"
    )


def _state_ticks(metrics: Mapping[str, Any], state: str) -> int:
    return int(
        (metrics.get("station_feedback_state_station_ticks") or {}).get(
            state, 0
        )
    )


def _transition_count(metrics: Mapping[str, Any], transition: str) -> int:
    return int(
        (metrics.get("station_feedback_transitions") or {}).get(
            transition, 0
        )
    )


def _load_runs(
    args: argparse.Namespace,
    points: Sequence[tuple[float, str]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    payloads: dict[tuple[str, str, int], dict[str, Any]] = {}
    physical_violations = 0
    for multiplier, tag in points:
        for mode in STATION_FEEDBACK_RELEASE_AWARE_EXPERIMENT_MODES:
            for seed in args.seeds:
                path = _output_path(
                    args.output_root, tag, mode, args.load, int(seed)
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = _read_json(path)
                meta = payload.get("meta") or {}
                audit = payload.get("audit") or {}
                if payload.get("schema_version") != SCHEMA_VERSION:
                    raise RuntimeError(f"unexpected schema in {path}")
                if (
                    meta.get("experiment_contract_version")
                    != EXPERIMENT_CONTRACT_VERSION
                ):
                    raise RuntimeError(f"unexpected contract in {path}")
                if not bool(audit.get("passed")):
                    raise RuntimeError(f"arm audit failed in {path}")
                if (
                    meta.get("station_feedback_mode") != mode
                    or meta.get("arm_key") != ARM_KEYS[mode]
                    or meta.get("load") != args.load
                    or int(meta.get("seed", -1)) != int(seed)
                    or int(meta.get("ticks", -1)) != int(args.sim_ticks)
                ):
                    raise RuntimeError(f"metadata mismatch in {path}")

                config = meta.get("station_feedback_config") or {}
                release_aware = mode in (
                    STATION_FEEDBACK_MODE_SHADOW_V3,
                    STATION_FEEDBACK_MODE_ACTIVE_V3,
                )
                if release_aware != (
                    config.get("release_signal")
                    == STATION_FEEDBACK_RELEASE_SIGNAL_SERVICE_DUE
                ):
                    raise RuntimeError(f"release profile mismatch in {path}")

                metrics = payload.get("metrics") or {}
                station = payload.get("station_admission_audit") or {}
                manifest = payload.get("manifest") or {}
                reference = payload.get("paired_reference") or {}
                physical_violations += int(
                    station.get("physical_capacity_violation_count", 0)
                )
                if int(
                    metrics.get(
                        "station_feedback_transition_trace_dropped", 0
                    )
                ) != 0:
                    raise RuntimeError(f"feedback trace dropped in {path}")

                total_orders = int(manifest.get("total_orders", 0))
                completed = float(metrics.get("completed_orders") or 0.0)
                row = {
                    "multiplier": float(multiplier),
                    "tag": tag,
                    "mode": mode,
                    "arm": ARM_KEYS[mode],
                    "seed": int(seed),
                    "sim_ticks": int(args.sim_ticks),
                    "total_orders": total_orders,
                    "completed_orders": completed,
                    "completed_orders_per_1000_sim_ticks": (
                        1000.0 * completed / float(args.sim_ticks)
                    ),
                    "clearance_ratio": completed / max(total_orders, 1),
                    "open_order_count": float(
                        metrics.get("open_order_count") or 0.0
                    ),
                    "pending_order_count": float(
                        metrics.get("pending_order_count") or 0.0
                    ),
                    "deadlock_ratio_mean": float(
                        metrics.get("deadlock_ratio_mean") or 0.0
                    ),
                    "stall_ratio_mean": float(
                        metrics.get("stall_ratio_mean") or 0.0
                    ),
                    "capacity_rejections": float(
                        metrics.get("station_capacity_rejections") or 0.0
                    ),
                    "wall_time_s": float(metrics.get("wall_time_s") or 0.0),
                    "feedback_forced_defers": int(
                        metrics.get("station_feedback_forced_defers", 0)
                    ),
                    "feedback_suppressed_station_ticks": int(
                        metrics.get(
                            "station_feedback_batch_filter_suppressed_station_ticks",
                            0,
                        )
                    ),
                    "feedback_suppressed_context_ticks": int(
                        metrics.get(
                            "station_feedback_batch_filter_suppressed_context_ticks",
                            0,
                        )
                    ),
                    "service_release_due_station_ticks": int(
                        metrics.get(
                            "station_feedback_service_release_due_station_ticks",
                            0,
                        )
                    ),
                    "release_overdue_station_ticks": int(
                        metrics.get(
                            "station_feedback_release_overdue_station_ticks", 0
                        )
                    ),
                    "release_overdue_tick_sum": int(
                        metrics.get(
                            "station_feedback_release_overdue_tick_sum", 0
                        )
                    ),
                    "due_exit_blocked_station_ticks": int(
                        metrics.get(
                            "station_feedback_due_exit_blocked_station_ticks", 0
                        )
                    ),
                    "brake_station_ticks": _state_ticks(metrics, "brake"),
                    "recovery_station_ticks": _state_ticks(
                        metrics, "recovery"
                    ),
                    "locked_station_ticks": _state_ticks(metrics, "locked"),
                    "brake_to_recovery": _transition_count(
                        metrics, "brake->recovery"
                    ),
                    "recovery_to_open": _transition_count(
                        metrics, "recovery->open"
                    ),
                    "brake_to_locked": _transition_count(
                        metrics, "brake->locked"
                    ),
                    "reference_completed_orders": float(
                        (reference.get("metrics") or {}).get(
                            "completed_orders", 0.0
                        ) or 0.0
                    ),
                    "manifest_sha256": manifest.get("content_sha256"),
                    "path": path.as_posix(),
                }
                rows.append(row)
                payloads[(tag, mode, int(seed))] = payload
    _mark_collapses(rows)
    return rows, payloads, physical_violations


def _shadow_equivalence(
    payloads: Mapping[tuple[str, str, int], Mapping[str, Any]],
    points: Sequence[tuple[float, str]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    mismatches = []
    for _multiplier, tag in points:
        for seed in seeds:
            off = payloads[(tag, STATION_FEEDBACK_MODE_OFF, int(seed))]
            shadow = payloads[(tag, STATION_FEEDBACK_MODE_SHADOW_V3, int(seed))]
            off_metrics = off.get("metrics") or {}
            shadow_metrics = shadow.get("metrics") or {}
            different = [
                key
                for key in BEHAVIOR_EQUIVALENCE_KEYS
                if off_metrics.get(key) != shadow_metrics.get(key)
            ]
            if (
                (off.get("manifest") or {}).get("content_sha256")
                != (shadow.get("manifest") or {}).get("content_sha256")
                or different
            ):
                mismatches.append({
                    "tag": tag,
                    "seed": int(seed),
                    "mismatched_metrics": different,
                })
    return {
        "passed": not mismatches,
        "checked_pairs": len(points) * len(seeds),
        "mismatches": mismatches,
    }


def _curves(
    rows: Sequence[Mapping[str, Any]],
    points: Sequence[tuple[float, str]],
    min_clearance: float,
    max_collapse: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["tag"]), str(row["mode"]))].append(row)
    curves = []
    for multiplier, tag in points:
        for mode in STATION_FEEDBACK_RELEASE_AWARE_EXPERIMENT_MODES:
            group = grouped[(tag, mode)]
            clearance = _mean(row["clearance_ratio"] for row in group)
            collapse = _mean(float(row["collapsed"]) for row in group)
            curves.append({
                "multiplier": float(multiplier),
                "tag": tag,
                "mode": mode,
                "arm": ARM_KEYS[mode],
                "mean_completed_orders": _mean(
                    row["completed_orders"] for row in group
                ),
                "mean_clearance_ratio": clearance,
                "collapse_rate": collapse,
                "mean_pending_orders": _mean(
                    row["pending_order_count"] for row in group
                ),
                "mean_deadlock_ratio": _mean(
                    row["deadlock_ratio_mean"] for row in group
                ),
                "mean_capacity_rejections": _mean(
                    row["capacity_rejections"] for row in group
                ),
                "mean_forced_defers": _mean(
                    row["feedback_forced_defers"] for row in group
                ),
                "mean_release_overdue_station_ticks": _mean(
                    row["release_overdue_station_ticks"] for row in group
                ),
                "mean_locked_station_ticks": _mean(
                    row["locked_station_ticks"] for row in group
                ),
                "brake_to_recovery": sum(
                    int(row["brake_to_recovery"]) for row in group
                ),
                "recovery_to_open": sum(
                    int(row["recovery_to_open"]) for row in group
                ),
                "brake_to_locked": sum(
                    int(row["brake_to_locked"]) for row in group
                ),
                "sustainable": bool(
                    clearance >= min_clearance and collapse <= max_collapse
                ),
            })
    return curves


def _contrasts(
    rows: Sequence[Mapping[str, Any]],
    points: Sequence[tuple[float, str]],
) -> list[dict[str, Any]]:
    by_key = {
        (str(row["tag"]), str(row["mode"]), int(row["seed"])): row
        for row in rows
    }
    pairs = (
        ("active_v2_minus_off", STATION_FEEDBACK_MODE_ACTIVE_V2, STATION_FEEDBACK_MODE_OFF),
        ("active_v3_minus_off", STATION_FEEDBACK_MODE_ACTIVE_V3, STATION_FEEDBACK_MODE_OFF),
        ("active_v3_minus_active_v2", STATION_FEEDBACK_MODE_ACTIVE_V3, STATION_FEEDBACK_MODE_ACTIVE_V2),
    )
    result = []
    seeds = sorted({int(row["seed"]) for row in rows})
    for multiplier, tag in points:
        for label, left_mode, right_mode in pairs:
            deltas = []
            deadlock_deltas = []
            pending_deltas = []
            wins = ties = losses = 0
            for seed in seeds:
                left = by_key[(tag, left_mode, seed)]
                right = by_key[(tag, right_mode, seed)]
                delta = float(left["completed_orders"]) - float(
                    right["completed_orders"]
                )
                deltas.append(delta)
                deadlock_deltas.append(
                    float(left["deadlock_ratio_mean"])
                    - float(right["deadlock_ratio_mean"])
                )
                pending_deltas.append(
                    float(left["pending_order_count"])
                    - float(right["pending_order_count"])
                )
                wins += int(delta > 0)
                ties += int(delta == 0)
                losses += int(delta < 0)
            result.append({
                "multiplier": float(multiplier),
                "tag": tag,
                "contrast": label,
                "mean_completed_delta": _mean(deltas),
                "median_completed_delta": float(statistics.median(deltas)),
                "min_completed_delta": min(deltas),
                "max_completed_delta": max(deltas),
                "mean_deadlock_delta": _mean(deadlock_deltas),
                "mean_pending_delta": _mean(pending_deltas),
                "wins": wins,
                "ties": ties,
                "losses": losses,
            })
    return result


def _critical_multiplier(curves: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result = {}
    for mode in STATION_FEEDBACK_RELEASE_AWARE_EXPERIMENT_MODES:
        passing = [
            float(row["multiplier"])
            for row in curves
            if row["mode"] == mode and bool(row["sustainable"])
        ]
        result[mode] = max(passing) if passing else 0.0
    return result


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Station-feedback service-due capacity frontier",
        "",
        "V3 counts release delay only after the service robot is due to exit "
        "and the normal two-phase handoff grace has expired. Busy service time "
        "and admission rejection cannot independently create V3 BRAKE.",
        "",
        "## Integrity",
        "",
        f"- Complete paired runs: {summary['integrity']['expected_run_count']}",
        f"- All arm audits passed: {summary['integrity']['all_arm_audits_passed']}",
        f"- Off/shadow_v3 equivalence: {summary['shadow_equivalence']['passed']}",
        f"- Physical capacity violations: {summary['integrity']['physical_capacity_violations']}",
        "",
        "## Frontier curves",
        "",
        "| Multiplier | Mode | Completed | Clearance | Collapse | Pending | Deadlock | Forced defers | Overdue ticks | B->R | R->O | B->L | Sustainable |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["curves"]:
        lines.append(
            f"| {row['multiplier']:.2f} | {row['mode']} | "
            f"{row['mean_completed_orders']:.1f} | "
            f"{row['mean_clearance_ratio']:.3f} | "
            f"{row['collapse_rate']:.2f} | "
            f"{row['mean_pending_orders']:.1f} | "
            f"{row['mean_deadlock_ratio']:.3f} | "
            f"{row['mean_forced_defers']:.1f} | "
            f"{row['mean_release_overdue_station_ticks']:.1f} | "
            f"{row['brake_to_recovery']} | {row['recovery_to_open']} | "
            f"{row['brake_to_locked']} | "
            f"{'yes' if row['sustainable'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Paired contrasts",
        "",
        "| Multiplier | Contrast | Mean completed delta | Median | Min | Max | Pending delta | W/T/L |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["contrasts"]:
        lines.append(
            f"| {row['multiplier']:.2f} | {row['contrast']} | "
            f"{row['mean_completed_delta']:.1f} | "
            f"{row['median_completed_delta']:.1f} | "
            f"{row['min_completed_delta']:.0f} | "
            f"{row['max_completed_delta']:.0f} | "
            f"{row['mean_pending_delta']:.1f} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )
    lines.extend([
        "",
        "Primary success requires active_v3 to preserve off/shadow safety, "
        "remove the V2 DR oscillation and catastrophic LOCKED tail, and avoid "
        "paired throughput loss. Lower rejection alone is not success.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--load", default="high")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--points", nargs="+", required=True)
    parser.add_argument("--sim-ticks", type=int, required=True)
    parser.add_argument("--min-mean-clearance-ratio", type=float, default=0.90)
    parser.add_argument("--max-collapse-rate", type=float, default=0.20)
    args = parser.parse_args()
    points = _parse_points(args.points)
    rows, payloads, physical_violations = _load_runs(args, points)
    expected = (
        len(points)
        * len(STATION_FEEDBACK_RELEASE_AWARE_EXPERIMENT_MODES)
        * len(args.seeds)
    )
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} runs, found {len(rows)}")
    shadow = _shadow_equivalence(payloads, points, args.seeds)
    if not shadow["passed"]:
        raise RuntimeError("shadow_v3 changed simulator behavior")
    if physical_violations:
        raise RuntimeError("physical station capacity was violated")
    curves = _curves(
        rows,
        points,
        float(args.min_mean_clearance_ratio),
        float(args.max_collapse_rate),
    )
    contrasts = _contrasts(rows, points)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment_contract_version": EXPERIMENT_CONTRACT_VERSION,
        "integrity": {
            "expected_run_count": expected,
            "actual_run_count": len(rows),
            "all_arm_audits_passed": True,
            "physical_capacity_violations": physical_violations,
        },
        "shadow_equivalence": shadow,
        "curves": curves,
        "contrasts": contrasts,
        "critical_multiplier": _critical_multiplier(curves),
    }
    validation = args.output_root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    _atomic_json(validation / "station_feedback_release_aware.json", summary)
    _write_csv(validation / "station_feedback_release_aware_per_run.csv", rows)
    _write_csv(validation / "station_feedback_release_aware_curves.csv", curves)
    _write_csv(
        validation / "station_feedback_release_aware_contrasts.csv", contrasts
    )
    (validation / "station_feedback_release_aware.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    print(validation / "station_feedback_release_aware.md")


if __name__ == "__main__":
    main()
