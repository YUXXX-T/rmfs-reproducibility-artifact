"""Validate and summarize the paired station-feedback closed-loop sweep."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from WorldModel.evaluation.run_phase_c_psi_dynamic_admission import (
    _atomic_json,
    _read_json,
)
from WorldModel.evaluation.run_phase_c_station_feedback_closed_loop import (
    ARM_KEYS,
    SUPPORTED_SCHEMA_VERSIONS,
)
from WorldModel.core.station_feedback_control import (
    STATION_FEEDBACK_MODE_ACTIVE,
    STATION_FEEDBACK_MODE_OFF,
    STATION_FEEDBACK_MODE_SHADOW,
    STATION_FEEDBACK_MODES,
)


SUMMARY_SCHEMA_VERSION = "phase_c_station_feedback_closed_loop_summary_v2"
REFERENCE_KEY = "reference_s1_j1_eta_v3"

BEHAVIOR_EQUIVALENCE_KEYS = (
    "completed_orders",
    "completed_tasks",
    "avg_task_duration",
    "avg_excess_delay",
    "open_order_count",
    "pending_order_count",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "stall_ratio_mean",
    "stall_ratio_max",
    "station_capacity_rejections",
    "station_over_capacity_grants",
    "station_committed_over_capacity_tick_count",
    "station_ticks_with_any_committed_over_capacity",
    "station_context_defer_decisions",
    "station_context_defer_selected",
    "station_context_defer_debt_updates",
    "station_context_defer_liveness_bound_violations",
)


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return float(statistics.fmean(rows)) if rows else 0.0


def _output_path(root: Path, tag: str, mode: str, load: str, seed: int) -> Path:
    return (
        root / tag / "per_arm" / ARM_KEYS[mode]
        / f"{load}_seed{seed}.json"
    )


def _parse_points(values: Sequence[str]) -> list[tuple[float, str]]:
    points = []
    seen = set()
    for raw in values:
        multiplier_text, separator, tag = str(raw).partition(":")
        if not separator or not multiplier_text or not tag:
            raise SystemExit(
                f"invalid --points entry {raw!r}; expected multiplier:tag"
            )
        multiplier = float(multiplier_text)
        if multiplier <= 0.0 or tag in seen:
            raise SystemExit(f"invalid or duplicate frontier point: {raw!r}")
        seen.add(tag)
        points.append((multiplier, tag))
    return points


def _state_ticks(metrics: Mapping[str, Any], state: str) -> int:
    values = metrics.get("station_feedback_state_station_ticks") or {}
    return int(values.get(state, 0))


def _load_runs(args: argparse.Namespace, points: Sequence[tuple[float, str]]):
    payloads: dict[tuple[str, str, int], dict[str, Any]] = {}
    per_run: list[dict[str, Any]] = []
    physical_violations = 0
    for multiplier, tag in points:
        for mode in args.modes:
            for seed in args.seeds:
                path = _output_path(
                    args.output_root, tag, mode, args.load, int(seed)
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = _read_json(path)
                meta = payload.get("meta") or {}
                audit = payload.get("audit") or {}
                if payload.get("schema_version") not in (
                    SUPPORTED_SCHEMA_VERSIONS
                ):
                    raise RuntimeError(f"unexpected schema in {path}")
                if not bool(audit.get("passed")):
                    raise RuntimeError(f"arm audit failed in {path}")
                if (
                    meta.get("station_feedback_mode") != mode
                    or meta.get("load") != args.load
                    or int(meta.get("seed", -1)) != int(seed)
                    or int(meta.get("ticks", -1)) != int(args.sim_ticks)
                ):
                    raise RuntimeError(f"metadata mismatch in {path}")

                manifest = payload.get("manifest") or {}
                reference = payload.get("paired_reference") or {}
                metrics = payload.get("metrics") or {}
                station = payload.get("station_admission_audit") or {}
                total_orders = int(manifest.get("total_orders", 0))
                completed = float(metrics.get("completed_orders") or 0.0)
                clearance = completed / max(total_orders, 1)
                physical_violations += int(
                    station.get("physical_capacity_violation_count", 0)
                )
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
                    "clearance_ratio": clearance,
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
                    "wall_time_s": float(
                        metrics.get("wall_time_s") or 0.0
                    ),
                    "assignment_time_ms_mean": float(
                        metrics.get("assignment_time_ms_mean") or 0.0
                    ),
                    "dynamic_probe_steps": int(
                        metrics.get("dynamic_probe_steps", 0)
                    ),
                    "station_context_defer_steps": int(
                        metrics.get("station_context_defer_steps", 0)
                    ),
                    "feedback_would_defer_evaluations": int(
                        metrics.get(
                            "station_feedback_would_defer_evaluations", 0
                        )
                    ),
                    "feedback_forced_defers": int(
                        metrics.get("station_feedback_forced_defers", 0)
                    ),
                    "feedback_batch_filter_suppressed_batches": int(
                        metrics.get(
                            "station_feedback_batch_filter_suppressed_batches",
                            0,
                        )
                    ),
                    "feedback_batch_filter_suppressed_station_ticks": int(
                        metrics.get(
                            "station_feedback_batch_filter_suppressed_station_ticks",
                            0,
                        )
                    ),
                    "feedback_batch_filter_suppressed_context_ticks": int(
                        metrics.get(
                            "station_feedback_batch_filter_suppressed_context_ticks",
                            0,
                        )
                    ),
                    "feedback_batch_filter_unique_contexts": int(
                        metrics.get(
                            "station_feedback_batch_filter_unique_contexts",
                            0,
                        )
                    ),
                    "feedback_brake_station_ticks": _state_ticks(
                        metrics, "brake"
                    ),
                    "feedback_locked_station_ticks": _state_ticks(
                        metrics, "locked"
                    ),
                    "reference_completed_orders": float(
                        (reference.get("metrics") or {}).get(
                            "completed_orders", 0.0
                        ) or 0.0
                    ),
                    "reference_deadlock_ratio_mean": float(
                        (reference.get("metrics") or {}).get(
                            "deadlock_ratio_mean", 0.0
                        ) or 0.0
                    ),
                    "reference_stall_ratio_mean": float(
                        (reference.get("metrics") or {}).get(
                            "stall_ratio_mean", 0.0
                        ) or 0.0
                    ),
                    "reference_capacity_rejections": float(
                        (reference.get("metrics") or {}).get(
                            "station_capacity_rejections", 0.0
                        ) or 0.0
                    ),
                    "manifest_sha256": manifest.get("content_sha256"),
                    "path": path.as_posix(),
                }
                per_run.append(row)
                payloads[(tag, mode, int(seed))] = payload

    return payloads, per_run, physical_violations


def _shadow_equivalence(
    payloads: Mapping[tuple[str, str, int], Mapping[str, Any]],
    points: Sequence[tuple[float, str]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    checks = []
    mismatches = []
    for _multiplier, tag in points:
        for seed in seeds:
            off = payloads[(tag, STATION_FEEDBACK_MODE_OFF, int(seed))]
            shadow = payloads[(tag, STATION_FEEDBACK_MODE_SHADOW, int(seed))]
            off_metrics = off.get("metrics") or {}
            shadow_metrics = shadow.get("metrics") or {}
            key_mismatches = [
                key for key in BEHAVIOR_EQUIVALENCE_KEYS
                if off_metrics.get(key) != shadow_metrics.get(key)
            ]
            manifest_equal = (
                (off.get("manifest") or {}).get("content_sha256")
                == (shadow.get("manifest") or {}).get("content_sha256")
            )
            passed = manifest_equal and not key_mismatches
            checks.append({
                "tag": tag,
                "seed": int(seed),
                "passed": passed,
                "mismatched_metrics": key_mismatches,
            })
            if not passed:
                mismatches.append(checks[-1])
    return {
        "passed": not mismatches,
        "checked_pairs": len(checks),
        "behavior_metric_keys": list(BEHAVIOR_EQUIVALENCE_KEYS),
        "mismatches": mismatches,
    }


def _mark_collapses(per_run: list[dict[str, Any]]) -> None:
    best: dict[tuple[str, int], float] = defaultdict(float)
    for row in per_run:
        key = (str(row["tag"]), int(row["seed"]))
        best[key] = max(
            best[key],
            float(row["completed_orders"]),
            float(row["reference_completed_orders"]),
        )
    for row in per_run:
        ceiling = best[(str(row["tag"]), int(row["seed"]))]
        row["empirical_completion_ceiling"] = ceiling
        row["completion_efficiency"] = (
            float(row["completed_orders"]) / max(ceiling, 1.0)
        )
        row["collapsed"] = bool(
            row["completion_efficiency"]
            < float(0.8)
        )


def _curves(
    per_run: Sequence[Mapping[str, Any]],
    points: Sequence[tuple[float, str]],
    min_clearance: float,
    max_collapse: float,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_run:
        by_key[(str(row["tag"]), str(row["mode"]))].append(row)

    curves = []
    for multiplier, tag in points:
        for mode in STATION_FEEDBACK_MODES:
            rows = by_key[(tag, mode)]
            curve = {
                "multiplier": float(multiplier),
                "tag": tag,
                "mode": mode,
                "arm": ARM_KEYS[mode],
                "mean_completed_orders": _mean(
                    row["completed_orders"] for row in rows
                ),
                "mean_completed_orders_per_1000_sim_ticks": _mean(
                    row["completed_orders_per_1000_sim_ticks"] for row in rows
                ),
                "mean_clearance_ratio": _mean(
                    row["clearance_ratio"] for row in rows
                ),
                "collapse_rate": _mean(
                    float(bool(row["collapsed"])) for row in rows
                ),
                "mean_deadlock_ratio": _mean(
                    row["deadlock_ratio_mean"] for row in rows
                ),
                "mean_stall_ratio": _mean(
                    row["stall_ratio_mean"] for row in rows
                ),
                "mean_capacity_rejections": _mean(
                    row["capacity_rejections"] for row in rows
                ),
                "mean_feedback_forced_defers": _mean(
                    row["feedback_forced_defers"] for row in rows
                ),
                "mean_dynamic_probe_steps": _mean(
                    row["dynamic_probe_steps"] for row in rows
                ),
                "mean_feedback_batch_filter_suppressed_station_ticks": _mean(
                    row[
                        "feedback_batch_filter_suppressed_station_ticks"
                    ]
                    for row in rows
                ),
                "mean_feedback_batch_filter_unique_contexts": _mean(
                    row["feedback_batch_filter_unique_contexts"]
                    for row in rows
                ),
                "mean_brake_station_ticks": _mean(
                    row["feedback_brake_station_ticks"] for row in rows
                ),
                "mean_locked_station_ticks": _mean(
                    row["feedback_locked_station_ticks"] for row in rows
                ),
            }
            curve["sustainable"] = bool(
                curve["mean_clearance_ratio"] >= min_clearance
                and curve["collapse_rate"] <= max_collapse
            )
            curves.append(curve)

        reference_rows = by_key[(tag, STATION_FEEDBACK_MODE_OFF)]
        reference_completed = [
            float(row["reference_completed_orders"]) for row in reference_rows
        ]
        total_orders = [float(row["total_orders"]) for row in reference_rows]
        reference_clearance = [
            completed / max(total, 1.0)
            for completed, total in zip(reference_completed, total_orders)
        ]
        reference_collapsed = []
        for row, completed in zip(reference_rows, reference_completed):
            ceiling = float(row["empirical_completion_ceiling"])
            reference_collapsed.append(completed / max(ceiling, 1.0) < 0.8)
        curves.append({
            "multiplier": float(multiplier),
            "tag": tag,
            "mode": REFERENCE_KEY,
            "arm": REFERENCE_KEY,
            "mean_completed_orders": _mean(reference_completed),
            "mean_completed_orders_per_1000_sim_ticks": (
                1000.0 * _mean(reference_completed)
                / float(reference_rows[0]["sim_ticks"])
                if reference_rows else 0.0
            ),
            "mean_clearance_ratio": _mean(reference_clearance),
            "collapse_rate": _mean(float(value) for value in reference_collapsed),
            "mean_deadlock_ratio": _mean(
                row["reference_deadlock_ratio_mean"]
                for row in reference_rows
            ),
            "mean_stall_ratio": _mean(
                row["reference_stall_ratio_mean"]
                for row in reference_rows
            ),
            "mean_capacity_rejections": _mean(
                row["reference_capacity_rejections"]
                for row in reference_rows
            ),
            "mean_feedback_forced_defers": 0.0,
            "mean_dynamic_probe_steps": 0.0,
            "mean_feedback_batch_filter_suppressed_station_ticks": 0.0,
            "mean_feedback_batch_filter_unique_contexts": 0.0,
            "mean_brake_station_ticks": 0.0,
            "mean_locked_station_ticks": 0.0,
            "sustainable": bool(
                _mean(reference_clearance) >= min_clearance
                and _mean(float(value) for value in reference_collapsed)
                <= max_collapse
            ),
        })
    return curves


def _contrasts(
    per_run: Sequence[Mapping[str, Any]],
    points: Sequence[tuple[float, str]],
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["tag"]), str(row["mode"]), int(row["seed"])): row
        for row in per_run
    }
    rows = []
    for multiplier, tag in points:
        for comparison, left, right in (
            (
                "active_minus_off",
                STATION_FEEDBACK_MODE_ACTIVE,
                STATION_FEEDBACK_MODE_OFF,
            ),
            (
                "active_minus_shadow",
                STATION_FEEDBACK_MODE_ACTIVE,
                STATION_FEEDBACK_MODE_SHADOW,
            ),
        ):
            deltas = []
            deadlock_deltas = []
            wins = ties = losses = 0
            for seed in sorted({int(row["seed"]) for row in per_run}):
                left_row = lookup[(tag, left, seed)]
                right_row = lookup[(tag, right, seed)]
                delta = float(left_row["completed_orders"]) - float(
                    right_row["completed_orders"]
                )
                deltas.append(delta)
                deadlock_deltas.append(
                    float(left_row["deadlock_ratio_mean"])
                    - float(right_row["deadlock_ratio_mean"])
                )
                wins += int(delta > 0)
                ties += int(delta == 0)
                losses += int(delta < 0)
            rows.append({
                "multiplier": float(multiplier),
                "tag": tag,
                "comparison": comparison,
                "mean_completed_delta": _mean(deltas),
                "mean_deadlock_delta": _mean(deadlock_deltas),
                "wins": wins,
                "ties": ties,
                "losses": losses,
            })

        active_rows = [
            lookup[(tag, STATION_FEEDBACK_MODE_ACTIVE, int(seed))]
            for seed in sorted({int(row["seed"]) for row in per_run})
        ]
        reference_deltas = [
            float(row["completed_orders"])
            - float(row["reference_completed_orders"])
            for row in active_rows
        ]
        rows.append({
            "multiplier": float(multiplier),
            "tag": tag,
            "comparison": "active_minus_s1_j1_eta_v3_reference",
            "mean_completed_delta": _mean(reference_deltas),
            "mean_deadlock_delta": None,
            "wins": sum(delta > 0 for delta in reference_deltas),
            "ties": sum(delta == 0 for delta in reference_deltas),
            "losses": sum(delta < 0 for delta in reference_deltas),
        })
    return rows


def _critical(curves: Sequence[Mapping[str, Any]], points) -> dict[str, Any]:
    result = {}
    for mode in (*STATION_FEEDBACK_MODES, REFERENCE_KEY):
        by_tag = {
            str(row["tag"]): row for row in curves if row["mode"] == mode
        }
        contiguous = None
        highest = None
        still_contiguous = True
        for multiplier, tag in points:
            passed = bool(by_tag[tag]["sustainable"])
            if passed:
                highest = float(multiplier)
                if still_contiguous:
                    contiguous = float(multiplier)
            else:
                still_contiguous = False
        result[mode] = {
            "contiguous_critical_multiplier": contiguous,
            "highest_passing_multiplier": highest,
        }
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Station-feedback closed-loop capacity frontier",
        "",
        "The `off`, `shadow_v1`, and `active_v1` arms replay identical ETA V3 "
        "capacity-frontier manifests. `shadow_v1` must be behaviorally identical "
        "to `off`; only `active_v1` may change dispatch decisions.",
        "",
        "## Integrity",
        "",
        f"- Complete paired runs: {summary['integrity']['expected_run_count']}",
        f"- All arm audits passed: {summary['integrity']['all_arm_audits_passed']}",
        f"- Off/shadow behavioral equivalence: {summary['shadow_equivalence']['passed']}",
        f"- Physical capacity violations: {summary['integrity']['physical_capacity_violations']}",
        "",
        "## Frontier curves",
        "",
        "| Multiplier | Mode | Completed | Clearance | Collapse | Deadlock | Forced defer | Batch station suppress | Unique suppressed contexts | Dynamic steps | BRAKE station-ticks | LOCKED station-ticks | Sustainable |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["curves"]:
        lines.append(
            f"| {row['multiplier']:.2f} | {row['mode']} | "
            f"{row['mean_completed_orders']:.1f} | "
            f"{row['mean_clearance_ratio']:.3f} | "
            f"{row['collapse_rate']:.2f} | "
            f"{row['mean_deadlock_ratio']:.3f} | "
            f"{row['mean_feedback_forced_defers']:.1f} | "
            f"{row['mean_feedback_batch_filter_suppressed_station_ticks']:.1f} | "
            f"{row['mean_feedback_batch_filter_unique_contexts']:.1f} | "
            f"{row['mean_dynamic_probe_steps']:.1f} | "
            f"{row['mean_brake_station_ticks']:.1f} | "
            f"{row['mean_locked_station_ticks']:.1f} | "
            f"{'yes' if row['sustainable'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Paired contrasts",
        "",
        "| Multiplier | Contrast | Completed delta | Deadlock delta | W/T/L |",
        "|---:|---|---:|---:|---:|",
    ])
    for row in summary["contrasts"]:
        deadlock = (
            "n/a" if row["mean_deadlock_delta"] is None
            else f"{row['mean_deadlock_delta']:.4f}"
        )
        lines.append(
            f"| {row['multiplier']:.2f} | {row['comparison']} | "
            f"{row['mean_completed_delta']:.1f} | {deadlock} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )
    lines.extend([
        "",
        "This remains a finite-horizon empirical operating frontier, not a "
        "queueing-theoretic stability proof.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--load", default="high")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=STATION_FEEDBACK_MODES,
        default=list(STATION_FEEDBACK_MODES),
    )
    parser.add_argument(
        "--points",
        nargs="+",
        default=[
            "0.8:m080",
            "1.0:m100",
            "1.2:m120",
            "1.4:m140",
            "1.6:m160",
        ],
    )
    parser.add_argument("--sim-ticks", type=int, default=1500)
    parser.add_argument("--min-mean-clearance-ratio", type=float, default=0.90)
    parser.add_argument("--max-collapse-rate", type=float, default=0.20)
    args = parser.parse_args()

    if tuple(args.modes) != tuple(STATION_FEEDBACK_MODES):
        raise SystemExit(
            "formal analysis requires off, shadow_v1, and active_v1"
        )
    points = _parse_points(args.points)
    payloads, per_run, physical_violations = _load_runs(args, points)
    _mark_collapses(per_run)
    shadow = _shadow_equivalence(payloads, points, args.seeds)
    curves = _curves(
        per_run,
        points,
        float(args.min_mean_clearance_ratio),
        float(args.max_collapse_rate),
    )
    contrasts = _contrasts(per_run, points)
    expected = len(points) * len(args.modes) * len(args.seeds)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "config": {
            "load": args.load,
            "seeds": list(args.seeds),
            "points": [
                {"multiplier": multiplier, "tag": tag}
                for multiplier, tag in points
            ],
            "modes": list(args.modes),
            "sim_ticks": int(args.sim_ticks),
            "min_mean_clearance_ratio": float(
                args.min_mean_clearance_ratio
            ),
            "max_collapse_rate": float(args.max_collapse_rate),
            "collapse_efficiency_ratio": 0.8,
        },
        "integrity": {
            "expected_run_count": expected,
            "actual_run_count": len(per_run),
            "all_arm_audits_passed": len(per_run) == expected,
            "physical_capacity_violations": int(physical_violations),
        },
        "shadow_equivalence": shadow,
        "critical_multiplier": _critical(curves, points),
        "curves": curves,
        "contrasts": contrasts,
    }
    if (
        len(per_run) != expected
        or physical_violations != 0
        or not shadow["passed"]
    ):
        raise RuntimeError("closed-loop summary integrity checks failed")

    validation = args.output_root / "validation"
    _atomic_json(validation / "station_feedback_closed_loop.json", summary)
    _write_csv(validation / "station_feedback_closed_loop_per_run.csv", per_run)
    _write_csv(validation / "station_feedback_closed_loop_curves.csv", curves)
    _write_csv(
        validation / "station_feedback_closed_loop_contrasts.csv", contrasts
    )
    markdown = _markdown(summary)
    (validation / "station_feedback_closed_loop.md").write_text(
        markdown, encoding="utf-8"
    )
    print(markdown)


if __name__ == "__main__":
    main()
