"""Validate and summarize the paired early-BRAKE station-feedback sweep."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from WorldModel.core.station_feedback_control import (
    STATION_FEEDBACK_EARLY_BRAKE_EXPERIMENT_MODES,
    STATION_FEEDBACK_MODE_ACTIVE,
    STATION_FEEDBACK_MODE_ACTIVE_V2,
    STATION_FEEDBACK_MODE_OFF,
    STATION_FEEDBACK_MODE_SHADOW_V2,
    station_feedback_mode_uses_early_brake,
)
from WorldModel.evaluation.analyze_phase_c_station_feedback_closed_loop import (
    BEHAVIOR_EQUIVALENCE_KEYS,
    REFERENCE_KEY,
    _mark_collapses,
    _mean,
    _parse_points,
    _write_csv,
)
from WorldModel.evaluation.run_phase_c_psi_dynamic_admission import (
    _atomic_json,
    _read_json,
)
from WorldModel.evaluation.run_phase_c_station_feedback_early_brake import (
    ARM_KEYS,
    EXPERIMENT_CONTRACT_VERSION,
    SCHEMA_VERSION,
)


SUMMARY_SCHEMA_VERSION = "phase_c_station_feedback_early_brake_summary_v1"


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


def _mapping_int(metrics: Mapping[str, Any], key: str, item: str) -> int:
    return int((metrics.get(key) or {}).get(item, 0))


def _load_runs(
    args: argparse.Namespace,
    points: Sequence[tuple[float, str]],
):
    payloads: dict[tuple[str, str, int], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    physical_violations = 0
    for multiplier, tag in points:
        for mode in STATION_FEEDBACK_EARLY_BRAKE_EXPERIMENT_MODES:
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
                if bool(config.get("early_brake_enabled", False)) != (
                    station_feedback_mode_uses_early_brake(mode)
                ):
                    raise RuntimeError(f"feedback profile mismatch in {path}")

                manifest = payload.get("manifest") or {}
                reference = payload.get("paired_reference") or {}
                metrics = payload.get("metrics") or {}
                station = payload.get("station_admission_audit") or {}
                physical_violations += int(
                    station.get("physical_capacity_violation_count", 0)
                )
                total_orders = int(manifest.get("total_orders", 0))
                completed = float(metrics.get("completed_orders") or 0.0)
                transitions = metrics.get("station_feedback_transitions") or {}
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
                    "assignment_time_ms_mean": float(
                        metrics.get("assignment_time_ms_mean") or 0.0
                    ),
                    "dynamic_probe_steps": int(
                        metrics.get("dynamic_probe_steps", 0)
                    ),
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
                    "feedback_unique_contexts": int(
                        metrics.get(
                            "station_feedback_batch_filter_unique_contexts", 0
                        )
                    ),
                    "committed_pressure_station_ticks": int(
                        metrics.get(
                            "station_feedback_committed_pressure_station_ticks",
                            0,
                        )
                    ),
                    "early_brake_condition_station_ticks": int(
                        metrics.get(
                            "station_feedback_early_brake_condition_station_ticks",
                            0,
                        )
                    ),
                    "early_reason_station_ticks": _mapping_int(
                        metrics,
                        "station_feedback_dominant_reason_station_ticks",
                        "early_committed_pressure",
                    ),
                    "caution_station_ticks": _mapping_int(
                        metrics, "station_feedback_state_station_ticks", "caution"
                    ),
                    "brake_station_ticks": _mapping_int(
                        metrics, "station_feedback_state_station_ticks", "brake"
                    ),
                    "recovery_station_ticks": _mapping_int(
                        metrics, "station_feedback_state_station_ticks", "recovery"
                    ),
                    "locked_station_ticks": _mapping_int(
                        metrics, "station_feedback_state_station_ticks", "locked"
                    ),
                    "brake_to_recovery": int(
                        transitions.get("brake->recovery", 0)
                    ),
                    "recovery_to_open": int(
                        transitions.get("recovery->open", 0)
                    ),
                    "brake_to_locked": int(
                        transitions.get("brake->locked", 0)
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
                rows.append(row)
                payloads[(tag, mode, int(seed))] = payload
    return payloads, rows, physical_violations


def _paired_integrity(payloads, points, seeds) -> dict[str, Any]:
    mismatches = []
    shadow_mismatches = []
    for _multiplier, tag in points:
        for seed in seeds:
            arms = [
                payloads[(tag, mode, int(seed))]
                for mode in STATION_FEEDBACK_EARLY_BRAKE_EXPERIMENT_MODES
            ]
            manifests = {
                (payload.get("manifest") or {}).get("content_sha256")
                for payload in arms
            }
            if len(manifests) != 1:
                mismatches.append({"tag": tag, "seed": int(seed)})

            off = payloads[(tag, STATION_FEEDBACK_MODE_OFF, int(seed))]
            shadow = payloads[
                (tag, STATION_FEEDBACK_MODE_SHADOW_V2, int(seed))
            ]
            off_metrics = off.get("metrics") or {}
            shadow_metrics = shadow.get("metrics") or {}
            changed = [
                key for key in BEHAVIOR_EQUIVALENCE_KEYS
                if off_metrics.get(key) != shadow_metrics.get(key)
            ]
            if changed:
                shadow_mismatches.append({
                    "tag": tag,
                    "seed": int(seed),
                    "mismatched_metrics": changed,
                })
    return {
        "manifest_pairing_passed": not mismatches,
        "manifest_mismatches": mismatches,
        "off_shadow_v2_equivalence_passed": not shadow_mismatches,
        "off_shadow_v2_mismatches": shadow_mismatches,
    }


def _curves(rows, points, min_clearance, max_collapse):
    by_key = defaultdict(list)
    for row in rows:
        by_key[(str(row["tag"]), str(row["mode"]))].append(row)
    result = []
    for multiplier, tag in points:
        for mode in STATION_FEEDBACK_EARLY_BRAKE_EXPERIMENT_MODES:
            group = by_key[(tag, mode)]
            curve = {
                "multiplier": float(multiplier),
                "tag": tag,
                "mode": mode,
                "arm": ARM_KEYS[mode],
                "mean_completed_orders": _mean(
                    row["completed_orders"] for row in group
                ),
                "mean_clearance_ratio": _mean(
                    row["clearance_ratio"] for row in group
                ),
                "collapse_rate": _mean(
                    float(bool(row["collapsed"])) for row in group
                ),
                "mean_deadlock_ratio": _mean(
                    row["deadlock_ratio_mean"] for row in group
                ),
                "mean_stall_ratio": _mean(
                    row["stall_ratio_mean"] for row in group
                ),
                "mean_capacity_rejections": _mean(
                    row["capacity_rejections"] for row in group
                ),
                "mean_dynamic_probe_steps": _mean(
                    row["dynamic_probe_steps"] for row in group
                ),
                "mean_forced_defers": _mean(
                    row["feedback_forced_defers"] for row in group
                ),
                "mean_early_brake_ticks": _mean(
                    row["early_reason_station_ticks"] for row in group
                ),
                "mean_recovery_ticks": _mean(
                    row["recovery_station_ticks"] for row in group
                ),
                "total_brake_to_recovery": sum(
                    row["brake_to_recovery"] for row in group
                ),
                "total_recovery_to_open": sum(
                    row["recovery_to_open"] for row in group
                ),
                "total_brake_to_locked": sum(
                    row["brake_to_locked"] for row in group
                ),
                "mean_locked_ticks": _mean(
                    row["locked_station_ticks"] for row in group
                ),
            }
            curve["sustainable"] = bool(
                curve["mean_clearance_ratio"] >= min_clearance
                and curve["collapse_rate"] <= max_collapse
            )
            result.append(curve)

        reference_rows = by_key[(tag, STATION_FEEDBACK_MODE_OFF)]
        completed = [
            float(row["reference_completed_orders"])
            for row in reference_rows
        ]
        clearance = [
            value / max(float(row["total_orders"]), 1.0)
            for value, row in zip(completed, reference_rows)
        ]
        collapsed = [
            value / max(float(row["empirical_completion_ceiling"]), 1.0)
            < 0.8
            for value, row in zip(completed, reference_rows)
        ]
        result.append({
            "multiplier": float(multiplier),
            "tag": tag,
            "mode": REFERENCE_KEY,
            "arm": REFERENCE_KEY,
            "mean_completed_orders": _mean(completed),
            "mean_clearance_ratio": _mean(clearance),
            "collapse_rate": _mean(float(value) for value in collapsed),
            "mean_deadlock_ratio": _mean(
                row["reference_deadlock_ratio_mean"] for row in reference_rows
            ),
            "mean_stall_ratio": _mean(
                row["reference_stall_ratio_mean"] for row in reference_rows
            ),
            "mean_capacity_rejections": _mean(
                row["reference_capacity_rejections"] for row in reference_rows
            ),
            "mean_dynamic_probe_steps": 0.0,
            "mean_forced_defers": 0.0,
            "mean_early_brake_ticks": 0.0,
            "mean_recovery_ticks": 0.0,
            "total_brake_to_recovery": 0,
            "total_recovery_to_open": 0,
            "total_brake_to_locked": 0,
            "mean_locked_ticks": 0.0,
            "sustainable": bool(
                _mean(clearance) >= min_clearance
                and _mean(float(value) for value in collapsed)
                <= max_collapse
            ),
        })
    return result


def _contrasts(rows, points, seeds):
    lookup = {
        (str(row["tag"]), str(row["mode"]), int(row["seed"])): row
        for row in rows
    }
    comparisons = (
        ("active_v2_minus_off", STATION_FEEDBACK_MODE_ACTIVE_V2,
         STATION_FEEDBACK_MODE_OFF),
        ("active_v2_minus_active_v1", STATION_FEEDBACK_MODE_ACTIVE_V2,
         STATION_FEEDBACK_MODE_ACTIVE),
        ("active_v1_minus_off", STATION_FEEDBACK_MODE_ACTIVE,
         STATION_FEEDBACK_MODE_OFF),
    )
    result = []
    for multiplier, tag in points:
        for name, left, right in comparisons:
            completed_deltas = []
            deadlock_deltas = []
            rejection_deltas = []
            wins = ties = losses = 0
            for seed in seeds:
                left_row = lookup[(tag, left, int(seed))]
                right_row = lookup[(tag, right, int(seed))]
                delta = (
                    float(left_row["completed_orders"])
                    - float(right_row["completed_orders"])
                )
                completed_deltas.append(delta)
                deadlock_deltas.append(
                    float(left_row["deadlock_ratio_mean"])
                    - float(right_row["deadlock_ratio_mean"])
                )
                rejection_deltas.append(
                    float(left_row["capacity_rejections"])
                    - float(right_row["capacity_rejections"])
                )
                wins += int(delta > 0)
                ties += int(delta == 0)
                losses += int(delta < 0)
            result.append({
                "multiplier": float(multiplier),
                "tag": tag,
                "comparison": name,
                "mean_completed_delta": _mean(completed_deltas),
                "mean_deadlock_delta": _mean(deadlock_deltas),
                "mean_capacity_rejection_delta": _mean(rejection_deltas),
                "wins": wins,
                "ties": ties,
                "losses": losses,
            })
    return result


def _critical(curves, points):
    result = {}
    modes = (*STATION_FEEDBACK_EARLY_BRAKE_EXPERIMENT_MODES, REFERENCE_KEY)
    for mode in modes:
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


def _markdown(summary: Mapping[str, Any]) -> str:
    integrity = summary["integrity"]
    lines = [
        "# Station-feedback early-BRAKE capacity frontier",
        "",
        "The early controller is opt-in and station-local. ETA V3 admission, "
        "World Model, Dynamic-J, pipeline+phi V2, S1, order generation, task "
        "lifecycle, and path planning remain unchanged.",
        "",
        "## Integrity",
        "",
        f"- Complete paired runs: {integrity['complete_runs']}",
        f"- All arm audits passed: {integrity['all_arm_audits_passed']}",
        f"- Manifest pairing passed: {integrity['manifest_pairing_passed']}",
        f"- Off/shadow_v2 equivalence: "
        f"{integrity['off_shadow_v2_equivalence_passed']}",
        f"- Physical capacity violations: "
        f"{integrity['physical_capacity_violations']}",
        "",
        "## Frontier curves",
        "",
        "| Multiplier | Mode | Completed | Clearance | Collapse | Deadlock | "
        "Early brake ticks | Recovery ticks | B->R | R->O | B->L | Sustainable |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["curves"]:
        lines.append(
            f"| {row['multiplier']:.2f} | {row['mode']} | "
            f"{row['mean_completed_orders']:.1f} | "
            f"{row['mean_clearance_ratio']:.3f} | "
            f"{row['collapse_rate']:.2f} | "
            f"{row['mean_deadlock_ratio']:.3f} | "
            f"{row['mean_early_brake_ticks']:.1f} | "
            f"{row['mean_recovery_ticks']:.1f} | "
            f"{row['total_brake_to_recovery']} | "
            f"{row['total_recovery_to_open']} | "
            f"{row['total_brake_to_locked']} | "
            f"{'yes' if row['sustainable'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Paired contrasts",
        "",
        "| Multiplier | Contrast | Completed delta | Deadlock delta | "
        "Rejection delta | W/T/L |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for row in summary["contrasts"]:
        lines.append(
            f"| {row['multiplier']:.2f} | {row['comparison']} | "
            f"{row['mean_completed_delta']:.1f} | "
            f"{row['mean_deadlock_delta']:.4f} | "
            f"{row['mean_capacity_rejection_delta']:.1f} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )
    lines.extend([
        "",
        "A useful early-BRAKE result requires observed BRAKE->RECOVERY and "
        "RECOVERY->OPEN transitions, fewer absorbing LOCKED transitions, and "
        "no paired throughput loss.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--load", default="high")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--points", nargs="+", required=True)
    parser.add_argument("--sim-ticks", type=int, default=1500)
    parser.add_argument("--min-mean-clearance-ratio", type=float, default=0.90)
    parser.add_argument("--max-collapse-rate", type=float, default=0.20)
    args = parser.parse_args()
    points = _parse_points(args.points)
    payloads, rows, physical_violations = _load_runs(args, points)
    expected = (
        len(points)
        * len(STATION_FEEDBACK_EARLY_BRAKE_EXPERIMENT_MODES)
        * len(args.seeds)
    )
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} runs, loaded {len(rows)}")
    integrity = _paired_integrity(payloads, points, args.seeds)
    if not integrity["manifest_pairing_passed"]:
        raise RuntimeError("paired manifests differ across early-BRAKE arms")
    if not integrity["off_shadow_v2_equivalence_passed"]:
        raise RuntimeError("shadow_v2 changed simulator behavior")
    if physical_violations:
        raise RuntimeError("physical station capacity violation detected")

    _mark_collapses(rows)
    curves = _curves(
        rows,
        points,
        float(args.min_mean_clearance_ratio),
        float(args.max_collapse_rate),
    )
    contrasts = _contrasts(rows, points, args.seeds)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "integrity": {
            "complete_runs": len(rows),
            "all_arm_audits_passed": True,
            "physical_capacity_violations": physical_violations,
            **integrity,
        },
        "config": {
            "load": args.load,
            "seeds": [int(seed) for seed in args.seeds],
            "points": [
                {"multiplier": multiplier, "tag": tag}
                for multiplier, tag in points
            ],
            "sim_ticks": int(args.sim_ticks),
            "modes": list(STATION_FEEDBACK_EARLY_BRAKE_EXPERIMENT_MODES),
            "min_mean_clearance_ratio": float(
                args.min_mean_clearance_ratio
            ),
            "max_collapse_rate": float(args.max_collapse_rate),
        },
        "curves": curves,
        "contrasts": contrasts,
        "critical_multiplier": _critical(curves, points),
    }
    validation = args.output_root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    _atomic_json(validation / "station_feedback_early_brake.json", summary)
    _write_csv(validation / "station_feedback_early_brake_per_run.csv", rows)
    _write_csv(validation / "station_feedback_early_brake_curves.csv", curves)
    _write_csv(
        validation / "station_feedback_early_brake_contrasts.csv", contrasts
    )
    markdown = _markdown(summary)
    (validation / "station_feedback_early_brake.md").write_text(
        markdown, encoding="utf-8"
    )
    print(markdown)


if __name__ == "__main__":
    main()


__all__ = [
    "SUMMARY_SCHEMA_VERSION",
    "_contrasts",
    "_paired_integrity",
]
