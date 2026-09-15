"""Analyze the paired dispatch-preserving ETA queue capacity frontier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from WorldModel.evaluation.analyze_phase_c_station_admission_capacity_frontier import (
    MANIFEST_CONTRACT_SCHEMA_VERSION,
    _atomic_json,
    _atomic_text,
    _max_mapping,
    _mean,
    _mean_ci95,
)


SCHEMA_VERSION = "phase_c_dispatch_preserving_eta_queue_frontier_summary_v1"
ARM_RESULT_SCHEMA_VERSION = "phase_c_dispatch_preserving_eta_queue_arm_v1"

POLICIES = ("s1_j1", "s0_j0", "greedy")
ADMISSIONS = ("eta_v3_control", "dispatch_eta_queue_v1")
ARM_LAYOUT = {
    f"{policy}_{admission}": (policy, admission)
    for policy in POLICIES
    for admission in ADMISSIONS
}
DEFAULT_ARMS = tuple(ARM_LAYOUT)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    return float(value)


def _result_path(
    output_root: Path, tag: str, arm: str, load: str, seed: int
) -> Path:
    return output_root / tag / "per_arm" / arm / f"{load}_seed{seed}.json"


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mean_metrics = (
        "total_orders",
        "completed_orders",
        "offered_orders_per_1000_arrival_ticks",
        "completed_orders_per_1000_sim_ticks",
        "clearance_ratio",
        "open_order_count",
        "open_order_ratio",
        "pending_order_count",
        "deadlock_ratio_mean",
        "stall_ratio_mean",
        "capacity_rejections",
        "waiting_assigned_ratio",
        "waiting_duration_p95_ticks",
        "waiting_promotions",
        "dispatch_queue_bypass_grants",
        "unresolved_waiter_count_final",
    )
    result: dict[str, Any] = {"run_count": len(rows)}
    for metric in mean_metrics:
        values = [
            float(row[metric])
            for row in rows
            if row.get(metric) is not None
        ]
        result[f"{metric}_mean"] = _mean(values)
        result[f"{metric}_ci95"] = _mean_ci95(values)
    result.update({
        "collapse_seed_count": sum(bool(row["collapsed"]) for row in rows),
        "collapse_rate": (
            sum(bool(row["collapsed"]) for row in rows) / len(rows)
            if rows else None
        ),
        "max_committed_load": max(
            (float(row["max_committed_load"]) for row in rows), default=0.0
        ),
        "max_occupancy": max(
            (float(row["max_occupancy"]) for row in rows), default=0.0
        ),
        "physical_capacity_violations_sum": sum(
            int(row["physical_capacity_violations"]) for row in rows
        ),
        "dynamic_hard_limit_violations_sum": sum(
            int(row["dynamic_hard_limit_violations"]) for row in rows
        ),
        "dispatch_priority_fallback_sum": sum(
            int(row["dispatch_priority_fallback_count"]) for row in rows
        ),
        "runs": [dict(row) for row in rows],
    })
    return result


DELTA_METRICS = (
    "completed_orders",
    "completed_orders_per_1000_sim_ticks",
    "clearance_ratio",
    "open_order_count",
    "deadlock_ratio_mean",
    "stall_ratio_mean",
    "capacity_rejections",
)


def _summarize_deltas(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"paired_seed_count": len(pairs)}
    for metric in DELTA_METRICS:
        values = [float(row[f"{metric}_delta"]) for row in pairs]
        result[f"{metric}_delta_mean"] = _mean(values)
        result[f"{metric}_delta_ci95"] = _mean_ci95(values)
    completed = [float(row["completed_orders_delta"]) for row in pairs]
    result["throughput_wins_ties_losses"] = {
        "wins": sum(value > 0 for value in completed),
        "ties": sum(value == 0 for value in completed),
        "losses": sum(value < 0 for value in completed),
    }
    result["pairs"] = [dict(row) for row in pairs]
    return result


def _paired_delta(
    left: Mapping[int, Mapping[str, Any]],
    right: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    pairs = []
    for seed in sorted(set(left) & set(right)):
        row = {"seed": int(seed)}
        for metric in DELTA_METRICS:
            row[f"{metric}_delta"] = (
                float(left[seed][metric]) - float(right[seed][metric])
            )
        row["left_collapsed"] = bool(left[seed]["collapsed"])
        row["right_collapsed"] = bool(right[seed]["collapsed"])
        pairs.append(row)
    result = _summarize_deltas(pairs)
    result["collapse_transition_counts"] = {
        "both_healthy": sum(
            not row["left_collapsed"] and not row["right_collapsed"]
            for row in pairs
        ),
        "left_only_healthy": sum(
            not row["left_collapsed"] and row["right_collapsed"]
            for row in pairs
        ),
        "right_only_healthy": sum(
            row["left_collapsed"] and not row["right_collapsed"]
            for row in pairs
        ),
        "both_collapsed": sum(
            row["left_collapsed"] and row["right_collapsed"]
            for row in pairs
        ),
    }
    return result


def _difference_in_differences(
    s1_queue: Mapping[int, Mapping[str, Any]],
    s1_control: Mapping[int, Mapping[str, Any]],
    baseline_queue: Mapping[int, Mapping[str, Any]],
    baseline_control: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    seeds = sorted(
        set(s1_queue)
        & set(s1_control)
        & set(baseline_queue)
        & set(baseline_control)
    )
    pairs = []
    for seed in seeds:
        row = {"seed": int(seed)}
        for metric in DELTA_METRICS:
            s1_effect = (
                float(s1_queue[seed][metric])
                - float(s1_control[seed][metric])
            )
            baseline_effect = (
                float(baseline_queue[seed][metric])
                - float(baseline_control[seed][metric])
            )
            row[f"{metric}_delta"] = s1_effect - baseline_effect
            row[f"s1_{metric}_queue_effect"] = s1_effect
            row[f"baseline_{metric}_queue_effect"] = baseline_effect
        pairs.append(row)
    return _summarize_deltas(pairs)


def _frontier(
    arm: str,
    points: Sequence[Mapping[str, Any]],
    curves: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    passing = []
    contiguous = None
    still_contiguous = True
    statuses = []
    for point in points:
        passed = bool(curves[str(point["tag"])][arm]["sustainable"])
        value = float(point["value"])
        statuses.append({
            "multiplier": value,
            "tag": str(point["tag"]),
            "passed": passed,
        })
        if passed:
            passing.append(value)
        if still_contiguous and passed:
            contiguous = value
        else:
            still_contiguous = False
    return {
        "contiguous_critical_multiplier": contiguous,
        "highest_passing_multiplier": max(passing) if passing else None,
        "points": statuses,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_ci(value: Any, digits: int = 3) -> str:
    if not isinstance(value, Sequence) or len(value) != 2:
        return "n/a"
    return f"[{float(value[0]):.{digits}f}, {float(value[1]):.{digits}f}]"


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Dispatch-preserving ETA queue capacity frontier",
        "",
        "ETA V3 control and the new queue use identical ETA capacity limits. "
        "The treatment adds only an explicit post-PICK waiting lifecycle "
        "ordered by each policy's own dispatch sequence.",
        "",
        "## Frontier curves",
        "",
        "| Load | Arm | Completed/1000 | Clearance | Collapse | Deadlock | Waiting | Rejections | Sustainable |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for point in summary["multipliers"]:
        tag = str(point["tag"])
        for arm, row in summary["curves"][tag].items():
            lines.append(
                f"| {float(point['value']):.2f} | {arm} | "
                f"{_fmt(row['completed_orders_per_1000_sim_ticks_mean'], 1)} | "
                f"{_fmt(row['clearance_ratio_mean'])} | "
                f"{_fmt(row['collapse_rate'], 2)} | "
                f"{_fmt(row['deadlock_ratio_mean_mean'])} | "
                f"{_fmt(row['waiting_assigned_ratio_mean'])} | "
                f"{_fmt(row['capacity_rejections_mean'], 1)} | "
                f"{'yes' if row['sustainable'] else 'no'} |"
            )

    lines.extend([
        "",
        "## Queue effect within policy",
        "",
        "Positive order delta means the new queue beats ETA V3 for the same policy.",
        "",
        "| Load | Policy | Orders delta | 95% CI | W/T/L | Deadlock delta | Open-order delta |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary["queue_effects"]:
        contrast = row["contrast"]
        wtl = contrast["throughput_wins_ties_losses"]
        lines.append(
            f"| {row['multiplier']:.2f} | {row['policy']} | "
            f"{_fmt(contrast['completed_orders_delta_mean'], 1)} | "
            f"{_fmt_ci(contrast['completed_orders_delta_ci95'], 1)} | "
            f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
            f"{_fmt(contrast['deadlock_ratio_mean_delta_mean'])} | "
            f"{_fmt(contrast['open_order_count_delta_mean'], 1)} |"
        )

    lines.extend([
        "",
        "## S1+J1 under the new queue",
        "",
        "| Load | Baseline | Orders delta | 95% CI | W/T/L | Deadlock delta |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for row in summary["new_queue_policy_contrasts"]:
        contrast = row["contrast"]
        wtl = contrast["throughput_wins_ties_losses"]
        lines.append(
            f"| {row['multiplier']:.2f} | {row['baseline']} | "
            f"{_fmt(contrast['completed_orders_delta_mean'], 1)} | "
            f"{_fmt_ci(contrast['completed_orders_delta_ci95'], 1)} | "
            f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
            f"{_fmt(contrast['deadlock_ratio_mean_delta_mean'])} |"
        )

    lines.extend([
        "",
        "## Difference in differences",
        "",
        "Positive values mean the queue helps S1+J1 more than it helps the baseline.",
        "",
        "| Load | Baseline | Orders DiD | 95% CI | Deadlock DiD | Clearance DiD |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for row in summary["difference_in_differences"]:
        contrast = row["contrast"]
        lines.append(
            f"| {row['multiplier']:.2f} | {row['baseline']} | "
            f"{_fmt(contrast['completed_orders_delta_mean'], 1)} | "
            f"{_fmt_ci(contrast['completed_orders_delta_ci95'], 1)} | "
            f"{_fmt(contrast['deadlock_ratio_mean_delta_mean'])} | "
            f"{_fmt(contrast['clearance_ratio_delta_mean'])} |"
        )

    lines.extend([
        "",
        "## Critical multiplier",
        "",
        "| Arm | Contiguous critical | Highest passing |",
        "|---|---:|---:|",
    ])
    for arm, row in summary["frontiers"].items():
        lines.append(
            f"| {arm} | {_fmt(row['contiguous_critical_multiplier'], 2)} | "
            f"{_fmt(row['highest_passing_multiplier'], 2)} |"
        )

    integrity = summary["integrity"]
    lines.extend([
        "",
        "## Integrity",
        "",
        f"- Missing results: {len(integrity['missing_results'])}",
        f"- All arm audits passed: {integrity['all_arm_audits_passed']}",
        f"- Manifest pairing passed: {integrity['manifest_pairing_passed']}",
        f"- Policy fingerprint pairing passed: {integrity['policy_pairing_passed']}",
        f"- Queue lifecycle audits passed: {integrity['queue_audits_passed']}",
        f"- Physical capacity violations: {integrity['physical_capacity_violations_all_runs']}",
        f"- Dynamic hard-limit violations: {integrity['dynamic_hard_limit_violations_all_runs']}",
        f"- Dispatch-priority fallback count: {integrity['dispatch_priority_fallback_all_runs']}",
        "",
        "The critical multiplier is a finite-horizon empirical operating "
        "frontier, not a proof of queueing-theoretic stability.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--load", choices=("low", "mid", "high"), default="high")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--sim-ticks", type=int, default=1500)
    parser.add_argument("--arrival-horizon-ticks", type=int, default=1500)
    parser.add_argument("--collapse-efficiency-ratio", type=float, default=0.80)
    parser.add_argument("--min-mean-clearance-ratio", type=float, default=0.90)
    parser.add_argument("--max-collapse-rate", type=float, default=0.20)
    args = parser.parse_args()

    if args.sim_ticks <= 0 or args.arrival_horizon_ticks <= 0:
        raise SystemExit("tick horizons must be positive")
    if len(set(args.arms)) != len(args.arms):
        raise SystemExit("--arms must be unique")
    unknown = [arm for arm in args.arms if arm not in ARM_LAYOUT]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    if set(args.arms) != set(DEFAULT_ARMS):
        raise SystemExit(
            "this paired analyzer requires all six control/treatment arms"
        )

    contract_path = args.source_root / "capacity_frontier_manifest_contract.json"
    contract = _read_json(contract_path)
    if contract.get("schema_version") != MANIFEST_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unexpected capacity-frontier manifest contract schema")
    if contract.get("load") != args.load:
        raise ValueError("manifest contract load does not match --load")
    if [int(seed) for seed in contract.get("seeds", [])] != [
        int(seed) for seed in args.seeds
    ]:
        raise ValueError("manifest contract seeds do not match --seeds")
    if int(contract.get("target_ticks", -1)) != args.arrival_horizon_ticks:
        raise ValueError("manifest target horizon does not match")
    multipliers = sorted(
        list(contract.get("multipliers") or []),
        key=lambda row: float(row["value"]),
    )
    if not multipliers:
        raise ValueError("manifest contract contains no multipliers")
    failed_prefixes = [
        row for row in (contract.get("nested_prefix_checks") or [])
        if not bool(row.get("previous_prefix_is_subset"))
    ]
    if failed_prefixes:
        raise ValueError("manifest contract contains failed prefix checks")

    raw: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    missing_results = []
    all_audits = True
    manifest_pairing = True
    queue_audits = True
    physical_violations = 0
    hard_limit_violations = 0
    dispatch_fallbacks = 0
    policy_fingerprints: dict[str, set[str]] = {}

    for point in multipliers:
        tag = str(point["tag"])
        raw[tag] = {arm: {} for arm in args.arms}
        for seed in args.seeds:
            expected_manifest = contract["per_seed"][str(seed)]["scaled"][tag]
            for arm in args.arms:
                path = _result_path(
                    args.output_root, tag, arm, args.load, int(seed)
                )
                if not path.is_file():
                    missing_results.append(path.as_posix())
                    continue
                payload = _read_json(path)
                if payload.get("schema_version") != ARM_RESULT_SCHEMA_VERSION:
                    raise ValueError(f"unexpected arm result schema in {path}")
                meta = payload.get("meta") or {}
                policy, admission = ARM_LAYOUT[arm]
                if (
                    meta.get("arm_key") != arm
                    or meta.get("policy_key") != policy
                    or meta.get("admission_key") != admission
                ):
                    raise ValueError(f"arm metadata mismatch in {path}")
                if (
                    meta.get("load") != args.load
                    or int(meta.get("seed", -1)) != int(seed)
                    or int(meta.get("ticks", -1)) != args.sim_ticks
                ):
                    raise ValueError(f"load/seed/tick mismatch in {path}")
                audit_passed = bool((payload.get("audit") or {}).get("passed"))
                all_audits &= audit_passed
                manifest = payload.get("manifest") or {}
                manifest_ok = (
                    manifest.get("content_sha256")
                    == expected_manifest["manifest_sha256"]
                    and int(manifest.get("total_orders", -1))
                    == int(expected_manifest["total_orders"])
                )
                manifest_pairing &= manifest_ok
                fingerprint = str(
                    ((meta.get("policy_contract") or {}).get(
                        "fingerprint_sha256"
                    ))
                )
                policy_fingerprints.setdefault(policy, set()).add(fingerprint)

                metrics = payload.get("metrics") or {}
                station = payload.get("station_admission_audit") or {}
                waiting = payload.get("dispatch_waiting_audit")
                queue_audits &= (
                    waiting is None
                    if admission == "eta_v3_control"
                    else bool((waiting or {}).get("passed"))
                )
                physical = int(
                    station.get("physical_capacity_violation_count", 0)
                )
                hard = int(
                    station.get("dynamic_hard_limit_violation_count", 0)
                )
                fallback = int(
                    (waiting or {}).get("dispatch_priority_fallback_count", 0)
                )
                physical_violations += physical
                hard_limit_violations += hard
                dispatch_fallbacks += fallback
                total_orders = int(expected_manifest["total_orders"])
                completed = _number(metrics.get("completed_orders"))
                open_orders = _number(
                    metrics.get("open_order_count"), total_orders - completed
                )
                row = {
                    "multiplier": float(point["value"]),
                    "tag": tag,
                    "arm": arm,
                    "policy": policy,
                    "admission": admission,
                    "seed": int(seed),
                    "total_orders": total_orders,
                    "completed_orders": completed,
                    "offered_orders_per_1000_arrival_ticks": (
                        1000.0 * total_orders / args.arrival_horizon_ticks
                    ),
                    "completed_orders_per_1000_sim_ticks": (
                        1000.0 * completed / args.sim_ticks
                    ),
                    "clearance_ratio": (
                        completed / total_orders if total_orders else 1.0
                    ),
                    "open_order_count": open_orders,
                    "open_order_ratio": (
                        open_orders / total_orders if total_orders else 0.0
                    ),
                    "pending_order_count": _number(
                        metrics.get("pending_order_count")
                    ),
                    "deadlock_ratio_mean": _number(
                        metrics.get("deadlock_ratio_mean")
                    ),
                    "stall_ratio_mean": _number(
                        metrics.get("stall_ratio_mean")
                    ),
                    "capacity_rejections": _number(
                        metrics.get("station_capacity_rejections")
                    ),
                    "waiting_assigned_ratio": _number(
                        metrics.get("waiting_assigned_ratio")
                    ),
                    "waiting_duration_p95_ticks": (
                        metrics.get("waiting_duration_p95_ticks")
                    ),
                    "waiting_promotions": _number(
                        metrics.get("waiting_promotions")
                    ),
                    "dispatch_queue_bypass_grants": _number(
                        metrics.get("dispatch_queue_bypass_grants")
                    ),
                    "unresolved_waiter_count_final": _number(
                        metrics.get("unresolved_waiter_count_final")
                    ),
                    "max_committed_load": _max_mapping(
                        station.get("max_committed_load")
                    ),
                    "max_occupancy": _max_mapping(
                        station.get("max_occupancy")
                    ),
                    "physical_capacity_violations": physical,
                    "dynamic_hard_limit_violations": hard,
                    "dispatch_priority_fallback_count": fallback,
                    "manifest_sha256": manifest.get("content_sha256"),
                    "audit_passed": audit_passed,
                    "path": path.as_posix(),
                }
                raw[tag][arm][int(seed)] = row

    if missing_results:
        raise RuntimeError(
            f"missing {len(missing_results)} results; first={missing_results[0]}"
        )
    policy_pairing = all(
        len(values) == 1 and "None" not in values and "" not in values
        for values in policy_fingerprints.values()
    )

    curves: dict[str, dict[str, Any]] = {}
    for point in multipliers:
        tag = str(point["tag"])
        curves[tag] = {}
        for seed in args.seeds:
            ceiling = max(
                raw[tag][arm][int(seed)]["completed_orders"]
                for arm in args.arms
            )
            for arm in args.arms:
                row = raw[tag][arm][int(seed)]
                row["empirical_completion_ceiling"] = ceiling
                row["completion_efficiency"] = (
                    row["completed_orders"] / ceiling if ceiling else 1.0
                )
                row["collapsed"] = bool(
                    row["completion_efficiency"]
                    < args.collapse_efficiency_ratio
                )
        for arm in args.arms:
            aggregate = _aggregate([
                raw[tag][arm][int(seed)] for seed in args.seeds
            ])
            aggregate["sustainable"] = bool(
                float(aggregate["clearance_ratio_mean"])
                >= args.min_mean_clearance_ratio
                and float(aggregate["collapse_rate"])
                <= args.max_collapse_rate
            )
            curves[tag][arm] = aggregate

    frontiers = {
        arm: _frontier(arm, multipliers, curves) for arm in args.arms
    }
    queue_effects = []
    new_queue_policy_contrasts = []
    did_rows = []
    for point in multipliers:
        tag = str(point["tag"])
        value = float(point["value"])
        for policy in POLICIES:
            queue_effects.append({
                "multiplier": value,
                "tag": tag,
                "policy": policy,
                "contrast": _paired_delta(
                    raw[tag][f"{policy}_dispatch_eta_queue_v1"],
                    raw[tag][f"{policy}_eta_v3_control"],
                ),
            })
        for baseline in ("s0_j0", "greedy"):
            new_queue_policy_contrasts.append({
                "multiplier": value,
                "tag": tag,
                "baseline": baseline,
                "contrast": _paired_delta(
                    raw[tag]["s1_j1_dispatch_eta_queue_v1"],
                    raw[tag][f"{baseline}_dispatch_eta_queue_v1"],
                ),
            })
            did_rows.append({
                "multiplier": value,
                "tag": tag,
                "baseline": baseline,
                "contrast": _difference_in_differences(
                    raw[tag]["s1_j1_dispatch_eta_queue_v1"],
                    raw[tag]["s1_j1_eta_v3_control"],
                    raw[tag][f"{baseline}_dispatch_eta_queue_v1"],
                    raw[tag][f"{baseline}_eta_v3_control"],
                ),
            })

    frontier_shifts = []
    for policy in POLICIES:
        control = frontiers[f"{policy}_eta_v3_control"][
            "contiguous_critical_multiplier"
        ]
        queue = frontiers[f"{policy}_dispatch_eta_queue_v1"][
            "contiguous_critical_multiplier"
        ]
        frontier_shifts.append({
            "policy": policy,
            "eta_v3_control": control,
            "dispatch_eta_queue_v1": queue,
            "shift": (
                float(queue) - float(control)
                if queue is not None and control is not None
                else None
            ),
        })

    summary = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "load": args.load,
            "seeds": [int(seed) for seed in args.seeds],
            "sim_ticks": int(args.sim_ticks),
            "arrival_horizon_ticks": int(args.arrival_horizon_ticks),
            "source_root": args.source_root.as_posix(),
            "output_root": args.output_root.as_posix(),
            "arms": list(args.arms),
            "run_count": len(args.arms) * len(args.seeds) * len(multipliers),
        },
        "thresholds": {
            "collapse_efficiency_ratio": float(
                args.collapse_efficiency_ratio
            ),
            "min_mean_clearance_ratio": float(
                args.min_mean_clearance_ratio
            ),
            "max_collapse_rate": float(args.max_collapse_rate),
        },
        "manifest_contract": {
            "path": contract_path.as_posix(),
            "contract_sha256": contract.get("contract_sha256"),
            "scaling_formula": contract.get("scaling_formula"),
        },
        "multipliers": multipliers,
        "curves": curves,
        "frontiers": frontiers,
        "frontier_shifts": frontier_shifts,
        "queue_effects": queue_effects,
        "new_queue_policy_contrasts": new_queue_policy_contrasts,
        "difference_in_differences": did_rows,
        "integrity": {
            "missing_results": missing_results,
            "all_arm_audits_passed": bool(all_audits),
            "manifest_pairing_passed": bool(manifest_pairing),
            "policy_pairing_passed": bool(policy_pairing),
            "policy_fingerprints": {
                policy: sorted(values)
                for policy, values in policy_fingerprints.items()
            },
            "queue_audits_passed": bool(queue_audits),
            "physical_capacity_violations_all_runs": int(
                physical_violations
            ),
            "dynamic_hard_limit_violations_all_runs": int(
                hard_limit_violations
            ),
            "dispatch_priority_fallback_all_runs": int(
                dispatch_fallbacks
            ),
        },
    }
    summary["integrity"]["passed"] = bool(
        all_audits
        and manifest_pairing
        and policy_pairing
        and queue_audits
        and physical_violations == 0
        and hard_limit_violations == 0
        and dispatch_fallbacks == 0
    )
    if not summary["integrity"]["passed"]:
        raise RuntimeError("dispatch-preserving frontier integrity checks failed")

    validation = args.output_root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    json_path = validation / "dispatch_preserving_eta_queue_frontier.json"
    markdown_path = validation / "dispatch_preserving_eta_queue_frontier.md"
    per_run_path = validation / "dispatch_preserving_eta_queue_per_run.csv"
    curves_path = validation / "dispatch_preserving_eta_queue_curves.csv"
    contrast_path = validation / "dispatch_preserving_eta_queue_contrasts.csv"
    did_path = validation / "dispatch_preserving_eta_queue_did.csv"
    _atomic_json(json_path, summary)
    _atomic_text(markdown_path, _markdown(summary))

    per_run_fields = [
        "multiplier", "tag", "arm", "policy", "admission", "seed",
        "total_orders", "completed_orders",
        "completed_orders_per_1000_sim_ticks", "clearance_ratio",
        "open_order_count", "deadlock_ratio_mean", "stall_ratio_mean",
        "capacity_rejections", "waiting_assigned_ratio",
        "waiting_duration_p95_ticks", "waiting_promotions",
        "dispatch_queue_bypass_grants", "unresolved_waiter_count_final",
        "max_committed_load", "max_occupancy", "completion_efficiency",
        "collapsed", "audit_passed", "path",
    ]
    with per_run_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_run_fields)
        writer.writeheader()
        for point in multipliers:
            tag = str(point["tag"])
            for arm in args.arms:
                for seed in args.seeds:
                    source = raw[tag][arm][int(seed)]
                    writer.writerow({key: source.get(key) for key in per_run_fields})

    curve_fields = [
        "multiplier", "tag", "arm", "run_count",
        "completed_orders_mean", "completed_orders_per_1000_sim_ticks_mean",
        "clearance_ratio_mean", "collapse_rate", "deadlock_ratio_mean_mean",
        "stall_ratio_mean_mean", "open_order_count_mean",
        "capacity_rejections_mean", "waiting_assigned_ratio_mean",
        "waiting_duration_p95_ticks_mean", "waiting_promotions_mean",
        "dispatch_queue_bypass_grants_mean", "sustainable",
    ]
    with curves_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_fields)
        writer.writeheader()
        for point in multipliers:
            tag = str(point["tag"])
            for arm in args.arms:
                writer.writerow({
                    "multiplier": point["value"],
                    "tag": tag,
                    "arm": arm,
                    **{
                        key: curves[tag][arm].get(key)
                        for key in curve_fields
                        if key not in ("multiplier", "tag", "arm")
                    },
                })

    contrast_fields = [
        "kind", "multiplier", "tag", "policy", "baseline",
        "completed_orders_delta_mean", "completed_orders_delta_ci95",
        "deadlock_ratio_mean_delta_mean", "clearance_ratio_delta_mean",
        "open_order_count_delta_mean", "wins", "ties", "losses",
    ]
    with contrast_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=contrast_fields)
        writer.writeheader()
        for kind, rows in (
            ("queue_effect", queue_effects),
            ("new_queue_policy", new_queue_policy_contrasts),
        ):
            for row in rows:
                contrast = row["contrast"]
                wtl = contrast["throughput_wins_ties_losses"]
                writer.writerow({
                    "kind": kind,
                    "multiplier": row["multiplier"],
                    "tag": row["tag"],
                    "policy": row.get("policy", "s1_j1"),
                    "baseline": row.get("baseline"),
                    "completed_orders_delta_mean": contrast.get(
                        "completed_orders_delta_mean"
                    ),
                    "completed_orders_delta_ci95": json.dumps(
                        contrast.get("completed_orders_delta_ci95")
                    ),
                    "deadlock_ratio_mean_delta_mean": contrast.get(
                        "deadlock_ratio_mean_delta_mean"
                    ),
                    "clearance_ratio_delta_mean": contrast.get(
                        "clearance_ratio_delta_mean"
                    ),
                    "open_order_count_delta_mean": contrast.get(
                        "open_order_count_delta_mean"
                    ),
                    **wtl,
                })

    did_fields = [
        "multiplier", "tag", "baseline",
        "completed_orders_delta_mean", "completed_orders_delta_ci95",
        "deadlock_ratio_mean_delta_mean", "clearance_ratio_delta_mean",
        "open_order_count_delta_mean",
    ]
    with did_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=did_fields)
        writer.writeheader()
        for row in did_rows:
            contrast = row["contrast"]
            writer.writerow({
                "multiplier": row["multiplier"],
                "tag": row["tag"],
                "baseline": row["baseline"],
                "completed_orders_delta_mean": contrast.get(
                    "completed_orders_delta_mean"
                ),
                "completed_orders_delta_ci95": json.dumps(
                    contrast.get("completed_orders_delta_ci95")
                ),
                "deadlock_ratio_mean_delta_mean": contrast.get(
                    "deadlock_ratio_mean_delta_mean"
                ),
                "clearance_ratio_delta_mean": contrast.get(
                    "clearance_ratio_delta_mean"
                ),
                "open_order_count_delta_mean": contrast.get(
                    "open_order_count_delta_mean"
                ),
            })

    print(json.dumps({
        "summary_json": json_path.as_posix(),
        "summary_markdown": markdown_path.as_posix(),
        "per_run_csv": per_run_path.as_posix(),
        "curves_csv": curves_path.as_posix(),
        "contrasts_csv": contrast_path.as_posix(),
        "did_csv": did_path.as_posix(),
        "integrity": summary["integrity"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
