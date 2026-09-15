"""Summarize the paired Phase-C station-admission capacity frontier.

The analyzer is intentionally independent of the simulator.  It audits the
scaled manifest contract and a selected subset of factorial arm JSON files,
then reports offered/achieved throughput curves, collapse probability,
paired policy contrasts, and an explicitly thresholded empirical critical
load multiplier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "phase_c_station_admission_capacity_frontier_summary_v1"
MANIFEST_CONTRACT_SCHEMA_VERSION = "phase_c_capacity_frontier_manifest_contract_v1"
ARM_RESULT_SCHEMA_VERSION = "phase_c_sj_admission_factorial_arm_v1"
DEFAULT_ARMS = (
    "s1_j1_physical_only",
    "s1_j1_committed_v1",
    "s1_j1_eta_v3",
    "s0_j0_physical_only",
    "s0_j0_committed_v1",
    "s0_j0_eta_v3",
    "greedy_physical_only",
    "greedy_committed_v1",
    "greedy_eta_v3",
)
ADMISSION_SUFFIXES = ("physical_only", "committed_v1", "eta_v3", "fifo_v2")
T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _mean(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values]
    return statistics.fmean(rows) if rows else None


def _mean_ci95(values: Iterable[float]) -> list[float] | None:
    rows = [float(value) for value in values]
    if not rows:
        return None
    mean = statistics.fmean(rows)
    if len(rows) < 2:
        return [mean, mean]
    critical = T_CRITICAL_975.get(len(rows) - 1, 1.96)
    half = critical * statistics.stdev(rows) / math.sqrt(len(rows))
    return [mean - half, mean + half]


def _max_mapping(value: Any) -> float:
    if not isinstance(value, Mapping) or not value:
        return 0.0
    return max(float(item) for item in value.values())


def _split_arm(arm: str) -> tuple[str, str]:
    for admission in sorted(ADMISSION_SUFFIXES, key=len, reverse=True):
        suffix = f"_{admission}"
        if arm.endswith(suffix):
            return arm[: -len(suffix)], admission
    raise ValueError(f"cannot parse admission suffix from arm: {arm}")


def _result_path(
    output_root: Path, tag: str, arm: str, load: str, seed: int
) -> Path:
    return output_root / tag / "per_arm" / arm / f"{load}_seed{seed}.json"


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {
        "total_orders": "total_orders",
        "completed_orders": "completed_orders",
        "offered_orders_per_1000_arrival_ticks": (
            "offered_orders_per_1000_arrival_ticks"
        ),
        "completed_orders_per_1000_sim_ticks": (
            "completed_orders_per_1000_sim_ticks"
        ),
        "clearance_ratio": "clearance_ratio",
        "open_order_count": "open_order_count",
        "open_order_ratio": "open_order_ratio",
        "pending_order_count": "pending_order_count",
        "deadlock_ratio_mean": "deadlock_ratio",
        "stall_ratio_mean": "stall_ratio",
        "capacity_rejections": "capacity_rejections",
    }
    result: dict[str, Any] = {"run_count": len(rows)}
    for source_metric, output_metric in metrics.items():
        values = [float(row[source_metric]) for row in rows]
        result[f"{output_metric}_mean"] = _mean(values)
        result[f"{output_metric}_ci95"] = _mean_ci95(values)
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
        "runs": [dict(row) for row in rows],
    })
    return result


def _paired_contrast(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left = {int(row["seed"]): row for row in left_rows}
    right = {int(row["seed"]): row for row in right_rows}
    seeds = sorted(set(left) & set(right))
    pairs = []
    for seed in seeds:
        completed_delta = (
            float(left[seed]["completed_orders"])
            - float(right[seed]["completed_orders"])
        )
        pairs.append({
            "seed": seed,
            "completed_orders_delta": completed_delta,
            "throughput_per_1000_delta": (
                float(left[seed]["completed_orders_per_1000_sim_ticks"])
                - float(right[seed]["completed_orders_per_1000_sim_ticks"])
            ),
            "clearance_ratio_delta": (
                float(left[seed]["clearance_ratio"])
                - float(right[seed]["clearance_ratio"])
            ),
            "deadlock_ratio_delta": (
                float(left[seed]["deadlock_ratio_mean"])
                - float(right[seed]["deadlock_ratio_mean"])
            ),
            "left_collapsed": bool(left[seed]["collapsed"]),
            "right_collapsed": bool(right[seed]["collapsed"]),
        })
    completed = [float(row["completed_orders_delta"]) for row in pairs]
    throughput = [float(row["throughput_per_1000_delta"]) for row in pairs]
    clearance = [float(row["clearance_ratio_delta"]) for row in pairs]
    deadlock = [float(row["deadlock_ratio_delta"]) for row in pairs]
    return {
        "paired_seed_count": len(pairs),
        "completed_orders_delta_mean": _mean(completed),
        "completed_orders_delta_ci95": _mean_ci95(completed),
        "throughput_per_1000_delta_mean": _mean(throughput),
        "throughput_per_1000_delta_ci95": _mean_ci95(throughput),
        "clearance_ratio_delta_mean": _mean(clearance),
        "clearance_ratio_delta_ci95": _mean_ci95(clearance),
        "deadlock_ratio_delta_mean": _mean(deadlock),
        "throughput_wins_ties_losses": {
            "wins": sum(value > 0 for value in completed),
            "ties": sum(value == 0 for value in completed),
            "losses": sum(value < 0 for value in completed),
        },
        "collapse_transition_counts": {
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
        },
        "pairs": pairs,
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
        "# Phase-C station-admission capacity frontier",
        "",
        "The same long base order stream is time-scaled at each multiplier. "
        "Higher multipliers select a longer nested prefix and compress it into "
        "the same arrival horizon.",
        "",
        "## Sustainability definition",
        "",
        f"- Mean clearance ratio >= {summary['thresholds']['min_mean_clearance_ratio']:.2f}",
        f"- Collapse rate <= {summary['thresholds']['max_collapse_rate']:.2f}",
        f"- Collapse means completed orders < {summary['thresholds']['collapse_efficiency_ratio']:.2f} "
        "of the best arm for the same multiplier/seed.",
        "- The reported contiguous critical multiplier stops at the first failed frontier point.",
        "",
        "## Critical multiplier",
        "",
        "| Arm | Contiguous critical | Highest passing |",
        "|---|---:|---:|",
    ]
    for arm, row in summary["frontiers"].items():
        lines.append(
            f"| {arm} | {_fmt(row['contiguous_critical_multiplier'], 2)} | "
            f"{_fmt(row['highest_passing_multiplier'], 2)} |"
        )
    lines.extend([
        "",
        "## Frontier curves",
        "",
        "| Multiplier | Arm | Offered/1000 | Completed/1000 | Clearance | Collapse | Deadlock | Open orders | Sustainable |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for point in summary["multipliers"]:
        tag = point["tag"]
        multiplier = point["value"]
        for arm, row in summary["curves"][tag].items():
            lines.append(
                f"| {multiplier:.2f} | {arm} | "
                f"{_fmt(row['offered_orders_per_1000_arrival_ticks_mean'], 1)} | "
                f"{_fmt(row['completed_orders_per_1000_sim_ticks_mean'], 1)} | "
                f"{_fmt(row['clearance_ratio_mean'], 3)} | "
                f"{_fmt(row['collapse_rate'], 2)} | "
                f"{_fmt(row['deadlock_ratio_mean'], 3)} | "
                f"{_fmt(row['open_order_count_mean'], 1)} | "
                f"{'yes' if row['sustainable'] else 'no'} |"
            )
    lines.extend([
        "",
        "## Paired S1+J1 contrasts",
        "",
        "| Multiplier | Admission | Baseline | Orders delta | 95% CI | W/T/L | Deadlock delta |",
        "|---:|---|---|---:|---:|---:|---:|",
    ])
    for row in summary["paired_contrasts"]:
        wtl = row["contrast"]["throughput_wins_ties_losses"]
        lines.append(
            f"| {row['multiplier']:.2f} | {row['admission']} | {row['baseline']} | "
            f"{_fmt(row['contrast']['completed_orders_delta_mean'], 1)} | "
            f"{_fmt_ci(row['contrast']['completed_orders_delta_ci95'], 1)} | "
            f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
            f"{_fmt(row['contrast']['deadlock_ratio_delta_mean'], 3)} |"
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
        f"- Physical capacity violations: {integrity['physical_capacity_violations_all_runs']}",
        "",
        "This finite-horizon critical multiplier is an empirical operating "
        "frontier under the stated thresholds, not a proof of queueing-theoretic stability.",
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
    if not 0 < args.collapse_efficiency_ratio <= 1:
        raise SystemExit("--collapse-efficiency-ratio must be in (0, 1]")
    if not 0 <= args.min_mean_clearance_ratio <= 1:
        raise SystemExit("--min-mean-clearance-ratio must be in [0, 1]")
    if not 0 <= args.max_collapse_rate <= 1:
        raise SystemExit("--max-collapse-rate must be in [0, 1]")
    if len(set(args.arms)) != len(args.arms):
        raise SystemExit("--arms must be unique")

    contract_path = args.source_root / "capacity_frontier_manifest_contract.json"
    contract = _read_json(contract_path)
    if contract.get("schema_version") != MANIFEST_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unexpected capacity-frontier manifest contract schema")
    if contract.get("load") != args.load:
        raise ValueError("manifest contract load does not match --load")
    if [int(seed) for seed in contract.get("seeds", [])] != [int(seed) for seed in args.seeds]:
        raise ValueError("manifest contract seeds do not match --seeds")
    if int(contract.get("target_ticks", -1)) != args.arrival_horizon_ticks:
        raise ValueError("manifest target horizon does not match --arrival-horizon-ticks")
    multipliers = list(contract.get("multipliers") or [])
    if not multipliers:
        raise ValueError("manifest contract contains no multipliers")
    tags = [str(point.get("tag")) for point in multipliers]
    values = [float(point.get("value")) for point in multipliers]
    if len(set(tags)) != len(tags) or len(set(values)) != len(values):
        raise ValueError("manifest contract multiplier tags and values must be unique")
    failed_prefixes = [
        row for row in (contract.get("nested_prefix_checks") or [])
        if not bool(row.get("previous_prefix_is_subset"))
    ]
    if failed_prefixes:
        raise ValueError("manifest contract contains failed nested-prefix checks")
    if contract.get("unit_reference_required"):
        for seed in args.seeds:
            unit_rows = [
                row
                for row in contract["per_seed"][str(seed)]["scaled"].values()
                if float(row.get("multiplier", -1.0)) == 1.0
            ]
            if len(unit_rows) != 1 or not bool(
                (unit_rows[0].get("unit_reference") or {}).get("passed")
            ):
                raise ValueError(f"required 1.0x reference check failed for seed {seed}")
    for arm in args.arms:
        _split_arm(arm)

    raw: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    missing_results = []
    all_audits = True
    manifest_pairing = True
    physical_violations = 0
    policy_fingerprints: dict[str, set[str]] = {}

    for point in multipliers:
        tag = str(point["tag"])
        raw[tag] = {arm: {} for arm in args.arms}
        for seed in args.seeds:
            expected_manifest = contract["per_seed"][str(seed)]["scaled"][tag]
            for arm in args.arms:
                path = _result_path(args.output_root, tag, arm, args.load, int(seed))
                if not path.is_file():
                    missing_results.append(path.as_posix())
                    continue
                payload = _read_json(path)
                if payload.get("schema_version") != ARM_RESULT_SCHEMA_VERSION:
                    raise ValueError(f"unexpected arm result schema in {path}")
                audit = payload.get("audit") or {}
                passed = bool(audit.get("passed"))
                all_audits &= passed
                manifest = payload.get("manifest") or {}
                manifest_ok = (
                    manifest.get("content_sha256")
                    == expected_manifest["manifest_sha256"]
                    and int(manifest.get("total_orders", -1))
                    == int(expected_manifest["total_orders"])
                )
                manifest_pairing &= manifest_ok
                meta = payload.get("meta") or {}
                if meta.get("arm_key") != arm:
                    raise ValueError(f"arm mismatch in {path}")
                if meta.get("load") != args.load or int(meta.get("seed", -1)) != int(seed):
                    raise ValueError(f"load/seed mismatch in {path}")
                if int(meta.get("ticks", -1)) != args.sim_ticks:
                    raise ValueError(f"tick mismatch in {path}")
                policy, admission = _split_arm(arm)
                fingerprint = str(
                    ((meta.get("policy_contract") or {}).get("fingerprint_sha256"))
                )
                policy_fingerprints.setdefault(policy, set()).add(fingerprint)
                metrics = payload.get("metrics") or {}
                station = payload.get("station_admission_audit") or {}
                violations = int(station.get("physical_capacity_violation_count", 0))
                physical_violations += violations
                total_orders = int(expected_manifest["total_orders"])
                completed = float(metrics.get("completed_orders", 0.0))
                open_orders = float(metrics.get("open_order_count", total_orders - completed))
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
                    "pending_order_count": float(metrics.get("pending_order_count", 0.0)),
                    "deadlock_ratio_mean": float(metrics.get("deadlock_ratio_mean", 0.0)),
                    "stall_ratio_mean": float(metrics.get("stall_ratio_mean", 0.0)),
                    "capacity_rejections": float(
                        metrics.get("station_capacity_rejections", 0.0)
                    ),
                    "max_committed_load": _max_mapping(station.get("max_committed_load")),
                    "max_occupancy": _max_mapping(station.get("max_occupancy")),
                    "physical_capacity_violations": violations,
                    "manifest_sha256": manifest.get("content_sha256"),
                    "audit_passed": passed,
                    "path": path.as_posix(),
                }
                raw[tag][arm][int(seed)] = row

    if missing_results:
        raise RuntimeError(
            f"missing {len(missing_results)} frontier results; first={missing_results[0]}"
        )
    policy_pairing = all(
        len(fingerprints) == 1 and "None" not in fingerprints
        for fingerprints in policy_fingerprints.values()
    )

    curves: dict[str, dict[str, Any]] = {}
    for point in multipliers:
        tag = str(point["tag"])
        curves[tag] = {}
        for seed in args.seeds:
            ceiling = max(
                raw[tag][arm][int(seed)]["completed_orders"] for arm in args.arms
            )
            for arm in args.arms:
                row = raw[tag][arm][int(seed)]
                row["empirical_completion_ceiling"] = ceiling
                row["completion_efficiency"] = (
                    row["completed_orders"] / ceiling if ceiling else 1.0
                )
                row["collapsed"] = (
                    row["completion_efficiency"]
                    < args.collapse_efficiency_ratio
                )
        for arm in args.arms:
            aggregate = _aggregate(
                [raw[tag][arm][int(seed)] for seed in args.seeds]
            )
            aggregate["sustainable"] = bool(
                float(aggregate["clearance_ratio_mean"])
                >= args.min_mean_clearance_ratio
                and float(aggregate["collapse_rate"])
                <= args.max_collapse_rate
            )
            curves[tag][arm] = aggregate

    frontiers: dict[str, Any] = {}
    sorted_points = sorted(multipliers, key=lambda row: float(row["value"]))
    for arm in args.arms:
        passing = [
            float(point["value"])
            for point in sorted_points
            if curves[str(point["tag"])][arm]["sustainable"]
        ]
        contiguous = None
        point_status = []
        still_contiguous = True
        for point in sorted_points:
            passed = bool(curves[str(point["tag"])][arm]["sustainable"])
            point_status.append({
                "multiplier": float(point["value"]),
                "tag": str(point["tag"]),
                "passed": passed,
            })
            if still_contiguous and passed:
                contiguous = float(point["value"])
            else:
                still_contiguous = False
        frontiers[arm] = {
            "contiguous_critical_multiplier": contiguous,
            "highest_passing_multiplier": max(passing) if passing else None,
            "points": point_status,
        }

    admissions = sorted({_split_arm(arm)[1] for arm in args.arms})
    paired_contrasts = []
    for point in sorted_points:
        tag = str(point["tag"])
        for admission in admissions:
            left_arm = f"s1_j1_{admission}"
            if left_arm not in args.arms:
                continue
            for baseline in ("s0_j0", "greedy"):
                right_arm = f"{baseline}_{admission}"
                if right_arm not in args.arms:
                    continue
                paired_contrasts.append({
                    "multiplier": float(point["value"]),
                    "tag": tag,
                    "admission": admission,
                    "left": "s1_j1",
                    "baseline": baseline,
                    "contrast": _paired_contrast(
                        [raw[tag][left_arm][int(seed)] for seed in args.seeds],
                        [raw[tag][right_arm][int(seed)] for seed in args.seeds],
                    ),
                })

    frontier_shifts = []
    for admission in admissions:
        left_arm = f"s1_j1_{admission}"
        if left_arm not in frontiers:
            continue
        for baseline in ("s0_j0", "greedy"):
            right_arm = f"{baseline}_{admission}"
            if right_arm not in frontiers:
                continue
            left_value = frontiers[left_arm]["contiguous_critical_multiplier"]
            right_value = frontiers[right_arm]["contiguous_critical_multiplier"]
            frontier_shifts.append({
                "admission": admission,
                "left": "s1_j1",
                "baseline": baseline,
                "s1_j1_contiguous_critical_multiplier": left_value,
                "baseline_contiguous_critical_multiplier": right_value,
                "shift": (
                    float(left_value) - float(right_value)
                    if left_value is not None and right_value is not None
                    else None
                ),
            })

    manifest_counts = {}
    for point in sorted_points:
        tag = str(point["tag"])
        counts = [
            int(contract["per_seed"][str(seed)]["scaled"][tag]["total_orders"])
            for seed in args.seeds
        ]
        manifest_counts[tag] = {
            "multiplier": float(point["value"]),
            "mean_total_orders": _mean(counts),
            "min_total_orders": min(counts),
            "max_total_orders": max(counts),
            "per_seed": {str(seed): count for seed, count in zip(args.seeds, counts)},
        }

    summary: dict[str, Any] = {
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
            "collapse_efficiency_ratio": float(args.collapse_efficiency_ratio),
            "min_mean_clearance_ratio": float(args.min_mean_clearance_ratio),
            "max_collapse_rate": float(args.max_collapse_rate),
        },
        "manifest_contract": {
            "path": contract_path.as_posix(),
            "contract_sha256": contract.get("contract_sha256"),
            "scaling_formula": contract.get("scaling_formula"),
        },
        "multipliers": sorted_points,
        "manifest_counts": manifest_counts,
        "curves": curves,
        "frontiers": frontiers,
        "frontier_shifts": frontier_shifts,
        "paired_contrasts": paired_contrasts,
        "integrity": {
            "missing_results": missing_results,
            "all_arm_audits_passed": bool(all_audits),
            "manifest_pairing_passed": bool(manifest_pairing),
            "policy_pairing_passed": bool(policy_pairing),
            "policy_fingerprints": {
                policy: sorted(values)
                for policy, values in policy_fingerprints.items()
            },
            "physical_capacity_violations_all_runs": int(physical_violations),
        },
    }
    summary["integrity"]["passed"] = bool(
        all_audits
        and manifest_pairing
        and policy_pairing
        and physical_violations == 0
    )

    validation = args.output_root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    json_path = validation / "station_admission_capacity_frontier.json"
    markdown_path = validation / "station_admission_capacity_frontier.md"
    per_run_path = validation / "station_admission_capacity_frontier_per_run.csv"
    curve_path = validation / "station_admission_capacity_frontier_curves.csv"
    contrast_path = validation / "station_admission_capacity_frontier_contrasts.csv"
    _atomic_json(json_path, summary)
    _atomic_text(markdown_path, _markdown(summary))

    per_run_fields = [
        "multiplier", "tag", "arm", "policy", "admission", "seed",
        "total_orders", "completed_orders",
        "offered_orders_per_1000_arrival_ticks",
        "completed_orders_per_1000_sim_ticks", "clearance_ratio",
        "open_order_count", "open_order_ratio", "pending_order_count",
        "deadlock_ratio_mean", "stall_ratio_mean", "capacity_rejections",
        "max_committed_load", "max_occupancy",
        "empirical_completion_ceiling", "completion_efficiency", "collapsed",
        "physical_capacity_violations", "manifest_sha256", "audit_passed", "path",
    ]
    with per_run_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_run_fields)
        writer.writeheader()
        for point in sorted_points:
            tag = str(point["tag"])
            for arm in args.arms:
                for seed in args.seeds:
                    writer.writerow({key: raw[tag][arm][int(seed)].get(key) for key in per_run_fields})

    curve_fields = [
        "multiplier", "tag", "arm", "run_count",
        "total_orders_mean", "completed_orders_mean",
        "offered_orders_per_1000_arrival_ticks_mean",
        "completed_orders_per_1000_sim_ticks_mean", "clearance_ratio_mean",
        "open_order_count_mean", "open_order_ratio_mean",
        "deadlock_ratio_mean", "stall_ratio_mean",
        "capacity_rejections_mean", "collapse_seed_count", "collapse_rate",
        "max_committed_load", "max_occupancy", "sustainable",
    ]
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_fields)
        writer.writeheader()
        for point in sorted_points:
            tag = str(point["tag"])
            for arm in args.arms:
                aggregate = curves[tag][arm]
                writer.writerow({
                    "multiplier": float(point["value"]),
                    "tag": tag,
                    "arm": arm,
                    **{key: aggregate.get(key) for key in curve_fields[3:]},
                })

    contrast_fields = [
        "multiplier", "tag", "admission", "left", "baseline",
        "completed_orders_delta_mean", "completed_orders_ci95_low",
        "completed_orders_ci95_high", "throughput_per_1000_delta_mean",
        "clearance_ratio_delta_mean", "deadlock_ratio_delta_mean",
        "wins", "ties", "losses",
    ]
    with contrast_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=contrast_fields)
        writer.writeheader()
        for row in paired_contrasts:
            contrast = row["contrast"]
            ci = contrast["completed_orders_delta_ci95"] or [None, None]
            wtl = contrast["throughput_wins_ties_losses"]
            writer.writerow({
                "multiplier": row["multiplier"],
                "tag": row["tag"],
                "admission": row["admission"],
                "left": row["left"],
                "baseline": row["baseline"],
                "completed_orders_delta_mean": contrast["completed_orders_delta_mean"],
                "completed_orders_ci95_low": ci[0],
                "completed_orders_ci95_high": ci[1],
                "throughput_per_1000_delta_mean": contrast["throughput_per_1000_delta_mean"],
                "clearance_ratio_delta_mean": contrast["clearance_ratio_delta_mean"],
                "deadlock_ratio_delta_mean": contrast["deadlock_ratio_delta_mean"],
                "wins": wtl["wins"],
                "ties": wtl["ties"],
                "losses": wtl["losses"],
            })

    print(json.dumps({
        "summary_json": json_path.as_posix(),
        "summary_markdown": markdown_path.as_posix(),
        "per_run_csv": per_run_path.as_posix(),
        "curves_csv": curve_path.as_posix(),
        "contrasts_csv": contrast_path.as_posix(),
        "frontier_shifts": frontier_shifts,
        "integrity": summary["integrity"],
    }, indent=2, ensure_ascii=False))
    if not summary["integrity"]["passed"]:
        raise RuntimeError(
            "capacity-frontier integrity checks failed; inspect validation outputs"
        )


if __name__ == "__main__":
    main()
