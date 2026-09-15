"""Replay aligned analytic Layer-5 snapshots under isolated fixed-context truth.

The replay does not run future arrivals or a continuation scheduler.  It asks
whether, at the exact online decision state, analytic-A agrees with true H=5
work drift and the pure-WM/fused ranking agrees with native-H=10 system cost.
The resulting
priority list is the input-selection mechanism for a later Phase-C World
Model dataset; it does not train another head.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pickle
import random
from typing import Mapping, Sequence

import numpy as np
import torch

from WorldModel.data.counterfactual_rollout import evaluate_candidate_rollout
from WorldModel.core.analytic_work_relief import extract_work_horizon_endpoint
from WorldModel.graph_builder import build_static_graph
from WorldState.order_state import Order
from WorldState.task_state import Task


SCHEMA_VERSION = "analytic_work_layer5_isolated_replay_v1"


def _scalar(value) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _argmin(rows: Sequence[Mapping], key: str) -> int:
    return int(min(rows, key=lambda row: (float(row[key]), int(row["robot_id"])))[
        "robot_id"
    ])


def _normalised_regret(
    rows: Sequence[Mapping], robot_id: int, key: str
) -> float:
    values = [float(row[key]) for row in rows]
    selected = next(
        float(row[key]) for row in rows if int(row["robot_id"]) == int(robot_id)
    )
    value_range = max(values) - min(values)
    if value_range == 0.0:
        return 0.0
    return float((selected - min(values)) / value_range)


def _pairwise_concordance(
    rows: Sequence[Mapping], predictor: str, target: str
) -> float | None:
    concordant = 0.0
    pairs = 0
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            target_delta = float(rows[left][target]) - float(rows[right][target])
            if target_delta == 0.0:
                continue
            pred_delta = (
                float(rows[left][predictor]) - float(rows[right][predictor])
            )
            if pred_delta == 0.0:
                concordant += 0.5
            elif pred_delta * target_delta > 0.0:
                concordant += 1.0
            pairs += 1
    return float(concordant / pairs) if pairs else None


def _future_system_endpoint_metrics(result: Mapping) -> dict:
    future = result.get("future_system_labels")
    if future is None or len(future) == 0:
        return {}
    endpoint = future[-1]
    return {
        "endpoint_wait_or_stall": _scalar(endpoint[0]),
        "endpoint_average_excess_delay": _scalar(endpoint[1]),
        "endpoint_station_pressure": _scalar(endpoint[2] + endpoint[3]),
        "endpoint_station_queue_delta": _scalar(endpoint[2]),
        "endpoint_station_load_imbalance": _scalar(endpoint[3]),
        "endpoint_bottleneck_cvar": _scalar(endpoint[4]),
    }


def _process_snapshot(
    arguments: tuple[str, dict, int, int, bool]
) -> dict | None:
    (
        path_value,
        lyapunov_config,
        analytic_horizon,
        system_horizon,
        only_modified,
    ) = arguments
    path = Path(path_value)
    with path.open("rb") as handle:
        snapshot = pickle.load(handle)
    online = snapshot.get("online_decision") or {}
    if not bool(snapshot.get("online_decision_trace_aligned")):
        raise ValueError(f"unaligned decision snapshot: {path}")
    modified = bool(online.get("work_drift_modified_decision"))
    if only_modified and not modified:
        return None
    if int(online.get("work_drift_horizon", -1)) != int(analytic_horizon):
        raise ValueError(
            f"snapshot does not use analytic H={analytic_horizon}: {path}"
        )

    world = snapshot["world_snapshot"]
    (
        _edge_index,
        node_map,
        _inv_node_map,
        local_capacity,
        bottleneck_score,
        _node_type,
        adj,
    ) = build_static_graph(world.map_state)
    trace_by_robot = {
        int(row["robot_id"]): row for row in online.get("candidates", ())
    }

    saved_task_id = Task._next_id
    saved_order_id = Order._next_id
    saved_python_rng = random.getstate()
    saved_numpy_rng = np.random.get_state()
    saved_torch_rng = torch.random.get_rng_state()
    candidate_rows = []
    try:
        for candidate in snapshot["candidates"]:
            robot_id = int(candidate["robot_id"])
            traced = trace_by_robot.get(robot_id)
            if traced is None:
                raise ValueError(
                    f"candidate {robot_id} is absent from online trace: {path}"
                )
            Task._next_id = int(snapshot["task_next_id"])
            Order._next_id = int(snapshot["order_next_id"])
            random.setstate(snapshot["python_rng_state"])
            np.random.set_state(snapshot["numpy_rng_state"])
            torch.random.set_rng_state(snapshot["torch_rng_state"])
            result = evaluate_candidate_rollout(
                world,
                candidate,
                snapshot["fixed_context"],
                snapshot["config"],
                snapshot["path_planner_state"],
                int(system_horizon),
                node_map,
                local_capacity,
                bottleneck_score,
                adj,
                reservation_window=int(
                    lyapunov_config.get("reservation_window", 10)
                ),
                record_lyapunov_l0=True,
                lyapunov_l0_config=lyapunov_config,
                rollout_continuation_mode="isolated",
            )
            mask = result.get("future_mask")
            if mask is None or not bool((mask > 0.5).all()):
                return {
                    "_replay_skipped": True,
                    "snapshot": path.as_posix(),
                    "reason": (
                        "right_censored_or_invalid_candidate:"
                        f"robot={robot_id}"
                    ),
                }
            start = result["lyapunov_l0_start"]
            try:
                analytic_endpoint = extract_work_horizon_endpoint(
                    result, int(analytic_horizon)
                )
            except (KeyError, TypeError, ValueError) as exc:
                return {
                    "_replay_skipped": True,
                    "snapshot": path.as_posix(),
                    "reason": f"invalid_work_trajectory:{type(exc).__name__}",
                }
            candidate_rows.append({
                "robot_id": robot_id,
                "wm_score": float(traced["score"]),
                "analytic_work_drift": float(traced["work_drift_raw"]),
                "combined_score": float(
                    traced["work_drift_combined_score"]
                ),
                "true_work_drift_h5": float(
                    analytic_endpoint["components"]["work"]
                    - start["components"]["work"]
                ),
                "true_realized_cost_h10": _scalar(result["realized_cost"]),
                "rollout_blocked_moves": int(result["rollout_blocked_moves"]),
                "rollout_vertex_conflicts": int(
                    result["rollout_vertex_conflicts"]
                ),
                "rollout_swap_conflicts": int(
                    result["rollout_swap_conflicts"]
                ),
                **_future_system_endpoint_metrics(result),
            })
    finally:
        Task._next_id = saved_task_id
        Order._next_id = saved_order_id
        random.setstate(saved_python_rng)
        np.random.set_state(saved_numpy_rng)
        torch.random.set_rng_state(saved_torch_rng)

    if len(candidate_rows) < 2:
        return {
            "_replay_skipped": True,
            "snapshot": path.as_posix(),
            "reason": "fewer_than_two_valid_candidates",
        }
    baseline_robot = int(online["baseline_robot"])
    selected_robot = int(online["selected_robot"])
    analytic_robot = _argmin(candidate_rows, "analytic_work_drift")
    true_work_robot = _argmin(candidate_rows, "true_work_drift_h5")
    true_cost_robot = _argmin(candidate_rows, "true_realized_cost_h10")
    baseline_row = next(
        row for row in candidate_rows if row["robot_id"] == baseline_robot
    )
    selected_row = next(
        row for row in candidate_rows if row["robot_id"] == selected_robot
    )
    analytic_row = next(
        row for row in candidate_rows if row["robot_id"] == analytic_robot
    )

    work_change = (
        selected_row["true_work_drift_h5"]
        - baseline_row["true_work_drift_h5"]
    )
    cost_change = (
        selected_row["true_realized_cost_h10"]
        - baseline_row["true_realized_cost_h10"]
    )
    tolerance = 1e-12
    min_true_work = min(row["true_work_drift_h5"] for row in candidate_rows)
    min_true_cost = min(row["true_realized_cost_h10"] for row in candidate_rows)
    analytic_top1_matches = (
        analytic_row["true_work_drift_h5"] <= min_true_work + tolerance
    )
    wm_top1_matches = (
        baseline_row["true_realized_cost_h10"] <= min_true_cost + tolerance
    )
    if not modified:
        category = "NO_FUSION_CHANGE_CONTROL"
    elif not analytic_top1_matches:
        category = "ANALYTIC_RANKING_MISMATCH"
    elif work_change > tolerance:
        category = "FUSION_SELECTED_WORSE_WORK_DESPITE_ANALYTIC_RANKING"
    elif cost_change > tolerance:
        category = "WORK_IMPROVES_BUT_H10_SYSTEM_COST_WORSENS"
    else:
        category = "ANALYTIC_SELECTION_H5_FAVOURABLE"

    reasons = []
    if len(candidate_rows) > 10:
        reasons.append("candidate_count_above_training_top_m")
    if modified:
        reasons.append("analytic_modified_selection")
    if modified and cost_change > tolerance:
        reasons.append("harmful_combined_selection_under_isolated_truth")
    if not wm_top1_matches:
        reasons.append("wm_ranking_error_under_isolated_truth")
    if not analytic_top1_matches:
        reasons.append("analytic_work_ranking_error")
    if int(snapshot["decision_tick"]) > 200:
        # The original Phase-B collection arms were 200 ticks long.  This is
        # a support-shift flag for Phase-C selection, not by itself proof of
        # statistical OOD.
        reasons.append("long_run_tail_state")
    priority = (
        4 * int("harmful_combined_selection_under_isolated_truth" in reasons)
        + 3 * int("wm_ranking_error_under_isolated_truth" in reasons)
        + 2 * int("analytic_work_ranking_error" in reasons)
        + int("candidate_count_above_training_top_m" in reasons)
        + int("long_run_tail_state" in reasons)
    )
    return {
        "snapshot": path.as_posix(),
        "run_id": snapshot.get("run_id"),
        "arm_label": snapshot.get("arm_label"),
        "seed": snapshot.get("seed"),
        "load": (snapshot.get("online_decision") or {}).get("load")
        or snapshot.get("load")
        or (snapshot.get("online_decision") or {}).get("config_path"),
        "decision_tick": int(snapshot["decision_tick"]),
        "order_id": int(snapshot["fixed_context"]["order_id"]),
        "pod_id": int(snapshot["fixed_context"]["pod_id"]),
        "candidate_group_id": snapshot["candidate_group_id"],
        "candidate_count": len(candidate_rows),
        "baseline_robot": baseline_robot,
        "analytic_robot": analytic_robot,
        "selected_robot": selected_robot,
        "true_work_robot": true_work_robot,
        "true_cost_robot": true_cost_robot,
        "modified": modified,
        "analytic_top1_matches_true_work": analytic_top1_matches,
        "wm_top1_matches_true_cost": wm_top1_matches,
        "category": category,
        "selected_minus_baseline_true_work_drift": float(work_change),
        "selected_minus_baseline_true_realized_cost": float(cost_change),
        "wm_pairwise_vs_true_cost": _pairwise_concordance(
            candidate_rows, "wm_score", "true_realized_cost_h10"
        ),
        "analytic_pairwise_vs_true_work": _pairwise_concordance(
            candidate_rows, "analytic_work_drift", "true_work_drift_h5"
        ),
        "combined_pairwise_vs_true_cost": _pairwise_concordance(
            candidate_rows, "combined_score", "true_realized_cost_h10"
        ),
        "wm_normalised_cost_regret": _normalised_regret(
            candidate_rows, baseline_robot, "true_realized_cost_h10"
        ),
        "combined_normalised_cost_regret": _normalised_regret(
            candidate_rows, selected_robot, "true_realized_cost_h10"
        ),
        "phase_c_priority_score": int(priority),
        "phase_c_reasons": reasons,
        "candidates": candidate_rows,
    }


def _mean(rows: Sequence[Mapping], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(sum(values) / len(values)) if values else None


def _online_summary(path: str | None) -> dict | None:
    if not path:
        return None
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    aggregate = report.get("aggregate") or {}
    baseline = aggregate.get("WorldModel") or {}
    analytic = aggregate.get("WorldModel+AnalyticWorkH5") or {}
    if not baseline or not analytic:
        raise ValueError("paired report lacks analytic Layer-5 arm labels")
    return {
        "wm_label_cost_improvement": float(
            baseline["wm_label_cost_mean"] - analytic["wm_label_cost_mean"]
        ),
        "completed_orders_improvement": float(
            analytic["completed_orders_mean"]
            - baseline["completed_orders_mean"]
        ),
        "open_order_count_reduction": float(
            baseline["open_order_count_mean"]
            - analytic["open_order_count_mean"]
        ),
    }


def _future_trace_index(snapshot_root: Path) -> dict[tuple[int, int, int, int], dict]:
    traces_root = snapshot_root.parent / "decision_traces"
    index = {}
    if not traces_root.is_dir():
        return index
    for path in sorted(traces_root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("assigner") != "WorldModel+AnalyticWorkH5":
                    continue
                key = (
                    int(row["seed"]),
                    int(row["tick"]),
                    int(row["order_id"]),
                    int(row["pod_id"]),
                )
                index[key] = {
                    name: value
                    for name, value in row.items()
                    if str(name).startswith("future_")
                }
    return index


def _selected_snapshot_files(
    root: Path,
    *,
    only_modified: bool,
    max_snapshots: int | None,
) -> tuple[list[Path], int]:
    """Resolve aligned files, using snapindex metadata before replay.

    When ``only_modified`` is requested, interval controls are removed before
    the expensive simulator replay.  A cap is sampled uniformly over time
    rather than taking an early-run prefix.
    """

    all_files = sorted(root.rglob("*.pkl"))
    available = len(all_files)
    files = all_files
    if only_modified:
        indexed = []
        for index_path in sorted(root.rglob("snapindex_*.json")):
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for row in index.get("decisions", ()):
                if row.get("snapshot_selection_reason") != (
                    "analytic_modified_selection"
                ):
                    continue
                path = index_path.parent / str(row["file"])
                if path.is_file():
                    indexed.append(path)
        files = sorted(set(indexed))
    if max_snapshots is not None and len(files) > max(0, int(max_snapshots)):
        limit = max(0, int(max_snapshots))
        if limit == 0:
            files = []
        elif limit == 1:
            files = [files[len(files) // 2]]
        else:
            indices = np.linspace(0, len(files) - 1, limit, dtype=int)
            files = [files[int(index)] for index in indices]
    return files, available


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--lyapunov-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paired-online-report", default=None)
    parser.add_argument("--analytic-horizon", type=int, default=5)
    parser.add_argument("--system-horizon", type=int, default=10)
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--only-modified", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if int(args.analytic_horizon) != 5:
        raise SystemExit("the frozen analytic-A replay horizon is H=5")
    if int(args.system_horizon) != 10:
        raise SystemExit("the frozen World-Model system truth horizon is H=10")
    if int(args.workers) <= 0:
        raise SystemExit("--workers must be positive")

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite replay report: {output}")
    snapshot_files, snapshot_files_available = _selected_snapshot_files(
        Path(args.snapshot_dir),
        only_modified=bool(args.only_modified),
        max_snapshots=args.max_snapshots,
    )
    if not snapshot_files:
        raise SystemExit("no decision snapshots found")
    lyapunov_config = json.loads(
        Path(args.lyapunov_config).read_text(encoding="utf-8")
    )
    work = [
        (
            path.as_posix(),
            lyapunov_config,
            int(args.analytic_horizon),
            int(args.system_horizon),
            bool(args.only_modified),
        )
        for path in snapshot_files
    ]
    if int(args.workers) == 1:
        replayed = [_process_snapshot(value) for value in work]
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            replayed = list(executor.map(_process_snapshot, work))
    skipped = [
        row for row in replayed
        if isinstance(row, Mapping) and row.get("_replay_skipped")
    ]
    rows = [
        row for row in replayed
        if row is not None and not row.get("_replay_skipped")
    ]
    if not rows:
        raise SystemExit("no eligible aligned snapshots were replayed")

    future_index = _future_trace_index(Path(args.snapshot_dir))
    future_trace_joined = 0
    for row in rows:
        key = (
            int(row["seed"]),
            int(row["decision_tick"]),
            int(row["order_id"]),
            int(row["pod_id"]),
        )
        future = future_index.get(key)
        if future:
            row["online_future_windows"] = future
            future_trace_joined += 1
    tail_values = [
        float(row["online_future_windows"]["future_200_wm_label_cost"])
        for row in rows
        if row.get("online_future_windows", {}).get(
            "future_200_wm_label_cost"
        ) is not None
        and int(row["online_future_windows"].get("future_200_count", 0)) == 200
    ]
    tail_q90 = float(np.quantile(tail_values, 0.90)) if tail_values else None
    if tail_q90 is not None:
        for row in rows:
            value = row.get("online_future_windows", {}).get(
                "future_200_wm_label_cost"
            )
            if (
                value is None
                or int(row.get("online_future_windows", {}).get(
                    "future_200_count", 0
                )) != 200
                or float(value) < tail_q90
            ):
                continue
            if "online_future_200_cost_tail" not in row["phase_c_reasons"]:
                row["phase_c_reasons"].append("online_future_200_cost_tail")
                row["phase_c_priority_score"] += 2

    modified = [row for row in rows if row["modified"]]
    harmful = [
        row for row in modified
        if row["selected_minus_baseline_true_realized_cost"] > 1e-12
    ]
    analytic_mismatch = [
        row for row in rows
        if not bool(row["analytic_top1_matches_true_work"])
    ]
    modified_analytic_mismatch = [
        row for row in analytic_mismatch if row["modified"]
    ]
    fusion_work_harm = [
        row for row in modified
        if row["selected_minus_baseline_true_work_drift"] > 1e-12
    ]
    work_cost_conflict = [
        row for row in modified
        if row["selected_minus_baseline_true_work_drift"] <= 1e-12
        and row["selected_minus_baseline_true_realized_cost"] > 1e-12
    ]
    wm_errors = [
        row for row in rows if not bool(row["wm_top1_matches_true_cost"])
    ]
    online = _online_summary(args.paired_online_report)
    analytic_pairwise = _mean(rows, "analytic_pairwise_vs_true_work")

    if analytic_pairwise is not None and analytic_pairwise < 0.5:
        verdict = "ANALYTIC_H5_DIRECTION_FAILURE"
        next_action = (
            "audit analytic virtual-post-action parity before Phase-C retraining; "
            "ordinary top-1 errors alone are expected from a nominal free-flow model"
        )
    elif fusion_work_harm:
        verdict = "FUSION_CAN_SELECT_WORSE_TRUE_H5_WORK_THAN_PURE_WM"
        next_action = (
            "inspect the fixed group-range/lambda fusion; this consequence does "
            "not by itself invalidate the analytic work ledger"
        )
    elif work_cost_conflict:
        verdict = "LYAPUNOV_WORK_SIGNAL_VALID_BUT_FUSION_OBJECTIVE_CONFLICT"
        next_action = (
            "retain snapshots as Phase-C causal evidence and revisit how the fixed "
            "work objective is combined with the World-Model system cost"
        )
    elif online and online["wm_label_cost_improvement"] < 0.0 and not harmful:
        verdict = "ISOLATED_H5_WORK_H10_COST_VALID_CLOSED_LOOP_FEEDBACK_FAILURE"
        next_action = (
            "use long-run tail snapshots for Phase-C World-Model data; the H5 "
            "analytic action direction is not the reproduced failure"
        )
    elif wm_errors:
        verdict = "WORLD_MODEL_H10_RANKING_ERRORS_PHASEC_CANDIDATES_AVAILABLE"
        next_action = (
            "export the prioritised fixed-context snapshots into Phase C and "
            "compare them with the training distribution before retraining "
            "the base World Model, without adding an auxiliary head"
        )
    else:
        verdict = "ANALYTIC_H5_MECHANISM_SUPPORTED_ON_REPLAYED_SNAPSHOTS"
        next_action = "continue the paired 1000-tick development test"

    priorities = sorted(
        [row for row in rows if row["phase_c_reasons"]],
        key=lambda row: (
            -int(row["phase_c_priority_score"]),
            int(row["decision_tick"]),
            str(row["snapshot"]),
        ),
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "role": "LAYER5_FAILURE_LOCALISATION_AND_PHASEC_SELECTION",
        "verdict": verdict,
        "next_action": next_action,
        "analytic_work_horizon": int(args.analytic_horizon),
        "system_cost_horizon": int(args.system_horizon),
        "rollout_continuation_mode": "isolated",
        "future_orders_generated": False,
        "continuation_policy_used": False,
        "learned_auxiliary_head_used": False,
        "snapshot_dir": str(args.snapshot_dir),
        "snapshot_files_available": snapshot_files_available,
        "snapshot_files_seen": len(snapshot_files),
        "snapshot_groups_replayed": len(rows),
        "snapshot_groups_skipped": len(skipped),
        "skipped_replay_examples": skipped[:50],
        "future_trace_groups_joined": future_trace_joined,
        "future_200_wm_label_cost_q90": tail_q90,
        "modified_groups": len(modified),
        "harmful_modified_groups": len(harmful),
        "analytic_work_top1_mismatch_groups": len(analytic_mismatch),
        "modified_analytic_mismatch_groups": len(modified_analytic_mismatch),
        "fusion_selected_worse_work_groups": len(fusion_work_harm),
        "work_improves_but_system_cost_worsens_groups": len(work_cost_conflict),
        "wm_true_cost_top1_error_groups": len(wm_errors),
        "rates": {
            "modified": len(modified) / len(rows),
            "harmful_among_modified": len(harmful) / max(len(modified), 1),
            "analytic_top1_mismatch_among_modified": (
                len(modified_analytic_mismatch) / max(len(modified), 1)
            ),
            "wm_true_cost_top1_error": len(wm_errors) / len(rows),
        },
        "ranking": {
            "wm_pairwise_vs_true_cost_mean": _mean(
                rows, "wm_pairwise_vs_true_cost"
            ),
            "analytic_pairwise_vs_true_work_mean": _mean(
                rows, "analytic_pairwise_vs_true_work"
            ),
            "combined_pairwise_vs_true_cost_mean": _mean(
                rows, "combined_pairwise_vs_true_cost"
            ),
            "wm_normalised_cost_regret_mean": _mean(
                rows, "wm_normalised_cost_regret"
            ),
            "combined_normalised_cost_regret_mean": _mean(
                rows, "combined_normalised_cost_regret"
            ),
        },
        "paired_online_summary": online,
        "phase_c_priority_snapshots": [
            {
                "snapshot": row["snapshot"],
                "seed": row["seed"],
                "decision_tick": row["decision_tick"],
                "candidate_group_id": row["candidate_group_id"],
                "candidate_count": row["candidate_count"],
                "priority_score": row["phase_c_priority_score"],
                "reasons": row["phase_c_reasons"],
                "category": row["category"],
            }
            for row in priorities
        ],
        "groups": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {output}")
    print(f"verdict: {verdict}")
    print(
        "replayed/modified/harmful = "
        f"{len(rows)}/{len(modified)}/{len(harmful)}"
    )
    print(
        "analytic top1 mismatches = "
        f"{len(analytic_mismatch)}; fusion-work harm = {len(fusion_work_harm)}; "
        f"work-cost conflicts = {len(work_cost_conflict)}"
    )
    print(f"Phase-C priority snapshots = {len(priorities)}")


if __name__ == "__main__":
    main()
