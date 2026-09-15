"""Replay every captured failure snapshot under a common continuation policy.

The online run determines which robot was selected.  This diagnostic restores
the exact pre-assignment state, forces each candidate robot in turn, and then
uses the same Greedy continuation for every candidate.  The common
continuation isolates within-context robot ranking from later policy choices;
it is diagnostic evidence, not a new closed-loop certification result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from Policies.TaskAssigner import GreedyTaskAssigner
from WorldModel.core.costs import compute_realized_cost
from WorldModel.data.counterfactual_rollout import evaluate_candidate_rollout
from WorldModel.evaluation.phase_c_failure_diagnostic_protocol import (
    CASES,
    COUNTERFACTUAL_DELAY_SCALE,
    COUNTERFACTUAL_HORIZONS,
    COUNTERFACTUAL_SCHEMA_VERSION,
    OUTPUT_ROOT,
    RUN_SCHEMA_VERSION,
    run_id,
    sha256_file,
)
from WorldModel.graph_builder import build_static_graph
from WorldState.order_state import Order
from WorldState.task_state import Task


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"existing counterfactual report differs: {path}")
        print(f"[audit] unchanged {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _persistent_deadlock_onset(
    values: Sequence[float], *, window: int = 50, threshold: float = 0.5
) -> int | None:
    for index in range(max(0, len(values) - 2 * window + 1)):
        if (
            fmean(values[index:index + window]) >= threshold
            and fmean(values[index + window:index + 2 * window]) >= threshold
        ):
            return index
    return None


def _pairwise_concordance(
    rows: Sequence[Mapping[str, Any]], predictor: str, target: str
) -> float | None:
    concordant = 0.0
    pairs = 0
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            pred_delta = float(rows[left][predictor]) - float(
                rows[right][predictor]
            )
            target_delta = float(rows[left][target]) - float(
                rows[right][target]
            )
            if target_delta == 0.0:
                continue
            if pred_delta == 0.0:
                concordant += 0.5
            elif pred_delta * target_delta > 0.0:
                concordant += 1.0
            pairs += 1
    return float(concordant / pairs) if pairs else None


def _prefix_metrics(system: torch.Tensor, horizon: int) -> dict[str, float]:
    prefix = system[:horizon]
    risk = prefix[:, 6]
    terminal = risk[-min(50, horizon):]
    return {
        "discounted_realized_cost": float(compute_realized_cost(prefix)),
        "completed_orders_delta": float(prefix[:, 5].sum().item()),
        "unified_risk_mean": float(risk.mean().item()),
        "unified_risk_max": float(risk.max().item()),
        "unified_risk_terminal_mean": float(terminal.mean().item()),
    }


def _candidate_trace(online: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["robot_id"]): dict(row)
        for row in online.get("candidates", ())
        if row.get("robot_id") is not None
    }


def _replay_snapshot(arguments: tuple[str, str, tuple[int, ...]]) -> dict[str, Any]:
    path_value, arm, horizons = arguments
    path = Path(path_value)
    with path.open("rb") as handle:
        snapshot = pickle.load(handle)

    if not bool(snapshot.get("online_decision_trace_aligned")):
        raise ValueError(f"unaligned diagnostic snapshot: {path}")
    online = snapshot.get("online_decision") or {}
    selected_robot = online.get("selected_robot")
    baseline_robot = online.get("baseline_robot")
    if selected_robot is None or baseline_robot is None:
        return {
            "snapshot": path.as_posix(),
            "valid": False,
            "reason": "online decision has no selected/baseline robot",
        }

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
    trace_by_robot = _candidate_trace(online)
    max_horizon = max(horizons)

    saved_task_id = Task._next_id
    saved_order_id = Order._next_id
    saved_python_rng = random.getstate()
    saved_numpy_rng = np.random.get_state()
    saved_torch_rng = torch.random.get_rng_state()
    candidate_rows = []
    try:
        for candidate in snapshot.get("candidates", ()):
            robot_id = int(candidate["robot_id"])
            trace = trace_by_robot.get(robot_id)
            if trace is None:
                raise ValueError(
                    f"candidate robot {robot_id} missing from trace: {path}"
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
                max_horizon,
                node_map,
                local_capacity,
                bottleneck_score,
                adj,
                reservation_window=int(
                    (snapshot.get("lyapunov_l0_config") or {}).get(
                        "reservation_window", 1
                    )
                ),
                delay_scale=float(COUNTERFACTUAL_DELAY_SCALE),
                rollout_continuation_mode="behavior",
                continuation_order_generator=snapshot["order_generator_state"],
                continuation_task_assigner=GreedyTaskAssigner(),
            )
            mask = result["future_mask"]
            if mask.numel() != max_horizon or not bool((mask > 0.5).all()):
                continue
            system = result["future_system_labels"]
            base_score = _scalar(trace.get("score"))
            conversion_score = _scalar(trace.get("score_conv"))
            policy_score = (
                conversion_score
                if arm == "phasec_s1" and conversion_score is not None
                else base_score
            )
            if base_score is None or policy_score is None:
                raise ValueError(f"candidate trace lacks score: {path}")
            candidate_rows.append({
                "robot_id": robot_id,
                "wm_score": base_score,
                "conversion_score": conversion_score,
                "policy_score": policy_score,
                "route_len": int(trace.get("route_len", 0)),
                "risk_max_prediction": _scalar(trace.get("risk_max")),
                "rollout_vertex_conflicts_h200": int(
                    result["rollout_vertex_conflicts"]
                ),
                "rollout_swap_conflicts_h200": int(
                    result["rollout_swap_conflicts"]
                ),
                "rollout_blocked_moves_h200": int(
                    result["rollout_blocked_moves"]
                ),
                "rollout_generated_orders_h200": int(
                    result["rollout_generated_orders"]
                ),
                "rollout_assigned_tasks_h200": int(
                    result["rollout_assigned_tasks"]
                ),
                "horizons": {
                    str(horizon): _prefix_metrics(system, horizon)
                    for horizon in horizons
                },
            })
    finally:
        Task._next_id = saved_task_id
        Order._next_id = saved_order_id
        random.setstate(saved_python_rng)
        np.random.set_state(saved_numpy_rng)
        torch.random.set_rng_state(saved_torch_rng)

    if len(candidate_rows) < 2:
        return {
            "snapshot": path.as_posix(),
            "valid": False,
            "reason": "fewer than two valid counterfactual candidates",
        }
    return {
        "snapshot": path.as_posix(),
        "snapshot_sha256": sha256_file(path),
        "valid": True,
        "run_id": snapshot.get("run_id"),
        "arm": arm,
        "load": snapshot.get("load"),
        "seed": int(snapshot.get("seed")),
        "decision_tick": int(snapshot["decision_tick"]),
        "candidate_group_id": snapshot["candidate_group_id"],
        "order_id": int(snapshot["fixed_context"]["order_id"]),
        "pod_id": int(snapshot["fixed_context"]["pod_id"]),
        "station_id": int(snapshot["fixed_context"]["station_id"]),
        "selected_robot": int(selected_robot),
        "baseline_robot": int(baseline_robot),
        "candidates": sorted(candidate_rows, key=lambda row: row["robot_id"]),
    }


def _normalised_regret(values: Sequence[float], selected: float) -> float:
    span = max(values) - min(values)
    return 0.0 if span == 0.0 else float((selected - min(values)) / span)


def _group_ranking_summary(group: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    rows = []
    for candidate in group["candidates"]:
        metrics = candidate["horizons"][str(horizon)]
        rows.append({
            **candidate,
            "true_cost": float(metrics["discounted_realized_cost"]),
            "completion": float(metrics["completed_orders_delta"]),
            "terminal_risk": float(metrics["unified_risk_terminal_mean"]),
        })
    selected_robot = int(group["selected_robot"])
    baseline_robot = int(group["baseline_robot"])
    selected = next(row for row in rows if row["robot_id"] == selected_robot)
    baseline = next(row for row in rows if row["robot_id"] == baseline_robot)
    costs = [row["true_cost"] for row in rows]
    tolerance = 1e-9
    policy_argmin = min(
        rows, key=lambda row: (row["policy_score"], row["robot_id"])
    )
    dominating = [
        int(row["robot_id"])
        for row in rows
        if row["robot_id"] != selected_robot
        and row["completion"] >= selected["completion"] - tolerance
        and row["terminal_risk"] <= selected["terminal_risk"] + tolerance
        and (
            row["completion"] > selected["completion"] + tolerance
            or row["terminal_risk"] < selected["terminal_risk"] - tolerance
        )
    ]
    best_cost = min(rows, key=lambda row: (row["true_cost"], row["robot_id"]))
    return {
        "horizon": int(horizon),
        "best_true_cost_robot": int(best_cost["robot_id"]),
        "trace_policy_argmin_robot": int(policy_argmin["robot_id"]),
        "trace_policy_argmin_matches_selected": (
            int(policy_argmin["robot_id"]) == selected_robot
        ),
        "selected_true_cost": selected["true_cost"],
        "best_true_cost": best_cost["true_cost"],
        "selected_cost_regret": selected["true_cost"] - best_cost["true_cost"],
        "selected_normalised_cost_regret": _normalised_regret(
            costs, selected["true_cost"]
        ),
        "selected_top1_matches_true_cost": (
            selected["true_cost"] <= best_cost["true_cost"] + tolerance
        ),
        "wm_pairwise_vs_true_cost": _pairwise_concordance(
            rows, "wm_score", "true_cost"
        ),
        "policy_pairwise_vs_true_cost": _pairwise_concordance(
            rows, "policy_score", "true_cost"
        ),
        "selected_completion": selected["completion"],
        "selected_terminal_risk": selected["terminal_risk"],
        "pareto_dominated_selected": bool(dominating),
        "pareto_dominating_robots": dominating,
        "selected_minus_baseline_true_cost": (
            selected["true_cost"] - baseline["true_cost"]
        ),
    }


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return fmean(present) if present else None


def _aggregate_groups(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for horizon in COUNTERFACTUAL_HORIZONS:
        rankings = [group["ranking"][str(horizon)] for group in groups]
        regrets = [
            float(row["selected_normalised_cost_regret"]) for row in rankings
        ]
        modified = [
            (group, row)
            for group, row in zip(groups, rankings)
            if int(group["selected_robot"]) != int(group["baseline_robot"])
        ]
        result[str(horizon)] = {
            "groups": len(rankings),
            "selected_true_cost_top1_rate": (
                sum(bool(row["selected_top1_matches_true_cost"]) for row in rankings)
                / max(len(rankings), 1)
            ),
            "selected_pareto_dominated_rate": (
                sum(bool(row["pareto_dominated_selected"]) for row in rankings)
                / max(len(rankings), 1)
            ),
            "trace_policy_argmin_mismatch_groups": sum(
                not bool(row["trace_policy_argmin_matches_selected"])
                for row in rankings
            ),
            "wm_pairwise_vs_true_cost_mean": _mean(
                row["wm_pairwise_vs_true_cost"] for row in rankings
            ),
            "policy_pairwise_vs_true_cost_mean": _mean(
                row["policy_pairwise_vs_true_cost"] for row in rankings
            ),
            "selected_normalised_cost_regret_mean": _mean(regrets),
            "selected_normalised_cost_regret_p90": _quantile(regrets, 0.90),
            "conversion_modified_groups": len(modified),
            "conversion_harmful_true_cost_groups": sum(
                float(row["selected_minus_baseline_true_cost"]) > 1e-9
                for _, row in modified
            ),
        }
    return result


def _trajectory_context(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "diagnostic_summary.json")
    if summary.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError(f"wrong run schema: {run_dir}")
    trajectory = run_dir / str(summary["outputs"]["trajectory"])
    deadlock = []
    completion_ticks = []
    with trajectory.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            deadlock.append(float(row["risk"]["deadlock_ratio"]))
            if int(row.get("completed_orders_delta", 0)) > 0:
                completion_ticks.append(int(row["tick"]))
    return {
        "persistent_deadlock_onset_tick": _persistent_deadlock_onset(deadlock),
        "last_completion_tick": completion_ticks[-1] if completion_ticks else None,
        "deadlock_ratio_max": max(deadlock) if deadlock else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 2, 1))
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    output_root = Path(args.output_root)
    bundle = _read_json(output_root / "phase_c_failure_diagnostic_protocol.json")
    protocol_sha = str((bundle.get("protocol") or {}).get("protocol_sha256", ""))
    if not protocol_sha:
        raise ValueError("diagnostic bundle lacks protocol hash")

    case_inputs = []
    expected_snapshots = 0
    for case in CASES:
        load = str(case["load"])
        seed = int(case["seed"])
        arm = str(case["snapshot_arm"])
        rid = run_id(arm, load, seed)
        run_dir = output_root / "runs" / rid
        index = _read_json(run_dir / "snapshots" / f"snapindex_{rid}.json")
        files = sorted((run_dir / "snapshots").glob("*.pkl"))
        if len(files) != int(index.get("n_snapshots", -1)):
            raise ValueError(f"snapshot index count mismatch: {rid}")
        expected_snapshots += len(files)
        case_inputs.append((case, run_dir, files))

    output = output_root / "phase_c_failure_counterfactuals.json"
    if output.is_file():
        existing = _read_json(output)
        resume_checks = {
            "schema": existing.get("schema_version") == COUNTERFACTUAL_SCHEMA_VERSION,
            "protocol": existing.get("protocol_sha256") == protocol_sha,
            "passed": bool(existing.get("passed")),
            "snapshot_count": int(existing.get("snapshot_groups_expected", -1))
            == expected_snapshots,
        }
        if not all(resume_checks.values()):
            failed = [key for key, passed in resume_checks.items() if not passed]
            raise ValueError(f"cannot resume counterfactual report: {failed}")
        print(f"[resume] counterfactual report: {output}")
        return

    jobs = [
        (path.as_posix(), str(case["snapshot_arm"]), COUNTERFACTUAL_HORIZONS)
        for case, _run_dir, files in case_inputs
        for path in files
    ]
    print(
        f"[counterfactual] snapshots={len(jobs)} workers={args.workers} "
        f"horizons={list(COUNTERFACTUAL_HORIZONS)}"
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        replayed = list(executor.map(_replay_snapshot, jobs, chunksize=1))
    replay_by_path = {row["snapshot"]: row for row in replayed}

    case_reports = []
    failures = []
    for case, run_dir, files in case_inputs:
        context = _trajectory_context(run_dir)
        onset = context["persistent_deadlock_onset_tick"]
        groups = []
        invalid = []
        for path in files:
            group = replay_by_path[path.as_posix()]
            if not bool(group.get("valid")):
                invalid.append(group)
                continue
            tick = int(group["decision_tick"])
            group["failure_timing"] = {
                "persistent_deadlock_onset_tick": onset,
                "ticks_before_persistent_deadlock": (
                    int(onset) - tick if onset is not None else None
                ),
                "within_200_ticks_before_onset": bool(
                    onset is not None and int(onset) - 200 <= tick <= int(onset)
                ),
            }
            group["ranking"] = {
                str(horizon): _group_ranking_summary(group, horizon)
                for horizon in COUNTERFACTUAL_HORIZONS
            }
            groups.append(group)
        groups.sort(key=lambda row: (row["decision_tick"], row["candidate_group_id"]))
        before_onset = [
            group for group in groups
            if group["failure_timing"]["within_200_ticks_before_onset"]
        ]
        case_passed = not invalid and len(groups) == len(files)
        if not case_passed:
            failures.append(
                f"{run_dir.name}: valid={len(groups)} expected={len(files)}"
            )
        case_reports.append({
            "run_id": run_dir.name,
            "load": case["load"],
            "seed": int(case["seed"]),
            "arm": case["snapshot_arm"],
            "question": case["question"],
            "passed": case_passed,
            "trajectory": context,
            "snapshot_groups": len(files),
            "valid_replayed_groups": len(groups),
            "invalid_groups": invalid,
            "aggregate_all_decisions": _aggregate_groups(groups),
            "aggregate_pre_failure_200_ticks": _aggregate_groups(before_onset),
            "groups": groups,
        })

    report = {
        "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "protocol_sha256": protocol_sha,
        "passed": not failures and len(replayed) == expected_snapshots,
        "role": "WITHIN_CONTEXT_RANKING_DIAGNOSTIC",
        "continuation_policy": "common_greedy_behavior_after_forced_first_action",
        "continuation_interpretation": (
            "candidate comparisons isolate the first robot choice under one "
            "shared continuation; they do not prove cross-context joint optimality"
        ),
        "horizons": list(COUNTERFACTUAL_HORIZONS),
        "system_label_delay_scale": float(COUNTERFACTUAL_DELAY_SCALE),
        "snapshot_groups_expected": expected_snapshots,
        "snapshot_groups_observed": len(replayed),
        "failures": failures,
        "cases": case_reports,
    }
    _atomic_write_json(output, report)
    if not report["passed"]:
        raise RuntimeError("counterfactual replay failed: " + "; ".join(failures))
    print(f"[complete] counterfactual report: {output}")


if __name__ == "__main__":
    main()
