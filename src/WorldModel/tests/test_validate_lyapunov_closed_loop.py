import unittest

import numpy as np
import torch

from WorldModel.evaluation.validate_lyapunov_closed_loop import (
    analyze_stream,
    evaluate_payloads,
)


def _payload(seed, *, increasing=False):
    ticks = np.arange(120, dtype=np.float32)
    direction = 1.0 if increasing else -1.0
    work = 10.0 + direction * 0.01 * ticks
    components = np.column_stack((
        work,
        np.full_like(ticks, 0.10),
        np.zeros_like(ticks),
        np.zeros_like(ticks),
        np.full_like(ticks, 0.30),
    ))
    total = components.sum(axis=1)
    summary = np.column_stack((
        components,
        total,
        work,
        work,
        np.full_like(ticks, 0.2),
        np.full_like(ticks, 0.3),
        np.full_like(ticks, 1.0),
        np.full_like(ticks, 1.0),
    ))
    progress = np.zeros((ticks.size, 5), dtype=np.float32)
    if increasing:
        progress[1:, 2] = 0.01
    else:
        progress[1:, 0] = 0.01
    return {
        "schema_version": "td_stream_v1",
        "run_id": f"Greedy_low_seed{seed}",
        "arm_label": "Greedy",
        "seed": seed,
        "config": "Config/world_model_config_PP_48_low.json",
        "lyapunov_l0_enabled": True,
        "lyapunov_l0_config": {
            "work_weight": 1.0,
            "station_weight": 1.0,
            "traffic_weight": 0.0,
            "stall_weight": 0.0,
            "plan_fail_weight": 0.0,
            "arrival_weight": 1.0,
        },
        "tick_seq": torch.tensor(ticks),
        "lyapunov_l0_summary": torch.tensor(summary),
        "lyapunov_l0_summary_names": [
            "L_work", "L_station", "L_traffic", "L_stall", "L_arrival",
            "L_total", "station_work_sum", "station_work_max",
            "station_queue_ratio_mean", "station_queue_ratio_max",
            "arrival_bin_max", "reservation_conflicts",
        ],
        "productive_progress": torch.tensor(progress),
        "productive_progress_names": [
            "productive", "reverse", "arrival", "replan_residual",
            "route_plan_churn",
        ],
        "risk_seq_full": torch.tensor(total / total.max()),
    }


class ClosedLoopLyapunovValidationTests(unittest.TestCase):
    def test_stream_invariants_and_negative_high_drift(self):
        report = analyze_stream(
            _payload(1),
            drift_horizons=(5,),
            primary_horizon=5,
            risk_horizon=5,
            burn_in=10,
            min_bin_points=10,
            bootstrap_repeats=20,
        )
        self.assertTrue(report["invariants"]["pass"])
        self.assertLess(report["primary_high_L_drift"]["mean"], 0.0)
        relation = report["primary_continuous_drift_relation"]
        self.assertIn("predicted_drift_at_observed_L_p95", relation)
        self.assertLess(
            relation["predicted_drift_at_observed_L_p95"], 0.0
        )
        self.assertLess(report["backlog"]["tail_slope_per_tick"], 0.0)

    def test_cross_run_gate_passes_stable_synthetic_policy(self):
        report = evaluate_payloads(
            [("low_seed1.pt", _payload(1)), ("low_seed2.pt", _payload(2))],
            drift_horizons=(5,),
            primary_horizon=5,
            risk_horizon=5,
            burn_in=10,
            min_bin_points=10,
            min_runs_per_group=2,
            min_component_active_samples=20,
            bootstrap_repeats=50,
        )
        self.assertEqual(report["verdict"], "PASS_CLOSED_LOOP_STABILITY")
        self.assertEqual(
            report["configured_active_components"],
            ["L_arrival", "L_station", "L_work"],
        )
        self.assertEqual(report["undercovered_components"], [])
        group = report["groups"]["low|Greedy"]
        self.assertLess(
            group[
                "continuous_predicted_drift_at_L_p95_ci95_across_runs"
            ][1],
            0.0,
        )

    def test_cross_run_gate_detects_policy_backlog_growth(self):
        report = evaluate_payloads(
            [
                ("low_seed1.pt", _payload(1, increasing=True)),
                ("low_seed2.pt", _payload(2, increasing=True)),
            ],
            drift_horizons=(5,),
            primary_horizon=5,
            risk_horizon=5,
            burn_in=10,
            min_bin_points=10,
            min_runs_per_group=2,
            min_component_active_samples=20,
            bootstrap_repeats=50,
        )
        self.assertEqual(report["verdict"], "CLOSED_LOOP_POLICY_FAILURE")


if __name__ == "__main__":
    unittest.main()
