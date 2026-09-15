import unittest

from WorldModel.evaluation.evaluate_lyapunov_oracle import (
    audit_analytic_invariants,
    audit_isolated_semantics,
    build_short_horizon_certification,
    evaluate_samples,
)


def _sample(group, robot, heuristic, delta_l, productive, realized):
    return {
        "run_id": "synthetic_low_seed1",
        "simulation_seed": 1,
        "candidate_group_id": group,
        "candidate_key": f"{group}_r{robot}",
        "candidate_info": {"robot_id": robot},
        "heuristic_cost": float(heuristic),
        "realized_cost": float(realized),
        "lyapunov_l0_delta": float(delta_l),
        "lyapunov_l0_progress": {
            "productive_total": float(productive),
            "reverse_total": 0.0,
            "arrival_total": 0.0,
            "replan_residual_total": 0.0,
            "route_plan_churn_total": 0.0,
        },
        "rollout_blocked_moves": 0,
    }


def _snapshot(total, work_sum, tick=0):
    return {
        "schema_version": "lyapunov_l1_snapshot_v3",
        "tick": int(tick),
        "total": float(total),
        "components": {
            "work": float(total),
            "station": 0.0,
            "traffic": 0.0,
            "stall": 0.0,
            "arrival": 0.0,
        },
        "station_work": {1: float(work_sum)},
    }


def _strict_sample():
    sample = _sample("g1", 1, 0.0, -0.25, 0.5, 0.0)
    sample.update({
        "lyapunov_l0_collection_schema_version": "lyapunov_l1_collection_v3",
        "rollout_continuation_mode": "isolated",
        "continuation_policy": "isolated_forced_candidate",
        "rollout_generated_orders": 0,
        "rollout_assigned_tasks": 0,
        "lyapunov_l0_valid": True,
        "fixed_context": {"order_id": 1, "pod_id": 10},
        "lyapunov_l0_config": {
            "work_weight": 1.0,
            "station_weight": 1.0,
            "traffic_weight": 1.0,
            "stall_weight": 1.0,
            "plan_fail_weight": 0.5,
            "arrival_weight": 1.0,
        },
        "future_mask": [1.0, 1.0],
        "lyapunov_l0_start": _snapshot(1.0, 2.0, tick=5),
        "lyapunov_l0_post_action": _snapshot(1.1, 2.0, tick=5),
        "lyapunov_l0_end": _snapshot(0.75, 1.5, tick=7),
        "lyapunov_l0_immediate_delta": 0.1,
    })
    sample["lyapunov_l0_start"]["completed_chain_keys"] = []
    sample["lyapunov_l0_post_action"]["completed_chain_keys"] = []
    sample["lyapunov_l0_end"]["completed_chain_keys"] = []
    sample["lyapunov_l0_progress"]["horizon"] = 2
    return sample


class LyapunovOracleEvaluationTests(unittest.TestCase):
    def test_strict_isolated_audit_and_invariants_accept_closed_row(self):
        sample = _strict_sample()
        semantic = audit_isolated_semantics([sample])
        invariant = audit_analytic_invariants([sample])
        self.assertTrue(semantic["passed"])
        self.assertTrue(invariant["passed"])

    def test_strict_isolated_audit_rejects_behavior_continuation(self):
        sample = _strict_sample()
        sample["rollout_continuation_mode"] = "behavior"
        sample["continuation_policy"] = "GreedyTaskAssigner"
        sample["rollout_generated_orders"] = 1
        sample["rollout_assigned_tasks"] = 2
        sample["lyapunov_l0_progress"]["arrival_total"] = 1.0
        semantic = audit_isolated_semantics([sample])
        self.assertFalse(semantic["passed"])
        self.assertFalse(
            semantic["checks"]["isolated_continuation_mode"]["passed"]
        )
        self.assertFalse(
            semantic["checks"]["zero_external_arrival_work"]["passed"]
        )

    def test_strict_isolated_audit_rejects_snapshot_schema_or_start_mismatch(self):
        left = _strict_sample()
        right = _strict_sample()
        right["candidate_key"] = "g1_r2"
        right["candidate_info"] = {"robot_id": 2}
        right["lyapunov_l0_start"] = _snapshot(1.1, 2.0)
        right["lyapunov_l0_start"]["schema_version"] = (
            "lyapunov_l0_snapshot_v1"
        )
        semantic = audit_isolated_semantics([left, right])
        self.assertFalse(semantic["passed"])
        self.assertFalse(
            semantic["checks"]["current_snapshot_schema"]["passed"]
        )
        self.assertFalse(
            semantic["checks"]["same_start_snapshot_within_group"]["passed"]
        )

    def test_strict_isolated_audit_rejects_mixed_functional_configs(self):
        left = _strict_sample()
        right = _strict_sample()
        right["candidate_key"] = "g1_r2"
        right["candidate_info"] = {"robot_id": 2}
        right["lyapunov_l0_config"] = dict(right["lyapunov_l0_config"])
        right["lyapunov_l0_config"]["traffic_weight"] = 2.0
        semantic = audit_isolated_semantics([left, right])
        self.assertFalse(semantic["passed"])
        self.assertFalse(
            semantic["checks"]["frozen_lyapunov_config"]["passed"]
        )

    def test_strict_isolated_audit_rejects_horizon_mismatch(self):
        sample = _strict_sample()
        sample["lyapunov_l0_progress"]["horizon"] = 1
        semantic = audit_isolated_semantics([sample])
        self.assertFalse(semantic["passed"])
        self.assertFalse(
            semantic["checks"]["rollout_horizon_consistent"]["passed"]
        )

    def test_strict_isolated_audit_rejects_completed_candidate(self):
        sample = _strict_sample()
        sample["lyapunov_l0_start"]["completed_chain_keys"] = [[1, 10]]
        semantic = audit_isolated_semantics([sample])
        self.assertFalse(semantic["passed"])
        self.assertFalse(
            semantic["checks"]["candidate_not_completed_at_start"]["passed"]
        )

    def test_invariant_audit_rejects_assignment_work_loss(self):
        sample = _strict_sample()
        sample["lyapunov_l0_post_action"]["station_work"][1] = 1.0
        invariant = audit_analytic_invariants([sample])
        self.assertFalse(invariant["passed"])
        self.assertFalse(
            invariant["checks"]["assignment_preserves_work_mass"]["passed"]
        )

    def test_short_horizon_gate_distinguishes_pass_and_redesign(self):
        method = "registered"
        method_summary = {
            "groups": 3,
            "flips": 3,
            "realized_cost_delta": {"mean": -0.2},
            "productive_progress_delta": {"mean": 0.1},
            "lyapunov_improvement": {"mean": 0.5},
        }
        report = {
            "groups": 3,
            "manifest": {
                load: {"groups": 1} for load in ("low", "mid", "high")
            },
            "pairwise": {
                "practical_gap_slices": {
                    "0.1": {
                        "pairs": 3,
                        "realized_cost_direction_accuracy": 1.0,
                    }
                }
            },
            "methods": {method: method_summary},
            "by_load": {
                load: {"methods": {method: method_summary}}
                for load in ("low", "mid", "high")
            },
            "matched_random_placebo": {
                method: {"realized_cost_delta": {"p05": -0.1}}
            },
            "component_scale": {
                "usable_samples": 6,
                "inactive_components": [],
                "tie_rate": 0.0,
                "max_dominant_component": "work",
                "max_dominant_rate": 0.5,
                "active_component_p90_scale_ratio": 2.0,
                "gates": {
                    "all_components_activated": True,
                    "component_dominance_pass": True,
                    "component_dominance_limit": 0.8,
                    "active_p90_scale_ratio_pass": True,
                    "active_p90_scale_ratio_limit": 10.0,
                },
            },
        }
        audit = {"passed": True}
        gate = build_short_horizon_certification(
            report,
            method=method,
            semantic_audit=audit,
            invariant_audit=audit,
            min_groups=3,
            min_groups_per_load=1,
            min_method_flips=3,
            min_pairwise_pairs=3,
        )
        self.assertEqual(gate["verdict"], "PASS_SHORT_HORIZON_ORACLE")
        self.assertEqual(
            gate["protocol_status"], "legacy_gap_based_diagnostic"
        )
        self.assertFalse(gate["scale_invariant_validity_claim"])

        report["pairwise"]["practical_gap_slices"]["0.1"][
            "realized_cost_direction_accuracy"
        ] = 0.4
        gate = build_short_horizon_certification(
            report,
            method=method,
            semantic_audit=audit,
            invariant_audit=audit,
            min_groups=3,
            min_groups_per_load=1,
            min_method_flips=3,
            min_pairwise_pairs=3,
        )
        self.assertEqual(gate["verdict"], "REDESIGN_REQUIRED")

    def test_progress_floor_prevents_delay_only_candidate(self):
        samples = [
            _sample("g1", 1, 0, 5, 1.0, 10),
            _sample("g1", 2, 1, 0, 1.0, 12),
            _sample("g2", 3, 0, 3, 1.0, 10),
            _sample("g2", 4, 1, 0, 0.5, 12),
        ]
        report = evaluate_samples(
            samples,
            lambdas=(1.0,),
            progress_epsilons=(0.0,),
            gap_thresholds=(0.1,),
            random_repeats=4,
        )
        unconstrained = report["methods"]["base_plus_delta_lambda=1"]
        constrained = report["methods"][
            "base_plus_delta_lambda=1_progress_floor_eps=0"
        ]
        self.assertEqual(unconstrained["flips"], 2)
        self.assertEqual(constrained["flips"], 1)
        self.assertGreaterEqual(
            constrained["productive_progress_delta"]["mean"], 0.0
        )

    def test_report_marks_heuristic_base_as_proxy(self):
        samples = [
            _sample("g1", 1, 0, 1, 1.0, 1),
            _sample("g1", 2, 1, 0, 1.0, 2),
        ]
        report = evaluate_samples(samples, random_repeats=2)
        self.assertTrue(report["semantics"]["base_field_is_proxy"])
        self.assertFalse(report["semantics"]["online_claim"])
        self.assertFalse(
            report["semantics"]["raw_gap_thresholds_are_validity_gates"]
        )
        self.assertEqual(report["groups"], 1)

    def test_component_scale_flags_arrival_dominance(self):
        samples = [
            _sample("g1", 1, 0, 4.0, 1.0, 1.0),
            _sample("g1", 2, 1, 5.0, 1.0, 1.0),
        ]
        for index, sample in enumerate(samples):
            start = _snapshot(1.0, 2.0)
            end = _snapshot(1.0, 2.0)
            end["components"]["arrival"] = 4.0 + index
            end["total"] = 5.0 + index
            sample["lyapunov_l0_start"] = start
            sample["lyapunov_l0_end"] = end
        report = evaluate_samples(samples, random_repeats=2)
        scale = report["component_scale"]
        self.assertEqual(scale["max_dominant_component"], "arrival")
        self.assertFalse(scale["gates"]["component_dominance_pass"])

    def test_practical_drift_margin_rejects_numerical_flip(self):
        samples = [
            _sample("g1", 1, 0.0, 1.0, 1.0, 1.0),
            _sample("g1", 2, 0.0, 0.95, 1.0, 1.0),
        ]
        report = evaluate_samples(
            samples,
            lambdas=(1.0,),
            progress_epsilons=(0.0,),
            selection_margins=(0.1,),
            random_repeats=2,
        )
        unguarded = report["methods"][
            "base_plus_delta_lambda=1_progress_floor_eps=0"
        ]
        guarded = report["methods"][
            "base_plus_delta_lambda=1_progress_floor_eps=0_min_drift_gain=0.1"
        ]
        self.assertEqual(unguarded["flips"], 1)
        self.assertEqual(guarded["flips"], 0)


if __name__ == "__main__":
    unittest.main()
