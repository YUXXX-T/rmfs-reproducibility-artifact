import copy
import unittest

from WorldModel.evaluation.evaluate_lyapunov_validity import (
    evaluate_action_controllability,
    evaluate_closed_loop_attachment,
    evaluate_drift_prediction,
    evaluate_five_layer_validity,
    evaluate_incremental_information,
    evaluate_state_potential,
)


def _snapshot(total, *, station_work=10.0, tick=0):
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
        "station_work": {1: float(station_work)},
        "completed_chain_keys": [],
    }


def _sample(
    group,
    candidate,
    *,
    seed,
    delta_l,
    wm_score=0.0,
    realized_cost=0.0,
    predicted_delta=None,
    productive=0.0,
):
    start_total = 10.0
    start_work = 10.0
    end_work = start_work - float(productive)
    sample = {
        "run_id": f"isolated_low_seed{seed}",
        "simulation_seed": int(seed),
        "candidate_group_id": str(group),
        "candidate_key": f"{group}_candidate{candidate}",
        "candidate_info": {"robot_id": int(candidate)},
        "lyapunov_l0_collection_schema_version": "lyapunov_l1_collection_v3",
        "rollout_continuation_mode": "isolated",
        "continuation_policy": "isolated_forced_candidate",
        "rollout_generated_orders": 0,
        "rollout_assigned_tasks": 0,
        "lyapunov_l0_valid": True,
        "fixed_context": {"order_id": int(seed), "pod_id": int(seed) + 1000},
        "lyapunov_l0_config": {
            "work_weight": 1.0,
            "station_weight": 1.0,
            "traffic_weight": 0.0,
            "stall_weight": 0.0,
            "plan_fail_weight": 0.0,
            "arrival_weight": 1.0,
        },
        "future_mask": [1.0, 1.0],
        "lyapunov_l0_start": _snapshot(start_total, station_work=start_work, tick=0),
        "lyapunov_l0_post_action": _snapshot(start_total, station_work=start_work, tick=0),
        "lyapunov_l0_end": _snapshot(
            start_total + float(delta_l),
            station_work=end_work,
            tick=2,
        ),
        "lyapunov_l0_immediate_delta": 0.0,
        "lyapunov_l0_delta": float(delta_l),
        "lyapunov_l0_progress": {
            "horizon": 2,
            "productive_total": float(productive),
            "reverse_total": 0.0,
            "arrival_total": 0.0,
            "replan_residual_total": 0.0,
            "route_plan_churn_total": 0.0,
        },
        "wm_score": float(wm_score),
        "heuristic_cost": float(wm_score),
        "realized_cost": float(realized_cost),
        "rollout_blocked_moves": 0.0,
        "rollout_vertex_conflicts": 0.0,
        "rollout_swap_conflicts": 0.0,
    }
    if predicted_delta is not None:
        sample["predicted_delta_l"] = float(predicted_delta)
    return sample


def _informative_samples(group_count=12):
    drift_patterns = (
        (-1.0, 1.0, 0.0),
        (0.0, -1.0, 1.0),
        (1.0, 0.0, -1.0),
    )
    wm_values = (-1.0, 0.0, 1.0)
    samples = []
    for group_index in range(group_count):
        pattern = drift_patterns[group_index % len(drift_patterns)]
        for candidate, (wm_score, delta_l) in enumerate(
            zip(wm_values, pattern), start=1
        ):
            # The group intercept is deliberately present.  A valid layer-3
            # test must remove it through within-context centring.
            outcome = 100.0 + 0.25 * group_index + 2.0 * wm_score + 3.0 * delta_l
            samples.append(
                _sample(
                    f"g{group_index}",
                    candidate,
                    seed=401 + group_index,
                    delta_l=delta_l,
                    wm_score=wm_score,
                    realized_cost=outcome,
                    predicted_delta=delta_l,
                )
            )
    return samples


def _closed_loop_pass_report():
    return {
        "schema_version": "lyapunov_closed_loop_validation_v2",
        "semantics": {
            "five_layer_role": "layer5_normal_arrival_closed_loop_stability",
        },
        "verdict": "PASS_CLOSED_LOOP_STABILITY",
        "passed": True,
        "failed_policy_groups": [],
        "incomplete_groups": [],
        "undercovered_components": [],
        "recommendations": [],
    }


class FiveLayerLyapunovValidityTests(unittest.TestCase):
    def test_layer5_rejects_untyped_or_inconsistent_pass_report(self):
        untyped = _closed_loop_pass_report()
        untyped["semantics"] = {}
        report = evaluate_closed_loop_attachment(untyped)
        self.assertFalse(report["supported"])
        self.assertEqual(report["status"], "INCOMPATIBLE_CLOSED_LOOP_REPORT")

        inconsistent = _closed_loop_pass_report()
        inconsistent["verdict"] = "CLOSED_LOOP_POLICY_FAILURE"
        report = evaluate_closed_loop_attachment(inconsistent)
        self.assertFalse(report["supported"])
        self.assertEqual(report["status"], "INCONSISTENT_CLOSED_LOOP_REPORT")

    def test_layer1_supports_state_monitor_without_action_claim(self):
        samples = _informative_samples(group_count=3)
        report = evaluate_state_potential(samples, strict_isolated=True)

        self.assertTrue(report["supported"])
        self.assertFalse(report["semantics"]["action_selection_claim"])
        self.assertEqual(
            report["semantics"]["productive_progress_role"],
            "ledger_closure_only",
        )

    def test_layer1_observes_service_and_replan_transition_contracts(self):
        service = _sample(
            "service",
            1,
            seed=390,
            delta_l=-1.0,
            productive=1.0,
        )
        replan = _sample("replan", 1, seed=391, delta_l=0.0)
        replan["lyapunov_l0_progress"]["route_plan_churn_total"] = 1.0

        report = evaluate_state_potential([service, replan])
        observed = report["observed_transition_contract"]

        self.assertEqual(
            observed["physical_service_work_potential"]["tested"], 1
        )
        self.assertTrue(
            observed["physical_service_work_potential"]["passed"]
        )
        self.assertEqual(
            observed["route_replan_cannot_fake_work_dissipation"]["tested"],
            1,
        )
        self.assertTrue(
            observed["route_replan_cannot_fake_work_dissipation"]["passed"]
        )
        self.assertEqual(
            observed["status"], "SUPPORTED_ON_OBSERVED_TRANSITIONS"
        )

    def test_layer1_rejects_constant_zero_core_potential(self):
        samples = _informative_samples(group_count=2)
        for sample in samples:
            sample["lyapunov_l0_config"]["work_weight"] = 0.0

        report = evaluate_state_potential(samples)

        self.assertFalse(report["supported"])
        self.assertEqual(report["status"], "STATE_FUNCTIONAL_CONTRACT_FAILED")
        checks = report["analytic_state_perturbations"]["configs"][0]["checks"]
        self.assertFalse(checks["positive_core_work_weight"])
        self.assertFalse(
            checks["adding_unfinished_work_strictly_increases_core_potential"]
        )

    def test_layer1_blocks_an_observed_transition_violation(self):
        invalid_service = _sample(
            "invalid_service",
            1,
            seed=392,
            delta_l=1.0,
            productive=1.0,
        )

        report = evaluate_state_potential([invalid_service])
        observed = report["observed_transition_contract"]

        self.assertFalse(report["supported"])
        self.assertEqual(report["status"], "STATE_TRANSITION_CONTRACT_FAILED")
        self.assertEqual(observed["status"], "FAILED_ON_OBSERVED_TRANSITIONS")
        self.assertEqual(
            observed["violated_checks"], ["physical_service_work_potential"]
        )

    def test_layer2_is_scale_invariant_and_has_no_absolute_gap_gate(self):
        samples = _informative_samples(group_count=6)
        baseline = evaluate_action_controllability(
            samples,
            bootstrap_repeats=40,
            random_seed=17,
        )

        scaled = copy.deepcopy(samples)
        factor = 1e-7
        for sample in scaled:
            old_delta = sample["lyapunov_l0_delta"]
            new_delta = factor * old_delta
            sample["lyapunov_l0_delta"] = new_delta
            sample["lyapunov_l0_end"]["total"] = (
                sample["lyapunov_l0_start"]["total"] + new_delta
            )
            sample["lyapunov_l0_end"]["components"]["work"] = (
                sample["lyapunov_l0_end"]["total"]
            )

        rescaled = evaluate_action_controllability(
            scaled,
            bootstrap_repeats=40,
            random_seed=17,
        )

        self.assertEqual(baseline["status"], "ACTION_CONTROLLABLE_DRIFT_SUPPORTED")
        self.assertEqual(rescaled["status"], baseline["status"])
        self.assertFalse(baseline["semantics"]["raw_gap_gate"])
        self.assertAlmostEqual(
            baseline["normalised_within_group_drift_range"]["mean"],
            rescaled["normalised_within_group_drift_range"]["mean"],
            places=8,
        )
        self.assertFalse(
            baseline["ledger_coupled_evidence"]["primary_validity_evidence"]
        )

    def test_same_seed_across_loads_is_one_independent_cluster(self):
        samples = []
        for seed in (701, 702):
            for load in ("low", "high"):
                for candidate, delta in ((1, -1.0), (2, 1.0)):
                    sample = _sample(
                        f"{load}_{seed}",
                        candidate,
                        seed=seed,
                        delta_l=delta,
                    )
                    sample["run_id"] = f"isolated_{load}_seed{seed}"
                    sample["load_level"] = load
                    samples.append(sample)

        report = evaluate_action_controllability(
            samples,
            bootstrap_repeats=20,
            random_seed=19,
        )

        self.assertEqual(report["independent_clusters"], 2)
        self.assertEqual(
            report["semantics"]["cluster_unit"],
            "simulation_seed across all loads; run/source only when seed is absent",
        )

    def test_action_insensitive_potential_is_explicitly_monitor_only(self):
        samples = []
        for group_index in range(4):
            for candidate in range(1, 4):
                samples.append(
                    _sample(
                        f"constant{group_index}",
                        candidate,
                        seed=500 + group_index,
                        delta_l=0.0,
                    )
                )

        report = evaluate_five_layer_validity(
            samples,
            bootstrap_repeats=20,
            placebo_repeats=10,
        )

        self.assertEqual(
            report["layer2_action_controllable_drift"]["status"],
            "ACTION_INSENSITIVE_STATE_MONITOR_ONLY",
        )
        self.assertEqual(
            report["decision"]["recommended_role"],
            "STATE_MONITOR_ONLY_ACTION_INSENSITIVE",
        )
        self.assertTrue(report["decision"]["monitor_only_is_valid_outcome"])
        self.assertFalse(
            report["decision"]["online_action_score_enabled_by_this_report"]
        )

    def test_inconclusive_action_controllability_blocks_later_claims(self):
        samples = [
            _sample("single", 1, seed=610, delta_l=-1.0),
        ]
        report = evaluate_five_layer_validity(
            samples,
            wm_score_field="wm_score",
            predicted_drift_field="predicted_delta_l",
            closed_loop_report=_closed_loop_pass_report(),
            bootstrap_repeats=20,
            placebo_repeats=10,
        )
        self.assertEqual(
            report["decision"]["recommended_role"],
            "STATE_MONITOR_ACTION_CONTROLLABILITY_INCONCLUSIVE",
        )
        self.assertFalse(
            report["decision"]["online_action_score_enabled_by_this_report"]
        )

    def test_layer3_requires_real_wm_and_keeps_progress_secondary(self):
        samples = _informative_samples()
        missing = evaluate_incremental_information(
            samples,
            wm_score_field=None,
            bootstrap_repeats=20,
            placebo_repeats=10,
        )
        self.assertEqual(
            missing["status"],
            "NOT_EVALUATED_MISSING_TRUE_WM_SCORE_FIELD",
        )

        report = evaluate_incremental_information(
            samples,
            wm_score_field="wm_score",
            bootstrap_repeats=60,
            placebo_repeats=40,
            random_seed=23,
        )
        self.assertEqual(report["status"], "INCREMENTAL_INFORMATION_SUPPORTED")
        self.assertEqual(report["primary_outcome"], "realized_cost")
        self.assertGreater(
            report["independent_outcomes"]["realized_cost"]
            ["normalised_rmse_improvement"],
            0.0,
        )
        self.assertFalse(
            report["ledger_coupled_evidence"]["primary_validity_evidence"]
        )

    def test_layer4_exact_centred_prediction_has_zero_regret(self):
        samples = _informative_samples(group_count=6)
        report = evaluate_drift_prediction(
            samples,
            predicted_drift_field="predicted_delta_l",
            bootstrap_repeats=40,
            random_seed=29,
        )

        self.assertEqual(
            report["status"],
            "WORLD_MODEL_DRIFT_ESTIMATION_SUPPORTED",
        )
        self.assertAlmostEqual(report["normalised_rmse"], 0.0, places=12)
        self.assertAlmostEqual(
            report["ranking"]["top1_min_drift_accuracy"], 1.0, places=12
        )
        self.assertAlmostEqual(
            report["ranking"]["pairwise_concordance"], 1.0, places=12
        )
        self.assertAlmostEqual(
            report["ranking"]["normalised_selection_regret_fraction_of_group_range"]
            ["mean"],
            0.0,
            places=12,
        )

    def test_all_five_supported_layers_enable_online_auxiliary_claim(self):
        report = evaluate_five_layer_validity(
            _informative_samples(),
            wm_score_field="wm_score",
            predicted_drift_field="predicted_delta_l",
            closed_loop_report=_closed_loop_pass_report(),
            strict_isolated=True,
            bootstrap_repeats=60,
            placebo_repeats=40,
            random_seed=31,
        )

        self.assertEqual(report["verdict"], "ONLINE_ACTION_AUXILIARY_SUPPORTED")
        self.assertTrue(report["passed"])
        self.assertFalse(report["semantics"]["raw_delta_L_gap_is_validity_gate"])
        self.assertFalse(report["semantics"]["productive_progress_primary_evidence"])
        for key in (
            "layer1_state_potential",
            "layer2_action_controllable_drift",
            "layer3_incremental_information",
            "layer4_world_model_drift_estimation",
            "layer5_closed_loop_stability",
        ):
            self.assertTrue(report[key]["supported"], key)


if __name__ == "__main__":
    unittest.main()
