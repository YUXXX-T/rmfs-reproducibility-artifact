import math
import unittest

from WorldModel.evaluation.evaluate_lyapunov_validity import (
    evaluate_incremental_information,
)


COMPONENTS = ("work", "station", "traffic", "stall", "arrival")


def _sample(
    *,
    seed,
    load,
    group,
    candidate,
    wm,
    outcome,
    work=0.0,
    station=0.0,
    traffic=0.0,
    stall=0.0,
    arrival=0.0,
):
    component_delta = {
        "work": float(work),
        "station": float(station),
        "traffic": float(traffic),
        "stall": float(stall),
        "arrival": float(arrival),
    }
    return {
        "run_id": f"isolated_{load}_seed{seed}",
        "simulation_seed": int(seed),
        "load_level": str(load),
        "candidate_group_id": str(group),
        "candidate_key": f"{group}_candidate{candidate}",
        "lyapunov_l0_valid": True,
        "lyapunov_l0_delta": float(sum(component_delta.values())),
        "lyapunov_l0_component_delta": component_delta,
        "wm_score": float(wm),
        "realized_cost": float(outcome),
    }


def _evaluate(samples):
    return evaluate_incremental_information(
        samples,
        wm_score_field="wm_score",
        outcome_fields=("realized_cost",),
        bootstrap_repeats=30,
        placebo_repeats=10,
        random_seed=811,
    )


def _load_interaction_samples():
    """Low is intentionally over-represented while high reverses drift sign."""
    samples = []
    work_pattern = (-1.0, 0.0, 1.0)
    wm_pattern = (0.5, -1.0, 0.5)
    groups_per_seed = {"low": 4, "mid": 1, "high": 1}
    drift_sign = {"low": 1.0, "mid": 1.0, "high": -1.0}
    for seed in range(721, 727):
        for load, group_count in groups_per_seed.items():
            for group_index in range(group_count):
                group = f"{load}_{seed}_{group_index}"
                # Rotate both patterns without changing their orthogonality.
                offset = (seed + group_index) % 3
                for candidate in range(3):
                    work = work_pattern[(candidate + offset) % 3]
                    wm = wm_pattern[(candidate + offset) % 3]
                    outcome = 50.0 + 0.75 * wm + drift_sign[load] * work
                    samples.append(
                        _sample(
                            seed=seed,
                            load=load,
                            group=group,
                            candidate=candidate,
                            wm=wm,
                            outcome=outcome,
                            work=work,
                        )
                    )
    return samples


class LyapunovLayer3DiagnosticTests(unittest.TestCase):
    def test_near_zero_by_load_variation_never_explodes_nrmse(self):
        samples = []
        wm_pattern = (-1.0, 0.0, 1.0)
        work_pattern = (0.0, 1.0, -1.0)
        for seed in range(701, 707):
            for load in ("low", "high"):
                group = f"{load}_{seed}"
                for candidate, (wm, work) in enumerate(
                    zip(wm_pattern, work_pattern)
                ):
                    if load == "low":
                        outcome = 100.0 + 2.0 * wm + work
                    else:
                        # This is below the float32 label-resolution guard after
                        # candidate-group centring.  It is not evidence for or
                        # against a load gate.
                        outcome = 100.0 + 1e-10 * candidate
                    samples.append(
                        _sample(
                            seed=seed,
                            load=load,
                            group=group,
                            candidate=candidate,
                            wm=wm,
                            outcome=outcome,
                            work=work,
                        )
                    )

        report = _evaluate(samples)
        primary = report["independent_outcomes"]["realized_cost"]
        high = primary["by_load"]["high"]

        self.assertEqual(high["status"], "NEAR_ZERO_VARIATION")
        self.assertEqual(
            high["target_variation"]["status"], "NEAR_ZERO_VARIATION"
        )
        self.assertIsNone(high["normalised_rmse_improvement"])
        self.assertIsNone(high["wm_only_normalised_rmse"])
        self.assertIsNone(high["wm_plus_drift_normalised_rmse"])
        self.assertTrue(math.isfinite(high["wm_only_rmse"]))
        self.assertTrue(math.isfinite(high["wm_plus_drift_rmse"]))

    def test_load_interaction_detects_heterogeneity_but_does_not_enable_gate(self):
        report = _evaluate(_load_interaction_samples())
        diagnostics = report["load_component_diagnostics_v2"]
        outcome = diagnostics["by_outcome"]["realized_cost"]
        shared = outcome["shared_coefficient_model"]["comparison"]
        conditioned = outcome["load_conditioned_model"]

        self.assertEqual(
            shared["load_heterogeneity_classification"],
            "LOAD_HETEROGENEOUS_POINT_ESTIMATES",
        )
        self.assertGreater(
            shared["by_load"]["low"]["normalised_rmse_improvement"], 0.0
        )
        self.assertGreater(
            shared["by_load"]["mid"]["normalised_rmse_improvement"], 0.0
        )
        self.assertLess(
            shared["by_load"]["high"]["normalised_rmse_improvement"], 0.0
        )
        self.assertEqual(
            conditioned["status"], "DIAGNOSTIC_ONLY_NOT_AN_ONLINE_GATE"
        )
        for load in ("low", "mid", "high"):
            self.assertGreater(
                conditioned["comparison"]["by_load"][load]
                ["normalised_rmse_improvement"],
                0.0,
            )

        # A better load-interaction fit diagnoses coefficient mismatch.  It
        # must not silently certify a soft gate or universal online auxiliary.
        self.assertFalse(diagnostics["semantics"]["enables_soft_gate"])
        self.assertFalse(report["supported"])
        self.assertFalse(
            report["primary_load_robustness"]
            ["all_informative_loads_positive_with_seed_cluster_ci"]
        )
        self.assertFalse(
            diagnostics["semantics"]["station_load_imbalance_is_L_station"]
        )

    def test_component_ablation_locates_stable_work_and_harmful_arrival(self):
        samples = []
        x_pattern = (-1.0, 0.0, 1.0)
        for seed_index, seed in enumerate(range(741, 747)):
            arrival_sign = 1.0 if seed_index % 2 == 0 else -1.0
            group = f"mid_{seed}"
            for candidate, work in enumerate(x_pattern):
                # Work has a stable relation to the independent outcome.
                # Arrival changes sign by held-out seed.  In leave-seed-out
                # validation its fitted relation therefore anti-generalises.
                arrival = 3.0 * arrival_sign * work
                samples.append(
                    _sample(
                        seed=seed,
                        load="mid",
                        group=group,
                        candidate=candidate,
                        wm=0.0,
                        outcome=20.0 + 2.0 * work,
                        work=work,
                        arrival=arrival,
                    )
                )

        report = _evaluate(samples)
        diagnostics = report["load_component_diagnostics_v2"]
        ablation = diagnostics["by_outcome"]["realized_cost"][
            "component_ablation"
        ]
        work = ablation["work"]
        arrival = ablation["arrival"]

        self.assertEqual(work["status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(arrival["status"], "DIAGNOSTIC_ONLY")
        work_gain = work["comparison"]["overall"][
            "normalised_rmse_improvement"
        ]
        arrival_gain = arrival["comparison"]["overall"][
            "normalised_rmse_improvement"
        ]
        self.assertGreater(work_gain, 0.5)
        self.assertLess(arrival_gain, 0.0)
        self.assertGreater(work_gain, arrival_gain)
        self.assertEqual(
            ablation["all_active_components_free_coefficients"]["status"],
            "DIAGNOSTIC_ONLY_NOT_A_REWEIGHTED_L_CERTIFICATE",
        )

    def test_seed_folds_span_all_loads_and_load_equal_summary_is_explicit(self):
        report = _evaluate(_load_interaction_samples())
        primary = report["independent_outcomes"]["realized_cost"]
        outcome = report["load_component_diagnostics_v2"]["by_outcome"]
        outcome = outcome["realized_cost"]
        shared = outcome["shared_coefficient_model"]

        self.assertEqual(primary["independent_seed_clusters"], 6)
        self.assertEqual(primary["folds"], 6)
        for fold in shared["wm_plus_drift_fold_diagnostics"]:
            self.assertEqual(len(fold["held_out_clusters"]), 1)
            self.assertEqual(fold["test_loads"], ["high", "low", "mid"])

        comparison = shared["comparison"]
        valid = [
            row["normalised_rmse_improvement"]
            for row in comparison["by_load"].values()
            if row["normalised_rmse_improvement"] is not None
        ]
        expected_load_equal = sum(valid) / len(valid)
        self.assertAlmostEqual(
            comparison["load_equal_mean_normalised_rmse_improvement"],
            expected_load_equal,
            places=12,
        )
        # Low has four times as many groups per seed, so this must not be an
        # alias for the sample-weighted overall metric.
        self.assertNotAlmostEqual(
            comparison["load_equal_mean_normalised_rmse_improvement"],
            comparison["overall"]["normalised_rmse_improvement"],
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
