import unittest
from unittest import mock

import WorldModel.evaluation.evaluate_lyapunov_validity as validity

from WorldModel.evaluation.evaluate_lyapunov_validity import (
    DEFAULT_OUTCOME_FIELDS,
    DEPENDENT_LEDGER_FIELD,
    FORMAL_WORK_ONLY_BOOTSTRAP_REPEATS,
    FORMAL_WORK_ONLY_MINIMUM_SEED_CLUSTERS,
    FORMAL_WORK_ONLY_PLACEBO_REPEATS,
    FORMAL_WORK_ONLY_RANDOM_SEED,
    _attach_centred_component_deltas,
    _centred_rows,
    _formal_work_only_candidate_contract,
    _formal_work_only_protocol,
    _groups,
    evaluate_incremental_information,
)


LOAD_OFFSET = {"low": 0.0, "mid": 7.0, "high": 15.0}
FROZEN_PROTOCOL_SHA256 = (
    "external-fingerprint-omitted"
)


def _formal_samples(*, seeds=range(411, 421), omit_pair=None):
    """Synthetic paired-load data with stable work and unstable arrival drift."""
    samples = []
    work_pattern = (-1.0, 0.0, 1.0)
    wm_pattern = (0.5, -1.0, 0.5)
    for seed_index, seed in enumerate(seeds):
        for load in ("low", "mid", "high"):
            if omit_pair == (int(seed), load):
                continue
            offset = (seed_index + ("low", "mid", "high").index(load)) % 3
            group = f"{load}_seed{seed}"
            arrival_sign = 1.0 if seed_index % 2 == 0 else -1.0
            for candidate in range(3):
                work = work_pattern[(candidate + offset) % 3]
                wm = wm_pattern[(candidate + offset) % 3]
                # Arrival is deliberately high-variance and changes its
                # relation by seed.  The frozen candidate must remain work,
                # not silently fall back to total Delta L.
                arrival = 4.0 * arrival_sign * work
                intercept = 100.0 + LOAD_OFFSET[load] + 0.1 * seed_index

                def outcome(wm_weight, work_weight, extra=0.0):
                    return (
                        intercept
                        + extra
                        + wm_weight * wm
                        + work_weight * work
                    )

                samples.append({
                    "run_id": f"formal_{load}_seed{seed}",
                    "simulation_seed": int(seed),
                    "load_level": load,
                    "candidate_group_id": group,
                    "candidate_key": f"{group}_candidate{candidate}",
                    "lyapunov_l0_valid": True,
                    "lyapunov_l0_delta": float(work + arrival),
                    "lyapunov_l0_component_delta": {
                        "work": float(work),
                        "station": 0.0,
                        "traffic": 0.0,
                        "stall": 0.0,
                        "arrival": float(arrival),
                    },
                    "wm_score": float(wm),
                    "realized_cost": outcome(0.7, 2.0),
                    "_validation_outcomes": {
                        "wait_or_stall_mean": outcome(0.4, 1.6, 20.0),
                        "average_excess_delay_mean": outcome(0.6, 1.3, 40.0),
                        "station_queue_delta_mean": outcome(0.2, 0.4, 5.0),
                        "station_load_imbalance_mean": outcome(0.2, 0.2, 3.0),
                        "bottleneck_cvar_mean": outcome(0.5, 0.5, 10.0),
                        "completed_orders_cost": outcome(0.3, 0.3, 8.0),
                        "deadlock_risk_max": outcome(0.5, 1.8, 30.0),
                    },
                    "rollout_blocked_moves": outcome(0.3, 0.25, 6.0),
                    "rollout_vertex_conflicts": 0.0,
                    "rollout_swap_conflicts": 0.0,
                })
    return samples


def _evaluate(samples, *, minimum_independent_clusters=None):
    # Keep the end-to-end synthetic test cheap while exercising the exact
    # same frozen-parameter equality check.  Production constants are tested
    # separately and remain 5000/2000/20260715.
    with mock.patch.multiple(
        validity,
        FORMAL_WORK_ONLY_BOOTSTRAP_REPEATS=80,
        FORMAL_WORK_ONLY_PLACEBO_REPEATS=40,
    ):
        return evaluate_incremental_information(
            samples,
            wm_score_field="wm_score",
            wm_score_provenance_verified=True,
            candidate_component="work",
            minimum_independent_clusters=minimum_independent_clusters,
            bootstrap_repeats=80,
            placebo_repeats=40,
            random_seed=FORMAL_WORK_ONLY_RANDOM_SEED,
        )


def _formal_rows(samples):
    fields = [
        "wm_score",
        "lyapunov_l0_delta",
        *DEFAULT_OUTCOME_FIELDS,
        DEPENDENT_LEDGER_FIELD,
    ]
    centred, _ = _centred_rows(_groups(samples), fields)
    component_fields = _attach_centred_component_deltas(centred)
    return centred, component_fields["work"]


class FormalWorkOnlyLayer3Tests(unittest.TestCase):
    def test_paired_ten_seed_work_candidate_is_formally_supported(self):
        self.assertEqual(
            _formal_work_only_protocol()["protocol_sha256"],
            FROZEN_PROTOCOL_SHA256,
        )
        report = _evaluate(_formal_samples())
        contract = report["formal_candidate_contract"]

        self.assertEqual(report["candidate_component"], "work")
        self.assertIn(".work", report["candidate_predictor_field"])
        self.assertEqual(report["status"], "INCREMENTAL_INFORMATION_SUPPORTED")
        self.assertTrue(report["supported"])
        self.assertTrue(contract["passed"])
        self.assertTrue(contract["data_contract"]["passed"])
        self.assertEqual(
            contract["data_contract"]["independent_seed_clusters"], 10
        )
        self.assertEqual(contract["data_contract"]["paired_seed_clusters"], 10)

        protocol = contract["protocol"]
        self.assertTrue(protocol["frozen"])
        self.assertEqual(protocol["candidate_component"], "work")
        self.assertEqual(protocol["primary_outcome"], "realized_cost")
        self.assertEqual(protocol["required_loads"], ["low", "mid", "high"])
        self.assertEqual(protocol["resampling"]["bootstrap_repeats"], 80)
        self.assertEqual(protocol["resampling"]["placebo_repeats"], 40)
        self.assertTrue(contract["resampling_contract"]["passed"])

        primary = contract["primary_evidence"]
        self.assertTrue(primary["passed"])
        self.assertTrue(all(primary["checks"].values()))
        self.assertEqual(len(primary["seed_point_improvements"]), 10)
        self.assertTrue(all(
            value > 0.0
            for value in primary["seed_point_improvements"].values()
        ))
        self.assertEqual(len(primary["work_coefficients_by_fold"]), 10)
        self.assertTrue(all(
            value > 0.0 for value in primary["work_coefficients_by_fold"]
        ))
        self.assertGreaterEqual(contract["supportive_secondary_outcomes"], 2)
        self.assertTrue(all(contract["secondary_checks"].values()))
        self.assertTrue(all(contract["guardrail_checks"].values()))

    def test_production_resampling_contract_is_frozen(self):
        protocol = _formal_work_only_protocol()
        self.assertEqual(
            protocol["resampling"],
            {
                "bootstrap_repeats": FORMAL_WORK_ONLY_BOOTSTRAP_REPEATS,
                "placebo_repeats": FORMAL_WORK_ONLY_PLACEBO_REPEATS,
                "random_seed": FORMAL_WORK_ONLY_RANDOM_SEED,
            },
        )

        samples = _formal_samples()
        diagnostic = _evaluate(samples)
        centred, predictor_field = _formal_rows(samples)
        contract = _formal_work_only_candidate_contract(
            diagnostic["independent_outcomes"],
            centred,
            primary_outcome="realized_cost",
            candidate_predictor_field=predictor_field,
            minimum_independent_clusters=(
                FORMAL_WORK_ONLY_MINIMUM_SEED_CLUSTERS
            ),
            wm_score_provenance_verified=True,
            bootstrap_repeats=80,
            placebo_repeats=40,
            random_seed=FORMAL_WORK_ONLY_RANDOM_SEED,
        )
        self.assertFalse(contract["resampling_contract"]["passed"])
        self.assertEqual(
            contract["status"],
            "FORMAL_WORK_ONLY_RESAMPLING_CONTRACT_FAILED",
        )
        self.assertFalse(contract["passed"])

    def test_formal_work_minimum_seed_count_is_not_tunable(self):
        with self.assertRaises(ValueError):
            _evaluate(
                _formal_samples(),
                minimum_independent_clusters=(
                    FORMAL_WORK_ONLY_MINIMUM_SEED_CLUSTERS - 1
                ),
            )

    def test_missing_seed_load_pair_fails_frozen_data_contract(self):
        report = _evaluate(
            _formal_samples(omit_pair=(417, "high"))
        )
        contract = report["formal_candidate_contract"]

        self.assertFalse(contract["data_contract"]["passed"])
        self.assertFalse(
            contract["data_contract"]["checks"]
            ["every_seed_has_exact_paired_low_mid_high_arms"]
        )
        self.assertEqual(contract["data_contract"]["paired_seed_clusters"], 9)
        self.assertFalse(contract["passed"])
        self.assertFalse(report["supported"])

    def test_nine_seed_development_data_cannot_formally_certify(self):
        report = _evaluate(_formal_samples(seeds=range(401, 410)))
        contract = report["formal_candidate_contract"]

        self.assertEqual(
            contract["data_contract"]["independent_seed_clusters"], 9
        )
        self.assertFalse(
            contract["data_contract"]["checks"]
            ["minimum_independent_seed_clusters"]
        )
        self.assertFalse(contract["passed"])
        self.assertFalse(report["supported"])

    def test_heuristic_cost_cannot_formally_certify_work_candidate(self):
        samples = _formal_samples()
        for sample in samples:
            sample["heuristic_cost"] = sample["wm_score"]
        report = evaluate_incremental_information(
            samples,
            wm_score_field="heuristic_cost",
            candidate_component="work",
            bootstrap_repeats=20,
            placebo_repeats=10,
            random_seed=23,
        )
        contract = report["formal_candidate_contract"]

        self.assertEqual(
            report["status"],
            "PROXY_ONLY_NOT_A_WORLD_MODEL_INCREMENTAL_TEST",
        )
        self.assertFalse(
            contract["semantic_checks"]
            ["uses_checkpoint_scored_world_model_value"]
        )
        self.assertFalse(contract["passed"])
        self.assertFalse(report["supported"])

    def test_default_total_mode_remains_backward_compatible(self):
        samples = _formal_samples()
        implicit = evaluate_incremental_information(
            samples,
            wm_score_field="wm_score",
            bootstrap_repeats=30,
            placebo_repeats=10,
            random_seed=19,
        )
        explicit = evaluate_incremental_information(
            samples,
            wm_score_field="wm_score",
            candidate_component="total",
            bootstrap_repeats=30,
            placebo_repeats=10,
            random_seed=19,
        )

        self.assertEqual(implicit["candidate_component"], "total")
        self.assertIsNone(implicit["formal_candidate_contract"])
        self.assertEqual(implicit["status"], explicit["status"])
        self.assertEqual(
            implicit["independent_outcomes"]["realized_cost"]
            ["normalised_rmse_improvement"],
            explicit["independent_outcomes"]["realized_cost"]
            ["normalised_rmse_improvement"],
        )


if __name__ == "__main__":
    unittest.main()
