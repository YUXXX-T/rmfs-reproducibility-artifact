import unittest
from unittest import mock

import WorldModel.evaluation.evaluate_lyapunov_validity as validity

from WorldModel.evaluation.evaluate_lyapunov_validity import (
    FORMAL_WORK_ONLY_RANDOM_SEED,
    WORK_GROUP_RANGE_FIELD,
    _attach_work_group_range,
    _centred_rows,
    _formal_work_group_range_protocol,
    _formal_work_only_protocol,
    _groups,
    evaluate_incremental_information,
)


RAW_V1_PROTOCOL_SHA256 = (
    "external-fingerprint-omitted"
)
GROUP_RANGE_V2_PROTOCOL_SHA256 = (
    "external-fingerprint-omitted"
)


def _samples(*, seeds=range(421, 431)):
    rows = []
    directions = (-1.5, -0.5, 0.5, 1.5)
    wm_pattern = (1.0, -1.0, -1.0, 1.0)
    loads = ("low", "mid", "high")
    for seed_index, seed in enumerate(seeds):
        for load_index, load in enumerate(loads):
            for group_index in range(2):
                group = f"seed{seed}_{load}_group{group_index}"
                scale = (
                    0.003
                    * (1.0 + load_index)
                    * (40.0 if seed == 426 and group_index == 0 else 1.0)
                )
                offset = (seed_index + load_index + group_index) % 4
                for candidate in range(4):
                    position = (candidate + offset) % 4
                    direction = directions[position]
                    work = scale * direction
                    wm = wm_pattern[position]
                    # For four evenly spaced candidates, range-normalised work
                    # is direction / 3.0 irrespective of the context scale.
                    group_range = direction / 3.0
                    intercept = 100.0 + 10.0 * load_index + 0.1 * seed_index

                    def outcome(wm_weight, drift_weight, extra=0.0):
                        return (
                            intercept
                            + extra
                            + wm_weight * wm
                            + drift_weight * group_range
                        )

                    rows.append({
                        "run_id": f"range_{load}_seed{seed}",
                        "simulation_seed": int(seed),
                        "load_level": load,
                        "candidate_group_id": group,
                        "candidate_key": f"{group}_candidate{candidate}",
                        "lyapunov_l0_valid": True,
                        "lyapunov_l0_delta": float(work),
                        "lyapunov_l0_component_delta": {
                            "work": float(work),
                            "station": 0.0,
                            "traffic": 0.0,
                            "stall": 0.0,
                            "arrival": 0.0,
                        },
                        "wm_score": float(wm),
                        "realized_cost": outcome(0.5, 2.0),
                        "_validation_outcomes": {
                            "wait_or_stall_mean": outcome(0.3, 1.8, 20.0),
                            "average_excess_delay_mean": outcome(0.4, 1.5, 40.0),
                            "station_queue_delta_mean": outcome(0.2, 0.6, 5.0),
                            "station_load_imbalance_mean": outcome(0.2, 0.4, 3.0),
                            "bottleneck_cvar_mean": outcome(0.3, 0.5, 10.0),
                            "completed_orders_cost": outcome(0.2, 0.3, 8.0),
                            "deadlock_risk_max": outcome(0.4, 1.7, 30.0),
                        },
                        "rollout_blocked_moves": outcome(0.2, 0.25, 6.0),
                        "rollout_vertex_conflicts": 0.0,
                        "rollout_swap_conflicts": 0.0,
                    })
    return rows


def _evaluate(samples):
    with mock.patch.multiple(
        validity,
        FORMAL_WORK_ONLY_BOOTSTRAP_REPEATS=80,
        FORMAL_WORK_ONLY_PLACEBO_REPEATS=40,
    ):
        return evaluate_incremental_information(
            samples,
            wm_score_field="wm_score",
            wm_score_provenance_verified=True,
            candidate_component="work_group_range",
            minimum_independent_clusters=10,
            bootstrap_repeats=80,
            placebo_repeats=40,
            random_seed=FORMAL_WORK_ONLY_RANDOM_SEED,
        )


class WorkGroupRangeV2Tests(unittest.TestCase):
    def test_raw_v1_protocol_hash_is_unchanged(self):
        self.assertEqual(
            _formal_work_only_protocol()["protocol_sha256"],
            RAW_V1_PROTOCOL_SHA256,
        )

    def test_protocol_freezes_untouched_seed_set_and_no_gap(self):
        protocol = _formal_work_group_range_protocol()
        self.assertEqual(protocol["candidate_component"], "work_group_range")
        self.assertEqual(protocol["required_simulation_seeds"], list(range(421, 431)))
        self.assertFalse(protocol["candidate_set_contract"]["raw_gap_gate"])
        self.assertFalse(protocol["candidate_set_contract"]["load_gate"])
        self.assertEqual(
            protocol["protocol_sha256"],
            GROUP_RANGE_V2_PROTOCOL_SHA256,
        )

    def test_transform_is_invariant_to_context_scale(self):
        samples = _samples(seeds=(421, 426))
        fields = ["wm_score", "lyapunov_l0_delta", "realized_cost"]
        rows, _ = _centred_rows(_groups(samples), fields)
        field = _attach_work_group_range(rows)
        self.assertEqual(field, WORK_GROUP_RANGE_FIELD)
        grouped = {}
        for row in rows:
            grouped.setdefault(row["group"], []).append(row[field])
        signatures = {
            tuple(round(value, 12) for value in sorted(values))
            for values in grouped.values()
        }
        self.assertEqual(
            signatures,
            {(-0.5, round(-1.0 / 6.0, 12), round(1.0 / 6.0, 12), 0.5)},
        )

    def test_untouched_seeds_can_satisfy_v2_and_emit_preflight(self):
        report = _evaluate(_samples())
        contract = report["formal_candidate_contract"]
        self.assertEqual(report["candidate_component"], "work_group_range")
        self.assertEqual(report["status"], "INCREMENTAL_INFORMATION_SUPPORTED")
        self.assertTrue(report["supported"])
        self.assertTrue(contract["passed"])
        self.assertTrue(
            contract["data_contract"]["checks"]["exact_frozen_simulation_seed_set"]
        )
        self.assertEqual(
            contract["protocol"]["required_simulation_seeds"],
            list(range(421, 431)),
        )
        preflight = report["work_group_range_preflight"]
        self.assertEqual(preflight["status"], "DEVELOPMENT_DIAGNOSTIC_COMPLETE")
        self.assertGreater(
            preflight["leave_one_candidate_out_stability"]["overall"]["trials"],
            0,
        )
        self.assertEqual(
            preflight["leave_one_candidate_out_stability"]
            ["online_all_idle_above_observed_max"],
            "NOT_EVALUATED",
        )

    def test_development_seeds_cannot_be_relabelled_as_formal_v2(self):
        report = _evaluate(_samples(seeds=range(411, 421)))
        contract = report["formal_candidate_contract"]
        self.assertFalse(contract["passed"])
        self.assertFalse(
            contract["data_contract"]["checks"]["exact_frozen_simulation_seed_set"]
        )
        self.assertEqual(
            contract["status"],
            "FORMAL_WORK_GROUP_RANGE_DATA_CONTRACT_FAILED",
        )
        self.assertFalse(report["supported"])


if __name__ == "__main__":
    unittest.main()
