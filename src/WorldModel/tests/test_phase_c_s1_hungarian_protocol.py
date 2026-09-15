import unittest

from Policies.TaskAssigner import WorldModelTaskAssigner
from WorldModel.evaluation.phase_c_s1_hungarian_protocol import (
    ARMS,
    GREEDY_LABEL,
    HUNGARIAN_LABEL,
    LOADS,
    PHASEC_CONFIG,
    PHASEC_LABEL,
    PHASEC_S1_LABEL,
    S1_CONFIG,
    SEEDS,
    TICKS,
    TOP_M,
    formal_protocol,
)


class PhaseCS1HungarianProtocolTest(unittest.TestCase):
    def test_formal_scope_is_exact(self):
        self.assertEqual(LOADS, ("low", "mid", "high"))
        self.assertEqual(SEEDS, tuple(range(491, 501)))
        self.assertEqual(TICKS, 1500)
        self.assertEqual(TOP_M, 10)
        self.assertEqual(
            ARMS,
            (
                GREEDY_LABEL,
                HUNGARIAN_LABEL,
                PHASEC_LABEL,
                PHASEC_S1_LABEL,
            ),
        )

    def test_no_assign_and_new_dispatch_are_excluded(self):
        for config in (PHASEC_CONFIG, S1_CONFIG):
            self.assertFalse(config["include_no_assign_candidate"])
            self.assertEqual(config["dispatch_potential_mode"], "off")
            self.assertEqual(config["work_drift_mode"], "off")
            self.assertEqual(config["lyapunov_l0_mode"], "off")

    def test_s1_is_the_frozen_c3_l025_configuration(self):
        self.assertEqual(S1_CONFIG["energy_scoring_mode"], "conversion")
        self.assertEqual(S1_CONFIG["energy_drift_signal"], "combo")
        self.assertEqual(S1_CONFIG["energy_gate_mode"], "sigmoid")
        self.assertEqual(S1_CONFIG["energy_gate_window"], 200)
        self.assertEqual(S1_CONFIG["energy_gate_warmup"], 50)
        self.assertEqual(S1_CONFIG["energy_gate_quantile"], 0.8)
        self.assertEqual(S1_CONFIG["energy_conv_lambda"], 0.25)
        self.assertEqual(
            S1_CONFIG["long_risk_beta"], {"terminal": 0.5, "cvar": 0.2}
        )

    def test_assigner_accepts_both_frozen_configs(self):
        base = WorldModelTaskAssigner(
            checkpoint_path="unused.pt", top_m=TOP_M, **PHASEC_CONFIG
        )
        s1 = WorldModelTaskAssigner(
            checkpoint_path="unused.pt",
            top_m=TOP_M,
            energy_conv_random_flip_seed=0,
            **S1_CONFIG,
        )
        self.assertFalse(base.include_no_assign_candidate)
        self.assertEqual(base.energy_scoring_mode, "off")
        self.assertFalse(s1.include_no_assign_candidate)
        self.assertEqual(s1.energy_scoring_mode, "conversion")
        self.assertEqual(s1.energy_conv_lambda, 0.25)

    def test_protocol_declares_throughput_non_gating(self):
        protocol = formal_protocol()
        self.assertFalse(
            protocol["primary_analysis"]["throughput_is_hard_gate"]
        )
        self.assertTrue(protocol["forbidden"]["native_no_assign"])
        self.assertTrue(protocol["forbidden"]["post_test_s1_tuning"])


if __name__ == "__main__":
    unittest.main()

