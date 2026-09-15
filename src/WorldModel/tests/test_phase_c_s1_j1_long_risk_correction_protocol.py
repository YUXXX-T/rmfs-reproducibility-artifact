from __future__ import annotations

import unittest

from WorldModel.evaluation.phase_c_s1_j1_long_risk_correction_protocol import (
    ALL_ARM_KEYS,
    COMPARISONS,
    INTERACTIONS,
    NEW_ARM_KEYS,
    NEW_ARM_SPECS,
    SOURCE_ROOT,
    formal_protocol,
    selector_config,
)
from WorldModel.evaluation.run_phase_c_s1_j1_long_risk_correction import (
    interaction_delta_from_values,
)
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


class PhaseCS1J1LongRiskCorrectionProtocolTest(unittest.TestCase):
    def test_exact_new_arm_matrix(self) -> None:
        self.assertEqual(
            NEW_ARM_KEYS,
            ("s0_j1", "s1_event_j1", "s1_terminal_j1", "s1_combo_j1"),
        )
        self.assertEqual(len(ALL_ARM_KEYS), 10)
        self.assertEqual(NEW_ARM_SPECS["s0_j1"]["selector"], "s0")
        self.assertIsNone(NEW_ARM_SPECS["s0_j1"]["signal"])
        self.assertEqual(
            NEW_ARM_SPECS["s1_event_j1"]["signal"], "event_logit"
        )
        self.assertEqual(
            NEW_ARM_SPECS["s1_terminal_j1"]["signal"], "terminal"
        )
        self.assertEqual(NEW_ARM_SPECS["s1_combo_j1"]["signal"], "combo")

    def test_selector_configs_are_isolated(self) -> None:
        s0 = selector_config("s0_j1")
        event = selector_config("s1_event_j1")
        terminal = selector_config("s1_terminal_j1")
        combo = selector_config("s1_combo_j1")
        self.assertEqual(s0["energy_scoring_mode"], "off")
        self.assertEqual(event["energy_drift_signal"], "event_logit")
        self.assertEqual(terminal["energy_drift_signal"], "terminal")
        self.assertEqual(combo["energy_drift_signal"], "combo")
        event["energy_drift_signal"] = "changed"
        self.assertEqual(
            selector_config("s1_event_j1")["energy_drift_signal"],
            "event_logit",
        )

    def test_protocol_is_physical_only_without_station_queue(self) -> None:
        protocol = formal_protocol(SOURCE_ROOT)
        station = protocol["station_admission"]
        self.assertEqual(station["mode"], STATION_ADMISSION_PHYSICAL_ONLY)
        self.assertTrue(station["physical_occupancy_cap_enforced"])
        self.assertFalse(station["in_transit_committed_cap_enforced"])
        self.assertFalse(station["eta_controller"])
        self.assertFalse(station["dispatch_preserving_station_queue"])
        self.assertEqual(protocol["formal_test"]["new_simulation_count"], 120)

    def test_required_main_effect_and_interaction_contrasts_exist(self) -> None:
        comparison_names = {row[0] for row in COMPARISONS}
        self.assertIn("event_j1_minus_event_j0", comparison_names)
        self.assertIn("combo_j1_minus_event_j1", comparison_names)
        self.assertIn("combo_j1_minus_greedy", comparison_names)
        self.assertIn("combo_j1_minus_hungarian", comparison_names)
        interaction_names = {row[0] for row in INTERACTIONS}
        self.assertEqual(
            interaction_names,
            {"event_s1_x_j1", "terminal_s1_x_j1", "combo_s1_x_j1"},
        )

    def test_difference_in_differences_sign(self) -> None:
        # J1 adds 10 to S0 and 35 to S1, so the S1 x J1 interaction is +25.
        self.assertEqual(
            interaction_delta_from_values(100, 110, 120, 155),
            25.0,
        )
        # An equal J1 main effect in S0 and S1 has zero interaction.
        self.assertEqual(
            interaction_delta_from_values(100, 130, 120, 150),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
