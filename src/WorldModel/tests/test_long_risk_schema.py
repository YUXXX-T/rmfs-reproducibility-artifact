import unittest

import torch

from Policies.TaskAssigner.WorldModelTaskAssigner.world_model_task_assigner import (
    WorldModelTaskAssigner,
)
from WorldModel.core.long_risk_schema import (
    LONG_RISK_OUTPUT_DIM,
    LONG_RISK_OUTPUT_NAMES,
    LONG_RISK_SCHEMA_VERSION,
    decode_long_risk_predictions,
    long_risk_drift_signal_key,
    long_risk_runtime_contract,
)
from WorldModel.core.model import LongRiskHead


SENTINEL = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


class LongRiskSchemaTest(unittest.TestCase):
    def test_sentinel_decodes_event_and_quantile_combo_correctly(self):
        decoded = decode_long_risk_predictions(SENTINEL)

        self.assertAlmostEqual(decoded["long_risk_peak_q90"], 10.0)
        self.assertAlmostEqual(decoded["long_risk_peak_q95"], 20.0)
        self.assertAlmostEqual(decoded["long_risk_cvar_q90"], 30.0)
        self.assertAlmostEqual(decoded["long_risk_terminal_q90"], 40.0)
        self.assertAlmostEqual(decoded["long_risk_delta_group_q90"], 50.0)
        self.assertAlmostEqual(decoded["long_risk_event_logit"], 60.0)
        self.assertAlmostEqual(decoded["long_risk_quantile_combo"], 33.0)
        self.assertAlmostEqual(decoded["long_risk_combo_q90"], 33.0)

    def test_assigner_runtime_uses_the_same_strict_decoder(self):
        assigner = WorldModelTaskAssigner(
            checkpoint_path="unused-unit-test.pt"
        )
        decoded = assigner._long_risk_components(
            {"long_risk_preds": torch.tensor(SENTINEL)}
        )

        self.assertAlmostEqual(decoded["long_risk_event_logit"], 60.0)
        self.assertAlmostEqual(decoded["long_risk_quantile_combo"], 33.0)
        self.assertEqual(
            long_risk_drift_signal_key("combo"),
            "long_risk_quantile_combo",
        )
        self.assertEqual(
            long_risk_drift_signal_key("event_logit"),
            "long_risk_event_logit",
        )

    def test_s1_conversion_reads_combo_and_legacy_event_explicitly(self):
        decoded = decode_long_risk_predictions(SENTINEL)

        combo_assigner = WorldModelTaskAssigner(
            checkpoint_path="unused-unit-test.pt",
            energy_scoring_mode="conversion",
            energy_drift_signal="combo",
            energy_gate_mode="off",
        )
        combo_record = {"score": 0.0, "best_robot": 1, **decoded}
        combo_assigner._apply_energy_conversion([combo_record])

        event_assigner = WorldModelTaskAssigner(
            checkpoint_path="unused-unit-test.pt",
            energy_scoring_mode="conversion",
            energy_drift_signal="event_logit",
            energy_gate_mode="off",
        )
        event_record = {"score": 0.0, "best_robot": 1, **decoded}
        event_assigner._apply_energy_conversion([event_record])

        self.assertAlmostEqual(combo_record["energy_conv_signal"], 33.0)
        self.assertAlmostEqual(event_record["energy_conv_signal"], 60.0)

    def test_present_vector_must_have_exactly_six_channels(self):
        for bad in ([1.0] * 5, [1.0] * 7):
            with self.subTest(length=len(bad)):
                with self.assertRaisesRegex(ValueError, "expected 6 channels"):
                    decode_long_risk_predictions(bad)

    def test_head_and_metadata_share_the_canonical_contract(self):
        head = LongRiskHead(latent_dim=8)
        output = head(torch.zeros(4, 8), torch.zeros(4, 8))
        contract = long_risk_runtime_contract()

        self.assertEqual(output.shape, (LONG_RISK_OUTPUT_DIM,))
        self.assertEqual(head.LONG_RISK_DIM, LONG_RISK_OUTPUT_DIM)
        self.assertEqual(contract["schema_version"], LONG_RISK_SCHEMA_VERSION)
        self.assertEqual(tuple(contract["output_order"]), LONG_RISK_OUTPUT_NAMES)


if __name__ == "__main__":
    unittest.main()
