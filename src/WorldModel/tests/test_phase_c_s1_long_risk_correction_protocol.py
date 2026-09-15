import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from WorldModel.core.long_risk_schema import (
    LONG_RISK_SCHEMA_VERSION,
    long_risk_runtime_contract,
)
from WorldModel.evaluation.phase_c_s1_long_risk_correction_protocol import (
    ARM_KEYS,
    LOADS,
    SEEDS,
    S1_SIGNAL_CONFIGS,
    TICKS,
    formal_protocol,
)
from WorldModel.evaluation.run_phase_c_s1_long_risk_correction import (
    _audit_metrics,
    _make_assigner,
)


class PhaseCS1LongRiskCorrectionProtocolTest(unittest.TestCase):
    def test_formal_scope_is_fresh_and_complete(self):
        self.assertEqual(LOADS, ("low", "mid", "high"))
        self.assertEqual(SEEDS, tuple(range(631, 641)))
        self.assertEqual(TICKS, 1500)
        self.assertEqual(
            ARM_KEYS,
            (
                "greedy",
                "hungarian",
                "phasec",
                "s1_event",
                "s1_terminal",
                "s1_combo",
            ),
        )

    def test_signal_arms_differ_only_by_explicit_signal(self):
        common = None
        for signal in ("event_logit", "terminal", "combo"):
            config = dict(S1_SIGNAL_CONFIGS[signal])
            self.assertEqual(config.pop("energy_drift_signal"), signal)
            if common is None:
                common = config
            else:
                self.assertEqual(config, common)

    def test_assigners_expose_expected_runtime_contract(self):
        expected = {
            "s1_event": "event_logit",
            "s1_terminal": "terminal",
            "s1_combo": "combo",
        }
        for arm, signal in expected.items():
            with self.subTest(arm=arm):
                assigner = _make_assigner(arm)
                self.assertEqual(assigner.energy_scoring_mode, "conversion")
                self.assertEqual(assigner.energy_drift_signal, signal)
                self.assertEqual(
                    assigner.long_risk_schema_version,
                    LONG_RISK_SCHEMA_VERSION,
                )

    def test_protocol_embeds_canonical_combo_contract(self):
        protocol = formal_protocol()
        embedded = protocol["world_model"]["long_risk_runtime_contract"]
        self.assertEqual(embedded, long_risk_runtime_contract())
        self.assertEqual(
            embedded["quantile_combo_weights"],
            {"peak_q95": 0.2, "cvar_q90": 0.3, "terminal_q90": 0.5},
        )
        self.assertTrue(protocol["formal_test"]["fresh_seed_range"])
        self.assertFalse(
            protocol["interpretation"]["station_admission_modification"]
        )

    def test_per_arm_audit_rejects_signal_mismatch(self):
        manifest = {"manifest_sha256": "abc", "total_orders": 123}
        metrics = {
            "order_arrival_manifest_sha256": "abc",
            "order_arrival_count": 123,
            "order_arrival_replayed": True,
            "model_assign_calls": 10,
            "fallback_greedy_calls": 0,
            "native_no_assign_enabled": False,
            "long_risk_schema_version": LONG_RISK_SCHEMA_VERSION,
            "long_risk_runtime_contract": long_risk_runtime_contract(),
            "energy_conv_contexts": 10,
            "energy_scoring_mode": "conversion",
            "energy_drift_signal": "combo",
        }
        with TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "orders.json"
            manifest_path.write_text("{}", encoding="utf-8")
            passed = _audit_metrics(
                "s1_combo", metrics, manifest_path, manifest
            )
            self.assertTrue(passed["passed"])

            wrong = dict(metrics)
            wrong["energy_drift_signal"] = "event_logit"
            rejected = _audit_metrics(
                "s1_combo", wrong, manifest_path, manifest
            )
            self.assertFalse(rejected["passed"])
            self.assertFalse(
                rejected["checks"]["s1_signal_matches_arm"]
            )


if __name__ == "__main__":
    unittest.main()
