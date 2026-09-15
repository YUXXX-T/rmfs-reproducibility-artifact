"""Unit tests for the isolated psi_pre v2 contract."""

from __future__ import annotations

import unittest

import torch

from WorldModel.data.build_phase_c_psi_pre_dataset import _baseline_channels
from WorldModel.evaluation.phase_c_psi_pre_v2_protocol import (
    PRIMARY_CHANNELS,
    SERVICE_CHANNELS,
    TARGET_CHANNELS,
    TARGET_SOURCE_INDICES,
    TRAFFIC_DIAGNOSTIC_CHANNELS,
    TRAFFIC_PRIMARY_CHANNELS,
    formal_protocol,
)


class PsiPreV2ContractTests(unittest.TestCase):
    def test_channel_roles_are_explicit_and_non_overlapping(self):
        self.assertEqual(len(TARGET_CHANNELS), 5)
        self.assertEqual(PRIMARY_CHANNELS, SERVICE_CHANNELS + TRAFFIC_PRIMARY_CHANNELS)
        self.assertEqual(TRAFFIC_DIAGNOSTIC_CHANNELS, ("region_blocked_max",))
        self.assertNotIn("region_blocked_max", PRIMARY_CHANNELS)
        self.assertEqual(TARGET_SOURCE_INDICES, (0, 1, 2, 3, 7))

    def test_bce_logits_are_sigmoided_before_region_aggregation(self):
        logits = torch.tensor([
            [0.0, 0.2, -10.0, 10.0, 0.4, 0.5],
            [0.0, 0.4, -10.0, 10.0, 0.6, 0.7],
        ])
        station = torch.zeros(1, 2)
        raw = _baseline_channels(logits, station, [[0, 1]])
        corrected = _baseline_channels(
            logits, station, [[0, 1]], apply_node_bce_sigmoid=True
        )
        # region_wait_mean is output column 4 (node channel 2), and
        # region_blocked_mean is output column 6 (node channel 3).
        self.assertLess(float(corrected[0, 4]), 0.01)
        self.assertGreater(float(corrected[0, 6]), 0.99)
        self.assertNotAlmostEqual(float(raw[0, 4]), float(corrected[0, 4]))
        # Continuous density/congestion channels are unchanged by the fix.
        self.assertAlmostEqual(float(raw[0, 2]), float(corrected[0, 2]), places=6)
        self.assertAlmostEqual(float(raw[0, 8]), float(corrected[0, 8]), places=6)

    def test_protocol_marks_v2_as_offline_only(self):
        protocol = formal_protocol()
        self.assertEqual(protocol["channel_contract"]["primary_channels"], list(PRIMARY_CHANNELS))
        self.assertTrue(protocol["channel_contract"]["blocked_max_role"].startswith("diagnostic"))
        self.assertFalse(protocol["diagnostic_bars"]["online_connection_allowed"])
        self.assertTrue(protocol["forbidden"]["seeds_501_510"])


if __name__ == "__main__":
    unittest.main()
