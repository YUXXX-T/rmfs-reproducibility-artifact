import unittest

import numpy as np
import torch

from WorldModel.evaluation.analyze_phase_c_station_congestion_correlations import (
    _auc,
    _flatten_station,
    _partial_spearman,
    _rankdata,
    _spearman,
)
from WorldModel.evaluation.phase_c_station_congestion_correlation_protocol import (
    LOCKED_REFERENCE_SEEDS,
    SEEDS,
    canonical_sha256,
    protocol_payload,
    run_id,
)
from WorldModel.evaluation.station_congestion_trace_probe import (
    StationCongestionTraceProbe,
    _cvar,
    _nodes_within_hops,
)


class PhaseCStationCongestionCorrelationTests(unittest.TestCase):
    def test_protocol_hash_and_seed_isolation(self):
        payload = protocol_payload("a" * 64)
        claimed = payload.pop("protocol_sha256")
        self.assertEqual(canonical_sha256(payload), claimed)
        self.assertFalse(set(SEEDS) & set(LOCKED_REFERENCE_SEEDS))
        self.assertEqual(run_id("phasec_s1", "high", 511), "phasec_s1_high_seed511")
        with self.assertRaisesRegex(ValueError, "outside frozen"):
            run_id("phasec_s1", "high", 501)

    def test_graph_hop_region_is_deterministic(self):
        adjacency = {
            0: [1],
            1: [0, 2, 3],
            2: [1, 4],
            3: [1],
            4: [2],
        }
        self.assertEqual(_nodes_within_hops([0], adjacency, 0), frozenset({0}))
        self.assertEqual(_nodes_within_hops([0], adjacency, 1), frozenset({0, 1}))
        self.assertEqual(
            _nodes_within_hops([0], adjacency, 2),
            frozenset({0, 1, 2, 3}),
        )

    def test_cvar_uses_upper_tail(self):
        self.assertAlmostEqual(_cvar([0.0, 1.0, 2.0, 10.0], 0.25), 10.0)
        self.assertAlmostEqual(_cvar([0.0, 1.0, 2.0, 10.0], 0.50), 6.0)
        self.assertEqual(_cvar([]), 0.0)

    def test_node_channel_summary_preserves_local_extremes(self):
        values = torch.zeros(3, 10)
        values[:, 1] = torch.tensor([0.1, 0.5, 0.9])
        values[:, 8] = torch.tensor([0.0, 0.5, 1.0])
        summary = StationCongestionTraceProbe._node_channel_summary(
            values, [0, 1, 2]
        )
        self.assertAlmostEqual(summary["node_density_mean"], 0.5, places=6)
        self.assertAlmostEqual(summary["node_density_max"], 0.9, places=6)
        self.assertGreater(summary["node_bottleneck_weighted_density"], 0.5)

    def test_rank_spearman_and_auc(self):
        values = np.asarray([5.0, 1.0, 1.0, 9.0])
        ranks = _rankdata(values)
        self.assertEqual(ranks.tolist(), [3.0, 1.5, 1.5, 4.0])
        self.assertAlmostEqual(
            _spearman(
                np.asarray([1.0, 2.0, 3.0]),
                np.asarray([3.0, 2.0, 1.0]),
            ),
            -1.0,
        )
        self.assertAlmostEqual(
            _auc(
                np.asarray([0.1, 0.2, 0.8, 0.9]),
                np.asarray([False, False, True, True]),
            ),
            1.0,
        )

    def test_partial_spearman_removes_tick_confound(self):
        rows = []
        for tick in range(30):
            rows.append({
                "feature": float(tick),
                "target": float(tick),
                "tick_fraction": tick / 29.0,
                "global_open_order_count": float(tick),
                "global_active_robot_ratio": 0.5,
                "load": "low",
                "arm": "phasec",
                "station_id": 0,
            })
        value = _partial_spearman(
            rows, "feature", "target", station_scope=True
        )
        self.assertTrue(np.isnan(value) or abs(value) < 1e-6)

    def test_flatten_station_keeps_same_tick_controls_and_regions(self):
        meta = {"run_id": "r", "arm": "phasec", "load": "mid", "seed": 511}
        system = {"open_order_count": 12, "active_robot_ratio": 0.75}
        station = {
            "station_id": 2,
            "station_seed_nodes": [1, 2],
            "queue_occupancy_ratio": 0.5,
            "regions": {"h3": {"mobility_impairment_ratio": 0.25}},
        }
        row = _flatten_station(meta, 10, system, station)
        self.assertEqual(row["station_id"], 2)
        self.assertEqual(row["global_open_order_count"], 12.0)
        self.assertEqual(row["h3_mobility_impairment_ratio"], 0.25)


if __name__ == "__main__":
    unittest.main()

