import json
import tempfile
import unittest
from pathlib import Path

from WorldModel.evaluation.pc_endpoint_recovery_analysis import (
    _headline,
)
from WorldModel.evaluation.pc_endpoint_recovery import (
    _parse_run_id,
    _select_sources,
    _write_recovery_collection,
)


class StationCongestionEndpointRecoveryTests(unittest.TestCase):
    def test_parse_run_id(self):
        self.assertEqual(
            _parse_run_id("phasec_s1_high_seed530"), ("high", 530)
        )
        with self.assertRaises(ValueError):
            _parse_run_id("phasec_s1_high_seed500")

    def test_selection_prefers_finalized_and_excludes_zero_staging(self):
        finalized = {
            "phasec_s1_low_seed521": {
                "run_id": "phasec_s1_low_seed521",
                "load": "low",
                "seed": 521,
                "snapshot_count": 8,
            }
        }
        staging = {
            "phasec_s1_low_seed521": [{
                "run_id": "phasec_s1_low_seed521",
                "load": "low",
                "seed": 521,
                "snapshot_count": 3,
                "snapshot_index": Path("ignored"),
            }],
            "phasec_s1_low_seed522": [{
                "run_id": "phasec_s1_low_seed522",
                "load": "low",
                "seed": 522,
                "snapshot_count": 0,
                "snapshot_index": Path(__file__),
                "source_dir": Path("zero"),
            }],
        }

        included, excluded = _select_sources(finalized, staging)

        self.assertEqual(len(included), 1)
        self.assertEqual(included[0]["snapshot_count"], 8)
        excluded_by_id = {row["run_id"]: row for row in excluded}
        self.assertEqual(
            excluded_by_id["phasec_s1_low_seed522"]["reason"],
            "zero_snapshots",
        )

    def test_recovery_collection_is_explicitly_not_formal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            snapshot_dir = source_dir / "snapshots"
            snapshot_dir.mkdir(parents=True)
            index = snapshot_dir / "snapindex_run.json"
            index.write_text("{}\n", encoding="utf-8")
            first = snapshot_dir / "first.pkl"
            second = snapshot_dir / "second.pkl"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            order_manifest = source_dir / "order_manifest.json"
            order_manifest.write_text("{}\n", encoding="utf-8")
            source = {
                "run_id": "phasec_s1_mid_seed521",
                "load": "mid",
                "seed": 521,
                "source_kind": "postrun_underfilled_staging",
                "source_dir": source_dir,
                "source_summary": None,
                "snapshot_dir": snapshot_dir,
                "snapshot_index": index,
                "snapshot_files": [first, second],
                "snapshot_count": 2,
                "order_manifest": order_manifest,
            }
            output = _write_recovery_collection(
                root / "recovery",
                source,
                protocol_sha256="abc",
                copy_mode="copy",
            )
            summary_path = (
                root / "recovery/collections/phasec_s1_mid_seed521/"
                "collection_summary.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(output["snapshots"], 2)
            self.assertTrue(summary["exploratory_underfilled"])
            self.assertFalse(summary["formal_protocol_passed"])
            self.assertEqual(
                summary["audit"]["scope"],
                "exploratory_recovery_integrity_only",
            )

    def test_headline_marks_run_macro_as_primary(self):
        metrics = {
            "per_run_spearman_mean": 0.7,
            "per_run_cluster_ci95": [0.5, 0.9],
            "run_sign_consistency": 0.8,
            "pooled_spearman": 0.9,
            "mae": 0.1,
            "rmse": 0.2,
            "bias": 0.0,
            "candidate_ranking_at_context_station": {},
            "sign": None,
        }
        base = {
            "layers": {
                "A": {
                    subset: {channel: dict(metrics) for channel in (
                        "traffic", "service"
                    )}
                    for subset in ("all_stations", "context_station")
                }
            }
        }

        headline = _headline(base)

        self.assertEqual(
            headline["A"]["all_stations"]["traffic"]
            ["primary_per_run_spearman_mean"],
            0.7,
        )
        self.assertEqual(
            headline["A"]["all_stations"]["traffic"]
            ["secondary_pooled_spearman"],
            0.9,
        )


if __name__ == "__main__":
    unittest.main()
