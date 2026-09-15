from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from WorldModel.evaluation.analyze_phase_c_station_context_defer_pipeline_phi import (
    analyse,
)
from WorldModel.evaluation.run_phase_c_station_context_defer_pipeline_phi import (
    ARM_KEY,
    BASELINE_ARM_KEY,
    READY_MAX_V1_ARM_KEY,
    SCHEMA_VERSION,
)


class StationContextDeferPipelinePhiAnalysisTests(unittest.TestCase):
    def test_partial_report_keeps_primary_and_mechanism_contrasts_separate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "per_arm" / ARM_KEY / "low_seed551.json"
            output.parent.mkdir(parents=True)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "meta": {"load": "low", "seed": 551, "ticks": 1500},
                "reference": {
                    "arm": BASELINE_ARM_KEY,
                    "metrics": {
                        "completed_orders": 100,
                        "completed_tasks": 200,
                        "deadlock_ratio_mean": 0.10,
                        "deadlock_ratio_max": 0.20,
                        "stall_ratio_mean": 0.30,
                        "stall_ratio_max": 0.40,
                        "pending_order_count": 10,
                        "open_order_count": 20,
                    },
                },
                "ready_max_v1_reference": {
                    "arm": READY_MAX_V1_ARM_KEY,
                    "metrics": {
                        "completed_orders": 90,
                        "completed_tasks": 210,
                        "deadlock_ratio_mean": 0.05,
                        "deadlock_ratio_max": 0.10,
                        "stall_ratio_mean": 0.20,
                        "stall_ratio_max": 0.30,
                        "pending_order_count": 20,
                        "open_order_count": 30,
                    },
                    "defer": {
                        "rate": 0.80,
                        "batches": 50,
                        "all_remaining_batches": 30,
                    },
                },
                "audit": {
                    "passed": True,
                    "checks": {
                        "pipeline_phi_risk_mode": True,
                        "ready_contention_is_diagnostic_only": True,
                    },
                },
                "station_admission_audit": {
                    "passed": True,
                    "capacity_violation_count": 0,
                    "token_mismatch_count": 0,
                },
                "metrics": {
                    "completed_orders": 98,
                    "completed_tasks": 220,
                    "deadlock_ratio_mean": 0.07,
                    "deadlock_ratio_max": 0.15,
                    "stall_ratio_mean": 0.25,
                    "stall_ratio_max": 0.35,
                    "pending_order_count": 12,
                    "open_order_count": 22,
                    "station_context_defer_evaluations": 100,
                    "station_context_defer_decisions": 20,
                    "station_context_defer_rate": 0.20,
                    "station_context_defer_batches": 50,
                    "station_context_defer_all_remaining_batches": 5,
                    "station_context_defer_ready_would_dominate_evaluations": 70,
                    "station_context_defer_risk_mean": 0.3,
                    "station_context_defer_pipeline_excess_mean": 0.2,
                    "station_context_defer_ready_contention_mean": 0.8,
                    "station_context_defer_phi_pressure_mean": 0.1,
                    "station_context_defer_eligible_ticks_max": 10,
                    "station_context_defer_liveness_bound_violations": 0,
                },
            }
            output.write_text(json.dumps(payload), encoding="utf-8")

            report = analyse(root, allow_partial=True)
            primary = report["contrasts"]["vs_dynamic_j_admission_v1"]
            mechanism = report["contrasts"]["vs_ready_max_defer_v1"]
            self.assertEqual(
                primary["overall"]["metric_deltas"]
                ["completed_orders"]["mean"],
                -2.0,
            )
            self.assertEqual(
                mechanism["overall"]["metric_deltas"]
                ["completed_orders"]["mean"],
                8.0,
            )
            self.assertAlmostEqual(
                report["defer_mechanism"]
                ["ready_would_dominate_rate"]["mean"],
                0.70,
            )
            self.assertAlmostEqual(
                report["defer_mechanism"]
                ["paired_delta_vs_ready_max_v1"]
                ["defer_rate"]["mean"],
                -0.60,
            )
            self.assertTrue(
                report["summary"]["mechanism_and_lifecycle_passed"]
            )
            self.assertFalse(report["summary"]["complete"])


if __name__ == "__main__":
    unittest.main()
