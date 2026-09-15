import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from WorldModel.evaluation import run_hungarian_comparison as module


class HungarianComparisonTests(unittest.TestCase):
    def test_paired_summary_uses_explicit_metric_direction(self):
        per_seed = {
            1: {
                module.BASELINE_LABEL: {
                    "completed_orders": 10,
                    "avg_excess_delay": 5.0,
                },
                module.HUNGARIAN_LABEL: {
                    "completed_orders": 12,
                    "avg_excess_delay": 4.0,
                },
            },
            2: {
                module.BASELINE_LABEL: {
                    "completed_orders": 10,
                    "avg_excess_delay": 5.0,
                },
                module.HUNGARIAN_LABEL: {
                    "completed_orders": 9,
                    "avg_excess_delay": 7.0,
                },
            },
        }

        summary = module._paired_summary(per_seed)

        self.assertEqual(
            summary["completed_orders"]["favourable_improvement_mean"],
            0.5,
        )
        self.assertEqual(
            summary["avg_excess_delay"]["favourable_improvement_mean"],
            -0.5,
        )

    def test_runner_reuses_wm_and_replays_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.json"
            manifests = root / "manifests"
            manifests.mkdir()
            output = root / "result.json"
            manifest = {
                "schema_version": "test_manifest_v1",
                "manifest_sha256": "same-stream",
                "total_orders": 3,
                "orders": [],
            }
            (manifests / "layer5_low_seed1_orders.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            baseline = {
                "meta": {
                    "load": "low",
                    "config": "Config/world_model_config_PP_48_low.json",
                    "seeds": [1],
                    "ticks": 20,
                },
                "per_seed": {
                    "1": {
                        module.BASELINE_LABEL: {
                            "completed_orders": 2,
                            "avg_excess_delay": 4.0,
                            "order_arrival_manifest_sha256": "same-stream",
                            "order_arrival_count": 3,
                        }
                    }
                },
            }
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            hungarian_metrics = {
                "completed_orders": 3,
                "avg_excess_delay": 3.0,
                "order_arrival_manifest_sha256": "same-stream",
                "order_arrival_count": 3,
            }
            argv = [
                "run_hungarian_comparison.py",
                "--baseline-results",
                str(baseline_path),
                "--manifests-dir",
                str(manifests),
                "--config",
                "Config/world_model_config_PP_48_low.json",
                "--load",
                "low",
                "--seeds",
                "1",
                "--ticks",
                "20",
                "--output",
                str(output),
            ]

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    module,
                    "_run_one_assigner",
                    return_value=hungarian_metrics,
                ) as run,
            ):
                module.main()

            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(result["meta"]["pure_world_model_reused"])
            self.assertFalse(result["meta"]["world_model_rerun"])
            self.assertTrue(
                result["meta"]["hungarian_is_independent_baseline"]
            )
            self.assertEqual(
                result["schema_version"],
                "worldmodel_hungarian_paired_comparison_v2",
            )
            self.assertEqual(
                result["protocol"]["task_chain_commit"],
                "shared_fixed_context_commit",
            )
            self.assertEqual(
                result["protocol"]["hungarian_implementation_version"],
                "manhattan_global_shared_fixed_context_commit_v2",
            )
            self.assertTrue(
                result["protocol"]["station_queue_contract"]
            )
            self.assertEqual(
                result["protocol"]["return_source"],
                "station_exit_or_service_fallback",
            )
            self.assertEqual(
                len(result["protocol"]["hungarian_source_sha256"]),
                64,
            )
            self.assertEqual(
                len(result["protocol"]["shared_commit_source_sha256"]),
                64,
            )
            self.assertFalse(
                result["protocol"]["world_model_loaded_by_hungarian_arm"]
            )
            self.assertFalse(
                result["protocol"]["hungarian_called_inside_world_model"]
            )
            self.assertTrue(
                result["manifest_audit"]["1"]["paired_hash_match"]
            )
            self.assertEqual(run.call_count, 1)
            call_args, call_kwargs = run.call_args
            self.assertIsInstance(call_args[1], module.HungarianTaskAssigner)
            self.assertEqual(call_args[2:4], (1, 20))
            self.assertEqual(
                Path(call_kwargs["recorded_orders_path"]),
                manifests / "layer5_low_seed1_orders.json",
            )
            self.assertEqual(
                result["paired_comparison"]["completed_orders"]
                ["favourable_improvement_mean"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
