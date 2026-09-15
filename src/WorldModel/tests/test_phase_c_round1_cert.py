import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from WorldModel.evaluation.phase_c_round1_cert_protocol import (
    CERTIFICATION_SEEDS,
    DECISION_TRACE_SCHEMA_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_FORMAL_PROTOCOL_SHA256,
    LOAD_CONFIGS,
    ONLINE_REPORT_SCHEMA_VERSION,
    PER_SEED_REPORT_SCHEMA_VERSION,
    TRAJECTORY_BIN_EDGES,
    formal_protocol,
    sha256_file,
    validate_exact_seed_set,
)
from WorldModel.evaluation.run_phase_c_round1_online_pair import (
    BASELINE_LABEL,
    CANDIDATE_LABEL,
    _compact_candidate_trace,
    _pair_audit,
    _resume_seed_payload,
)
from WorldModel.evaluation.validate_phase_c_round1_cert import (
    _audit_artifacts,
    _bin_index,
    _canonical_manifest_sha256,
    _load_online_rows,
    _metric_report,
)
from WorldModel.evaluation.evaluate_online_v6 import (
    ORDER_ARRIVAL_MANIFEST_SCHEMA_VERSION,
)


class PhaseCRound1FormalProtocolTests(unittest.TestCase):
    def test_protocol_is_hashed_and_reserves_exact_online_matrix(self):
        protocol = formal_protocol()
        self.assertEqual(
            protocol["protocol_sha256"], EXPECTED_FORMAL_PROTOCOL_SHA256
        )
        self.assertEqual(
            tuple(protocol["formal_test"]["seeds"]), CERTIFICATION_SEEDS
        )
        self.assertEqual(protocol["formal_test"]["ticks"], 1500)
        self.assertEqual(
            tuple(protocol["formal_test"]["trajectory_bin_edges"]),
            TRAJECTORY_BIN_EDGES,
        )
        self.assertEqual(
            protocol["acceptance"]["primary"]["metric"],
            "completion_fraction",
        )
        self.assertTrue(protocol["forbidden"]["post_test_checkpoint_selection"])

    def test_exact_certification_seed_set(self):
        validate_exact_seed_set(list(reversed(CERTIFICATION_SEEDS)))
        with self.assertRaises(ValueError):
            validate_exact_seed_set(CERTIFICATION_SEEDS[:-1])


class PhaseCRound1FormalStatisticsTests(unittest.TestCase):
    def test_primary_bootstrap_clusters_three_loads_by_seed(self):
        rows = []
        for seed in (471, 472):
            for load in ("low", "mid", "high"):
                rows.append({
                    "seed": seed,
                    "load": load,
                    "baseline": {
                        "order_arrival_count": 100,
                        "completed_orders": 20,
                    },
                    "candidate": {
                        "order_arrival_count": 100,
                        "completed_orders": 30,
                    },
                })
        report = _metric_report(
            rows,
            "completion_fraction",
            repeats=100,
            seed=7,
        )
        self.assertEqual(report["overall"]["n"], 2)
        self.assertAlmostEqual(report["overall"]["mean"], 0.1)
        self.assertTrue(report["favourable_point_estimate"])

    def test_relative_delay_uses_frozen_baseline_scale(self):
        rows = [{
            "seed": 471,
            "load": "low",
            "baseline": {
                "order_arrival_count": 20,
                "avg_excess_delay": 10.0,
            },
            "candidate": {
                "order_arrival_count": 20,
                "avg_excess_delay": 10.5,
            },
        }]
        report = _metric_report(
            rows,
            "avg_excess_delay_relative",
            repeats=10,
            seed=1,
        )
        self.assertAlmostEqual(report["overall"]["mean"], 0.05)
        self.assertFalse(report["favourable_point_estimate"])


class PhaseCRound1FormalTraceTests(unittest.TestCase):
    def test_compact_trace_matches_native_no_assign_counters(self):
        assigner = SimpleNamespace(decision_trace_records=[
            {
                "tick": 10,
                "context_idx": 0,
                "selected_action_type": "assign_robot",
                "action_status": "selected_before_guard",
            },
            {
                "tick": 1200,
                "context_idx": 1,
                "selected_action_type": "no_assign",
                "action_status": "native_no_assign_selected",
            },
        ])
        metrics = {
            "decision_trace_contexts": 2,
            "decision_contexts_total": 2,
            "native_no_assign_contexts": 2,
            "native_no_assign_scored": 2,
            "native_no_assign_selected": 1,
            "native_no_assign_assignment_selected": 1,
            "decision_trace_dropped": 0,
        }
        rows, summary = _compact_candidate_trace(
            assigner, metrics, load="low", seed=471
        )
        self.assertTrue(summary["passed"])
        self.assertEqual(rows[1]["selected_action_type"], "no_assign")
        self.assertEqual(_bin_index(0), 0)
        self.assertEqual(_bin_index(1499), 3)
        self.assertIsNone(_bin_index(1500))

    def test_per_seed_resume_is_hash_bound_to_manifest_and_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "orders.json"
            trace = root / "trace.jsonl"
            seed_report = root / "seed.json"
            manifest.write_text("{}\n", encoding="utf-8")
            trace.write_text("{}\n", encoding="utf-8")
            payload = {
                "schema_version": PER_SEED_REPORT_SCHEMA_VERSION,
                "meta": {
                    "formal": True,
                    "load": "low",
                    "seed": 471,
                    "ticks": 1500,
                    "formal_protocol_sha256": "protocol",
                    "frozen_bundle_sha256": "bundle",
                },
                "arms": {
                    BASELINE_LABEL: {},
                    CANDIDATE_LABEL: {},
                },
                "pair_audit": {"passed": True},
                "comparison": {},
                "artifacts": {
                    "order_manifest": {"sha256": sha256_file(manifest)},
                    "candidate_decision_trace": {
                        "sha256": sha256_file(trace),
                        "summary": {"passed": True},
                    },
                },
            }
            seed_report.write_text(json.dumps(payload), encoding="utf-8")
            resumed = _resume_seed_payload(
                seed_report,
                formal=True,
                load="low",
                seed=471,
                ticks=1500,
                protocol_sha256="protocol",
                bundle_sha256="bundle",
                manifest_path=manifest,
                trace_path=trace,
            )
            self.assertEqual(resumed["meta"]["seed"], 471)
            trace.write_text("changed\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _resume_seed_payload(
                    seed_report,
                    formal=True,
                    load="low",
                    seed=471,
                    ticks=1500,
                    protocol_sha256="protocol",
                    bundle_sha256="bundle",
                    manifest_path=manifest,
                    trace_path=trace,
                )

    def test_complete_synthetic_matrix_passes_integrity_audit(self):
        protocol = formal_protocol()
        bundle_sha = "frozen-bundle-sha"
        config_artifacts = {
            load: {"sha256": f"config-{load}"}
            for load in ("low", "mid", "high")
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir = root / "order_manifests"
            trace_dir = root / "decision_traces"
            per_seed_dir = root / "per_seed"
            for path in (manifest_dir, trace_dir, per_seed_dir):
                path.mkdir()
            online_paths = []
            for load in ("low", "mid", "high"):
                per_seed = {}
                pair_audits = {}
                artifacts_by_seed = {}
                for seed in CERTIFICATION_SEEDS:
                    manifest_path = (
                        manifest_dir
                        / f"phasec_r1_{load}_seed{seed}_orders.json"
                    )
                    manifest = {
                        "schema_version": (
                            ORDER_ARRIVAL_MANIFEST_SCHEMA_VERSION
                        ),
                        "orders": [{
                            "tick": 0,
                            "order_id": 0,
                            "station_id": 0,
                            "sku_demands": {"A": 1},
                        }],
                    }
                    manifest["total_orders"] = 1
                    manifest["manifest_sha256"] = (
                        _canonical_manifest_sha256(manifest)
                    )
                    manifest_path.write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                    common = {
                        "online_robot_candidate_scope": "all_idle",
                        "fallback_greedy_calls": 0,
                        "fallback_greedy_ratio": 0.0,
                        "order_arrival_manifest_schema_version": (
                            ORDER_ARRIVAL_MANIFEST_SCHEMA_VERSION
                        ),
                        "order_arrival_manifest_sha256": manifest[
                            "manifest_sha256"
                        ],
                        "order_arrival_count": 1,
                    }
                    baseline = {
                        **common,
                        "order_arrival_replayed": False,
                    }
                    candidate = {
                        **common,
                        "order_arrival_replayed": True,
                        "native_no_assign_enabled": True,
                        "native_no_assign_contexts": 2,
                        "native_no_assign_scored": 2,
                        "native_no_assign_selected": 1,
                        "native_no_assign_assignment_selected": 1,
                        "decision_trace_contexts": 2,
                        "decision_contexts_total": 2,
                        "decision_trace_dropped": 0,
                    }
                    trace_path = (
                        trace_dir
                        / f"phasec_r1_{load}_seed{seed}_candidate_decisions.jsonl"
                    )
                    trace_rows = [
                        {
                            "schema_version": DECISION_TRACE_SCHEMA_VERSION,
                            "load": load,
                            "seed": seed,
                            "record_index": 0,
                            "tick": 100,
                            "context_idx": 0,
                            "selected_action_type": "no_assign",
                            "action_status": "native_no_assign_selected",
                        },
                        {
                            "schema_version": DECISION_TRACE_SCHEMA_VERSION,
                            "load": load,
                            "seed": seed,
                            "record_index": 1,
                            "tick": 1200,
                            "context_idx": 1,
                            "selected_action_type": "assign_robot",
                            "action_status": "selected_before_guard",
                        },
                    ]
                    trace_path.write_text(
                        "".join(json.dumps(row) + "\n" for row in trace_rows),
                        encoding="utf-8",
                    )
                    artifacts = {
                        "order_manifest": {
                            "sha256": sha256_file(manifest_path),
                        },
                        "candidate_decision_trace": {
                            "sha256": sha256_file(trace_path),
                            "summary": {"passed": True},
                        },
                    }
                    arms = {
                        BASELINE_LABEL: baseline,
                        CANDIDATE_LABEL: candidate,
                    }
                    pair = _pair_audit(baseline, candidate)
                    seed_path = (
                        per_seed_dir
                        / f"phasec_r1_{load}_seed{seed}_paired.json"
                    )
                    seed_payload = {
                        "schema_version": PER_SEED_REPORT_SCHEMA_VERSION,
                        "meta": {
                            "formal": True,
                            "load": load,
                            "seed": seed,
                            "ticks": 1500,
                            "formal_protocol_sha256": protocol[
                                "protocol_sha256"
                            ],
                            "frozen_bundle_sha256": bundle_sha,
                        },
                        "arms": arms,
                        "pair_audit": pair,
                        "comparison": {},
                        "artifacts": artifacts,
                    }
                    seed_path.write_text(
                        json.dumps(seed_payload), encoding="utf-8"
                    )
                    per_seed[str(seed)] = arms
                    pair_audits[str(seed)] = pair
                    artifacts_by_seed[str(seed)] = {
                        **artifacts,
                        "per_seed_report": {
                            "sha256": sha256_file(seed_path),
                        },
                    }
                report = {
                    "schema_version": ONLINE_REPORT_SCHEMA_VERSION,
                    "meta": {
                        "formal": True,
                        "role": (
                            "FORMAL_PHASE_C_ROUND1_PAIRED_CLOSED_LOOP_COLLECTION"
                        ),
                        "formal_protocol_sha256": protocol[
                            "protocol_sha256"
                        ],
                        "frozen_bundle_sha256": bundle_sha,
                        "load": load,
                        "ticks": 1500,
                        "top_m": 10,
                        "config": Path(LOAD_CONFIGS[load]).as_posix(),
                        "config_sha256": f"config-{load}",
                        "seeds": list(CERTIFICATION_SEEDS),
                        "paired_arms": [BASELINE_LABEL, CANDIDATE_LABEL],
                        "baseline_checkpoint": {
                            "sha256": EXPECTED_BASELINE_SHA256
                        },
                        "candidate_checkpoint": {
                            "sha256": EXPECTED_CANDIDATE_SHA256
                        },
                        "baseline_native_horizon": 3,
                        "candidate_native_horizon": 10,
                        "external_assignment_baseline": False,
                        "td_target_or_head": False,
                        "work_drift_or_residual_head": False,
                        "lyapunov_online_scoring": False,
                        "hard_no_assign_gate": False,
                    },
                    "per_seed": per_seed,
                    "pair_audits": pair_audits,
                    "per_seed_artifacts": artifacts_by_seed,
                }
                online_path = root / f"online_{load}.json"
                online_path.write_text(json.dumps(report), encoding="utf-8")
                online_paths.append(online_path)

            rows, implementation = _load_online_rows(
                online_paths,
                protocol=protocol,
                bundle_sha256=bundle_sha,
                config_artifacts=config_artifacts,
            )
            self.assertTrue(implementation["passed"])
            self.assertEqual(len(rows), 30)
            artifact_audit = _audit_artifacts(
                rows,
                manifest_dir=manifest_dir,
                trace_dir=trace_dir,
                per_seed_dir=per_seed_dir,
                protocol_sha256=protocol["protocol_sha256"],
                bundle_sha256=bundle_sha,
            )
            self.assertTrue(artifact_audit["integrity_passed"])
            self.assertTrue(artifact_audit["terminal_assignments_passed"])


if __name__ == "__main__":
    unittest.main()
