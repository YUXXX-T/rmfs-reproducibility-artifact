"""Development protocol for contemporaneous station/system congestion audit.

This branch is intentionally isolated from the frozen 491--510 comparison.
It reuses only the already-frozen policy/config/checkpoint contract, creates
fresh Greedy order manifests for seeds 511--520, and never reads 501--510
outcomes while selecting congestion components.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = "phase_c_station_congestion_correlation_protocol_v1"
FROZEN_BUNDLE_SCHEMA_VERSION = (
    "phase_c_station_congestion_correlation_bundle_v1"
)
RUN_SCHEMA_VERSION = "phase_c_station_congestion_correlation_run_v1"
TRACE_SCHEMA_VERSION = "phase_c_station_congestion_trace_v1"
REPORT_SCHEMA_VERSION = "phase_c_station_congestion_correlation_report_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
SOURCE_ROOT = Path(os.environ.get(
    "PHASEC_STATION_CORR_SOURCE_ROOT",
    str(BASE_ROOT / "phasec_s1_hungarian_cert_491_500_v1"),
))
SOURCE_BUNDLE = Path(os.environ.get(
    "PHASEC_STATION_CORR_SOURCE_BUNDLE",
    str(SOURCE_ROOT / "phase_c_s1_hungarian_frozen_protocol.json"),
))
OUTPUT_ROOT = Path(os.environ.get(
    "PHASEC_STATION_CORR_OUT",
    str(BASE_ROOT / "station_congestion_correlation_dev_511_520_v1"),
))

LOADS = ("low", "mid", "high")
SEEDS = tuple(range(511, 521))
LOCKED_REFERENCE_SEEDS = tuple(range(501, 511))
ARMS = ("greedy", "hungarian", "phasec", "phasec_s1")
TICKS = 1500
FRAME_STRIDE = 5
REGION_HOPS = (1, 3, 5)
PRIMARY_REGION_HOPS = 3
TRAILING_WINDOW = 20
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20260803

SOURCE_FILES = (
    "WorldModel/evaluation/phase_c_station_congestion_correlation_protocol.py",
    "WorldModel/evaluation/station_congestion_trace_probe.py",
    "WorldModel/evaluation/freeze_phase_c_station_congestion_correlation.py",
    "WorldModel/evaluation/run_phase_c_station_congestion_correlation_arm.py",
    "WorldModel/evaluation/analyze_phase_c_station_congestion_correlations.py",
    "WorldModel/evaluation/run_phase_c_station_congestion_correlation_cpu.slurm",
    "WorldModel/evaluation/run_phase_c_station_congestion_correlation_ubuntu.sh",
    "WorldModel/tests/test_phase_c_station_congestion_correlation.py",
    "WorldModel/evaluation/run_phase_c_s1_hungarian_arm.py",
    "WorldModel/evaluation/phase_c_s1_hungarian_protocol.py",
    "WorldModel/evaluation/evaluate_online_v6.py",
    "WorldModel/evaluation/td_stream_probe.py",
    "WorldModel/graph/graph_builder.py",
    "WorldModel/core/model.py",
    "Policies/TaskAssigner/WorldModelTaskAssigner/world_model_task_assigner.py",
    "Policies/TaskAssigner/HungarianTaskAssigner/hungarian_task_assigner.py",
    "Policies/TaskAssigner/GreedyTaskAssigner/greedy_task_assigner.py",
    "Policies/TaskAssigner/context_assignment.py",
    "Engine/simulation_engine.py",
)


def canonical_sha256(payload: Mapping) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_id(arm: str, load: str, seed: int) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if load not in LOADS:
        raise ValueError(f"unknown load: {load}")
    if int(seed) not in SEEDS:
        raise ValueError(f"seed is outside frozen development block: {seed}")
    return f"{arm}_{load}_seed{int(seed)}"


def protocol_payload(source_bundle_sha256: str) -> dict:
    if set(SEEDS) & set(LOCKED_REFERENCE_SEEDS):
        raise RuntimeError("development seeds overlap locked 501--510 reference")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "rank contemporaneous physical indicators of station-local and "
            "system congestion before training a station-conditioned phi head"
        ),
        "source_policy_contract": {
            "root": SOURCE_ROOT.as_posix(),
            "bundle": SOURCE_BUNDLE.as_posix(),
            "bundle_sha256": str(source_bundle_sha256),
            "reuse": "checkpoint/config/policy semantics only",
            "reuse_source_manifests_or_results": False,
        },
        "development": {
            "loads": list(LOADS),
            "seeds": list(SEEDS),
            "ticks": TICKS,
            "arms": list(ARMS),
            "manifest_source": "greedy",
            "exact_manifest_replay": True,
            "frame_stride": FRAME_STRIDE,
        },
        "station_regions": {
            "graph_hops": list(REGION_HOPS),
            "primary_graph_hops": PRIMARY_REGION_HOPS,
            "seeds": "station service node plus entry and exit nodes",
            "overlap_allowed": True,
        },
        "time_semantics": {
            "phi_supervision": "same-state same-tick only",
            "future_ticks_in_state_target": False,
            "past_only_trailing_window": TRAILING_WINDOW,
            "saved_frames_align_with_analysis_ticks": True,
        },
        "analysis": {
            "primary_correlation": "Spearman",
            "partial_controls": [
                "load",
                "arm",
                "tick_fraction",
                "global_open_order_count",
                "global_active_robot_ratio",
            ],
            "station_fixed_effect_control": True,
            "within_tick_station_ranking": True,
            "cluster_unit": "run_id",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "report_feature_redundancy": True,
            "no_automatic_feature_selection": True,
        },
        "locked_reference": {
            "seeds": list(LOCKED_REFERENCE_SEEDS),
            "read_outcomes_for_feature_selection": False,
            "modify_existing_artifacts": False,
            "final_one_shot_comparison_only_after_freeze": True,
        },
        "forbidden": {
            "policy_change": True,
            "assigner_default_change": True,
            "future_label_as_phi_target": True,
            "global_zero_no_assign_as_defer_context": True,
            "501_510_feature_or_scale_selection": True,
            "overwrite_existing_phase_c_outputs": True,
        },
    }
    return {**payload, "protocol_sha256": canonical_sha256(payload)}
