"""Frozen protocol for the corrected Combo-S1 + static-J1 paper campaigns."""

from __future__ import annotations

from pathlib import Path

from WorldModel.core.long_risk_schema import long_risk_runtime_contract
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    PSI_HEAD_CHECKPOINT,
    PSI_SCALE_CONTRACT,
)
from WorldModel.evaluation.phase_c_s1_long_risk_correction_protocol import (
    CANDIDATE_CHECKPOINT,
    LOAD_CONFIGS,
    REPORT_METRICS,
    S1_SIGNAL_CONFIGS,
    TOP_M,
    canonical_sha256,
    sha256_file,
)
from WorldModel.evaluation.phase_c_s1_hungarian_protocol import PHASEC_CONFIG
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


SCHEMA_VERSION = "phase_c_combo_s1_j1_paper_protocol_v1"
BUNDLE_SCHEMA_VERSION = "phase_c_combo_s1_j1_paper_bundle_v1"
RUN_SCHEMA_VERSION = "phase_c_combo_s1_j1_paper_run_v1"
MAIN_SUMMARY_SCHEMA_VERSION = "phase_c_combo_s1_j1_main_summary_v1"
FRONTIER_SUMMARY_SCHEMA_VERSION = "phase_c_combo_s1_j1_frontier_summary_v1"
LEGACY_AUDIT_SCHEMA_VERSION = "phase_c_combo_s1_j1_legacy_source_audit_v1"
FRONTIER_AUDIT_SCHEMA_VERSION = "phase_c_combo_s1_j1_frontier_source_audit_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")

LEGACY_SEEDS = tuple(range(551, 561))
FRESH_SEEDS = tuple(range(641, 651))
LOADS = ("low", "mid", "high")
TICKS = 1500

LEGACY_HIGH_MANIFEST_ROOT = (
    BASE_ROOT / "psi_dispatch_dynamic_eta_overbooking_stress_source_551_560_v1"
)
LEGACY_LOW_MID_MANIFEST_ROOT = (
    BASE_ROOT / "station_admission_sj_factorial_low_mid_source_551_560_v1"
)
LEGACY_HIGH_REFERENCE_ROOT = (
    BASE_ROOT / "station_admission_sj_factorial_551_560_v1"
)
LEGACY_LOW_MID_REFERENCE_ROOT = (
    BASE_ROOT / "station_admission_sj_factorial_low_mid_551_560_v1"
)

MAIN_LEGACY_OUTPUT_ROOT = (
    BASE_ROOT / "phasec_combo_s1_j1_main_replacement_551_560_v1"
)
MAIN_FRESH_OUTPUT_ROOT = (
    BASE_ROOT / "phasec_combo_s1_j1_main_replacement_641_650_v1"
)

FRONTIER_SOURCE_ROOT = (
    BASE_ROOT / "station_admission_capacity_frontier_source_high_551_560_v1"
)
FRONTIER_OUTPUT_ROOT = (
    BASE_ROOT / "phasec_combo_s1_j1_paired_arrival_frontier_high_551_560_v1"
)
FRONTIER_SEEDS = LEGACY_SEEDS
FRONTIER_POINTS = (
    ("m080", 0.8),
    ("m100", 1.0),
    ("m120", 1.2),
    ("m140", 1.4),
    ("m160", 1.6),
)
FRONTIER_LOAD = "high"
FRONTIER_COLLAPSE_EFFICIENCY = 0.80
FRONTIER_SUSTAINABLE_MEAN_CLEARANCE = 0.90
FRONTIER_SUSTAINABLE_MAX_COLLAPSE_RATE = 0.20

RUN_ARM_LABELS = {
    "greedy": "GreedySequentialManhattan",
    "hungarian": "HungarianBatchManhattan",
    "phasec": "PhaseCRobotOnlyS0J0",
    "combo_j1": "CorrectedQuantileComboS1StaticJ1",
}
RUN_ARMS = tuple(RUN_ARM_LABELS)

LEGACY_REFERENCE_ARM_DIRS = {
    "greedy": "greedy_physical_only",
    "hungarian": "hungarian_physical_only",
    "phasec": "s0_j0_physical_only",
    "s0_j1": "s0_j1_physical_only",
    "legacy_event_j1": "s1_j1_physical_only",
}
LEGACY_REFERENCE_LABELS = {
    "greedy": RUN_ARM_LABELS["greedy"],
    "hungarian": RUN_ARM_LABELS["hungarian"],
    "phasec": RUN_ARM_LABELS["phasec"],
    "s0_j1": "PhaseCS0StaticJ1",
    "legacy_event_j1": "LegacyEventLogitS1StaticJ1",
}

MAIN_LEGACY_COMPARISONS = tuple(
    (f"combo_j1_minus_{baseline}", baseline, "combo_j1")
    for baseline in LEGACY_REFERENCE_ARM_DIRS
)
MAIN_FRESH_COMPARISONS = tuple(
    (f"combo_j1_minus_{baseline}", baseline, "combo_j1")
    for baseline in ("greedy", "hungarian", "phasec")
)
FRONTIER_COMPARISONS = MAIN_FRESH_COMPARISONS

DERIVED_REPORT_METRICS = tuple(REPORT_METRICS) + ("completion_ratio",)


def selector_config() -> dict:
    """Return a detached corrected quantile-combo S1 configuration."""

    return dict(S1_SIGNAL_CONFIGS["combo"])


def formal_protocol() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "replace the historical event-logit S1+J1 headline result with "
            "the corrected quantile-combo S1+J1 policy, then measure a truly "
            "paired offered-load frontier"
        ),
        "runtime_policy": {
            "final_method": RUN_ARM_LABELS["combo_j1"],
            "checkpoint": str(CANDIDATE_CHECKPOINT),
            "top_m": TOP_M,
            "s0_config": dict(PHASEC_CONFIG),
            "combo_s1_config": selector_config(),
            "psi_head_checkpoint": Path(PSI_HEAD_CHECKPOINT).as_posix(),
            "psi_scale_contract": Path(PSI_SCALE_CONTRACT).as_posix(),
            "context_scheduler": "static ascending J once per proposal batch",
            "long_risk_runtime_contract": long_risk_runtime_contract(),
        },
        "station_admission": {
            "mode": STATION_ADMISSION_PHYSICAL_ONLY,
            "physical_occupancy_cap_enforced": True,
            "in_transit_committed_cap_enforced": False,
            "station_queue": False,
            "eta_controller": False,
        },
        "main_replacement": {
            "preferred_seeds": list(LEGACY_SEEDS),
            "fallback_fresh_seeds": list(FRESH_SEEDS),
            "loads": list(LOADS),
            "ticks": TICKS,
            "preferred_behavior": (
                "reuse all exact historical manifests and unaffected baseline "
                "outputs; simulate only corrected combo_j1"
            ),
            "preferred_new_simulation_count": (
                len(LOADS) * len(LEGACY_SEEDS)
            ),
            "fallback_behavior": (
                "if any historical manifest is unavailable, use the complete "
                "fresh seed group and simulate greedy, hungarian, phasec, and "
                "corrected combo_j1 on newly recorded paired manifests"
            ),
            "fallback_new_simulation_count": (
                len(LOADS) * len(FRESH_SEEDS) * len(RUN_ARMS)
            ),
            "legacy_manifest_roots": {
                "high": LEGACY_HIGH_MANIFEST_ROOT.as_posix(),
                "low_mid": LEGACY_LOW_MID_MANIFEST_ROOT.as_posix(),
            },
            "legacy_reference_roots": {
                "high": LEGACY_HIGH_REFERENCE_ROOT.as_posix(),
                "low_mid": LEGACY_LOW_MID_REFERENCE_ROOT.as_posix(),
            },
            "legacy_reference_arms": dict(LEGACY_REFERENCE_ARM_DIRS),
            "legacy_comparisons": [
                {"name": name, "baseline": baseline, "candidate": candidate}
                for name, baseline, candidate in MAIN_LEGACY_COMPARISONS
            ],
            "fresh_comparisons": [
                {"name": name, "baseline": baseline, "candidate": candidate}
                for name, baseline, candidate in MAIN_FRESH_COMPARISONS
            ],
        },
        "paired_arrival_frontier": {
            "source_root": FRONTIER_SOURCE_ROOT.as_posix(),
            "load": FRONTIER_LOAD,
            "seeds": list(FRONTIER_SEEDS),
            "ticks": TICKS,
            "points": [
                {"tag": tag, "multiplier": multiplier}
                for tag, multiplier in FRONTIER_POINTS
            ],
            "arms": dict(RUN_ARM_LABELS),
            "pairing_contract": (
                "all points are nested prefixes of one long stream per seed; "
                "arrival tick is floor(base_tick / multiplier)"
            ),
            "new_simulation_count": (
                len(FRONTIER_POINTS) * len(FRONTIER_SEEDS) * len(RUN_ARMS)
            ),
            "collapse_efficiency_ratio": FRONTIER_COLLAPSE_EFFICIENCY,
            "collapse_reference": (
                "completed orders divided by the empirical best completed "
                "orders among all frozen arms for the same multiplier and seed"
            ),
            "sustainable_mean_clearance_ratio": (
                FRONTIER_SUSTAINABLE_MEAN_CLEARANCE
            ),
            "sustainable_max_collapse_rate": (
                FRONTIER_SUSTAINABLE_MAX_COLLAPSE_RATE
            ),
            "critical_multiplier_definition": (
                "largest contiguous sustainable point starting at the lowest "
                "tested multiplier; also report the highest isolated passing point"
            ),
            "comparisons": [
                {"name": name, "baseline": baseline, "candidate": candidate}
                for name, baseline, candidate in FRONTIER_COMPARISONS
            ],
        },
        "reported_metrics": list(DERIVED_REPORT_METRICS),
        "forbidden": {
            "legacy_event_logit_described_as_combo": True,
            "mixed_manifests_within_a_paired_cell": True,
            "committed_or_eta_admission": True,
            "dynamic_j": True,
            "station_feedback": True,
            "post_run_tuning": True,
            "seed_or_load_dropping": True,
        },
    }


def protocol_sha256() -> str:
    return canonical_sha256(formal_protocol())


__all__ = [
    "BASE_ROOT",
    "BUNDLE_SCHEMA_VERSION",
    "DERIVED_REPORT_METRICS",
    "FRONTIER_AUDIT_SCHEMA_VERSION",
    "FRONTIER_COLLAPSE_EFFICIENCY",
    "FRONTIER_COMPARISONS",
    "FRONTIER_LOAD",
    "FRONTIER_OUTPUT_ROOT",
    "FRONTIER_POINTS",
    "FRONTIER_SEEDS",
    "FRONTIER_SOURCE_ROOT",
    "FRONTIER_SUMMARY_SCHEMA_VERSION",
    "FRONTIER_SUSTAINABLE_MAX_COLLAPSE_RATE",
    "FRONTIER_SUSTAINABLE_MEAN_CLEARANCE",
    "FRESH_SEEDS",
    "LEGACY_AUDIT_SCHEMA_VERSION",
    "LEGACY_HIGH_MANIFEST_ROOT",
    "LEGACY_HIGH_REFERENCE_ROOT",
    "LEGACY_LOW_MID_MANIFEST_ROOT",
    "LEGACY_LOW_MID_REFERENCE_ROOT",
    "LEGACY_REFERENCE_ARM_DIRS",
    "LEGACY_REFERENCE_LABELS",
    "LEGACY_SEEDS",
    "LOADS",
    "MAIN_FRESH_COMPARISONS",
    "MAIN_FRESH_OUTPUT_ROOT",
    "MAIN_LEGACY_COMPARISONS",
    "MAIN_LEGACY_OUTPUT_ROOT",
    "MAIN_SUMMARY_SCHEMA_VERSION",
    "RUN_ARMS",
    "RUN_ARM_LABELS",
    "RUN_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "TICKS",
    "formal_protocol",
    "protocol_sha256",
    "selector_config",
    "sha256_file",
]
