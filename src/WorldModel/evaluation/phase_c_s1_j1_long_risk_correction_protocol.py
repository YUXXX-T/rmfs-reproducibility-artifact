"""Frozen protocol for the corrected LongRiskHead S1 x static-J1 test."""

from __future__ import annotations

from pathlib import Path

from WorldModel.core.long_risk_schema import long_risk_runtime_contract
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import (
    PSI_HEAD_CHECKPOINT,
    PSI_SCALE_CONTRACT,
)
from WorldModel.evaluation.phase_c_s1_long_risk_correction_protocol import (
    ARM_LABELS as SOURCE_ARM_LABELS,
    CANDIDATE_CHECKPOINT,
    LOAD_CONFIGS,
    LOADS,
    PHASEC_CONFIG,
    REPORT_METRICS,
    SEEDS,
    S1_SIGNAL_CONFIGS,
    TICKS,
    TOP_M,
    canonical_sha256,
    sha256_file,
)
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


SCHEMA_VERSION = "phase_c_s1_j1_long_risk_correction_protocol_v1"
BUNDLE_SCHEMA_VERSION = "phase_c_s1_j1_long_risk_correction_bundle_v1"
PER_SEED_SCHEMA_VERSION = "phase_c_s1_j1_long_risk_correction_seed_v1"
SUMMARY_SCHEMA_VERSION = "phase_c_s1_j1_long_risk_correction_summary_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
SOURCE_ROOT = BASE_ROOT / "phasec_s1_long_risk_correction_631_640_v1"
OUTPUT_ROOT = BASE_ROOT / "phasec_s1_j1_long_risk_correction_631_640_v1"

NEW_ARM_SPECS = {
    "s0_j1": {
        "label": "PhaseCS0StaticJ1",
        "selector": "s0",
        "signal": None,
    },
    "s1_event_j1": {
        "label": "PhaseCS1LegacyEventLogitStaticJ1",
        "selector": "s1",
        "signal": "event_logit",
    },
    "s1_terminal_j1": {
        "label": "PhaseCS1TerminalQ90StaticJ1",
        "selector": "s1",
        "signal": "terminal",
    },
    "s1_combo_j1": {
        "label": "PhaseCS1QuantileComboV2StaticJ1",
        "selector": "s1",
        "signal": "combo",
    },
}
NEW_ARM_KEYS = tuple(NEW_ARM_SPECS)
NEW_ARM_LABELS = {
    key: str(spec["label"]) for key, spec in NEW_ARM_SPECS.items()
}
ALL_ARM_LABELS = {**SOURCE_ARM_LABELS, **NEW_ARM_LABELS}
ALL_ARM_KEYS = tuple(ALL_ARM_LABELS)

# These paired contrasts answer main effects before any station-queue test.
COMPARISONS = (
    ("s0_j1_minus_phasec", "phasec", "s0_j1"),
    ("event_j1_minus_event_j0", "s1_event", "s1_event_j1"),
    ("terminal_j1_minus_terminal_j0", "s1_terminal", "s1_terminal_j1"),
    ("combo_j1_minus_combo_j0", "s1_combo", "s1_combo_j1"),
    ("event_j1_minus_s0_j1", "s0_j1", "s1_event_j1"),
    ("terminal_j1_minus_s0_j1", "s0_j1", "s1_terminal_j1"),
    ("combo_j1_minus_s0_j1", "s0_j1", "s1_combo_j1"),
    ("combo_j1_minus_event_j1", "s1_event_j1", "s1_combo_j1"),
    ("combo_j1_minus_terminal_j1", "s1_terminal_j1", "s1_combo_j1"),
    ("s0_j1_minus_greedy", "greedy", "s0_j1"),
    ("event_j1_minus_greedy", "greedy", "s1_event_j1"),
    ("terminal_j1_minus_greedy", "greedy", "s1_terminal_j1"),
    ("combo_j1_minus_greedy", "greedy", "s1_combo_j1"),
    ("s0_j1_minus_hungarian", "hungarian", "s0_j1"),
    ("event_j1_minus_hungarian", "hungarian", "s1_event_j1"),
    ("terminal_j1_minus_hungarian", "hungarian", "s1_terminal_j1"),
    ("combo_j1_minus_hungarian", "hungarian", "s1_combo_j1"),
    ("event_j1_minus_phasec", "phasec", "s1_event_j1"),
    ("terminal_j1_minus_phasec", "phasec", "s1_terminal_j1"),
    ("combo_j1_minus_phasec", "phasec", "s1_combo_j1"),
)

# (S1+J1 - S1+J0) - (S0+J1 - S0+J0), evaluated per load/seed.
INTERACTIONS = (
    (
        "event_s1_x_j1",
        "phasec",
        "s0_j1",
        "s1_event",
        "s1_event_j1",
    ),
    (
        "terminal_s1_x_j1",
        "phasec",
        "s0_j1",
        "s1_terminal",
        "s1_terminal_j1",
    ),
    (
        "combo_s1_x_j1",
        "phasec",
        "s0_j1",
        "s1_combo",
        "s1_combo_j1",
    ),
)


def selector_config(arm: str) -> dict:
    """Return a detached scorer configuration for one new J1 arm."""

    spec = NEW_ARM_SPECS[arm]
    signal = spec["signal"]
    if signal is None:
        return dict(PHASEC_CONFIG)
    return dict(S1_SIGNAL_CONFIGS[str(signal)])


def formal_protocol(source_root: str | Path = SOURCE_ROOT) -> dict:
    source_root = Path(source_root)
    protocol: dict = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "paired physical-only test of static J1 with Phase-C S0, "
            "historical event-logit S1, terminal-q90 S1, and the corrected "
            "explicit quantile-combo S1"
        ),
        "formal_test": {
            "loads": list(LOADS),
            "seeds": list(SEEDS),
            "ticks": TICKS,
            "top_m": TOP_M,
            "source_root": source_root.as_posix(),
            "source_arms_reused": dict(SOURCE_ARM_LABELS),
            "new_arms": dict(NEW_ARM_LABELS),
            "new_simulation_count": len(NEW_ARM_KEYS) * len(LOADS) * len(SEEDS),
            "combined_arm_count": len(ALL_ARM_KEYS),
            "exact_source_manifest_replay_required": True,
        },
        "station_admission": {
            "mode": STATION_ADMISSION_PHYSICAL_ONLY,
            "physical_occupancy_cap_enforced": True,
            "in_transit_committed_cap_enforced": False,
            "eta_controller": False,
            "fifo_waiting": False,
            "dispatch_preserving_station_queue": False,
        },
        "world_model": {
            "checkpoint": str(CANDIDATE_CHECKPOINT),
            "psi_head_checkpoint": Path(PSI_HEAD_CHECKPOINT).as_posix(),
            "psi_scale_contract": Path(PSI_SCALE_CONTRACT).as_posix(),
            "s0_config": dict(PHASEC_CONFIG),
            "s1_signal_configs": {
                key: dict(value) for key, value in S1_SIGNAL_CONFIGS.items()
            },
            "long_risk_runtime_contract": long_risk_runtime_contract(),
        },
        "factorial_interpretation": {
            "j0": "fixed candidate-context prefix; source campaign",
            "j1": "one static ascending-J order per proposal batch",
            "within_context_selector": (
                "S0, event-logit S1, terminal-q90 S1, or corrected combo S1"
            ),
            "interaction_definition": (
                "(S1+J1 - S1+J0) - (S0+J1 - S0+J0)"
            ),
        },
        "paired_comparisons": [
            {"name": name, "baseline": baseline, "candidate": candidate}
            for name, baseline, candidate in COMPARISONS
        ],
        "interaction_contrasts": [
            {
                "name": name,
                "s0_j0": s0_j0,
                "s0_j1": s0_j1,
                "s1_j0": s1_j0,
                "s1_j1": s1_j1,
            }
            for name, s0_j0, s0_j1, s1_j0, s1_j1 in INTERACTIONS
        ],
        "reported_metrics": list(REPORT_METRICS),
        "forbidden": {
            "station_queue": True,
            "committed_or_eta_admission": True,
            "dynamic_j": True,
            "post_run_signal_or_lambda_tuning": True,
            "seed_or_load_dropping": True,
            "checkpoint_reselection": True,
            "native_no_assign": True,
            "new_station_feedback": True,
        },
    }
    return protocol


def protocol_sha256(source_root: str | Path = SOURCE_ROOT) -> str:
    return canonical_sha256(formal_protocol(source_root))


__all__ = [
    "ALL_ARM_KEYS",
    "ALL_ARM_LABELS",
    "BUNDLE_SCHEMA_VERSION",
    "COMPARISONS",
    "INTERACTIONS",
    "NEW_ARM_KEYS",
    "NEW_ARM_LABELS",
    "NEW_ARM_SPECS",
    "OUTPUT_ROOT",
    "PER_SEED_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SOURCE_ROOT",
    "SUMMARY_SCHEMA_VERSION",
    "formal_protocol",
    "protocol_sha256",
    "selector_config",
    "sha256_file",
]
