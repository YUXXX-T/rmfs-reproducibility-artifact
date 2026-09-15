"""Frozen protocol for the corrected LongRiskHead S1 signal comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from WorldModel.core.long_risk_schema import long_risk_runtime_contract
from WorldModel.evaluation.phase_c_s1_hungarian_protocol import (
    PHASEC_CONFIG,
    S1_CONFIG,
)


SCHEMA_VERSION = "phase_c_s1_long_risk_correction_protocol_v1"
BUNDLE_SCHEMA_VERSION = "phase_c_s1_long_risk_correction_bundle_v1"
PER_SEED_SCHEMA_VERSION = "phase_c_s1_long_risk_correction_seed_v1"
SUMMARY_SCHEMA_VERSION = "phase_c_s1_long_risk_correction_summary_v1"

BASE_ROOT = "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1"
OUTPUT_ROOT = f"{BASE_ROOT}/phasec_s1_long_risk_correction_631_640_v1"
CANDIDATE_CHECKPOINT = f"{BASE_ROOT}/model_round1_v1/best_regret_world_model.pt"

LOADS = ("low", "mid", "high")
LOAD_CONFIGS = {
    "low": "Config/world_model_config_PP_48_low.json",
    "mid": "Config/world_model_config_PP_48_mid.json",
    "high": "Config/world_model_config_PP_48_high.json",
}
SEEDS = tuple(range(631, 641))
TICKS = 1500
TOP_M = 10

ARM_LABELS = {
    "greedy": "GreedySequentialManhattan",
    "hungarian": "HungarianBatchManhattan",
    "phasec": "PhaseCRobotOnly",
    "s1_event": "PhaseCS1LegacyEventLogit",
    "s1_terminal": "PhaseCS1TerminalQ90",
    "s1_combo": "PhaseCS1QuantileComboV2",
}
ARM_KEYS = tuple(ARM_LABELS)

S1_SIGNAL_CONFIGS = {
    signal: {**S1_CONFIG, "energy_drift_signal": signal}
    for signal in ("event_logit", "terminal", "combo")
}

REPORT_METRICS = (
    "completed_orders",
    "completed_tasks",
    "avg_task_duration",
    "avg_excess_delay",
    "open_order_count",
    "pending_order_count",
    "stall_ratio_mean",
    "stall_ratio_max",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "congestion_events",
    "severe_events",
    "risk_rate_per_100",
    "wall_time_s",
    "assignment_time_ms_mean",
)

COMPARISONS = (
    ("s1_combo_minus_phasec", "phasec", "s1_combo"),
    ("s1_combo_minus_greedy", "greedy", "s1_combo"),
    ("s1_combo_minus_hungarian", "hungarian", "s1_combo"),
    ("s1_combo_minus_legacy_event", "s1_event", "s1_combo"),
    ("s1_combo_minus_terminal", "s1_terminal", "s1_combo"),
    ("legacy_event_minus_phasec", "phasec", "s1_event"),
    ("terminal_minus_phasec", "phasec", "s1_terminal"),
)


def canonical_sha256(payload: Mapping) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formal_protocol() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "fresh-seed paired revalidation of the corrected LongRiskHead "
            "quantile-combo S1 runtime against the historical event-logit "
            "semantics, terminal-only S1, Phase C, Greedy, and Hungarian"
        ),
        "formal_test": {
            "loads": list(LOADS),
            "seeds": list(SEEDS),
            "ticks": TICKS,
            "top_m": TOP_M,
            "arms": dict(ARM_LABELS),
            "manifest_source": ARM_LABELS["greedy"],
            "one_manifest_per_load_seed": True,
            "exact_manifest_replay_required": True,
            "fresh_seed_range": True,
        },
        "world_model": {
            "checkpoint": CANDIDATE_CHECKPOINT,
            "phasec_config": dict(PHASEC_CONFIG),
            "s1_signal_configs": {
                key: dict(value)
                for key, value in S1_SIGNAL_CONFIGS.items()
            },
            "long_risk_runtime_contract": long_risk_runtime_contract(),
        },
        "interpretation": {
            "s1_event": (
                "honest reproduction of the historical formal S1 signal"
            ),
            "s1_terminal": "single-tail-channel ablation",
            "s1_combo": (
                "corrected explicit 0.2*peak_q95 + 0.3*cvar_q90 + "
                "0.5*terminal_q90 signal"
            ),
            "within_context_only": True,
            "cross_context_reordering": False,
            "station_admission_modification": False,
        },
        "forbidden": {
            "post_run_signal_or_lambda_tuning": True,
            "seed_or_load_dropping": True,
            "checkpoint_reselection": True,
            "native_no_assign": True,
            "new_station_feedback": True,
        },
        "reported_metrics": list(REPORT_METRICS),
        "paired_comparisons": [
            {"name": name, "baseline": baseline, "candidate": candidate}
            for name, baseline, candidate in COMPARISONS
        ],
    }

