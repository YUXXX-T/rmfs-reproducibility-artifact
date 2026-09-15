"""Run one paired early-BRAKE station-feedback capacity-frontier arm.

This runner reuses the audited closed-loop execution path while adding one
explicitly isolated controller profile.  ETA V3 admission and its hard
committed ceiling remain engine-owned and unchanged.  The V2 controller only
defers new contexts for one station after sustained soft committed pressure is
corroborated by at least two station-flow signals.

The formal factorial contains four arms:

* ``off``: frozen Dynamic-J + pipeline+phi V2 + S1 baseline;
* ``shadow_v2``: observe the early-BRAKE profile without changing decisions;
* ``active_v1``: the previous late-BRAKE closed loop;
* ``active_v2``: the early-BRAKE profile with station-local batch filtering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from WorldModel.core.station_feedback_control import (
    STATION_FEEDBACK_EARLY_BRAKE_EXPERIMENT_MODES,
    STATION_FEEDBACK_MODE_ACTIVE,
    STATION_FEEDBACK_MODE_ACTIVE_V2,
    STATION_FEEDBACK_MODE_OFF,
    STATION_FEEDBACK_MODE_SHADOW_V2,
    StationFeedbackConfig,
    station_feedback_mode_uses_early_brake,
)
from WorldModel.evaluation.phase_c_psi_dispatch_ablation_protocol import LOADS
from WorldModel.evaluation.run_phase_c_station_admission_restoration import (
    DEFAULT_BUNDLE,
    FORMAL_TICKS,
)
from WorldModel.evaluation.run_phase_c_station_feedback_closed_loop import (
    BASE_ROOT,
    DEFAULT_REFERENCE_ARM,
    DEFAULT_REFERENCE_ROOT,
    DEFAULT_SOURCE_ROOT,
    _run,
    _validate_args,
)


SCHEMA_VERSION = "phase_c_station_feedback_early_brake_arm_v1"
EXPERIMENT_CONTRACT_VERSION = (
    "phase_c_station_feedback_early_brake_capacity_frontier_v1"
)
DEFAULT_OUTPUT_ROOT = (
    BASE_ROOT / "station_feedback_closed_loop_capacity_frontier_high_551_560_v3"
    / "m100"
)

ARM_KEYS = {
    STATION_FEEDBACK_MODE_OFF: (
        "s1_dynamic_j_pipeline_phi_eta_v3_feedback_off"
    ),
    STATION_FEEDBACK_MODE_SHADOW_V2: (
        "s1_dynamic_j_pipeline_phi_eta_v3_feedback_shadow_v2"
    ),
    STATION_FEEDBACK_MODE_ACTIVE: (
        "s1_dynamic_j_pipeline_phi_eta_v3_feedback_active_v1"
    ),
    STATION_FEEDBACK_MODE_ACTIVE_V2: (
        "s1_dynamic_j_pipeline_phi_eta_v3_feedback_active_v2"
    ),
}


def _feedback_config(
    service_ticks: int,
    feedback_mode: str,
) -> StationFeedbackConfig:
    return StationFeedbackConfig.for_service_ticks(
        service_ticks,
        early_brake_enabled=station_feedback_mode_uses_early_brake(
            feedback_mode
        ),
    )


def _validate_early_args(args: argparse.Namespace) -> None:
    _validate_args(args)
    if float(args.max_committed_multiplier) <= 1.4:
        raise SystemExit(
            "early committed-pressure multiplier must remain below the "
            "ETA hard committed multiplier"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--station-feedback-mode",
        choices=STATION_FEEDBACK_EARLY_BRAKE_EXPERIMENT_MODES,
        required=True,
    )
    parser.add_argument("--load", choices=LOADS, default="high")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ticks", type=int, default=FORMAL_TICKS)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT
    )
    parser.add_argument(
        "--reference-arm", default=DEFAULT_REFERENCE_ARM
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frozen-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--feedback-trace-max-records", type=int, default=1000)
    parser.add_argument("--policy-trace-max-records", type=int, default=0)
    parser.add_argument("--station-trace-stride", type=int, default=10)
    parser.add_argument("--station-trace-max-records", type=int, default=300)
    parser.add_argument("--max-committed-multiplier", type=float, default=2.0)
    parser.add_argument("--healthy-extra-ratio", type=float, default=1.0)
    parser.add_argument("--caution-extra-ratio", type=float, default=0.5)
    parser.add_argument("--brake-extra-ratio", type=float, default=0.0)
    parser.add_argument("--eta-near-ticks", type=int, default=8)
    parser.add_argument("--eta-mid-ticks", type=int, default=20)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    _validate_early_args(args)
    _run(
        args,
        arm_keys=ARM_KEYS,
        schema_version=SCHEMA_VERSION,
        contract_version=EXPERIMENT_CONTRACT_VERSION,
        feedback_config_factory=_feedback_config,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ARM_KEYS",
    "DEFAULT_OUTPUT_ROOT",
    "EXPERIMENT_CONTRACT_VERSION",
    "SCHEMA_VERSION",
    "_feedback_config",
]
