"""Run the paired service-due station-feedback capacity-frontier arm.

This V4 experiment keeps ETA V3 admission, Dynamic-J, pipeline+phi V2, S1,
the World Model, order generation, task lifecycle, and path planning frozen.
It changes only the explicitly selected station-feedback detector:

* ``off``: no feedback controller;
* ``shadow_v3``: audit service-due release warnings without changing choices;
* ``active_v2``: replay the previous two-of-three early-BRAKE profile;
* ``active_v3``: defer only after a service robot is due to exit, the normal
  two-phase handoff grace has expired, and the exit remains blocked.

The ordinary station service countdown is never counted as release drought in
V3.  Admission rejection remains diagnostic and cannot independently create a
V3 BRAKE or CAUTION state.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from WorldModel.core.station_feedback_control import (
    STATION_FEEDBACK_MODE_ACTIVE_V2,
    STATION_FEEDBACK_MODE_ACTIVE_V3,
    STATION_FEEDBACK_MODE_OFF,
    STATION_FEEDBACK_MODE_SHADOW_V3,
    STATION_FEEDBACK_RELEASE_AWARE_EXPERIMENT_MODES,
    StationFeedbackConfig,
    station_feedback_mode_uses_release_aware_brake,
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


SCHEMA_VERSION = "phase_c_station_feedback_release_aware_arm_v1"
EXPERIMENT_CONTRACT_VERSION = (
    "phase_c_station_feedback_release_aware_capacity_frontier_v1"
)
DEFAULT_OUTPUT_ROOT = (
    BASE_ROOT / "station_feedback_closed_loop_capacity_frontier_high_551_560_v4"
    / "m100"
)

ARM_KEYS = {
    STATION_FEEDBACK_MODE_OFF: (
        "s1_dynamic_j_pipeline_phi_eta_v3_feedback_off"
    ),
    STATION_FEEDBACK_MODE_SHADOW_V3: (
        "s1_dynamic_j_pipeline_phi_eta_v3_feedback_shadow_v3"
    ),
    STATION_FEEDBACK_MODE_ACTIVE_V2: (
        "s1_dynamic_j_pipeline_phi_eta_v3_feedback_active_v2"
    ),
    STATION_FEEDBACK_MODE_ACTIVE_V3: (
        "s1_dynamic_j_pipeline_phi_eta_v3_feedback_active_v3"
    ),
}


def _feedback_config(
    service_ticks: int,
    feedback_mode: str,
) -> StationFeedbackConfig:
    if station_feedback_mode_uses_release_aware_brake(feedback_mode):
        return StationFeedbackConfig.for_service_due_release(
            service_ticks,
            early_brake_enabled=True,
        )
    return StationFeedbackConfig.for_service_ticks(
        service_ticks,
        early_brake_enabled=(
            feedback_mode == STATION_FEEDBACK_MODE_ACTIVE_V2
        ),
    )


def _validate_release_aware_args(args: argparse.Namespace) -> None:
    _validate_args(args)
    if float(args.max_committed_multiplier) <= 1.4:
        raise SystemExit(
            "feedback committed-pressure multiplier must remain below the "
            "ETA hard committed multiplier"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--station-feedback-mode",
        choices=STATION_FEEDBACK_RELEASE_AWARE_EXPERIMENT_MODES,
        required=True,
    )
    parser.add_argument("--load", choices=LOADS, default="high")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ticks", type=int, default=FORMAL_TICKS)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT
    )
    parser.add_argument("--reference-arm", default=DEFAULT_REFERENCE_ARM)
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
    _validate_release_aware_args(args)
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
