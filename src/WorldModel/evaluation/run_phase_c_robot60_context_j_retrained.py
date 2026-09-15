"""Run only the 60-robot-adapted context-J arm on seeds 601--610.

This is a non-destructive supplement to the frozen four-arm 60-robot block in
``phasec_s1_robot60_601_610_v1``.  It replays those exact order manifests and
combines the 30 new runs with the existing Greedy, Hungarian, Phase-C, and S1
rows.  The World Model, station-phi head, S1 robot scorer, and FIFO-V2
admission remain frozen; only the context-J head is the one trained from the
60-robot 611--630 behavior-continuation data.

The implementation deliberately wraps the earlier zero-shot supplement
runner instead of modifying it, so its frozen protocol and prior outputs stay
reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from WorldModel.evaluation import run_phase_c_robot60_context_j_missing as base


SCHEMA_VERSION = "phase_c_robot60_context_j_retrained_run_v1"
PROTOCOL_SCHEMA_VERSION = "phase_c_robot60_context_j_retrained_protocol_v1"
SUMMARY_SCHEMA_VERSION = "phase_c_robot60_context_j_retrained_summary_v1"

BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "ctxj_robot60_retrained_601_610_v1"
CONTEXT_HEAD = BASE_ROOT / (
    "context_dispatch_j_head_train_robot60_611_630_v1/"
    "best_context_j_head.pt"
)

ARM = "learned_j_robot60_adapted"
ARM_LABEL = "PhaseCLearnedJRobot60AdaptedFifoV2"
RUNNER_PATH = Path(
    "WorldModel/evaluation/run_phase_c_robot60_context_j_retrained.py"
)
BASE_RUNNER_PATH = Path(
    "WorldModel/evaluation/run_phase_c_robot60_context_j_missing.py"
)


def _configure_base() -> None:
    base.__doc__ = __doc__
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.PROTOCOL_SCHEMA_VERSION = PROTOCOL_SCHEMA_VERSION
    base.SUMMARY_SCHEMA_VERSION = SUMMARY_SCHEMA_VERSION
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    base.ARM = ARM
    base.COMBINED_ARMS = base.REFERENCE_ARMS + (ARM,)
    base.ARM_LABEL = ARM_LABEL
    base.NEW_CONTEXT_HEAD = CONTEXT_HEAD
    base.RUNNER_PATH = RUNNER_PATH


_configure_base()
_base_make_protocol = base._make_protocol
_base_run_arm = base._run_arm
_base_summarise = base._summarise


def _make_protocol(reference_root: Path) -> dict[str, Any]:
    bundle = _base_make_protocol(reference_root)
    protocol = bundle["protocol"]
    protocol["purpose"] = (
        "evaluate the 60-robot-adapted context-J head on the fixed paired "
        "601-610 robot-count regression block"
    )
    protocol["context_j_head"].update(
        {
            "training_block": "robot60_611_630",
            "training_robot_count": 60,
            "training_seeds": list(range(611, 631)),
            "evaluation_seeds_excluded_from_training": True,
        }
    )
    protocol["adaptation"] = {
        "world_model_retrained": False,
        "station_phi_retrained": False,
        "context_j_retrained": True,
        "context_j_training_robot_count": 60,
        "s1_retuned": False,
        "fifo_modified": False,
        "evaluation_kind": "fixed_held_out_robot_count_adaptation_regression",
        "zero_shot_context_j": False,
        "allowed_config_changes": ["robots.num_robots"],
    }
    protocol.pop("zero_shot", None)
    protocol["code_hashes"]["base_supplement_runner"] = {
        "path": BASE_RUNNER_PATH.as_posix(),
        "sha256": base.sha256_file(BASE_RUNNER_PATH),
    }
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol": protocol,
        "protocol_sha256": base._canonical_sha(protocol),
    }


def _run_arm(
    root: Path,
    load: str,
    seed: int,
    ticks: int,
    trace_max: int,
) -> None:
    _base_run_arm(root, load, seed, ticks, trace_max)
    output = base._output_path(root, load, seed)
    payload = base._read_json(output)
    payload["generalization"] = {
        "source_world_model_robot_count": 48,
        "target_robot_count": 60,
        "core_world_model_frozen": True,
        "station_phi_frozen": True,
        "s1_robot_scorer_frozen": True,
        "context_j_training_robot_count": 60,
        "context_j_adapted": True,
        "zero_shot_context_j": False,
        "evaluation_kind": "fixed_held_out_robot_count_adaptation_regression",
        "changed_config_fields": ["robots.num_robots"],
    }
    base._atomic_json(output, payload)


def _rewrite_result_hashes(root: Path) -> None:
    lines: list[str] = []
    for path in sorted(root.glob("per_arm/**/*.json")):
        lines.append(
            f"{base.sha256_file(path)}  {path.relative_to(root).as_posix()}"
        )
    for path in (
        root / "online_summary.json",
        base._protocol_path(root),
        base._frozen_inputs_path(root),
    ):
        lines.append(
            f"{base.sha256_file(path)}  {path.relative_to(root).as_posix()}"
        )
    (root / "results.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _summarise(root: Path) -> None:
    _base_summarise(root)
    path = root / "online_summary.json"
    report = base._read_json(path)
    report["context_j_training_robot_count"] = 60
    report["evaluation_kind"] = (
        "fixed_held_out_robot_count_adaptation_regression"
    )
    report["audit"].pop("zero_shot_checkpoints_unchanged", None)
    report["audit"].update(
        {
            "frozen_world_model_station_phi_and_s1_unchanged": True,
            "adapted_robot60_context_j_used": True,
            "evaluation_seeds_excluded_from_context_j_training": True,
        }
    )
    base._atomic_json(path, report)
    _rewrite_result_hashes(root)


base._make_protocol = _make_protocol
base._run_arm = _run_arm
base._summarise = _summarise


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
