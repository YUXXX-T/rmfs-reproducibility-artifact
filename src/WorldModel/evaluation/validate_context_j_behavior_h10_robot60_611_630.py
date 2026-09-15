"""Strict audit for the 60-robot 611--630 context-J label bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import WorldModel.evaluation.validate_context_j_behavior_h10_571_590 as base
from WorldModel.evaluation.context_j_behavior_h10_robot60_protocol import (
    REPORT_SCHEMA_VERSION,
    SEEDS,
    TEST_SEEDS,
    TRAIN_SEEDS,
    VAL_SEEDS,
)


EXPECTED_ROBOT_COUNT = 60
EXPECTED_FIFO_MODE = "committed_capacity_fifo_v2"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _install_robot60_split_contract() -> None:
    base.EXPECTED_SEEDS = tuple(SEEDS)
    original_signal = base._signal_for_run

    def signal_for_run(*args, **kwargs):
        result = original_signal(*args, **kwargs)
        seed = int(result["seed"])
        if seed in TRAIN_SEEDS:
            result["split"] = "train"
        elif seed in VAL_SEEDS:
            result["split"] = "val"
        elif seed in TEST_SEEDS:
            result["split"] = "test"
        else:
            raise ValueError(f"seed outside frozen 611--630 split: {seed}")
        return result

    base._signal_for_run = signal_for_run


def _robot60_fifo_checks(
    *, data_root: Path, bundle: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = bundle["protocol"]
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for load in base.EXPECTED_LOADS:
        expected_config = Path(protocol["inputs"]["loads"][load]).resolve()
        config_payload = json.loads(expected_config.read_text(encoding="utf-8"))
        config_is_60 = (
            int(config_payload.get("robots", {}).get("num_robots", -1))
            == EXPECTED_ROBOT_COUNT
        )
        for seed in SEEDS:
            run = data_root / "runs" / f"{load}_seed{seed}"
            marker_path = run / "run_complete.json"
            generation_path = run / "gen_config.json"
            metadata_path = run / "behavior_h10_data_meta.json"
            fifo_path = run / "fifo_admission_audit.json"
            checks = {
                "active_config_is_60": config_is_60,
                "marker_present": marker_path.is_file(),
                "generation_present": generation_path.is_file(),
                "metadata_present": metadata_path.is_file(),
                "fifo_audit_present": fifo_path.is_file(),
            }
            marker = (
                json.loads(marker_path.read_text(encoding="utf-8"))
                if marker_path.is_file() else {}
            )
            generation = (
                json.loads(generation_path.read_text(encoding="utf-8"))
                if generation_path.is_file() else {}
            )
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.is_file() else {}
            )
            fifo = (
                json.loads(fifo_path.read_text(encoding="utf-8"))
                if fifo_path.is_file() else {}
            )
            params = metadata.get("generation_params") or {}
            checks.update({
                "marker_robot_count": int(marker.get("num_robots", -1))
                == EXPECTED_ROBOT_COUNT,
                "marker_fifo_mode": marker.get("station_admission")
                == EXPECTED_FIFO_MODE,
                "generation_robot_count": int(generation.get("num_robots", -1))
                == EXPECTED_ROBOT_COUNT,
                "metadata_robot_count": int(params.get("num_robots", -1))
                == EXPECTED_ROBOT_COUNT,
                "generation_config_matches": (
                    bool(generation.get("config_path"))
                    and Path(str(generation["config_path"])).resolve()
                    == expected_config
                ),
                "fifo_hash_matches": (
                    fifo_path.is_file()
                    and marker.get("fifo_audit_sha256") == _digest(fifo_path)
                ),
                "fifo_passed": bool(fifo.get("passed")),
                "fifo_mode": fifo.get("mode") == EXPECTED_FIFO_MODE,
                "fifo_num_agents": int(fifo.get("num_agents", -1))
                == EXPECTED_ROBOT_COUNT,
                "fifo_ticks": int(fifo.get("ticks", -1))
                == int(protocol["collection"]["ticks"]),
            })
            passed = all(checks.values())
            row = {
                "load": load,
                "seed": int(seed),
                "checks": checks,
                "passed": passed,
            }
            rows.append(row)
            if not passed:
                failed.append({
                    "load": load,
                    "seed": int(seed),
                    "failed_checks": [
                        name for name, value in checks.items() if not value
                    ],
                })
    return rows, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--pair-epsilon", type=float, default=0.01)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=30)
    args = parser.parse_args()
    if args.pair_epsilon < 0:
        raise SystemExit("--pair-epsilon must be non-negative")

    _install_robot60_split_contract()
    result = base.run_audit(
        data_root=args.data_root,
        bundle_path=args.bundle,
        save_json=args.save_json,
        pair_epsilon=args.pair_epsilon,
        allow_partial=args.allow_partial,
        run_model_validation=False,
        model_report_path=None,
        torch_threads=args.torch_threads,
    )
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    robot_rows, robot_failed = _robot60_fifo_checks(
        data_root=args.data_root, bundle=bundle
    )
    result["schema_version"] = REPORT_SCHEMA_VERSION
    result["robot60_fifo_audit"] = {
        "expected_runs": 60,
        "completed_runs": len(robot_rows),
        "failed_runs": robot_failed,
        "runs": robot_rows,
    }
    result["gates"].update({
        "all_runs_use_60_robots": not robot_failed,
        "all_fifo_v2_invariants_passed": not robot_failed,
    })
    result["gates"]["all_gates_passed"] = all(
        bool(value)
        for name, value in result["gates"].items()
        if name != "all_gates_passed"
    )
    result["interpretation"]["next_step"] = (
        "fit J60 with train 611--622, select only on val 623--626, "
        "and hold out test 627--630; seeds 601--610 remain excluded"
    )
    args.save_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not result["gates"]["all_gates_passed"] and not args.allow_partial:
        raise RuntimeError(
            "robot60 context-J label audit failed; see gates/robot60_fifo_audit"
        )

    print("=" * 88)
    print("Context-J robot60 behavior H=10 label audit")
    print("=" * 88)
    print("runs =", result["completed_runs"], "/", result["expected_runs"])
    for split, row in result["splits"].items():
        print(
            split,
            "frames=", row["multi_context_frames"],
            "contexts=", row["context_rows"],
            "non_tie_pairs=", row["non_tie_context_pairs"],
        )
    print("all_gates_passed =", result["gates"]["all_gates_passed"])
    print("JSON saved:", args.save_json)


if __name__ == "__main__":
    main()
