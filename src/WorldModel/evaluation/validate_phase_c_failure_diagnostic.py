"""Validate and summarize the targeted Phase-C failure diagnostic outputs."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any

from WorldModel.evaluation.phase_c_failure_diagnostic_protocol import (
    ARMS,
    CASES,
    COUNTERFACTUAL_SCHEMA_VERSION,
    OUTPUT_ROOT,
    RUN_SCHEMA_VERSION,
    TICKS,
    VALIDATION_SCHEMA_VERSION,
    run_id,
    sha256_file,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{number}")
            rows.append(value)
    return rows


def _verify_hash_manifest(root: Path, path: Path) -> tuple[bool, int]:
    count = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(None, 1)
        target = root / relative.strip()
        if not target.is_file() or sha256_file(target) != expected:
            return False, count
        count += 1
    return True, count


def _persistent_deadlock_onset(
    deadlock: list[float], *, window: int = 50, threshold: float = 0.5
) -> int | None:
    for tick in range(0, max(0, len(deadlock) - 2 * window + 1)):
        first = deadlock[tick:tick + window]
        second = deadlock[tick + window:tick + 2 * window]
        if fmean(first) >= threshold and fmean(second) >= threshold:
            return tick
    return None


def _trajectory_summary(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    deadlock = [float(row["risk"]["deadlock_ratio"]) for row in rows]
    completion_ticks = [
        int(row["tick"])
        for row in rows
        if int(row.get("completed_orders_delta", 0)) > 0
    ]
    modified = []
    decision_count = 0
    for row in rows:
        decisions = row.get("decisions") or []
        decision_count += len(decisions)
        if any(
            bool(decision.get("energy_conversion_modified"))
            for decision in decisions
        ):
            modified.append(int(row["tick"]))
    return {
        "ticks": len(rows),
        "decision_contexts": decision_count,
        "first_deadlock_ge_0p5_tick": next(
            (index for index, value in enumerate(deadlock) if value >= 0.5),
            None,
        ),
        "persistent_deadlock_onset_tick": _persistent_deadlock_onset(deadlock),
        "deadlock_ratio_max": max(deadlock) if deadlock else None,
        "last_completion_tick": completion_ticks[-1] if completion_ticks else None,
        "energy_conversion_modified_ticks": sorted(set(modified)),
    }


def _write_exact(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"existing validation differs: {path}")
        print(f"[audit] unchanged {path}")
        return
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"[write] {path}")


def main() -> None:
    protocol_path = OUTPUT_ROOT / "phase_c_failure_diagnostic_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol_bundle = _read_json(protocol_path)
    protocol_sha = str((protocol_bundle.get("protocol") or {}).get(
        "protocol_sha256", ""
    ))
    if not protocol_sha:
        raise ValueError("diagnostic protocol lacks protocol_sha256")

    counterfactual_path = OUTPUT_ROOT / "phase_c_failure_counterfactuals.json"
    if not counterfactual_path.is_file():
        raise FileNotFoundError(counterfactual_path)
    counterfactual = _read_json(counterfactual_path)
    counterfactual_checks = {
        "schema": (
            counterfactual.get("schema_version")
            == COUNTERFACTUAL_SCHEMA_VERSION
        ),
        "protocol": counterfactual.get("protocol_sha256") == protocol_sha,
        "passed": bool(counterfactual.get("passed")),
        "case_count": len(counterfactual.get("cases") or []) == len(CASES),
    }

    run_reports = []
    failures = []
    output_hash_rows = [
        f"{sha256_file(protocol_path)}  {protocol_path.as_posix()}"
    ]
    for case in CASES:
        load = str(case["load"])
        seed = int(case["seed"])
        snapshot_arm = str(case["snapshot_arm"])
        for arm in ARMS:
            rid = run_id(arm, load, seed)
            run_dir = OUTPUT_ROOT / "runs" / rid
            summary_path = run_dir / "diagnostic_summary.json"
            manifest_path = run_dir / "run_outputs.sha256"
            if not summary_path.is_file() or not manifest_path.is_file():
                failures.append(f"missing output for {rid}")
                continue
            summary = _read_json(summary_path)
            manifest_ok, manifest_entries = _verify_hash_manifest(
                run_dir, manifest_path
            )
            checks = {
                "schema": summary.get("schema_version") == RUN_SCHEMA_VERSION,
                "protocol": summary.get("protocol_sha256") == protocol_sha,
                "audit": bool((summary.get("audit") or {}).get("passed")),
                "output_hashes": manifest_ok,
                "output_manifest_hash": (
                    summary.get("run_outputs_sha256")
                    == sha256_file(manifest_path)
                ),
                "snapshot_scope": bool(summary.get("capture_snapshots"))
                == (arm == snapshot_arm),
            }
            trajectory_name = str(summary["outputs"]["trajectory"])
            trajectory_path = run_dir / trajectory_name
            trajectory = _trajectory_summary(trajectory_path)
            checks["trajectory_ticks"] = trajectory["ticks"] == TICKS
            if not all(checks.values()):
                failed = [key for key, passed in checks.items() if not passed]
                failures.append(f"{rid}: {failed}")
            run_reports.append({
                "run_id": rid,
                "arm": arm,
                "load": load,
                "seed": seed,
                "capture_snapshots": bool(summary.get("capture_snapshots")),
                "snapshot_count": int(summary["outputs"]["snapshot_count"]),
                "trace_rows": int(summary["outputs"]["decision_trace_rows"]),
                "output_manifest_entries": manifest_entries,
                "checks": checks,
                "metrics": {
                    "completion_fraction": (
                        float(summary["metrics"]["completed_orders"])
                        / max(float(summary["metrics"]["order_arrival_count"]), 1.0)
                    ),
                    "deadlock_ratio_mean": float(
                        summary["metrics"]["deadlock_ratio_mean"]
                    ),
                    "completed_orders": int(
                        summary["metrics"]["completed_orders"]
                    ),
                },
                "trajectory": trajectory,
            })
            output_hash_rows.extend([
                f"{sha256_file(summary_path)}  {summary_path.as_posix()}",
                f"{sha256_file(manifest_path)}  {manifest_path.as_posix()}",
            ])

    report = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "protocol_sha256": protocol_sha,
        "passed": (
            not failures
            and len(run_reports) == len(CASES) * len(ARMS)
            and all(counterfactual_checks.values())
        ),
        "scope": {
            "cases": [dict(case) for case in CASES],
            "arms": list(ARMS),
            "expected_runs": len(CASES) * len(ARMS),
            "observed_runs": len(run_reports),
        },
        "failures": failures,
        "counterfactual": {
            "path": counterfactual_path.as_posix(),
            "sha256": sha256_file(counterfactual_path),
            "checks": counterfactual_checks,
            "cases": [
                {
                    "run_id": case.get("run_id"),
                    "passed": case.get("passed"),
                    "snapshot_groups": case.get("snapshot_groups"),
                    "aggregate_all_decisions": case.get(
                        "aggregate_all_decisions"
                    ),
                    "aggregate_pre_failure_200_ticks": case.get(
                        "aggregate_pre_failure_200_ticks"
                    ),
                }
                for case in counterfactual.get("cases", ())
            ],
        },
        "runs": run_reports,
    }
    report_path = OUTPUT_ROOT / "phase_c_failure_diagnostic_validation.json"
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    _write_exact(report_path, encoded)
    if not report["passed"]:
        failed_counterfactual = [
            key for key, passed in counterfactual_checks.items() if not passed
        ]
        details = list(failures)
        if failed_counterfactual:
            details.append(f"counterfactual: {failed_counterfactual}")
        raise RuntimeError("diagnostic validation failed: " + "; ".join(details))

    output_hash_rows.append(
        f"{sha256_file(report_path)}  {report_path.as_posix()}"
    )
    output_hash_rows.append(
        f"{sha256_file(counterfactual_path)}  {counterfactual_path.as_posix()}"
    )
    _write_exact(
        OUTPUT_ROOT / "diagnostic_outputs.sha256",
        "\n".join(output_hash_rows) + "\n",
    )
    print(f"[complete] validated {len(run_reports)} diagnostic runs")


if __name__ == "__main__":
    main()
