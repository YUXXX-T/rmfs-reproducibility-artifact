"""Short paired smoke proving that the Lyapunov shadow changes no action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Policies.TaskAssigner.WorldModelTaskAssigner.world_model_task_assigner import (
    WorldModelTaskAssigner,
)
from WorldModel.evaluation.diagnose_phase_c_lyapunov_shadow import (
    reproduce_audit,
)
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
from WorldModel.evaluation.phase_c_postcert_lyapunov import (
    LyapunovShadowWorldModelTaskAssigner,
    shadow_config_dict,
)


SCHEMA_VERSION = "phase_c_postcert_lyapunov_shadow_smoke_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lyapunov-config", required=True)
    parser.add_argument("--recorded-orders", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--top-m", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        "config": Path(args.config),
        "checkpoint": Path(args.checkpoint),
        "lyapunov_config": Path(args.lyapunov_config),
        "recorded_orders": Path(args.recorded_orders),
        "output": Path(args.output),
    }
    for name, path in paths.items():
        if name != "output" and not path.is_file():
            raise FileNotFoundError(path)
    if paths["output"].exists():
        raise FileExistsError(f"refusing to overwrite {paths['output']}")
    with paths["lyapunov_config"].open("r", encoding="utf-8") as handle:
        lyapunov_config = json.load(handle)

    baseline = WorldModelTaskAssigner(
        checkpoint_path=str(paths["checkpoint"]),
        top_m=int(args.top_m),
        include_no_assign_candidate=True,
        decision_trace_enabled=True,
        decision_trace_max_records=200000,
    )
    baseline_metrics = _run_one_assigner(
        str(paths["config"]),
        baseline,
        int(args.seed),
        int(args.ticks),
        trace_label="PhaseCWorldModelSmokeBaseline",
        recorded_orders_path=str(paths["recorded_orders"]),
    )

    shadow = LyapunovShadowWorldModelTaskAssigner(
        checkpoint_path=str(paths["checkpoint"]),
        top_m=int(args.top_m),
        decision_trace_max_records=200000,
        lyapunov_l0_config=lyapunov_config,
        shadow_horizons=(5, 10, 20, 50),
    )
    shadow_metrics = _run_one_assigner(
        str(paths["config"]),
        shadow,
        int(args.seed),
        int(args.ticks),
        trace_label="PhaseCWorldModelSmokeShadow",
        recorded_orders_path=str(paths["recorded_orders"]),
    )
    audit = reproduce_audit(
        reference_report={
            "arms": {"PhaseCWorldModel": baseline_metrics},
        },
        reference_trace=baseline.decision_trace_records,
        observed_metrics=shadow_metrics,
        observed_trace=shadow.decision_trace_records,
    )
    checks = {
        "paired_nonperturbation": bool(audit["passed"]),
        "nonempty_shadow": bool(shadow.lyapunov_shadow_records),
        "one_shadow_per_decision": (
            len(shadow.lyapunov_shadow_records)
            == len(shadow.decision_trace_records)
        ),
        "all_horizons_present": all(
            set(row["horizons"]) == {"5", "10", "20", "50"}
            for row in shadow.lyapunov_shadow_records
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "checks": checks,
        "passed": all(checks.values()),
        "reproduction_audit": audit,
        "baseline_metrics": baseline_metrics,
        "shadow_metrics": shadow_metrics,
        "shadow_config": shadow_config_dict(shadow),
        "shadow_records": len(shadow.lyapunov_shadow_records),
    }
    if not report["passed"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    with paths["output"].open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print("Phase-C Lyapunov shadow non-perturbation PASS")
    print("saved:", paths["output"])
    print("shadow records =", len(shadow.lyapunov_shadow_records))


if __name__ == "__main__":
    main()
