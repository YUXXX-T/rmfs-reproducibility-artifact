"""Run ``WorldModel.generate_data`` with FIFO-V2 station admission enabled.

This additive wrapper leaves the historical generator's default semantics
unchanged.  It temporarily replaces the imported SimulationEngine class so
the station admission mode is selected immediately after engine construction,
then writes a per-run invariant audit beside the generated dataset.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from WorldModel.evaluation.run_phase_c_psi_dynamic_admission_wait import (
    FifoWaitingAuditProbe,
)
from WorldState.station_state import STATION_ADMISSION_COMMITTED_FIFO_V2


AUDIT_FILENAME = "fifo_admission_audit.json"


def _argument_value(name: str) -> str | None:
    for index, value in enumerate(sys.argv[1:]):
        if value == name and index + 2 <= len(sys.argv[1:]):
            return sys.argv[index + 2]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix):]
    return None


def main() -> None:
    output_value = _argument_value("--output-dir")
    if not output_value:
        raise SystemExit("FIFO-V2 wrapper requires an explicit --output-dir")
    output_dir = Path(output_value)
    trace_max = max(0, int(os.environ.get("FIFO_TRACE_MAX_RECORDS", "0")))

    import Engine.simulation_engine as engine_module
    from WorldModel import generate_data as generator

    original_engine = engine_module.SimulationEngine
    holder: dict[str, object] = {}

    class FifoV2SimulationEngine(original_engine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.world.station_state.set_admission_mode(
                STATION_ADMISSION_COMMITTED_FIFO_V2
            )
            probe = FifoWaitingAuditProbe(
                self, trace_max_records=trace_max
            )
            self.on_tick_callbacks.append(probe.on_tick)
            holder["engine"] = self
            holder["probe"] = probe

    engine_module.SimulationEngine = FifoV2SimulationEngine
    try:
        generator.main()
    finally:
        engine_module.SimulationEngine = original_engine

    probe = holder.get("probe")
    engine = holder.get("engine")
    if not isinstance(probe, FifoWaitingAuditProbe) or engine is None:
        raise RuntimeError("FIFO-V2 wrapper did not observe a generated engine")
    summary = probe.summary()
    summary.update({
        "schema_version": "context_j_fifo_v2_collection_audit_v1",
        "num_agents": int(len(engine.world.agents)),
        "ticks": int(engine.world.tick),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / AUDIT_FILENAME
    audit_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not bool(summary.get("passed")):
        raise RuntimeError(f"FIFO-V2 collection invariant failed: {audit_path}")
    print(f"[audit] FIFO-V2 collection passed: {audit_path}")


if __name__ == "__main__":
    main()
