from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tick_operation_order_matches_documented_contract() -> None:
    source = (ROOT / "src/Engine/simulation_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    tick = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_tick"
    )
    lines = ast.get_source_segment(source, tick)
    assert lines is not None
    markers = [
        "self.order_generator.generate(self.world)",
        "self._refill_order_backlog(tick)",
        "self.task_assigner.assign(self.world)",
        "self.world.station_state.tick(self.world)",
        "self._plan_and_activate(tick)",
        "self._move_agents(tick)",
        "self._detect_conflicts(tick, prev_positions)",
        "self._handle_actions(tick)",
        "self._check_order_completion(tick)",
        "self.metrics.record(self.world, vertex_conflicts, assign_ms, plan_ms)",
        "self.world.advance_tick()",
    ]
    positions = [lines.index(marker) for marker in markers]
    assert positions == sorted(positions)


def test_tick_document_points_to_canonical_engine() -> None:
    text = (ROOT / "docs/simulator/tick_semantics.md").read_text(encoding="utf-8")
    assert "src/Engine/simulation_engine.py" in text
    assert "increment `world.tick`" in text
