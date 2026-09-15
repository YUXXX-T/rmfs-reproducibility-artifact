#!/usr/bin/env python3
"""Render compact four- and six-station layout previews from committed JSON."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "pod": "#DCE6EA",
    "service": "#0F4D92",
    "queue": "#7FA6C9",
    "buffer": "#D9A441",
    "entry": "#4B8F8C",
    "exit": "#B65C5C",
}


def _cell(axis, row: int, col: int, color: str, *, edge: str = "white") -> None:
    axis.add_patch(
        Rectangle((col, row), 1, 1, facecolor=color, edgecolor=edge, linewidth=0.55)
    )


def render(path: Path, output: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    layout = payload["map"]
    rows, cols = int(layout["rows"]), int(layout["cols"])
    figure, axis = plt.subplots(figsize=(4.15, 4.15))
    axis.set_facecolor("#FAFBFC")

    for zone in layout.get("pod_zones", []):
        row0 = int(zone["origin_row"])
        col0 = int(zone["origin_col"])
        for row in range(row0, row0 + int(zone["num_rows"])):
            for col in range(col0, col0 + int(zone["num_cols"])):
                _cell(axis, row, col, COLORS["pod"])

    for station in layout["stations"]:
        queue = station["queue"]
        for row, col in queue.get("queue", []):
            _cell(axis, row, col, COLORS["queue"])
        for row, col in queue.get("buffer", []):
            _cell(axis, row, col, COLORS["buffer"])
        for role in ("entry", "exit", "service"):
            row, col = queue[role]
            _cell(axis, row, col, COLORS[role])
        service_row, service_col = queue["service"]
        axis.text(
            service_col + 0.5,
            service_row + 0.52,
            str(station["id"]),
            ha="center",
            va="center",
            color="white",
            fontsize=7.2,
            weight="bold",
        )

    axis.set(xlim=(0, cols), ylim=(rows, 0), aspect="equal")
    axis.set_xticks(range(0, cols + 1, 5))
    axis.set_yticks(range(0, rows + 1, 5))
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    axis.set_title(
        f"{len(layout['stations'])}-station layout | {rows}x{cols}, "
        f"{payload['robots']['num_robots']} robots",
        fontsize=10,
        weight="bold",
        pad=8,
    )
    axis.grid(color="#D7DEE3", linewidth=0.45, alpha=0.85)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#75818A")
        spine.set_linewidth(0.7)
    legend = [
        Patch(facecolor=COLORS[key], label=label)
        for key, label in (
            ("pod", "Pod storage"),
            ("service", "Service"),
            ("queue", "Queue"),
            ("buffer", "Buffer"),
            ("entry", "Entry"),
            ("exit", "Exit"),
        )
    ]
    axis.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=7.2,
        columnspacing=1.1,
        handlelength=1.4,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def main() -> None:
    for suffix in ("s4", "s6"):
        source = ROOT / "maps" / f"map20_{suffix}.json"
        output = ROOT / "maps" / f"map20_{suffix}.png"
        render(source, output)
        docs_output = ROOT / "docs/assets" / output.name
        docs_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output, docs_output)
        print(f"Wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
