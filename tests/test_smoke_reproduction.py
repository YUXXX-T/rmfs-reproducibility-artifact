from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lightweight_reproduction_matches_committed_tables() -> None:
    subprocess.run([sys.executable, "scripts/reproduce_tables.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/compare_generated.py"], cwd=ROOT, check=True)


def test_mechanism_validation_and_figure_generation(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/reproduce_mechanism.py", "--check"], cwd=ROOT, check=True
    )
    subprocess.run(
        [sys.executable, "scripts/generate_fig05_mechanism.py", "--output-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
    )
    assert (tmp_path / "fig05_station_lock_mechanism.pdf").stat().st_size > 1000
    assert (tmp_path / "fig05_station_lock_mechanism.png").stat().st_size > 1000
    assert (tmp_path / "fig05_station_lock_mechanism.summary.json").is_file()
