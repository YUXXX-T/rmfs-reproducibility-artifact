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


def test_runtime_figure_generation(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/generate_fig06_runtime.py", "--output-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
    )
    for name in (
        "fig06_runtime_benchmark.pdf",
        "fig06_runtime_benchmark.png",
        "fig06s_runtime_wall_time.pdf",
        "fig06s_runtime_wall_time.png",
    ):
        assert (tmp_path / name).stat().st_size > 1000


def test_crossload_runtime_figure_generation(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_station6_runtime_crossload.py",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
    )
    for name in (
        "station6_runtime_assignment.pdf",
        "station6_runtime_assignment.png",
        "station6_runtime_wall.pdf",
        "station6_runtime_wall.png",
    ):
        assert (tmp_path / name).stat().st_size > 1000
