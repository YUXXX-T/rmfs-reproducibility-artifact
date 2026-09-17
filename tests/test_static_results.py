from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_static_github_figures_are_committed_and_match_docs_copies() -> None:
    for name in (
        "results_overview.png",
        "paired_effects_overview.png",
        "density_scale_completed_orders_factorial.png",
        "density_scale_four_endpoint_summary.png",
        "fig06_runtime_benchmark.png",
        "fig06s_runtime_wall_time.png",
        "station6_runtime_assignment.png",
        "station6_runtime_wall.png",
    ):
        canonical = ROOT / "artifacts/figures" / name
        docs_copy = ROOT / "docs/assets" / name
        assert canonical.stat().st_size > 10_000
        assert canonical.read_bytes() == docs_copy.read_bytes()


def test_repository_pages_embed_static_figures() -> None:
    required = {
        "README.md": (
            "artifacts/figures/results_overview.png",
            "artifacts/figures/paired_effects_overview.png",
            "artifacts/figures/density_scale_four_endpoint_summary.png",
            "artifacts/figures/fig06_runtime_benchmark.png",
        ),
        "artifacts/tables/README.md": (
            "../figures/results_overview.png",
            "../figures/paired_effects_overview.png",
            "../figures/density_scale_completed_orders_factorial.png",
            "../figures/density_scale_four_endpoint_summary.png",
            "../figures/fig06_runtime_benchmark.png",
            "../figures/fig06s_runtime_wall_time.png",
            "../figures/station6_runtime_assignment.png",
            "../figures/station6_runtime_wall.png",
        ),
    }
    for relative, references in required.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert all(reference in text for reference in references)
