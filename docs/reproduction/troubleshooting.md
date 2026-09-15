# Troubleshooting

## Top-level imports fail

Install editable mode (`python -m pip install -e .`) or set `PYTHONPATH=src`. The original `Engine/Policies/WorldModel` package names are intentional.

## Old NumPy/Pydantic pins fail

Use Python 3.10 for the full simulator. For statistics and docs only, use `requirements-artifact.txt`, which does not install PyTorch or historical training pins.

## Checkpoints are missing

Lightweight tables and Fig. 5 do not need them. Full simulation with WM arms does; consult `checkpoints/checkpoint_manifest.json`, place each release asset at its expected path, and run `python scripts/verify_checkpoints.py`.

## External MAPF solver executable is missing

The main PP and in-tree PIBT paths do not need EECBS/LNS2/LaCAM2. Their wrappers are retained but third-party source/build outputs are not vendored.

## Generated files differ

First run `python scripts/verify_artifacts.py`. Confirm Python version, bootstrap seed/resample count, event-data row count, and that the committed raw CSVs were not edited. PDF bytes can differ across Matplotlib versions; compare the event summary and rendered data semantics.
