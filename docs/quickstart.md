# Quick start

## Lightweight evidence reproduction

Use Python 3.10 for the reference environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-artifact.txt
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py --figure fig05
python scripts/verify_artifacts.py
python -m pytest
```

The scripts read committed evidence and write only to `artifacts/generated/`. Use `python scripts/compare_generated.py` to compare generated tabular and statistical content with the committed paper artifacts.

## Documentation

```bash
python -m mkdocs build --strict
python -m mkdocs serve
```

## Full simulator

The simulator dependency set includes PyTorch and the historical package pins:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/smoke_simulation.py
```

The full 750-run campaign and model training require separately distributed checkpoints; see [Full experiment](reproduction/full_experiment.md).
