.PHONY: help setup test reproduce table-main paired-ci station6 fig05 map-previews result-plots verify docs serve smoke-sim

PYTHON ?= python

help:
	@echo "setup       Install statistics/docs dependencies"
	@echo "test        Run artifact contract tests"
	@echo "reproduce   Regenerate all lightweight tables and figures"
	@echo "table-main  Regenerate the main aggregate table"
	@echo "paired-ci   Regenerate Proposed-vs-JSQ/WM-Base paired CIs"
	@echo "station6    Regenerate the six-station table and paired summary"
	@echo "fig05       Regenerate aggregate temporal Fig. 5"
	@echo "map-previews Regenerate four- and six-station layout previews"
	@echo "result-plots Regenerate static GitHub result figures"
	@echo "verify      Check schemas, file indexes, counts, and anonymity"
	@echo "docs        Build the MkDocs site"
	@echo "serve       Serve documentation locally"
	@echo "smoke-sim   Run a short real-simulator baseline job (full dependencies)"

setup:
	$(PYTHON) -m pip install -r requirements-artifact.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest
	$(PYTHON) -m compileall -q src scripts tests

reproduce:
	$(PYTHON) scripts/reproduce_tables.py
	$(PYTHON) scripts/reproduce_figures.py --figure all
	$(PYTHON) scripts/generate_results_overview.py

table-main:
	$(PYTHON) scripts/reproduce_tables.py --only main

paired-ci:
	$(PYTHON) scripts/reproduce_tables.py --only paired-ci

station6:
	$(PYTHON) scripts/reproduce_tables.py --only station6

fig05:
	$(PYTHON) scripts/reproduce_figures.py --figure fig05

map-previews:
	$(PYTHON) scripts/generate_map_previews.py

result-plots:
	$(PYTHON) scripts/generate_results_overview.py

verify:
	$(PYTHON) scripts/verify_artifacts.py

docs:
	$(PYTHON) -m mkdocs build --strict

serve:
	$(PYTHON) -m mkdocs serve

smoke-sim:
	$(PYTHON) scripts/smoke_simulation.py
