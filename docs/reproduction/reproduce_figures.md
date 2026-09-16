# Reproduce figures

Run:

```bash
python scripts/reproduce_figures.py --figure fig05
```

The script reads `artifacts/raw/station_lock_events.csv`, validates the fixed event counts and medians, and writes PDF/PNG to `artifacts/generated/figures`. It does not require the multi-gigabyte trace bundle because the committed event table is the audited detector output.

To re-detect events from per-tick traces, download the external trace bundle and run `src/WorldModel/evaluation/analyze_lock_precedence.py`; then compare the resulting event table to the committed one. Other committed paper figures are retained as frozen outputs because their underlying compact data is included, while Fig. 5 is the primary scripted aggregate mechanism reproduction.

Regenerate the runtime main and supplementary figures with:

```bash
python scripts/reproduce_figures.py --figure fig06
```

This reads the anonymous 50-run and pooled inputs under
`artifacts/raw/figure_inputs/`, recomputes the paired Wilcoxon tests and
latency-driver regressions, validates them against the frozen statistics, and
writes Fig. 6/6s to `artifacts/generated/fig06`. It does not load a checkpoint
or import model-side code.

Use `--figure all` to regenerate Figs. 5, 6, and 6s together.

The fixed-four-station scale-transfer figures are independently regenerated
from compact run-level evidence with:

```bash
python scripts/reproduce_density_scale.py
```

This produces both the 3×3 completed-order factorial and the four-endpoint
map-size summary without checkpoints or per-tick traces.
