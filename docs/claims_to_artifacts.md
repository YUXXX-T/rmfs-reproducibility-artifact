# Claims to artifacts

| Paper claim | Evidence | Raw data | Reproduction command |
|---|---|---|---|
| Main PP high-load improvement over Greedy | Main table / Fig. 2 / [static result overview](assets/results_overview.png) | `artifacts/raw/main_per_seed.csv` | `make table-main` |
| Complete Proposed–JSQ and Proposed–WM-Base paired intervals | [Static paired-effect plot](assets/paired_effects_overview.png) / paired comparison supplement | `artifacts/raw/main_per_seed.csv` | `make paired-ci` |
| Station lock temporally precedes collapse in detected paired events | Fig. 5 | `artifacts/raw/station_lock_events.csv` | `make fig05` |
| Run-level collapse uses both completed orders and mean deadlock ratio | Collapse summary | main per-seed rows + station events | `python scripts/reproduce_mechanism.py` |
| Adaptation transfers to a six-station layout on held-out seeds | Six-station table / [static result overview](assets/results_overview.png) | `artifacts/raw/station6_per_seed.csv` | `make station6` |
| Frozen policy transfers to larger maps and proportional fleets under four fixed stations | [Scale-transfer figures and paired analysis](results/density_scale_results.md) | `artifacts/raw/density_scale_per_seed.csv` | `make density-scale` |
| Proposed has higher dispatch cost than analytic baselines; device differences do not establish end-to-end speedup | [Runtime figures and three-load analysis](results/runtime_results.md) | `artifacts/raw/figure_inputs/station6_runtime_chart_summary.csv` | `make fig06` |
| Arrival streams are paired within every load/seed cell | Manifest audit | `manifests/main_50seed`, `manifests/station6_10seed`, `manifests/density_scale_10seed` | `make verify` |

The intended chain is:

```text
paper claim → immutable per-seed evidence → deterministic analysis → generated figure/table
```

Arm-name mapping is explicit: `ComboS1J1` is the proposed method and `PhaseC` is the WM-Base control in the 50-seed artifact.
