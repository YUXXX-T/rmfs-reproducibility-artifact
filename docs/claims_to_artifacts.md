# Claims to artifacts

| Paper claim | Evidence | Raw data | Reproduction command |
|---|---|---|---|
| Main PP high-load improvement over Greedy | Main table / Fig. 2 / [interactive view](results/explorer.md) | `artifacts/raw/main_per_seed.csv` | `make table-main` |
| Complete Proposed–JSQ and Proposed–WM-Base paired intervals | [Interactive forest plot](results/explorer.md) / paired comparison supplement | `artifacts/raw/main_per_seed.csv` | `make paired-ci` |
| Station lock temporally precedes collapse in detected paired events | Fig. 5 | `artifacts/raw/station_lock_events.csv` | `make fig05` |
| Run-level collapse uses both completed orders and mean deadlock ratio | Collapse summary | main per-seed rows + station events | `python scripts/reproduce_mechanism.py` |
| Adaptation transfers to a six-station layout on held-out seeds | Six-station table / [interactive view](results/explorer.md) | `artifacts/raw/station6_per_seed.csv` | `make station6` |
| Reported runtime observations come from the same frozen runs | Runtime table | `runtime/raw_measurements.csv` | `python scripts/reproduce_runtime.py` |
| Arrival streams are paired within every load/seed cell | Manifest audit | `manifests/main_50seed`, `manifests/station6_10seed` | `make verify` |

The intended chain is:

```text
paper claim → immutable per-seed evidence → deterministic analysis → generated figure/table
```

Arm-name mapping is explicit: `ComboS1J1` is the proposed method and `PhaseC` is the WM-Base control in the 50-seed artifact.
