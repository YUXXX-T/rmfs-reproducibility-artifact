# Runtime evidence

The paper runtime evidence is organized around the dedicated six-station
benchmark on held-out seeds 721–730 at low, mid, and high load:

- [Main runtime figure](../artifacts/figures/fig06_runtime_benchmark.png)
- [Supplementary wall-time figure](../artifacts/figures/fig06s_runtime_wall_time.png)
- [Cross-load assignment-cost figure](../artifacts/figures/station6_runtime_assignment.png)
- [Cross-load per-seed wall-time figure](../artifacts/figures/station6_runtime_wall.png)
- [Complete interpretation](../docs/results/runtime_results.md)
- [Measurement protocol](../docs/experiments/runtime_protocol.md)
- [Hardware record](hardware.md)
- Figure inputs: `artifacts/raw/figure_inputs/fig06_runtime_benchmark_*` and
  `station6_runtime_chart_summary.csv` plus the three load-specific run files
- Reproduction: `make fig06`

`raw_measurements.csv` and `summary.csv` are older broad campaign timing
extracts. They remain available for audit, but Fig. 6/6s uses the dedicated
synchronized benchmark described above. The original collection JSON is not
published because its audit section contains absolute worker paths and static
file fingerprints; the committed CSV/JSON extract retains every plotted and
reported numeric value.
