# Runtime evidence

The paper runtime evidence is organized around the dedicated six-station,
mid-load benchmark on held-out seeds 721–730:

- [Main runtime figure](../artifacts/figures/fig06_runtime_benchmark.png)
- [Supplementary wall-time figure](../artifacts/figures/fig06s_runtime_wall_time.png)
- [Complete interpretation](../docs/results/runtime_results.md)
- [Measurement protocol](../docs/experiments/runtime_protocol.md)
- [Hardware record](hardware.md)
- Figure inputs: `artifacts/raw/figure_inputs/fig06_runtime_benchmark_*`
- Reproduction: `python scripts/generate_fig06_runtime.py`

`raw_measurements.csv` and `summary.csv` are older broad campaign timing
extracts. They remain available for audit, but Fig. 6/6s uses the dedicated
synchronized benchmark described above. The original collection JSON is not
published because its audit section contains absolute worker paths and static
file fingerprints; the committed CSV/JSON extract retains every plotted and
reported numeric value.
