# Interactive results explorer

CSV files remain the canonical, machine-readable evidence. This page adds a
review layer over the same frozen rows: select an experiment, load, metric, and
reference policy; inspect the chart; then audit or download the filtered table.
No values are recomputed in the browser.

<div
  id="rmfs-results-explorer"
  class="rmfs-explorer"
  data-source="../../assets/data/results_explorer.json"
>
  <div class="rmfs-loading" role="status">Loading frozen result tables…</div>
</div>

<noscript>
JavaScript is required for the interactive controls. The key completed-order
tables are rendered below, and every source CSV remains directly downloadable.
</noscript>

## Quick reading

| Evaluation | Load | Greedy | JSQ | WM-Base | ComboS1J1 |
|---|---|---:|---:|---:|---:|
| Four-station PP, 50 seeds | High | 318.36 | 367.74 | 352.28 | **383.42** |
| Six-station adaptation, 10 seeds | High | 667.4 | — | 620.0 | **696.6** |

Values are mean completed orders after 1,500 ticks. The explorer exposes the
other throughput, delay, backlog, congestion, risk, and runtime metrics without
requiring readers to manipulate CSV files manually.

## Direct data downloads

<div class="rmfs-download-grid">
  <a class="rmfs-download-card" href="../../assets/data/table_main.csv" download>
    <strong>Four-station aggregates</strong>
    <span>15 policy–load rows · mean, SD, and n</span>
    <code>table_main.csv</code>
  </a>
  <a class="rmfs-download-card" href="../../assets/data/paired_confidence_intervals.csv" download>
    <strong>Four-station paired effects</strong>
    <span>Proposed–JSQ and Proposed–WM-Base intervals</span>
    <code>paired_confidence_intervals.csv</code>
  </a>
  <a class="rmfs-download-card" href="../../assets/data/table_station6.csv" download>
    <strong>Six-station aggregates</strong>
    <span>12 policy–load rows · mean, SD, and n</span>
    <code>table_station6.csv</code>
  </a>
  <a class="rmfs-download-card" href="../../assets/data/station6_paired_confidence_intervals.csv" download>
    <strong>Six-station paired effects</strong>
    <span>Proposed–Greedy and Proposed–WM-Base intervals</span>
    <code>station6_paired_confidence_intervals.csv</code>
  </a>
</div>

!!! note "How to read the uncertainty"

    The aggregate view shows the mean and standard deviation across seeds; it
    is descriptive and is not a confidence interval. The paired-effects view
    shows the paper's paired bootstrap 95% confidence intervals. Effects are
    always reported as `ComboS1J1 − baseline`; negative effects are favorable
    for metrics marked “lower is better.”

