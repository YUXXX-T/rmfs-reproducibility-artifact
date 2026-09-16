# Result tables

These figures and tables render directly on the GitHub file page. No local
download, notebook, or JavaScript is needed.

![Aggregate policy outcomes](../figures/results_overview.png)

![Paired bootstrap effects](../figures/paired_effects_overview.png)

## Main four-station evaluation

Mean completed orders over 50 paired seeds, after 1,500 simulation ticks:

| Load | Greedy | Hungarian | JSQ | WM-Base | ComboS1J1 |
|---|---:|---:|---:|---:|---:|
| Low | 356.42 | 341.14 | 350.84 | 345.14 | **380.56** |
| Mid | 358.30 | 325.36 | 337.52 | 356.90 | **361.56** |
| High | 318.36 | 290.44 | 367.74 | 352.28 | **383.42** |

## Six-station adaptation

Mean completed orders over 10 held-out seeds:

| Load | Greedy | Hungarian | WM-Base | ComboS1J1 |
|---|---:|---:|---:|---:|
| Low | 640.9 | 518.1 | 640.7 | **642.1** |
| Mid | 608.5 | 543.1 | **609.8** | 594.1 |
| High | 667.4 | 504.9 | 620.0 | **696.6** |

## Fixed-four-station map/fleet scale transfer

The frozen model and policy are evaluated without retraining across three map
sizes, three proportional robot densities, three loads, and ten seeds.

![Four-endpoint scale-transfer summary](../figures/density_scale_four_endpoint_summary.png)

| Map | Greedy | Hungarian | JSQ | WM-Base | Proposed |
|---:|---:|---:|---:|---:|---:|
| 20×20 | 337.1 | 351.0 | 341.0 | 353.4 | 339.8 |
| 30×30 | 285.1 | 276.4 | 289.0 | 307.7 | **322.3** |
| 40×40 | 299.9 | 283.7 | 291.0 | 343.2 | **361.8** |

Values pool three densities × three loads × ten seeds (n=90 per policy and
map). The figure bands use stratified load–seed cluster-bootstrap 95%
intervals. Proposed − JSQ is +34.31 completed orders overall, with 95% CI
[20.12, 49.03].

![All 27 scale-transfer cells](../figures/density_scale_completed_orders_factorial.png)

The exact summaries are in `table_density_scale.csv` and
`table_density_scale_cells.csv`; paired effects and Pareto membership are in
the adjacent `statistics/` directory.

Bold marks the largest mean in each row and is descriptive, not a claim of
statistical significance. The paired-effect figure above reports the paired
bootstrap 95% confidence intervals used for inference.

All tables are regenerated from per-seed evidence with `make reproduce`.
The static summaries are regenerated with `make result-plots`; maintainers use
`python scripts/generate_results_overview.py --freeze` only when intentionally
refreshing the committed copies.
