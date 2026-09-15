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

Bold marks the largest mean in each row and is descriptive, not a claim of
statistical significance. The paired-effect figure above reports the paired
bootstrap 95% confidence intervals used for inference.

All tables are regenerated from per-seed evidence with `make reproduce`.
The static summaries are regenerated with `make result-plots`; maintainers use
`python scripts/generate_results_overview.py --freeze` only when intentionally
refreshing the committed copies.
