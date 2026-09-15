# Result tables

These CSV files are the machine-readable aggregate evidence. For visual
inspection, filtering, sorting, and paired-CI plots, open the
[interactive results explorer](../../docs/results/explorer.md). When the
documentation is deployed with GitHub Pages, the same page provides direct CSV
downloads and runs entirely in the browser.

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
statistical significance. Use the paired interval CSV files under
`artifacts/statistics/` or the explorer's paired-effects view for inference.

## Files

- `table_main.csv`: full four-station means, standard deviations, and counts.
- `table_baselines.csv`: the common comparison-metric subset.
- `table_station6.csv`: six-station means, standard deviations, and counts.

All tables are regenerated from per-seed evidence with `make reproduce`.
