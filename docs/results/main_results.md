# Main PP results

[Open the interactive comparison](explorer.md) to change the load and metric,
inspect mean ± SD graphically, sort the displayed values, or download the
current view.

Frozen means over seeds 900–949 are:

| Load | Greedy | Hungarian | JSQ | WM-Base | ComboS1J1 |
|---|---:|---:|---:|---:|---:|
| Low | 356.42 | 341.14 | 350.84 | 345.14 | 380.56 |
| Mid | 358.30 | 325.36 | 337.52 | 356.90 | 361.56 |
| High | 318.36 | 290.44 | 367.74 | 352.28 | 383.42 |

Values are completed orders after 1,500 ticks. Relative to Greedy, ComboS1J1 changes high-load completion by +65.06 orders with paired 95% CI [15.76, 114.86], while mean deadlock ratio changes by −0.128522 with CI [−0.223687, −0.035251]. Low/mid effects are reported rather than hidden: completion deltas are +24.14 [−9.46, 56.20] and +3.26 [−43.28, 48.14].

These frozen values are generated from `artifacts/raw/main_per_seed.csv`. The full metric table includes delay, backlog, congestion events, risk rate, completion fraction, and timing fields.
