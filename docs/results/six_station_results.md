# Six-station results

[Open the interactive comparison](explorer.md) and choose “Six-station
adaptation” to explore every reported metric and paired interval.

Held-out means over seeds 721–730 are:

| Load | Greedy | Hungarian | WM-Base | ComboS1J1 |
|---|---:|---:|---:|---:|
| Low | 640.9 | 518.1 | 640.7 | 642.1 |
| Mid | 608.5 | 543.1 | 609.8 | 594.1 |
| High | 667.4 | 504.9 | 620.0 | 696.6 |

At high load, ComboS1J1 adds 29.2 completed orders relative to Greedy and 76.6 relative to WM-Base. Mean deadlock-ratio differences are −0.066697 and −0.102383, respectively. Mid load is not uniformly favorable: completion is 14.4 below Greedy and 15.7 below WM-Base.

The proper conclusion is therefore load-conditioned generalization after six-station adaptation, not uniform dominance and not zero-shot transfer. `make station6` regenerates the aggregate and paired tables from all 120 rows.
