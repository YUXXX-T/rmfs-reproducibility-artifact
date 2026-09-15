# Seed splits

No reported evaluation seed is selected by outcome.

| Purpose | Train | Validation | Offline test | Online test |
|---|---|---|---|---|
| World-model long-risk labels | 681–687 | 688–689 | 690 | 691–700 |
| J1 station predictor | 511–516 | 517–518 | 519–520 | separate online campaigns |
| PIBT adaptation | 461–467 | 468–469 | 470 | 551–560 |
| Six-station core | 711–717 | 718–719 | 720 | 721–730 |
| Six-station long-risk labels/J1 predictor | 731–737 | 738–739 | 740 | 721–730 |
| Main PP paper evaluation | — | — | — | 900–949 |

The six-station held-out block 721–730 is disjoint from both 711–720 and 731–740. Machine-readable partitions are in `splits/`.
