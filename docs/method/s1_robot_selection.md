# S1 robot selection

S1 operates **within one fixed context**. It begins with the nearest `top_m=10` idle robots and applies the world-model consequence score to select one candidate. Its reported long-risk signal is:

```text
0.2 × peak_q95 + 0.3 × cvar_q90 + 0.5 × terminal_q90
```

The conversion gate is deterministic, uses a 200-sample window with 50-sample warm-up and the 0.8 quantile, and has conversion coefficient 0.25. The active energy channels weight station, bottleneck, and severe-congestion terms by 1, 1, and 2. `NO_ASSIGN`, station injection, and random flips are disabled.

S1 scores are context-local. They are not valid cross-context priorities; J1 owns that decision scope.
