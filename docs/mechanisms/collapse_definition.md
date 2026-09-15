# Collapse definition

A run is labeled collapsed only when both endpoint conditions hold:

```text
completed_orders < 300
AND
deadlock_ratio_mean >= 0.4
```

This is the definition used for the paper’s high-load collapse counts: 22/50 for Greedy and 15/50 for ComboS1J1. Collapse is therefore **not** determined by throughput alone.

After this run-level label is fixed, an onset tick is localized inside labeled runs from an independent order-side trace: smoothed completion rate falls below 25% of its first-third median while pending + in-progress backlog is at least 1.5 times its early median (and at least one), sustained for three samples. A low-rate episode in a run that fails the endpoint conjunction is retained as an all-run diagnostic but does not enter the paper-aligned collapse curve or lead distribution.
