# Paired comparisons

[Open the interactive paired-effects view](explorer.md) to switch between
metrics and baselines and inspect the bootstrap confidence intervals against
the zero-effect line.

The proposed arm is compared with JSQ and WM-Base on every load and every common scalar metric. The CSV stores the raw arm-minus-baseline mean, percentile interval, favorable direction, paired count, and wins/losses/ties.

At high load, Proposed–JSQ completed-order difference is +15.68 with 95% CI [−28.62, 58.62]. Proposed–WM-Base is +31.14 [−17.34, 79.34]. These intervals cross zero; they must not be summarized as confirmed pairwise advantages. The stronger confirmed high-load comparison in this block is versus Greedy: +65.06 [15.76, 114.86].

See `artifacts/statistics/paired_confidence_intervals.csv` and run `make paired-ci`. Reporting the complete table guards against selecting only favorable metrics or loads.
