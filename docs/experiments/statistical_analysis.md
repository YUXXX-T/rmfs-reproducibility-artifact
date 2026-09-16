# Statistical analysis

For a metric `m`, load `l`, proposed arm `A`, baseline `B`, and paired seed `s`, the analysis first computes:

```text
d_s = m(A, l, s) - m(B, l, s)
```

It then resamples the 50 seed-level deltas with replacement, keeping the pairing intact, and reports the mean delta with a percentile 95% interval. The main protocol uses 10,000 bootstrap resamples, deterministic seed 20260905, and order-statistic indices `floor(.025 × (B−1))` and `ceil(.975 × (B−1))`.

All deltas retain the literal `arm minus baseline` sign. The `direction` column states whether higher or lower is favorable, preventing sign-flipping ambiguity. Wins/losses/ties use the favorable direction but do not replace interval estimates.

The complete Proposed–JSQ and Proposed–WM-Base outputs, not only selected significant endpoints, are in `artifacts/statistics/paired_confidence_intervals.csv`.

## Scale-transfer campaign

The map/fleet scale analysis uses 30 load–seed clusters. Within each map size,
the three density-specific differences are averaged before resampling; the
overall scope averages all nine map–density differences. Bootstrap resampling
is stratified by load so low, mid, and high remain equally weighted. It uses
50,000 draws and deterministic seed 20260916. These intervals therefore
quantify paired transfer effects across arrival realizations rather than
treating the three densities as independent replications.
