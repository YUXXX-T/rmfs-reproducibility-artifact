# Station-lock definition

Station-lock onset is an operational trace event, not a simulator state label. At each sampled interval, the detector differences cumulative admission attempts for each station, clips negative differences to zero, and computes:

```text
maximum station interval attempts / total interval attempts
```

The series is smoothed with a centered five-sample rolling mean. Main-figure onset is the first sample of the first three-sample run at or above 0.8. With a trace stride of 10 ticks, the smoothing span is approximately 50 ticks and the sustain requirement covers approximately 30 ticks.

The comparison asks whether dispatch policies differ in how quickly station demand becomes severely concentrated under the same offered load and physical-only admission contract. A 0.9 threshold is reported as sensitivity evidence.
