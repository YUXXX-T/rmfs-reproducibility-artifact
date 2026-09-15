# Runtime results

Frozen runtime summaries are in `runtime/summary.csv`. Model arms have recorded mean assignment-call latency; baseline rows may lack that field. Whole-run wall time is available for all arms but has large between-seed variance because collapsed/healthy dynamics and machine scheduling change the amount of work.

For this reason, runtime is presented as observed experimental cost with sample count and standard deviation, not as a portable speedup claim. No missing baseline latency is replaced with zero. See [Runtime protocol](../experiments/runtime_protocol.md) for limitations.
