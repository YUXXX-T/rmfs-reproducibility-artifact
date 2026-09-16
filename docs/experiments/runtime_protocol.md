# Runtime protocol

## Dedicated dispatch benchmark

Fig. 6 uses a dedicated benchmark rather than timing fields opportunistically
collected during the main campaign.

| Item | Fixed setting |
|---|---|
| Environment | six-station 20×20 adaptation, 48 robots, PP planner |
| Workload | mid load |
| Seeds | 721–730 (ten held-out seeds) |
| Pairing | identical recorded arrival manifest within each seed |
| Horizon | 1,500 simulation ticks per run |
| Configurations | Greedy, JSQ, Hungarian, Proposed CPU, Proposed GPU |
| Execution | sequential and interleaved by seed |
| Assignment measurement | synchronized wall time around every `task_assigner.assign()` call |
| CUDA timing | `torch.cuda.synchronize()` immediately before and after the measured call |
| Initialization | recorded separately and excluded from steady-state assignment latency |
| Warm-up | 30 ticks for each method/device before measurement |
| Calls | 1,500 per seed; 15,000 pooled per configuration |
| Candidate setting | proposed top-M = 10 |

For each run, the benchmark records mean, median, p95, maximum, and total
steady-state assignment time. It also records initialization time, whole-run
wall time, completed orders, simulated ticks per second, and—where
applicable—model-inference counts and the policy's internal inference timer.

The primary runtime endpoint is the externally measured, synchronized
`assign()` latency. Internal policy timing is used only to decompose that
latency. In particular, `model_inference_time_ms_mean` is total inference time
divided by the number of inference calls, not by the number of assignment
calls. The inference share is therefore calculated as:

```text
(mean time per inference × number of inference calls)
------------------------------------------------------
       total externally measured assign() time
```

The CPU/GPU analysis pairs the ten observations by seed and applies a
two-sided Wilcoxon signed-rank test. The latency-driver panel fits ordinary
least squares across the ten CPU observations, with inference calls per tick
as the predictor and mean `assign()` latency as the outcome.

## Whole-run timing

Episode wall time includes the simulator, task assignment, path planning,
model inference, Python/runtime overhead, and operating-system scheduling.
The number and difficulty of decisions also change with run dynamics. For
that reason, Fig. 6s is descriptive and the paper does not present its values
as portable algorithmic speedups.

The 300-order reference in Fig. 6s identifies a low-throughput region only.
It is not the formal paper collapse label, which additionally requires mean
deadlock ratio of at least 0.4.

## Frozen data boundary

The anonymous artifact retains all per-run and pooled values required to
recompute Fig. 6/6s. Absolute worker paths, checkpoint/manifest fingerprints,
and the worker hostname from the collection JSON are excluded. Hardware and
software fields relevant to interpretation are recorded separately in
`runtime/hardware.md`.

`runtime/raw_measurements.csv` and `runtime/summary.csv` are the broader timing
extracts from the main and adaptation campaigns. They are retained for audit,
but they do not replace the controlled benchmark above and are not the source
of Fig. 6.
