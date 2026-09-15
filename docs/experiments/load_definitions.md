# Load definitions

| Parameter | Low | Mid | High |
|---|---:|---:|---:|
| Scheduled order interval (ticks) | 5 | 4 | 3 |
| Initial pending pool | 4 | 8 | 12 |
| Pre-assignment pending floor | 2 | 4 | 6 |
| Distinct SKUs per order | 2 | 2 | 2 |
| Quantity per selected SKU | 1–3 | 1–3 | 1–3 |
| Station service duration (ticks) | 5 | 5 | 5 |

All loads use Zipf exponent 1.5, parallel task execution, 48 robots, pickup/drop-off durations of two ticks, home returns, and 1,500 evaluation ticks. The same load parameters are used on the four- and six-station layouts.

The exact source JSON is under `configs/simulator`; `configs/workloads` extracts the load-changing fields. Initial pool and refill floor mean that the order interval alone is not a complete load definition.
