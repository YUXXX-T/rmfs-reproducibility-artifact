# Hardware record

The frozen result rows do not contain a complete machine identity (CPU model, GPU model, RAM, OS image, thread affinity, and background-load control). The original PP runner records whole-run wall time and selected assignment-call latency, but those fields alone are insufficient to reconstruct a hardware-normalized benchmark.

Accordingly, this artifact reports the observations and their variability without claiming portable latency. Before a non-anonymous archival release, add the original hardware record here if it can be recovered from scheduler logs; do not infer it from worker counts or package metadata.
