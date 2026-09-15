# Runtime protocol

The frozen job outputs contain whole-run wall time for every arm and mean assignment-call latency where the evaluator recorded it. `runtime/raw_measurements.csv` preserves these observations; `runtime/summary.csv` reports arithmetic means, sample standard deviations, and non-missing counts by layout/load/arm.

Wall time includes simulation, path planning, Python overhead, model inference, process scheduling, and machine load. It is useful for reproducing the campaign’s operational cost but is not a hardware-independent algorithmic benchmark. Baseline assignment latency is missing in some historical rows and is never imputed as zero.

The frozen JSON did not encode a complete CPU/GPU model, operating system image, thread affinity, and background-load record. `runtime/hardware.md` marks that limitation explicitly instead of inventing hardware metadata.
