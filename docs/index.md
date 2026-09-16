# RMFS world-model artifact

This site is the detailed companion to the anonymous paper **“Congestion-Robust RMFS Dispatch via a Horizon-Separated Action-Conditioned World Model.”** It is organized as an evidence graph, not as a second paper.

Use [Claims to artifacts](claims_to_artifacts.md) to move directly from a paper statement to its frozen evidence and reproduction command. The [training pipeline](reproduction/training_pipeline.md) documents data, optimization stages, checkpoint provenance, and seed separation beyond the brief commands in [Quick start](quickstart.md). The result figures below render directly in repository and anonymous-mirror views.

![Aggregate results across load and station layouts](assets/results_overview.png)

The main PP evaluation fixes 50 seeds (900–949), three offered-load definitions, five dispatch arms, 1,500 ticks, and exact per-cell arrival-manifest replay. The six-station experiment is a seed-disjoint adaptation study, not a zero-shot claim. The mechanism analysis uses the paper’s joint completed-orders/deadlock endpoint definition before localizing a collapse time from an independent completion-rate-plus-backlog trace.

The fixed-four-station scale-transfer campaign adds 1,350 runs over three map
sizes and three proportional robot densities, with the model and dispatch
parameters frozen. Its [static figures and paired results](results/density_scale_results.md)
are also directly visible in repository mirrors.

!!! note "Double-blind snapshot"
    Author identities, personal paths, repository-owner URLs, and large binary assets are intentionally absent. The anonymous citation metadata should be replaced after review.
