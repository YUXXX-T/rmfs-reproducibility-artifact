# Map-size and fleet-density generalization

This campaign tests whether the frozen dispatcher transfers across graph size
and robot density while retaining the same four-station service interface. It
is a **zero-shot scale-transfer** experiment, not a station-count experiment.

The full factorial contains three square maps (20×20, 30×30, and 40×40),
three robot densities (0.12, 0.15, and 0.18), three offered-load regimes, ten
seeds (701–710), and five dispatch policies. The corresponding robot counts
are 48/60/72, 108/135/162, and 192/240/288. Every run uses the PP motion
planner, four stations, `physical_only_legacy` station admission, and 1,500
simulation ticks. This gives 3 × 3 × 3 × 10 × 5 = **1,350 completed runs**.

The 27 exact layout/load JSON configurations are in
`configs/evaluation/density_scale_variants/`; the 30 portable order streams
are in `manifests/density_scale_10seed/`. The JSON configuration's default
`simulation.max_ticks` is overridden to 1,500 by the campaign evaluator;
likewise its default random seed and order generator are replaced at runtime
with the selected evaluation seed and the paired recorded-order stream. The
committed stream files omit the source runner's embedded validation digest,
in keeping with this double-blind artifact's no-static-fingerprint policy;
they expose the complete order events for inspection and analysis. A full
runner rerun can regenerate its own validated manifests in a scratch area.

The world model, long-risk output, and J1 predictor remain frozen. S1 and J1
are not retuned for the new map or fleet sizes. The same arrival realization
is reused across every policy and map–density variant within each load–seed
cell. Source-side audits of layout connectivity, robot/pod reachability,
initialization, run completeness, and cross-variant manifest pairing all
passed before the compact artifact table was exported.

For a map-size summary, each load–seed cluster first averages the three robot
densities. A stratified bootstrap resamples the ten clusters separately within
low, mid, and high load, preserving equal weight for the three load regimes.
The paired policy analysis applies the same procedure to within-variant policy
differences. It uses 50,000 resamples and deterministic seed 20260916. The
machine-readable contract is
`configs/evaluation/density_scale_10seed.yaml`.

The scope is deliberately bounded: these runs support transfer across map
size and proportional fleet size under a fixed station interface. They do not
show zero-shot transfer to a different station count, and their incidental
wall times are not used as controlled runtime measurements.
