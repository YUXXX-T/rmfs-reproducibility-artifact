# Artifact evidence

- `raw/`: flattened per-seed metrics for the main, six-station, and
  map/fleet-scale campaigns, Fig. 5 event times, and compact figure inputs.
- `protocols/`: sanitized frozen protocol/summary JSON with portable paths.
- `statistics/`: paired intervals, collapse counts, station-lock summaries, and bootstrap settings.
- `tables/`: paper-ready aggregate CSV files plus a GitHub-rendered summary page.
- `figures/`: frozen paper figures and static result dashboards that render directly on GitHub.
- `generated/`: ignored workspace for deterministic reruns.

`artifact_manifest.json` indexes committed evidence by portable relative path
and role. It deliberately does not record byte sizes because text line endings
differ across Windows and Linux checkouts. The 750-row main CSV, 120-row
six-station CSV, and 1,350-row scale-transfer CSV retain scalar metrics but
omit absolute paths, wall-time observations, protocol fingerprints, verbose
station traces, and duplicated nested audit structures.

The Fig. 6 runtime inputs are the exception to the general omission of timing
fields: they come from a dedicated synchronized benchmark and are stored under
`raw/figure_inputs/`. Machine paths and file fingerprints from its collection
audit are not retained.
