# Artifact evidence

- `raw/`: flattened per-seed metrics, Fig. 5 event times, and compact figure inputs.
- `protocols/`: sanitized frozen protocol/summary JSON with portable paths.
- `statistics/`: paired intervals, collapse counts, station-lock summaries, and bootstrap settings.
- `tables/`: paper-ready aggregate CSV files plus a rendered GitHub summary and a link to the interactive explorer.
- `figures/`: frozen vector/raster paper figures.
- `generated/`: ignored workspace for deterministic reruns.

`artifact_manifest.json` indexes committed evidence by path, role, and byte size. The 750-row main CSV and 120-row six-station CSV retain scalar metrics but omit absolute paths, verbose station traces, and duplicated nested audit structures.
