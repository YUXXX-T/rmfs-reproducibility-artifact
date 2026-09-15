# Artifact evidence

- `raw/`: flattened per-seed metrics, Fig. 5 event times, and compact figure inputs.
- `protocols/`: sanitized frozen protocol/summary JSON with portable paths.
- `statistics/`: paired intervals, collapse counts, station-lock summaries, and bootstrap settings.
- `tables/`: paper-ready aggregate CSV files plus a GitHub-rendered summary page.
- `figures/`: frozen paper figures and static result dashboards that render directly on GitHub.
- `generated/`: ignored workspace for deterministic reruns.

`artifact_manifest.json` indexes committed evidence by portable relative path and role. It deliberately does not record byte sizes because text line endings differ across Windows and Linux checkouts. The 750-row main CSV and 120-row six-station CSV retain scalar metrics but omit absolute paths, verbose station traces, and duplicated nested audit structures.
