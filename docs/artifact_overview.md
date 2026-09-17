# Artifact overview

## Included

- First-party simulator, policy, planner, world-model, training, and evaluation Python source under `src/`.
- Exact four- and six-station simulator JSON configurations, 27 map/fleet-scale
  layout/load JSON configurations, and concise YAML contracts.
- All 150 main-evaluation paired arrival manifests, 30 six-station held-out
  manifests, and 30 map/fleet-scale manifests reused across all nine variants.
- Flattened per-seed result rows for the main, six-station, and 1,350-run
  fixed-four-station scale-transfer campaigns, with identity-bearing path and
  fingerprint fields removed.
- Paired bootstrap outputs, station-lock events, collapse summaries, the
  dedicated 150-run, three-load CPU/GPU runtime extract, paper tables, and
  figures.
- Reproduction, validation, CI, and documentation infrastructure.

## Distributed separately

- Neural-network checkpoints (`.pt`, `.pth`, `.ckpt`).
- Training tensors, snapshots, and multi-gigabyte per-tick trace bundles.
- Optional third-party EECBS, MAPF-LNS2, and LaCAM2 source/build trees.

`checkpoints/checkpoint_manifest.json` records expected release assets and
paths. The 210 arrival files and their schema/order-count index make the
paired offered load directly inspectable. The statistics-only path is complete
without large files; full retraining and trace re-detection require the
external bundle.

## Source-layout decision

The evaluated code imports top-level packages such as `Engine`, `Policies`, `WorldModel`, and `WorldState`. They are therefore kept under `src/` with the same names. This preserves code behavior while still providing an installable source layout. External-solver wrappers are present, but their vendored upstream repositories are intentionally absent.

Review sanitization removes identity-bearing result fields, changes two local
training-output defaults to the ignored `artifacts/generated/checkpoints`
directory, and represents static external-asset fingerprints embedded in
legacy validation utilities by the neutral `external-fingerprint-omitted`
placeholder. The latter prevents cross-repository linkage and is not used by
the paper-table, figure, or artifact-verification commands. Simulator, policy,
planner, and paper-statistics semantics are unchanged.

## Frozen versus generated outputs

Committed evidence is immutable under `artifacts/raw`, `artifacts/statistics`, `artifacts/tables`, and `artifacts/figures`. Reproduction scripts write to `artifacts/generated` so a rerun cannot silently replace paper evidence.
