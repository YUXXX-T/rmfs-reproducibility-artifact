# Reproducibility appendix

This web appendix is the canonical review version; no identity-bearing PDF metadata is bundled.

## Checklist

- Simulator environment and tick order: documented and linked to code.
- Low/mid/high definitions: exact JSON plus YAML extraction.
- Pod, station, return, service, and arrival rules: documented.
- State/action/node/station/system/long-risk channels: machine-readable schemas.
- Training scale, data generation, two-stage optimization, J1 auxiliary
  training, and seed separation: the
  [training pipeline](reproduction/training_pipeline.md), source, and splits.
- Frozen-Greedy continuation: implementation and declared horizons.
- Greedy, Hungarian, JSQ, WM-Base, S1/J1, PP, and PIBT: configuration files.
- Per-seed raw result rows and paired arrival manifests: committed with schema and count validation.
- Paired bootstrap intervals: deterministic script and complete Proposed–JSQ/WM-Base CSV.
- Dedicated synchronized runtime benchmark: paired CPU/GPU data, hardware and
  timing protocol, static Fig. 6/6s, and a deterministic plotting script.
- Collapse and station-lock rules: separate versioned detection configs.
- Sensitivity: station-lock thresholds 0.6/0.7/0.8/0.9, with 0.8 preregistered for the main figure.

Large checkpoints and tick traces are addressed through manifests rather than silently omitted.
