# Paired arrival manifests

`main_50seed/` contains one manifest per `(load, seed)` for loads low/mid/high and seeds 900–949. Every dispatch arm replays the same cell manifest. `station6_10seed/` contains the corresponding three-load block for held-out seeds 721–730. `density_scale_10seed/` contains the 30 order streams for loads low/mid/high and seeds 701–710; each stream is reused across all nine map–density variants and five dispatch arms.

`metadata.csv` records campaign, relative path, schema version, and order count for all 210 files. Manifests contain arrival ticks, order IDs, station IDs, and SKU demands; they contain no policy outcomes or static fingerprints.
