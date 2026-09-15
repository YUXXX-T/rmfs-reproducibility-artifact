# Paired arrival manifests

`main_50seed/` contains one manifest per `(load, seed)` for loads low/mid/high and seeds 900–949. Every dispatch arm replays the same cell manifest. `station6_10seed/` contains the corresponding three-load block for held-out seeds 721–730.

`metadata.csv` records campaign, relative path, schema version, and order count for all 180 files. Manifests contain arrival ticks, order IDs, station IDs, and SKU demands; they contain no policy outcomes.
