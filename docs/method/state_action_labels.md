# State, action, and label channels

The exact channel order is machine-readable:

- `schemas/state_features.yaml`: 10 node, 6 edge, and `5+S` demand channels;
- `schemas/action_features.yaml`: 8 node-action, 6 global-action, and 4 audit-only edge-action channels;
- `schemas/node_labels.yaml`: 6 node and 2 per-station future labels;
- `schemas/system_labels.yaml`: 7 future system labels;
- `schemas/long_risk_labels.yaml`: 6 long-risk outputs and the derived quantile combination.

The reported four-station model has demand dimension 9; the six-station adapter expands only the station-specific tail from four to six channels, producing dimension 11. The grid node/edge schemas do not depend on station count.

Action scope is fixed-context robot choice. Robot start, pod, station, return position, three route legs, route pressure, distances, station queue, and order size are encoded. `NO_ASSIGN` has an all-zero representation in the general implementation but is disabled in the reported main policy.

The authoritative executable definitions are `src/WorldModel/graph/graph_builder.py` and `src/WorldModel/core/long_risk_schema.py`; the YAML schemas are review-friendly mirrors checked by tests.
