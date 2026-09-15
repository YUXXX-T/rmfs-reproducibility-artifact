# Entities and lifecycle

An order specifies SKU quantities and a destination station. Its lifecycle is `PENDING → IN_PROGRESS → COMPLETED`. The upstream retriever selects one or more pods; each unfinished pod chain creates three robot tasks:

```text
PICK pod at current/home position
  → DELIVER carried pod to station queue/service
  → RETURN pod to the selected return location
```

Task states are `PENDING`, `ASSIGNED`, `IN_PROGRESS`, `COMPLETED`, or `CANCELLED`. The reported home-return policy sends a pod to its original home. An order completes only after all required pod chains are terminal and delivered. The action-conditioned WM does not change the chosen pod, station, return position, or order decomposition; it selects a robot for a fixed context.
