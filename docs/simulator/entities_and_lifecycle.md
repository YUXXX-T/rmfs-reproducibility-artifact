# Entities and lifecycle

`PICK -> DELIVER -> RETURN` names the three task **types** created for one
selected pod. It is not the complete execution lifecycle. The simulator tracks
four coupled state machines: order, task, robot, and station/pod state.

## From an order to executable work

An order contains SKU quantities and one destination station. The pod
retriever may select one or more pods to cover those quantities. Each selected
pod defines an independent chain assigned to one robot:

```text
order demand
  -> select pod(s)
  -> select a robot for each dispatchable (order, pod, station) context
  -> create PICK, DELIVER, and RETURN task records for each selected pod
```

A multi-pod order can therefore have several chains, potentially executed by
different robots and overlapping in time. The order is complete only when its
selected pods have been delivered and all task records belonging to the order
are terminal. Finishing one pod chain is not necessarily an order completion.

## Normal lifecycle of one pod chain

The full normal path is:

```text
robot IDLE
  -> assignment committed
  -> travel to pod
  -> pickup countdown
  -> pod attached to robot
  -> station admission / possible admission wait
  -> travel to station entry
  -> entry check-in
  -> queue or buffer motion
  -> station service countdown
  -> inventory delivered to order
  -> wait for an unblocked station exit
  -> move to exit and finalize DELIVER
  -> travel to return location
  -> drop-off countdown
  -> pod detached
  -> robot IDLE
```

For the reported `HomeReturnPlanner`, the return location is the pod's
original home cell. The pod remains attached while the robot is in the station
queue, service slot, and exit phase; it is detached only after the RETURN
countdown completes.

## Task states

The three task types are `PICK`, `DELIVER`, and `RETURN`, but every task also
has its own state:

```text
PENDING -> ASSIGNED -> IN_PROGRESS -> COMPLETED
              |             |
              +-> CANCELLED <-+
```

In the normal committed chain, all three task records are created as
`ASSIGNED`. They become `IN_PROGRESS` sequentially. Their completion points
are deliberately different:

| Task | Becomes active | Becomes complete |
|---|---|---|
| `PICK` | The robot starts the route to the selected pod. | The robot reaches the pod and the configured pickup countdown finishes. |
| `DELIVER` | Station admission succeeds and the robot starts or resumes its route to the station entry. | Service has finished **and** the robot has left the occupied station slot and reached the station exit. |
| `RETURN` | The completed DELIVER releases the next task in the chain. | The robot reaches the selected return cell and the configured drop-off countdown finishes. |

Thus `DELIVER` includes more than motion toward a station. Queueing, service,
exit backpressure, and exit finalization are part of that task's execution.

## Robot states

`AgentStatus` records the operational phase independently of `TaskStatus`:

| Robot state | Meaning |
|---|---|
| `IDLE` | Available for dispatch. |
| `MOVING_TO_POD` | Traveling to the PICK destination or completing its pickup wait. |
| `WAITING_ASSIGNED` | Carrying a pod but waiting for explicit station admission. This appears only in admission modes with an explicit waiting queue, not in the main `physical_only_legacy` protocol. |
| `CARRYING` | Carrying a pod toward the station entry or transitioning between delivery and return. |
| `QUEUING` | Checked in at the station entry and moving through entry, buffer, or queue cells. |
| `DELIVERING` | Occupying the service cell and completing station service. |
| `EXITING` | Service is finished, but the station slot/exit transition has not yet been finalized. A blocked exit keeps the robot in this state and preserves backpressure. |
| `RETURNING` | Carrying the pod from the station exit to its return cell. |
| `MOVING` | Generic movement state retained for nonstandard/repositioning paths. |

Action countdowns (`pickup_duration`, `station_process_duration`, and
`dropoff_duration`) are stored separately in `wait_ticks`. A robot can
therefore retain an operational status while being unable to move for the
remaining action ticks.

## Station and pod transitions

Station admission and physical occupancy are separate concepts. After
admission, a robot travels to the entry; after check-in it becomes `QUEUING`
and is advanced at most one station-zone cell per tick toward the service
slot. The counted physical capacity consists of service, queue, and buffer
slots. The entry is a handoff cell and is not included in that capacity.

After service, the robot enters `EXITING`. If the exit cell is occupied, the
robot remains in its station slot and continues to exert backpressure. When
the exit becomes free, the station moves the robot to it; on the following
exit-finalization step the DELIVER task becomes `COMPLETED`, after which RETURN
can begin.

The pod itself has no separate enum, but its effective lifecycle is:

```text
stationary at home and available
  -> reserved by an assigned/in-progress task chain
  -> carried through pickup, station travel, queue, service, and exit
  -> carried on the return trip
  -> put down at the planned return cell and available again
```

SKU inventory is decremented when station service completes and the pod is
marked delivered for the order.

## Order states and multi-pod completion

The order state is:

```text
PENDING -> IN_PROGRESS -> COMPLETED
    ^             |
    +-------------+  incomplete cancelled chain / retry
```

`PENDING` includes orders awaiting a robot or awaiting completion of their
selected dispatch contexts. Once the selected pod chains are covered by
active or already fulfilled work, the order becomes `IN_PROGRESS`. It becomes
`COMPLETED` only when all of its task records are terminal and its delivered
pod set covers the selected pod set.

## Failure and retry path

If a robot reaches a PICK destination but the pod is missing, already carried,
or no longer at that cell, the engine does not continue the nominal chain:

1. it marks the PICK and its related active DELIVER/RETURN records
   `CANCELLED`;
2. it searches for another available pod that covers the order demand;
3. if found, it creates a replacement PICK/DELIVER/RETURN chain and preserves
   the relevant dispatch ordering metadata;
4. if no replacement is available, it releases the robot;
5. when the order's tasks are terminal but its selected pods were not fully
   delivered, it clears the stale selection and returns the order to
   `PENDING` for retrieval and dispatch again.

This retry branch is why `CANCELLED` is terminal for an individual task but
does not imply successful order completion.

## Implementation map

| Concern | Canonical source |
|---|---|
| Order state and delivered-pod ledger | `src/WorldState/order_state.py` |
| Task types, states, and chain queries | `src/WorldState/task_state.py` |
| Robot operational states and waits | `src/WorldState/agent_state.py` |
| Context creation and task-chain commit | `src/Policies/TaskAssigner/context_assignment.py` |
| Station entry, queue cascade, service slot, and exit | `src/WorldState/station_state.py` |
| Action countdowns, delivery completion, retry, and order completion | `src/Engine/simulation_engine.py` |

The action-conditioned world model acts at robot selection for a fixed
`(order, pod, station, return)` context. It does not remove any of the physical
phases above and does not redefine their transition semantics.
